from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.GA.surrogate_features import feature_dicts_to_matrix
from solver.GA.surrogate_qrf import QRFSurrogate, SurrogateSample


def _sample(i: int) -> SurrogateSample:
    deterministic_makespan = float(i + 100)
    R = 1.0 + 0.001 * i
    return SurrogateSample(
        candidate_id=i,
        features={
            "deterministic_makespan": deterministic_makespan,
            "mean_worker_load": float(i % 10),
            "max_worker_load": float(i % 20),
        },
        deterministic_makespan=deterministic_makespan,
        robust_makespan=R * deterministic_makespan,
        robust_makespan_stdev=0.1 * float(i % 5),
        R=R,
        n_simulations=10,
    )


def _fitted_surrogate() -> QRFSurrogate:
    surrogate = QRFSurrogate(
        min_samples_before_fit=20,
        n_estimators=20,
        min_samples_leaf=1,
        random_state=123,
    )
    surrogate.add_samples([_sample(i) for i in range(40)])
    assert surrogate.fit()
    return surrogate


def test_feature_dicts_to_matrix_stable_column_order():
    X, feature_names = feature_dicts_to_matrix(
        [{"b": 2.0, "a": 1.0}, {"a": 3.0, "c": 4.0}]
    )

    assert feature_names == ["a", "b", "c"]
    assert X.tolist() == [[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]]

    X_fixed, fixed_names = feature_dicts_to_matrix(
        [{"b": 2.0, "a": 1.0}],
        feature_names=["b", "missing", "a"],
    )

    assert fixed_names == ["b", "missing", "a"]
    assert X_fixed.tolist() == [[2.0, 0.0, 1.0]]


def test_is_ready_false_before_enough_samples():
    surrogate = QRFSurrogate(min_samples_before_fit=3)
    surrogate.add_sample(_sample(0))

    assert not surrogate.is_ready()


def test_fit_returns_false_before_enough_samples():
    surrogate = QRFSurrogate(min_samples_before_fit=3)
    surrogate.add_samples([_sample(0), _sample(1)])

    assert surrogate.fit() is False


def test_fit_trains_after_enough_synthetic_samples():
    surrogate = _fitted_surrogate()

    assert surrogate.model is not None
    assert surrogate.feature_names == [
        "deterministic_makespan",
        "max_worker_load",
        "mean_worker_load",
    ]


def test_predict_one_returns_ordered_quantiles():
    surrogate = _fitted_surrogate()

    prediction = surrogate.predict_one(
        candidate_id=999,
        features={
            "deterministic_makespan": 130.0,
            "mean_worker_load": 3.0,
            "max_worker_load": 13.0,
        },
        deterministic_makespan=130.0,
    )

    assert prediction.q10_R <= prediction.q50_R <= prediction.q90_R
    assert prediction.uncertainty_R == pytest.approx(prediction.q90_R - prediction.q10_R)
    assert prediction.predicted_robust_makespan == pytest.approx(
        prediction.q50_R * 130.0
    )
    assert prediction.score == pytest.approx(prediction.q90_R * 130.0)


def test_predict_before_fit_raises_clear_exception():
    surrogate = QRFSurrogate(min_samples_before_fit=1)

    with pytest.raises(RuntimeError, match="QRFSurrogate is not fitted yet."):
        surrogate.predict_one(1, {"deterministic_makespan": 100.0}, 100.0)


def test_select_for_real_evaluation_returns_at_least_min_count():
    surrogate = _fitted_surrogate()
    records = [
        {
            "candidate_id": i,
            "features": {
                "deterministic_makespan": float(100 + i),
                "mean_worker_load": float(i % 10),
                "max_worker_load": float(i % 20),
            },
            "deterministic_makespan": float(100 + i),
        }
        for i in range(30)
    ]
    predictions = surrogate.predict_many(records)

    selected = surrogate.select_for_real_evaluation(
        predictions,
        top_fraction=0.02,
        uncertain_fraction=0.005,
        random_fraction=0.005,
        min_count=5,
        rng=123,
    )

    assert len(selected) >= 5
    assert selected <= {record["candidate_id"] for record in records}


def test_save_and_load_keep_usable_model(tmp_path: Path):
    surrogate = _fitted_surrogate()
    path = tmp_path / "qrf.pkl"

    surrogate.save(path)
    loaded = QRFSurrogate.load(path)

    prediction = loaded.predict_one(
        candidate_id=1000,
        features={
            "deterministic_makespan": 135.0,
            "mean_worker_load": 5.0,
            "max_worker_load": 15.0,
        },
        deterministic_makespan=135.0,
    )

    assert prediction.q10_R <= prediction.q50_R <= prediction.q90_R
