from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.GA.wfjssp_ga import WFJSSPGA, WorkerGAConfig


def _config(**kwargs) -> WorkerGAConfig:
    values = {
        "durations": [
            [[3.0]],
            [[4.0]],
        ],
        "job_sequence": [0, 1],
        "population_size": 2,
        "offspring_amount": 2,
        "use_stochastic_evaluation": True,
        "n_simulations": 2,
        "seed": 123,
    }
    values.update(kwargs)
    return WorkerGAConfig(**values)


def test_worker_ga_config_accepts_surrogate_options():
    cfg = _config(use_surrogate_evaluation=True)

    assert cfg.use_surrogate_evaluation is True
    assert cfg.surrogate_warmup_real_candidates == 300
    assert cfg.surrogate_n_estimators == 300


def test_wfjssp_ga_initializes_surrogate_when_enabled_for_stochastic_eval():
    ga = WFJSSPGA(_config(use_surrogate_evaluation=True))

    assert ga.surrogate is not None
    assert ga.surrogate.min_samples_before_fit == 300
    assert ga.surrogate.max_training_samples == 5000


def test_surrogate_retrain_interval_grows_with_sample_count():
    ga = WFJSSPGA(
        _config(
            use_surrogate_evaluation=True,
            surrogate_warmup_real_candidates=10,
            surrogate_retrain_interval_real_candidates=100,
            surrogate_retrain_interval_growth_samples=50,
            surrogate_retrain_interval_growth_factor=2.0,
            surrogate_max_retrain_interval_real_candidates=500,
        )
    )

    assert ga.surrogate is not None
    assert ga._current_surrogate_retrain_interval() == 100

    ga.surrogate.add_samples([object()] * 60)
    assert ga._current_surrogate_retrain_interval() == 200

    ga.surrogate.add_samples([object()] * 200)
    assert ga._current_surrogate_retrain_interval() == 500


def test_evaluate_batch_empty_does_not_fail_with_surrogate_enabled():
    ga = WFJSSPGA(_config(use_surrogate_evaluation=True))

    ga.evaluate_batch([])

    assert ga.function_evaluations == 0
    assert ga.surrogate_predictions == 0


def test_surrogate_is_none_when_disabled():
    ga = WFJSSPGA(_config(use_surrogate_evaluation=False))

    assert ga.surrogate is None
