from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.GA.wfjssp_ga import WFJSSPGA, WFJSSPIndividual, WorkerGAConfig


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


def test_local_search_selects_better_top_uncertain_and_random_candidates():
    ga = WFJSSPGA(
        _config(
            use_surrogate_evaluation=True,
            local_search_top_k=1,
            local_search_uncertain_k=1,
            local_search_random_k=1,
            seed=123,
        )
    )
    origin = WFJSSPIndividual(ga.config, ga.rng, randomize=False)
    origin.fitness = {"makespan": 100.0, "fitness_source": "real"}
    predictions = [
        SimpleNamespace(candidate_id=1, predicted_robust_makespan=99.0, uncertainty_R=0.1),
        SimpleNamespace(candidate_id=2, predicted_robust_makespan=120.0, uncertainty_R=0.2),
        SimpleNamespace(candidate_id=3, predicted_robust_makespan=80.0, uncertainty_R=0.3),
        SimpleNamespace(candidate_id=4, predicted_robust_makespan=110.0, uncertainty_R=5.0),
    ]
    records_by_id = {prediction.candidate_id: {"origin": origin} for prediction in predictions}

    selected = ga._select_local_search_candidate_ids(predictions, records_by_id)

    assert {1, 3, 4} <= selected
    assert len(selected) == 4


def test_local_search_replaces_origin_only_after_real_improvement():
    ga = WFJSSPGA(
        _config(
            use_stochastic_evaluation=True,
            use_surrogate_evaluation=True,
            local_search_interval=1,
            local_search_origin_count=1,
            local_search_neighbors_per_origin=1,
            local_search_top_k=1,
            seed=123,
        )
    )
    origin = WFJSSPIndividual(ga.config, ga.rng, randomize=False)
    origin.sequence = [0, 1]
    origin.assignments = [0, 0]
    origin.workers = [0, 0]
    origin.fitness = {"makespan": 10.0, "fitness_source": "real"}
    ga.population = [origin]

    class FakeSurrogate:
        model = object()
        samples = []

        def is_ready(self):
            return True

        def predict_many(self, records):
            return [
                SimpleNamespace(
                    candidate_id=records[0]["candidate_id"],
                    predicted_robust_makespan=9.0,
                    uncertainty_R=0.0,
                )
            ]

        def fit(self):
            return True

    ga.surrogate = FakeSurrogate()

    def fake_evaluate_real(ind, candidate_id=None):
        ind.fitness = {"makespan": 9.0, "fitness_source": "real", "candidate_id": candidate_id}
        return 9.0

    ga.evaluate_real = fake_evaluate_real

    ga.run_local_search()

    assert ga.population[0].fitness["makespan"] == 9.0
    assert ga.last_local_search_metrics["local_search_replacements"] == 1
