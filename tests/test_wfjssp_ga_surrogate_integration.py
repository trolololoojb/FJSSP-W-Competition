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


def _local_search_config(**kwargs) -> WorkerGAConfig:
    values = {
        "durations": [
            [[3.0, 4.0], [2.0, 3.0]],
            [[5.0, 6.0], [4.0, 5.0]],
            [[4.0, 5.0], [3.0, 4.0]],
            [[6.0, 7.0], [5.0, 6.0]],
            [[2.0, 3.0], [4.0, 5.0]],
            [[7.0, 8.0], [6.0, 7.0]],
        ],
        "job_sequence": [0, 0, 1, 1, 2, 2],
        "population_size": 1,
        "offspring_amount": 1,
        "use_stochastic_evaluation": True,
        "use_surrogate_evaluation": True,
        "n_simulations": 2,
        "seed": 123,
    }
    values.update(kwargs)
    return WorkerGAConfig(**values)


def _real_origin(ga: WFJSSPGA) -> WFJSSPIndividual:
    origin = WFJSSPIndividual(ga.config, ga.rng, randomize=False)
    origin.sequence = list(ga.config.job_sequence)
    origin.assignments = [0] * ga.config.n_operations
    origin.workers = [0] * ga.config.n_operations
    origin.fitness = {"makespan": 100.0, "fitness_source": "real"}
    return origin


def test_worker_ga_config_accepts_surrogate_options():
    cfg = _config(use_surrogate_evaluation=True)

    assert cfg.use_surrogate_evaluation is True
    assert cfg.surrogate_warmup_real_candidates == 300
    assert cfg.surrogate_n_estimators == 300
    assert cfg.local_search_real_eval_limit_per_origin == 0
    assert cfg.local_search_min_predicted_improvement == 0.0


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


def test_local_search_applies_min_predicted_improvement_and_origin_eval_limit():
    ga = WFJSSPGA(
        _config(
            use_surrogate_evaluation=True,
            local_search_top_k=4,
            local_search_real_eval_limit_per_origin=2,
            local_search_min_predicted_improvement=5.0,
            seed=123,
        )
    )
    origin = WFJSSPIndividual(ga.config, ga.rng, randomize=False)
    origin.fitness = {"makespan": 100.0, "fitness_source": "real"}
    predictions = [
        SimpleNamespace(candidate_id=1, predicted_robust_makespan=99.0, uncertainty_R=0.1),
        SimpleNamespace(candidate_id=2, predicted_robust_makespan=96.0, uncertainty_R=0.2),
        SimpleNamespace(candidate_id=3, predicted_robust_makespan=95.0, uncertainty_R=0.3),
        SimpleNamespace(candidate_id=4, predicted_robust_makespan=90.0, uncertainty_R=0.4),
    ]
    records_by_id = {prediction.candidate_id: {"origin": origin} for prediction in predictions}

    selected = ga._select_local_search_candidate_ids(predictions, records_by_id)

    assert selected == {3, 4}


def test_local_search_runs_only_when_parameters_are_positive():
    ga = WFJSSPGA(
        _local_search_config(
            local_search_interval=0,
            local_search_origin_count=1,
            local_search_neighbors_per_origin=1,
        )
    )

    class FakeSurrogate:
        model = object()

        def is_ready(self):
            return True

    ga.surrogate = FakeSurrogate()

    assert not ga._local_search_enabled()

    ga.config.local_search_interval = 1
    assert ga._local_search_enabled()


def test_local_search_removes_duplicate_neighbors():
    ga = WFJSSPGA(_local_search_config(local_search_neighbors_per_origin=100))
    origin = _real_origin(ga)
    decoded = ga._decode_individual(origin)

    neighbors = ga._generate_local_neighbors(origin, decoded, limit=100)
    keys = [ga.individual_key(neighbor) for _, neighbor in neighbors]

    assert len(keys) == len(set(keys))


def test_local_search_generates_worker_machine_swap_and_shift_neighbors():
    ga = WFJSSPGA(_local_search_config(local_search_neighbors_per_origin=100))
    origin = _real_origin(ga)
    decoded = ga._decode_individual(origin)

    neighbors = ga._generate_local_neighbors(origin, decoded, limit=100)
    kinds = {kind for kind, _ in neighbors}

    assert {"worker", "machine_worker", "sequence_swap", "sequence_shift"} <= kinds


def test_local_search_real_eval_limit_per_origin_is_enforced():
    ga = WFJSSPGA(
        _local_search_config(
            local_search_interval=1,
            local_search_origin_count=1,
            local_search_neighbors_per_origin=50,
            local_search_top_k=50,
            local_search_real_eval_limit_per_origin=3,
        )
    )
    origin = _real_origin(ga)
    ga.population = [origin]

    class FakeSurrogate:
        model = object()
        samples = []

        def is_ready(self):
            return True

        def predict_many(self, records):
            return [
                SimpleNamespace(
                    candidate_id=record["candidate_id"],
                    predicted_robust_makespan=10.0 + index,
                    uncertainty_R=0.0,
                )
                for index, record in enumerate(records)
            ]

        def fit(self):
            return True

    ga.surrogate = FakeSurrogate()

    def fake_evaluate_real(ind, candidate_id=None):
        ind.fitness = {"makespan": 99.0, "fitness_source": "real", "candidate_id": candidate_id}
        return 99.0

    ga.evaluate_real = fake_evaluate_real

    ga.run_local_search()

    assert ga.last_local_search_metrics["local_search_real_evaluations"] == 3


def test_local_search_worse_real_neighbor_does_not_replace_origin():
    ga = WFJSSPGA(
        _local_search_config(
            local_search_interval=1,
            local_search_origin_count=1,
            local_search_neighbors_per_origin=20,
            local_search_top_k=1,
        )
    )
    origin = _real_origin(ga)
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
                    predicted_robust_makespan=50.0,
                    uncertainty_R=0.0,
                )
            ]

        def fit(self):
            return True

    ga.surrogate = FakeSurrogate()

    def fake_evaluate_real(ind, candidate_id=None):
        ind.fitness = {"makespan": 101.0, "fitness_source": "real", "candidate_id": candidate_id}
        return 101.0

    ga.evaluate_real = fake_evaluate_real

    ga.run_local_search()

    assert ga.population[0] is origin
    assert ga.population[0].fitness["makespan"] == 100.0
    assert ga.last_local_search_metrics["local_search_replacements"] == 0


def test_local_search_better_real_neighbor_replaces_origin():
    ga = WFJSSPGA(
        _local_search_config(
            local_search_interval=1,
            local_search_origin_count=1,
            local_search_neighbors_per_origin=20,
            local_search_top_k=1,
        )
    )
    origin = _real_origin(ga)
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
                    predicted_robust_makespan=50.0,
                    uncertainty_R=0.0,
                )
            ]

        def fit(self):
            return True

    ga.surrogate = FakeSurrogate()

    def fake_evaluate_real(ind, candidate_id=None):
        ind.fitness = {"makespan": 99.0, "fitness_source": "real", "candidate_id": candidate_id}
        return 99.0

    ga.evaluate_real = fake_evaluate_real

    ga.run_local_search()

    assert ga.population[0].fitness["makespan"] == 99.0
    assert ga.last_local_search_metrics["local_search_replacements"] == 1
