from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.compare_hpo_component_factorial_scenario2 as component_runner  # noqa: E402
from scripts.compare_hpo_component_factorial_scenario2 import (  # noqa: E402
    ALL_VARIANTS,
    LEGACY_EXPERIMENT,
    LOCAL_SEARCH_KEYS,
    NEW_VARIANTS,
    QUALITY_NONINFERIORITY_MARGIN,
    REFERENCE_VARIANTS,
    RL_KEYS,
    _ensure_task_metadata,
    _ok_rows_by_run,
    _validate_resume_state,
    atomic_json,
    bootstrap_geomean_ci,
    build_component_configs,
    design_digest,
    evaluate_hypotheses,
    reference_tree_digest,
    roots_overlap,
    task_specs,
    task_design_payload,
    validate_matrix,
)
from solver.GA.wfjssp_ga import WFJSSPGA, WorkerGAConfig  # noqa: E402


PLAIN = "hpo_plain_ga"
PIPELINE = "hpo_no_rl"
RL_ONLY = "hpo_rl_only"
COMBINED = "hpo_with_rl"

EXPECTED_LOCAL_SEARCH_KEYS = {
    "local_search_interval",
    "local_search_origin_count",
    "local_search_neighbors_per_origin",
    "local_search_top_k",
    "local_search_uncertain_k",
    "local_search_random_k",
    "local_search_real_eval_limit_per_origin",
    "local_search_min_predicted_improvement",
}
EXPECTED_RL_KEYS = {
    "enable_rl_mutation_control",
    "rl_gamma",
    "rl_lambda",
    "rl_clip_epsilon",
    "rl_value_coef",
    "rl_warmup_generations",
    "rl_history_length",
    "rl_hidden_size",
    "rl_entropy_coef",
    "rl_learning_rate",
    "rl_update_interval",
}


def _hpo_ga_config() -> dict[str, object]:
    return {
        "population_size": 360,
        "offspring_amount": 2160,
        "elitism_rate": 0.05,
        "restart_generations": 800,
        "use_surrogate_evaluation": True,
        "surrogate_warmup_real_candidates": 600,
        "surrogate_top_fraction": 0.01,
        "surrogate_uncertain_fraction": 0.005,
        "surrogate_random_fraction": 0.0075,
        "surrogate_min_real_per_generation": 5,
        "surrogate_retrain_interval_real_candidates": 75,
        "surrogate_n_estimators": 200,
        "surrogate_min_samples_leaf": 5,
        "surrogate_max_features": "sqrt",
        "surrogate_n_jobs": 2,
        "surrogate_max_training_samples": 5_000,
        "surrogate_retrain_interval_growth_samples": 5_000,
        "surrogate_retrain_interval_growth_factor": 2.0,
        "surrogate_max_retrain_interval_real_candidates": 1_000,
        "local_search_interval": 10,
        "local_search_origin_count": 3,
        "local_search_neighbors_per_origin": 256,
        "local_search_top_k": 8,
        "local_search_uncertain_k": 6,
        "local_search_random_k": 2,
        "local_search_real_eval_limit_per_origin": 12,
        "local_search_min_predicted_improvement": 2.0,
        "enable_rl_mutation_control": False,
    }


def _reference_configs() -> dict[str, dict[str, object]]:
    no_rl = _hpo_ga_config()
    with_rl = deepcopy(no_rl)
    with_rl.update(
        {
            "enable_rl_mutation_control": True,
            "rl_gamma": 0.99,
            "rl_lambda": 0.95,
            "rl_clip_epsilon": 0.2,
            "rl_value_coef": 0.5,
            "rl_warmup_generations": 10,
            "rl_history_length": 3,
            "rl_hidden_size": 32,
            "rl_entropy_coef": 0.01,
            "rl_learning_rate": 0.0001,
            "rl_update_interval": 8,
        }
    )
    base_source = "final_rank03_race2_rank04_race1_rank08_TPE0071_8834ed6804"
    return {
        PIPELINE: {
            "ga_config": no_rl,
            "internal_simulations": 12,
            "base_source": base_source,
            "rl_source": None,
        },
        COMBINED: {
            "ga_config": with_rl,
            "internal_simulations": 12,
            "base_source": base_source,
            "rl_source": "rl_lr1e-04_u008",
        },
    }


def _complete_matrix(
    reference_configs: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Accept builders returning either the two new cells or all four cells."""
    built = build_component_configs(reference_configs)
    matrix = deepcopy(reference_configs)
    matrix.update(built)
    return matrix


def _ga_config(
    configs: dict[str, dict[str, object]], variant: str
) -> dict[str, object]:
    entry = configs[variant]
    ga_config = entry.get("ga_config", entry)
    assert isinstance(ga_config, dict)
    return ga_config


def _changed_keys(left: dict[str, object], right: dict[str, object]) -> set[str]:
    missing = object()
    return {
        key
        for key in set(left) | set(right)
        if left.get(key, missing) != right.get(key, missing)
    }


def _is_supported(decisions: dict[str, object], hypothesis: str) -> bool:
    value = decisions[hypothesis]
    if isinstance(value, dict):
        value = value["supported"]
    return bool(value)


def _comparison(
    *, quality_ratio_ci_high: float, raw_fe_ratio_ci_high: float
) -> dict[str, float]:
    """Canonical input row consumed by ``evaluate_hypotheses``.

    Both values are upper confidence bounds for candidate/baseline ratios;
    smaller values favour the candidate.
    """
    return {
        "quality_ci_high": quality_ratio_ci_high,
        "raw_fe_ci_high": raw_fe_ratio_ci_high,
    }


def test_public_variant_and_component_constants_are_unambiguous() -> None:
    assert tuple(NEW_VARIANTS) == (PLAIN, RL_ONLY)
    assert tuple(REFERENCE_VARIANTS) == (PIPELINE, COMBINED)
    assert set(ALL_VARIANTS) == {PLAIN, PIPELINE, RL_ONLY, COMBINED}
    assert len(tuple(ALL_VARIANTS)) == 4
    assert set(LOCAL_SEARCH_KEYS) == EXPECTED_LOCAL_SEARCH_KEYS
    assert set(RL_KEYS) == EXPECTED_RL_KEYS
    assert QUALITY_NONINFERIORITY_MARGIN == pytest.approx(0.02)


def test_task_specs_schedule_only_the_two_missing_cells_on_all_instances() -> None:
    instances_dir = REPO_ROOT / "instances" / "fjssp-w"
    expected_instances = sorted(path.name for path in instances_dir.glob("*.fjs"))
    assert len(expected_instances) == 30

    specs = task_specs(instances_dir)

    assert len(specs) == 60
    assert Counter(variant for variant, _ in specs) == {PLAIN: 30, RL_ONLY: 30}
    assert not ({variant for variant, _ in specs} & set(REFERENCE_VARIANTS))
    for variant in NEW_VARIANTS:
        assert sorted(instance for cell, instance in specs if cell == variant) == expected_instances
    assert len(set(specs)) == len(specs)


def test_output_and_reference_roots_must_be_disjoint(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    output = tmp_path / "output"
    reference.mkdir()
    output.mkdir()

    assert not roots_overlap(reference, output)
    assert roots_overlap(reference, reference)
    assert roots_overlap(reference, reference / "child")
    assert roots_overlap(output / "child", output)

    alias = tmp_path / "reference_alias"
    alias.symlink_to(reference, target_is_directory=True)
    assert roots_overlap(alias, reference)


def test_component_matrix_has_only_the_intended_pairwise_differences() -> None:
    references = _reference_configs()
    original_references = deepcopy(references)

    configs = _complete_matrix(references)
    validate_matrix(configs)

    assert references == original_references, "the reference artifacts must not be mutated"
    assert set(configs) == set(ALL_VARIANTS)

    plain = _ga_config(configs, PLAIN)
    pipeline = _ga_config(configs, PIPELINE)
    rl_only = _ga_config(configs, RL_ONLY)
    combined = _ga_config(configs, COMBINED)
    pipeline_keys = {"use_surrogate_evaluation", *LOCAL_SEARCH_KEYS}

    assert _changed_keys(plain, pipeline) == pipeline_keys
    assert _changed_keys(rl_only, combined) == pipeline_keys
    assert _changed_keys(plain, rl_only) == set(RL_KEYS)
    assert _changed_keys(pipeline, combined) == set(RL_KEYS)

    for config in (plain, rl_only):
        assert config["use_surrogate_evaluation"] is False
        for key in LOCAL_SEARCH_KEYS:
            assert config[key] == 0
    for config in (pipeline, combined):
        assert config["use_surrogate_evaluation"] is True
        assert config["local_search_interval"] == 10
        assert config["local_search_origin_count"] == 3
        assert config["local_search_neighbors_per_origin"] == 256

    for key in (
        "surrogate_warmup_real_candidates",
        "surrogate_top_fraction",
        "surrogate_uncertain_fraction",
        "surrogate_random_fraction",
        "surrogate_min_real_per_generation",
        "surrogate_retrain_interval_real_candidates",
        "surrogate_n_estimators",
        "surrogate_min_samples_leaf",
        "surrogate_max_features",
        "surrogate_n_jobs",
        "surrogate_max_training_samples",
        "surrogate_retrain_interval_growth_samples",
        "surrogate_retrain_interval_growth_factor",
        "surrogate_max_retrain_interval_real_candidates",
    ):
        assert plain[key] == pipeline[key] == rl_only[key] == combined[key]

    for key, value in {
        "population_size": 360,
        "offspring_amount": 2160,
        "elitism_rate": 0.05,
        "restart_generations": 800,
    }.items():
        assert all(_ga_config(configs, variant)[key] == value for variant in ALL_VARIANTS)

    assert configs[PLAIN]["internal_simulations"] == 12
    assert configs[RL_ONLY]["internal_simulations"] == 12
    assert configs[PLAIN]["base_source"] == configs[PIPELINE]["base_source"]
    assert configs[RL_ONLY]["base_source"] == configs[COMBINED]["base_source"]
    assert configs[PLAIN]["rl_source"] is None
    assert configs[RL_ONLY]["rl_source"] == configs[COMBINED]["rl_source"]


@pytest.mark.parametrize(
    ("variant", "key", "value"),
    [
        (PLAIN, "population_size", 361),
        (PLAIN, "local_search_interval", 1),
        (RL_ONLY, "use_surrogate_evaluation", True),
        (RL_ONLY, "rl_learning_rate", 0.5),
    ],
)
def test_validate_matrix_rejects_component_or_base_parameter_drift(
    variant: str, key: str, value: object
) -> None:
    configs = _complete_matrix(_reference_configs())
    _ga_config(configs, variant)[key] = value

    with pytest.raises(ValueError):
        validate_matrix(configs)


def test_rl_only_config_constructs_a_ga_without_surrogate_or_local_search() -> None:
    configs = _complete_matrix(_reference_configs())
    rl_only = deepcopy(_ga_config(configs, RL_ONLY))
    cfg = WorkerGAConfig(
        durations=[[[3.0]], [[4.0]]],
        job_sequence=[0, 1],
        use_stochastic_evaluation=True,
        n_simulations=2,
        seed=123,
        rl_seed=123,
        **rl_only,
    )
    ga = WFJSSPGA(cfg)

    assert ga.config.enable_rl_mutation_control is True
    assert ga.surrogate is None
    assert ga._local_search_enabled() is False
    for key in LOCAL_SEARCH_KEYS:
        assert getattr(ga.config, key) == 0

    # Some implementations create the controller in ``run`` and therefore do
    # not expose an attribute at construction time. If one is exposed here, it
    # must represent an active controller rather than a disabled placeholder.
    for attribute in ("rl_agent", "rl_mutation_agent", "mutation_controller"):
        if hasattr(ga, attribute):
            assert getattr(ga, attribute) is not None


def test_bootstrap_geomean_ci_is_seeded_and_exact_for_constant_ratios() -> None:
    first = bootstrap_geomean_ci([0.97] * 12, samples=250, seed=20260824)
    second = bootstrap_geomean_ci([0.97] * 12, samples=250, seed=20260824)

    assert first == second
    assert first[0] == pytest.approx(0.97)
    assert first[1] == pytest.approx(0.97)


def test_reference_dataset_digest_changes_only_with_reference_data(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    output = tmp_path / "output"
    instance = "example.fjs"
    (reference / "experiment_manifest.json").parent.mkdir(parents=True)
    (reference / "experiment_manifest.json").write_text("{}\n", encoding="utf-8")
    for variant in REFERENCE_VARIANTS:
        directory = reference / variant / "example"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text("{}\n", encoding="utf-8")
        (directory / "raw_results.jsonl").write_text(
            '{"status":"ok"}\n', encoding="utf-8"
        )

    before = reference_tree_digest(reference, [instance])
    output.mkdir()
    (output / "summary.json").write_text("{}\n", encoding="utf-8")
    assert reference_tree_digest(reference, [instance]) == before

    changed = reference / REFERENCE_VARIANTS[0] / "example" / "raw_results.jsonl"
    changed.write_text('{"status":"changed"}\n', encoding="utf-8")
    assert reference_tree_digest(reference, [instance]) != before


def test_incomplete_result_file_is_rejected_before_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "raw_results.jsonl"
    rows = [
        {
            "experiment": LEGACY_EXPERIMENT,
            "variant": PIPELINE,
            "instance": "example.fjs",
            "run": run,
            "status": "ok",
        }
        for run in range(1, 10)
    ]
    rows.append(
        {
            "experiment": LEGACY_EXPERIMENT,
            "variant": PIPELINE,
            "instance": "example.fjs",
            "run": 10,
            "status": "failed",
        }
    )
    path.write_text("".join(f"{__import__('json').dumps(row)}\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.compare_hpo_component_factorial_scenario2.validate_result_row",
        lambda *args, **kwargs: None,
    )

    incomplete = _ok_rows_by_run(
        path,
        variant=PIPELINE,
        instance="example.fjs",
        uncertainty={},
        legacy=True,
        require_complete=False,
    )
    assert set(incomplete) == set(range(1, 10))
    with pytest.raises(ValueError, match="missing successful runs \\[10\\]"):
        _ok_rows_by_run(
            path,
            variant=PIPELINE,
            instance="example.fjs",
            uncertainty={},
            legacy=True,
            require_complete=True,
        )


def test_resume_requires_exact_design_and_effective_config(tmp_path: Path) -> None:
    payload = {
        "variant": PLAIN,
        "reference_dataset_digest": "reference-v1",
        "ga_config": {"population_size": 360},
    }
    digest = design_digest(payload)
    config_digest = design_digest(payload["ga_config"])
    effective = {
        "variant": PLAIN,
        "design_digest": digest,
        "config_digest": config_digest,
        "ga_config": payload["ga_config"],
    }

    fresh = tmp_path / "fresh"
    created = _ensure_task_metadata(
        fresh,
        resume=True,
        payload=payload,
        digest=digest,
        config_digest=config_digest,
        effective_config=effective,
        manifest={
            "design_digest": digest,
            "config_digest": config_digest,
            "design": payload,
            "created_at_unix_s": 1.0,
        },
    )
    assert created
    assert (fresh / "manifest.json").is_file()
    assert (fresh / "effective_config.json").is_file()

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    atomic_json(incomplete / "effective_config.json", effective)
    with pytest.raises(RuntimeError, match="incomplete task metadata"):
        _validate_resume_state(
            incomplete,
            resume=True,
            payload=payload,
            digest=digest,
            config_digest=config_digest,
            effective_config=effective,
        )

    complete = tmp_path / "complete"
    complete.mkdir()
    atomic_json(
        complete / "manifest.json",
        {
            "design_digest": digest,
            "config_digest": config_digest,
            "design": payload,
        },
    )
    atomic_json(complete / "effective_config.json", effective)
    _validate_resume_state(
        complete,
        resume=True,
        payload=payload,
        digest=digest,
        config_digest=config_digest,
        effective_config=effective,
    )
    manifest_before = (complete / "manifest.json").read_bytes()
    effective_before = (complete / "effective_config.json").read_bytes()
    created = _ensure_task_metadata(
        complete,
        resume=True,
        payload=payload,
        digest=digest,
        config_digest=config_digest,
        effective_config=effective,
        manifest={
            "design_digest": digest,
            "config_digest": config_digest,
            "design": payload,
            "created_at_unix_s": 999.0,
        },
    )
    assert not created
    assert (complete / "manifest.json").read_bytes() == manifest_before
    assert (complete / "effective_config.json").read_bytes() == effective_before
    with pytest.raises(RuntimeError, match="use --resume"):
        _validate_resume_state(
            complete,
            resume=False,
            payload=payload,
            digest=digest,
            config_digest=config_digest,
            effective_config=effective,
        )

    changed_payload = deepcopy(payload)
    changed_payload["reference_dataset_digest"] = "reference-v2"
    with pytest.raises(RuntimeError, match="design"):
        _validate_resume_state(
            complete,
            resume=True,
            payload=changed_payload,
            digest=design_digest(changed_payload),
            config_digest=config_digest,
            effective_config=effective,
        )


def test_task_design_requires_and_pins_reference_dataset_digest(tmp_path: Path) -> None:
    instance = "example.fjs"
    (tmp_path / instance).write_text("instance data\n", encoding="utf-8")
    design = {
        "configs": build_component_configs(_reference_configs()),
        "protocol": {},
        "pinned_hashes": {},
        "source_file_hashes": {},
        "environment_versions": {},
        "reference_dataset_digest": None,
    }
    args = SimpleNamespace(instances_dir=tmp_path)

    with pytest.raises(ValueError, match="reference dataset digest"):
        task_design_payload(args, design, PLAIN, instance)

    design["reference_dataset_digest"] = "reference-v1"
    first = task_design_payload(args, design, PLAIN, instance)
    design["reference_dataset_digest"] = "reference-v2"
    second = task_design_payload(args, design, PLAIN, instance)
    assert design_digest(first) != design_digest(second)


def test_run_task_always_requests_full_reference_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GateReached(RuntimeError):
        pass

    def load_design_spy(args: object, *, validate_reference_rows: bool) -> dict[str, object]:
        assert validate_reference_rows is True
        raise GateReached

    monkeypatch.setattr(component_runner, "load_design", load_design_spy)
    args = SimpleNamespace(
        instances_dir=REPO_ROOT / "instances" / "fjssp-w",
        task_index=0,
    )
    with pytest.raises(GateReached):
        component_runner.run_task(args)


def test_hypothesis_decisions_apply_two_percent_noninferiority_margin() -> None:
    comparisons = {
        # H1: fewer real evaluations and quality just inside the allowed bound.
        "pipeline_effect_without_rl": _comparison(
            quality_ratio_ci_high=1.0 + QUALITY_NONINFERIORITY_MARGIN - 1e-6,
            raw_fe_ratio_ci_high=0.95,
        ),
        # H2 (and one half of H3): final quality is the primary endpoint.
        "rl_effect_with_pipeline": _comparison(
            quality_ratio_ci_high=0.99,
            raw_fe_ratio_ci_high=1.05,
        ),
        # The other half of H3 also requires statistically better quality.
        "pipeline_effect_with_rl": _comparison(
            quality_ratio_ci_high=0.99,
            raw_fe_ratio_ci_high=1.10,
        ),
    }

    decisions = evaluate_hypotheses(comparisons)

    assert _is_supported(decisions, "H1")
    assert _is_supported(decisions, "H2")
    assert _is_supported(decisions, "H3")


def test_hypothesis_decisions_reject_degradation_and_inconclusive_effects() -> None:
    comparisons = {
        # A large FE reduction cannot rescue quality outside the 2% margin.
        "pipeline_effect_without_rl": _comparison(
            quality_ratio_ci_high=1.020001,
            raw_fe_ratio_ci_high=0.80,
        ),
        # H2 remains unsupported because quality is primary, even with lower FE.
        "rl_effect_with_pipeline": _comparison(
            quality_ratio_ci_high=1.01,
            raw_fe_ratio_ci_high=0.80,
        ),
        # FE superiority cannot replace the second H3 quality comparison.
        "pipeline_effect_with_rl": _comparison(
            quality_ratio_ci_high=1.01,
            raw_fe_ratio_ci_high=0.80,
        ),
    }

    decisions = evaluate_hypotheses(comparisons)

    assert not _is_supported(decisions, "H1")
    assert not _is_supported(decisions, "H2")
    assert not _is_supported(decisions, "H3")
