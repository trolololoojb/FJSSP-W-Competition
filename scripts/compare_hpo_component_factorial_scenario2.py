#!/usr/bin/env python3
"""Run and summarize the missing cells of the Scenario-2 component factorial.

The experiment crosses the surrogate/local-search pipeline with RL mutation
control under the fixed HPO base configuration.  Only the two missing cells
(`hpo_plain_ga` and `hpo_rl_only`) can be executed.  The completed
`hpo_no_rl` and `hpo_with_rl` cells are imported read-only from the earlier
HPO/RL factorial experiment.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENT = "scenario2_hpo_component_factorial"
LEGACY_EXPERIMENT = "scenario2_hpo_rl_factorial"
DESIGN_SCHEMA_VERSION = 1

NEW_VARIANTS = ("hpo_plain_ga", "hpo_rl_only")
REFERENCE_VARIANTS = ("hpo_no_rl", "hpo_with_rl")
ALL_VARIANTS = (
    "hpo_plain_ga",
    "hpo_no_rl",
    "hpo_rl_only",
    "hpo_with_rl",
)

COMPARISONS = (
    ("pipeline_effect_without_rl", "hpo_plain_ga", "hpo_no_rl"),
    ("rl_effect_without_pipeline", "hpo_plain_ga", "hpo_rl_only"),
    ("rl_effect_with_pipeline", "hpo_no_rl", "hpo_with_rl"),
    ("pipeline_effect_with_rl", "hpo_rl_only", "hpo_with_rl"),
    ("combined_effect", "hpo_plain_ga", "hpo_with_rl"),
)

LOCAL_SEARCH_KEYS = {
    "local_search_interval",
    "local_search_origin_count",
    "local_search_neighbors_per_origin",
    "local_search_top_k",
    "local_search_uncertain_k",
    "local_search_random_k",
    "local_search_real_eval_limit_per_origin",
    "local_search_min_predicted_improvement",
}
RL_KEYS = {
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
PIPELINE_KEYS = {"use_surrogate_evaluation"} | LOCAL_SEARCH_KEYS

EXPECTED_INSTANCE_COUNT = 30
EXPECTED_RUNS = 10
EXPECTED_INTERNAL_SIMULATIONS = 12
EXPECTED_FINAL_SIMULATIONS = 50
EXPECTED_MAX_FUNCTION_EVALUATIONS = 5_000_000
EXPECTED_TIME_LIMIT_S = 129_600
EXPECTED_WORKERS = 10
EXPECTED_SIMULATION_WORKERS = 2
EXPECTED_SURROGATE_N_JOBS = 2
PROGRESS_INTERVAL_EVALUATIONS = 50_000
FINAL_SIMULATION_SEED_OFFSET = 2_000_000_000
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_803
QUALITY_NONINFERIORITY_MARGIN = 0.02

PINNED_HPO_ID = "final_rank03_race2_rank04_race1_rank08_TPE0071_8834ed6804"
PINNED_RL_ID = "rl_lr1e-04_u008"
PINNED_HPO_ARTIFACT_SHA256 = (
    "9269d06b074a93a9d527d0eeb2bc7d25882e588d5eff74c8d12de8ababf4530f"
)
PINNED_RL_ARTIFACT_SHA256 = (
    "1ee29862ce818a658427e098f559c05583abaaf54d392d3d2eec1b6f43548490"
)
PINNED_UNCERTAINTY_SHA256 = (
    "79263971b5aa53d829e49506cca428a057a79ffa6cd2549487b8e59130e07051"
)
PINNED_REFERENCE_MANIFEST_SHA256 = (
    "afc010ddc9594a0c82903b5c1e64b32935dd260e99593b21868615a5ea231db9"
)

PROTOCOL = {
    "n_runs": EXPECTED_RUNS,
    "internal_simulations": EXPECTED_INTERNAL_SIMULATIONS,
    "final_simulations": EXPECTED_FINAL_SIMULATIONS,
    "max_function_evaluations": EXPECTED_MAX_FUNCTION_EVALUATIONS,
    "time_limit_s": EXPECTED_TIME_LIMIT_S,
    "workers": EXPECTED_WORKERS,
    "simulation_workers": EXPECTED_SIMULATION_WORKERS,
    "surrogate_n_jobs": EXPECTED_SURROGATE_N_JOBS,
    "progress_interval_evaluations": PROGRESS_INTERVAL_EVALUATIONS,
    "final_simulation_seed_offset": FINAL_SIMULATION_SEED_OFFSET,
    "keep_multiple": False,
    "do_restart": False,
}

SOURCE_FILES = (
    Path(__file__).resolve(),
    REPO_ROOT / "solver/GA/wfjssp_ga.py",
    REPO_ROOT / "solver/GA/parallel_simulation.py",
    REPO_ROOT / "solver/GA/rl_mutation_agent.py",
    REPO_ROOT / "solver/GA/surrogate_features.py",
    REPO_ROOT / "solver/GA/surrogate_qrf.py",
    REPO_ROOT / "scripts/run_scenario2_submission.py",
    REPO_ROOT / "util/benchmark_parser.py",
    REPO_ROOT / "util/encoding.py",
    REPO_ROOT / "util/evaluation.py",
    REPO_ROOT / "util/graph.py",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(
    path: Path,
    fields: list[str],
    rows: Iterable[dict[str, Any]],
    *,
    delimiter: str = ",",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        fh.flush()


def instance_names(instances_dir: Path) -> list[str]:
    names = sorted(path.name for path in instances_dir.glob("*.fjs"))
    if len(names) != EXPECTED_INSTANCE_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_INSTANCE_COUNT} Scenario-2 instances; "
            f"found {len(names)} in {instances_dir}"
        )
    return names


def task_specs(instances_dir: Path) -> list[tuple[str, str]]:
    return [
        (variant, instance)
        for variant in NEW_VARIANTS
        for instance in instance_names(instances_dir)
    ]


def task_dir(root: Path, variant: str, instance: str) -> Path:
    return root / variant / instance.removesuffix(".fjs")


def roots_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def validate_pinned_file(path: Path, expected_sha256: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Pinned {label} hash mismatch for {path}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return actual


def _zero_local_search(config: dict[str, Any]) -> None:
    for key in LOCAL_SEARCH_KEYS:
        config[key] = 0.0 if key == "local_search_min_predicted_improvement" else 0


def build_component_configs(
    reference_configs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Derive all four matrix cells from the actually executed B/D configs."""
    missing = [key for key in REFERENCE_VARIANTS if key not in reference_configs]
    if missing:
        raise ValueError(f"Reference manifest lacks variants: {', '.join(missing)}")

    no_rl = copy.deepcopy(reference_configs["hpo_no_rl"])
    with_rl = copy.deepcopy(reference_configs["hpo_with_rl"])
    if not isinstance(no_rl.get("ga_config"), dict) or not isinstance(
        with_rl.get("ga_config"), dict
    ):
        raise ValueError("Reference variants must contain ga_config objects")

    plain = copy.deepcopy(no_rl)
    plain_ga = plain["ga_config"]
    plain_ga["use_surrogate_evaluation"] = False
    plain_ga["enable_rl_mutation_control"] = False
    _zero_local_search(plain_ga)
    plain["rl_source"] = None

    rl_only = copy.deepcopy(with_rl)
    rl_only_ga = rl_only["ga_config"]
    rl_only_ga["use_surrogate_evaluation"] = False
    rl_only_ga["enable_rl_mutation_control"] = True
    _zero_local_search(rl_only_ga)

    return {
        "hpo_plain_ga": plain,
        "hpo_no_rl": no_rl,
        "hpo_rl_only": rl_only,
        "hpo_with_rl": with_rl,
    }


def differing_keys(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    return {
        key
        for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    }


def validate_matrix(configs: dict[str, dict[str, Any]]) -> None:
    if set(configs) != set(ALL_VARIANTS) or len(configs) != len(ALL_VARIANTS):
        raise ValueError(
            f"Matrix variants must be {ALL_VARIANTS}, got {tuple(configs)}"
        )
    for variant, wrapper in configs.items():
        if int(wrapper.get("internal_simulations", 0)) != EXPECTED_INTERNAL_SIMULATIONS:
            raise ValueError(f"{variant} must use 12 internal simulations")
        if wrapper.get("base_source") != PINNED_HPO_ID:
            raise ValueError(f"{variant} does not use pinned HPO base {PINNED_HPO_ID}")
        expected_rl_source = PINNED_RL_ID if variant in {
            "hpo_rl_only",
            "hpo_with_rl",
        } else None
        if wrapper.get("rl_source") != expected_rl_source:
            raise ValueError(f"Unexpected RL source for {variant}")

    ga = {variant: configs[variant]["ga_config"] for variant in ALL_VARIANTS}
    for variant, config in ga.items():
        required_core = {
            "population_size": 360,
            "offspring_amount": 2160,
            "elitism_rate": 0.05,
            "restart_generations": 800,
            "surrogate_n_jobs": EXPECTED_SURROGATE_N_JOBS,
        }
        mismatches = {
            key: (config.get(key), expected)
            for key, expected in required_core.items()
            if config.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Pinned HPO fields differ for {variant}: {mismatches}")

    for variant in ("hpo_plain_ga", "hpo_rl_only"):
        config = ga[variant]
        if config.get("use_surrogate_evaluation") is not False:
            raise ValueError(f"Surrogate must be disabled for {variant}")
        nonzero = {key: config.get(key) for key in LOCAL_SEARCH_KEYS if config.get(key) != 0}
        if nonzero:
            raise ValueError(f"Local search must be explicitly zero for {variant}: {nonzero}")
    for variant in ("hpo_no_rl", "hpo_with_rl"):
        if ga[variant].get("use_surrogate_evaluation") is not True:
            raise ValueError(f"Surrogate must be enabled for {variant}")
        if any(float(ga[variant].get(key, 0)) <= 0 for key in (
            "local_search_interval",
            "local_search_origin_count",
            "local_search_neighbors_per_origin",
        )):
            raise ValueError(f"Local search is not configured for {variant}")

    exact_differences = (
        ("hpo_plain_ga", "hpo_no_rl", PIPELINE_KEYS),
        ("hpo_rl_only", "hpo_with_rl", PIPELINE_KEYS),
        ("hpo_plain_ga", "hpo_rl_only", RL_KEYS),
        ("hpo_no_rl", "hpo_with_rl", RL_KEYS),
    )
    for left, right, expected in exact_differences:
        actual = differing_keys(ga[left], ga[right])
        if actual != expected:
            raise ValueError(
                f"Unexpected factor differences {left}<->{right}: "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )


def source_file_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in SOURCE_FILES
    }


def environment_versions() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in ("numpy", "torch", "quantile-forest", "scikit-learn"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {"python": sys.version, "packages": packages}


def git_provenance() -> dict[str, Any]:
    def run(*command: str) -> str | None:
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    # Limit status to executable source files so generated result paths cannot
    # bloat every task manifest, while an uncommitted runner is still visible.
    source_paths = [str(path.relative_to(REPO_ROOT)) for path in SOURCE_FILES]
    status = run("git", "status", "--short", "--", *source_paths)
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "status": status,
    }


def _validate_source_artifacts(
    hpo_artifact: dict[str, Any],
    rl_artifact: dict[str, Any],
    reference_configs: dict[str, dict[str, Any]],
) -> None:
    if hpo_artifact.get("config_id") != PINNED_HPO_ID:
        raise ValueError("Pinned HPO artifact has an unexpected config_id")
    if rl_artifact.get("config_id") != PINNED_RL_ID:
        raise ValueError("Pinned RL artifact has an unexpected config_id")
    if rl_artifact.get("source_config_id") != PINNED_HPO_ID:
        raise ValueError("Pinned RL artifact was not tuned on the pinned HPO base")
    if int(hpo_artifact.get("internal_simulations", 0)) != EXPECTED_INTERNAL_SIMULATIONS:
        raise ValueError("Pinned HPO artifact does not use 12 internal simulations")
    if int(rl_artifact.get("internal_simulations", 0)) != EXPECTED_INTERNAL_SIMULATIONS:
        raise ValueError("Pinned RL artifact does not use 12 internal simulations")

    expected_no_rl = copy.deepcopy(hpo_artifact.get("ga_config"))
    expected_with_rl = copy.deepcopy(rl_artifact.get("ga_config"))
    if not isinstance(expected_no_rl, dict) or not isinstance(expected_with_rl, dict):
        raise ValueError("Pinned artifacts lack ga_config objects")
    expected_no_rl["enable_rl_mutation_control"] = False
    expected_no_rl["surrogate_n_jobs"] = EXPECTED_SURROGATE_N_JOBS
    expected_with_rl["enable_rl_mutation_control"] = True
    expected_with_rl["surrogate_n_jobs"] = EXPECTED_SURROGATE_N_JOBS
    if expected_no_rl != reference_configs["hpo_no_rl"]["ga_config"]:
        raise ValueError("Legacy hpo_no_rl config differs from the pinned HPO artifact")
    if expected_with_rl != reference_configs["hpo_with_rl"]["ga_config"]:
        raise ValueError("Legacy hpo_with_rl config differs from the pinned RL artifact")


def _load_uncertainty(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("metadata", {}).get("scenario") != 2:
        raise ValueError(f"{path} is not a Scenario-2 uncertainty file")
    if not isinstance(payload.get("instances"), dict):
        raise ValueError(f"{path} has no 'instances' object")
    return payload


def _uncertainty_for(
    payload: dict[str, Any], instance: str, run: int
) -> tuple[int, list[list[float]]]:
    instance_data = payload["instances"].get(instance)
    if not isinstance(instance_data, dict):
        raise KeyError(f"No uncertainty parameters for {instance}")
    runs = instance_data.get("runs")
    run_data = runs.get(str(run)) if isinstance(runs, dict) else None
    if not isinstance(run_data, dict):
        raise KeyError(f"No uncertainty parameters for {instance} run {run}")
    parameters = run_data.get("uncertainty_parameters")
    if not isinstance(parameters, list):
        raise ValueError(f"Invalid uncertainty parameters for {instance} run {run}")
    converted = [[float(value) for value in row] for row in parameters]
    return int(run_data["seed"]), converted


def _validate_manifest_protocol(
    manifest: dict[str, Any],
    *,
    variant: str,
    instance: str,
    wrapper: dict[str, Any],
) -> None:
    expected = {
        "experiment": LEGACY_EXPERIMENT,
        "variant": variant,
        "instance": instance,
        "n_runs": EXPECTED_RUNS,
        "internal_simulations": EXPECTED_INTERNAL_SIMULATIONS,
        "final_simulations": EXPECTED_FINAL_SIMULATIONS,
        "max_function_evaluations": EXPECTED_MAX_FUNCTION_EVALUATIONS,
        "time_limit_s": EXPECTED_TIME_LIMIT_S,
        "workers": EXPECTED_WORKERS,
        "simulation_workers": EXPECTED_SIMULATION_WORKERS,
        "surrogate_n_jobs": EXPECTED_SURROGATE_N_JOBS,
        "ga_config": wrapper["ga_config"],
        "base_source": wrapper["base_source"],
        "rl_source": wrapper["rl_source"],
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(
            f"Legacy task manifest mismatch for {variant}/{instance}: "
            f"{', '.join(mismatches)}"
        )


def _validate_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or (positive and converted <= 0):
        raise ValueError(f"{label} must be finite{' and positive' if positive else ''}")
    return converted


def _validate_json_int(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a JSON integer")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def validate_result_row(
    row: dict[str, Any],
    *,
    variant: str,
    instance: str,
    uncertainty: dict[str, Any],
    legacy: bool,
    expected_design_digest: str | None = None,
    expected_config_digest: str | None = None,
) -> None:
    expected_experiment = LEGACY_EXPERIMENT if legacy else EXPERIMENT
    if row.get("experiment") != expected_experiment:
        raise ValueError(f"Unexpected experiment in {variant}/{instance} run row")
    if row.get("variant") != variant or row.get("config_id") != variant:
        raise ValueError(f"Variant/config_id mismatch in {variant}/{instance}")
    if row.get("instance") != instance or row.get("status") != "ok":
        raise ValueError(f"Identity/status mismatch in {variant}/{instance}")
    run = _validate_json_int(row.get("run"), "run", positive=True)
    if run not in range(1, EXPECTED_RUNS + 1):
        raise ValueError(f"Invalid run number in {variant}/{instance}: {run}")
    expected_seed, expected_uncertainty = _uncertainty_for(uncertainty, instance, run)
    if _validate_json_int(row.get("seed"), "seed") != expected_seed:
        raise ValueError(f"Seed mismatch in {variant}/{instance} run {run}")
    if row.get("uncertainty_parameters") != expected_uncertainty:
        raise ValueError(f"Uncertainty mismatch in {variant}/{instance} run {run}")
    if (
        _validate_json_int(
            row.get("internal_simulations"), "internal_simulations", positive=True
        )
        != EXPECTED_INTERNAL_SIMULATIONS
    ):
        raise ValueError(f"Internal simulation mismatch in {variant}/{instance} run {run}")
    if not legacy and row.get("design_digest") != expected_design_digest:
        raise ValueError(f"Design digest mismatch in {variant}/{instance} run {run}")
    if not legacy and row.get("config_digest") != expected_config_digest:
        raise ValueError(f"Config digest mismatch in {variant}/{instance} run {run}")
    expected_rl_enabled = variant in {"hpo_rl_only", "hpo_with_rl"}
    if row.get("rl_enabled") is not expected_rl_enabled:
        raise ValueError(f"RL activation mismatch in {variant}/{instance} run {run}")
    if not legacy:
        if row.get("surrogate_enabled") is not False:
            raise ValueError(f"Surrogate unexpectedly active in {variant}/{instance} run {run}")
        if row.get("local_search_configured") is not False:
            raise ValueError(f"Local search unexpectedly active in {variant}/{instance} run {run}")
        if (
            _validate_json_int(
                row.get("final_simulations"), "final_simulations", positive=True
            )
            != EXPECTED_FINAL_SIMULATIONS
        ):
            raise ValueError(f"Final simulation count mismatch in {variant}/{instance} run {run}")
        slurm = row.get("slurm")
        if not isinstance(slurm, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in slurm.items()
        ):
            raise ValueError(f"Invalid SLURM provenance in {variant}/{instance} run {run}")

    fitness = _validate_number(row.get("fitness"), "fitness", positive=True)
    robust = _validate_number(
        row.get("final_robust_makespan"), "final_robust_makespan", positive=True
    )
    if not math.isclose(fitness, robust, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"fitness != final_robust_makespan in {variant}/{instance} run {run}")
    for key in (
        "final_robust_stdev",
        "final_R",
        "deterministic_makespan",
        "runtime_s",
    ):
        _validate_number(row.get(key), key, positive=key in {"deterministic_makespan", "runtime_s"})

    best_found = _validate_json_int(
        row.get("function_evaluations"), "function_evaluations", positive=True
    )
    raw = _validate_json_int(
        row.get("raw_function_evaluations"),
        "raw_function_evaluations",
        positive=True,
    )
    if best_found <= 0 or raw <= 0 or best_found > raw:
        raise ValueError(f"Invalid function evaluation counters in {variant}/{instance} run {run}")
    if raw > EXPECTED_MAX_FUNCTION_EVALUATIONS:
        raise ValueError(f"Raw FE exceeds protocol limit in {variant}/{instance} run {run}")
    if raw % EXPECTED_INTERNAL_SIMULATIONS != 0:
        raise ValueError(f"Raw FE is not a 12-simulation multiple in {variant}/{instance} run {run}")

    final_results = row.get("final_simulation_results")
    if not isinstance(final_results, list) or len(final_results) != EXPECTED_FINAL_SIMULATIONS:
        raise ValueError(f"Expected 50 final simulations in {variant}/{instance} run {run}")
    for value in final_results:
        _validate_number(value, "final_simulation_result")

    recomputed_mean = statistics.mean(float(value) for value in final_results)
    recomputed_stdev = statistics.stdev(float(value) for value in final_results)
    deterministic = float(row["deterministic_makespan"])
    recomputed_r = recomputed_mean / deterministic
    consistency = {
        "fitness": (fitness, recomputed_mean),
        "final_robust_stdev": (float(row["final_robust_stdev"]), recomputed_stdev),
        "final_R": (float(row["final_R"]), recomputed_r),
    }
    inconsistent = [
        key
        for key, (stored, recomputed) in consistency.items()
        if not math.isclose(stored, recomputed, rel_tol=1e-12, abs_tol=1e-9)
    ]
    if inconsistent:
        raise ValueError(
            f"Final simulation aggregate mismatch in {variant}/{instance} run {run}: "
            f"{', '.join(inconsistent)}"
        )

    schedules = (
        row.get("start_times"),
        row.get("machine_assignments"),
        row.get("worker_assignments"),
    )
    if not all(isinstance(values, list) for values in schedules):
        raise ValueError(f"Schedule fields must be lists in {variant}/{instance} run {run}")
    if len({len(values) for values in schedules}) != 1 or not schedules[0]:
        raise ValueError(f"Schedule lengths differ in {variant}/{instance} run {run}")
    for value in schedules[0]:
        _validate_number(value, "start_time")
    for label, assignments in zip(
        ("machine_assignment", "worker_assignment"), schedules[1:]
    ):
        for value in assignments:
            _validate_json_int(value, label)
    _validate_json_int(row.get("generations"), "generations", positive=True)


def _ok_rows_by_run(
    path: Path,
    *,
    variant: str,
    instance: str,
    uncertainty: dict[str, Any],
    legacy: bool,
    expected_design_digest: str | None = None,
    expected_config_digest: str | None = None,
    require_complete: bool,
) -> dict[int, dict[str, Any]]:
    ok: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        status = row.get("status")
        if status not in {"ok", "failed"}:
            raise ValueError(f"Unknown status in {path}: {status!r}")
        if status != "ok":
            if row.get("experiment") != (LEGACY_EXPERIMENT if legacy else EXPERIMENT):
                raise ValueError(f"Failed row has wrong experiment in {path}")
            if row.get("variant") != variant or row.get("instance") != instance:
                raise ValueError(f"Failed row identity mismatch in {path}")
            run = _validate_json_int(row.get("run"), "run", positive=True)
            if run not in range(1, EXPECTED_RUNS + 1):
                raise ValueError(f"Failed row has invalid run in {path}")
            if not legacy and row.get("design_digest") != expected_design_digest:
                raise ValueError(f"Failed row design mismatch in {path} run {run}")
            if not legacy and row.get("config_digest") != expected_config_digest:
                raise ValueError(f"Failed row config mismatch in {path} run {run}")
            continue
        validate_result_row(
            row,
            variant=variant,
            instance=instance,
            uncertainty=uncertainty,
            legacy=legacy,
            expected_design_digest=expected_design_digest,
            expected_config_digest=expected_config_digest,
        )
        run = int(row["run"])
        if run in ok:
            if canonical_json(ok[run]) != canonical_json(row):
                raise ValueError(f"Conflicting successful duplicate in {path} for run {run}")
            raise ValueError(f"Duplicate successful row in {path} for run {run}")
        ok[run] = row
    if require_complete and set(ok) != set(range(1, EXPECTED_RUNS + 1)):
        missing = sorted(set(range(1, EXPECTED_RUNS + 1)) - set(ok))
        raise ValueError(f"Incomplete results in {path}; missing successful runs {missing}")
    return ok


def reference_tree_digest(reference_root: Path, names: list[str]) -> str:
    entries: list[tuple[str, str]] = []
    root_manifest = reference_root / "experiment_manifest.json"
    entries.append(("experiment_manifest.json", sha256_file(root_manifest)))
    for variant in REFERENCE_VARIANTS:
        for instance in names:
            directory = task_dir(reference_root, variant, instance)
            for filename in ("manifest.json", "raw_results.jsonl"):
                path = directory / filename
                entries.append((str(path.relative_to(reference_root)), sha256_file(path)))
    return sha256_json(entries)


def new_tree_digest(output_root: Path, names: list[str]) -> str:
    entries: list[tuple[str, str]] = []
    for variant in NEW_VARIANTS:
        for instance in names:
            directory = task_dir(output_root, variant, instance)
            for filename in ("manifest.json", "effective_config.json", "raw_results.jsonl"):
                path = directory / filename
                entries.append((str(path.relative_to(output_root)), sha256_file(path)))
    return sha256_json(entries)


def _validate_reference_rows(
    reference_root: Path,
    names: list[str],
    configs: dict[str, dict[str, Any]],
    uncertainty: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in REFERENCE_VARIANTS:
        variant_root = reference_root / variant
        discovered = {
            path.name for path in variant_root.iterdir() if path.is_dir()
        } if variant_root.is_dir() else set()
        expected_directories = {name.removesuffix(".fjs") for name in names}
        if discovered != expected_directories:
            raise ValueError(
                f"Legacy {variant} instance directories differ: "
                f"missing={sorted(expected_directories - discovered)}, "
                f"extra={sorted(discovered - expected_directories)}"
            )
        rows: list[dict[str, Any]] = []
        for instance in names:
            directory = task_dir(reference_root, variant, instance)
            manifest = read_json(directory / "manifest.json")
            _validate_manifest_protocol(
                manifest,
                variant=variant,
                instance=instance,
                wrapper=configs[variant],
            )
            keyed = _ok_rows_by_run(
                directory / "raw_results.jsonl",
                variant=variant,
                instance=instance,
                uncertainty=uncertainty,
                legacy=True,
                require_complete=True,
            )
            rows.extend(keyed[run] for run in sorted(keyed))
        if len(rows) != EXPECTED_INSTANCE_COUNT * EXPECTED_RUNS:
            raise ValueError(f"Expected 300 rows for {variant}, got {len(rows)}")
        rows_by_variant[variant] = rows

    left = {(row["instance"], row["run"]): row for row in rows_by_variant["hpo_no_rl"]}
    right = {(row["instance"], row["run"]): row for row in rows_by_variant["hpo_with_rl"]}
    if set(left) != set(right):
        raise ValueError("Legacy B/D result keysets differ")
    for key in sorted(left):
        if left[key]["seed"] != right[key]["seed"]:
            raise ValueError(f"Legacy B/D seed mismatch for {key}")
        if left[key]["uncertainty_parameters"] != right[key]["uncertainty_parameters"]:
            raise ValueError(f"Legacy B/D uncertainty mismatch for {key}")
    return rows_by_variant


def load_design(
    args: argparse.Namespace,
    *,
    validate_reference_rows: bool,
) -> dict[str, Any]:
    if roots_overlap(args.output_root, args.reference_root):
        raise ValueError("Output root and read-only reference root must be disjoint")
    names = instance_names(args.instances_dir)

    hashes = {
        "hpo_artifact": validate_pinned_file(
            args.hpo_config, PINNED_HPO_ARTIFACT_SHA256, "HPO artifact"
        ),
        "rl_artifact": validate_pinned_file(
            args.rl_config, PINNED_RL_ARTIFACT_SHA256, "RL artifact"
        ),
        "uncertainty": validate_pinned_file(
            args.uncertainty_json, PINNED_UNCERTAINTY_SHA256, "uncertainty file"
        ),
        "reference_manifest": validate_pinned_file(
            args.reference_root / "experiment_manifest.json",
            PINNED_REFERENCE_MANIFEST_SHA256,
            "reference experiment manifest",
        ),
    }

    reference_manifest = read_json(args.reference_root / "experiment_manifest.json")
    expected_root_values = {
        "experiment": LEGACY_EXPERIMENT,
        "instance_count": EXPECTED_INSTANCE_COUNT,
        "n_runs": EXPECTED_RUNS,
        "final_simulations": EXPECTED_FINAL_SIMULATIONS,
        "max_function_evaluations": EXPECTED_MAX_FUNCTION_EVALUATIONS,
        "time_limit_s": EXPECTED_TIME_LIMIT_S,
    }
    root_mismatches = [
        key
        for key, expected in expected_root_values.items()
        if reference_manifest.get(key) != expected
    ]
    if root_mismatches:
        raise ValueError(
            "Reference experiment manifest protocol mismatch: "
            + ", ".join(root_mismatches)
        )
    artifact_configs = reference_manifest.get("artifact_configs")
    if not isinstance(artifact_configs, dict):
        raise ValueError("Reference experiment manifest lacks artifact_configs")
    reference_configs = {
        variant: copy.deepcopy(artifact_configs[variant])
        for variant in REFERENCE_VARIANTS
        if variant in artifact_configs
    }
    configs = build_component_configs(reference_configs)
    validate_matrix(configs)

    hpo_artifact = read_json(args.hpo_config)
    rl_artifact = read_json(args.rl_config)
    _validate_source_artifacts(hpo_artifact, rl_artifact, configs)
    uncertainty = _load_uncertainty(args.uncertainty_json)

    for variant in REFERENCE_VARIANTS:
        for instance in names:
            manifest = read_json(task_dir(args.reference_root, variant, instance) / "manifest.json")
            _validate_manifest_protocol(
                manifest,
                variant=variant,
                instance=instance,
                wrapper=configs[variant],
            )

    reference_rows = None
    reference_digest = None
    if validate_reference_rows:
        reference_digest_before = reference_tree_digest(args.reference_root, names)
        reference_rows = _validate_reference_rows(
            args.reference_root, names, configs, uncertainty
        )
        reference_digest = reference_tree_digest(args.reference_root, names)
        if reference_digest != reference_digest_before:
            raise RuntimeError("Reference dataset changed while it was being validated")

    return {
        "configs": configs,
        "protocol": copy.deepcopy(PROTOCOL),
        "instances": names,
        "uncertainty": uncertainty,
        "pinned_hashes": hashes,
        "reference_manifest": reference_manifest,
        "reference_rows": reference_rows,
        "reference_dataset_digest": reference_digest,
        "source_file_hashes": source_file_hashes(),
        "environment_versions": environment_versions(),
    }


def task_design_payload(
    args: argparse.Namespace,
    design: dict[str, Any],
    variant: str,
    instance: str,
) -> dict[str, Any]:
    if variant not in NEW_VARIANTS:
        raise ValueError(f"Only missing cells can be executed; got {variant}")
    reference_dataset_digest = design.get("reference_dataset_digest")
    if not isinstance(reference_dataset_digest, str) or not reference_dataset_digest:
        raise ValueError("Task designs require a validated reference dataset digest")
    wrapper = design["configs"][variant]
    return {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "variant": variant,
        "components": {
            "surrogate_and_local_search": variant not in {
                "hpo_plain_ga",
                "hpo_rl_only",
            },
            "rl": variant == "hpo_rl_only",
        },
        "instance": instance,
        "instance_sha256": sha256_file(args.instances_dir / instance),
        "ga_config": wrapper["ga_config"],
        "internal_simulations": wrapper["internal_simulations"],
        "base_source": wrapper["base_source"],
        "rl_source": wrapper["rl_source"],
        "protocol": design["protocol"],
        "pinned_hashes": design["pinned_hashes"],
        "reference_dataset_digest": reference_dataset_digest,
        "source_file_hashes": design["source_file_hashes"],
        "environment_versions": design["environment_versions"],
    }


def design_digest(payload: dict[str, Any]) -> str:
    return sha256_json(payload)


def _expected_effective_config(
    variant: str,
    wrapper: dict[str, Any],
    digest: str,
    config_digest: str,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "design_digest": digest,
        "config_digest": config_digest,
        **wrapper,
    }


def _validate_resume_state(
    directory: Path,
    *,
    resume: bool,
    payload: dict[str, Any],
    digest: str,
    config_digest: str,
    effective_config: dict[str, Any],
) -> bool:
    manifest_path = directory / "manifest.json"
    effective_config_path = directory / "effective_config.json"
    raw_path = directory / "raw_results.jsonl"
    existing_paths = [
        path for path in (manifest_path, effective_config_path, raw_path) if path.exists()
    ]
    if existing_paths and not resume:
        raise RuntimeError(
            f"Task output already exists in {directory}; use --resume or a fresh root"
        )
    if not existing_paths:
        return False
    if not manifest_path.is_file() or not effective_config_path.is_file():
        raise RuntimeError(
            f"Cannot resume incomplete task metadata in {directory}; "
            "manifest.json and effective_config.json are both required"
        )
    previous_manifest = read_json(manifest_path)
    mismatches: list[str] = []
    if previous_manifest.get("design_digest") != digest:
        mismatches.append("design_digest")
    if previous_manifest.get("config_digest") != config_digest:
        mismatches.append("config_digest")
    if previous_manifest.get("design") != payload:
        mismatches.append("design")
    if read_json(effective_config_path) != effective_config:
        mismatches.append("effective_config")
    if mismatches:
        raise RuntimeError(
            f"Cannot resume changed design in {directory}: {', '.join(mismatches)}"
        )
    return True


def _ensure_task_metadata(
    directory: Path,
    *,
    resume: bool,
    payload: dict[str, Any],
    digest: str,
    config_digest: str,
    effective_config: dict[str, Any],
    manifest: dict[str, Any],
) -> bool:
    """Validate existing metadata or create it once without resume rewrites."""
    already_exists = _validate_resume_state(
        directory,
        resume=resume,
        payload=payload,
        digest=digest,
        config_digest=config_digest,
        effective_config=effective_config,
    )
    if already_exists:
        return False
    atomic_json(directory / "manifest.json", manifest)
    atomic_json(directory / "effective_config.json", effective_config)
    return True


def solve_run(task: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_scenario2_submission import UNCERTAINTY_SOURCE
    from solver.GA.parallel_simulation import run_n_simulations_parallel
    from solver.GA.wfjssp_ga import build_ga_from_worker_encoding, is_simulatable_schedule
    from util.benchmark_parser import WorkerBenchmarkParser
    from util.evaluation import makespan, translate

    encoding = WorkerBenchmarkParser().parse_benchmark(str(task["instance_path"]))
    seed = int(task["seed"])
    start_wall = time.time()
    ga_config = copy.deepcopy(task["ga_config"])
    ga_config.update(
        {
            "seed": seed,
            "rl_seed": seed,
            "use_stochastic_evaluation": True,
            "n_simulations": EXPECTED_INTERNAL_SIMULATIONS,
            "simulation_workers": EXPECTED_SIMULATION_WORKERS,
            "surrogate_n_jobs": EXPECTED_SURROGATE_N_JOBS,
            "uncertainty_parameters": task["uncertainty_parameters"],
        }
    )
    ga = build_ga_from_worker_encoding(encoding, **ga_config)
    result = ga.run(
        max_generations=None,
        time_limit_s=EXPECTED_TIME_LIMIT_S,
        max_function_evaluations=EXPECTED_MAX_FUNCTION_EVALUATIONS,
        progress_interval_evaluations=PROGRESS_INTERVAL_EVALUATIONS,
        keep_multiple=False,
        do_restart=False,
    )
    best = result["best"]
    start_times, machines, workers = translate(
        best.sequence, best.assignments, best.workers, encoding.durations()
    )
    start_times = [int(x) if float(x).is_integer() else float(x) for x in start_times]
    machines = [int(x) for x in machines]
    workers = [int(x) for x in workers]
    end_times = [
        float(start_times[i] + encoding.durations()[i][machines[i]][workers[i]])
        for i in range(len(start_times))
    ]
    if not is_simulatable_schedule(
        start_times, end_times, machines, workers, encoding.job_sequence()
    ):
        raise ValueError("Best decoded schedule is not simulatable")

    simulation_args = (
        start_times,
        end_times,
        machines,
        workers,
        encoding.job_sequence(),
        encoding.durations(),
        task["uncertainty_parameters"],
        EXPECTED_FINAL_SIMULATIONS,
    )
    try:
        final_results, robust_makespan, robust_stdev, final_r = run_n_simulations_parallel(
            *simulation_args,
            uncertainty_source=UNCERTAINTY_SOURCE,
            processing_times=True,
            workers=EXPECTED_SIMULATION_WORKERS,
            seed=seed + FINAL_SIMULATION_SEED_OFFSET,
        )
    except TypeError:
        final_results, robust_makespan, robust_stdev, final_r = run_n_simulations_parallel(
            *simulation_args,
            processing_times=True,
            workers=EXPECTED_SIMULATION_WORKERS,
            seed=seed + FINAL_SIMULATION_SEED_OFFSET,
        )

    raw_evaluations = int(result["function_evaluations"])
    best_found_evaluations = int(
        result.get("best_found_function_evaluations", raw_evaluations)
    )
    if not (0 < best_found_evaluations <= raw_evaluations <= EXPECTED_MAX_FUNCTION_EVALUATIONS):
        raise ValueError(
            "Invalid FE counters: "
            f"best={best_found_evaluations}, raw={raw_evaluations}"
        )
    local_search_configured = all(
        int(ga_config[key]) > 0
        for key in (
            "local_search_interval",
            "local_search_origin_count",
            "local_search_neighbors_per_origin",
        )
    )
    return {
        "experiment": EXPERIMENT,
        "variant": task["variant"],
        "config_id": task["variant"],
        "instance": task["instance"],
        "run": int(task["run"]),
        "seed": seed,
        "status": "ok",
        "design_digest": task["design_digest"],
        "config_digest": task["config_digest"],
        "fitness": float(robust_makespan),
        "final_robust_makespan": float(robust_makespan),
        "final_robust_stdev": float(robust_stdev),
        "final_R": float(final_r),
        "deterministic_makespan": float(
            makespan(start_times, machines, workers, encoding.durations())
        ),
        "function_evaluations": best_found_evaluations,
        "raw_function_evaluations": raw_evaluations,
        "runtime_s": float(result.get("runtime_s", time.time() - start_wall)),
        "generations": int(result["generations"]),
        "rl_enabled": bool(ga_config["enable_rl_mutation_control"]),
        "surrogate_enabled": bool(ga_config["use_surrogate_evaluation"]),
        "local_search_configured": local_search_configured,
        "internal_simulations": EXPECTED_INTERNAL_SIMULATIONS,
        "final_simulations": EXPECTED_FINAL_SIMULATIONS,
        "start_times": start_times,
        "machine_assignments": machines,
        "worker_assignments": workers,
        "uncertainty_parameters": task["uncertainty_parameters"],
        "final_simulation_results": [float(x) for x in final_results],
        "slurm": task.get("slurm", {}),
    }


def _slurm_provenance() -> dict[str, str]:
    names = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_PARTITION",
        "SLURMD_NODENAME",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def run_task(args: argparse.Namespace) -> None:
    specs = task_specs(args.instances_dir)
    if args.task_index < 0 or args.task_index >= len(specs):
        raise IndexError(f"--task-index must be in 0..{len(specs) - 1}")
    design = load_design(args, validate_reference_rows=True)
    variant, instance = specs[args.task_index]
    if variant not in NEW_VARIANTS:
        raise RuntimeError(f"Refusing to execute completed reference variant {variant}")
    wrapper = design["configs"][variant]
    payload = task_design_payload(args, design, variant, instance)
    digest = design_digest(payload)
    config_digest = sha256_json(wrapper["ga_config"])

    directory = task_dir(args.output_root, variant, instance)
    raw_path = directory / "raw_results.jsonl"
    effective_config = _expected_effective_config(
        variant, wrapper, digest, config_digest
    )
    manifest = {
        "experiment": EXPERIMENT,
        "variant": variant,
        "instance": instance,
        "design_digest": digest,
        "config_digest": config_digest,
        "design": payload,
        "output_root": str(args.output_root),
        "reference_root": str(args.reference_root),
        "git": git_provenance(),
        "created_at_unix_s": time.time(),
    }
    _ensure_task_metadata(
        directory,
        resume=args.resume,
        payload=payload,
        digest=digest,
        config_digest=config_digest,
        effective_config=effective_config,
        manifest=manifest,
    )

    uncertainty = design["uncertainty"]
    completed: dict[int, dict[str, Any]] = {}
    if raw_path.exists():
        completed = _ok_rows_by_run(
            raw_path,
            variant=variant,
            instance=instance,
            uncertainty=uncertainty,
            legacy=False,
            expected_design_digest=digest,
            expected_config_digest=config_digest,
            require_complete=False,
        )

    run_tasks: list[dict[str, Any]] = []
    for run in range(1, EXPECTED_RUNS + 1):
        if run in completed:
            continue
        seed, parameters = _uncertainty_for(uncertainty, instance, run)
        run_tasks.append(
            {
                "variant": variant,
                "instance": instance,
                "instance_path": str(args.instances_dir / instance),
                "run": run,
                "seed": seed,
                "uncertainty_parameters": parameters,
                "ga_config": wrapper["ga_config"],
                "design_digest": digest,
                "config_digest": config_digest,
                "slurm": _slurm_provenance(),
            }
        )

    print(
        f"variant={variant} instance={instance} pending_runs={len(run_tasks)} "
        f"design_digest={digest}",
        flush=True,
    )
    failures = 0

    def save(task: dict[str, Any], row: dict[str, Any]) -> None:
        nonlocal failures
        if row.get("status") != "ok":
            failures += 1
        append_jsonl(raw_path, row)
        print(
            f"done variant={variant} instance={instance} run={task['run']} "
            f"status={row['status']}",
            flush=True,
        )

    def failed(task: dict[str, Any], exc: BaseException) -> dict[str, Any]:
        return {
            "experiment": EXPERIMENT,
            "variant": variant,
            "instance": instance,
            "run": int(task["run"]),
            "seed": int(task["seed"]),
            "status": "failed",
            "design_digest": digest,
            "config_digest": config_digest,
            "uncertainty_parameters": task["uncertainty_parameters"],
            "error": repr(exc),
        }

    if run_tasks:
        with ProcessPoolExecutor(max_workers=min(EXPECTED_WORKERS, len(run_tasks))) as pool:
            futures = {pool.submit(solve_run, task): task for task in run_tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = failed(task, exc)
                save(task, row)
    if failures:
        raise RuntimeError(f"{failures} runs failed for {variant}/{instance}")


def geometric_mean(values: list[float]) -> float | None:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(statistics.mean(math.log(value) for value in values))


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_geomean_ci(
    ratios: list[float],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    if not ratios or samples <= 0:
        return None, None
    if any(value <= 0 or not math.isfinite(value) for value in ratios):
        raise ValueError("Bootstrap ratios must be finite and positive")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        resample = [ratios[rng.randrange(len(ratios))] for _ in ratios]
        estimate = geometric_mean(resample)
        if estimate is not None:
            estimates.append(estimate)
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def evaluate_hypotheses(
    comparison_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    h1_comparison = comparison_map["pipeline_effect_without_rl"]
    h2_comparison = comparison_map["rl_effect_with_pipeline"]
    h3_other_component = comparison_map["pipeline_effect_with_rl"]

    h1_quality_noninferior = (
        float(h1_comparison["quality_ci_high"])
        < 1.0 + QUALITY_NONINFERIORITY_MARGIN
    )
    h1_raw_fe_reduced = float(h1_comparison["raw_fe_ci_high"]) < 1.0
    h2_quality_superior = float(h2_comparison["quality_ci_high"]) < 1.0
    h3_quality_superior_to_rl_only = (
        float(h3_other_component["quality_ci_high"]) < 1.0
    )
    decisions = {
        "H1": {
            "supported": h1_quality_noninferior and h1_raw_fe_reduced,
            "quality_noninferior": h1_quality_noninferior,
            "raw_function_evaluations_reduced": h1_raw_fe_reduced,
            "quality_noninferiority_margin": QUALITY_NONINFERIORITY_MARGIN,
            "comparison": "pipeline_effect_without_rl",
        },
        "H2": {
            "supported": h2_quality_superior,
            "quality_superior": h2_quality_superior,
            "primary_endpoint": "final robust makespan",
            "comparison": "rl_effect_with_pipeline",
        },
        "H3": {
            "supported": h2_quality_superior and h3_quality_superior_to_rl_only,
            "quality_superior_to_surrogate_pipeline_only": h2_quality_superior,
            "quality_superior_to_rl_only": h3_quality_superior_to_rl_only,
            "comparisons": [
                "rl_effect_with_pipeline",
                "pipeline_effect_with_rl",
            ],
        },
    }
    for decision in decisions.values():
        decision["decision_criterion_met"] = decision["supported"]
        decision["supported_within_protocol"] = decision["supported"]
        decision["evidence_scope"] = "retrospective_component_ablation"
    return decisions


def _hardware_provenance(
    rows_by_variant: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        variant: {
            "recorded": False,
            "reason": "legacy experiment did not record SLURM node or partition",
        }
        for variant in REFERENCE_VARIANTS
    }
    for variant in NEW_VARIANTS:
        slurm_rows = [row.get("slurm", {}) for row in rows_by_variant[variant]]
        result[variant] = {
            "recorded": any(bool(value) for value in slurm_rows),
            "rows_without_slurm_metadata": sum(not bool(value) for value in slurm_rows),
            "partitions": sorted(
                {
                    value["SLURM_JOB_PARTITION"]
                    for value in slurm_rows
                    if "SLURM_JOB_PARTITION" in value
                }
            ),
            "nodes": sorted(
                {
                    value["SLURMD_NODENAME"]
                    for value in slurm_rows
                    if "SLURMD_NODENAME" in value
                }
            ),
            "array_job_ids": sorted(
                {
                    value["SLURM_ARRAY_JOB_ID"]
                    for value in slurm_rows
                    if "SLURM_ARRAY_JOB_ID" in value
                }
            ),
        }
    return result


def _load_new_rows(
    args: argparse.Namespace,
    design: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in NEW_VARIANTS:
        variant_root = args.output_root / variant
        discovered = {
            path.name for path in variant_root.iterdir() if path.is_dir()
        } if variant_root.is_dir() else set()
        expected_directories = {
            name.removesuffix(".fjs") for name in design["instances"]
        }
        if discovered != expected_directories:
            raise ValueError(
                f"New {variant} instance directories differ: "
                f"missing={sorted(expected_directories - discovered)}, "
                f"extra={sorted(discovered - expected_directories)}"
            )
        rows: list[dict[str, Any]] = []
        for instance in design["instances"]:
            directory = task_dir(args.output_root, variant, instance)
            payload = task_design_payload(args, design, variant, instance)
            expected_digest = design_digest(payload)
            manifest = read_json(directory / "manifest.json")
            config_digest = sha256_json(design["configs"][variant]["ga_config"])
            expected_effective = _expected_effective_config(
                variant,
                design["configs"][variant],
                expected_digest,
                config_digest,
            )
            if (
                manifest.get("design_digest") != expected_digest
                or manifest.get("config_digest") != config_digest
                or manifest.get("design") != payload
            ):
                raise ValueError(f"Task manifest design mismatch for {variant}/{instance}")
            if read_json(directory / "effective_config.json") != expected_effective:
                raise ValueError(f"Effective config mismatch for {variant}/{instance}")
            keyed = _ok_rows_by_run(
                directory / "raw_results.jsonl",
                variant=variant,
                instance=instance,
                uncertainty=design["uncertainty"],
                legacy=False,
                expected_design_digest=expected_digest,
                expected_config_digest=config_digest,
                require_complete=True,
            )
            rows.extend(keyed[run] for run in sorted(keyed))
        if len(rows) != EXPECTED_INSTANCE_COUNT * EXPECTED_RUNS:
            raise ValueError(f"Expected 300 rows for {variant}, got {len(rows)}")
        rows_by_variant[variant] = rows
    return rows_by_variant


def _validate_existing_new_outputs(
    args: argparse.Namespace,
    design: dict[str, Any],
) -> dict[str, int]:
    """Fail early on incompatible partial A/C outputs without requiring completion."""
    successful_counts = {variant: 0 for variant in NEW_VARIANTS}
    expected_by_directory = {
        instance.removesuffix(".fjs"): instance for instance in design["instances"]
    }
    for variant in NEW_VARIANTS:
        variant_root = args.output_root / variant
        if not variant_root.exists():
            continue
        if not variant_root.is_dir():
            raise ValueError(f"Expected a directory at {variant_root}")
        directories = [path for path in variant_root.iterdir() if path.is_dir()]
        extras = sorted(path.name for path in directories if path.name not in expected_by_directory)
        if extras:
            raise ValueError(f"Unexpected partial-output directories for {variant}: {extras}")
        for directory in directories:
            instance = expected_by_directory[directory.name]
            payload = task_design_payload(args, design, variant, instance)
            digest = design_digest(payload)
            wrapper = design["configs"][variant]
            config_digest = sha256_json(wrapper["ga_config"])
            effective_config = _expected_effective_config(
                variant, wrapper, digest, config_digest
            )
            metadata_exists = _validate_resume_state(
                directory,
                resume=True,
                payload=payload,
                digest=digest,
                config_digest=config_digest,
                effective_config=effective_config,
            )
            raw_path = directory / "raw_results.jsonl"
            if metadata_exists and raw_path.exists():
                rows = _ok_rows_by_run(
                    raw_path,
                    variant=variant,
                    instance=instance,
                    uncertainty=design["uncertainty"],
                    legacy=False,
                    expected_design_digest=digest,
                    expected_config_digest=config_digest,
                    require_complete=False,
                )
                successful_counts[variant] += len(rows)
    return successful_counts


def _key_rows(rows: list[dict[str, Any]], variant: str) -> dict[tuple[str, int], dict[str, Any]]:
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["instance"]), int(row["run"]))
        if key in keyed:
            raise ValueError(f"Duplicate result key for {variant}: {key}")
        keyed[key] = row
    return keyed


def _cap_reached(raw_function_evaluations: int) -> bool:
    return raw_function_evaluations >= (
        EXPECTED_MAX_FUNCTION_EVALUATIONS - EXPECTED_INTERNAL_SIMULATIONS
    )


def _metric_ci(ratios: list[float], seed_offset: int) -> tuple[float, float, float]:
    estimate = geometric_mean(ratios)
    low, high = bootstrap_geomean_ci(
        ratios,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED + seed_offset,
    )
    if estimate is None or low is None or high is None:
        raise ValueError("Could not calculate a comparison metric")
    return estimate, low, high


def summarize(args: argparse.Namespace) -> None:
    design = load_design(args, validate_reference_rows=True)
    reference_digest_before = design["reference_dataset_digest"]
    if not isinstance(design["reference_rows"], dict):
        raise RuntimeError("Reference rows were not loaded")
    new_digest_before = new_tree_digest(args.output_root, design["instances"])
    new_rows = _load_new_rows(args, design)
    new_digest_after_load = new_tree_digest(args.output_root, design["instances"])
    if new_digest_after_load != new_digest_before:
        raise RuntimeError("New A/C dataset changed while it was being validated")
    rows_by_variant = {**new_rows, **design["reference_rows"]}
    rows_by_variant = {variant: rows_by_variant[variant] for variant in ALL_VARIANTS}
    keyed = {
        variant: _key_rows(rows_by_variant[variant], variant)
        for variant in ALL_VARIANTS
    }
    expected_keys = {
        (instance, run)
        for instance in design["instances"]
        for run in range(1, EXPECTED_RUNS + 1)
    }
    for variant in ALL_VARIANTS:
        if set(keyed[variant]) != expected_keys:
            raise ValueError(f"Incomplete or unexpected result keys for {variant}")
    for key in sorted(expected_keys):
        seeds = {int(keyed[variant][key]["seed"]) for variant in ALL_VARIANTS}
        uncertainties = {
            canonical_json(keyed[variant][key]["uncertainty_parameters"])
            for variant in ALL_VARIANTS
        }
        if len(seeds) != 1 or len(uncertainties) != 1:
            raise ValueError(f"Four-cell pairing mismatch for {key}")

    variant_summary: list[dict[str, Any]] = []
    variant_instance_summary: list[dict[str, Any]] = []
    for variant in ALL_VARIANTS:
        rows = rows_by_variant[variant]
        raw_values = [int(row["raw_function_evaluations"]) for row in rows]
        cap_count = sum(_cap_reached(value) for value in raw_values)
        variant_summary.append(
            {
                "variant": variant,
                "source": "new" if variant in NEW_VARIANTS else "read_only_reference",
                "successful_runs": len(rows),
                "expected_runs": EXPECTED_INSTANCE_COUNT * EXPECTED_RUNS,
                "mean_fitness": statistics.mean(float(row["fitness"]) for row in rows),
                "median_fitness": statistics.median(float(row["fitness"]) for row in rows),
                "mean_raw_function_evaluations": statistics.mean(raw_values),
                "mean_best_found_function_evaluations": statistics.mean(
                    int(row["function_evaluations"]) for row in rows
                ),
                "mean_runtime_s": statistics.mean(float(row["runtime_s"]) for row in rows),
                "mean_final_robust_stdev": statistics.mean(
                    float(row["final_robust_stdev"]) for row in rows
                ),
                "mean_final_R": statistics.mean(float(row["final_R"]) for row in rows),
                "fe_cap_saturated_runs": cap_count,
                "fe_cap_saturation_rate": cap_count / len(rows),
                "final_validation_simulations_per_run": EXPECTED_FINAL_SIMULATIONS,
            }
        )
        for instance in design["instances"]:
            instance_rows = [row for row in rows if row["instance"] == instance]
            instance_raw = [int(row["raw_function_evaluations"]) for row in instance_rows]
            variant_instance_summary.append(
                {
                    "variant": variant,
                    "instance": instance,
                    "successful_runs": len(instance_rows),
                    "median_fitness": statistics.median(
                        float(row["fitness"]) for row in instance_rows
                    ),
                    "mean_fitness": statistics.mean(
                        float(row["fitness"]) for row in instance_rows
                    ),
                    "mean_raw_function_evaluations": statistics.mean(instance_raw),
                    "mean_best_found_function_evaluations": statistics.mean(
                        int(row["function_evaluations"]) for row in instance_rows
                    ),
                    "mean_runtime_s": statistics.mean(
                        float(row["runtime_s"]) for row in instance_rows
                    ),
                    "fe_cap_saturated_runs": sum(_cap_reached(value) for value in instance_raw),
                }
            )

    paired_run_rows: list[dict[str, Any]] = []
    instance_comparison_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    instance_metric_maps: dict[str, dict[str, dict[str, float]]] = {}
    for comparison_index, (comparison, baseline, candidate) in enumerate(COMPARISONS):
        metric_maps = {
            "quality": {},
            "raw_fe": {},
            "best_found_fe": {},
            "runtime": {},
            "robust_stdev": {},
        }
        final_r_deltas: list[float] = []
        for instance in design["instances"]:
            keys = [(instance, run) for run in range(1, EXPECTED_RUNS + 1)]
            base_rows = [keyed[baseline][key] for key in keys]
            candidate_rows = [keyed[candidate][key] for key in keys]
            for run, (base, cand) in enumerate(zip(base_rows, candidate_rows), start=1):
                paired_run_rows.append(
                    {
                        "comparison": comparison,
                        "baseline": baseline,
                        "candidate": candidate,
                        "instance": instance,
                        "run": run,
                        "seed": base["seed"],
                        "baseline_fitness": base["fitness"],
                        "candidate_fitness": cand["fitness"],
                        "fitness_ratio": float(cand["fitness"]) / float(base["fitness"]),
                        "baseline_raw_function_evaluations": base["raw_function_evaluations"],
                        "candidate_raw_function_evaluations": cand["raw_function_evaluations"],
                        "raw_function_evaluations_ratio": int(cand["raw_function_evaluations"])
                        / int(base["raw_function_evaluations"]),
                        "baseline_best_found_function_evaluations": base["function_evaluations"],
                        "candidate_best_found_function_evaluations": cand["function_evaluations"],
                        "baseline_runtime_s": base["runtime_s"],
                        "candidate_runtime_s": cand["runtime_s"],
                    }
                )

            base_quality = statistics.median(float(row["fitness"]) for row in base_rows)
            candidate_quality = statistics.median(
                float(row["fitness"]) for row in candidate_rows
            )
            base_raw = statistics.mean(
                int(row["raw_function_evaluations"]) for row in base_rows
            )
            candidate_raw = statistics.mean(
                int(row["raw_function_evaluations"]) for row in candidate_rows
            )
            base_best = statistics.mean(int(row["function_evaluations"]) for row in base_rows)
            candidate_best = statistics.mean(
                int(row["function_evaluations"]) for row in candidate_rows
            )
            base_runtime = statistics.mean(float(row["runtime_s"]) for row in base_rows)
            candidate_runtime = statistics.mean(
                float(row["runtime_s"]) for row in candidate_rows
            )
            base_stdev = statistics.mean(
                float(row["final_robust_stdev"]) for row in base_rows
            )
            candidate_stdev = statistics.mean(
                float(row["final_robust_stdev"]) for row in candidate_rows
            )
            base_final_r = statistics.mean(float(row["final_R"]) for row in base_rows)
            candidate_final_r = statistics.mean(
                float(row["final_R"]) for row in candidate_rows
            )
            instance_ratios = {
                "quality": candidate_quality / base_quality,
                "raw_fe": candidate_raw / base_raw,
                "best_found_fe": candidate_best / base_best,
                "runtime": candidate_runtime / base_runtime,
                "robust_stdev": candidate_stdev / base_stdev,
            }
            for metric, ratio in instance_ratios.items():
                metric_maps[metric][instance] = ratio
            final_r_deltas.append(candidate_final_r - base_final_r)
            instance_comparison_rows.append(
                {
                    "comparison": comparison,
                    "baseline": baseline,
                    "candidate": candidate,
                    "instance": instance,
                    "paired_runs": EXPECTED_RUNS,
                    "baseline_median_fitness": base_quality,
                    "candidate_median_fitness": candidate_quality,
                    "quality_ratio": instance_ratios["quality"],
                    "baseline_mean_raw_function_evaluations": base_raw,
                    "candidate_mean_raw_function_evaluations": candidate_raw,
                    "raw_fe_ratio": instance_ratios["raw_fe"],
                    "baseline_mean_best_found_function_evaluations": base_best,
                    "candidate_mean_best_found_function_evaluations": candidate_best,
                    "best_found_fe_ratio": instance_ratios["best_found_fe"],
                    "runtime_ratio": instance_ratios["runtime"],
                    "final_robust_stdev_ratio": instance_ratios["robust_stdev"],
                    "final_R_delta": candidate_final_r - base_final_r,
                    "quality_outcome": "win"
                    if instance_ratios["quality"] < 1
                    else "loss"
                    if instance_ratios["quality"] > 1
                    else "tie",
                    "raw_fe_outcome": "win"
                    if instance_ratios["raw_fe"] < 1
                    else "loss"
                    if instance_ratios["raw_fe"] > 1
                    else "tie",
                }
            )

        instance_metric_maps[comparison] = metric_maps
        metrics: dict[str, tuple[float, float, float]] = {}
        for metric_index, (metric, values_by_instance) in enumerate(metric_maps.items()):
            metrics[metric] = _metric_ci(
                list(values_by_instance.values()),
                seed_offset=comparison_index * 10 + metric_index,
            )
        quality = metrics["quality"]
        raw_fe = metrics["raw_fe"]
        comparison_rows.append(
            {
                "comparison": comparison,
                "baseline": baseline,
                "candidate": candidate,
                "paired_runs": EXPECTED_INSTANCE_COUNT * EXPECTED_RUNS,
                "paired_instances": EXPECTED_INSTANCE_COUNT,
                "quality_geometric_mean_ratio": quality[0],
                "quality_improvement_percent": 100.0 * (1.0 - quality[0]),
                "quality_ci_low": quality[1],
                "quality_ci_high": quality[2],
                "quality_superior": quality[2] < 1.0,
                "quality_noninferior_2pct": quality[2]
                < 1.0 + QUALITY_NONINFERIORITY_MARGIN,
                "raw_fe_geometric_mean_ratio": raw_fe[0],
                "raw_fe_reduction_percent": 100.0 * (1.0 - raw_fe[0]),
                "raw_fe_ci_low": raw_fe[1],
                "raw_fe_ci_high": raw_fe[2],
                "raw_fe_reduced": raw_fe[2] < 1.0,
                "best_found_fe_geometric_mean_ratio": metrics["best_found_fe"][0],
                "best_found_fe_ci_low": metrics["best_found_fe"][1],
                "best_found_fe_ci_high": metrics["best_found_fe"][2],
                "runtime_geometric_mean_ratio": metrics["runtime"][0],
                "runtime_ci_low": metrics["runtime"][1],
                "runtime_ci_high": metrics["runtime"][2],
                "final_robust_stdev_geometric_mean_ratio": metrics["robust_stdev"][0],
                "final_robust_stdev_ci_low": metrics["robust_stdev"][1],
                "final_robust_stdev_ci_high": metrics["robust_stdev"][2],
                "mean_final_R_delta": statistics.mean(final_r_deltas),
                "quality_wins": sum(value < 1 for value in metric_maps["quality"].values()),
                "quality_ties": sum(value == 1 for value in metric_maps["quality"].values()),
                "quality_losses": sum(value > 1 for value in metric_maps["quality"].values()),
                "raw_fe_wins": sum(value < 1 for value in metric_maps["raw_fe"].values()),
                "raw_fe_ties": sum(value == 1 for value in metric_maps["raw_fe"].values()),
                "raw_fe_losses": sum(value > 1 for value in metric_maps["raw_fe"].values()),
            }
        )

    comparison_map = {row["comparison"]: row for row in comparison_rows}
    hypotheses = evaluate_hypotheses(comparison_map)
    hardware_provenance = _hardware_provenance(rows_by_variant)
    rl_without = instance_metric_maps["rl_effect_without_pipeline"]
    rl_with = instance_metric_maps["rl_effect_with_pipeline"]
    interaction: dict[str, Any] = {
        "definition": "(hpo_with_rl/hpo_no_rl)/(hpo_rl_only/hpo_plain_ga)",
        "interpretation": "exploratory; ratios below 1 favour positive component interaction",
        "paired_instances": EXPECTED_INSTANCE_COUNT,
    }
    for metric_index, metric in enumerate(("quality", "raw_fe")):
        ratios = [
            rl_with[metric][instance] / rl_without[metric][instance]
            for instance in design["instances"]
        ]
        estimate, low, high = _metric_ci(ratios, seed_offset=100 + metric_index)
        interaction[f"{metric}_ratio_of_ratios"] = estimate
        interaction[f"{metric}_ci_low"] = low
        interaction[f"{metric}_ci_high"] = high

    if reference_tree_digest(args.reference_root, design["instances"]) != reference_digest_before:
        raise RuntimeError("Read-only reference dataset changed before summary output")
    if new_tree_digest(args.output_root, design["instances"]) != new_digest_before:
        raise RuntimeError("New A/C dataset changed before summary output")

    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output_root / "variant_summary.csv", list(variant_summary[0]), variant_summary)
    atomic_csv(
        args.output_root / "variant_instance_summary.csv",
        list(variant_instance_summary[0]),
        variant_instance_summary,
    )
    atomic_csv(
        args.output_root / "paired_run_comparison.csv",
        list(paired_run_rows[0]),
        paired_run_rows,
    )
    atomic_csv(
        args.output_root / "instance_comparison.csv",
        list(instance_comparison_rows[0]),
        instance_comparison_rows,
    )
    atomic_csv(
        args.output_root / "comparison_summary.csv",
        list(comparison_rows[0]),
        comparison_rows,
    )
    report = {
        "experiment": EXPERIMENT,
        "evidence_scope": "retrospective_component_ablation",
        "limitations": [
            (
                "The HPO and RL configurations were selected on overlapping Scenario-2 "
                "instances; decision flags are in-sample ablation criteria, not independent "
                "confirmatory evidence."
            ),
            (
                "The 36-hour-or-5M-FE stopping rule is hardware-dependent. Legacy B/D "
                "hardware was not recorded, so raw-FE differences cannot be attributed "
                "causally to components without that qualification."
            ),
            (
                "Legacy runs contain no anytime checkpoints; endpoint raw-FE can be "
                "uninformative when the common 5M cap is saturated."
            ),
        ],
        "primary_unit": "instance",
        "quality_metric": (
            "geometric mean across instances of candidate/baseline median final robust makespan"
        ),
        "evaluation_metric": (
            "geometric mean across instances of candidate/baseline mean raw_function_evaluations"
        ),
        "raw_fe_definition": (
            "individual stochastic simulation replications during search; do not multiply by 12"
        ),
        "final_validation_simulations": {
            "per_run": EXPECTED_FINAL_SIMULATIONS,
            "included_in_raw_function_evaluations": False,
        },
        "quality_noninferiority_margin": QUALITY_NONINFERIORITY_MARGIN,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "comparisons": comparison_rows,
        "hypotheses": hypotheses,
        "interaction": interaction,
        "hardware_provenance": hardware_provenance,
        "fe_limit_saturation": {
            "threshold": EXPECTED_MAX_FUNCTION_EVALUATIONS
            - EXPECTED_INTERNAL_SIMULATIONS,
            "warning": (
                "Endpoint raw-FE ratios can be uninformative when many runs saturate the common cap."
            ),
        },
        "variant_summary": variant_summary,
    }
    atomic_json(args.output_root / "comparison_report.json", report)
    atomic_json(
        args.output_root / "experiment_manifest.json",
        {
            "experiment": EXPERIMENT,
            "schema_version": DESIGN_SCHEMA_VERSION,
            "variants": list(ALL_VARIANTS),
            "new_variants": list(NEW_VARIANTS),
            "reference_variants": list(REFERENCE_VARIANTS),
            "comparisons": [list(item) for item in COMPARISONS],
            "output_root": str(args.output_root),
            "reference_root": str(args.reference_root),
            "reference_access": "read_only",
            "reference_dataset_digest": reference_digest_before,
            "new_dataset_digest": new_digest_before,
            "protocol": design["protocol"],
            "evidence_scope": "retrospective_component_ablation",
            "quality_noninferiority_margin": QUALITY_NONINFERIORITY_MARGIN,
            "artifact_configs": design["configs"],
            "pinned_hashes": design["pinned_hashes"],
            "source_file_hashes": design["source_file_hashes"],
            "environment_versions": design["environment_versions"],
            "hardware_provenance": hardware_provenance,
            "git_at_summary": git_provenance(),
        },
    )

    reference_digest_after = reference_tree_digest(
        args.reference_root, design["instances"]
    )
    if reference_digest_after != reference_digest_before:
        raise RuntimeError("Read-only reference dataset changed while summarizing")
    if new_tree_digest(args.output_root, design["instances"]) != new_digest_before:
        raise RuntimeError("New A/C dataset changed while summarizing")
    print(f"Wrote component-factorial outputs to {args.output_root}")


def preflight(args: argparse.Namespace) -> None:
    design = load_design(args, validate_reference_rows=True)
    reference_rows = design["reference_rows"]
    if not isinstance(reference_rows, dict):
        raise RuntimeError("Reference validation did not load rows")
    counts = {variant: len(reference_rows[variant]) for variant in REFERENCE_VARIANTS}
    partial_counts = _validate_existing_new_outputs(args, design)
    print(
        "Preflight OK: "
        f"reference_counts={counts} new_tasks={len(task_specs(args.instances_dir))} "
        f"existing_new_successes={partial_counts} "
        f"reference_digest={design['reference_dataset_digest']}"
    )


def print_tasks(args: argparse.Namespace) -> None:
    for index, (variant, instance) in enumerate(task_specs(args.instances_dir)):
        print(f"{index}\t{variant}\t{instance}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "print-tasks", "run-task", "summarize"),
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--instances-dir", type=Path, default=Path("instances/fjssp-w"))
    parser.add_argument(
        "--uncertainty-json",
        type=Path,
        default=Path("config/scenario2_uncertainty.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/hpo_component_factorial_scenario2"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("results/hpo_rl_factorial_scenario2"),
        help="Read-only root containing completed hpo_no_rl and hpo_with_rl runs.",
    )
    parser.add_argument(
        "--hpo-config",
        type=Path,
        default=Path(
            "results/hpo_scenario2/final/results/"
            f"{PINNED_HPO_ID}/config.json"
        ),
    )
    parser.add_argument(
        "--rl-config",
        type=Path,
        default=Path(f"results/hpo_rl_scenario2/results/{PINNED_RL_ID}/config.json"),
    )
    args = parser.parse_args()
    if args.command != "run-task" and args.resume:
        parser.error("--resume is only valid with run-task")
    return args


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        preflight(args)
    elif args.command == "print-tasks":
        print_tasks(args)
    elif args.command == "run-task":
        run_task(args)
    elif args.command == "summarize":
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
