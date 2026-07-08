#!/usr/bin/env python3
"""Literature-informed HPO runner for Scenario 2 of FJSSP-W.


The script implements the MA-friendly search plan from the research note,
without the optional RL mutation controller:

1. screening: C0-C3 + random configs
2. tpe: Optuna/TPE configs guided by screening results
3. race1: top 15 configs with larger budget
4. race2: top 8 configs with larger budget
5. final: top 5 configs on holdout instances

It is designed for Slurm job arrays: one array task evaluates one full
configuration on the phase-specific instance/seed set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.GA.parallel_simulation import run_n_simulations_parallel
from solver.GA.wfjssp_ga import build_ga_from_worker_encoding, is_simulatable_schedule
from util.benchmark_parser import WorkerBenchmarkParser
from util.evaluation import makespan, translate
from util.hyperparameters import write_hyperparameters_txt
from scripts.run_scenario2_submission import GA_CONFIG, UNCERTAINTY_SOURCE, load_uncertainty, uncertainty_for


# ---------------------------------------------------------------------------
# Search plan
# ---------------------------------------------------------------------------

PHASES: dict[str, dict[str, Any]] = {
    "screening": {
        "n_configs": 48,
        "source_phase": None,
        "instance_pool": "train",
        "instance_count": 8,
        "run_count": 2,
        "max_function_evaluations": 250_000,
        "final_simulations": 20,
        "time_limit_s": None,
    },
    "tpe": {
        "n_configs": 72,
        "source_phase": "screening",
        "instance_pool": "train",
        "instance_count": 12,
        "run_count": 3,
        "max_function_evaluations": 750_000,
        "final_simulations": 20,
        "time_limit_s": None,
    },
    "race1": {
        "n_configs": 15,
        "source_phase": "tpe",
        "instance_pool": "train",
        "instance_count": 18,
        "run_count": 4,
        "max_function_evaluations": 1_500_000,
        "final_simulations": 50,
        "time_limit_s": None,
    },
    "race2": {
        "n_configs": 8,
        "source_phase": "race1",
        "instance_pool": "train_val",
        "instance_count": 24,
        "run_count": 6,
        "max_function_evaluations": 3_000_000,
        "final_simulations": 50,
        "time_limit_s": None,
    },
    "final": {
        "n_configs": 5,
        "source_phase": "race2",
        "instance_pool": "holdout",
        "instance_count": 6,
        "run_count": 10,
        "max_function_evaluations": 5_000_000,
        "final_simulations": 50,
        "time_limit_s": None,
    },
}

CHOICES: dict[str, list[Any]] = {
    "ga.population_size": [160, 200, 240, 280, 360],
    "ga.offspring_ratio": [4, 5, 6],
    "ga.elitism_rate": [0.05, 0.08, 0.10],
    "ga.restart_generations": [400, 800, 1200],
    "stoch.internal_simulations": [8, 10, 12, 16],
    "sur.warmup_real_candidates": [600, 800, 1000, 1200],
    "sur.n_estimators": [200, 300, 400, 500],
    "sur.min_samples_leaf": [2, 3, 5, 7],
    "sur.retrain_interval_real_candidates": [50, 75, 100, 150],
    "sur.top_fraction": [0.01, 0.015, 0.02, 0.03],
    "sur.uncertain_fraction": [0.0025, 0.005, 0.0075, 0.01],
    "sur.random_fraction": [0.0025, 0.005, 0.0075, 0.01],
    "sur.min_real_per_generation": [4, 5, 6, 8],
    "sur.max_training_samples": [3000, 5000, 8000],
    "ls.use": [True, False],
    "ls.interval": [10, 15, 20, 30],
    "ls.origin_count": [2, 3, 4],
    "ls.neighbors_per_origin": [128, 200, 256],
    "ls.top_k": [4, 6, 8, 10],
    "ls.uncertain_k": [2, 4, 6],
    "ls.random_k": [1, 2, 3, 4],
    "ls.real_eval_limit_per_origin": [8, 10, 12, 16],
    "ls.min_predicted_improvement": [0.0, 1.0, 2.0, 5.0],
}

START_PARAMS: list[dict[str, Any]] = [
    {
        "config_id": "C0_repo_reproduction",
        "source": "repo baseline",
        "params": {
            "ga.population_size": 200,
            "ga.offspring_ratio": 5,
            "ga.elitism_rate": 0.10,
            "ga.restart_generations": 800,
            "stoch.internal_simulations": 10,
            "sur.warmup_real_candidates": 1000,
            "sur.n_estimators": 300,
            "sur.min_samples_leaf": 3,
            "sur.retrain_interval_real_candidates": 100,
            "sur.top_fraction": 0.020,
            "sur.uncertain_fraction": 0.005,
            "sur.random_fraction": 0.005,
            "sur.min_real_per_generation": 5,
            "sur.max_training_samples": 5000,
            "ls.use": True,
            "ls.interval": 20,
            "ls.origin_count": 3,
            "ls.neighbors_per_origin": 200,
            "ls.top_k": 8,
            "ls.uncertain_k": 4,
            "ls.random_k": 3,
            "ls.real_eval_limit_per_origin": 12,
            "ls.min_predicted_improvement": 5.0,
        },
    },
    {
        "config_id": "C1_sparse_literature_start",
        "source": "literature-informed sparse start",
        "params": {
            "ga.population_size": 160,
            "ga.offspring_ratio": 4,
            "ga.elitism_rate": 0.08,
            "ga.restart_generations": 400,
            "stoch.internal_simulations": 8,
            "sur.warmup_real_candidates": 600,
            "sur.n_estimators": 300,
            "sur.min_samples_leaf": 5,
            "sur.retrain_interval_real_candidates": 75,
            "sur.top_fraction": 0.020,
            "sur.uncertain_fraction": 0.0075,
            "sur.random_fraction": 0.0075,
            "sur.min_real_per_generation": 5,
            "sur.max_training_samples": 3000,
            "ls.use": True,
            "ls.interval": 15,
            "ls.origin_count": 3,
            "ls.neighbors_per_origin": 128,
            "ls.top_k": 6,
            "ls.uncertain_k": 4,
            "ls.random_k": 2,
            "ls.real_eval_limit_per_origin": 10,
            "ls.min_predicted_improvement": 2.0,
        },
    },
    {
        "config_id": "C2_stronger_memetic_search",
        "source": "stronger local-search hypothesis",
        "params": {
            "ga.population_size": 280,
            "ga.offspring_ratio": 4,
            "ga.elitism_rate": 0.08,
            "ga.restart_generations": 800,
            "stoch.internal_simulations": 10,
            "sur.warmup_real_candidates": 800,
            "sur.n_estimators": 400,
            "sur.min_samples_leaf": 5,
            "sur.retrain_interval_real_candidates": 100,
            "sur.top_fraction": 0.015,
            "sur.uncertain_fraction": 0.005,
            "sur.random_fraction": 0.005,
            "sur.min_real_per_generation": 6,
            "sur.max_training_samples": 5000,
            "ls.use": True,
            "ls.interval": 10,
            "ls.origin_count": 4,
            "ls.neighbors_per_origin": 256,
            "ls.top_k": 10,
            "ls.uncertain_k": 6,
            "ls.random_k": 2,
            "ls.real_eval_limit_per_origin": 12,
            "ls.min_predicted_improvement": 1.0,
        },
    },
    {
        "config_id": "C3_conservative_surrogate",
        "source": "conservative surrogate hypothesis",
        "params": {
            "ga.population_size": 240,
            "ga.offspring_ratio": 5,
            "ga.elitism_rate": 0.05,
            "ga.restart_generations": 1200,
            "stoch.internal_simulations": 12,
            "sur.warmup_real_candidates": 1200,
            "sur.n_estimators": 500,
            "sur.min_samples_leaf": 7,
            "sur.retrain_interval_real_candidates": 150,
            "sur.top_fraction": 0.010,
            "sur.uncertain_fraction": 0.010,
            "sur.random_fraction": 0.010,
            "sur.min_real_per_generation": 4,
            "sur.max_training_samples": 8000,
            "ls.use": True,
            "ls.interval": 20,
            "ls.origin_count": 2,
            "ls.neighbors_per_origin": 128,
            "ls.top_k": 4,
            "ls.uncertain_k": 4,
            "ls.random_k": 4,
            "ls.real_eval_limit_per_origin": 8,
            "ls.min_predicted_improvement": 5.0,
        },
    },
]


# ---------------------------------------------------------------------------
# Config conversion and generation
# ---------------------------------------------------------------------------

def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def short_hash(value: Any) -> str:
    return hashlib.sha1(canonical_json(value).encode("utf-8")).hexdigest()[:10]


def params_to_ga_config(params: dict[str, Any]) -> dict[str, Any]:
    pop = int(params["ga.population_size"])
    offspring_ratio = int(params["ga.offspring_ratio"])
    use_ls = bool(params.get("ls.use", True))

    cfg = {
        "population_size": pop,
        "offspring_amount": pop * offspring_ratio,
        "elitism_rate": float(params["ga.elitism_rate"]),
        "restart_generations": int(params["ga.restart_generations"]),
        "use_surrogate_evaluation": True,
        "surrogate_warmup_real_candidates": int(params["sur.warmup_real_candidates"]),
        "surrogate_top_fraction": float(params["sur.top_fraction"]),
        "surrogate_uncertain_fraction": float(params["sur.uncertain_fraction"]),
        "surrogate_random_fraction": float(params["sur.random_fraction"]),
        "surrogate_min_real_per_generation": int(params["sur.min_real_per_generation"]),
        "surrogate_retrain_interval_real_candidates": int(params["sur.retrain_interval_real_candidates"]),
        "surrogate_n_estimators": int(params["sur.n_estimators"]),
        "surrogate_min_samples_leaf": int(params["sur.min_samples_leaf"]),
        "surrogate_max_features": "sqrt",
        "surrogate_max_training_samples": int(params["sur.max_training_samples"]),
        "enable_rl_mutation_control": False,
        # stable current defaults; not part of the first search wave
        "surrogate_retrain_interval_growth_samples": 5_000,
        "surrogate_retrain_interval_growth_factor": 2.0,
        "surrogate_max_retrain_interval_real_candidates": 1_000,
        "surrogate_n_jobs": -1,
    }

    if use_ls:
        cfg.update(
            {
                "local_search_interval": int(params["ls.interval"]),
                "local_search_origin_count": int(params["ls.origin_count"]),
                "local_search_neighbors_per_origin": int(params["ls.neighbors_per_origin"]),
                "local_search_top_k": int(params["ls.top_k"]),
                "local_search_uncertain_k": int(params["ls.uncertain_k"]),
                "local_search_random_k": int(params["ls.random_k"]),
                "local_search_real_eval_limit_per_origin": int(params["ls.real_eval_limit_per_origin"]),
                "local_search_min_predicted_improvement": float(params["ls.min_predicted_improvement"]),
            }
        )
    else:
        cfg.update(
            {
                "local_search_interval": 0,
                "local_search_origin_count": 0,
                "local_search_neighbors_per_origin": 0,
                "local_search_top_k": 0,
                "local_search_uncertain_k": 0,
                "local_search_random_k": 0,
                "local_search_real_eval_limit_per_origin": 0,
                "local_search_min_predicted_improvement": 0.0,
            }
        )

    return cfg


def make_config(config_id: str, params: dict[str, Any], source: str, parent: str | None = None) -> dict[str, Any]:
    return {
        "config_id": config_id,
        "source": source,
        "parent_config_id": parent,
        "params": dict(params),
        "ga_config": params_to_ga_config(params),
        "internal_simulations": int(params["stoch.internal_simulations"]),
        "rl_note": "RL mutation control is intentionally disabled in this search.",
    }


def sample_random_params(rng: random.Random) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in [
        "ga.population_size",
        "ga.offspring_ratio",
        "ga.elitism_rate",
        "ga.restart_generations",
        "stoch.internal_simulations",
        "sur.warmup_real_candidates",
        "sur.n_estimators",
        "sur.min_samples_leaf",
        "sur.retrain_interval_real_candidates",
        "sur.top_fraction",
        "sur.uncertain_fraction",
        "sur.random_fraction",
        "sur.min_real_per_generation",
        "sur.max_training_samples",
        "ls.use",
    ]:
        params[key] = rng.choice(CHOICES[key])

    real_eval_fraction = (
        float(params["sur.top_fraction"])
        + float(params["sur.uncertain_fraction"])
        + float(params["sur.random_fraction"])
    )
    # Keep real simulation pressure bounded. Try again in caller if violated.
    if real_eval_fraction > 0.05:
        raise ValueError("surrogate real-evaluation fraction too high")

    if params["ls.use"]:
        for key in [
            "ls.interval",
            "ls.origin_count",
            "ls.neighbors_per_origin",
            "ls.top_k",
            "ls.uncertain_k",
            "ls.random_k",
            "ls.real_eval_limit_per_origin",
            "ls.min_predicted_improvement",
        ]:
            params[key] = rng.choice(CHOICES[key])
    return params


def suggest_optuna_params(trial: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in [
        "ga.population_size",
        "ga.offspring_ratio",
        "ga.elitism_rate",
        "ga.restart_generations",
        "stoch.internal_simulations",
        "sur.warmup_real_candidates",
        "sur.n_estimators",
        "sur.min_samples_leaf",
        "sur.retrain_interval_real_candidates",
        "sur.top_fraction",
        "sur.uncertain_fraction",
        "sur.random_fraction",
        "sur.min_real_per_generation",
        "sur.max_training_samples",
        "ls.use",
    ]:
        params[key] = trial.suggest_categorical(key, CHOICES[key])

    real_eval_fraction = (
        float(params["sur.top_fraction"])
        + float(params["sur.uncertain_fraction"])
        + float(params["sur.random_fraction"])
    )
    if real_eval_fraction > 0.05:
        # This should be rare with the defined choices, but keeps the rule explicit.
        raise RuntimeError("invalid surrogate real-evaluation fraction")

    if params["ls.use"]:
        for key in [
            "ls.interval",
            "ls.origin_count",
            "ls.neighbors_per_origin",
            "ls.top_k",
            "ls.uncertain_k",
            "ls.random_k",
            "ls.real_eval_limit_per_origin",
            "ls.min_predicted_improvement",
        ]:
            params[key] = trial.suggest_categorical(key, CHOICES[key])
    return params


# ---------------------------------------------------------------------------
# Instance split
# ---------------------------------------------------------------------------

def worker_count_from_name(name: str) -> int:
    try:
        return int(name.split("_workers")[0].split("_")[-1])
    except Exception:
        return 0


def family_key(name: str) -> str:
    stem = name.replace(".fjs", "")
    parts = stem.split("_")
    if parts and parts[0].startswith("2") and len(parts) >= 3:
        return "_".join(parts[:3])
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return parts[0]


def stratified_split(instance_names: list[str]) -> dict[str, list[str]]:
    """Deterministic 18/6/6 split, roughly stratified by family and size."""
    groups: dict[str, list[str]] = {}
    for name in sorted(instance_names):
        groups.setdefault(family_key(name), []).append(name)
    for names in groups.values():
        names.sort(key=lambda n: (worker_count_from_name(n), n))

    train: list[str] = []
    val: list[str] = []
    holdout: list[str] = []
    pattern = ["train", "train", "train", "val", "holdout"]
    counters = {"train": 0, "val": 0, "holdout": 0}
    limits = {"train": 18, "val": 6, "holdout": 6}

    ordered: list[str] = []
    for fam in sorted(groups):
        ordered.extend(groups[fam])

    for idx, name in enumerate(ordered):
        preferred = pattern[idx % len(pattern)]
        candidates = [preferred, "train", "val", "holdout"]
        for bucket in candidates:
            if counters[bucket] < limits[bucket]:
                if bucket == "train":
                    train.append(name)
                elif bucket == "val":
                    val.append(name)
                else:
                    holdout.append(name)
                counters[bucket] += 1
                break
    return {"train": train, "val": val, "holdout": holdout, "train_val": train + val}


def load_instance_names(instances_dir: Path) -> list[str]:
    names = sorted(p.name for p in instances_dir.glob("*.fjs"))
    if not names:
        raise FileNotFoundError(f"No .fjs files found in {instances_dir}")
    return names


def phase_instance_names(phase: str, instances_dir: Path) -> list[str]:
    spec = PHASES[phase]
    split = stratified_split(load_instance_names(instances_dir))
    pool = list(split[str(spec["instance_pool"])])
    count = int(spec["instance_count"])
    return pool[:count]


# ---------------------------------------------------------------------------
# Plan IO
# ---------------------------------------------------------------------------

def phase_dir(hpo_root: Path, phase: str) -> Path:
    return hpo_root / phase


def configs_path(hpo_root: Path, phase: str) -> Path:
    return phase_dir(hpo_root, phase) / "configs.jsonl"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(canonical_json(row) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_configs(hpo_root: Path, phase: str) -> list[dict[str, Any]]:
    path = configs_path(hpo_root, phase)
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"No configs found at {path}. Run make-plan first.")
    return rows


def make_screening_plan(hpo_root: Path, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in START_PARAMS:
        cfg = make_config(item["config_id"], item["params"], item["source"])
        configs.append(cfg)
        seen.add(canonical_json(cfg["params"]))

    target = int(PHASES["screening"]["n_configs"])
    attempts = 0
    while len(configs) < target:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("Could not generate enough unique random configs")
        try:
            params = sample_random_params(rng)
        except ValueError:
            continue
        key = canonical_json(params)
        if key in seen:
            continue
        seen.add(key)
        config_id = f"R{len(configs) - len(START_PARAMS) + 1:04d}_{short_hash(params)}"
        configs.append(make_config(config_id, params, "random screening"))
    return configs


def load_summary_rows(hpo_root: Path, phase: str) -> list[dict[str, Any]]:
    path = phase_dir(hpo_root, phase) / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run summarize for phase {phase} first.")
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def configs_by_id(hpo_root: Path, phase: str) -> dict[str, dict[str, Any]]:
    return {cfg["config_id"]: cfg for cfg in read_configs(hpo_root, phase)}


def make_promoted_plan(hpo_root: Path, phase: str) -> list[dict[str, Any]]:
    spec = PHASES[phase]
    source_phase = str(spec["source_phase"])
    rows = load_summary_rows(hpo_root, source_phase)
    src_configs = configs_by_id(hpo_root, source_phase)
    selected = [row for row in rows if row.get("status") == "ok"][: int(spec["n_configs"])]
    if not selected:
        raise RuntimeError(f"No successful configs in {source_phase}/summary.csv")

    out: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        parent_id = row["config_id"]
        parent = src_configs[parent_id]
        new_id = f"{phase}_rank{rank:02d}_{parent_id}"
        out.append(make_config(new_id, parent["params"], f"promoted from {source_phase}", parent=parent_id))
    return out


def make_tpe_plan(hpo_root: Path, seed: int) -> list[dict[str, Any]]:
    try:
        import optuna
        from optuna.distributions import CategoricalDistribution
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required for phase 'tpe'. Install with: pip install -r requirements-hpo.txt"
        ) from exc

    source_phase = "screening"
    rows = load_summary_rows(hpo_root, source_phase)
    src_configs = configs_by_id(hpo_root, source_phase)

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=8, multivariate=True, group=True, constant_liar=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    for row in rows:
        if row.get("status") != "ok":
            continue
        cfg = src_configs.get(row["config_id"])
        if cfg is None:
            continue
        params = cfg["params"]
        distributions = {key: CategoricalDistribution(CHOICES[key]) for key in params.keys()}
        trial = optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            value=float(row["score"]),
            user_attrs={"source_config_id": row["config_id"]},
        )
        study.add_trial(trial)

    configs: list[dict[str, Any]] = []
    seen = {canonical_json(cfg["params"]) for cfg in src_configs.values()}
    target = int(PHASES["tpe"]["n_configs"])
    attempts = 0
    while len(configs) < target:
        attempts += 1
        if attempts > target * 50:
            raise RuntimeError("TPE generated too many duplicate configs")
        trial = study.ask()
        try:
            params = suggest_optuna_params(trial)
        except RuntimeError:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        key = canonical_json(params)
        if key in seen:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        seen.add(key)
        config_id = f"TPE{len(configs) + 1:04d}_{short_hash(params)}"
        configs.append(make_config(config_id, params, "TPE guided by screening summary"))
        # Keep the trial as RUNNING while asking more configs. constant_liar then discourages duplicates.
    return configs


def make_plan(args: argparse.Namespace) -> None:
    hpo_root = args.hpo_root.resolve()
    phase = args.phase
    if phase == "screening":
        configs = make_screening_plan(hpo_root, seed=args.seed)
    elif phase == "tpe":
        configs = make_tpe_plan(hpo_root, seed=args.seed)
    elif phase in {"race1", "race2", "final"}:
        configs = make_promoted_plan(hpo_root, phase)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    path = configs_path(hpo_root, phase)
    write_jsonl(path, configs)
    (phase_dir(hpo_root, phase) / "phase_spec.json").write_text(
        json.dumps(PHASES[phase], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(configs)} configs to {path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(to_builtin(row), sort_keys=True, allow_nan=False) + "\n")
        fh.flush()


def load_completed_ok(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                completed[(str(row["instance"]), int(row["run"]))] = row
    return completed


def solve_hpo_run_task(task: dict[str, Any]) -> dict[str, Any]:
    parser = WorkerBenchmarkParser()
    encoding = parser.parse_benchmark(str(task["instance_path"]))
    instance_name = str(task["instance"])
    run = int(task["run"])
    seed = int(task["seed"])
    uncertainty_parameters = task["uncertainty_parameters"]
    start_wall = time.time()

    ga_kwargs = dict(GA_CONFIG)
    ga_kwargs.update(task["ga_config"])
    ga_kwargs["enable_rl_mutation_control"] = False
    ga_kwargs["seed"] = seed
    ga_kwargs["rl_seed"] = seed
    ga_kwargs["use_stochastic_evaluation"] = True
    ga_kwargs["n_simulations"] = int(task["internal_simulations"])
    ga_kwargs["simulation_workers"] = int(task["simulation_workers"])
    ga_kwargs["surrogate_n_jobs"] = int(task["surrogate_n_jobs"])
    ga_kwargs["uncertainty_parameters"] = uncertainty_parameters

    ga = build_ga_from_worker_encoding(encoding, **ga_kwargs)
    result = ga.run(
        max_generations=None,
        time_limit_s=task.get("time_limit_s"),
        max_function_evaluations=int(task["max_function_evaluations"]),
        progress_interval_evaluations=int(task.get("progress_interval_evaluations", 100_000)),
        keep_multiple=False,
        do_restart=False,
    )
    best = result["best"]

    start_times, machines, workers = translate(
        best.sequence,
        best.assignments,
        best.workers,
        encoding.durations(),
    )
    start_times = [int(x) if float(x).is_integer() else float(x) for x in start_times]
    machines = [int(x) for x in machines]
    workers = [int(x) for x in workers]
    end_times = [
        start_times[i] + encoding.durations()[i][machines[i]][workers[i]]
        for i in range(len(start_times))
    ]
    end_times = [float(x) for x in end_times]

    if not is_simulatable_schedule(start_times, end_times, machines, workers, encoding.job_sequence()):
        raise ValueError("Best decoded schedule is not simulatable")

    try:
        final_results, robust_makespan, robust_makespan_stdev, final_r = run_n_simulations_parallel(
            start_times,
            end_times,
            machines,
            workers,
            encoding.job_sequence(),
            encoding.durations(),
            uncertainty_parameters,
            int(task["final_simulations"]),
            uncertainty_source=UNCERTAINTY_SOURCE,
            processing_times=True,
            workers=int(task["simulation_workers"]),
            seed=seed + 2_000_000_000,
        )
    except TypeError:
        final_results, robust_makespan, robust_makespan_stdev, final_r = run_n_simulations_parallel(
            start_times,
            end_times,
            machines,
            workers,
            encoding.job_sequence(),
            encoding.durations(),
            uncertainty_parameters,
            int(task["final_simulations"]),
            processing_times=True,
            workers=int(task["simulation_workers"]),
            seed=seed + 2_000_000_000,
        )

    raw_function_evaluations = int(result["function_evaluations"])
    function_evaluations = int(result.get("best_found_function_evaluations", raw_function_evaluations))
    deterministic_makespan = float(makespan(start_times, machines, workers, encoding.durations()))

    return {
        "phase": task["phase"],
        "config_id": task["config_id"],
        "instance": instance_name,
        "run": run,
        "seed": seed,
        "status": "ok",
        "fitness": float(robust_makespan),
        "final_robust_makespan": float(robust_makespan),
        "final_robust_stdev": float(robust_makespan_stdev),
        "final_R": float(final_r),
        "deterministic_makespan": deterministic_makespan,
        "function_evaluations": function_evaluations,
        "raw_function_evaluations": raw_function_evaluations,
        "runtime_s": float(result.get("runtime_s", time.time() - start_wall)),
        "generations": int(result["generations"]),
        "surrogate_samples": int(result.get("surrogate_samples", 0)),
        "surrogate_fit_count": int(result.get("surrogate_fit_count", 0)),
        "surrogate_predictions": int(result.get("surrogate_predictions", 0)),
        "surrogate_real_candidate_evaluations": int(result.get("surrogate_real_candidate_evaluations", 0)),
        "start_times": start_times,
        "machine_assignments": machines,
        "worker_assignments": workers,
        "uncertainty_parameters": uncertainty_parameters,
        "final_simulation_results": [float(x) for x in final_results],
    }


def run_task(args: argparse.Namespace) -> None:
    hpo_root = args.hpo_root.resolve()
    phase = args.phase
    configs = read_configs(hpo_root, phase)
    if args.task_index < 0 or args.task_index >= len(configs):
        raise IndexError(f"task-index {args.task_index} outside 0..{len(configs)-1}")
    config = configs[args.task_index]
    phase_spec = dict(PHASES[phase])
    if args.max_function_evaluations is not None:
        phase_spec["max_function_evaluations"] = int(args.max_function_evaluations)
    if args.final_simulations is not None:
        phase_spec["final_simulations"] = int(args.final_simulations)
    if args.time_limit_s is not None:
        phase_spec["time_limit_s"] = int(args.time_limit_s)

    out_dir = phase_dir(hpo_root, phase) / "results" / config["config_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    instance_names = phase_instance_names(phase, args.instances_dir)
    uncertainty_payload = load_uncertainty(args.uncertainty_json)
    instance_by_name = {p.name: p for p in args.instances_dir.glob("*.fjs")}
    raw_path = out_dir / "raw_results.jsonl"
    completed = load_completed_ok(raw_path) if args.resume else {}

    tasks: list[dict[str, Any]] = []
    for instance_name in instance_names:
        for run in range(1, int(phase_spec["run_count"]) + 1):
            if (instance_name, run) in completed:
                continue
            seed, uncertainty_parameters = uncertainty_for(uncertainty_payload, instance_name, run)
            tasks.append(
                {
                    "phase": phase,
                    "config_id": config["config_id"],
                    "instance": instance_name,
                    "instance_path": str(instance_by_name[instance_name]),
                    "run": run,
                    "seed": seed,
                    "uncertainty_parameters": uncertainty_parameters,
                    "ga_config": config["ga_config"],
                    "internal_simulations": int(config["internal_simulations"]),
                    "final_simulations": int(phase_spec["final_simulations"]),
                    "max_function_evaluations": int(phase_spec["max_function_evaluations"]),
                    "time_limit_s": phase_spec.get("time_limit_s"),
                    "simulation_workers": int(args.simulation_workers),
                    "surrogate_n_jobs": int(args.surrogate_n_jobs),
                    "progress_interval_evaluations": int(args.progress_interval_evaluations),
                }
            )

    manifest = {
        "phase": phase,
        "phase_spec": phase_spec,
        "config_id": config["config_id"],
        "instances": instance_names,
        "tasks_total": len(instance_names) * int(phase_spec["run_count"]),
        "tasks_pending_at_start": len(tasks),
        "workers": int(args.workers),
        "simulation_workers": int(args.simulation_workers),
        "surrogate_n_jobs": int(args.surrogate_n_jobs),
        "uncertainty_json": str(args.uncertainty_json),
        "instances_dir": str(args.instances_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_hyperparameters_txt(
        out_dir,
        run_metadata=manifest,
        ga_config=config["ga_config"],
        run_config={
            "max_generations": None,
            "time_limit_s": phase_spec.get("time_limit_s"),
            "max_function_evaluations": phase_spec["max_function_evaluations"],
            "progress_interval_evaluations": args.progress_interval_evaluations,
            "keep_multiple": False,
            "do_restart": False,
            "final_simulations": phase_spec["final_simulations"],
            "internal_simulations": config["internal_simulations"],
        },
        notes=[
            "HPO trial config. RL mutation control is disabled.",
            "One Slurm array task evaluates one complete configuration for this phase.",
        ],
    )

    print(f"Phase={phase} config={config['config_id']} pending_runs={len(tasks)} out={out_dir}", flush=True)
    failures: list[dict[str, Any]] = []
    if args.workers <= 1:
        for task in tasks:
            try:
                row = solve_hpo_run_task(task)
            except Exception as exc:  # noqa: BLE001
                row = {"phase": phase, "config_id": config["config_id"], "instance": task["instance"], "run": task["run"], "status": "failed", "error": repr(exc)}
                failures.append(row)
            append_jsonl(raw_path, row)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = {pool.submit(solve_hpo_run_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = {"phase": phase, "config_id": config["config_id"], "instance": task["instance"], "run": task["run"], "status": "failed", "error": repr(exc)}
                    failures.append(row)
                append_jsonl(raw_path, row)
                print(
                    f"done config={config['config_id']} instance={task['instance']} run={task['run']} status={row['status']}",
                    flush=True,
                )

    summarize_one_config(out_dir)
    if failures and not args.allow_failed_runs:
        raise RuntimeError(f"{len(failures)} runs failed for {config['config_id']}; see {raw_path}")


def summarize_one_config(out_dir: Path) -> None:
    rows = [row for row in read_jsonl(out_dir / "raw_results.jsonl") if row.get("status") == "ok"]
    if not rows:
        return
    fitnesses = [float(row["fitness"]) for row in rows]
    payload = {
        "successful_runs": len(rows),
        "best_fitness": min(fitnesses),
        "mean_fitness": statistics.mean(fitnesses),
        "median_fitness": statistics.median(fitnesses),
        "std_fitness": statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
        "mean_runtime_s": statistics.mean(float(row["runtime_s"]) for row in rows),
        "mean_function_evaluations": statistics.mean(int(row["function_evaluations"]) for row in rows),
    }
    (out_dir / "trial_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase summary and ranking
# ---------------------------------------------------------------------------

def config_result_dir(hpo_root: Path, phase: str, config_id: str) -> Path:
    return phase_dir(hpo_root, phase) / "results" / config_id


def median_by_instance(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get("status") == "ok":
            grouped.setdefault(str(row["instance"]), []).append(float(row["fitness"]))
    return {inst: statistics.median(vals) for inst, vals in grouped.items() if vals}


def summarize_phase(args: argparse.Namespace) -> None:
    hpo_root = args.hpo_root.resolve()
    phase = args.phase
    configs = read_configs(hpo_root, phase)
    phase_spec = PHASES[phase]
    expected_runs = int(phase_spec["instance_count"]) * int(phase_spec["run_count"])

    raw_by_config: dict[str, list[dict[str, Any]]] = {}
    for cfg in configs:
        raw_path = config_result_dir(hpo_root, phase, cfg["config_id"]) / "raw_results.jsonl"
        raw_by_config[cfg["config_id"]] = read_jsonl(raw_path)

    c0_rows = raw_by_config.get("C0_repo_reproduction", [])
    reference = median_by_instance(c0_rows)
    if not reference:
        # Fallback: best median per instance across all configs.
        medians: dict[str, list[float]] = {}
        for rows in raw_by_config.values():
            for inst, med in median_by_instance(rows).items():
                medians.setdefault(inst, []).append(med)
        reference = {inst: min(vals) for inst, vals in medians.items() if vals}

    summary_rows: list[dict[str, Any]] = []
    for cfg in configs:
        rows = raw_by_config[cfg["config_id"]]
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        fail_count = len([row for row in rows if row.get("status") == "failed"])
        if not ok_rows:
            score = math.inf
            status = "failed"
        else:
            norm_values: list[float] = []
            stdev_values: list[float] = []
            for row in ok_rows:
                ref = reference.get(str(row["instance"]))
                if ref is None or ref <= 0:
                    continue
                norm_values.append(float(row["fitness"]) / ref)
                stdev_values.append(float(row.get("final_robust_stdev", 0.0)) / ref)
            if not norm_values:
                score = math.inf
                status = "failed"
            else:
                fail_penalty = max(0, expected_runs - len(ok_rows)) / max(1, expected_runs)
                score = statistics.mean(norm_values) + 0.05 * statistics.mean(stdev_values) + 0.10 * fail_penalty
                status = "ok"

        fitnesses = [float(row["fitness"]) for row in ok_rows]
        summary_rows.append(
            {
                "status": status,
                "rank": 0,
                "config_id": cfg["config_id"],
                "parent_config_id": cfg.get("parent_config_id") or "",
                "source": cfg.get("source") or "",
                "score": score,
                "successful_runs": len(ok_rows),
                "failed_runs": fail_count,
                "expected_runs": expected_runs,
                "best_fitness": min(fitnesses) if fitnesses else math.inf,
                "mean_fitness": statistics.mean(fitnesses) if fitnesses else math.inf,
                "median_fitness": statistics.median(fitnesses) if fitnesses else math.inf,
                "std_fitness": statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
                "mean_runtime_s": statistics.mean(float(row.get("runtime_s", 0.0)) for row in ok_rows) if ok_rows else math.inf,
                "mean_function_evaluations": statistics.mean(int(row.get("function_evaluations", 0)) for row in ok_rows) if ok_rows else math.inf,
                "params": canonical_json(cfg["params"]),
            }
        )

    summary_rows.sort(key=lambda row: (row["status"] != "ok", float(row["score"]), float(row["mean_fitness"])))
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank

    out_path = phase_dir(hpo_root, phase) / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "status",
        "config_id",
        "parent_config_id",
        "source",
        "score",
        "successful_runs",
        "failed_runs",
        "expected_runs",
        "best_fitness",
        "mean_fitness",
        "median_fitness",
        "std_fitness",
        "mean_runtime_s",
        "mean_function_evaluations",
        "params",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter=",")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    (phase_dir(hpo_root, phase) / "reference_by_instance.json").write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote phase summary to {out_path}")
    if summary_rows:
        best = summary_rows[0]
        print(f"Best: rank=1 config={best['config_id']} score={float(best['score']):.6f}")


def print_split(args: argparse.Namespace) -> None:
    split = stratified_split(load_instance_names(args.instances_dir))
    print(json.dumps(split, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["make-plan", "run-task", "summarize", "print-split"])
    parser.add_argument("--phase", choices=sorted(PHASES), default="screening")
    parser.add_argument("--hpo-root", type=Path, default=Path(os.environ.get("HPO_ROOT", "results/hpo_scenario2")))
    parser.add_argument("--instances-dir", type=Path, default=Path("instances/fjssp-w"))
    parser.add_argument("--uncertainty-json", type=Path, default=Path("config/scenario2_uncertainty.json"))
    parser.add_argument("--seed", type=int, default=20260615)

    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--simulation-workers", type=int, default=1)
    parser.add_argument("--surrogate-n-jobs", type=int, default=1)
    parser.add_argument("--progress-interval-evaluations", type=int, default=100_000)
    parser.add_argument("--max-function-evaluations", type=int, default=None)
    parser.add_argument("--final-simulations", type=int, default=None)
    parser.add_argument("--time-limit-s", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failed-runs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "make-plan":
        make_plan(args)
    elif args.command == "run-task":
        run_task(args)
    elif args.command == "summarize":
        summarize_phase(args)
    elif args.command == "print-split":
        print_split(args)
    else:
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
