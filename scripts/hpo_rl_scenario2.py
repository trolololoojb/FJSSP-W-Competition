#!/usr/bin/env python3
"""Small RL-only parameter search for the Scenario-2 HPO winner.

The final rank-1 configuration of the existing HPO is loaded from its result
files.  Only the RL learning rate and update interval are varied; all other GA,
surrogate, local-search, stochastic-evaluation, and RL parameters stay fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEARNING_RATES = (1e-4, 3e-4, 1e-3)
UPDATE_INTERVALS = (8, 16, 32)

# Reijnen et al.'s published PPO configuration uses these values.  The paper
# itself only says that the model parameters follow Schulman et al. (2017).
FIXED_RL_PARAMS: dict[str, Any] = {
    "enable_rl_mutation_control": True,
    "rl_gamma": 0.99,
    "rl_lambda": 0.95,
    "rl_clip_epsilon": 0.2,
    "rl_value_coef": 0.5,
    "rl_warmup_generations": 10,
    "rl_history_length": 3,
    "rl_hidden_size": 32,
    "rl_entropy_coef": 0.01,
}

DEFAULT_SOURCE_HPO_ROOT = Path("results/hpo_scenario2")
DEFAULT_RL_HPO_ROOT = Path("results/hpo_rl_scenario2")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(canonical_json(row) + "\n")
    tmp_path.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        fh.flush()


def load_hpo_winner(source_hpo_root: Path) -> dict[str, Any]:
    """Load rank 1 from the previous HPO instead of duplicating its values."""
    summary_path = source_hpo_root / "final" / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Previous HPO summary not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if row.get("status") == "ok"]
    if not rows:
        raise RuntimeError(f"No successful configuration in {summary_path}")

    winner = min(rows, key=lambda row: int(row["rank"]))
    config_path = source_hpo_root / "final" / "results" / winner["config_id"] / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Rank-1 configuration not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def make_configs(source_hpo_root: Path) -> list[dict[str, Any]]:
    winner = load_hpo_winner(source_hpo_root)
    base_ga_config = dict(winner["ga_config"])
    configs: list[dict[str, Any]] = []

    for learning_rate, update_interval in product(LEARNING_RATES, UPDATE_INTERVALS):
        ga_config = dict(base_ga_config)
        ga_config.update(FIXED_RL_PARAMS)
        ga_config["rl_learning_rate"] = learning_rate
        ga_config["rl_update_interval"] = update_interval
        config_id = f"rl_lr{learning_rate:.0e}_u{update_interval:03d}"
        configs.append(
            {
                "config_id": config_id,
                "source_config_id": winner["config_id"],
                "source": "final rank 1 of the non-RL Scenario-2 HPO",
                "ga_config": ga_config,
                "internal_simulations": int(winner["internal_simulations"]),
                "searched_params": {
                    "rl_learning_rate": learning_rate,
                    "rl_update_interval": update_interval,
                },
                "fixed_rl_params": dict(FIXED_RL_PARAMS),
            }
        )
    return configs


def worker_count_from_name(name: str) -> int:
    try:
        return int(name.split("_workers")[0].split("_")[-1])
    except ValueError:
        return 0


def family_key(name: str) -> str:
    parts = name.removesuffix(".fjs").split("_")
    if parts and parts[0].startswith("2") and len(parts) >= 3:
        return "_".join(parts[:3])
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


def training_instances(instances_dir: Path, count: int) -> list[str]:
    """Use the same deterministic training split as the first HPO."""
    names = sorted(path.name for path in instances_dir.glob("*.fjs"))
    if not names:
        raise FileNotFoundError(f"No .fjs files found in {instances_dir}")

    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(family_key(name), []).append(name)
    for group in groups.values():
        group.sort(key=lambda name: (worker_count_from_name(name), name))

    ordered = [name for family in sorted(groups) for name in groups[family]]
    limits = {"train": 18, "val": 6, "holdout": 6}
    buckets = {key: [] for key in limits}
    pattern = ("train", "train", "train", "val", "holdout")
    for index, name in enumerate(ordered):
        preferred = pattern[index % len(pattern)]
        for bucket in (preferred, "train", "val", "holdout"):
            if len(buckets[bucket]) < limits[bucket]:
                buckets[bucket].append(name)
                break

    if count < 1 or count > len(buckets["train"]):
        raise ValueError(f"--instance-count must be between 1 and {len(buckets['train'])}")
    return buckets["train"][:count]


def configs_path(hpo_root: Path) -> Path:
    return hpo_root / "configs.jsonl"


def load_configs(hpo_root: Path) -> list[dict[str, Any]]:
    configs = read_jsonl(configs_path(hpo_root))
    if not configs:
        raise FileNotFoundError(f"No plan at {configs_path(hpo_root)}; run 'prepare' first")
    return configs


def prepare(args: argparse.Namespace) -> None:
    configs = make_configs(args.source_hpo_root)
    instances = training_instances(args.instances_dir, args.instance_count)
    args.hpo_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(configs_path(args.hpo_root), configs)
    phase_spec = {
        "n_configs": len(configs),
        "learning_rates": list(LEARNING_RATES),
        "update_intervals": list(UPDATE_INTERVALS),
        "fixed_rl_params": FIXED_RL_PARAMS,
        "source_hpo_root": str(args.source_hpo_root),
        "source_config_id": configs[0]["source_config_id"],
        "instances": instances,
        "n_runs": args.n_runs,
        "internal_simulations": configs[0]["internal_simulations"],
        "final_simulations": args.final_simulations,
        "max_function_evaluations": args.max_function_evaluations,
        "time_limit_s": args.time_limit_s,
    }
    (args.hpo_root / "phase_spec.json").write_text(
        json.dumps(phase_spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(configs)} configurations in {args.hpo_root}")


def effective_rows(path: Path) -> list[dict[str, Any]]:
    """Deduplicate retries, preferring a successful result for each run."""
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = (str(row.get("instance")), int(row.get("run", -1)))
        previous = keyed.get(key)
        if previous is None or row.get("status") == "ok" or previous.get("status") != "ok":
            keyed[key] = row
    return list(keyed.values())


def run_task(args: argparse.Namespace) -> None:
    # Heavy solver dependencies are imported only for actual cluster workers.
    from scripts.hpo_scenario2 import solve_hpo_run_task
    from scripts.run_scenario2_submission import load_uncertainty, uncertainty_for

    configs = load_configs(args.hpo_root)
    if args.task_index < 0 or args.task_index >= len(configs):
        raise IndexError(f"--task-index outside 0..{len(configs) - 1}")

    config = configs[args.task_index]
    instances = training_instances(args.instances_dir, args.instance_count)
    instance_paths = {path.name: path for path in args.instances_dir.glob("*.fjs")}
    uncertainty_payload = load_uncertainty(args.uncertainty_json)

    out_dir = args.hpo_root / "results" / config["config_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.jsonl"
    if raw_path.exists() and raw_path.stat().st_size and not args.resume:
        raise RuntimeError(f"Results already exist at {raw_path}; use --resume or a fresh --hpo-root")

    manifest = {
        "config_id": config["config_id"],
        "source_config_id": config["source_config_id"],
        "searched_params": config["searched_params"],
        "fixed_rl_params": config["fixed_rl_params"],
        "ga_config": config["ga_config"],
        "instances": instances,
        "n_runs": args.n_runs,
        "internal_simulations": int(config["internal_simulations"]),
        "final_simulations": args.final_simulations,
        "max_function_evaluations": args.max_function_evaluations,
        "time_limit_s": args.time_limit_s,
        "workers": args.workers,
        "simulation_workers": args.simulation_workers,
        "surrogate_n_jobs": args.surrogate_n_jobs,
    }
    manifest_path = out_dir / "manifest.json"
    if raw_path.exists() and raw_path.stat().st_size:
        if not manifest_path.exists():
            raise RuntimeError(f"Cannot resume {raw_path}: {manifest_path} is missing")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        compatibility_keys = (
            "ga_config",
            "instances",
            "n_runs",
            "internal_simulations",
            "final_simulations",
            "max_function_evaluations",
            "time_limit_s",
        )
        mismatches = [key for key in compatibility_keys if previous.get(key) != manifest.get(key)]
        if mismatches:
            raise RuntimeError(
                f"Cannot resume {out_dir} with changed fields: {', '.join(mismatches)}"
            )
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    completed = {
        (str(row["instance"]), int(row["run"]))
        for row in effective_rows(raw_path)
        if row.get("status") == "ok"
    }
    tasks: list[dict[str, Any]] = []
    for instance in instances:
        for run in range(1, args.n_runs + 1):
            if (instance, run) in completed:
                continue
            seed, uncertainty_parameters = uncertainty_for(uncertainty_payload, instance, run)
            tasks.append(
                {
                    "phase": "rl_mini_hpo",
                    "config_id": config["config_id"],
                    "instance": instance,
                    "instance_path": str(instance_paths[instance]),
                    "run": run,
                    "seed": seed,
                    "uncertainty_parameters": uncertainty_parameters,
                    "ga_config": config["ga_config"],
                    "internal_simulations": int(config["internal_simulations"]),
                    "final_simulations": args.final_simulations,
                    "max_function_evaluations": args.max_function_evaluations,
                    "time_limit_s": args.time_limit_s,
                    "simulation_workers": args.simulation_workers,
                    "surrogate_n_jobs": args.surrogate_n_jobs,
                    "progress_interval_evaluations": args.progress_interval_evaluations,
                }
            )

    print(
        f"config={config['config_id']} pending_runs={len(tasks)} output={out_dir}",
        flush=True,
    )
    failures = 0

    def save_result(task: dict[str, Any], row: dict[str, Any]) -> None:
        nonlocal failures
        if row.get("status") != "ok":
            failures += 1
        append_jsonl(raw_path, row)
        print(
            f"done config={config['config_id']} instance={task['instance']} "
            f"run={task['run']} status={row['status']}",
            flush=True,
        )

    if args.workers <= 1:
        for task in tasks:
            try:
                row = solve_hpo_run_task(task)
            except Exception as exc:  # noqa: BLE001
                row = {
                    "phase": "rl_mini_hpo",
                    "config_id": config["config_id"],
                    "instance": task["instance"],
                    "run": task["run"],
                    "status": "failed",
                    "error": repr(exc),
                }
            save_result(task, row)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(solve_hpo_run_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "phase": "rl_mini_hpo",
                        "config_id": config["config_id"],
                        "instance": task["instance"],
                        "run": task["run"],
                        "status": "failed",
                        "error": repr(exc),
                    }
                save_result(task, row)

    if failures and not args.allow_failed_runs:
        raise RuntimeError(f"{failures} runs failed; see {raw_path}")


def summarize(args: argparse.Namespace) -> None:
    configs = load_configs(args.hpo_root)
    instances = training_instances(args.instances_dir, args.instance_count)
    expected_runs = len(instances) * args.n_runs
    rows_by_config: dict[str, list[dict[str, Any]]] = {}
    medians_by_config: dict[str, dict[str, float]] = {}

    for config in configs:
        raw_path = args.hpo_root / "results" / config["config_id"] / "raw_results.jsonl"
        rows = [row for row in effective_rows(raw_path) if row.get("status") == "ok"]
        rows_by_config[config["config_id"]] = rows
        grouped: dict[str, list[float]] = {}
        for row in rows:
            grouped.setdefault(str(row["instance"]), []).append(float(row["fitness"]))
        medians_by_config[config["config_id"]] = {
            instance: statistics.median(values) for instance, values in grouped.items()
        }

    reference = {
        instance: min(
            medians[instance]
            for medians in medians_by_config.values()
            if instance in medians
        )
        for instance in instances
        if any(instance in medians for medians in medians_by_config.values())
    }

    summary_rows: list[dict[str, Any]] = []
    for config in configs:
        rows = rows_by_config[config["config_id"]]
        fitnesses = [float(row["fitness"]) for row in rows]
        normalized = [
            float(row["fitness"]) / reference[str(row["instance"])]
            for row in rows
            if str(row["instance"]) in reference and reference[str(row["instance"])] > 0
        ]
        normalized_stdev = [
            float(row.get("final_robust_stdev", 0.0)) / reference[str(row["instance"])]
            for row in rows
            if str(row["instance"]) in reference and reference[str(row["instance"])] > 0
        ]
        missing_runs = max(0, expected_runs - len(rows))
        if not rows or not normalized:
            status = "failed"
            score = math.inf
        else:
            status = "ok" if missing_runs == 0 else "incomplete"
            score = (
                statistics.mean(normalized)
                + 0.05 * statistics.mean(normalized_stdev)
                + 0.10 * missing_runs / expected_runs
            )

        params = config["searched_params"]
        summary_rows.append(
            {
                "rank": 0,
                "status": status,
                "config_id": config["config_id"],
                "rl_learning_rate": params["rl_learning_rate"],
                "rl_update_interval": params["rl_update_interval"],
                "score": score,
                "successful_runs": len(rows),
                "expected_runs": expected_runs,
                "mean_fitness": statistics.mean(fitnesses) if fitnesses else math.inf,
                "median_fitness": statistics.median(fitnesses) if fitnesses else math.inf,
                "best_fitness": min(fitnesses) if fitnesses else math.inf,
                "std_fitness": statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
                "mean_runtime_s": (
                    statistics.mean(float(row.get("runtime_s", 0.0)) for row in rows)
                    if rows
                    else math.inf
                ),
            }
        )

    status_order = {"ok": 0, "incomplete": 1, "failed": 2}
    summary_rows.sort(
        key=lambda row: (status_order[str(row["status"])], float(row["score"]))
    )
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank

    args.hpo_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.hpo_root / "summary.csv"
    fields = list(summary_rows[0]) if summary_rows else []
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.hpo_root / "reference_by_instance.json").write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {summary_path}")
    if summary_rows:
        best = summary_rows[0]
        print(
            f"Best: {best['config_id']} lr={best['rl_learning_rate']} "
            f"update_interval={best['rl_update_interval']} score={float(best['score']):.6f}"
        )


def print_configs(args: argparse.Namespace) -> None:
    for index, config in enumerate(make_configs(args.source_hpo_root)):
        params = config["searched_params"]
        print(
            f"{index}\t{config['config_id']}\t"
            f"lr={params['rl_learning_rate']}\tupdate_interval={params['rl_update_interval']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run-task", "summarize", "print-configs"))
    parser.add_argument(
        "--hpo-root",
        type=Path,
        default=Path(os.environ.get("RL_HPO_ROOT", DEFAULT_RL_HPO_ROOT)),
    )
    parser.add_argument("--source-hpo-root", type=Path, default=DEFAULT_SOURCE_HPO_ROOT)
    parser.add_argument("--instances-dir", type=Path, default=Path("instances/fjssp-w"))
    parser.add_argument(
        "--uncertainty-json", type=Path, default=Path("config/scenario2_uncertainty.json")
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--instance-count", type=int, default=8)
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--final-simulations", type=int, default=20)
    parser.add_argument("--max-function-evaluations", type=int, default=5_000_000)
    parser.add_argument("--time-limit-s", type=int, default=7_200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--simulation-workers", type=int, default=1)
    parser.add_argument("--surrogate-n-jobs", type=int, default=1)
    parser.add_argument("--progress-interval-evaluations", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failed-runs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run-task":
        run_task(args)
    elif args.command == "summarize":
        summarize(args)
    elif args.command == "print-configs":
        print_configs(args)
    else:
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())