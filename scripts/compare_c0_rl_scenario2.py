#!/usr/bin/env python3
"""Compare Scenario-2 C0 against C0 with RL mutation control enabled."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VARIANTS = ("c0_no_rl", "c0_with_rl")
HPO_WINNER_INTERNAL_SIMULATIONS = 12


def variant_ga_config(variant: str, surrogate_n_jobs: int) -> dict[str, Any]:
    from scripts.run_scenario2_submission import GA_CONFIG

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    cfg = dict(GA_CONFIG)
    cfg["enable_rl_mutation_control"] = variant == "c0_with_rl"
    cfg["surrogate_n_jobs"] = surrogate_n_jobs
    return cfg


def instance_names(instances_dir: Path) -> list[str]:
    names = sorted(path.name for path in instances_dir.glob("*.fjs"))
    if not names:
        raise FileNotFoundError(f"No .fjs files found in {instances_dir}")
    return names


def task_specs(instances_dir: Path) -> list[tuple[str, str]]:
    return [(variant, name) for variant in VARIANTS for name in instance_names(instances_dir)]


def task_output_dir(output_root: Path, variant: str, instance_name: str) -> Path:
    return output_root / variant / instance_name.removesuffix(".fjs")


def raw_results_path(output_root: Path, variant: str, instance_name: str) -> Path:
    return task_output_dir(output_root, variant, instance_name) / "raw_results.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def load_completed_ok(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") == "ok":
            completed[(str(row["instance"]), int(row["run"]))] = row
    return completed


def solve_compare_run_task(task: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_scenario2_submission import UNCERTAINTY_SOURCE
    from solver.GA.parallel_simulation import run_n_simulations_parallel
    from solver.GA.wfjssp_ga import build_ga_from_worker_encoding, is_simulatable_schedule
    from util.benchmark_parser import WorkerBenchmarkParser
    from util.evaluation import makespan, translate

    parser = WorkerBenchmarkParser()
    encoding = parser.parse_benchmark(str(task["instance_path"]))
    instance_name = str(task["instance"])
    run = int(task["run"])
    seed = int(task["seed"])
    uncertainty_parameters = task["uncertainty_parameters"]
    start_wall = time.time()

    ga_kwargs = dict(task["ga_config"])
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
        "phase": "c0_rl_comparison",
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
        "rl_enabled": bool(ga_kwargs.get("enable_rl_mutation_control", False)),
        "start_times": start_times,
        "machine_assignments": machines,
        "worker_assignments": workers,
        "uncertainty_parameters": uncertainty_parameters,
        "final_simulation_results": [float(x) for x in final_results],
    }


def run_task(args: argparse.Namespace) -> None:
    from scripts.run_scenario2_submission import load_uncertainty, uncertainty_for

    tasks = task_specs(args.instances_dir)
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise IndexError(f"--task-index {args.task_index} outside 0..{len(tasks) - 1}")

    variant, instance_name = tasks[args.task_index]
    out_dir = task_output_dir(args.output_root, variant, instance_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_results_path(args.output_root, variant, instance_name)
    completed = load_completed_ok(raw_path) if args.resume else {}

    uncertainty_payload = load_uncertainty(args.uncertainty_json)
    instance_path = args.instances_dir / instance_name
    ga_config = variant_ga_config(variant, args.surrogate_n_jobs)

    manifest = {
        "comparison": "scenario2_c0_vs_c0_rl",
        "variant": variant,
        "instance": instance_name,
        "n_runs": args.n_runs,
        "internal_simulations": args.internal_simulations,
        "final_simulations": args.final_simulations,
        "max_function_evaluations": args.max_function_evaluations,
        "time_limit_s": args.time_limit_s,
        "workers": args.workers,
        "simulation_workers": args.simulation_workers,
        "surrogate_n_jobs": args.surrogate_n_jobs,
        "ga_config": ga_config,
    }
    manifest_path = out_dir / "manifest.json"
    if raw_path.exists() and raw_path.stat().st_size:
        if not manifest_path.exists():
            raise RuntimeError(
                f"Cannot resume {raw_path}: results exist without a manifest. "
                "Use a fresh --output-root."
            )
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        compatibility_keys = (
            "variant",
            "instance",
            "internal_simulations",
            "final_simulations",
            "max_function_evaluations",
            "time_limit_s",
            "ga_config",
        )
        mismatches = [
            key
            for key in compatibility_keys
            if previous_manifest.get(key) != manifest.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                f"Cannot mix results with a changed experiment configuration in {out_dir}; "
                f"mismatched manifest fields: {', '.join(mismatches)}. "
                "Use a fresh --output-root."
            )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_tasks: list[dict[str, Any]] = []
    for run in range(1, args.n_runs + 1):
        if (instance_name, run) in completed:
            continue
        seed, uncertainty_parameters = uncertainty_for(uncertainty_payload, instance_name, run)
        run_tasks.append(
            {
                "phase": "c0_rl_comparison",
                "config_id": variant,
                "instance": instance_name,
                "instance_path": str(instance_path),
                "run": run,
                "seed": seed,
                "uncertainty_parameters": uncertainty_parameters,
                "ga_config": ga_config,
                "internal_simulations": args.internal_simulations,
                "final_simulations": args.final_simulations,
                "max_function_evaluations": args.max_function_evaluations,
                "time_limit_s": args.time_limit_s,
                "simulation_workers": args.simulation_workers,
                "surrogate_n_jobs": args.surrogate_n_jobs,
                "progress_interval_evaluations": args.progress_interval_evaluations,
            }
        )

    print(
        f"variant={variant} instance={instance_name} pending_runs={len(run_tasks)} out={out_dir}",
        flush=True,
    )
    failures = 0
    if args.workers <= 1:
        for task in run_tasks:
            try:
                row = solve_compare_run_task(task)
            except Exception as exc:  # noqa: BLE001
                row = {
                    "phase": "c0_rl_comparison",
                    "config_id": variant,
                    "instance": task["instance"],
                    "run": task["run"],
                    "status": "failed",
                    "error": repr(exc),
                }
                failures += 1
            append_jsonl(raw_path, row)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(solve_compare_run_task, task): task for task in run_tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "phase": "c0_rl_comparison",
                        "config_id": variant,
                        "instance": task["instance"],
                        "run": task["run"],
                        "status": "failed",
                        "error": repr(exc),
                    }
                    failures += 1
                append_jsonl(raw_path, row)
                print(
                    f"done variant={variant} instance={task['instance']} run={task['run']} status={row['status']}",
                    flush=True,
                )

    if failures and not args.allow_failed_runs:
        raise RuntimeError(f"{failures} runs failed for {variant}/{instance_name}; see {raw_path}")


def ok_rows(output_root: Path, variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in sorted((output_root / variant).glob("*/raw_results.jsonl")):
        rows.extend(row for row in read_jsonl(raw_path) if row.get("status") == "ok")
    return rows


def median(values: list[float]) -> float:
    return statistics.median(values) if values else math.inf


def summarize(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows_by_variant = {variant: ok_rows(args.output_root, variant) for variant in VARIANTS}
    keyed: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        variant: {(row["instance"], int(row["run"])): row for row in rows}
        for variant, rows in rows_by_variant.items()
    }
    common_keys = sorted(set(keyed["c0_no_rl"]) & set(keyed["c0_with_rl"]))

    pair_path = args.output_root / "paired_run_comparison.csv"
    with pair_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "instance",
            "run",
            "no_rl_fitness",
            "with_rl_fitness",
            "ratio_with_rl_over_no_rl",
            "delta_with_rl_minus_no_rl",
            "no_rl_runtime_s",
            "with_rl_runtime_s",
            "runtime_ratio_with_rl_over_no_rl",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key in common_keys:
            base = keyed["c0_no_rl"][key]
            rl = keyed["c0_with_rl"][key]
            base_fit = float(base["fitness"])
            rl_fit = float(rl["fitness"])
            base_rt = float(base.get("runtime_s", 0.0))
            rl_rt = float(rl.get("runtime_s", 0.0))
            writer.writerow(
                {
                    "instance": key[0],
                    "run": key[1],
                    "no_rl_fitness": base_fit,
                    "with_rl_fitness": rl_fit,
                    "ratio_with_rl_over_no_rl": rl_fit / base_fit if base_fit > 0 else math.inf,
                    "delta_with_rl_minus_no_rl": rl_fit - base_fit,
                    "no_rl_runtime_s": base_rt,
                    "with_rl_runtime_s": rl_rt,
                    "runtime_ratio_with_rl_over_no_rl": rl_rt / base_rt if base_rt > 0 else math.inf,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    expected_runs = len(instance_names(args.instances_dir)) * args.n_runs
    for variant in VARIANTS:
        rows = rows_by_variant[variant]
        fitnesses = [float(row["fitness"]) for row in rows]
        runtimes = [float(row.get("runtime_s", 0.0)) for row in rows]
        summary_rows.append(
            {
                "variant": variant,
                "successful_runs": len(rows),
                "expected_runs": expected_runs,
                "mean_fitness": statistics.mean(fitnesses) if fitnesses else math.inf,
                "median_fitness": median(fitnesses),
                "best_fitness": min(fitnesses) if fitnesses else math.inf,
                "std_fitness": statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
                "mean_runtime_s": statistics.mean(runtimes) if runtimes else math.inf,
                "median_runtime_s": median(runtimes),
            }
        )

    ratios = [
        float(keyed["c0_with_rl"][key]["fitness"]) / float(keyed["c0_no_rl"][key]["fitness"])
        for key in common_keys
        if float(keyed["c0_no_rl"][key]["fitness"]) > 0
    ]
    deltas = [
        float(keyed["c0_with_rl"][key]["fitness"]) - float(keyed["c0_no_rl"][key]["fitness"])
        for key in common_keys
    ]
    rl_wins = sum(1 for delta in deltas if delta < 0)

    summary_path = args.output_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "variant",
            "successful_runs",
            "expected_runs",
            "mean_fitness",
            "median_fitness",
            "best_fitness",
            "std_fitness",
            "mean_runtime_s",
            "median_runtime_s",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    report = {
        "expected_runs_per_variant": expected_runs,
        "paired_runs": len(common_keys),
        "rl_win_count": rl_wins,
        "no_rl_win_count": len(deltas) - rl_wins,
        "mean_ratio_with_rl_over_no_rl": statistics.mean(ratios) if ratios else math.inf,
        "median_ratio_with_rl_over_no_rl": median(ratios),
        "mean_delta_with_rl_minus_no_rl": statistics.mean(deltas) if deltas else math.inf,
        "median_delta_with_rl_minus_no_rl": median(deltas),
        "summary_csv": str(summary_path),
        "paired_csv": str(pair_path),
    }
    (args.output_root / "comparison_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {pair_path}")
    print(
        "RL ratio with/no-RL: "
        f"mean={report['mean_ratio_with_rl_over_no_rl']:.6f}, "
        f"median={report['median_ratio_with_rl_over_no_rl']:.6f}, "
        f"paired_runs={len(common_keys)}"
    )


def print_tasks(args: argparse.Namespace) -> None:
    for idx, (variant, instance) in enumerate(task_specs(args.instances_dir)):
        print(f"{idx}\t{variant}\t{instance}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run-task", "summarize", "print-tasks"])
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--instances-dir", type=Path, default=Path("instances/fjssp-w"))
    parser.add_argument("--uncertainty-json", type=Path, default=Path("config/scenario2_uncertainty.json"))
    parser.add_argument("--output-root", type=Path, default=Path("results/c0_rl_scenario2"))
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument(
        "--internal-simulations",
        type=int,
        default=HPO_WINNER_INTERNAL_SIMULATIONS,
        help="Simulations per stochastic GA evaluation (default: final HPO winner value).",
    )
    parser.add_argument("--final-simulations", type=int, default=50)
    parser.add_argument("--max-function-evaluations", type=int, default=5_000_000)
    parser.add_argument("--time-limit-s", type=int, default=7_200)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--simulation-workers", type=int, default=2)
    parser.add_argument("--surrogate-n-jobs", type=int, default=2)
    parser.add_argument("--progress-interval-evaluations", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failed-runs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run-task":
        run_task(args)
    elif args.command == "summarize":
        summarize(args)
    elif args.command == "print-tasks":
        print_tasks(args)
    else:
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
