#!/usr/bin/env python3
"""Run the official Scenario-2 FJSSP-W submission pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.GA.parallel_simulation import run_n_simulations_parallel
from solver.GA.wfjssp_ga import build_ga_from_worker_encoding, is_simulatable_schedule
from util.benchmark_parser import WorkerBenchmarkParser
from util.evaluation import makespan, translate


SCENARIO = 2
INTERNAL_EVAL_SIMULATIONS = 10
FINAL_EVAL_SIMULATIONS = 50
UNCERTAINTY_SOURCE = "worker"

GA_CONFIG: dict[str, Any] = {
    "population_size": 200,
    "offspring_amount": 1000,
    "use_surrogate_evaluation": True,
    "surrogate_warmup_real_candidates": 1000,
    "surrogate_top_fraction": 0.02,
    "surrogate_uncertain_fraction": 0.005,
    "surrogate_random_fraction": 0.005,
    "surrogate_min_real_per_generation": 5,
    "surrogate_retrain_interval_real_candidates": 100,
    "surrogate_n_estimators": 300,
    "surrogate_min_samples_leaf": 3,
    "surrogate_max_features": "sqrt",
    "surrogate_n_jobs": -1,
    "surrogate_max_training_samples": 5_000,
    "surrogate_retrain_interval_growth_samples": 5_000,
    "surrogate_retrain_interval_growth_factor": 2.0,
    "surrogate_max_retrain_interval_real_candidates": 1_000,
    "local_search_interval": 20,
    "local_search_origin_count": 3,
    "local_search_neighbors_per_origin": 200,
    "local_search_top_k": 8,
    "local_search_uncertain_k": 4,
    "local_search_random_k": 3,
    "local_search_real_eval_limit_per_origin": 12,
    "local_search_min_predicted_improvement": 5.0,
    "elitism_rate": 0.1,
    "restart_generations": 800,
    "enable_rl_mutation_control": False,
    "rl_update_interval": 16,
    "rl_warmup_generations": 10,
    "rl_history_length": 3,
    "rl_learning_rate": 1e-3,
    "rl_hidden_size": 32,
    "rl_gamma": 0.99,
    "rl_lambda": 0.95,
    "rl_clip_epsilon": 0.2,
    "rl_entropy_coef": 0.01,
    "rl_value_coef": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, default=Path("instances/fjssp-w"))
    parser.add_argument("--uncertainty-json", type=Path, default=Path("config/scenario2_uncertainty.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/scenario2_submission"))
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--internal-simulations", type=int, default=INTERNAL_EVAL_SIMULATIONS)
    parser.add_argument("--final-simulations", type=int, default=FINAL_EVAL_SIMULATIONS)
    parser.add_argument(
        "--simulation-workers",
        type=int,
        default=1,
        help="Parallel worker processes used inside each stochastic simulation call.",
    )
    parser.add_argument(
        "--time-limit-s",
        type=int,
        default=None,
        help="Optional wall-clock time limit in seconds for each GA run.",
    )
    parser.add_argument("--max-function-evaluations", type=int, default=5_000_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failed-runs", action="store_true")
    parser.add_argument("--instances", nargs="+", help="Optional instance filenames to run.")
    parser.add_argument("--limit-runs", type=int, help="Stop after this many unfinished runs.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel GA worker processes. Use 0 to use all available CPU cores.",
    )
    parser.add_argument(
        "--surrogate-n-jobs",
        type=int,
        default=None,
        help=(
            "Threads used by each run's surrogate model. Defaults to 1 with parallel "
            "workers and -1 with a single worker."
        ),
    )
    parser.add_argument(
        "--disable-local-search",
        action="store_true",
        help="Disable GA local search while keeping EA and surrogate evaluation enabled.",
    )
    return parser.parse_args()


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


def compact_json(value: Any) -> str:
    return json.dumps(to_builtin(value), separators=(",", ":"))


def load_uncertainty(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("metadata", {}).get("scenario") != 2:
        raise ValueError(f"{path} is not a Scenario-2 uncertainty file.")
    if "instances" not in payload:
        raise ValueError(f"{path} has no 'instances' object.")
    return payload


def selected_instance_files(instances_dir: Path, selected_names: list[str] | None) -> list[Path]:
    all_files = sorted(instances_dir.glob("*.fjs"))
    if selected_names is None:
        return all_files
    by_name = {path.name: path for path in all_files}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown instance(s): {missing}")
    return [by_name[name] for name in selected_names]


def load_completed_ok(raw_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if not raw_path.exists():
        return completed
    with raw_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{raw_path}:{line_number}: invalid JSONL row: {exc}") from exc
            if row.get("status") == "ok":
                completed[(str(row["instance"]), int(row["run"]))] = row
    return completed


def uncertainty_for(payload: dict[str, Any], instance_name: str, run: int) -> tuple[int, list[list[float]]]:
    instance_data = payload["instances"].get(instance_name)
    if instance_data is None:
        raise KeyError(f"No uncertainty parameters for {instance_name}.")
    run_data = instance_data["runs"].get(str(run))
    if run_data is None:
        raise KeyError(f"No uncertainty parameters for {instance_name} run {run}.")
    params = [[float(value) for value in row] for row in run_data["uncertainty_parameters"]]
    return int(run_data["seed"]), params


def resolve_worker_count(requested_workers: int) -> int:
    if requested_workers < 0:
        raise ValueError("--workers must be >= 0.")
    if requested_workers == 0:
        return max(1, os.process_cpu_count() or 1)
    return requested_workers


def resolve_surrogate_n_jobs(workers: int, requested_n_jobs: int | None) -> int:
    if requested_n_jobs is not None:
        if requested_n_jobs == 0:
            raise ValueError("--surrogate-n-jobs must not be 0.")
        return requested_n_jobs
    return 1 if workers > 1 else -1


def solve_run(
    instance_name: str,
    run: int,
    seed: int,
    encoding: Any,
    uncertainty_parameters: list[list[float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    start_wall = time.time()
    ga_kwargs = dict(GA_CONFIG)
    if getattr(args, "disable_local_search", False):
        ga_kwargs.update(
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
    if getattr(args, "surrogate_n_jobs", None) is not None:
        ga_kwargs["surrogate_n_jobs"] = args.surrogate_n_jobs
    ga_kwargs.update(
        {
            "seed": seed,
            "rl_seed": seed,
            "use_stochastic_evaluation": True,
            "uncertainty_parameters": uncertainty_parameters,
            "n_simulations": args.internal_simulations,
            "simulation_workers": args.simulation_workers,
        }
    )
    run_config = {
        "max_generations": None,
        "time_limit_s": args.time_limit_s,
        "max_function_evaluations": args.max_function_evaluations,
        "progress_interval_evaluations": 50_000,
        "keep_multiple": False,
        "do_restart": False,
    }

    ga = build_ga_from_worker_encoding(encoding, **ga_kwargs)
    result = ga.run(**run_config)
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
        raise ValueError("Best decoded schedule is not simulatable.")

    try:
        final_results, robust_makespan, robust_makespan_stdev, final_r = run_n_simulations_parallel(
            start_times,
            end_times,
            machines,
            workers,
            encoding.job_sequence(),
            encoding.durations(),
            uncertainty_parameters,
            args.final_simulations,
            uncertainty_source=UNCERTAINTY_SOURCE,
            processing_times=True,
            workers=args.simulation_workers,
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
            args.final_simulations,
            processing_times=True,
            workers=args.simulation_workers,
            seed=seed + 2_000_000_000,
        )

    raw_function_evaluations = int(result["function_evaluations"])
    function_evaluations = int(
        result.get("best_found_function_evaluations", raw_function_evaluations)
    )
    if function_evaluations > args.max_function_evaluations:
        raise ValueError(
            f"FunctionEvaluations {function_evaluations} exceeds limit {args.max_function_evaluations}."
        )

    deterministic_makespan = float(makespan(start_times, machines, workers, encoding.durations()))
    return {
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
        "start_times": start_times,
        "machine_assignments": machines,
        "worker_assignments": workers,
        "uncertainty_parameters": uncertainty_parameters,
        "final_simulation_results": [float(x) for x in final_results],
    }


def solve_run_task(task: dict[str, Any]) -> dict[str, Any]:
    parser = WorkerBenchmarkParser()
    encoding = parser.parse_benchmark(str(task["instance_path"]))
    args = argparse.Namespace(
        internal_simulations=task["internal_simulations"],
        final_simulations=task["final_simulations"],
        time_limit_s=task["time_limit_s"],
        max_function_evaluations=task["max_function_evaluations"],
        surrogate_n_jobs=task["surrogate_n_jobs"],
        disable_local_search=task["disable_local_search"],
        simulation_workers=task["simulation_workers"],
    )
    return solve_run(
        task["instance"],
        int(task["run"]),
        int(task["seed"]),
        encoding,
        task["uncertainty_parameters"],
        args,
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(to_builtin(row), sort_keys=True, allow_nan=False) + "\n")
        fh.flush()


def write_csv_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    uncertainty_json: Path,
    expected_instances: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row["instance"], int(row["run"])))

    def replace_atomically(path: Path, write_fn: Any) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        write_fn(tmp_path)
        tmp_path.replace(path)

    official_fields = [
        "Instance",
        "Fitness",
        "FunctionEvaluations",
        "StartTimes",
        "MachineAssignments",
        "WorkerAssignments",
        "UncertaintyParameters",
    ]
    def write_official(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=official_fields, delimiter=";")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "Instance": row["instance"],
                        "Fitness": row["fitness"],
                        "FunctionEvaluations": row["function_evaluations"],
                        "StartTimes": compact_json(row["start_times"]),
                        "MachineAssignments": compact_json(row["machine_assignments"]),
                        "WorkerAssignments": compact_json(row["worker_assignments"]),
                        "UncertaintyParameters": compact_json(row["uncertainty_parameters"]),
                    }
                )

    replace_atomically(output_dir / "submission_scenario2.csv", write_official)

    metadata_fields = [
        "Instance",
        "Run",
        "Seed",
        "Fitness",
        "FunctionEvaluations",
        "StartTimes",
        "MachineAssignments",
        "WorkerAssignments",
        "UncertaintyParameters",
        "DeterministicMakespan",
        "FinalRobustStdev",
        "FinalR",
        "RuntimeSeconds",
        "Generations",
    ]
    def write_metadata(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=metadata_fields, delimiter=";")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "Instance": row["instance"],
                        "Run": row["run"],
                        "Seed": row["seed"],
                        "Fitness": row["fitness"],
                        "FunctionEvaluations": row["function_evaluations"],
                        "StartTimes": compact_json(row["start_times"]),
                        "MachineAssignments": compact_json(row["machine_assignments"]),
                        "WorkerAssignments": compact_json(row["worker_assignments"]),
                        "UncertaintyParameters": compact_json(row["uncertainty_parameters"]),
                        "DeterministicMakespan": row["deterministic_makespan"],
                        "FinalRobustStdev": row["final_robust_stdev"],
                        "FinalR": row["final_R"],
                        "RuntimeSeconds": row["runtime_s"],
                        "Generations": row["generations"],
                    }
                )

    replace_atomically(output_dir / "submission_scenario2_with_metadata.csv", write_metadata)

    run_fields = [
        "Instance",
        "Run",
        "Seed",
        "Fitness",
        "DeterministicMakespan",
        "FinalRobustStdev",
        "FinalR",
        "FunctionEvaluations",
        "RawFunctionEvaluations",
        "RuntimeSeconds",
        "Generations",
    ]
    def write_runs(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=run_fields, delimiter=";")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "Instance": row["instance"],
                        "Run": row["run"],
                        "Seed": row["seed"],
                        "Fitness": row["fitness"],
                        "DeterministicMakespan": row["deterministic_makespan"],
                        "FinalRobustStdev": row["final_robust_stdev"],
                        "FinalR": row["final_R"],
                        "FunctionEvaluations": row["function_evaluations"],
                        "RawFunctionEvaluations": row["raw_function_evaluations"],
                        "RuntimeSeconds": row["runtime_s"],
                        "Generations": row["generations"],
                    }
                )

    replace_atomically(output_dir / "run_results.csv", write_runs)

    summary_fields = [
        "Instance",
        "SuccessfulRuns",
        "BestFitness",
        "MeanFitness",
        "StdFitness",
        "BestRun",
        "BestSeed",
        "MeanRuntimeSeconds",
        "MeanFunctionEvaluations",
    ]
    def write_summary(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=summary_fields, delimiter=";")
            writer.writeheader()
            for instance in sorted({row["instance"] for row in rows}):
                instance_rows = [row for row in rows if row["instance"] == instance]
                fitnesses = [float(row["fitness"]) for row in instance_rows]
                best = min(instance_rows, key=lambda row: float(row["fitness"]))
                writer.writerow(
                    {
                        "Instance": instance,
                        "SuccessfulRuns": len(instance_rows),
                        "BestFitness": min(fitnesses),
                        "MeanFitness": statistics.mean(fitnesses),
                        "StdFitness": statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
                        "BestRun": best["run"],
                        "BestSeed": best["seed"],
                        "MeanRuntimeSeconds": statistics.mean(float(row["runtime_s"]) for row in instance_rows),
                        "MeanFunctionEvaluations": statistics.mean(
                            int(row["function_evaluations"]) for row in instance_rows
                        ),
                    }
                )

    replace_atomically(output_dir / "instance_summary.csv", write_summary)

    manifest = {
        "scenario": 2,
        "n_instances": expected_instances,
        "n_runs_per_instance": args.n_runs,
        "total_expected_runs": expected_instances * args.n_runs,
        "total_successful_runs": len(rows),
        "time_limit_s": args.time_limit_s,
        "max_function_evaluations": args.max_function_evaluations,
        "internal_simulations": args.internal_simulations,
        "final_simulations": args.final_simulations,
        "workers": args.workers,
        "surrogate_n_jobs": args.surrogate_n_jobs,
        "simulation_workers": args.simulation_workers,
        "local_search_enabled": not bool(args.disable_local_search),
        "uncertainty_json": str(uncertainty_json),
        "official_csv": str(output_dir / "submission_scenario2.csv"),
        "metadata_csv": str(output_dir / "submission_scenario2_with_metadata.csv"),
        "function_evaluations_note": (
            "FunctionEvaluations uses best_found_function_evaluations from WFJSSPGA.run "
            "when available, otherwise final run function_evaluations."
        ),
    }
    def write_manifest(path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")

    replace_atomically(output_dir / "submission_manifest.json", write_manifest)


def main() -> int:
    args = parse_args()
    try:
        args.workers = resolve_worker_count(args.workers)
        args.surrogate_n_jobs = resolve_surrogate_n_jobs(args.workers, args.surrogate_n_jobs)
        uncertainty_payload = load_uncertainty(args.uncertainty_json)
        instance_files = selected_instance_files(args.instances_dir, args.instances)
        if not instance_files:
            raise ValueError(f"No .fjs files found in {args.instances_dir}.")

        raw_path = args.output_dir / "raw_results.jsonl"
        completed = load_completed_ok(raw_path) if args.resume else {}
        failures = []
        total_expected = len(instance_files) * args.n_runs
        started = 0
        tasks = []

        for instance_position, instance_path in enumerate(instance_files, start=1):
            for run in range(1, args.n_runs + 1):
                key = (instance_path.name, run)
                progress_index = (instance_position - 1) * args.n_runs + run
                if key in completed:
                    print(f"[{progress_index}/{total_expected}] skip instance={instance_path.name} run={run}")
                    continue
                if args.limit_runs is not None and started >= args.limit_runs:
                    break
                seed, uncertainty_parameters = uncertainty_for(uncertainty_payload, instance_path.name, run)
                tasks.append(
                    {
                        "progress_index": progress_index,
                        "total_expected": total_expected,
                        "instance_path": str(instance_path),
                        "instance": instance_path.name,
                        "run": run,
                        "seed": seed,
                        "uncertainty_parameters": uncertainty_parameters,
                        "internal_simulations": args.internal_simulations,
                        "final_simulations": args.final_simulations,
                        "time_limit_s": args.time_limit_s,
                        "max_function_evaluations": args.max_function_evaluations,
                        "surrogate_n_jobs": args.surrogate_n_jobs,
                        "simulation_workers": args.simulation_workers,
                        "disable_local_search": args.disable_local_search,
                    }
                )
                started += 1
            if args.limit_runs is not None and started >= args.limit_runs:
                break

        if tasks:
            active_workers = min(args.workers, len(tasks))
            print(
                f"Running {len(tasks)} unfinished run(s) with {active_workers} worker(s) "
                f"and surrogate_n_jobs={args.surrogate_n_jobs}.",
                flush=True,
            )

        def record_failure(task: dict[str, Any], exc: BaseException) -> None:
            error_row = {
                "instance": task["instance"],
                "run": task["run"],
                "seed": task["seed"],
                "status": "error",
                "error": str(exc),
            }
            append_jsonl(raw_path, error_row)
            failures.append(error_row)
            print(
                f"Error in {task['instance']} run {task['run']}: {exc}",
                file=sys.stderr,
                flush=True,
            )

        def record_success(task: dict[str, Any], row: dict[str, Any]) -> None:
            append_jsonl(raw_path, row)
            completed[(task["instance"], int(task["run"]))] = row
            write_csv_outputs(
                args.output_dir,
                list(completed.values()),
                args,
                args.uncertainty_json,
                expected_instances=len(instance_files),
            )
            print(
                f"[{task['progress_index']}/{task['total_expected']}] "
                f"done instance={task['instance']} run={task['run']} "
                f"fitness={row['fitness']}",
                flush=True,
            )

        if completed:
            write_csv_outputs(
                args.output_dir,
                list(completed.values()),
                args,
                args.uncertainty_json,
                expected_instances=len(instance_files),
            )

        if tasks and min(args.workers, len(tasks)) == 1:
            for task in tasks:
                print(
                    f"[{task['progress_index']}/{task['total_expected']}] "
                    f"instance={task['instance']} run={task['run']} seed={task['seed']}",
                    flush=True,
                )
                try:
                    row = solve_run_task(task)
                    record_success(task, row)
                except Exception as exc:
                    record_failure(task, exc)
        elif tasks:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
                future_to_task = {executor.submit(solve_run_task, task): task for task in tasks}
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        row = future.result()
                        record_success(task, row)
                    except Exception as exc:
                        record_failure(task, exc)

        ok_rows = list(completed.values())
        write_csv_outputs(
            args.output_dir,
            ok_rows,
            args,
            args.uncertainty_json,
            expected_instances=len(instance_files),
        )

        expected_success = len(instance_files) * args.n_runs
        if failures and not args.allow_failed_runs:
            raise RuntimeError(f"{len(failures)} run(s) failed. See {raw_path}.")
        if len(ok_rows) != expected_success and not args.allow_failed_runs:
            raise RuntimeError(f"Expected {expected_success} successful runs, found {len(ok_rows)}.")
        print(f"Wrote Scenario-2 outputs to {args.output_dir} ({len(ok_rows)} successful runs).")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
