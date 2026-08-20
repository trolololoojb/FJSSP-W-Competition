#!/usr/bin/env python3
"""Run and summarize the pre-registered 2x2 Scenario-2 HPO/RL comparison.

The four variants cross the stored competition configuration and the stored
non-RL HPO winner with RL disabled/enabled.  Configurations are loaded from
result artifacts deliberately; the mutable GA_CONFIG in the submission runner
is not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VARIANTS = (
    "competition_no_rl",
    "hpo_no_rl",
    "competition_with_rl",
    "hpo_with_rl",
)
COMPARISONS = (
    ("non_rl_hpo_effect", "competition_no_rl", "hpo_no_rl"),
    ("rl_effect_on_competition", "competition_no_rl", "competition_with_rl"),
    ("rl_effect_on_hpo", "hpo_no_rl", "hpo_with_rl"),
    ("combined_effect", "competition_no_rl", "hpo_with_rl"),
)
EXPECTED_INSTANCE_COUNT = 30
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def atomic_csv(
    path: Path,
    fields: list[str],
    rows: Iterable[dict[str, Any]],
    *,
    delimiter: str = ",",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        fh.flush()


def rank_one(summary_path: Path) -> dict[str, str]:
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    winners = [
        row
        for row in rows
        if row.get("rank") == "1" and row.get("status") == "ok"
    ]
    if len(winners) != 1:
        raise RuntimeError(
            f"Expected exactly one successful rank-1 row in {summary_path}, got {len(winners)}"
        )
    return winners[0]


def load_artifact_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    submission = read_json(args.competition_manifest)
    competition_artifact = read_json(args.competition_config)
    competition_ga = submission.get("ga_config") or competition_artifact.get("ga_config")
    if not isinstance(competition_ga, dict) or not competition_ga:
        raise ValueError(
            "Competition ga_config is absent from both stored artifacts: "
            f"{args.competition_manifest} and {args.competition_config}"
        )
    competition_internal = int(submission.get("internal_simulations", 0))
    if competition_internal <= 0:
        raise ValueError(f"Invalid internal_simulations in {args.competition_manifest}")
    artifact_internal = int(competition_artifact.get("internal_simulations", 0))
    if artifact_internal != competition_internal:
        raise ValueError(
            "Competition artifacts disagree on internal_simulations: "
            f"manifest={competition_internal}, config={artifact_internal}"
        )
    artifact_ga = competition_artifact.get("ga_config")
    if submission.get("ga_config") is not None and artifact_ga != competition_ga:
        raise ValueError("Competition manifest and C0 artifact contain different ga_config values")

    hpo_winner = rank_one(args.hpo_final_root / "summary.csv")
    hpo_id = str(hpo_winner["config_id"])
    hpo_artifact = read_json(args.hpo_final_root / "results" / hpo_id / "config.json")
    hpo_ga = hpo_artifact.get("ga_config")
    if not isinstance(hpo_ga, dict) or not hpo_ga:
        raise ValueError(f"Missing ga_config for non-RL HPO winner {hpo_id}")
    hpo_internal = int(hpo_artifact.get("internal_simulations", 0))
    if hpo_internal <= 0:
        raise ValueError(f"Invalid internal_simulations for non-RL HPO winner {hpo_id}")

    rl_winner = rank_one(args.rl_hpo_root / "summary.csv")
    rl_id = str(rl_winner["config_id"])
    rl_artifact = read_json(args.rl_hpo_root / "results" / rl_id / "config.json")
    rl_ga = rl_artifact.get("ga_config")
    if not isinstance(rl_ga, dict) or not rl_ga:
        raise ValueError(f"Missing ga_config for RL HPO winner {rl_id}")
    if rl_artifact.get("source_config_id") != hpo_id:
        raise ValueError(
            "RL winner was not tuned on the selected non-RL HPO winner: "
            f"expected {hpo_id}, got {rl_artifact.get('source_config_id')}"
        )
    missing_rl = sorted(key for key in RL_KEYS if key not in rl_ga)
    if missing_rl:
        raise ValueError(f"RL winner lacks required parameters: {', '.join(missing_rl)}")

    rl_values = {key: rl_ga[key] for key in RL_KEYS}

    def build(base: dict[str, Any], enabled: bool) -> dict[str, Any]:
        config = dict(base)
        if enabled:
            config.update(rl_values)
            config["enable_rl_mutation_control"] = True
        else:
            config["enable_rl_mutation_control"] = False
        config["surrogate_n_jobs"] = args.surrogate_n_jobs
        return config

    configs = {
        "competition_no_rl": {
            "ga_config": build(competition_ga, False),
            "internal_simulations": competition_internal,
            "base_source": [str(args.competition_manifest), str(args.competition_config)],
            "rl_source": None,
        },
        "hpo_no_rl": {
            "ga_config": build(hpo_ga, False),
            "internal_simulations": hpo_internal,
            "base_source": hpo_id,
            "rl_source": None,
        },
        "competition_with_rl": {
            "ga_config": build(competition_ga, True),
            "internal_simulations": competition_internal,
            "base_source": [str(args.competition_manifest), str(args.competition_config)],
            "rl_source": rl_id,
        },
        "hpo_with_rl": {
            "ga_config": build(hpo_ga, True),
            "internal_simulations": hpo_internal,
            "base_source": hpo_id,
            "rl_source": rl_id,
        },
    }

    for left, right in (
        ("competition_no_rl", "competition_with_rl"),
        ("hpo_no_rl", "hpo_with_rl"),
    ):
        no_rl = configs[left]["ga_config"]
        with_rl = configs[right]["ga_config"]
        non_rl_differences = sorted(
            key
            for key in set(no_rl) | set(with_rl)
            if key not in RL_KEYS and no_rl.get(key) != with_rl.get(key)
        )
        if non_rl_differences:
            raise ValueError(
                f"{left} and {right} differ outside RL fields: {', '.join(non_rl_differences)}"
            )
    return configs


def instance_names(instances_dir: Path) -> list[str]:
    names = sorted(path.name for path in instances_dir.glob("*.fjs"))
    if len(names) != EXPECTED_INSTANCE_COUNT:
        raise ValueError(
            f"Competition comparison requires exactly {EXPECTED_INSTANCE_COUNT} instances; "
            f"found {len(names)} in {instances_dir}"
        )
    return names


def task_specs(instances_dir: Path) -> list[tuple[str, str]]:
    names = instance_names(instances_dir)
    return [(variant, instance) for variant in VARIANTS for instance in names]


def task_dir(output_root: Path, variant: str, instance: str) -> Path:
    return output_root / variant / instance.removesuffix(".fjs")


def effective_rows(path: Path) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = (str(row.get("instance")), int(row.get("run", -1)))
        previous = keyed.get(key)
        if previous is None or row.get("status") == "ok" or previous.get("status") != "ok":
            keyed[key] = row
    return list(keyed.values())


def solve_run(task: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_scenario2_submission import UNCERTAINTY_SOURCE
    from solver.GA.parallel_simulation import run_n_simulations_parallel
    from solver.GA.wfjssp_ga import build_ga_from_worker_encoding, is_simulatable_schedule
    from util.benchmark_parser import WorkerBenchmarkParser
    from util.evaluation import makespan, translate

    encoding = WorkerBenchmarkParser().parse_benchmark(str(task["instance_path"]))
    seed = int(task["seed"])
    start_wall = time.time()
    ga_config = dict(task["ga_config"])
    ga_config.update(
        {
            "seed": seed,
            "rl_seed": seed,
            "use_stochastic_evaluation": True,
            "n_simulations": int(task["internal_simulations"]),
            "simulation_workers": int(task["simulation_workers"]),
            "surrogate_n_jobs": int(task["surrogate_n_jobs"]),
            "uncertainty_parameters": task["uncertainty_parameters"],
        }
    )
    ga = build_ga_from_worker_encoding(encoding, **ga_config)
    result = ga.run(
        max_generations=None,
        time_limit_s=int(task["time_limit_s"]),
        max_function_evaluations=int(task["max_function_evaluations"]),
        progress_interval_evaluations=int(task["progress_interval_evaluations"]),
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
        int(task["final_simulations"]),
    )
    try:
        final_results, robust_makespan, robust_stdev, final_r = run_n_simulations_parallel(
            *simulation_args,
            uncertainty_source=UNCERTAINTY_SOURCE,
            processing_times=True,
            workers=int(task["simulation_workers"]),
            seed=seed + 2_000_000_000,
        )
    except TypeError:
        final_results, robust_makespan, robust_stdev, final_r = run_n_simulations_parallel(
            *simulation_args,
            processing_times=True,
            workers=int(task["simulation_workers"]),
            seed=seed + 2_000_000_000,
        )

    raw_evaluations = int(result["function_evaluations"])
    evaluations = int(result.get("best_found_function_evaluations", raw_evaluations))
    if evaluations > int(task["max_function_evaluations"]):
        raise ValueError(
            f"FunctionEvaluations {evaluations} exceeds {task['max_function_evaluations']}"
        )
    return {
        "experiment": "scenario2_hpo_rl_factorial",
        "variant": task["variant"],
        "config_id": task["variant"],
        "instance": task["instance"],
        "run": int(task["run"]),
        "seed": seed,
        "status": "ok",
        "fitness": float(robust_makespan),
        "final_robust_makespan": float(robust_makespan),
        "final_robust_stdev": float(robust_stdev),
        "final_R": float(final_r),
        "deterministic_makespan": float(
            makespan(start_times, machines, workers, encoding.durations())
        ),
        "function_evaluations": evaluations,
        "raw_function_evaluations": raw_evaluations,
        "runtime_s": float(result.get("runtime_s", time.time() - start_wall)),
        "generations": int(result["generations"]),
        "rl_enabled": bool(ga_config["enable_rl_mutation_control"]),
        "internal_simulations": int(task["internal_simulations"]),
        "start_times": start_times,
        "machine_assignments": machines,
        "worker_assignments": workers,
        "uncertainty_parameters": task["uncertainty_parameters"],
        "final_simulation_results": [float(x) for x in final_results],
    }


def run_task(args: argparse.Namespace) -> None:
    from scripts.run_scenario2_submission import load_uncertainty, uncertainty_for

    specs = task_specs(args.instances_dir)
    if args.task_index < 0 or args.task_index >= len(specs):
        raise IndexError(f"--task-index must be in 0..{len(specs) - 1}")
    configs = load_artifact_configs(args)
    variant, instance = specs[args.task_index]
    variant_config = configs[variant]
    out_dir = task_dir(args.output_root, variant, instance)
    raw_path = out_dir / "raw_results.jsonl"
    manifest_path = out_dir / "manifest.json"
    config_path = out_dir / "effective_config.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "scenario2_hpo_rl_factorial",
        "variant": variant,
        "instance": instance,
        "n_runs": args.n_runs,
        "internal_simulations": variant_config["internal_simulations"],
        "final_simulations": args.final_simulations,
        "max_function_evaluations": args.max_function_evaluations,
        "time_limit_s": args.time_limit_s,
        "workers": args.workers,
        "simulation_workers": args.simulation_workers,
        "surrogate_n_jobs": args.surrogate_n_jobs,
        "uncertainty_json": str(args.uncertainty_json),
        "ga_config": variant_config["ga_config"],
        "base_source": variant_config["base_source"],
        "rl_source": variant_config["rl_source"],
    }
    if raw_path.exists() and raw_path.stat().st_size:
        if not args.resume:
            raise RuntimeError(f"Results exist at {raw_path}; use --resume or a fresh output root")
        if not manifest_path.exists():
            raise RuntimeError(f"Cannot resume {raw_path} without {manifest_path}")
        previous = read_json(manifest_path)
        compatibility = (
            "variant",
            "instance",
            "n_runs",
            "internal_simulations",
            "final_simulations",
            "max_function_evaluations",
            "time_limit_s",
            "uncertainty_json",
            "ga_config",
        )
        mismatches = [key for key in compatibility if previous.get(key) != manifest.get(key)]
        if mismatches:
            raise RuntimeError(
                f"Cannot resume changed experiment in {out_dir}: {', '.join(mismatches)}"
            )
    atomic_json(manifest_path, manifest)
    atomic_json(config_path, {"variant": variant, **variant_config})

    completed = {
        (str(row["instance"]), int(row["run"]))
        for row in effective_rows(raw_path)
        if row.get("status") == "ok"
    }
    uncertainty = load_uncertainty(args.uncertainty_json)
    run_tasks: list[dict[str, Any]] = []
    for run in range(1, args.n_runs + 1):
        if (instance, run) in completed:
            continue
        seed, parameters = uncertainty_for(uncertainty, instance, run)
        run_tasks.append(
            {
                "variant": variant,
                "instance": instance,
                "instance_path": str(args.instances_dir / instance),
                "run": run,
                "seed": seed,
                "uncertainty_parameters": parameters,
                "ga_config": variant_config["ga_config"],
                "internal_simulations": variant_config["internal_simulations"],
                "final_simulations": args.final_simulations,
                "max_function_evaluations": args.max_function_evaluations,
                "time_limit_s": args.time_limit_s,
                "simulation_workers": args.simulation_workers,
                "surrogate_n_jobs": args.surrogate_n_jobs,
                "progress_interval_evaluations": args.progress_interval_evaluations,
            }
        )

    print(f"variant={variant} instance={instance} pending_runs={len(run_tasks)}", flush=True)
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
            "experiment": "scenario2_hpo_rl_factorial",
            "variant": variant,
            "instance": instance,
            "run": int(task["run"]),
            "seed": int(task["seed"]),
            "status": "failed",
            "error": repr(exc),
        }

    if args.workers <= 1:
        for task in run_tasks:
            try:
                row = solve_run(task)
            except Exception as exc:  # noqa: BLE001
                row = failed(task, exc)
            save(task, row)
    elif run_tasks:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(run_tasks))) as pool:
            futures = {pool.submit(solve_run, task): task for task in run_tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = failed(task, exc)
                save(task, row)
    if failures and not args.allow_failed_runs:
        raise RuntimeError(f"{failures} runs failed for {variant}/{instance}")


def all_ok_rows(output_root: Path, variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_root / variant).glob("*/raw_results.jsonl")):
        rows.extend(row for row in effective_rows(path) if row.get("status") == "ok")
    return rows


def geometric_mean(values: list[float]) -> float | None:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(statistics.mean(math.log(value) for value in values))


def positive_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0 or numerator < 0:
        return None
    return numerator / denominator


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
    ratios: list[float], *, samples: int, seed: int
) -> tuple[float | None, float | None]:
    if not ratios or samples <= 0:
        return None, None
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resample = [ratios[rng.randrange(len(ratios))] for _ in ratios]
        estimate = geometric_mean(resample)
        if estimate is not None:
            estimates.append(estimate)
    estimates.sort()
    if not estimates:
        return None, None
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def write_variant_outputs(
    output_root: Path,
    variant: str,
    rows: list[dict[str, Any]],
    expected_runs: int,
) -> None:
    variant_root = output_root / variant
    ordered = sorted(rows, key=lambda row: (str(row["instance"]), int(row["run"])))
    official_fields = [
        "Instance",
        "Fitness",
        "FunctionEvaluations",
        "StartTimes",
        "MachineAssignments",
        "WorkerAssignments",
        "UncertaintyParameters",
    ]
    official_rows = [
        {
            "Instance": row["instance"],
            "Fitness": row["fitness"],
            "FunctionEvaluations": row["function_evaluations"],
            "StartTimes": compact_json(row["start_times"]),
            "MachineAssignments": compact_json(row["machine_assignments"]),
            "WorkerAssignments": compact_json(row["worker_assignments"]),
            "UncertaintyParameters": compact_json(row["uncertainty_parameters"]),
        }
        for row in ordered
    ]
    atomic_csv(
        variant_root / "submission_scenario2.csv",
        official_fields,
        official_rows,
        delimiter=";",
    )
    run_fields = [
        "variant",
        "instance",
        "run",
        "seed",
        "fitness",
        "deterministic_makespan",
        "final_robust_stdev",
        "final_R",
        "function_evaluations",
        "raw_function_evaluations",
        "runtime_s",
        "generations",
    ]
    atomic_csv(
        variant_root / "run_results.csv",
        run_fields,
        ({field: row.get(field) for field in run_fields} for row in ordered),
    )
    atomic_json(
        variant_root / "submission_manifest.json",
        {
            "experiment": "scenario2_hpo_rl_factorial",
            "variant": variant,
            "expected_runs": expected_runs,
            "successful_runs": len(ordered),
            "complete": len(ordered) == expected_runs,
            "official_csv": str(variant_root / "submission_scenario2.csv"),
        },
    )


def summarize(args: argparse.Namespace) -> None:
    names = instance_names(args.instances_dir)
    configs = load_artifact_configs(args)
    expected_per_variant = len(names) * args.n_runs
    rows_by_variant = {variant: all_ok_rows(args.output_root, variant) for variant in VARIANTS}
    keyed = {
        variant: {(str(row["instance"]), int(row["run"])): row for row in rows}
        for variant, rows in rows_by_variant.items()
    }
    for variant, rows in rows_by_variant.items():
        write_variant_outputs(args.output_root, variant, rows, expected_per_variant)

    variant_summary: list[dict[str, Any]] = []
    variant_instance_summary: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = rows_by_variant[variant]
        fitness = [float(row["fitness"]) for row in rows]
        variant_summary.append(
            {
                "variant": variant,
                "successful_runs": len(rows),
                "missing_runs": expected_per_variant - len(rows),
                "expected_runs": expected_per_variant,
                "mean_fitness": statistics.mean(fitness) if fitness else None,
                "median_fitness": statistics.median(fitness) if fitness else None,
                "mean_runtime_s": statistics.mean(float(row["runtime_s"]) for row in rows)
                if rows
                else None,
                "mean_function_evaluations": statistics.mean(
                    int(row["function_evaluations"]) for row in rows
                )
                if rows
                else None,
                "mean_final_R": statistics.mean(float(row["final_R"]) for row in rows)
                if rows
                else None,
            }
        )
        for instance in names:
            instance_rows_for_variant = [
                row for row in rows if str(row["instance"]) == instance
            ]
            instance_fitness = [
                float(row["fitness"]) for row in instance_rows_for_variant
            ]
            variant_instance_summary.append(
                {
                    "variant": variant,
                    "instance": instance,
                    "successful_runs": len(instance_rows_for_variant),
                    "missing_runs": args.n_runs - len(instance_rows_for_variant),
                    "median_fitness": statistics.median(instance_fitness)
                    if instance_fitness
                    else None,
                    "mean_fitness": statistics.mean(instance_fitness)
                    if instance_fitness
                    else None,
                    "fitness_stdev": statistics.stdev(instance_fitness)
                    if len(instance_fitness) > 1
                    else 0.0 if instance_fitness else None,
                    "mean_runtime_s": statistics.mean(
                        float(row["runtime_s"]) for row in instance_rows_for_variant
                    )
                    if instance_rows_for_variant
                    else None,
                    "mean_function_evaluations": statistics.mean(
                        int(row["function_evaluations"])
                        for row in instance_rows_for_variant
                    )
                    if instance_rows_for_variant
                    else None,
                    "mean_final_robust_stdev": statistics.mean(
                        float(row["final_robust_stdev"])
                        for row in instance_rows_for_variant
                    )
                    if instance_rows_for_variant
                    else None,
                    "mean_final_R": statistics.mean(
                        float(row["final_R"]) for row in instance_rows_for_variant
                    )
                    if instance_rows_for_variant
                    else None,
                }
            )
    atomic_csv(args.output_root / "variant_summary.csv", list(variant_summary[0]), variant_summary)
    atomic_csv(
        args.output_root / "variant_instance_summary.csv",
        list(variant_instance_summary[0]),
        variant_instance_summary,
    )

    paired_run_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    comparison_instance_ratios: dict[str, dict[str, float]] = {}
    for comparison, baseline, candidate in COMPARISONS:
        common = sorted(set(keyed[baseline]) & set(keyed[candidate]))
        for instance, run in common:
            base = keyed[baseline][(instance, run)]
            cand = keyed[candidate][(instance, run)]
            if int(base["seed"]) != int(cand["seed"]):
                raise ValueError(
                    f"Seed mismatch in {comparison} for {instance} run {run}: "
                    f"{base['seed']} != {cand['seed']}"
                )
            if base.get("uncertainty_parameters") != cand.get("uncertainty_parameters"):
                raise ValueError(
                    f"Uncertainty mismatch in {comparison} for {instance} run {run}"
                )
            base_fitness = float(base["fitness"])
            candidate_fitness = float(cand["fitness"])
            paired_run_rows.append(
                {
                    "comparison": comparison,
                    "baseline": baseline,
                    "candidate": candidate,
                    "instance": instance,
                    "run": run,
                    "seed": base["seed"],
                    "baseline_fitness": base_fitness,
                    "candidate_fitness": candidate_fitness,
                    "fitness_ratio": candidate_fitness / base_fitness,
                    "improvement_percent": 100.0 * (1.0 - candidate_fitness / base_fitness),
                    "baseline_runtime_s": base["runtime_s"],
                    "candidate_runtime_s": cand["runtime_s"],
                    "baseline_function_evaluations": base["function_evaluations"],
                    "candidate_function_evaluations": cand["function_evaluations"],
                    "baseline_final_R": base["final_R"],
                    "candidate_final_R": cand["final_R"],
                }
            )

        ratios_by_instance: dict[str, float] = {}
        runtime_ratios: list[float] = []
        evaluation_ratios: list[float] = []
        robust_stdev_ratios: list[float] = []
        final_r_deltas: list[float] = []
        for instance in names:
            instance_keys = [key for key in common if key[0] == instance]
            base_values = [float(keyed[baseline][key]["fitness"]) for key in instance_keys]
            candidate_values = [float(keyed[candidate][key]["fitness"]) for key in instance_keys]
            if not base_values or not candidate_values:
                continue
            base_median = statistics.median(base_values)
            candidate_median = statistics.median(candidate_values)
            ratio = candidate_median / base_median
            base_runtime = statistics.mean(
                float(keyed[baseline][key]["runtime_s"]) for key in instance_keys
            )
            candidate_runtime = statistics.mean(
                float(keyed[candidate][key]["runtime_s"]) for key in instance_keys
            )
            base_evaluations = statistics.mean(
                int(keyed[baseline][key]["function_evaluations"])
                for key in instance_keys
            )
            candidate_evaluations = statistics.mean(
                int(keyed[candidate][key]["function_evaluations"])
                for key in instance_keys
            )
            base_robust_stdev = statistics.mean(
                float(keyed[baseline][key]["final_robust_stdev"])
                for key in instance_keys
            )
            candidate_robust_stdev = statistics.mean(
                float(keyed[candidate][key]["final_robust_stdev"])
                for key in instance_keys
            )
            base_final_r = statistics.mean(
                float(keyed[baseline][key]["final_R"]) for key in instance_keys
            )
            candidate_final_r = statistics.mean(
                float(keyed[candidate][key]["final_R"]) for key in instance_keys
            )
            runtime_ratio = positive_ratio(candidate_runtime, base_runtime)
            evaluation_ratio = positive_ratio(candidate_evaluations, base_evaluations)
            robust_stdev_ratio = positive_ratio(
                candidate_robust_stdev, base_robust_stdev
            )
            if runtime_ratio is not None and runtime_ratio > 0:
                runtime_ratios.append(runtime_ratio)
            if evaluation_ratio is not None and evaluation_ratio > 0:
                evaluation_ratios.append(evaluation_ratio)
            if robust_stdev_ratio is not None and robust_stdev_ratio > 0:
                robust_stdev_ratios.append(robust_stdev_ratio)
            final_r_deltas.append(candidate_final_r - base_final_r)
            ratios_by_instance[instance] = ratio
            instance_rows.append(
                {
                    "comparison": comparison,
                    "baseline": baseline,
                    "candidate": candidate,
                    "instance": instance,
                    "paired_runs": len(instance_keys),
                    "baseline_median_fitness": base_median,
                    "candidate_median_fitness": candidate_median,
                    "fitness_ratio": ratio,
                    "improvement_percent": 100.0 * (1.0 - ratio),
                    "baseline_mean_runtime_s": base_runtime,
                    "candidate_mean_runtime_s": candidate_runtime,
                    "runtime_ratio": runtime_ratio,
                    "baseline_mean_function_evaluations": base_evaluations,
                    "candidate_mean_function_evaluations": candidate_evaluations,
                    "function_evaluations_ratio": evaluation_ratio,
                    "baseline_mean_final_robust_stdev": base_robust_stdev,
                    "candidate_mean_final_robust_stdev": candidate_robust_stdev,
                    "final_robust_stdev_ratio": robust_stdev_ratio,
                    "baseline_mean_final_R": base_final_r,
                    "candidate_mean_final_R": candidate_final_r,
                    "final_R_delta": candidate_final_r - base_final_r,
                    "outcome": "win" if ratio < 1.0 else "loss" if ratio > 1.0 else "tie",
                }
            )
        comparison_instance_ratios[comparison] = ratios_by_instance
        ratios = list(ratios_by_instance.values())
        ci_low, ci_high = bootstrap_geomean_ci(
            ratios,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        comparison_rows.append(
            {
                "comparison": comparison,
                "baseline": baseline,
                "candidate": candidate,
                "paired_runs": len(common),
                "paired_instances": len(ratios),
                "geometric_mean_ratio": geometric_mean(ratios),
                "improvement_percent": (
                    100.0 * (1.0 - geometric_mean(ratios))
                    if geometric_mean(ratios) is not None
                    else None
                ),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "geometric_mean_runtime_ratio": geometric_mean(runtime_ratios),
                "geometric_mean_function_evaluations_ratio": geometric_mean(
                    evaluation_ratios
                ),
                "geometric_mean_final_robust_stdev_ratio": geometric_mean(
                    robust_stdev_ratios
                ),
                "mean_final_R_delta": statistics.mean(final_r_deltas)
                if final_r_deltas
                else None,
                "wins": sum(ratio < 1.0 for ratio in ratios),
                "ties": sum(ratio == 1.0 for ratio in ratios),
                "losses": sum(ratio > 1.0 for ratio in ratios),
            }
        )

    atomic_csv(
        args.output_root / "paired_run_comparison.csv",
        list(paired_run_rows[0]) if paired_run_rows else ["comparison"],
        paired_run_rows,
    )
    atomic_csv(
        args.output_root / "instance_comparison.csv",
        list(instance_rows[0]) if instance_rows else ["comparison"],
        instance_rows,
    )
    atomic_csv(
        args.output_root / "comparison_summary.csv",
        list(comparison_rows[0]) if comparison_rows else ["comparison"],
        comparison_rows,
    )

    c0_rl = comparison_instance_ratios.get("rl_effect_on_competition", {})
    hpo_rl = comparison_instance_ratios.get("rl_effect_on_hpo", {})
    interaction_instances = sorted(set(c0_rl) & set(hpo_rl))
    interaction_ratios = [hpo_rl[name] / c0_rl[name] for name in interaction_instances]
    interaction_low, interaction_high = bootstrap_geomean_ci(
        interaction_ratios,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed + 1,
    )
    interaction = {
        "definition": "(hpo_with_rl / hpo_no_rl) / (competition_with_rl / competition_no_rl)",
        "paired_instances": len(interaction_instances),
        "geometric_mean_ratio_of_ratios": geometric_mean(interaction_ratios),
        "mean_log_interaction": statistics.mean(math.log(x) for x in interaction_ratios)
        if interaction_ratios
        else None,
        "bootstrap_ci_low": interaction_low,
        "bootstrap_ci_high": interaction_high,
    }
    atomic_json(
        args.output_root / "comparison_report.json",
        {
            "experiment": "scenario2_hpo_rl_factorial",
            "primary_unit": "instance",
            "primary_metric": "geometric mean of candidate/baseline median makespan ratios",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "expected_runs_per_variant": expected_per_variant,
            "variant_summary": variant_summary,
            "comparisons": comparison_rows,
            "interaction": interaction,
        },
    )
    atomic_json(
        args.output_root / "experiment_manifest.json",
        {
            "experiment": "scenario2_hpo_rl_factorial",
            "variants": list(VARIANTS),
            "comparisons": [list(item) for item in COMPARISONS],
            "instance_count": len(names),
            "n_runs": args.n_runs,
            "final_simulations": args.final_simulations,
            "max_function_evaluations": args.max_function_evaluations,
            "time_limit_s": args.time_limit_s,
            "uncertainty_json": str(args.uncertainty_json),
            "artifact_configs": configs,
        },
    )
    print(f"Wrote factorial comparison outputs to {args.output_root}")


def print_tasks(args: argparse.Namespace) -> None:
    for index, (variant, instance) in enumerate(task_specs(args.instances_dir)):
        print(f"{index}\t{variant}\t{instance}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run-task", "summarize", "print-tasks"))
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--instances-dir", type=Path, default=Path("instances/fjssp-w"))
    parser.add_argument(
        "--uncertainty-json", type=Path, default=Path("config/scenario2_uncertainty.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/hpo_rl_factorial_scenario2")
    )
    parser.add_argument(
        "--competition-manifest",
        type=Path,
        default=Path("results/scenario2_submission/submission_manifest.json"),
    )
    parser.add_argument(
        "--competition-config",
        type=Path,
        default=Path(
            "results/hpo_scenario2/screening/results/"
            "C0_repo_reproduction/config.json"
        ),
        help="Stored C0 config used when the legacy submission manifest has no ga_config.",
    )
    parser.add_argument(
        "--hpo-final-root", type=Path, default=Path("results/hpo_scenario2/final")
    )
    parser.add_argument(
        "--rl-hpo-root", type=Path, default=Path("results/hpo_rl_scenario2")
    )
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--final-simulations", type=int, default=50)
    parser.add_argument("--max-function-evaluations", type=int, default=5_000_000)
    parser.add_argument("--time-limit-s", type=int, default=129_600)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--simulation-workers", type=int, default=2)
    parser.add_argument("--surrogate-n-jobs", type=int, default=2)
    parser.add_argument("--progress-interval-evaluations", type=int, default=50_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-failed-runs", action="store_true")
    args = parser.parse_args()
    if args.n_runs != 10:
        raise ValueError("Competition protocol requires --n-runs 10")
    if args.final_simulations != 50:
        raise ValueError("Competition protocol requires --final-simulations 50")
    if args.max_function_evaluations != 5_000_000:
        raise ValueError("Competition protocol requires --max-function-evaluations 5000000")
    if args.workers < 1 or args.simulation_workers < 1 or args.surrogate_n_jobs == 0:
        raise ValueError("Worker counts must be positive and surrogate-n-jobs must not be zero")
    return args


def main() -> int:
    args = parse_args()
    if args.command == "run-task":
        run_task(args)
    elif args.command == "summarize":
        summarize(args)
    elif args.command == "print-tasks":
        print_tasks(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
