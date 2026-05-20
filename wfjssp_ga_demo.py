"""WFJSSP-GA demo workflow as a Python script.

This script is a cleaned version of ``wfjssp_ga_demo.ipynb``.
Notebook display output, local plots and inline statistics were removed.
The solving, final evaluation and file export logic follows the notebook.
"""

from pathlib import Path
from datetime import datetime
import json
import os
import random

import numpy as np
import pandas as pd

from solver.GA.wfjssp_ga import build_ga_from_worker_encoding, is_simulatable_schedule
from util.benchmark_parser import WorkerBenchmarkParser
from util.evaluation import makespan, translate
from util.graph import run_n_simulations
from util.uncertainty import create_uncertainty_vector


INSTANCES_DIR = Path("instances/fjssp-w")
RESULTS_DIR = Path("results/wfjssp_ga_competition")

SCENARIO = 2  # 1 = deterministisch, 2 = Unsicherheit
STRICT_COMPETITION_MODE = False

# Optional: Datei mit offiziellen Unsicherheitsparametern.
# Erwartetes Format: {"instance_name.fjs": [[alpha, beta, offset], ...], ...}
OFFICIAL_UNCERTAINTY_FILE = None

# Standardverhalten der Demo: Parameter werden lokal erzeugt.
ALLOW_DEMO_UNCERTAINTY_FALLBACK = True
DEMO_UNCERTAINTY_SEED = 123
DEMO_UNCERTAINTY_FACTOR = 10.0
DEMO_UNCERTAINTY_OFFSET = 1.0
UNCERTAINTY_SOURCE = "worker"

# Competition-relevante Wiederholungen
N_INDEPENDENT_RUNS = 1
INTERNAL_EVAL_SIMULATIONS = 10
FINAL_EVAL_SIMULATIONS = 50

# Zum schnellen Testen kann die Auswahl reduziert werden.
# SELECTED_INSTANCES = sorted(p.name for p in INSTANCES_DIR.glob("*.fjs"))
SELECTED_INSTANCES = ["6_Fattahi_20_workers.fjs"]

RUN_SEEDS = [1000 + i for i in range(N_INDEPENDENT_RUNS)]

# RL steuert die Gewichtung der drei Mutationsoperatoren:
# [Sequenzmutation, Maschinenmutation, Workermutation]
ENABLE_RL_MUTATION_CONTROL = False

GA_CONFIG = {
    "population_size": 200, 
    "offspring_amount": 1000, 
    "use_surrogate_evaluation": True,
    "surrogate_warmup_real_candidates": 1000,
    "surrogate_top_fraction": 0.02,
    "surrogate_uncertain_fraction": 0.005,
    "surrogate_random_fraction": 0.005,
    "surrogate_min_real_per_generation": 5,
    "surrogate_retrain_interval_real_candidates": 100, # bestimmt die Anzahl echter Kandidaten zwischen zwei Retrainings, wenn kein Wachstum (surrogate_retrain_interval_growth_samples) vorliegt
    "surrogate_n_estimators": 300, 
    "surrogate_min_samples_leaf": 3,
    "surrogate_max_features": "sqrt",
    "surrogate_n_jobs": -1,
    "surrogate_max_training_samples": 5_000,
    "surrogate_retrain_interval_growth_samples": 5_000,
    "surrogate_retrain_interval_growth_factor": 2.0,
    "surrogate_max_retrain_interval_real_candidates": 1_000,  # bestimmt die maximale Anzahl echter Kandidaten zwischen zwei Retrainings, unabhängig von Wachstum
    "elitism_rate": 0.1,
    "restart_generations": 800,
    "enable_rl_mutation_control": ENABLE_RL_MUTATION_CONTROL,
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

RUN_CONFIG = {
    "max_generations": None,
    "time_limit_s": 22000, 
    "max_function_evaluations": None,
    "progress_interval_evaluations": 50_000,
    "keep_multiple": False,
    "do_restart": False,
}

# Optionales Weights-&-Biases-Logging. Mit WANDB_MODE="offline" kann lokal
# geloggt und spaeter per `wandb sync` hochgeladen werden.
USE_WANDB = True
WANDB_PROJECT = "wfjssp-ga"
WANDB_ENTITY = None
WANDB_MODE = os.environ.get("WANDB_MODE")
WANDB_LOG_ARTIFACTS = True
EXPERIMENT_NAME = None


def load_official_uncertainty_map(path):
    if path is None:
        return None
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_uncertainty_parameters(instance_name, encoding, uncertainty_map):
    if SCENARIO != 2:
        return None, "disabled"

    if uncertainty_map is not None:
        if instance_name not in uncertainty_map:
            raise KeyError(f"No uncertainty parameters found for {instance_name}")
        return uncertainty_map[instance_name], "official"

    if STRICT_COMPETITION_MODE:
        raise ValueError(
            "Scenario 2 in strict competition mode requires OFFICIAL_UNCERTAINTY_FILE with provided parameters."
        )

    if not ALLOW_DEMO_UNCERTAINTY_FALLBACK:
        return None, "missing"

    random.seed(DEMO_UNCERTAINTY_SEED)
    resource_count = encoding.durations().shape[2] if UNCERTAINTY_SOURCE == "worker" else encoding.durations().shape[1]
    params = create_uncertainty_vector(
        resource_count,
        factor=DEMO_UNCERTAINTY_FACTOR,
        offset=DEMO_UNCERTAINTY_OFFSET,
    )
    return params, "demo_fallback"


def safe_workload_balance(machine_assignments, worker_assignments, durations):
    if not worker_assignments:
        return 0.0

    n_workers = max(worker_assignments) + 1
    working_time = [0] * n_workers
    for i in range(len(worker_assignments)):
        worker = int(worker_assignments[i])
        machine = int(machine_assignments[i])
        working_time[worker] += durations[i][machine][worker]

    mean_working_time = np.mean(working_time)
    result = 0.0
    for worker_time in working_time:
        result += (mean_working_time - worker_time) ** 2
    return float(result)


def solve_instance_with_ga(encoding, seed, uncertainty_parameters=None, history_callback=None):
    ga_kwargs = dict(GA_CONFIG)
    ga_kwargs["seed"] = seed
    ga_kwargs["rl_seed"] = seed

    if SCENARIO == 2:
        ga_kwargs.update(
            {
                "use_stochastic_evaluation": True,
                "uncertainty_parameters": uncertainty_parameters,
                "n_simulations": INTERNAL_EVAL_SIMULATIONS,
            }
        )

    ga = build_ga_from_worker_encoding(encoding, **ga_kwargs)
    result = ga.run(**RUN_CONFIG, history_callback=history_callback)
    best = result["best"]

    start_times, machines, workers = translate(
        best.sequence,
        best.assignments,
        best.workers,
        encoding.durations(),
    )
    deterministic_makespan = float(makespan(start_times, machines, workers, encoding.durations()))
    worker_balance = safe_workload_balance(machines, workers, encoding.durations())

    raw_function_evaluations = int(result["function_evaluations"])
    competition_function_evaluations = raw_function_evaluations

    row = {
        "seed": seed,
        "runtime_s": float(result["runtime_s"]),
        "generations": int(result["generations"]),
        "raw_function_evaluations": raw_function_evaluations,
        "function_evaluations": competition_function_evaluations,
        "restarts": int(result["restarts"]),
        "deterministic_makespan": deterministic_makespan,
        "workload_balance": worker_balance,
        "sequence": list(best.sequence),
        "machines": list(best.assignments),
        "workers": list(best.workers),
        "start_times": list(start_times),
        "rl_enabled": bool(ga_kwargs.get("enable_rl_mutation_control", False)),
        "history": [dict(point) for point in result.get("history", [])],
    }

    if SCENARIO == 2:
        row["internal_robust_makespan"] = float(best.fitness.get("makespan"))
        row["internal_robust_stdev"] = float(best.fitness.get("robust_makespan_stdev"))
        row["internal_R"] = float(best.fitness.get("R"))
    else:
        row["internal_robust_makespan"] = None
        row["internal_robust_stdev"] = None
        row["internal_R"] = None

    return row


def evaluate_final_solution(row, encoding, uncertainty_parameters=None):
    if SCENARIO == 1:
        row["final_objective"] = row["deterministic_makespan"]
        row["final_robust_makespan"] = None
        row["final_robust_stdev"] = None
        row["final_R"] = None
        row["final_eval_simulations"] = 0
        row["final_eval_status"] = "deterministic"
        return row

    end_times = [
        row["start_times"][i] + encoding.durations()[i][int(row["machines"][i])][int(row["workers"][i])]
        for i in range(len(row["start_times"]))
    ]
    if not is_simulatable_schedule(
        row["start_times"],
        end_times,
        row["machines"],
        row["workers"],
        encoding.job_sequence(),
    ):
        row["final_objective"] = float("inf")
        row["final_robust_makespan"] = float("inf")
        row["final_robust_stdev"] = None
        row["final_R"] = None
        row["final_eval_simulations"] = 0
        row["final_simulation_results"] = []
        row["final_eval_status"] = "rejected_unsimulatable"
        return row

    try:
        results, robust_makespan, robust_makespan_stdev, R = run_n_simulations(
            row["start_times"],
            end_times,
            row["machines"],
            row["workers"],
            encoding.job_sequence(),
            encoding.durations(),
            uncertainty_parameters,
            FINAL_EVAL_SIMULATIONS,
            uncertainty_source=UNCERTAINTY_SOURCE,
            processing_times=True,
        )
    except (RecursionError, Exception):
        row["final_objective"] = float("inf")
        row["final_robust_makespan"] = float("inf")
        row["final_robust_stdev"] = None
        row["final_R"] = None
        row["final_eval_simulations"] = 0
        row["final_simulation_results"] = []
        row["final_eval_status"] = "simulation_error"
        return row

    row["final_objective"] = float(robust_makespan)
    row["final_robust_makespan"] = float(robust_makespan)
    row["final_robust_stdev"] = float(robust_makespan_stdev)
    row["final_R"] = float(R)
    row["final_eval_simulations"] = FINAL_EVAL_SIMULATIONS
    row["final_simulation_results"] = [float(x) for x in results]
    row["final_eval_status"] = "ok"
    return row


def drop_large_columns(df):
    cols = [
        "sequence",
        "machines",
        "workers",
        "start_times",
        "history",
        "final_simulation_results",
    ]
    keep = [c for c in df.columns if c not in cols]
    return df[keep].copy()


def get_wandb_module():
    if not USE_WANDB:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Installiere wandb oder setze USE_WANDB = False.") from exc
    return wandb


def init_wandb_run(instance_name=None, run_idx=None, seed=None, run_kind="ga-run"):
    wandb = get_wandb_module()
    if wandb is None:
        return None

    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        "scenario": SCENARIO,
        "experiment": EXPERIMENT_NAME,
        "started_at": started_at,
        "instance": instance_name,
        "run": run_idx,
        "seed": seed,
        "selected_instances": SELECTED_INSTANCES,
        "n_independent_runs": N_INDEPENDENT_RUNS,
        "internal_eval_simulations": INTERNAL_EVAL_SIMULATIONS,
        "final_eval_simulations": FINAL_EVAL_SIMULATIONS,
        "uncertainty_source": UNCERTAINTY_SOURCE,
        **{f"ga/{key}": value for key, value in GA_CONFIG.items()},
        **{f"run/{key}": value for key, value in RUN_CONFIG.items()},
    }
    name_parts = [run_kind, started_at]
    if EXPERIMENT_NAME:
        name_parts.append(str(EXPERIMENT_NAME))
    if instance_name is not None:
        name_parts.append(Path(instance_name).stem)
    if run_idx is not None:
        name_parts.append(f"run-{run_idx}")
    if seed is not None:
        name_parts.append(f"seed-{seed}")

    init_kwargs = {
        "project": WANDB_PROJECT,
        "entity": WANDB_ENTITY,
        "name": "__".join(name_parts),
        "job_type": run_kind,
        "config": config,
    }
    if WANDB_MODE:
        init_kwargs["mode"] = WANDB_MODE

    run = wandb.init(**init_kwargs)
    if hasattr(run, "define_metric"):
        run.define_metric("fitness_by_evaluation/function_evaluations")
        run.define_metric(
            "fitness_by_evaluation/*",
            step_metric="fitness_by_evaluation/function_evaluations",
        )
        run.define_metric("fitness_by_generation/generation")
        run.define_metric(
            "fitness_by_generation/*",
            step_metric="fitness_by_generation/generation",
        )
        run.define_metric("surrogate/function_evaluations")
        run.define_metric("surrogate/*", step_metric="surrogate/function_evaluations")
    return run


def log_wandb_history_point(wandb_run, point):
    if wandb_run is None:
        return

    mutation_mix = point.get("mutation_mix") or [None, None, None]
    payload = {
        "fitness_by_evaluation/function_evaluations": int(point["function_evaluations"]),
        "fitness_by_evaluation/generation": int(point["generation"]),
        "fitness_by_evaluation/best_makespan": float(point["best_makespan"]),
        "fitness_by_evaluation/overall_best_makespan": float(point["overall_best_makespan"]),
        "fitness_by_generation/generation": int(point["generation"]),
        "fitness_by_generation/function_evaluations": int(point["function_evaluations"]),
        "fitness_by_generation/best_makespan": float(point["best_makespan"]),
        "fitness_by_generation/overall_best_makespan": float(point["overall_best_makespan"]),
        "ga/mutation_probability": float(point["mutation_probability"]),
        "ga/runtime_s": float(point["runtime_s"]),
        "ga/population_size": int(point["population_size"]),
        "ga/offspring_amount": int(point["offspring_amount"]),
        "ga/restarts": int(point["restarts"]),
        "ga/mix_sequence": mutation_mix[0],
        "ga/mix_machine": mutation_mix[1],
        "ga/mix_worker": mutation_mix[2],
        "ga/seq_mutations": int(point.get("seq_mutations", 0)),
        "ga/machine_mutations": int(point.get("machine_mutations", 0)),
        "ga/worker_mutations": int(point.get("worker_mutations", 0)),
        "surrogate/function_evaluations": int(point["function_evaluations"]),
        "surrogate/ready": bool(point.get("surrogate_ready", False)),
        "surrogate/samples": int(point.get("surrogate_samples", 0)),
        "surrogate/fit_count": int(point.get("surrogate_fit_count", 0)),
        "surrogate/predictions_total": int(point.get("surrogate_predictions", 0)),
        "surrogate/real_candidate_evaluations_total": int(
            point.get("surrogate_real_candidate_evaluations", 0)
        ),
        "surrogate/batch_predictions": int(point.get("surrogate_batch_predictions", 0)),
        "surrogate/batch_real_evaluations": int(point.get("surrogate_batch_real_evaluations", 0)),
        "surrogate/batch_surrogate_evaluations": int(
            point.get("surrogate_batch_surrogate_evaluations", 0)
        ),
        "surrogate/batch_real_fraction": float(point.get("surrogate_batch_real_fraction", 0.0)),
        "surrogate/batch_mean_uncertainty_R": float(
            point.get("surrogate_batch_mean_uncertainty_R", 0.0)
        ),
        "surrogate/batch_mae_R": point.get("surrogate_batch_mae_R"),
        "surrogate/batch_rmse_R": point.get("surrogate_batch_rmse_R"),
        "surrogate/batch_mae_robust_makespan": point.get("surrogate_batch_mae_robust_makespan"),
        "surrogate/batch_score_bias": point.get("surrogate_batch_score_bias"),
        "surrogate/batch_interval_coverage": point.get("surrogate_batch_interval_coverage"),
        "surrogate/batch_conservative_score_rate": point.get(
            "surrogate_batch_conservative_score_rate"
        ),
        "surrogate/batch_spearman_score": point.get("surrogate_batch_spearman_score"),
        "surrogate/validation_count": int(point.get("surrogate_validation_count", 0)),
        "surrogate/cumulative_mae_R": point.get("surrogate_cumulative_mae_R"),
        "surrogate/cumulative_rmse_R": point.get("surrogate_cumulative_rmse_R"),
        "surrogate/cumulative_interval_coverage": point.get(
            "surrogate_cumulative_interval_coverage"
        ),
        "surrogate/cumulative_conservative_score_rate": point.get(
            "surrogate_cumulative_conservative_score_rate"
        ),
    }
    wandb_run.log(payload)


def log_wandb_run_metrics(wandb_run, row, log_history=False):
    if wandb_run is None:
        return

    if log_history:
        for point in row.get("history", []):
            log_wandb_history_point(wandb_run, point)

    wandb_run.log(
        {
            "run/runtime_s": float(row["runtime_s"]),
            "run/generations": int(row["generations"]),
            "run/function_evaluations": int(row["function_evaluations"]),
            "run/deterministic_makespan": float(row["deterministic_makespan"]),
            "run/final_objective": float(row["final_objective"]),
            "run/final_R": row.get("final_R"),
            "run/final_eval_status": row.get("final_eval_status"),
        }
    )


def log_wandb_artifacts(wandb_run, artifact_paths):
    if wandb_run is None or not WANDB_LOG_ARTIFACTS:
        return

    wandb = get_wandb_module()
    artifact = wandb.Artifact("wfjssp-ga-results", type="results")
    for path in artifact_paths.values():
        artifact.add_file(str(path))
    wandb_run.log_artifact(artifact)


def build_results():
    uncertainty_map = load_official_uncertainty_map(OFFICIAL_UNCERTAINTY_FILE)
    parser = WorkerBenchmarkParser()

    all_rows = []
    instance_summaries = []

    for instance_name in SELECTED_INSTANCES:
        instance_path = INSTANCES_DIR / instance_name
        encoding = parser.parse_benchmark(str(instance_path))
        uncertainty_parameters, uncertainty_mode = get_uncertainty_parameters(instance_name, encoding, uncertainty_map)

        instance_rows = []
        for run_idx, seed in enumerate(RUN_SEEDS, start=1):
            wandb_run = init_wandb_run(instance_name, run_idx, seed, run_kind="ga-run")
            try:
                history_callback = (
                    None
                    if wandb_run is None
                    else lambda point, run=wandb_run: log_wandb_history_point(run, point)
                )
                row = solve_instance_with_ga(
                    encoding,
                    seed,
                    uncertainty_parameters,
                    history_callback=history_callback,
                )
                row["instance"] = instance_name
                row["run"] = run_idx
                row["scenario"] = SCENARIO
                row["uncertainty_mode"] = uncertainty_mode
                row = evaluate_final_solution(row, encoding, uncertainty_parameters)
                log_wandb_run_metrics(wandb_run, row)
            finally:
                if wandb_run is not None:
                    wandb_run.finish()
            instance_rows.append(row)
            all_rows.append(row)

        instance_df = pd.DataFrame(instance_rows)
        metric_col = "final_objective"
        best_idx = instance_df[metric_col].idxmin()
        best_row = instance_df.loc[best_idx]
        instance_summaries.append(
            {
                "instance": instance_name,
                "best_run": int(best_row["run"]),
                "best_seed": int(best_row["seed"]),
                "best_final_objective": float(best_row["final_objective"]),
                "best_deterministic_makespan": float(best_row["deterministic_makespan"]),
                "best_function_evaluations": int(best_row["function_evaluations"]),
                "mean_final_objective": float(instance_df[metric_col].mean()),
                "std_final_objective": float(instance_df[metric_col].std(ddof=0)),
            }
        )

    results_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(instance_summaries)
    return results_df, summary_df


def build_ranking(results_df):
    metric_col = "final_objective"
    secondary_col = "final_R" if SCENARIO == 2 else "workload_balance"

    ranking_df = results_df.copy()
    ranking_df["secondary_sort"] = ranking_df[secondary_col].fillna(np.inf)
    ranking_df = ranking_df.sort_values(["instance", metric_col, "secondary_sort", "function_evaluations"])
    ranking_df["rank_within_instance"] = ranking_df.groupby("instance").cumcount() + 1
    return ranking_df


def write_outputs(results_df, summary_df, ranking_df):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    compact_results_df = drop_large_columns(results_df)
    compact_results_df = compact_results_df.sort_values(["instance", "run"]).reset_index(drop=True)
    summary_df = summary_df.sort_values("instance").reset_index(drop=True)

    compact_results_path = RESULTS_DIR / "run_results.csv"
    summary_path = RESULTS_DIR / "instance_summary.csv"
    ranking_path = RESULTS_DIR / "ranking_results.csv"
    solutions_json_path = RESULTS_DIR / "solutions.json"

    compact_results_df.to_csv(compact_results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    ranking_df[
        [
            "instance",
            "run",
            "seed",
            "final_objective",
            "final_R",
            "workload_balance",
            "function_evaluations",
            "rank_within_instance",
        ]
    ].to_csv(ranking_path, index=False)

    solutions_payload = []
    for row in results_df.to_dict(orient="records"):
        solutions_payload.append(
            {
                "instance": row["instance"],
                "run": int(row["run"]),
                "seed": int(row["seed"]),
                "final_objective": float(row["final_objective"]),
                "deterministic_makespan": float(row["deterministic_makespan"]),
                "function_evaluations": int(row["function_evaluations"]),
                "sequence": [int(x) for x in row["sequence"]],
                "machines": [int(x) for x in row["machines"]],
                "workers": [int(x) for x in row["workers"]],
                "start_times": [int(x) for x in row["start_times"]],
            }
        )

    with solutions_json_path.open("w", encoding="utf-8") as fh:
        json.dump(solutions_payload, fh)

    return {
        "run_results": compact_results_path,
        "instance_summary": summary_path,
        "ranking_results": ranking_path,
        "solutions_json": solutions_json_path,
    }


def main():
    results_df, summary_df = build_results()
    ranking_df = build_ranking(results_df)
    artifact_paths = write_outputs(results_df, summary_df, ranking_df)
    summary_run = init_wandb_run(run_kind="summary")
    try:
        log_wandb_artifacts(summary_run, artifact_paths)
    finally:
        if summary_run is not None:
            summary_run.finish()
    return artifact_paths


if __name__ == "__main__":
    main()
