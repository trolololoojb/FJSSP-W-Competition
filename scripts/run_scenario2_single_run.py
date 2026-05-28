#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_scenario2_submission import (
    load_uncertainty,
    uncertainty_for,
    solve_run,
    load_completed_ok,
    append_jsonl,
    write_csv_outputs,
)
from util.benchmark_parser import WorkerBenchmarkParser


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--instances-dir", type=Path, default=Path("instances/fjssp-w"))
    parser.add_argument("--uncertainty-json", type=Path, default=Path("config/scenario2_uncertainty.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--internal-simulations", type=int, default=10)
    parser.add_argument("--final-simulations", type=int, default=50)
    parser.add_argument("--simulation-workers", type=int, default=1)
    parser.add_argument("--time-limit-s", type=int, default=129600)
    parser.add_argument("--max-function-evaluations", type=int, default=5_000_000)
    parser.add_argument("--surrogate-n-jobs", type=int, default=1)
    parser.add_argument("--disable-local-search", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_results.jsonl"

    completed = load_completed_ok(raw_path) if args.resume else {}
    key = (args.instance, args.run)

    out_args = SimpleNamespace(
        n_runs=1,
        workers=1,
        time_limit_s=args.time_limit_s,
        max_function_evaluations=args.max_function_evaluations,
        internal_simulations=args.internal_simulations,
        final_simulations=args.final_simulations,
        surrogate_n_jobs=args.surrogate_n_jobs,
        simulation_workers=args.simulation_workers,
        disable_local_search=args.disable_local_search,
    )

    if key in completed:
        row = completed[key]
        write_csv_outputs(args.output_dir, [row], out_args, args.uncertainty_json, expected_instances=1)
        print(f"SKIP existing: {args.instance} run {args.run}", flush=True)
        return 0

    instance_path = args.instances_dir / args.instance
    if not instance_path.exists():
        raise FileNotFoundError(instance_path)

    uncertainty_payload = load_uncertainty(args.uncertainty_json)
    seed, uncertainty_parameters = uncertainty_for(uncertainty_payload, args.instance, args.run)

    print(f"START {args.instance} run {args.run} seed={seed}", flush=True)
    print(
        f"time_limit_s={args.time_limit_s}, "
        f"simulation_workers={args.simulation_workers}, "
        f"surrogate_n_jobs={args.surrogate_n_jobs}",
        flush=True,
    )

    encoding = WorkerBenchmarkParser().parse_benchmark(str(instance_path))

    solve_args = SimpleNamespace(
        internal_simulations=args.internal_simulations,
        final_simulations=args.final_simulations,
        time_limit_s=args.time_limit_s,
        max_function_evaluations=args.max_function_evaluations,
        surrogate_n_jobs=args.surrogate_n_jobs,
        disable_local_search=args.disable_local_search,
        simulation_workers=args.simulation_workers,
    )

    row = solve_run(
        args.instance,
        args.run,
        seed,
        encoding,
        uncertainty_parameters,
        solve_args,
    )

    append_jsonl(raw_path, row)
    write_csv_outputs(args.output_dir, [row], out_args, args.uncertainty_json, expected_instances=1)

    print(
        f"DONE {args.instance} run {args.run} "
        f"fitness={row['fitness']} runtime_s={row['runtime_s']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
