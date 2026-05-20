#!/usr/bin/env python3
"""Generate reproducible Scenario-2 uncertainty parameters."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.benchmark_parser import WorkerBenchmarkParser
from util.uncertainty import create_uncertainty_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, default=Path("instances/fjssp-w"))
    parser.add_argument("--output", type=Path, default=Path("config/scenario2_uncertainty.json"))
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--factor", type=float, default=10.0)
    parser.add_argument("--offset", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print a summary without writing JSON.")
    return parser.parse_args()


def worker_count(encoding: Any) -> int:
    durations = encoding.durations()
    shape = getattr(durations, "shape", None)
    if shape is not None and len(shape) >= 3:
        return int(shape[2])
    return len(durations[0][0])


def seeded_uncertainty(n_workers: int, seed: int, factor: float, offset: float) -> list[list[float]]:
    state = random.getstate()
    try:
        random.seed(seed)
        params = create_uncertainty_vector(n_workers, factor=factor, offset=offset)
    finally:
        random.setstate(state)
    return [[float(value) for value in row] for row in params]


def validate_payload(payload: dict[str, Any], instance_files: list[Path]) -> None:
    metadata = payload["metadata"]
    n_runs = int(metadata["n_runs"])
    factor = float(metadata["factor"])
    expected_offset = float(metadata["offset"])
    instances = payload["instances"]

    if len(instances) != len(instance_files):
        raise ValueError(f"Expected {len(instance_files)} instances, found {len(instances)} in payload.")

    expected_names = {path.name for path in instance_files}
    if set(instances) != expected_names:
        missing = sorted(expected_names - set(instances))
        extra = sorted(set(instances) - expected_names)
        raise ValueError(f"Instance mismatch. Missing={missing}, extra={extra}")

    for instance_name, instance_data in instances.items():
        n_workers = int(instance_data["n_workers"])
        runs = instance_data["runs"]
        if len(runs) != n_runs:
            raise ValueError(f"{instance_name}: expected {n_runs} runs, found {len(runs)}.")
        for run_idx in range(1, n_runs + 1):
            run_key = str(run_idx)
            if run_key not in runs:
                raise ValueError(f"{instance_name}: missing run {run_key}.")
            params = runs[run_key]["uncertainty_parameters"]
            if len(params) != n_workers:
                raise ValueError(f"{instance_name} run {run_key}: expected {n_workers} worker rows.")
            for row in params:
                if not isinstance(row, list) or len(row) != 3:
                    raise ValueError(f"{instance_name} run {run_key}: invalid parameter row {row!r}.")
                alpha, beta, offset = row
                if not all(isinstance(value, (int, float)) for value in row):
                    raise ValueError(f"{instance_name} run {run_key}: non-numeric parameter row {row!r}.")
                if not all(math.isfinite(float(value)) for value in row):
                    raise ValueError(f"{instance_name} run {run_key}: non-finite parameter row {row!r}.")
                if not math.isclose(float(beta), factor * float(alpha), rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"{instance_name} run {run_key}: beta != factor * alpha.")
                if not math.isclose(float(offset), expected_offset, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"{instance_name} run {run_key}: offset mismatch.")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    instances_dir = args.instances_dir
    instance_files = sorted(instances_dir.glob("*.fjs"))
    if not instance_files:
        raise ValueError(f"No .fjs files found in {instances_dir}.")

    parser = WorkerBenchmarkParser()
    instances: dict[str, Any] = {}
    for instance_path in instance_files:
        encoding = parser.parse_benchmark(str(instance_path))
        n_workers = worker_count(encoding)
        runs = {}
        for run_index_zero_based in range(args.n_runs):
            seed = args.base_seed + run_index_zero_based
            runs[str(run_index_zero_based + 1)] = {
                "seed": seed,
                "uncertainty_parameters": seeded_uncertainty(
                    n_workers,
                    seed=seed,
                    factor=args.factor,
                    offset=args.offset,
                ),
            }
        instances[instance_path.name] = {"n_workers": n_workers, "runs": runs}

    payload = {
        "metadata": {
            "scenario": 2,
            "source": "util.uncertainty.create_uncertainty_vector",
            "n_runs": args.n_runs,
            "base_seed": args.base_seed,
            "factor": args.factor,
            "offset": args.offset,
            "seed_scheme": "seed = base_seed + run_index_zero_based",
            "instances_dir": str(instances_dir),
        },
        "instances": instances,
    }
    validate_payload(payload, instance_files)
    return payload


def main() -> int:
    args = parse_args()
    try:
        payload = build_payload(args)
        if args.dry_run:
            print(f"Dry run OK: generated {len(payload['instances'])} instances x {args.n_runs} runs.")
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"Wrote {args.output} with {len(payload['instances'])} instances x {args.n_runs} runs.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
