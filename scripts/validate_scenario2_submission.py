#!/usr/bin/env python3
"""Validate Scenario-2 submission CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solver.GA.wfjssp_ga import is_simulatable_schedule
from util.benchmark_parser import WorkerBenchmarkParser
from util.evaluation import makespan


OFFICIAL_FIELDS = [
    "Instance",
    "Fitness",
    "FunctionEvaluations",
    "StartTimes",
    "MachineAssignments",
    "WorkerAssignments",
    "UncertaintyParameters",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, default=Path("instances/fjssp-w"))
    parser.add_argument("--uncertainty-json", type=Path, default=Path("config/scenario2_uncertainty.json"))
    parser.add_argument("--submission-csv", type=Path, default=Path("results/scenario2_submission/submission_scenario2.csv"))
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("results/scenario2_submission/submission_scenario2_with_metadata.csv"),
    )
    parser.add_argument("--expected-runs", type=int, default=10)
    parser.add_argument("--max-function-evaluations", type=int, default=5_000_000)
    parser.add_argument("--dry-run", action="store_true", help="Run all validations without modifying files.")
    parser.add_argument("--skip-simulatable-check", action="store_true")
    return parser.parse_args()


def read_semicolon_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        return list(reader.fieldnames or []), list(reader)


def parse_json_list(value: str, field: str, errors: list[str], context: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        errors.append(f"{context}: {field} is not valid JSON: {exc}")
        return None
    if not isinstance(parsed, list):
        errors.append(f"{context}: {field} is not a list.")
    return parsed


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def worker_count(encoding: Any) -> int:
    durations = encoding.durations()
    shape = getattr(durations, "shape", None)
    if shape is not None and len(shape) >= 3:
        return int(shape[2])
    return len(durations[0][0])


def validate() -> list[str]:
    args = parse_args()
    errors: list[str] = []

    if not args.submission_csv.exists():
        errors.append(f"Submission CSV does not exist: {args.submission_csv}")
        return errors
    if not args.metadata_csv.exists():
        errors.append(f"Metadata CSV does not exist: {args.metadata_csv}")
        return errors
    if not args.uncertainty_json.exists():
        errors.append(f"Uncertainty JSON does not exist: {args.uncertainty_json}")
        return errors

    official_fields, official_rows = read_semicolon_csv(args.submission_csv)
    metadata_fields, metadata_rows = read_semicolon_csv(args.metadata_csv)
    if official_fields != OFFICIAL_FIELDS:
        errors.append(f"Official CSV columns mismatch: expected {OFFICIAL_FIELDS}, got {official_fields}")
    if "Run" not in metadata_fields or "Seed" not in metadata_fields:
        errors.append("Metadata CSV must contain Run and Seed columns.")
    if len(official_rows) != len(metadata_rows):
        errors.append(f"Official and metadata row counts differ: {len(official_rows)} != {len(metadata_rows)}")

    with args.uncertainty_json.open("r", encoding="utf-8") as fh:
        uncertainty_payload = json.load(fh)

    instance_files = sorted(args.instances_dir.glob("*.fjs"))
    instance_names = {path.name for path in instance_files}
    if len(instance_files) != 30:
        errors.append(f"Expected 30 .fjs instances, found {len(instance_files)} in {args.instances_dir}.")

    expected_total = len(instance_files) * args.expected_runs
    if len(official_rows) != expected_total:
        errors.append(f"Expected {expected_total} submission rows, found {len(official_rows)}.")

    counts = Counter(row.get("Instance", "") for row in official_rows)
    if set(counts) != instance_names:
        errors.append(
            f"Submission instance set mismatch. Missing={sorted(instance_names - set(counts))}, "
            f"extra={sorted(set(counts) - instance_names)}"
        )
    for instance_name in sorted(instance_names):
        if counts[instance_name] != args.expected_runs:
            errors.append(f"{instance_name}: expected {args.expected_runs} rows, found {counts[instance_name]}.")

    parser = WorkerBenchmarkParser()
    encodings = {path.name: parser.parse_benchmark(str(path)) for path in instance_files}

    paired_rows = zip(official_rows, metadata_rows)
    for index, (row, metadata) in enumerate(paired_rows, start=1):
        context = f"row {index} ({row.get('Instance', '<missing>')})"
        instance = row.get("Instance", "")
        if metadata.get("Instance") != instance:
            errors.append(f"{context}: metadata Instance does not match official row.")
        if instance not in encodings:
            continue

        try:
            run = int(metadata.get("Run", ""))
        except ValueError:
            errors.append(f"{context}: metadata Run is not an integer.")
            run = -1
        try:
            int(metadata.get("Seed", ""))
        except ValueError:
            errors.append(f"{context}: metadata Seed is not an integer.")

        try:
            fitness = float(row.get("Fitness", "nan"))
            if not math.isfinite(fitness) or fitness <= 0:
                errors.append(f"{context}: Fitness must be finite and > 0.")
        except ValueError:
            errors.append(f"{context}: Fitness is not numeric.")

        try:
            function_evaluations = int(row.get("FunctionEvaluations", ""))
            if function_evaluations > args.max_function_evaluations:
                errors.append(f"{context}: FunctionEvaluations exceeds {args.max_function_evaluations}.")
            if function_evaluations < 0:
                errors.append(f"{context}: FunctionEvaluations is negative.")
        except ValueError:
            errors.append(f"{context}: FunctionEvaluations is not an integer.")

        start_times = parse_json_list(row.get("StartTimes", ""), "StartTimes", errors, context)
        machines = parse_json_list(row.get("MachineAssignments", ""), "MachineAssignments", errors, context)
        workers = parse_json_list(row.get("WorkerAssignments", ""), "WorkerAssignments", errors, context)
        params = parse_json_list(row.get("UncertaintyParameters", ""), "UncertaintyParameters", errors, context)
        if any(value is None for value in (start_times, machines, workers, params)):
            continue

        encoding = encodings[instance]
        n_operations = len(encoding.job_sequence())
        n_workers = worker_count(encoding)
        if len(start_times) != n_operations:
            errors.append(f"{context}: StartTimes length {len(start_times)} != {n_operations}.")
        if len(machines) != n_operations:
            errors.append(f"{context}: MachineAssignments length {len(machines)} != {n_operations}.")
        if len(workers) != n_operations:
            errors.append(f"{context}: WorkerAssignments length {len(workers)} != {n_operations}.")
        if len(params) != n_workers:
            errors.append(f"{context}: UncertaintyParameters length {len(params)} != {n_workers}.")

        if not all(is_number(value) for value in start_times):
            errors.append(f"{context}: StartTimes contains non-finite or non-numeric values.")
        if not all(isinstance(value, int) for value in machines):
            errors.append(f"{context}: MachineAssignments must contain integers.")
        if not all(isinstance(value, int) for value in workers):
            errors.append(f"{context}: WorkerAssignments must contain integers.")
        for param_index, param_row in enumerate(params):
            if not isinstance(param_row, list) or len(param_row) != 3:
                errors.append(f"{context}: uncertainty row {param_index} must have length 3.")
            elif not all(is_number(value) for value in param_row):
                errors.append(f"{context}: uncertainty row {param_index} contains non-finite values.")

        if run > 0:
            try:
                expected_params = uncertainty_payload["instances"][instance]["runs"][str(run)]["uncertainty_parameters"]
                if params != expected_params:
                    errors.append(f"{context}: uncertainty parameters do not match JSON for run {run}.")
            except KeyError:
                errors.append(f"{context}: no uncertainty JSON entry for run {run}.")

        if len(start_times) == len(machines) == len(workers) == n_operations:
            durations = encoding.durations()
            end_times = []
            assignment_valid = True
            for op_index, (start, machine, worker) in enumerate(zip(start_times, machines, workers)):
                try:
                    duration = durations[op_index][machine][worker]
                except Exception:
                    errors.append(f"{context}: invalid machine/worker index at operation {op_index}.")
                    assignment_valid = False
                    break
                if duration <= 0:
                    errors.append(f"{context}: machine/worker assignment has zero duration at operation {op_index}.")
                    assignment_valid = False
                    break
                end_times.append(float(start) + float(duration))
            if assignment_valid:
                if not args.skip_simulatable_check and not is_simulatable_schedule(
                    start_times,
                    end_times,
                    machines,
                    workers,
                    encoding.job_sequence(),
                ):
                    errors.append(f"{context}: schedule is not simulatable.")
                try:
                    deterministic = float(makespan(start_times, machines, workers, durations))
                    if not math.isfinite(deterministic) or deterministic <= 0:
                        errors.append(f"{context}: deterministic makespan is invalid.")
                except Exception as exc:
                    errors.append(f"{context}: deterministic makespan failed: {exc}")

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Validation OK: 300 Scenario-2 submission rows are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
