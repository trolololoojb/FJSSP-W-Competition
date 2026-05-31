#!/usr/bin/env python3
"""Backfill hyperparameters.txt files for existing result directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_scenario2_submission import effective_ga_config, effective_run_config
from util.hyperparameters import result_hyperparameter_dir, write_hyperparameters_txt


def _read_first_jsonl(path: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return json.loads(line)
    return None


def _namespace_from_manifest(manifest: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        internal_simulations=manifest.get("internal_simulations", 10),
        final_simulations=manifest.get("final_simulations", 50),
        time_limit_s=manifest.get("time_limit_s"),
        max_function_evaluations=manifest.get("max_function_evaluations"),
        workers=manifest.get("workers", 1),
        surrogate_n_jobs=manifest.get("surrogate_n_jobs"),
        simulation_workers=manifest.get("simulation_workers", 1),
        disable_local_search=not bool(manifest.get("local_search_enabled", True)),
    )


def _write_from_manifest(path: Path) -> Path:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    args = _namespace_from_manifest(manifest)
    target_dir = result_hyperparameter_dir(path.parent)
    run_metadata = {
        key: manifest.get(key)
        for key in [
            "scenario",
            "n_instances",
            "n_runs_per_instance",
            "total_expected_runs",
            "total_successful_runs",
            "internal_simulations",
            "final_simulations",
            "workers",
            "simulation_workers",
            "surrogate_n_jobs",
            "time_limit_s",
            "max_function_evaluations",
            "local_search_enabled",
            "uncertainty_json",
        ]
        if key in manifest
    }
    ga_config = manifest.get("ga_config") or effective_ga_config(args)
    run_config = manifest.get("run_config") or effective_run_config(args)
    return write_hyperparameters_txt(
        target_dir,
        run_metadata=run_metadata,
        ga_config=ga_config,
        run_config=run_config,
        notes=[
            "Backfilled from submission_manifest.json.",
            f"Source result dir: {path.parent}",
        ],
    )


def _infer_task_args(result_dir: Path) -> SimpleNamespace:
    is_retry = "scenario2_task_results_behnkegeiger_retry" in result_dir.parts
    return SimpleNamespace(
        internal_simulations=10,
        final_simulations=50,
        time_limit_s=129600,
        max_function_evaluations=5_000_000,
        workers=1,
        surrogate_n_jobs=10 if is_retry else 1,
        simulation_workers=10 if is_retry else 5,
        disable_local_search=False,
    )


def _write_from_result_json(path: Path) -> Path:
    row = json.loads(path.read_text(encoding="utf-8"))
    args = _infer_task_args(path.parent)
    target_dir = result_hyperparameter_dir(path.parent)
    return write_hyperparameters_txt(
        target_dir,
        run_metadata={
            "scenario": 2,
            "example_instance": row.get("instance"),
            "example_run": row.get("run"),
            "example_seed": row.get("seed"),
            "example_status": row.get("status"),
            "internal_simulations": args.internal_simulations,
            "final_simulations": args.final_simulations,
            "simulation_workers": args.simulation_workers,
            "surrogate_n_jobs": args.surrogate_n_jobs,
            "time_limit_s": args.time_limit_s,
            "max_function_evaluations": args.max_function_evaluations,
            "local_search_enabled": not args.disable_local_search,
            "uncertainty_json": "config/scenario2_uncertainty.json",
        },
        ga_config=effective_ga_config(args),
        run_config=effective_run_config(args),
        notes=[
            "Backfilled from result.json and the matching SLURM runner defaults.",
            f"Source result dir: {path.parent}",
        ],
    )


def _write_from_raw_results(path: Path) -> Path:
    row = _read_first_jsonl(path) or {}
    target_dir = result_hyperparameter_dir(path.parent)
    args = SimpleNamespace(
        internal_simulations=10,
        final_simulations=50,
        time_limit_s=129600,
        max_function_evaluations=5_000_000,
        workers=10,
        surrogate_n_jobs=2,
        simulation_workers=2,
        disable_local_search=False,
    )
    return write_hyperparameters_txt(
        target_dir,
        run_metadata={
            "scenario": 2,
            "example_instance": row.get("instance", path.parent.name + ".fjs"),
            "n_runs_per_instance": 10,
            "workers": args.workers,
            "internal_simulations": args.internal_simulations,
            "final_simulations": args.final_simulations,
            "simulation_workers": args.simulation_workers,
            "surrogate_n_jobs": args.surrogate_n_jobs,
            "time_limit_s": args.time_limit_s,
            "max_function_evaluations": args.max_function_evaluations,
            "local_search_enabled": not args.disable_local_search,
            "uncertainty_json": "config/scenario2_uncertainty.json",
        },
        ga_config=effective_ga_config(args),
        run_config=effective_run_config(args),
        notes=[
            "Backfilled from raw_results.jsonl and submit_scenario2_instances.sh defaults.",
            f"Source result dir: {path.parent}",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("results"), Path("results_old")])
    args = parser.parse_args()

    written: dict[Path, Path] = {}
    for root in args.roots:
        if not root.exists():
            continue
        for manifest_path in sorted(root.rglob("submission_manifest.json")):
            written[result_hyperparameter_dir(manifest_path.parent)] = _write_from_manifest(manifest_path)
        for result_path in sorted(root.rglob("result.json")):
            target_dir = result_hyperparameter_dir(result_path.parent)
            if target_dir not in written:
                written[target_dir] = _write_from_result_json(result_path)
        for raw_path in sorted(root.rglob("raw_results.jsonl")):
            target_dir = result_hyperparameter_dir(raw_path.parent)
            if target_dir not in written:
                written[target_dir] = _write_from_raw_results(raw_path)

    print(f"Wrote {len(written)} hyperparameter file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
