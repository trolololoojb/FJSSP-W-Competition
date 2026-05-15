

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def _safe_stats(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        _finite_float(np.mean(arr)),
        _finite_float(np.std(arr)),
        _finite_float(np.min(arr)),
        _finite_float(np.max(arr)),
    )


def _as_list(values: Any) -> list:
    if values is None:
        return []
    try:
        return list(values)
    except TypeError:
        return []


def _duration_for_assignment(
    durations: Any,
    operation: Any,
    machine: Any,
    worker: Any,
) -> float:
    if isinstance(durations, Mapping):
        keys = (
            (operation, machine, worker),
            (operation, machine),
            (machine, worker),
            operation,
        )
        for key in keys:
            if key in durations:
                return _finite_float(durations[key])
        return 0.0

    try:
        value = durations[operation][machine][worker]
    except (TypeError, KeyError, IndexError):
        try:
            value = durations[operation][machine]
        except (TypeError, KeyError, IndexError):
            try:
                value = durations[operation]
            except (TypeError, KeyError, IndexError):
                return 0.0
    return _finite_float(value)


def _uncertainty_for_worker(uncertainty_parameters: Any, worker: Any) -> float:
    if uncertainty_parameters is None:
        return 0.0

    if isinstance(uncertainty_parameters, Mapping):
        if worker in uncertainty_parameters:
            value = uncertainty_parameters[worker]
            if isinstance(value, Mapping):
                for key in ("uncertainty", "std", "stdev", "sigma"):
                    if key in value:
                        return _finite_float(value[key])
                return 0.0
            if isinstance(value, (list, tuple, np.ndarray)):
                return _finite_float(np.mean(np.abs(np.asarray(value, dtype=float))))
            return _finite_float(value)
        return 0.0

    try:
        value = uncertainty_parameters[worker]
    except (TypeError, KeyError, IndexError):
        return 0.0
    if isinstance(value, (list, tuple, np.ndarray)):
        return _finite_float(np.mean(np.abs(np.asarray(value, dtype=float))))
    return _finite_float(value)


def featurize_candidate(
    sequence,
    machine_assignments,
    worker_assignments,
    start_times,
    durations,
    job_sequence,
    uncertainty_parameters=None,
) -> dict:
    """Return a flat dict of numeric features for one decoded candidate."""

    machines = _as_list(machine_assignments)
    workers = _as_list(worker_assignments)
    starts = _as_list(start_times)
    jobs = _as_list(job_sequence)

    op_durations: list[float] = []
    machine_loads: dict[Any, float] = {}
    worker_loads: dict[Any, float] = {}
    completion_times: list[float] = []

    for op_idx, machine in enumerate(machines):
        if op_idx >= len(workers):
            continue
        worker = workers[op_idx]
        duration = _duration_for_assignment(durations, op_idx, machine, worker)
        if duration <= 0.0:
            continue

        op_durations.append(duration)
        machine_loads[machine] = machine_loads.get(machine, 0.0) + duration
        worker_loads[worker] = worker_loads.get(worker, 0.0) + duration

        start = _finite_float(starts[op_idx]) if op_idx < len(starts) else 0.0
        completion_times.append(start + duration)

    mean_duration, std_duration, min_duration, max_duration = _safe_stats(op_durations)
    machine_load_values = list(machine_loads.values())
    worker_load_values = list(worker_loads.values())
    mean_machine_load, std_machine_load, _, max_machine_load = _safe_stats(machine_load_values)
    mean_worker_load, std_worker_load, _, max_worker_load = _safe_stats(worker_load_values)

    features = {
        "n_operations": _finite_float(len(machines)),
        "n_machines": _finite_float(len(set(machines))),
        "n_workers": _finite_float(len(set(workers))),
        "n_jobs": _finite_float(len(set(jobs))),
        "deterministic_makespan": _finite_float(max(completion_times) if completion_times else 0.0),
        "mean_operation_duration": mean_duration,
        "std_operation_duration": std_duration,
        "min_operation_duration": min_duration,
        "max_operation_duration": max_duration,
        "mean_machine_load": mean_machine_load,
        "max_machine_load": max_machine_load,
        "std_machine_load": std_machine_load,
        "mean_worker_load": mean_worker_load,
        "max_worker_load": max_worker_load,
        "std_worker_load": std_worker_load,
        "worker_load_imbalance": _finite_float(max_worker_load - mean_worker_load),
        "machine_load_imbalance": _finite_float(max_machine_load - mean_machine_load),
    }

    if uncertainty_parameters is not None:
        worker_uncertainties = {
            worker: _uncertainty_for_worker(uncertainty_parameters, worker)
            for worker in set(workers)
        }
        uncertainty_values = list(worker_uncertainties.values())
        mean_uncertainty, std_uncertainty, _, max_uncertainty = _safe_stats(uncertainty_values)

        weighted_values = []
        for worker, load in worker_loads.items():
            weighted_values.append(worker_uncertainties.get(worker, 0.0) * load)
        total_worker_load = sum(worker_load_values)
        weighted_mean = (
            sum(weighted_values) / total_worker_load if total_worker_load > 0.0 else 0.0
        )

        features.update(
            {
                "mean_worker_uncertainty": mean_uncertainty,
                "max_worker_uncertainty": max_uncertainty,
                "std_worker_uncertainty": std_uncertainty,
                "weighted_mean_worker_uncertainty": _finite_float(weighted_mean),
                "weighted_max_worker_uncertainty": _finite_float(
                    max(weighted_values) if weighted_values else 0.0
                ),
            }
        )

    return {key: _finite_float(value) for key, value in features.items()}


def feature_dicts_to_matrix(feature_dicts, feature_names=None):
    """Convert list of feature dicts to numpy matrix and stable feature-name list."""

    records = list(feature_dicts or [])
    if feature_names is None:
        feature_names = sorted({key for record in records for key in record.keys()})
    else:
        feature_names = list(feature_names)

    X = np.zeros((len(records), len(feature_names)), dtype=float)
    for row_idx, record in enumerate(records):
        for col_idx, name in enumerate(feature_names):
            X[row_idx, col_idx] = _finite_float(record.get(name, 0.0))

    return X, feature_names
