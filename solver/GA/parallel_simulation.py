from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor
import random
import statistics
from typing import Sequence

from util.graph import run_n_simulations


_EXECUTORS: dict[int, ProcessPoolExecutor] = {}


def _shutdown_executors() -> None:
    for executor in _EXECUTORS.values():
        executor.shutdown(wait=False, cancel_futures=True)
    _EXECUTORS.clear()


atexit.register(_shutdown_executors)


def _executor(max_workers: int) -> ProcessPoolExecutor:
    executor = _EXECUTORS.get(max_workers)
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=max_workers)
        _EXECUTORS[max_workers] = executor
    return executor


def _chunk_sizes(n_items: int, requested_workers: int) -> list[int]:
    if n_items <= 0:
        return []
    # util.graph.run_n_simulations computes statistics.stdev, so each chunk
    # must contain at least two simulations.
    workers = min(max(1, int(requested_workers)), max(1, n_items // 2))
    if workers <= 1:
        return [n_items]

    base = n_items // workers
    remainder = n_items % workers
    sizes = [base + (1 if idx < remainder else 0) for idx in range(workers)]

    while len(sizes) > 1 and sizes[-1] < 2:
        sizes[-2] += sizes[-1]
        sizes.pop()
    return sizes


def _run_chunk(payload: tuple) -> list[float]:
    (
        s,
        e,
        m,
        w,
        js,
        d,
        uncertainty_parameters,
        n_simulations,
        uncertainty_source,
        processing_times,
        machine_breakdowns,
        worker_unavailabilites,
        seed,
    ) = payload

    if seed is not None:
        random.seed(seed)
    results, _, _, _ = run_n_simulations(
        s,
        e,
        m,
        w,
        js,
        d,
        uncertainty_parameters,
        n_simulations,
        uncertainty_source=uncertainty_source,
        processing_times=processing_times,
        machine_breakdowns=machine_breakdowns,
        worker_unavailabilites=worker_unavailabilites,
    )
    return [float(value) for value in results]


def run_n_simulations_parallel(
    s: Sequence,
    e: Sequence,
    m: Sequence,
    w: Sequence,
    js: Sequence,
    d: Sequence,
    uncertainty_parameters: Sequence,
    n_simulations: int,
    *,
    uncertainty_source: str = "worker",
    processing_times: bool = False,
    machine_breakdowns: bool = False,
    worker_unavailabilites: bool = False,
    workers: int = 1,
    seed: int | None = None,
) -> tuple[list[float], float, float, float]:
    """Run the original competition simulation in parallel chunks.

    The implementation intentionally delegates every simulation chunk to
    util.graph.run_n_simulations so the competition simulation logic remains
    unchanged.
    """
    n_simulations = int(n_simulations)
    workers = int(workers)
    if workers <= 1 or n_simulations < 4:
        return run_n_simulations(
            s,
            e,
            m,
            w,
            js,
            d,
            uncertainty_parameters,
            n_simulations,
            uncertainty_source=uncertainty_source,
            processing_times=processing_times,
            machine_breakdowns=machine_breakdowns,
            worker_unavailabilites=worker_unavailabilites,
        )

    sizes = _chunk_sizes(n_simulations, workers)
    if len(sizes) <= 1:
        return run_n_simulations(
            s,
            e,
            m,
            w,
            js,
            d,
            uncertainty_parameters,
            n_simulations,
            uncertainty_source=uncertainty_source,
            processing_times=processing_times,
            machine_breakdowns=machine_breakdowns,
            worker_unavailabilites=worker_unavailabilites,
        )

    seed_rng = random.Random(seed)
    payloads = []
    for chunk_size in sizes:
        chunk_seed = None if seed is None else seed_rng.randrange(0, 2**32)
        payloads.append(
            (
                s,
                e,
                m,
                w,
                js,
                d,
                uncertainty_parameters,
                chunk_size,
                uncertainty_source,
                processing_times,
                machine_breakdowns,
                worker_unavailabilites,
                chunk_seed,
            )
        )

    results: list[float] = []
    for chunk_results in _executor(len(payloads)).map(_run_chunk, payloads):
        results.extend(chunk_results)

    robust_makespan = statistics.mean(results)
    robust_makespan_stdev = statistics.stdev(results)
    r_value = robust_makespan / max(e)
    return results, robust_makespan, robust_makespan_stdev, r_value
