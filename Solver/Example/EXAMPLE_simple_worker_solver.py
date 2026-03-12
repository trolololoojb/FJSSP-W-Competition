from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.benchmark_parser import WorkerBenchmarkParser
from util.evaluation import makespan


@dataclass
class Candidate:
    job: int
    operation_index: int
    machine: int
    worker: int
    start_time: int
    end_time: int


class SimpleWorkerSolver:
    """
    Sehr einfache Konstruktionsheuristik fuer FJSSP-W:
    In jedem Schritt wird ueber alle aktuell freigegebenen Operationen
    die Maschinen-/Worker-Kombination mit der fruehesten Fertigstellung
    ausgewaehlt.
    """

    def __init__(self, durations, job_sequence) -> None:
        self.durations = durations
        self.job_sequence = job_sequence
        self.n_operations = len(job_sequence)
        self.n_machines = len(durations[0])
        self.n_workers = len(durations[0][0])
        self.n_jobs = max(job_sequence) + 1

        self.job_to_operations: list[list[int]] = [[] for _ in range(self.n_jobs)]
        for operation_index, job in enumerate(job_sequence):
            self.job_to_operations[job].append(operation_index)

    def solve(self) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
        next_operation_per_job = [0] * self.n_jobs
        job_ready_time = [0] * self.n_jobs
        machine_ready_time = [0] * self.n_machines
        worker_ready_time = [0] * self.n_workers

        start_times = [0] * self.n_operations
        machine_assignments = [-1] * self.n_operations
        worker_assignments = [-1] * self.n_operations
        sequence: list[int] = []
        scheduled_operations: list[int] = []

        while len(sequence) < self.n_operations:
            best = self._select_best_candidate(
                next_operation_per_job=next_operation_per_job,
                job_ready_time=job_ready_time,
                machine_ready_time=machine_ready_time,
                worker_ready_time=worker_ready_time,
            )

            start_times[best.operation_index] = best.start_time
            machine_assignments[best.operation_index] = best.machine
            worker_assignments[best.operation_index] = best.worker

            next_operation_per_job[best.job] += 1
            job_ready_time[best.job] = best.end_time
            machine_ready_time[best.machine] = best.end_time
            worker_ready_time[best.worker] = best.end_time

            sequence.append(best.job)
            scheduled_operations.append(best.operation_index)

        return (
            sequence,
            start_times,
            machine_assignments,
            worker_assignments,
            scheduled_operations,
        )

    def _select_best_candidate(
        self,
        next_operation_per_job: list[int],
        job_ready_time: list[int],
        machine_ready_time: list[int],
        worker_ready_time: list[int],
    ) -> Candidate:
        best: Candidate | None = None

        for job in range(self.n_jobs):
            operation_pos = next_operation_per_job[job]
            if operation_pos >= len(self.job_to_operations[job]):
                continue

            operation_index = self.job_to_operations[job][operation_pos]

            for machine in range(self.n_machines):
                for worker in range(self.n_workers):
                    duration = int(self.durations[operation_index][machine][worker])
                    if duration <= 0:
                        continue

                    start_time = max(
                        job_ready_time[job],
                        machine_ready_time[machine],
                        worker_ready_time[worker],
                    )
                    end_time = start_time + duration

                    candidate = Candidate(
                        job=job,
                        operation_index=operation_index,
                        machine=machine,
                        worker=worker,
                        start_time=start_time,
                        end_time=end_time,
                    )

                    if self._is_better(candidate, best):
                        best = candidate

        if best is None:
            raise ValueError("Keine zulaessige Operation gefunden. Instanz oder Solverzustand ist inkonsistent.")

        return best

    def _is_better(self, candidate: Candidate, current: Candidate | None) -> bool:
        if current is None:
            return True
        if candidate.end_time != current.end_time:
            return candidate.end_time < current.end_time
        if candidate.start_time != current.start_time:
            return candidate.start_time < current.start_time
        if candidate.job != current.job:
            return candidate.job < current.job
        if candidate.machine != current.machine:
            return candidate.machine < current.machine
        return candidate.worker < current.worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Einfacher Beispiel-Solver fuer FJSSP-W-Instanzen."
    )
    parser.add_argument(
        "--instance",
        default="instances/Example_Instances_FJSSP-WF/Fattahi17.fjs",
        help="Pfad zu einer .fjs Instanz im FJSSP-W-Format.",
    )
    parser.add_argument(
        "--show-ops",
        action="store_true",
        help="Gibt die geplanten Operationen im Detail aus.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance_path = Path(args.instance)
    if not instance_path.exists():
        raise FileNotFoundError(f"Instanz nicht gefunden: {instance_path}")

    parser = WorkerBenchmarkParser()
    encoding = parser.parse_benchmark(str(instance_path))

    solver = SimpleWorkerSolver(encoding.durations(), encoding.job_sequence())
    sequence, start_times, machine_assignments, worker_assignments, operation_order = solver.solve()
    final_makespan = makespan(
        start_times=start_times,
        machine_assignments=machine_assignments,
        worker_assignments=worker_assignments,
        durations=encoding.durations(),
    )

    print(f"Instanz: {instance_path}")
    print(f"Jobs: {encoding.n_jobs()}")
    print(f"Maschinen: {encoding.n_machines()}")
    print(f"Operationen: {encoding.n_operations()}")
    print(f"Job-Reihenfolge der Konstruktion: {sequence}")
    print(f"Maschinenzuweisung pro Operation: {machine_assignments}")
    print(f"Workerzuweisung pro Operation: {worker_assignments}")
    print(f"Startzeiten pro Operation: {start_times}")
    print(f"Makespan: {int(final_makespan)}")

    if args.show_ops:
        print("\nGeplante Operationen:")
        for operation_index in operation_order:
            machine = machine_assignments[operation_index]
            worker = worker_assignments[operation_index]
            start = start_times[operation_index]
            duration = int(encoding.durations()[operation_index][machine][worker])
            end = start + duration
            job = encoding.job_sequence()[operation_index]
            print(
                f"op={operation_index:02d} job={job:02d} "
                f"machine={machine:02d} worker={worker:02d} "
                f"start={start:04d} end={end:04d} duration={duration:04d}"
            )


if __name__ == "__main__":
    main()
