from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple
import math
import random
import time

from util.evaluation import translate, makespan
from util.graph import run_n_simulations


def is_simulatable_schedule(
    start_times: Sequence[int],
    end_times: Sequence[int],
    machine_assignments: Sequence[int],
    worker_assignments: Sequence[int],
    job_sequence: Sequence[int],
) -> bool:
    """
    Validate that a decoded schedule can be safely passed to stochastic simulation.

    The check is intentionally conservative: candidates that violate temporal
    consistency, resource constraints, or acyclicity are rejected early and can
    be scored with +inf instead of crashing the GA.
    """
    n_operations = len(start_times)
    if not (
        len(end_times) == n_operations
        and len(machine_assignments) == n_operations
        and len(worker_assignments) == n_operations
        and len(job_sequence) == n_operations
    ):
        return False

    for i in range(n_operations):
        if end_times[i] < start_times[i]:
            return False

    adjacency: Dict[int, Set[int]] = defaultdict(set)
    indegree = [0] * n_operations

    def add_edge(source: int, target: int) -> bool:
        if source == target:
            return False
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1
        return True

    job_operations: Dict[int, List[int]] = defaultdict(list)
    for op_index, job_id in enumerate(job_sequence):
        job_operations[job_id].append(op_index)

    for operations in job_operations.values():
        for predecessor, successor in zip(operations, operations[1:]):
            if end_times[predecessor] > start_times[successor]:
                return False
            if not add_edge(predecessor, successor):
                return False

    def validate_resource(assignments: Sequence[int]) -> bool:
        grouped_operations: Dict[int, List[int]] = defaultdict(list)
        for op_index, resource_id in enumerate(assignments):
            grouped_operations[resource_id].append(op_index)

        for operations in grouped_operations.values():
            operations.sort(key=lambda op_index: (start_times[op_index], end_times[op_index], op_index))
            for predecessor, successor in zip(operations, operations[1:]):
                if start_times[successor] < end_times[predecessor]:
                    return False
                if not add_edge(predecessor, successor):
                    return False
        return True

    if not validate_resource(machine_assignments):
        return False
    if not validate_resource(worker_assignments):
        return False

    queue = deque(op_index for op_index, degree in enumerate(indegree) if degree == 0)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for successor in adjacency[current]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)

    return visited == n_operations


@dataclass
class WorkerGAConfig:
    """
    Englisch:
    Configuration for the worker assignment genetic algorithm. Expects durations and job sequence to be provided, along with various parameters controlling the genetic algorithm's behavior. 
    Derived fields are computed in __post_init__ and include counts of jobs, machines, workers, operations, as well as precomputed lists of available machines and workers for each operation, 
    and the maximum dissimilarity used for population initialization.
    Deutsch: 
    Konfiguration für den genetischen Algorithmus zur Arbeitszuweisung. Erwartet werden die Dauer und die Job-Sequenz, zusammen mit verschiedenen Parametern, die das Verhalten des genetischen Algorithmus steuern.
    Abgeleitete Felder werden in __post_init__ berechnet und umfassen die Anzahl der Jobs, Maschinen, Arbeiter, Operationen sowie vorgefertigte Listen verfügbarer Maschinen und Arbeiter für jede Operation und die maximale Dissimilarität, die für die Initialisierung der Population verwendet wird.
    """
    durations: Sequence  # expected shape: [n_ops][n_machines][n_workers]
    job_sequence: List[int]  # operation blocks by job, e.g. [0,0,0,1,1,2,...]
    population_size: int = 100 # number of individuals in the population
    offspring_amount: int = 400 # number of offspring created each generation, including the elite
    mutation_probability: Optional[float] = None # if None, will be set to 1/(2*n_operations)
    max_mutation_probability: float = 0.1 # maximum mutation probability before restart, if using adaptive mutation
    elitism_rate: float = 0.1 # portion of the population to carry over directly to the next generation
    max_elitism_rate: float = 0.1 # maximum elitism rate during restart, as a portion of the population size
    tournament_size: int = 0  # 0 => max(1, pop/10), number of individuals competing in tournament selection
    max_tournament_rate: float = 0.2 # maximum tournament size during restart, as a portion of the population size
    population_size_growth_rate: float = 1.25 # growth rate for population and offspring amount during restart
    restart_generations: int = 100 # number of generations with no improvement before performing a restart
    duration_variety: float = 1.0 # variety of durations in the instance, used to adjust parameters during restart (1.0 means no adjustment, <1.0 reduces elitism and tournament size more for low variety)
    use_dissimilarity: bool = True # whether to use dissimilarity in the fitness function and population initialization. dissimilarity is a measure of how different two individuals are, used to maintain diversity in the population. If True, new individuals will be initialized to be sufficiently dissimilar from the existing population, and the dissimilarity will also be considered during selection and mutation.
    max_initialization_attempts: int = 100 # maximum attempts to find a sufficiently dissimilar individual during population initialization before relaxing the dissimilarity requirement
    distance_adjustment_rate: float = 0.75 # rate at which to relax the dissimilarity requirement during population initialization if too many attempts fail
    use_stochastic_evaluation: bool = False # whether to use stochastic evaluation with simulations to estimate robust makespan, instead of deterministic evaluation. If True, the fitness will include "makespan" as the estimated robust makespan, "robust_makespan_stdev" as the standard deviation of the makespan across simulations, and "R" as the robustness measure (e.g. the 95th percentile of the makespan distribution). The translate function will be used to get start times and assignments, and then run_n_simulations will be called to perform the stochastic evaluation.
    uncertainty_parameters: Optional[List[List[float]]] = None # optional parameters for the uncertainty model used in stochastic evaluation, expected shape [n_ops][n_machines], where each entry is a parameter (e.g. standard deviation) for the processing time distribution of that operation on that machine. If None, no uncertainty will be applied and the deterministic durations will be used in the simulations.
    n_simulations: int = 100 # number of simulations to run for stochastic evaluation when use_stochastic_evaluation is True
    seed: Optional[int] = None # random seed for reproducibility

    # derived fields
    n_jobs: int = field(init=False)
    n_machines: int = field(init=False)
    n_workers: int = field(init=False)
    n_operations: int = field(init=False)
    job_start_indices: List[int] = field(init=False)
    available_machines: List[List[int]] = field(init=False)
    available_workers: List[List[List[int]]] = field(init=False)
    max_dissimilarity: float = field(init=False)

    def __post_init__(self) -> None:
        """
        Englisch:
        Compute derived fields based on the provided durations and job sequence. This includes counting the number of operations, machines, workers, and jobs, as well as precomputing the available machines and workers for each operation.
        Deutsch:
        Berechnung der abgeleiteten Felder basierend auf den bereitgestellten Dauern und der Job-Sequenz. Dies umfasst die Zählung der Anzahl der Operationen, Maschinen, Arbeiter und Jobs sowie die Vorberechnung der verfügbaren Maschinen und Arbeiter für jede Operation.
        Args:            None
        Returns:         None
        Raises:          ValueError if job ids in the job sequence are not contiguous integers starting from 0
        """
        self.n_operations = len(self.job_sequence)
        self.n_machines = len(self.durations[0])
        self.n_workers = len(self.durations[0][0])
        self.n_jobs = len(set(self.job_sequence))
        if self.mutation_probability is None:
            self.mutation_probability = 1.0 / (2.0 * self.n_operations)
        self.job_start_indices = self._compute_job_start_indices(self.job_sequence)
        self.available_machines = []
        self.available_workers = []
        for op in range(self.n_operations):
            op_machines: List[int] = []
            op_workers: List[List[int]] = [[] for _ in range(self.n_machines)]
            for m in range(self.n_machines):
                valid_workers = [w for w in range(self.n_workers) if self.durations[op][m][w] > 0]
                op_workers[m] = valid_workers
                if valid_workers:
                    op_machines.append(m)
            self.available_machines.append(op_machines)
            self.available_workers.append(op_workers)
        self.max_dissimilarity = self._determine_max_dissimilarity()

    @staticmethod
    def _compute_job_start_indices(job_sequence: List[int]) -> List[int]:
        """
        Englisch:
        Computes the starting indices for each job in the operation sequence.
        Deutsch:
        Berechnet die Startindizes für jeden Job in der Operationssequenz.
        Args:
            job_sequence (List[int]): The sequence of job IDs for each operation.
        Returns:
            List[int]: List of starting indices for each job.
        """
        seen = {}
        starts: List[int] = []
        for idx, job in enumerate(job_sequence):
            if job not in seen:
                seen[job] = idx
                starts.append(idx)
        # assumes jobs are encoded as 0..n-1 like the benchmark code
        if sorted(seen) != list(range(len(seen))):
            raise ValueError("job ids must be contiguous 0..n-1 for this port")
        return [seen[j] for j in range(len(seen))]

    def _determine_max_dissimilarity(self) -> float:
        """
        Englisch:
        Determines the maximum possible dissimilarity value for population initialization.
        Deutsch:
        Bestimmt den maximal möglichen Unähnlichkeitswert für die Populationsinitialisierung.
        Args:
            None
        Returns:
            float: The maximum dissimilarity value.
        """
        result = 0.0
        for op in range(self.n_operations):
            result += len(self.available_machines[op])
            max_workers = 0
            for m in self.available_machines[op]:
                max_workers = max(max_workers, len(self.available_workers[op][m]))
            result += max_workers
        result += self.n_operations
        return result


class WFJSSPIndividual:
    """
    Englisch:
    Represents an individual solution in the genetic algorithm, containing the operation sequence, machine assignments, worker assignments, fitness value, and feasibility flag.
    Deutsch:
    Repräsentiert eine individuelle Lösung im genetischen Algorithmus, enthält die Operationssequenz, Maschinenzuweisungen, Arbeiterzuweisungen, Fitnesswert und Machbarkeitsflag.
    """
    def __init__(self, config: WorkerGAConfig, rng: random.Random, randomize: bool = True) -> None:
        """
        Englisch:
        Initializes a new individual with the given configuration and random number generator. If randomize is True, generates a random solution.
        Deutsch:
        Initialisiert ein neues Individuum mit der gegebenen Konfiguration und dem Zufallszahlengenerator. Wenn randomize True ist, generiert eine zufällige Lösung.
        Args:
            config (WorkerGAConfig): The configuration for the genetic algorithm.
            rng (random.Random): Random number generator.
            randomize (bool, optional): Whether to randomize the individual. Defaults to True.
        Returns:
            None
        """
        self.config = config
        self.rng = rng
        self.sequence = [0] * config.n_operations
        self.assignments = [0] * config.n_operations
        self.workers = [0] * config.n_operations
        self.fitness = {"makespan": math.inf}
        self.feasible = True
        if randomize:
            self.randomize()

    def copy(self) -> "WFJSSPIndividual":
        """
        Englisch:
        Creates a deep copy of the individual.
        Deutsch:
        Erstellt eine tiefe Kopie des Individuums.
        Args:
            None
        Returns:
            WFJSSPIndividual: A copy of the individual.
        """
        other = WFJSSPIndividual(self.config, self.rng, randomize=False)
        other.sequence = self.sequence[:]
        other.assignments = self.assignments[:]
        other.workers = self.workers[:]
        other.fitness = dict(self.fitness)
        other.feasible = self.feasible
        return other

    def equals(self, other: "WFJSSPIndividual") -> bool:
        """
        Englisch:
        Checks if this individual is equal to another based on sequence, assignments, and workers.
        Deutsch:
        Überprüft, ob dieses Individuum einem anderen basierend auf Sequenz, Zuweisungen und Arbeitern entspricht.
        Args:
            other (WFJSSPIndividual): The other individual to compare with.
        Returns:
            bool: True if equal, False otherwise.
        """
        return (
            self.sequence == other.sequence
            and self.assignments == other.assignments
            and self.workers == other.workers
        )

    def randomize(self) -> None:
        """
        Englisch:
        Randomly initializes the sequence, assignments, and workers for the individual.
        Deutsch:
        Initialisiert zufällig die Sequenz, Zuweisungen und Arbeiter für das Individuum.
        Args:
            None
        Returns:
            None
        """
        self.sequence = self.config.job_sequence[:]
        self.rng.shuffle(self.sequence)
        for i in range(self.config.n_operations):
            self.assignments[i] = self.rng.choice(self.config.available_machines[i])
        for i in range(self.config.n_operations):
            self.workers[i] = self.rng.choice(self.config.available_workers[i][self.assignments[i]])

    def dissimilarity(self, other: "WFJSSPIndividual") -> float:
        """
        Englisch:
        Calculates the dissimilarity between this individual and another, used for population diversity.
        Deutsch:
        Berechnet die Unähnlichkeit zwischen diesem Individuum und einem anderen, verwendet für Populationsvielfalt.
        Args:
            other (WFJSSPIndividual): The other individual to compare with.
        Returns:
            float: The dissimilarity value.
        """
        result = 0.0
        for i in range(self.config.n_operations):
            if self.assignments[i] != other.assignments[i]:
                result += len(self.config.available_machines[i])
            if self.sequence[i] != other.sequence[i]:
                result += 1.0
            if self.workers[i] != other.workers[i]:
                # keep the same slightly odd indexing logic as the C# version
                result += len(self.config.available_workers[i][other.assignments[i]])
        return result

    @classmethod
    def from_population(
        cls,
        config: WorkerGAConfig,
        rng: random.Random,
        population: List["WFJSSPIndividual"],
    ) -> "WFJSSPIndividual":
        """
        Englisch:
        Creates a new individual that is sufficiently dissimilar from the existing population.
        Deutsch:
        Erstellt ein neues Individuum, das ausreichend unähnlich zur bestehenden Population ist.
        Args:
            config (WorkerGAConfig): The configuration for the genetic algorithm.
            rng (random.Random): Random number generator.
            population (List[WFJSSPIndividual]): The current population.
        Returns:
            WFJSSPIndividual: A new individual.
        """
        ind = cls(config, rng, randomize=False)
        if not population or not config.use_dissimilarity:
            ind.randomize()
            return ind
        min_distance = config.max_dissimilarity
        attempts = 0
        while True:
            if attempts > config.max_initialization_attempts:
                min_distance *= config.distance_adjustment_rate
                attempts = 0
            ind.randomize()
            ds = [ind.dissimilarity(p) for p in population]
            avg = sum(ds) / len(ds) if ds else math.inf
            if avg >= min_distance:
                return ind
            attempts += 1

    @classmethod
    def crossover(
        cls,
        config: WorkerGAConfig,
        rng: random.Random,
        parent_a: "WFJSSPIndividual",
        parent_b: "WFJSSPIndividual",
    ) -> "WFJSSPIndividual":
        """
        Englisch:
        Performs crossover between two parent individuals to create a child.
        Deutsch:
        Führt Crossover zwischen zwei Eltern-Individuen durch, um ein Kind zu erstellen.
        Args:
            config (WorkerGAConfig): The configuration for the genetic algorithm.
            rng (random.Random): Random number generator.
            parent_a (WFJSSPIndividual): First parent.
            parent_b (WFJSSPIndividual): Second parent.
        Returns:
            WFJSSPIndividual: The child individual.
        """
        child = cls(config, rng, randomize=False)
        jobs = sorted(set(config.job_sequence))
        a_jobs, b_jobs = set(), set()
        for job in jobs:
            if rng.random() < 0.5:
                a_jobs.add(job)
            else:
                b_jobs.add(job)
        parent_b_values = [job for job in parent_b.sequence if job in b_jobs]
        b_index = 0
        for i in range(config.n_operations):
            if parent_a.sequence[i] in a_jobs:
                child.sequence[i] = parent_a.sequence[i]
            else:
                child.sequence[i] = parent_b_values[b_index]
                b_index += 1
            if rng.random() < 0.5:
                child.assignments[i] = parent_a.assignments[i]
                child.workers[i] = parent_a.workers[i]
            else:
                child.assignments[i] = parent_b.assignments[i]
                child.workers[i] = parent_b.workers[i]
        return child

    def mutate(self, p: float) -> None:
        """
        Englisch:
        Mutates the individual with probability p, changing sequence, assignments, or workers.
        Deutsch:
        Mutatiert das Individuum mit Wahrscheinlichkeit p, ändert Sequenz, Zuweisungen oder Arbeiter.
        Args:
            p (float): Mutation probability.
        Returns:
            None
        """
        for i in range(self.config.n_operations):
            if self.rng.random() < p:
                attempts = 0
                while True:
                    swap = self.rng.randrange(self.config.n_operations)
                    attempts += 1
                    if self.sequence[swap] != self.sequence[i] or attempts >= 100:
                        break
                self.sequence[i], self.sequence[swap] = self.sequence[swap], self.sequence[i]

            if self.rng.random() < p and len(self.config.available_machines[i]) > 1:
                attempts = 0
                while True:
                    idx = self.rng.randrange(len(self.config.available_machines[i]))
                    attempts += 1
                    new_machine = self.config.available_machines[i][idx]
                    if new_machine != self.assignments[i] or attempts >= 100:
                        break
                self.assignments[i] = new_machine
                if self.workers[i] not in self.config.available_workers[i][self.assignments[i]]:
                    self.workers[i] = self.rng.choice(self.config.available_workers[i][self.assignments[i]])

            if self.rng.random() < p and len(self.config.available_workers[i][self.assignments[i]]) > 1:
                attempts = 0
                while True:
                    idx = self.rng.randrange(len(self.config.available_workers[i][self.assignments[i]]))
                    attempts += 1
                    new_worker = self.config.available_workers[i][self.assignments[i]][idx]
                    if new_worker != self.workers[i] or attempts >= 100:
                        break
                self.workers[i] = new_worker


class WFJSSPGA:
    """
    Englisch:
    Main genetic algorithm class for solving the Flexible Job Shop Scheduling Problem with Workers (FJSSP-W). Manages the population, evaluation, and evolution process.
    Deutsch:
    Hauptklasse des genetischen Algorithmus zur Lösung des Flexible Job Shop Scheduling Problems mit Workern (FJSSP-W). Verwaltet die Population, Evaluierung und Evolutionsprozess.
    """
    DEFAULT_TIME_LIMIT_S = 20 * 60
    DEFAULT_MAX_FUNCTION_EVALUATIONS = 5_000_000
    DEFAULT_PROGRESS_INTERVAL_EVALUATIONS = 50_000

    def __init__(self, config: WorkerGAConfig) -> None:
        """
        Englisch:
        Initializes the genetic algorithm with the given configuration.
        Deutsch:
        Initialisiert den genetischen Algorithmus mit der gegebenen Konfiguration.
        Args:
            config (WorkerGAConfig): The configuration for the genetic algorithm.
        Returns:
            None
        """
        self.config = config
        self.rng = random.Random(config.seed)
        self.population: List[WFJSSPIndividual] = []
        self.function_evaluations = 0
        self._run_start_time: Optional[float] = None
        self._next_progress_evaluation: Optional[int] = None
        self._progress_interval_evaluations: Optional[int] = None

    def _maybe_print_progress(self) -> None:
        if (
            self._run_start_time is None
            or self._next_progress_evaluation is None
            or self._progress_interval_evaluations is None
        ):
            return

        while self.function_evaluations >= self._next_progress_evaluation:
            runtime_s = time.time() - self._run_start_time
            print(
                f"Progress: {self.function_evaluations:,} Funktionsevaluierungen, "
                f"Laufzeit {runtime_s:.2f}s",
                flush=True,
            )
            self._next_progress_evaluation += self._progress_interval_evaluations

    def _count_function_evaluation(self, amount: int = 1) -> None:
        self.function_evaluations += amount
        self._maybe_print_progress()

    def evaluate(self, ind: WFJSSPIndividual) -> float:
        """
        Englisch:
        Evaluates the fitness of an individual by simulating the scheduling and calculating the makespan.
        Deutsch:
        Bewertet die Fitness eines Individuums, indem das Scheduling simuliert und der Makespan berechnet wird.
        Args:
            ind (WFJSSPIndividual): The individual to evaluate.
        Returns:
            float: The makespan value.
        """
        
        # Wenn das Individuum als unzulässig markiert ist, setzen wir den Fitnesswert auf unendlich und geben unendlich zurück
        if not ind.feasible:
            ind.fitness["makespan"] = math.inf
            return math.inf

        try:
            start_times, machine_assignments, worker_assignments = translate(
                ind.sequence, ind.assignments, ind.workers, self.config.durations
            )
        except Exception:
            # Invalid solution
            ind.fitness["makespan"] = math.inf
            self._count_function_evaluation()
            return math.inf

        if self.config.use_stochastic_evaluation:
            end_times = [
                start_times[i] + self.config.durations[i][machine_assignments[i]][worker_assignments[i]]
                for i in range(len(start_times))
            ]
            if not is_simulatable_schedule(
                start_times,
                end_times,
                machine_assignments,
                worker_assignments,
                self.config.job_sequence,
            ):
                ind.fitness["makespan"] = math.inf
                self._count_function_evaluation()
                return math.inf
            try:
                results, robust_makespan, robust_makespan_stdev, R = run_n_simulations(
                    start_times,
                    end_times,
                    machine_assignments,
                    worker_assignments,
                    self.config.job_sequence,
                    self.config.durations,
                    self.config.uncertainty_parameters,
                    self.config.n_simulations,
                    processing_times=True
                )
            except (RecursionError, Exception):
                ind.fitness["makespan"] = math.inf
                self._count_function_evaluation()
                return math.inf
            ind.fitness["makespan"] = robust_makespan
            ind.fitness["robust_makespan_stdev"] = robust_makespan_stdev
            ind.fitness["R"] = R
            self._count_function_evaluation(len(results))
            return robust_makespan
        else:
            makespan_val = makespan(start_times, machine_assignments, worker_assignments, self.config.durations)
            ind.fitness["makespan"] = makespan_val
            self._count_function_evaluation()
            return makespan_val

    def create_population(self, population_size: int, stop_condition=None) -> None:
        """
        Englisch:
        Creates an initial population of individuals.
        Deutsch:
        Erstellt eine anfängliche Population von Individuen.
        Args:
            population_size (int): The size of the population to create.
        Returns:
            None
        """
        self.population = []
        for _ in range(population_size):
            if stop_condition is not None and stop_condition():
                break
            ind = WFJSSPIndividual.from_population(self.config, self.rng, self.population)
            self.evaluate(ind)
            self.population.append(ind)
        self.population.sort(key=lambda x: x.fitness["makespan"])

    def tournament_selection(self, tournament_size: int) -> WFJSSPIndividual:
        """
        Englisch:
        Selects an individual using tournament selection.
        Deutsch:
        Wählt ein Individuum mittels Turnierselektion aus.
        Args:
            tournament_size (int): The size of the tournament.
        Returns:
            WFJSSPIndividual: The selected individual.
        """
        if tournament_size == 0:
            tournament_size = max(1, len(self.population) // 10)
        participants = self.rng.sample(range(len(self.population)), k=tournament_size)
        winner = self.population[participants[0]]
        for idx in participants[1:]:
            if self.population[idx].fitness["makespan"] < winner.fitness["makespan"]:
                winner = self.population[idx]
        return winner

    def recombine(self, tournament_size: int) -> WFJSSPIndividual:
        """
        Englisch:
        Recombines two parents to create a child individual.
        Deutsch:
        Rekombiniert zwei Eltern, um ein Kind-Individuum zu erstellen.
        Args:
            tournament_size (int): The size of the tournament for selection.
        Returns:
            WFJSSPIndividual: The child individual.
        """
        parent_a = self.tournament_selection(tournament_size)
        attempts = 0
        while True:
            parent_b = self.tournament_selection(tournament_size)
            attempts += 1
            if not parent_a.equals(parent_b) or attempts >= 100:
                break
        return WFJSSPIndividual.crossover(self.config, self.rng, parent_a, parent_b)

    def create_offspring(
        self,
        offspring_amount: int,
        tournament_size: int,
        mutation_probability: float,
        stop_condition=None,
    ) -> List[WFJSSPIndividual]:
        """
        Englisch:
        Creates a new generation of offspring through recombination and mutation.
        Deutsch:
        Erstellt eine neue Generation von Nachkommen durch Rekombination und Mutation.
        Args:
            offspring_amount (int): Number of offspring to create.
            tournament_size (int): Size of the tournament for selection.
            mutation_probability (float): Probability of mutation.
        Returns:
            List[WFJSSPIndividual]: List of offspring individuals.
        """
        offspring: List[WFJSSPIndividual] = []
        for _ in range(offspring_amount):
            if stop_condition is not None and stop_condition():
                break
            child = self.recombine(tournament_size)
            child.mutate(mutation_probability)
            self.evaluate(child)
            offspring.append(child)
        return offspring

    @staticmethod
    def _get_all_equal(best: WFJSSPIndividual, individuals: List[WFJSSPIndividual]) -> List[WFJSSPIndividual]:
        """
        Englisch:
        Returns all individuals with the same fitness as the best individual.
        Deutsch:
        Gibt alle Individuen mit der gleichen Fitness wie das beste Individuum zurück.
        Args:
            best (WFJSSPIndividual): The best individual.
            individuals (List[WFJSSPIndividual]): List of individuals to check.
        Returns:
            List[WFJSSPIndividual]: List of individuals with equal fitness.
        """
        return [x for x in individuals if x.fitness["makespan"] == best.fitness["makespan"]]

    @staticmethod
    def _update_mutation_probability(p: float, generation: int, last_progress: int, max_wait: int, max_p: float) -> float:
        """
        Englisch:
        Updates the mutation probability based on stagnation.
        Deutsch:
        Aktualisiert die Mutationswahrscheinlichkeit basierend auf Stagnation.
        Args:
            p (float): Current mutation probability.
            generation (int): Current generation number.
            last_progress (int): Last generation with progress.
            max_wait (int): Maximum wait generations.
            max_p (float): Maximum mutation probability.
        Returns:
            float: Updated mutation probability.
        """
        return p + ((((generation - last_progress) * (1.0 / max_wait)) ** 4) * max_p)

    def run(
        self,
        max_generations: Optional[int] = None,
        time_limit_s: Optional[float] = DEFAULT_TIME_LIMIT_S,
        target_fitness: Optional[float] = None,
        max_function_evaluations: Optional[int] = DEFAULT_MAX_FUNCTION_EVALUATIONS,
        progress_interval_evaluations: Optional[int] = DEFAULT_PROGRESS_INTERVAL_EVALUATIONS,
        keep_multiple: bool = True,
        do_restart: bool = True,
    ) -> dict:
        """
        Englisch:
        Runs the genetic algorithm until one of the stopping criteria is met.
        Deutsch:
        Führt den genetischen Algorithmus aus, bis eines der Stoppkriterien erfüllt ist.
        Args:
            max_generations (Optional[int], optional): Maximum number of generations. Defaults to None.
            time_limit_s (Optional[float], optional): Time limit in seconds. Defaults to 1200.
            target_fitness (Optional[float], optional): Target fitness value. Defaults to None.
            max_function_evaluations (Optional[int], optional): Maximum function evaluations. Defaults to 5_000_000.
            progress_interval_evaluations (Optional[int], optional): Print progress after this many function evaluations. Defaults to 50_000.
            keep_multiple (bool, optional): Whether to keep multiple best individuals. Defaults to True.
            do_restart (bool, optional): Whether to perform restarts. Defaults to True.
        Returns:
            dict: Results including best individual, population, etc.
        """
        self.function_evaluations = 0
        start_time = time.time()
        self._run_start_time = start_time
        self._progress_interval_evaluations = progress_interval_evaluations
        self._next_progress_evaluation = (
            progress_interval_evaluations
            if progress_interval_evaluations is not None and progress_interval_evaluations > 0
            else None
        )

        def stop_limit_reached() -> bool:
            if time_limit_s is not None and (time.time() - start_time) >= time_limit_s:
                return True
            if max_function_evaluations is not None and self.function_evaluations >= max_function_evaluations:
                return True
            return False

        self.create_population(self.config.population_size, stop_condition=stop_limit_reached)
        if not self.population:
            raise RuntimeError("No individual could be evaluated before a stop limit was reached.")
        population_size = self.config.population_size
        offspring_amount = self.config.offspring_amount
        tournament_size = self.config.tournament_size
        mutation_probability = float(self.config.mutation_probability)
        max_mutation_probability = self.config.max_mutation_probability
        elitism = int(self.config.elitism_rate * population_size)
        max_wait = self.config.restart_generations

        overall_best = self._get_all_equal(self.population[0], self.population)
        current_best = self._get_all_equal(self.population[0], self.population)
        last_progress = 0
        generation = 0
        restarts = 0
        history = []

        def record_history() -> None:
            history.append(
                {
                    "generation": generation,
                    "best_makespan": float(self.population[0].fitness["makespan"]),
                    "overall_best_makespan": float(overall_best[0].fitness["makespan"]),
                    "function_evaluations": int(self.function_evaluations),
                    "mutation_probability": float(mutation_probability),
                    "population_size": int(population_size),
                    "offspring_amount": int(offspring_amount),
                    "restarts": int(restarts),
                    "runtime_s": float(time.time() - start_time),
                }
            )

        record_history()

        while True:
            best_fitness = overall_best[0].fitness["makespan"]
            if max_generations is not None and generation >= max_generations:
                break
            if stop_limit_reached():
                break
            if target_fitness is not None and best_fitness <= target_fitness:
                break

            if generation > 0 and last_progress < generation - 1:
                mutation_probability = self._update_mutation_probability(
                    mutation_probability, generation, last_progress, max_wait, max_mutation_probability
                )

            if do_restart and mutation_probability > max_mutation_probability:
                max_population_size = 400
                max_offspring_amount = max_population_size * 4
                population_size = min(max_population_size, int(self.config.population_size_growth_rate * population_size))
                offspring_amount = min(max_offspring_amount, int(self.config.population_size_growth_rate * offspring_amount))
                elitism = max(0, int(population_size * self.config.max_elitism_rate * self.config.duration_variety))
                tournament_size = max(1, int(population_size * self.config.max_tournament_rate * self.config.duration_variety))
                current_best = []
                self.create_population(population_size, stop_condition=stop_limit_reached)
                if not self.population:
                    break
                mutation_probability = float(self.config.mutation_probability)
                last_progress = generation
                restarts += 1
                if stop_limit_reached():
                    break

            offspring = self.create_offspring(
                offspring_amount,
                tournament_size,
                mutation_probability,
                stop_condition=stop_limit_reached,
            )
            pool = offspring + self.population[:elitism]
            pool.sort(key=lambda x: x.fitness["makespan"])
            self.population = pool[:population_size]

            if not current_best or self.population[0].fitness["makespan"] < current_best[0].fitness["makespan"]:
                current_best = self._get_all_equal(self.population[0], self.population) if keep_multiple else [self.population[0]]
                if current_best[0].fitness["makespan"] < overall_best[0].fitness["makespan"]:
                    overall_best = self._get_all_equal(self.population[0], self.population) if keep_multiple else [current_best[0]]
                elif keep_multiple and current_best[0].fitness["makespan"] == overall_best[0].fitness["makespan"]:
                    known = set(id(x) for x in overall_best)
                    for ind in current_best:
                        if id(ind) not in known:
                            overall_best.append(ind)
                            known.add(id(ind))
                last_progress = generation
            elif keep_multiple and self.population[0].fitness["makespan"] == current_best[0].fitness["makespan"]:
                equals = self._get_all_equal(self.population[0], self.population)
                known_current = set(id(x) for x in current_best)
                for ind in equals:
                    if id(ind) not in known_current:
                        current_best.append(ind)
                        known_current.add(id(ind))
                if current_best[0].fitness["makespan"] == overall_best[0].fitness["makespan"]:
                    known_best = set(id(x) for x in overall_best)
                    for ind in current_best:
                        if id(ind) not in known_best:
                            overall_best.append(ind)
                            known_best.add(id(ind))

            generation += 1
            record_history()

        return {
            "best": overall_best[0],
            "best_all_equal": overall_best,
            "population": self.population,
            "generations": generation,
            "function_evaluations": self.function_evaluations,
            "runtime_s": time.time() - start_time,
            "restarts": restarts,
            "history": history,
        }


# convenience helper -------------------------------------------------------

def build_ga_from_worker_encoding(worker_encoding, **kwargs) -> WFJSSPGA:
    """
    Englisch:
    Convenience function to build a WFJSSPGA instance from a worker encoding object that provides durations() and job_sequence() methods.
    Deutsch:
    Hilfsfunktion, um eine WFJSSPGA-Instanz aus einem Worker-Encoding-Objekt zu erstellen, das durations() und job_sequence() Methoden bereitstellt.
    Args:
        worker_encoding: Object with durations() and job_sequence() methods.
        **kwargs: Additional keyword arguments for WorkerGAConfig.
    Returns:
        WFJSSPGA: The initialized genetic algorithm instance.
    """
    cfg = WorkerGAConfig(
        durations=worker_encoding.durations(),
        job_sequence=list(worker_encoding.job_sequence()),
        **kwargs,
    )
    return WFJSSPGA(cfg)
