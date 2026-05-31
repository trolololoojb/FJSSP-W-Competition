# Scenario 2 Submission: SA-RGA

This repository contains my submission for **Scenario 2** of the FJSSP-W Competition: the Flexible Job Shop Scheduling Problem with Worker Flexibility under worker-dependent stochastic processing-time uncertainty.

## Participant

**Johann Bartels**  
Leibniz Universität Hannover  
Email: johann.ulrich.bartels@stud.uni-hannover.de

## Solver

**SA-RGA**: Surrogate-Assisted Robust Genetic Algorithm

The solver is based on the FJSSP-W genetic algorithm by Hutter, Steinberger, and Hellwig. The original C# implementation was translated to Python and adapted to the official competition repository.

The main extensions for this submission are:

- Scenario-2 robust evaluation with fixed worker-dependent uncertainty parameters
- direct use of the competition parser and evaluation utilities
- a quantile-random-forest surrogate model for reducing expensive simulation calls
- local search around promising candidates
- explicit schedule validation before stochastic simulation
- CSV export in the official Scenario-2 submission format

An optional RL-based mutation controller is included in the code base, but it was disabled for the submitted final runs.

## Scenario

This submission targets **Scenario 2**.

The objective is robust makespan minimization under stochastic processing times. One stochastic simulation is counted as one function evaluation. Therefore, evaluating a candidate with 10 simulations consumes 10 function evaluations.

Final submitted rows were reevaluated with 50 stochastic simulations.

## Repository Structure

Important files and directories:

```text
solver/GA/
    Genetic algorithm, mutation logic, surrogate model, local search, and optional RL components.

scripts/generate_scenario2_uncertainty.py
    Generates the fixed Scenario-2 uncertainty parameters.

scripts/run_scenario2_submission.py
    Main Scenario-2 solver and CSV writer.

scripts/validate_scenario2_submission.py
    Validates the official Scenario-2 CSV output.

submit_scenario2_instances.sh
    Slurm array script for running all 30 instances and merging raw results.

submit_behnkegeiger60_retry.sh
    Separate retry script for the BehnkeGeiger60 instance.

config/scenario2_uncertainty.json
    Fixed uncertainty parameters used during optimization, final evaluation, and CSV export.

results/scenario2_submission/
    Final generated submission CSV files.

used_nodes.txt
used_nodes_hardware.txt
used_nodes_sinfo.txt
    Hardware and Slurm node information for the final runs.
```

## Installation

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The final Slurm scripts were executed in a Linux HPC environment. The main array script loads a Python module where available and then activates `.venv`.

## Uncertainty Parameters

The submitted Scenario-2 runs use fixed uncertainty parameters stored in:

```text
config/scenario2_uncertainty.json
```

For the submitted results, this file should **not** be regenerated. It is part of the submission and must remain unchanged for validation and reproduction of the final CSV.

To generate a new uncertainty file for a fresh experiment, run:

```bash
python scripts/generate_scenario2_uncertainty.py
```

## Quick Smoke Test

A small local test can be executed with:

```bash
python scripts/run_scenario2_submission.py \
    --instances 6_Fattahi_20_workers.fjs \
    --n-runs 2 \
    --limit-runs 2 \
    --allow-failed-runs \
    --output-dir results/scenario2_smoke
```

This is only a functional test. It is not the final competition run.

## Full Scenario-2 Run

The complete experiment consists of 30 benchmark instances and 10 independent runs per instance.

On a local or interactive environment, the solver can be started with:

```bash
python scripts/run_scenario2_submission.py --resume
```

For the final submitted runs, the Slurm array script was used:

```bash
bash submit_scenario2_instances.sh submit
```

The script creates one array task per benchmark instance and runs 10 independent runs per instance.

Useful status command:

```bash
squeue -u "$USER"
```

## Merge Final Results

After all instance jobs have finished, merge the raw results into the final Scenario-2 submission directory:

```bash
bash submit_scenario2_instances.sh merge
```

This writes the final CSV files to:

```text
results/scenario2_submission/
```

and then runs the validator automatically.

A successful full validation prints:

```text
Validation OK: 300 Scenario-2 submission rows are valid.
```

## Validation

The official Scenario-2 CSV can be validated with:

```bash
python scripts/validate_scenario2_submission.py
```

The validator checks:

- exact Scenario-2 column order
- number of rows
- 30 benchmark instances
- 10 runs per instance
- JSON formatting of vector columns
- function-evaluation limit
- consistency with `config/scenario2_uncertainty.json`
- basic decoded schedule feasibility and simulability

## Output Files

The important generated outputs are:

```text
results/scenario2_submission/submission_scenario2.csv
results/scenario2_submission/submission_scenario2_with_metadata.csv
results/scenario2_submission/run_results.csv
results/scenario2_submission/instance_summary.csv
results/scenario2_submission/submission_manifest.json
config/scenario2_uncertainty.json
```

The official result file is:

```text
results/scenario2_submission/submission_scenario2.csv
```

It uses the Scenario-2 format:

```text
Instance;Fitness;FunctionEvaluations;StartTimes;MachineAssignments;WorkerAssignments;UncertaintyParameters
```

`StartTimes`, `MachineAssignments`, and `WorkerAssignments` are ordered by the fixed operation order of the corresponding benchmark instance. `UncertaintyParameters` contains the fixed worker-dependent uncertainty parameters used for the corresponding run.

## Important Packaging Note

The directory

```text
results/scenario2_submission/
```

may be ignored by Git. For the final submission archive, include the generated CSV and metadata files explicitly, or add them intentionally with `git add -f` if they should be versioned.

## Hardware

The final Scenario-2 run was executed as CPU-based Python jobs on the Leibniz University Hannover Slurm cluster.

The main Scenario-2 Slurm array used:

```text
Job name:          sc2inst
Array tasks:       one task per benchmark instance
CPUs per task:     26
Memory per CPU:    2500 MB
Wall time:         42:00:00
Internal sims:     10
Final sims:        50
Max evaluations:   5,000,000
Run workers:       10
Simulation workers: 2
Surrogate jobs:    2
```

The completed jobs were distributed over heterogeneous nodes in the `mpp.share`, `amo`, and `taurus` partitions:

```text
mpp.share nodes: 128 CPUs, 500 GB RAM, AMD, AVX512, NDR200, year 2025
amo nodes:        40 CPUs, 180 GB RAM, Intel, AVX512, HDR100, year 2020
taurus nodes:     32 CPUs, 120 GB RAM, Intel, AVX512, FDR10, year 2019
```

Detailed node information is included in:

```text
used_nodes.txt
used_nodes_hardware.txt
used_nodes_sinfo.txt
```

## Submitted Materials

The submission archive should contain:

```text
README.md
technical_report_scenario2.pdf
config/scenario2_uncertainty.json
results/scenario2_submission/submission_scenario2.csv
results/scenario2_submission/submission_scenario2_with_metadata.csv
results/scenario2_submission/submission_manifest.json
scripts/
solver/
util/
instances/
requirements.txt
submit_scenario2_instances.sh
submit_behnkegeiger60_retry.sh
used_nodes.txt
used_nodes_hardware.txt
used_nodes_sinfo.txt
```

The primary file for the competition ranking is:

```text
results/scenario2_submission/submission_scenario2.csv
```

## References

- FJSSP-W Competition repository: https://github.com/jrc-rodec/FJSSP-W-Competition
- FJSSP-W GA basis by Hutter, Steinberger, and Hellwig: https://github.com/jrc-rodec/FJSSPW-GA
