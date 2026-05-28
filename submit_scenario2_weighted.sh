#!/bin/bash -l
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

cd "$REPO_DIR"

PYTHON="$REPO_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python venv not found at $PYTHON" >&2
    exit 2
fi

RUNS=10
TIME_LIMIT_S=129600
SLURM_TIME="37:00:00"

INTERNAL_SIMULATIONS=10
FINAL_SIMULATIONS=50
MAX_FUNCTION_EVALUATIONS=5000000

# INSTANZ-INDIZES in der sortierten Liste von instances/fjssp-w/*.fjs.
# Index 0 ist die erste Datei nach sortierter Reihenfolge.
SLOW_INSTANCE_IDS="2 3 4 8 9 16 19 21 22 23 24 25"

PARTITION="mpp.share"
MEM_PER_CPU="3900M"

CPUS_PER_JOB=12
SIM_WORKERS_PER_RUN=5

SLOW_CONCURRENT=52
FAST_CONCURRENT=12

TASK_DIR="chunks"
RESULT_TASK_DIR="results/scenario2_task_results"

SLOW_PAIR_FILE="${TASK_DIR}/scenario2_pairs_slow.tsv"
FAST_PAIR_FILE="${TASK_DIR}/scenario2_pairs_fast.tsv"
INSTANCE_ORDER_FILE="${TASK_DIR}/scenario2_instance_order.tsv"

if [[ "${1:-}" == "merge" ]]; then
    "$PYTHON" - <<'PY'
from argparse import Namespace
from pathlib import Path
import json
import sys

from scripts.run_scenario2_submission import selected_instance_files, write_csv_outputs

RUNS = 10
TIME_LIMIT_S = 129600
INTERNAL_SIMULATIONS = 10
FINAL_SIMULATIONS = 50
MAX_FUNCTION_EVALUATIONS = 5_000_000

pair_files = [
    Path("chunks/scenario2_pairs_slow.tsv"),
    Path("chunks/scenario2_pairs_fast.tsv"),
]

rows = {}
missing = []

for pair_file in pair_files:
    if not pair_file.exists():
        print(f"Missing pair file: {pair_file}", file=sys.stderr)
        sys.exit(2)

    with pair_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")

            for i in (0, 3):
                if i >= len(parts) or not parts[i]:
                    continue

                instance, run_s, kind = parts[i:i+3]
                run = int(run_s)
                inst_base = instance.removesuffix(".fjs")

                path = (
                    Path("results/scenario2_task_results")
                    / inst_base
                    / f"run_{run:02d}"
                    / "result.json"
                )

                if not path.exists():
                    missing.append(f"{instance} run {run}")
                    continue

                row = json.loads(path.read_text(encoding="utf-8"))
                rows[(row["instance"], int(row["run"]))] = row

if missing:
    print(f"Missing {len(missing)} result(s):", file=sys.stderr)
    for item in missing[:40]:
        print("  " + item, file=sys.stderr)
    sys.exit(2)

instance_files = selected_instance_files(Path("instances/fjssp-w"), None)
expected = len(instance_files) * RUNS

if len(rows) != expected:
    print(f"Expected {expected} rows, found {len(rows)}.", file=sys.stderr)
    sys.exit(2)

args = Namespace(
    n_runs=RUNS,
    time_limit_s=TIME_LIMIT_S,
    max_function_evaluations=MAX_FUNCTION_EVALUATIONS,
    internal_simulations=INTERNAL_SIMULATIONS,
    final_simulations=FINAL_SIMULATIONS,
    workers=2,
    surrogate_n_jobs=1,
    simulation_workers=5,
    disable_local_search=False,
)

ordered_rows = [rows[key] for key in sorted(rows.keys())]

write_csv_outputs(
    Path("results/scenario2_submission"),
    ordered_rows,
    args,
    Path("config/scenario2_uncertainty.json"),
    expected_instances=len(instance_files),
)

print(f"Merged {len(ordered_rows)} rows into results/scenario2_submission/")
PY
    exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$TASK_DIR" "$RESULT_TASK_DIR" logs

    "$PYTHON" - <<'PY'
from pathlib import Path

RUNS = 10
SLOW_INSTANCE_IDS = {2, 3, 4, 8, 9, 16, 19, 21, 22, 23, 24, 25}

instances = sorted(p.name for p in Path("instances/fjssp-w").glob("*.fjs"))

slow = []
fast = []

Path("chunks").mkdir(exist_ok=True)

with open("chunks/scenario2_instance_order.tsv", "w", encoding="utf-8") as f:
    f.write("index\tkind\tinstance\n")

    for idx, instance in enumerate(instances):
        kind = "slow" if idx in SLOW_INSTANCE_IDS else "fast"
        f.write(f"{idx}\t{kind}\t{instance}\n")

        target = slow if kind == "slow" else fast

        for run in range(1, RUNS + 1):
            target.append((instance, run, kind))

def write_pairs(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(0, len(rows), 2):
            a = rows[i]
            b = rows[i + 1] if i + 1 < len(rows) else ("", "", "")
            f.write("\t".join(map(str, (*a, *b))) + "\n")

write_pairs("chunks/scenario2_pairs_slow.tsv", slow)
write_pairs("chunks/scenario2_pairs_fast.tsv", fast)

print("Instance order written to: chunks/scenario2_instance_order.tsv")
print()
print("Slow instances:")
for idx, instance in enumerate(instances):
    if idx in SLOW_INSTANCE_IDS:
        print(f"  {idx}: {instance}")

print()
print(f"slow tasks: {len(slow)} -> pair jobs: {(len(slow) + 1) // 2}")
print(f"fast tasks: {len(fast)} -> pair jobs: {(len(fast) + 1) // 2}")
print(f"total tasks: {len(slow) + len(fast)}")
PY

    N_SLOW=$(wc -l < "$SLOW_PAIR_FILE")
    N_FAST=$(wc -l < "$FAST_PAIR_FILE")

    COMMON_EXPORT="ALL,RUNS=${RUNS},TIME_LIMIT_S=${TIME_LIMIT_S},INTERNAL_SIMULATIONS=${INTERNAL_SIMULATIONS},FINAL_SIMULATIONS=${FINAL_SIMULATIONS},MAX_FUNCTION_EVALUATIONS=${MAX_FUNCTION_EVALUATIONS},SIM_WORKERS_PER_RUN=${SIM_WORKERS_PER_RUN}"

    SLOW_JOB=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name=sc2_slow \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_JOB" \
        --mem-per-cpu="$MEM_PER_CPU" \
        --time="$SLURM_TIME" \
        --array="0-$((N_SLOW - 1))%${SLOW_CONCURRENT}" \
        --output="logs/sc2_slow_%A_%a.out" \
        --error="logs/sc2_slow_%A_%a.err" \
        --export="${COMMON_EXPORT},PAIR_FILE=${SLOW_PAIR_FILE}" \
        "$0")

    FAST_JOB=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name=sc2_fast \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_JOB" \
        --mem-per-cpu="$MEM_PER_CPU" \
        --time="$SLURM_TIME" \
        --array="0-$((N_FAST - 1))%${FAST_CONCURRENT}" \
        --output="logs/sc2_fast_%A_%a.out" \
        --error="logs/sc2_fast_%A_%a.err" \
        --export="${COMMON_EXPORT},PAIR_FILE=${FAST_PAIR_FILE}" \
        "$0")

    echo "Submitted slow job: $SLOW_JOB"
    echo "Submitted fast job: $FAST_JOB"
    echo
    echo "Max running jobs: $((SLOW_CONCURRENT + FAST_CONCURRENT))"
    echo "Max CPUs: $(((SLOW_CONCURRENT + FAST_CONCURRENT) * CPUS_PER_JOB))"
    echo
    echo "Expected result.json files: 300"
    echo
    echo "Status:"
    echo "  squeue -u \$USER"
    echo
    echo "Check instance order:"
    echo "  column -t chunks/scenario2_instance_order.tsv | less"
    echo
    echo "Merge after completion:"
    echo "  bash submit_scenario2_weighted.sh merge"

    exit 0
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PAIR_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PAIR_FILE")
IFS=$'\t' read -r INSTANCE1 RUN1 KIND1 INSTANCE2 RUN2 KIND2 <<< "$PAIR_LINE"

run_one() {
    local INSTANCE="$1"
    local RUN="$2"
    local KIND="$3"

    [[ -z "$INSTANCE" ]] && return 0

    local RUN_PAD
    RUN_PAD=$(printf "%02d" "$RUN")

    local INST_BASE="${INSTANCE%.fjs}"
    local OUT_DIR="${RESULT_TASK_DIR}/${INST_BASE}/run_${RUN_PAD}"
    local RESULT_JSON="${OUT_DIR}/result.json"

    mkdir -p "$OUT_DIR"

    if [[ -s "$RESULT_JSON" ]]; then
        echo "Already done: $INSTANCE run $RUN"
        return 0
    fi

    echo "Start: $INSTANCE run $RUN kind=$KIND sim_workers=$SIM_WORKERS_PER_RUN"
    echo "Repo: $REPO_DIR"
    echo "Python: $PYTHON"
    echo "Output: $OUT_DIR"

    "$PYTHON" - "$INSTANCE" "$RUN" "$OUT_DIR" <<'PY'
from pathlib import Path
import json
import os
import sys

from scripts.run_scenario2_submission import (
    load_uncertainty,
    solve_run_task,
    to_builtin,
    uncertainty_for,
)

instance = sys.argv[1]
run = int(sys.argv[2])
out_dir = Path(sys.argv[3])

payload = load_uncertainty(Path("config/scenario2_uncertainty.json"))
seed, uncertainty_parameters = uncertainty_for(payload, instance, run)

task = {
    "progress_index": 1,
    "total_expected": 1,
    "instance_path": str(Path("instances/fjssp-w") / instance),
    "instance": instance,
    "run": run,
    "seed": seed,
    "uncertainty_parameters": uncertainty_parameters,
    "internal_simulations": int(os.environ["INTERNAL_SIMULATIONS"]),
    "final_simulations": int(os.environ["FINAL_SIMULATIONS"]),
    "time_limit_s": int(os.environ["TIME_LIMIT_S"]),
    "max_function_evaluations": int(os.environ["MAX_FUNCTION_EVALUATIONS"]),
    "surrogate_n_jobs": 1,
    "simulation_workers": int(os.environ["SIM_WORKERS_PER_RUN"]),
    "disable_local_search": False,
}

row = solve_run_task(task)

tmp = out_dir / "result.json.tmp"
final = out_dir / "result.json"

tmp.write_text(
    json.dumps(to_builtin(row), sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
tmp.replace(final)

print(f"Wrote {final}")
PY
}

run_one "$INSTANCE1" "$RUN1" "$KIND1" &
PID1=$!

run_one "$INSTANCE2" "$RUN2" "$KIND2" &
PID2=$!

STATUS=0

wait "$PID1" || STATUS=1
wait "$PID2" || STATUS=1

exit "$STATUS"