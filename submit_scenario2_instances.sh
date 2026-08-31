#!/bin/bash -l
set -euo pipefail

MODE="${1:-submit}"

PARTITION="${PARTITION:-mpp.share,amo,taurus,lena,haku,smp}"
CPUS_PER_TASK="${CPUS_PER_TASK:-26}"
CONCURRENT="${CONCURRENT:-30}"
MEM_PER_CPU="${MEM_PER_CPU:-2500M}"
SLURM_TIME="${SLURM_TIME:-42:00:00}"

TIME_LIMIT_S="${TIME_LIMIT_S:-129600}"   # 36h
RUN_WORKERS="${RUN_WORKERS:-10}"
SIMULATION_WORKERS="${SIMULATION_WORKERS:-2}"
SURROGATE_N_JOBS="${SURROGATE_N_JOBS:-2}"

INTERNAL_SIMULATIONS="${INTERNAL_SIMULATIONS:-10}"
FINAL_SIMULATIONS="${FINAL_SIMULATIONS:-50}"
MAX_FUNCTION_EVALUATIONS="${MAX_FUNCTION_EVALUATIONS:-5000000}"

TASK_FILE="chunks/scenario2_instances.tsv"

create_tasks() {
mkdir -p chunks logs results/scenario2_instances

python - <<'PY'
from pathlib import Path

slow = [
    "0_BehnkeGeiger_60_workers.fjs",
    "1_Brandimarte_12_workers.fjs",
    "1_Brandimarte_14_workers.fjs",
    "2a_Hurink_sdata_38_workers.fjs",
    "2a_Hurink_sdata_40_workers.fjs",
    "2c_Hurink_rdata_38_workers.fjs",
    "2d_Hurink_vdata_30_workers.fjs",
    "3_DPpaulli_15_workers.fjs",
    "3_DPpaulli_18_workers.fjs",
    "3_DPpaulli_1_workers.fjs",
    "3_DPpaulli_9_workers.fjs",
    "4_ChambersBarnes_10_workers.fjs",
]

all_instances = sorted(p.name for p in Path("instances/fjssp-w").glob("*.fjs"))
all_set = set(all_instances)

ordered = [x for x in slow if x in all_set]
ordered += [x for x in all_instances if x not in set(ordered)]

with open("chunks/scenario2_instances.tsv", "w") as f:
    for name in ordered:
        f.write(name + "\n")

print(f"Wrote {len(ordered)} instances")
PY
}

submit_jobs() {
create_tasks

N_TASKS=$(wc -l < "$TASK_FILE")
MAX_ARRAY=$((N_TASKS - 1))

JOBID=$(sbatch --parsable \
    --partition="$PARTITION" \
    --job-name=sc2inst \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem-per-cpu="$MEM_PER_CPU" \
    --time="$SLURM_TIME" \
    --array="0-${MAX_ARRAY}%${CONCURRENT}" \
    --output="logs/sc2inst_%A_%a.out" \
    --error="logs/sc2inst_%A_%a.err" \
    --export="ALL,TASK_FILE=${TASK_FILE},TIME_LIMIT_S=${TIME_LIMIT_S},RUN_WORKERS=${RUN_WORKERS},SIMULATION_WORKERS=${SIMULATION_WORKERS},SURROGATE_N_JOBS=${SURROGATE_N_JOBS},INTERNAL_SIMULATIONS=${INTERNAL_SIMULATIONS},FINAL_SIMULATIONS=${FINAL_SIMULATIONS},MAX_FUNCTION_EVALUATIONS=${MAX_FUNCTION_EVALUATIONS}" \
    "$0" worker)

echo "Submitted job array: $JOBID"
echo "30 Instanzen parallel, je 10 Runs parallel"
echo "Max CPUs: $((CONCURRENT * CPUS_PER_TASK))"
echo "Status: squeue -u \$USER"
echo "Merge später: bash $0 merge"
}

worker() {
cd "$SLURM_SUBMIT_DIR"

module load Python/3.11.6 2>/dev/null || true
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

INSTANCE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
SAFE_INSTANCE="${INSTANCE%.fjs}"
OUT_DIR="results/scenario2_instances/$SAFE_INSTANCE"

echo "INSTANCE=$INSTANCE"
echo "OUT_DIR=$OUT_DIR"
echo "CPUS=$SLURM_CPUS_PER_TASK"
echo "RUN_WORKERS=$RUN_WORKERS"
echo "SIMULATION_WORKERS=$SIMULATION_WORKERS"
echo "SURROGATE_N_JOBS=$SURROGATE_N_JOBS"
echo "TIME_LIMIT_S=$TIME_LIMIT_S"

python scripts/run_scenario2_submission.py \
    --resume \
    --instances "$INSTANCE" \
    --output-dir "$OUT_DIR" \
    --n-runs 10 \
    --time-limit-s "$TIME_LIMIT_S" \
    --max-function-evaluations "$MAX_FUNCTION_EVALUATIONS" \
    --internal-simulations "$INTERNAL_SIMULATIONS" \
    --final-simulations "$FINAL_SIMULATIONS" \
    --workers "$RUN_WORKERS" \
    --simulation-workers "$SIMULATION_WORKERS" \
    --surrogate-n-jobs "$SURROGATE_N_JOBS"
}

merge_results() {
python - <<'PY'
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys

from scripts.run_scenario2_submission import write_csv_outputs

rows_by_key = {}

for raw_path in sorted(Path("results/scenario2_instances").glob("*/raw_results.jsonl")):
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows_by_key[(row["instance"], int(row["run"]))] = row

instances = sorted(p.name for p in Path("instances/fjssp-w").glob("*.fjs"))
expected = [(inst, run) for inst in instances for run in range(1, 11)]

missing = [key for key in expected if key not in rows_by_key]
rows = [rows_by_key[key] for key in expected if key in rows_by_key]

args = SimpleNamespace(
    n_runs=10,
    workers=int(os.environ.get("RUN_WORKERS", "10")),
    time_limit_s=int(os.environ.get("TIME_LIMIT_S", "129600")),
    max_function_evaluations=int(os.environ.get("MAX_FUNCTION_EVALUATIONS", "5000000")),
    internal_simulations=int(os.environ.get("INTERNAL_SIMULATIONS", "10")),
    final_simulations=int(os.environ.get("FINAL_SIMULATIONS", "50")),
    surrogate_n_jobs=int(os.environ.get("SURROGATE_N_JOBS", "2")),
    simulation_workers=int(os.environ.get("SIMULATION_WORKERS", "2")),
    disable_local_search=False,
)

write_csv_outputs(
    Path("results/scenario2_submission"),
    rows,
    args,
    Path("config/scenario2_uncertainty.json"),
    expected_instances=30,
)

print(f"Merged OK rows: {len(rows)}")

if missing:
    print("Missing runs:")
    for instance, run in missing[:80]:
        print(f"  {instance} run {run}")
    if len(missing) > 80:
        print(f"  ... and {len(missing) - 80} more")
    sys.exit(2)

print("All 300 runs present.")
PY

python scripts/validate_scenario2_submission.py
}

case "$MODE" in
    submit) submit_jobs ;;
    worker) worker ;;
    merge) merge_results ;;
    *) echo "Usage: bash $0 submit|worker|merge"; exit 2 ;;
esac