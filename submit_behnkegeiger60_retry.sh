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

INSTANCE="0_BehnkeGeiger_60_workers.fjs"
RUNS=10

TIME_LIMIT_S=129600
SLURM_TIME="37:00:00"

INTERNAL_SIMULATIONS=10
FINAL_SIMULATIONS=50
MAX_FUNCTION_EVALUATIONS=5000000

PARTITION="mpp.share"
MEM_PER_CPU="3900M"

# 2 Runs pro SLURM-Job.
# Pro Run:
#   1 Hauptprozess
#   10 Simulation-Worker
#   2 Surrogate-Jobs
# = ca. 13 CPUs pro Run
# => 2 Runs * 13 = 26 CPUs pro Job
CPUS_PER_JOB=26
SIM_WORKERS_PER_RUN=10
SURROGATE_N_JOBS_PER_RUN=2

TASK_DIR="chunks"
RESULT_TASK_DIR="results/scenario2_task_results"
PAIR_FILE="${TASK_DIR}/behnkegeiger60_retry_pairs.tsv"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p "$TASK_DIR" "$RESULT_TASK_DIR" logs tmp

    cat > "$PAIR_FILE" <<EOF
$INSTANCE	1	$INSTANCE	2
$INSTANCE	3	$INSTANCE	4
$INSTANCE	5	$INSTANCE	6
$INSTANCE	7	$INSTANCE	8
$INSTANCE	9	$INSTANCE	10
EOF

    N_JOBS=$(wc -l < "$PAIR_FILE")

    JOB_ID=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name=bg60_retry \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_JOB" \
        --mem-per-cpu="$MEM_PER_CPU" \
        --time="$SLURM_TIME" \
        --array="0-$((N_JOBS - 1))%5" \
        --output="logs/bg60_retry_%A_%a.out" \
        --error="logs/bg60_retry_%A_%a.err" \
        --export="ALL,PAIR_FILE=${PAIR_FILE},RESULT_TASK_DIR=${RESULT_TASK_DIR},TIME_LIMIT_S=${TIME_LIMIT_S},INTERNAL_SIMULATIONS=${INTERNAL_SIMULATIONS},FINAL_SIMULATIONS=${FINAL_SIMULATIONS},MAX_FUNCTION_EVALUATIONS=${MAX_FUNCTION_EVALUATIONS},SIM_WORKERS_PER_RUN=${SIM_WORKERS_PER_RUN},SURROGATE_N_JOBS_PER_RUN=${SURROGATE_N_JOBS_PER_RUN}" \
        "$0")

    echo "Submitted BehnkeGeiger60 retry job: $JOB_ID"
    echo "Jobs: $N_JOBS"
    echo "CPUs per job: $CPUS_PER_JOB"
    echo "Simulation workers per run: $SIM_WORKERS_PER_RUN"
    echo "Surrogate jobs per run: $SURROGATE_N_JOBS_PER_RUN"
    echo "Max CPUs: $((N_JOBS * CPUS_PER_JOB))"
    echo
    echo "Status:"
    echo "  squeue -u \$USER"
    echo
    echo "Progress:"
    echo "  find results/scenario2_task_results/0_BehnkeGeiger_60_workers -name result.json | wc -l"
    echo
    echo "Errors:"
    echo "  grep -R '\"status\": \"error\"' results/scenario2_task_results/0_BehnkeGeiger_60_workers -n"

    exit 0
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TMP_BASE="$REPO_DIR/tmp/bg60_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMP_BASE"

export TMPDIR="$TMP_BASE"
export TMP="$TMP_BASE"
export TEMP="$TMP_BASE"
export JOBLIB_TEMP_FOLDER="$TMP_BASE"

trap 'rm -rf "$TMP_BASE"' EXIT

PAIR_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PAIR_FILE")
IFS=$'\t' read -r INSTANCE1 RUN1 INSTANCE2 RUN2 <<< "$PAIR_LINE"

is_done_ok() {
    local RESULT_JSON="$1"

    [[ -s "$RESULT_JSON" ]] || return 1

    "$PYTHON" - "$RESULT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

try:
    row = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)

sys.exit(0 if row.get("status") == "ok" else 1)
PY
}

run_one() {
    local INSTANCE="$1"
    local RUN="$2"

    [[ -z "$INSTANCE" ]] && return 0
    [[ -z "$RUN" ]] && return 0

    local RUN_PAD
    RUN_PAD=$(printf "%02d" "$RUN")

    local INST_BASE="${INSTANCE%.fjs}"
    local OUT_DIR="${RESULT_TASK_DIR}/${INST_BASE}/run_${RUN_PAD}"
    local RESULT_JSON="${OUT_DIR}/result.json"

    mkdir -p "$OUT_DIR"

    if is_done_ok "$RESULT_JSON"; then
        echo "Already OK: $INSTANCE run $RUN"
        return 0
    fi

    rm -f "$RESULT_JSON" "${RESULT_JSON}.tmp"

    echo "Start retry: $INSTANCE run $RUN"
    echo "Repo: $REPO_DIR"
    echo "Python: $PYTHON"
    echo "TMPDIR: $TMPDIR"
    echo "Output: $OUT_DIR"
    echo "Simulation workers per run: $SIM_WORKERS_PER_RUN"
    echo "Surrogate jobs per run: $SURROGATE_N_JOBS_PER_RUN"

    "$PYTHON" - "$INSTANCE" "$RUN" "$OUT_DIR" <<'PY'
from pathlib import Path
import json
import os
import sys
import traceback

from scripts.run_scenario2_submission import (
    load_uncertainty,
    solve_run_task,
    to_builtin,
    uncertainty_for,
)

instance = sys.argv[1]
run = int(sys.argv[2])
out_dir = Path(sys.argv[3])

try:
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
        "surrogate_n_jobs": int(os.environ["SURROGATE_N_JOBS_PER_RUN"]),
        "simulation_workers": int(os.environ["SIM_WORKERS_PER_RUN"]),
        "disable_local_search": False,
    }

    row = solve_run_task(task)

except Exception as exc:
    traceback.print_exc()
    row = {
        "instance": instance,
        "run": run,
        "seed": locals().get("seed", None),
        "status": "error",
        "error": str(exc),
    }

tmp = out_dir / "result.json.tmp"
final = out_dir / "result.json"

tmp.write_text(
    json.dumps(to_builtin(row), sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
tmp.replace(final)

print(f"Wrote {final} status={row.get('status')}")

if row.get("status") != "ok":
    raise SystemExit(1)
PY
}

run_one "$INSTANCE1" "$RUN1" &
PID1=$!

run_one "$INSTANCE2" "$RUN2" &
PID2=$!

STATUS=0

wait "$PID1" || STATUS=1
wait "$PID2" || STATUS=1

exit "$STATUS"