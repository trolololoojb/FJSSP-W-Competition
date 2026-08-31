#!/bin/bash -l
set -euo pipefail

MODE="${1:-submit}"

PARTITION="${PARTITION:-mpp.share,amo,taurus,lena,haku,smp}"
CPUS_PER_TASK="${CPUS_PER_TASK:-26}"
MEM_PER_CPU="${MEM_PER_CPU:-2500M}"
SLURM_TIME="${SLURM_TIME:-42:00:00}"
CONCURRENT="${CONCURRENT:-8}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/results/c0_rl_scenario2}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.6}"
SCRIPT="scripts/compare_c0_rl_scenario2.py"

N_RUNS="${N_RUNS:-10}"
INTERNAL_SIMULATIONS="${INTERNAL_SIMULATIONS:-12}"
FINAL_SIMULATIONS="${FINAL_SIMULATIONS:-50}"
MAX_FUNCTION_EVALUATIONS="${MAX_FUNCTION_EVALUATIONS:-5000000}"
EA_TIME_LIMIT_S="${EA_TIME_LIMIT_S:-7200}"
RUN_WORKERS="${RUN_WORKERS:-10}"
SIMULATION_WORKERS="${SIMULATION_WORKERS:-2}"
SURROGATE_N_JOBS="${SURROGATE_N_JOBS:-2}"

prepare_env() {
    module load "$PYTHON_MODULE" 2>/dev/null || true
    if [[ -d .venv ]]; then
        source .venv/bin/activate
    fi
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
}

usage() {
    cat <<EOF
Usage:
  bash submit_c0_rl_scenario2.sh submit
  bash submit_c0_rl_scenario2.sh summarize
  bash submit_c0_rl_scenario2.sh tasks
  bash submit_c0_rl_scenario2.sh worker      # internal Slurm array mode

Environment variables:
  OUTPUT_ROOT=$OUTPUT_ROOT
  CONCURRENT=$CONCURRENT
  CPUS_PER_TASK=$CPUS_PER_TASK
  EA_TIME_LIMIT_S=$EA_TIME_LIMIT_S
  N_RUNS=$N_RUNS
  RUN_WORKERS=$RUN_WORKERS
  SIMULATION_WORKERS=$SIMULATION_WORKERS
  SURROGATE_N_JOBS=$SURROGATE_N_JOBS
EOF
}

common_args=(
    --output-root "$OUTPUT_ROOT"
    --n-runs "$N_RUNS"
    --internal-simulations "$INTERNAL_SIMULATIONS"
    --final-simulations "$FINAL_SIMULATIONS"
    --max-function-evaluations "$MAX_FUNCTION_EVALUATIONS"
    --time-limit-s "$EA_TIME_LIMIT_S"
    --workers "$RUN_WORKERS"
    --simulation-workers "$SIMULATION_WORKERS"
    --surrogate-n-jobs "$SURROGATE_N_JOBS"
)

submit_jobs() {
    prepare_env
    mkdir -p "$OUTPUT_ROOT/logs"
    N_TASKS=$(python "$SCRIPT" print-tasks "${common_args[@]}" | wc -l)
    if [[ "$N_TASKS" -le 0 ]]; then
        echo "No comparison tasks generated" >&2
        exit 2
    fi
    MAX_ARRAY=$((N_TASKS - 1))

    JOBID=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name="c0_rl_s2" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem-per-cpu="$MEM_PER_CPU" \
        --time="$SLURM_TIME" \
        --array="0-${MAX_ARRAY}%${CONCURRENT}" \
        --output="$OUTPUT_ROOT/logs/c0_rl_%A_%a.out" \
        --error="$OUTPUT_ROOT/logs/c0_rl_%A_%a.err" \
        --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},N_RUNS=${N_RUNS},INTERNAL_SIMULATIONS=${INTERNAL_SIMULATIONS},FINAL_SIMULATIONS=${FINAL_SIMULATIONS},MAX_FUNCTION_EVALUATIONS=${MAX_FUNCTION_EVALUATIONS},EA_TIME_LIMIT_S=${EA_TIME_LIMIT_S},RUN_WORKERS=${RUN_WORKERS},SIMULATION_WORKERS=${SIMULATION_WORKERS},SURROGATE_N_JOBS=${SURROGATE_N_JOBS},PYTHON_MODULE=${PYTHON_MODULE}" \
        "$0" worker)

    echo "Submitted C0 vs C0+RL comparison as job array: $JOBID"
    echo "Tasks: $N_TASKS (2 variants x all instances), concurrency: $CONCURRENT"
    echo "Status: squeue -u \$USER -r"
}

worker() {
    cd "$SLURM_SUBMIT_DIR"
    prepare_env
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    echo "OUTPUT_ROOT=$OUTPUT_ROOT"
    echo "EA_TIME_LIMIT_S=$EA_TIME_LIMIT_S"
    echo "RUN_WORKERS=$RUN_WORKERS"

    python "$SCRIPT" run-task \
        "${common_args[@]}" \
        --task-index "$SLURM_ARRAY_TASK_ID" \
        --resume \
        --allow-failed-runs
}

summarize() {
    prepare_env
    python "$SCRIPT" summarize "${common_args[@]}"
}

tasks() {
    prepare_env
    python "$SCRIPT" print-tasks "${common_args[@]}"
}

case "$MODE" in
    help|-h|--help) usage ;;
    submit) submit_jobs ;;
    worker) worker ;;
    summarize) summarize ;;
    tasks) tasks ;;
    *) usage; exit 2 ;;
esac
