#!/bin/bash -l
set -euo pipefail

MODE="${1:-help}"

SCRIPT="scripts/compare_hpo_rl_factorial_scenario2.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/results/hpo_rl_factorial_scenario2}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.6}"

# General LUIS CPU partitions only.  Keeping this explicit prevents Slurm
# from extending an unspecified request to opportunistic FCH partitions such
# as ai or stahl.  GPU resources are neither requested nor required.
PARTITION="${PARTITION:-mpp.share,amo,taurus,lena,haku,smp}"
CPUS_PER_TASK="${CPUS_PER_TASK:-26}"
MEM_PER_CPU="${MEM_PER_CPU:-2500M}"
SLURM_TIME="${SLURM_TIME:-42:00:00}"
CONCURRENT="${CONCURRENT:-8}"

N_RUNS="${N_RUNS:-10}"
FINAL_SIMULATIONS="${FINAL_SIMULATIONS:-50}"
MAX_FUNCTION_EVALUATIONS="${MAX_FUNCTION_EVALUATIONS:-5000000}"
EA_TIME_LIMIT_S="${EA_TIME_LIMIT_S:-129600}"
RUN_WORKERS="${RUN_WORKERS:-10}"
SIMULATION_WORKERS="${SIMULATION_WORKERS:-2}"
SURROGATE_N_JOBS="${SURROGATE_N_JOBS:-2}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260803}"

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

common_args=(
    --output-root "$OUTPUT_ROOT"
    --n-runs "$N_RUNS"
    --final-simulations "$FINAL_SIMULATIONS"
    --max-function-evaluations "$MAX_FUNCTION_EVALUATIONS"
    --time-limit-s "$EA_TIME_LIMIT_S"
    --workers "$RUN_WORKERS"
    --simulation-workers "$SIMULATION_WORKERS"
    --surrogate-n-jobs "$SURROGATE_N_JOBS"
    --bootstrap-samples "$BOOTSTRAP_SAMPLES"
    --bootstrap-seed "$BOOTSTRAP_SEED"
)

usage() {
    cat <<EOF
Usage:
  bash submit_hpo_rl_factorial_scenario2.sh submit
  bash submit_hpo_rl_factorial_scenario2.sh summarize
  bash submit_hpo_rl_factorial_scenario2.sh tasks
  bash submit_hpo_rl_factorial_scenario2.sh worker    # internal array mode

The experiment contains 120 array tasks (4 variants x 30 instances),
10 paired runs per task, 5,000,000 function evaluations, 50 final
simulations, and a 36-hour solver limit. Submission never summarizes
automatically; run the summarize command manually after the array finishes.

Optional environment overrides:
  OUTPUT_ROOT=$OUTPUT_ROOT
  PARTITION=$PARTITION
  CONCURRENT=$CONCURRENT
  CPUS_PER_TASK=$CPUS_PER_TASK
  MEM_PER_CPU=$MEM_PER_CPU
  SLURM_TIME=$SLURM_TIME
  PYTHON_MODULE=$PYTHON_MODULE
EOF
}

submit_jobs() {
    mkdir -p "$OUTPUT_ROOT/logs"
    sbatch_args=(
        --parsable
        --job-name=hpo_rl_2x2_s2
        --nodes=1
        --ntasks=1
        --cpus-per-task="$CPUS_PER_TASK"
        --mem-per-cpu="$MEM_PER_CPU"
        --time="$SLURM_TIME"
        --array="0-119%${CONCURRENT}"
        --output="$OUTPUT_ROOT/logs/hpo_rl_2x2_%A_%a.out"
        --error="$OUTPUT_ROOT/logs/hpo_rl_2x2_%A_%a.err"
        --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},N_RUNS=${N_RUNS},FINAL_SIMULATIONS=${FINAL_SIMULATIONS},MAX_FUNCTION_EVALUATIONS=${MAX_FUNCTION_EVALUATIONS},EA_TIME_LIMIT_S=${EA_TIME_LIMIT_S},RUN_WORKERS=${RUN_WORKERS},SIMULATION_WORKERS=${SIMULATION_WORKERS},SURROGATE_N_JOBS=${SURROGATE_N_JOBS},BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES},BOOTSTRAP_SEED=${BOOTSTRAP_SEED},PYTHON_MODULE=${PYTHON_MODULE}"
    )
    sbatch_args+=(--partition="$PARTITION")
    job_id=$(sbatch "${sbatch_args[@]}" "$0" worker)
    echo "Submitted 2x2 HPO/RL Scenario-2 array as job $job_id."
    echo "After all 120 tasks finish, run: bash $0 summarize"
}

worker() {
    cd "$SLURM_SUBMIT_DIR"
    prepare_env
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    echo "OUTPUT_ROOT=$OUTPUT_ROOT"
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-$CPUS_PER_TASK}" \
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
