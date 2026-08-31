#!/bin/bash -l
set -euo pipefail

MODE="${1:-help}"

PARTITION="${PARTITION:-mpp.share,amo,taurus,lena,haku,smp}"
CPUS_PER_TASK="${CPUS_PER_TASK:-26}"
MEM_PER_CPU="${MEM_PER_CPU:-2500M}"
SLURM_TIME="${SLURM_TIME:-08:00:00}"
CONCURRENT="${CONCURRENT:-8}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.6}"

RL_HPO_ROOT="${RL_HPO_ROOT:-$PWD/results/hpo_rl_scenario2}"
SOURCE_HPO_ROOT="${SOURCE_HPO_ROOT:-$PWD/results/hpo_scenario2}"
SCRIPT="scripts/hpo_rl_scenario2.py"

INSTANCE_COUNT="${INSTANCE_COUNT:-8}"
N_RUNS="${N_RUNS:-2}"
FINAL_SIMULATIONS="${FINAL_SIMULATIONS:-20}"
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

common_args=(
    --hpo-root "$RL_HPO_ROOT"
    --source-hpo-root "$SOURCE_HPO_ROOT"
    --instance-count "$INSTANCE_COUNT"
    --n-runs "$N_RUNS"
    --final-simulations "$FINAL_SIMULATIONS"
    --max-function-evaluations "$MAX_FUNCTION_EVALUATIONS"
    --time-limit-s "$EA_TIME_LIMIT_S"
    --workers "$RUN_WORKERS"
    --simulation-workers "$SIMULATION_WORKERS"
    --surrogate-n-jobs "$SURROGATE_N_JOBS"
)

usage() {
    cat <<EOF
Usage:
  bash submit_hpo_rl_scenario2.sh prepare
  bash submit_hpo_rl_scenario2.sh submit
  bash submit_hpo_rl_scenario2.sh summarize
  bash submit_hpo_rl_scenario2.sh configs

Defaults: 9 configs, 8 training instances, 2 runs, 7200 seconds per GA run.
Override cluster settings through PARTITION, CONCURRENT, CPUS_PER_TASK,
MEM_PER_CPU, SLURM_TIME, PYTHON_MODULE, or the run variables in this script.
EOF
}

prepare() {
    prepare_env
    python "$SCRIPT" prepare "${common_args[@]}"
}

submit() {
    prepare
    mkdir -p "$RL_HPO_ROOT/logs"
    N_CONFIGS=$(wc -l < "$RL_HPO_ROOT/configs.jsonl")
    MAX_ARRAY=$((N_CONFIGS - 1))
    JOB_ID=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name="hpo_rl_s2" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem-per-cpu="$MEM_PER_CPU" \
        --time="$SLURM_TIME" \
        --array="0-${MAX_ARRAY}%${CONCURRENT}" \
        --output="$RL_HPO_ROOT/logs/hpo_rl_%A_%a.out" \
        --error="$RL_HPO_ROOT/logs/hpo_rl_%A_%a.err" \
        --export="ALL,RL_HPO_ROOT=${RL_HPO_ROOT},SOURCE_HPO_ROOT=${SOURCE_HPO_ROOT},INSTANCE_COUNT=${INSTANCE_COUNT},N_RUNS=${N_RUNS},FINAL_SIMULATIONS=${FINAL_SIMULATIONS},MAX_FUNCTION_EVALUATIONS=${MAX_FUNCTION_EVALUATIONS},EA_TIME_LIMIT_S=${EA_TIME_LIMIT_S},RUN_WORKERS=${RUN_WORKERS},SIMULATION_WORKERS=${SIMULATION_WORKERS},SURROGATE_N_JOBS=${SURROGATE_N_JOBS},PYTHON_MODULE=${PYTHON_MODULE}" \
        "$0" worker)

    echo "Submitted RL mini-HPO as Slurm array job $JOB_ID ($N_CONFIGS configs)."
    echo "After completion: bash submit_hpo_rl_scenario2.sh summarize"
}

worker() {
    cd "$SLURM_SUBMIT_DIR"
    prepare_env
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

configs() {
    prepare_env
    python "$SCRIPT" print-configs "${common_args[@]}"
}

case "$MODE" in
    help|-h|--help) usage ;;
    prepare) prepare ;;
    submit) submit ;;
    worker) worker ;;
    summarize) summarize ;;
    configs) configs ;;
    *) usage; exit 2 ;;
esac