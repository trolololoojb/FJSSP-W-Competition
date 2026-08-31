#!/bin/bash -l
set -euo pipefail

MODE="${1:-help}"

SCRIPT="scripts/compare_hpo_component_factorial_scenario2.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/results/hpo_component_factorial_scenario2}"
REFERENCE_ROOT="${REFERENCE_ROOT:-$PWD/results/hpo_rl_factorial_scenario2}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.6}"

# Scheduler and process-allocation settings are operational only. Scientific
# protocol values are fixed and validated by the Python experiment runner.
PARTITION="${PARTITION:-mpp.share,amo,taurus,lena,haku,smp}"
CPUS_PER_TASK="${CPUS_PER_TASK:-26}"
MEM_PER_CPU="${MEM_PER_CPU:-2500M}"
SLURM_TIME="${SLURM_TIME:-42:00:00}"
CONCURRENT="${CONCURRENT:-8}"
EXPECTED_TASKS=60

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
    --reference-root "$REFERENCE_ROOT"
)

usage() {
    cat <<EOF
Usage:
  bash $0 submit
  bash $0 preflight
  bash $0 summarize
  bash $0 tasks
  bash $0 worker      # internal Slurm array mode

The experiment runs only the two missing HPO component variants
(plain GA and RL-only): 2 variants x 30 instances = 60 array tasks.
The existing no-RL and full variants are read from REFERENCE_ROOT and are
never submitted, copied, or modified. Scientific protocol parameters are
fixed and validated by the experiment runner.

Optional operational environment overrides:
  OUTPUT_ROOT=$OUTPUT_ROOT
  REFERENCE_ROOT=$REFERENCE_ROOT
  PARTITION=$PARTITION
  CONCURRENT=$CONCURRENT
  CPUS_PER_TASK=$CPUS_PER_TASK
  MEM_PER_CPU=$MEM_PER_CPU
  SLURM_TIME=$SLURM_TIME
  PYTHON_MODULE=$PYTHON_MODULE
EOF
}

task_count() {
    python "$SCRIPT" print-tasks "${common_args[@]}" | awk 'NF { count++ } END { print count + 0 }'
}

validated_task_count() {
    local count
    count="$(task_count)"
    if [[ ! "$count" =~ ^[0-9]+$ ]] || (( count != EXPECTED_TASKS )); then
        echo "Expected exactly $EXPECTED_TASKS component-ablation tasks, got '$count'." >&2
        exit 2
    fi
    printf '%s\n' "$count"
}

assert_task_count() {
    validated_task_count >/dev/null
}

run_preflight() {
    python "$SCRIPT" preflight "${common_args[@]}"
    assert_task_count
}

submit_jobs() {
    prepare_env
    run_preflight
    mkdir -p "$OUTPUT_ROOT/logs"

    local n_tasks max_array job_id
    n_tasks="$(validated_task_count)"
    max_array=$((n_tasks - 1))
    job_id=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name="hpo_component_2x2_s2" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem-per-cpu="$MEM_PER_CPU" \
        --time="$SLURM_TIME" \
        --array="0-${max_array}%${CONCURRENT}" \
        --output="$OUTPUT_ROOT/logs/hpo_component_2x2_%A_%a.out" \
        --error="$OUTPUT_ROOT/logs/hpo_component_2x2_%A_%a.err" \
        --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},REFERENCE_ROOT=${REFERENCE_ROOT},PYTHON_MODULE=${PYTHON_MODULE}" \
        "$0" worker)

    echo "Submitted Scenario-2 HPO component ablation as job $job_id."
    echo "Tasks: $n_tasks (2 new variants x 30 instances), concurrency: $CONCURRENT"
    echo "After all tasks finish, run: bash $0 summarize"
}

preflight() {
    prepare_env
    run_preflight
    echo "Preflight passed: exactly $EXPECTED_TASKS tasks."
}

worker() {
    cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required in worker mode}"
    prepare_env
    : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required in worker mode}"

    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    echo "OUTPUT_ROOT=$OUTPUT_ROOT"
    echo "REFERENCE_ROOT=$REFERENCE_ROOT"
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-$CPUS_PER_TASK}" \
        python "$SCRIPT" run-task \
        "${common_args[@]}" \
        --task-index "$SLURM_ARRAY_TASK_ID" \
        --resume
}

summarize() {
    prepare_env
    python "$SCRIPT" summarize "${common_args[@]}"
}

tasks() {
    prepare_env
    assert_task_count
    python "$SCRIPT" print-tasks "${common_args[@]}"
}

case "$MODE" in
    help|-h|--help) usage ;;
    submit) submit_jobs ;;
    preflight) preflight ;;
    summarize) summarize ;;
    tasks) tasks ;;
    worker) worker ;;
    *) usage; exit 2 ;;
esac
