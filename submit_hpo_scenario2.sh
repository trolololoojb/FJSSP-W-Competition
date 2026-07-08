#!/bin/bash -l
set -euo pipefail

MODE="${1:-help}"
PHASE="${2:-screening}"

# LUH cluster defaults, close to the existing submit_scenario2_instances.sh.
PARTITION="${PARTITION:-mpp.share,amo,taurus,lena,haku,smp}"
CPUS_PER_TASK="${CPUS_PER_TASK:-26}"
MEM_PER_CPU="${MEM_PER_CPU:-2500M}"
SLURM_TIME="${SLURM_TIME:-42:00:00}"
CONCURRENT="${CONCURRENT:-8}"
EA_TIME_LIMIT_S="${EA_TIME_LIMIT_S:-7200}"

# Per configuration job: run several instance/run tasks in parallel.
RUN_WORKERS="${RUN_WORKERS:-10}"
SIMULATION_WORKERS="${SIMULATION_WORKERS:-2}"
SURROGATE_N_JOBS="${SURROGATE_N_JOBS:-2}"

# Keep HPO logs and results inside the repository by default.
if [[ -z "${HPO_ROOT:-}" ]]; then
    HPO_ROOT="$PWD/results/hpo_scenario2"
fi

PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.6}"
SCRIPT="scripts/hpo_scenario2.py"

usage() {
    cat <<EOF
Usage:
  bash submit_hpo_scenario2.sh prepare <phase>
  bash submit_hpo_scenario2.sh submit <phase>
  bash submit_hpo_scenario2.sh summarize <phase>
  bash submit_hpo_scenario2.sh next <phase>
  bash submit_hpo_scenario2.sh worker <phase>      # internal Slurm array mode

Phases:
  screening -> tpe -> race1 -> race2 -> final

Typical sequence:
  bash submit_hpo_scenario2.sh submit screening
  bash submit_hpo_scenario2.sh summarize screening
  bash submit_hpo_scenario2.sh submit tpe
  bash submit_hpo_scenario2.sh summarize tpe
  bash submit_hpo_scenario2.sh submit race1
  bash submit_hpo_scenario2.sh summarize race1
  bash submit_hpo_scenario2.sh submit race2
  bash submit_hpo_scenario2.sh summarize race2
  bash submit_hpo_scenario2.sh submit final
  bash submit_hpo_scenario2.sh summarize final

Useful environment variables:
  HPO_ROOT=$HPO_ROOT
  PARTITION=$PARTITION
  CONCURRENT=$CONCURRENT
  CPUS_PER_TASK=$CPUS_PER_TASK
  EA_TIME_LIMIT_S=$EA_TIME_LIMIT_S
  RUN_WORKERS=$RUN_WORKERS
  SIMULATION_WORKERS=$SIMULATION_WORKERS
  SURROGATE_N_JOBS=$SURROGATE_N_JOBS
EOF
}

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

make_plan() {
    prepare_env
    mkdir -p "$HPO_ROOT/$PHASE" "$HPO_ROOT/logs"
    python "$SCRIPT" make-plan --phase "$PHASE" --hpo-root "$HPO_ROOT"
}

submit_phase() {
    make_plan
    CONFIGS="$HPO_ROOT/$PHASE/configs.jsonl"
    N_CONFIGS=$(wc -l < "$CONFIGS")
    if [[ "$N_CONFIGS" -le 0 ]]; then
        echo "No configs in $CONFIGS" >&2
        exit 2
    fi
    MAX_ARRAY=$((N_CONFIGS - 1))
    mkdir -p "$HPO_ROOT/logs"

    JOBID=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name="hpo_${PHASE}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem-per-cpu="$MEM_PER_CPU" \
        --time="$SLURM_TIME" \
        --array="0-${MAX_ARRAY}%${CONCURRENT}" \
        --output="$HPO_ROOT/logs/hpo_${PHASE}_%A_%a.out" \
        --error="$HPO_ROOT/logs/hpo_${PHASE}_%A_%a.err" \
        --export="ALL,PHASE=${PHASE},HPO_ROOT=${HPO_ROOT},RUN_WORKERS=${RUN_WORKERS},SIMULATION_WORKERS=${SIMULATION_WORKERS},SURROGATE_N_JOBS=${SURROGATE_N_JOBS},PYTHON_MODULE=${PYTHON_MODULE},EA_TIME_LIMIT_S=${EA_TIME_LIMIT_S}" \
        "$0" worker "$PHASE")

    echo "Submitted HPO phase '$PHASE' as job array: $JOBID"
    echo "Configs: $N_CONFIGS, concurrency: $CONCURRENT"
    echo "Status: squeue -u \$USER"
}

worker() {
    cd "$SLURM_SUBMIT_DIR"
    prepare_env
    echo "PHASE=$PHASE"
    echo "HPO_ROOT=$HPO_ROOT"
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    echo "RUN_WORKERS=$RUN_WORKERS"
    echo "SIMULATION_WORKERS=$SIMULATION_WORKERS"
    echo "SURROGATE_N_JOBS=$SURROGATE_N_JOBS"
    echo "EA_TIME_LIMIT_S=$EA_TIME_LIMIT_S"

    EXTRA_ARGS=()
    if [[ "$EA_TIME_LIMIT_S" != "none" ]]; then
        EXTRA_ARGS+=(--time-limit-s "$EA_TIME_LIMIT_S")
    fi

    python "$SCRIPT" run-task \
        --phase "$PHASE" \
        --hpo-root "$HPO_ROOT" \
        --task-index "$SLURM_ARRAY_TASK_ID" \
        --workers "$RUN_WORKERS" \
        --simulation-workers "$SIMULATION_WORKERS" \
        --surrogate-n-jobs "$SURROGATE_N_JOBS" \
        "${EXTRA_ARGS[@]}" \
        --resume \
        --allow-failed-runs
}

summarize() {
    prepare_env
    python "$SCRIPT" summarize --phase "$PHASE" --hpo-root "$HPO_ROOT"
}

next_phase() {
    case "$PHASE" in
        tpe) summarize_phase="screening" ;;
        race1) summarize_phase="tpe" ;;
        race2) summarize_phase="race1" ;;
        final) summarize_phase="race2" ;;
        *) echo "next only makes sense for tpe, race1, race2, final" >&2; exit 2 ;;
    esac
    prepare_env
    python "$SCRIPT" summarize --phase "$summarize_phase" --hpo-root "$HPO_ROOT"
    python "$SCRIPT" make-plan --phase "$PHASE" --hpo-root "$HPO_ROOT"
}

case "$MODE" in
    help|-h|--help) usage ;;
    prepare) make_plan ;;
    submit) submit_phase ;;
    worker) worker ;;
    summarize) summarize ;;
    next) next_phase ;;
    *) usage; exit 2 ;;
esac
