#!/bin/bash
#SBATCH --job-name=scenario2p
#SBATCH --partition=amo,taurus,lena,haku,mpp.share
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=3900M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs_parallel/%x_%A_%a.out
#SBATCH --error=logs_parallel/%x_%A_%a.err

set -euo pipefail

cd /bigwork/nhwijoha/FJSSP-W-Competition
source .venv/bin/activate

mkdir -p logs_parallel

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

TASK_ID="${SLURM_ARRAY_TASK_ID}"
TASK_PADDED="$(printf "%02d" "$TASK_ID")"

CHUNK_FILE="chunks/chunk_${TASK_PADDED}.txt"
OUT_DIR="results/scenario2_parallel_chunk_${TASK_PADDED}"

if [[ ! -f "$CHUNK_FILE" ]]; then
  echo "Missing chunk file: $CHUNK_FILE"
  exit 1
fi

mapfile -t INSTANCES < "$CHUNK_FILE"

case "$TASK_ID" in
  2|3|4|8|9|16|19|21|22|23|24|25)
    RUN_WORKERS=7
    SIM_WORKERS=5
    SURROGATE_JOBS=2
    ;;
  *)
    RUN_WORKERS=10
    SIM_WORKERS=3
    SURROGATE_JOBS=2
    ;;
esac

echo "Task: $TASK_ID"
echo "Instances: ${INSTANCES[*]}"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Run workers: $RUN_WORKERS"
echo "Simulation workers per run: $SIM_WORKERS"
echo "Surrogate jobs per run: $SURROGATE_JOBS"
echo "Output: $OUT_DIR"

srun --cpu-bind=cores python scripts/run_scenario2_submission.py \
  --resume \
  --instances "${INSTANCES[@]}" \
  --output-dir "$OUT_DIR" \
  --workers "$RUN_WORKERS" \
  --simulation-workers "$SIM_WORKERS" \
  --surrogate-n-jobs "$SURROGATE_JOBS" \
  --time-limit-s 129600 \
  --max-function-evaluations 5000000
