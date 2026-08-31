#!/bin/bash
#SBATCH --job-name=scen2slow

#SBATCH --time=47:59:00
#SBATCH --cpus-per-task=90
#SBATCH --mem-per-cpu=3900M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=2,3,4,8,9,16,19,21,22,23,24,25%8
#SBATCH --output=logs_parallel/%x_%A_%a.out
#SBATCH --error=logs_parallel/%x_%A_%a.err

set -euo pipefail

cd /bigwork/nhwijoha/FJSSP-W-Competition
source .venv/bin/activate

mkdir -p logs_timelimit

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

TASK_ID="${SLURM_ARRAY_TASK_ID}"
TASK_PADDED="$(printf "%02d" "$TASK_ID")"

CHUNK_FILE="chunks/chunk_${TASK_PADDED}.txt"
OUT_DIR="results/scenario2_parallel_chunk_${TASK_PADDED}"

mapfile -t INSTANCES < "$CHUNK_FILE"

echo "Task: $TASK_ID"
echo "Instances: ${INSTANCES[*]}"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Output: $OUT_DIR"

srun --cpu-bind=cores python scripts/run_scenario2_submission.py \
  --resume \
  --instances "${INSTANCES[@]}" \
  --output-dir "$OUT_DIR" \
  --workers 10 \
  --simulation-workers 5 \
  --surrogate-n-jobs 4 \
  --max-function-evaluations 5000000