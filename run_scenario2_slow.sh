#!/bin/bash -l
#SBATCH --job-name=scenario2
#SBATCH --partition=mpp.share,amo,taurus,lena,haku
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=47:59:00
#SBATCH --array=2,3,4,8,9,16,19,21,22,23,24,25%10
#SBATCH --output=logs/scenario2_%A_%a.out
#SBATCH --error=logs/scenario2_%A_%a.err

cd "$SLURM_SUBMIT_DIR"

module load Python/3.11.6
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CHUNK_FILE=$(printf "chunks/chunk_%02d.txt" "$SLURM_ARRAY_TASK_ID")
mapfile -t INSTANCES < "$CHUNK_FILE"

OUT_DIR=$(printf "results/scenario2_submission_chunk_%02d" "$SLURM_ARRAY_TASK_ID")

echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Instance: ${INSTANCES[*]}"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Output: $OUT_DIR"

srun python scripts/run_scenario2_submission.py \
  --resume \
  --instances "${INSTANCES[@]}" \
  --output-dir "$OUT_DIR" \
  --workers 10 \
  --surrogate-n-jobs 4