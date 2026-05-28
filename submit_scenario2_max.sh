#!/bin/bash -l
set -euo pipefail

MODE="${1:-submit}"

PARTITION="${PARTITION:-mpp.share,amo,taurus,lena,haku,smp}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
CONCURRENT="${CONCURRENT:-50}"
MEM_PER_CPU="${MEM_PER_CPU:-2500M}"
SLURM_TIME="${SLURM_TIME:-38:00:00}"

TIME_LIMIT_S="${TIME_LIMIT_S:-129600}"   # 36h GA-Zeitlimit
INTERNAL_SIMULATIONS="${INTERNAL_SIMULATIONS:-10}"
FINAL_SIMULATIONS="${FINAL_SIMULATIONS:-50}"
MAX_FUNCTION_EVALUATIONS="${MAX_FUNCTION_EVALUATIONS:-5000000}"

export TIME_LIMIT_S INTERNAL_SIMULATIONS FINAL_SIMULATIONS MAX_FUNCTION_EVALUATIONS CPUS_PER_TASK

TASK_FILE="chunks/scenario2_tasks.tsv"
RUNNER="scripts/run_scenario2_single_run.py"

create_runner() {
mkdir -p scripts

cat > "$RUNNER" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_scenario2_submission import (
    load_uncertainty,
    uncertainty_for,
    solve_run,
    load_completed_ok,
    append_jsonl,
    write_csv_outputs,
)
from util.benchmark_parser import WorkerBenchmarkParser


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--instances-dir", type=Path, default=Path("instances/fjssp-w"))
    parser.add_argument("--uncertainty-json", type=Path, default=Path("config/scenario2_uncertainty.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--internal-simulations", type=int, default=10)
    parser.add_argument("--final-simulations", type=int, default=50)
    parser.add_argument("--simulation-workers", type=int, default=1)
    parser.add_argument("--time-limit-s", type=int, default=129600)
    parser.add_argument("--max-function-evaluations", type=int, default=5_000_000)
    parser.add_argument("--surrogate-n-jobs", type=int, default=1)
    parser.add_argument("--disable-local-search", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_results.jsonl"

    completed = load_completed_ok(raw_path) if args.resume else {}
    key = (args.instance, args.run)

    out_args = SimpleNamespace(
        n_runs=1,
        workers=1,
        time_limit_s=args.time_limit_s,
        max_function_evaluations=args.max_function_evaluations,
        internal_simulations=args.internal_simulations,
        final_simulations=args.final_simulations,
        surrogate_n_jobs=args.surrogate_n_jobs,
        simulation_workers=args.simulation_workers,
        disable_local_search=args.disable_local_search,
    )

    if key in completed:
        row = completed[key]
        write_csv_outputs(args.output_dir, [row], out_args, args.uncertainty_json, expected_instances=1)
        print(f"SKIP existing: {args.instance} run {args.run}", flush=True)
        return 0

    instance_path = args.instances_dir / args.instance
    if not instance_path.exists():
        raise FileNotFoundError(instance_path)

    uncertainty_payload = load_uncertainty(args.uncertainty_json)
    seed, uncertainty_parameters = uncertainty_for(uncertainty_payload, args.instance, args.run)

    print(f"START {args.instance} run {args.run} seed={seed}", flush=True)
    print(
        f"time_limit_s={args.time_limit_s}, "
        f"simulation_workers={args.simulation_workers}, "
        f"surrogate_n_jobs={args.surrogate_n_jobs}",
        flush=True,
    )

    encoding = WorkerBenchmarkParser().parse_benchmark(str(instance_path))

    solve_args = SimpleNamespace(
        internal_simulations=args.internal_simulations,
        final_simulations=args.final_simulations,
        time_limit_s=args.time_limit_s,
        max_function_evaluations=args.max_function_evaluations,
        surrogate_n_jobs=args.surrogate_n_jobs,
        disable_local_search=args.disable_local_search,
        simulation_workers=args.simulation_workers,
    )

    row = solve_run(
        args.instance,
        args.run,
        seed,
        encoding,
        uncertainty_parameters,
        solve_args,
    )

    append_jsonl(raw_path, row)
    write_csv_outputs(args.output_dir, [row], out_args, args.uncertainty_json, expected_instances=1)

    print(
        f"DONE {args.instance} run {args.run} "
        f"fitness={row['fitness']} runtime_s={row['runtime_s']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

chmod +x "$RUNNER"
}

create_tasks() {
mkdir -p chunks logs results/scenario2_runs

python - <<'PY'
from pathlib import Path

runs = range(1, 11)

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

instances_dir = Path("instances/fjssp-w")
all_instances = sorted(p.name for p in instances_dir.glob("*.fjs"))
all_set = set(all_instances)

missing_slow = [name for name in slow if name not in all_set]
if missing_slow:
    print("WARNING: slow instance(s) not found:")
    for name in missing_slow:
        print("  ", name)

ordered = [name for name in slow if name in all_set]
ordered += [name for name in all_instances if name not in set(ordered)]

tasks = []
for instance in ordered:
    for run in runs:
        tasks.append((instance, run))

out = Path("chunks/scenario2_tasks.tsv")
with out.open("w", encoding="utf-8") as f:
    for instance, run in tasks:
        f.write(f"{instance}\t{run}\n")

print(f"Wrote {len(tasks)} tasks to {out}")
print(f"Slow tasks first: {len([t for t in tasks if t[0] in slow])}")
PY
}

submit_jobs() {
create_runner
create_tasks

N_TASKS=$(wc -l < "$TASK_FILE")
MAX_ARRAY=$((N_TASKS - 1))

SBATCH_ARGS=(
    --job-name=sc2max
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    --mem-per-cpu="$MEM_PER_CPU"
    --time="$SLURM_TIME"
    --array="0-${MAX_ARRAY}%${CONCURRENT}"
    --output="logs/sc2max_%A_%a.out"
    --error="logs/sc2max_%A_%a.err"
    --export="ALL,TASK_FILE=${TASK_FILE},TIME_LIMIT_S=${TIME_LIMIT_S},INTERNAL_SIMULATIONS=${INTERNAL_SIMULATIONS},FINAL_SIMULATIONS=${FINAL_SIMULATIONS},MAX_FUNCTION_EVALUATIONS=${MAX_FUNCTION_EVALUATIONS}"
)

if [[ -n "$PARTITION" ]]; then
    SBATCH_ARGS=(--partition="$PARTITION" "${SBATCH_ARGS[@]}")
fi

JOBID=$(sbatch --parsable "${SBATCH_ARGS[@]}" "$0" worker)

echo "Submitted job array: $JOBID"
echo "Tasks: $N_TASKS"
echo "Max running tasks: $CONCURRENT"
echo "CPUs per task: $CPUS_PER_TASK"
echo "Max CPUs used: $((CONCURRENT * CPUS_PER_TASK))"
echo
echo "Status:"
echo "  squeue -u \$USER"
echo
echo "Merge after all jobs are done:"
echo "  bash $0 merge"
}

worker() {
cd "$SLURM_SUBMIT_DIR"

module load Python/3.11.6 2>/dev/null || true
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TASK_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
IFS=$'\t' read -r INSTANCE RUN <<< "$TASK_LINE"

SAFE_INSTANCE="${INSTANCE%.fjs}"
OUT_DIR=$(printf "results/scenario2_runs/%s/run_%02d" "$SAFE_INSTANCE" "$RUN")

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "INSTANCE=$INSTANCE"
echo "RUN=$RUN"
echo "OUT_DIR=$OUT_DIR"
echo "CPUS=$SLURM_CPUS_PER_TASK"
echo "TIME_LIMIT_S=$TIME_LIMIT_S"

python "$RUNNER" \
    --resume \
    --instance "$INSTANCE" \
    --run "$RUN" \
    --output-dir "$OUT_DIR" \
    --time-limit-s "$TIME_LIMIT_S" \
    --internal-simulations "$INTERNAL_SIMULATIONS" \
    --final-simulations "$FINAL_SIMULATIONS" \
    --max-function-evaluations "$MAX_FUNCTION_EVALUATIONS" \
    --simulation-workers "$SLURM_CPUS_PER_TASK" \
    --surrogate-n-jobs "$SLURM_CPUS_PER_TASK"
}

merge_results() {
python - <<'PY'
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys

from scripts.run_scenario2_submission import write_csv_outputs

task_file = Path("chunks/scenario2_tasks.tsv")
out_dir = Path("results/scenario2_submission")
uncertainty_json = Path("config/scenario2_uncertainty.json")

expected = []
with task_file.open("r", encoding="utf-8") as f:
    for line in f:
        instance, run = line.rstrip("\n").split("\t")
        expected.append((instance, int(run)))

rows_by_key = {}

for raw_path in sorted(Path("results/scenario2_runs").glob("*/run_*/raw_results.jsonl")):
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            key = (row["instance"], int(row["run"]))
            rows_by_key[key] = row

missing = [key for key in expected if key not in rows_by_key]
rows = [rows_by_key[key] for key in expected if key in rows_by_key]

args = SimpleNamespace(
    n_runs=10,
    workers=1,
    time_limit_s=int(os.environ.get("TIME_LIMIT_S", "129600")),
    max_function_evaluations=int(os.environ.get("MAX_FUNCTION_EVALUATIONS", "5000000")),
    internal_simulations=int(os.environ.get("INTERNAL_SIMULATIONS", "10")),
    final_simulations=int(os.environ.get("FINAL_SIMULATIONS", "50")),
    surrogate_n_jobs=int(os.environ.get("CPUS_PER_TASK", "16")),
    simulation_workers=int(os.environ.get("CPUS_PER_TASK", "16")),
    disable_local_search=False,
)

write_csv_outputs(
    out_dir,
    rows,
    args,
    uncertainty_json,
    expected_instances=30,
)

print(f"Merged OK rows: {len(rows)}")
print(f"Output: {out_dir}")

if missing:
    print("\nMissing runs:")
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
    submit)
        submit_jobs
        ;;
    worker)
        worker
        ;;
    merge)
        merge_results
        ;;
    *)
        echo "Usage: bash $0 submit|worker|merge"
        exit 2
        ;;
esac