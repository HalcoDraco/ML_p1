#!/bin/bash
# Submit from the repository root; runs a NumPy experiment in the project env.
# Example: sbatch --array=0-14 -c 4 --mem=24G experiments/job.sbatch experiments/mlp_run.py --name m5
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=outputs/slurm/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
repo=${SLURM_SUBMIT_DIR:-$PWD}
cd "$repo"
echo "$(date '+%F %T') $(hostname) task ${SLURM_ARRAY_TASK_ID:-0}: $*"
uv run --no-sync --project "$repo" python "$@"
echo "$(date '+%F %T') done"
