#!/bin/bash
# One `p1` command as a slurm job, or one task of a job array, on SCITAS jed.
# Submit it with slurm/submit.sh, which freezes a copy of the code first.
#
# In the p1 arguments, @TASK@ becomes the array task id, @FOLD@ the task id
# modulo $FOLDS (default 5) and @VARIANT@ the task id divided by $FOLDS, so
# one array covers every (grid variant, fold) pair.
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out

set -euo pipefail
repo=$1  # repository whose .venv runs the frozen code
shift
# np.bincount is single-threaded; BLAS (logistic regression) gets the job's CPUs.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
task=${SLURM_ARRAY_TASK_ID:-0}
folds=${FOLDS:-5}
args=()
for arg in "$@"; do
    arg=${arg//@TASK@/$task}
    arg=${arg//@FOLD@/$((task % folds))}
    arg=${arg//@VARIANT@/$((task / folds))}
    args+=("$arg")
done
echo "$(date '+%F %T') $(hostname) code $(pwd): p1 ${args[*]}"
# -m with the frozen copy as working directory: its p1/ comes first on sys.path.
uv run --no-sync --project "$repo" python -m p1.cli "${args[@]}"
echo "$(date '+%F %T') done"
