#!/bin/bash
# Scratch job: run a python script of /scratch/pgonzale/p1x with the project's
# env plus the benchmark libraries. Array task id -> SLURM_ARRAY_TASK_ID.
#SBATCH --partition=academic
#SBATCH --qos=academic
#SBATCH --account=master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/pgonzale/p1x/logs/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
cd /scratch/pgonzale/p1x
echo "$(date '+%F %T') $(hostname) task ${SLURM_ARRAY_TASK_ID:-0}: $*"
uv run --no-sync --project /home/pgonzale/ML_p1 --with lightgbm==4.7.0 \
    --with xgboost==3.4.1 --with catboost==1.2.10 python "$@"
echo "$(date '+%F %T') done"
