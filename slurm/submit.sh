#!/bin/bash
# Submit slurm/p1.sbatch on a frozen copy of the code.
#
# The code (p1/, configs/, *.py) is copied to outputs/code/<time>/, where
# data/ and outputs/ link back to the repository, so editing the code later
# does not change the jobs that are still queued.
#
# Usage: slurm/submit.sh [sbatch options] -- [p1 arguments]
#   slurm/submit.sh -J gbdt --array=0-4 -- cv configs/gbdt.toml --fold @TASK@
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
options=()
while [[ $# -gt 0 && $1 != "--" ]]; do
    options+=("$1")
    shift
done
shift  # the "--"
code="$repo/outputs/code/$(date +%m%d-%H%M%S)-$$"
mkdir -p "$code" "$repo/outputs/slurm"
cp -rp "$repo/p1" "$repo/configs" "$repo"/*.py "$code/"
ln -s "$repo/data" "$code/data"
ln -s "$repo/outputs" "$code/outputs"
sbatch --parsable --chdir="$code" "${options[@]}" "$repo/slurm/p1.sbatch" "$repo" "$@"
