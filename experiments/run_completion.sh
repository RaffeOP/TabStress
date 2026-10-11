#!/usr/bin/env bash
# Completion runs for the revision blocks that were first run on a subset of datasets. Resumable: each call
# skips tasks already listed in results/metrics/revision/done_tasks.txt. Kills its worker pool on exit.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
trap 'pkill -P $$; exit 1' INT TERM
PY=.venv/bin/python
$PY experiments/run_revision.py --workers 8 --blocks deploy_cat train_cat
$PY experiments/run_revision.py --workers 8 --blocks train_mech --seeds-train 42 --include-excluded
$PY experiments/run_revision.py --workers 8 --blocks tuned --include-excluded
$PY experiments/run_revision.py --workers 8 --blocks train_mech --seeds-train 42 43 44 --include-excluded
echo "COMPLETION DONE"
