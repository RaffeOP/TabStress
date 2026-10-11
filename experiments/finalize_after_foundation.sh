#!/usr/bin/env bash
# Waits for the TabICL run to finish, then rebuilds and verifies the manuscript (full verification)
# and, only if verification passes, repackages the Overleaf bundle. One status line is appended to
# results/metrics/revision/finalize.log; the full output goes to finalize_full.log.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/metrics/revision/finalize.log
FULL=results/metrics/revision/finalize_full.log

while pgrep -f run_foundation.py > /dev/null || pgrep -f run_final_pipeline.sh > /dev/null; do sleep 60; done
echo "$(date '+%F %T') foundation run ended; $(grep -c . results/metrics/revision/done_foundation.txt) foundation tasks done" >> "$LOG"

if bash experiments/run_final_pipeline.sh > "$FULL" 2>&1; then
  bash experiments/build_overleaf_bundle.sh >> "$FULL" 2>&1
  echo "$(date '+%F %T') FINALIZE PASS: $(grep -E 'checks|passed' "$FULL" | tail -1); $(grep Pages "$FULL" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
else
  echo "$(date '+%F %T') FINALIZE FAIL (bundle not rebuilt): $(grep -iE 'FAIL|error' "$FULL" | head -5 | tr '\n' ' ')" >> "$LOG"
fi
