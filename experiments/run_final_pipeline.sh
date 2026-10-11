#!/usr/bin/env bash
# Final analysis and build of the TabStress manuscript, in dependency order. Stops at the first
# failing step. Run from the repository root after every experiment block is complete:
#   bash experiments/run_final_pipeline.sh            (full verification, including the harness re-run)
#   bash experiments/run_final_pipeline.sh --quick    (skips the harness re-run in the verifier)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
VERIFY_ARGS=()
if [[ "${1:-}" == "--quick" ]]; then VERIFY_ARGS=(--skip-harness); fi

step() { echo; echo "=== $* ==="; }

step "1/10 completeness of the result files"
$PY - <<'EOF'
import os, pandas as pd
EXT = "results/metrics/extended"
expected = {"base": 9840, "cc_binary": 2520, "native_nan": 3000, "catboost_default": 820, "xgb_default": 820}
ext = pd.concat([pd.read_csv(os.path.join(EXT, f)) for f in os.listdir(EXT) if f.endswith(".csv")])
key = ["variant", "seed", "dataset", "model", "modality", "eta"]
got = ext.groupby("variant").size().to_dict()
bad = {v: (got.get(v, 0), n) for v, n in expected.items() if got.get(v, 0) != n}
dups = int(ext.duplicated(key).sum())
grid = pd.read_csv("results/metrics/benchmark_results.csv")
patched = os.path.exists("results/metrics/original_patch_log.csv")
if bad or dups or len(grid) != 5280 or not patched:
    raise SystemExit(f"incomplete: {bad}, duplicates {dups}, grid rows {len(grid)}, original patch applied {patched}")
print("all blocks complete:", got, "| original grid 5,280 rows, patch applied")
EOF

step "2/10 primary grid";            $PY experiments/build_primary.py
step "3/10 operator audit";          $PY experiments/operator_audit.py > /dev/null
step "4/10 forensic audit (primary)"; $PY experiments/forensic_audit.py --primary > /dev/null
step "5/10 forensic audit (original)"; $PY experiments/forensic_audit.py > /dev/null
step "6/10 extended analyses";       $PY experiments/analyze_primary_extras.py > /dev/null
step "6b/10 corrections, data quality, velocity"; PYTHONPATH=. $PY experiments/analyze_fix_extras.py > /dev/null
step "6c/10 revision analyses (protocols, mechanisms, tuning, artifacts, knee validation)"
$PY experiments/knee_validation.py > /dev/null 2>&1
PYTHONPATH=. $PY experiments/analyze_revision.py > /dev/null
# TabICL (Protocol B, seven-model comparison): reads the complete seeds only, no torch needed
PYTHONPATH=. $PY experiments/analyze_foundation.py > /dev/null
# sensitivity runs (TabICL settings, compound corruption) and the Bayesian / bootstrap statistics
PYTHONPATH=. $PY experiments/analyze_sensitivity.py > /dev/null
step "7/10 figures";                 $PY experiments/build_primary_figures.py > /dev/null 2>&1
$PY experiments/build_revision_figures.py > /dev/null 2>&1
step "8/10 tables and atlas";        $PY experiments/build_manuscript_tables.py --primary > /dev/null
PYTHONPATH=. $PY experiments/build_revision_tables.py
step "9/10 assemble and compile"
$PY experiments/assemble_manuscript.py
if [[ -s springer_project/build/dropped_blocks.txt ]]; then
  echo "conditional blocks were dropped; the data are not complete"; exit 1
fi
(cd springer_project && tectonic -X compile TabStress.tex --keep-logs > build/tectonic.out 2>&1) || {
  tail -30 springer_project/build/tectonic.out; exit 1; }
(cd springer_project && tectonic -X compile TabStress_Supplementary.tex > build/tectonic_supp.out 2>&1) || {
  tail -30 springer_project/build/tectonic_supp.out; exit 1; }
pdfinfo springer_project/TabStress.pdf | grep Pages
pdfinfo springer_project/TabStress_Supplementary.pdf | grep Pages
step "10/10 verification";           $PY experiments/verify_manuscript.py "${VERIFY_ARGS[@]+"${VERIFY_ARGS[@]}"}"
