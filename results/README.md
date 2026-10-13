# results/ — status of every directory

Read this before using any file under `results/`. The status column is the authority: a file is confirmatory evidence only if its directory is marked **CONFIRMATORY**. Table and figure provenance (script, input, configuration) is in `PIPELINE_MANIFEST.md`, Section 5.

| Directory | Status | Contents | Used by |
|---|---|---|---|
| `metrics/extended/` | CONFIRMATORY (primary grid, matched budget) | Seed-level cell CSVs for the primary grid; `catboost_default__seed42.csv` and `xgb_default__seed42.csv` are library-default runs (sensitivity only) | `build_primary.py`, `build_manuscript_tables.py` |
| `metrics/revision/` | CONFIRMATORY (revision blocks) | `deploy*`, `train*`, `compound`, `foundation*`, `pol_ordering*`, `tuned__seed42.csv` (tuned configurations, seed 42 only); resumable state (`done_*.txt`, `foundation_failures.txt`) | `analyze_revision.py`, `analyze_foundation.py`, `build_revision_tables.py` |
| `metrics/artifact/` | SENSITIVITY (corrupted or cleaned data copies) | `base`, `cc_binary`, `native_nan` files (seeds 42–44); rows are keyed by `dataset` and cover the three cleaned copies (`*_clean`) and the five deduplicated copies (`*_dedup`) | `analyze_sensitivity.py`, `build_revision_tables.py` |
| `metrics/ablation/` | SENSITIVITY (MLP early-stopping ablation, seed 42) | `mlp_no_es__seed42.csv` | `run_mlp_ablation.py`; quoted in Section 5.7 |
| `analysis/primary/` | DERIVED (from `metrics/extended/`) | `cells.csv`, `cells_by_seed.csv`, `cells_gaussian_allcol_by_seed.csv` | tables and figures of the primary grid |
| `analysis/revision/` | DERIVED (from `metrics/revision/` and `metrics/artifact/`) | Revision tables (`tab_*.tex`), trajectories, `revision.json`, `sensitivity_extra.json`, `foundation.json`, `knee_validation.json`, `rerun_audit.json` | manuscript inputs |
| `analysis/audit_primary/` | DERIVED (audit numbers) | `audit_numbers.json`, `extras.json`, `operator_audit.json`, `fix_extras.json`, `friedman.csv`, `wilcoxon_holm.csv` | manuscript text, `verify_manuscript.py` |
| `analysis/latex_primary/` | DERIVED (LaTeX inputs) | `tab_*.tex`, `atlas_primary.tex` | `TabStress_Supplementary.tex` |
| `analysis/audit/`, `analysis/latex/` | FIRST SUBMISSION (superseded) | Outputs of the first submission; read only by `forensic_audit.py` and `verify_manuscript.py` as the earlier reference point. Not inputs to any current table | — |
| `metrics/backup_before_original_patch/`, `metrics/extended_backup_gaussfix/`, `metrics/extended_backup_tiefix/` | SUPERSEDED (before correction) | Pre-correction cell files. Read only as the "before" side of the correction audit (`analyze_fix_extras.py`, `recompute_original_cells.py`). Never a table input | correction audit only |
| `superseded/` | SUPERSEDED | Logs of superseded runs (`superseded/logs/`), including every stale progress, failure or finalization log that used to sit in `metrics/revision/`. Prefix `revision_` marks logs that came from `metrics/revision/` | none |
| `metrics/raw/` | INPUT (raw OpenML downloads) | 21 CSV files (including `volkert_backend.csv`); read by `run_benchmark.py`, `run_volkert_reverse.py` and the first-submission recomputation | first-submission audit |
| `metrics/benchmark_results.csv`, `metrics/original_patch_log.csv`, `metrics/original_recompute.log`, `metrics/artifact_run.log` | FIRST SUBMISSION / CORRECTION RECORD | Earlier benchmark output and the record of the correction audit (`recompute_original_cells.py`); indexed in PIPELINE_MANIFEST.md, Section 5.3 | correction audit only |

Rules:
1. A table or figure must cite only files in CONFIRMATORY, DERIVED or SENSITIVITY directories, as listed in PIPELINE_MANIFEST.md Section 5.
2. SUPERSEDED and FIRST SUBMISSION files may not be added to any manuscript input list.
