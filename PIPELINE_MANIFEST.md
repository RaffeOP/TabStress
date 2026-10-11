# TabStress Pipeline Manifest & Changelog

**Manuscript:** TabStress: Nonparametric Robustness Profiling of Tabular Classifiers under Controlled Data Corruption  
**Authors:** Shashank Mouli Satapathy & Devarsh Patel (Vellore Institute of Technology)  
**Pipeline Version:** 2.1 (second-revision package), built on repository commit `749cc1c` (branch `feature/reviewer-revision`); the second-revision edits listed in Section 3 sit on top of that commit and are identified by the final commit hash recorded in the release tag.
**Final pipeline entry point:** `experiments/run_final_pipeline.sh` (everything it reads and writes is listed in Sections 1 and 4).

---

## 1. Pipeline Architecture & Execution Flow

The TabStress replication pipeline executes end-to-end via:
```bash
bash experiments/run_final_pipeline.sh --quick
```
Or for full reproduction including test-cell re-evaluations:
```bash
bash experiments/run_final_pipeline.sh
```

### Stage Summary & File Mappings

| Stage | Script / Module | Input | Output / Artifacts | Description |
|---|---|---|---|---|
| **1. Completeness Audit** | Inline bash check | `results/metrics/extended/` | Stored cell count validation | Verifies 22,280 cells and 111,400 fits across all blocks |
| **2. Primary Grid** | `experiments/build_primary.py` | `results/metrics/extended/` | `results/analysis/primary/cells.csv` | Assembles seed-averaged cells and per-seed results |
| **3. Operator Audit** | `experiments/operator_audit.py` | Primary grid & seeds | `results/analysis/primary/operator_audit.json` | Audits coupling, target rates, and flat blocks |
| **4. Forensic Audits** | `experiments/forensic_audit.py` | Primary & original data | Audit JSON logs | Verifies fold independence, target leakage absence |
| **5. Extended Extras** | `experiments/analyze_primary_extras.py` | Primary data & baselines | `results/analysis/primary/extras.json` | Computes native handling, budget sensitivity, variance ratios |
| **6. Velocity & Quality** | `experiments/analyze_fix_extras.py` | Grid & repair metrics | `results/analysis/audit_primary/fix_extras.json` | Evaluates degenerate predictors (amazon_access) and repair |
| **7. Revision Analyses** | `experiments/analyze_revision.py` | Extended blocks | `results/analysis/revision/` | Confirmatory rank tests, Protocol A vs B, mechanism runs |
| **8. Foundation Benchmark** | `experiments/analyze_foundation.py` | `results/metrics/revision/foundation__*` | `results/analysis/revision/foundation.json` | 7-model Protocol B comparison with TabICL |
| **9. Sensitivity & Bayes** | `experiments/analyze_sensitivity.py` | Compound runs & TabICL defaults | `results/analysis/revision/sensitivity_extra.json` | Bayesian signed-rank tests (ROPE), compound additivity |
| **10. Knee Validation** | `experiments/knee_validation.py` | Synthetic & empirical curves | `results/analysis/revision/knee_validation.json` | Knee reliability on known hinge curves and noise sweeps |
| **11. Figure Generation** | `experiments/build_primary_figures.py`<br>`experiments/build_revision_figures.py` | Numerical results | `figures/*.png` | Generates 6 main figures and 30+ supplementary atlas figures |
| **12. Table Assembly** | `experiments/build_manuscript_tables.py`<br>`experiments/build_revision_tables.py` | Analysis JSONs & CSVs | `results/analysis/latex_primary/`<br>`results/analysis/revision/` | Typesets LaTeX tables (main-text and supplementary tables) |
| **13. Manuscript Assembly** | `experiments/assemble_manuscript.py` | `springer_project/build/TabStress_main.tex` | `springer_project/TabStress.tex`<br>`springer_project/TabStress_Supplementary.tex` | Inlines tables, macros, and dynamic values |
| **14. Tectonic Build** | `tectonic` engine | Assembled `.tex` files | `TabStress.pdf`<br>`TabStress_Supplementary.pdf` | Typesets main paper (<45 pp) and supplementary document |
| **15. Verification Suite** | `experiments/verify_manuscript.py` | Typeset texts, data files | 306 automated assertions | Strict mechanical quality gate (0 failures required) |

---

## 2. Dataset Preprocessing & Artifact Cleaning

The benchmark incorporates 20 curated tabular classification datasets from OpenML. Specific dataset artifacts and their resolution protocols are documented below:

1. **`spambase` (OpenML ID 44):**
   - *Artifact:* Contains 391 duplicate feature vectors with identical labels (rule R1) and 3 feature vectors that occur twice with contradictory labels (rule R2, 6 rows). R1 keeps the first occurrence; R2 removes every row of a conflicting vector, so 397 of 4,601 rows are removed and 4,204 remain.
   - *Resolution:* In sensitivity runs (`results/analysis/revision/sensitivity_extra.json`), duplicate rows (R1) and conflicting rows (R2) are dropped prior to stratified fold splitting. Omnibus test rankings remain stable ($\tau \ge 0.87$).
2. **`churn` (OpenML ID 40701):**
   - *Artifact:* Contains an administrative non-predictive customer phone identifier column.
   - *Resolution:* Dropped the identifier column; confirmed zero impact on relative degradation profiles.
3. **`MiniBooNE` (OpenML ID 41150):**
   - *Artifact:* Contained 468 sentinel rows recorded with `-999.0` sentinel missing values in signal features.
   - *Resolution:* Sentinel rows filtered; rank tests recomputed with zero change in model ordering.
4. **`amazon_access` (OpenML ID 4135):**
   - *Artifact:* Severe class imbalance ($\pi_{\max} = 0.94$) leads to degenerate majority-class output for unregularized MLP.
   - *Resolution:* Explicitly disclosed and isolated; confirmed that primary rank leaders are invariant to excluding this dataset.

Exact rules, per-dataset counts and the sentinel criterion (all features equal to -999) are in Supplementary Section S10 (`results/analysis/revision/tab_cleaning_counts.tex`, produced by `experiments/dataset_cleaning_counts.py`). The primary benchmark uses the uploads unchanged; cleaning applies only to the artifact-sensitivity copies.

### Terminology (identical to the article's methodology section)

- **Evaluation cell (cell):** one combination of dataset, model family, corruption operator, intensity tier eta and random seed, evaluated by stratified five-fold cross-validation.
- **Model fit:** one training event of one model on one training fold. Under Protocol A each cell entails five fits; under Protocol B the five fits per dataset, model and seed are shared by all eleven tiers.
- **Evaluation:** the scoring of one fitted model on one evaluation fold at one corruption tier; a cell comprises five evaluations.
- **Protocol A = training robustness:** the learning procedure is trained and evaluated on data from the same corrupted process. **Protocol B = inference robustness:** a model trained on clean data is evaluated on increasingly corrupted inputs.

---

## 3. Pipeline Evolution & Changelog

### Version 2.1 (Current Second Revision)
- **Protocol Separation:** Formally decoupled Protocol A (corruption-aware training robustness) from Protocol B (inference robustness) across all metrics and reporting.
- **Metric Refinement:** Established balanced Area Under the Degradation Curve ($\AUDC_{\mathrm{bal}}$) as the primary confirmatory metric; documented zeroing of $\aMRR$ when above-baseline skill is absent ($A_0 \le B_0$).
- **Knee Estimator Clarification:** Natural cubic spline knee $\etaknee$ re-framed as a population-level descriptive summary rather than an exact per-curve failure indicator; synthetic hinge curve simulations highlighted per-curve sampling variance.
- **Pretrained Foundation Model:** Integrated TabICLv2 into Protocol B under a constrained evaluation regime (4 estimators, 3,000 context rows) with secondary default-setting sensitivity checks (8 estimators).
- **Multiple Testing Rigour:** Separated omnibus Friedman/Iman--Davenport tests from post-hoc pairwise Wilcoxon tests and effect-size concordance ($W$).
- **Supplement restructuring:** The open-problems table was removed; the supplement is organised as Sections S1--S10 (derivations, corpus, per-seed and native-handling tables, Protocol B tests, sensitivity runs, atlas, cleaning rules); the `gap_coverage_supp.tex` input is no longer used.
- **Continuous-only Gaussian:** the continuous-attribute Gaussian operator (`deploy_cont`, `train_cont` blocks of `experiments/run_revision.py`) is the principal Gaussian experiment (`tab_gaussian_principal.tex`); the all-column, categorical-aware and categorical-only variants are reported separately.
- **BibTeX Normalization:** Resolved missing publisher address fields across seminal and recent references (`breiman1984classification`, `donoho1983notion`, `gopfert2020adversarial`, `qu2025tabicl`), eradicating all `\blocation{???}` formatting artifacts.

### Version 1.1 (First Revision)
- **Natural Spline Knots:** Transitioned knee estimator from unpenalized univariate spline to natural cubic spline interpolation with zero boundary curvature ($S''(0)=S''(\etamax)=0$).
- **Seed Replication:** Scaled evaluation grid from single seed (42) to three seeds (42, 43, 44), generating 600 seed-averaged trajectories.
- **Categorical & Missingness Extensions:** Added categorical-aware and categorical-only operators, as well as rank-based MAR and MNAR missingness mechanisms.

### Version 1.0 (Initial Submission)
- Initial release with 6 classical supervised learners, 5 corruption operators, 20 OpenML datasets, and basic AUDC/MRR metrics under seed 42.

---

## 4. Main-Text Tables and Figures: Generating Script and Input Files

| Article item | Generating script | Input result files |
|---|---|---|
| Table 4 (dataset inventory, `tab_inventory_main.tex`) | `experiments/build_revision_tables.py` | `data/*.npz`, `results/analysis/revision/openml_categorical.json` |
| Table 5 (accounting, `tab_accounting.tex`) | `experiments/build_revision_tables.py` | `results/analysis/revision/revision.json` (from `analyze_revision.py`), `results/metrics/extended/*.csv`, `results/metrics/revision/*.csv` |
| Combined statistics (`tab_combined.tex`), confirmatory family | `experiments/analyze_revision.py` | `results/analysis/primary/cells_by_seed.csv`, `results/analysis/revision/trajectories_revision.csv` |
| Categorical operators (`tab_categorical.tex`) | `experiments/build_revision_tables.py` | `results/analysis/revision/trajectories_categorical_A.csv`, `trajectories_categorical_B.csv` |
| Principal Gaussian experiment, continuous attributes only (`tab_gaussian_principal.tex`) | `experiments/build_revision_tables.py` (from `experiments/analyze_revision.py`, `continuous_gaussian_analysis`) | `results/metrics/revision/deploy_cont__seed4{2,3,4}.csv`, `results/metrics/revision/train_cont__seed4{2,3,4}.csv` (run with `experiments/run_revision.py --blocks deploy_cont train_cont`), `results/analysis/revision/trajectories_gauss_principal_{A,B}.csv`, `trajectories_gauss_cont_{A,B}.csv`, `revision.json` |
| Table 9, knee-estimator sensitivity (`tab_ablation.tex`) | `experiments/build_manuscript_tables.py` | `results/analysis/audit_primary/audit_numbers.json` |
| Supplementary Table S7, library defaults and budget (`tab_budget.tex`) | `experiments/build_manuscript_tables.py` | `results/analysis/primary/cells.csv`, `results/metrics/extended/xgb_default__seed42.csv`, `catboost_default__seed42.csv` |
| Knee and artifact sensitivity (`tab_sensitivity.tex`) | `experiments/analyze_sensitivity.py`, `experiments/build_revision_tables.py` | `results/analysis/revision/revision.json`, `results/metrics/artifact/*.csv` |
| Protocol A vs B (`tab_protocol.tex`) | `experiments/build_revision_tables.py` | `results/analysis/revision/trajectories_protocol_b.csv`, `trajectories_revision.csv` |
| TabICL configurations and foundation comparison (`tab_tabicl_configs.tex`, `tab_foundation.tex`) | `experiments/build_revision_tables.py`, `experiments/analyze_foundation.py` | `results/metrics/revision/foundation__*`, `results/analysis/revision/foundation.json` |
| Figures `fig_curves`, `fig_overall`, `fig_ranking`, `fig_knee` | `experiments/build_revision_figures.py` | `results/analysis/revision/trajectories_revision.csv`, `results/analysis/primary/cells.csv` |
| Supplement: CD diagrams, atlas, per-dataset curves | `experiments/build_primary_figures.py`, `experiments/build_manuscript_tables.py` | `results/analysis/primary/cells.csv` |
| Supplement: pol class-ordering sensitivity (`tab_pol_ordering.tex`) | `experiments/run_pol_ordering_sensitivity.py` | `results/metrics/revision/pol_ordering__seed4{2,3,4}.csv`, `pol_ordering_summary.json`, `results/analysis/primary/cells.csv` |
| Supplement: cleaning counts (`tab_cleaning_counts.tex`) | `experiments/dataset_cleaning_counts.py` | `data/*.npz` |

### Superseded outputs

Superseded and clearly unreferenced run logs were moved to `results/superseded/logs/` (`extended_run.log`, `extended_run2.log`, `extended_ccfix.log`, `extended_xgbdefault.log`). Three directories are kept in `results/metrics/` although they are superseded, because `experiments/analyze_fix_extras.py` and `experiments/recompute_original_cells.py` read them as the "before correction" side of the correction audit; they are never inputs to a table: `backup_before_original_patch/`, `extended_backup_gaussfix/`, `extended_backup_tiefix/`. The top-level `backup/`, `overleaf_package_backup/` and `TabStress_Overleaf_Bundle_backup.zip` are earlier submission bundles (git-ignored) and are not part of the pipeline. `results/analysis/latex/` and `results/analysis/audit/` are first-submission outputs still read by `forensic_audit.py` and `verify_manuscript.py`; the manuscript inputs are those listed above.

### Audit

`python3 experiments/audit_manifest.py` checks that every file in the block below exists and prints its sha256; it exits non-zero if any file is missing.

```files
experiments/run_final_pipeline.sh
experiments/build_primary.py
experiments/analyze_revision.py
experiments/analyze_sensitivity.py
experiments/analyze_foundation.py
experiments/build_revision_tables.py
experiments/build_revision_figures.py
experiments/build_manuscript_tables.py
experiments/build_primary_figures.py
experiments/run_pol_ordering_sensitivity.py
experiments/dataset_cleaning_counts.py
experiments/verify_manuscript.py
results/analysis/primary/cells.csv
results/analysis/primary/cells_by_seed.csv
results/analysis/revision/revision.json
results/analysis/revision/confirmatory_family.csv
results/analysis/revision/tab_inventory_main.tex
results/analysis/revision/tab_accounting.tex
results/analysis/revision/tab_combined.tex
results/analysis/revision/tab_categorical.tex
results/analysis/revision/tab_gaussian_principal.tex
results/metrics/revision/deploy_cont__seed42.csv
results/metrics/revision/deploy_cont__seed43.csv
results/metrics/revision/deploy_cont__seed44.csv
results/metrics/revision/train_cont__seed42.csv
results/metrics/revision/train_cont__seed43.csv
results/metrics/revision/train_cont__seed44.csv
results/analysis/revision/trajectories_gauss_principal_A.csv
results/analysis/revision/trajectories_gauss_principal_B.csv
results/analysis/revision/tab_protocol.tex
results/analysis/revision/tab_sensitivity.tex
results/analysis/revision/tab_tabicl_configs.tex
results/analysis/revision/tab_foundation.tex
results/analysis/revision/tab_pol_ordering.tex
results/analysis/revision/tab_cleaning_counts.tex
results/analysis/revision/cleaning_counts.csv
results/metrics/revision/pol_ordering__seed42.csv
results/metrics/revision/pol_ordering__seed43.csv
results/metrics/revision/pol_ordering__seed44.csv
results/metrics/revision/pol_ordering_summary.json
springer_project/build/TabStress_main.tex
springer_project/build/TabStress_supp_main.tex
```
