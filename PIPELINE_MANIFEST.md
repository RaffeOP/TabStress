# TabStress Pipeline Manifest & Changelog

**Manuscript:** TabStress: Nonparametric Robustness Profiling of Tabular Classifiers under Controlled Data Corruption  
**Authors:** Devarsh Patel & Shashank Mouli Satapathy (Vellore Institute of Technology)  
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
| **3. Operator Audit** | `experiments/operator_audit.py` | Primary grid & seeds | `results/analysis/audit_primary/operator_audit.json` | Audits coupling, target rates, and flat blocks |
| **4. Forensic Audits** | `experiments/forensic_audit.py` | Primary & original data | Audit JSON logs | Verifies fold independence, target leakage absence |
| **5. Extended Extras** | `experiments/analyze_primary_extras.py` | Primary data & baselines | `results/analysis/audit_primary/extras.json` | Computes native handling, budget sensitivity, variance ratios |
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

Superseded and clearly unreferenced run logs were moved to `results/superseded/logs/`: the four `extended_*.log` files, and every stale progress, failure or finalization log that used to sit in `results/metrics/revision/` (prefixed `revision_`, for example `revision_finalize.log`, which records a failed finalization that was later superseded). The resumable state files that scripts read (`done_*.txt`, `foundation_failures.txt`, `run_foundation.log`) remain in `results/metrics/revision/`. `results/README.md` gives the status of every `results/` directory.

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

---

## 5. Pipeline Provenance & Execution Map

This section is the authority for which script, input and configuration produced each table and figure. `results/README.md` gives the status of every directory under `results/`.

### 5.1 Configuration flags

| Flag | Meaning | Set in | Output files |
|---|---|---|---|
| Matched budget (primary) | six families, 100 boosting rounds, learning rate 0.1, RF 100 trees, MLP as in Section 4.2 of the article | `experiments/run_extended.py` | `results/metrics/extended/{base,cc_binary,native_nan}__seed{42,43,44}.csv`; the same harness is used by every revision block |
| Library default, XGBoost | 100 rounds, learning rate 0.3, depth 6 (sensitivity only) | `experiments/run_extended.py` (comment at line 78) | `results/metrics/extended/xgb_default__seed42.csv` |
| Library default, CatBoost | 1,000 rounds, automatic learning rate (sensitivity only) | `experiments/run_extended.py` | `results/metrics/extended/catboost_default__seed42.csv` |
| Tuned configurations | 6 random-search trials per dataset and family on clean data, inner 3-fold CV; seed 42 | `experiments/run_revision.py` (`TUNED_TRIALS = 6`, `SEARCH_SPACE`) | `results/metrics/revision/tuned__seed42.csv`; search space in Supplementary Table S1 (`tab:search`) |
| Protocol A (training robustness) | model trained and evaluated on the corrupted process; corruption applied inside each fold | `run_extended.py` (primary grid and variants `base`, `cc_binary`, `native_nan`); `run_revision.py` blocks `train_mech`, `train_cont`, `train_cat` | `results/metrics/extended/*`, `results/metrics/revision/train_*__seed*.csv` |
| Protocol B (inference robustness), imputed | clean-trained models, corrupted validation folds, median imputation | `run_revision.py` block `deploy` | `results/metrics/revision/deploy__seed*.csv` |
| Protocol B, native handling | as above, no imputer, five tree families, missingness operators only | `run_revision.py` block `deploy` | `results/metrics/revision/deploy__seed*.csv` |
| Continuous-only Gaussian | Gaussian noise on continuous attributes only (principal experiment) | `run_revision.py` blocks `deploy_cont`, `train_cont` | `results/metrics/revision/deploy_cont__seed*.csv`, `train_cont__seed*.csv` |
| Categorical-aware Gaussian | Gaussian on all columns, categorical-aware | `run_revision.py` blocks `deploy_cat`, `train_cat` | `results/metrics/revision/deploy_cat__seed*.csv`, `train_cat__seed*.csv` |
| Compound corruption | two operators together on six datasets | `run_revision.py` block `compound` | `results/metrics/revision/compound__seed*.csv` |
| Class ordering (pol) | pol target ordering: lexicographic (primary) and numeric (sensitivity) | `experiments/run_pol_ordering_sensitivity.py` | `results/metrics/revision/pol_ordering__seed*.csv`, `pol_ordering_summary.json` |
| TabICL, constrained (main) | 4 estimators, in-context set capped at 3,000 rows, median imputation before the model, Bioresponse excluded; Protocol B only, seeds 42–44 | `experiments/run_foundation.py` (`results/metrics/revision/foundation_settings.json`) | `results/metrics/revision/foundation__seed*.csv`, `foundation_failures.txt` |
| TabICL, default-settings sensitivity | 8 estimators, no row cap; six datasets, seed 42 | `results/metrics/revision/foundation_default_settings.json` | `results/metrics/revision/foundation_default__seed42.csv` |
| Deduplicated copies | five datasets with duplicate rows (nomao, pol, volkert, phoneme, ozone_level) | `experiments/run_dedup_sensitivity.py`, `experiments/complete_dedup_sensitivity.py` | `results/metrics/artifact/{base,cc_binary,native_nan}__seed*.csv` (rows `*_dedup`), see 5.4 |
| Cleaned copies | spambase (397 rows), churn (ID column), MiniBooNE (468 sentinel rows) | `experiments/run_artifact_sensitivity.py` | `results/metrics/artifact/{base,cc_binary,native_nan}__seed*.csv` (rows `*_clean`), see 5.4 |
| MLP early-stopping ablation | MLP with early stopping disabled, symmetric label noise, seed 42, all twenty datasets | `experiments/run_mlp_ablation.py` | `results/metrics/ablation/mlp_no_es__seed42.csv` |

### 5.2 Table and figure map

Generated by `python3 experiments/provenance_map.py` from the two manuscript files. The script reads each `\label{tab:...}` and `\label{fig:...}`, identifies whether it is inlined from a generated file, typeset from a figure file, or hand-typed, and names the generating script. `python3 experiments/provenance_map.py --check` exits non-zero if any label has no known generator.

| Manuscript file | Label | Typeset from | Generating script |
|---|---|---|---|
| TabStress.tex | `tab:comparison` | `hand-typed` | hand-typed in the .tex file (no generator) |
| TabStress.tex | `tab:gaps` | `hand-typed` | hand-typed in the .tex file (no generator) |
| TabStress.tex | `tab:roles` | `hand-typed` | hand-typed in the .tex file (no generator) |
| TabStress.tex | `fig:pipeline` | `sections/methodology_flowchart.tex` | hand-typed TikZ (sections/methodology_flowchart.tex) |
| TabStress.tex | `tab:inventory` | `../results/analysis/revision/tab_inventory_main.tex` | experiments/build_revision_tables.py |
| TabStress.tex | `tab:accounting` | `../results/analysis/revision/tab_accounting.tex` | experiments/build_revision_tables.py |
| TabStress.tex | `fig:curves` | `fig_curves.png` | experiments/build_revision_figures.py |
| TabStress.tex | `fig:overall` | `fig_overall.png` | experiments/build_revision_figures.py |
| TabStress.tex | `tab:combined` | `../results/analysis/revision/tab_combined.tex` | experiments/analyze_revision.py |
| TabStress.tex | `tab:amazon` | `hand-typed` | hand-typed in the .tex file (no generator) |
| TabStress.tex | `tab:gaussian_principal` | `../results/analysis/revision/tab_gaussian_principal.tex` | experiments/build_revision_tables.py |
| TabStress.tex | `tab:categorical` | `../results/analysis/revision/tab_categorical.tex` | experiments/build_revision_tables.py |
| TabStress.tex | `fig:ranking` | `fig_ranking.png` | experiments/build_revision_figures.py |
| TabStress.tex | `tab:ablation` | `../results/analysis/latex_primary/tab_ablation.tex` | experiments/build_manuscript_tables.py |
| TabStress.tex | `fig:knee` | `fig_knee.png` | experiments/build_revision_figures.py |
| TabStress.tex | `tab:sensitivity` | `../results/analysis/revision/tab_sensitivity.tex` | experiments/analyze_sensitivity.py, experiments/build_revision_tables.py |
| TabStress.tex | `tab:protocol` | `../results/analysis/revision/tab_protocol.tex` | experiments/build_revision_tables.py |
| TabStress.tex | `tab:tabicl_configs` | `../results/analysis/revision/tab_tabicl_configs.tex` | experiments/build_revision_tables.py |
| TabStress.tex | `tab:foundation` | `../results/analysis/revision/tab_foundation.tex` | experiments/analyze_foundation.py |
| TabStress_Supplementary.tex | `tab:search` | `../results/analysis/revision/tab_search_space.tex` | experiments/build_revision_tables.py |
| TabStress_Supplementary.tex | `tab:inventory` | `../results/analysis/latex_primary/tab_inventory.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:friedman` | `../results/analysis/latex_primary/tab_friedman.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:ranks` | `../results/analysis/latex_primary/tab_ranks.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:seeds` | `../results/analysis/latex_primary/tab_seeds.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:native` | `../results/analysis/latex_primary/tab_native.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:budget` | `../results/analysis/latex_primary/tab_budget.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:protocolb_tests` | `../results/analysis/revision/tab_protocol_b_tests.tex` | experiments/build_revision_tables.py |
| TabStress_Supplementary.tex | `fig:cd_all` | `cd_mrr_symmetric_label.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `tab:endurance_mrr` | `../results/analysis/latex_primary/tab_endurance.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:endurance_audc` | `../results/analysis/latex_primary/tab_endurance.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_Bioresponse` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_qsar_biodeg` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_eye_movements` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_gesture_phase` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_ozone_level` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_steel_plates_fault` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_vehicle` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_electricity` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_pol` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_miniboone` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_bank_marketing` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_churn` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_amazon_access` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_nomao` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_spambase` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_madelon` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_hill_valley` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_jungle_chess` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_phoneme` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:traj_volkert` | `../results/analysis/latex_primary/tab_trajectories.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_Bioresponse` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_qsar_biodeg` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_eye_movements` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_gesture_phase` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_ozone_level` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_steel_plates_fault` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_vehicle` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_electricity` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_pol` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_miniboone` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_bank_marketing` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_churn` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_amazon_access` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_nomao` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_spambase` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_madelon` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_hill_valley` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_jungle_chess` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_phoneme` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:grid_volkert` | `../results/analysis/latex_primary/tab_accuracy_grids.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `fig:atlas_Bioresponse` | `curves_Bioresponse.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_qsar-biodeg` | `curves_Bioresponse.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_g1` | `curves_Bioresponse.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_eye_movements` | `curves_eye_movements.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_gesture_phase` | `curves_eye_movements.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_ozone_level` | `curves_ozone_level.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_steel-plates-fault` | `curves_steel-plates-fault.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_vehicle` | `curves_steel-plates-fault.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_g2` | `curves_steel-plates-fault.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_electricity` | `curves_electricity.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_pol` | `curves_electricity.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_miniboone` | `curves_miniboone.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_bank-marketing` | `curves_bank-marketing.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_churn` | `curves_bank-marketing.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_g3` | `curves_bank-marketing.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_amazon_access` | `curves_amazon_access.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_nomao` | `curves_amazon_access.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_spambase` | `curves_spambase.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_madelon` | `curves_madelon.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_hill_valley` | `curves_madelon.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_g4` | `curves_madelon.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_jungle_chess` | `curves_jungle_chess.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_phoneme` | `curves_jungle_chess.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `fig:atlas_volkert` | `curves_volkert.png` | experiments/build_primary_figures.py |
| TabStress_Supplementary.tex | `tab:wilcoxon` | `../results/analysis/latex_primary/tab_wilcoxon.tex` | experiments/build_manuscript_tables.py |
| TabStress_Supplementary.tex | `tab:sens_settings` | `../results/analysis/revision/tab_sensitivity_extra.tex` | experiments/analyze_sensitivity.py |
| TabStress_Supplementary.tex | `tab:sens_compound` | `../results/analysis/revision/tab_sensitivity_extra.tex` | experiments/analyze_sensitivity.py |
| TabStress_Supplementary.tex | `tab:sens_bayes` | `../results/analysis/revision/tab_sensitivity_extra.tex` | experiments/analyze_sensitivity.py |
| TabStress_Supplementary.tex | `tab:sens_gap` | `../results/analysis/revision/tab_sensitivity_extra.tex` | experiments/analyze_sensitivity.py |
| TabStress_Supplementary.tex | `tab:cleaning_counts` | `../results/analysis/revision/tab_cleaning_counts.tex` | experiments/dataset_cleaning_counts.py |
| TabStress_Supplementary.tex | `tab:pol_ordering` | `../results/analysis/revision/tab_pol_ordering.tex` | experiments/run_pol_ordering_sensitivity.py |
| TabStress_Supplementary.tex | `tab:tabicl_pd` | `hand-typed` | hand-typed in the .tex file (no generator) |

### 5.3 Run and recomputation logs

Logs that are kept and what they record:

| Log | Records |
|---|---|
| `results/analysis/revision/rerun_audit.log`, `rerun_audit.json` | Recomputation of 120 stored Protocol A cells (10 blocks, sampled from every block and seed) with the current harness; maximum absolute difference $1.1\times10^{-16}$ |
| `results/metrics/original_recompute.log`, `original_patch_log.csv` | Gaussian-operator correction (constant columns received noise scale 1 instead of 0): control cells reproduced to $1.1\\times10^{-16}$, then every replaced value logged before and after (`recompute_original_cells.py`) |
| `results/metrics/artifact_run.log` | Progress of the artifact-sensitivity run (2,664 tasks) |
| `results/metrics/revision/run_foundation.log`, `done_foundation.txt`, `foundation_failures.txt` | TabICL run: settings, the Bioresponse exclusion with its memory reason, and the resumable task list |
| `results/metrics/revision/done_tasks.txt` | Resumable task list of the revision blocks (`run_completion.sh`) |
| `results/superseded/logs/` | Superseded run and finalization logs, prefixed by their source directory |

### 5.4 Post-hoc corrections and sensitivity runs: index

| Correction or run | Input data | Generating script | Per-cell outputs | Summary location |
|---|---|---|---|---|
| spambase, 397 rows (391 duplicate rows, rule R1; 6 rows of 3 conflicting vectors, rule R2) | `data/spambase_clean.npz` | `experiments/run_artifact_sensitivity.py` | `results/metrics/artifact/{base,cc_binary,native_nan}__seed{42,43,44}.csv`, rows with `dataset = spambase_clean` (committed) | `results/analysis/revision/revision.json` → `artifacts/cleaned`; `results/analysis/revision/tab_sensitivity.tex` |
| churn, administrative identifier column dropped | `data/churn_clean.npz` | `experiments/run_artifact_sensitivity.py` | `results/metrics/artifact/{base,cc_binary,native_nan}__seed{42,43,44}.csv`, rows with `dataset = churn_clean` (committed) | as above; `results/analysis/audit_primary/audit_numbers.json` (inventory) |
| MiniBooNE, 468 sentinel rows (all features equal to −999) removed | `data/miniboone_clean.npz` | `experiments/run_artifact_sensitivity.py` | `results/metrics/artifact/{base,cc_binary,native_nan}__seed{42,43,44}.csv`, rows with `dataset = miniboone_clean` (committed) | as above; `results/analysis/audit_primary/audit_numbers.json` |
| Deduplicated corpus, five datasets (nomao, pol, volkert, phoneme, ozone_level) | `data/{name}_dedup.npz` | `experiments/run_dedup_sensitivity.py`, `experiments/complete_dedup_sensitivity.py` | `results/metrics/artifact/{base,cc_binary,native_nan}__seed{42,43,44}.csv`, rows with `dataset = {name}_dedup` (committed) | `results/analysis/revision/revision.json` → `artifacts/dedup`, `baselines`; `results/analysis/revision/tab_sensitivity_extra.tex` |
| Duplicate-row counts | `data/*.npz` | `experiments/duplicate_scan.py` | printed counts | Section 4.2 of the article |
| Cleaning counts and the sentinel criterion | `data/*.npz` | `experiments/dataset_cleaning_counts.py` | `results/analysis/revision/cleaning_counts.csv` | `results/analysis/revision/tab_cleaning_counts.tex` (Supplementary Section S10) |
| MLP early-stopping ablation | `data/*.npz` | `experiments/run_mlp_ablation.py` | `results/metrics/ablation/mlp_no_es__seed42.csv` (committed) | Section 5.7 of the article |
| pol class ordering | `data/pol.npz` | `experiments/run_pol_ordering_sensitivity.py` | `results/metrics/revision/pol_ordering__seed*.csv` (committed) | `results/analysis/revision/tab_pol_ordering.tex` |

### 5.5 Storage of the cleaned and deduplicated runs

Per-cell outputs of all eight cleaned or deduplicated datasets are committed. They are rows of `results/metrics/artifact/{base,cc_binary,native_nan}__seed{42,43,44}.csv`, keyed by the `dataset` column (`spambase_clean`, `churn_clean`, `miniboone_clean`, `nomao_dedup`, `pol_dedup`, `volkert_dedup`, `phoneme_dedup`, `ozone_level_dedup`). Each binary dataset has 186 `base`, 60 `cc_binary` and 50 `native_nan` cells per seed; the multiclass `pol_dedup` and `volkert_dedup` have 246 `base` (including pair-flip) and 50 `native_nan` cells per seed. No cell is duplicated. Stale revision logs were moved to `results/superseded/logs/` (Section 4, "Superseded outputs").

### 5.6 Audit commands

```bash
python3 experiments/provenance_map.py --check   # every table and figure label has a generator
python3 experiments/audit_manifest.py           # every file in the Section 4 block exists
python3 experiments/verify_manuscript.py        # 306 checks against the typeset manuscript
```
