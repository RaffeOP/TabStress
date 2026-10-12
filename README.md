# TabStress: Nonparametric Robustness Profiling of Tabular Classifiers under Controlled Data Corruption

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Benchmarks: OpenML](https://img.shields.io/badge/Data-OpenML-orange.svg)](https://www.openml.org/)

Official code, data, and reproduction package for the empirical benchmark:  
**"TabStress: Nonparametric Robustness Profiling of Tabular Classifiers under Controlled Data Corruption"**

*Shashank Mouli Satapathy and Devarsh Patel*  
School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, India.

---

## 1. Overview & Research Scope

TabStress provides a comprehensive, non-parametric evaluation framework that maps the continuous degradation trajectories of tabular classifiers across systematically controlled data corruptions.

While standard tabular benchmarks evaluate models exclusively at corruption level zero (clean data), deployed systems regularly encounter label noise, measurement channel drift, missing values, and corrupted categoricals. TabStress formulates robustness as a continuous **response curve** over eleven corruption intensity tiers $\eta \in [0, 0.50]$, benchmarking:

- **6 Classical Classifier Families:** CART, Random Forest, XGBoost, LightGBM, CatBoost, and Multilayer Perceptron (MLP).
- **Tabular Foundation Model:** TabICL (in-context tabular foundation model evaluated under inference deployment).
- **20 Curated OpenML Benchmark Datasets:** Covering binary and multiclass tasks ($K=2$ to $11$), balanced to severely imbalanced distributions ($\pi_{\max}=0.16$ to $0.94$), and diverse feature spaces ($D=8$ to $1{,}776$).
- **Controlled Corruption Operators:**
  1. *Symmetric Label Noise:* Uniform label corruption across all classes.
  2. *Class-Conditional Label Noise:* Asymmetric label corruption (pair-flip on multiclass, minority-to-majority on binary).
  3. *Additive Gaussian Feature Noise:* Zero-mean Gaussian perturbation with variance scaled to feature dispersion (continuous-only principal experiment, with all-column sensitivity control).
  4. *Missing Completely at Random (MCAR):* Bernoulli feature masking with median imputation vs. native tree routing.
  5. *Missing at Random (MAR):* Logistic probability masking conditioned on observed predictors.
  6. *Missing Not at Random (MNAR):* Extreme-quantile masking conditioned on feature values themselves.
  7. *Type-Aware Categorical Noise:* Injection of valid but incorrect discrete categories.
  8. *Compound Corruption:* Simultaneous multi-channel degradation (label corruption + feature noise/missingness).

The benchmark encompasses **22,280 Protocol A cells (111,400 model fits)** and **25,740 Protocol B cells (3,300 model fits)** across three fixed random seeds (42, 43, 44) and hyperparameter regimes.

---

## 2. Evaluation Estimands & Metrics

### Protocol Distinction
- **Protocol A (Training Robustness):** Training and test folds are corrupted jointly. Measures the resilience of the *learning procedure* when training under persistent data corruption.
- **Protocol B (Inference Robustness):** Models are trained strictly on clean historical data and evaluated against corrupted test inputs at deployment. Measures the resilience of a *frozen model* in production.

### Core Metrics
- **$\mathrm{AUDC}_{\mathrm{bal}}$ (Balanced Area Under the Degradation Curve — Primary Metric):** Area under the balanced accuracy trajectory across tiers $\eta \in [0, 0.50]$, preventing artificial inflation caused by majority-class drift under heavy corruption.
- **$\mathrm{AUDC}$ (Standard AUDC):** Unweighted integrated accuracy over the corruption sweep.
- **$\mathrm{MRR}$ (Mean Relative Retention):** Ratio of corrupted endurance to clean baseline accuracy ($\mathrm{AUDC} / \mathrm{acc}_0$).
- **$\mathrm{aMRR}$ (Baseline-Adjusted Retention):** Normalized retention accounting for the naive chance baseline floor.
- **$\eta_{\mathrm{knee}}$ (Empirical Maximum-Curvature Knee):** Knee point detected via cubic spline interpolation identifying the onset of accelerated performance degradation.

---

## 3. Key Findings

1. **Clean Accuracy Does Not Predict Robustness (RQ1):** The most accurate model on clean data retains superior performance under deployment corruption on at most **18%** of datasets under Protocol B.
2. **Protocol Mismatch Inverts Model Ranks (RQ2):** Evaluating only under Protocol A significantly overstates real-world resilience; models rank differently under deployment deterioration.
3. **Catastrophic Failure of Native Missing-Value Routing:** Native tree handling collapses severely when missingness emerges at test time for models trained on clean data, whereas median imputation remains stable.
4. **Resilience of Neural Representations:** While gradient-boosted trees lead on clean data, MLPs exhibit superior relative retention ($\mathrm{MRR}$) under additive Gaussian feature noise at deployment.

---

## 4. Repository Structure

```
TabStress/
├── data/                      # 20 benchmark OpenML datasets (.npz archives)
├── src/                       # Core TabStress Python library
│   ├── noise_injector.py      # Vectorized corruption operators
│   ├── breakdown_math.py      # Natural cubic spline interpolation & knee estimation
│   ├── models.py              # Model wrappers (scikit-learn, XGBoost, LightGBM, CatBoost)
│   ├── data_loader.py         # OpenML dataset ingestion & caching
│   └── statistical_tests.py   # Friedman, Nemenyi, and pairwise Wilcoxon-Holm tests
├── experiments/               # Benchmarking, audits, and reproduction scripts
│   ├── run_final_pipeline.sh  # End-to-end build & verifier pipeline
│   ├── run_benchmark.py       # Primary benchmark harness
│   ├── run_revision.py        # Protocol B & extended mechanism runner
│   ├── verify_manuscript.py   # 306-assertion mathematical & empirical quality gate
│   ├── audit_manifest.py      # Cryptographic SHA-256 verification of all paper assets
│   ├── build_revision_tables.py
│   └── build_revision_figures.py
├── results/                   # Benchmark metric logs and generated analysis tables
│   ├── metrics/               # Raw evaluation cell CSVs (seeds 42, 43, 44)
│   └── analysis/              # Consolidated metrics, rank tests, and LaTeX tables
├── springer_project/          # LaTeX manuscript sources, figures, and bib files
│   ├── build/TabStress_main.tex
│   ├── build/TabStress_supp_main.tex
│   ├── TabStress.bib          # 60 fully verified archival bibliographic entries
│   └── figures/               # Critical-difference diagrams, knee curves, atlas panels
├── PIPELINE_MANIFEST.md       # Cryptographic provenance manifest mapping tables to scripts
├── requirements.txt           # Pinned dependencies for classical models (Python 3.12.6)
├── requirements-tabicl.txt    # Dedicated environment dependencies for TabICL
└── LICENSE                    # MIT License
```

---

## 5. Environment Setup & Installation

The core benchmark requires **Python 3.12.6**.

```bash
# 1. Clone repository
git clone https://github.com/RaffeOP/TabStress.git
cd TabStress

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

*Note:* Foundation model evaluation (TabICL under Protocol B) uses a dedicated environment pinned in `requirements-tabicl.txt` (`tabicl==2.2.0`, `torch==2.14.1`).

---

## 6. Reproducing the Benchmark

### 1. Verify Pipeline & Audit Provenance
Validate the integrity of all pre-computed cells, figures, and tables against the manifest:
```bash
python3 experiments/audit_manifest.py
```

### 2. Run the End-to-End Build and Verification
To execute the complete statistical analysis, compile the manuscript, and run all 306 verification assertions:
```bash
bash experiments/run_final_pipeline.sh
```

### 3. Running Individual Benchmark Cells
To run single evaluation cells or test corruption operators:
```bash
python3 experiments/run_extended.py
```

---

## 7. License

This repository is distributed under the [MIT License](LICENSE).

---

## 8. Citation

If you use TabStress in your research or benchmarks, please cite:

```bibtex
@article{patel2026tabstress,
  title={{TabStress}: Nonparametric Robustness Profiling of Tabular Classifiers under Controlled Data Corruption},
  author={Patel, Devarsh and Satapathy, Shashank Mouli},
  year={2026}
}
```
