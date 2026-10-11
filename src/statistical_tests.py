"""
TabStress: Non-Parametric Statistical Testing & CD Diagram Engine
Implements the Janez Demšar (JMLR 2006) protocol:
1. Friedman Non-Parametric Rank Test
2. Nemenyi Post-Hoc Critical Difference (CD) computation & diagram plotting
3. Cohen's d pairwise effect sizes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats, create_report

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Computes Cohen's d effect size using pooled standard deviation."""
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    diff = np.nanmean(g1) - np.nanmean(g2)
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.nanvar(g1, ddof=1), np.nanvar(g2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return float(diff / s_pooled)

def run_demsar_analysis(df_results: pd.DataFrame, alpha: float = 0.05, output_cd_plot: str = None) -> dict:
    """
    Executes Demšar non-parametric statistical testing over multi-dataset results.
    df_results: DataFrame where rows are datasets and columns are model families.
    """
    # Pin the Demšar (2006) path: autorank otherwise switches to RM-ANOVA + Tukey HSD
    # whenever a sample happens to pass its normality/homoscedasticity pre-tests.
    res = autorank(df_results, alpha=alpha, verbose=False, force_mode="nonparametric")
    
    if output_cd_plot:
        fig, ax = plt.subplots(figsize=(8, 3))
        plot_stats(res, ax=ax, allow_insignificant=True)
        plt.tight_layout()
        plt.savefig(output_cd_plot, dpi=300, bbox_inches="tight")
        plt.close()

    report = create_report(res)
    return {
        "result_object": res,
        "p_value": res.pvalue,
        "cd": getattr(res, "cd", None),
        "rankdf": res.rankdf,
        "report": report
    }
