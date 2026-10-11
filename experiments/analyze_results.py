import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.breakdown_math import compute_critical_breakdown_threshold
from src.statistical_tests import run_demsar_analysis

INPUT_CSV = "results/metrics/benchmark_results.csv"
OUTPUT_DIR = "results/analysis"
FIG_DIRS = ["paper/figures", "springer_project/figures"]

# Academic Springer styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "lines.linewidth": 1.5,
    "lines.markersize": 4
})

def analyze_breakdowns():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} does not exist.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fd in FIG_DIRS:
        os.makedirs(fd, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} evaluation records.")

    inflection_records = []
    datasets = sorted(df["dataset"].unique())
    models = sorted(df["model"].unique())
    modalities = sorted(df["modality"].unique())

    print(f"Analyzing {len(datasets)} datasets, {len(models)} models, {len(modalities)} modalities...")

    for d_name in datasets:
        for m_name in models:
            for mod in modalities:
                sub = df[(df["dataset"] == d_name) & (df["model"] == m_name) & (df["modality"] == mod)].sort_values("eta")
                if len(sub) < 3:
                    continue

                x = sub["eta"].to_numpy(dtype=np.float64)
                y = sub["mean_acc"].to_numpy(dtype=np.float64)

                metrics = compute_critical_breakdown_threshold(x, y)
                inflection_records.append({
                    "dataset": d_name,
                    "model": m_name,
                    "modality": mod,
                    "baseline_acc": float(y[0]),
                    "final_acc": float(y[-1]),
                    "eta_star": metrics["eta_star"],
                    "max_curvature": metrics["max_curvature"],
                    "d_rate": metrics["d_rate"],
                    "audc": metrics["audc"],
                    "mrr": metrics["mrr"]
                })

    df_inflect = pd.DataFrame(inflection_records)
    inflect_path = os.path.join(OUTPUT_DIR, "inflection_thresholds.csv")
    df_inflect.to_csv(inflect_path, index=False)
    print(f"✅ Knee analysis saved to: {inflect_path} ({len(df_inflect)} entries)")

    # 1. Demšar Statistical Analysis and Critical Difference Diagrams
    for mod in modalities:
        sub_mod = df_inflect[df_inflect["modality"] == mod]
        if len(sub_mod) == 0:
            continue

        pivot_mrr = sub_mod.pivot(index="dataset", columns="model", values="mrr").dropna()
        if pivot_mrr.shape[0] >= 3 and pivot_mrr.shape[1] >= 3:
            for fd in FIG_DIRS:
                cd_plot_path = os.path.join(fd, f"cd_diagram_{mod}.png")
                try:
                    stat_res = run_demsar_analysis(pivot_mrr, alpha=0.05, output_cd_plot=cd_plot_path)
                    print(f"Demšar analysis for {mod}: Friedman p={stat_res['p_value']:.4e} -> {cd_plot_path}")
                except Exception as e:
                    print(f"[WARNING] Demšar analysis for {mod} encountered: {e}")

    # 2. Complete Exhaustive Degradation Trajectories for ALL Datasets
    print(f"Generating degradation curves for all {len(datasets)} datasets...")
    for d_name in datasets:
        n_mods = len(modalities)
        fig, axes = plt.subplots(1, n_mods, figsize=(4.2 * n_mods, 3.6), sharey=True)
        if n_mods == 1:
            axes = [axes]

        for ax_idx, mod in enumerate(modalities):
            ax = axes[ax_idx]
            sub = df[(df["dataset"] == d_name) & (df["modality"] == mod)]

            for m_name in models:
                m_sub = sub[sub["model"] == m_name].sort_values("eta")
                if len(m_sub) > 0:
                    ax.plot(m_sub["eta"], m_sub["mean_acc"], marker="o", markersize=3, label=m_name.upper())

            mod_label = mod.replace("_", " ").title()
            ax.set_title(f"{mod_label}", fontsize=11, fontweight="bold")
            ax.set_xlabel(r"Noise Tier ($\eta$)", fontsize=10)
            if ax_idx == 0:
                ax.set_ylabel("Stratified 5-Fold Accuracy", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)

        axes[0].legend(loc="lower left", fontsize=8, framealpha=0.8)
        fig.suptitle(f"Continuous Degradation Trajectories: {d_name}", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()

        for fd in FIG_DIRS:
            curve_plot_path = os.path.join(fd, f"degradation_curve_{d_name}.png")
            plt.savefig(curve_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    # 3. Macro Robustness Comparison (MRR Violin & Boxplots across all datasets)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df_inflect, x="model", y="mrr", hue="modality", ax=axes[0], palette="Set2")
    axes[0].set_title("Marginal Robustness Retention (MRR) Distribution", fontweight="bold")
    axes[0].set_xlabel("Model Architecture")
    axes[0].set_ylabel("MRR Score")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].tick_params(axis="x", rotation=15)

    sns.boxplot(data=df_inflect, x="model", y="eta_star", hue="modality", ax=axes[1], palette="Set2")
    axes[1].set_title(r"Critical Knee Threshold ($\eta_{\mathrm{knee}}$) Distribution", fontweight="bold")
    axes[1].set_xlabel("Model Architecture")
    axes[1].set_ylabel(r"Knee ($\eta_{\mathrm{knee}}$)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    for fd in FIG_DIRS:
        mrr_box_path = os.path.join(fd, "macro_robustness_comparison.png")
        plt.savefig(mrr_box_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated macro_robustness_comparison.png")

    # 4. Generate Publication-Grade LaTeX Summary Tables
    # Summary of MRR and eta* per model and modality
    summary_pivot_mrr = df_inflect.groupby(["model", "modality"])["mrr"].agg(["mean", "std"]).reset_index()
    summary_pivot_eta = df_inflect.groupby(["model", "modality"])["eta_star"].agg(["mean", "std"]).reset_index()

    latex_table_path = os.path.join(OUTPUT_DIR, "summary_table_latex.tex")
    with open(latex_table_path, "w") as f:
        f.write("% ==========================================================================\n")
        f.write("% Table: Aggregate Empirical Robustness Profiles across 20 Benchmark Datasets\n")
        f.write("% ==========================================================================\n")
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Aggregate Knee and Endurance Statistics across the 20-Dataset Gold Standard Tabular Benchmark. Evaluated across 4 Synthetic Noise Modalities (Mean $\\pm$ Standard Deviation).}\n")
        f.write("\\label{tab:aggregate_knee_results}\n")
        f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llcccc@{}}\n\\toprule\n")
        f.write("Model Family & Modality & Baseline Acc ($\\eta=0$) & Terminal Acc ($\\eta=0.5$) & $\\eta_{\mathrm{knee}}$ Knee & MRR (\\%) \\\\\n\\midrule\n")

        for m_name in models:
            m_sub = df_inflect[df_inflect["model"] == m_name]
            for mod in modalities:
                sub_mm = m_sub[m_sub["modality"] == mod]
                if len(sub_mm) == 0:
                    continue
                b_mean, b_std = sub_mm["baseline_acc"].mean(), sub_mm["baseline_acc"].std()
                f_mean, f_std = sub_mm["final_acc"].mean(), sub_mm["final_acc"].std()
                e_mean, e_std = sub_mm["eta_star"].mean(), sub_mm["eta_star"].std()
                r_mean, r_std = sub_mm["mrr"].mean(), sub_mm["mrr"].std()

                mod_clean = mod.replace("_", " ").title()
                f.write(f"{m_name.upper()} & {mod_clean} & {b_mean:.3f}$\\pm${b_std:.3f} & {f_mean:.3f}$\\pm${f_std:.3f} & {e_mean:.3f}$\\pm${e_std:.3f} & {r_mean*100:.1f}\\%$\\pm${r_std*100:.1f}\\% \\\\\n")
            f.write("\\midrule\n")

        f.write("\\bottomrule\n\\end{tabular*}\n\\end{table*}\n")

    print(f"Generated publication-grade LaTeX summary table: {latex_table_path}")
    print("\nComplete Analysis Pipeline Finished Successfully!")

if __name__ == "__main__":
    analyze_breakdowns()
