"""Figures for the seed-averaged five-condition analysis.

  springer_project/figures/curves_<dataset>.png   degradation curves, mean over three seeds,
                                                  band = +/- 1 SD across seeds, one panel per condition
  springer_project/figures/cd_<metric>_<condition>.png   Nemenyi CD diagrams (autorank, non-parametric mode forced)
  springer_project/figures/macro_primary.png      MRR, AUDC and eta* per model, one column per condition

Colour encodes the model family in a fixed order validated for colour-vision deficiency
(dataviz reference palette); every family also has its own marker so that identity never
rests on colour alone, including in greyscale print."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autorank import autorank, plot_stats

CELLS = "results/analysis/primary/cells.csv"
TRAJ = "results/analysis/audit_primary/trajectory_audit.csv"
FIG = "springer_project/figures"
MODELS = ["cart", "randomforest", "xgboost", "lightgbm", "catboost", "mlp"]
LABEL = {"cart": "CART", "randomforest": "RF", "xgboost": "XGBoost", "lightgbm": "LightGBM",
         "catboost": "CatBoost", "mlp": "MLP"}
COLOR = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARKER = dict(zip(MODELS, ["o", "s", "^", "D", "v", "P"]))
CONDITIONS = ["symmetric_label", "class_conditional", "gaussian_feature", "mcar_imputed", "mcar_native"]
CTITLE = {"symmetric_label": "Symmetric label", "class_conditional": "Class-conditional label",
          "gaussian_feature": "Gaussian feature", "mcar_imputed": "MCAR, median-imputed",
          "mcar_native": "MCAR, native handling"}
CSHORT = {"symmetric_label": "Symmetric\nlabel", "class_conditional": "Class-cond.\nlabel",
          "gaussian_feature": "Gaussian\nfeature", "mcar_imputed": "MCAR,\nimputed", "mcar_native": "MCAR,\nnative"}
INVENTORY = "results/analysis/audit_primary/audit_numbers.json"
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#e4e3dd"

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False})


def info_panel(ax, handles, labels, meta):
    """Sixth panel of an atlas figure: the legend and the size of the dataset."""
    ax.axis("off")
    ax.legend(handles, labels, loc="upper left", ncol=2, frameon=False, borderaxespad=0.1, handlelength=2.2,
              columnspacing=1.0)
    rows = f"N = {meta['N']:,}" + (f" (used {meta['N_eff']:,})" if meta["N_eff"] < meta["N"] else "")
    reference = "dashed line: majority-class rate" + ("; dotted: 1/K" if 1.0 / meta["K"] < meta["majority_rate"] - 0.02 else "")
    ax.text(0.03, 0.02, "\n".join([rows, f"D = {meta['D']}, K = {meta['K']}",
                                     rf"majority rate $\pi_{{\max}}$ = {meta['majority_rate']:.2f}", reference]),
            transform=ax.transAxes, va="bottom", ha="left", color=MUTED)


def curves(cells, inventory):
    """One figure per dataset, drawn at print size: five condition panels and an information panel."""
    for d, g in cells.groupby("dataset"):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.1), sharey=True)
        panels = axes.ravel()
        chance = 1.0 / inventory[d]["K"]
        majority = inventory[d]["majority_rate"]
        for ax, cond in zip(panels, CONDITIONS):
            sub = g[g.condition == cond]
            ax.axhline(majority, color=MUTED, linestyle="--", linewidth=0.8, zorder=1)
            if chance < majority - 0.02:
                ax.axhline(chance, color=MUTED, linestyle=":", linewidth=0.8, zorder=1)
            for m in MODELS:
                s = sub[sub.model == m].sort_values("eta")
                ax.fill_between(s.eta, s.mean_acc - s.sd_seed, s.mean_acc + s.sd_seed,
                                color=COLOR[m], alpha=0.14, linewidth=0)
                ax.plot(s.eta, s.mean_acc, color=COLOR[m], marker=MARKER[m], markersize=3.0,
                        linewidth=1.1, markeredgewidth=0, label=LABEL[m])
            ax.set_title(CTITLE[cond], color=INK)
            ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
            ax.grid(True, color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
        for ax in (axes[1, 0], axes[1, 1], axes[0, 2]):
            ax.set_xlabel(r"intensity $\eta$")
        for ax in axes[:, 0]:
            ax.set_ylabel("accuracy")
        handles, labels = panels[0].get_legend_handles_labels()
        info_panel(panels[5], handles, labels, inventory[d])
        fig.tight_layout(h_pad=0.9, w_pad=0.5)
        fig.savefig(os.path.join(FIG, f"curves_{d}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def cd_diagrams(traj):
    """Nemenyi critical-difference diagrams, drawn at the size at which they are printed (half
    the text width). Same analysis as src.statistical_tests.run_demsar_analysis."""
    out = {}
    for metric in ["mrr", "audc"]:
        for cond in CONDITIONS:
            pv = traj[traj.modality == cond].pivot(index="dataset", columns="model", values=metric)[MODELS]
            res = autorank(pv.rename(columns=LABEL), alpha=0.05, verbose=False, force_mode="nonparametric")
            fig, ax = plt.subplots(figsize=(3.9, 1.7))
            plot_stats(res, ax=ax, allow_insignificant=True)
            fig.savefig(os.path.join(FIG, f"cd_{metric}_{cond}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
            out[f"{metric}|{cond}"] = float(res.pvalue)
    return out


def macro(traj):
    """Small multiples: one row per condition, one column per score, drawn at print size."""
    cols = [("mrr", "MRR"), ("audc", "AUDC"), ("eta_def", r"$\eta_{\mathrm{knee}}$")]
    t = traj.copy()
    t["eta_def"] = t["eta_star"].where(t["kappa_max"] >= 1e-9)
    fig, axes = plt.subplots(len(CONDITIONS), len(cols), figsize=(7.2, 8.2), sharey="col", sharex=True)
    for i, cond in enumerate(CONDITIONS):
        for j, (col, lab) in enumerate(cols):
            ax = axes[i, j]
            data = [t[(t.modality == cond) & (t.model == m)][col].dropna().to_numpy() for m in MODELS]
            bp = ax.boxplot(data, widths=0.62, patch_artist=True, showfliers=True,
                            medianprops=dict(color=INK, linewidth=1.1),
                            flierprops=dict(marker="o", markersize=2.2, markerfacecolor=MUTED, markeredgewidth=0),
                            whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                            boxprops=dict(linewidth=0.7, edgecolor=MUTED))
            for patch, m in zip(bp["boxes"], MODELS):
                patch.set_facecolor(COLOR[m])
                patch.set_alpha(0.85)
            ax.grid(True, axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(lab, color=INK)
            if j == 0:
                ax.set_ylabel(CSHORT[cond])
    for ax in axes[-1]:
        ax.set_xticks(range(1, len(MODELS) + 1))
        ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=40, ha="right")
    fig.tight_layout(h_pad=0.7, w_pad=0.8)
    fig.savefig(os.path.join(FIG, "macro_primary.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    cells = pd.read_csv(CELLS)
    traj = pd.read_csv(TRAJ)
    with open(INVENTORY) as fh:
        inventory = json.load(fh)["inventory"]
    curves(cells, inventory)
    p = cd_diagrams(traj)
    macro(traj)
    print("figures written; autorank chi-square p:", {k: f"{v:.2e}" for k, v in p.items()})


if __name__ == "__main__":
    main()
