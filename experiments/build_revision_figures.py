"""Main-text figures of the revised manuscript (revision points 30-31).

  fig_curves.png    representative degradation curves under both protocols
  fig_overall.png   balanced AUDC of every model under every condition, Protocols A and B
  fig_ranking.png   one multi-panel ranking figure: (a) MRR, (b) AUDC, (c) balanced MRR, (d) balanced AUDC
  fig_knee.png      knee-estimator reliability: synthetic recovery, seed agreement, pooled knees
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.gaussian_principal import load_cont, substitute_continuous

OUT = "springer_project/figures"
REV = "results/analysis/revision"
MODELS = ["cart", "randomforest", "xgboost", "lightgbm", "catboost", "mlp"]
NAME = {"cart": "CART", "randomforest": "RF", "xgboost": "XGBoost", "lightgbm": "LightGBM",
        "catboost": "CatBoost", "mlp": "MLP"}
COLOR = dict(zip(MODELS, ["#7f7f7f", "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd"]))
MARKER = dict(zip(MODELS, ["o", "s", "^", "D", "v", "P"]))
COND_A = {"symmetric_label": "Symmetric label", "class_conditional": "Class-conditional label",
          "gaussian_feature": "Gaussian feature", "mcar_imputed": "MCAR (imputed)", "mcar_native": "MCAR (native)"}
COND_B = {"gaussian_feature": "Gaussian feature", "mcar_imputed": "MCAR (imputed)",
          "mcar_native": "MCAR (native)", "mar_imputed": "MAR (imputed)", "mnar_imputed": "MNAR (imputed)"}
CD = 1.686
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7})


def deploy_cells():
    files = [os.path.join("results/metrics/revision", f) for f in os.listdir("results/metrics/revision")
             if f.startswith("deploy__")]
    dep = substitute_continuous(pd.concat([pd.read_csv(f) for f in files]), load_cont("deploy_cont"), "modality")
    return dep[dep.pipeline == "imputed"].groupby(["modality", "dataset", "model", "eta"], as_index=False).mean_acc.mean()


def fig_curves():
    cells = pd.read_csv("results/analysis/primary/cells.csv")
    dep = deploy_cells()
    datasets = [("phoneme", "phoneme (binary, $\\pi_{\\max}=0.71$)"),
                ("bank-marketing", "bank-marketing (binary, $\\pi_{\\max}=0.88$)"),
                ("volkert", "volkert (10 classes)")]
    panels = [("A", "symmetric_label", "Symmetric label noise (A)"), ("A", "gaussian_feature", "Gaussian noise, Protocol A"),
              ("B", "gaussian_feature", "Gaussian noise, Protocol B")]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex=True)
    for i, (d, dlabel) in enumerate(datasets):
        for j, (proto, cond, title) in enumerate(panels):
            ax = axes[i, j]
            src = cells[(cells.condition == cond) & (cells.dataset == d)] if proto == "A" else \
                dep[(dep.modality == cond) & (dep.dataset == d)]
            for m in MODELS:
                g = src[src.model == m].sort_values("eta")
                ax.plot(g.eta, g.mean_acc, color=COLOR[m], marker=MARKER[m], ms=2.5, lw=1, label=NAME[m])
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"{dlabel}\naccuracy", fontsize=7)
            if i == 2:
                ax.set_xlabel("corruption intensity $\\eta$")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(ncol=2, fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_curves.png"), dpi=220)
    plt.close(fig)


def boxes(ax, t, cond, title):
    data = [t[(t.condition == cond) & (t.model == m)].audc_bal.to_numpy() for m in MODELS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops={"color": "black"}, flierprops={"ms": 2})
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COLOR[m])
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, 7), [NAME[m] for m in MODELS], rotation=60, fontsize=6.5)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def fig_overall():
    ta = pd.read_csv(os.path.join(REV, "trajectories_revision.csv"))
    tb = pd.read_csv(os.path.join(REV, "trajectories_protocol_b.csv"))
    fig, axes = plt.subplots(2, 5, figsize=(7.4, 4.6), sharey=True)
    for j, (c, label) in enumerate(COND_A.items()):
        boxes(axes[0, j], ta, c, label)
    for j, (c, label) in enumerate(COND_B.items()):
        if c in set(tb.condition):
            boxes(axes[1, j], tb, c, label)
    axes[0, 0].set_ylabel("Protocol A: balanced AUDC")
    axes[1, 0].set_ylabel("Protocol B: balanced AUDC")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_overall.png"), dpi=220)
    plt.close(fig)


def rank_panel(ax, fam, metric, title, show_labels):
    for i, cond in enumerate(COND_A):
        r = fam[(fam.metric == metric) & (fam.condition == cond)].iloc[0]
        ranks = eval(r["ranks"]) if isinstance(r["ranks"], str) else r["ranks"]
        best = min(ranks.values())
        ax.plot([best, best + CD], [i, i], color="black", lw=2.5, alpha=0.25, solid_capstyle="butt")
        for m in MODELS:
            ax.scatter(ranks[m], i, color=COLOR[m], marker=MARKER[m], s=22, zorder=3, label=NAME[m] if i == 0 else None)
        p = r["p_holm_confirmatory"] if metric in ("audc_bal", "audc", "mrr") else r["p_holm_all"]
        ax.text(6.6, i, f"p={p:.1e}" if p < 1e-3 else f"p={p:.3f}", va="center", fontsize=6)
    ax.set_yticks(range(len(COND_A)), list(COND_A.values()) if show_labels else [""] * len(COND_A))
    ax.set_xlim(0.8, 8.3)
    ax.set_ylim(-0.6, len(COND_A) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("mean Friedman rank (1 = best)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)


def fig_ranking():
    fam = pd.read_csv(os.path.join(REV, "confirmatory_family.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))
    panels = [("mrr", "(a) MRR"), ("audc", "(b) AUDC"), ("mrr_bal", "(c) Balanced MRR"),
              ("audc_bal", "(d) Balanced AUDC (primary)")]
    for ax, (metric, title), lab in zip(axes.ravel(), panels, [True, False, True, False]):
        rank_panel(ax, fam, metric, title, lab)
    axes[0, 0].legend(ncol=6, fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 1.32))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ranking.png"), dpi=220)
    plt.close(fig)


def fig_knee():
    kv = json.load(open(os.path.join(REV, "knee_validation.json")))["detail"]
    rev = json.load(open(os.path.join(REV, "revision.json")))
    ta = pd.read_csv(os.path.join(REV, "trajectories_revision.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    sds = [0.0, 0.0025, 0.005, 0.01, 0.02]
    for est, ls, lab in [("spline", "-", "spline $\\eta_{\\mathrm{knee}}$"), ("discrete", "--", "second difference")]:
        for knee, col in [("early", "#1f77b4"), ("middle", "#ff7f0e"), ("late", "#2ca02c")]:
            vals = [kv[f"{knee}_knee"][f"sd_{sd}"][est]["within_one_tier"] for sd in sds]
            axes[0].plot(sds, vals, ls=ls, color=col, marker="o", ms=3, label=f"{lab}, {knee}")
    axes[0].set(xlabel="noise SD per tier", ylabel="recovered within one tier", ylim=(0, 1.05),
                title="(a) Synthetic known knees")
    axes[0].legend(fontsize=5.2)
    ks = rev["knee_seeds"]
    names = [("audc_bal", "AUDC$_{bal}$"), ("mrr", "MRR"), ("eta_discrete", "2nd diff."),
             ("eta_knee_tier", "$\\eta_{knee}$ (tier)"), ("eta_knee", "$\\eta_{knee}$")]
    axes[1].bar(range(len(names)), [ks[k]["spearman_min"] for k, _ in names], color="#4c72b0")
    axes[1].set_xticks(range(len(names)), [n for _, n in names], rotation=30, fontsize=6.5)
    axes[1].set(ylabel="min. seed-to-seed Spearman", ylim=(0, 1.05), title="(b) Replicability across seeds")
    data = [ta[ta.condition == c].eta_knee_tier.dropna() for c in COND_A]
    axes[2].boxplot(data, widths=0.6, medianprops={"color": "black"})
    axes[2].set_xticks(range(1, 6), ["Sym.", "Cl.-cond.", "Gauss.", "MCAR imp.", "MCAR nat."], rotation=30, fontsize=6.5)
    axes[2].set(ylabel="$\\eta_{\\mathrm{knee}}$ (tier resolution)", title="(c) Population distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_knee.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    fig_ranking()
    fig_knee()
    if os.path.exists(os.path.join(REV, "trajectories_protocol_b.csv")):
        fig_curves()
        fig_overall()
