"""Generates every numeric LaTeX table of the manuscript from the audit outputs,
so that no figure in the paper is typed by hand. Output: results/analysis/latex/."""

import json
from decimal import ROUND_HALF_UP, Decimal
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import BENCHMARK_SUITE

AUDIT = "results/analysis/audit"
OUT = "results/analysis/latex"
BENCH = "results/metrics/benchmark_results.csv"
FLAT_KAPPA = 1e-9

MODELS = ["cart", "randomforest", "xgboost", "lightgbm", "catboost", "mlp"]
MODEL_TEX = {"cart": "CART", "randomforest": "RF", "xgboost": "XGBoost",
             "lightgbm": "LightGBM", "catboost": "CatBoost", "mlp": "MLP"}
MODALITIES = ["symmetric_label", "asymmetric_label", "gaussian_feature", "mcar_missing"]
MOD_TEX = {"symmetric_label": "Symmetric label", "asymmetric_label": "Pair-flip label",
           "gaussian_feature": "Gaussian feature", "mcar_missing": "MCAR + median imp."}
CONDITIONS = ["symmetric_label", "class_conditional", "gaussian_feature", "mcar_imputed", "mcar_native"]
COND_TEX = {"symmetric_label": "Symmetric label", "class_conditional": "Class-cond.\\ label",
            "gaussian_feature": "Gaussian feature", "mcar_imputed": "MCAR, imputed", "mcar_native": "MCAR, native"}
PRIMARY = False
WORDS = {4: "four", 5: "five"}
GROUPS = {
    "Science \\& sensing": ["Bioresponse", "qsar-biodeg", "eye_movements", "gesture_phase", "ozone_level"],
    "Industrial \\& physical": ["steel-plates-fault", "vehicle", "electricity", "pol", "miniboone"],
    "Business \\& web": ["bank-marketing", "churn", "amazon_access", "nomao", "spambase"],
    "Synthetic, game \\& perceptual": ["madelon", "hill_valley", "jungle_chess", "phoneme", "volkert"],
}
DATASETS = [d for g in GROUPS.values() for d in g]



def _rho_range(r):
    """Format a Spearman range, collapsing it when both ends round alike."""
    lo, hi = f"{r['spearman_min']:.2f}", f"{r['spearman_max']:.2f}"
    return lo if lo == hi else f"{lo}--{hi}"

def r2(x):
    """Two-decimal rounding, half away from zero: mean ranks over 20 datasets are exact
    multiples of 1/40, so binary round-half-even would print 2.125 as 2.12."""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def tex(name):
    return name.replace("_", "\\_")


def pfmt(p):
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}"


def write(name, body):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(body)


def inventory(audit):
    inv = audit["inventory"]
    rows = []
    for grp, ds in GROUPS.items():
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{grp}}}}} \\\\")
        for d in ds:
            v = inv[d]
            rows.append(f"{tex(d)} & {BENCHMARK_SUITE[d]} & {v['N']:,} & {v['N_eff']:,} & "
                        f"{v['D']} & {v['K']} & {v['majority_rate']:.2f} \\\\")
    write("tab_inventory.tex", "\n".join([
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Benchmark corpus as loaded from the cached OpenML matrices. $N$ is the original sample size, "
        "$N_{\\mathrm{eff}}$ the size after the one-off stratified subsample to $10{,}000$ rows, $D$ the number of "
        "columns after encoding, $K$ the number of classes and $\\pi_{\\max}$ the majority-class rate.}",
        "\\label{tab:inventory}",
        "\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
        "Dataset & OpenML ID & $N$ & $N_{\\mathrm{eff}}$ & $D$ & $K$ & $\\pi_{\\max}$ \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))


def friedman(fried, audit):
    rows = []
    for metric, label in [("mrr", "MRR"), ("audc", "AUDC"), ("audc_bal_mrr", "MRR$_{\\mathrm{bal}}$")]:
        sub = fried[fried.metric == metric].set_index("modality")
        for i, mod in enumerate(MODALITIES):
            r = sub.loc[mod]
            ranks = audit["mean_ranks"][f"{metric}|{mod}"]
            best = min(ranks, key=ranks.get)
            worst = max(ranks, key=ranks.get)
            rows.append(f"{label if i == 0 else ''} & {MOD_TEX[mod]} & {r.chi2_F:.2f} & {r.F_ID:.2f} & "
                        f"{pfmt(r.p_ID)} & {pfmt(r.p_ID_holm_across_modalities)} & "
                        f"{MODEL_TEX[best]} ({r2(ranks[best])}) & {MODEL_TEX[worst]} ({r2(ranks[worst])}) \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    write("tab_friedman.tex", "\n".join([
        "\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
        ("\\caption{Friedman omnibus tests on seed-averaged trajectories over $N=20$ datasets and $k=6$ models ($\\chi^2_F$ on $5$ degrees of freedom; " if PRIMARY else "\\caption{Friedman omnibus tests over $N=20$ datasets and $k=6$ models ($\\chi^2_F$ on $5$ degrees of freedom; ")
        + f"Iman--Davenport $F_F$ on $(5,95)$). $p_{{\\mathrm{{Holm}}}}$ adjusts the {WORDS[len(MODALITIES)]} {'condition' if PRIMARY else 'modality'}-wise tests of each metric. "
        "The Nemenyi critical difference is $\\mathrm{CD}=1.686$ for every row. Mean ranks of the best and worst model "
        "are in parentheses (1 = best).}",
        "\\label{tab:friedman}",
        "\\begin{tabular}{@{}llrrllll@{}}", "\\toprule",
        "Metric & Condition & $\\chi^2_F$ & $F_F$ & $p$ & $p_{\\mathrm{Holm}}$ & Best (rank) & Worst (rank) \\\\",
        "\\midrule", *rows, "\\end{tabular}", "\\end{table*}", ""]))


def ranks_table(audit):
    rows = []
    for metric, label in [("mrr", "MRR"), ("audc", "AUDC")]:
        for i, mod in enumerate(MODALITIES):
            r = audit["mean_ranks"][f"{metric}|{mod}"]
            best = min(r.values())
            cells = [f"\\textbf{{{r2(r[m])}}}" if r[m] == best else r2(r[m]) for m in MODELS]
            rows.append(f"{label if i == 0 else ''} & {MOD_TEX[mod]} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    head = " & ".join(MODEL_TEX[m] for m in MODELS)
    write("tab_ranks.tex", "\n".join([
        "\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
        ("\\caption{Mean Friedman ranks of the seed-averaged trajectories over the 20 datasets (1 = best, 6 = worst; bold = best in row). Two models " if PRIMARY else "\\caption{Mean Friedman ranks over the 20 datasets (1 = best, 6 = worst; bold = best in row). Two models ")
        + "differ significantly at $\\alpha=0.05$ under Nemenyi when their ranks differ by more than $1.686$.}",
        "\\label{tab:ranks}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll" + "r" * 6 + "@{}}", "\\toprule",
        f"Metric & Condition & {head} \\\\", "\\midrule", *rows, "\\end{tabular*}", "\\end{table*}", ""]))


def ablation(audit):
    a = audit["ablations"]
    edge = json.load(open(os.path.join(AUDIT, "fix_extras.json")))["ablation_edge"] if PRIMARY else None
    keys = {"legacy_univariate_vs_natural": "legacy", "notaknot_vs_natural": "notaknot",
            "coarse6_vs_natural": "coarse6", "smoothing_gcv_vs_natural": "smoothing_gcv"}
    rows = []
    for key, label in [("legacy_univariate_vs_natural", "Smoothing spline, $s=10^{-3}$, $\\arg\\max\\lvert S''\\rvert$ (legacy)"),
                       ("notaknot_vs_natural", "Interpolating cubic, not-a-knot ends"),
                       ("coarse6_vs_natural", "Natural cubic on 6 tiers ($\\Delta\\eta=0.10$)"),
                       ("smoothing_gcv_vs_natural", "Penalised cubic smoothing spline, GCV")]:
        v = a[key]
        rows.append(f"{label} & {v['median_abs_diff']:.3f} & {100 * v['within_0p05']:.1f} & "
                    f"{v['spearman']:.2f} & {100 * (edge[keys[key]] if edge else v['edge_frac']):.1f} \\\\")
    if edge:
        tail = (f"whose estimates are interior (Supplementary Proposition~S1): {edge['reference_count']} of the "
                f"{edge['n_defined']} defined knees lie within $0.01$ of the boundary. ``Edge'' is the share of "
                "defined knees within $0.01$ of $0$ or $0.5$; trajectories with $\\kappa\\equiv0$ have no "
                "knee and are left out. The 500-point grid is a numerical search grid, not experimental resolution (11 tiers); read $\\eta_{\\mathrm{knee}}$ at tier level, as a secondary population-level descriptor.}")
    else:
        tail = (f"whose estimates sit on the boundary in {100 * a['natural_edge_frac']:.1f}\\% of trajectories. "
                "``Edge'' counts estimates within $0.01$ of $0$ or $0.5$.}")
    write("tab_ablation.tex", "\n".join([
        "\\begin{table}[t]", "\\centering", "\\small",
        f"\\caption{{Sensitivity of $\\eta_{{\\mathrm{{knee}}}}$ to the estimator, over all {len(MODALITIES) * 120} {'seed-averaged ' if PRIMARY else ''}trajectories. Each row is compared with the "
        "reference estimator (interpolating natural cubic spline, 11 tiers, $\\arg\\max\\kappa$ on a 500-point grid), " + tail,
        "\\label{tab:ablation}",
        "\\begin{tabular}{@{}p{4.3cm}rrrr@{}}", "\\toprule",
        "Estimator & med.\\ $\\lvert\\Delta\\eta_{\\mathrm{knee}}\\rvert$ & $\\le 0.05$ (\\%) & Spearman & Edge (\\%) \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))


TIER_STEP = 0.05  # spacing of the eleven experimental intensity tiers


def tier(value):
    """Nearest experimental tier: the 500-point search grid is numerical, not experimental resolution."""
    return round(value / TIER_STEP) * TIER_STEP


def trajectory_tables(t):
    blocks = []
    for d in DATASETS:
        sub = t[t.dataset == d]
        rows = []
        for mod in MODALITIES:
            for j, m in enumerate(MODELS):
                r = sub[(sub.modality == mod) & (sub.model == m)].iloc[0]
                if r.kappa_max < FLAT_KAPPA:
                    eta = "flat"
                else:
                    eta = f"{tier(r.eta_star):.2f} [{tier(r.eta_boot_lo):.2f}, {tier(r.eta_boot_hi):.2f}]"
                rows.append(f"{MOD_TEX[mod] if j == 0 else ''} & {MODEL_TEX[m]} & {r.acc0:.3f} & {r.acc_end:.3f} & "
                            f"{r.audc:.3f} & {r.mrr:.3f} & {eta} \\\\")
            rows.append("\\midrule")
        rows[-1] = "\\bottomrule"
        blocks.append("\n".join([
            "\\begin{table}[p]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3.5pt}",
            f"\\caption{{Trajectory summaries for \\texttt{{{tex(d)}}}" + (", accuracy averaged over seeds 42, 43 and 44" if PRIMARY else "")
            + ". $\\eta_{\\mathrm{knee}}$ and its 95\\% parametric-bootstrap interval ($B=200$" + ("; the standard error of each tier is $\\max(\\mathrm{SD}_{\\mathrm{seed}},\\,\\mathrm{SD}_{\\mathrm{fold}}/2)/\\sqrt3$" if PRIMARY else "")
            + "); ``flat'' marks a trajectory with $\\kappa\\equiv 0$, for which $\\eta_{\\mathrm{knee}}$ is undefined. Values are rounded to the nearest of the eleven experimental tiers (spacing $0.05$); the knee is a coarse, secondary population-level descriptor.}",
            f"\\label{{tab:traj_{d.replace('-', '_')}}}",
            "\\begin{tabular}{@{}llrrrrl@{}}", "\\toprule",
            "Condition & Model & $\\mathrm{Acc}(0)$ & $\\mathrm{Acc}(0.5)$ & AUDC & MRR & $\\eta_{\\mathrm{knee}}$ [95\\% CI] \\\\",
            "\\midrule", *rows, "\\end{tabular}", "\\end{table}", ""]))
    write("tab_trajectories.tex", "\n".join(blocks))


def accuracy_grids(df):
    etas = sorted(df["eta"].unique())
    blocks = []
    for d in DATASETS:
        sub = df[df.dataset == d]
        rows = []
        for mod in MODALITIES:
            for j, m in enumerate(MODELS):
                g = sub[(sub.modality == mod) & (sub.model == m)].sort_values("eta")
                vals = " & ".join(f"{100 * v:.1f}" for v in g["mean_acc"])
                rows.append(f"{MOD_TEX[mod].replace('\\\\ ', ' ') if j == 0 else ''} & {MODEL_TEX[m]} & {vals} \\\\")
            rows.append("\\midrule")
        rows[-1] = "\\bottomrule"
        head = " & ".join(f"{e:.2f}" for e in etas)
        blocks.append("\n".join([
            "\\begin{table}[p]", "\\centering", "\\tiny", "\\setlength{\\tabcolsep}{2.6pt}",
            (f"\\caption{{Accuracy (\\%) of every cell for \\texttt{{{tex(d)}}} by intensity $\\eta$: five-fold mean, averaged over seeds 42, 43 and 44.}}" if PRIMARY else f"\\caption{{Five-fold mean accuracy (\\%) of every cell for \\texttt{{{tex(d)}}}, by noise tier $\\eta$.}}"),
            f"\\label{{tab:grid_{d.replace('-', '_')}}}",
            "\\begin{tabular}{@{}ll" + "r" * len(etas) + "@{}}", "\\toprule",
            f"Condition & Model & {head} \\\\", "\\midrule", *rows, "\\end{tabular}", "\\end{table}", ""]))
    write("tab_accuracy_grids.tex", "\n".join(blocks))


def wilcoxon(wil):
    rows = []
    for metric, label in [("mrr", "MRR"), ("audc", "AUDC")]:
        for mod in MODALITIES:
            sub = wil[(wil.metric == metric) & (wil.modality == mod)]
            rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{label}, {MOD_TEX[mod]}}}}} \\\\")
            for r in sub.itertuples():
                agree = "" if r.wilcoxon_sig == r.nemenyi_sig else "$\\dagger$"
                rows.append(f"{MODEL_TEX[r.a]}--{MODEL_TEX[r.b]} & "
                            f"${r.median_diff:+.4f}$ & {pfmt(r.p_raw)} & {pfmt(r.p_holm)} & "
                            f"{'yes' if r.wilcoxon_sig else 'no'}{agree} \\\\")
            rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    write("tab_wilcoxon.tex", "\n".join([
        "{\\scriptsize", "\\begin{longtable}{@{}lrrrl@{}}",
        "\\caption{Pairwise Wilcoxon signed-rank tests (Pratt zero handling) over the 20 datasets, with Holm "
        "adjustment over the 15 pairs of each block. $\\dagger$ marks disagreement with the Nemenyi verdict.}"
        "\\label{tab:wilcoxon}\\\\",
        "\\toprule", "Pair & Median diff. & $p$ & $p_{\\mathrm{Holm}}$ & Sig. \\\\", "\\midrule",
        "\\endfirsthead", "\\toprule", "Pair & Median diff. & $p$ & $p_{\\mathrm{Holm}}$ & Sig. \\\\",
        "\\midrule", "\\endhead", *rows, "\\end{longtable}", "}", ""]))


def endurance_matrices(t):
    blocks = []
    for metric, label, scale in [("mrr", "MRR (\\%)", 100), ("audc", "AUDC (\\%)", 100)]:
        m = t.groupby(["dataset", "model"])[metric].mean().unstack()[MODELS] * scale
        rows = []
        for grp, ds in GROUPS.items():
            rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{grp}}}}} \\\\")
            for d in ds:
                best = m.loc[d].max()
                cells = [f"\\textbf{{{v:.1f}}}" if np.isclose(v, best) else f"{v:.1f}" for v in m.loc[d]]
                rows.append(f"{tex(d)} & " + " & ".join(cells) + " \\\\")
        mean = m.mean()
        cells = [f"\\textbf{{{v:.1f}}}" if np.isclose(v, mean.max()) else f"{v:.1f}" for v in mean]
        rows += ["\\midrule", "Mean over datasets & " + " & ".join(cells) + " \\\\"]
        head = " & ".join(MODEL_TEX[x] for x in MODELS)
        blocks.append("\n".join([
            "\\begin{table}[p]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
            (f"\\caption{{Per-dataset {label.split()[0]} of the seed-averaged trajectories, averaged over the five conditions (bold = best in row). "
             "The average is descriptive only: the conditions sit on different intensity scales "
             "(Section~3.2 of the main article), and the two masking conditions share the MLP, which cannot take missing values. "
             "Per-condition values are in Supplementary Section~S5.}"
             if PRIMARY else
             f"\\caption{{Per-dataset {label.split()[0]}, averaged over the four modalities (bold = best in row). "
             "The average is descriptive only: the modalities sit on different intensity scales "
             "(Section~3.2 of the main article), and on binary datasets the two label modalities are one corruption law "
             "(Proposition~A2 of the main article), which therefore enters the average twice. Per-modality values are in "
             "Supplementary Section~S5.}"),
            f"\\label{{tab:endurance_{metric}}}",
            "\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule", f"Dataset & {head} \\\\", "\\midrule",
            *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))
    write("tab_endurance.tex", "\n".join(blocks))


def macro_figure(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    d = t.copy()
    d.loc[d.kappa_max < FLAT_KAPPA, "eta_star"] = np.nan
    d["Model"] = d["model"].map(MODEL_TEX)
    d["Modality"] = d["modality"].map(MOD_TEX)
    order = [MODEL_TEX[m] for m in MODELS]
    hue_order = [MOD_TEX[m] for m in MODALITIES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, title in [(axes[0], "mrr", "Marginal robustness retention (MRR)"),
                           (axes[1], "eta_star", r"Maximum-curvature knee $\eta_{\mathrm{knee}}$")]:
        sns.boxplot(data=d, x="Model", y=col, hue="Modality", order=order, hue_order=hue_order,
                    ax=ax, palette="Set2", fliersize=2, linewidth=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(col.upper() if col == "mrr" else r"$\eta_{\mathrm{knee}}$")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].legend_.remove()
    plt.tight_layout()
    plt.savefig("springer_project/figures/macro_breakdown_profile.png", dpi=300, bbox_inches="tight")
    plt.close()


ATLAS_GROUPS = [("g1", "chemistry, biology, atmospheric and human-motion sensing", "Science \\& sensing"),
                ("g2", "industrial, physical and network measurements", "Industrial \\& physical"),
                ("g3", "business, marketing and web data", "Business \\& web"),
                ("g4", "synthetic, game, speech and image-derived tasks", "Synthetic, game \\& perceptual")]


ATLAS_COND = {"symmetric_label": "symmetric label noise", "class_conditional": "class-conditional label noise",
              "gaussian_feature": "Gaussian feature noise", "mcar_imputed": "imputed masking",
              "mcar_native": "native masking"}


def atlas_section(t):
    """Section of the paper that embeds all twenty seed-averaged degradation figures. The group
    statistics sit in the figure captions, so that the short text of the section does not leave a
    near-empty page in front of the float pages; every number is computed here."""
    refs = [f"{title.lower()} (Figure~\\ref{{fig:atlas_{gid}}})" for gid, title, _ in ATLAS_GROUPS]
    out = ["\\subsection{The empirical atlas}\\label{sec:atlas}\\label{app:atlas}", "",
           "This section shows the complete degradation trajectory of every benchmark dataset, so that no "
           "dataset-specific behaviour is hidden behind the rank statistics of Section~5 of the main article. Each figure "
           "plots accuracy, averaged over the three seeds, against intensity for the six model families, with one "
           "panel per corruption condition. The shaded band is $\\pm1$ sample standard deviation across the three seeds, the "
           "dashed horizontal line is the majority-class rate $\\pi_{\\max}$ (the accuracy of a constant prediction) and, "
           "where it lies clearly below $\\pi_{\\max}$, the dotted line is the uniform-guess accuracy $1/K$; the sixth "
           "panel gives the legend and the size of the dataset. The datasets are grouped by application domain: "
           f"{', '.join(refs[:-1])} and {refs[-1]}. The caption of each group gives the median MRR of its "
           "$5\\times 6=30$ trajectories under each condition and the range of mean AUDC over the six families. "
           "Conditions sit on different intensity scales (Section~3.2 of the main article), so these numbers are "
           "descriptive and do not rank corruption types.", ""]
    for gid, title, key in ATLAS_GROUPS:
        ds = GROUPS[key]
        g = t[t.dataset.isin(ds)]
        med = g.groupby("modality").mrr.median()
        aud = g.groupby("model").audc.mean()
        parts = [f"${med[c]:.3f}$ ({ATLAS_COND[c]})" for c in CONDITIONS]
        stats_text = (f" Median MRR of the group: {', '.join(parts[:-1])} and {parts[-1]}; mean AUDC over its datasets and all five conditions from "
                      f"${aud.min():.3f}$ ({MODEL_TEX[aud.idxmin()]}) to ${aud.max():.3f}$ ({MODEL_TEX[aud.idxmax()]}).")
        for chunk, cont in [(c, i > 0) for i, c in enumerate([ds[:2], ds[2:4], ds[4:]]) if c]:
            out += ["\\begin{figure}[tbp]", "\\ContinuedFloat" if cont else "", "\\centering"]
            for i, d in enumerate(chunk):
                out += ["\\begin{subfigure}{\\linewidth}", "  \\centering",
                        f"  \\includegraphics[width=\\linewidth]{{curves_{d}.png}}",
                        f"  \\caption{{{tex(d)}}}\\label{{fig:atlas_{d}}}",
                        "\\end{subfigure}" + ("\\\\[4pt]" if i < len(chunk) - 1 else "")]
            cap = f"Degradation atlas for {title}" + (" (continued)." if cont else "." + stats_text) + " Band: $\\pm1$ sample SD across three seeds; dashed line: $\\pi_{\\max}$."
            out += [f"\\caption{{{cap}}}"] + ([] if cont else [f"\\label{{fig:atlas_{gid}}}"]) + ["\\end{figure}", ""]
    # flush the atlas float pages here, so that the figures of the next section are not queued behind them
    out.append("\\clearpage")
    write("atlas_primary.tex", "\n".join(l for l in out if l is not None) + "\n")


def native_table(extras):
    rows = []
    for m in ["cart", "randomforest", "xgboost", "lightgbm", "catboost"]:
        v = extras["native"][m]
        rows.append(f"{MODEL_TEX[m]} & ${v['median_delta_audc']:+.4f}$ & ${v['median_delta_mrr']:+.4f}$ & "
                    f"{v['native_better']}/{v['n']} & {pfmt(v['p_audc'])} & {pfmt(v['p_audc_holm'])} \\\\")
    write("tab_native.tex", "\n".join([
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Native missing-value handling against median imputation under MCAR masking, on seed-averaged "
        "trajectories. Differences are native minus imputed, as medians over the 20 datasets; ``better'' counts the "
        "datasets on which native handling gives the higher AUDC. $p$: Wilcoxon signed-rank test on AUDC, "
        "$p_{\\mathrm{Holm}}$ adjusted over the five families.}",
        "\\label{tab:native}",
        "\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
        "Model & $\\Delta$AUDC & $\\Delta$MRR & Better & $p$ & $p_{\\mathrm{Holm}}$ \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]))


def budget_table(extras):
    b = extras["budget"]
    lib = {"catboost": "CatBoost (1{,}000 rounds, automatic rate)", "xgboost": "XGBoost (rate 0.3)"}
    rows = []
    for m in ["catboost", "xgboost"]:
        for i, mod in enumerate(MODALITIES_ORIGINAL):
            v = b["per_library"][f"{m}|{mod}"]
            rows.append(f"{lib[m] if i == 0 else ''} & {MOD_TEX[mod]} & ${v['median_delta_acc0']:+.4f}$ & "
                        f"${v['median_delta_audc']:+.4f}$ & ${v['median_delta_mrr']:+.4f}$ & "
                        f"{v['default_better_audc']}/20 & {pfmt(v['p_audc_holm'])} \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    rank_rows = []
    for metric, label in [("mrr", "MRR"), ("audc", "AUDC")]:
        for i, mod in enumerate(MODALITIES_ORIGINAL):
            r = b["rankings"][f"{metric}|{mod}"]
            rank_rows.append(f"{label if i == 0 else ''} & {MOD_TEX[mod]} & {MODEL_TEX[r['best_matched']]} & "
                             f"{MODEL_TEX[r['best_defaults']]} & {MODEL_TEX[r['worst_defaults']]} & "
                             f"{r['kendall_tau']:.2f} & {pfmt(r['p_ID_defaults'])} \\\\")
        rank_rows.append("\\midrule")
    rank_rows[-1] = "\\bottomrule"
    write("tab_budget.tex", "\n".join([
        "\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\caption{Matched budget against library defaults, seed 42, under symmetric label noise, pair-flip label noise "
        "applied to all twenty datasets, Gaussian feature noise and median-imputed masking. Top: effect of switching "
        "one library to its shipped defaults (default minus matched, medians over the 20 datasets; ``better'' counts "
        "datasets with higher AUDC; Wilcoxon $p$, Holm-adjusted over the eight tests). LightGBM, random forest and CART "
        "already run at their defaults in the matched grid. Bottom: Friedman leaders with every library at its "
        "defaults, Kendall $\\tau$ between the matched and the default mean-rank vectors, and the Iman--Davenport $p$ "
        "of the default configuration.}",
        "\\label{tab:budget}",
        "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
        "Library & Operator & $\\Delta\\mathrm{Acc}(0)$ & $\\Delta$AUDC & $\\Delta$MRR & Better & $p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule", *rows, "\\end{tabular}", "",
        "\\vspace{6pt}",
        "\\begin{tabular}{@{}llllrrr@{}}", "\\toprule",
        "Metric & Operator & Best (matched) & Best (defaults) & Worst (defaults) & $\\tau$ & $p$ \\\\",
        "\\midrule", *rank_rows, "\\end{tabular}", "\\end{table*}", ""]))


def seeds_table(extras):
    sd = extras["seeds"]
    rows = []
    for metric, label in [("mrr", "MRR"), ("audc", "AUDC")]:
        for i, cond in enumerate(MODALITIES):
            v = sd["per_condition"][f"{metric}|{cond}"]
            best = ", ".join(sorted({MODEL_TEX[b] for b in v["best_per_seed"]}))
            worst = ", ".join(sorted({MODEL_TEX[b] for b in v["worst_per_seed"]}))
            rows.append(f"{label if i == 0 else ''} & {MOD_TEX[cond]} & {pfmt(v['max_p_ID'])} & "
                        f"{v['kendall_tau_min']:.2f} & {best} & {worst} \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    rel = sd["reliability"]
    write("tab_seeds.tex", "\n".join([
        "\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Replication over seeds 42, 43 and 44 (split, corruption and model initialisation all change). "
        "Largest Iman--Davenport $p$ over the three seeds; smallest Kendall $\\tau$ between the three seeds' "
        "mean-rank vectors; the models ranked best and worst in any seed. Seed-to-seed Spearman correlation of the "
        f"{rel['eta_star']['n_trajectories']} trajectory summaries: "
        f"MRR {_rho_range(rel['mrr'])}, AUDC {_rho_range(rel['audc'])}, "
        f"$\\eta_{{\\mathrm{{knee}}}}$ {_rho_range(rel['eta_star'])}.}}",
        "\\label{tab:seeds}",
        "\\begin{tabular}{@{}llrrll@{}}", "\\toprule",
        "Metric & Condition & max $p$ & min $\\tau$ & Best in any seed & Worst in any seed \\\\", "\\midrule",
        *rows, "\\end{tabular}", "\\end{table*}", ""]))


def main():
    global AUDIT, OUT, MODALITIES, MOD_TEX, PRIMARY, MODALITIES_ORIGINAL
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", action="store_true", help="tables for the seed-averaged five-condition grid")
    args = ap.parse_args()
    MODALITIES_ORIGINAL = list(MODALITIES)
    original_tex = dict(MOD_TEX)
    if args.primary:
        PRIMARY = True
        AUDIT, OUT = "results/analysis/audit_primary", "results/analysis/latex_primary"
        MODALITIES, MOD_TEX = CONDITIONS, {**original_tex, **COND_TEX}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(AUDIT, "audit_numbers.json")) as fh:
        audit = json.load(fh)
    t = pd.read_csv(os.path.join(AUDIT, "trajectory_audit.csv"))
    inventory(audit)
    friedman(pd.read_csv(os.path.join(AUDIT, "friedman.csv")), audit)
    ranks_table(audit)
    ablation(audit)
    trajectory_tables(t)
    wilcoxon(pd.read_csv(os.path.join(AUDIT, "wilcoxon_holm.csv")))
    endurance_matrices(t)
    if args.primary:
        cells = pd.read_csv("results/analysis/primary/cells.csv").rename(columns={"condition": "modality"})
        accuracy_grids(cells)
        extras = json.load(open(os.path.join(AUDIT, "extras.json")))
        atlas_section(t)
        native_table(extras)
        if extras.get("budget") and extras["budget"].get("complete"):
            budget_table(extras)
        if extras.get("seeds"):
            seeds_table(extras)
    else:
        accuracy_grids(pd.read_csv(BENCH))
        macro_figure(t)
    print(sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
