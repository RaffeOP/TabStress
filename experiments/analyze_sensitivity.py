"""Sensitivity runs and two extra statistics for the supplement.

1. TabICL settings check: default settings (8 estimators, no row cap) against the main run (4 estimators, 3,000-row
   cap) on the datasets of the default run, seed 42, Protocol B.
2. Compound corruption (label noise or Gaussian noise together with MCAR missingness at the same intensity),
   Protocol A, against the single-operator results of the same datasets and seeds.
3. Bayesian signed-rank test (Benavoli et al., 2017) between the best and the second-best family on balanced AUDC.
4. Bootstrap confidence intervals over datasets for the median TabICL minus best-classical MRR gap.

Writes results/analysis/revision/sensitivity_extra.json and tab_sensitivity_extra.tex."""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_foundation as af                                         # noqa: E402
import analyze_revision as ar                                           # noqa: E402

SEED_SETTINGS = 42
COMPOUND_SEEDS = [42, 43, 44]
COMPOUND_PARTS = {"label_mcar": ("symmetric_label", "mcar_imputed"),
                  "gauss_mcar": ("gaussian_feature", "mcar_imputed")}
COMPOUND_LABEL = {"label_mcar": "Label noise + MCAR", "gauss_mcar": "Gaussian + MCAR"}
SINGLE_LABEL = {"symmetric_label": "label noise", "gaussian_feature": "Gaussian", "mcar_imputed": "MCAR"}
RANK_SCORES = ["mrr", "audc_bal"]
ROPE = 0.01
PRIOR_STRENGTH = 0.5
N_MONTE_CARLO = 20000
N_BOOTSTRAP = 10000
RNG_SEED = 20240611
CLASSICAL = af.CLASSICAL


def mean_ranks(traj, models, metric, cond):
    pv = traj[(traj.condition == cond) & traj.model.isin(models)].pivot(index="dataset", columns="model",
                                                                          values=metric)[models]
    R, _ = ar.friedman_block(pv)
    return {m: float(R[m]) for m in models}


def tabicl_settings(bases):
    default = pd.read_csv(os.path.join(ar.REV_METRICS, f"foundation_default__seed{SEED_SETTINGS}.csv"))
    main = ar.load_block("foundation")
    datasets = sorted(default.dataset.unique())
    main = main[(main.seed == SEED_SETTINGS) & main.dataset.isin(datasets) & main.modality.isin(af.OPERATORS)]
    main = main.drop_duplicates(["seed", "dataset", "model", "modality", "eta"], keep="last")
    dep = ar.load_block("deploy")
    dep = dep[(dep.seed == SEED_SETTINGS) & (dep.pipeline == "imputed") & dep.model.isin(CLASSICAL)
              & dep.modality.isin(af.OPERATORS) & dep.dataset.isin(datasets)]
    dep = dep.drop_duplicates(["seed", "dataset", "model", "modality", "eta"], keep="last")
    trajs = {}
    for name, fnd in (("main", main), ("default", default)):
        raw = ar.deploy_conditions(pd.concat([dep, fnd], ignore_index=True))
        raw = raw[raw.condition.isin(af.CONDITIONS)]
        trajs[name] = ar.trajectories(ar.seed_mean(raw, ["condition", "dataset", "model"]),
                                      ["condition", "dataset", "model"], bases)
    clean = {n: t[(t.condition == "gaussian_feature") & (t.model == af.FOUNDATION)].set_index("dataset").acc0
             for n, t in trajs.items()}
    diff = clean["default"] - clean["main"]
    out = {"seed": SEED_SETTINGS, "datasets": datasets, "n_datasets": len(datasets),
           "clean": {"mean_main": float(clean["main"].mean()), "mean_default": float(clean["default"].mean()),
                     "mean_diff": float(diff.mean()), "max_abs_diff": float(diff.abs().max()),
                     "n_default_higher": int((diff > 1e-12).sum()),
                     "per_dataset": {d: [float(clean["main"][d]), float(clean["default"][d])] for d in datasets}},
           "ranks": {}}
    for metric in RANK_SCORES:
        for cond in af.CONDITIONS:
            rec = {}
            for name, t in trajs.items():
                r = mean_ranks(t, af.POOL, metric, cond)
                rec[name] = dict(tabicl_rank=r[af.FOUNDATION], position=int(1 + sum(v < r[af.FOUNDATION]
                                                                                    for v in r.values())),
                                 leader=min(r, key=r.get))
            out["ranks"][f"{metric}|{cond}"] = rec
    shifts = [abs(v["default"]["tabicl_rank"] - v["main"]["tabicl_rank"]) for v in out["ranks"].values()]
    out["max_rank_shift"] = float(max(shifts))
    out["n_leader_changes"] = int(sum(v["default"]["leader"] != v["main"]["leader"] for v in out["ranks"].values()))
    out["n_cells"] = len(out["ranks"])
    out["tabicl_last_or_sixth_main"] = int(sum(v["main"]["position"] >= 6 for k, v in out["ranks"].items()
                                              if k.startswith("mrr")))
    out["tabicl_last_or_sixth_default"] = int(sum(v["default"]["position"] >= 6 for k, v in out["ranks"].items()
                                                 if k.startswith("mrr")))
    return out


def compound_trajectories(bases, datasets):
    comp = ar.load_block("compound")
    comp = comp[comp.seed.isin(COMPOUND_SEEDS) & (comp.eta > 0)].drop_duplicates(
        ["seed", "dataset", "model", "modality", "eta"], keep="last")
    dep = ar.load_block("deploy")
    clean = dep[dep.seed.isin(COMPOUND_SEEDS) & (dep.pipeline == "imputed") & (dep.eta == 0)
                & (dep.modality == "gaussian_feature") & dep.dataset.isin(datasets) & dep.model.isin(CLASSICAL)]
    clean = clean.drop_duplicates(["seed", "dataset", "model"], keep="last")
    parts = [pd.concat([g, clean.assign(modality=op)]).assign(condition=op) for op, g in comp.groupby("modality")]
    cells = ar.seed_mean(pd.concat(parts, ignore_index=True), ["condition", "dataset", "model"])
    by_seed = pd.read_csv(os.path.join(ar.PRIMARY, "cells_by_seed.csv"))
    singles = by_seed[by_seed.seed.isin(COMPOUND_SEEDS) & by_seed.dataset.isin(datasets)
                      & by_seed.condition.isin(SINGLE_LABEL)]
    single_cells = ar.seed_mean(singles, ["condition", "dataset", "model"])
    return ar.trajectories(pd.concat([cells, single_cells], ignore_index=True), ["condition", "dataset", "model"],
                           bases)


def compound_analysis(bases):
    comp = ar.load_block("compound")
    datasets = sorted(comp.dataset.unique())
    traj = compound_trajectories(bases, datasets)
    out = {"datasets": datasets, "n_datasets": len(datasets), "seeds": COMPOUND_SEEDS, "ops": {}}
    for op, (a, b) in COMPOUND_PARTS.items():
        rec = {"tau": {}, "ranks": {}, "additivity": {}}
        for metric in RANK_SCORES:
            rc = mean_ranks(traj, CLASSICAL, metric, op)
            rec["ranks"][metric] = rc
            rec["tau"][metric] = {}
            for single in (a, b):
                rs = mean_ranks(traj, CLASSICAL, metric, single)
                rec["tau"][metric][single] = float(stats.kendalltau([rc[m] for m in CLASSICAL],
                                                                    [rs[m] for m in CLASSICAL])[0])
            rec["tau"][metric]["min"] = min(rec["tau"][metric][a], rec["tau"][metric][b])
        idx = ["dataset", "model"]
        tc, ta, tb = (traj[traj.condition == c].set_index(idx) for c in (op, a, b))
        for label, lo, hi in (("acc", "acc0", "audc"), ("bal", "bal0", "audc_bal")):
            loss = lambda t: t[lo] - t[hi]                              # noqa: E731
            d = (loss(tc) - loss(ta) - loss(tb)).dropna()
            rec["additivity"][label] = dict(median=float(d.median()), share_sub_additive=float((d < 0).mean()),
                                            n=int(len(d)), median_compound_loss=float(loss(tc).median()),
                                            median_sum_single_loss=float((loss(ta) + loss(tb)).median()))
        out["ops"][op] = rec
    out["tau_mrr_dominant_min"] = float(min(v["tau"]["mrr"][COMPOUND_PARTS[op][0]] for op, v in out["ops"].items()))
    out["tau_mrr_mcar_min"] = float(min(v["tau"]["mrr"]["mcar_imputed"] for v in out["ops"].values()))
    out["tau_min"] = float(min(v["tau"][m]["min"] for v in out["ops"].values() for m in RANK_SCORES))
    out["median_additivity_acc"] = {op: v["additivity"]["acc"]["median"] for op, v in out["ops"].items()}
    out["additivity_acc_max"] = float(max(out["median_additivity_acc"].values()))
    out["additivity_acc_min"] = float(min(out["median_additivity_acc"].values()))
    out["sub_additive_share_min"] = float(min(v["additivity"]["acc"]["share_sub_additive"]
                                               for v in out["ops"].values()))
    return out


def bayesian_signed_rank(diff, rng):
    """Dirichlet-process Bayesian signed-rank test (Benavoli et al., 2017): posterior probabilities that the median
    difference is below, inside or above the region of practical equivalence."""
    z = np.concatenate([[0.0], np.asarray(diff, float)])
    weights = rng.dirichlet(np.concatenate([[PRIOR_STRENGTH], np.ones(len(z) - 1)]), size=N_MONTE_CARLO)
    walsh = (z[:, None] + z[None, :]) / 2
    masks = [walsh < -ROPE, np.abs(walsh) <= ROPE, walsh > ROPE]
    theta = np.stack([np.einsum("si,ij,sj->s", weights, m.astype(float), weights) for m in masks], axis=1)
    winner = theta.argmax(axis=1)
    return dict(p_left=float(theta[:, 0].mean()), p_rope=float(theta[:, 1].mean()), p_right=float(theta[:, 2].mean()),
                decision_left=float((winner == 0).mean()), decision_rope=float((winner == 1).mean()),
                decision_right=float((winner == 2).mean()))


def bayes_analysis(rng):
    traj = pd.read_csv(os.path.join(ar.OUT, "trajectories_revision.csv"))
    out = {"rope": ROPE, "prior_strength": PRIOR_STRENGTH, "n_samples": N_MONTE_CARLO, "conditions": {}}
    for cond in ar.CONDITIONS:
        pv = traj[traj.condition == cond].pivot(index="dataset", columns="model", values="audc_bal")[ar.MODELS]
        R, _ = ar.friedman_block(pv)
        best, second = R.index[0], R.index[1]
        res = bayesian_signed_rank((pv[best] - pv[second]).to_numpy(), rng)
        res.update(best=best, second=second, rank_gap=float(R.iloc[1] - R.iloc[0]),
                   median_diff=float((pv[best] - pv[second]).median()))
        out["conditions"][cond] = res
    cs = out["conditions"].values()
    out["max_decision_best"] = float(max(c["decision_right"] for c in cs))
    out["n_best_prob_over_95"] = int(sum(c["decision_right"] > 0.95 for c in cs))
    out["n_rope_prob_over_50"] = int(sum(c["p_rope"] > 0.5 for c in cs))
    out["n_rope_decision_largest"] = int(sum(c["decision_rope"] > max(c["decision_right"], c["decision_left"])
                                             for c in cs))
    return out


def gap_intervals(rng):
    traj = pd.read_csv(os.path.join(ar.OUT, "trajectories_foundation.csv"))
    out = {"n_bootstrap": N_BOOTSTRAP, "conditions": {}}
    for cond in af.CONDITIONS:
        pv = traj[traj.condition == cond].pivot(index="dataset", columns="model", values="mrr")
        gap = (pv[af.FOUNDATION] - pv[CLASSICAL].max(axis=1)).to_numpy()
        draws = rng.integers(0, len(gap), size=(N_BOOTSTRAP, len(gap)))
        meds = np.median(gap[draws], axis=1)
        lo, hi = np.percentile(meds, [2.5, 97.5])
        out["conditions"][cond] = dict(median=float(np.median(gap)), lo=float(lo), hi=float(hi), n=int(len(gap)))
    out["all_upper_below_zero"] = bool(all(c["hi"] < 0 for c in out["conditions"].values()))
    out["upper_max"] = float(max(c["hi"] for c in out["conditions"].values()))
    out["lower_min"] = float(min(c["lo"] for c in out["conditions"].values()))
    return out


def signed(x):
    return "0.000" if abs(x) < 0.0005 else f"{x:+.3f}"


def table_settings(s):
    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             "\\caption{TabICL settings check (seed 42, Protocol~B, "
             f"{s['n_datasets']} datasets): main run (4 estimators, in-context set capped at 3,000 rows) against "
             "default settings (8 estimators, no cap). Mean rank of TabICL among the seven models and, in "
             "parentheses, its position.}", "\\label{tab:sens_settings}",
             "\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
             " & & \\multicolumn{2}{c}{MRR} & \\multicolumn{2}{c}{AUDC$_{\\mathrm{bal}}$} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}", "Operator & & Main & Default & Main & Default \\\\", "\\midrule"]
    for cond in af.CONDITIONS:
        cells = []
        for metric in RANK_SCORES:
            for name in ("main", "default"):
                r = s["ranks"][f"{metric}|{cond}"][name]
                cells.append(f"{r['tabicl_rank']:.2f} ({r['position']})")
        lines.append(f"{af.COND_LABEL[cond]} & & " + " & ".join(cells) + " \\\\")
    c = s["clean"]
    lines += ["\\midrule", f"Clean accuracy & & \\multicolumn{{2}}{{c}}{{{c['mean_main']:.3f}}} & "
              f"\\multicolumn{{2}}{{c}}{{{c['mean_default']:.3f}}} \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return lines


def table_compound(c):
    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             "\\caption{Compound corruption under Protocol~A (" f"{c['n_datasets']} datasets, seeds "
             f"{', '.join(map(str, c['seeds']))}, six classical families). $\\tau$: Kendall correlation between the mean-rank "
             "vector under the compound operator and under each component alone. Additivity: median over "
             "dataset--model pairs of the compound AUDC loss (clean value minus AUDC) minus the sum of the two "
             "single-operator losses; negative values are sub-additive.}", "\\label{tab:sens_compound}",
             "\\begin{tabular}{@{}lllrrr@{}}", "\\toprule",
             "Compound & Score & Component & $\\tau$ & Additivity & Sub-additive \\\\", "\\midrule"]
    for op, rec in c["ops"].items():
        a, b = COMPOUND_PARTS[op]
        for k, metric in enumerate(RANK_SCORES):
            for j, single in enumerate((a, b)):
                add = ""
                share = ""
                if k == 0 and j == 0:
                    add = f"${signed(rec['additivity']['acc']['median'])}$"
                    share = f"{100 * rec['additivity']['acc']['share_sub_additive']:.0f}\\%"
                lines.append(f"{COMPOUND_LABEL[op] if (k, j) == (0, 0) else ''} & "
                             f"{'MRR' if metric == 'mrr' else 'AUDC$_{\\mathrm{bal}}$'} & {SINGLE_LABEL[single]} & "
                             f"{rec['tau'][metric][single]:.2f} & {add} & {share} \\\\")
        lines.append("\\midrule")
    lines[-1:] = ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return lines


def table_bayes(b, g):
    label = {"symmetric_label": "Symmetric label", "class_conditional": "Class-conditional",
             "gaussian_feature": "Gaussian", "mcar_imputed": "MCAR (imputed)", "mcar_native": "MCAR (native)"}
    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             f"\\caption{{Bayesian signed-rank test (ROPE $\\pm{ROPE}$ on balanced AUDC) between the best and the "
             "second-best family under each confirmatory condition: posterior probability that the best family is "
             "better (right), practically equivalent (ROPE) or worse (left), and the share of posterior samples in "
             "which the best family wins the decision.}", "\\label{tab:sens_bayes}",
             "\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
             "Condition & Best / second & $P(\\mathrm{best})$ & $P(\\mathrm{ROPE})$ & $P(\\mathrm{second})$ & Wins \\\\",
             "\\midrule"]
    for cond, r in b["conditions"].items():
        lines.append(f"{label[cond]} & {af.NICE[r['best']]} / {af.NICE[r['second']]} & {r['p_right']:.2f} & "
                     f"{r['p_rope']:.2f} & {r['p_left']:.2f} & {r['decision_right']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "",
              "\\begin{table}[t]", "\\centering", "\\footnotesize",
              "\\caption{Median MRR difference between TabICL and the best classical family per dataset, with a "
              f"bootstrap 95\\% interval over datasets ({g['n_bootstrap']:,} resamples).}}".replace(",", "{,}"),
              "\\label{tab:sens_gap}", "\\begin{tabular}{@{}lrr@{}}", "\\toprule",
              "Operator & Median & 95\\% interval \\\\", "\\midrule"]
    for cond, r in g["conditions"].items():
        lines.append(f"{af.COND_LABEL[cond]} & ${signed(r['median'])}$ & $[{signed(r['lo'])},\\,{signed(r['hi'])}]$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return lines


def main():
    rng = np.random.default_rng(RNG_SEED)
    bases = ar.dataset_baselines()
    out = {"tabicl_settings": tabicl_settings(bases), "compound": compound_analysis(bases),
           "bayes": bayes_analysis(rng), "gap_ci": gap_intervals(rng)}
    json.dump(ar.json_safe(out), open(os.path.join(ar.OUT, "sensitivity_extra.json"), "w"), indent=1)
    lines = (table_settings(out["tabicl_settings"]) + [""] + table_compound(out["compound"]) + [""]
             + table_bayes(out["bayes"], out["gap_ci"]))
    open(os.path.join(ar.OUT, "tab_sensitivity_extra.tex"), "w").write("\n".join(lines) + "\n")
    print(json.dumps(ar.json_safe(out), indent=1))


if __name__ == "__main__":
    main()
