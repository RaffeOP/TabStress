"""Forensic audit of the TabStress benchmark: integrity, spline estimators,
Demšar statistics, metric confounds and ablations. Writes
results/analysis/audit/audit_numbers.json plus CSVs for every check."""

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline, UnivariateSpline, make_smoothing_spline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.breakdown_math import compute_critical_breakdown_threshold

BENCH = "results/metrics/benchmark_results.csv"
RAW_DIR = "results/metrics/raw"
OUT = "results/analysis/audit"
GRID = np.round(np.arange(0, 0.501, 0.05), 2)
COARSE = np.round(np.arange(0, 0.501, 0.10), 2)
DENSE_N = 500
BOOT_B = 200
EDGE_TOL = 0.01
ALPHA = 0.05
SEED = 42


def curvature_argmax(x, y, bc="natural"):
    cs = CubicSpline(x, y, bc_type=bc, axis=0)
    xd = np.linspace(x.min(), x.max(), DENSE_N)
    d1, d2 = cs(xd, 1), cs(xd, 2)
    kappa = np.abs(d2) / np.power(1.0 + d1 ** 2, 1.5)
    return xd[np.argmax(kappa, axis=0)], kappa.max(axis=0)


def legacy_eta(x, y):
    sp = UnivariateSpline(x, y, s=0.001, k=3)
    xd = np.linspace(x.min(), x.max(), DENSE_N)
    return float(xd[np.argmax(np.abs(sp.derivative(2)(xd)))])


def smoothing_eta(x, y):
    sp = make_smoothing_spline(x, y)
    xd = np.linspace(x.min(), x.max(), DENSE_N)
    d1, d2 = sp(xd, 1), sp(xd, 2)
    kappa = np.abs(d2) / np.power(1.0 + d1 ** 2, 1.5)
    return float(xd[np.argmax(kappa)]), float(kappa.max())


def at_edge(e):
    return (np.asarray(e) <= EDGE_TOL) | (np.asarray(e) >= 0.5 - EDGE_TOL)


def integrity(df):
    raw = pd.concat(
        [pd.read_csv(os.path.join(RAW_DIR, f)) for f in sorted(os.listdir(RAW_DIR))
         if f.endswith(".csv") and not f.endswith("_backend.csv")],
        ignore_index=True,
    )
    key = ["dataset", "model", "modality", "eta"]
    raw["eta"] = raw["eta"].round(3)
    dup = raw[raw.duplicated(key, keep=False)]
    rerun_dev = float(dup.groupby(key)["mean_acc"].agg(lambda s: s.max() - s.min()).max()) if len(dup) else 0.0
    metric_cols = ["mean_acc", "std_acc", "mean_bal_acc", "mean_f1"]
    expected = len(df["dataset"].unique()) * 6 * 4 * 11
    return {
        "rows": int(len(df)),
        "expected_rows": int(expected),
        "unique_keys": int(df.drop_duplicates(key).shape[0]),
        "nan_cells": int(df[metric_cols].isna().sum().sum()),
        "out_of_unit_interval": int(((df[metric_cols] < 0) | (df[metric_cols] > 1)).sum().sum()),
        "zero_mean_acc": int((df["mean_acc"] == 0).sum()),
        "max_fold_std": float(df["std_acc"].max()),
        "fold_std_gt_0p15": int((df["std_acc"] > 0.15).sum()),
        "fold_std_median": float(df["std_acc"].median()),
        "rerun_duplicate_rows": int(len(dup)),
        "rerun_max_abs_acc_diff": rerun_dev,
    }


def trajectory_table(df):
    rows = []
    rng = np.random.default_rng(SEED)
    for (d, m, mod), g in df.groupby(["dataset", "model", "modality"]):
        g = g.sort_values("eta")
        x = g["eta"].to_numpy(float)
        y = g["mean_acc"].to_numpy(float)
        # SE of the plotted mean: given explicitly for seed-averaged grids, otherwise the
        # ddof=0 standard deviation over five folds converted to the SE of their mean.
        se = (g["se"] if "se" in g else g["std_acc"] / 2.0).to_numpy(float)
        ref = compute_critical_breakdown_threshold(x, y)
        e_nak, _ = curvature_argmax(x, y, "not-a-knot")
        cmask = np.isin(np.round(x, 2), COARSE)
        e_coarse, _ = curvature_argmax(x[cmask], y[cmask])
        e_ss, k_ss = smoothing_eta(x, y)
        yb = y[:, None] + rng.normal(0, 1, (len(y), BOOT_B)) * se[:, None]
        e_boot, _ = curvature_argmax(x, yb)
        audc_b = np.trapezoid(yb, x, axis=0) / (x[-1] - x[0])
        rows.append({
            "dataset": d, "model": m, "modality": mod,
            "acc0": y[0], "acc_end": y[-1], "bal_acc0": g["mean_bal_acc"].iloc[0],
            "eta_star": ref["eta_star"], "kappa_max": ref["max_curvature"],
            "audc": ref["audc"], "mrr": ref["mrr"], "d_rate": ref["d_rate"],
            "audc_bal": np.trapezoid(g["mean_bal_acc"], x) / (x[-1] - x[0]),
            "eta_legacy": legacy_eta(x, y), "eta_notaknot": float(e_nak),
            "eta_coarse": float(e_coarse), "eta_smooth": e_ss, "kappa_smooth": k_ss,
            "eta_boot_sd": float(np.std(e_boot)),
            "eta_boot_lo": float(np.quantile(e_boot, 0.025)),
            "eta_boot_hi": float(np.quantile(e_boot, 0.975)),
            "eta_boot_within_0p05": float(np.mean(np.abs(e_boot - ref["eta_star"]) <= 0.05)),
            "mrr_boot_sd": float(np.std(audc_b / yb[0])),
            "max_abs_d2_raw": float(np.max(np.abs(np.diff(y, 2)))) / 0.05 ** 2,
        })
    t = pd.DataFrame(rows)
    t["audc_bal_mrr"] = t["audc_bal"] / t["bal_acc0"]
    return t


TIE_DECIMALS = 12      # values equal to floating-point noise are ties (flat trajectories give MRR = 1 +- 1e-16)


def friedman_block(pivot):
    pivot = pivot.round(TIE_DECIMALS)
    n, k = pivot.shape
    ranks = (-pivot).rank(axis=1, method="average")
    R = ranks.mean(axis=0)
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2)
    q = stats.studentized_range.ppf(1 - ALPHA, k, 1e6) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    ties = int((ranks.apply(lambda r: r.duplicated().any(), axis=1)).sum())
    return R.sort_values(), {
        "n": n, "k": k, "chi2_F": float(chi2), "df": k - 1,
        "p_chi2": float(stats.chi2.sf(chi2, k - 1)),
        "F_ID": float(ff), "df_ID": [k - 1, (k - 1) * (n - 1)],
        "p_ID": float(stats.f.sf(ff, k - 1, (k - 1) * (n - 1))),
        "p_scipy_tiecorr": float(stats.friedmanchisquare(*[pivot[c] for c in pivot.columns]).pvalue),
        "q_alpha": float(q), "CD": float(cd), "datasets_with_ties": ties,
    }


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def statistics(t):
    fried, ranks_out, wil, assumptions = [], {}, [], []
    for metric in ["mrr", "audc", "audc_bal_mrr"]:
        block = []
        for mod in sorted(t["modality"].unique()):
            pv = t[t.modality == mod].pivot(index="dataset", columns="model", values=metric)
            R, res = friedman_block(pv)
            res.update(metric=metric, modality=mod)
            block.append(res)
            ranks_out[f"{metric}|{mod}"] = {k: round(float(v), 3) for k, v in R.items()}
            pairs = list(itertools.combinations(pv.columns, 2))
            raw_p = [stats.wilcoxon(pv[a], pv[b], zero_method="pratt").pvalue for a, b in pairs]
            adj = holm(raw_p)
            for (a, b), p0, p1 in zip(pairs, raw_p, adj):
                nem = abs(R[a] - R[b]) > res["CD"]
                wil.append({"metric": metric, "modality": mod, "a": a, "b": b,
                            "median_diff": float((pv[a] - pv[b]).median()),
                            "p_raw": float(p0), "p_holm": float(p1),
                            "wilcoxon_sig": bool(p1 < ALPHA), "nemenyi_sig": bool(nem)})
            if metric == "mrr":
                sw = {c: float(stats.shapiro(pv[c]).pvalue) for c in pv.columns}
                assumptions.append({"modality": mod,
                                    "shapiro_min_p": min(sw.values()),
                                    "shapiro_reject_count": int(sum(p < ALPHA for p in sw.values())),
                                    "bartlett_p": float(stats.bartlett(*[pv[c] for c in pv.columns]).pvalue),
                                    "levene_p": float(stats.levene(*[pv[c] for c in pv.columns]).pvalue)})
        adj = holm([b["p_ID"] for b in block])
        for b, a in zip(block, adj):
            b["p_ID_holm_across_modalities"] = float(a)
        fried.extend(block)
    return pd.DataFrame(fried), ranks_out, pd.DataFrame(wil), pd.DataFrame(assumptions)


def confounds(t):
    out = {}
    for mod, g in t.groupby("modality"):
        within = []
        for _, gd in g.groupby("dataset"):
            within.append(stats.spearmanr(gd["acc0"], gd["mrr"]).statistic)
        out[mod] = {
            "spearman_acc0_mrr_pooled": float(stats.spearmanr(g["acc0"], g["mrr"]).statistic),
            "spearman_acc0_mrr_within_dataset_median": float(np.nanmedian(within)),
            "mrr_gt_1": int((g["mrr"] > 1).sum()),
            "median_relative_loss_at_0p5": float(np.median(1 - g["acc_end"] / g["acc0"])),
        }
    return out


def binary_test_retest(t, k_by_dataset):
    binary = [d for d, k in k_by_dataset.items() if k == 2]
    s = t[t.dataset.isin(binary) & (t.modality == "symmetric_label")].set_index(["dataset", "model"])
    a = t[t.dataset.isin(binary) & (t.modality == "asymmetric_label")].set_index(["dataset", "model"])
    a = a.loc[s.index]
    return {
        "n_pairs": int(len(s)), "n_binary_datasets": len(binary),
        "acc_end_abs_diff_median": float(np.median(np.abs(s["acc_end"] - a["acc_end"]))),
        "mrr_pearson": float(stats.pearsonr(s["mrr"], a["mrr"]).statistic),
        "mrr_abs_diff_median": float(np.median(np.abs(s["mrr"] - a["mrr"]))),
        "eta_spearman": float(stats.spearmanr(s["eta_star"], a["eta_star"]).statistic),
        "eta_abs_diff_median": float(np.median(np.abs(s["eta_star"] - a["eta_star"]))),
        "eta_within_0p05_frac": float(np.mean(np.abs(s["eta_star"] - a["eta_star"]) <= 0.05)),
    }


def dataset_inventory():
    rows = {}
    for f in sorted(os.listdir("data")):
        if not f.endswith(".npz"):
            continue
        z = np.load(os.path.join("data", f), allow_pickle=True)
        y = z["y"]
        counts = np.bincount(y.astype(int))
        rows[f[:-4]] = {"N": int(len(y)), "N_eff": int(min(len(y), 10000)),
                        "D": int(z["X"].shape[1]), "K": int(len(counts)),
                        "majority_rate": float(counts.max() / counts.sum())}
    return rows


def ablations(t):
    def cmp(col):
        d = np.abs(t[col] - t["eta_star"])
        return {"median_abs_diff": float(d.median()),
                "within_0p05": float((d <= 0.05).mean()),
                "spearman": float(stats.spearmanr(t[col], t["eta_star"]).statistic),
                "edge_frac": float(at_edge(t[col]).mean())}
    return {
        "natural_edge_frac": float(at_edge(t["eta_star"]).mean()),
        "legacy_univariate_vs_natural": cmp("eta_legacy"),
        "notaknot_vs_natural": cmp("eta_notaknot"),
        "coarse6_vs_natural": cmp("eta_coarse"),
        "smoothing_gcv_vs_natural": cmp("eta_smooth"),
        "smoothing_near_linear_frac": float((t["kappa_smooth"] < 1e-3).mean()),
        "bootstrap_eta_sd_median": float(t["eta_boot_sd"].median()),
        "bootstrap_eta_sd_p90": float(t["eta_boot_sd"].quantile(0.9)),
        "bootstrap_within_0p05_median": float(t["eta_boot_within_0p05"].median()),
        "bootstrap_ci_width_median": float((t["eta_boot_hi"] - t["eta_boot_lo"]).median()),
        "bootstrap_mrr_sd_median": float(t["mrr_boot_sd"].median()),
    }


NAME = {"cart": "CART", "randomforest": "RF", "xgboost": "XGBoost", "lightgbm": "LightGBM",
        "catboost": "CatBoost", "mlp": "MLP"}


def derived(t, fried, ranks, inv):
    """Every quantity the manuscript prose quotes, so that no number is computed by hand."""
    out = {"friedman": {}, "friedman_max_p_holm": {}, "leaders": {}, "mean_by_model": {}, "eta_pooled": {}}
    for r in fried.itertuples():
        out["friedman"][f"{r.metric}|{r.modality}"] = {"chi2_F": r.chi2_F, "F_ID": r.F_ID, "p_ID": r.p_ID,
                                                        "p_holm": r.p_ID_holm_across_modalities}
    for metric, g in fried.groupby("metric"):
        out["friedman_max_p_holm"][metric] = float(g.p_ID_holm_across_modalities.max())
    for key, rk in ranks.items():
        order = sorted(rk, key=rk.get)
        out["leaders"][key] = {"best": NAME[order[0]], "second": NAME[order[1]], "worst": NAME[order[-1]],
                               "order": [NAME[m] for m in order], "position": {m: order.index(m) + 1 for m in order}}
    for metric in ["audc", "mrr", "acc0", "acc_end"]:
        for mod, g in t.groupby("modality"):
            out["mean_by_model"][f"{metric}|{mod}"] = g.groupby("model")[metric].mean().to_dict()
    acc0 = t.groupby("model")["acc0"].mean()
    out["acc0_by_model"] = acc0.to_dict()
    out["acc0_best_model"] = NAME[acc0.idxmax()]
    d = t[t.kappa_max >= 1e-9]
    for mod, g in d.groupby("modality"):
        out["eta_pooled"][mod] = {"median": float(g.eta_star.median()), "q25": float(g.eta_star.quantile(.25)),
                                  "q75": float(g.eta_star.quantile(.75)),
                                  "by_model": g.groupby("model").eta_star.median().to_dict()}
    eta = d.eta_star.to_numpy()
    dist = np.abs(eta[:, None] - np.arange(0, 0.501, 0.05)[None, :]).min(1)
    flat = t[t.kappa_max < 1e-9]
    out["lattice"] = {"defined": int(len(eta)), "on_tier_share": float((dist <= 0.002).mean()),
                      "max_off_tier": float(dist.max())}
    out["flat"] = {"count": int(len(flat)), "datasets": sorted(flat.dataset.unique()),
                   "models": sorted(flat.model.unique()), "conditions": sorted(flat.modality.unique())}
    out["trajectories"] = int(len(t))
    out["mrr_gt_1"] = int((t.mrr > 1).sum())
    # the corpus is the 20 datasets of the study: the *_clean copies are duplicates of three of them
    corpus = [v for key, v in inv.items() if not key.endswith("_clean")]
    out["inventory_summary"] = {
        "n_datasets": len(corpus),
        "n_binary": sum(v["K"] == 2 for v in corpus),
        "n_multiclass": sum(v["K"] > 2 for v in corpus),
        "n_subsampled": sum(v["N"] > 10000 for v in corpus),
        "n_majority_ge_0p70": sum(v["majority_rate"] >= 0.70 for v in corpus)}
    return out


def primary_integrity(df):
    metric_cols = ["mean_acc", "mean_bal_acc", "mean_f1"]
    n_ds, n_cond = df.dataset.nunique(), df.modality.nunique()
    return {"rows": int(len(df)), "expected_rows": int(n_ds * 6 * n_cond * 11),
            "unique_keys": int(df.drop_duplicates(["dataset", "model", "modality", "eta"]).shape[0]),
            "nan_cells": int(df[metric_cols].isna().sum().sum()),
            "out_of_unit_interval": int(((df[metric_cols] < 0) | (df[metric_cols] > 1)).sum().sum()),
            "seeds_per_cell_min": int(df.n_seeds.min()), "seeds_per_cell_max": int(df.n_seeds.max()),
            "seed_sd_median": float(df.sd_seed.median()), "seed_sd_max": float(df.sd_seed.max())}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", action="store_true",
                    help="audit the seed-averaged five-condition grid (results/analysis/primary/cells.csv)")
    args = ap.parse_args()
    out = "results/analysis/audit_primary" if args.primary else OUT
    os.makedirs(out, exist_ok=True)
    if args.primary:
        df = pd.read_csv("results/analysis/primary/cells.csv").rename(columns={"condition": "modality"})
        # SE of a three-seed mean: the across-seed spread has only two degrees of freedom and can be
        # near zero by chance, so the larger of the seed-based and the fold-based estimate is used.
        n = df["n_seeds"].to_numpy(float)
        df["se"] = np.maximum(df["sd_seed"].to_numpy(float), df["fold_std"].to_numpy(float) / 2.0) / np.sqrt(n)
    else:
        df = pd.read_csv(BENCH)
    inv = dataset_inventory()
    t = trajectory_table(df)
    fried, ranks, wil, assum = statistics(t)
    k_by = {d: v["K"] for d, v in inv.items()}

    t.to_csv(os.path.join(out, "trajectory_audit.csv"), index=False)
    fried.to_csv(os.path.join(out, "friedman.csv"), index=False)
    wil.to_csv(os.path.join(out, "wilcoxon_holm.csv"), index=False)
    assum.to_csv(os.path.join(out, "parametric_assumptions.csv"), index=False)

    summary = {
        "integrity": primary_integrity(df) if args.primary else integrity(df),
        "inventory": inv,
        "ablations": ablations(t),
        "confounds": confounds(t),
        "mean_ranks": ranks,
        "wilcoxon_vs_nemenyi_agreement": {
            m: float((g["wilcoxon_sig"] == g["nemenyi_sig"]).mean()) for m, g in wil.groupby("metric")},
        "median_eta_star_by_model_modality": t.groupby(["modality", "model"])["eta_star"].median()
            .round(3).unstack().to_dict(),
    }
    if not args.primary:
        summary["binary_sym_vs_asym_test_retest"] = binary_test_retest(t, k_by)
    summary.update(derived(t, fried, ranks, inv))
    summary["assumptions"] = {"shapiro_min_p": float(assum.shapiro_min_p.min()),
                              "shapiro_rejects_min": int(assum.shapiro_reject_count.min()),
                              "shapiro_rejects_max": int(assum.shapiro_reject_count.max()),
                              "bartlett_levene_min_p": float(assum[["bartlett_p", "levene_p"]].min().min())}
    summary["post_hoc_counts"] = {f"{m}|{c}": {"wilcoxon": int(g.wilcoxon_sig.sum()), "nemenyi": int(g.nemenyi_sig.sum()),
                                               "disagree": int((g.wilcoxon_sig != g.nemenyi_sig).sum())}
                                  for (m, c), g in wil.groupby(["metric", "modality"])}
    summary["post_hoc_pairs_per_metric"] = int(wil.groupby("metric").size().iloc[0])
    bal = fried[fried.metric == "audc_bal_mrr"]
    summary["balanced_summary"] = {"n_conditions": int(len(bal)),
                                   "n_significant_holm": int((bal.p_ID_holm_across_modalities < 0.05).sum()),
                                   "not_significant": sorted(bal[bal.p_ID_holm_across_modalities >= 0.05].modality)}
    with open(os.path.join(out, "audit_numbers.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(json.dumps({k: summary[k] for k in ["integrity", "ablations", "confounds"]}, indent=1, default=float))
    print(fried[["metric", "modality", "chi2_F", "F_ID", "p_ID", "p_ID_holm_across_modalities",
                 "CD", "datasets_with_ties"]].to_string(index=False))
    print(assum.to_string(index=False))


if __name__ == "__main__":
    main()
