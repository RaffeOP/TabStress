"""Analyses that the extended runs make possible, on the primary five-condition grid.

  seeds          per-seed Friedman tests and rank agreement; seed-to-seed reliability of
                 eta*, MRR and AUDC (cells_by_seed.csv, three seeds)
  native         native missing-value handling versus median imputation under MCAR,
                 per tree family, on seed-averaged trajectories
  budget         CatBoost at library defaults (1,000 rounds) versus 100 rounds, seed 42
  threshold      minority->majority noise on binary datasets: terminal accuracy versus the
                 majority-class rate, terminal balanced accuracy

Writes results/analysis/audit_primary/extras.json."""

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(__file__))
from forensic_audit import friedman_block, holm
from src.breakdown_math import compute_critical_breakdown_threshold

PRIMARY = "results/analysis/primary"
AUDIT = "results/analysis/audit_primary"
BASE = "results/metrics/benchmark_results.csv"
EXT = "results/metrics/extended"
MODELS = ["cart", "randomforest", "xgboost", "lightgbm", "catboost", "mlp"]
TREES = MODELS[:5]
CONDITIONS = ["symmetric_label", "class_conditional", "gaussian_feature", "mcar_imputed", "mcar_native"]
ORIGINAL = ["symmetric_label", "asymmetric_label", "gaussian_feature", "mcar_missing"]
FLAT = 1e-9


def trajectories(cells, group_cols, value="mean_acc"):
    rec = []
    for key, g in cells.groupby(group_cols):
        g = g.sort_values("eta")
        if len(g) != 11:
            raise ValueError(f"incomplete trajectory {key}: {len(g)} tiers")
        x, y = g.eta.to_numpy(float), g[value].to_numpy(float)
        r = compute_critical_breakdown_threshold(x, y)
        rec.append(dict(zip(group_cols, key), acc0=y[0], acc_end=y[-1], audc=r["audc"], mrr=r["mrr"],
                        bal_end=g.mean_bal_acc.iloc[-1],
                        eta_star=r["eta_star"] if r["max_curvature"] >= FLAT else np.nan))
    return pd.DataFrame(rec)


def seeds_analysis(by_seed):
    t = trajectories(by_seed, ["seed", "condition", "dataset", "model"])
    seeds = sorted(t.seed.unique())
    out = {"seeds": [int(s) for s in seeds], "per_condition": {}, "reliability": {}}
    for metric in ["mrr", "audc"]:
        for cond in CONDITIONS:
            ranks, ps = [], []
            for s in seeds:
                pv = t[(t.seed == s) & (t.condition == cond)].pivot(index="dataset", columns="model", values=metric)
                R, f = friedman_block(pv)
                ranks.append(R.reindex(MODELS))
                ps.append(f["p_ID"])
            taus = [stats.kendalltau(a, b).statistic for a, b in itertools.combinations(ranks, 2)]
            out["per_condition"][f"{metric}|{cond}"] = {
                "p_ID_per_seed": [float(p) for p in ps],
                "max_p_ID": float(max(ps)),
                "kendall_tau_min": float(min(taus)),
                "best_per_seed": [r.idxmin() for r in ranks],
                "worst_per_seed": [r.idxmax() for r in ranks],
                "ranks_per_seed": [{k: round(float(v), 3) for k, v in r.items()} for r in ranks]}
    blocks = out["per_condition"]
    out["summary"] = {"n_blocks": len(blocks),
                      "same_best_all_seeds": sum(len(set(v["best_per_seed"])) == 1 for v in blocks.values()),
                      "same_worst_all_seeds": sum(len(set(v["worst_per_seed"])) == 1 for v in blocks.values()),
                      "kendall_tau_min": float(min(v["kendall_tau_min"] for v in blocks.values())),
                      "max_p_ID": float(max(v["max_p_ID"] for v in blocks.values()))}
    out["margin"] = margin_analysis(by_seed, t)
    key = ["condition", "dataset", "model"]
    w = t.pivot_table(index=key, columns="seed", values=["eta_star", "mrr", "audc"])
    for col in ["eta_star", "mrr", "audc"]:
        pairs = list(itertools.combinations(seeds, 2))
        rho = [stats.spearmanr(w[(col, a)], w[(col, b)], nan_policy="omit").statistic for a, b in pairs]
        within = [float(np.nanmean(np.abs(w[(col, a)] - w[(col, b)]) <= 0.05)) for a, b in pairs]
        out["reliability"][col] = {"spearman_min": float(min(rho)), "spearman_max": float(max(rho)),
                                   "within_0p05_mean": float(np.mean(within)),
                                   "n_trajectories": int(len(w))}
    return out


def top_two_margin(wide):
    """wide: rows = models of one comparison unit, columns = seeds. Compares the gap between the two
    best seed-mean scores with the larger of their two seed standard deviations."""
    mean, sd = wide.mean(axis=1), wide.std(axis=1, ddof=1)
    order = mean.sort_values(ascending=False).index
    first, second = order[0], order[1]
    margin = float(mean[first] - mean[second])
    spread = float(max(sd[first], sd[second]))
    best_per_seed = wide.idxmax(axis=0)
    return {"margin": margin, "spread": spread, "tied": margin == 0.0,
            "below": margin < spread, "same_best": bool((best_per_seed == best_per_seed.iloc[0]).all())}


def margin_analysis(by_seed, t):
    """Is the gap between the two best families larger than the seed-to-seed variation?
    cell:  accuracy at one (condition, dataset, eta); the clean cell (eta = 0) is shared by all
           conditions and counted once per dataset
    block: AUDC of one (condition, dataset)"""
    cells = by_seed[(by_seed.eta > 0) | (by_seed.condition == CONDITIONS[0])]
    rec = [top_two_margin(g.pivot(index="model", columns="seed", values="mean_acc"))
           for _, g in cells.groupby(["condition", "dataset", "eta"])]
    blk = [top_two_margin(g.pivot(index="model", columns="seed", values="audc"))
           for _, g in t.groupby(["condition", "dataset"])]
    out = {}
    for name, r in [("cell", pd.DataFrame(rec)), ("block", pd.DataFrame(blk))]:
        out[name] = {"n": int(len(r)), "below_seed_sd_share": float(r.below.mean()),
                     "tied": int(r.tied.sum()), "same_best_share": float(r.same_best.mean()),
                     "median_margin": float(r.margin.median()), "median_spread": float(r.spread.median())}
    return out


def native_analysis(traj):
    a = traj[traj.modality == "mcar_native"].set_index(["dataset", "model"])
    b = traj[traj.modality == "mcar_imputed"].set_index(["dataset", "model"])
    per = {}
    for m in TREES:
        x, y = a.xs(m, level="model"), b.xs(m, level="model").loc[a.xs(m, level="model").index]
        d_audc, d_mrr = x.audc - y.audc, x.mrr - y.mrr
        per[m] = {"median_delta_audc": float(d_audc.median()), "mean_delta_audc": float(d_audc.mean()),
                  "median_delta_mrr": float(d_mrr.median()),
                  "native_better": int((d_audc > 0).sum()), "n": int(len(d_audc)),
                  "p_audc": float(stats.wilcoxon(x.audc, y.audc, zero_method="pratt").pvalue)}
    adj = holm([v["p_audc"] for v in per.values()])
    for (m, v), p in zip(per.items(), adj):
        v["p_audc_holm"] = float(p)
    per["summary"] = {"n_lean_imputed": sum(v["median_delta_audc"] < 0 for v in per.values()),
                      "sig_native_better": sum(v["p_audc_holm"] < 0.05 and v["median_delta_audc"] > 0 for v in per.values()),
                      "sig_native_worse": sum(v["p_audc_holm"] < 0.05 and v["median_delta_audc"] < 0 for v in per.values()),
                      "max_abs_median_delta_audc": max(abs(v["median_delta_audc"]) for v in per.values()),
                      "min_p_holm": min(v["p_audc_holm"] for v in per.values())}
    return per


def load_variant(name, model):
    frames = [pd.read_csv(os.path.join(EXT, f)) for f in os.listdir(EXT) if f.startswith(name + "__")]
    if not frames:
        return None
    v = pd.concat(frames)
    v["eta"] = v.eta.round(2)
    if len(v) != 20 * (1 + 4 * 10):
        return False
    clean = v[v.modality == "clean"]
    v = pd.concat([v[v.modality != "clean"]] + [clean.assign(modality=m) for m in ORIGINAL])
    return trajectories(v, ["modality", "dataset"]).assign(model=model)


def budget_analysis():
    """Matched budget (main grid, seed 42) against every library at its shipped defaults:
    CatBoost 1,000 rounds with automatic learning rate, XGBoost learning rate 0.3; LightGBM,
    random forest and CART defaults already coincide with the matched grid."""
    base = pd.read_csv(BASE)
    base["eta"] = base.eta.round(2)
    t_base = trajectories(base, ["modality", "dataset", "model"])
    variants = {"catboost": load_variant("catboost_default", "catboost"),
                "xgboost": load_variant("xgb_default", "xgboost")}
    status = {m: (v is not None and v is not False) for m, v in variants.items()}
    out = {"complete": all(status.values()), "available": status, "per_library": {}, "rankings": {}}
    tests = []
    for m, tv in variants.items():
        if not status[m]:
            continue
        for mod in ORIGINAL:
            d = tv[tv.modality == mod].set_index("dataset")
            c = t_base[(t_base.modality == mod) & (t_base.model == m)].set_index("dataset").loc[d.index]
            rec = {"median_delta_acc0": float((d.acc0 - c.acc0).median()),
                   "median_delta_audc": float((d.audc - c.audc).median()),
                   "median_delta_mrr": float((d.mrr - c.mrr).median()),
                   "default_better_audc": int((d.audc > c.audc).sum()),
                   "p_audc": float(stats.wilcoxon(d.audc, c.audc, zero_method="pratt").pvalue)}
            out["per_library"][f"{m}|{mod}"] = rec
            tests.append(f"{m}|{mod}")
    adj = holm([out["per_library"][k]["p_audc"] for k in tests])
    for k, p in zip(tests, adj):
        out["per_library"][k]["p_audc_holm"] = float(p)
    if out["complete"]:
        t_def = t_base.copy()
        for m, tv in variants.items():
            t_def = t_def[t_def.model != m]
            t_def = pd.concat([t_def, tv], ignore_index=True)
        for mod in ORIGINAL:
            for metric in ["mrr", "audc"]:
                R0, f0 = friedman_block(t_base[t_base.modality == mod].pivot(index="dataset", columns="model", values=metric))
                R1, f1 = friedman_block(t_def[t_def.modality == mod].pivot(index="dataset", columns="model", values=metric))
                R0, R1 = R0.reindex(MODELS), R1.reindex(MODELS)
                out["rankings"][f"{metric}|{mod}"] = {
                    "matched": {k: round(float(v), 3) for k, v in R0.items()},
                    "defaults": {k: round(float(v), 3) for k, v in R1.items()},
                    "best_matched": R0.idxmin(), "best_defaults": R1.idxmin(),
                    "worst_defaults": R1.idxmax(),
                    "kendall_tau": float(stats.kendalltau(R0, R1).statistic),
                    "p_ID_defaults": float(f1["p_ID"])}
        taus = [v["kendall_tau"] for v in out["rankings"].values()]
        out["rankings_summary"] = {"kendall_tau_min": float(min(taus)), "kendall_tau_max": float(max(taus)),
                                   "cart_last_everywhere": all(v["worst_defaults"] == "cart" for v in out["rankings"].values())}
    return out


def cell_counts(by_seed):
    """Distinct cross-validated cells actually computed, per experiment block."""
    base = len(pd.read_csv(BASE))
    ext = pd.concat([pd.read_csv(os.path.join(EXT, f)) for f in os.listdir(EXT) if f.endswith(".csv")])
    per = ext.groupby("variant").size().to_dict()
    total = base + int(sum(per.values()))
    return {"original_grid": base, "extended_by_variant": {k: int(v) for k, v in per.items()},
            "total_cells": total, "total_fits": 5 * total,
            "seeds_complete": sorted(int(x) for x in by_seed.seed.unique()),
            "n_seeds": int(by_seed.seed.nunique())}


def threshold_analysis(traj, inventory):
    binary = [d for d, v in inventory.items() if v["K"] == 2]
    t = traj[(traj.modality == "class_conditional") & traj.dataset.isin(binary)].copy()
    t["pi_max"] = t.dataset.map(lambda d: inventory[d]["majority_rate"])
    t["gap"] = t.acc_end - t.pi_max
    imb = t[t.pi_max >= 0.85]
    bal = t[t.pi_max <= 0.55]
    sym = traj[(traj.modality == "symmetric_label") & traj.dataset.isin(binary)]
    multi = [d for d, v in inventory.items() if v["K"] > 2]
    cm = traj[(traj.modality == "class_conditional") & traj.dataset.isin(multi)].set_index(["dataset", "model"])
    sm = traj[(traj.modality == "symmetric_label") & traj.dataset.isin(multi)].set_index(["dataset", "model"]).loc[cm.index]
    multiclass = {"n_pairs": int(len(cm)), "n_lower_under_pairflip": int((cm.acc_end < sm.acc_end).sum()),
                  "median_acc_end_pairflip": float(cm.acc_end.median()), "median_acc_end_symmetric": float(sm.acc_end.median())}
    return {"multiclass": multiclass, "n_binary": len(binary),
            "imbalanced_datasets": sorted(imb.dataset.unique()),
            "n_imbalanced": int(imb.dataset.nunique()), "n_balanced": int(bal.dataset.nunique()),
            "median_bal_end_symmetric_binary": float(sym.bal_end.median()),
            "imbalanced_max_abs_gap": float(imb.gap.abs().max()),
            "imbalanced_median_bal_end": float(imb.bal_end.median()),
            "balanced_datasets": sorted(bal.dataset.unique()),
            "balanced_median_gap": float(bal.gap.median()),
            "median_bal_end_class_conditional": float(t.bal_end.median()),
            "median_bal_end_symmetric": float(sym.bal_end.median()),
            "median_acc_end_class_conditional": float(t.acc_end.median()),
            "median_acc_end_symmetric": float(sym.acc_end.median())}


def variance_ratio(traj):
    """Per condition: variance of MRR between datasets (means over models) over the variance
    between models (means over datasets). Large values: the dataset, not the model, decides."""
    out = {}
    for c in CONDITIONS:
        g = traj[traj.modality == c]
        out[c] = float(g.groupby("dataset").mrr.mean().var() / g.groupby("model").mrr.mean().var())
    return out


def main():
    by_seed = pd.read_csv(os.path.join(PRIMARY, "cells_by_seed.csv"))
    traj = pd.read_csv(os.path.join(AUDIT, "trajectory_audit.csv"))
    inventory = json.load(open(os.path.join(AUDIT, "audit_numbers.json")))["inventory"]
    traj_bal = trajectories(pd.read_csv(os.path.join(PRIMARY, "cells.csv")), ["condition", "dataset", "model"])
    traj = traj.merge(traj_bal[["condition", "dataset", "model", "bal_end"]].rename(columns={"condition": "modality"}),
                      on=["modality", "dataset", "model"])
    out = {"counts": cell_counts(by_seed),
           "seeds": seeds_analysis(by_seed) if by_seed.seed.nunique() > 1 else None,
           "native": native_analysis(traj),
           "budget": budget_analysis(),
           "threshold": threshold_analysis(traj, inventory),
           "variance_ratio": variance_ratio(traj)}
    with open(os.path.join(AUDIT, "extras.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps({k: (v if k != "seeds" else (v and v["reliability"])) for k, v in out.items()}, indent=1,
                     default=float)[:3000])


if __name__ == "__main__":
    main()
