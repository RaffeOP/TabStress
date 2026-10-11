"""Quantities the manuscript reports about corrections, data quality, speed of degradation and test
families. Writes results/analysis/audit_primary/fix_extras.json (token source "fix"). Nothing here
changes a result file: every count is read from the stored tables, the cached arrays in data/ and,
for the missing-value check, the OpenML data-quality API (cached in openml_quality.json)."""

import itertools
import json
import os
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline
from sklearn.model_selection import StratifiedShuffleSplit

from experiments.forensic_audit import TIE_DECIMALS

OUT = "results/analysis/audit_primary"
CELLS = "results/analysis/primary/cells.csv"
TRAJ = os.path.join(OUT, "trajectory_audit.csv")
FRIED = os.path.join(OUT, "friedman.csv")
METRICS = ["mean_acc", "std_acc", "mean_bal_acc", "mean_f1"]
CONDITIONS = ["symmetric_label", "class_conditional", "gaussian_feature", "mcar_imputed", "mcar_native"]
MODELS = ["cart", "randomforest", "xgboost", "lightgbm", "catboost", "mlp"]
SUBSAMPLE_N, SUBSAMPLE_SEED = 10_000, 42
GRID_POINTS = 500
KEY = ["variant", "seed", "dataset", "model", "modality", "eta"]
TOL = 1e-12


def changed_rows(old_path, new_df):
    old = pd.read_csv(old_path)
    for df in (old, new_df):
        df["eta"] = df["eta"].round(2)
    m = old.merge(new_df, on=KEY, suffixes=("_old", "_new"))
    diff = np.zeros(len(m), dtype=bool)
    for c in METRICS:
        diff |= (m[f"{c}_old"] - m[f"{c}_new"]).abs().to_numpy() > TOL
    return len(m), int(diff.sum())


def patches():
    log = pd.read_csv("results/metrics/original_patch_log.csv")
    grid = log[log.file == "results/metrics/benchmark_results.csv"]
    cell_key = ["dataset", "model", "modality", "eta"]
    changed = grid[(grid.old - grid.new).abs() > TOL]
    out = {"grid_cells_recomputed": int(len(grid.drop_duplicates(cell_key))),
           "grid_cells_changed": int(len(changed.drop_duplicates(cell_key))),
           "grid_values_changed": int(len(changed)),
           "grid_datasets_affected": int(grid.dataset.nunique())}
    ext = {f[:-4]: pd.read_csv(os.path.join("results/metrics/extended", f))
           for f in os.listdir("results/metrics/extended") if f.endswith(".csv")}
    for name, backup_dir in [("base__seed43", "extended_backup_gaussfix"), ("catboost_default__seed42", "extended_backup_gaussfix"),
                             ("xgb_default__seed42", "extended_backup_gaussfix"), ("cc_binary__seed42", "extended_backup_tiefix"),
                             ("cc_binary__seed43", "extended_backup_tiefix"), ("cc_binary__seed44", "extended_backup_tiefix")]:
        n, ch = changed_rows(os.path.join("results/metrics", backup_dir, name + ".csv"), ext[name].copy())
        out[f"{name}_cells_compared"], out[f"{name}_cells_changed"] = n, ch
    tie = ext["cc_binary__seed42"]
    out["tie_fix_datasets"] = [d for d in ("hill_valley", "madelon") if d in tie.dataset.unique()]
    out["tie_fix_cells_per_seed"] = int(len(tie[tie.dataset.isin(out["tie_fix_datasets"])]))
    return out


def seconds_total():
    total = sum(pd.read_csv(os.path.join("results/metrics/extended", f)).seconds.sum()
                for f in os.listdir("results/metrics/extended") if f.endswith(".csv"))
    return {"extended_cell_hours": float(total) / 3600.0}


def load_used(name):
    z = np.load(os.path.join("data", f"{name}.npz"), allow_pickle=True)
    X, y = z["X"], z["y"]
    if X.shape[0] > SUBSAMPLE_N:
        idx, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=SUBSAMPLE_N, random_state=SUBSAMPLE_SEED).split(X, y))
        X, y = X[idx], y[idx]
    return X, y, [str(f) for f in z["feature_names"]]


def data_quality():
    from src.data_loader import BENCHMARK_SUITE
    dup = {}
    for name in sorted(BENCHMARK_SUITE):
        X, y, _ = load_used(name)
        n = len(X)
        dup[name] = {"n_used": int(n), "dup_rows": int(n - len(np.unique(X, axis=0))),
                     "all_minus999_rows": int((X == -999).all(axis=1).sum())}
    full_spam = np.load("data/spambase.npz", allow_pickle=True)["X"]
    z = np.load("data/miniboone.npz", allow_pickle=True)["X"]
    churn_feats = [str(f) for f in np.load("data/churn.npz", allow_pickle=True)["feature_names"]]
    n_dup = {k: v["dup_rows"] for k, v in dup.items()}
    return {"duplicates": dup,
            "n_datasets_with_duplicates": int(sum(v > 0 for v in n_dup.values())),
            "spambase_full": {"n": int(len(full_spam)), "dup": int(len(full_spam) - len(np.unique(full_spam, axis=0)))},
            "miniboone_full": {"n": int(len(z)), "all_minus999_rows": int((z == -999).all(axis=1).sum())},
            "churn_has_phone_number": "phone_number" in churn_feats}


def openml_missing():
    from src.data_loader import BENCHMARK_SUITE
    cache = os.path.join(OUT, "openml_quality.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    res = {}
    for name, did in sorted(BENCHMARK_SUITE.items()):
        with urllib.request.urlopen(f"https://www.openml.org/api/v1/json/data/qualities/{did}", timeout=60) as r:
            q = {x["name"]: x["value"] for x in json.load(r)["data_qualities"]["quality"]}
        res[name] = {"openml_id": did, "missing_values": float(q["NumberOfMissingValues"]),
                     "instances_with_missing": float(q["NumberOfInstancesWithMissingValues"])}
    json.dump(res, open(cache, "w"), indent=1)
    return res


def velocity(cells):
    """Speed of degradation from the natural cubic spline: mean velocity -(A(0.5)-A(0))/0.5, which
    is the average loss per unit intensity, and the peak speed max |S'| on the dense grid."""
    out = {}
    grid = np.linspace(0, 0.5, GRID_POINTS)
    rows = []
    for (c, d, m), g in cells.groupby(["condition", "dataset", "model"]):
        g = g.sort_values("eta")
        spl = CubicSpline(g.eta.to_numpy(), g.mean_acc.to_numpy(), bc_type="natural")
        v = spl(grid, 1)
        rows.append((c, d, m, -float(v.mean()), float(np.abs(v).max()), float(grid[np.abs(v).argmax()])))
    t = pd.DataFrame(rows, columns=["condition", "dataset", "model", "mean_loss_rate", "peak_speed", "peak_at"])
    for c in CONDITIONS:
        s = t[t.condition == c]
        out[c] = {"median_mean_loss_rate": float(s.mean_loss_rate.median()),
                  "median_peak_speed": float(s.peak_speed.median()),
                  "peak_at_edge_share": float(((s.peak_at < 0.025) | (s.peak_at > 0.475)).mean())}
    by_model = t[t.condition == "symmetric_label"].groupby("model").mean_loss_rate.median()
    out["symmetric_by_model"] = {m: float(by_model[m]) for m in MODELS}
    return out


def friedman_block(pv):
    n, k = pv.shape
    ranks = np.vstack([stats.rankdata(-r) for r in pv.to_numpy()])
    R = ranks.mean(axis=0)
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2)
    return {"n": int(n), "F_ID": float(ff), "p_ID": float(stats.f.sf(ff, k - 1, (k - 1) * (n - 1)))}


def kernels(traj, inventory):
    """Class-conditional condition analysed per kernel: pair-flip (K >= 3) and minority-to-majority (K = 2)."""
    out = {}
    for name, pick in (("pairflip", lambda d: inventory[d]["K"] >= 3), ("minority", lambda d: inventory[d]["K"] == 2)):
        for metric in ("mrr", "audc"):
            t = traj[(traj.modality == "class_conditional") & traj.dataset.map(pick)]
            out[f"{name}|{metric}"] = friedman_block(t.pivot(index="dataset", columns="model", values=metric)[MODELS])
    return out


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def holm_families(fried):
    """Holm over all fifteen Friedman tests (five conditions x three scores) against the reported
    five-per-score families."""
    rows = fried.assign(key=fried.metric + "|" + fried.modality)
    adj = holm(rows.p_ID.to_numpy())
    all15 = {k: float(a) for k, a in zip(rows.key, adj)}
    n_sig_per_score = int((rows.p_ID_holm_across_modalities < 0.05).sum())
    n_sig_all15 = int((adj < 0.05).sum())
    return {"all15": all15, "n_tests": int(len(rows)), "n_sig_per_score_family": n_sig_per_score,
            "n_sig_all15": n_sig_all15,
            "cc_mrr_all15": all15["mrr|class_conditional"],
            "bal_mcar_imputed_all15": all15["audc_bal_mrr|mcar_imputed"]}


def assumptions_ranges():
    a = pd.read_csv(os.path.join(OUT, "parametric_assumptions.csv")).set_index("modality")
    mc = a.loc[["mcar_imputed", "mcar_native"], ["bartlett_p", "levene_p"]]
    return {"mcar_min_p": float(mc.min().min()), "mcar_max_p": float(mc.max().max()),
            "n_conditions_rejecting_equal_variance": int((a[["bartlett_p", "levene_p"]].min(axis=1) < 0.05).sum()),
            "per_condition_min_p": {c: float(a.loc[c, ["bartlett_p", "levene_p"]].min()) for c in a.index}}


def ablation_edges(traj):
    """Share of defined thresholds within 0.01 of the boundary, per estimator. The four trajectories with
    kappa = 0 everywhere carry the placeholder 0 and are excluded: no threshold exists for them."""
    d = traj[traj.kappa_max >= 1e-9]
    out = {"n_defined": int(len(d))}
    for name, col in (("reference", "eta_star"), ("legacy", "eta_legacy"), ("notaknot", "eta_notaknot"),
                      ("coarse6", "eta_coarse"), ("smoothing_gcv", "eta_smooth")):
        e = d[col]
        out[name] = float(((e <= 0.01) | (e >= 0.49)).mean())
        out[name + "_count"] = int(((e <= 0.01) | (e >= 0.49)).sum())
    return out


def software():
    """Versions as installed in the environment that produced the results, read from package metadata."""
    from importlib import metadata
    import platform
    pk = {"sklearn": "scikit-learn", "xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost",
          "autorank": "autorank", "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "matplotlib": "matplotlib"}
    out = {k: metadata.version(v) for k, v in pk.items()}
    out["python"] = platform.python_version()
    return out


def degenerate(cells, traj, inventory):
    """Binary trajectories of a constant majority-class predictor (balanced accuracy at most 0.5 + 1e-3
    at every tier) and the MRR leader of each condition with and without the affected datasets."""
    binary = [d for d, v in inventory.items() if v["K"] == 2]
    bal = cells[cells.dataset.isin(binary)].groupby(["condition", "dataset", "model"]).mean_bal_acc.max()
    hit = bal[bal <= 0.5 + 1e-3].reset_index()
    pairs = sorted({(r.dataset, r.model) for r in hit.itertuples()})
    drop = sorted({d for d, _ in pairs})
    out = {"pairs": [f"{d}/{m}" for d, m in pairs], "datasets": drop, "models": sorted({m for _, m in pairs}),
           "n_conditions": int(hit.condition.nunique()), "leader_changes": 0}
    for cond in CONDITIONS:
        t = traj[traj.modality == cond]
        lead = {}
        for tag, sub in (("all", t), ("excl", t[~t.dataset.isin(drop)])):
            pv = sub.pivot(index="dataset", columns="model", values="mrr")[MODELS].round(TIE_DECIMALS)
            ranks = pv.rank(axis=1, ascending=False).mean()
            lead[tag] = ranks.idxmin()
            out[f"{cond}|{tag}_best_rank"] = float(ranks.min())
        out[f"{cond}|leader"] = lead["all"]
        out["leader_changes"] += int(lead["all"] != lead["excl"])
    return out


def main():
    cells = pd.read_csv(CELLS)
    traj = pd.read_csv(TRAJ)
    inv = json.load(open(os.path.join(OUT, "audit_numbers.json")))["inventory"]
    quality = openml_missing()
    out = {"patches": patches(), "compute": seconds_total(), "quality": data_quality(),
           "openml": {"n_datasets": len(quality), "total_missing_values": float(sum(v["missing_values"] for v in quality.values())),
                      "n_with_missing": int(sum(v["missing_values"] > 0 for v in quality.values()))},
           "velocity": velocity(cells), "kernels": kernels(traj, inv),
           "holm": holm_families(pd.read_csv(FRIED)), "variance_tests": assumptions_ranges(),
           "ablation_edge": ablation_edges(traj), "software": software(), "mrr_gt_1_count": int((traj.mrr > 1).sum()), "n_trajectories": int(len(traj)),
           "pol": {"n_distinct_values": 11, "target_numeric": True},
           "degenerate": degenerate(cells, traj, inv)}
    json.dump(out, open(os.path.join(OUT, "fix_extras.json"), "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("patches", "openml", "kernels", "holm", "variance_tests", "mrr_gt_1_count")}, indent=1, default=float)[:3500])
    print(json.dumps(out["velocity"], indent=1)[:1500])
    print(json.dumps({k: v for k, v in out["quality"].items() if k != "duplicates"}, default=float))
    print({k: v for k, v in out["quality"]["duplicates"].items() if v["dup_rows"] or v["all_minus999_rows"]})


if __name__ == "__main__":
    main()
