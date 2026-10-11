"""Analyses added in the revision. Writes results/analysis/revision/revision.json (token source "rev"),
CSV tables and LaTeX tables under results/analysis/revision/.

  metrics       balanced AUDC (primary robustness measure), AUDC, MRR, balanced MRR and the
                chance/majority-adjusted retention ratios on the primary (Protocol A) grid
  confirmatory  pre-specified hypothesis family {balanced AUDC, AUDC, MRR} x five conditions, Holm over 15;
                sensitivity metrics reported with Holm over the full set of 30 tests
  clean_vs_robust  does clean accuracy predict robustness?
  knee          effective resolution and seed reliability of the spline knee and of the discrete
                second-difference knee on real trajectories
  deploy        Protocol B (clean training, corrupted evaluation) versus Protocol A
  mechanisms    MAR and MNAR missingness (both protocols)
  categorical   categorical-aware and categorical-only corruption on the datasets with nominal attributes
  tuned         tuned versus matched-budget configurations (seed 42)
  artifacts     cleaned spambase / churn / MiniBooNE, and exclusion of the flagged datasets
  accounting    explicit derivation of every cell and fit count
"""

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(__file__))
from forensic_audit import TIE_DECIMALS, friedman_block, holm
from gaussian_principal import GAUSS_OP, load_cont, no_continuous_datasets, substitute_continuous
from src.breakdown_math import compute_critical_breakdown_threshold

PRIMARY = "results/analysis/primary"
REV_METRICS = "results/metrics/revision"
ART_METRICS = "results/metrics/artifact"
OUT = "results/analysis/revision"
MODELS = ["cart", "randomforest", "xgboost", "lightgbm", "catboost", "mlp"]
CONDITIONS = ["symmetric_label", "class_conditional", "gaussian_feature", "mcar_imputed", "mcar_native"]
CONFIRMATORY = ["audc_bal", "audc", "mrr"]
SENSITIVITY = ["mrr_bal", "amrr", "amrr_bal"]
ALL_METRICS = CONFIRMATORY + SENSITIVITY
ALPHA = 0.05
TIER = 0.05
MIN_HEADROOM = 0.01
FLAT = 1e-9
ARTIFACT_SETS = {"spambase": "spambase_clean", "churn": "churn_clean", "miniboone": "miniboone_clean"}


def dataset_baselines():
    """Majority-class rate and number of classes of every dataset as it enters the experiment."""
    from experiments.run_extended import data
    from src.data_loader import BENCHMARK_SUITE
    out = {}
    for d in list(BENCHMARK_SUITE) + list(ARTIFACT_SETS.values()):
        _, y, k = data(d)
        out[d] = {"pi_max": float(np.bincount(y).max() / len(y)), "K": int(k)}
    return out


def adjusted_retention(audc, acc0, baseline):
    """(AUDC - b) / (Acc(0) - b): the share of above-baseline skill retained over the sweep. A model with
    less than MIN_HEADROOM of skill above the baseline at eta = 0 has nothing to retain and scores 0."""
    headroom = acc0 - baseline
    return 0.0 if headroom < MIN_HEADROOM else (audc - baseline) / headroom


def discrete_knee(y):
    return float(np.round(0.05 * (1 + int(np.argmax(np.abs(y[:-2] - 2 * y[1:-1] + y[2:])))), 2))


def trajectory_metrics(g, base):
    """All revision metrics of one 11-tier trajectory (rows sorted by eta)."""
    x, y, yb = g.eta.to_numpy(float), g.mean_acc.to_numpy(float), g.mean_bal_acc.to_numpy(float)
    span = x[-1] - x[0]
    r = compute_critical_breakdown_threshold(x, y)
    audc_bal = float(np.trapezoid(yb, x) / span)
    knee = r["eta_star"] if r["max_curvature"] >= FLAT else np.nan
    return dict(acc0=y[0], bal0=yb[0], acc_end=y[-1], audc=r["audc"], mrr=r["mrr"], audc_bal=audc_bal,
                mrr_bal=audc_bal / yb[0], amrr=adjusted_retention(r["audc"], y[0], base["pi_max"]),
                amrr_bal=adjusted_retention(audc_bal, yb[0], 1.0 / base["K"]), eta_knee=knee,
                eta_knee_tier=np.round(knee / TIER) * TIER if np.isfinite(knee) else np.nan,
                eta_discrete=discrete_knee(y))


def trajectories(cells, keys, bases):
    rows = []
    for key, g in cells.groupby(keys):
        g = g.sort_values("eta")
        if len(g) != 11:
            raise ValueError(f"incomplete trajectory {key}: {len(g)} tiers")
        rec = dict(zip(keys, key))
        rec.update(trajectory_metrics(g, bases[rec["dataset"]]))
        rows.append(rec)
    return pd.DataFrame(rows)


def friedman_family(t, metrics, conditions, cond_col="condition"):
    """Friedman/Iman-Davenport per (metric, condition), Wilcoxon-Holm pairs, Nemenyi; one row per test."""
    rows = []
    for metric in metrics:
        for cond in conditions:
            pv = t[t[cond_col] == cond].pivot(index="dataset", columns="model", values=metric)[MODELS].round(TIE_DECIMALS)
            R, res = friedman_block(pv)
            pairs = list(itertools.combinations(MODELS, 2))
            p_pairs = holm([stats.wilcoxon(pv[a], pv[b], zero_method="pratt").pvalue for a, b in pairs])
            # Kendall's coefficient of concordance, the effect size of the Friedman test
            kendall_w = res["chi2_F"] / (res["n"] * (res["k"] - 1))
            rows.append(dict(metric=metric, condition=cond, F_ID=res["F_ID"], p=res["p_ID"], CD=res["CD"], W=kendall_w,
                             best=R.index[0], worst=R.index[-1], best_rank=float(R.iloc[0]),
                             worst_rank=float(R.iloc[-1]), ranks={m: float(R[m]) for m in MODELS},
                             n_wilcoxon_sig=int(np.sum(p_pairs < ALPHA)),
                             n_nemenyi_sig=int(sum(abs(R[a] - R[b]) > res["CD"] for a, b in pairs))))
    return pd.DataFrame(rows)


def confirmatory(traj):
    fam = friedman_family(traj, ALL_METRICS, CONDITIONS)
    conf = fam.metric.isin(CONFIRMATORY)
    fam.loc[conf, "p_holm_confirmatory"] = holm(fam.loc[conf, "p"])
    fam["p_holm_all"] = holm(fam["p"])
    fam["role"] = np.where(conf, "confirmatory", "sensitivity")
    fam.loc[fam.metric == "audc_bal", "role"] = "primary"
    out = {"n_confirmatory": int(conf.sum()), "n_all": int(len(fam)),
           "confirmatory_max_p_holm": float(fam.loc[conf, "p_holm_confirmatory"].max()),
           "confirmatory_n_rejected": int((fam.loc[conf, "p_holm_confirmatory"] < ALPHA).sum()),
           "all_n_rejected": int((fam["p_holm_all"] < ALPHA).sum()),
           "tests": {f"{r.metric}|{r.condition}": {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                                                    for k, v in r._asdict().items() if k != "Index"}
                     for r in fam.itertuples()}}
    for metric in ALL_METRICS:
        sub = fam[fam.metric == metric]
        col = "p_holm_confirmatory" if metric in CONFIRMATORY else "p_holm_all"
        out[f"{metric}_n_sig"] = int((sub[col] < ALPHA).sum())
        out[f"{metric}_max_p"] = float(sub[col].max())
        out[f"{metric}_not_sig"] = [c for c, p in zip(sub.condition, sub[col]) if p >= ALPHA]
    out["kendall_w_min"] = float(fam.loc[conf, "W"].min())
    out["kendall_w_max"] = float(fam.loc[conf, "W"].max())
    out["rank_agreement"] = rank_agreement(fam)
    out["leader_counts"] = {metric: fam[fam.metric == metric].best.value_counts().to_dict() for metric in ALL_METRICS}
    out["worst_counts"] = {metric: fam[fam.metric == metric].worst.value_counts().to_dict() for metric in ALL_METRICS}
    return fam, out


def rank_agreement(fam):
    """Kendall tau between the mean-rank vectors of the primary measure and every other metric."""
    out = {}
    for cond in CONDITIONS:
        ref = fam[(fam.metric == "audc_bal") & (fam.condition == cond)].ranks.iloc[0]
        for metric in ALL_METRICS[1:]:
            other = fam[(fam.metric == metric) & (fam.condition == cond)].ranks.iloc[0]
            out[f"{metric}|{cond}"] = float(stats.kendalltau([ref[m] for m in MODELS], [other[m] for m in MODELS])[0])
    vals = list(out.values())
    same_leader = {}
    for cond in CONDITIONS:
        sub = fam[fam.condition == cond].set_index("metric").best
        same_leader[cond] = sorted(set(sub[ALL_METRICS]))
    return {"tau": out, "tau_min": float(min(vals)), "tau_median": float(np.median(vals)),
            "leaders_by_condition": same_leader,
            "amrr_vs_mrr_tau": {c: out_tau(fam, "amrr", "mrr", c) for c in CONDITIONS}}


def out_tau(fam, a, b, cond):
    ra = fam[(fam.metric == a) & (fam.condition == cond)].ranks.iloc[0]
    rb = fam[(fam.metric == b) & (fam.condition == cond)].ranks.iloc[0]
    return float(stats.kendalltau([ra[m] for m in MODELS], [rb[m] for m in MODELS])[0])


def clean_vs_robust(traj):
    """Clean accuracy as a predictor of robustness: per dataset and condition, is the most accurate clean
    model also the most robust one, and how do clean and robustness ranks correlate?"""
    out = {}
    for cond in CONDITIONS:
        sub = traj[traj.condition == cond]
        same, taus = {"audc_bal": [], "mrr": [], "audc": []}, {"audc_bal": [], "mrr": [], "audc": []}
        for _, g in sub.groupby("dataset"):
            for metric in same:
                same[metric].append(g.acc0.idxmax() == g[metric].idxmax())
                taus[metric].append(stats.kendalltau(g.acc0, g[metric])[0])
        out[cond] = {m: {"same_best_share": float(np.mean(same[m])), "tau_median": float(np.nanmedian(taus[m]))}
                     for m in same}
    out["same_best_share_mrr_overall"] = float(np.mean([out[c]["mrr"]["same_best_share"] for c in CONDITIONS]))
    out["same_best_share_audc_bal_overall"] = float(np.mean([out[c]["audc_bal"]["same_best_share"]
                                                             for c in CONDITIONS]))
    out["tau_mrr_min"] = float(min(out[c]["mrr"]["tau_median"] for c in CONDITIONS))
    out["tau_mrr_max"] = float(max(out[c]["mrr"]["tau_median"] for c in CONDITIONS))
    return out


def knee_reliability(by_seed, bases):
    """Seed-to-seed agreement of the spline knee and of the discrete knee on real trajectories."""
    t = trajectories(by_seed, ["seed", "condition", "dataset", "model"], bases)
    out = {}
    for est in ["eta_knee", "eta_knee_tier", "eta_discrete", "audc_bal", "mrr"]:
        wide = t.pivot_table(index=["condition", "dataset", "model"], columns="seed", values=est)
        rhos, within = [], []
        for a, b in itertools.combinations(wide.columns, 2):
            ok = wide[[a, b]].dropna()
            rhos.append(stats.spearmanr(ok[a], ok[b])[0])
            within.append(float(np.mean(np.abs(ok[a] - ok[b]) <= TIER + 1e-9)))
        out[est] = {"spearman_min": float(min(rhos)), "spearman_max": float(max(rhos)),
                    "within_one_tier_mean": float(np.mean(within))}
    return out


def knee_resolution(traj):
    k = traj.eta_knee.dropna()
    on_tier = np.abs(k - np.round(k / TIER) * TIER) <= 0.002
    agree = (traj.eta_knee_tier - traj.eta_discrete).abs().dropna()
    pooled = {c: {"median_tier": float(traj[traj.condition == c].eta_knee_tier.median()),
                  "q25": float(traj[traj.condition == c].eta_knee_tier.quantile(0.25)),
                  "q75": float(traj[traj.condition == c].eta_knee_tier.quantile(0.75)),
                  "median_discrete": float(traj[traj.condition == c].eta_discrete.median())}
              for c in CONDITIONS}
    return {"on_tier_share": float(on_tier.mean()), "n_defined": int(len(k)),
            "spline_vs_discrete_within_one_tier": float((agree <= TIER + 1e-9).mean()),
            "spline_vs_discrete_exact": float((agree <= 1e-9).mean()), "pooled": pooled}


def load_block(name):
    files = [os.path.join(REV_METRICS, f) for f in os.listdir(REV_METRICS) if f.startswith(name + "__")]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def seed_mean(df, keys):
    return df.groupby(keys + ["eta"], as_index=False)[["mean_acc", "mean_bal_acc"]].mean()


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def principal_deploy(dep):
    """Protocol B with the Gaussian condition = continuous-only noise (all-column cells replaced on the six
    datasets with nominal attributes); see gaussian_principal.py."""
    return substitute_continuous(dep, load_cont("deploy_cont"), "modality", seeds=sorted(dep.seed.unique()))


def allcol_primary(bases):
    """Protocol A trajectories of the all-column Gaussian operator (stress test) on all twenty datasets: the
    primary cells on the fourteen datasets without nominal attributes plus the replaced all-column cells."""
    from gaussian_principal import cat_datasets
    prim = pd.read_csv(os.path.join(PRIMARY, "cells.csv"))
    by_seed = pd.read_csv(os.path.join(PRIMARY, "cells_by_seed.csv"))
    allcol = pd.read_csv(os.path.join(PRIMARY, "cells_gaussian_allcol_by_seed.csv"))
    dsets = cat_datasets()
    keep = by_seed[(by_seed.condition == GAUSS_OP) & ~by_seed.dataset.isin(dsets)]
    clean = by_seed[(by_seed.condition == GAUSS_OP) & by_seed.dataset.isin(dsets) & (by_seed.eta == 0)]
    mine = pd.concat([keep, clean, allcol], ignore_index=True).assign(condition=GAUSS_OP)
    cells = seed_mean(mine, ["condition", "dataset", "model"])
    return trajectories(cells, ["condition", "dataset", "model"], bases), \
        pd.concat([by_seed[by_seed.condition != GAUSS_OP], mine], ignore_index=True)


def deploy_conditions(dep):
    dep = dep[dep.modality != "type_aware_feature"]      # superseded by the categorical-aware operators
    """Protocol B trajectories as named conditions; under native masking the MLP keeps its imputer, as
    in the primary grid."""
    names = {"gaussian_feature": "gaussian_feature", "type_aware_feature": "type_aware_feature",
             "mcar_missing": "mcar", "mar_missing": "mar", "mnar_missing": "mnar"}
    parts = []
    for (pipe, op), g in dep.groupby(["pipeline", "modality"]):
        base = names[op]
        if pipe == "imputed":
            cond = base if op in ("gaussian_feature", "type_aware_feature") else f"{base}_imputed"
            parts.append(g.assign(condition=cond))
            if op in ("mcar_missing", "mar_missing", "mnar_missing"):
                parts.append(g[g.model == "mlp"].assign(condition=f"{base}_native"))
        else:
            parts.append(g.assign(condition=f"{base}_native"))
    return pd.concat(parts, ignore_index=True)


def paired_shift(ta, tb, metric):
    m = ta.merge(tb, on=["dataset", "model"], suffixes=("_a", "_b"))
    diff = m[f"{metric}_b"] - m[f"{metric}_a"]
    p = stats.wilcoxon(diff, zero_method="pratt").pvalue if (diff != 0).any() else 1.0
    return {"median_delta": float(diff.median()), "share_b_lower": float((diff < 0).mean()),
            "p_wilcoxon": float(p), "n_pairs": int(len(diff))}


def rank_tau(fa, fb, metric):
    ra, rb = fa[fa.metric == metric].ranks.iloc[0], fb[fb.metric == metric].ranks.iloc[0]
    return float(stats.kendalltau([ra[m] for m in MODELS], [rb[m] for m in MODELS])[0])


def comparison_summary(comp):
    """Counts over every (condition, score) block of a protocol or configuration comparison."""
    cells = [comp[c][m] for c in comp for m in comp[c]]
    return {"n_blocks": len(cells), "same_best": int(sum(r["best_a"] == r["best_b"] for r in cells)),
            "same_worst": int(sum(r["worst_a"] == r["worst_b"] for r in cells)),
            "tau_min": float(min(r["tau"] for r in cells)), "tau_median": float(np.median([r["tau"] for r in cells])),
            "n_lower_sig": int(sum(r["median_delta"] < 0 and r["p_wilcoxon"] < ALPHA for r in cells)),
            "n_higher_sig": int(sum(r["median_delta"] > 0 and r["p_wilcoxon"] < ALPHA for r in cells)),
            "max_abs_median_delta": float(max(abs(r["median_delta"]) for r in cells)),
            "n_cart_worst_both": int(sum(r["worst_a"] == r["worst_b"] == "cart" for r in cells))}


def protocol_comparison(traj_a, traj_b, pairs):
    """Protocol A versus Protocol B on matched conditions: paired shift of each score, rank agreement and
    leaders under both protocols."""
    out = {}
    for cond_a, cond_b in pairs:
        ta = traj_a[traj_a.condition == cond_a]
        tb = traj_b[traj_b.condition == cond_b]
        fa = friedman_family(ta, CONFIRMATORY, [cond_a])
        fb = friedman_family(tb, CONFIRMATORY, [cond_b])
        rec = {}
        for metric in CONFIRMATORY:
            rec[metric] = paired_shift(ta, tb, metric)
            rec[metric].update(tau=rank_tau(fa, fb, metric),
                               best_a=fa[fa.metric == metric].best.iloc[0],
                               best_b=fb[fb.metric == metric].best.iloc[0],
                               worst_a=fa[fa.metric == metric].worst.iloc[0],
                               worst_b=fb[fb.metric == metric].worst.iloc[0])
        out[cond_b] = rec
    return out


def deployment(primary_traj, bases):
    dep = load_block("deploy")
    if dep is None:
        return None, None
    raw = deploy_conditions(principal_deploy(dep))
    cells = seed_mean(raw, ["condition", "dataset", "model"])
    complete = raw.groupby("condition").size()
    full = [c for c, n in complete.items() if n == 20 * 6 * 11 * 3]
    if not full:
        return None, None
    tb = trajectories(cells[cells.condition.isin(full)], ["condition", "dataset", "model"], bases)
    fam = friedman_family(tb, ALL_METRICS, sorted(full))
    fam["p_holm"] = holm(fam.p)
    seeds = sorted(int(s) for s in dep.seed.unique())
    pairs = [(c, c) for c in ["gaussian_feature", "mcar_imputed", "mcar_native"] if c in full]
    comp = protocol_comparison(primary_traj, tb, pairs)
    out = {"seeds": seeds, "conditions": sorted(full), "comparison": comp, "summary": comparison_summary(comp),
           "tests": {f"{r.metric}|{r.condition}": dict(best=r.best, worst=r.worst, F_ID=float(r.F_ID),
                                                       p_holm=float(r.p_holm), n_wilcoxon_sig=int(r.n_wilcoxon_sig))
                     for r in fam.itertuples()},
           "n_sig_holm": int((fam.p_holm < ALPHA).sum()), "n_tests": int(len(fam)),
           "terminal": {c: float(tb[tb.condition == c].acc_end.median()) for c in full},
           "audc_bal_median": {c: float(tb[tb.condition == c].audc_bal.median()) for c in full},
           "mrr_median": {c: float(tb[tb.condition == c].mrr.median()) for c in full}}
    out["native_vs_imputed"] = native_vs_imputed_b(tb, full)
    out["clean_vs_robust"] = {c: float(np.mean([g.acc0.idxmax() == g.mrr.idxmax()
                                                for _, g in tb[tb.condition == c].groupby("dataset")]))
                              for c in full}
    return out, tb


def native_vs_imputed_b(tb, conditions):
    """Under Protocol B: native minus imputed AUDC per tree family and mechanism, Wilcoxon with Holm over all
    tests of the block."""
    rows = []
    for mech in ["mcar", "mar", "mnar"]:
        if f"{mech}_native" not in conditions or f"{mech}_imputed" not in conditions:
            continue
        nat = tb[tb.condition == f"{mech}_native"].set_index(["dataset", "model"])
        imp = tb[tb.condition == f"{mech}_imputed"].set_index(["dataset", "model"])
        for m in MODELS[:5]:
            d = (nat.xs(m, level="model").audc - imp.xs(m, level="model").audc)
            rows.append(dict(mech=mech, model=m, median_delta=float(d.median()), share_native_lower=float((d < 0).mean()),
                             p=float(stats.wilcoxon(d, zero_method="pratt").pvalue)))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p)
    out = {f"{r.model}|{r.mech}": dict(median_delta=r.median_delta, share_native_lower=r.share_native_lower,
                                        p_holm=float(r.p_holm)) for r in df.itertuples()}
    boost = df[df.model.isin(["xgboost", "lightgbm", "catboost"])]
    out["summary"] = {"n_tests": int(len(df)), "n_native_worse_sig": int(((df.median_delta < 0) & (df.p_holm < ALPHA)).sum()),
                      "n_native_better_sig": int(((df.median_delta > 0) & (df.p_holm < ALPHA)).sum()),
                      "worst_delta": float(df.median_delta.min()), "worst": df.loc[df.median_delta.idxmin(), ["model", "mech"]].tolist(),
                      "boosters_max_delta": float(boost.median_delta.max()),
                      "rf_mcar_delta": float(df[(df.model == "randomforest") & (df.mech == "mcar")].median_delta.iloc[0])}
    return out


def mechanism_rows(tm):
    """Analysable Protocol A rows of the MAR and MNAR block, and the seeds that are complete.

    The superseded type-aware rows are dropped and resumed runs are de-duplicated. A seed counts only when it holds
    every planned cell (all datasets x models x operators x noisy tiers), so a partially written seed file
    (seeds 43 and 44 are appended while this runs) never enters the analysis."""
    from experiments.run_extended import MODELS as models, TIERS as tiers
    from experiments.run_revision import TRAIN_MECH_EXCLUDED, TRAIN_MECH_OPS
    from src.data_loader import BENCHMARK_SUITE
    tm = tm[tm.modality.isin(TRAIN_MECH_OPS) & ~tm.dataset.isin(TRAIN_MECH_EXCLUDED) & (tm.eta > 0)].copy()
    tm = tm.drop_duplicates(["seed", "dataset", "model", "modality", "eta"], keep="last")
    planned = len([d for d in BENCHMARK_SUITE if d not in TRAIN_MECH_EXCLUDED]) \
        * len(models) * len(TRAIN_MECH_OPS) * len([e for e in tiers if e > 0])
    counts = tm.groupby("seed").size()
    seeds = sorted(int(s) for s, n in counts.items() if n == planned)
    return tm[tm.seed.isin(seeds)], seeds


def train_mechanisms(dep_traj, bases):
    """Protocol A under MAR and MNAR masking against Protocol B on the same datasets, folds and seeds. Both are
    the mean over the complete seeds (one seed while only seed 42 is finished)."""
    from experiments.run_revision import TRAIN_MECH_EXCLUDED
    tm = load_block("train_mech")
    dep = load_block("deploy")
    if tm is None or dep is None:
        return None
    tm, seeds = mechanism_rows(tm)
    if not seeds:
        return {"complete": False}
    dep = dep[dep.seed.isin(seeds)]
    clean = dep[(dep.pipeline == "imputed") & (dep.eta == 0) & (dep.modality == "gaussian_feature")]
    names = {"mar_missing": "mar_imputed", "mnar_missing": "mnar_imputed"}
    parts = [pd.concat([g, clean.assign(modality=op)]).assign(condition=names[op])
             for op, g in tm.groupby("modality")]
    cells = seed_mean(pd.concat(parts, ignore_index=True), ["condition", "dataset", "model"])
    done = sorted(cells.dataset.unique())
    n_data = len(done)
    counts = cells.groupby("condition").size()
    full = [c for c, n in counts.items() if n == n_data * 6 * 11]
    if not full:
        return {"complete": False}
    ta = trajectories(cells[cells.condition.isin(full)], ["condition", "dataset", "model"], bases)
    dep_mean = seed_mean(deploy_conditions(dep[dep.dataset.isin(done)]), ["condition", "dataset", "model"])
    tb = trajectories(dep_mean[dep_mean.condition.isin(full)], ["condition", "dataset", "model"], bases)
    fam = friedman_family(ta, CONFIRMATORY, full)
    fam["p_holm"] = holm(fam.p)
    return {"complete": True, "conditions": full, "n_datasets": n_data, "datasets": done,
            "seeds": seeds, "n_seeds": len(seeds),
            "all_planned_done": n_data == 20 - len(TRAIN_MECH_EXCLUDED),
            "max_p_holm_audc_bal": float(fam[fam.metric == "audc_bal"].p_holm.max()),
            "max_p_holm": float(fam.p_holm.max()),
            "tests": {f"{r.metric}|{r.condition}": dict(best=r.best, worst=r.worst, p_holm=float(r.p_holm),
                                                             ranks=r.ranks)
                      for r in fam.itertuples()},
            "comparison": (comp := protocol_comparison(ta, tb, [(c, c) for c in full])),
            "summary": comparison_summary(comp),
            "audc_bal_median_a": {c: float(ta[ta.condition == c].audc_bal.median()) for c in full},
            "audc_bal_median_b": {c: float(tb[tb.condition == c].audc_bal.median()) for c in full}}


def categorical_analysis(allcol_traj, bases):
    """Categorical-aware (Gaussian on continuous + whole-attribute resampling of nominal attributes) and
    categorical-only corruption, both protocols, seeds 42-44, on the datasets with nominal attributes; compared
    with the all-column Gaussian operator (Gaussian noise on the integer codes of the nominal attributes too) on
    the same datasets, folds and seeds."""
    from experiments.run_revision import cat_datasets
    dsets = cat_datasets()
    dc, tc, dep = load_block("deploy_cat"), load_block("train_cat"), load_block("deploy")
    if dc is None or tc is None:
        return None
    n_exp = len(dsets) * 6 * 3
    if dc.groupby(["dataset", "model", "seed"]).ngroups < n_exp or len(tc) < n_exp * 2 * 10:
        return {"complete": False}
    tb = trajectories(seed_mean(dc, ["modality", "dataset", "model"]).rename(columns={"modality": "condition"}),
                      ["condition", "dataset", "model"], bases)
    clean = dep[(dep.pipeline == "imputed") & (dep.eta == 0) & (dep.modality == "gaussian_feature")
                & dep.dataset.isin(dsets)]
    parts = [pd.concat([g, clean.assign(modality=op)]).assign(condition=op) for op, g in tc.groupby("modality")]
    ta = trajectories(seed_mean(pd.concat(parts, ignore_index=True), ["condition", "dataset", "model"]),
                      ["condition", "dataset", "model"], bases)
    gb = deploy_conditions(dep[dep.dataset.isin(dsets)])
    gb = trajectories(seed_mean(gb[gb.condition == "gaussian_feature"], ["condition", "dataset", "model"]),
                      ["condition", "dataset", "model"], bases)
    ga = allcol_traj[allcol_traj.dataset.isin(dsets)]
    out = {"complete": True, "datasets": dsets, "n_datasets": len(dsets),
           "n_attributes": {d: len(__import__("experiments.run_revision", fromlist=["x"]).categorical_groups(d))
                            for d in dsets}}
    for proto, t, g in [("A", ta, ga), ("B", tb, gb)]:
        rec = {}
        for op in ["cat_aware_feature", "cat_only_feature"]:
            sub = t[t.condition == op]
            rec[op] = {"audc_bal_median": float(sub.audc_bal.median()), "mrr_median": float(sub.mrr.median()),
                       "terminal_loss_median": float((sub.acc0 - sub.acc_end).median()),
                       "best_audc_bal": sub.groupby("model").audc_bal.mean().idxmax(),
                       "worst_audc_bal": sub.groupby("model").audc_bal.mean().idxmin()}
            m = sub.merge(g, on=["dataset", "model"], suffixes=("", "_g"))
            d = m.audc_bal - m.audc_bal_g
            rec[op]["vs_gaussian_median_delta"] = float(d.median())
            rec[op]["vs_gaussian_share_higher"] = float((d > 0).mean())
            rec[op]["vs_gaussian_p"] = float(stats.wilcoxon(d, zero_method="pratt").pvalue) if (d != 0).any() else 1.0
        rec["gaussian_audc_bal_median"] = float(g.audc_bal.median())
        out[proto] = rec
    ta.to_csv(os.path.join(OUT, "trajectories_categorical_A.csv"), index=False)
    tb.to_csv(os.path.join(OUT, "trajectories_categorical_B.csv"), index=False)
    return out


PRINCIPAL = "gaussian_principal"
CONT_OP = "gauss_cont_feature"


def continuous_gaussian_trajectories(bases):
    """Continuous-only Gaussian trajectories on the six datasets with nominal attributes, both protocols.
    Clean (eta = 0) rows come from the Protocol B clean fits, as in categorical_analysis."""
    from experiments.run_revision import cat_datasets
    dsets = cat_datasets()
    dc, tc, dep = load_block("deploy_cont"), load_block("train_cont"), load_block("deploy")
    if dc is None or tc is None or dep is None:
        return None, None, dsets
    n_exp = len(dsets) * 6 * 3
    if dc.groupby(["dataset", "model", "seed"]).ngroups < n_exp or len(tc) < n_exp * 10:
        return None, None, dsets
    keys = ["condition", "dataset", "model"]
    tb = trajectories(seed_mean(dc, ["modality", "dataset", "model"]).rename(columns={"modality": "condition"}),
                      keys, bases)
    clean = dep[(dep.pipeline == "imputed") & (dep.eta == 0) & (dep.modality == "gaussian_feature")
                & dep.dataset.isin(dsets)]
    ta = trajectories(seed_mean(pd.concat([tc, clean.assign(modality=CONT_OP)], ignore_index=True),
                                ["modality", "dataset", "model"]).rename(columns={"modality": "condition"}),
                      keys, bases)
    return ta, tb, dsets


def six_dataset_shift(cont, all_column, dsets):
    """Median audc_bal of the continuous-only and all-column operators on the six nominal datasets, paired by
    (dataset, model), with the Wilcoxon p of the paired difference."""
    sub = all_column[all_column.dataset.isin(dsets)]
    m = cont.merge(sub, on=["dataset", "model"], suffixes=("_c", "_a"))
    diff = m.audc_bal_c - m.audc_bal_a
    p = stats.wilcoxon(diff, zero_method="pratt").pvalue if (diff != 0).any() else 1.0
    return {"median_cont": float(m.audc_bal_c.median()), "median_all": float(m.audc_bal_a.median()),
            "median_delta": float(diff.median()), "p": float(p)}


def omnibus_record(fam):
    """Friedman / Iman-Davenport omnibus for audc_bal and mrr of one condition, with audc_bal mean ranks."""
    rec = {}
    for metric in ["audc_bal", "mrr"]:
        row = fam[fam.metric == metric].iloc[0]
        rec[metric] = {"F": float(row.F_ID), "p": float(row.p), "best": row.best, "worst": row.worst}
    rec["audc_bal"]["mean_ranks"] = {m: float(v) for m, v in fam[fam.metric == "audc_bal"].iloc[0].ranks.items()}
    return rec


def principal_record(cont, all_column, dsets):
    """Principal condition over all twenty datasets (continuous-only on the six, all-column elsewhere) against
    the all-column operator on the same twenty datasets, one protocol."""
    others = all_column[~all_column.dataset.isin(dsets)]
    principal = pd.concat([others.assign(condition=PRINCIPAL), cont.assign(condition=PRINCIPAL)],
                          ignore_index=True)
    metrics = ["audc_bal", "mrr"]
    fp = friedman_family(principal, metrics, [PRINCIPAL])
    fa = friedman_family(all_column.assign(condition="gaussian_feature"), metrics, ["gaussian_feature"])
    rec = omnibus_record(fp)
    ranks_p = fp[fp.metric == "audc_bal"].iloc[0].ranks
    ranks_a = fa[fa.metric == "audc_bal"].iloc[0].ranks
    rec["all_column"] = omnibus_record(fa)
    rec["tau_vs_all_column_audc_bal"] = float(stats.kendalltau([ranks_p[m] for m in MODELS],
                                                              [ranks_a[m] for m in MODELS])[0])
    rec["same_best"] = bool(rec["audc_bal"]["best"] == fa[fa.metric == "audc_bal"].iloc[0].best)
    rec["same_worst"] = bool(rec["audc_bal"]["worst"] == fa[fa.metric == "audc_bal"].iloc[0].worst)
    rec["six"] = six_dataset_shift(cont, all_column, dsets)
    # a dataset without continuous attributes is not perturbed at all by the principal operator: a flat block
    nocont = no_continuous_datasets()
    rec["no_continuous_datasets"] = nocont
    sub = principal[~principal.dataset.isin(nocont)]
    rec["excluding_no_continuous"] = omnibus_record(friedman_family(sub, metrics, [PRINCIPAL]))
    rec["excluding_no_continuous"]["n_datasets"] = int(sub.dataset.nunique())
    return rec, principal


def continuous_gaussian_analysis(allcol_traj, bases):
    """Principal Gaussian experiment: Gaussian noise on continuous attributes only. On the six datasets with
    nominal attributes this differs from the all-column operator; elsewhere the two are identical, so the
    all-column trajectories are reused. Protocols A and B, seeds 42-44, twenty datasets. This is the check that
    the primary Gaussian condition (continuous-only on the six datasets) is built from the same cells, and the
    comparison of the principal condition with the all-column stress test."""
    cont_a, cont_b, dsets = continuous_gaussian_trajectories(bases)
    dep = load_block("deploy")
    if cont_a is None or dep is None:
        return {"complete": False}
    raw_b = deploy_conditions(dep)
    gb_all = trajectories(seed_mean(raw_b[raw_b.condition == "gaussian_feature"],
                                    ["condition", "dataset", "model"]),
                          ["condition", "dataset", "model"], bases)
    ga_all = allcol_traj
    out = {"complete": True, "n_datasets_cont": len(dsets), "n_datasets": int(ga_all.dataset.nunique())}
    for proto, cont, all_column in [("A", cont_a, ga_all), ("B", cont_b, gb_all)]:
        out[proto], principal = principal_record(cont, all_column, dsets)
        principal.to_csv(os.path.join(OUT, f"trajectories_gauss_principal_{proto}.csv"), index=False)
        cont.to_csv(os.path.join(OUT, f"trajectories_gauss_cont_{proto}.csv"), index=False)
    return out


def tuned_analysis(by_seed, bases):
    """Tuned configurations (seed 42) against the matched budget (seed 42) on the same folds."""
    tu = load_block("tuned")
    if tu is None:
        return None
    from experiments.run_revision import TUNED_EXCLUDED
    tu = tu[~tu.dataset.isin(TUNED_EXCLUDED)].copy()
    n_data = tu.dataset.nunique()
    tu.loc[tu.protocol == "B", "condition"] = "gaussian_feature_B"
    # run_tuned evaluates the clean fit once per operator; both copies are identical, one is kept
    clean_a = tu[(tu.protocol == "A") & (tu.modality == "clean")].drop_duplicates(["dataset", "model"])
    parts = [tu[tu.condition == "gaussian_feature_B"]]
    for op in ["symmetric_label", "gaussian_feature"]:
        parts.append(pd.concat([tu[(tu.protocol == "A") & (tu.modality == op)], clean_a]).assign(condition=op))
    cells = pd.concat(parts, ignore_index=True)
    per_ds = cells.groupby("dataset").size()
    done = sorted(per_ds[per_ds == 3 * 6 * 11].index)
    cells = cells[cells.dataset.isin(done)]
    n_data = len(done)
    counts = cells.groupby("condition").size()
    full = [c for c, n in counts.items() if n == n_data * 6 * 11] if n_data >= 10 else []
    if not full:
        return {"complete": False}
    t_tuned = trajectories(cells[cells.condition.isin(full)], ["condition", "dataset", "model"], bases)
    matched = by_seed[(by_seed.seed == 42) & by_seed.condition.isin(["symmetric_label", "gaussian_feature"])
                      & by_seed.dataset.isin(done)]
    t_matched = trajectories(matched, ["condition", "dataset", "model"], bases)
    dep = load_block("deploy")
    dep42 = seed_mean(deploy_conditions(dep[(dep.seed == 42)]), ["condition", "dataset", "model"])
    t_mb = trajectories(dep42[(dep42.condition == "gaussian_feature") & dep42.dataset.isin(done)],
                        ["condition", "dataset", "model"], bases)
    t_matched = pd.concat([t_matched, t_mb.assign(condition="gaussian_feature_B")], ignore_index=True)
    params = tu.drop_duplicates(["dataset", "model"])
    out = {"complete": True, "conditions": full, "trials": 6, "n_datasets": int(n_data), "datasets": done,
           "comparison": (comp := protocol_comparison(t_matched, t_tuned, [(c, c) for c in full])),
           "summary": comparison_summary(comp),
           "acc0_gain_median": {m: float((t_tuned[(t_tuned.model == m) & (t_tuned.condition == full[0])]
                                          .set_index("dataset").acc0 -
                                          t_matched[(t_matched.model == m) & (t_matched.condition == full[0])]
                                          .set_index("dataset").acc0).median()) for m in MODELS},
           "n_configs": int(len(params))}
    gains = list(out["acc0_gain_median"].values())
    out["acc0_gain_min"], out["acc0_gain_max"] = float(min(gains)), float(max(gains))
    return out


def artifacts(primary_traj, bases):
    """Rank tests with the three cleaned datasets substituted, and with the flagged datasets excluded. The cleaned
    runs exist for the all-column Gaussian operator only, so this block takes the all-column Gaussian cells as its
    baseline (like for like); the other four conditions are the primary ones."""
    out = {}
    flagged = ["spambase", "churn", "miniboone", "amazon_access"]
    base_fam = friedman_family(primary_traj, CONFIRMATORY, CONDITIONS)
    excl = friedman_family(primary_traj[~primary_traj.dataset.isin(flagged)], CONFIRMATORY, CONDITIONS)
    out["excluded"] = summarize_change(base_fam, excl, flagged)
    files = [os.path.join(ART_METRICS, f) for f in os.listdir(ART_METRICS)] if os.path.isdir(ART_METRICS) else []
    if not files:
        return out
    art = pd.concat([pd.read_csv(f) for f in files if f.endswith(".csv")], ignore_index=True)
    if art.groupby("seed").size().min() < 888 or art.seed.nunique() < 3:
        out["cleaned_complete"] = False
        return out
    cells = artifact_conditions(art)
    t_clean = trajectories(cells, ["condition", "dataset", "model"], bases)
    t_clean["dataset"] = t_clean.dataset.map({v: k for k, v in ARTIFACT_SETS.items()})
    swapped = pd.concat([primary_traj[~primary_traj.dataset.isin(ARTIFACT_SETS)], t_clean], ignore_index=True)
    out["cleaned"] = summarize_change(base_fam, friedman_family(swapped, CONFIRMATORY, CONDITIONS), list(ARTIFACT_SETS))
    out["cleaned"]["max_abs_delta_audc"] = float(
        (t_clean.set_index(["condition", "dataset", "model"]).audc -
         primary_traj.set_index(["condition", "dataset", "model"]).audc).abs().max())
    out["cleaned_complete"] = True
    return out


def artifact_conditions(art):
    """Seed-averaged primary conditions for the cleaned datasets (same construction as build_primary)."""
    clean = art[art.modality == "clean"]
    m = {"symmetric_label": ("base", "symmetric_label"), "class_conditional": ("cc_binary", "minority_to_majority"),
         "gaussian_feature": ("base", "gaussian_feature"), "mcar_imputed": ("base", "mcar_missing"),
         "mcar_native": ("native_nan", "mcar_missing")}
    parts = []
    for cond, (variant, mod) in m.items():
        g = art[(art.variant == variant) & (art.modality == mod)]
        if cond == "mcar_native":
            g = pd.concat([g, art[(art.variant == "base") & (art.modality == "mcar_missing") & (art.model == "mlp")]])
        parts.append(pd.concat([g, clean]).assign(condition=cond))
    return seed_mean(pd.concat(parts, ignore_index=True), ["condition", "dataset", "model"])


def summarize_change(base_fam, new_fam, datasets):
    out = {"datasets": datasets, "same_best": 0, "same_worst": 0, "n_blocks": 0, "taus": [],
           "max_p_holm": float(holm(new_fam.p).max())}
    for (metric, cond), g in base_fam.groupby(["metric", "condition"]):
        h = new_fam[(new_fam.metric == metric) & (new_fam.condition == cond)].iloc[0]
        r0, r1 = g.ranks.iloc[0], h.ranks
        out["n_blocks"] += 1
        out["same_best"] += int(g.best.iloc[0] == h.best)
        out["same_worst"] += int(g.worst.iloc[0] == h.worst)
        out["taus"].append(float(stats.kendalltau([r0[m] for m in MODELS], [r1[m] for m in MODELS])[0]))
    out["tau_min"] = float(min(out["taus"]))
    out["n_sig_holm"] = int((holm(new_fam.p) < ALPHA).sum())
    return out


def accounting():
    """Every count quoted in the paper, derived from its factors (revision point 9)."""
    n_d, n_m, n_trees, n_bin, n_noisy, n_tiers = 20, 6, 5, 14, 10, 11
    original = n_d * n_m * 4 * n_tiers                       # seed 42, four original operators, eta = 0 per operator
    base_extra = 2 * n_d * n_m * (1 + 4 * n_noisy)           # seeds 43, 44: one shared clean cell + four operators
    cc_binary = 3 * n_bin * n_m * n_noisy                    # minority-to-majority, seeds 42-44
    native = 3 * n_d * n_trees * n_noisy                     # native masking, seeds 42-44
    defaults = 2 * n_d * (1 + 4 * n_noisy)                   # XGBoost and CatBoost defaults, seed 42
    primary_grid = n_d * n_m * 5 * n_tiers * 3
    out = {"original_seed42": original, "base_seeds_43_44": base_extra, "cc_binary": cc_binary,
           "native_nan": native, "library_defaults": defaults,
           "total_cells": original + base_extra + cc_binary + native + defaults,
           "primary_grid_cells": primary_grid, "primary_grid_cells_per_seed": primary_grid // 3}
    out["total_fits"] = 5 * out["total_cells"]
    # Of the computed cells, the primary analysis uses per seed one clean cell and 10 noisy tiers for each
    # of the five conditions (MLP rows of the native condition are its imputed rows): 5,920 distinct cells.
    clean = n_d * n_m
    noisy_used = n_d * n_m * n_noisy * 3 + 6 * n_m * n_noisy + n_bin * n_m * n_noisy + n_d * n_trees * n_noisy
    out["primary_distinct_per_seed"] = clean + noisy_used
    out["primary_distinct"] = 3 * out["primary_distinct_per_seed"]
    out["pairflip_binary_test_retest"] = 3 * n_bin * n_m * n_noisy
    out["duplicate_clean_seed42"] = 3 * clean
    out["primary_reused_clean_rows"] = 3 * 4 * clean
    out["primary_reused_mlp_native_rows"] = 3 * n_d * (n_tiers - 1)
    assert out["primary_distinct"] + out["pairflip_binary_test_retest"] + out["duplicate_clean_seed42"] \
        + defaults == out["total_cells"]
    assert out["primary_distinct"] + out["primary_reused_clean_rows"] + out["primary_reused_mlp_native_rows"] \
        == primary_grid
    dep = load_block("deploy")
    if dep is not None:
        out["deploy_rows"] = int((dep.modality != "type_aware_feature").sum())
        out["deploy_fits"] = int(5 * dep.drop_duplicates(["seed", "dataset", "model", "pipeline"]).shape[0])
    from experiments.run_revision import TUNED_EXCLUDED
    mech = load_block("train_mech")
    if mech is not None:
        rows, mech_seeds = mechanism_rows(mech)
        out["train_mech_rows"] = int(len(rows))
        out["train_mech_seeds"] = mech_seeds
    tuned = load_block("tuned")
    if tuned is not None:
        # rows of datasets dropped from a block for compute reasons are kept on disk but not analysed
        out["tuned_rows"] = int(((~tuned.dataset.isin(TUNED_EXCLUDED)) & (tuned.seed == 42)
                                 & (tuned.modality != "type_aware_feature")).sum())
    for name in ["deploy_cat", "train_cat"]:
        blk = load_block(name)
        if blk is not None:
            out[f"{name}_rows"] = int(len(blk))
    return out


def latex_p(p):
    if p >= 1e-3:
        return f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def latex_combined(fam, path):
    """Table merging omnibus significance and mean ranks (revision point 34)."""
    nice = {"cart": "CART", "randomforest": "RF", "xgboost": "XGBoost", "lightgbm": "LightGBM",
            "catboost": "CatBoost", "mlp": "MLP"}
    mnames = {"audc_bal": "AUDC$_{\\mathrm{bal}}$", "audc": "AUDC", "mrr": "MRR", "mrr_bal": "MRR$_{\\mathrm{bal}}$",
              "amrr": "aMRR", "amrr_bal": "aMRR$_{\\mathrm{bal}}$"}
    cnames = {"symmetric_label": "Symmetric", "class_conditional": "Class-cond.", "gaussian_feature": "Gaussian",
              "mcar_imputed": "MCAR imp.", "mcar_native": "MCAR nat."}
    lines = ["\\begin{table*}[t]", "\\centering", "\\footnotesize",
             "\\caption{Cross-model comparison on the primary grid (Protocol~A, $N=20$ datasets, $k=6$ models, "
             "seed-averaged). Best and worst: lowest and highest mean Friedman rank (in parentheses). $p$: "
             "Iman--Davenport $p$-value (omnibus: do mean ranks differ?), Holm-adjusted over the confirmatory family of 15 "
             "tests (upper block) or over the sensitivity family, i.e.\ jointly over all 30 tests (lower block). $W$: Kendall's coefficient of concordance, the effect size of the Friedman "
             "test (0: no agreement among datasets, 1: identical ranking on every dataset). Pairs: model pairs "
             "separated by Holm-adjusted Wilcoxon / Nemenyi (CD$=1.686$) out of 15 (pairwise support). $W$ is the practical magnitude of the ordering; a small $p$ with low $W$ and few separated pairs is not a stable ordering. "
             "Metric interpretations: $\\AUDC_{\\mathrm{bal}}$ (primary) measures absolute endurance under class imbalance; "
             "$\\AUDC$ measures raw area under the degradation curve; $\\MRR$ measures relative accuracy retention normalised by clean performance ($A_0$); "
             "$\\MRR_{\\mathrm{bal}}$, $\\aMRR$ and $\\aMRR_{\\mathrm{bal}}$ are complementary views, not primary outcomes.}",
             "\\label{tab:combined}", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{@{}llllrrr@{}}", "\\toprule",
             "Metric & Corruption & Best (rank) & Worst (rank) & adj.\\ $p$ & $W$ & Pairs W/N \\\\", "\\midrule"]
    for role, block in [("confirmatory", CONFIRMATORY), ("sensitivity", SENSITIVITY)]:
        if role == "sensitivity":
            lines.append("\\midrule")
            lines.append("\\multicolumn{7}{@{}l}{\\emph{Sensitivity metrics (not confirmatory)}}\\\\")
        for metric in block:
            for cond in CONDITIONS:
                r = fam[(fam.metric == metric) & (fam.condition == cond)].iloc[0]
                p = r.p_holm_confirmatory if role == "confirmatory" else r.p_holm_all
                ptxt = latex_p(p)
                lines.append(f"{mnames[metric] if cond == CONDITIONS[0] else ''} & {cnames[cond]} & "
                             f"{nice[r.best]} ({r.best_rank:.2f}) & {nice[r.worst]} ({r.worst_rank:.2f}) & "
                             f"{ptxt} & {r.W:.2f} & {r.n_wilcoxon_sig}/{r.n_nemenyi_sig} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    bases = dataset_baselines()
    cells = pd.read_csv(os.path.join(PRIMARY, "cells.csv"))
    by_seed = pd.read_csv(os.path.join(PRIMARY, "cells_by_seed.csv"))
    traj = trajectories(cells, ["condition", "dataset", "model"], bases)
    traj.to_csv(os.path.join(OUT, "trajectories_revision.csv"), index=False)
    allcol_traj, by_seed_allcol = allcol_primary(bases)
    fam, conf = confirmatory(traj)
    fam.to_csv(os.path.join(OUT, "confirmatory_family.csv"), index=False)
    latex_combined(fam, os.path.join(OUT, "tab_combined.tex"))
    rev = {"baselines": bases, "confirmatory": conf, "clean_vs_robust": clean_vs_robust(traj),
           "knee_resolution": knee_resolution(traj), "knee_seeds": knee_reliability(by_seed, bases),
           "accounting": accounting(), "artifacts": artifacts(pd.concat([traj[traj.condition != GAUSS_OP], allcol_traj], ignore_index=True), bases),
           "amrr_zeroed": int((traj.acc0 - traj.dataset.map(lambda d: bases[d]["pi_max"]) < MIN_HEADROOM).sum())}
    ra = os.path.join(OUT, "rerun_audit.json")
    if os.path.exists(ra):
        audit = json.load(open(ra))
        rev["rerun_audit"] = {k: audit[k] for k in ["n_cells", "max_abs_diff", "n_blocks"]}
    kv = os.path.join(OUT, "knee_validation.json")
    if os.path.exists(kv):
        rev["knee_validation"] = json.load(open(kv))["headline"]
    dep, tb = deployment(traj, bases)
    if dep is not None:
        rev["deploy"] = dep
        tb.to_csv(os.path.join(OUT, "trajectories_protocol_b.csv"), index=False)
        rev["mechanisms"] = train_mechanisms(tb, bases)
        rev["tuned"] = tuned_analysis(by_seed_allcol, bases)
        rev["categorical"] = categorical_analysis(allcol_traj, bases)
        rev["gauss_cont"] = continuous_gaussian_analysis(allcol_traj, bases)
    json.dump(json_safe(rev), open(os.path.join(OUT, "revision.json"), "w"), indent=1)
    print(json.dumps(json_safe({k: rev[k] for k in ["confirmatory", "clean_vs_robust", "knee_resolution",
                                                    "knee_seeds", "accounting", "artifacts"]}), indent=1)[:9000])


if __name__ == "__main__":
    main()
