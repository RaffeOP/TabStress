"""Audit of the MSNIE corruption operators on every benchmark dataset.

Measures what each operator actually does, against Definition 1 of the manuscript, on the
training folds of seed 42 (the same data and splits as the experiment):

  symmetric   realised flip rate against eta; replacements are other valid classes, spread
              uniformly over the K-1 alternatives
  pair-flip   (K >= 3) realised flip rate against eta; every flip goes to (k+1) mod K
  minority    (K = 2) realised flip rate within the minority class against eta; the other class
              is never touched; the minority class is the same in every fold
  gaussian    realised noise variance against eta^2 sigma_j^2 per column; columns constant in
              the corrupted matrix (sigma_j = 0), which the implementation perturbs with scale 1
  mcar        realised missing fraction against eta
  nesting     whether the corruption at eta = 0.2 is contained in the corruption at eta = 0.4,
              fold by fold, as a shared random stream implies

and checks the stored results: no (seed, condition, dataset) block may be constant across all
intensities for every model unless every model sits at the majority-class rate (the signature
of an operator that corrupts nothing).

Deviations are reported as z-scores against binomial (rates) or chi-square (variances) sampling
error; |z| > 4 fails. Writes results/analysis/audit_primary/operator_audit.json."""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(__file__))
from gaussian_principal import no_continuous_datasets
from run_extended import data
from src.data_loader import BENCHMARK_SUITE
from src.noise_injector import MSNIE

NO_CONTINUOUS = no_continuous_datasets()

SEED = 42
ETAS = [0.1, 0.3, 0.5]
Z_MAX = 4.0
OUT = "results/analysis/audit_primary/operator_audit.json"
BY_SEED = "results/analysis/primary/cells_by_seed.csv"
INVENTORY = "results/analysis/audit_primary/audit_numbers.json"


def rate_z(observed, nominal, n):
    return (observed - nominal) / np.sqrt(nominal * (1 - nominal) / n)


def folds(X, y):
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))


def label_checks(y_tr, k, minority):
    """Realised behaviour of the label operators on one training fold."""
    rec = {"sym_z": [], "sym_bad_label": 0, "sym_self": 0, "sym_uniform_p": [],
           "pf_z": [], "pf_wrong_target": 0, "m2m_z": [], "m2m_other_touched": 0}
    for eta in ETAS:
        noisy = MSNIE(SEED).inject_symmetric_label_noise(y_tr, eta, num_classes=k)
        changed = noisy != y_tr
        rec["sym_z"].append(rate_z(changed.mean(), eta, len(y_tr)))
        rec["sym_bad_label"] += int((~np.isin(noisy, np.arange(k))).sum())
        if k >= 3:
            offset = (noisy[changed] - y_tr[changed]) % k
            rec["sym_self"] += int((offset == 0).sum())
            counts = np.bincount(offset, minlength=k)[1:]
            rec["sym_uniform_p"].append(float(stats.chisquare(counts).pvalue))
            noisy = MSNIE(SEED).inject_asymmetric_label_noise_eta(y_tr, eta, num_classes=k)
            changed = noisy != y_tr
            rec["pf_z"].append(rate_z(changed.mean(), eta, len(y_tr)))
            rec["pf_wrong_target"] += int(((noisy[changed] - y_tr[changed]) % k != 1).sum())
        else:
            noisy = MSNIE(SEED).inject_minority_to_majority_noise(y_tr, eta, minority=minority)
            changed = noisy != y_tr
            in_m = y_tr == minority
            rec["m2m_z"].append(rate_z(changed[in_m].mean(), eta, int(in_m.sum())))
            rec["m2m_other_touched"] += int(changed[~in_m].sum())
    return rec


def feature_checks(X_tr):
    """Realised Gaussian noise variance and MCAR missing fraction on one training matrix."""
    n = X_tr.shape[0]
    sd = np.nanstd(X_tr, axis=0)
    live = sd > 0
    rec = {"gauss_z": [], "mcar_z": [], "const_cols": int((~live).sum()), "n_cols": int(X_tr.shape[1]),
           "const_noise_sd_ratio": []}
    for eta in ETAS:
        diff = MSNIE(SEED).inject_gaussian_feature_noise(X_tr, eta) - X_tr
        ratio = diff[:, live].var(axis=0) / (eta * sd[live]) ** 2
        rec["gauss_z"].append(float(np.max(np.abs(ratio - 1) / np.sqrt(2 / (n - 1)))))
        if (~live).any():   # documented implementation detail: constant columns get scale 1
            rec["const_noise_sd_ratio"].append(float(np.median(diff[:, ~live].std(axis=0) / eta)))
        masked = np.isnan(MSNIE(SEED).inject_mcar_missingness(X_tr, eta))
        rec["mcar_z"].append(rate_z(masked.mean(), eta, masked.size))
    return rec


def nesting(X, y, k, minority, fold_list):
    """Fold-by-fold containment of the corruption at eta = 0.2 in the corruption at eta = 0.4,
    replaying the order in which a cell consumes the random stream (run_extended.run_cell)."""
    lo, hi = 0.2, 0.4
    ops = ["symmetric", "pairflip" if k >= 3 else "minority", "gaussian", "mcar"]
    out = {op: [] for op in ops}
    for op in ops:
        inj_lo, inj_hi = MSNIE(SEED), MSNIE(SEED)
        for tr, va in fold_list:
            if op in ("symmetric", "pairflip", "minority"):
                f = {"symmetric": lambda inj, e: inj.inject_symmetric_label_noise(y[tr], e, num_classes=k),
                     "pairflip": lambda inj, e: inj.inject_asymmetric_label_noise_eta(y[tr], e, num_classes=k),
                     "minority": lambda inj, e: inj.inject_minority_to_majority_noise(y[tr], e, minority=minority)}[op]
                a, b = f(inj_lo, lo), f(inj_hi, hi)
                ch_a = a != y[tr]
                set_nested = bool(np.all((b != y[tr])[ch_a]))
                out[op].append(set_nested and bool(np.all(a[ch_a] == b[ch_a])))
                out.setdefault(op + "_set", []).append(set_nested)
            elif op == "gaussian":
                ok = True
                for M in (X[tr], X[va]):
                    da = inj_lo.inject_gaussian_feature_noise(M, lo) - M
                    db = inj_hi.inject_gaussian_feature_noise(M, hi) - M
                    ok &= bool(np.allclose(db, da * hi / lo))
                out[op].append(ok)
            else:
                ok = True
                for M in (X[tr], X[va]):
                    ma = np.isnan(inj_lo.inject_mcar_missingness(M, lo))
                    mb = np.isnan(inj_hi.inject_mcar_missingness(M, hi))
                    ok &= bool(np.all(mb[ma]))
                out[op].append(ok)
    return out


def flat_blocks(inventory):
    """(seed, condition, dataset) blocks in which every model is constant over all eleven tiers.
    'suspect' blocks have some model away from the majority-class rate: corruption that changed
    nothing; the others are blocks in which every model predicts the majority class throughout.
    Gaussian blocks of a dataset without any continuous attribute are flat by construction (the principal
    continuous-only operator has nothing to perturb there) and are listed apart."""
    if not os.path.exists(BY_SEED):
        return None
    by = pd.read_csv(BY_SEED)
    suspect, majority, no_continuous = [], [], []
    for (s, c, d), g in by.groupby(["seed", "condition", "dataset"]):
        spread = g.groupby("model").mean_acc.agg(lambda a: a.max() - a.min())
        if (spread < 1e-12).all():
            if c == "gaussian_feature" and d in NO_CONTINUOUS:
                no_continuous.append(f"{c}/{d}/seed{s}")
                continue
            off = (g.mean_acc - inventory[d]["majority_rate"]).abs().max()
            (suspect if off > 0.005 else majority).append(f"{c}/{d}/seed{s}")
    return {"suspect": suspect, "all_at_majority_rate": majority,
            "no_continuous_attributes": no_continuous, "n_blocks": int(by.groupby(["seed", "condition", "dataset"]).ngroups)}


def main():
    inventory = json.load(open(INVENTORY))["inventory"]
    per, nest = {}, {}
    for d in BENCHMARK_SUITE:
        X, y, k = data(d)
        X = X.astype(np.float64)
        fl = folds(X, y)
        minority = MSNIE.minority_class(y) if k == 2 else None
        tr = fl[0][0]
        per[d] = {**label_checks(y[tr], k, minority), **feature_checks(X[tr]), "K": k}
        const = [int((np.nanstd(X[idx], axis=0) == 0).sum()) for f in fl for idx in f]
        per[d]["const_cols_any_matrix_max"] = max(const)
        nest[d] = nesting(X, y, k, minority, fl)

    def zmax(key):
        vals = [abs(z) for v in per.values() for z in v[key]]
        return float(max(vals)) if vals else 0.0

    multi = [d for d in per if per[d]["K"] >= 3]
    binary = [d for d in per if per[d]["K"] == 2]
    share = lambda op, ds, sl=slice(None): float(np.mean([x for d in ds for x in nest[d].get(op, [])[sl]]))
    out = {
        "symmetric": {"max_abs_z": zmax("sym_z"), "invalid_labels": sum(v["sym_bad_label"] for v in per.values()),
                      "self_replacements": sum(v["sym_self"] for v in per.values()),
                      "uniformity_min_p": min(p for v in per.values() for p in v["sym_uniform_p"])},
        "pairflip": {"max_abs_z": zmax("pf_z"), "wrong_target": sum(v["pf_wrong_target"] for v in per.values()),
                     "n_datasets": len(multi)},
        "minority": {"max_abs_z": zmax("m2m_z"), "other_class_touched": sum(v["m2m_other_touched"] for v in per.values()),
                     "n_datasets": len(binary),
                     "tie_datasets": sorted(d for d in binary if inventory[d]["majority_rate"] == 0.5)},
        "gaussian": {"max_abs_z": zmax("gauss_z"),
                     "datasets_with_constant_columns": sorted(d for d in per if per[d]["const_cols_any_matrix_max"] > 0),
                     "constant_share_max": float(max(v["const_cols_any_matrix_max"] / v["n_cols"] for v in per.values())),
                     "constant_noise_sd_over_eta": [r for v in per.values() for r in v["const_noise_sd_ratio"]]},
        "mcar": {"max_abs_z": zmax("mcar_z")},
        "nesting": {"symmetric_binary_all_folds": share("symmetric", binary),
                    "symmetric_multiclass_first_fold": share("symmetric", multi, slice(0, 1)),
                    "symmetric_multiclass_later_folds": share("symmetric", multi, slice(1, None)),
                    "symmetric_multiclass_first_fold_set": share("symmetric_set", multi, slice(0, 1)),
                    "symmetric_multiclass_later_folds_set": share("symmetric_set", multi, slice(1, None)),
                    "pairflip": share("pairflip", multi), "minority": share("minority", binary),
                    "gaussian": share("gaussian", per), "mcar": share("mcar", per)},
        "flat_blocks": flat_blocks(inventory),
        "z_threshold": Z_MAX,
        "per_dataset": per,
    }
    out["label_max_abs_z"] = max(out["symmetric"]["max_abs_z"], out["pairflip"]["max_abs_z"], out["minority"]["max_abs_z"])
    out["all_pass"] = bool(
        out["label_max_abs_z"] <= Z_MAX and out["gaussian"]["max_abs_z"] <= Z_MAX and out["mcar"]["max_abs_z"] <= Z_MAX
        and out["symmetric"]["invalid_labels"] == 0 and out["symmetric"]["self_replacements"] == 0
        and out["symmetric"]["uniformity_min_p"] > 1e-3 and out["pairflip"]["wrong_target"] == 0
        and out["minority"]["other_class_touched"] == 0
        and out["flat_blocks"] is not None and out["flat_blocks"]["suspect"] == [])
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "per_dataset"}, indent=1, default=float))


if __name__ == "__main__":
    main()
