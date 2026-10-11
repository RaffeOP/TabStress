"""Primary analysis grid: the three seeds (42 from the original 5,280-cell run, 43 and 44
from run_extended.py) averaged cell by cell, over five corruption conditions.

  symmetric_label    symmetric label noise (all datasets)
  class_conditional  pair-flip noise on multiclass datasets, minority->majority noise on binary
                     datasets (where the pair-flip kernel would equal symmetric noise)
  gaussian_feature   additive Gaussian noise on the continuous attributes (principal Gaussian condition);
                     on the six datasets with nominal attributes the continuous-only cells of
                     results/metrics/revision/train_cont replace the all-column cells (see gaussian_principal.py)
  mcar_imputed       MCAR masking, median imputation for every model
  mcar_native        MCAR masking, native missing-value handling for the five tree families;
                     the MLP, which cannot take missing values, keeps median imputation

Writes results/analysis/primary/cells_by_seed.csv (one row per seed and cell) and
results/analysis/primary/cells.csv (seed mean, seed SD and per-seed accuracy), and
results/analysis/primary/cells_gaussian_allcol_by_seed.csv (the all-column Gaussian cells that the substitution
replaces, kept as the stress test).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.gaussian_principal import ALLCOL_OP, allcol_rows, load_cont, substitute_continuous

BASE = "results/metrics/benchmark_results.csv"
EXT = "results/metrics/extended"
OUT = "results/analysis/primary"
SEEDS = [42, 43, 44]
MODELS = ["cart", "randomforest", "xgboost", "lightgbm", "catboost", "mlp"]
TREES = MODELS[:5]
CONDITIONS = ["symmetric_label", "class_conditional", "gaussian_feature", "mcar_imputed", "mcar_native"]
METRICS = ["mean_acc", "mean_bal_acc", "mean_f1", "std_acc"]


def load_ext():
    frames = [pd.read_csv(os.path.join(EXT, f)) for f in sorted(os.listdir(EXT)) if f.endswith(".csv")]
    e = pd.concat(frames, ignore_index=True)
    e["eta"] = e["eta"].round(2)
    return e.drop_duplicates(["variant", "seed", "dataset", "model", "modality", "eta"])


def base_grid(base, ext, seed):
    """Original four modalities for one seed, eta = 0 included for every modality."""
    if seed == 42:
        g = base.copy()
    else:
        e = ext[(ext.variant == "base") & (ext.seed == seed)]
        clean = e[e.modality == "clean"]
        g = pd.concat([e[e.modality != "clean"]] + [clean.assign(modality=m) for m in
                                                     ["symmetric_label", "asymmetric_label", "gaussian_feature",
                                                      "mcar_missing"]], ignore_index=True)
    g = g.copy()
    g["eta"] = g["eta"].round(2)
    return g[["dataset", "model", "modality", "eta"] + METRICS].assign(seed=seed)


def conditions_for_seed(base, ext, seed, binary):
    g = base_grid(base, ext, seed)
    out = [g[g.modality == "symmetric_label"].assign(condition="symmetric_label"),
           g[g.modality == "gaussian_feature"].assign(condition="gaussian_feature"),
           g[g.modality == "mcar_missing"].assign(condition="mcar_imputed")]
    multi = g[(g.modality == "asymmetric_label") & ~g.dataset.isin(binary)]
    cc = ext[(ext.variant == "cc_binary") & (ext.seed == seed)][["dataset", "model", "eta"] + METRICS]
    clean = g[(g.eta == 0.0) & (g.modality == "symmetric_label")]
    cc0 = clean[clean.dataset.isin(binary)][["dataset", "model", "eta"] + METRICS]
    out.append(pd.concat([multi.drop(columns="modality"), cc.assign(seed=seed), cc0.assign(seed=seed)],
                         ignore_index=True).assign(condition="class_conditional"))
    nn = ext[(ext.variant == "native_nan") & (ext.seed == seed)][["dataset", "model", "eta"] + METRICS]
    nn0 = clean[clean.model.isin(TREES)][["dataset", "model", "eta"] + METRICS]
    mlp = g[(g.modality == "mcar_missing") & (g.model == "mlp")][["dataset", "model", "eta"] + METRICS]
    out.append(pd.concat([nn.assign(seed=seed), nn0.assign(seed=seed), mlp.assign(seed=seed)],
                         ignore_index=True).assign(condition="mcar_native"))
    res = pd.concat(out, ignore_index=True).drop(columns="modality", errors="ignore")
    return res[["seed", "condition", "dataset", "model", "eta"] + METRICS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true", help="build from whatever seeds are complete")
    args = ap.parse_args()
    base, ext = pd.read_csv(BASE), load_ext()
    binary = sorted(ext[ext.variant == "cc_binary"].dataset.unique())
    n_ds = base.dataset.nunique()
    expected = n_ds * 6 * 11
    frames, used, allcol = [], [], []
    cont_a = load_cont("train_cont")
    for seed in SEEDS:
        try:
            c = conditions_for_seed(base, ext, seed, binary)
        except Exception as exc:
            print(f"seed {seed}: unavailable ({exc!r})")
            continue
        counts = c.groupby("condition").size()
        complete = all(counts.get(k, 0) == expected for k in CONDITIONS) and \
            not c.duplicated(["condition", "dataset", "model", "eta"]).any()
        print(f"seed {seed}: {dict(counts)} complete={complete}")
        if complete:
            allcol.append(allcol_rows(c, "condition"))
            c = substitute_continuous(c, cont_a, "condition", seeds=[seed])
            frames.append(c)
            used.append(seed)
        elif not args.allow_partial:
            raise SystemExit(f"seed {seed} incomplete; rerun when run_extended.py has finished")
    if not frames:
        raise SystemExit("no complete seed")
    os.makedirs(OUT, exist_ok=True)
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(os.path.join(OUT, "cells_by_seed.csv"), index=False)
    pd.concat(allcol, ignore_index=True).assign(condition=ALLCOL_OP).to_csv(
        os.path.join(OUT, "cells_gaussian_allcol_by_seed.csv"), index=False)
    key = ["condition", "dataset", "model", "eta"]
    agg = long.groupby(key).agg(mean_acc=("mean_acc", "mean"), sd_seed=("mean_acc", "std"),
                                mean_bal_acc=("mean_bal_acc", "mean"), mean_f1=("mean_f1", "mean"),
                                fold_std=("std_acc", "mean"), n_seeds=("seed", "nunique")).reset_index()
    wide = long.pivot_table(index=key, columns="seed", values="mean_acc")
    wide.columns = [f"acc_seed{s}" for s in wide.columns]
    agg = agg.merge(wide.reset_index(), on=key)
    agg["sd_seed"] = agg["sd_seed"].fillna(0.0)
    agg.to_csv(os.path.join(OUT, "cells.csv"), index=False)
    print(f"seeds used: {used}; {len(agg)} cells over {len(CONDITIONS)} conditions -> {OUT}/cells.csv")


if __name__ == "__main__":
    main()
