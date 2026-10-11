"""Principal Gaussian-noise condition: additive Gaussian noise on the continuous attributes only.

On the datasets without nominal attributes the continuous-only operator is identical to the all-column
operator, so the existing all-column cells are reused. On the datasets with nominal attributes
(run_revision.cat_datasets) the all-column cells at eta > 0 are replaced by the continuous-only cells of the
same dataset, model, seed and eta; the clean (eta = 0) fits are shared. The all-column operator stays available
as the separately reported mathematical stress test (key ALLCOL_OP)."""

import os

import pandas as pd

REV_METRICS = "results/metrics/revision"
GAUSS_OP = "gaussian_feature"
CONT_OP = "gauss_cont_feature"
ALLCOL_OP = "gaussian_allcol"
KEYS = ["seed", "dataset", "model", "eta"]


def cat_datasets():
    from experiments.run_revision import cat_datasets as cats
    return cats()


def load_cont(block):
    """Continuous-only rows of one protocol (train_cont for A, deploy_cont for B), noisy tiers only."""
    files = [os.path.join(REV_METRICS, f) for f in sorted(os.listdir(REV_METRICS)) if f.startswith(block + "__")]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["eta"] = df["eta"].round(2)
    return df[df.eta > 0]


def substitute_continuous(df, cont, op_col, seeds=None):
    """`df` with its Gaussian (eta > 0) rows on the nominal-attribute datasets replaced by the matching
    continuous-only rows. Raises if any replaced cell has no matching continuous-only cell (so a seed or
    tier the continuous-only run lacks is never silently filled with all-column data)."""
    dsets = cat_datasets()
    df = df.copy()
    df["eta"] = df["eta"].round(2)
    gauss = (df[op_col] == GAUSS_OP) & df.dataset.isin(dsets) & (df.eta > 0)
    if seeds is not None:
        cont = cont[cont.seed.isin(seeds)]
    old = df[gauss]
    keys = ["seed", "dataset", "model", "eta"]
    cont = cont.drop_duplicates(keys)
    need = old.drop_duplicates(keys)[keys]
    have = need.merge(cont[keys], on=keys, how="left", indicator=True)
    missing = int((have["_merge"] != "both").sum())
    if missing:
        raise SystemExit(f"continuous-only Gaussian cells missing for {missing} primary cells; not substituting")
    new = cont.merge(need, on=keys)
    new[op_col] = GAUSS_OP
    cols = [c for c in df.columns]
    for c in cols:
        if c not in new.columns:
            new[c] = float("nan")
    return pd.concat([df[~gauss], new[cols]], ignore_index=True)


def allcol_rows(df, op_col):
    """The all-column Gaussian rows of the nominal-attribute datasets that the substitution replaces, under the
    stress-test key."""
    dsets = cat_datasets()
    return df[(df[op_col] == GAUSS_OP) & df.dataset.isin(dsets) & (df.eta.round(2) > 0)].assign(**{op_col: ALLCOL_OP})


def no_continuous_datasets():
    """Datasets on which the continuous-only operator corrupts nothing: every column is a nominal attribute."""
    from experiments.run_revision import categorical_groups, load_dataset
    out = []
    for d in cat_datasets():
        n_cols = load_dataset(d)[0].shape[1]
        n_nominal = int(sum(len(g) for g in categorical_groups(d)))
        if n_nominal >= n_cols:
            out.append(d)
    return out
