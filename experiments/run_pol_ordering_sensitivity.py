"""Sensitivity of the class-conditional (pair-flip) block to the class ordering on pol (revision point 38).

pol (OpenML 201) has an ordinal target: eleven values 0, 10, ..., 100. The loader label-encodes the target
as strings, so the encoded order is lexicographic ('0','10','100','20',...,'90') and the cyclic pair-flip
i -> (i+1) mod K corrupts e.g. 10 -> 100 and 100 -> 20. This script re-runs the pair-flip block on pol with
(a) the encoded (lexicographic) ordering used in the paper and (b) the numeric ordering 0 -> 10 -> ... -> 100 -> 0,
with the same protocol, folds, seeds, tiers and models as experiments/run_extended.py (Protocol A).
Output: results/metrics/revision/pol_ordering__seed<seed>.csv (same schema as the extended blocks, plus `ordering`)
and results/metrics/revision/pol_ordering_summary.json (cross-dataset tests with pol replaced)."""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(var, "1")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.run_extended import MODELS, TIERS, build, data
from src.data_loader import load_dataset
from src.noise_injector import MSNIE

OUT_DIR = "results/metrics/revision"
CELLS = "results/analysis/primary/cells.csv"
DATASET = "pol"
SEEDS = (42, 43, 44)
ORDERINGS = ("encoded", "numeric")
GRID_SPAN = 0.5


def class_rank_permutation(ordering):
    """rank[c] = position of encoded class c in the chosen cyclic order."""
    names = load_dataset(DATASET)[3]
    if ordering == "encoded":
        return np.arange(len(names))
    return np.argsort(np.argsort(names.astype(float)))


def run_cell(task):
    seed, model, ordering, eta = task
    X, y, k = data(DATASET)
    rank = class_rank_permutation(ordering)
    inverse = np.argsort(rank)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    inj = MSNIE(random_state=seed)
    acc, bal = [], []
    for tr, va in skf.split(X, y):
        ytr = inj.inject_asymmetric_label_noise_eta(rank[y[tr]], eta=eta, num_classes=k)
        ytr = inverse[ytr]
        clf = build("base", model, seed)
        clf.fit(X[tr], ytr)
        pred = np.asarray(clf.predict(X[va])).ravel().astype(y.dtype)
        acc.append(accuracy_score(y[va], pred))
        bal.append(balanced_accuracy_score(y[va], pred))
    return dict(seed=seed, dataset=DATASET, model=model, ordering=ordering, eta=eta,
                mean_acc=float(np.mean(acc)), mean_bal_acc=float(np.mean(bal)))


def run_all(workers):
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = [(s, m, o, e) for s in SEEDS for m in MODELS for o in ORDERINGS for e in TIERS]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_cell, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            rows.append(fut.result())
            if i % 20 == 0:
                print(i, len(tasks), time.strftime("%H:%M:%S"), flush=True)
    df = pd.DataFrame(rows)
    for seed, g in df.groupby("seed"):
        g.sort_values(["ordering", "model", "eta"]).to_csv(os.path.join(OUT_DIR, f"pol_ordering__seed{seed}.csv"), index=False)
    return df


def trajectory_scores(g):
    g = g.sort_values("eta")
    x, bal = g.eta.to_numpy(), g.mean_bal_acc.to_numpy()
    audc_bal = float(np.trapezoid(bal, x) / GRID_SPAN)
    return audc_bal, audc_bal / bal[0]


def cross_dataset_tests(df):
    """Friedman/Iman-Davenport over the 20 datasets for the class-conditional condition, with pol taken from each ordering."""
    cells = pd.read_csv(CELLS)
    cc = cells[cells.condition == "class_conditional"]
    out = {}
    pol_mean = df.groupby(["ordering", "model", "eta"], as_index=False)[["mean_acc", "mean_bal_acc"]].mean()
    for ordering in ORDERINGS:
        scores = {}
        for (dataset, model), g in cc[cc.dataset != DATASET].groupby(["dataset", "model"]):
            scores[(dataset, model)] = trajectory_scores(g)
        for model, g in pol_mean[pol_mean.ordering == ordering].groupby("model"):
            scores[(DATASET, model)] = trajectory_scores(g)
        res = {}
        for j, metric in enumerate(["audc_bal", "mrr_bal"]):
            wide = pd.Series({k: v[j] for k, v in scores.items()}).unstack()
            chi2, _ = stats.friedmanchisquare(*[wide[m].to_numpy() for m in MODELS])
            n, k_models = len(wide), len(MODELS)
            stat = (n - 1) * chi2 / (n * (k_models - 1) - chi2)
            p = stats.f.sf(stat, k_models - 1, (k_models - 1) * (n - 1))
            ranks = wide.rank(axis=1, ascending=False).mean()
            res[metric] = {"iman_davenport_F": float(stat), "friedman_chi2": float(chi2), "p": float(p), "mean_rank": ranks.round(3).to_dict(),
                           "n_datasets": int(len(wide)), "pol_values": wide.loc[DATASET].round(4).to_dict()}
        out[ordering] = res
    return out


def write_table(summary):
    """LaTeX table: pol per-model scores under both orderings and the cross-dataset omnibus test."""
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{Class-conditional (pair-flip) noise on pol: encoded (lexicographic) versus numeric class order. "
             r"Scores are means over seeds 42--44; the last two rows give the Iman--Davenport test over all twenty "
             r"datasets with pol taken from the respective ordering.}\label{tab:pol_ordering}",
             r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{$\AUDC_{\mathrm{bal}}$} & \multicolumn{2}{c}{$\MRR_{\mathrm{bal}}$} \\",
             r"Model & encoded & numeric & encoded & numeric \\", r"\midrule"]
    for model in MODELS:
        v = [summary[o][m]["pol_values"][model] for m in ("audc_bal", "mrr_bal") for o in ORDERINGS]
        lines.append(f"{model} & " + " & ".join(f"{x:.3f}" for x in v) + r" \\")
    lines.append(r"\midrule")
    f = [summary[o][m]["iman_davenport_F"] for m in ("audc_bal", "mrr_bal") for o in ORDERINGS]
    p = [summary[o][m]["p"] for m in ("audc_bal", "mrr_bal") for o in ORDERINGS]
    lines.append(r"$F_{\mathrm{ID}}$ & " + " & ".join(f"{x:.2f}" for x in f) + r" \\")
    lines.append(r"$p$ & " + " & ".join(f"{x:.1e}" if x < 0.01 else f"{x:.2f}" for x in p) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open("results/analysis/revision/tab_pol_ordering.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main(workers=4, reuse=False):
    if reuse:
        df = pd.concat([pd.read_csv(os.path.join(OUT_DIR, f"pol_ordering__seed{s}.csv")) for s in SEEDS])
    else:
        df = run_all(workers)
    summary = cross_dataset_tests(df)
    with open(os.path.join(OUT_DIR, "pol_ordering_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    write_table(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4, reuse="--reuse" in sys.argv)
