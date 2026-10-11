"""Recompute cells of the original seed-42 grid after a correction to a corruption operator,
and patch them into the grid.

Used once, for the Gaussian operator: an earlier version gave columns that are constant in the
corrupted matrix a noise scale of 1 instead of 0. Only datasets with such columns change (see
experiments/operator_audit.py); every other cell is bit-identical under both versions.

The cells are recomputed with the extended harness (run_extended.run_cell, variant "base",
seed 42), which reproduces stored seed-42 cells exactly; two control cells from unaffected
datasets are recomputed first and must match all four stored metrics. Results are kept in
results/metrics/original_recomputed.csv (resumable). When every requested cell is present, the
rows of results/metrics/benchmark_results.csv and of results/metrics/raw/<dataset>.csv are
patched in place, after backing both up, and every replaced value is logged in
results/metrics/original_patch_log.csv."""

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from run_extended import MODELS, TIERS, run_cell

GRID = "results/metrics/benchmark_results.csv"
RAW_DIR = "results/metrics/raw"
STORE = "results/metrics/original_recomputed.csv"
LOG = "results/metrics/original_patch_log.csv"
BACKUP = "results/metrics/backup_before_original_patch"
METRICS = ["mean_acc", "std_acc", "mean_bal_acc", "mean_f1"]
KEY = ["dataset", "model", "modality", "eta"]
CONTROLS = [("phoneme", "cart", "gaussian_feature", 0.3), ("spambase", "lightgbm", "gaussian_feature", 0.2)]


def control_check():
    grid = pd.read_csv(GRID)
    grid["eta"] = grid.eta.round(2)
    worst = 0.0
    for d, m, mod, e in CONTROLS:
        r = run_cell(("base", 42, d, m, mod, e))
        stored = grid[(grid.dataset == d) & (grid.model == m) & (grid.modality == mod) & (grid.eta == e)].iloc[0]
        worst = max(worst, max(abs(r[c] - stored[c]) for c in METRICS))
    if worst > 1e-12:
        raise SystemExit(f"harness does not reproduce the stored grid (max deviation {worst}); not patching")
    print(f"control cells reproduced (max deviation {worst:.1e})", flush=True)


def compute(tasks, workers):
    done = set()
    if os.path.exists(STORE):
        s = pd.read_csv(STORE)
        done = set(zip(s.dataset, s.model, s.modality, s.eta.round(2)))
    todo = [t for t in tasks if (t[2], t[3], t[4], round(t[5], 2)) not in done]
    print(f"{len(tasks) - len(todo)} done, {len(todo)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_cell, t): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            row = {k: rec[k] for k in KEY + METRICS + ["seconds"]}
            pd.DataFrame([row]).to_csv(STORE, mode="a", header=not os.path.exists(STORE), index=False)
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} recomputed", flush=True)


def patch(tasks):
    new = pd.read_csv(STORE)
    new["eta"] = new.eta.round(2)
    want = {(t[2], t[3], t[4], round(t[5], 2)) for t in tasks}
    have = set(zip(new.dataset, new.model, new.modality, new.eta))
    if not want <= have:
        raise SystemExit(f"{len(want - have)} cells still missing; not patching")
    new = new.set_index(KEY)
    os.makedirs(BACKUP, exist_ok=True)
    log = []
    datasets = sorted({t[2] for t in tasks})
    for path in [GRID] + [os.path.join(RAW_DIR, f"{d}.csv") for d in datasets]:
        dest = os.path.join(BACKUP, os.path.basename(path) if path == GRID else "raw__" + os.path.basename(path))
        if not os.path.exists(dest):
            shutil.copy2(path, dest)
        df = pd.read_csv(path)
        eta = df.eta.round(2)
        n = 0
        for i in df.index:
            k = (df.at[i, "dataset"], df.at[i, "model"], df.at[i, "modality"], eta[i])
            if k in want:
                for c in METRICS:
                    log.append({"file": path, "dataset": k[0], "model": k[1], "modality": k[2], "eta": k[3],
                                "metric": c, "old": df.at[i, c], "new": new.loc[k, c]})
                    df.at[i, c] = new.loc[k, c]
                n += 1
        df.to_csv(path, index=False)
        print(f"patched {n} rows in {path}", flush=True)
    pd.DataFrame(log).to_csv(LOG, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--modality", default="gaussian_feature")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    tasks = [("base", 42, d, m, args.modality, e) for d in args.datasets for m in MODELS for e in TIERS if e > 0]
    control_check()
    compute(tasks, args.workers)
    patch(tasks)


if __name__ == "__main__":
    main()
