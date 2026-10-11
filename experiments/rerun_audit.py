"""Re-computation audit (revision point 10): a random sample of stored cells from every Protocol A block
and seed is recomputed from scratch with the current, corrected harness and compared with the stored
accuracy. Writes results/analysis/revision/rerun_audit.json."""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(var, "1")

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.run_extended import run_cell

EXT = "results/metrics/extended"
PER_BLOCK = 12
SKIP_DATASETS = {"volkert"}   # the slowest dataset is left out to bound the audit's run time


def sample():
    rng = np.random.default_rng(2026)
    tasks = []
    for f in sorted(os.listdir(EXT)):
        if not f.endswith(".csv") or f.startswith("catboost_default"):
            continue
        df = pd.read_csv(os.path.join(EXT, f))
        df = df[~df.dataset.isin(SKIP_DATASETS)]
        for _, r in df.iloc[rng.choice(len(df), PER_BLOCK, replace=False)].iterrows():
            tasks.append(((r.variant, int(r.seed), r.dataset, r.model, r.modality, float(r.eta)), float(r.mean_acc)))
    grid = pd.read_csv("results/metrics/benchmark_results.csv")
    grid = grid[~grid.dataset.isin(SKIP_DATASETS)]
    for _, r in grid.iloc[rng.choice(len(grid), PER_BLOCK, replace=False)].iterrows():
        tasks.append((("base", 42, r.dataset, r.model, r.modality, float(r.eta)), float(r.mean_acc)))
    return tasks


def main(workers):
    tasks = sample()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        recs = list(pool.map(run_cell, [t for t, _ in tasks]))
    diffs = [abs(r["mean_acc"] - stored) for r, (_, stored) in zip(recs, tasks)]
    out = {"n_cells": len(diffs), "max_abs_diff": float(max(diffs)), "n_blocks": len({t[0][:2] for t in tasks}),
           "cells": [dict(task=list(t), stored=s, recomputed=r["mean_acc"]) for (t, s), r in zip(tasks, recs)]}
    os.makedirs("results/analysis/revision", exist_ok=True)
    json.dump(out, open("results/analysis/revision/rerun_audit.json", "w"), indent=1)
    print(out["n_cells"], out["max_abs_diff"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
