"""Duplicate-removal sensitivity for the datasets not covered by run_artifact_sensitivity.

For each dataset, exact duplicate feature rows are removed (first occurrence kept), and every
feature vector whose duplicates carry conflicting labels is dropped entirely, the same rule as
spambase_clean. The protocol is then re-run exactly as in the primary grid (Protocol A, seeds
42-44). Deduplication happens before the 10,000-row subsample, as in the main pipeline.
Results: results/metrics/artifact/<variant>__seed<seed>.csv
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(var, "1")

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments import run_artifact_sensitivity as artifact
from experiments.run_extended import run_cell
from src.data_loader import load_dataset

OUT_DIR = "results/metrics/artifact"
SOURCES = ["nomao", "pol", "volkert", "phoneme", "ozone_level"]
DEDUP_VARIANTS = [f"{name}_dedup" for name in SOURCES]
DEFAULT_WORKERS = 8
PRIMARY_SEEDS = (42, 43, 44)


def build_dedup_copy(name):
    X, y, f, c = load_dataset(name)
    df = pd.DataFrame(X)
    conflicting = df.assign(y=y).groupby(list(df.columns)).y.transform("nunique") > 1
    keep = ~df.duplicated(keep="first").to_numpy() & ~conflicting.to_numpy()
    np.savez_compressed(f"data/{name}_dedup.npz", X=X[keep], y=y[keep], feature_names=f, class_names=c)


def main(workers=DEFAULT_WORKERS, seeds=PRIMARY_SEEDS):
    for name in SOURCES:
        build_dedup_copy(name)
    artifact.CLEANED = DEDUP_VARIANTS
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = artifact.plan(seeds)
    print(len(tasks), "tasks", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_cell, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            rec = fut.result()
            path = os.path.join(OUT_DIR, f"{rec['variant']}__seed{rec['seed']}.csv")
            pd.DataFrame([rec]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            if i % 100 == 0:
                print(i, flush=True)


if __name__ == "__main__":
    main(workers=int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKERS)
