"""Artifact-removal sensitivity (revision point 39). Builds cleaned copies of the three datasets with
documented artifacts and re-runs the primary protocol (Protocol A, seeds 42-44) on them:
  spambase_clean   duplicate feature rows removed (first occurrence kept; rows whose duplicates carry
                   conflicting labels dropped entirely)
  churn_clean      administrative phone_number identifier column removed
  miniboone_clean  sentinel rows with every feature equal to -999 removed
Results: results/metrics/artifact/<variant>__seed<seed>.csv, same schema as the extended blocks."""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(var, "1")

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.run_extended import MODALITIES, MODELS, TIERS, TREES, run_cell
from src.data_loader import load_dataset

OUT_DIR = "results/metrics/artifact"
CLEANED = ["spambase_clean", "churn_clean", "miniboone_clean"]


def build_clean_copies():
    X, y, f, c = load_dataset("spambase")
    df = pd.DataFrame(X)
    conflicting = df.assign(y=y).groupby(list(df.columns)).y.transform("nunique") > 1
    keep = ~df.duplicated(keep="first") & ~conflicting.to_numpy()
    np.savez_compressed("data/spambase_clean.npz", X=X[keep], y=y[keep], feature_names=f, class_names=c)
    X, y, f, c = load_dataset("churn")
    cols = np.array([n != "phone_number" for n in f])
    np.savez_compressed("data/churn_clean.npz", X=X[:, cols], y=y, feature_names=f[cols], class_names=c)
    X, y, f, c = load_dataset("miniboone")
    keep = ~(X == -999).all(axis=1)
    np.savez_compressed("data/miniboone_clean.npz", X=X[keep], y=y[keep], feature_names=f, class_names=c)


def plan(seeds):
    noisy = [e for e in TIERS if e > 0]
    tasks = []
    for s in seeds:
        for d in CLEANED:
            tasks += [("base", s, d, m, "clean", 0.0) for m in MODELS]
            tasks += [("base", s, d, m, mod, e) for m in MODELS for mod in MODALITIES if mod != "asymmetric_label"
                      for e in noisy]
            tasks += [("cc_binary", s, d, m, "minority_to_majority", e) for m in MODELS for e in noisy]
            tasks += [("native_nan", s, d, m, "mcar_missing", e) for m in TREES for e in noisy]
    return tasks


def main(workers=8, seeds=(42, 43, 44)):
    build_clean_copies()
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = plan(seeds)
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
    main(workers=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
