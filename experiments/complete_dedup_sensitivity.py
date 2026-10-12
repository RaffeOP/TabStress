"""Finish the deduplication sensitivity run: re-run every planned cell that is missing from disk.

run_dedup_sensitivity.py appends each finished cell to results/metrics/artifact/<variant>__seed<seed>.csv.
If those files are rewritten while the run is in progress, completed cells are lost. This script
rebuilds the full plan, finds the cells absent from the files, and runs only those.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(var, "1")

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments import run_artifact_sensitivity as artifact
from experiments.run_dedup_sensitivity import DEDUP_VARIANTS, OUT_DIR, PRIMARY_SEEDS, build_dedup_copy, SOURCES
from experiments.run_extended import MODELS, TIERS, run_cell

KEY_COLUMNS = ["variant", "seed", "dataset", "model", "modality", "eta"]
DEFAULT_WORKERS = 8


MULTICLASS_VARIANTS = {"pol_dedup", "volkert_dedup"}


def planned_tasks():
    artifact.CLEANED = DEDUP_VARIANTS
    tasks = artifact.plan(PRIMARY_SEEDS)
    # minority-to-majority noise is defined for binary labels only; pol and volkert are multiclass
    tasks = [t for t in tasks if not (t[0] == "cc_binary" and t[2] in MULTICLASS_VARIANTS)]
    return tasks + multiclass_class_conditional_tasks()


def multiclass_class_conditional_tasks():
    """Class-conditional noise on multiclass datasets is the pair-flip (asymmetric_label) operator in the primary grid."""
    noisy = [e for e in TIERS if e > 0]
    return [("base", seed, dataset, model, "asymmetric_label", eta)
            for seed in PRIMARY_SEEDS for dataset in sorted(MULTICLASS_VARIANTS)
            for model in MODELS for eta in noisy]


def completed_keys():
    if not os.path.isdir(OUT_DIR):
        return set()
    files = [os.path.join(OUT_DIR, f) for f in os.listdir(OUT_DIR) if f.endswith(".csv")]
    if not files:
        return set()
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame = frame[frame.dataset.isin(DEDUP_VARIANTS)]
    return set(map(tuple, frame[KEY_COLUMNS].round({"eta": 4}).astype(str).values.tolist()))


def task_key(task):
    variant, seed, dataset, model, modality, eta = task
    return (variant, str(seed), dataset, model, modality, str(round(eta, 4)))


def missing_tasks():
    done = completed_keys()
    return [t for t in planned_tasks() if task_key(t) not in done]


def append_record(rec):
    path = os.path.join(OUT_DIR, f"{rec['variant']}__seed{rec['seed']}.csv")
    pd.DataFrame([rec]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def main(workers=DEFAULT_WORKERS):
    for name in SOURCES:
        if not os.path.exists(f"data/{name}_dedup.npz"):
            build_dedup_copy(name)
    tasks = missing_tasks()
    print(len(tasks), "missing cells", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_cell, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            append_record(fut.result())
            if i % 100 == 0:
                print(i, flush=True)
    print("remaining after run:", len(missing_tasks()), flush=True)


if __name__ == "__main__":
    main(workers=int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKERS)
