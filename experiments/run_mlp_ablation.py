"""MLP early-stopping ablation under symmetric label noise (Protocol A, seed 42).

The primary MLP uses early stopping on a 10% hold-out of the training fold. Under Protocol A that
hold-out carries corrupted labels, so early stopping is itself affected by the corruption. This
script re-runs the MLP with early stopping disabled, on the same folds and the same corruption
draws as the primary grid, for every dataset and every noisy tier.
Results: results/metrics/ablation/mlp_no_es__seed42.csv
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(var, "1")

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments import run_extended
from experiments.run_extended import TIERS, run_cell
from src.models import get_model

ABLATION_VARIANT = "mlp_no_es"
OUT_DIR = "results/metrics/ablation"
SEED = 42
PRIMARY_ORIGINAL = run_extended.build


def build_with_ablation(variant, model, seed):
    """Disable early stopping for the ablation variant; every other variant is built as before."""
    if variant == ABLATION_VARIANT:
        pipe = get_model("mlp", random_state=seed)
        pipe.set_params(clf__early_stopping=False)
        return run_extended.single_thread(pipe)
    return PRIMARY_ORIGINAL(variant, model, seed)


run_extended.build = build_with_ablation


def run_task(task):
    return run_cell(task)


def plan(datasets):
    noisy = [e for e in TIERS if e > 0]
    return [(ABLATION_VARIANT, SEED, d, "mlp", "symmetric_label", e) for d in datasets for e in noisy]


def main(workers=8):
    datasets = sorted(run_extended.BENCHMARK_SUITE)
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = plan(datasets)
    print(len(tasks), "tasks", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            rows.append(fut.result())
            if i % 25 == 0:
                print(i, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"mlp_no_es__seed{SEED}.csv"), index=False)


if __name__ == "__main__":
    main(workers=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
