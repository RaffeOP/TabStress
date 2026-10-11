"""
TabStress: Parallel Multi-Process Benchmark Runner
Executes benchmark across 20 curated datasets using worker pools.
Each dataset writes to an isolated raw CSV in results/metrics/raw/{dataset}.csv.
Once all datasets finish, all raw CSVs are merged into results/metrics/benchmark_results.csv.
"""

import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.run_benchmark import execute_benchmark, merge_all_results, RAW_DIR, MODELS, MODALITIES, NOISE_TIERS
from src.data_loader import BENCHMARK_SUITE

DATASET_ORDER = [
    # Fast / small datasets first
    "vehicle", "hill_valley", "qsar-biodeg", "steel-plates-fault", "ozone_level",
    "madelon", "churn", "spambase", "phoneme", "Bioresponse",
    # Medium datasets
    "gesture_phase", "eye_movements", "pol", "amazon_access",
    # Large datasets (subsampled to 10k)
    "nomao", "jungle_chess", "electricity", "bank-marketing", "volkert", "miniboone"
]

def run_single_dataset(d_name):
    t0 = time.time()
    expected = len(MODELS) * len(MODALITIES) * len(NOISE_TIERS)
    raw_csv = os.path.join(RAW_DIR, f"{d_name}.csv")

    if os.path.exists(raw_csv):
        try:
            with open(raw_csv, "r") as f:
                lines = sum(1 for _ in f) - 1
            if lines >= expected:
                return d_name, True, f"Already complete ({lines}/{expected} records)", 0.0
        except Exception:
            pass

    print(f"\n[Worker] Launching evaluation for {d_name} (Target: {expected} runs)...")
    try:
        execute_benchmark(target_datasets=[d_name])
        elapsed = time.time() - t0
        return d_name, True, f"Finished in {elapsed:.1f}s", elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return d_name, False, f"Failed with error: {e}", elapsed

def main():
    parser = argparse.ArgumentParser(description="TabStress Parallel Benchmark Runner")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent dataset workers (default: 2)")
    parser.add_argument("--datasets", nargs="+", default=None, help="Specific datasets to run")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    target_datasets = args.datasets if args.datasets else DATASET_ORDER

    print("=" * 70)
    print(f"TabStress Parallel Benchmark Coordinator")
    print(f"   Total Datasets: {len(target_datasets)}")
    print(f"   Worker Concurrency: {args.workers}")
    print(f"   Total Target Runs: {len(target_datasets) * len(MODELS) * len(MODALITIES) * len(NOISE_TIERS)}")
    print("=" * 70)

    start_time = time.time()
    results = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_dataset = {executor.submit(run_single_dataset, d): d for d in target_datasets}
        for future in as_completed(future_to_dataset):
            d_name = future_to_dataset[future]
            try:
                name, success, msg, dur = future.result()
                status_icon = "[OK]" if success else "[FAIL]"
                print(f"{status_icon} [{name}] {msg}")
                results[name] = (success, dur)
            except Exception as exc:
                print(f"[FAIL] [{d_name}] Generated an unhandled exception: {exc}")
                results[d_name] = (False, 0.0)

    print("\n" + "=" * 70)
    print("All benchmark worker tasks completed! Finalizing dataset merge...")
    merge_all_results()
    total_elapsed = time.time() - start_time
    print(f"Total parallel runtime: {total_elapsed/60:.2f} minutes")
    print("=" * 70)

if __name__ == "__main__":
    main()
