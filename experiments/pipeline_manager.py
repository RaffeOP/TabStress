"""
TabStress Pipeline Manager
Monitors active benchmarks, maintains a concurrency limit of 3 workers on 8-core Apple Silicon,
automatically starts pending datasets (bank-marketing, volkert, miniboone) as slots free up,
and executes the full analytical synthesis pipeline (analyze_results.py) upon 100% completion.
"""

import os
import sys
import time
import subprocess
import pandas as pd

RAW_DIR = "results/metrics/raw"
BENCHMARK_SCRIPT = "experiments/run_benchmark.py"
ANALYZE_SCRIPT = "experiments/analyze_results.py"
PYTHON_BIN = sys.executable

ALL_20_DATASETS = [
    "vehicle", "hill_valley", "qsar-biodeg", "steel-plates-fault", "ozone_level",
    "madelon", "churn", "spambase", "phoneme", "Bioresponse",
    "gesture_phase", "eye_movements", "pol", "amazon_access",
    "nomao", "jungle_chess", "electricity", "bank-marketing", "volkert", "miniboone"
]

PENDING_QUEUE = ["bank-marketing", "volkert", "miniboone"]
MAX_CONCURRENT = 3
TARGET_RUNS_PER_DATASET = 264

def get_dataset_run_count(dataset_name):
    csv_path = os.path.join(RAW_DIR, f"{dataset_name}.csv")
    if not os.path.exists(csv_path):
        return 0
    try:
        df = pd.read_csv(csv_path)
        # Unique evaluation tuples
        tuples = set(zip(df["dataset"], df["model"], df["modality"], df["eta"].round(3)))
        return len(tuples)
    except Exception:
        return 0

def get_active_benchmark_pids():
    try:
        res = subprocess.run(["pgrep", "-f", "run_benchmark.py"], stdout=subprocess.PIPE, text=True)
        pids = [int(p) for p in res.stdout.strip().split() if p]
        return pids
    except Exception:
        return []

def is_dataset_running(dataset_name):
    try:
        res = subprocess.run(["pgrep", "-f", f"run_benchmark.py.*{dataset_name}"], stdout=subprocess.PIPE, text=True)
        return bool(res.stdout.strip())
    except Exception:
        return False

def launch_dataset_worker(dataset_name):
    log_file = open(f"results/metrics/raw/{dataset_name}.log", "a")
    proc = subprocess.Popen(
        [PYTHON_BIN, BENCHMARK_SCRIPT, "--datasets", dataset_name],
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    print(f"[Pipeline Manager] Spawned worker for {dataset_name} (PID: {proc.pid})")
    return proc

def main():
    print("=" * 70)
    print("TabStress Pipeline Manager Initialized")
    print(f"   Target Datasets: {len(ALL_20_DATASETS)}")
    print(f"   Max Concurrency: {MAX_CONCURRENT}")
    print("=" * 70)

    # Dynamically find all incomplete datasets, sorted by highest completion first (quickest to finalize)
    incomplete = [d for d in ALL_20_DATASETS if get_dataset_run_count(d) < TARGET_RUNS_PER_DATASET]
    incomplete.sort(key=lambda d: get_dataset_run_count(d), reverse=True)
    queue = [d for d in incomplete if not is_dataset_running(d)]

    while True:
        completed = {}
        total_completed_runs = 0
        for d in ALL_20_DATASETS:
            cnt = get_dataset_run_count(d)
            completed[d] = cnt
            total_completed_runs += min(cnt, TARGET_RUNS_PER_DATASET)

        pct = (total_completed_runs / (len(ALL_20_DATASETS) * TARGET_RUNS_PER_DATASET)) * 100
        
        # Dynamically refresh queue of datasets needing work
        incomplete = [d for d in ALL_20_DATASETS if get_dataset_run_count(d) < TARGET_RUNS_PER_DATASET and not is_dataset_running(d)]
        incomplete.sort(key=lambda d: get_dataset_run_count(d), reverse=True)
        queue = incomplete

        # Check running datasets
        active_pids = get_active_benchmark_pids()
        num_active = len(active_pids)

        print(f"\n[{time.strftime('%X')}] Total Progress: {total_completed_runs}/{len(ALL_20_DATASETS)*TARGET_RUNS_PER_DATASET} runs ({pct:.1f}%) | Active Workers: {num_active}")
        
        for d in ALL_20_DATASETS:
            cnt = completed[d]
            status = "COMPLETE" if cnt >= TARGET_RUNS_PER_DATASET else ("RUNNING" if is_dataset_running(d) else "QUEUED")
            if status != "COMPLETE":
                print(f"    - {d:20s}: {cnt:3d}/{TARGET_RUNS_PER_DATASET} [{status}]")

        if total_completed_runs >= len(ALL_20_DATASETS) * TARGET_RUNS_PER_DATASET:
            print("\nALL 20 DATASETS REACHED 100% COMPLETION (5,280 RUNS)!")
            print("Running final result merge & analytical synthesis...")
            from experiments.run_benchmark import merge_all_results
            merge_all_results()
            print("Executing experiments/analyze_results.py...")
            subprocess.run([PYTHON_BIN, ANALYZE_SCRIPT], check=True)
            print("All analytical synthesis, degradation plots, and LaTeX tables generated successfully!")
            break

        # Fill all available worker slots up to MAX_CONCURRENT
        while num_active < MAX_CONCURRENT and queue:
            next_ds = queue.pop(0)
            if get_dataset_run_count(next_ds) < TARGET_RUNS_PER_DATASET and not is_dataset_running(next_ds):
                launch_dataset_worker(next_ds)
                num_active += 1
                time.sleep(1)

        time.sleep(15)

if __name__ == "__main__":
    main()
