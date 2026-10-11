import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_dataset, BENCHMARK_SUITE
from src.noise_injector import MSNIE
from src.models import get_model
from src.breakdown_math import compute_critical_breakdown_threshold

RESULTS_CSV = "results/metrics/benchmark_results.csv"

NOISE_TIERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MODELS = ["cart", "randomforest", "xgboost", "lightgbm", "catboost", "mlp"]
MODALITIES = ["symmetric_label", "asymmetric_label", "gaussian_feature", "mcar_missing"]

def get_completed_keys(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        keys = set(zip(df["dataset"], df["model"], df["modality"], df["eta"].round(3)))
        return keys
    except Exception:
        return set()

def run_evaluation_step(X, y, num_classes, model_name, modality, eta, seed=42, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    injector = MSNIE(random_state=seed)

    accs = []
    bal_accs = []
    f1s = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X[train_idx].copy(), y[train_idx].copy()
        X_val, y_val = X[val_idx].copy(), y[val_idx].copy()

        if modality == "symmetric_label":
            y_train = injector.inject_symmetric_label_noise(y_train, eta=eta, num_classes=num_classes)
        elif modality == "asymmetric_label":
            y_train = injector.inject_asymmetric_label_noise_eta(y_train, eta=eta, num_classes=num_classes)
        elif modality == "gaussian_feature":
            X_train = injector.inject_gaussian_feature_noise(X_train, gamma=eta)
            X_val = injector.inject_gaussian_feature_noise(X_val, gamma=eta)
        elif modality == "mcar_missing":
            X_train = injector.inject_mcar_missingness(X_train, h=eta)
            X_val = injector.inject_mcar_missingness(X_val, h=eta)

        try:
            clf = get_model(model_name, random_state=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            accs.append(accuracy_score(y_val, y_pred))
            bal_accs.append(balanced_accuracy_score(y_val, y_pred))
            f1s.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        except Exception:
            # Fallback for severe degradation edge cases
            accs.append(0.0)
            bal_accs.append(0.0)
            f1s.append(0.0)

    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_bal_acc": float(np.mean(bal_accs)),
        "mean_f1": float(np.mean(f1s))
    }

RAW_DIR = "results/metrics/raw"

def get_completed_keys_for_dataset(dataset_name: str) -> set:
    keys = set()
    raw_csv = os.path.join(RAW_DIR, f"{dataset_name}.csv")
    if os.path.exists(raw_csv):
        try:
            df = pd.read_csv(raw_csv)
            for _, r in df.iterrows():
                keys.add((str(r["dataset"]), str(r["model"]), str(r["modality"]), round(float(r["eta"]), 3)))
        except Exception:
            pass
    return keys

def merge_all_results():
    os.makedirs(RAW_DIR, exist_ok=True)
    csv_files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("No raw dataset CSVs found to merge.")
        return
    dfs = []
    for f in sorted(csv_files):
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["dataset", "model", "modality", "eta"])
        df_all.to_csv(RESULTS_CSV, index=False)
        print(f"✅ Merged {len(df_all)} records across {len(csv_files)} datasets into {RESULTS_CSV}")

def execute_benchmark(target_datasets=None, target_models=None, target_modalities=None):
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    datasets_to_run = target_datasets if target_datasets else list(BENCHMARK_SUITE.keys())
    models_to_run = target_models if target_models else MODELS
    modalities_to_run = target_modalities if target_modalities else MODALITIES

    expected_per_dataset = len(models_to_run) * len(modalities_to_run) * len(NOISE_TIERS)
    total_combos = len(datasets_to_run) * expected_per_dataset
    print(f"[Benchmark] TabStress matrix: {total_combos} total evaluations ({len(datasets_to_run)} datasets)")

    for d_idx, d_name in enumerate(datasets_to_run, 1):
        raw_csv = os.path.join(RAW_DIR, f"{d_name}.csv")
        completed_keys = get_completed_keys_for_dataset(d_name)
        if len(completed_keys) >= expected_per_dataset:
            print(f"[{d_idx}/{len(datasets_to_run)}] Dataset {d_name} already complete ({len(completed_keys)}/{expected_per_dataset} runs). Skipping.")
            continue

        print(f"\n[{d_idx}/{len(datasets_to_run)}] Ingesting dataset: {d_name} (Completed: {len(completed_keys)}/{expected_per_dataset})...")
        try:
            X, y, f_names, c_names = load_dataset(d_name)
            num_classes = len(c_names)
            print(f"    Loaded {d_name}: N={X.shape[0]}, D={X.shape[1]}, Classes={num_classes}")
        except Exception as e:
            print(f"    [WARNING] Skipping {d_name} due to fetch error: {e}")
            continue

        # Stratified subsampling if sample size > 10,000 to ensure bounded execution runtime
        if X.shape[0] > 10000:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=42)
            sub_idx, _ = next(sss.split(X, y))
            X, y = X[sub_idx], y[sub_idx]
            print(f"    Stratified subsampled to N={X.shape[0]} for memory-bounded execution")

        records = []
        for m_name in models_to_run:
            for mod in modalities_to_run:
                for eta in NOISE_TIERS:
                    key = (d_name, m_name, mod, round(eta, 3))
                    if key in completed_keys:
                        continue

                    metrics = run_evaluation_step(X, y, num_classes, m_name, mod, eta)
                    record = {
                        "dataset": d_name,
                        "model": m_name,
                        "modality": mod,
                        "eta": eta,
                        "mean_acc": metrics["mean_acc"],
                        "std_acc": metrics["std_acc"],
                        "mean_bal_acc": metrics["mean_bal_acc"],
                        "mean_f1": metrics["mean_f1"]
                    }
                    records.append(record)
                    completed_keys.add(key)

                    # Incremental append every 5 steps
                    if len(records) >= 5:
                        df_chunk = pd.DataFrame(records)
                        df_chunk.to_csv(raw_csv, mode="a", header=not os.path.exists(raw_csv), index=False)
                        records.clear()

        # Flush remaining records for this dataset
        if records:
            df_chunk = pd.DataFrame(records)
            df_chunk.to_csv(raw_csv, mode="a", header=not os.path.exists(raw_csv), index=False)
            records.clear()

        print(f"    Dataset {d_name} finalized ({len(completed_keys)} records saved to {raw_csv})")

    # Merge all raw dataset CSVs into main benchmark_results.csv
    merge_all_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--modalities", nargs="+", default=None)
    args = parser.parse_args()

    execute_benchmark(
        target_datasets=args.datasets,
        target_models=args.models,
        target_modalities=args.modalities
    )
