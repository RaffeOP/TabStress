import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_dataset
from src.noise_injector import MSNIE
from src.models import get_model

MASTER_CSV = "results/metrics/raw/volkert.csv"
BACKEND_CSV = "results/metrics/raw/volkert_backend.csv"

# Reverse order models and modalities
MODELS_REVERSE = ["lightgbm"]
MODALITIES_REVERSE = ["mcar_missing"]
NOISE_TIERS_REVERSE = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]

def get_existing_keys():
    keys = set()
    for path in [MASTER_CSV, BACKEND_CSV]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                for _, r in df.iterrows():
                    keys.add((str(r["dataset"]), str(r["model"]), str(r["modality"]), round(float(r["eta"]), 3)))
            except Exception:
                pass
    return keys

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
            accs.append(0.0)
            bal_accs.append(0.0)
            f1s.append(0.0)

    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_bal_acc": float(np.mean(bal_accs)),
        "mean_f1": float(np.mean(f1s))
    }

def main():
    print("[Reverse Worker] Initializing Volkert reverse-order benchmark worker...")
    X, y, f_names, c_names = load_dataset("volkert")
    num_classes = len(c_names)

    if X.shape[0] > 10000:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=42)
        sub_idx, _ = next(sss.split(X, y))
        X, y = X[sub_idx], y[sub_idx]
        print(f"    Subsampled to N={X.shape[0]} samples, {X.shape[1]} features, {num_classes} classes.")

    total_runs_done = 0

    for m_name in MODELS_REVERSE:
        for mod in MODALITIES_REVERSE:
            for eta in NOISE_TIERS_REVERSE:
                existing_keys = get_existing_keys()
                key = ("volkert", m_name, mod, round(eta, 3))
                if key in existing_keys:
                    continue

                print(f"[Reverse Worker] Running: volkert | {m_name} | {mod} | eta={eta:.2f} ...", flush=True)
                metrics = run_evaluation_step(X, y, num_classes, m_name, mod, eta)

                record = {
                    "dataset": "volkert",
                    "model": m_name,
                    "modality": mod,
                    "eta": eta,
                    "mean_acc": metrics["mean_acc"],
                    "std_acc": metrics["std_acc"],
                    "mean_bal_acc": metrics["mean_bal_acc"],
                    "mean_f1": metrics["mean_f1"]
                }

                df_record = pd.DataFrame([record])
                df_record.to_csv(
                    BACKEND_CSV,
                    mode="a",
                    header=not os.path.exists(BACKEND_CSV),
                    index=False
                )
                total_runs_done += 1
                print(f"   Done: acc={metrics['mean_acc']:.4f} (Saved to {BACKEND_CSV})", flush=True)

    print(f"\n[Reverse Worker] Completed {total_runs_done} runs from the back!")

if __name__ == "__main__":
    main()
