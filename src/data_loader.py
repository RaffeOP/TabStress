import os
import ssl
import certifi
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

try:
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass

# Curated 20-Dataset Gold Standard Tabular Benchmark Corpus
# Sourced from Grinsztajn (NeurIPS 2022), McElfresh (NeurIPS 2023), OpenML-CC18, and FICO Challenge
BENCHMARK_SUITE = {
    "Bioresponse": 4134,
    "phoneme": 1489,
    "nomao": 1486,
    "eye_movements": 1044,
    "churn": 40701,
    "steel-plates-fault": 1504,
    "qsar-biodeg": 1494,
    "pol": 201,
    "jungle_chess": 41027,
    "electricity": 151,
    "bank-marketing": 1461,
    "spambase": 44,
    "vehicle": 54,
    "miniboone": 41150,
    "volkert": 41166,
    "madelon": 1485,
    "amazon_access": 4135,
    "gesture_phase": 4538,
    "ozone_level": 1487,
    "hill_valley": 1479
}

def load_dataset(dataset_name_or_id, data_dir="data", as_numpy=True):
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, f"{dataset_name_or_id}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], data["feature_names"], data["class_names"]

    if isinstance(dataset_name_or_id, str) and dataset_name_or_id in BENCHMARK_SUITE:
        target_id = BENCHMARK_SUITE[dataset_name_or_id]
        bunch = fetch_openml(data_id=target_id, as_frame=True, parser="auto")
    elif isinstance(dataset_name_or_id, int):
        bunch = fetch_openml(data_id=dataset_name_or_id, as_frame=True, parser="auto")
    else:
        bunch = fetch_openml(name=dataset_name_or_id, as_frame=True, parser="auto")

    df_X = bunch.data
    s_y = bunch.target

    # Encode categoricals gracefully: one-hot for low cardinality (<= 25), OrdinalEncoder for high cardinality
    df_X_proc = pd.DataFrame(index=df_X.index)
    for col in df_X.columns:
        if df_X[col].dtype.name in ['category', 'object']:
            if df_X[col].nunique() <= 25:
                dummies = pd.get_dummies(df_X[col], prefix=col, drop_first=True)
                df_X_proc = pd.concat([df_X_proc, dummies], axis=1)
            else:
                le_col = LabelEncoder()
                df_X_proc[col] = le_col.fit_transform(df_X[col].astype(str))
        else:
            df_X_proc[col] = df_X[col]

    feature_names = np.array(df_X_proc.columns.tolist())

    le = LabelEncoder()
    y_encoded = le.fit_transform(s_y.astype(str))
    class_names = np.array([str(c) for c in le.classes_])

    X_mat = df_X_proc.to_numpy(dtype=np.float64)
    if np.isnan(X_mat).any():
        col_medians = np.nanmedian(X_mat, axis=0)
        col_medians[np.isnan(col_medians)] = 0.0
        inds = np.where(np.isnan(X_mat))
        X_mat[inds] = np.take(col_medians, inds[1])

    np.savez_compressed(
        cache_path,
        X=X_mat,
        y=y_encoded,
        feature_names=feature_names,
        class_names=class_names
    )

    return X_mat, y_encoded, feature_names, class_names
