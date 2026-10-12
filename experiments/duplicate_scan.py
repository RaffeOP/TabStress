"""Count exact duplicate feature rows in each cached dataset upload.

Reports the number of rows that repeat an earlier feature vector (features only, labels ignored)
and the number of feature vectors that carry conflicting labels. Output is printed as a table and
is the source of the duplicate counts quoted in the manuscript's dataset-artifacts paragraph.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_dataset

DATASETS = ["nomao", "pol", "volkert", "phoneme", "ozone_level", "spambase", "miniboone"]


def scan_dataset(name):
    X, y, _, _ = load_dataset(name)
    df = pd.DataFrame(X)
    duplicate_rows = int(df.duplicated().sum())
    conflicting_vectors = int((df.assign(y=y).groupby(list(df.columns)).y.nunique() > 1).sum())
    return {"dataset": name, "rows": X.shape[0], "duplicate_rows": duplicate_rows,
            "conflicting_vectors": conflicting_vectors}


def main():
    records = [scan_dataset(name) for name in DATASETS]
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
