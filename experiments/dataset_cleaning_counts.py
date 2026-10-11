"""Row counts before and after each cleaning rule of the artifact-sensitivity analysis (revision point 34).

Rules (applied only to the cleaned copies of run_artifact_sensitivity.py; the primary benchmark uses the
uploads unchanged apart from loader encoding and the 10,000-row stratified subsample):
  R1 exact duplicates: feature-vector duplicates of an earlier row with the same label are removed (first kept)
  R2 conflicting duplicates: every row of a feature vector that occurs with more than one label is removed
  R3 sentinel rows: rows in which every feature equals -999 are removed
Output: results/analysis/revision/cleaning_counts.csv"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import BENCHMARK_SUITE, load_dataset

SENTINEL = -999
EOL = chr(92) * 2
OUT = "results/analysis/revision/cleaning_counts.csv"


def count_rules(name):
    X, y, _, _ = load_dataset(name)
    df = pd.DataFrame(X)
    n = len(df)
    conflicting = (df.assign(y=y).groupby(list(df.columns)).y.transform("nunique") > 1).to_numpy()
    dup = df.duplicated(keep="first").to_numpy()
    sentinel = (X == SENTINEL).all(axis=1)
    removed_r12 = dup | conflicting
    return dict(dataset=name, n_rows=n, n_features=X.shape[1], r1_duplicate_only=int((dup & ~conflicting).sum()),
                r2_conflicting_rows=int(conflicting.sum()), rows_after_r1_r2=int(n - removed_r12.sum()),
                r3_sentinel_rows=int(sentinel.sum()), rows_after_r3=int(n - sentinel.sum()))


def write_table(df):
    flagged = df[(df.r1_duplicate_only + df.r2_conflicting_rows + df.r3_sentinel_rows) > 0]
    header = ("Dataset & Rows & R1 duplicates & R2 conflicting & After R1+R2 & R3 sentinel & After R3 " + EOL)
    caption = (f"Rows before and after each cleaning rule, for the datasets on which at least one rule applies "
               f"(the other {len(df) - len(flagged)} datasets are unaffected). R1 and R2 act on feature vectors, R3 on "
               "sentinel rows. Only the cleaned copies of spambase (R1 and R2), churn (identifier column) and MiniBooNE "
               "(R3) enter the artifact analysis; the other rows are reported for transparency.")
    lines = [r"\begin{table}[htbp]", r"\centering\small", r"\caption{" + caption + r"}\label{tab:cleaning_counts}",
             r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule", header, r"\midrule"]
    for r in flagged.itertuples():
        cells = [r.dataset.replace("_", r"\_")] + [f"{v:,}".replace(",", "{,}") for v in (
            r.n_rows, r.r1_duplicate_only, r.r2_conflicting_rows, r.rows_after_r1_r2, r.r3_sentinel_rows, r.rows_after_r3)]
        lines.append(" & ".join(cells) + " " + EOL)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open("results/analysis/revision/tab_cleaning_counts.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    df = pd.DataFrame([count_rules(d) for d in BENCHMARK_SUITE])
    df.to_csv(OUT, index=False)
    write_table(df)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
