"""Extended TabStress experiments that close the design gaps of the 5,280-run grid.

Variants (results in results/metrics/extended/<variant>__seed<seed>.csv):
  cc_binary         minority->majority label noise on the binary datasets (train labels only); the
                    minority class is fixed per dataset, ties (madelon, hill_valley) go to the larger label
  native_nan        MCAR masking with NO imputation for the five tree families
  catboost_default  CatBoost at library defaults (1,000 rounds, automatic learning rate)
  xgb_default       XGBoost at library defaults (100 rounds, learning rate 0.3)
  base              the original four modalities, all six families, for additional seeds

The protocol is identical to experiments/run_benchmark.py (same cached data, same
10,000-row stratified subsample with seed 42, stratified 5-fold CV, corruption inside
the fold) except that the seed of the split, the corruption and the model varies, and each
cell runs single-threaded so that cells can run in parallel. Workers only compute; the main
process is the only writer, so result files cannot be corrupted by concurrent appends.
At eta = 0 every modality yields the same fit, so clean cells are computed once per
(dataset, model, seed) and stored under modality "clean".
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(var, "1")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import BENCHMARK_SUITE, load_dataset
from src.models import get_model
from src.noise_injector import MSNIE

OUT_DIR = "results/metrics/extended"
TIERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MODELS = ["cart", "randomforest", "xgboost", "lightgbm", "catboost", "mlp"]
TREES = ["cart", "randomforest", "xgboost", "lightgbm", "catboost"]
MODALITIES = ["symmetric_label", "asymmetric_label", "gaussian_feature", "mcar_missing"]
SUBSAMPLE_N = 10_000
SUBSAMPLE_SEED = 42
KEY = ["variant", "seed", "dataset", "model", "modality", "eta"]


@lru_cache(maxsize=4)
def data(name):
    X, y, _, _ = load_dataset(name)
    if X.shape[0] > SUBSAMPLE_N:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=SUBSAMPLE_N, random_state=SUBSAMPLE_SEED)
        idx, _ = next(sss.split(X, y))
        X, y = X[idx], y[idx]
    return X, y, len(np.unique(y))


def single_thread(pipe):
    clf = pipe.named_steps["clf"]
    if "n_jobs" in clf.get_params():
        clf.set_params(n_jobs=1)
    if "thread_count" in clf.get_params():
        clf.set_params(thread_count=1)
    return pipe


def build(variant, model, seed):
    if variant == "native_nan" and model in TREES:
        pipe = get_model(model, random_state=seed)
        pipe.steps = [s for s in pipe.steps if s[0] != "imputer"]
        return single_thread(pipe)
    if variant == "xgb_default":
        import xgboost as xgb
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        # library defaults: 100 rounds, learning rate 0.3, depth 6 (the matched grid uses 0.1)
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("clf", xgb.XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=1))])
    if variant == "catboost_default":
        from catboost import CatBoostClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("clf", CatBoostClassifier(random_state=seed, verbose=0, thread_count=1))])
    return single_thread(get_model(model, random_state=seed))


def run_cell(task):
    variant, seed, dataset, model, modality, eta = task
    X, y, k = data(dataset)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    inj = MSNIE(random_state=seed)
    acc, bal, f1 = [], [], []
    t0 = time.time()
    for tr, va in skf.split(X, y):
        Xtr, ytr, Xva, yva = X[tr].copy(), y[tr].copy(), X[va].copy(), y[va].copy()
        if modality == "symmetric_label":
            ytr = inj.inject_symmetric_label_noise(ytr, eta=eta, num_classes=k)
        elif modality == "asymmetric_label":
            ytr = inj.inject_asymmetric_label_noise_eta(ytr, eta=eta, num_classes=k)
        elif modality == "minority_to_majority":
            ytr = inj.inject_minority_to_majority_noise(ytr, eta=eta, minority=MSNIE.minority_class(y))
        elif modality == "gaussian_feature":
            Xtr = inj.inject_gaussian_feature_noise(Xtr, gamma=eta)
            Xva = inj.inject_gaussian_feature_noise(Xva, gamma=eta)
        elif modality == "mcar_missing":
            Xtr = inj.inject_mcar_missingness(Xtr, h=eta)
            Xva = inj.inject_mcar_missingness(Xva, h=eta)
        elif modality != "clean":
            raise ValueError(modality)
        clf = build(variant, model, seed)
        clf.fit(Xtr, ytr)
        pred = np.asarray(clf.predict(Xva)).ravel().astype(yva.dtype)
        acc.append(accuracy_score(yva, pred))
        bal.append(balanced_accuracy_score(yva, pred))
        f1.append(f1_score(yva, pred, average="macro", zero_division=0))
    return dict(variant=variant, seed=seed, dataset=dataset, model=model, modality=modality, eta=eta,
                mean_acc=float(np.mean(acc)), std_acc=float(np.std(acc)), min_fold_acc=float(np.min(acc)),
                mean_bal_acc=float(np.mean(bal)), mean_f1=float(np.mean(f1)), seconds=round(time.time() - t0, 2))


def binary_datasets():
    return [d for d in BENCHMARK_SUITE if len(np.unique(load_dataset(d)[1])) == 2]


def plan(seeds_new, seeds_extra):
    tasks = []
    noisy = [e for e in TIERS if e > 0]
    for seed in seeds_new:
        for d in binary_datasets():
            tasks += [("cc_binary", seed, d, m, "minority_to_majority", e) for m in MODELS for e in noisy]
        for d in BENCHMARK_SUITE:
            tasks += [("native_nan", seed, d, m, "mcar_missing", e) for m in TREES for e in noisy]
    for d in BENCHMARK_SUITE:
        tasks.append(("xgb_default", 42, d, "xgboost", "clean", 0.0))
        tasks += [("xgb_default", 42, d, "xgboost", mod, e) for mod in MODALITIES for e in noisy]
    for d in BENCHMARK_SUITE:
        tasks.append(("catboost_default", 42, d, "catboost", "clean", 0.0))
        tasks += [("catboost_default", 42, d, "catboost", mod, e) for mod in MODALITIES for e in noisy]
    for seed in seeds_extra:
        for d in BENCHMARK_SUITE:
            tasks += [("base", seed, d, m, "clean", 0.0) for m in MODELS]
            tasks += [("base", seed, d, m, mod, e) for m in MODELS for mod in MODALITIES for e in noisy]
    return tasks


def out_path(variant, seed):
    return os.path.join(OUT_DIR, f"{variant}__seed{seed}.csv")


def done_keys():
    keys = set()
    if not os.path.isdir(OUT_DIR):
        return keys
    for f in os.listdir(OUT_DIR):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(OUT_DIR, f))
            keys |= set(zip(df.variant, df.seed, df.dataset, df.model, df.modality, df.eta.round(3)))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds-new", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--seeds-extra", type=int, nargs="+", default=[43, 44])
    ap.add_argument("--limit", type=int, default=None, help="run only the first N pending tasks")
    ap.add_argument("--only", nargs="+", default=None, help="restrict to these variants")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    done = done_keys()
    tasks = [t for t in plan(args.seeds_new, args.seeds_extra)
             if (t[0], t[1], t[2], t[3], t[4], round(t[5], 3)) not in done
             and (args.only is None or t[0] in args.only)]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"{len(done)} cells already done, {len(tasks)} pending, {args.workers} workers", flush=True)

    buffers, t_start, n = {}, time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                print(f"FAILED {task}: {exc!r}", flush=True)
                continue
            path = out_path(rec["variant"], rec["seed"])
            pd.DataFrame([rec]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            n += 1
            if n % 50 == 0 or n == len(tasks):
                rate = n / (time.time() - t_start)
                print(f"{n}/{len(tasks)} cells, {rate * 60:.1f}/min, ETA {(len(tasks) - n) / rate / 60:.0f} min",
                      flush=True)


if __name__ == "__main__":
    main()
