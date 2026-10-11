"""Revision experiments for TabStress (results in results/metrics/revision/<block>__seed<seed>.csv).

Blocks
  deploy        Protocol B (deployment/inference robustness): each model is fitted once per fold on
                CLEAN training data and evaluated on validation folds corrupted at every tier by the
                Gaussian, type-aware, MCAR, MAR and MNAR operators. Pipelines: "imputed" (median
                imputation, all six families) and "native" (no imputer, the five tree families,
                missingness operators only).
  train_mech    Protocol A (corruption-aware training robustness) for the new operators: training
                AND validation features corrupted (MAR, MNAR, type-aware), imputed pipeline.
  tuned         Configuration sensitivity: hyperparameters chosen by a small random search on clean
                data (inner 3-fold CV), then Protocol A (symmetric label noise, Gaussian noise) and
                Protocol B (Gaussian noise) with the tuned configuration.

Data, subsampling and outer CV are identical to experiments/run_extended.py. Workers only compute;
the main process is the only writer.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(var, "1")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.run_extended import MODELS, TIERS, TREES, build, data
from src.data_loader import BENCHMARK_SUITE
from src.data_loader import load_dataset
from src.noise_injector import MSNIE, categorical_aware_noise, continuous_gaussian_noise

OUT_DIR = "results/metrics/revision"
FEATURE_OPS = ["gaussian_feature", "type_aware_feature", "mcar_missing", "mar_missing", "mnar_missing"]
MISSING_OPS = ["mcar_missing", "mar_missing", "mnar_missing"]
TRAIN_MECH_OPS = ["mar_missing", "mnar_missing"]
# Compound corruption (Protocol A): two operators at the same tier at once, on six fast datasets
COMPOUND_OPS = ["label_mcar", "gauss_mcar"]
COMPOUND_DATASETS = ["vehicle", "qsar-biodeg", "steel-plates-fault", "ozone_level", "phoneme", "spambase"]
# Categorical-aware operators run on the datasets that have nominal attributes in their OpenML metadata
# (results/analysis/revision/openml_categorical.json); on the others the categorical-aware operator is the
# Gaussian operator by definition.
CAT_OPS = ["cat_aware_feature", "cat_only_feature"]
# Principal Gaussian experiment: Gaussian noise on continuous attributes only (nominal columns left clean)
CONT_OPS = ["gauss_cont_feature"]
CAT_META = "results/analysis/revision/openml_categorical.json"
# Protocol A with the MAR and MNAR operators runs on all twenty datasets, so nothing is excluded. (An earlier
# version skipped the six most expensive datasets, volkert, Bioresponse, pol, madelon, gesture_phase and miniboone,
# for compute reasons; that restriction was lifted when the runs were completed.)
TRAIN_MECH_EXCLUDED = set()
TUNED_TRIALS = 6
TUNED_DATASETS = list(BENCHMARK_SUITE)
TUNED_EXCLUDED = set()
SEARCH_SPACE = {
    "cart": {"max_depth": [None, 4, 8, 12, 16], "min_samples_leaf": [1, 2, 5, 10, 20]},
    "randomforest": {"max_features": ["sqrt", 0.3, 0.5], "min_samples_leaf": [1, 2, 5], "n_estimators": [100, 200]},
    "xgboost": {"max_depth": [3, 4, 6, 8], "learning_rate": [0.03, 0.1, 0.2], "n_estimators": [100, 200],
                "subsample": [0.7, 1.0], "colsample_bytree": [0.6, 1.0]},
    "lightgbm": {"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.1, 0.2], "n_estimators": [100, 200],
                 "min_child_samples": [5, 20, 50]},
    "catboost": {"depth": [4, 6, 8], "learning_rate": [0.03, 0.1, 0.2], "iterations": [100, 200],
                 "l2_leaf_reg": [1, 3, 10]},
    "mlp": {"hidden_layer_sizes": [(64,), (128, 64), (128, 64, 32), (256, 128)], "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-3, 3e-3]},
}


def discrete_columns(X):
    """Integer-coded columns (one-hot indicators, ordinal codes, counts) receive the categorical-aware
    resampling operator; every other column is treated as continuous."""
    return np.array([np.all(np.equal(np.mod(X[:, j], 1), 0)) for j in range(X.shape[1])])


def categorical_groups(dataset):
    """Encoded column indices of every nominal OpenML attribute of `dataset`: the dummy columns
    '<attr>_<level>' of a one-hot coded attribute, or the single ordinal column '<attr>'."""
    import json
    nominal = json.load(open(CAT_META)).get(dataset, [])
    names = [str(n) for n in load_dataset(dataset)[2]]
    owner = {}
    for j, n in enumerate(names):
        if n in nominal:                       # an ordinal-coded attribute keeps its own name
            owner[j] = n
            continue
        prefixes = [a for a in nominal if n.startswith(a + "_") and n not in nominal]
        if prefixes:                           # a dummy column belongs to the longest matching attribute
            owner[j] = max(prefixes, key=len)
    groups = [np.array([j for j, a in owner.items() if a == attr]) for attr in nominal]
    return [g for g in groups if len(g)]


def cat_datasets():
    import json
    return [d for d, cats in json.load(open(CAT_META)).items() if cats]


def mar_driver(X, seed):
    """The fully observed driver column of the MAR operator: a fixed random continuous column (any
    column if none is continuous), chosen once per (dataset, seed)."""
    cont = np.where(~discrete_columns(X))[0]
    pool = cont if len(cont) else np.arange(X.shape[1])
    return int(np.random.default_rng(seed).choice(pool))


def corrupt_features(inj, X, op, eta, ctx):
    if op == "gaussian_feature":
        return inj.inject_gaussian_feature_noise(X, gamma=eta)
    if op == "type_aware_feature":
        return inj.inject_type_aware_feature_noise(X, gamma=eta, discrete=ctx["discrete"], reference=ctx["reference"])
    if op == "gauss_cont_feature":
        return continuous_gaussian_noise(inj, X, eta, ctx["groups"])
    if op in ("cat_aware_feature", "cat_only_feature"):
        return categorical_aware_noise(inj, X, eta, ctx["groups"], ctx["reference"],
                                       continuous=op == "cat_aware_feature")
    if op == "mcar_missing":
        return inj.inject_mcar_missingness(X, h=eta)
    if op == "mar_missing":
        return inj.inject_mar_missingness(X, h=eta, driver=ctx["driver"])
    if op == "mnar_missing":
        return inj.inject_mnar_missingness(X, h=eta)
    raise ValueError(op)


def pipeline(kind, model, seed, params=None):
    pipe = build("native_nan" if kind == "native" else "base", model, seed)
    if params:
        pipe.named_steps["clf"].set_params(**params)
    return pipe


def scores(y, pred):
    return accuracy_score(y, pred), balanced_accuracy_score(y, pred)


def rows_from(acc, meta):
    out = []
    for (op, eta), vals in acc.items():
        a = np.array(vals)
        out.append(dict(meta, modality=op, eta=eta, mean_acc=float(a[:, 0].mean()), std_acc=float(a[:, 0].std()),
                        mean_bal_acc=float(a[:, 1].mean())))
    return out


def run_deploy(seed, dataset, model, kind, params=None, ops=None, block="deploy"):
    """Protocol B: one clean fit per fold, evaluation on every corrupted copy of the validation fold."""
    X, y, _ = data(dataset)
    ops = ops or (MISSING_OPS if kind == "native" else FEATURE_OPS)
    ctx = {"discrete": discrete_columns(X), "driver": mar_driver(X, seed), "groups": categorical_groups(dataset)}
    acc, t0 = {}, time.time()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        clf = pipeline(kind, model, seed, params).fit(X[tr], y[tr])
        ctx["reference"] = X[tr]
        for i, op in enumerate(ops):
            for eta in TIERS:
                inj = MSNIE(random_state=seed * 100_000 + fold * 1_000 + i * 100 + int(round(eta * 100)))
                Xva = corrupt_features(inj, X[va], op, eta, ctx)
                pred = np.asarray(clf.predict(Xva)).ravel().astype(y.dtype)
                acc.setdefault((op, eta), []).append(scores(y[va], pred))
    meta = dict(block=block, protocol="B", pipeline=kind, seed=seed, dataset=dataset, model=model)
    rows = rows_from(acc, meta)
    for r in rows:
        r["seconds"] = round(time.time() - t0, 2)
    return rows


def run_train(seed, dataset, model, op, eta, params=None, block="train_mech"):
    """Protocol A: training and validation features (or training labels) corrupted at the same tier."""
    X, y, k = data(dataset)
    ctx = {"discrete": discrete_columns(X), "driver": mar_driver(X, seed), "groups": categorical_groups(dataset)}
    inj = MSNIE(random_state=seed)
    acc, t0 = {}, time.time()
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y):
        Xtr, ytr, Xva = X[tr], y[tr].copy(), X[va]
        ctx["reference"] = X[tr]
        if op == "symmetric_label":
            ytr = inj.inject_symmetric_label_noise(ytr, eta=eta, num_classes=k)
        elif op in COMPOUND_OPS:
            if op == "label_mcar":
                ytr = inj.inject_symmetric_label_noise(ytr, eta=eta, num_classes=k)
            else:
                Xtr = corrupt_features(inj, Xtr, "gaussian_feature", eta, ctx)
                Xva = corrupt_features(inj, Xva, "gaussian_feature", eta, ctx)
            Xtr = corrupt_features(inj, Xtr, "mcar_missing", eta, ctx)
            Xva = corrupt_features(inj, Xva, "mcar_missing", eta, ctx)
        elif op != "clean":
            Xtr = corrupt_features(inj, Xtr, op, eta, ctx)
            Xva = corrupt_features(inj, Xva, op, eta, ctx)
        clf = pipeline("imputed", model, seed, params).fit(Xtr, ytr)
        pred = np.asarray(clf.predict(Xva)).ravel().astype(y.dtype)
        acc.setdefault((op, eta), []).append(scores(y[va], pred))
    rows = rows_from(acc, dict(block=block, protocol="A", pipeline="imputed", seed=seed, dataset=dataset, model=model))
    rows[0]["seconds"] = round(time.time() - t0, 2)
    return rows


def tune(dataset, model, seed):
    """Random search (TUNED_TRIALS configurations, inner 3-fold CV) on the clean data of the first outer
    training split; returns the best configuration and its inner-CV accuracy."""
    X, y, _ = data(dataset)
    tr, _ = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y))
    rng = np.random.default_rng(seed)
    space = SEARCH_SPACE[model]
    best, best_score = None, -1.0
    for _ in range(TUNED_TRIALS):
        params = {k: v[rng.integers(len(v))] for k, v in space.items()}
        pipe = pipeline("imputed", model, seed, params)
        s = cross_val_score(clone(pipe), X[tr], y[tr], cv=StratifiedKFold(3, shuffle=True, random_state=seed)).mean()
        if s > best_score:
            best, best_score = params, s
    return best, best_score


def run_tuned(seed, dataset, model):
    params, inner = tune(dataset, model, seed)
    rows = run_deploy(seed, dataset, model, "imputed", params=params, ops=["gaussian_feature"], block="tuned")
    for op in ["symmetric_label", "gaussian_feature"]:
        for eta in TIERS:
            rows += run_train(seed, dataset, model, op if eta > 0 else "clean", eta, params=params, block="tuned")
    for r in rows:
        r["params"], r["inner_cv_acc"] = repr(params), inner
    return rows


def plan(blocks, seeds_deploy, seeds_train, include_excluded=False):
    tasks = []
    order = sorted(BENCHMARK_SUITE, key=lambda d: d != "volkert")   # start the slowest dataset first
    if "deploy" in blocks:
        for s in seeds_deploy:
            tasks += [("deploy", s, d, m, "imputed") for d in order for m in MODELS]
            tasks += [("deploy", s, d, m, "native") for d in order for m in TREES]
    if "tuned" in blocks:
        tuned_sets = list(BENCHMARK_SUITE) if include_excluded else TUNED_DATASETS
        tasks += [("tuned", 42, d, m) for d in tuned_sets for m in MODELS]
    if "deploy_cat" in blocks:
        tasks += [("deploy_cat", s, d, m) for s in seeds_deploy for d in cat_datasets() for m in MODELS]
    if "train_cat" in blocks:
        tasks += [("train_cat", s, d, m, op, e) for s in seeds_deploy for d in cat_datasets() for m in MODELS
                  for op in CAT_OPS for e in TIERS if e > 0]
    if "deploy_cont" in blocks:
        tasks += [("deploy_cont", s, d, m) for s in seeds_deploy for d in cat_datasets() for m in MODELS]
    if "train_cont" in blocks:
        tasks += [("train_cont", s, d, m, op, e) for s in seeds_deploy for d in cat_datasets() for m in MODELS
                  for op in CONT_OPS for e in TIERS if e > 0]
    if "train_mech" in blocks:
        for s in seeds_train:
            tasks += [("train_mech", s, d, m, op, e) for d in order
                      if include_excluded or d not in TRAIN_MECH_EXCLUDED
                      for m in MODELS for op in TRAIN_MECH_OPS
                      for e in TIERS if e > 0]
    if "compound" in blocks:
        tasks += [("compound", s, d, m, op, e) for s in seeds_train for d in COMPOUND_DATASETS
                  for m in MODELS for op in COMPOUND_OPS for e in TIERS if e > 0]
    return tasks


def execute(task):
    if task[0] == "deploy":
        return run_deploy(*task[1:])
    if task[0] == "tuned":
        return run_tuned(*task[1:])
    if task[0] == "deploy_cat":
        return run_deploy(*task[1:], "imputed", ops=CAT_OPS, block="deploy_cat")
    if task[0] == "train_cat":
        return run_train(*task[1:], block="train_cat")
    if task[0] == "deploy_cont":
        return run_deploy(*task[1:], "imputed", ops=CONT_OPS, block="deploy_cont")
    if task[0] == "train_cont":
        return run_train(*task[1:], block="train_cont")
    if task[0] == "compound":
        return run_train(*task[1:], block="compound")
    return run_train(*task[1:])


def task_id(t):
    return "|".join(map(str, t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--blocks", nargs="+", default=["deploy", "tuned", "train_mech"])
    ap.add_argument("--seeds-deploy", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--seeds-train", type=int, nargs="+", default=[42])
    ap.add_argument("--include-excluded", action="store_true",
                    help="also run the datasets omitted for compute (completion runs; resumable)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    log = os.path.join(OUT_DIR, "done_tasks.txt")
    done = set(open(log).read().split("\n")) if os.path.exists(log) else set()
    tasks = [t for t in plan(args.blocks, args.seeds_deploy, args.seeds_train, args.include_excluded) if task_id(t) not in done]
    print(f"{len(done)} tasks done, {len(tasks)} pending", flush=True)
    t_start, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:
                print(f"FAILED {task}: {exc!r}", flush=True)
                continue
            path = os.path.join(OUT_DIR, f"{task[0]}__seed{task[1]}.csv")
            pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            with open(log, "a") as fh:
                fh.write(task_id(task) + "\n")
            n += 1
            if n % 20 == 0 or n == len(tasks):
                rate = n / (time.time() - t_start)
                print(f"{n}/{len(tasks)} tasks, ETA {(len(tasks) - n) / rate / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
