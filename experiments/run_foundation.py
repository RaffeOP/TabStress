"""Pretrained tabular foundation model (TabICL) under Protocol B, imputed pipeline.

Results go to results/metrics/revision/foundation__seed<seed>.csv with the schema of the deploy block
(block="foundation", protocol="B", pipeline="imputed", model="tabicl").

Protocol B as in experiments/run_revision.run_deploy: per fold, ONE fit on CLEAN training data (a median
imputer fitted on the clean training fold, then the in-context learner), then evaluation on every corrupted
copy of the validation fold. Corruption is drawn exactly as in run_deploy: MSNIE seeded with
seed * 100_000 + fold * 1_000 + i * 100 + round(eta * 100), where i is the index of the operator in
FEATURE_OPS, so the corrupted validation data are bit-identical to those of the six classical families.

Run with the TabICL environment:  .venv-tabpfn/bin/python experiments/run_foundation.py
Single process, single writer; resumable through results/metrics/revision/done_foundation.txt.
"""

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
import types

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
for _missing in ("xgboost", "catboost"):          # classical-family libraries are not needed here; the stubs
    try:                                          # only let run_revision (whose operators we reuse) import
        __import__(_missing)
    except ImportError:
        sys.modules[_missing] = types.ModuleType(_missing)
        if _missing == "catboost":
            sys.modules[_missing].CatBoostClassifier = object
from experiments.run_extended import TIERS, data                                         # noqa: E402
from experiments.run_revision import FEATURE_OPS, corrupt_features, discrete_columns, mar_driver  # noqa: E402
from src.data_loader import BENCHMARK_SUITE                                              # noqa: E402
from src.noise_injector import MSNIE                                                     # noqa: E402

OUT_DIR = "results/metrics/revision"
DONE_LOG = os.path.join(OUT_DIR, "done_foundation.txt")
RUN_LOG = os.path.join(OUT_DIR, "run_foundation.log")
SETTINGS_FILE = os.path.join(OUT_DIR, "foundation_settings.json")
UNSUPPORTED = {"Bioresponse": "1,776 features: column-wise embedding does not fit the 8 GB unified memory of the "
                              "machine (MPS out of memory / 5 GB output buffer with 3,000 context rows, 4 estimators); "
                              "predicting 300 rows took more than 8 minutes, so the 31,000 test rows per fold are infeasible"}
OPS = ["gaussian_feature", "mcar_missing", "mar_missing", "mnar_missing"]
MODEL_NAME = "tabicl"
N_FOLDS = 5
TORCH_THREADS = 2
DEFAULT_N_ESTIMATORS = 8     # TabICLClassifier default
DEFAULT_CHUNK_ROWS = 2_000
NO_CACHE_CHUNK_ROWS = 20_000
MAX_NATIVE_CLASSES = 10     # TabICL's native class limit; more classes use hierarchical classification
OFFLOAD_DIR = os.path.join(tempfile.gettempdir(), "tabicl_offload")
MEMORY_SAFETY_FACTOR = float(os.environ.get("TABICL_SAFETY", "0.05"))
DEFAULT_BATCH_SIZE = 8       # TabICLClassifier default: ensemble members forwarded together


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a") as fh:
        fh.write(line + "\n")


def make_classifier(seed, n_estimators, device, kv_cache="kv", batch_size=DEFAULT_BATCH_SIZE):
    from tabicl import TabICLClassifier
    from tabicl import InferenceConfig
    cache = {"kv": True, "repr": "repr", "none": False}[kv_cache]
    config = InferenceConfig()                    # library defaults, except the memory safety factor: the
    for stage in (config.COL_CONFIG, config.ROW_CONFIG, config.ICL_CONFIG):   # inference manager otherwise sizes
        stage.safety_factor = MEMORY_SAFETY_FACTOR                            # its batches to most free memory
        stage.disk_offload_dir = OFFLOAD_DIR      # outputs too large for RAM go to disk instead of failing
    return TabICLClassifier(n_estimators=n_estimators, device=device, random_state=seed, kv_cache=cache,
                            batch_size=batch_size, inference_config=config)


def cap_training_set(Xtr, ytr, max_train, seed, fold):
    """Stratified subsample of the in-context training set, seeded per (seed, fold)."""
    if max_train is None or len(ytr) <= max_train:
        return Xtr, ytr
    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_train, random_state=seed * 1_000 + fold)
    idx, _ = next(sss.split(Xtr, ytr))
    return Xtr[idx], ytr[idx]


def corrupted_copies(X_va, X_ref, ctx, seed, fold, ops):
    """The corrupted copies of the validation fold for `ops`, drawn exactly as run_deploy does (the generator
    of every (operator, tier) is seeded independently, so drawing per operator changes nothing). The clean copy
    (eta = 0, identical for every operator) is key None when `ops` is None."""
    ctx["reference"] = X_ref
    if ops is None:
        return {None: X_va}
    copies = {}
    for op in ops:
        i = FEATURE_OPS.index(op)
        for eta in TIERS:
            if eta == 0:
                continue
            inj = MSNIE(random_state=seed * 100_000 + fold * 1_000 + i * 100 + int(round(eta * 100)))
            copies[(op, eta)] = corrupt_features(inj, X_va, op, eta, ctx)
    return copies


def predict_copies(clf, imputer, copies, batched, chunk_rows=DEFAULT_CHUNK_ROWS):
    """Predicted labels for every copy. Batched: the copies are stacked and sent through predict in chunks of
    `chunk_rows` rows (chunking only bounds memory; rows are independent given the fitted context)."""
    keys = list(copies)
    imputed = [imputer.transform(copies[k]) for k in keys]
    if not batched:
        return {k: np.asarray(clf.predict(a)) for k, a in zip(keys, imputed)}
    stacked = np.vstack(imputed)
    pred = np.concatenate([np.asarray(clf.predict(stacked[i:i + chunk_rows]))
                           for i in range(0, len(stacked), chunk_rows)])
    return dict(zip(keys, np.split(pred, np.cumsum([len(a) for a in imputed])[:-1])))


def run_fold(clf, imputer, X_va, y_va, X_tr, ctx, seed, fold, batched, acc, chunk_rows=DEFAULT_CHUNK_ROWS):
    """Predict the clean copy and, operator by operator, its ten corrupted copies; free memory in between."""
    for ops in [None] + [[op] for op in OPS]:
        copies = corrupted_copies(X_va, X_tr, ctx, seed, fold, ops)
        preds = predict_copies(clf, imputer, copies, batched, chunk_rows)
        for (op, eta), pred in ([((o, 0.0), preds[None]) for o in OPS] if ops is None else preds.items()):
            pred = pred.astype(y_va.dtype)
            acc.setdefault((op, eta), []).append((accuracy_score(y_va, pred), balanced_accuracy_score(y_va, pred)))
        del copies, preds
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def run_dataset(seed, dataset, n_estimators, max_train, device, batched=True, kv_cache="kv"):
    torch.set_num_threads(TORCH_THREADS)
    X, y, n_classes = data(dataset)
    chunk_rows = DEFAULT_CHUNK_ROWS
    if kv_cache != "none" and n_classes > MAX_NATIVE_CLASSES:
        # TabICL cannot cache the context of a problem with more classes than its native limit: without a cache
        # every predict call recomputes the context, so use few large calls (one operator group per call)
        kv_cache, chunk_rows = "none", NO_CACHE_CHUNK_ROWS
        log(f"  {dataset}: {n_classes} classes > {MAX_NATIVE_CLASSES}, running without KV cache")
    ctx = {"discrete": discrete_columns(X), "driver": mar_driver(X, seed), "groups": []}
    acc, t0 = {}, time.time()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        t_fold = time.time()
        imputer = SimpleImputer(strategy="median").fit(X[tr])
        Xtr, ytr = cap_training_set(X[tr], y[tr], max_train, seed, fold)
        clf = make_classifier(seed, n_estimators, device, kv_cache).fit(imputer.transform(Xtr), ytr)
        run_fold(clf, imputer, X[va], y[va], X[tr], ctx, seed, fold, batched, acc, chunk_rows)
        log(f"  seed {seed} {dataset} fold {fold} done in {time.time() - t_fold:.0f}s "
            f"(train {len(ytr)}, val {len(va)}, d={X.shape[1]}, rss {rss_gb():.1f} GB)")
        del clf
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    meta = dict(block="foundation", protocol="B", pipeline="imputed", seed=seed, dataset=dataset, model=MODEL_NAME)
    return [dict(meta, modality=op, eta=eta, mean_acc=float(np.mean([a for a, _ in v])),
                 std_acc=float(np.std([a for a, _ in v])), mean_bal_acc=float(np.mean([b for _, b in v])),
                 seconds=round(time.time() - t0, 2)) for (op, eta), v in acc.items()]


def rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def task_id(seed, dataset):
    return f"foundation|{seed}|{dataset}|{MODEL_NAME}|imputed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--datasets", nargs="+", default=list(BENCHMARK_SUITE))
    ap.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    ap.add_argument("--max-train", type=int, default=None, help="cap on the in-context training set (stratified)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-batch", action="store_true", help="one predict call per corrupted copy")
    ap.add_argument("--kv-cache", default="kv", choices=["kv", "repr", "none"],
                    help="cache of the training context: kv (fast, memory-heavy), repr (smaller), none (recompute)")
    ap.add_argument("--tag", default="", help="settings-sensitivity run: writes foundation_<tag>__* files, "
                                               "so the main results are never mixed with it")
    args = ap.parse_args()
    global DONE_LOG, RUN_LOG, SETTINGS_FILE
    stem = f"foundation_{args.tag}" if args.tag else "foundation"
    if args.tag:
        DONE_LOG = os.path.join(OUT_DIR, f"done_{stem}.txt")
        RUN_LOG = os.path.join(OUT_DIR, f"run_{stem}.log")
        SETTINGS_FILE = os.path.join(OUT_DIR, f"{stem}_settings.json")
    failures = os.path.join(OUT_DIR, f"{stem}_failures.txt")
    torch.set_num_threads(TORCH_THREADS)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    settings = dict(model="tabicl", n_estimators=args.n_estimators, max_train=args.max_train, device=args.device,
                    batched=not args.no_batch, kv_cache=args.kv_cache, chunk_rows=DEFAULT_CHUNK_ROWS,
                    memory_safety_factor=MEMORY_SAFETY_FACTOR, torch=torch.__version__, torch_threads=TORCH_THREADS)
    from importlib.metadata import version
    settings["tabicl_version"] = version("tabicl")
    json.dump(settings, open(SETTINGS_FILE, "w"), indent=1)
    done = set(open(DONE_LOG).read().split("\n")) if os.path.exists(DONE_LOG) else set()
    for dataset, reason in UNSUPPORTED.items():
        if dataset in args.datasets:
            log(f"SKIPPED {dataset}: {reason}")
            open(failures, "a").write(f"all\t{dataset}\t{reason}\n")
    tasks = [(s, d) for s in args.seeds for d in args.datasets
             if task_id(s, d) not in done and d not in UNSUPPORTED]
    log(f"settings {settings}; {len(done)} tasks done, {len(tasks)} pending")
    t_start = time.time()
    for n, (seed, dataset) in enumerate(tasks, 1):
        t0 = time.time()
        try:
            # a fresh process per task, so that no accelerator memory survives from the previous dataset
            with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
                rows = pool.apply(run_dataset, (seed, dataset, args.n_estimators, args.max_train, args.device,
                                                not args.no_batch, args.kv_cache))
        except Exception as exc:                  # record why a dataset failed and continue
            log(f"FAILED seed {seed} {dataset}: {exc!r}")
            with open(failures, "a") as fh:
                fh.write(f"{seed}\t{dataset}\t{exc!r}\n")
            if args.device == "mps":
                torch.mps.empty_cache()
            continue
        path = os.path.join(OUT_DIR, f"{stem}__seed{seed}.csv")
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        with open(DONE_LOG, "a") as fh:
            fh.write(task_id(seed, dataset) + "\n")
        log(f"DONE seed {seed} {dataset} in {(time.time() - t0) / 60:.1f} min; {n}/{len(tasks)} tasks, "
            f"elapsed {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
