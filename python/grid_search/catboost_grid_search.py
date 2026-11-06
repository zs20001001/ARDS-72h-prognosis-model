import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "data")
TRAIN = os.path.join(DATA_DIR, "training set.xlsx")
# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
TARGET = "outcome"

CV_FOLDS = 10
CV_SEED = 42
MODEL_SEED = 42
THREAD_COUNT = 1

RANGE = {
    "depth": (6, 7),
    "learning_rate": (0.05, 0.06),
    "iterations": (180, 200),
    "l2_leaf_reg": (100.0, 120.0),
    "random_strength": (6.0, 7.0),
    "border_count": (80, 96),
}

STEP = {
    "depth": 1,
    "learning_rate": 0.01,
    "iterations": 20,
    "l2_leaf_reg": 20.0,
    "random_strength": 1.0,
    "border_count": 16,
}

INT_AXES = ("depth", "iterations", "border_count")
KEYS = list(RANGE.keys())

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(OUT_DIR, "catboost_grid_search.csv")
OUT_JSON = os.path.join(OUT_DIR, "catboost_grid_search_best.json")

assert CV_SEED == 42

_CACHE = {}


def data():
    if not _CACHE:
        from sklearn.model_selection import StratifiedKFold
        frame = pd.read_excel(TRAIN)
        X = frame[FEATURES]
        y = frame[TARGET].astype(int).values
        _CACHE["X"] = X
        _CACHE["y"] = y
        _CACHE["folds"] = list(StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED).split(X, y))
    return _CACHE


def levels(key):
    lo, hi = RANGE[key]
    if key in INT_AXES:
        return list(range(int(lo), int(hi) + 1, int(STEP[key])))
    n = int(round((hi - lo) / STEP[key])) + 1
    return [round(lo + i * STEP[key], 10) for i in range(n)]


def cv_auc(params):
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    cache = data()
    X, y = cache["X"], cache["y"]
    folds = []
    for tr, te in cache["folds"]:
        model = CatBoostClassifier(
            **params, loss_function="Logloss", eval_metric="AUC",
            random_seed=MODEL_SEED, thread_count=THREAD_COUNT,
            verbose=False, allow_writing_files=False)
        model.fit(X.iloc[tr], y[tr])
        folds.append(float(roc_auc_score(y[te],
                                         model.predict_proba(X.iloc[te])[:, 1])))
    folds = np.asarray(folds)
    row = dict(params)
    row["cv_auc"] = float(folds.mean())
    row["cv_std"] = float(folds.std(ddof=0))
    row["cv_se"] = float(folds.std(ddof=0) / np.sqrt(len(folds)))
    row["fold_aucs"] = " ".join("%.6f" % v for v in folds)
    return row


def main():
    start = time.time()
    grids = [levels(k) for k in KEYS]
    combos = list(product(*grids))

    print("=" * 78)
    print("CatBoost grid search on the training set")
    print("=" * 78)
    print("cv_seed      = %d" % CV_SEED)
    print("model_seed   = %d" % MODEL_SEED)
    print("thread_count = %d" % THREAD_COUNT)
    print("folds        = %d" % CV_FOLDS)
    print()
    for key, grid in zip(KEYS, grids):
        lo, hi = RANGE[key]
        print("  %-17s [%s, %s]  step %-5s levels %s"
              % (key, lo, hi, STEP[key], grid))
    print()
    print("combinations = %d" % len(combos))
    print("=" * 78, flush=True)

    rows = []
    workers = min(10, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(cv_auc, dict(zip(KEYS, combo))) for combo in combos]
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 16 == 0 or done == len(combos):
                print("  progress %3d/%d   %.1f min"
                      % (done, len(combos), (time.time() - start) / 60), flush=True)

    df = pd.DataFrame(rows).sort_values(
        "cv_auc", ascending=False, kind="mergesort").reset_index(drop=True)
    df[KEYS + ["cv_auc", "cv_std", "cv_se", "fold_aucs"]].to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig")

    print()
    print("top 8 by cv auc")
    print(df.head(8)[KEYS + ["cv_auc", "cv_std"]].to_string(index=False))

    champ = df.iloc[0]
    gap = float(champ.cv_auc - df.cv_auc.iloc[1]) if len(df) > 1 else float("nan")

    print()
    print("best combination (%d combinations searched)" % len(combos))
    for key in KEYS:
        value = int(champ[key]) if key in INT_AXES else float(champ[key])
        print("  %-17s %s" % (key, value))
    print("  %-17s %.6f" % ("cv auc", champ.cv_auc))
    print("  %-17s %.6f" % ("cv sd", champ.cv_std))
    print("  %-17s %s" % ("fold aucs", champ.fold_aucs))
    print("  %-17s %+.6f" % ("gap to runner-up", gap))

    payload = {
        "model": "catboost",
        "cv_folds": CV_FOLDS,
        "seeds": {"cv_seed": CV_SEED, "model_seed": MODEL_SEED,
                  "thread_count": THREAD_COUNT},
        "range": {key: list(RANGE[key]) for key in KEYS},
        "step": dict(STEP),
        "levels": {key: [(int(v) if key in INT_AXES else float(v)) for v in grid]
                   for key, grid in zip(KEYS, grids)},
        "n_combos": len(combos),
        "ranking": "highest mean 10-fold cv auc",
        "best_params": {key: (int(champ[key]) if key in INT_AXES
                              else float(champ[key])) for key in KEYS},
        "cv_auc": float(champ.cv_auc),
        "cv_std": float(champ.cv_std),
        "cv_se": float(champ.cv_se),
        "fold_aucs": [float(v) for v in str(champ.fold_aucs).split()],
        "gap_over_runner_up": gap,
        "runner_up_cv_auc": float(df.cv_auc.iloc[1]) if len(df) > 1 else None,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print("saved: %s" % os.path.basename(OUT_CSV))
    print("saved: %s" % os.path.basename(OUT_JSON))
    print("elapsed %.1f min" % ((time.time() - start) / 60))


if __name__ == "__main__":
    main()
