import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "data")
TRAIN = os.path.join(DATA_DIR, "training_set.xlsx")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "xgboost_grid_search")

# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["peep", "d3_sofa", "d3_lac", "d3_a_ado2"]
OUTCOME = "outcome"

SEED = 42
N_SPLITS = 10

RANGE = {
    "max_depth": (3, 4),
    "learning_rate": (0.50, 0.60),
    "n_estimators": (1, 2),
    "min_child_weight": (1, 10),
    "subsample": (0.60, 0.90),
    "colsample_bytree": (0.50, 0.90),
}

LEVELS_MODE = "endpoints"
INT_PARAMS = {"max_depth", "n_estimators", "min_child_weight"}

BASE_PARAMS = dict(
    objective="binary:logistic",
    booster="gbtree",
    tree_method="hist",
    max_bin=64,
    scale_pos_weight=1.0,
    seed=SEED,
    nthread=1,
    verbosity=0,
)


def load_train(path=TRAIN):
    if not os.path.exists(path):
        raise SystemExit("file not found: %s" % path)
    engine = "xlrd" if path.lower().endswith(".xls") else "openpyxl"
    frame = pd.read_excel(path, sheet_name=0, engine=engine)
    frame = frame.loc[:, ~frame.columns.duplicated()]

    missing = [c for c in FEATURES + [OUTCOME] if c not in frame.columns]
    if missing:
        raise SystemExit("missing columns in training set: %s" % missing)

    X = frame[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = pd.to_numeric(frame[OUTCOME], errors="coerce").astype(int).to_numpy()
    return X, y


def cv_auc(params, X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    aucs = []
    for tr_i, va_i in skf.split(X, y):
        bst = xgb.train(params,
                        xgb.DMatrix(X[tr_i], label=y[tr_i], nthread=1),
                        int(params["n_estimators"]))
        prob = bst.predict(xgb.DMatrix(X[va_i], nthread=1))
        aucs.append(float(roc_auc_score(y[va_i], prob)))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1)), aucs


def complexity(params):
    return float(params["max_depth"]) * 100 + float(params["n_estimators"])


def axis_values(name, lo, hi):
    if LEVELS_MODE == "dense":
        raw = [lo, (lo + hi) / 2.0, hi]
    else:
        raw = [lo, hi]

    if name in INT_PARAMS:
        out, seen = [], set()
        for value in raw:
            ivalue = int(round(value))
            if ivalue not in seen:
                seen.add(ivalue)
                out.append(ivalue)
        return out
    return [round(float(v), 6) for v in raw]


def build_grid():
    bad = {k: v for k, v in RANGE.items() if not float(v[1]) > float(v[0])}
    if bad:
        raise SystemExit("invalid range, upper bound not above lower: %s" % bad)

    axes = {name: axis_values(name, lo, hi) for name, (lo, hi) in RANGE.items()}
    names = list(RANGE.keys())
    grid = [dict(zip(names, combo))
            for combo in itertools.product(*[axes[n] for n in names])]

    print("levels mode: %s" % LEVELS_MODE)
    for name in names:
        print("  %-18s %s" % (name, axes[name]))
    print("  total combinations = %d" % len(grid))
    return grid


def main():
    start = time.time()
    print("=" * 84)
    print("XGBoost grid search on the training set")
    print("ranking criterion: highest mean 10-fold cv auc")
    print("random_state = %d" % SEED)
    print("=" * 84)

    X, y = load_train()
    print()
    print("training set %d cases (%d events), features %s"
          % (X.shape[0], int(y.sum()), FEATURES))
    print()
    print("parameter range:")
    for name, (lo, hi) in RANGE.items():
        print("  %-18s [%s, %s]" % (name, lo, hi))
    print()

    grid = build_grid()

    print()
    print("searching ...")
    rows = []
    for i, combo in enumerate(grid, 1):
        params = {**BASE_PARAMS, **combo}
        mean, sd, folds = cv_auc(params, X, y)
        rows.append({
            **combo,
            "cv_auc_mean": mean,
            "cv_auc_sd": sd,
            "cv_folds": "|".join("%.4f" % v for v in folds),
            "complexity": complexity(combo),
        })
        if i % 20 == 0 or i == len(grid):
            print("  [%3d/%d] best cv so far = %.4f"
                  % (i, len(grid), max(r["cv_auc_mean"] for r in rows)))

    df = pd.DataFrame(rows).sort_values(
        ["cv_auc_mean", "complexity"],
        ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "grid_all.csv"), index=False,
              encoding="utf-8-sig")

    head = ["rank", "max_depth", "learning_rate", "n_estimators",
            "min_child_weight", "subsample", "colsample_bytree",
            "cv_auc_mean", "cv_auc_sd"]

    print()
    print("=" * 84)
    print("results, descending cv auc (top %d)" % min(15, len(df)))
    print("=" * 84)
    print(df[head].head(15).to_string(index=False,
                                     float_format=lambda x: "%.4f" % x))

    top = df.iloc[0]
    best_params = {k: (int(round(float(top[k]))) if k in INT_PARAMS
                       else round(float(top[k]), 6)) for k in RANGE}

    print()
    print("-" * 84)
    print("best combination")
    print("-" * 84)
    for k, v in best_params.items():
        print("  %-18s = %s" % (k, v))
    print()
    print("  cv auc              %.4f +- %.4f (fold sd, ddof=1)"
          % (top.cv_auc_mean, top.cv_auc_sd))
    print("  fold aucs           %s" % top.cv_folds)
    print("  fold range          %.4f - %.4f"
          % (min(float(v) for v in top.cv_folds.split("|")),
             max(float(v) for v in top.cv_folds.split("|"))))
    print("  cv auc distribution max %.4f / median %.4f / min %.4f"
          % (df.cv_auc_mean.max(), df.cv_auc_mean.median(), df.cv_auc_mean.min()))

    payload = {
        "model": "xgboost",
        "criterion": "highest mean 10-fold cv auc",
        "random_state": SEED,
        "n_splits": N_SPLITS,
        "levels_mode": LEVELS_MODE,
        "range": {k: list(v) for k, v in RANGE.items()},
        "n_candidates": int(len(df)),
        "best_params": best_params,
        "cv_auc_mean": float(top.cv_auc_mean),
        "cv_auc_sd": float(top.cv_auc_sd),
        "cv_folds": [float(v) for v in top.cv_folds.split("|")],
    }
    with open(os.path.join(OUT_DIR, "grid_best.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print("saved: %s/grid_all.csv, grid_best.json" % OUT_DIR)
    print("elapsed %.1f min" % ((time.time() - start) / 60))


if __name__ == "__main__":
    main()
