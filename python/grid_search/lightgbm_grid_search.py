import itertools
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "data")
TRAIN = os.path.join(DATA_DIR, "training set.xlsx")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "lightgbm_grid_search")

# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
TARGET = "outcome"
CV_FOLDS = 10
CV_SEED = 42
N_JOBS = 8

RANGE = dict(
    num_leaves=[3, 4],
    min_child_samples=[20, 30],
    learning_rate=[0.006, 0.008],
    n_estimators=[150, 200],
    feature_fraction=[0.9, 1.0],
    bagging_fraction=[0.8, 1.0],
    lambda_l1=[3.0, 10.0],
    lambda_l2=[0.0, 3.0],
)

AXES = ["num_leaves", "min_child_samples", "learning_rate", "n_estimators",
        "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2"]
INT_AXES = ("num_leaves", "min_child_samples", "n_estimators")

FIXED_PARAMS = dict(
    objective="binary", boosting_type="gbdt", metric="auc",
    max_depth=-1, max_bin=63, min_split_gain=0.0, bagging_freq=1,
    scale_pos_weight=1.0, is_unbalance=False, verbose=-1,
    deterministic=True, force_col_wise=True, num_threads=1,
    seed=42, bagging_seed=43, feature_fraction_seed=44,
)

DATASET_PARAMS = dict(max_bin=FIXED_PARAMS["max_bin"], min_data_in_bin=3,
                      data_random_seed=42, feature_pre_filter=False, verbose=-1)


def load_excel(path):
    try:
        return pd.read_excel(path, engine="calamine")
    except Exception:
        return pd.read_excel(path)


def load_xy(path):
    frame = load_excel(path)
    X = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(frame[TARGET], errors="coerce").astype(int).values
    return X, y


def build_folds(X, y):
    return list(StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                random_state=CV_SEED).split(X, y))


def evaluate(combo, X, y, folds):
    params = dict(FIXED_PARAMS,
                  num_leaves=int(combo["num_leaves"]),
                  min_data_in_leaf=int(combo["min_child_samples"]),
                  learning_rate=float(combo["learning_rate"]),
                  feature_fraction=float(combo["feature_fraction"]),
                  bagging_fraction=float(combo["bagging_fraction"]),
                  lambda_l1=float(combo["lambda_l1"]),
                  lambda_l2=float(combo["lambda_l2"]))
    if params["bagging_fraction"] >= 1.0:
        params["bagging_freq"] = 0
    n_trees = int(combo["n_estimators"])

    oof = np.zeros(len(y), dtype=float)
    aucs = []
    for tr, te in folds:
        dtr = lgb.Dataset(X.iloc[tr], label=y[tr], free_raw_data=False,
                          params=DATASET_PARAMS)
        model = lgb.train(params, dtr, num_boost_round=n_trees)
        oof[te] = model.predict(X.iloc[te], num_iteration=n_trees)
        aucs.append(roc_auc_score(y[te], oof[te]))

    aucs = np.array(aucs)
    return dict(cv_pooled=float(roc_auc_score(y, oof)),
                cv_fold_mean=float(aucs.mean()),
                cv_fold_sd=float(aucs.std(ddof=1)),
                cv_fold_se=float(aucs.std(ddof=1) / np.sqrt(len(aucs))))


def main():
    start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    for axis in AXES:
        assert axis in RANGE
        assert len(RANGE[axis]) >= 2
        assert RANGE[axis][0] != RANGE[axis][-1]

    X, y = load_xy(TRAIN)
    folds = build_folds(X, y)
    combos = [dict(zip(AXES, values))
              for values in itertools.product(*[RANGE[a] for a in AXES])]

    print("=" * 96)
    print("LightGBM grid search on the training set")
    print("=" * 96)
    print("training %d cases (%d events) | fold sizes %s"
          % (len(y), int(y.sum()), [len(b) for _, b in folds]))
    print("cv_seed = %d | early stopping off | %d combinations"
          % (CV_SEED, len(combos)))
    print()
    for axis in AXES:
        print("  %-20s %6s ~ %-6s %s"
              % (axis, RANGE[axis][0], RANGE[axis][-1], RANGE[axis]))

    t1 = time.time()
    results = Parallel(n_jobs=N_JOBS, batch_size=4, verbose=0)(
        delayed(evaluate)(combo, X, y, folds) for combo in combos)
    df = pd.DataFrame(combos)
    for key in ("cv_pooled", "cv_fold_mean", "cv_fold_sd", "cv_fold_se"):
        df[key] = [r[key] for r in results]
    df.insert(0, "src_idx", np.arange(len(df)))

    df = df.sort_values(["cv_fold_mean", "num_leaves", "min_child_samples"],
                        ascending=[False, True, False]).reset_index(drop=True)
    df.round(6).to_csv(os.path.join(OUT_DIR, "all_combos.csv"),
                       index=False, encoding="utf-8-sig")
    df.head(10).round(6).to_csv(os.path.join(OUT_DIR, "top10.csv"),
                                index=False, encoding="utf-8-sig")

    print()
    print("done in %.1fs | cv auc range %.6f - %.6f"
          % (time.time() - t1, df.cv_fold_mean.min(), df.cv_fold_mean.max()))
    print()
    print("top 10 by mean fold auc")
    print(df.head(10)[AXES + ["cv_pooled", "cv_fold_mean", "cv_fold_sd"]]
          .to_string(index=False, float_format=lambda v: "%.4f" % v))

    best = df.iloc[0]
    n_tie = int((df.cv_fold_mean.sub(best.cv_fold_mean).abs() < 1e-12).sum())
    print()
    print("best combination (cv ties: %d)" % n_tie)
    for axis in AXES:
        print("  %-20s %s" % (axis, best[axis]))
    print("  %-20s %.9f" % ("cv auc fold mean", best.cv_fold_mean))
    print("  %-20s %.9f" % ("cv auc pooled", best.cv_pooled))
    print("  %-20s %.6f" % ("cv fold sd", best.cv_fold_sd))
    print("  %-20s %.6f" % ("cv fold se", best.cv_fold_se))

    payload = dict(
        range_bounds={a: [RANGE[a][0], RANGE[a][-1]] for a in AXES},
        range_levels={a: RANGE[a] for a in AXES},
        params={a: (int(best[a]) if a in INT_AXES else float(best[a])) for a in AXES},
        cv_pooled=float(best.cv_pooled),
        cv_fold_mean=float(best.cv_fold_mean),
        cv_fold_sd=float(best.cv_fold_sd),
        cv_fold_se=float(best.cv_fold_se),
        n_combos=len(combos), n_tie=n_tie,
        cv_protocol=dict(splitter="StratifiedKFold", n_splits=CV_FOLDS,
                         shuffle=True, random_state=CV_SEED,
                         n_samples=len(y), fold_sizes=[len(g) for _, g in folds],
                         metric="mean of the 10 per-fold aucs",
                         early_stopping=False,
                         ranking="cv_fold_mean descending"),
    )
    with open(os.path.join(OUT_DIR, "best.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print("saved to %s | elapsed %.1fs" % (OUT_DIR, time.time() - start))


if __name__ == "__main__":
    main()
