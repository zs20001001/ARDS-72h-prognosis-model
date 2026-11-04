import json
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parents[1] / "data"
TRAIN_FILE = "training_set.xlsx"

TARGET = "outcome"
# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]

RANGE = {
    "criterion": ["entropy"],
    "max_depth": [2, 3],
    "min_samples_split": [2, 3, 5, 8, 10, 15, 20, 30],
    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
    "ccp_alpha": [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006,
                  0.007, 0.008, 0.009, 0.010, 0.011, 0.012,
                  0.014, 0.017, 0.020],
}
NAMES = list(RANGE.keys())

FIXED = dict(
    splitter="best",
    max_features=None,
    max_leaf_nodes=None,
    class_weight=None,
    min_impurity_decrease=0.0,
    min_weight_fraction_leaf=0.0,
    random_state=42,
)

CV_SEED = 42
N_SPLITS = 10
N_JOBS = -1
CHUNK = 64

OUT_RANK = BASE_DIR / "decision_tree_grid_search.csv"
OUT_BEST = BASE_DIR / "decision_tree_grid_search_best.json"


def resolve_path(name):
    candidates = [DATA_DIR / name, BASE_DIR / name]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("training file not found, tried: %s"
                            % ", ".join(str(p) for p in candidates))


def load_train():
    path = resolve_path(TRAIN_FILE)
    frame = pd.read_excel(path, sheet_name=0)
    missing = [c for c in FEATURES + [TARGET] if c not in frame.columns]
    if missing:
        raise KeyError("missing columns: %s" % missing)
    X = frame[FEATURES].apply(pd.to_numeric, errors="coerce").values.astype(float)
    y = frame[TARGET].astype(int).values
    if sorted(set(y.tolist())) != [0, 1]:
        raise ValueError("outcome must be 0/1, got %s" % sorted(set(y.tolist())))
    print("  training set n=%d  events=%d (%.1f%%)  missing=%d  file=%s"
          % (len(y), int(y.sum()), 100 * y.mean(), int(np.isnan(X).sum()),
             path.name))
    return X, y


def evaluate(X, y, folds, params):
    p = dict(FIXED)
    p.update(params)

    fold_auc = []
    for a, b in folds:
        model = DecisionTreeClassifier(**p).fit(X[a], y[a])
        if len(np.unique(y[b])) < 2:
            fold_auc.append(float("nan"))
        else:
            fold_auc.append(float(roc_auc_score(y[b],
                                                model.predict_proba(X[b])[:, 1])))

    fold_auc = np.asarray(fold_auc, dtype=float)
    full = DecisionTreeClassifier(**p).fit(X, y)
    row = dict(params)
    row.update({
        "cv_auc": float(np.nanmean(fold_auc)),
        "cv_auc_sd": float(np.nanstd(fold_auc, ddof=1)),
        "n_leaves": int(full.get_n_leaves()),
        "tree_depth": int(full.get_depth()),
    })
    return row


def evaluate_chunk(X, y, folds, chunk):
    return [evaluate(X, y, folds, combo) for combo in chunk]


def main():
    print("=" * 78)
    print("Decision tree grid search on the training set")
    print("=" * 78)

    print()
    print("[1/4] loading training set")
    X, y = load_train()

    print()
    print("[2/4] building %d-fold stratified cv (seed=%d)" % (N_SPLITS, CV_SEED))
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                 random_state=CV_SEED).split(X, y))
    print("  folds = %d, events per test fold = %s"
          % (len(folds), [int(y[b].sum()) for _, b in folds]))

    print()
    print("[3/4] expanding the search range")
    combos = [dict(zip(NAMES, values))
              for values in product(*[RANGE[n] for n in NAMES])]
    for n in NAMES:
        v = RANGE[n]
        span = "%s ~ %s" % (v[0], v[-1]) if len(v) > 1 else "%s" % v[0]
        print("  %-18s %2d levels   %s" % (n, len(v), span))
    print("  combinations = %d, fits = %d"
          % (len(combos), len(combos) * (N_SPLITS + 1)))

    print()
    print("[4/4] parallel evaluation")
    start = time.time()
    n_proc = N_JOBS if N_JOBS > 0 else (os.cpu_count() or 4)
    chunks = [combos[i:i + CHUNK] for i in range(0, len(combos), CHUNK)]
    parts = Parallel(n_jobs=n_proc, batch_size=1)(
        delayed(evaluate_chunk)(X, y, folds, chunk) for chunk in chunks)
    rows = [r for part in parts for r in part]
    print("  done in %.1fs" % (time.time() - start))

    df = pd.DataFrame(rows).sort_values(
        ["cv_auc", "n_leaves", "tree_depth"] + NAMES,
        ascending=[False, True, True] + [True] * len(NAMES),
    ).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    df[["rank"] + NAMES + ["cv_auc", "cv_auc_sd", "n_leaves", "tree_depth"]] \
        .to_csv(OUT_RANK, index=False, encoding="utf-8-sig")

    print()
    print("=" * 78)
    print("results")
    print("=" * 78)
    print("combinations = %d" % len(df))
    print("cv auc range = %.6f ~ %.6f" % (df.cv_auc.min(), df.cv_auc.max()))
    print()
    print("top 10 by cv auc")
    print(df.head(10)[["rank", "criterion", "max_depth", "min_samples_split",
                       "min_samples_leaf", "ccp_alpha", "cv_auc", "cv_auc_sd",
                       "n_leaves", "tree_depth"]].to_string(index=False))

    top = df.iloc[0]
    ties = df[np.isclose(df.cv_auc, top.cv_auc, atol=1e-12)]

    print()
    print("best combination")
    print("  criterion         = %s" % top.criterion)
    print("  max_depth         = %d" % int(top.max_depth))
    print("  min_samples_split = %d" % int(top.min_samples_split))
    print("  min_samples_leaf  = %d" % int(top.min_samples_leaf))
    print("  ccp_alpha         = %g" % top.ccp_alpha)
    print("  cv auc            = %.6f +- %.6f" % (top.cv_auc, top.cv_auc_sd))
    print("  tree              = %d leaves, depth %d"
          % (int(top.n_leaves), int(top.tree_depth)))
    print("  combinations at the same cv auc = %d" % len(ties))

    payload = {
        "model": "decision tree",
        "search_space": RANGE,
        "n_combos": int(len(df)),
        "params": {
            "criterion": str(top.criterion),
            "max_depth": int(top.max_depth),
            "min_samples_split": int(top.min_samples_split),
            "min_samples_leaf": int(top.min_samples_leaf),
            "ccp_alpha": float(top.ccp_alpha),
            **{k: v for k, v in FIXED.items()},
        },
        "cv": {"folds": N_SPLITS, "seed": CV_SEED,
               "metric": "mean of per-fold auc",
               "cv_auc": float(top.cv_auc),
               "cv_auc_sd": float(top.cv_auc_sd)},
        "ranking": "cv auc descending, then fewer leaves, shallower tree, smaller parameters",
        "tree": {"n_leaves": int(top.n_leaves), "depth": int(top.tree_depth)},
        "n_ties_at_max": int(len(ties)),
    }
    OUT_BEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print()
    print("saved: %s / %s" % (OUT_RANK.name, OUT_BEST.name))
    print("=" * 78)


if __name__ == "__main__":
    main()
