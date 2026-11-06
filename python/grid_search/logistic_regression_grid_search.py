import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "data")
TRAIN = os.path.join(DATA_DIR, "training set.xlsx")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
TARGET = "outcome"

C_LOW = 0.1
C_HIGH = 10.0
N_GRID = 80
N_FOLDS = 10
SEED = 42

OUT_CSV = os.path.join(OUT_DIR, "logistic_regression_grid_search.csv")
OUT_JSON = os.path.join(OUT_DIR, "logistic_regression_grid_search_best.json")


def load_training_data(path=TRAIN, features=FEATURES, target=TARGET):
    df = pd.read_excel(path)
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=features + [target]).reset_index(drop=True)
    return df[features].to_numpy(float), df[target].to_numpy(int)


def build_model(C):
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, solver="lbfgs", max_iter=5000,
                                  random_state=SEED)),
    ])


def cv_fold_aucs(X, y, C, cv):
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        model = build_model(C).fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]
        scores.append(float(roc_auc_score(y[test_idx], prob)))
    return np.asarray(scores)


def grid_search(X, y, lo=C_LOW, hi=C_HIGH, n=N_GRID, n_folds=N_FOLDS, seed=SEED):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    grid = np.geomspace(lo, hi, n)
    rows = []
    for C in grid:
        folds = cv_fold_aucs(X, y, float(C), cv)
        rows.append({"C": float(C), "cv_auc": float(folds.mean()),
                     "cv_sd": float(folds.std(ddof=1)), "fold_aucs": folds})
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "fold_aucs"}
                          for r in rows])
    best_idx = max(range(len(rows)),
                   key=lambda i: (rows[i]["cv_auc"], -rows[i]["C"]))
    return rows, best_idx, table


def main():
    X, y = load_training_data()
    print("training set: %d cases, %d events" % (len(y), int(y.sum())))

    rows, best_idx, table = grid_search(X, y)
    best = rows[best_idx]

    out = []
    for r in rows:
        rec = {"C": r["C"], "cv_auc": r["cv_auc"], "cv_sd": r["cv_sd"]}
        for k, v in enumerate(r["fold_aucs"], 1):
            rec["fold%d" % k] = round(float(v), 8)
        out.append(rec)
    pd.DataFrame(out).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("range searched   : C in [%g, %g] (%d log-spaced points)"
          % (C_LOW, C_HIGH, N_GRID))
    print("cv protocol      : StratifiedKFold(%d, shuffle=True, random_state=%d)"
          % (N_FOLDS, SEED))
    print("best combination : C = %.6f" % best["C"])
    print("10-fold cv auc   : %.6f (sd %.6f)" % (best["cv_auc"], best["cv_sd"]))
    print("fold aucs        : %s"
          % "  ".join("%.6f" % v for v in best["fold_aucs"]))
    print("cv auc over grid : %.6f - %.6f"
          % (table.cv_auc.min(), table.cv_auc.max()))
    print("saved: %s" % os.path.basename(OUT_CSV))

    payload = {
        "model": "logistic regression (standardisation + l2)",
        "range": {"C_low": C_LOW, "C_high": C_HIGH, "n_points": N_GRID},
        "grid": [round(float(v), 8) for v in np.geomspace(C_LOW, C_HIGH, N_GRID)],
        "cv": {"folds": N_FOLDS, "seed": SEED,
               "metric": "mean of per-fold auc"},
        "ranking": "highest cv auc, ties resolved towards the smaller C",
        "best_params": {"C": float(best["C"])},
        "cv_auc": float(best["cv_auc"]),
        "cv_sd": float(best["cv_sd"]),
        "fold_aucs": [round(float(v), 6) for v in best["fold_aucs"]],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("saved: %s" % os.path.basename(OUT_JSON))


if __name__ == "__main__":
    main()
