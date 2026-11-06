import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "data")
TRAIN = os.path.join(DATA_DIR, "training set.xlsx")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
TARGET = "outcome"

CV_FOLDS = 10
CV_SEED = 42

GRID = {
    "n_estimators": [100, 300, 400, 600],
    "max_depth": [8, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

# n_jobs = -1 builds the trees of one forest in parallel. The trees are
# independent and the seed is fixed, so this affects run time only.
FIXED = dict(max_features=4, criterion="gini", bootstrap=True,
             class_weight=None, random_state=42, n_jobs=-1)

OUT_CSV = os.path.join(OUT_DIR, "random_forest_grid_search.csv")
OUT_JSON = os.path.join(OUT_DIR, "random_forest_grid_search_best.json")


def evaluate(X, y, n, d, s, l):
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=CV_SEED)
    fold_auc = np.empty(CV_FOLDS)
    oof = np.zeros(len(y))
    for k, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        model = RandomForestClassifier(
            n_estimators=n, max_depth=d, min_samples_split=s,
            min_samples_leaf=l, **FIXED).fit(X[tr_idx], y[tr_idx])
        prob = model.predict_proba(X[te_idx])[:, 1]
        oof[te_idx] = prob
        fold_auc[k] = roc_auc_score(y[te_idx], prob)
    return fold_auc.mean(), fold_auc.std(ddof=1), fold_auc, roc_auc_score(y, oof)


def main():
    df = pd.read_excel(TRAIN, engine="openpyxl")
    X = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(int)

    combos = [(n, d, s, l)
              for n in GRID["n_estimators"]
              for d in GRID["max_depth"]
              for s in GRID["min_samples_split"]
              for l in GRID["min_samples_leaf"]]

    print("=" * 78)
    print("Random forest grid search on the training set")
    print("=" * 78)
    print("grid:", GRID)
    print("training set %d cases (%d events) | %d-fold CV | seed %d | %d combinations"
          % (len(y), int(y.sum()), CV_FOLDS, CV_SEED, len(combos)))
    print()

    rows, start = [], time.time()
    for i, (n, d, s, l) in enumerate(combos, 1):
        mean, sd, fold_auc, oof_auc = evaluate(X, y, n, d, s, l)
        rows.append({"n_estimators": n, "max_depth": d,
                     "min_samples_split": s, "min_samples_leaf": l,
                     "cv_auc": mean, "cv_sd": sd, "oof_auc": oof_auc,
                     "fold_aucs": fold_auc})
        print("[%2d/%d] %3d/%2d/%2d/%2d  cv_auc=%.8f  sd=%.6f  oof=%.8f  (%.0fs)"
              % (i, len(combos), n, d, s, l, mean, sd, oof_auc, time.time() - start))

    order = sorted(rows, key=lambda r: (-r["cv_auc"], r["cv_sd"],
                                        r["n_estimators"], r["max_depth"],
                                        r["min_samples_split"],
                                        r["min_samples_leaf"]))
    best = order[0]

    print()
    print("=" * 78)
    print("top 8 by cv auc")
    print("=" * 78)
    print("%16s%13s%11s%13s" % ("params", "cv_auc", "cv_sd", "oof_auc"))
    for r in order[:8]:
        tag = "%d/%d/%d/%d" % (r["n_estimators"], r["max_depth"],
                               r["min_samples_split"], r["min_samples_leaf"])
        print("%16s%13.6f%11.6f%13.6f"
              % (tag, r["cv_auc"], r["cv_sd"], r["oof_auc"]))

    print()
    print("=" * 78)
    print("best combination")
    print("=" * 78)
    print("n_estimators      %d" % best["n_estimators"])
    print("max_depth         %d" % best["max_depth"])
    print("min_samples_split %d" % best["min_samples_split"])
    print("min_samples_leaf  %d" % best["min_samples_leaf"])
    print("cv auc            %.8f (sd %.6f)" % (best["cv_auc"], best["cv_sd"]))
    print("fold aucs         %s" % "  ".join("%.6f" % v for v in best["fold_aucs"]))
    if len(order) > 1:
        print("gap to runner-up  %.8f" % (best["cv_auc"] - order[1]["cv_auc"]))

    out = []
    for r in rows:
        rec = {k: r[k] for k in ("n_estimators", "max_depth", "min_samples_split",
                                 "min_samples_leaf")}
        rec["cv_auc"] = round(float(r["cv_auc"]), 10)
        rec["cv_sd"] = round(float(r["cv_sd"]), 10)
        rec["oof_auc"] = round(float(r["oof_auc"]), 10)
        for k, v in enumerate(r["fold_aucs"], 1):
            rec["fold%d" % k] = round(float(v), 8)
        out.append(rec)
    pd.DataFrame(out).sort_values("cv_auc", ascending=False).to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig")

    payload = {
        "model": "random forest",
        "search_grid": GRID,
        "n_combos": len(combos),
        "cv": {"folds": CV_FOLDS, "seed": CV_SEED,
               "splitter": "StratifiedKFold(shuffle=True)",
               "metric": "mean of per-fold auc"},
        "ranking": "highest cv auc, then smaller sd, then smaller parameters",
        "fixed_params": {k: v for k, v in FIXED.items() if k != "n_jobs"},
        "best_params": {
            "n_estimators": best["n_estimators"],
            "max_depth": best["max_depth"],
            "min_samples_split": best["min_samples_split"],
            "min_samples_leaf": best["min_samples_leaf"],
        },
        "cv_auc": round(float(best["cv_auc"]), 8),
        "cv_sd": round(float(best["cv_sd"]), 6),
        "oof_auc": round(float(best["oof_auc"]), 8),
        "fold_aucs": [round(float(v), 6) for v in best["fold_aucs"]],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print("saved: %s" % os.path.basename(OUT_CSV))
    print("saved: %s" % os.path.basename(OUT_JSON))
    print("elapsed %.1fs" % (time.time() - start))


if __name__ == "__main__":
    main()
