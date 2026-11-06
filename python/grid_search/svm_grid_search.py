import itertools
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "data")
TRAIN = os.path.join(DATA_DIR, "training set.xlsx")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
TARGET = "outcome"

C_RANGE = (0.1, 10.0)
G_RANGE = (1.0, 10.0)
STEP_DEC = 0.25

CV_FOLDS = 10
CV_SEED = 42

FIXED_SVC = dict(
    kernel="rbf", degree=3, coef0=0.0, class_weight=None,
    shrinking=True, tol=1e-3, cache_size=500, max_iter=-1,
    decision_function_shape="ovr", random_state=42,
)

OUT_CSV = os.path.join(OUT_DIR, "svm_grid_search.csv")
OUT_JSON = os.path.join(OUT_DIR, "svm_grid_search_best.json")


def log_grid(lo, hi, step):
    e_lo, e_hi = np.log10(lo), np.log10(hi)
    n = int(round((e_hi - e_lo) / step)) + 1
    exps = np.linspace(e_lo, e_hi, n)
    return exps, 10.0 ** exps


def build_model(C, gamma):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=C, gamma=gamma, **FIXED_SVC)),
    ])


def cv_folds(X, y):
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    return list(skf.split(X, y))


def main():
    df = pd.read_excel(TRAIN)
    X = df[FEATURES].values.astype(float)
    y = df[TARGET].values.astype(int)

    c_grid = log_grid(*C_RANGE, STEP_DEC)[1]
    g_grid = log_grid(*G_RANGE, STEP_DEC)[1]

    print("=" * 78)
    print("SVM grid search on the training set")
    print("=" * 78)
    print("training set : %s" % os.path.basename(TRAIN))
    print("cases        : %d (positive %d / negative %d)"
          % (len(y), int(y.sum()), int((1 - y).sum())))
    print("features     : %s" % ", ".join(FEATURES))
    print("range        : C in [%g, %g], gamma in [%g, %g]"
          % (C_RANGE[0], C_RANGE[1], G_RANGE[0], G_RANGE[1]))
    print("step         : %g decades" % STEP_DEC)
    print("cv           : StratifiedKFold(%d, shuffle=True, random_state=%d)"
          % (CV_FOLDS, CV_SEED))
    print("combinations : %d x %d = %d"
          % (len(c_grid), len(g_grid), len(c_grid) * len(g_grid)))
    print("=" * 78)

    folds = cv_folds(X, y)

    records = []
    total = len(c_grid) * len(g_grid)
    for i, (C, gamma) in enumerate(itertools.product(c_grid, g_grid), 1):
        fold_auc, oof = [], np.full(len(y), np.nan)
        for tr_idx, va_idx in folds:
            model = build_model(C, gamma).fit(X[tr_idx], y[tr_idx])
            score = model.decision_function(X[va_idx])
            oof[va_idx] = score
            fold_auc.append(float(roc_auc_score(y[va_idx], score)))

        mean_auc = float(np.mean(fold_auc))
        sd_auc = float(np.std(fold_auc, ddof=1))
        records.append({
            "C": float(C),
            "gamma": float(gamma),
            "cv_auc_mean": mean_auc,
            "cv_auc_sd": sd_auc,
            "cv_auc_se": sd_auc / np.sqrt(CV_FOLDS),
            "fold_aucs": " ".join("%.6f" % v for v in fold_auc),
        })
        if i % 10 == 0 or i == total:
            print("  ... %d/%d done" % (i, total), flush=True)

    res = pd.DataFrame([{k: v for k, v in r.items() if k != "fold_aucs"}
                        for r in records])
    res = res.sort_values(["cv_auc_mean", "C"],
                          ascending=[False, True]).reset_index(drop=True)
    res.insert(0, "rank", np.arange(1, len(res) + 1))
    res.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    best_auc = max(r["cv_auc_mean"] for r in records)
    candidates = [r for r in records if abs(r["cv_auc_mean"] - best_auc) < 1e-12]
    best = min(candidates, key=lambda r: r["C"])

    print()
    print("=" * 78)
    print("best combination")
    print("=" * 78)
    print(" C      = %r" % best["C"])
    print(" gamma  = %r" % best["gamma"])
    print(" cv auc = %.6f (sd %.6f)" % (best["cv_auc_mean"], best["cv_auc_sd"]))
    print(" fold aucs: %s" % best["fold_aucs"])
    print()
    print(" top 10")
    print(" %4s%12s%12s%10s%9s" % ("rank", "C", "gamma", "cv_auc", "sd"))
    for _, r in res.head(10).iterrows():
        print(" %4d%12.6g%12.6g%10.4f%9.4f"
              % (int(r["rank"]), r.C, r.gamma, r.cv_auc_mean, r.cv_auc_sd))
    print("=" * 78)

    payload = {
        "model": "SVM (SVC, kernel='rbf')",
        "range": {
            "C": list(C_RANGE),
            "gamma": list(G_RANGE),
            "step_decades": STEP_DEC,
            "n_combos": int(len(res)),
            "C_levels": [round(float(v), 6) for v in c_grid],
            "gamma_levels": [round(float(v), 6) for v in g_grid],
        },
        "cv": {"folds": CV_FOLDS, "seed": CV_SEED, "stratified": True,
               "metric": "roc_auc, mean over folds"},
        "ranking": "highest mean cv auc, ties resolved towards the smaller C",
        "best_params": {"C": float(best["C"]), "gamma": float(best["gamma"])},
        "cv_auc": best["cv_auc_mean"],
        "cv_auc_sd": best["cv_auc_sd"],
        "fold_aucs": [float(v) for v in str(best["fold_aucs"]).split()],
        "fixed_params": {k: (v if not isinstance(v, float) else float(v))
                         for k, v in FIXED_SVC.items()},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print("saved: %s / %s" % (os.path.basename(OUT_CSV), os.path.basename(OUT_JSON)))


if __name__ == "__main__":
    main()
