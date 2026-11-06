# -*- coding: utf-8 -*-
"""Extended XGBoost grid, used to answer comment 6 of the second-round report.

The published grid restricted XGBoost to one or two boosting rounds. This script
re-runs the search over a wider axis, on the training cohort only and by the same
stratified 10-fold cross-validation used everywhere else, so that no validation
cohort is touched:

    learning rate  0.05, 0.10, 0.30, 0.50, 0.60
    rounds         1, 2, 5, 10, 20, 50, 100
    max depth      3, 4

That is 70 configurations; the remaining parameters (subsample 0.60,
colsample_bytree 0.90, min_child_weight 1, max_bin 64) are held at the values
selected in the original search. The full grid is written to CSV and is the file supplied
as Supplementary File S5b.

Result (training cohort, n = 284): the best cell is learning rate 0.30, 5 rounds,
depth 4, with a mean CV AUROC of 0.9189 against 0.9034 for the selected CatBoost
configuration. The difference is 0.016, below the standard error of the fold mean
(0.020), and the top five cells lie within 0.0014 of each other; the manuscript
therefore reports the two algorithms as indistinguishable rather than replacing
the model.

Usage
-----
    python xgboost_extended_grid.py --data "../../data/training set.xlsx" \
        --out xgboost_extended_grid.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
OUTCOME = "outcome"
SEED = 42
N_SPLITS = 10

LEARNING_RATES = [0.05, 0.10, 0.30, 0.50, 0.60]
ROUNDS = [1, 2, 5, 10, 20, 50, 100]
DEPTHS = [3, 4]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=HERE.parents[1] / "data" / "training set.xlsx")
    ap.add_argument("--out", type=Path, default=HERE / "xgboost_extended_grid.csv")
    args = ap.parse_args()

    df = pd.read_excel(args.data) if args.data.suffix.lower() in {".xlsx", ".xls"} \
        else pd.read_csv(args.data)
    df.columns = [str(c).strip() for c in df.columns]
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[OUTCOME], errors="coerce").astype(int).to_numpy()

    from sklearn.metrics import roc_auc_score

    rows = []
    for lr in LEARNING_RATES:
        for rounds in ROUNDS:
            for depth in DEPTHS:
                params = dict(
                    objective="binary:logistic", booster="gbtree", tree_method="hist",
                    max_bin=64, learning_rate=lr, max_depth=depth, min_child_weight=1,
                    subsample=0.60, colsample_bytree=0.90, seed=SEED, nthread=1,
                    verbosity=0,
                )
                skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
                cv_aucs, train_aucs = [], []
                for tr, va in skf.split(X, y):
                    dtrain = xgb.DMatrix(X[tr], label=y[tr], nthread=1)
                    dvalid = xgb.DMatrix(X[va], nthread=1)
                    booster = xgb.train(params, dtrain, num_boost_round=rounds)
                    cv_aucs.append(roc_auc_score(y[va], booster.predict(dvalid)))
                    train_aucs.append(roc_auc_score(y[tr], booster.predict(dtrain)))
                rows.append(dict(
                    learning_rate=lr, boosting_rounds=rounds, max_depth=depth,
                    mean_cv_auroc=float(np.mean(cv_aucs)),
                    sd_cv_auroc=float(np.std(cv_aucs, ddof=1)),
                    mean_train_auroc=float(np.mean(train_aucs)),
                ))
                print("%-5s %-3d d%d  CV %.4f (SD %.4f)  train %.4f"
                      % (lr, rounds, depth, rows[-1]["mean_cv_auroc"],
                         rows[-1]["sd_cv_auroc"], rows[-1]["mean_train_auroc"]))

    result = pd.DataFrame(rows).sort_values("mean_cv_auroc", ascending=False)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    best = result.iloc[0]
    print("\nbest configuration: learning rate %.2f, %d rounds, depth %d, "
          "mean CV AUROC %.4f (SD %.4f)"
          % (best.learning_rate, best.boosting_rounds, best.max_depth,
             best.mean_cv_auroc, best.sd_cv_auroc))
    print(f"{len(result)} configurations written to {args.out.name}")


if __name__ == "__main__":
    main()
