# -*- coding: utf-8 -*-
"""Nested cross-validation that repeats the selection inside every training fold.

Protocol (Table S14, row 7): stratified outer 10-fold cross-validation repeated
five times with random_state 42 + r, r = 0...4, giving 50 training folds of 256
patients and test folds of 28. Feature selection runs on the 256 patients of the
training fold only, never on the test fold and never on either validation cohort;
the classifier is then fitted on the selected set and scored once on the held-out
fold.

The fallback rule of Table S14, row 8 applies unchanged: an empty intersection is
replaced by the Boruta-confirmed set, and an empty Boruta result by the full
candidate set. The branch taken in each fold is counted and reported, which is
how the manuscript states that only one branch ever occurred.

This answers a different question from the Harrell bootstrap: it is a
*conditional* estimate of whether selecting the variables once, before
cross-validation, inflates discrimination, with imputation and the hyperparameter
grid held at their full-cohort values.

Usage
-----
    python nested_cv_selection.py --data "../../data/training set.xlsx" \
        --models CatBoost --out results/nested_cv.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "evaluation"))

from models import MODEL_ZOO, fit_model, positive_class_probability  # noqa: E402
from performance import auroc  # noqa: E402
from feature_selection import select_features  # noqa: E402

N_SPLITS = 10
N_REPEATS = 5
BASE_SEED = 42


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=HERE.parents[1] / "data" / "training set.xlsx")
    ap.add_argument("--outcome", default="outcome")
    ap.add_argument("--candidates", type=Path,
                    default=HERE.parents[1] / "data" / "candidate_predictors.txt")
    ap.add_argument("--models", nargs="*", default=list(MODEL_ZOO))
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--boruta-trees", type=int, default=500,
                    help="trees per Boruta forest; lower it only for a smoke test")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "nested_cv.csv")
    args = ap.parse_args()

    df = pd.read_excel(args.data) if args.data.suffix.lower() in {".xlsx", ".xls"} \
        else pd.read_csv(args.data)
    df.columns = [str(c).strip() for c in df.columns]
    if args.candidates and args.candidates.exists():
        names = [ln.strip() for ln in args.candidates.read_text().splitlines() if ln.strip()]
    else:
        names = [c for c in df.columns if c != args.outcome]
    X = df[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[args.outcome], errors="coerce").astype(int).to_numpy()

    from sklearn.model_selection import StratifiedKFold

    rows = []
    for r in range(args.repeats):
        seed = BASE_SEED + r
        skf = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(X, y), start=1):
            cols, branch, n_boruta, n_lasso = select_features(
                X[tr], y[tr], seed=seed, n_trees=args.boruta_trees)
            for name in args.models:
                model = fit_model(name, X[tr][:, cols], y[tr])
                p = positive_class_probability(model, X[te][:, cols])
                rows.append(dict(
                    repeat=r, fold=k, model=name,
                    n_train=len(tr), n_test=len(te),
                    n_predictors=len(cols), branch=branch,
                    n_boruta=n_boruta, n_lasso=n_lasso,
                    auroc=auroc(y[te], p),
                ))
            print(f"repeat {r + 1}/{args.repeats} fold {k}/{args.splits}: "
                  f"{len(cols)} variables ({branch})", flush=True)

    per_fold = pd.DataFrame(rows)
    summary = (
        per_fold.groupby("model")["auroc"]
        .agg(cv_auroc="mean", sd="std", n="count")
        .reset_index()
        .sort_values("cv_auroc", ascending=False)
    )
    branches = Counter(per_fold.drop_duplicates(["repeat", "fold"])["branch"])
    sizes = per_fold.drop_duplicates(["repeat", "fold"])["n_predictors"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(args.out, index=False)
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)

    print("\nmodel                  mean CV AUROC   SD")
    for _, row in summary.iterrows():
        print("%-22s %12.4f  %5.3f" % (row["model"], row["cv_auroc"], row["sd"]))
    print("\nfallback branches:", dict(branches))
    print("variables per fold: median %d, range %d-%d"
          % (int(np.median(sizes)), int(sizes.min()), int(sizes.max())))
    print(f"\nfiles: {args.out.name}, {args.out.stem}_summary.csv")


if __name__ == "__main__":
    main()
