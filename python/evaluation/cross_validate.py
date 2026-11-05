# -*- coding: utf-8 -*-
"""Stratified 10-fold cross-validation of the final predictor set (Table S14, row 5).

Protocol: stratified 10-fold cross-validation in the 284 training patients,
``shuffle = TRUE``, ``random_state = 42``, with the fold assignment held identical
across the eight algorithms so that the models are compared on the same
partitions. Nothing is selected or tuned inside the loop: the predictor set is the
four final variables and the hyperparameters are those of ``models.py``. The
estimate is therefore the cross-validated discrimination of a fixed
specification, which is the quantity the algorithm ranking uses, not an estimate
of the optimism of the whole development procedure.

Usage
-----
    python cross_validate.py --data ../../data/training_set.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "resampling"))
sys.path.insert(0, str(HERE.parent / "evaluation"))

from models import MODEL_ZOO, fit_model, positive_class_probability  # noqa: E402
from performance import auroc  # noqa: E402

N_SPLITS = 10
SEED = 42
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=HERE.parents[1] / "data" / "training_set.xlsx")
    ap.add_argument("--outcome", default="outcome")
    ap.add_argument("--features", nargs="*", default=FEATURES)
    ap.add_argument("--models", nargs="*", default=list(MODEL_ZOO))
    ap.add_argument("--out", type=Path, default=HERE / "results" / "cross_validation.csv")
    args = ap.parse_args()

    df = pd.read_excel(args.data) if args.data.suffix.lower() in {".xlsx", ".xls"} \
        else pd.read_csv(args.data)
    df.columns = [str(c).strip() for c in df.columns]
    X = df[args.features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[args.outcome], errors="coerce").astype(int).to_numpy()

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))          # computed once, shared by all models

    rows = []
    apparent_map = {}
    for name in args.models:
        apparent_map[name] = auroc(y, positive_class_probability(fit_model(name, X, y), X))
        for k, (tr, te) in enumerate(folds, start=1):
            model = fit_model(name, X[tr], y[tr])
            rows.append(dict(model=name, fold=k,
                             auroc=auroc(y[te], positive_class_probability(model, X[te]))))
        aucs = np.array([r["auroc"] for r in rows if r["model"] == name])
        print("%-22s apparent %.4f   CV %.4f (SD %.4f, range %.3f-%.3f)"
              % (name, apparent_map[name], aucs.mean(), aucs.std(ddof=1),
                 aucs.min(), aucs.max()))

    per_fold = pd.DataFrame(rows)
    summary = (per_fold.groupby("model")["auroc"]
               .agg(cv_auroc="mean", sd="std", fold_min="min", fold_max="max", n="count")
               .reset_index())
    summary["apparent_auroc"] = summary["model"].map(apparent_map)
    summary = summary.sort_values("cv_auroc", ascending=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(args.out, index=False)
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    print(f"\nfiles: {args.out.name}, {args.out.stem}_summary.csv")


if __name__ == "__main__":
    main()
