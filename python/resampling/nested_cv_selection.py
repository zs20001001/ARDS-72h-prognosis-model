# -*- coding: utf-8 -*-
"""Nested cross-validation that repeats the selection inside every training fold.

Protocol (Table S14, row 7): stratified outer 10-fold cross-validation repeated
five times with random_state 42 + r, r = 0...4, giving 50 training folds of 256
patients and test folds of 28. Feature selection runs on the 256 patients of the
training fold only, never on the test fold and never on either validation cohort;
the classifier is then fitted on the selected set and scored once on the held-out
fold.

Two arms are run, and they are the two rows of Supplementary Table S4f:

``reselect``
    Boruta and LASSO are re-run inside every training fold and the classifier is
    fitted on the resulting set (median 7 predictors per fold, range 1-12).

``fixed``
    The four predictors of the final model (``d3_a_ado2``, ``peep``, ``d3_sofa``,
    ``d3_lac``) are used unchanged in every fold, which is the paired arm that
    Table S4f compares against.

Patient-level out-of-fold predictions are kept, not only the fold-level AUROCs:
every held-out patient contributes exactly one probability per repeat, and the
predictions of the two arms are aligned on the same patient identifier. That file
is what ``reproduce_table_s4f.py`` needs in order to compute the pooled AUROC,
the AUROC of the predictions averaged over the five repeats, and the paired
comparison reported in Table S4f. Averaging fold-level AUROCs, which is what the
v1.0.4 script did, cannot produce either of the latter two quantities.

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
        --models CatBoost --arm both --out results/nested_cv.csv

Outputs, next to ``--out``:

    <stem>.csv        one row per arm x model x repeat x fold (AUROC per fold)
    <stem>_oof.csv    one row per arm x model x repeat x held-out patient
    <stem>_summary.csv  fold-level AUROC averaged within each arm

If the workbook has no patient identifier column, the row index is used; the
data have one row per patient, so the row index identifies the patient within
this analysis.
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

FIXED_PREDICTORS = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
ID_CANDIDATES = ("patient_id", "subject_id", "stay_id", "id", "pid")


def patient_ids(df: pd.DataFrame, n: int) -> np.ndarray:
    """One identifier per row, stable across repeats."""
    for column in ID_CANDIDATES:
        if column in df.columns:
            return df[column].astype(str).to_numpy()
    return np.array([f"row{i}" for i in range(n)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=HERE.parents[1] / "data" / "training set.xlsx")
    ap.add_argument("--outcome", default="outcome")
    ap.add_argument("--candidates", type=Path,
                    default=HERE.parents[1] / "data" / "candidate_predictors.txt")
    ap.add_argument("--models", nargs="*", default=list(MODEL_ZOO))
    ap.add_argument("--arm", choices=("both", "reselect", "fixed"), default="both",
                    help="both = the two rows of Table S4f in one run")
    ap.add_argument("--fixed-predictors", nargs="*", default=FIXED_PREDICTORS)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--boruta-trees", type=int, default=500,
                    help="trees per Boruta forest; lower it only for a smoke test")
    ap.add_argument("--jobs", type=int, default=1,
                    help="cores used to build the trees of one forest; the result "
                         "does not depend on it")
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
    pids = patient_ids(df, len(y))

    arms = ("reselect", "fixed") if args.arm == "both" else (args.arm,)
    if "fixed" in arms:
        missing = [f for f in args.fixed_predictors if f not in names]
        if missing:
            raise SystemExit(f"fixed predictors not in the candidate list: {missing}")
        fixed_cols = [names.index(f) for f in args.fixed_predictors]

    from sklearn.model_selection import StratifiedKFold

    fold_rows, oof_rows = [], []
    for r in range(args.repeats):
        seed = BASE_SEED + r
        skf = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(X, y), start=1):
            selections = {}
            if "reselect" in arms:
                selections["reselect"] = select_features(
                    X[tr], y[tr], seed=seed, n_trees=args.boruta_trees,
                    n_jobs=args.jobs)
            if "fixed" in arms:
                selections["fixed"] = (fixed_cols, "fixed", len(fixed_cols), len(fixed_cols))

            for arm in arms:
                cols, branch, n_boruta, n_lasso = selections[arm]
                for name in args.models:
                    model = fit_model(name, X[tr][:, cols], y[tr])
                    p = positive_class_probability(model, X[te][:, cols])
                    fold_rows.append(dict(
                        arm=arm, repeat=r + 1, fold=k, model=name,
                        n_train=len(tr), n_test=len(te),
                        n_predictors=len(cols), branch=branch,
                        n_boruta=n_boruta, n_lasso=n_lasso,
                        auroc=auroc(y[te], p),
                    ))
                    oof_rows.extend(dict(
                        arm=arm, model=name, repeat=r + 1, fold=k,
                        patient_id=pid, y=int(yy), p=float(pp),
                    ) for pid, yy, pp in zip(pids[te], y[te], p))
            detail = ", ".join(f"{a}:{len(selections[a][0])}v" for a in arms)
            print(f"repeat {r + 1}/{args.repeats} fold {k}/{args.splits}: {detail}",
                  flush=True)

    per_fold = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(oof_rows)
    summary = (
        per_fold.groupby(["arm", "model"])["auroc"]
        .agg(cv_auroc="mean", sd="std", n="count")
        .reset_index()
        .sort_values(["arm", "cv_auroc"], ascending=[True, False])
    )
    branches = Counter(per_fold.loc[per_fold.arm == "reselect"]
                       .drop_duplicates(["repeat", "fold"])["branch"])
    sizes = per_fold.loc[per_fold.arm == "reselect"].drop_duplicates(
        ["repeat", "fold"])["n_predictors"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(args.out, index=False)
    oof.to_csv(args.out.with_name(args.out.stem + "_oof.csv"), index=False)
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)

    print("\narm        model                mean fold AUROC   SD")
    for _, row in summary.iterrows():
        print("%-10s %-22s %14.4f  %5.3f"
              % (row["arm"], row["model"], row["cv_auroc"], row["sd"]))
    if sizes.size:
        print("\nfallback branches (reselect arm):", dict(branches))
        print("variables per fold: median %d, range %d-%d"
              % (int(np.median(sizes)), int(sizes.min()), int(sizes.max())))
    print(f"\nfiles: {args.out.name}, {args.out.stem}_oof.csv, {args.out.stem}_summary.csv")
    print("next: python reproduce_table_s4f.py --oof "
          f"{args.out.with_name(args.out.stem + '_oof.csv').name} --model CatBoost")


if __name__ == "__main__":
    main()
