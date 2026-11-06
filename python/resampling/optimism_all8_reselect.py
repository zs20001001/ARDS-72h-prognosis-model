# -*- coding: utf-8 -*-
"""Harrell bootstrap optimism with re-selection inside every resample.

This is the analysis behind Table S8 (section D) and Table S4b: 200 patient-level
bootstrap resamples of the 284 training patients; inside each resample the two
selectors are re-run, their intersection is used as the predictor set, and the
classifier is refitted from scratch. Optimism is the mean difference between the
AUROC of a bootstrap model on its own resample and its AUROC on the original
training cohort, and the optimism-corrected AUROC is the apparent AUROC of the
original fit minus that mean:

    optimism_b   = AUROC(D_b, D_b) - AUROC(D_b, D)
    optimism     = mean_b optimism_b
    corrected    = AUROC(D, D) - optimism

Three arms are available and are meant to be compared:

``--arm reselect``  (default) the whole development procedure: selection and
                    fitting are both repeated inside the resample. This is the
                    estimate reported as the optimism-corrected AUROC.
``--arm fixed``     the predictor set is held at the four final variables and
                    only the classifier is refitted, which isolates the bias of
                    fitting from the bias of selection.
``--arm oob``       the same selection-and-fitting step as ``reselect``, but the
                    model is scored on the patients left out of its own resample
                    instead of on the original cohort. Those patients took part in
                    neither the selection nor the fitting, so the score is an
                    out-of-bag rather than an apparent one, and the optimism
                    carries the sampling noise of a test set of roughly 105
                    patients (Table S8, section D, second variant).

All three arms use the same resamples, so the per-replicate differences are paired.

Imputation and hyperparameters are *not* repeated: both are held at their
full-cohort values, and the estimate is conditional on them (Table S14, row 6).

Usage
-----
    python optimism_all8_reselect.py --data ../../data/training_set.xlsx \
        --n-boot 200 --arm reselect --out results/optimism_reselect.csv
    python optimism_all8_reselect.py --data ../../data/training_set.xlsx \
        --n-boot 200 --arm oob --models CatBoost --out results/optimism_oob.csv

Runtime: the selection step dominates. With BorutaPy at 500 trees, expect roughly
1-3 min per resample on one core; use ``--n-jobs`` and ``--n-boot`` accordingly.
A short smoke run with ``--n-boot 5 --boruta-trees 50`` finishes in about a
minute and is enough to verify the wiring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# performance.py lives one level up, in python/evaluation.
sys.path.insert(0, str(HERE.parent / "evaluation"))

from models import MODEL_ZOO, fit_model, positive_class_probability  # noqa: E402
from performance import auroc  # noqa: E402
from feature_selection import select_features  # noqa: E402

FIXED_FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]


def load_cohort(path: Path, outcome: str, candidates_path: Path | None = None):
    """Load the training cohort and the candidate predictor list."""
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if outcome not in df.columns:
        raise SystemExit(f"outcome column {outcome!r} not found in {path}")

    if candidates_path and candidates_path.exists():
        names = [ln.strip() for ln in candidates_path.read_text().splitlines() if ln.strip()]
        missing = [n for n in names if n not in df.columns]
        if missing:
            raise SystemExit(f"candidate predictors missing from the data: {missing[:5]} ...")
    else:
        names = [c for c in df.columns if c != outcome]

    X = df[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[outcome], errors="coerce").astype(int).to_numpy()
    return X, y, names


def one_resample(b: int, X, y, names, arm: str, seed: int, models, fixed_idx,
                 boruta_trees: int = 500, select_seed: int = 500):
    """Select (reselect / oob arms), refit every model, return per-model optimism."""
    rng = np.random.default_rng(seed + b)
    idx = rng.integers(0, len(y), len(y))
    Xb, yb = X[idx], y[idx]

    if arm in ("reselect", "oob"):
        cols, branch, n_boruta, n_lasso = select_features(Xb, yb, seed=select_seed,
                                                          n_trees=boruta_trees)
    else:
        cols, branch, n_boruta, n_lasso = fixed_idx, "fixed", np.nan, np.nan

    # Patients that are in none of the resampled rows: the out-of-bag set.
    if arm == "oob":
        test_idx = np.setdiff1d(np.arange(len(y)), np.unique(idx))
        if len(np.unique(y[test_idx])) < 2:
            return []
    else:
        test_idx = np.arange(len(y))

    rows = []
    for name in models:
        model = fit_model(name, Xb[:, cols], yb)
        auc_boot = auroc(yb, positive_class_probability(model, Xb[:, cols]))
        auc_test = auroc(y[test_idx], positive_class_probability(model, X[test_idx][:, cols]))
        rows.append(
            dict(
                replicate=b,
                model=name,
                n_predictors=len(cols),
                branch=branch,
                n_boruta=n_boruta,
                n_lasso=n_lasso,
                n_test=len(test_idx),
                auc_resample=auc_boot,
                auc_test=auc_test,
                optimism=auc_boot - auc_test,
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path,
                    default=HERE.parents[1] / "data" / "training_set.xlsx")
    ap.add_argument("--outcome", default="outcome")
    ap.add_argument("--candidates", type=Path,
                    default=HERE.parents[1] / "data" / "candidate_predictors.txt")
    ap.add_argument("--arm", choices=["reselect", "fixed", "oob"], default="reselect")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=499,
                    help="resample b is drawn under the seed seed + b, so the "
                         "default 499 gives seeds 500-699 and reproduces the "
                         "published 200 resamples (Table S14, row 6)")
    ap.add_argument("--select-seed", type=int, default=500,
                    help="seed of the two selectors inside every resample; fixed "
                         "at 500 in the published run (Table S14, row 6)")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--boruta-trees", type=int, default=500,
                    help="trees per Boruta forest; 500 is the published value, "
                         "lower it only for a smoke test")
    ap.add_argument("--models", nargs="*", default=list(MODEL_ZOO),
                    help="subset of the eight classifiers")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "optimism_all8.csv")
    args = ap.parse_args()

    X, y, names = load_cohort(args.data, args.outcome, args.candidates)
    fixed_idx = [names.index(f) for f in FIXED_FEATURES if f in names]
    print(f"cohort n = {len(y)}  events = {int(y.sum())}  "
          f"candidates = {X.shape[1]}  arm = {args.arm}  resamples = {args.n_boot}")

    # Apparent (resubstitution) AUROC of the original fit, one per model.
    apparent = {}
    if args.arm in ("reselect", "oob"):
        cols, branch, n_boruta, n_lasso = select_features(
            X, y, seed=args.select_seed, n_trees=args.boruta_trees)
        print(f"full-cohort selection: {len(cols)} variables ({branch}); "
              f"Boruta {n_boruta}, LASSO {n_lasso}")
        if len(cols) == len(names):
            print("  note: the full-cohort intersection was empty, so all "
                  "candidates are used")
    else:
        cols = fixed_idx
        print(f"fixed arm: fitting on {[names[i] for i in cols]}")

    for name in args.models:
        model = fit_model(name, X[:, cols], y)
        apparent[name] = auroc(y, positive_class_probability(model, X[:, cols]))

    # Bootstrap loop, parallel over resamples so that each worker writes nothing.
    if args.n_jobs > 1:
        from joblib import Parallel, delayed

        chunks = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(one_resample)(b, X, y, names, args.arm, args.seed, args.models,
                                  fixed_idx, args.boruta_trees, args.select_seed)
            for b in range(1, args.n_boot + 1)
        )
        rows = [r for chunk in chunks for r in chunk]
    else:
        rows = []
        for b in range(1, args.n_boot + 1):
            rows += one_resample(b, X, y, names, args.arm, args.seed, args.models,
                                 fixed_idx, args.boruta_trees, args.select_seed)
            if b % 10 == 0:
                print(f"  {b}/{args.n_boot} resamples done", flush=True)

    per_rep = pd.DataFrame(rows)
    summary = (
        per_rep.groupby("model")["optimism"]
        .agg(optimism="mean", sd="std", n="count")
        .reset_index()
    )
    summary["apparent_auroc"] = summary["model"].map(apparent)
    summary["corrected_auroc"] = summary["apparent_auroc"] - summary["optimism"]
    # Size of the scoring set: 284 in the reselect and fixed arms, the out-of-bag
    # count (median about 105) in the oob arm.
    summary["median_n_test"] = summary["model"].map(
        per_rep.groupby("model")["n_test"].median())
    summary["ci_low"] = summary.apply(
        lambda r: np.percentile(
            per_rep.loc[per_rep.model == r["model"], "optimism"], 2.5), axis=1)
    summary["ci_high"] = summary.apply(
        lambda r: np.percentile(
            per_rep.loc[per_rep.model == r["model"], "optimism"], 97.5), axis=1)
    summary = summary.sort_values("corrected_auroc", ascending=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_rep.to_csv(args.out.with_name(args.out.stem + "_per_replicate.csv"), index=False)
    summary.to_csv(args.out, index=False)

    runs = json.dumps(
        dict(data=str(args.data), arm=args.arm, n_boot=args.n_boot, seed=args.seed,
             select_seed=args.select_seed,
             models=list(args.models), n=len(y), events=int(y.sum()),
             scoring="out-of-bag patients of each resample" if args.arm == "oob"
                     else "original training cohort"),
        ensure_ascii=False,
    )
    args.out.with_name(args.out.stem + "_run.json").write_text(runs, encoding="utf-8")

    print("\nmodel                 apparent  optimism (95% CI)      corrected")
    for _, r in summary.iterrows():
        print("%-21s %8.4f  %+.4f (%+.4f to %+.4f)  %8.4f"
              % (r["model"], r["apparent_auroc"], r["optimism"],
                 r["ci_low"], r["ci_high"], r["corrected_auroc"]))
    print(f"\nfiles: {args.out.name}, "
          f"{args.out.stem}_per_replicate.csv, {args.out.stem}_run.json")


if __name__ == "__main__":
    main()
