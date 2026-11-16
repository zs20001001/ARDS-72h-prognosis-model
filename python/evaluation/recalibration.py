# -*- coding: utf-8 -*-
"""Two-parameter logistic recalibration, evaluated by cross-validation inside the
cohort being recalibrated (Table S14, rows 9 and 13).

This is the analysis in Supplementary Table S11D and in the response to comment 8.
It is **model updating with internal evaluation of that update**, not independent
external validation of the original model: the same patients supply the
coefficients and the performance estimate, which is why the update is fitted and
scored by cross-validation rather than resubstitution.

Protocol
--------
1. logit(p') = a + b * logit(p), fitted without penalisation.
2. Stratified 10-fold cross-validation within the cohort (random_state 42): the
   two coefficients are fitted on nine folds and applied to the held-out fold, so
   every patient receives an out-of-fold recalibrated probability.
3. AUROC (unchanged by a monotone transform, reported for completeness), Brier
   score and calibration slope are reported for the original and for the
   out-of-fold recalibrated predictions.
4. 95% intervals: 2,000 patient-level resamples (seed 7) in which the *entire*
   step-2 procedure is repeated, so the interval carries the uncertainty of the
   recalibration itself and not only the sampling noise of the cohort.

   A bootstrap resample necessarily contains copies of the same patient. The
   fold split is therefore made on the patient identifier, not on the row: every
   copy of a patient is kept in the same fold, so a copy can never enter both the
   recalibration training folds and the evaluation fold of the same replicate.
   This is the point raised in comment 6 of the third review; v1.0.4 split the
   bootstrapped rows with an ordinary StratifiedKFold, which allowed exactly that
   overlap. The point estimate is unchanged: it is computed on the observed rows,
   where every patient appears once.

Usage
-----
    python recalibration.py --data "data/external validation set.xlsx" \
        --outcome outcome --p-col p --group-col patient_id --out results/recalibration.csv

``--group-col`` defaults to ``patient_id`` when the workbook has such a column
(``subject_id`` and ``stay_id`` are also recognised); without one the script
falls back to splitting rows and says so.

The same command with "data/internal validation set.xlsx" recalibrates the frozen model inside
the internal validation cohort.

The input needs one row per patient, the binary outcome and the predicted
probability of the frozen model. To generate that column from the model bundle:

    import joblib, pandas as pd
    bundle = joblib.load("catboost_theta6_model.pkl")
    df["p"] = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from performance import (  # noqa: E402
    auroc,
    brier,
    calibration,
)

N_SPLITS = 10
SEED = 42
N_BOOT = 2000
BOOT_SEED = 7
DEFAULT_GROUP_COLS = ("patient_id", "subject_id", "stay_id")


def logit(p, eps: float = 1e-9):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fit_two_parameter(y, p):
    """Coefficients (a, b) of logit(p') = a + b * logit(p), by Newton-Raphson."""
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(y)), logit(p)])
    beta = np.array([0.0, 1.0])
    for _ in range(200):
        mu = expit(design @ beta)
        w = mu * (1 - mu)
        grad = design.T @ (y - mu)
        hess = design.T @ (design * w[:, None])
        # A tiny ridge keeps the solve defined in degenerate resamples.
        step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def apply_two_parameter(a, b, p):
    return expit(a + b * logit(p))


def cross_validated_recalibration(y, p, n_splits: int = N_SPLITS, seed: int = SEED,
                                  groups=None):
    """Out-of-fold recalibrated probabilities; the cohort is never scored in-sample.

    When ``groups`` is given, the split is made on those values rather than on
    rows, so copies of one patient created by a bootstrap resample always land in
    the same fold. Without ``groups`` the split is the plain stratified 10-fold
    split of the observed cohort, where every patient appears once anyway.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    if groups is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splitter = skf.split(p.reshape(-1, 1), y)
    else:
        groups = np.asarray(groups)
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed)
        except TypeError:  # sklearn < 1.0: no shuffle argument
            sgkf = StratifiedGroupKFold(n_splits=n_splits)
        splitter = sgkf.split(p.reshape(-1, 1), y, groups)
    for tr, te in splitter:
        a, b = fit_two_parameter(y[tr], p[tr])
        out[te] = apply_two_parameter(a, b, p[te])
    return out


def metrics(y, p):
    return dict(
        auroc=auroc(y, p),
        brier=brier(y, p),
        slope=calibration(y, p)["slope"],
        citl=calibration(y, p)["citl"],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--outcome", default="outcome")
    ap.add_argument("--p-col", default="p")
    ap.add_argument("--group-col", default=None,
                    help="patient identifier used for the fold split and the bootstrap; "
                         "defaults to patient_id / subject_id / stay_id when present")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "recalibration.csv")
    args = ap.parse_args()

    df = pd.read_excel(args.data) if args.data.suffix.lower() in {".xlsx", ".xls"} \
        else pd.read_csv(args.data)
    df.columns = [str(c).strip() for c in df.columns]
    y = pd.to_numeric(df[args.outcome], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(df[args.p_col], errors="coerce").to_numpy(float)

    group_col = args.group_col
    if group_col is None:
        group_col = next((c for c in DEFAULT_GROUP_COLS if c in df.columns), None)
    if group_col and group_col in df.columns:
        groups = df[group_col].astype(str).to_numpy()
        repeated = len(groups) - len(np.unique(groups))
        print(f"group column: {group_col} "
              f"({len(np.unique(groups))} patients, {repeated} repeated rows)")
    else:
        groups = None
        if group_col:
            print(f"[warning] column {group_col!r} not found; splitting rows instead")
        else:
            print("[warning] no patient identifier column; the bootstrap split is made "
                  "on rows and copies of a patient may cross folds")

    a_full, b_full = fit_two_parameter(y, p)
    # Point estimate on the observed rows: each patient appears once, so the plain
    # stratified split is used and the estimate is the one reported previously.
    p_cv = cross_validated_recalibration(y, p)
    before = metrics(y, p)
    after = metrics(y, p_cv)

    print(f"cohort n = {len(y)}, events = {int(y.sum())}, prevalence = {y.mean():.3f}")
    print(f"recalibration on the full cohort: logit(p') = {a_full:+.3f} "
          f"{b_full:+.3f} * logit(p)")
    print("%-22s %8s %10s" % ("metric", "original", "recalibrated"))
    for key in ("auroc", "brier", "slope", "citl"):
        print("%-22s %8.4f %10.4f" % (key, before[key], after[key]))

    # Bootstrap: the 10-fold recalibration is repeated inside every resample.
    rng = np.random.default_rng(BOOT_SEED)
    draws = {k: [] for k in ("auroc", "brier", "slope", "citl")}
    for _ in range(args.n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        try:
            gb = groups[idx] if groups is not None else None
            p_cv_b = cross_validated_recalibration(yb, pb, groups=gb)
            m = metrics(yb, p_cv_b)
        except (ValueError, np.linalg.LinAlgError):
            continue
        for k in draws:
            draws[k].append(m[k])

    rows = []
    print("\nrecalibrated, with 95% percentile bootstrap intervals "
          f"({len(draws['auroc'])} resamples):")
    for k in ("auroc", "brier", "slope", "citl"):
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        rows.append(dict(metric=k, original=before[k], recalibrated=after[k],
                         ci_low=float(lo), ci_high=float(hi),
                         n_boot=len(draws[k])))
        print("  %-8s %8.4f (%8.4f to %8.4f)" % (k, after[k], lo, hi))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame({"outcome": y, "p_original": p, "p_recalibrated_cv": p_cv}).to_csv(
        args.out.with_name(args.out.stem + "_predictions.csv"), index=False)
    print(f"\nfiles: {args.out.name}, {args.out.stem}_predictions.csv")


if __name__ == "__main__":
    main()
