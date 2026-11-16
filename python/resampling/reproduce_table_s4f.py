# -*- coding: utf-8 -*-
"""Regenerate Supplementary Table S4f from patient-level out-of-fold predictions.

Table S4f compares two arms of the nested cross-validation, the fixed
four-predictor specification and the re-selection performed inside every
training fold, on three quantities:

1. **Pooled AUROC** -- all held-out patients of one repeat are pooled into a
   single AUROC; the mean and SD over the five repeats are reported.
2. **AUROC of predictions averaged over the repeats** -- every patient has five
   held-out predictions (one per repeat); they are averaged and a single AUROC is
   computed over the whole cohort, with a 95% percentile bootstrap interval
   obtained by resampling *patients*.
3. **Paired difference** -- the two arms are matched on the patient identifier
   and compared with the paired DeLong test on the averaged predictions. The
   mean of the five within-repeat differences is reported alongside.

Averaging the fifty fold-level AUROCs cannot produce quantities 2 and 3, which is
why ``nested_cv_selection.py`` writes the patient-level file this script reads.

Input
-----
One or more ``*_oof.csv`` files written by ``nested_cv_selection.py``, either one
file containing both arms or two files with one arm each. Required columns:
``arm``, ``model``, ``repeat``, ``fold``, ``patient_id``, ``y``, ``p``.

Usage
-----
    python reproduce_table_s4f.py --oof results/nested_cv_oof.csv \
        --folds results/nested_cv.csv --model CatBoost --out results/table_s4f.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "evaluation"))

from performance import auroc, delong_paired  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 7
ARM_LABELS = {"fixed": "Fixed four predictors", "reselect": "Re-selection inside the fold"}


def load_oof(paths) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    oof = pd.concat(frames, ignore_index=True)
    required = {"arm", "model", "repeat", "fold", "patient_id", "y", "p"}
    missing = required - set(oof.columns)
    if missing:
        raise SystemExit(f"missing columns in the OOF file(s): {sorted(missing)}")
    return oof


def check_complete(oof: pd.DataFrame) -> pd.DataFrame:
    """Every patient must be held out exactly once per repeat."""
    counts = (oof.groupby(["arm", "model", "repeat", "patient_id"]).size()
              .rename("n").reset_index())
    bad = counts[counts.n != 1]
    if not bad.empty:
        raise SystemExit(
            "some patients are held out more than once within a repeat:\n"
            + bad.head(10).to_string(index=False))
    per_repeat = oof.groupby(["arm", "model", "repeat"])["patient_id"].nunique()
    if per_repeat.nunique() != 1:
        raise SystemExit("the number of held-out patients differs between repeats:\n"
                         + per_repeat.to_string())
    return oof


def averaged_predictions(arm_df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: the mean of that patient's held-out predictions."""
    first = arm_df.groupby("patient_id", as_index=False).agg(y=("y", "first"))
    mean_p = arm_df.groupby("patient_id", as_index=False).agg(p=("p", "mean"))
    out = first.merge(mean_p, on="patient_id", how="inner")
    # guard against a patient carrying two different outcomes
    spread = arm_df.groupby("patient_id")["y"].nunique()
    if (spread > 1).any():
        raise SystemExit("a patient carries both outcome values")
    return out.sort_values("patient_id").reset_index(drop=True)


def pooled_per_repeat(arm_df: pd.DataFrame) -> np.ndarray:
    return np.array([
        auroc(g["y"], g["p"]) for _, g in arm_df.groupby("repeat", sort=True)
    ])


def bootstrap_auroc_ci(y, p, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Percentile interval from resampling patients, not rows-within-fold."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        draws.append(auroc(y[idx], p[idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi), len(draws)


def bootstrap_diff_ci(y, p1, p2, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Pairs of patients resampled together, so the arms stay matched."""
    y = np.asarray(y).astype(int)
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        draws.append(auroc(y[idx], p1[idx]) - auroc(y[idx], p2[idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def predictor_counts(folds_path, model: str):
    d = pd.read_csv(folds_path)
    d = d[(d.model == model) & (d.arm == "reselect")]
    if d.empty:
        return None
    per_fold = d.drop_duplicates(["repeat", "fold"])
    n = per_fold["n_predictors"]
    return int(np.median(n)), int(n.min()), int(n.max())


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", nargs="+", type=Path, required=True)
    ap.add_argument("--folds", type=Path, default=None,
                    help="fold-level CSV of nested_cv_selection.py, for predictor counts")
    ap.add_argument("--model", default="CatBoost")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "table_s4f.csv")
    args = ap.parse_args()

    oof = load_oof(args.oof)
    if args.model not in set(oof["model"]):
        raise SystemExit(f"model {args.model!r} not found; available: "
                         f"{sorted(oof['model'].unique())}")
    oof = oof[oof.model == args.model].copy()
    oof = check_complete(oof)

    counts = predictor_counts(args.folds, args.model) if args.folds else None
    results, averaged = {}, {}
    for arm in ("fixed", "reselect"):
        d = oof[oof.arm == arm]
        if d.empty:
            continue
        pooled = pooled_per_repeat(d)
        avg = averaged_predictions(d)
        auc_avg = auroc(avg.y, avg.p)
        lo, hi, n_boot = bootstrap_auroc_ci(avg.y, avg.p, args.n_boot)
        results[arm] = dict(pooled_mean=float(pooled.mean()),
                            pooled_sd=float(pooled.std(ddof=1)),
                            n_repeats=len(pooled),
                            auc_avg=float(auc_avg), lo=lo, hi=hi,
                            n_boot=n_boot, n=len(avg))
        averaged[arm] = avg
        print(f"{arm}: pooled {pooled.mean():.4f} +/- {pooled.std(ddof=1):.4f} "
              f"over {len(pooled)} repeats; averaged-prediction AUROC "
              f"{auc_avg:.4f} ({lo:.4f}-{hi:.4f})")

    rows = []
    for arm, r in results.items():
        if counts and arm == "reselect":
            label = f"{counts[0]} ({counts[1]}-{counts[2]})"
        else:
            label = "4"
        rows.append(dict(
            arm=ARM_LABELS.get(arm, arm),
            predictors_per_fold=label,
            pooled_auroc=f"{fmt(r['pooled_mean'])} +/- {fmt(r['pooled_sd'])}",
            averaged_auroc=f"{fmt(r['auc_avg'])} ({fmt(r['lo'])}-{fmt(r['hi'])})",
            difference="reference",
            p="--",
            n_patients=r["n"],
            n_boot=r["n_boot"],
        ))

    # ---- paired comparison on the averaged predictions ---------------------
    if {"fixed", "reselect"} <= set(averaged):
        a, b = averaged["fixed"], averaged["reselect"]
        merged = a.merge(b, on="patient_id", suffixes=("_fixed", "_reselect"))
        if not (merged.y_fixed == merged.y_reselect).all():
            raise SystemExit("the two arms disagree on the outcome of some patients")
        # difference is expressed as re-selection minus fixed, as in Table S4f
        d = delong_paired(merged.y_reselect, merged.p_reselect, merged.p_fixed)
        lo, hi = bootstrap_diff_ci(merged.y_reselect, merged.p_reselect, merged.p_fixed,
                                   args.n_boot)
        per_repeat = []
        for rep, g in oof.groupby("repeat"):
            gf = g[g.arm == "fixed"].set_index("patient_id")
            gr = g[g.arm == "reselect"].set_index("patient_id").loc[gf.index]
            per_repeat.append(auroc(gr.y, gr.p) - auroc(gf.y, gf.p))
        per_repeat = np.array(per_repeat)
        idx = [i for i, r in enumerate(rows) if r["arm"] == ARM_LABELS["reselect"]][0]
        rows[idx]["difference"] = f"{d['difference']:+.4f} ({lo:+.4f} to {hi:+.4f})"
        rows[idx]["p"] = f"{d['p']:.3f}" if d["p"] == d["p"] else "--"
        print(f"paired: difference {d['difference']:+.4f} "
              f"(DeLong {d['ci_low']:+.4f} to {d['ci_high']:+.4f}, P = {d['p']:.3f}); "
              f"bootstrap {lo:+.4f} to {hi:+.4f}; "
              f"mean of {len(per_repeat)} within-repeat differences "
              f"{per_repeat.mean():+.4f} +/- {per_repeat.std(ddof=1):.4f}")

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nfile: {args.out}")


if __name__ == "__main__":
    main()
