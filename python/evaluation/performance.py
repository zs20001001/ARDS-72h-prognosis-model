# -*- coding: utf-8 -*-
"""Performance metrics used throughout the manuscript (Table S14, rows 10-13).

Every function takes a binary outcome vector ``y`` (1 = death) and a vector of
predicted probabilities ``p`` for the same patients, in the same order.

Definitions follow Table S14:

* discrimination  - AUROC with the DeLong standard error and a 95% CI; two
  models scored on the same patients are compared with the paired DeLong test.
* calibration     - slope and calibration-in-the-large from a logistic model
  ``logit(y) = a + b * logit(p)``; the slope is ``b`` and the intercept ``a``.
* overall         - Brier score, Brier skill score = 1 - Brier / [p_obs(1 - p_obs)].
* threshold       - sensitivity, specificity, accuracy, PPV, NPV and MCC at a
  fixed cut-off (the frozen 0.495 unless stated otherwise).
* uncertainty     - percentile bootstrap with 2,000 patient-level resamples
  (seed 7), resampling patients and recomputing the statistic in each resample.

Run ``python performance.py`` for a self-check on simulated data.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

DEFAULT_THRESHOLD = 0.495
DEFAULT_BOOTSTRAP = 2000
DEFAULT_BOOTSTRAP_SEED = 7


# --------------------------------------------------------------------------- #
# DeLong
# --------------------------------------------------------------------------- #
def _midrank(x: np.ndarray) -> np.ndarray:
    """Mid-ranks, i.e. average ranks for ties (DeLong et al. 1988)."""
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(predictions: np.ndarray, n_pos: int):
    """Sun & Xu (2014) fast DeLong.

    ``predictions`` has one row per model and columns sorted so that the first
    ``n_pos`` columns are the patients with the outcome.
    """
    n_neg = predictions.shape[1] - n_pos
    pos, neg = predictions[:, :n_pos], predictions[:, n_pos:]
    tx = np.array([_midrank(pos[r]) for r in range(predictions.shape[0])])
    ty = np.array([_midrank(neg[r]) for r in range(predictions.shape[0])])
    tz = np.array([_midrank(predictions[r]) for r in range(predictions.shape[0])])
    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1.0) / 2.0 / n_neg
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / n_pos + sy / n_neg


def auroc(y, p) -> float:
    """AUROC, computed from the DeLong placement values (tie-corrected)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-y, kind="mergesort")
    aucs, _ = _fast_delong(np.atleast_2d(p[order]), int(y.sum()))
    return float(aucs[0])


def auroc_ci(y, p, alpha: float = 0.05):
    """AUROC with the DeLong standard error and a two-sided (1 - alpha) CI."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-y, kind="mergesort")
    aucs, cov = _fast_delong(np.atleast_2d(p[order]), int(y.sum()))
    se = float(np.sqrt(cov[0, 0]))
    z = norm.ppf(1 - alpha / 2)
    return dict(
        auroc=float(aucs[0]),
        se=se,
        ci_low=float(aucs[0] - z * se),
        ci_high=float(aucs[0] + z * se),
    )


def delong_paired(y, p1, p2):
    """Paired DeLong comparison of two models scored on the same patients."""
    y = np.asarray(y).astype(int)
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    order = np.argsort(-y, kind="mergesort")
    n_pos = int(y.sum())
    aucs, cov = _fast_delong(np.vstack((p1[order], p2[order])), n_pos)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(var))
    diff = float(aucs[0] - aucs[1])
    z = diff / se if se > 0 else np.nan
    p = float(2 * (1 - norm.cdf(abs(z)))) if se > 0 else np.nan
    return dict(
        auroc_1=float(aucs[0]),
        auroc_2=float(aucs[1]),
        difference=diff,
        se=se,
        ci_low=diff - 1.96 * se,
        ci_high=diff + 1.96 * se,
        z=z,
        p=p,
    )


# --------------------------------------------------------------------------- #
# Calibration, overall performance, threshold-dependent metrics
# --------------------------------------------------------------------------- #
def _logit(p, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration(y, p):
    """Calibration slope and calibration-in-the-large.

    ``logit(y) = a + b * logit(p)``: slope = b, CITL = a. A slope below 1 means
    the predictions are over-dispersed; a positive CITL means under-prediction.
    """
    y = np.asarray(y).astype(float)
    lp = _logit(p)
    design = np.column_stack([np.ones_like(lp), lp])
    # Logistic fit by Newton-Raphson on the two parameters, without penalisation.
    beta = np.array([0.0, 1.0])
    for _ in range(200):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1 - mu)
        grad = design.T @ (y - mu)
        hess = design.T @ (design * w[:, None])
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return dict(slope=float(beta[1]), citl=float(beta[0]))


def brier(y, p) -> float:
    y = np.asarray(y).astype(float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def brier_skill(y, p) -> float:
    """1 - Brier / [p_obs (1 - p_obs)], p_obs the observed prevalence."""
    y = np.asarray(y).astype(float)
    ref = float(np.mean(y)) * (1 - float(np.mean(y)))
    return float(1 - brier(y, p) / ref)


def threshold_metrics(y, p, threshold: float = DEFAULT_THRESHOLD):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    acc = (tp + tn) / len(y)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else np.nan
    return dict(
        threshold=threshold,
        sensitivity=float(sens),
        specificity=float(spec),
        accuracy=float(acc),
        ppv=float(ppv),
        npv=float(npv),
        mcc=float(mcc),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
def bootstrap_percentile(y, p, statistic, n_boot: int = DEFAULT_BOOTSTRAP,
                         seed: int = DEFAULT_BOOTSTRAP_SEED, alpha: float = 0.05):
    """Percentile bootstrap over patients.

    ``statistic`` takes ``(y, p)`` for one resample and returns a float. The
    whole statistic is recomputed inside each resample, so a statistic that
    itself involves refitting (the recalibrated model, for instance) carries its
    own fitting uncertainty into the interval.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            values.append(float(statistic(y[idx], p[idx])))
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            continue  # a resample with one outcome class carries no AUROC
    values = np.asarray(values, dtype=float)
    return dict(
        estimate=float(statistic(y, p)),
        ci_low=float(np.percentile(values, 100 * alpha / 2)),
        ci_high=float(np.percentile(values, 100 * (1 - alpha / 2))),
        n_boot=int(len(values)),
    )


def summary(y, p, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """All of the above for one model in one cohort."""
    out = {}
    out.update(auroc_ci(y, p))
    out.update(calibration(y, p))
    out["brier"] = brier(y, p)
    out["brier_skill"] = brier_skill(y, p)
    out.update(threshold_metrics(y, p, threshold))
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 206)
    p = np.clip(0.25 + 0.45 * y + rng.normal(0, 0.15, 206), 0.001, 0.999)

    print("AUROC      %.4f (95%% CI %.4f-%.4f, SE %.4f)"
          % (auroc_ci(y, p)["auroc"], auroc_ci(y, p)["ci_low"],
             auroc_ci(y, p)["ci_high"], auroc_ci(y, p)["se"]))
    cal = calibration(y, p)
    print("slope %.3f   CITL %+.3f" % (cal["slope"], cal["citl"]))
    print("Brier %.4f   Brier skill %.4f" % (brier(y, p), brier_skill(y, p)))
    tm = threshold_metrics(y, p)
    print("at %.3f: sens %.3f spec %.3f PPV %.3f NPV %.3f MCC %.3f"
          % (tm["threshold"], tm["sensitivity"], tm["specificity"],
             tm["ppv"], tm["npv"], tm["mcc"]))
    ci = bootstrap_percentile(y, p, auroc, n_boot=200)
    print("bootstrap AUROC %.4f (%.4f-%.4f, %d resamples)"
          % (ci["estimate"], ci["ci_low"], ci["ci_high"], ci["n_boot"]))
    pair = delong_paired(y, p, np.clip(p + rng.normal(0, 0.05, 206), 0.001, 0.999))
    print("paired DeLong diff %+.4f (%.4f to %.4f) P = %.4f"
          % (pair["difference"], pair["ci_low"], pair["ci_high"], pair["p"]))
