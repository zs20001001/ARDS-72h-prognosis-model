# -*- coding: utf-8 -*-
"""Two-stage feature selection with the fallback rule of the manuscript.

Stage 1  Boruta on a random forest (500 trees, maxRuns 100, significance level
         0.01) -> confirmed set.
Stage 2  LASSO logistic regression, 10-fold CV, 100-point regularisation path,
         variables kept at the one-standard-error rule.
Final    intersection of the two. The intersection is used as the predictor set
         of the final model; the two selectors are applied independently to the
         same data, so a variable has to pass both stages to be retained.

The analysis reported in the paper was run with Boruta 7.0.0 and glmnet in R
4.3.3 (Table S14, row 6), where the significance level was applied with the
Bonferroni adjustment of ``mcAdj = TRUE`` and the LASSO path was the one glmnet
generates for ``nlambda = 100``. This module is a re-implementation on
BorutaPy and scikit-learn: the same three settings are passed explicitly, but
the two libraries differ in how the importance of a shadow feature is computed
and in how the regularisation path is spaced, so a given resample need not
return exactly the same variable set as the R run. The published numbers are
those of the R run; this module is provided so that the rule, and not only its
output, is inspectable.

Fallback (Table S14, row 8). When a stage returns no variable, the Boruta
confirmed set is carried forward unchanged and the classifier is fitted on that
set. If Boruta itself confirms nothing, all candidate variables are used. The
branch that was taken is recorded by :func:`select_features` so that the
frequency of each fallback can be reported.

This module is deliberately restricted to *training* data; every call receives
only the rows of one training set or one training fold.
"""

from __future__ import annotations

import numpy as np

BORUTA_TREES = 500
BORUTA_MAX_RUNS = 100
BORUTA_PVALUE = 0.01
NFOLD = 10
LASSO_CS = np.logspace(-3, 0, 100)


def boruta_confirmed(X, y, seed: int = 42, n_trees: int = BORUTA_TREES):
    """Confirmed variables from BorutaPy; empty list when nothing is confirmed."""
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(
        n_estimators=n_trees, max_features="sqrt", n_jobs=1, random_state=seed
    )
    selector = BorutaPy(
        forest,
        n_estimators=n_trees,
        max_iter=BORUTA_MAX_RUNS,
        alpha=BORUTA_PVALUE,
        verbose=0,
        random_state=seed,
    )
    selector.fit(np.asarray(X, dtype=float), np.asarray(y).astype(int))
    return [int(i) for i in np.where(selector.support_)[0]]


def _l1_logistic(C: float, seed: int = 42):
    """L1-penalised logistic regression, spelled the way the local sklearn wants.

    sklearn 1.9 expresses the penalty through ``l1_ratio``; earlier releases use
    ``penalty="l1"``. Both give the same fit, so the helper keeps the script
    runnable on either without emitting a deprecation warning.
    """
    from sklearn.linear_model import LogisticRegression

    try:
        return LogisticRegression(C=C, solver="liblinear", l1_ratio=1.0,
                                  max_iter=1000, random_state=seed)
    except TypeError:
        return LogisticRegression(C=C, solver="liblinear", penalty="l1",
                                  max_iter=1000, random_state=seed)


def lasso_1se(X, y, seed: int = 42, nfold: int = NFOLD, cs=LASSO_CS):
    """Variables with a non-zero coefficient at the glmnet lambda.1se.

    X is standardised inside the selection step, as glmnet does, and the C grid
    spans 1e-3 to 1 in 100 steps because C and lambda are reciprocal. Ten-fold
    cross-validated binomial deviance is computed for every C, and the
    one-standard-error rule keeps the *most* regularised model whose deviance is
    within one standard error of the minimum, i.e. the smallest C on that
    plateau.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    y = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)

    deviance = np.zeros((nfold, len(cs)))
    for f, (tr, te) in enumerate(skf.split(Xs, y)):
        for j, C in enumerate(cs):
            model = _l1_logistic(C, seed).fit(Xs[tr], y[tr])
            p = np.clip(model.predict_proba(Xs[te])[:, 1], 1e-9, 1 - 1e-9)
            deviance[f, j] = -2 * np.mean(
                y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)
            )

    mean = deviance.mean(axis=0)
    se = deviance.std(axis=0, ddof=1) / np.sqrt(nfold)
    best = int(np.argmin(mean))
    plateau = np.where(mean <= mean[best] + se[best])[0]
    c_1se = float(np.asarray(cs)[plateau.min()])   # smallest C = largest lambda
    model = _l1_logistic(c_1se, seed).fit(Xs, y)
    return [int(i) for i in np.where(np.abs(model.coef_.ravel()) > 1e-8)[0]]


def select_features(X, y, seed: int = 42, n_trees: int = BORUTA_TREES):
    """Run both stages and return the predictor set actually used for fitting.

    Returns
    -------
    indices : list of int
        Column indices of the variables to fit on.
    branch : str
        ``"both"`` (intersection non-empty), ``"boruta_only"`` (LASSO returned
        nothing), ``"lasso_only"`` (Boruta confirmed nothing), or
        ``"all_vars"`` (both empty).
    n_boruta, n_lasso : int
        Size of each stage's set, before the fallback.
    """
    boruta = boruta_confirmed(X, y, seed=seed, n_trees=n_trees)
    lasso = lasso_1se(X, y, seed=seed)

    both = sorted(set(boruta) & set(lasso))
    if both:
        return both, "both", len(boruta), len(lasso)
    if boruta:
        return sorted(boruta), "boruta_only", len(boruta), len(lasso)
    if lasso:
        return sorted(lasso), "lasso_only", len(boruta), len(lasso)
    return list(range(np.asarray(X).shape[1])), "all_vars", len(boruta), len(lasso)
