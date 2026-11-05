# -*- coding: utf-8 -*-
"""The eight classifiers of the manuscript, with their final hyperparameters.

Each entry is a builder that returns a fitted model exposing
``predict_proba``-like output through :func:`positive_class_probability`.
The hyperparameters are exactly those in ``python/final_models/fit_*.py`` (the
values selected by the grid search), so a model refitted here reproduces the
published scores; nothing is retuned inside any resampling loop.

Import from the resampling and evaluation scripts:

    from models import MODEL_ZOO, positive_class_probability
"""

from __future__ import annotations

import numpy as np

SEED = 42

# --------------------------------------------------------------------------- #
# CatBoost
# --------------------------------------------------------------------------- #
CATBOOST_PARAMS = dict(
    depth=6,
    learning_rate=0.06,
    iterations=200,
    l2_leaf_reg=120.0,
    random_strength=6.0,
    border_count=96,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    thread_count=1,
    verbose=False,
    allow_writing_files=False,
)

# --------------------------------------------------------------------------- #
# XGBoost: two boosting rounds at eta = 0.6, the configuration actually
# evaluated in the published grid. The extended grid is in
# python/grid_search/xgboost_extended_grid.py.
# --------------------------------------------------------------------------- #
XGBOOST_PARAMS = dict(
    objective="binary:logistic",
    booster="gbtree",
    tree_method="hist",
    max_depth=4,
    eta=0.6,
    min_child_weight=1,
    subsample=0.60,
    colsample_bytree=0.90,
    max_bin=64,
    scale_pos_weight=1.0,
    seed=SEED,
    nthread=1,
    verbosity=0,
)
XGBOOST_ROUNDS = 2

LIGHTGBM_PARAMS = dict(
    objective="binary",
    boosting_type="gbdt",
    metric="auc",
    num_leaves=4,
    max_depth=-1,
    learning_rate=0.006,
    min_data_in_leaf=20,
    feature_fraction=1.0,
    bagging_fraction=1.0,
    bagging_freq=0,
    lambda_l1=3.0,
    lambda_l2=0.0,
    min_split_gain=0.0,
    scale_pos_weight=1.0,
    is_unbalance=False,
    seed=SEED,
    bagging_seed=43,
    feature_fraction_seed=44,
    deterministic=True,
    force_col_wise=True,
    verbose=-1,
)
LIGHTGBM_ROUNDS = 200

RANDOM_FOREST_PARAMS = dict(
    n_estimators=100,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features=4,
    criterion="gini",
    bootstrap=True,
    class_weight=None,
    random_state=SEED,
    n_jobs=1,
)

DECISION_TREE_PARAMS = dict(
    criterion="entropy",
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=5,
    ccp_alpha=0.007,
    splitter="best",
    max_features=None,
    random_state=SEED,
)

LOGISTIC_PARAMS = dict(C=0.1, solver="lbfgs", max_iter=5000)

SVM_PARAMS = dict(
    kernel="rbf",
    C=0.5623413251903491,
    gamma=1.7782794100389228,
    degree=3,
    coef0=0.0,
    probability=True,
    random_state=SEED,
)

GNB_PARAMS = dict(var_smoothing=48.32930239)


# --------------------------------------------------------------------------- #
# Builders. Each returns an object with a ``predict`` method that accepts a
# 2-D float array and returns P(death).
# --------------------------------------------------------------------------- #
class _Booster:
    """Small adapter so the two native-API boosters look like sklearn models."""

    def __init__(self, booster, library: str):
        self.booster = booster
        self.library = library

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.library == "xgboost":
            import xgboost as xgb

            return self.booster.predict(xgb.DMatrix(X, nthread=1))
        return self.booster.predict(X)


def build_catboost():
    from catboost import CatBoostClassifier

    return CatBoostClassifier(**CATBOOST_PARAMS)


def build_xgboost():
    import xgboost as xgb

    def fit(X, y):
        dtrain = xgb.DMatrix(X, label=y, nthread=1)
        booster = xgb.train(XGBOOST_PARAMS, dtrain, num_boost_round=XGBOOST_ROUNDS)
        return _Booster(booster, "xgboost")

    return _Lazy(fit)


def build_lightgbm():
    import lightgbm as lgb

    def fit(X, y):
        dataset = lgb.Dataset(X, label=y, free_raw_data=False)
        booster = lgb.train(LIGHTGBM_PARAMS, dataset, num_boost_round=LIGHTGBM_ROUNDS)
        return _Booster(booster, "lightgbm")

    return _Lazy(fit)


class _Lazy:
    """Uniform interface: a callable that takes (X, y) and returns a model."""

    def __init__(self, fit):
        self._fit = fit

    def fit(self, X, y):
        return self._fit(np.asarray(X, dtype=float), np.asarray(y).astype(int))


def build_random_forest():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**RANDOM_FOREST_PARAMS)


def build_decision_tree():
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(**DECISION_TREE_PARAMS)


def build_logistic():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(**LOGISTIC_PARAMS)


def build_svm():
    from sklearn.svm import SVC

    return SVC(**SVM_PARAMS)


def build_gnb():
    from sklearn.naive_bayes import GaussianNB

    return GaussianNB(**GNB_PARAMS)


MODEL_ZOO = {
    "CatBoost": build_catboost,
    "XGBoost": build_xgboost,
    "LightGBM": build_lightgbm,
    "Random forest": build_random_forest,
    "Decision tree": build_decision_tree,
    "Logistic regression": build_logistic,
    "SVM": build_svm,
    "Gaussian NB": build_gnb,
}


def positive_class_probability(model, X) -> np.ndarray:
    """P(death) for any of the eight models, whatever their native API."""
    X = np.asarray(X, dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def fit_model(name: str, X, y):
    """Fit one of the eight named models with its frozen hyperparameters."""
    return MODEL_ZOO[name]().fit(np.asarray(X, dtype=float), np.asarray(y).astype(int))
