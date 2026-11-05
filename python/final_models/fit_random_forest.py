from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TRAIN_PATH = Path(__file__).resolve().parents[2] / "data" / "training_set.xlsx"
# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
TARGET = "outcome"

# Hyperparameters selected by 10-fold CV grid search
# (python/grid_search/random_forest_grid_search.py), argmax at 100/8/2/2.
# The search pool was:
#   n_estimators      100 / 300 / 400 / 600
#   max_depth         8 / 10
#   min_samples_split 2 / 5
#   min_samples_leaf  1 / 2
PARAMS = dict(
    n_estimators=100,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features=4,
    criterion="gini",
    bootstrap=True,
    class_weight=None,
    random_state=42,
    # Single thread, so that the reported fit reproduces the setting stated in
    # Supplementary Table S14. Results are unchanged: trees are independent and
    # the seed is fixed, so n_jobs affects run time only.
    n_jobs=1,
)


def load_training_data(path=TRAIN_PATH):
    df = pd.read_excel(path)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = df[TARGET].astype(int).to_numpy()
    return X, y


def build_model():
    return RandomForestClassifier(**PARAMS)


def fit_model(X, y):
    return build_model().fit(X, y)


def main():
    X, y = load_training_data()
    model = fit_model(X, y)
    joblib.dump(model, "random_forest_model.joblib")
    print("RandomForest: n=%d, events=%d, trees=%d, max_depth=%d"
          % (len(y), int(y.sum()), model.n_estimators, model.max_depth))
    for name, imp in zip(FEATURES, model.feature_importances_):
        print("  %-12s importance=%.4f" % (name, imp))
    return model


if __name__ == "__main__":
    main()
