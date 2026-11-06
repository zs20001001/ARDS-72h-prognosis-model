from pathlib import Path

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

TRAIN_PATH = Path(__file__).resolve().parents[2] / "data" / "training set.xlsx"
# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
TARGET = "outcome"

PARAMS = dict(
    criterion="entropy",
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=5,
    ccp_alpha=0.007,
    splitter="best",
    max_features=None,
    max_leaf_nodes=None,
    class_weight=None,
    min_impurity_decrease=0.0,
    min_weight_fraction_leaf=0.0,
    random_state=42,
)


def load_training_data(path=TRAIN_PATH):
    df = pd.read_excel(path)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = df[TARGET].astype(int).to_numpy()
    return X, y


def build_model():
    return DecisionTreeClassifier(**PARAMS)


def fit_model(X, y):
    return build_model().fit(X, y)


def main():
    X, y = load_training_data()
    model = fit_model(X, y)
    joblib.dump(model, "decision_tree_model.joblib")
    print("DecisionTree: n=%d, events=%d, leaves=%d, depth=%d"
          % (len(y), int(y.sum()), model.get_n_leaves(), model.get_depth()))
    return model


if __name__ == "__main__":
    main()
