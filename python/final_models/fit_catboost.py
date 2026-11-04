from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

TRAIN_PATH = Path(__file__).resolve().parents[2] / "data" / "training_set.xlsx"
# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
TARGET = "outcome"

PARAMS = dict(
    depth=6,
    learning_rate=0.06,
    iterations=200,
    l2_leaf_reg=120.0,
    random_strength=6.0,
    border_count=96,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    thread_count=1,
    verbose=False,
    allow_writing_files=False,
)


def load_training_data(path=TRAIN_PATH):
    df = pd.read_excel(path)
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int).to_numpy()
    return X, y


def build_model():
    return CatBoostClassifier(**PARAMS)


def fit_model(X, y):
    return build_model().fit(X, y)


def main():
    X, y = load_training_data()
    model = fit_model(X, y)
    model.save_model("catboost_model.cbm")
    print("CatBoost: n=%d, events=%d, trees=%d"
          % (len(y), int(y.sum()), model.tree_count_))
    return model


if __name__ == "__main__":
    main()
