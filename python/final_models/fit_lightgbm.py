from pathlib import Path

import pandas as pd
import lightgbm as lgb

TRAIN_PATH = Path(__file__).resolve().parents[2] / "data" / "training_set.xlsx"
# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
TARGET = "outcome"

PARAMS = dict(
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
    seed=42,
    bagging_seed=43,
    feature_fraction_seed=44,
    deterministic=True,
    force_col_wise=True,
    num_threads=1,
    verbose=-1,
)

N_ROUNDS = 200

DATASET_PARAMS = dict(
    max_bin=63,
    min_data_in_bin=3,
    data_random_seed=42,
    feature_pre_filter=False,
    verbose=-1,
)


def load_training_data(path=TRAIN_PATH):
    df = pd.read_excel(path)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(int).to_numpy()
    return X, y


def build_dataset(X, y=None):
    return lgb.Dataset(X, label=y, free_raw_data=False, params=DATASET_PARAMS)


def fit_model(X, y):
    return lgb.train(PARAMS, build_dataset(X, y), num_boost_round=N_ROUNDS)


def main():
    X, y = load_training_data()
    model = fit_model(X, y)
    model.save_model("lightgbm_model.txt")
    print("LightGBM: n=%d, events=%d, trees=%d"
          % (len(y), int(y.sum()), model.num_trees()))
    return model


if __name__ == "__main__":
    main()
