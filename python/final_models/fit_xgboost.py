from pathlib import Path

import pandas as pd
import xgboost as xgb

TRAIN_PATH = Path(__file__).resolve().parents[2] / "data" / "training set.xlsx"
# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
TARGET = "outcome"

PARAMS = dict(
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
    seed=42,
    nthread=1,
    verbosity=0,
)

N_TREES = 2


def load_training_data(path=TRAIN_PATH):
    df = pd.read_excel(path)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = df[TARGET].astype(int).to_numpy()
    return X, y


def build_dmatrix(X, y=None):
    return xgb.DMatrix(X, label=y, nthread=PARAMS["nthread"])


def fit_model(X, y):
    return xgb.train(PARAMS, build_dmatrix(X, y), num_boost_round=N_TREES)


def main():
    X, y = load_training_data()
    model = fit_model(X, y)
    model.save_model("xgboost_model.json")
    print("XGBoost: n=%d, events=%d, trees=%d" % (len(y), int(y.sum()), N_TREES))
    return model


if __name__ == "__main__":
    main()
