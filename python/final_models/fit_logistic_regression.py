import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = Path(__file__).resolve().parents[2] / "data" / "training_set.xlsx"
# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
TARGET = "outcome"

C_FINAL = 0.1


def load_training_data(path=TRAIN_PATH):
    df = pd.read_excel(path)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = df[TARGET].astype(int).to_numpy()
    return X, y


def build_model(C=C_FINAL):
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, solver="lbfgs", max_iter=5000)),
    ])


def fit_model(X, y, C=C_FINAL):
    return build_model(C).fit(X, y)


def main():
    X, y = load_training_data()
    model = fit_model(X, y)
    joblib.dump(model, "logistic_regression_model.joblib")
    coef = model.named_steps["lr"].coef_[0]
    print("LogisticRegression: n=%d, events=%d, C=%.3f" % (len(y), int(y.sum()), C_FINAL))
    for name, b in zip(FEATURES, coef):
        print("  %-12s beta_per_SD=%+.4f  OR_per_SD=%.4f" % (name, b, math.exp(b)))
    return model


if __name__ == "__main__":
    main()
