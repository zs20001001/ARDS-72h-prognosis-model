from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = Path(__file__).resolve().parents[2] / "data" / "training set.xlsx"
# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
TARGET = "outcome"

VAR_SMOOTHING = 48.32930239


def load_training_data(path=TRAIN_PATH):
    df = pd.read_excel(path)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = df[TARGET].astype(int).to_numpy()
    return X, y


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gnb", GaussianNB(var_smoothing=VAR_SMOOTHING)),
    ])


def fit_model(X, y):
    return build_model().fit(X, y)


def main():
    X, y = load_training_data()
    model = fit_model(X, y)
    joblib.dump(model, "gaussian_nb_model.joblib")
    print("GaussianNB: n=%d, events=%d, var_smoothing=%.8f"
          % (len(y), int(y.sum()), VAR_SMOOTHING))
    return model


if __name__ == "__main__":
    main()
