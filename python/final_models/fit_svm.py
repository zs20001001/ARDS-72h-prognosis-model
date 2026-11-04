from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TRAIN_PATH = Path(__file__).resolve().parents[2] / "data" / "training_set.xlsx"
# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]
TARGET = "outcome"

PARAMS = dict(
    kernel="rbf",
    C=0.5623413251903491,
    gamma=1.7782794100389228,
    degree=3,
    coef0=0.0,
    class_weight=None,
    shrinking=True,
    tol=1e-3,
    cache_size=500,
    max_iter=-1,
    probability=True,
    random_state=42,
)


def load_training_data(path=TRAIN_PATH):
    df = pd.read_excel(path)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = df[TARGET].astype(int).to_numpy()
    return X, y


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(**PARAMS)),
    ])


def fit_model(X, y):
    return build_model().fit(X, y)


def main():
    X, y = load_training_data()
    model = fit_model(X, y)
    joblib.dump(model, "svm_model.joblib")
    print("SVM: n=%d, events=%d, support_vectors=%d"
          % (len(y), int(y.sum()), model.named_steps["svc"].support_.shape[0]))
    return model


if __name__ == "__main__":
    main()
