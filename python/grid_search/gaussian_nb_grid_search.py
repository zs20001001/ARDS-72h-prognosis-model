import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "data")
TRAIN = os.path.join(DATA_DIR, "training_set.xlsx")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "gaussian_nb_grid_search")

RANGE_LOW = 10.0
RANGE_HIGH = 100.0
N_POINTS = 20

LABEL = "outcome"
# `peep` is day-1 PEEP as stored in the source table (renamed d1_peep in R/01_boruta.R).
FEATURES = ["d3_a_ado2", "peep", "d3_sofa", "d3_lac"]

N_SPLITS = 10
CV_SEED = 42
N_JOBS = 1

TIE_TOL = 1e-10


def read_table(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("training file not found: %s" % os.path.abspath(path))
    ext = os.path.splitext(path)[1].lower()

    if ext in (".csv", ".txt"):
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, encoding="latin-1")

    if ext in (".xlsx", ".xlsm", ".xltx"):
        return pd.read_excel(path, sheet_name=0, engine="openpyxl")

    if ext == ".xls":
        return pd.read_excel(path, sheet_name=0, engine="xlrd")

    raise ValueError("unsupported file type: %s" % ext)


def to_xy(frame):
    missing = [c for c in FEATURES + [LABEL] if c not in frame.columns]
    if missing:
        raise KeyError("missing columns %s, found %s" % (missing, list(frame.columns)))
    X = frame[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = pd.to_numeric(frame[LABEL], errors="coerce").to_numpy()
    ok = ~np.isnan(y)
    if (~ok).any():
        print("  dropped %d rows with missing outcome" % int((~ok).sum()))
    return X[ok], y[ok].astype(int)


def make_model(var_smoothing):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gnb", GaussianNB(var_smoothing=var_smoothing)),
    ])


def cv_auc(model, X, y, n_splits, seed, n_jobs=1):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=n_jobs)
    return float(scores.mean()), float(scores.std()), scores


def main():
    print("=" * 78)
    print("Gaussian naive Bayes grid search on the training set")
    print("=" * 78)

    lo, hi = float(RANGE_LOW), float(RANGE_HIGH)
    if not (lo > 0 and hi > lo):
        raise ValueError("invalid range [%s, %s]" % (lo, hi))
    grid = np.logspace(np.log10(lo), np.log10(hi), N_POINTS)

    print()
    print("range  : var_smoothing in [%g, %g]" % (lo, hi))
    print("grid   : %d log-spaced points, %g ... %g"
          % (N_POINTS, grid[0], grid[-1]))
    print("ranking: highest mean %d-fold cv auc on the training set" % N_SPLITS)

    frame = read_table(TRAIN)
    X, y = to_xy(frame)

    print()
    print("training set %d cases (prevalence %.3f)" % (X.shape[0], y.mean()))
    print("features %s" % FEATURES)

    os.makedirs(OUT_DIR, exist_ok=True)

    print()
    print("%14s%13s%9s" % ("var_smoothing", "cv_auc", "sd"))
    print("-" * 36)

    rows, best = [], None
    for value in grid:
        value = float(value)
        mean, sd, _ = cv_auc(make_model(value), X, y, N_SPLITS, CV_SEED, N_JOBS)
        rows.append({"var_smoothing": value, "cv_auc": mean, "cv_auc_sd": sd})
        print("%14.6g%13.4f%9.4f" % (value, mean, sd))

        if best is None or (mean > best["cv_auc"] + TIE_TOL) or (
                abs(mean - best["cv_auc"]) <= TIE_TOL
                and value < best["var_smoothing"]):
            best = rows[-1]

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "grid_log.csv"), index=False,
                 encoding="utf-8-sig")

    best_value = float(best["var_smoothing"])
    cv_max = float(best["cv_auc"])
    tied = table[np.isclose(table["cv_auc"], cv_max, rtol=0, atol=1e-12)]

    if abs(np.log10(best_value) - np.log10(lo)) < 1e-9:
        position = "lower bound"
    elif abs(np.log10(best_value) - np.log10(hi)) < 1e-9:
        position = "upper bound"
    else:
        position = "interior"

    print()
    print("=" * 78)
    print("best combination")
    print("=" * 78)
    print("  var_smoothing  %.16g" % best_value)
    print("  position       %s (point %d of %d)"
          % (position, int(np.argmin(np.abs(grid - best_value))) + 1, len(grid)))
    print("  cv auc         %.4f +- %.4f" % (cv_max, best["cv_auc_sd"]))
    if len(tied) > 1:
        print("  tied points    %d, smallest var_smoothing selected" % len(tied))

    payload = {
        "model": "gaussian naive bayes",
        "range": {"var_smoothing_low": lo, "var_smoothing_high": hi,
                  "span_decades": float(np.log10(hi) - np.log10(lo)),
                  "n_points": int(N_POINTS)},
        "best_params": {"var_smoothing": best_value},
        "best_point_at": position,
        "best_point_index": int(np.argmin(np.abs(grid - best_value))) + 1,
        "ranking": "highest mean %d-fold stratified cv auc on the training set"
                   % N_SPLITS,
        "cv_auc": {"mean": cv_max, "sd": float(best["cv_auc_sd"])},
        "features": FEATURES,
        "tie_break_rule": "smallest var_smoothing among points with equal cv auc",
        "n_points_tied_at_max_cv": int(len(tied)),
        "tied_var_smoothing": [float(v) for v in tied["var_smoothing"]],
        "grid": [float(v) for v in grid],
        "cv_seed": CV_SEED,
        "n_cv_train": int(len(y)),
    }

    with open(os.path.join(OUT_DIR, "best_params.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print("saved to %s" % os.path.abspath(OUT_DIR))
    return payload


if __name__ == "__main__":
    main()
