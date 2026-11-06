# A 72-Hour Machine Learning Prognostic Model for ARDS Patients Requiring Invasive Mechanical Ventilation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23938386.svg)](https://doi.org/10.5281/zenodo.23938386)

Analysis code accompanying the manuscript:

> **Development and External Validation of a Machine Learning-Based 72-Hour Prognostic Model for
> Acute Respiratory Distress Syndrome Patients Undergoing Invasive Ventilation**
> Sheng Zhao et al. *(manuscript under review)*

This repository contains every script needed to reproduce the feature selection, hyperparameter
tuning and final model fitting reported in the paper. It does **not** contain patient-level data
(see [Data](#data)).

---

## Pipeline overview

| Step | Script(s) | What it does |
| :--- | :-------- | :----------- |
| 1 | `R/01_boruta.R` | Boruta feature selection on 69 candidate variables (500 trees, `maxRuns = 100`, `pValue = 0.01`), with `TentativeRoughFix` for tentative attributes |
| 2 | `R/02_lasso.R` | LASSO logistic regression, 10-fold cross-validation, variables retained at `lambda.1se` |
| 3 | `python/grid_search/*.py` | Exhaustive grid search for 8 classifiers, stratified 10-fold CV, ranked by mean CV AUC |
| 4 | `python/final_models/*.py` | Final models refit on the complete training set using the selected hyperparameters, then serialised to disk |
| 5 | `python/evaluation/cross_validate.py` | Stratified 10-fold cross-validation of the final four-variable specification, fold assignment shared by all eight models |
| 6 | `python/resampling/optimism_all8_reselect.py` | Harrell bootstrap with the two selectors re-run inside every resample; optimism and optimism-corrected AUROC for each of the eight models, a fixed-predictor arm for the paired comparison, and an out-of-bag arm in which the scoring patients take part in neither the selection nor the fitting |
| 7 | `python/resampling/nested_cv_selection.py` | Nested cross-validation (outer 10-fold × 5 repeats) that repeats the selection inside every training fold and records which fallback branch was taken |
| 8 | `python/evaluation/recalibration.py` | Two-parameter logistic recalibration evaluated by 10-fold cross-validation inside the cohort, with percentile bootstrap intervals from repeating the whole procedure |
| 9 | `python/evaluation/performance.py` | AUROC with DeLong standard error, paired DeLong, calibration slope and CITL, Brier and Brier skill, threshold-dependent metrics, percentile bootstrap |

The intersection of the Boruta and LASSO results gave **four predictors**, measured within the
first 72 h after initiation of invasive ventilation (time zero is intubation; Day 1 is 0–24 h and
Day 3 is 48–72 h):

| Variable | Meaning |
| :------- | :------ |
| `d3_a_ado2` | Alveolar–arterial oxygen gradient on day 3 |
| `peep` | Positive end-expiratory pressure on day 1 (stored as `peep` in the source table, renamed `d1_peep` by the R scripts) |
| `d3_sofa` | SOFA score on day 3 |
| `d3_lac` | Lactate on day 3 |

Classifiers evaluated: CatBoost, XGBoost, LightGBM, Random Forest, Decision Tree, Logistic
Regression, Support Vector Machine, Gaussian Naive Bayes.

---

## Repository structure

```
.
├── R/
│   ├── 01_boruta.R            # Boruta feature selection
│   └── 02_lasso.R             # LASSO feature selection
├── python/
│   ├── grid_search/
│   │   ├── catboost_grid_search.py
│   │   ├── decision_tree_grid_search.py
│   │   ├── gaussian_nb_grid_search.py
│   │   ├── lightgbm_grid_search.py
│   │   ├── logistic_regression_grid_search.py
│   │   ├── random_forest_grid_search.py
│   │   ├── svm_grid_search.py
│   │   ├── xgboost_grid_search.py
│   │   ├── xgboost_extended_grid.py     # 70-configuration extension (Supplementary File S5b)
│   │   └── xgboost_extended_grid.csv    # its output, as supplied with the manuscript
│   ├── final_models/
│   │   ├── fit_catboost.py
│   │   ├── fit_decision_tree.py
│   │   ├── fit_gaussian_nb.py
│   │   ├── fit_lightgbm.py
│   │   ├── fit_logistic_regression.py
│   │   ├── fit_random_forest.py
│   │   ├── fit_svm.py
│   │   └── fit_xgboost.py
│   ├── evaluation/
│   │   ├── performance.py          # AUROC/DeLong, calibration, Brier, threshold metrics, bootstrap
│   │   ├── cross_validate.py       # step 5
│   │   └── recalibration.py        # step 8
│   └── resampling/
│       ├── models.py               # the eight classifiers with their frozen hyperparameters
│       ├── feature_selection.py    # Boruta + LASSO(lambda.1se) + fallback rule
│       ├── optimism_all8_reselect.py  # step 6
│       └── nested_cv_selection.py  # step 7
├── data/
│   └── README.md              # required input format (data itself is not distributed)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

The extended XGBoost grid used in the response to comment 6 lives beside the other search scripts:
`python/grid_search/xgboost_extended_grid.py` (70 configurations) and its output
`python/grid_search/xgboost_extended_grid.csv`, which is the file supplied as Supplementary File S5b.

---

## Data

Patient-level data are **not** included in this repository because of patient privacy and the terms
of the institutional ethics approval. De-identified data are available from the corresponding
author on reasonable request.

To run the code, place the workbooks at:

```
data/training set.xlsx              # development cohort, required
data/internal validation set.xlsx   # internal validation cohort
data/external validation set.xlsx   # external validation cohort
```

The development cohort is the only file read by the selection, tuning, fitting and resampling
scripts; the two validation workbooks are read only when the frozen model is scored. Each first
worksheet must contain one row per patient with the columns listed in
[`data/README.md`](data/README.md). Absolute paths are not used anywhere; every script resolves the
data file relative to its own location.

---

## Software requirements

**Python** (developed and tested on Python 3.13):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**R** (≥ 4.2):

```r
install.packages(c("Boruta", "glmnet", "readxl"))
```

---

## Reproducing the analysis

Run the steps in order from the repository root. All random seeds are fixed: `42` for the Python
scripts, `500` for the R scripts.

### Step 1 — feature selection (R)

```bash
Rscript R/01_boruta.R
Rscript R/02_lasso.R
```

### Step 2 — hyperparameter tuning (Python)

Each grid-search script is independent and writes its own ranking table next to the script.
A full sweep over eight models takes several hours on a laptop; the CatBoost search alone evaluates
a large parameter grid with 10-fold CV.

```bash
python python/grid_search/catboost_grid_search.py
python python/grid_search/xgboost_grid_search.py
python python/grid_search/lightgbm_grid_search.py
python python/grid_search/random_forest_grid_search.py
python python/grid_search/decision_tree_grid_search.py
python python/grid_search/logistic_regression_grid_search.py
python python/grid_search/svm_grid_search.py
python python/grid_search/gaussian_nb_grid_search.py
```

### Step 3 — final models (Python)

The hyperparameters selected in Step 2 are hard-coded at the top of each script
(`PARAMS`, `C_FINAL`, …), so Step 3 can be run on its own.

```bash
python python/final_models/fit_catboost.py
python python/final_models/fit_xgboost.py
python python/final_models/fit_lightgbm.py
python python/final_models/fit_random_forest.py
python python/final_models/fit_decision_tree.py
python python/final_models/fit_logistic_regression.py
python python/final_models/fit_svm.py
python python/final_models/fit_gaussian_nb.py
```

### Step 4 — resampling, recalibration and performance evaluation

These scripts are the implementation of Supplementary Table S14 (rows 5-13). They read the
training cohort (and, for recalibration, a cohort file that already contains the predicted
probability column) and write CSV output next to the script unless `--out` is given.

```bash
# 10-fold cross-validation of the final four-variable specification
python python/evaluation/cross_validate.py --data "data/training set.xlsx"

# Harrell bootstrap with the selectors re-run inside each resample (whole development
# procedure), the fixed-predictor arm for the paired comparison, and the out-of-bag
# variant reported in Table S8, section D
python python/resampling/optimism_all8_reselect.py --data "data/training set.xlsx" \
    --arm reselect --n-boot 200
python python/resampling/optimism_all8_reselect.py --data "data/training set.xlsx" \
    --arm fixed --n-boot 200
python python/resampling/optimism_all8_reselect.py --data "data/training set.xlsx" \
    --arm oob --n-boot 200 --models CatBoost

# Nested cross-validation: selection repeated inside every training fold
python python/resampling/nested_cv_selection.py --data "data/training set.xlsx"

# Two-parameter logistic recalibration, cross-validated, with bootstrap intervals.
# The workbook is a validation cohort with the frozen model's probability added as column p
python python/evaluation/recalibration.py --data "data/external validation set.xlsx" \
    --outcome outcome --p-col p
python python/evaluation/recalibration.py --data "data/internal validation set.xlsx" \
    --outcome outcome --p-col p

# Metric library used by all of the above; running it prints a self-check
python python/evaluation/performance.py
```

Timing: the bootstrap is dominated by re-running Boruta in each resample (roughly 1-3 minutes per
resample at 500 trees on one core), so use `--n-jobs` to parallelise or `--n-boot` for a shorter run.
For a wiring check, `--n-boot 5 --boruta-trees 50` finishes in about a minute.

The predicted-probability column for `recalibration.py` is produced from the model bundle:

```python
import joblib, pandas as pd
bundle = joblib.load("catboost_theta6_model.pkl")
df["p"] = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
```

---

## Outputs

| Script | Output |
| :----- | :----- |
| `R/01_boruta.R` | `R/results_boruta_report/` — `attStats.csv`, `attStats_roughfix.csv`, `imp_history.csv`, `imp_history.png`, `imp_boxplot.png`, `boruta_fit.rds` |
| `R/02_lasso.R` | console output: coefficients and selected variables at `lambda.min` / `lambda.1se` |
| `python/grid_search/*.py` | `*_grid_search.csv` (full ranking), `*_grid_search_best.json` or `top10.csv` (best configuration), plus per-fold AUCs |
| `python/final_models/*.py` | serialised model (`*.cbm`, `*.json`, `*.txt`, `*.joblib`) and console output with the fitted coefficients or tree counts |
| `python/evaluation/cross_validate.py` | `results/cross_validation.csv` (per fold) and `results/cross_validation_summary.csv` |
| `python/resampling/optimism_all8_reselect.py` | `results/optimism_all8.csv` (optimism, corrected AUROC, CI per model), `*_per_replicate.csv`, `*_run.json` |
| `python/resampling/nested_cv_selection.py` | `results/nested_cv.csv` (per fold, with the fallback branch taken) and `results/nested_cv_summary.csv` |
| `python/evaluation/recalibration.py` | `results/recalibration.csv` (original vs recalibrated, with bootstrap intervals) and `*_predictions.csv` |
| `python/grid_search/xgboost_extended_grid.py` | `xgboost_extended_grid.csv` — 70 configurations, the file supplied as Supplementary File S5b |

Generated files are written next to the script that produced them and are excluded from version
control by `.gitignore`.

---

## Notes and caveats

- **Legacy column names.** The raw clinical export uses abbreviated legacy headers (`bai`, `jigan`,
  `qvjia`, …). `R/01_boruta.R` and `R/02_lasso.R` map them to the analysis names through the
  `OLD2NEW` dictionary at the top of each file; the Python scripts consume the four standardised
  predictor columns directly. The headers themselves contain no patient identifiers.
- **Day-1 PEEP naming.** The source column is `peep` rather than `d1_peep`; both refer to the same
  day-1 measurement used in the paper.
- **Random Forest** was tuned over a deliberately narrow pool (`n_estimators` 100/300/400/600,
  `max_depth` 8/10, `min_samples_split` 2/5, `min_samples_leaf` 1/2); the selected configuration was
  100/8/2/2 with `max_features = 4`, `criterion = "gini"`, `bootstrap = True`.
- **File naming.** Scripts are deliberately not named `catboost.py`, `xgboost.py` or `lightgbm.py`:
  such a file sits first on `sys.path` when run directly and would shadow the library of the same
  name. Keep the current suffixes if you rename anything.
- **Fallback rule.** If a selection stage returns no variable, the Boruta-confirmed set is carried
  forward unchanged; if Boruta itself confirms nothing, all candidate variables are used. The branch
  taken is returned by `feature_selection.select_features` and counted by
  `nested_cv_selection.py`. In the reported 50-fold run, only one branch occurred: 35 folds needed
  no fallback and 15 used the Boruta-only set.
- Results are sensitive to library versions, particularly CatBoost and scikit-learn. The versions in
  `requirements.txt` are the ones used for the manuscript.

---

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this code, please cite the associated article and this archived release.

- **DOI (all versions, recommended):** https://doi.org/10.5281/zenodo.23938386
- **This release:** v1.0.3. The version-specific DOI for an individual release is listed on the
  Zenodo record page; the concept DOI above always resolves to the newest release.

BibTeX for the archived software:

```bibtex
@software{zhao_ards_72h_prognosis_2026,
  author    = {Zhao, Sheng},
  title     = {A 72-Hour Machine Learning Prognostic Model for {ARDS} Patients
               Requiring Invasive Mechanical Ventilation},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.3},
  doi       = {10.5281/zenodo.23938386},
  url       = {https://doi.org/10.5281/zenodo.23938386}
}
```

## Contact

Sheng Zhao — Department of Critical Care Medicine, Zhengzhou Central Hospital Affiliated to
Zhengzhou University. E-mail: 1640472493@qq.com
