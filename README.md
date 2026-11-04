# A 72-Hour Machine Learning Prognostic Model for ARDS Patients Requiring Invasive Mechanical Ventilation

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

The intersection of the Boruta and LASSO results gave **four predictors**, measured within the
first 72 h of ICU admission:

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
│   │   └── xgboost_grid_search.py
│   └── final_models/
│       ├── fit_catboost.py
│       ├── fit_decision_tree.py
│       ├── fit_gaussian_nb.py
│       ├── fit_lightgbm.py
│       ├── fit_logistic_regression.py
│       ├── fit_random_forest.py
│       ├── fit_svm.py
│       └── fit_xgboost.py
├── data/
│   └── README.md              # required input format (data itself is not distributed)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Data

Patient-level data are **not** included in this repository because of patient privacy and the terms
of the institutional ethics approval. De-identified data are available from the corresponding
author on reasonable request.

To run the code, place the input workbook at:

```
data/training_set.xlsx
```

The first worksheet must contain one row per patient with the columns listed in
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

---

## Outputs

| Script | Output |
| :----- | :----- |
| `R/01_boruta.R` | `R/results_boruta_report/` — `attStats.csv`, `attStats_roughfix.csv`, `imp_history.csv`, `imp_history.png`, `imp_boxplot.png`, `boruta_fit.rds` |
| `R/02_lasso.R` | console output: coefficients and selected variables at `lambda.min` / `lambda.1se` |
| `python/grid_search/*.py` | `*_grid_search.csv` (full ranking), `*_grid_search_best.json` or `top10.csv` (best configuration), plus per-fold AUCs |
| `python/final_models/*.py` | serialised model (`*.cbm`, `*.json`, `*.txt`, `*.joblib`) and console output with the fitted coefficients or tree counts |

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
- Results are sensitive to library versions, particularly CatBoost and scikit-learn. The versions in
  `requirements.txt` are the ones used for the manuscript.

---

## License

Released under the [MIT License](LICENSE).

## Contact

Sheng Zhao — Department of Critical Care Medicine, Zhengzhou Central Hospital Affiliated to
Zhengzhou University. E-mail: 1640472493@qq.com
