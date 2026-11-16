# -*- coding: utf-8 -*-
# Nested cross-validation of the modelling procedure, implemented in R.
#
# This is the R counterpart of python/resampling/nested_cv_selection.py and the
# script referred to in the response to comment 6 of the third review: the
# selectors used for the reported results are Boruta and glmnet, both in R, so
# the resampling procedure that re-runs them inside every training fold is
# supplied in the same language. The Python module re-implements the same rule on
# BorutaPy and scikit-learn and is labelled as an approximate independent
# reimplementation in the README; the two are not expected to return identical
# variable sets in every fold.
#
# Protocol (Table S14, row 7): stratified outer 10-fold cross-validation repeated
# five times, seeds 42 + r. Selection runs on the 256 patients of the training
# fold only. Fallback (row 8): empty intersection -> Boruta set; empty Boruta ->
# all candidates. Two arms are produced, matching Supplementary Table S4f:
#
#   reselect  Boruta + LASSO re-run inside every training fold
#   fixed     the four predictors of the final model, unchanged in every fold
#
# Output (in R/results_nested_cv/):
#   nested_cv.csv      one row per arm x repeat x fold
#   nested_cv_oof.csv  one row per arm x repeat x held-out patient
#   nested_cv_summary.csv
#
# The OOF file has the same columns as the Python script's, so
# python/resampling/reproduce_table_s4f.py can be run on either output.
#
# Requirements: R 4.3.3, Boruta, glmnet, readxl, catboost (>= 0.2).
# Usage:
#   Rscript 03_nested_cv_selection.R [--repeats=5] [--splits=10] [--trees=500] [--maxruns=100]

suppressPackageStartupMessages(library(readxl))
suppressPackageStartupMessages(library(Boruta))
suppressPackageStartupMessages(library(glmnet))
suppressPackageStartupMessages(library(catboost))

BASE <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f, mustWork = FALSE)) else getwd()
})
OUT <- file.path(BASE, "results_nested_cv")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

arg <- function(name, default) {
  a <- commandArgs(trailingOnly = TRUE)
  hit <- grep(paste0("^--", name, "="), a, value = TRUE)
  if (length(hit)) as.numeric(sub("^[^=]+=", "", hit[1])) else default
}
N_REPEATS <- arg("repeats", 5)
N_SPLITS  <- arg("splits", 10)
N_TREES   <- arg("trees", 500)
MAX_RUNS  <- arg("maxruns", 100)
SEED0     <- 42

# Source column names -> the names used throughout the R scripts.
OLD2NEW <- c(
  year = "age", gental = "sex", gaoxueya = "hypertension",
  tangniaobing = "diabetes", xinzangbing = "chronic_cardiac_disease",
  shenbing = "chronic_kidney_disease", ganbing = "chronic_liver_disease",
  mianyi = "immune_system_disease", huxibing = "chronic_pulmonary_disease",
  aki = "aki", youyin = "inducing_factor",
  bai = "d1_wbc", birong = "d1_hematocrit", plt = "d1_platelet",
  zongdan = "d1_total_bilirubin", jigan = "d1_creatinine", ph = "d1_ph",
  pco2 = "d1_paco2", a_ado2 = "d1_aado2", lac = "d1_lactate",
  na = "d1_sodium", k = "d1_potassium", duobafendingan = "d1_dobutamine",
  t = "d1_temperature", hr = "d1_heart_rate", peep = "d1_peep",
  f = "d1_respiratory_rate", vt = "d1_tidal_volume", d1_p_f = "d1_pao2_fio2",
  d3_p_f = "d3_pao2_fio2", change_p_f = "delta_pao2_fio2",
  crrt = "crrt", ecmo = "ecmo", map = "d1_map",
  duobaan = "d1_dopamine", shenshang = "d1_epinephrine",
  qvjia = "d1_norepinephrine",
  d3_bai = "d3_wbc", d3_birong = "d3_hematocrit", d3_plt = "d3_platelet",
  d3_zongdan = "d3_total_bilirubin", d3_jigan = "d3_creatinine",
  d3_map = "d3_map", d3_duobaan = "d3_dopamine",
  d3_duobafendingan = "d3_dobutamine", d3_shenshang = "d3_epinephrine",
  d3_qvjia = "d3_norepinephrine", d3_ph = "d3_ph", d3_pco2 = "d3_paco2",
  d3_a_ado2 = "d3_aado2", d3_lac = "d3_lactate", d3_na = "d3_sodium",
  d3_k = "d3_potassium",
  d3_t = "d3_temperature", d3_hr = "d3_heart_rate", d3_PEEP = "d3_peep",
  d3_f = "d3_respiratory_rate", d3_VT = "d3_tidal_volume",
  d1_baidanbai = "d1_albumin", d1_pt = "d1_pt",
  d3_baidanbai = "d3_albumin", d3_pt = "d3_pt",
  d3_apacheii = "d3_apache_ii", d3_sofa = "d3_sofa",
  apacheii = "d1_apache_ii", sofa = "d1_sofa",
  change_apacheii = "delta_apache_ii", change_sofa = "delta_sofa",
  fuwowei_shifou = "prone_position"
)

VARS <- c("age", "sex", "hypertension", "diabetes", "chronic_cardiac_disease",
          "chronic_kidney_disease", "chronic_liver_disease", "immune_system_disease",
          "chronic_pulmonary_disease", "aki", "inducing_factor", "d1_wbc",
          "d1_hematocrit", "d1_platelet", "d1_total_bilirubin", "d1_creatinine",
          "d1_ph", "d1_paco2", "d1_aado2", "d1_lactate", "d1_sodium",
          "d1_potassium", "d1_dobutamine", "d3_potassium", "d1_temperature",
          "d1_heart_rate", "d1_peep", "d1_respiratory_rate", "d1_tidal_volume",
          "d1_pao2_fio2", "d3_pao2_fio2", "delta_pao2_fio2", "crrt", "ecmo",
          "d1_map", "d1_dopamine", "d1_epinephrine", "d1_norepinephrine",
          "d3_wbc", "d3_hematocrit", "d3_platelet", "d3_total_bilirubin",
          "d3_creatinine", "d3_map", "d3_dopamine", "d3_dobutamine",
          "d3_epinephrine", "d3_norepinephrine", "d3_ph", "d3_paco2",
          "d3_aado2", "d3_lactate", "d3_sodium", "d3_temperature",
          "d3_heart_rate", "d3_peep", "d3_respiratory_rate", "d3_tidal_volume",
          "d1_albumin", "d1_pt", "d3_albumin", "d3_pt", "d3_apache_ii",
          "d3_sofa", "d1_apache_ii", "d1_sofa", "delta_apache_ii",
          "delta_sofa", "prone_position")

FIXED <- c("d3_aado2", "d1_peep", "d3_sofa", "d3_lactate")

df <- as.data.frame(read_excel(file.path(BASE, "..", "data", "training set.xlsx")))
nm <- names(df)
hit <- nm %in% names(OLD2NEW)
nm[hit] <- unname(OLD2NEW[nm[hit]])
names(df) <- nm
stopifnot(all(FIXED %in% names(df)), all(VARS %in% names(df)), "outcome" %in% names(df))

X <- as.matrix(df[, VARS])
storage.mode(X) <- "double"
y <- as.integer(df$outcome)
stopifnot(!anyNA(X), !anyNA(y))

# One row per patient; the row index is the identifier when the workbook has none.
PID <- if ("patient_id" %in% names(df)) as.character(df$patient_id) else
  sprintf("row%d", seq_len(nrow(df)) - 1L)

# ---- stratified folds ------------------------------------------------------
stratified_folds <- function(y, k, seed) {
  set.seed(seed)
  fold <- integer(length(y))
  for (cls in sort(unique(y))) {
    idx <- which(y == cls)
    idx <- sample(idx)
    fold[idx] <- rep(seq_len(k), length.out = length(idx))
  }
  fold
}

# ---- selectors -------------------------------------------------------------
boruta_confirmed <- function(Xtr, ytr, seed) {
  set.seed(seed)
  bz <- Boruta(Xtr, factor(ytr), pValue = 0.01, mcAdj = TRUE,
               maxRuns = MAX_RUNS, ntree = N_TREES, doTrace = 0)
  getSelectedAttributes(bz, withTentative = FALSE)
}

lasso_1se <- function(Xtr, ytr, seed) {
  set.seed(seed)
  fit <- cv.glmnet(Xtr, ytr, family = "binomial", alpha = 1,
                   nlambda = 100, nfolds = 10, type.measure = "deviance")
  co <- as.matrix(coef(fit, s = "lambda.1se"))
  keep <- rownames(co)[co[, 1] != 0]
  setdiff(keep, "(Intercept)")
}

select_features <- function(Xtr, ytr, seed) {
  bor <- boruta_confirmed(Xtr, ytr, seed)
  las <- lasso_1se(Xtr, ytr, seed)
  both <- intersect(bor, las)
  if (length(both)) return(list(cols = both, branch = "both",
                                n_boruta = length(bor), n_lasso = length(las)))
  if (length(bor)) return(list(cols = bor, branch = "boruta_only",
                               n_boruta = length(bor), n_lasso = length(las)))
  if (length(las)) return(list(cols = las, branch = "lasso_only",
                               n_boruta = length(bor), n_lasso = length(las)))
  list(cols = colnames(Xtr), branch = "all_vars",
       n_boruta = length(bor), n_lasso = length(las))
}

# ---- CatBoost, the frozen configuration -----------------------------------
CAT_PARAMS <- list(
  iterations = 200, depth = 6, learning_rate = 0.06, l2_leaf_reg = 120,
  random_strength = 6, border_count = 96, loss_function = "Logloss",
  eval_metric = "AUC", random_seed = SEED0, verbose = 0, thread_count = 1
)

fit_predict <- function(Xtr, ytr, Xte) {
  pool <- catboost.load_pool(data = as.data.frame(Xtr), label = ytr)
  model <- catboost.train(pool, params = CAT_PARAMS)
  catboost.predict(model, catboost.load_pool(data = as.data.frame(Xte)),
                   prediction_type = "Probability")
}

# AUC without an extra dependency: rank-based and tie-corrected.
auc_rank <- function(y, p) {
  r <- rank(p)
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# ---- loop ------------------------------------------------------------------
fold_rows <- list()
oof_rows <- list()
for (r in seq_len(N_REPEATS)) {
  seed <- SEED0 + r - 1L
  fold <- stratified_folds(y, N_SPLITS, seed)
  for (k in seq_len(N_SPLITS)) {
    te <- which(fold == k)
    tr <- which(fold != k)
    sel <- select_features(X[tr, , drop = FALSE], y[tr], seed)
    arms <- list(reselect = list(cols = sel$cols, branch = sel$branch,
                                 nb = sel$n_boruta, nl = sel$n_lasso),
                 fixed = list(cols = FIXED, branch = "fixed",
                              nb = length(FIXED), nl = length(FIXED)))
    for (arm in names(arms)) {
      cols <- arms[[arm]]$cols
      p <- fit_predict(X[tr, cols, drop = FALSE], y[tr], X[te, cols, drop = FALSE])
      fold_rows[[length(fold_rows) + 1L]] <- data.frame(
        arm = arm, repeat = r, fold = k, model = "CatBoost",
        n_train = length(tr), n_test = length(te), n_predictors = length(cols),
        branch = arms[[arm]]$branch, n_boruta = arms[[arm]]$nb,
        n_lasso = arms[[arm]]$nl,
        auroc = auc_rank(y[te], as.numeric(p))
      )
      oof_rows[[length(oof_rows) + 1L]] <- data.frame(
        arm = arm, model = "CatBoost", repeat = r, fold = k,
        patient_id = PID[te], y = y[te], p = as.numeric(p)
      )
    }
    cat(sprintf("repeat %d/%d fold %d/%d: reselect %d vars (%s), fixed %d vars\n",
                r, N_REPEATS, k, N_SPLITS, length(sel$cols), sel$branch,
                length(FIXED)))
  }
}

per_fold <- do.call(rbind, fold_rows)
oof <- do.call(rbind, oof_rows)

summary_tbl <- aggregate(auroc ~ arm + model, data = per_fold,
                         FUN = function(v) c(mean = mean(v), sd = sd(v), n = length(v)))
summary_tbl <- data.frame(arm = summary_tbl$arm, model = summary_tbl$model,
                          cv_auroc = summary_tbl$auroc[, "mean"],
                          sd = summary_tbl$auroc[, "sd"],
                          n = summary_tbl$auroc[, "n"])

write.csv(per_fold, file.path(OUT, "nested_cv.csv"), row.names = FALSE)
write.csv(oof, file.path(OUT, "nested_cv_oof.csv"), row.names = FALSE)
write.csv(summary_tbl, file.path(OUT, "nested_cv_summary.csv"), row.names = FALSE)

print(summary_tbl, row.names = FALSE)
cat("\nfiles:", OUT, "\n")
cat("next: python ../python/resampling/reproduce_table_s4f.py ",
    "--oof ", file.path(OUT, "nested_cv_oof.csv"), " --model CatBoost\n", sep = "")
