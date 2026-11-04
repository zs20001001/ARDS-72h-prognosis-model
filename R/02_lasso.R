library(readxl)
library(glmnet)

BASE <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f, mustWork = FALSE)) else getwd()
})

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

f <- file.path(BASE, "..", "data", "training_set.xlsx")
df <- read_excel(f)

nm  <- names(df)
hit <- nm %in% names(OLD2NEW)
nm[hit] <- unname(OLD2NEW[nm[hit]])
names(df) <- nm

cat("dim:", dim(df), "\n")

xvars <- c("age", "sex", "hypertension", "diabetes", "chronic_cardiac_disease",
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

cat("n candidate vars:", length(xvars), "\n")

missing_cols <- setdiff(c(xvars, "outcome"), names(df))
cat("missing columns:", paste(missing_cols, collapse = ","), "\n")

dat <- df[, c(xvars, "outcome")]
dat[xvars] <- lapply(dat[xvars], as.numeric)
na_count <- sapply(dat, function(x) sum(is.na(x)))
cat("NA counts (nonzero):\n")
print(na_count[na_count > 0])

X <- as.matrix(dat[, xvars])
y <- as.integer(dat$outcome)
cat("y table:\n"); print(table(y))
cat("class of X columns:\n"); print(sapply(as.data.frame(X), class))

set.seed(500)
cv <- cv.glmnet(X, y, family = "binomial", alpha = 1, nlambda = 100, nfolds = 10,
                type.measure = "deviance", standardize = TRUE)
cat("lambda.min:", cv$lambda.min, " lambda.1se:", cv$lambda.1se, "\n")
cf <- coef(cv, s = "lambda.1se")
selected <- rownames(cf)[cf[, 1] != 0]
cat("selected at lambda.1se:\n"); print(selected)
print(cf)

cf2 <- coef(cv, s = "lambda.min")
cat("selected at lambda.min:\n"); print(rownames(cf2)[cf2[, 1] != 0])

d3k <- dat$d3_potassium
cat("\nd3_potassium summary:\n"); print(summary(d3k))
for (g in sort(unique(y))) {
  cat("group", g, "n=", sum(y == g), "\n")
  print(summary(d3k[y == g]))
}
cat("t.test:\n"); print(t.test(d3k ~ y))
cat("wilcox.test:\n"); print(wilcox.test(d3k ~ y))
