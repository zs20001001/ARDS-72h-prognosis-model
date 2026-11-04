suppressPackageStartupMessages(library(Boruta))
suppressPackageStartupMessages(library(readxl))

BASE <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f, mustWork = FALSE)) else getwd()
})
OUT <- file.path(BASE, "results_boruta_report")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

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

df <- as.data.frame(read_excel(file.path(BASE, "..", "data", "training_set.xlsx")))
nm  <- names(df)
hit <- nm %in% names(OLD2NEW)
nm[hit] <- unname(OLD2NEW[nm[hit]])
names(df) <- nm

X <- df[, VARS]
X[] <- lapply(X, as.numeric)
y <- factor(df$outcome)
stopifnot(nrow(X) == length(y), !anyNA(X), !anyNA(y))
cat(sprintf("n = %d, p = %d, outcome levels: %s\n",
            nrow(X), ncol(X), paste(levels(y), collapse = " / ")))

set.seed(500)
t0 <- Sys.time()
bz <- Boruta(X, y,
             pValue  = 0.01,
             mcAdj   = TRUE,
             maxRuns = 100,
             ntree   = 500,
             doTrace = 0)
cat(sprintf("Boruta finished: %d runs, %.1f sec\n",
            nrow(bz$ImpHistory),
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

st  <- attStats(bz)
dec <- bz$finalDecision
cat(sprintf("decisions: Confirmed %d / Tentative %d / Rejected %d\n",
            sum(dec == "Confirmed"), sum(dec == "Tentative"),
            sum(dec == "Rejected")))
cat("Confirmed:", paste(rownames(st)[st$decision == "Confirmed"],
                        collapse = ", "), "\n")

bz2 <- TentativeRoughFix(bz)
st2 <- attStats(bz2)
cat(sprintf("after TentativeRoughFix: Confirmed %d / Rejected %d\n",
            sum(st2$decision == "Confirmed"), sum(st2$decision == "Rejected")))

st$VariableName  <- rownames(st)
st2$VariableName <- rownames(st2)
write.csv(st[, c("VariableName", "meanImp", "medianImp", "minImp", "maxImp",
                 "normHits", "decision")],
          file.path(OUT, "attStats.csv"), row.names = FALSE, quote = FALSE)
write.csv(st2[, c("VariableName", "meanImp", "medianImp", "minImp", "maxImp",
                  "normHits", "decision")],
          file.path(OUT, "attStats_roughfix.csv"), row.names = FALSE, quote = FALSE)
saveRDS(bz, file.path(OUT, "boruta_fit.rds"))

write.csv(bz$ImpHistory, file.path(OUT, "imp_history.csv"),
          row.names = FALSE, quote = FALSE)

png(file.path(OUT, "imp_history.png"), width = 1600, height = 900, res = 150)
plotImpHistory(bz, xlab = "Run", ylab = "Importance (Z-score)")
dev.off()

imp  <- bz$ImpHistory
keep <- !grepl("^shadow", colnames(imp))
imp  <- imp[, keep, drop = FALSE]
imp  <- imp[, order(colMeans(imp)), drop = FALSE]
png(file.path(OUT, "imp_boxplot.png"), width = 1500, height = 1800, res = 150)
par(mar = c(4.5, 8.5, 3, 1))
boxplot(as.data.frame(imp), horizontal = TRUE, las = 1, cex.axis = 0.62,
        outline = FALSE, xlab = "Importance (Z-score)",
        main = "Boruta importance per variable (ntree = 500, maxRuns = 100)")
abline(v = median(bz$ImpHistory[, "shadowMax"]), col = "red", lty = 2)
mtext("red dashed line = median shadowMax", side = 3, line = 0.2,
      cex = 0.7, col = "red")
dev.off()

cat("output directory:", OUT, "\n")
