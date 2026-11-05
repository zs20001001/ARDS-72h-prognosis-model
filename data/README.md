# Input data

The patient-level workbook is **not distributed** with this repository. De-identified data are
available from the corresponding author on reasonable request.

Place the file here as:

```
data/training_set.xlsx
```

`pd.read_excel()` / `readxl::read_excel()` read the **first worksheet**, with one row per patient.

---

## Columns required by the Python scripts

| Column | Type | Description |
| :----- | :--- | :---------- |
| `d3_a_ado2` | numeric | Alveolar–arterial oxygen gradient, day 3 (mmHg) |
| `peep` | numeric | Positive end-expiratory pressure, day 1 (cmH2O) |
| `d3_sofa` | numeric | SOFA score, day 3 |
| `d3_lac` | numeric | Lactate, day 3 (mmol/L) |
| `outcome` | 0 / 1 | Primary outcome: `0` = survivor, `1` = non-survivor |

No missing values are expected in these five columns: the R scripts call
`stopifnot(!anyNA(X))`, and the Python scripts convert with `pd.to_numeric` and will fail loudly on
non-numeric entries.

---

## Candidate predictor list

The resampling scripts re-run the selection inside each resample, so they need to know which
columns are candidates. Provide them as one name per line:

```
data/candidate_predictors.txt
```

If the file is absent, every column of the workbook except `outcome` is treated as a candidate,
which is only correct when the workbook holds the analysis variables alone. Point the scripts at
another location with `--candidates`.

---

## Columns required by the R scripts

`R/01_boruta.R` and `R/02_lasso.R` expect the **full** raw export: 69 candidate predictors plus
`outcome`. The raw headers are legacy abbreviations and are renamed to analysis names through the
`OLD2NEW` dictionary at the top of each script, for example:

| Raw header | Analysis name |
| :--------- | :------------ |
| `year` | `age` |
| `gental` | `sex` |
| `bai` | `d1_wbc` |
| `jigan` | `d1_creatinine` |
| `qvjia` | `d1_norepinephrine` |
| `d3_lac` | `d3_lactate` |
| `fuwowei_shifou` | `prone_position` |

The complete mapping is in `OLD2NEW` in both R scripts; the resulting analysis variable list is
`VARS` (Boruta) and `xvars` (LASSO). If your export already uses the analysis names, the mapping
step is a no-op and the scripts run unchanged.

---

## A note on the two naming schemes

`peep` in the source table is the **day-1** PEEP and is renamed `d1_peep` by the R scripts. The
Python scripts keep the original `peep` header so that they run against the unmodified clinical
export; the paper refers to this variable as D1 PEEP.
