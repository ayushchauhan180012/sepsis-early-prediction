# Training Contract (FROZEN)

**Status: FROZEN — authoritative.** Approved Aug 2026.
This document defines the exact training/inference contract for the sepsis prediction
system. Do **not** change these definitions during implementation unless a future
explicit decision is made **and** documented in `DECISIONS.md`.

Companion docs: `AGENTS.md` (rules), `ARCHITECTURE.md` (system), `IMPLEMENTATION_PLAN.md` (roadmap).

---

## 1. Dataset

- Original training dataset: `baseline_dataset.csv` / original `baseline_dataset.psv`
  (pipe-delimited in the notebook; the verified copy is comma-delimited with identical
  content). Located **outside** the repository at `D:\sepsis_training_data\baseline_dataset.csv`.
- Shape: 790,215 rows × 14 columns; 20,336 patients; positive rate 1.1255%.

Columns (in file order):

```
HR, O2Sat, SBP, MAP, Resp, Temp, Lactate, WBC, Creatinine, Platelets,
Age, ICULOS, FutureSepsis, PatientID
```

## 2. Patient-Wise Train/Test Split

- **Patient-wise** split (no row leakage across patients).
- `test_size = 0.2`, `random_state = 42`, sklearn `train_test_split` with default shuffle.
- Split performed **before** any preprocessing.
- **16,268 training patients / 4,068 test patients** (n_test = ceil(0.2 × 20336)).
- Reproduced exactly with `np.random.RandomState(42)` permutation.

## 3. Frozen Vital Medians

Computed **after** the patient-wise split and **after** per-patient forward-fill,
exactly as in the original notebook (`train_df[vitals].median()` on the training split only).

| Vital | Frozen median |
|---|---|
| HR | **84.0** |
| O2Sat | **98.0** |
| SBP | **118.0** |
| MAP | **77.0** |
| Resp | **18.0** |
| Temp | **36.94** |

- These values **must NOT be recomputed during inference**.
- They are loaded from config/contract, never derived from live or cached data.
- Note: Temp's median is 36.94 (post-ffill), not 37.06 (raw NaN-skipped). The notebook
  computes medians after forward-fill; the ffill–then–median order is part of the contract.

## 4. Model Contract

The existing artifact `Backend/Model/hgb_sepsis_model.joblib` is **authoritative and frozen**.

- Estimator: `sklearn.ensemble.HistGradientBoostingClassifier`
  (`max_depth=6, learning_rate=0.1, max_iter=200, random_state=42`).
- Expects exactly **50 features**, in exactly this order (from `model.feature_names_in_`):

```
 0  HR
 1  O2Sat
 2  SBP
 3  MAP
 4  Resp
 5  Temp
 6  Lactate
 7  WBC
 8  Creatinine
 9  Platelets
10  Age
11  ICULOS
12  Lactate_missing
13  WBC_missing
14  Creatinine_missing
15  Platelets_missing
16  HR_delta6
17  O2Sat_delta6
18  SBP_delta6
19  MAP_delta6
20  Resp_delta6
21  Temp_delta6
22  HR_delta1
23  O2Sat_delta1
24  SBP_delta1
25  MAP_delta1
26  Resp_delta1
27  Temp_delta1
28  HR_roll6_std
29  O2Sat_roll6_std
30  SBP_roll6_std
31  MAP_roll6_std
32  Resp_roll6_std
33  Temp_roll6_std
34  HR_baseline_dev
35  O2Sat_baseline_dev
36  SBP_baseline_dev
37  MAP_baseline_dev
38  Resp_baseline_dev
39  Temp_baseline_dev
40  Lactate_recent_test
41  WBC_recent_test
42  Creatinine_recent_test
43  Platelets_recent_test
44  shock_index
45  resp_o2_ratio
46  map_hr_ratio
47  tachycardia
48  hypotension
49  tachypnea
```

- Classes: `[0, 1]`. `predict_proba(X)[:, 1]` is the sepsis risk probability.
- Inference must produce a feature vector with these exact names **in this exact order**.
  Because the model was fitted on a named DataFrame, feature-name validation applies.

## 5. Feature Semantics

### Vitals
`HR, O2Sat, SBP, MAP, Resp, Temp`

For each patient, chronologically:
1. **Forward-fill** vitals (per patient, chronological order).
2. Remaining leading NaNs are filled with the **frozen training medians** (§3).

### Labs
`Lactate, WBC, Creatinine, Platelets`

- Labs are **NOT imputed**.
- HGB receives remaining lab NaNs natively (missing-value handling).
- Missing indicators (`{lab}_missing`) are computed on raw lab values.

### Temporal Features (per vital, per patient, chronological)
- **delta1**: `current value − previous row`; unavailable first value → `0`.
- **delta6**: `current value − value 6 rows back`; unavailable value → `0`.
- **roll6_std**: `rolling(6, min_periods=1).std()`; NaN standard deviation → `0`.
- **baseline_dev**: `current imputed vital − mean of the FIRST SIX CHRONOLOGICAL
  STORED OBSERVATIONS` for that patient.

  **IMPORTANT:** This means FIRST SIX STORED OBSERVATIONS (rows), not ICU hours 1–6.
  The baseline is patient-specific, computed from the patient's own stored history,
  and is **not** a frozen global statistic. For patients with fewer than six stored
  rows, the baseline is the mean of the available rows (partial baseline). No fill is
  applied to `baseline_dev` (NaN is passed through; HGB handles it).

- **recent_test** (per lab): `not-null indicator` followed by
  `rolling(6, min_periods=1).max()`.

  **`recent_test` MUST use LABS, not vitals.**

### Additional Features (current row, on imputed vitals)
- `shock_index` = `HR / (SBP + 1)`
- `resp_o2_ratio` = `Resp / (O2Sat + 1)`
- `map_hr_ratio` = `MAP / (HR + 1)`
- `tachycardia` = `(HR > 100).astype(int)`
- `hypotension` = `(SBP < 90).astype(int)`
- `tachypnea` = `(Resp > 22).astype(int)`

## 6. Temporal Order

- Production **must explicitly sort** observations by `PatientID, ICULOS`
  **before** temporal feature computation and alert state updates.
- The original training file was **empirically verified** to already be ordered by
  `(PatientID, ICULOS)` (0 ICULOS decreases, 0 duplicates within patient), so explicit
  production sorting preserves training semantics exactly.
- Baseline = first six stored rows in chronological order (the training file's row order
  equals ICULOS order).

## 7. Alert Contract

Given the raw probability `p`:

1. `uncertain = (0.035 < p) & (p < 0.055)`  (strict inequalities; 0.035 and 0.055 are NOT uncertain)
2. If `uncertain`: `filtered_probability = 0`
3. `high_risk = filtered_probability >= 0.045`
4. `alert_raw = high_risk AND previous_high_risk`
5. **Persistence**: requires two consecutive high-risk observations.
6. **Cooldown**: an alert fires only when `ICULOS − last_alert_time >= 3`.
7. `last_alert_time` starts at **−999**.

Order of operations: **uncertainty filter → threshold → persistence → cooldown**.

- The **raw probability** must always be preserved separately from filtered probability
  and alert state.
- Alert state is per-patient and must **persist across requests**.

## 8. Verification Provenance

- Medians and split reproduced directly from `D:\sepsis_training_data\baseline_dataset.csv`
  following the notebook code exactly (verified: 16,268/4,068 patients; shape 790,215×14;
  positive rate 1.1255% — all matching notebook outputs).
- Temp=36.94 cross-validated against the evaluation notebook's `model_predictions.csv` head.
- 50-feature contract extracted directly from `model.feature_names_in_`.
- Alert parameters verified against `notebooks/early_sepsis_alert_system.ipynb` cell 12.

## 9. Change Control

Any deviation from this contract requires:

1. An explicit decision documented in `docs/DECISIONS.md`, and
2. A parity impact statement (what changes for inference vs. training).
