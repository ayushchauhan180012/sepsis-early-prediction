# Architecture

Status: **Context + current-state document** (updated Aug 2026).
The training/inference contract in `TRAINING_CONTRACT.md` is authoritative for correctness;
this document describes how the pieces fit together and what exists today.

---

## 1. Intended System Architecture

The intended direction of the project (for context, not a strict implementation spec):

```
ICU Monitor / Simulator
        ↓
FastAPI Ingestion (validation, patient identification)
        ↓
Real-Time Processing Pipeline
   Preprocessing → Feature Engineering → Model Inference → Risk Score → Alert Engine
        ↓
    ┌───┴───────────────┐
    ↓                   ↓
 Live Cache         Clinical Records
 (feature window)   (full raw history)
    ↓                   ↓
 Prediction          Alert Summaries
    ↓                   ↓
 Alerts              Analytics / Reports
    ↓
 Notifications (dashboard / email / SMS / pager)
    ↓
 Dashboard
```

Supporting layers (future):

- **Alert Summary DB** — summarized alert events (start, end, peak risk, duration,
  early-warning time, severity, acknowledgement status) for analytics/reporting.
- **Notification Service** — alert-triggered notifications. Conceptual; simulated backends only.
- **Dashboard / Frontend** — React dashboard (live monitoring, patient detail, reports). Deferred.
- **Automated Daily Reporting** — scheduled aggregation → PDF/CSV/dashboard data. Deferred.

## 2. Intended Responsibilities

| Component | Responsibility |
|---|---|
| Data Source | Hourly ICU observations: vitals, labs, demographics, PatientID, ICULOS |
| Ingestion Layer | FastAPI: receive observations, validate, identify patient, route into pipeline |
| Processing Pipeline | Preprocessing → feature engineering → model inference → risk score → alert engine |
| Prediction Cache / Live Data | Small sliding-window of recent observations for fast feature computation |
| Clinical Database | Permanent full patient history: demographics, raw observations, predictions, alerts, acknowledgements, outcomes |
| Alert Summary DB | Summarized alert events for analytics/reporting |
| Notification Service | Alert-triggered notifications (dashboard/email/SMS/pager) — conceptual |
| Dashboard / Frontend | Live monitoring + patient detail + reports — future |
| Automated Daily Reporting | Scheduled aggregation and report generation — future |

## 3. Current Implementation Status

### Existing
- **ML research** (notebooks): training, evaluation, baseline experiments. See `README.md` and `notebooks/`.
- **Model artifact**: `Backend/Model/hgb_sepsis_model.joblib` — frozen sklearn HGB, 50 features.
- **Health schema**: `Backend/Services/validation.py` — Pydantic `Health` model with realistic
  physiological bounds. Good, currently unused.
- **Synthetic simulator**: `Backend/Schema/sim_data.py` — hourly patient data generator
  (stable/stress/sepsis/recovering). Useful as a test data source.

### Broken / Scaffolding
- `Backend/app.py` — FastAPI stub; registers **no routes** (`app.get('/')` missing `@`).
- `Backend/Database/querry.py` — PostgreSQL helpers; `cursor = conn.cursor` (missing parens),
  hardcoded secrets, per-patient-table design, inconsistent DB names, `Update_Patient_Cache`
  targets a different table name than created.
- `Backend/Services/feature_engineering.py` — parity bugs: `{vital}_recent_test` instead of
  lab `_recent_test`, recomputes medians per request, broken `load_data` (`sql.Identifier({dict})`),
  placeholder DB credentials.
- `Backend/Services/pred_cache.py` — contradictory imports (unimportable), never calls the model,
  no risk scoring, returns raw `df.tail(1)`.

### Missing
- Model loading (sklearn not installed in either env), risk scoring, alert engine,
  prediction persistence, alert summaries, ingestion endpoints, tests, dependency manifest,
  `.gitignore`, config/secret management, Docker/CI, frontend, reporting.

## 4. Database Design (Logical)

A **single PostgreSQL database** with distinct logical responsibilities
(see `DECISIONS.md` decision D-007). The live "cache" is **derived** from the
full raw history — no separate 6-row cache (D-008).

| Logical role | Tables (proposed) | Responsibility |
|---|---|---|
| Clinical history | `patients`, `observations` | Demographics + full raw hourly observations |
| Feature state | derived from `observations` (bounded query window) | Input to feature engineering |
| Prediction history | `predictions` | Per-hour raw/filtered probability, high_risk, alert flags, ICULOS |
| Alert events | `alerts` | Alert start/end/duration/peak |
| Alert summaries | `alert_summaries` | Aggregated alert analytics |

## 5. Data Flow (Target)

1. Observation arrives at ingestion (validated by `Health`).
2. Observation is appended to the patient's raw history (`observations`), preserving ICULOS order.
3. Feature engineering replays the patient's chronological history and produces the 50
   features exactly per `TRAINING_CONTRACT.md`.
4. Model inference produces raw probability.
5. Alert engine applies the frozen alert contract (uncertainty band → threshold →
   persistence → cooldown) using persisted per-hour alert state.
6. `predictions` row persisted; alert events/summaries updated.

## 6. Divergences from the Intended Architecture (to be resolved by implementation)

1. Cache currently conflated with history (per-patient tables, 6-row limit).
2. No ordering guarantee in code — must explicitly sort by `PatientID, ICULOS`.
3. Training/inference feature parity not enforced by any shared spec.
4. No persisted alert state — engine must be stateful across requests.
5. No risk/prediction persistence.
6. No observability (logging, health checks, error handling).
