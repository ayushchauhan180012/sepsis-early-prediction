# Architecture

Status: **Context + current-state document** (updated Sep 2026).
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
 Notifications (simulated NoOp/Console in Phase 9; real providers deferred)
    ↓
 Dashboard (deferred)
```

Supporting layers (future):

- **Alert Summary DB** — summarized alert events (start, end, peak risk, duration,
  early-warning time, severity, acknowledgement status) for analytics/reporting.
- **Notification Service** — alert-triggered notifications. Implemented with simulated backends
  only (`NoOp` default, `Console`); real providers deferred.
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

### Complete (Phases 0–9)
- **ML research** (notebooks): training, evaluation, baseline experiments. See `README.md` and `notebooks/`.
- **Model artifact**: `Backend/Model/hgb_sepsis_model.joblib` — frozen sklearn HGB, 50 features (D-001).
- **Training contract**: `docs/TRAINING_CONTRACT.md` — frozen, authoritative (Phase 0).
- **Environment / deps**: `.gitignore`, `requirements.txt`, `Backend/config.py` with tracked frozen
  contract values; `.env` for environment-specific settings only (Phase 1, D-018…D-020).
- **Database layer**: Single PostgreSQL DB with five tables (`patients`, `observations`, `predictions`,
  `alerts`, `alert_summaries`). SQLAlchemy 2.x + psycopg v3. UPSERT semantics for (patient_id, ICULOS)
  uniqueness. Full raw history retained; live cache derived from it (Phases 2–3, D-007…D-009).
- **Feature engineering**: Deterministic, parity-verified transform producing exactly the 50-feature
  row the frozen model expects. Frozen medians loaded from config, never recomputed (Phase 3, D-003…D-006).
- **Prediction pipeline**: Model loaded once at startup via lifespan; `process_observation` orchestrates
  ingest → features → inference → persist → alert recompute → alert events → alert summary (Phases 4–5).
- **Alert engine**: Stateless recompute-from-history, frozen alert contract (Phase 5, D-012, O-7).
- **FastAPI ingestion**: `POST /predict` with ICULOS ordering enforcement, `GET /health`, request IDs,
  structured error handling (Phase 6). Since Phase 9, when a prediction raises an alert, a FastAPI
  background task dispatches a notification; delivery failures are caught and logged and never
  affect the prediction response.
- **Notifications**: `Backend/Services/notifications.py` — `NotificationChannel` ABC with simulated
  `NoOpNotification` (default, does nothing) and `ConsoleNotification` (logs via the app logger),
  selected by the config-driven `get_notification_channel()` factory (`NOTIFICATION_CHANNEL`,
  D-027). No real providers.
- **Readiness endpoint**: `GET /health/ready` checks model availability and database connectivity
  (lightweight SELECT 1 on the existing engine/session) and returns 200 only when both are ok;
  otherwise 503 with the failed component reported as `"degraded"`. `GET /health` remains the
  unchanged liveness check (D-028).
- **Analytics endpoints**: `GET /patients/{id}/trajectory` (risk trajectory + peak risk),
  `GET /patients/{id}/alerts` (alert summary statistics) (Phase 8).
- **Test suite**: `pytest==9.1.1`, 274 passed / 8 deselected (default run). Committed parity fixtures,
  in-memory SQLite for unit/integration tests, opt-in PostgreSQL integration tests (Phase 7, D-022…D-024).
- **Docker / Compose**: root `Dockerfile` based on `python:3.10-slim`, installs the pinned
  `requirements.txt` and copies `Backend/` including the frozen model artifact; `docker-compose.yml`
  runs the app with PostgreSQL (`postgres:16-alpine`), a named volume, `pg_isready` health check and
  health-gated startup (D-029). Build only — no registry push.
- **CI**: `.github/workflows/ci.yml` — GitHub Actions running the pytest suite, then a Docker build
  job, on pushes/PRs to `main`; no deployment (D-030).
- **Health schema**: `Backend/Services/validation.py` — Pydantic `Health` model with realistic
  physiological bounds; `PredictionResponse` model for API responses.
- **Synthetic simulator**: `Backend/Schema/sim_data.py` — hourly patient data generator
  (stable/stress/sepsis/recovering). Useful as a test data source.

### Deferred / Not Implemented
- React dashboard / frontend (Phase 9).
- Real notification providers (email / SMS / Slack) — simulated channels only (D-027).
- Cloud / Kubernetes deployment (D-029).
- Alembic / schema migrations (see §7 Deployment Bootstrap).
- Daily reporting / scheduled aggregation (D-026).
- Warning time as a live metric (D-025; retrospective evaluation only).

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

## 6. Divergences from the Intended Architecture (resolved)

All Phase 0–7 divergences are resolved:
1. ✅ Full raw history retained; live cache derived from bounded query window (Phase 2, D-008).
2. ✅ Explicit sort by `PatientID, ICULOS` enforced in feature engineering and alert engine (Phase 3, D-009).
3. ✅ Feature parity enforced by single source of truth (`feature_engineering.py`) and committed test fixtures (Phases 3+7, D-004).
4. ✅ Alert state persisted in `predictions`; recomputed from history on each request (Phase 5, O-7).
5. ✅ Risk/prediction persistence in `predictions` table (Phase 4).
6. ✅ Observability: structured logging, health endpoint, request IDs, error handling (Phase 6).

### Remaining (beyond Phase 9)
- Frontend / dashboard (React, deferred).
- Real notification providers (email/SMS/Slack, deferred).
- Cloud / Kubernetes deployment (deferred).
- Schema migration tooling (Alembic) — see §7.
- Daily reporting / scheduled aggregation (deferred, D-026).

## 7. Deployment Bootstrap (known limitation)

A freshly created PostgreSQL database does **not** automatically receive the application schema.
The runtime does not run DDL at startup: `Backend/Database/ddl.py` provides `create_all_tables()`
(and `drop_all_tables()` for test teardown) but documents that it is not invoked at application
startup. Before database-backed prediction operations can run against a new database, the schema
must be bootstrapped once, e.g.:

```
python -c "from Backend.Database.ddl import create_all_tables; create_all_tables()"
```

This is a documented deployment/bootstrap requirement — not a defect in the prediction-core logic.
Alembic or equivalent migration tooling remains deferred (not introduced in Phase 9).
