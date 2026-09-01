# AGENTS.md — Project Guidance for OpenCode Sessions

## Project Overview

Early Sepsis Prediction (6-hour ahead): a machine learning system that predicts the
risk of sepsis from ICU hourly time-series data, with a stateful alert engine.

- The **model** is a `HistGradientBoostingClassifier` (sklearn), serialized at
  `Backend/Model/hgb_sepsis_model.joblib`. It is **authoritative and frozen** — never retrain or replace it.
- The **training pipeline** is documented in `notebooks/early_sepsis_alert_system.ipynb`.
- The **frozen training/inference contract** (medians, 50-feature order, feature semantics,
  alert parameters) lives in **`docs/TRAINING_CONTRACT.md`** — it is the source of truth.

## Repository Layout

```
early_sepsis_prediction/
├── AGENTS.md                      ← this file
├── README.md                      ← research writeup / overview
├── .gitignore                     ← excludes venvs, .env, __pycache__, data files
├── .dockerignore                  ← excludes dev files from the Docker build context
├── requirements.txt               ← pinned deps (Phase 1)
├── .env.example                   ← env-specific config template (no secrets)
├── Dockerfile                     ← python:3.10-slim runtime image (Phase 9, D-029)
├── docker-compose.yml             ← local app + PostgreSQL orchestration (Phase 9, D-029)
├── .github/workflows/ci.yml       ← GitHub Actions CI: pytest + Docker build (Phase 9, D-030)
├── docs/
│   ├── ARCHITECTURE.md            ← intended + current architecture
│   ├── TRAINING_CONTRACT.md       ← FROZEN training/inference contract (authoritative)
│   ├── IMPLEMENTATION_PLAN.md     ← phased implementation roadmap
│   └── DECISIONS.md               ← decision log
├── notebooks/                     ← research: training, evaluation, baselines
│   ├── early_sepsis_alert_system.ipynb   ← main training notebook
│   ├── evaluation.ipynb
│   ├── sepsis_6h_prediction_baseline_experiments.ipynb   ← archived baselines
│   └── results/*.png
└── Backend/
    ├── app.py                     ← FastAPI entry: lifespan + routes (/health, /health/ready,
    │                                /predict, /patients/{id}/trajectory, /patients/{id}/alerts)
    ├── config.py                  ← tracked frozen contract values + Settings (Phase 1)
    ├── Model/hgb_sepsis_model.joblib  ← frozen model artifact
    ├── Database/
    │   ├── connection.py          ← SQLAlchemy engine / session factory + get_db dependency
    │   ├── schema.py              ← ORM models (patients, observations, predictions, alerts, alert_summaries)
    │   ├── operations.py          ← DB operations (CRUD, alert events/summaries, analytics queries)
    │   └── ddl.py                 ← create_all_tables / drop_all_tables bootstrap helpers
    ├── Services/
    │   ├── feature_engineering.py ← parity-verified 50-feature transform
    │   ├── pred_cache.py          ← pipeline orchestration (process_observation)
    │   ├── alert_engine.py        ← stateless recompute-from-history alert engine
    │   ├── notifications.py       ← notification channel abstraction (Phase 9, D-027)
    │   └── validation.py          ← Pydantic schemas (Health, PredictionResponse)
    └── Schema/sim_data.py         ← synthetic patient simulator (useful for tests)
```

## Non-Negotiable Rules

1. **Do not modify or retrain the model artifact** (`Backend/Model/hgb_sepsis_model.joblib`).
2. **Do not recompute training statistics during inference.** The six vital medians are frozen
   (see `docs/TRAINING_CONTRACT.md` §Frozen Medians) and must be loaded from config/contract, never
   recomputed from live data.
3. **Feature parity is mandatory.** Inference features must be produced in exactly the 50-feature
   order of `model.feature_names_in_`. Any change to feature definitions requires an explicit,
   documented decision.
4. **`recent_test` uses LAB columns** (`Lactate_recent_test`, ...), NOT vitals.
   `Services/feature_engineering.py` implements this correctly (the former `{vital}_recent_test`
   parity bug was fixed in Phase 3).
5. **Baseline semantics:** `baseline_dev` = current imputed vital − mean of the patient's
   **FIRST SIX STORED OBSERVATIONS** (chronological), NOT "ICU hours 1–6".
6. **Explicitly sort by `PatientID, ICULOS`** before temporal feature computation and alert
   state updates.
7. **Do not rely on a 6-row cache** for feature correctness. Retain full patient observation
   history (PostgreSQL is the intended store) and derive features from it.
8. **Labs are not imputed.** HGB handles remaining lab NaN natively.
9. **Raw probability must always be stored separately** from filtered probability and alert state.
10. **Do not over-engineer.** Single PostgreSQL database. No Redis, no multiple DBs, no
    microservices, no real notification providers, no React unless a later explicit decision is made.
11. **Preserve the existing project structure** unless there is a strong technical reason to change it.

## Environment / Commands

- Phase 1 (env/deps) is **complete**: the committed `Backend/myenv` was removed from the
  repository, `.gitignore` added, and pinned `requirements.txt` added (Python 3.10, pip).
- The frozen `scikit-learn==1.6.1` pin is load-compatibility-critical: the model pickle
  embeds `_sklearn_version=1.6.1` and a numpy>=2.0 array layout.
- Frozen contract values (vital medians, 50-feature order, alert params) live in the
  version-controlled `Backend/config.py` — not in `.env`. `.env` holds environment-specific
  values only (`DATABASE_URL`, credentials) and is git-ignored.
- To set up: `python -m venv Backend/myenv` then
  `Backend/myenv/Scripts/pip install -r requirements.txt` (or use your own venv).
- Common verification steps (read-only, no code changes):
  - Inspect the frozen contract: `docs/TRAINING_CONTRACT.md`
  - Read the decision log: `docs/DECISIONS.md`
  - Review phase status: `docs/IMPLEMENTATION_PLAN.md`

## Testing (Phase 7)

- Runner: `pytest` (pinned `pytest==9.1.1`). Config: `pytest.ini` at the repo root.
- Default run (no env vars needed, no DB, no training dataset):
  `python -m pytest` → **274 passed, 8 deselected**.
- The default `addopts = -m "not integration"` deselects the 8 PostgreSQL integration tests.
  They are opt-in and run ONLY when `TEST_DATABASE_URL` is explicitly set:
  `python -m pytest -m integration`. They never derive a test DB from `DATABASE_URL`, never
  touch the production/dev database (loud refusal if the URLs share a database), and skip
  cleanly when `TEST_DATABASE_URL` is absent or unreachable.
- Committed parity fixtures live in `tests/fixtures/` (regenerate ethically with
  `python tests/generate_fixtures.py --dataset D:/sepsis_training_data/baseline_dataset.csv`);
  the generator cross-checks production against the notebook and refuses to write on
  mismatch. Phase 3 and feature-parity tests skip the dataset-gated provenance check when
  the training CSV is absent.
- Tests must not modify production code. Never retrain or recompute frozen statistics during
  tests (D-003/D-023).

## Session Workflow

1. Read `docs/TRAINING_CONTRACT.md` first — it defines correctness for any implementation work.
2. Check `docs/DECISIONS.md` for already-made decisions before choosing alternatives.
3. Check `docs/IMPLEMENTATION_PLAN.md` for the current phase and its dependencies.
4. Follow the phases in dependency order; do not skip ahead.
5. Run correctness tests when available (Phase 7); until then, verify parity manually against the
   contract before considering a feature "done".

## Current Status (as of Sep 2026)

- **Phases 1–9 complete.** Model artifact frozen (D-001); training contract frozen and documented.
- Phases 2–7 delivered a single PostgreSQL DB, parity-verified feature engineering,
  a prediction pipeline (model loaded once, persist + alert recompute), a stateless alert
  engine, FastAPI ingestion endpoints, and the Phase 7 pytest suite (274 tests, committed
  parity fixtures, opt-in PostgreSQL integration tests).
- Phase 8 added analytics: risk-trajectory + peak-risk queries and
  `GET /patients/{id}/trajectory`, alert statistics from `alert_summaries` via
  `GET /patients/{id}/alerts`. Warning time and daily reporting deferred (D-025, D-026).
- Phase 9 added: notification channel abstraction with simulated NoOp/Console backends and
  background-task dispatch in `/predict` (D-027), the `GET /health/ready` readiness endpoint
  (D-028), Docker/Compose packaging (D-029), and GitHub Actions CI (D-030).
- See `docs/IMPLEMENTATION_PLAN.md` for the roadmap. React dashboard, real notification
  providers, cloud/Kubernetes deployment, and Alembic migrations remain deferred.
