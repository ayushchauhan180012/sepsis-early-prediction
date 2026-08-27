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
├── requirements.txt               ← pinned deps (Phase 1)
├── .env.example                   ← env-specific config template (no secrets)
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
    ├── app.py                     ← FastAPI entry (currently a stub, no routes registered)
    ├── config.py                  ← tracked settings: frozen contract values + Settings (Phase 1)
    ├── Model/hgb_sepsis_model.joblib  ← frozen model artifact
    ├── Database/querry.py         ← DB helpers (currently broken / needs rewrite)
    ├── Services/
    │   ├── feature_engineering.py ← preprocessing/features (currently has parity bugs)
    │   ├── pred_cache.py          ← pipeline orchestration (currently unimportable)
    │   └── validation.py          ← Pydantic Health schema (keep)
    └── Schema/sim_data.py         ← synthetic patient simulator (keep, useful for tests)
```

## Non-Negotiable Rules

1. **Do not modify or retrain the model artifact** (`Backend/Model/hgb_sepsis_model.joblib`).
2. **Do not recompute training statistics during inference.** The six vital medians are frozen
   (see `docs/TRAINING_CONTRACT.md` §Frozen Medians) and must be loaded from config/contract, never
   recomputed from live data.
3. **Feature parity is mandatory.** Inference features must be produced in exactly the 50-feature
   order of `model.feature_names_in_`. Any change to feature definitions requires an explicit,
   documented decision.
4. **`recent_test` uses LAB columns** (`Lactate_recent_test`, ...), NOT vitals. The current
   `Services/feature_engineering.py` computes `{vital}_recent_test` — this is a known parity bug to fix.
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
  `python -m pytest` → **215 passed, 8 deselected**.
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

## Current Status (as of Aug 2026)

- Phase 1 (local environment / dependencies) **complete**: committed `myenv` removed,
  `.gitignore` + pinned `requirements.txt` added, `Backend/config.py` holds tracked frozen
  contract values, `.env` (git-ignored) holds environment-specific settings. Fresh-venv
  install verified: model loads with `scikit-learn==1.6.1`.
- ML research complete (ROC-AUC ≈ 0.756; patient-level precision ≈ 0.62 / recall ≈ 0.42).
- Training contract frozen and documented.
- Backend is scaffolding only: `app.py` registers no routes; DB helpers, feature engineering,
  and the prediction cache are broken/not wired; no alert engine, no tests.
- See `docs/IMPLEMENTATION_PLAN.md` for the roadmap. Phase 2 (database) is next.
