# Decision Log

Format: `D-###` | date | decision | status.
FROZEN decisions require a documented re-decision before they can change.

---

## Training Contract Decisions

### D-001 — Model artifact is frozen and authoritative
- **Date:** Aug 2026 · **Status:** FROZEN
- `Backend/Model/hgb_sepsis_model.joblib` (sklearn HistGradientBoostingClassifier) must not
  be retrained, replaced, or modified. It defines the 50-feature input contract.

### D-002 — Patient-wise split parameters
- **Date:** Aug 2026 · **Status:** FROZEN
- `train_test_split(patients, test_size=0.2, random_state=42)`; 16,268 train / 4,068 test
  patients; split performed before preprocessing.

### D-003 — Frozen vital medians
- **Date:** Aug 2026 · **Status:** FROZEN
- HR=84.0, O2Sat=98.0, SBP=118.0, MAP=77.0, Resp=18.0, Temp=36.94. Computed after split and
  after per-patient forward-fill on the training set only. Must be loaded from config, never
  recomputed during inference.

### D-004 — Single feature specification (single source of truth)
- **Date:** Aug 2026 · **Status:** FROZEN (direction); implement in Phase 3
- One controlled feature spec (the 50 names + semantics in TRAINING_CONTRACT §4–§5) drives
  both inference and any training-parity verification. No second, independent feature
  implementation.

### D-005 — `recent_test` uses lab columns
- **Date:** Aug 2026 · **Status:** FROZEN
- `{lab}_recent_test` (`Lactate`, `WBC`, `Creatinine`, `Platelets`). The current
  `feature_engineering.py` computes `{vital}_recent_test` — a known parity bug to be fixed
  in Phase 3.

### D-006 — Baseline = first six stored observations
- **Date:** Aug 2026 · **Status:** FROZEN
- `baseline_dev` uses the mean of the patient's **first six chronological stored rows**
  (per vital), not ICU hours 1–6. Patient-specific, computed live from history, not a
  global frozen statistic. Partial baseline for patients with <6 stored rows.

### D-007 — Single PostgreSQL database
- **Date:** Aug 2026 · **Status:** FROZEN (direction)
- One database; distinct logical tables for history, predictions, alerts, summaries.
  No Redis, no multiple DBs, no microservices. Revisit only with explicit justification.

### D-008 — Full raw history is the feature source (no 6-row cache)
- **Date:** Aug 2026 · **Status:** FROZEN (direction)
- Retain full per-patient raw observation history in PostgreSQL and derive feature state
  from it. A 6-row cache is insufficient for delta6/baseline_dev/ffill correctness.
  The "live cache" is a bounded query over history, keeping cache/history responsibilities
  logically separate.

### D-009 — Explicit chronological sort
- **Date:** Aug 2026 · **Status:** FROZEN
- Production must sort by `PatientID, ICULOS` before temporal feature computation and alert
  state updates. The training file was verified already ordered, so this preserves semantics.

### D-010 — Labs not imputed
- **Date:** Aug 2026 · **Status:** FROZEN
- Labs remain NaN; HGB handles missing values natively. Only missing indicators are added.

### D-011 — Raw probability stored separately
- **Date:** Aug 2026 · **Status:** FROZEN
- Raw `predict_proba` probability persisted separately from filtered probability and alert
  state (needed for trajectories, peak risk, analytics).

### D-012 — Alert contract values
- **Date:** Aug 2026 · **Status:** FROZEN
- Uncertainty band `0.035 < p < 0.055` (strict), threshold `0.045`, persistence 2 consecutive
  hours, cooldown 3 hours, `last_alert_time` starts at −999. Order: uncertainty filter →
  threshold → persistence → cooldown. Values must not be changed without a documented decision.

## Architecture Decisions

### D-013 — Alert engine is stateful and persisted
- **Date:** Aug 2026 · **Status:** Direction
- Alert state (previous high_risk, last_alert_time) must persist across requests; derived
  from persisted per-hour flags in the predictions history.

### D-014 — Predictions persisted with patient + ICULOS
- **Date:** Aug 2026 · **Status:** Direction
- Every hourly prediction is stored with PatientID and ICULOS so risk trajectory, peak risk,
  alert history, warning time, and duration can be reconstructed.

### D-015 — Preserve existing project structure
- **Date:** Aug 2026 · **Status:** Direction
- Keep `Backend/{app.py, Database/, Services/, Schema/, Model/}` layout and the research
  notebooks. Finish the existing design rather than replacing it.

### D-016 — Notifications, dashboard, Docker, CI deferred
- **Date:** Aug 2026 · **Status:** Direction
- Notification providers, React dashboard, Docker, CI, cloud are explicitly deferred until
  the core prediction + alert system is logically correct and tested locally.

### D-017 — Alert event = maximal contiguous run
- **Date:** Aug 2026 · **Status:** Direction (confirm in Phase 5)
- An alert "event" is a maximal contiguous run of hours with `alert=1` per patient; peak
  risk = max raw probability in the run; duration = end − start + 1; early-warning time is
  derived retrospectively (needs outcome label).

## Phase 1 Decisions

### D-018 — Python 3.10 + pip + requirements.txt
- **Date:** Aug 2026 · **Status:** Confirmed
- Pin Python 3.10.11 and use `pip` with a pinned `requirements.txt` (no `pyproject.toml`).
- `scikit-learn==1.6.1` is a load-compatibility pin: the model pickle embeds
  `_sklearn_version=1.6.1` and a numpy>=2.0 array layout. `requirements.txt` added at repo root.

### D-019 — SQLAlchemy 2.x + psycopg v3 for the DB layer
- **Date:** Aug 2026 · **Status:** Confirmed
- DB layer (Phase 2) uses `SQLAlchemy==2.0.51` with `psycopg[binary]==3.3.4`.
- SQLAlchemy was already imported by the existing code, so this avoids introducing a new stack.

### D-020 — Frozen contract values are tracked config, not .env secrets
- **Date:** Aug 2026 · **Status:** Confirmed
- Frozen vital medians, 50-feature names, feature constants, and alert parameters live in the
  version-controlled `Backend/config.py` (sourced from `TRAINING_CONTRACT.md`).
- `.env` is reserved for environment-specific values only (`DATABASE_URL`, credentials,
  deployment-specific overrides) and is git-ignored. No secrets in code.

## Phase 7 Decisions

### D-022 — Lowercase PostgreSQL column naming
- **Date:** Aug 2026 · **Status:** Confirmed
- Database columns use PostgreSQL lowercase convention (`hr`, `o2sat`, `iculos`, …), and the
  training-contract uppercase names are mapped in Python (`Backend/Database/operations.py`,
  `observations_to_dataframe`). No D-021 was recorded — the codebase referenced only D-022
  (and no earlier D-021 evidence exists), so D-021 is intentionally left unused.

### D-023 — Test framework: pytest + committed fixtures (resolves O-8)
- **Date:** Aug 2026 · **Status:** Confirmed
- `pytest==9.1.1` (requires Python ≥3.10; verified against the D-018 pin of 3.10.11).
- Committed, deterministic fixtures in `tests/fixtures/`:
  - `feature_parity.csv` — raw observations + expected 50-feature rows for real patients
    (regression pin for `Services/feature_engineering.py`, valid without the training data).
  - `alert_scenarios.json` — probability streams + notebook cell-12 expected states
    (synthetic boundary/persistence/cooldown cases plus real-patient streams).
- Fixtures are produced by `tests/generate_fixtures.py`, which cross-checks the production
  transform against a verbatim reproduction of notebook cells 6/8/12 and **refuses to write
  on mismatch**, so a committed fixture is an independent record of the notebook output.
- Fixture inputs are pre-normalized; the pipeline path is covered by in-memory SQLite
  (Phases 3–6). No training dataset or database needed for the default CI run.

### D-024 — Confirmations for O-3…O-7
- **Date:** Aug 2026 · **Status:** Confirmed
- **O-3 (schema DDL):** five tables, lowercase columns, `UniqueConstraint(patient_id, iculos)`
  on observations & predictions, FKs → `patients.patient_id`, per-table indexes
  (`Backend/Database/schema.py`).
- **O-4 (out-of-order / duplicate ICULOS):** UPSERT semantics
  (`on_conflict_do_update`) for observations and predictions; reads are always
  ICULOS-ASC ordered (D-009).
- **O-5 (full-history replay vs incremental):** full raw observation history is the
  authoritative feature source (D-008); features are recomputed from it, never from a
  6-row cache.
- **O-6 (pre-baseline rows):** partial baseline (mean of available first rows, min 1) and no
  NaN passthrough for vitals (ffill + frozen medians); labs stay NaN (D-010).
- **O-7 (alert engine: recompute vs incremental):** stateless recompute-from-history
  (`Backend/Services/alert_engine.evaluate_alert_state`), pure/deterministic, verified
  against notebook cell 12.

---

## Phase 8 Decisions

### D-025 — Warning time deferred as a live metric
- **Date:** Aug 2026 · **Status:** Confirmed
- True warning time requires actual sepsis onset/outcome data, which the production
  pipeline does not persist. Deferred as a retrospective evaluation metric only.
  Phase 8 does not compute or expose warning time.

### D-026 — Daily reporting deferred
- **Date:** Aug 2026 · **Status:** Confirmed
- Daily report endpoint or scheduler is not implemented in Phase 8. The analytics
  query functions and API endpoints provide the data surface a future report would
  consume. No schedule or aggregation endpoint until explicitly requested.

## Open Items (to be decided during implementation)

| # | Open decision | Relevant phase |
|---|---|---|
| ~~O-1~~ | ~~Python version pin and package manager~~ → **D-018** (3.10.11, pip, requirements.txt) | 1 ✅ |
| ~~O-2~~ | ~~SQLAlchemy vs raw psycopg~~ → **D-019** (SQLAlchemy 2.x + psycopg v3) | 1 ✅ |
| ~~O-3~~ | ~~Exact schema DDL (columns, constraints, indexes)~~ → **D-024** | 2 ✅ |
| ~~O-4~~ | ~~Out-of-order / duplicate ICULOS policy (reject vs upsert)~~ → **D-024** (UPSERT) | 2 ✅ |
| ~~O-5~~ | ~~Full-history replay vs incremental feature state machine~~ → **D-024** (full replay) | 3 ✅ |
| ~~O-6~~ | ~~Pre-baseline rows: partial baseline + NaN passthrough semantics~~ → **D-024** | 3 ✅ |
| ~~O-7~~ | ~~Alert engine: recompute-from-history vs incremental state~~ → **D-024** (recompute) | 5 ✅ |
| ~~O-8~~ | ~~Test framework and fixture strategy~~ → **D-023** (pytest + committed fixtures) | 7 ✅ |
| ~~O-9~~ | ~~Warning time definition~~ → **D-025** (deferred; requires outcome labels) | 8 ✅ |
| ~~O-10~~ | ~~Daily report scope~~ → **D-026** (deferred; no scheduler/endpoint) | 8 ✅ |

## Change Log

- **Aug 2026** — Training contract frozen and documented. D-001…D-012 recorded.
- **Aug 2026** — Phase 1 (env/deps) executed. D-018…D-020 recorded.
- **Aug 2026** — Phase 7 (test suite completion) executed. D-022…D-024 recorded; all open
  items O-3…O-8 resolved. pytest, pytest.ini, shared conftest fixtures, committed parity
  fixtures, validation/DB/feature-parity/alert-scenario tests, opt-in PostgreSQL integration
  tests, and the phase-3 frozen-median provenance test added.
- **Aug 2026** — Phase 8 (analytics/reporting) executed. D-025…D-026 recorded; warning time
  and daily reporting deferred. Risk-trajectory, peak-risk, and alert-statistics queries
  added; GET API endpoints for patient trajectory and alert summaries.
