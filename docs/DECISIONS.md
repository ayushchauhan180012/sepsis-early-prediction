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

---

## Open Items (to be decided during implementation)

| # | Open decision | Relevant phase |
|---|---|---|
| O-1 | Python version pin and package manager (pip/requirements vs pyproject) | 1 |
| O-2 | SQLAlchemy vs raw psycopg for the DB layer | 1 |
| O-3 | Exact schema DDL (columns, constraints, indexes) | 2 |
| O-4 | Out-of-order / duplicate ICULOS policy (reject vs upsert) | 2 |
| O-5 | Full-history replay vs incremental feature state machine | 3 |
| O-6 | Pre-baseline rows: partial baseline + NaN passthrough semantics | 3 |
| O-7 | Alert engine: recompute-from-history vs incremental state | 5 |
| O-8 | Test framework and fixture strategy | 7 |

## Change Log

- **Aug 2026** — Training contract frozen and documented. D-001…D-012 recorded.
