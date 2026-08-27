"""Phase 7 — schema/constraint and CRUD tests on in-memory SQLite.

Covers ``Backend.Database.schema`` and ``Backend.Database.operations`` at the
ORM level: UPSERT semantics (O-4), ICULOS-ordered reads (D-009), FK
enforcement, NOT NULL constraints, and alert event/summary rebuild helpers.

Uses the shared ``db_session`` (in-memory SQLite) from conftest.py.  No
PostgreSQL is required; the SQLite in-memory engine is created per test.
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from Backend.Database.operations import (
    count_observations,
    ensure_patient,
    get_latest_prediction,
    get_patient_alerts,
    get_patient_history,
    get_patient_predictions,
    insert_alert,
    rebuild_alert_events,
    update_prediction_alert_fields,
    upsert_alert_summary,
    upsert_observation,
    upsert_prediction,
)
from Backend.Database.schema import AlertSummary, Observation, Patient, Prediction


def test_ensure_patient_is_idempotent(db_session):
    first = ensure_patient(db_session, "p7db", 45)
    db_session.flush()
    second = ensure_patient(db_session, "p7db", 45)
    db_session.flush()
    assert first.patient_id == second.patient_id == "p7db"
    from sqlalchemy import func, select

    total = db_session.execute(select(func.count(Patient.id))).scalar_one()
    assert total == 1


def test_observation_upsert_is_single_row(db_session):
    ensure_patient(db_session, "p7db", 45)
    upsert_observation(db_session, "p7db", 1, hr=80.0, sbp=120.0)
    db_session.flush()
    upsert_observation(db_session, "p7db", 1, hr=90.0, sbp=130.0)
    db_session.flush()
    rows = get_patient_history(db_session, "p7db")
    assert len(rows) == 1
    assert rows[0].hr == 90.0
    assert rows[0].sbp == 130.0
    assert count_observations(db_session, "p7db") == 1


def test_prediction_upsert_is_single_row(db_session):
    ensure_patient(db_session, "p7db", 45)
    upsert_prediction(
        db_session, "p7db", 2, raw_probability=0.4,
        filtered_probability=0.4, high_risk=False, alert=False,
    )
    db_session.flush()
    upsert_prediction(
        db_session, "p7db", 2, raw_probability=0.6,
        filtered_probability=0.6, high_risk=True, alert=True,
    )
    db_session.flush()
    rows = get_patient_predictions(db_session, "p7db")
    assert len(rows) == 1
    assert rows[0].alert is True


def test_get_patient_history_ordered_by_iculos(db_session):
    ensure_patient(db_session, "p7db", 45)
    for iculos in (3, 1, 2):
        upsert_observation(db_session, "p7db", iculos, hr=70.0 + iculos)
    db_session.flush()
    assert [o.iculos for o in get_patient_history(db_session, "p7db")] == [1, 2, 3]


def test_get_patient_predictions_ordered_by_iculos(db_session):
    ensure_patient(db_session, "p7db", 45)
    for iculos in (5, 3, 4, 1, 2):
        upsert_prediction(
            db_session, "p7db", iculos, raw_probability=0.01,
            filtered_probability=0.01, high_risk=False, alert=False,
        )
    db_session.flush()
    assert [p.iculos for p in get_patient_predictions(db_session, "p7db")] == [1, 2, 3, 4, 5]


def test_get_latest_prediction_returns_highest_iculos(db_session):
    ensure_patient(db_session, "p7db", 45)
    for iculos in (2, 5, 3):
        upsert_prediction(
            db_session, "p7db", iculos, raw_probability=0.01,
            filtered_probability=0.01, high_risk=False, alert=False,
        )
    db_session.flush()
    assert get_latest_prediction(db_session, "p7db").iculos == 5


def test_update_prediction_alert_fields_preserves_raw(db_session):
    ensure_patient(db_session, "p7db", 45)
    upsert_prediction(
        db_session, "p7db", 1, raw_probability=0.5,
        filtered_probability=0.2, high_risk=False, alert=False,
    )
    db_session.flush()
    update_prediction_alert_fields(
        db_session, "p7db", 1, filtered_probability=0.0, high_risk=True, alert=True,
    )
    db_session.flush()
    row = get_patient_predictions(db_session, "p7db")[0]
    assert row.raw_probability == 0.5  # immutable source of truth
    assert row.filtered_probability == 0.0
    assert row.high_risk is True
    assert row.alert is True


def test_observation_unique_constraint_enforced(db_session):
    ensure_patient(db_session, "p7db", 45)
    db_session.add(Observation(patient_id="p7db", iculos=1, hr=80.0))
    db_session.flush()
    db_session.add(Observation(patient_id="p7db", iculos=1, hr=99.0))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_prediction_requires_probability(db_session):
    ensure_patient(db_session, "p7db", 45)
    db_session.add(Prediction(patient_id="p7db", iculos=1, raw_probability=None,
                              filtered_probability=0.0, high_risk=False, alert=False))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_observation_fk_requires_existing_patient(db_session):
    db_session.add(Observation(patient_id="missing_patient", iculos=1, hr=80.0))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_insert_alert_records_duration(db_session):
    ensure_patient(db_session, "p7db", 45)
    alert = insert_alert(
        db_session, "p7db", alert_start_iculos=4, alert_end_iculos=7, peak_risk=0.9,
    )
    db_session.flush()
    assert alert.duration_hours == 4
    assert alert.peak_risk == 0.9


def test_get_patient_alerts_ordered_by_start(db_session):
    ensure_patient(db_session, "p7db", 45)
    insert_alert(db_session, "p7db", alert_start_iculos=20, alert_end_iculos=21, peak_risk=0.5)
    insert_alert(db_session, "p7db", alert_start_iculos=5, alert_end_iculos=6, peak_risk=0.6)
    db_session.flush()
    starts = [a.alert_start_iculos for a in get_patient_alerts(db_session, "p7db")]
    assert starts == [5, 20]


def test_rebuild_alert_events_is_idempotent(db_session):
    ensure_patient(db_session, "p7db", 45)
    for iculos, alert in [(1, False), (2, True), (3, True), (4, False), (5, True)]:
        upsert_prediction(
            db_session, "p7db", iculos, raw_probability=0.9 if alert else 0.02,
            filtered_probability=0.9 if alert else 0.02,
            high_risk=alert, alert=alert,
        )
    db_session.flush()
    first = rebuild_alert_events(db_session, "p7db")
    second = rebuild_alert_events(db_session, "p7db")
    assert [e.alert_start_iculos for e in first] == [2, 5]
    assert len(second) == len(first)


def test_upsert_alert_summary_removes_stale_when_no_alerts(db_session):
    ensure_patient(db_session, "p7db", 45)
    # Seed a stale summary as if leftover from previous alert state.
    db_session.add(AlertSummary(
        patient_id="p7db", total_alerts=1, total_alert_hours=2,
        first_alert_iculos=3, last_alert_iculos=4, max_peak_risk=0.8,
    ))
    db_session.flush()
    result = upsert_alert_summary(db_session, "p7db")
    db_session.flush()
    assert result is None
    from sqlalchemy import func, select

    remaining = db_session.execute(
        select(func.count(AlertSummary.id)).where(AlertSummary.patient_id == "p7db")
    ).scalar_one()
    assert remaining == 0