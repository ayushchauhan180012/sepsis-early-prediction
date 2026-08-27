"""Phase 7 — opt-in PostgreSQL integration tests (O-4, D-009, D-022).

These tests exercise the real ``operations``/``schema`` code against an actual
PostgreSQL server and are STRICTLY gated:

  * They run ONLY when ``TEST_DATABASE_URL`` is explicitly set in the
    environment.  They are never derived from the production ``DATABASE_URL``.
  * If ``TEST_DATABASE_URL`` is absent or unreachable, they skip cleanly.
  * If the test URL points at the SAME database as ``settings.database_url``
    (production/dev), they FAIL loudly rather than risk it.
  * All tables are created via ``Base.metadata.create_all`` and dropped via
    ``Base.metadata.drop_all`` ONLY on the test engine.

Default collection excludes these (``addopts = -m "not integration"`` in
pytest.ini).  Run them with:

    python -m pytest -m integration
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import sessionmaker

from Backend.config import settings
from Backend.Database.operations import (
    ensure_patient,
    get_patient_history,
    get_patient_predictions,
    rebuild_alert_events,
    upsert_alert_summary,
    upsert_observation,
    upsert_prediction,
)
from Backend.Database.schema import Base, Alert, Observation, Prediction

pytestmark = pytest.mark.integration

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


def _dbname(url: str) -> str:
    return (urlparse(url).path or "").lstrip("/").lower()


def _skip_or_fail_reason() -> str | None:
    if not TEST_DATABASE_URL:
        return "TEST_DATABASE_URL is not set (integration tests are opt-in)"
    if TEST_DATABASE_URL.lower().startswith("sqlite"):
        return "TEST_DATABASE_URL must point at PostgreSQL, not SQLite"
    prod_db = _dbname(settings.database_url or "")
    test_db = _dbname(TEST_DATABASE_URL)
    if prod_db and test_db and test_db == prod_db:
        raise SystemExit(
            "REFUSING to run integration tests against the same database as "
            f"DATABASE_URL (both are '{test_db}'). Point TEST_DATABASE_URL at a "
            "dedicated throwaway database instead."
        )
    return None


@pytest.fixture(scope="module")
def pg_engine():
    reason = _skip_or_fail_reason()
    if reason:
        pytest.skip(reason)

    engine = create_engine(TEST_DATABASE_URL, echo=False)
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except OperationalError as exc:  # unreachable / bad credentials
        pytest.skip(f"TEST_DATABASE_URL is not reachable: {exc}")
    except Exception as exc:
        pytest.fail(f"unexpected error connecting to TEST_DATABASE_URL: {exc}")

    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture()
def pg_session(pg_engine):
    TestSession = sessionmaker(bind=pg_engine, autocommit=False, autoflush=False)
    session = TestSession()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(autouse=True)
def clean_test_tables(pg_session):
    """Start each test from empty tables (test engine only — never prod)."""
    for table in ("alert_summaries", "alerts", "predictions", "observations", "patients"):
        pg_session.execute(text(f"DELETE FROM {table}"))
    pg_session.commit()
    yield


def test_all_tables_created(pg_engine):
    tables = set(pg_engine.table_names() if hasattr(pg_engine, "table_names") else [])
    if not tables:
        with pg_engine.connect() as conn:
            tables = {
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                ).fetchall()
            }
    assert {"patients", "observations", "predictions", "alerts", "alert_summaries"} <= tables


def test_observation_roundtrip_uses_lowercase_d022(pg_session):
    ensure_patient(pg_session, "p100000", 64)
    upsert_observation(pg_session, "p100000", 1, hr=80.0, o2sat=97.0, sbp=120.0)
    pg_session.flush()
    rows = get_patient_history(pg_session, "p100000")
    assert len(rows) == 1
    row = rows[0]
    assert row.hr == 80.0 and row.o2sat == 97.0 and row.sbp == 120.0
    assert row.map is None
    with pg_session.connection() as conn:
        cols = {
            r[0]
            for r in conn.exec_driver_sql(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'observations'"
            ).fetchall()
        }
    assert "hr" in cols and "o2sat" in cols and "ICULOS" not in cols
    assert "iculos" in cols


def test_observation_unique_patient_iculos(pg_session):
    ensure_patient(pg_session, "p100000", 64)
    pg_session.add(Observation(patient_id="p100000", iculos=1, hr=80.0))
    pg_session.flush()
    pg_session.add(Observation(patient_id="p100000", iculos=1, hr=81.0))
    with pytest.raises(IntegrityError):
        pg_session.flush()


def test_prediction_fk_enforced(pg_session):
    pg_session.add(Prediction(
        patient_id="no_such_patient", iculos=1, raw_probability=0.5,
        filtered_probability=0.5, high_risk=False, alert=False,
    ))
    with pytest.raises(IntegrityError):
        pg_session.flush()


def test_upsert_prediction_updates_in_place(pg_session):
    ensure_patient(pg_session, "p100000", 64)
    upsert_prediction(
        pg_session, "p100000", 1, raw_probability=0.5,
        filtered_probability=0.2, high_risk=False, alert=False,
    )
    pg_session.flush()
    upsert_prediction(
        pg_session, "p100000", 1, raw_probability=0.55,
        filtered_probability=0.55, high_risk=True, alert=True,
    )
    pg_session.flush()
    rows = get_patient_predictions(pg_session, "p100000")
    assert len(rows) == 1
    assert rows[0].alert is True and rows[0].raw_probability == 0.55


def test_predictions_ordered_by_iculos_asc(pg_session):
    ensure_patient(pg_session, "p100000", 64)
    for iculos in (5, 2, 4, 1, 3):
        upsert_prediction(
            pg_session, "p100000", iculos, raw_probability=0.5,
            filtered_probability=0.5, high_risk=False, alert=False,
        )
    pg_session.flush()
    assert [p.iculos for p in get_patient_predictions(pg_session, "p100000")] == [1, 2, 3, 4, 5]


def test_full_alert_flow(pg_session):
    ensure_patient(pg_session, "p100000", 64)
    for iculos, alert in [(1, False), (2, True), (3, True), (4, True), (5, False)]:
        upsert_prediction(
            pg_session, "p100000", iculos, raw_probability=0.9 if alert else 0.02,
            filtered_probability=0.9 if alert else 0.02,
            high_risk=alert, alert=alert,
        )
    pg_session.flush()

    events = rebuild_alert_events(pg_session, "p100000")
    assert [(e.alert_start_iculos, e.alert_end_iculos, e.duration_hours) for e in events] == [
        (2, 4, 3)
    ]
    assert events[0].peak_risk == 0.9

    summary = upsert_alert_summary(pg_session, "p100000")
    pg_session.flush()
    assert summary is not None
    assert summary.total_alerts == 1
    assert summary.total_alert_hours == 3
    assert summary.first_alert_iculos == 2
    assert summary.last_alert_iculos == 4
    assert summary.max_peak_risk == 0.9


def test_rebuild_alert_events_respects_gaps(pg_session):
    ensure_patient(pg_session, "p100000", 64)
    for iculos, alert in [(1, True), (2, True), (3, False), (4, True), (5, True)]:
        upsert_prediction(
            pg_session, "p100000", iculos, raw_probability=0.7 if alert else 0.02,
            filtered_probability=0.7 if alert else 0.02,
            high_risk=alert, alert=alert,
        )
    pg_session.flush()
    events = rebuild_alert_events(pg_session, "p100000")
    assert [(e.alert_start_iculos, e.alert_end_iculos) for e in events] == [(1, 2), (4, 5)]