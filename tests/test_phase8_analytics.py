"""Phase 8 correctness tests — Analytics / reporting queries and API endpoints.

Covers:
  A. get_risk_trajectory  — returns correct trajectory dicts from predictions
  B. get_peak_risk        — returns max raw_probability + its ICULOS
  C. get_alert_statistics — reads from alert_summaries table
  D. GET /patients/{pid}/trajectory — API endpoint returns trajectory + peak
  E. GET /patients/{pid}/alerts    — API endpoint returns alert summary
  F. Edge cases           — empty patient, no predictions, no alerts

API endpoint tests reuse the established in-memory-SQLite pattern with
``StaticPool`` (see ``test_phase6_api.py``): the shared connection lets data
seeded through a ``session_factory`` session be visible to the per-request
sessions created by the FastAPI ``get_db`` dependency override.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from Backend.Database.operations import (
    get_patient_history,
    get_risk_trajectory,
    get_peak_risk,
    get_alert_statistics,
    ensure_patient,
    upsert_observation,
    upsert_prediction,
    rebuild_alert_events,
    upsert_alert_summary,
)

REQUEST_ID_HEADER = "X-Request-ID"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _seed_predictions(session: Session, patient_id: str, rows: list[dict]) -> None:
    """Insert prediction rows directly into the DB.

    Each dict must have: iculos, raw_probability, filtered_probability,
    high_risk, alert.
    """
    ensure_patient(session, patient_id, age=65)
    for r in rows:
        upsert_prediction(
            session,
            patient_id,
            r["iculos"],
            raw_probability=r["raw_probability"],
            filtered_probability=r["filtered_probability"],
            high_risk=r["high_risk"],
            alert=r["alert"],
        )


def _seed_alerts_and_summary(session: Session, patient_id: str) -> None:
    """Rebuild alert events and summary from persisted predictions."""
    rebuild_alert_events(session, patient_id)
    upsert_alert_summary(session, patient_id)


def _make_predictions_sequence(
    probabilities: list[float],
    *,
    start_iculos: int = 1,
) -> list[dict]:
    """Build prediction row dicts from a list of raw probabilities.

    Applies the alert contract (uncertainty → threshold → persistence →
    cooldown) to derive filtered_probability, high_risk, and alert.
    """
    from Backend.Services.alert_engine import evaluate_alert_state

    inputs = [(start_iculos + i, p) for i, p in enumerate(probabilities)]
    states = evaluate_alert_state(inputs)
    return [
        {
            "iculos": s.iculos,
            "raw_probability": s.raw_probability,
            "filtered_probability": s.filtered_probability,
            "high_risk": s.high_risk,
            "alert": s.alert,
        }
        for s in states
    ]


# ── A. get_risk_trajectory ───────────────────────────────────────────────────

class TestRiskTrajectory:
    def test_returns_list_of_dicts(self, db_session: Session):
        rows = _make_predictions_sequence([0.02, 0.06, 0.08])
        _seed_predictions(db_session, "P-TRAJ1", rows)
        result = get_risk_trajectory(db_session, "P-TRAJ1")
        assert isinstance(result, list)
        assert len(result) == 3

    def test_keys_present(self, db_session: Session):
        rows = _make_predictions_sequence([0.02, 0.06])
        _seed_predictions(db_session, "P-TRAJ2", rows)
        result = get_risk_trajectory(db_session, "P-TRAJ2")
        for entry in result:
            assert set(entry.keys()) == {
                "iculos", "raw_probability", "filtered_probability",
                "high_risk", "alert",
            }

    def test_ordered_by_iculos_asc(self, db_session: Session):
        rows = _make_predictions_sequence([0.01, 0.05, 0.03, 0.07])
        _seed_predictions(db_session, "P-TRAJ3", rows)
        result = get_risk_trajectory(db_session, "P-TRAJ3")
        iculos_values = [r["iculos"] for r in result]
        assert iculos_values == sorted(iculos_values)

    def test_raw_probability_matches_input(self, db_session: Session):
        probs = [0.01, 0.06, 0.09, 0.03]
        rows = _make_predictions_sequence(probs)
        _seed_predictions(db_session, "P-TRAJ4", rows)
        result = get_risk_trajectory(db_session, "P-TRAJ4")
        for entry, expected_p in zip(result, probs):
            assert entry["raw_probability"] == pytest.approx(expected_p)

    def test_empty_for_no_predictions(self, db_session: Session):
        result = get_risk_trajectory(db_session, "P-TRAJ-NONE")
        assert result == []


# ── B. get_peak_risk ─────────────────────────────────────────────────────────

class TestPeakRisk:
    def test_returns_peak_and_iculos(self, db_session: Session):
        probs = [0.01, 0.08, 0.03, 0.06]
        rows = _make_predictions_sequence(probs)
        _seed_predictions(db_session, "P-PEAK1", rows)
        result = get_peak_risk(db_session, "P-PEAK1")
        assert result is not None
        assert result["peak_risk"] == pytest.approx(0.08)
        assert result["iculos"] == 2

    def test_first_element_is_peak(self, db_session: Session):
        probs = [0.10, 0.02, 0.03]
        rows = _make_predictions_sequence(probs)
        _seed_predictions(db_session, "P-PEAK2", rows)
        result = get_peak_risk(db_session, "P-PEAK2")
        assert result["peak_risk"] == pytest.approx(0.10)
        assert result["iculos"] == 1

    def test_last_element_is_peak(self, db_session: Session):
        probs = [0.01, 0.02, 0.12]
        rows = _make_predictions_sequence(probs)
        _seed_predictions(db_session, "P-PEAK3", rows)
        result = get_peak_risk(db_session, "P-PEAK3")
        assert result["peak_risk"] == pytest.approx(0.12)
        assert result["iculos"] == 3

    def test_none_for_no_predictions(self, db_session: Session):
        result = get_peak_risk(db_session, "P-PEAK-NONE")
        assert result is None


# ── C. get_alert_statistics ──────────────────────────────────────────────────

class TestAlertStatistics:
    def test_returns_none_for_no_alerts(self, db_session: Session):
        probs = [0.01, 0.01, 0.01]
        rows = _make_predictions_sequence(probs)
        _seed_predictions(db_session, "P-STAT1", rows)
        _seed_alerts_and_summary(db_session, "P-STAT1")
        result = get_alert_statistics(db_session, "P-STAT1")
        assert result is None

    def test_returns_summary_when_alerts_exist(self, db_session: Session):
        probs = [0.06, 0.07, 0.08, 0.07, 0.08, 0.07, 0.08]
        rows = _make_predictions_sequence(probs)
        _seed_predictions(db_session, "P-STAT2", rows)
        _seed_alerts_and_summary(db_session, "P-STAT2")
        result = get_alert_statistics(db_session, "P-STAT2")
        assert result is not None
        assert "total_alerts" in result
        assert "total_alert_hours" in result
        assert "first_alert_iculos" in result
        assert "last_alert_iculos" in result
        assert "max_peak_risk" in result

    def test_total_alerts_positive(self, db_session: Session):
        probs = [0.06, 0.07, 0.08, 0.07, 0.08, 0.07, 0.08]
        rows = _make_predictions_sequence(probs)
        _seed_predictions(db_session, "P-STAT3", rows)
        _seed_alerts_and_summary(db_session, "P-STAT3")
        result = get_alert_statistics(db_session, "P-STAT3")
        assert result["total_alerts"] >= 1
        assert result["total_alert_hours"] >= 1

    def test_peak_risk_in_summary_matches_highest_raw(self, db_session: Session):
        probs = [0.06, 0.07, 0.08, 0.07, 0.08, 0.07, 0.08]
        rows = _make_predictions_sequence(probs)
        _seed_predictions(db_session, "P-STAT4", rows)
        _seed_alerts_and_summary(db_session, "P-STAT4")
        result = get_alert_statistics(db_session, "P-STAT4")
        assert result["max_peak_risk"] == pytest.approx(max(probs))


# ── API infra (shared with endpoint tests) ───────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan; the shared app's model is loaded once."""
    from Backend.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def override_get_db(client, session_factory):
    """Route every ``get_db`` request to the shared StaticPool SQLite DB."""
    from Backend.app import app
    from Backend.Database.connection import get_db

    def _override():
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
def api_session(session_factory):
    """A session over the SAME StaticPool DB the API uses (for seeding)."""
    session = session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# ── D. GET /patients/{pid}/trajectory ────────────────────────────────────────

class TestTrajectoryEndpoint:
    def test_trajectory_returns_200(self, client, api_session, override_get_db):
        rows = _make_predictions_sequence([0.02, 0.06, 0.03])
        _seed_predictions(api_session, "P-API-TRJ1", rows)
        resp = client.get("/patients/P-API-TRJ1/trajectory")
        assert resp.status_code == 200

    def test_trajectory_structure(self, client, api_session, override_get_db):
        rows = _make_predictions_sequence([0.02, 0.06, 0.03])
        _seed_predictions(api_session, "P-API-TRJ2", rows)
        resp = client.get("/patients/P-API-TRJ2/trajectory")
        body = resp.json()
        assert body["patient_id"] == "P-API-TRJ2"
        assert isinstance(body["trajectory"], list)
        assert len(body["trajectory"]) == 3
        assert body["peak_risk"] is not None

    def test_trajectory_peak_risk_included(self, client, api_session, override_get_db):
        probs = [0.01, 0.08, 0.03]
        rows = _make_predictions_sequence(probs)
        _seed_predictions(api_session, "P-API-TRJ3", rows)
        resp = client.get("/patients/P-API-TRJ3/trajectory")
        body = resp.json()
        assert body["peak_risk"]["peak_risk"] == pytest.approx(0.08)
        assert body["peak_risk"]["iculos"] == 2

    def test_trajectory_empty_for_unknown_patient(self, client, override_get_db):
        resp = client.get("/patients/P-NONE/trajectory")
        assert resp.status_code == 200
        body = resp.json()
        assert body["trajectory"] == []
        assert body["peak_risk"] is None

    def test_trajectory_has_request_id(self, client, api_session, override_get_db):
        rows = _make_predictions_sequence([0.02])
        _seed_predictions(api_session, "P-API-TRJ5", rows)
        resp = client.get("/patients/P-API-TRJ5/trajectory")
        assert REQUEST_ID_HEADER in resp.headers


# ── E. GET /patients/{pid}/alerts ────────────────────────────────────────────

class TestAlertsEndpoint:
    def test_alerts_returns_200(self, client, api_session, override_get_db):
        rows = _make_predictions_sequence([0.01, 0.01, 0.01])
        _seed_predictions(api_session, "P-API-ALT1", rows)
        _seed_alerts_and_summary(api_session, "P-API-ALT1")
        resp = client.get("/patients/P-API-ALT1/alerts")
        assert resp.status_code == 200

    def test_alerts_null_when_no_alerts(self, client, api_session, override_get_db):
        rows = _make_predictions_sequence([0.01, 0.01, 0.01])
        _seed_predictions(api_session, "P-API-ALT2", rows)
        _seed_alerts_and_summary(api_session, "P-API-ALT2")
        resp = client.get("/patients/P-API-ALT2/alerts")
        body = resp.json()
        assert body["alert_summary"] is None

    def test_alerts_summary_when_alerts_exist(self, client, api_session, override_get_db):
        probs = [0.06, 0.07, 0.08, 0.07, 0.08, 0.07, 0.08]
        rows = _make_predictions_sequence(probs)
        _seed_predictions(api_session, "P-API-ALT3", rows)
        _seed_alerts_and_summary(api_session, "P-API-ALT3")
        resp = client.get("/patients/P-API-ALT3/alerts")
        body = resp.json()
        assert body["patient_id"] == "P-API-ALT3"
        assert body["alert_summary"] is not None
        assert body["alert_summary"]["total_alerts"] >= 1

    def test_alerts_has_request_id(self, client, api_session, override_get_db):
        rows = _make_predictions_sequence([0.01])
        _seed_predictions(api_session, "P-API-ALT4", rows)
        resp = client.get("/patients/P-API-ALT4/alerts")
        assert REQUEST_ID_HEADER in resp.headers

    def test_alerts_empty_for_unknown_patient(self, client, override_get_db):
        resp = client.get("/patients/P-NONE-ALT/alerts")
        assert resp.status_code == 200
        body = resp.json()
        assert body["alert_summary"] is None


# ── F. GET /patients/{pid}/observations ──────────────────────────────────────

def _seed_observations(session: Session, patient_id: str) -> None:
    """Insert a small, known observation history for an endpoint test."""
    ensure_patient(session, patient_id, age=65)
    upsert_observation(session, patient_id, 1, hr=84.0, o2sat=98.0,
                       sbp=118.0, map=77.0, resp=18.0, temp=36.94,
                       lactate=None, wbc=None, creatinine=None, platelets=None)
    upsert_observation(session, patient_id, 2, hr=90.0, o2sat=96.0,
                       sbp=110.0, map=72.0, resp=20.0, temp=37.1,
                       lactate=2.1, wbc=8.5, creatinine=1.1, platelets=210.0)


class TestObservationsEndpoint:
    def test_observations_returns_200(self, client, api_session, override_get_db):
        _seed_observations(api_session, "P-API-OBS1")
        resp = client.get("/patients/P-API-OBS1/observations")
        assert resp.status_code == 200

    def test_observations_structure_and_order(self, client, api_session, override_get_db):
        _seed_observations(api_session, "P-API-OBS2")
        resp = client.get("/patients/P-API-OBS2/observations")
        body = resp.json()
        assert body["patient_id"] == "P-API-OBS2"
        assert isinstance(body["observations"], list)
        assert len(body["observations"]) == 2
        assert [o["iculos"] for o in body["observations"]] == [1, 2]

    def test_observations_preserve_raw_values(self, client, api_session, override_get_db):
        _seed_observations(api_session, "P-API-OBS3")
        resp = client.get("/patients/P-API-OBS3/observations")
        body = resp.json()
        first, second = body["observations"]
        assert first["hr"] == pytest.approx(84.0)
        assert first["temp"] == pytest.approx(36.94)
        assert second["lactate"] == pytest.approx(2.1)
        assert second["wbc"] == pytest.approx(8.5)

    def test_observations_nullable_labs(self, client, api_session, override_get_db):
        _seed_observations(api_session, "P-API-OBS4")
        resp = client.get("/patients/P-API-OBS4/observations")
        body = resp.json()
        first = body["observations"][0]
        assert first["lactate"] is None
        assert first["wbc"] is None
        assert first["creatinine"] is None
        assert first["platelets"] is None

    def test_observations_has_request_id(self, client, api_session, override_get_db):
        _seed_observations(api_session, "P-API-OBS5")
        resp = client.get("/patients/P-API-OBS5/observations")
        assert REQUEST_ID_HEADER in resp.headers

    def test_observations_empty_for_unknown_patient(self, client, override_get_db):
        resp = client.get("/patients/P-NONE-OBS/observations")
        assert resp.status_code == 200
        body = resp.json()
        assert body["observations"] == []

    def test_observations_history_preserved(self, api_session):
        """get_patient_history returns raw ORM rows ordered by ICULOS."""
        _seed_observations(api_session, "P-OBS-HIST1")
        rows = get_patient_history(api_session, "P-OBS-HIST1")
        assert [o.iculos for o in rows] == [1, 2]
        assert rows[0].hr == 84.0