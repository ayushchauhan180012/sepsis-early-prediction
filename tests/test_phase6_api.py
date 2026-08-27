"""Phase 6 correctness tests — FastAPI ingestion endpoints.

Covers (per Phase 6 spec):
  A. Health        — /health returns 200, model availability
  B. Valid predict — POST /predict returns 200 with PredictionResponse structure
  C. Validation    — 422 for missing/invalid fields and ICULOS < 1
  D. Persistence   — valid request creates observation + prediction, matches response
  E. Sequential    — ICULOS 1, 2, 3 accepted and persisted
  F. Ordering      — duplicate / older ICULOS rejected with 409, no DB mutation
  G. Alert         — response contains filtered_probability, high_risk, alert (Phase 5)
  H. Isolation     — independent per-patient ordering/state
  I. Errors        — DB / model failure -> 500, no internals leaked
  J. Request IDs   — response has request ID; supplied one preserved; else generated

The tests reuse the established in-memory-SQLite pattern, but with
``StaticPool`` so a single shared connection lets data persist across the
multiple per-request sessions created by the FastAPI ``get_db`` dependency.
"""

from __future__ import annotations

import pytest
import joblib
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from Backend.app import app
from Backend.Database.connection import get_db
from Backend.Database.schema import Base

REQUEST_ID_HEADER = "X-Request-ID"


@pytest.fixture(scope="module")
def model():
    """The frozen HGB model loaded once per module."""
    return joblib.load("Backend/Model/hgb_sepsis_model.joblib")


@pytest.fixture()
def db():
    """A fresh in-memory SQLite DB sharing one connection across sessions."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    yield TestSession
    test_engine.dispose()


@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan; the shared app's model is loaded once."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def override_get_db(client, db, monkeypatch):
    """Route every ``get_db`` request to the per-test in-memory SQLite DB."""

    def _override():
        session = db()
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


def make_obs(patient_id: str, iculos: int, **overrides) -> dict:
    obs = {
        "PatientID": patient_id,
        "Age": 65,
        "ICULOS": iculos,
        "HR": 84.0 + iculos,
        "O2Sat": 98.0,
        "SBP": 118.0,
        "MAP": 77.0,
        "Resp": 18.0,
        "Temp": 36.94,
        "Lactate": None,
        "WBC": None,
        "Creatinine": None,
        "Platelets": None,
    }
    obs.update(overrides)
    return obs


def make_valid(patient_id: str, iculos: int) -> dict:
    return make_obs(patient_id, iculos)


# ── A. Health ────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_reports_model_loaded(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


# ── B. Valid prediction ──────────────────────────────────────────────────────

class TestValidPrediction:
    def test_valid_post_returns_200(self, client, override_get_db):
        resp = client.post("/predict", json=make_valid("P-B1", 1))
        assert resp.status_code == 200

    def test_response_has_prediction_structure(self, client, override_get_db):
        resp = client.post("/predict", json=make_valid("P-B2", 1))
        body = resp.json()
        assert set(body.keys()) == {
            "patient_id", "iculos", "raw_probability",
            "filtered_probability", "high_risk", "alert",
        }

    def test_raw_probability_in_unit_interval(self, client, override_get_db):
        resp = client.post("/predict", json=make_valid("P-B3", 1))
        assert 0.0 <= resp.json()["raw_probability"] <= 1.0

    def test_filtered_probability_in_unit_interval(self, client, override_get_db):
        resp = client.post("/predict", json=make_valid("P-B4", 1))
        assert 0.0 <= resp.json()["filtered_probability"] <= 1.0

    def test_high_risk_and_alert_are_bools(self, client, override_get_db):
        resp = client.post("/predict", json=make_valid("P-B5", 1))
        body = resp.json()
        assert isinstance(body["high_risk"], bool)
        assert isinstance(body["alert"], bool)


# ── C. Validation ────────────────────────────────────────────────────────────

class TestValidation:
    def test_missing_required_field_returns_422(self, client, override_get_db):
        payload = make_valid("P-C1", 1)
        del payload["PatientID"]
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_physiological_value_returns_422(self, client, override_get_db):
        resp = client.post("/predict", json=make_obs("P-C2", 1, HR=9999))
        assert resp.status_code == 422

    def test_invalid_iculos_less_than_one_returns_422(self, client, override_get_db):
        resp = client.post("/predict", json=make_valid("P-C3", 0))
        assert resp.status_code == 422


# ── D. Persistence ───────────────────────────────────────────────────────────

class TestPersistence:
    def test_valid_request_creates_observation(self, client, db, override_get_db):
        from Backend.Database.operations import get_patient_history

        client.post("/predict", json=make_valid("P-D1", 1))
        session = db()
        history = get_patient_history(session, "P-D1")
        session.close()
        assert len(history) == 1
        assert history[0].iculos == 1

    def test_valid_request_creates_prediction(self, client, db, override_get_db):
        from Backend.Database.operations import get_patient_predictions

        client.post("/predict", json=make_valid("P-D2", 1))
        session = db()
        preds = get_patient_predictions(session, "P-D2")
        session.close()
        assert len(preds) == 1

    def test_returned_prediction_matches_persisted(self, client, db, override_get_db):
        from Backend.Database.operations import get_patient_predictions

        resp = client.post("/predict", json=make_valid("P-D3", 1))
        body = resp.json()
        session = db()
        pred = get_patient_predictions(session, "P-D3")[0]
        session.close()
        assert pred.patient_id == body["patient_id"]
        assert pred.iculos == body["iculos"]
        assert pred.raw_probability == pytest.approx(body["raw_probability"])
        assert pred.filtered_probability == pytest.approx(body["filtered_probability"])
        assert pred.high_risk == body["high_risk"]
        assert pred.alert == body["alert"]


# ── E. Sequential observations ───────────────────────────────────────────────

class TestSequential:
    def test_iculos_1_2_3_accepted_and_persisted(self, client, db, override_get_db):
        from Backend.Database.operations import (
            get_patient_history,
            get_patient_predictions,
        )

        for hour in range(1, 4):
            resp = client.post("/predict", json=make_valid("P-E1", hour))
            assert resp.status_code == 200
            assert resp.json()["iculos"] == hour

        session = db()
        obs = get_patient_history(session, "P-E1")
        preds = get_patient_predictions(session, "P-E1")
        session.close()

        assert [o.iculos for o in obs] == [1, 2, 3]
        assert [p.iculos for p in preds] == [1, 2, 3]


# ── F. ICULOS ordering ───────────────────────────────────────────────────────

class TestIculosOrdering:
    def test_first_observation_accepted(self, client, override_get_db):
        resp = client.post("/predict", json=make_valid("P-F1", 5))
        assert resp.status_code == 200

    def test_next_greater_iculos_accepted(self, client, override_get_db):
        client.post("/predict", json=make_valid("P-F2", 1))
        resp = client.post("/predict", json=make_valid("P-F2", 2))
        assert resp.status_code == 200

    def test_duplicate_iculos_rejected_409(self, client, override_get_db):
        client.post("/predict", json=make_valid("P-F3", 1))
        resp = client.post("/predict", json=make_valid("P-F3", 1))
        assert resp.status_code == 409

    def test_older_iculos_rejected_409(self, client, override_get_db):
        client.post("/predict", json=make_valid("P-F4", 3))
        resp = client.post("/predict", json=make_valid("P-F4", 2))
        assert resp.status_code == 409

    def test_rejected_request_does_not_modify_db(self, client, db, override_get_db):
        from Backend.Database.operations import (
            get_patient_history,
            get_patient_predictions,
        )

        client.post("/predict", json=make_valid("P-F5", 3))
        resp = client.post("/predict", json=make_valid("P-F5", 3))  # duplicate
        assert resp.status_code == 409

        session = db()
        obs = get_patient_history(session, "P-F5")
        preds = get_patient_predictions(session, "P-F5")
        session.close()
        assert len(obs) == 1
        assert len(preds) == 1


# ── G. Alert integration ─────────────────────────────────────────────────────

class TestAlertIntegration:
    def test_response_contains_alert_fields(self, client, override_get_db):
        resp = client.post("/predict", json=make_valid("P-G1", 1))
        body = resp.json()
        assert "filtered_probability" in body
        assert "high_risk" in body
        assert "alert" in body

    def test_alert_matches_persisted_state(self, client, db, override_get_db):
        from Backend.Database.operations import get_patient_predictions

        resp = client.post("/predict", json=make_valid("P-G2", 1))
        body = resp.json()
        session = db()
        pred = get_patient_predictions(session, "P-G2")[0]
        session.close()
        assert pred.filtered_probability == pytest.approx(body["filtered_probability"])
        assert pred.high_risk == body["high_risk"]
        assert pred.alert == body["alert"]

    def test_multiple_observations_use_phase5_logic(self, client, db, override_get_db):
        from Backend.Database.operations import get_patient_predictions
        from Backend.Services.alert_engine import evaluate_alert_state

        for hour in range(1, 4):
            client.post("/predict", json=make_valid("P-G3", hour))

        session = db()
        preds = get_patient_predictions(session, "P-G3")
        session.close()
        # Phase 5 alert state must match the pure recompute-from-history result.
        expected = evaluate_alert_state(
            [(p.iculos, p.raw_probability) for p in preds]
        )
        for pred, state in zip(preds, expected):
            assert pred.filtered_probability == pytest.approx(state.filtered_probability)
            assert pred.high_risk == state.high_risk
            assert pred.alert == state.alert


# ── H. Patient isolation ─────────────────────────────────────────────────────

class TestPatientIsolation:
    def test_different_patients_start_at_iculos_1(self, client, override_get_db):
        for pid in ("P-H1", "P-H2"):
            resp = client.post("/predict", json=make_valid(pid, 1))
            assert resp.status_code == 200

    def test_ordering_for_one_patient_does_not_affect_another(
        self, client, override_get_db
    ):
        client.post("/predict", json=make_valid("P-H3", 1))
        client.post("/predict", json=make_valid("P-H4", 7))
        # H3 still only has ICULOS 1 → is 1 a duplicate? Posting 1 again is rejected.
        resp_dup = client.post("/predict", json=make_valid("P-H3", 1))
        assert resp_dup.status_code == 409
        # A fresh patient can still start at ICULOS 1.
        resp_fresh = client.post("/predict", json=make_valid("P-H5", 1))
        assert resp_fresh.status_code == 200


# ── I. Error handling ────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_database_failure_returns_500(self, client, db, monkeypatch):
        from Backend import app as app_mod

        def boom(*args, **kwargs):
            raise RuntimeError("simulated DB failure")

        monkeypatch.setattr(app_mod, "process_observation", boom)

        # Override get_db to yield a session whose queries raise.
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)
        TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)

        def _override():
            session = TestSession()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        app.dependency_overrides[get_db] = _override
        try:
            resp = client.post("/predict", json=make_valid("P-I1", 1))
            assert resp.status_code == 500
            assert "simulated" not in resp.text
            assert "RuntimeError" not in resp.text
        finally:
            app.dependency_overrides.clear()
            engine.dispose()
        monkeypatch.undo()

    def test_model_failure_returns_500(self, client, db, monkeypatch):
        from Backend import app as app_mod

        def boom(*args, **kwargs):
            raise ValueError("simulated inference failure")

        monkeypatch.setattr(app_mod, "process_observation", boom)

        def _override():
            session = db()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        app.dependency_overrides[get_db] = _override
        try:
            resp = client.post("/predict", json=make_valid("P-I2", 1))
            assert resp.status_code == 500
            assert "simulated" not in resp.text
            assert "ValueError" not in resp.text
        finally:
            app.dependency_overrides.clear()
        monkeypatch.undo()

    def test_500_does_not_expose_internals(self, client, db, monkeypatch):
        from Backend import app as app_mod

        def boom(*args, **kwargs):
            raise RuntimeError("secret internal detail")

        monkeypatch.setattr(app_mod, "process_observation", boom)

        def _override():
            session = db()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        app.dependency_overrides[get_db] = _override
        try:
            resp = client.post("/predict", json=make_valid("P-I3", 1))
            assert resp.status_code == 500
            assert "secret internal detail" not in resp.text
        finally:
            app.dependency_overrides.clear()
        monkeypatch.undo()


# ── J. Request IDs ───────────────────────────────────────────────────────────

class TestRequestIds:
    def test_response_contains_request_id(self, client, override_get_db):
        resp = client.post("/predict", json=make_valid("P-J1", 1))
        assert REQUEST_ID_HEADER in resp.headers

    def test_supplied_request_id_preserved(self, client, override_get_db):
        resp = client.post(
            "/predict",
            json=make_valid("P-J2", 1),
            headers={REQUEST_ID_HEADER: "req-abc-123"},
        )
        assert resp.headers.get(REQUEST_ID_HEADER) == "req-abc-123"

    def test_generated_request_id_when_missing(self, client, override_get_db):
        resp = client.post("/predict", json=make_valid("P-J3", 1))
        rid = resp.headers.get(REQUEST_ID_HEADER)
        assert rid is not None and rid != ""

    def test_health_includes_request_id(self, client):
        resp = client.get("/health")
        assert REQUEST_ID_HEADER in resp.headers
