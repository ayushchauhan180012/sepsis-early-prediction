"""Phase 9 Step 4 correctness tests — notification dispatch in /predict (D-027).

Covers:
  A. alert=False        -> no notification is scheduled
  B. alert=True         -> notification scheduled with patient_id + alert_data
  C. dispatch boundary  -> notification exceptions are caught and logged
  D. predict stays 200  -> notification failure never fails the prediction
  E. NoOp default       -> default channel runs without side effects
  F. predict unchanged  -> response shape/persistence/request-ID regressions

Mock strategy: the route references ``process_observation`` and ``_notify`` by
module attribute, so ``monkeypatch.setattr`` on ``Backend.app`` makes the tests
deterministic (no timing/sleeps). TestClient/Starlette runs background tasks as
part of the request lifecycle, so assertions are valid immediately after the
request returns.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from Backend import app as app_mod
from Backend.app import app
from Backend.Database.connection import get_db

REQUEST_ID_HEADER = "X-Request-ID"


@pytest.fixture()
def db(session_factory):
    """A session factory bound to a shared StaticPool in-memory SQLite DB."""
    return session_factory


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


def _alert_result(patient_id: str = "P-N", iculos: int = 1, alert: bool = True) -> dict:
    return {
        "patient_id": patient_id,
        "iculos": iculos,
        "raw_probability": 0.120,
        "filtered_probability": 0.120,
        "high_risk": True,
        "alert": alert,
    }


class _CapturingChannel:
    """Fake channel that records send() calls (no real notification)."""

    def __init__(self):
        self.calls = []

    def send(self, patient_id: str, alert_data: dict) -> None:
        self.calls.append((patient_id, alert_data))


class _RaisingChannel:
    """Fake channel whose send() always raises."""

    def send(self, patient_id: str, alert_data: dict) -> None:
        raise RuntimeError("simulated notification failure")


# ─── A. alert=False → no notification ──────────────────────────────────────

class TestAlertFalse:
    def test_no_notification_when_alert_false(self, client, override_get_db, monkeypatch):
        captured = []
        monkeypatch.setattr(
            app_mod, "_notify", lambda *a, **k: captured.append((a, k))
        )
        monkeypatch.setattr(
            app_mod, "process_observation",
            lambda *a, **k: _alert_result(alert=False),
        )
        resp = client.post("/predict", json=make_obs("P-FALSE", 1))
        assert resp.status_code == 200
        assert captured == []


# ─── B. alert=True → notification scheduled ────────────────────────────────

class TestAlertTrue:
    def test_notification_scheduled_when_alert_true(
        self, client, override_get_db, monkeypatch
    ):
        captured = []
        monkeypatch.setattr(
            app_mod, "_notify", lambda *a, **k: captured.append((a, k))
        )
        result = _alert_result("P-TRUE", 1, alert=True)
        monkeypatch.setattr(
            app_mod, "process_observation", lambda *a, **k: result
        )

        resp = client.post("/predict", json=make_obs("P-TRUE", 1))
        assert resp.status_code == 200
        assert len(captured) == 1

    def test_correct_patient_id_passed(self, client, override_get_db, monkeypatch):
        captured = []
        monkeypatch.setattr(
            app_mod, "_notify", lambda *a, **k: captured.append((a, k))
        )
        result = _alert_result("P-PATIENT", 1, alert=True)
        monkeypatch.setattr(
            app_mod, "process_observation", lambda *a, **k: result
        )

        client.post("/predict", json=make_obs("P-PATIENT", 1))
        assert captured[0][0][0] == "P-PATIENT"

    def test_correct_alert_data_passed(self, client, override_get_db, monkeypatch):
        captured = []
        monkeypatch.setattr(
            app_mod, "_notify", lambda *a, **k: captured.append((a, k))
        )
        result = _alert_result("P-DATA", 1, alert=True)
        monkeypatch.setattr(
            app_mod, "process_observation", lambda *a, **k: result
        )

        client.post("/predict", json=make_obs("P-DATA", 1))
        assert captured[0][0][1] == result

    def test_channel_send_receives_data(self, client, override_get_db, monkeypatch):
        channel = _CapturingChannel()
        monkeypatch.setattr(
            app_mod, "get_notification_channel", lambda *a, **k: channel
        )
        result = _alert_result("P-CH", 1, alert=True)
        monkeypatch.setattr(
            app_mod, "process_observation", lambda *a, **k: result
        )

        client.post("/predict", json=make_obs("P-CH", 1))
        assert channel.calls == [("P-CH", result)]


# ─── C. Dispatch boundary catches exceptions ────────────────────────────────

class TestDispatchBoundary:
    def test_notification_exception_is_caught(self, monkeypatch, caplog):
        monkeypatch.setattr(
            app_mod, "get_notification_channel", lambda *a, **k: _RaisingChannel()
        )
        caplog.set_level(logging.ERROR, logger="Backend.app")

        app_mod._notify("P-ERR", _alert_result())

        assert "notification dispatch failed" in caplog.text

    def test_notification_exception_does_not_propagate(self, client, override_get_db, monkeypatch):
        monkeypatch.setattr(
            app_mod, "get_notification_channel", lambda *a, **k: _RaisingChannel()
        )
        result = _alert_result("P-PROPG", 1, alert=True)
        monkeypatch.setattr(
            app_mod, "process_observation", lambda *a, **k: result
        )

        resp = client.post("/predict", json=make_obs("P-PROPG", 1))
        assert resp.status_code == 200
        assert resp.json() == result


# ─── D. Prediction stays successful on notification failure ──────────────────

class TestPredictSurvivesNotification:
    def test_response_ok_and_request_id_present(self, client, override_get_db, monkeypatch):
        monkeypatch.setattr(
            app_mod, "get_notification_channel", lambda *a, **k: _RaisingChannel()
        )
        monkeypatch.setattr(
            app_mod, "process_observation",
            lambda *a, **k: _alert_result("P-SURV", 1, alert=True),
        )

        resp = client.post("/predict", json=make_obs("P-SURV", 1))
        assert resp.status_code == 200
        assert REQUEST_ID_HEADER in resp.headers


# ─── E. NoOp remains the default ─────────────────────────────────────────────

class TestNoOpDefault:
    def test_default_channel_is_noop(self):
        from Backend.Services.notifications import (
            get_notification_channel,
            NoOpNotification,
        )
        assert isinstance(get_notification_channel(), NoOpNotification)

    def test_route_runs_with_default_noop_channel(self, client, override_get_db, monkeypatch):
        # Do NOT patch the channel: the default (NoOp) must run without error.
        result = _alert_result("P-NOOP", 1, alert=True)
        monkeypatch.setattr(
            app_mod, "process_observation", lambda *a, **k: result
        )

        resp = client.post("/predict", json=make_obs("P-NOOP", 1))
        assert resp.status_code == 200
        assert resp.json() == result


# ─── F. /predict response unchanged (regression) ─────────────────────────────

class TestPredictUnchanged:
    def test_real_predict_response_shape_unchanged(self, client, override_get_db):
        resp = client.post("/predict", json=make_obs("P-RUN", 1))
        assert resp.status_code == 200
        assert set(resp.json().keys()) == {
            "patient_id", "iculos", "raw_probability",
            "filtered_probability", "high_risk", "alert",
        }

    def test_real_predict_request_id_present(self, client, override_get_db):
        resp = client.post("/predict", json=make_obs("P-RID", 1))
        assert REQUEST_ID_HEADER in resp.headers