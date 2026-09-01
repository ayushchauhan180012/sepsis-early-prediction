"""Phase 9 Step 3 correctness tests — readiness endpoint (D-028).

Covers:
  A. Healthy model + DB -> 200 with expected checks structure
  B. Missing model -> 503 with model degraded
  C. Database failure -> 503 with database degraded
  D. Response structure / status fields
  E. X-Request-ID present

The tests reuse the established in-memory-SQLite + StaticPool pattern (shared
connection across per-request sessions) and the existing ``get_db`` dependency
override, exactly as ``test_phase6_api.py`` does.  No real PostgreSQL server is
required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from Backend.app import app
from Backend.Database.connection import get_db

REQUEST_ID_HEADER = "X-Request-ID"


@pytest.fixture()
def db(session_factory):
    """A session factory bound to a shared StaticPool in-memory SQLite DB.

    The shared connection (StaticPool) lets per-request sessions created by the
    ``get_db`` dependency override observe the same persisted data.
    """
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


class TestHealthy:
    def test_healthy_returns_200(self, client, override_get_db):
        resp = client.get("/health/ready")
        assert resp.status_code == 200

    def test_healthy_checks_structure(self, client, override_get_db):
        resp = client.get("/health/ready")
        body = resp.json()
        assert body["status"] == "ok"
        assert body["checks"] == {"model": "ok", "database": "ok"}


class TestMissingModel:
    def test_missing_model_returns_503(self, client, override_get_db):
        saved = app.state.model
        app.state.model = None
        try:
            resp = client.get("/health/ready")
            assert resp.status_code == 503
        finally:
            app.state.model = saved

    def test_missing_model_marks_model_degraded(self, client, override_get_db):
        saved = app.state.model
        app.state.model = None
        try:
            body = client.get("/health/ready").json()
            assert body["checks"]["model"] == "degraded"
            assert body["checks"]["database"] == "ok"
        finally:
            app.state.model = saved


class TestDatabaseFailure:
    def test_db_failure_returns_503(self, client, monkeypatch):
        class _BoomSession:
            def execute(self, *args, **kwargs):
                raise RuntimeError("simulated DB failure")

            def close(self):
                pass

        def _override():
            yield _BoomSession()

        app.dependency_overrides[get_db] = _override
        try:
            resp = client.get("/health/ready")
            assert resp.status_code == 503
        finally:
            app.dependency_overrides.clear()

    def test_db_failure_marks_database_degraded(self, client, monkeypatch):
        class _BoomSession:
            def execute(self, *args, **kwargs):
                raise RuntimeError("simulated")

            def close(self):
                pass

        def _override():
            yield _BoomSession()

        app.dependency_overrides[get_db] = _override
        try:
            body = client.get("/health/ready").json()
            assert body["checks"]["database"] == "degraded"
            assert body["checks"]["model"] == "ok"
        finally:
            app.dependency_overrides.clear()

    def test_db_failure_does_not_leak_internals(self, client, monkeypatch):
        class _BoomSession:
            def execute(self, *args, **kwargs):
                raise RuntimeError("secret internal detail")

            def close(self):
                pass

        def _override():
            yield _BoomSession()

        app.dependency_overrides[get_db] = _override
        try:
            resp = client.get("/health/ready")
            assert resp.status_code == 503
            assert "secret internal detail" not in resp.text
            assert "RuntimeError" not in resp.text
        finally:
            app.dependency_overrides.clear()


class TestRequestId:
    def test_readiness_includes_request_id(self, client, override_get_db):
        resp = client.get("/health/ready")
        assert REQUEST_ID_HEADER in resp.headers

    def test_supplied_request_id_preserved(self, client, override_get_db):
        resp = client.get(
            "/health/ready",
            headers={REQUEST_ID_HEADER: "req-ready-123"},
        )
        assert resp.headers.get(REQUEST_ID_HEADER) == "req-ready-123"


class TestHealthUnchanged:
    def test_legacy_health_still_returns_200(self, client, override_get_db):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "model_loaded": True}
