"""Phase 10.1 correctness tests — CORS + schema bootstrap.

Covers:
  A. CORS middleware — allowed development origins receive CORS headers,
     disallowed origins do not, and no wildcard ``*`` is configured.
  B. Schema bootstrap — the lifespan attempts to create the DB schema at
     startup and degrades gracefully (does not abort startup) when the
     database is unreachable, preserving SQLite test compatibility and the
     existing health/readiness behavior.

These tests reuse the established in-memory-SQLite + StaticPool pattern and
the existing ``get_db`` dependency override, exactly as the Phase 6/9 tests do.
No real PostgreSQL server is required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from Backend.app import app
from Backend.Database.connection import get_db

DEV_ORIGIN = "http://localhost:3000"
DEV_ORIGIN_ALT = "http://127.0.0.1:3000"
DISALLOWED_ORIGIN = "http://evil.example.com"


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


# ── A. CORS ──────────────────────────────────────────────────────────────────

class TestCors:
    def test_allowed_origin_receives_cors_headers(self, client):
        resp = client.get("/health", headers={"Origin": DEV_ORIGIN})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == DEV_ORIGIN

    def test_alternative_allowed_origin_receives_cors_headers(self, client):
        resp = client.get("/health", headers={"Origin": DEV_ORIGIN_ALT})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == DEV_ORIGIN_ALT

    def test_disallowed_origin_gets_no_cors_header(self, client):
        resp = client.get("/health", headers={"Origin": DISALLOWED_ORIGIN})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") is None

    def test_no_wildcard_origin_configured(self):
        origins = _cors_allow_origins()
        assert "*" not in [str(o) for o in origins]

    def test_configured_origins_are_development_origins(self):
        origins = [str(o) for o in _cors_allow_origins()]
        assert DEV_ORIGIN in origins
        assert DEV_ORIGIN_ALT in origins

    def test_cors_response_on_predict_allowed_origin(self, client, override_get_db):
        resp = client.options(
            "/predict",
            headers={
                "Origin": DEV_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code == 200
        allow_methods = resp.headers.get("access-control-allow-methods", "")
        assert "POST" in allow_methods


def _cors_middleware_kwargs():
    for mw in app.user_middleware:
        if mw.cls.__name__ == "CORSMiddleware":
            return mw.kwargs
    return None


def _cors_allow_origins():
    return _cors_middleware_kwargs()["allow_origins"]


# ── B. Schema bootstrap ──────────────────────────────────────────────────────

class TestSchemaBootstrap:
    def test_lifespan_attempts_create_all_tables(self, monkeypatch):
        """Lifespan calls create_all_tables on startup without crashing."""
        from Backend import app as app_mod

        called = []
        monkeypatch.setattr(app_mod, "create_all_tables",
                            lambda: called.append(True))
        # Re-instantiate lifespan so the patched function is used.
        with TestClient(app) as test_client:
            assert called, "create_all_tables was not invoked during startup"
            resp = test_client.get("/health")
            assert resp.status_code == 200

    def test_schema_bootstrap_failure_does_not_abort_startup(self, monkeypatch):
        """If DB bootstrap fails, the app still starts and health works."""
        from Backend import app as app_mod

        def boom():
            raise RuntimeError("database unreachable")

        monkeypatch.setattr(app_mod, "create_all_tables", boom)
        with TestClient(app) as test_client:
            resp = test_client.get("/health")
            assert resp.status_code == 200

    def test_readiness_reflects_failed_bootstrap(self, client, override_get_db):
        """If bootstrap fails, readiness reports database degraded."""
        class _BoomSession:
            def execute(self, *args, **kwargs):
                raise RuntimeError("simulated DB failure")

            def close(self):
                pass

        def _override():
            yield _BoomSession()

        app.dependency_overrides[get_db] = _override
        try:
            body = client.get("/health/ready").json()
            assert body["checks"]["database"] == "degraded"
        finally:
            app.dependency_overrides.clear()
