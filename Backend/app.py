"""FastAPI application — Phase 4 (lifespan) + Phase 6 (ingestion endpoints).

Lifespan loads the frozen HGB model exactly once via config-based path.
The model is stored on ``app.state.model`` for the prediction pipeline.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import joblib
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from Backend.config import settings
from Backend.Database.connection import get_db
from Backend.Database.schema import Observation
from Backend.Services.pred_cache import process_observation
from Backend.Services.validation import Health, PredictionResponse

log = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the HGB model once at startup; store on app.state."""
    log.info("Loading model from %s", settings.model_path)
    app.state.model = joblib.load(settings.model_path)
    log.info("Model loaded — classes=%s, n_features=%d",
             app.state.model.classes_.tolist(),
             app.state.model.n_features_in_)
    yield
    log.info("Shutting down")


app = FastAPI(lifespan=lifespan)


def _request_id(request: Request) -> str:
    """Return the incoming request ID, or generate a new one (Phase 6)."""
    supplied = request.headers.get(REQUEST_ID_HEADER)
    return supplied.strip() if supplied and supplied.strip() else str(uuid.uuid4())


def _latest_observation_iculos(session: Session, patient_id: str) -> int | None:
    """Return the highest stored ICULOS for a patient, or None if none stored."""
    stmt = (
        select(Observation.iculos)
        .where(Observation.patient_id == patient_id)
        .order_by(Observation.iculos.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


@app.get("/")
def home():
    return {"message": "welcome"}


@app.get("/health")
def health(request: Request, response: Response):
    """Liveness / model-availability check (Phase 6)."""
    request_id = _request_id(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    model = getattr(request.app.state, "model", None)
    if model is None:
        log.warning("health degraded — model not loaded (request_id=%s)", request_id)
        raise HTTPException(
            status_code=503,
            detail="model not loaded",
            headers={REQUEST_ID_HEADER: request_id},
        )
    log.info("health ok (request_id=%s)", request_id)
    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.post("/predict",
          response_model=PredictionResponse,
          responses={409: {"description": "ICULOS out of order"}})
def predict(
    request: Request,
    response: Response,
    obs: Health,
    session: Session = Depends(get_db),
):
    """Ingest one hourly observation, run the prediction pipeline, persist, and
    return the current risk + alert state.

    The prediction/alert business logic lives in
    ``pred_cache.process_observation``; this route only orchestrates it.
    """
    request_id = _request_id(request)
    response.headers[REQUEST_ID_HEADER] = request_id

    patient_id = obs.PatientID
    iculos = obs.ICULOS
    log.info("prediction request received — patient_id=%s iculos=%d request_id=%s",
             patient_id, iculos, request_id)

    # ICULOS ordering enforcement (REQ 5). Must happen BEFORE any DB write to
    # avoid silently upserting an older/duplicate observation via the API.
    try:
        latest = _latest_observation_iculos(session, patient_id)
    except Exception:
        log.exception("database read failed during ICULOS check — patient_id=%s "
                      "request_id=%s", patient_id, request_id)
        raise HTTPException(
            status_code=500,
            detail="internal database failure",
            headers={REQUEST_ID_HEADER: request_id},
        )

    if latest is not None and iculos <= latest:
        log.warning("ICULOS order violation — patient_id=%s iculos=%d latest=%d "
                    "request_id=%s", patient_id, iculos, latest, request_id)
        raise HTTPException(
            status_code=409,
            detail=f"ICULOS {iculos} is not greater than the latest stored "
                   f"ICULOS {latest} for patient {patient_id}",
            headers={REQUEST_ID_HEADER: request_id},
        )

    try:
        result = process_observation(
            session,
            obs.model_dump(),
            request.app.state.model,
        )
    except HTTPException:
        raise
    except Exception:
        log.exception("prediction failure — patient_id=%s iculos=%d request_id=%s",
                      patient_id, iculos, request_id)
        raise HTTPException(
            status_code=500,
            detail="internal prediction failure",
            headers={REQUEST_ID_HEADER: request_id},
        )

    log.info("prediction success — patient_id=%s iculos=%d raw_probability=%.4f "
             "request_id=%s", patient_id, iculos, result["raw_probability"],
             request_id)
    return result
