"""FastAPI application — Phase 4.

Lifespan loads the frozen HGB model exactly once via config-based path.
The model is stored on ``app.state.model`` for the prediction pipeline.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI

from Backend.config import settings

log = logging.getLogger(__name__)


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


@app.get("/")
def home():
    return {"message": "welcome"}
