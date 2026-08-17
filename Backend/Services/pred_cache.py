"""Prediction pipeline — orchestration layer (Phase 3: features only).

Wires together:
  1. Observation ingestion → Phase 2 ``operations`` (store to DB)
  2. Full history retrieval → Phase 2 ``operations``
  3. Feature engineering → ``feature_engineering`` (deterministic transform)
  4. Returns the 50-feature row ready for model inference (Phase 4)

**Model inference is NOT called here** — that responsibility belongs to
Phase 4 (``Backend/app.py`` lifespan loads the model; this module returns
the feature vector the model will consume).
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from Backend.Database.operations import (
    ensure_patient,
    upsert_observation,
    get_patient_history,
)
from Backend.Services.feature_engineering import (
    observations_to_dataframe,
    transform_patient_history,
    compute_feature_row,
)


def ingest_observation(
    session: Session,
    data: dict,
) -> None:
    """Persist one hourly observation and ensure the patient record exists.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        An open transactional session (caller commits).
    data : dict
        Observation payload with training-contract keys:
        ``PatientID, Age, ICULOS, HR, O2Sat, SBP, MAP, Resp, Temp,
        Lactate, WBC, Creatinine, Platelets``.
    """
    patient_id = data["PatientID"]
    age = data["Age"]
    ensure_patient(session, patient_id, age)
    upsert_observation(session, patient_id, data["ICULOS"],
                       hr=data.get("HR"), o2sat=data.get("O2Sat"),
                       sbp=data.get("SBP"), map=data.get("MAP"),
                       resp=data.get("Resp"), temp=data.get("Temp"),
                       lactate=data.get("Lactate"), wbc=data.get("WBC"),
                       creatinine=data.get("Creatinine"),
                       platelets=data.get("Platelets"))


def build_features(
    session: Session,
    patient_id: str,
    target_iculos: int | None = None,
) -> pd.DataFrame:
    """Retrieve patient history from DB and produce the 50-feature row.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        An open session (read-only; no writes here).
    patient_id : str
        The patient whose features to compute.
    target_iculos : int or None
        ICULOS hour to produce features for.  ``None`` → latest observation.

    Returns
    -------
    pd.DataFrame with exactly one row and the 50 frozen feature columns.

    Raises
    ------
    ValueError
        If the patient has no stored observations.
    """
    history = get_patient_history(session, patient_id)
    if not history:
        raise ValueError(f"No observations stored for patient {patient_id}")

    df = observations_to_dataframe(history)
    return compute_feature_row(df, target_iculos=target_iculos)


def process_observation(
    session: Session,
    data: dict,
) -> pd.DataFrame:
    """End-to-end: ingest → feature-engineer → return feature row.

    This is the main entry point for the Phase 3 pipeline.
    Model inference will be added in Phase 4.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        An open transactional session.
    data : dict
        Observation payload (training-contract keys).

    Returns
    -------
    pd.DataFrame with exactly one row and 50 feature columns.
    """
    ingest_observation(session, data)
    patient_id = data["PatientID"]
    iculos = data["ICULOS"]
    return build_features(session, patient_id, target_iculos=iculos)
