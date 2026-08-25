"""Prediction pipeline — orchestration layer (Phase 4: features + inference).

Wires together:
  1. Observation ingestion → Phase 2 ``operations`` (store to DB)
  2. Full history retrieval → Phase 2 ``operations``
  3. Feature engineering → ``feature_engineering`` (deterministic transform)
  4. Model inference → ``predict_risk`` (HGB ``predict_proba``)
  5. Prediction persistence → Phase 2 ``operations`` (``upsert_prediction``)

**Alert engine is NOT implemented here** — Phase 5 will replace the
temporary pass-through filter logic (``filtered_probability = raw_probability``,
``alert = False``).
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from Backend.Database.operations import (
    ensure_patient,
    get_patient,
    upsert_observation,
    get_patient_history,
    upsert_prediction,
)
from Backend.Services.feature_engineering import (
    observations_to_dataframe,
    transform_patient_history,
    compute_feature_row,
)


# ── Phase 3: ingest + features (unchanged) ──────────────────────────────────

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

    # Observation ORM objects do not carry age (it lives on the Patient table).
    # Inject it here so feature engineering has the required Age column.
    if "Age" not in df.columns or df["Age"].isna().all():
        patient = get_patient(session, patient_id)
        if patient is not None:
            df["Age"] = patient.age

    return compute_feature_row(df, target_iculos=target_iculos)


# ── Phase 4: model inference ─────────────────────────────────────────────────

def predict_risk(model, feature_row: pd.DataFrame) -> float:
    """Run HGB model inference on a single 50-feature row.

    Parameters
    ----------
    model : HistGradientBoostingClassifier
        The frozen model loaded via ``joblib.load`` (lifespan or test).
    feature_row : pd.DataFrame
        A single-row DataFrame with the 50 frozen feature columns.

    Returns
    -------
    float
        The sepsis risk probability from ``predict_proba(X)[:, 1]``.
    """
    proba = model.predict_proba(feature_row)
    return float(proba[:, 1][0])


# ── Phase 4: end-to-end orchestration ────────────────────────────────────────

def process_observation(
    session: Session,
    data: dict,
    model,
) -> dict:
    """End-to-end: ingest → features → inference → persist prediction.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        An open transactional session.
    data : dict
        Observation payload (training-contract keys).
    model : HistGradientBoostingClassifier
        The frozen model instance (passed explicitly, no global state).

    Returns
    -------
    dict with keys:
        ``patient_id``, ``iculos``, ``raw_probability``,
        ``filtered_probability``, ``high_risk``, ``alert``.
    """
    # 1. Persist observation
    ingest_observation(session, data)

    # 2. Retrieve full history + build 50-feature row
    patient_id = data["PatientID"]
    iculos = data["ICULOS"]
    feature_row = build_features(session, patient_id, target_iculos=iculos)

    # 3. Model inference
    raw_probability = predict_risk(model, feature_row)

    # ── Temporary pass-through for Phase 5 alert engine fields ────────────
    # Phase 5 will implement: uncertainty band → threshold → persistence →
    # cooldown.  For now, filtered_probability = raw_probability (no
    # filtering), high_risk uses the threshold only (no persistence/cooldown),
    # and alert is always False (no alert engine yet).
    filtered_probability = raw_probability
    high_risk = filtered_probability >= 0.045  # placeholder — Phase 5 will own this
    alert = False  # placeholder — Phase 5 will implement the alert engine
    # ── End temporary pass-through ────────────────────────────────────────

    # 4. Persist prediction
    upsert_prediction(
        session,
        patient_id,
        iculos,
        raw_probability=raw_probability,
        filtered_probability=filtered_probability,
        high_risk=high_risk,
        alert=alert,
    )

    return {
        "patient_id": patient_id,
        "iculos": iculos,
        "raw_probability": raw_probability,
        "filtered_probability": filtered_probability,
        "high_risk": high_risk,
        "alert": alert,
    }
