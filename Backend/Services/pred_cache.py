"""Prediction pipeline — orchestration layer (Phases 3–5).

Wires together:
  1. Observation ingestion → Phase 2 ``operations`` (store to DB)
  2. Full history retrieval → Phase 2 ``operations``
  3. Feature engineering → ``feature_engineing`` (deterministic transform)
  4. Model inference → ``predict_risk`` (HGB ``predict_proba``)
  5. Prediction persistence → Phase 2 ``operations`` (``upsert_prediction``)
  6. Alert state recomputation → ``alert_engine`` (Phase 5, recompute-from-history)
  7. Alert event rebuild → ``operations.rebuild_alert_events``
  8. Alert summary rebuild → ``operations.upsert_alert_summary``
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
    get_patient_predictions,
    update_prediction_alert_batch,
    rebuild_alert_events,
    upsert_alert_summary,
)
from Backend.Services.feature_engineering import (
    observations_to_dataframe,
    compute_feature_row,
)
from Backend.Services.alert_engine import evaluate_alert_state


# ── Phase 3: ingest + features ───────────────────────────────────────────────

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


# ── End-to-end orchestration (Phases 3 + 4 + 5) ─────────────────────────────

def process_observation(
    session: Session,
    data: dict,
    model,
) -> dict:
    """End-to-end: ingest → features → inference → persist → alert recompute.

    Flow:
        1. Persist observation
        2. Build 50-feature row from full patient history
        3. Model inference → raw probability
        4. Persist raw prediction (alert fields are temporary placeholders)
        5. Retrieve complete patient prediction history
        6. Recompute alert state from raw probabilities (Phase 5)
        7. Update ALL prediction alert-state fields
        8. Rebuild alert events (maximal contiguous runs)
        9. Rebuild alert summary
        10. Return current prediction state

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

    # 4. Persist raw prediction (temporary alert placeholders — overwritten in step 7)
    upsert_prediction(
        session,
        patient_id,
        iculos,
        raw_probability=raw_probability,
        filtered_probability=raw_probability,
        high_risk=False,
        alert=False,
    )

    # 5. Retrieve complete patient prediction history
    all_preds = get_patient_predictions(session, patient_id)

    # 6. Recompute alert state from raw probabilities
    predictions_input = [(p.iculos, p.raw_probability) for p in all_preds]
    alert_states = evaluate_alert_state(predictions_input)

    # 7. Update ALL prediction alert-state fields
    update_prediction_alert_batch(session, patient_id, alert_states)

    # 8. Rebuild alert events (maximal contiguous runs of alert=True)
    rebuild_alert_events(session, patient_id)

    # 9. Rebuild alert summary
    upsert_alert_summary(session, patient_id)

    # 10. Return current prediction state
    current = alert_states[-1]
    return {
        "patient_id": patient_id,
        "iculos": current.iculos,
        "raw_probability": current.raw_probability,
        "filtered_probability": current.filtered_probability,
        "high_risk": current.high_risk,
        "alert": current.alert,
    }
