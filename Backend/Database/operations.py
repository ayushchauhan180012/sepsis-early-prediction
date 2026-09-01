"""CRUD operations for the sepsis prediction database.

All writes use UPSERT semantics for (patient_id, iculos) uniqueness (O-4).
Reads return data ordered by ICULOS ASC (D-009).

Column mapping (D-022):
    DB lowercase  →  Python / Training contract uppercase
    hr            →  HR
    o2sat         →  O2Sat
    sbp           →  SBP
    map           →  MAP
    resp          →  Resp
    temp          →  Temp
    lactate       →  Lactate
    wbc           →  WBC
    creatinine    →  Creatinine
    platelets     →  Platelets
"""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from Backend.Database.schema import Patient, Observation, Prediction, Alert, AlertSummary


# ── Column name mapping (D-022) ─────────────────────────────────────────────

_DB_TO_contract = {
    "hr": "HR",
    "o2sat": "O2Sat",
    "sbp": "SBP",
    "map": "MAP",
    "resp": "Resp",
    "temp": "Temp",
    "lactate": "Lactate",
    "wbc": "WBC",
    "creatinine": "Creatinine",
    "platelets": "Platelets",
}

_contract_to_db = {v: k for k, v in _DB_TO_contract.items()}

VITALS_DB = [_contract_to_db[v] for v in ("HR", "O2Sat", "SBP", "MAP", "Resp", "Temp")]
LABS_DB   = [_contract_to_db[v] for v in ("Lactate", "WBC", "Creatinine", "Platelets")]


def contract_name_to_db(name: str) -> str:
    """Map a training-contract uppercase name to its DB lowercase column."""
    return _contract_to_db.get(name, name.lower())


def db_name_to_contract(name: str) -> str:
    """Map a DB lowercase column name to its training-contract uppercase name."""
    return _DB_TO_contract.get(name, name)


# ── patients ─────────────────────────────────────────────────────────────────

def ensure_patient(session: Session, patient_id: str, age: int) -> Patient:
    """Return existing patient or create a new one.  Idempotent."""
    stmt = select(Patient).where(Patient.patient_id == patient_id)
    patient = session.execute(stmt).scalar_one_or_none()
    if patient is None:
        patient = Patient(patient_id=patient_id, age=age)
        session.add(patient)
        session.flush()
    return patient


def get_patient(session: Session, patient_id: str) -> Patient | None:
    stmt = select(Patient).where(Patient.patient_id == patient_id)
    return session.execute(stmt).scalar_one_or_none()


# ── observations ─────────────────────────────────────────────────────────────

def upsert_observation(
    session: Session,
    patient_id: str,
    iculos: int,
    *,
    hr: float | None = None,
    o2sat: float | None = None,
    sbp: float | None = None,
    map: float | None = None,
    resp: float | None = None,
    temp: float | None = None,
    lactate: float | None = None,
    wbc: float | None = None,
    creatinine: float | None = None,
    platelets: float | None = None,
) -> None:
    """Insert or update a single hourly observation (O-4: UPSERT)."""
    stmt = pg_insert(Observation).values(
        patient_id=patient_id,
        iculos=iculos,
        hr=hr,
        o2sat=o2sat,
        sbp=sbp,
        map=map,
        resp=resp,
        temp=temp,
        lactate=lactate,
        wbc=wbc,
        creatinine=creatinine,
        platelets=platelets,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["patient_id", "iculos"],
        set_={
            "hr": stmt.excluded.hr,
            "o2sat": stmt.excluded.o2sat,
            "sbp": stmt.excluded.sbp,
            "map": stmt.excluded.map,
            "resp": stmt.excluded.resp,
            "temp": stmt.excluded.temp,
            "lactate": stmt.excluded.lactate,
            "wbc": stmt.excluded.wbc,
            "creatinine": stmt.excluded.creatinine,
            "platelets": stmt.excluded.platelets,
        },
    )
    session.execute(stmt)


def insert_observation_from_dict(session: Session, patient_id: str, data: dict) -> None:
    """Insert an observation from a dict with uppercase training-contract keys.

    Expected keys: HR, O2Sat, SBP, MAP, Resp, Temp, Lactate, WBC,
    Creatinine, Platelets, ICULOS, Age, PatientID.
    Missing optional keys (labs) are treated as None.
    """
    iculos = data["ICULOS"]
    kwargs = {}
    for contract_name in ("HR", "O2Sat", "SBP", "MAP", "Resp", "Temp",
                          "Lactate", "WBC", "Creatinine", "Platelets"):
        db_col = _contract_to_db[contract_name]
        kwargs[db_col] = data.get(contract_name)
    upsert_observation(session, patient_id, iculos, **kwargs)


def get_patient_history(session: Session, patient_id: str) -> list[Observation]:
    """Return the patient's full chronological observation history (D-008).

    Ordered by ICULOS ASC (D-009).  Never deletes old observations.
    """
    stmt = (
        select(Observation)
        .where(Observation.patient_id == patient_id)
        .order_by(Observation.iculos.asc())
    )
    return list(session.execute(stmt).scalars().all())


def count_observations(session: Session, patient_id: str) -> int:
    stmt = (
        select(text("count(*)"))
        .select_from(Observation)
        .where(Observation.patient_id == patient_id)
    )
    return session.execute(stmt).scalar_one()


def delete_patient_observations(session: Session, patient_id: str) -> int:
    """Delete all observations for a patient (test helper only)."""
    stmt = text("DELETE FROM observations WHERE patient_id = :pid")
    result = session.execute(stmt, {"pid": patient_id})
    return result.rowcount


def delete_patient(session: Session, patient_id: str) -> None:
    """Delete a patient and all cascade-linked rows (test helper only)."""
    session.execute(text("DELETE FROM observations WHERE patient_id = :pid"), {"pid": patient_id})
    session.execute(text("DELETE FROM predictions WHERE patient_id = :pid"), {"pid": patient_id})
    session.execute(text("DELETE FROM alerts WHERE patient_id = :pid"), {"pid": patient_id})
    session.execute(text("DELETE FROM alert_summaries WHERE patient_id = :pid"), {"pid": patient_id})
    session.execute(text("DELETE FROM patients WHERE patient_id = :pid"), {"pid": patient_id})


# ── predictions ──────────────────────────────────────────────────────────────

def upsert_prediction(
    session: Session,
    patient_id: str,
    iculos: int,
    *,
    raw_probability: float,
    filtered_probability: float,
    high_risk: bool,
    alert: bool,
) -> None:
    """Insert or update a single hourly prediction (O-4: UPSERT)."""
    stmt = pg_insert(Prediction).values(
        patient_id=patient_id,
        iculos=iculos,
        raw_probability=raw_probability,
        filtered_probability=filtered_probability,
        high_risk=high_risk,
        alert=alert,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["patient_id", "iculos"],
        set_={
            "raw_probability": stmt.excluded.raw_probability,
            "filtered_probability": stmt.excluded.filtered_probability,
            "high_risk": stmt.excluded.high_risk,
            "alert": stmt.excluded.alert,
        },
    )
    session.execute(stmt)


def get_patient_predictions(session: Session, patient_id: str) -> list[Prediction]:
    """Return all predictions for a patient, ordered by ICULOS ASC."""
    stmt = (
        select(Prediction)
        .where(Prediction.patient_id == patient_id)
        .order_by(Prediction.iculos.asc())
    )
    return list(session.execute(stmt).scalars().all())


def get_latest_prediction(session: Session, patient_id: str) -> Prediction | None:
    """Return the most recent prediction for a patient, or None."""
    stmt = (
        select(Prediction)
        .where(Prediction.patient_id == patient_id)
        .order_by(Prediction.iculos.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def update_prediction_alert_fields(
    session: Session,
    patient_id: str,
    iculos: int,
    *,
    filtered_probability: float,
    high_risk: bool,
    alert: bool,
) -> None:
    """Update alert-state fields on an existing prediction row.

    raw_probability is NOT updated — it is the immutable source of truth.
    """
    stmt = (
        select(Prediction)
        .where(Prediction.patient_id == patient_id)
        .where(Prediction.iculos == iculos)
    )
    pred = session.execute(stmt).scalar_one_or_none()
    if pred is not None:
        pred.filtered_probability = filtered_probability
        pred.high_risk = high_risk
        pred.alert = alert


def update_prediction_alert_batch(
    session: Session,
    patient_id: str,
    alert_states: list,
) -> None:
    """Update alert-state fields for a batch of predictions.

    Parameters
    ----------
    patient_id : str
    alert_states : list of ``alert_engine.AlertState`` objects.
        Each must have ``iculos``, ``filtered_probability``, ``high_risk``,
        ``alert``.  The patient's prediction rows must already exist.
    """
    for state in alert_states:
        update_prediction_alert_fields(
            session,
            patient_id,
            state.iculos,
            filtered_probability=state.filtered_probability,
            high_risk=state.high_risk,
            alert=state.alert,
        )


# ── alerts ────────────────────────────────────────────────────────────────────

def insert_alert(
    session: Session,
    patient_id: str,
    alert_start_iculos: int,
    alert_end_iculos: int,
    *,
    peak_risk: float,
) -> Alert:
    """Insert a single alert event row."""
    alert = Alert(
        patient_id=patient_id,
        alert_start_iculos=alert_start_iculos,
        alert_end_iculos=alert_end_iculos,
        duration_hours=alert_end_iculos - alert_start_iculos + 1,
        peak_risk=peak_risk,
    )
    session.add(alert)
    session.flush()
    return alert


def get_patient_alerts(session: Session, patient_id: str) -> list[Alert]:
    """Return alert events for a patient, ordered by start ICULOS ASC."""
    stmt = (
        select(Alert)
        .where(Alert.patient_id == patient_id)
        .order_by(Alert.alert_start_iculos.asc())
    )
    return list(session.execute(stmt).scalars().all())


def rebuild_alert_events(
    session: Session,
    patient_id: str,
) -> list[Alert]:
    """Delete existing alert events and rebuild from persisted predictions.

    An alert event is a maximal contiguous run of ``alert=True`` rows
    (D-017).  Idempotent — running twice produces the same result.
    """
    # Delete existing events for this patient
    session.execute(
        text("DELETE FROM alerts WHERE patient_id = :pid"),
        {"pid": patient_id},
    )

    predictions = get_patient_predictions(session, patient_id)

    # Find maximal contiguous runs of alert=True
    events: list[Alert] = []
    run_start: int | None = None
    run_peak: float = 0.0

    for pred in predictions:
        if pred.alert:
            if run_start is None:
                run_start = pred.iculos
                run_peak = pred.raw_probability
            else:
                run_peak = max(run_peak, pred.raw_probability)
        else:
            if run_start is not None:
                events.append(Alert(
                    patient_id=patient_id,
                    alert_start_iculos=run_start,
                    alert_end_iculos=pred.iculos - 1,
                    duration_hours=(pred.iculos - 1) - run_start + 1,
                    peak_risk=run_peak,
                ))
                run_start = None
                run_peak = 0.0

    # Close any open run at end of history
    if run_start is not None:
        events.append(Alert(
            patient_id=patient_id,
            alert_start_iculos=run_start,
            alert_end_iculos=predictions[-1].iculos,
            duration_hours=predictions[-1].iculos - run_start + 1,
            peak_risk=run_peak,
        ))

    for event in events:
        session.add(event)
    session.flush()
    return events


# ── alert summaries ───────────────────────────────────────────────────────────

def upsert_alert_summary(
    session: Session,
    patient_id: str,
) -> AlertSummary | None:
    """Rebuild the alert summary from alert events for a patient.

    If no alerts exist, deletes any stale summary and returns None.
    Idempotent — running twice produces the same result.
    """
    alerts = get_patient_alerts(session, patient_id)

    # Delete stale summary
    session.execute(
        text("DELETE FROM alert_summaries WHERE patient_id = :pid"),
        {"pid": patient_id},
    )

    if not alerts:
        return None

    total_alerts = len(alerts)
    total_alert_hours = sum(a.duration_hours for a in alerts)
    first_alert_iculos = alerts[0].alert_start_iculos
    last_alert_iculos = alerts[-1].alert_end_iculos
    max_peak_risk = max(a.peak_risk for a in alerts)

    summary = AlertSummary(
        patient_id=patient_id,
        total_alerts=total_alerts,
        total_alert_hours=total_alert_hours,
        first_alert_iculos=first_alert_iculos,
        last_alert_iculos=last_alert_iculos,
        max_peak_risk=max_peak_risk,
    )
    session.add(summary)
    session.flush()
    return summary


# ── analytics queries (Phase 8) ─────────────────────────────────────────────

def get_risk_trajectory(session: Session, patient_id: str) -> list[dict]:
    """Return the patient's full risk trajectory as a list of dicts.

    Each dict contains: ``iculos``, ``raw_probability``,
    ``filtered_probability``, ``high_risk``, ``alert``.
    Ordered by ICULOS ASC (D-009).  Returns an empty list if the patient
    has no persisted predictions.
    """
    preds = get_patient_predictions(session, patient_id)
    return [
        {
            "iculos": p.iculos,
            "raw_probability": p.raw_probability,
            "filtered_probability": p.filtered_probability,
            "high_risk": p.high_risk,
            "alert": p.alert,
        }
        for p in preds
    ]


def get_peak_risk(session: Session, patient_id: str) -> dict | None:
    """Return peak (maximum) raw risk and its ICULOS for a patient.

    Returns ``{"peak_risk": float, "iculos": int}`` or ``None`` if the
    patient has no persisted predictions.
    """
    preds = get_patient_predictions(session, patient_id)
    if not preds:
        return None
    peak = max(preds, key=lambda p: p.raw_probability)
    return {"peak_risk": peak.raw_probability, "iculos": peak.iculos}


def get_alert_statistics(session: Session, patient_id: str) -> dict | None:
    """Return aggregated alert statistics for a patient.

    Reads directly from the ``alert_summaries`` table (populated by
    ``upsert_alert_summary`` on every ``/predict`` call).

    Returns a dict with keys: ``total_alerts``, ``total_alert_hours``,
    ``first_alert_iculos``, ``last_alert_iculos``, ``max_peak_risk``.
    Returns ``None`` if no summary exists (patient has no alerts).
    """
    stmt = select(AlertSummary).where(AlertSummary.patient_id == patient_id)
    summary = session.execute(stmt).scalar_one_or_none()
    if summary is None:
        return None
    return {
        "total_alerts": summary.total_alerts,
        "total_alert_hours": summary.total_alert_hours,
        "first_alert_iculos": summary.first_alert_iculos,
        "last_alert_iculos": summary.last_alert_iculos,
        "max_peak_risk": summary.max_peak_risk,
    }


# ── test helpers ──────────────────────────────────────────────────────────────

def delete_patient_alerts(session: Session, patient_id: str) -> int:
    """Delete all alert events for a patient (test helper only)."""
    result = session.execute(
        text("DELETE FROM alerts WHERE patient_id = :pid"),
        {"pid": patient_id},
    )
    return result.rowcount


def delete_patient_alert_summaries(session: Session, patient_id: str) -> int:
    """Delete alert summaries for a patient (test helper only)."""
    result = session.execute(
        text("DELETE FROM alert_summaries WHERE patient_id = :pid"),
        {"pid": patient_id},
    )
    return result.rowcount
