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

from Backend.Database.schema import Patient, Observation, Prediction


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
