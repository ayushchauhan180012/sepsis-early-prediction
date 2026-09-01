"""SQLAlchemy ORM table definitions — single PostgreSQL database (D-007).

Five logical tables:
    patients        — static demographics
    observations    — authoritative raw hourly patient history (D-008)
    predictions     — per-hour risk scores and alert flags
    alerts          — derived alert events (populated in Phase 5)
    alert_summaries — aggregated alert analytics (queried in Phase 8)

Column names use lowercase PostgreSQL convention (D-022).
Training contract uppercase names are mapped in Python, not in SQL.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    func,
)

from Backend.Database.connection import Base


# ── patients ────────────────────────────────────────────────────────────────

class Patient(Base):
    __tablename__ = "patients"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    patient_id  = Column(String, nullable=False, unique=True, index=True)
    age         = Column(Integer, nullable=False)
    created_at  = Column(DateTime, nullable=False, server_default=func.now())


# ── observations ────────────────────────────────────────────────────────────

class Observation(Base):
    __tablename__ = "observations"
    __table_args__ = (
        UniqueConstraint("patient_id", "iculos", name="uq_observation_patient_iculos"),
        Index("idx_observation_patient_iculos", "patient_id", "iculos"),
    )

    id          = Column(Integer, primary_key=True, autoincrement=True)
    patient_id  = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    iculos      = Column(Integer, nullable=False)

    # Raw vitals
    hr          = Column(Float)
    o2sat       = Column(Float)
    sbp         = Column(Float)
    map         = Column(Float)
    resp        = Column(Float)
    temp        = Column(Float)

    # Raw labs (nullable — sparse testing)
    lactate     = Column(Float)
    wbc         = Column(Float)
    creatinine  = Column(Float)
    platelets   = Column(Float)

    received_at = Column(DateTime, nullable=False, server_default=func.now())


# ── predictions ─────────────────────────────────────────────────────────────

class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("patient_id", "iculos", name="uq_prediction_patient_iculos"),
        Index("idx_prediction_patient_iculos", "patient_id", "iculos"),
    )

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    patient_id          = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    iculos              = Column(Integer, nullable=False)
    raw_probability     = Column(Float, nullable=False)
    filtered_probability = Column(Float, nullable=False)
    high_risk           = Column(Boolean, nullable=False)
    alert               = Column(Boolean, nullable=False)
    predicted_at        = Column(DateTime, nullable=False, server_default=func.now())


# ── alerts ──────────────────────────────────────────────────────────────────

class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (
        Index("idx_alert_patient_start", "patient_id", "alert_start_iculos"),
    )

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    patient_id          = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    alert_start_iculos  = Column(Integer, nullable=False)
    alert_end_iculos    = Column(Integer)
    duration_hours      = Column(Integer)
    peak_risk           = Column(Float, nullable=False)
    created_at          = Column(DateTime, nullable=False, server_default=func.now())


# ── alert_summaries ─────────────────────────────────────────────────────────

class AlertSummary(Base):
    __tablename__ = "alert_summaries"
    __table_args__ = (
        Index("idx_alert_summary_patient", "patient_id"),
    )

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    patient_id          = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    total_alerts        = Column(Integer, nullable=False)
    total_alert_hours   = Column(Integer, nullable=False)
    first_alert_iculos  = Column(Integer)
    last_alert_iculos   = Column(Integer)
    max_peak_risk       = Column(Float)
    generated_at        = Column(DateTime, nullable=False, server_default=func.now())
