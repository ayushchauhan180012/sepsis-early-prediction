"""Phase 7 — Pydantic request/response validation tests.

Covers ``Backend.Services.validation``: the ``Health`` ingest schema and the
``PredictionResponse`` output schema.  These tests bind the API-level field
constraints to the frozen training contract (physiologic ranges, bounded
probabilities, ICULOS >= 1, non-negative labs).

No production code is modified by these tests — they only assert the existing
schemas' behavior.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from Backend.Services.validation import Health, PredictionResponse


def _base_payload(**overrides) -> dict:
    payload = {
        "HR": 88,
        "O2Sat": 97,
        "SBP": 118,
        "MAP": 78,
        "Resp": 18,
        "Temp": 36.9,
        "Lactate": 1.2,
        "WBC": 8.1,
        "Creatinine": 0.9,
        "Platelets": 210.0,
        "Age": 64,
        "ICULOS": 5,
        "PatientID": "p100000",
    }
    payload.update(overrides)
    return payload


# ── vital bounds (field constraints) ────────────────────────────────────────
# (field, too_low, lowest_valid, highest_valid, too_high)
VITAL_BOUNDS = [
    ("HR", 19, 20, 250, 251),
    ("O2Sat", -1, 0, 100, 101),
    ("SBP", 39, 40, 300, 301),
    ("MAP", 19, 20, 250, 251),
    ("Resp", -1, 1, 80, 81),
    ("Temp", 29.0, 30.0, 43.0, 44.0),
]


@pytest.mark.parametrize(
    "field,too_low,low_ok,high_ok,too_high",
    VITAL_BOUNDS,
    ids=[v[0] for v in VITAL_BOUNDS],
)
def test_vital_bounds(field, too_low, low_ok, high_ok, too_high):
    """Out-of-range vitals are rejected; boundary values are accepted."""
    with pytest.raises(ValidationError):
        Health(**_base_payload(**{field: too_low}))
    Health(**_base_payload(**{field: low_ok}))
    Health(**_base_payload(**{field: high_ok}))
    with pytest.raises(ValidationError):
        Health(**_base_payload(**{field: too_high}))


@pytest.mark.parametrize("field", ["HR", "SBP", "MAP", "Resp"])
def test_zero_vitals_rejected(field):
    """Zero vitals are rejected by the model validators."""
    with pytest.raises(ValidationError):
        Health(**_base_payload(**{field: 0}))


@pytest.mark.parametrize("lab", ["Lactate", "WBC", "Creatinine", "Platelets"])
def test_negative_labs_rejected(lab):
    """Labs must not be negative; zero is acceptable."""
    with pytest.raises(ValidationError):
        Health(**_base_payload(**{lab: -0.1}))
    Health(**_base_payload(**{lab: 0.0}))


def test_age_bounds():
    """Age is bounded to [0, 120]."""
    with pytest.raises(ValidationError):
        Health(**_base_payload(Age=-1))
    Health(**_base_payload(Age=0))
    Health(**_base_payload(Age=120))
    with pytest.raises(ValidationError):
        Health(**_base_payload(Age=121))


def test_iculos_must_be_positive():
    """ICULOS starts at 1 (ICU hour 1)."""
    with pytest.raises(ValidationError):
        Health(**_base_payload(ICULOS=0))
    Health(**_base_payload(ICULOS=1))


def test_temp_out_of_human_range_rejected():
    """Temp validator rejects values outside the realistic human range."""
    with pytest.raises(ValidationError):
        Health(**_base_payload(Temp=29.5))
    with pytest.raises(ValidationError):
        Health(**_base_payload(Temp=43.5))


def test_missing_patientid_rejected():
    with pytest.raises(ValidationError):
        Health(**_base_payload(PatientID=None))


def test_wrong_type_rejected():
    with pytest.raises(ValidationError):
        Health(**_base_payload(HR="one-hundred"))


def test_labs_default_to_none():
    parsed = Health(**_base_payload(Lactate=None, WBC=None, Creatinine=None, Platelets=None))
    assert parsed.Lactate is None
    assert parsed.Platelets is None


@pytest.mark.parametrize(
    "field,bad_low,bad_high,good_low,good_high",
    [
        ("raw_probability", -0.01, 1.01, 0.0, 1.0),
        ("filtered_probability", -0.01, 1.01, 0.0, 1.0),
    ],
    ids=["raw_probability", "filtered_probability"],
)
def test_prediction_response_probability_bounds(field, bad_low, bad_high, good_low, good_high):
    """Probabilities in PredictionResponse must stay within [0, 1]."""
    base = {
        "patient_id": "p100000",
        "iculos": 3,
        "raw_probability": 0.5,
        "filtered_probability": 0.1,
        "high_risk": False,
        "alert": False,
    }
    with pytest.raises(ValidationError):
        PredictionResponse(**{**base, field: bad_low})
    with pytest.raises(ValidationError):
        PredictionResponse(**{**base, field: bad_high})
    PredictionResponse(**{**base, field: good_low})
    PredictionResponse(**{**base, field: good_high})


def test_prediction_response_iculos_must_be_positive():
    with pytest.raises(ValidationError):
        PredictionResponse(
            patient_id="p1", iculos=0, raw_probability=0.5,
            filtered_probability=0.1, high_risk=False, alert=False,
        )