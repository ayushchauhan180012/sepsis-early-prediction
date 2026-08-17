"""Shared test fixtures for Phase 3 tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def single_patient_history():
    """A 12-row patient history with vitals, labs, and known ICULOS order.

    Row 0-5: first 6 observations (used for baseline_dev)
    Row 6-11: next 6 observations (target rows)
    """
    np.random.seed(42)
    n = 12
    hr_base = 90.0
    data = {
        "PatientID": ["P-TEST"] * n,
        "ICULOS": list(range(1, n + 1)),
        "Age": [65] * n,
        "HR": [hr_base + i * 2 for i in range(n)],
        "O2Sat": [97.0 + (i % 3) * 0.5 for i in range(n)],
        "SBP": [120.0 + i for i in range(n)],
        "MAP": [75.0 + i * 0.5 for i in range(n)],
        "Resp": [16.0 + (i % 4) for i in range(n)],
        "Temp": [36.8 + i * 0.05 for i in range(n)],
        "Lactate": [1.2 if i % 3 == 0 else np.nan for i in range(n)],
        "WBC": [8.0 if i % 4 == 0 else np.nan for i in range(n)],
        "Creatinine": [0.9 if i % 6 == 0 else np.nan for i in range(n)],
        "Platelets": [220.0 if i % 2 == 0 else np.nan for i in range(n)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def patient_with_leading_nan_vitals():
    """Patient with leading NaN vitals (first 3 rows all NaN for HR).
    Frozen medians should fill these after forward-fill."""
    n = 8
    data = {
        "PatientID": ["P-NAN"] * n,
        "ICULOS": list(range(1, n + 1)),
        "Age": [55] * n,
        "HR": [np.nan, np.nan, np.nan, 90.0, 92.0, 94.0, 96.0, 98.0],
        "O2Sat": [np.nan, 97.0, 98.0, 99.0, 98.5, 97.5, 98.0, 99.0],
        "SBP": [120.0, 121.0, np.nan, 119.0, 118.0, 120.0, 122.0, 121.0],
        "MAP": [75.0, 76.0, 77.0, 78.0, 77.5, 76.5, 77.0, 78.0],
        "Resp": [16.0, 17.0, 18.0, 16.0, 17.0, 18.0, 16.0, 17.0],
        "Temp": [36.8, 36.9, 37.0, 37.1, 37.0, 36.9, 36.8, 37.0],
        "Lactate": [np.nan] * n,
        "WBC": [np.nan] * n,
        "Creatinine": [np.nan] * n,
        "Platelets": [np.nan] * n,
    }
    return pd.DataFrame(data)


@pytest.fixture
def patient_short_history():
    """Patient with only 3 rows (fewer than baseline window of 6).
    Partial baseline should apply."""
    n = 3
    data = {
        "PatientID": ["P-SHORT"] * n,
        "ICULOS": list(range(1, n + 1)),
        "Age": [40] * n,
        "HR": [80.0, 85.0, 90.0],
        "O2Sat": [98.0, 97.0, 99.0],
        "SBP": [118.0, 120.0, 122.0],
        "MAP": [77.0, 78.0, 79.0],
        "Resp": [18.0, 17.0, 16.0],
        "Temp": [36.94, 37.0, 36.88],
        "Lactate": [np.nan] * n,
        "WBC": [np.nan] * n,
        "Creatinine": [np.nan] * n,
        "Platelets": [np.nan] * n,
    }
    return pd.DataFrame(data)


@pytest.fixture
def extreme_baseline_patient():
    """Patient where first 6 rows are far from last 6 rows.
    baseline_dev MUST use first 6, not trailing 6."""
    n = 12
    data = {
        "PatientID": ["P-EXTREME"] * n,
        "ICULOS": list(range(1, n + 1)),
        "Age": [70] * n,
        # First 6: HR=50 (very low); last 6: HR=120 (very high)
        "HR": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0,
               120.0, 121.0, 122.0, 123.0, 124.0, 125.0],
        "O2Sat": [98.0] * n,
        "SBP": [120.0] * n,
        "MAP": [77.0] * n,
        "Resp": [18.0] * n,
        "Temp": [36.94] * n,
        "Lactate": [np.nan] * n,
        "WBC": [np.nan] * n,
        "Creatinine": [np.nan] * n,
        "Platelets": [np.nan] * n,
    }
    return pd.DataFrame(data)


@pytest.fixture
def multi_patient_history():
    """Two patients with interleaved ICULOS to test groupby correctness."""
    data = {
        "PatientID": ["P-A"] * 6 + ["P-B"] * 6,
        "ICULOS": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
        "Age": [50] * 6 + [60] * 6,
        "HR": [80.0, 82.0, 84.0, 86.0, 88.0, 90.0,
               100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
        "O2Sat": [98.0] * 12,
        "SBP": [120.0] * 12,
        "MAP": [77.0] * 12,
        "Resp": [18.0] * 12,
        "Temp": [36.94] * 12,
        "Lactate": [np.nan] * 12,
        "WBC": [np.nan] * 12,
        "Creatinine": [np.nan] * 12,
        "Platelets": [np.nan] * 12,
    }
    return pd.DataFrame(data)
