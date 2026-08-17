"""Deterministic feature engineering — single source of truth (D-004).

Accepts a patient's chronological observation history as a DataFrame with
training-contract column names (uppercase) and produces the exact 50-feature
row the frozen model expects, per ``docs/TRAINING_CONTRACT.md``.

Design decisions:
  - This module is **pure-DataFrame logic** — no database access.
  - The caller provides a DataFrame sorted by ``PatientID, ICULOS`` (D-009).
  - Frozen medians are loaded from ``Backend.config``, never recomputed (D-003).
  - Labs are never imputed (D-010); ``HGB`` handles remaining NaN natively.
  - ``recent_test`` is computed on **labs only**, not vitals (D-005).
  - ``baseline_dev`` uses the **first six stored observations** (D-006).
"""

from __future__ import annotations

import pandas as pd

from Backend.config import (
    VITALS,
    LABS,
    FROZEN_MEDIANS,
    BASELINE_WINDOW,
    ROLL_WINDOW,
    DELTA6_LOOKBACK,
    DENOMINATOR_EPSILON,
    TACHYCARDIA_HR,
    HYPOTENSION_SBP,
    TACHYPNEA_RESP,
    FEATURE_NAMES,
)

# Column subsets for clarity
_RAW_COLS = ["PatientID", "ICULOS", "Age"] + VITALS + LABS


# ── helpers ──────────────────────────────────────────────────────────────────

def observations_to_dataframe(
    rows: list,
    *,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Convert a list of Observation ORM objects (or dicts) to a DataFrame
    with training-contract uppercase column names.

    Parameters
    ----------
    rows : list of Observation ORM objects or dicts.
        ORM objects are expected to have attribute names matching the DB
        schema (lowercase).  Dicts may use either casing.
    column_map : optional override mapping ``{db_attr: contract_name}``.
        Defaults to the Phase 2 D-022 mapping.

    Returns
    -------
    pd.DataFrame sorted by ``ICULOS`` ascending.
    """
    if column_map is None:
        column_map = {
            "patient_id": "PatientID",
            "iculos": "ICULOS",
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

    records: list[dict] = []
    for row in rows:
        if isinstance(row, dict):
            rec = {column_map.get(k, k): v for k, v in row.items()}
        else:
            rec = {}
            for attr, cname in column_map.items():
                rec[cname] = getattr(row, attr, None)
            if hasattr(row, "age"):
                rec["Age"] = row.age
        records.append(rec)

    df = pd.DataFrame(records)
    if "ICULOS" in df.columns:
        df = df.sort_values("ICULOS").reset_index(drop=True)
    return df


# ── core transform ───────────────────────────────────────────────────────────

def _impute_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill vitals per patient, then fill remaining leading NaNs
    with frozen training medians (§5)."""
    df = df.copy()
    df[VITALS] = df.groupby("PatientID")[VITALS].ffill()
    for vital in VITALS:
        df[vital] = df[vital].fillna(FROZEN_MEDIANS[vital])
    return df


def _add_lab_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``{lab}_missing`` columns based on **raw** (pre-imputed) lab values.
    Labs are never imputed (D-010)."""
    df = df.copy()
    for lab in LABS:
        df[f"{lab}_missing"] = df[lab].isnull().astype(int)
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute delta1, delta6, roll6_std, baseline_dev for each vital."""
    df = df.copy()

    for col in VITALS:
        grouped = df.groupby("PatientID")[col]

        # delta1: current − previous row, NaN → 0
        df[f"{col}_delta1"] = grouped.shift(0) - grouped.shift(1)
        df[f"{col}_delta1"] = df[f"{col}_delta1"].fillna(0)

        # delta6: current − value 6 rows back, NaN → 0
        df[f"{col}_delta6"] = grouped.shift(0) - grouped.shift(DELTA6_LOOKBACK)
        df[f"{col}_delta6"] = df[f"{col}_delta6"].fillna(0)

        # roll6_std: rolling(6, min_periods=1).std(), NaN → 0
        df[f"{col}_roll6_std"] = (
            grouped
            .rolling(window=ROLL_WINDOW, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
        df[f"{col}_roll6_std"] = df[f"{col}_roll6_std"].fillna(0)

        # baseline_dev: current value − mean of first BASELINE_WINDOW stored rows
        # (D-006).  For patients with <6 rows, partial baseline.  NaN passthrough
        # is not needed here since we use the available rows' mean.
        baseline_mean = grouped.transform(
            lambda x: x.iloc[: min(BASELINE_WINDOW, len(x))].mean()
        )
        df[f"{col}_baseline_dev"] = df[col] - baseline_mean

    return df


def _add_lab_recent_test(df: pd.DataFrame) -> pd.DataFrame:
    """``{lab}_recent_test``: not-null indicator → rolling(6, min_periods=1).max().
    **Labs only**, never vitals (D-005)."""
    df = df.copy()
    for lab in LABS:
        df[f"{lab}_recent_test"] = (
            df.groupby("PatientID")[lab]
            .transform(lambda x: x.notnull().astype(int).rolling(ROLL_WINDOW, min_periods=1).max())
        )
    return df


def _add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios and threshold flags on imputed vitals (§5)."""
    df = df.copy()
    df["shock_index"] = df["HR"] / (df["SBP"] + DENOMINATOR_EPSILON)
    df["resp_o2_ratio"] = df["Resp"] / (df["O2Sat"] + DENOMINATOR_EPSILON)
    df["map_hr_ratio"] = df["MAP"] / (df["HR"] + DENOMINATOR_EPSILON)
    df["tachycardia"] = (df["HR"] > TACHYCARDIA_HR).astype(int)
    df["hypotension"] = (df["SBP"] < HYPOTENSION_SBP).astype(int)
    df["tachypnea"] = (df["Resp"] > TACHYPNEA_RESP).astype(int)
    return df


def _assemble_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    """Select and reorder columns to match the frozen 50-feature order (§4)."""
    # Ensure all expected columns exist
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns after transformation: {missing}")
    return df[list(FEATURE_NAMES)].copy()


# ── public API ───────────────────────────────────────────────────────────────

def transform_patient_history(patient_history: pd.DataFrame) -> pd.DataFrame:
    """Apply the full deterministic transformation to a patient's observation history.

    Parameters
    ----------
    patient_history : pd.DataFrame
        Must contain columns: ``PatientID, ICULOS, Age`` plus the 6 vital
        columns and 4 lab columns (training-contract uppercase names).
        Data **should** be sorted by ``(PatientID, ICULOS)`` (D-009); this
        function enforces the sort.

    Returns
    -------
    pd.DataFrame
        Same row count, with all 50 engineered feature columns in the exact
        frozen order (``FEATURE_NAMES``).
    """
    df = patient_history.copy()
    if df.empty:
        raise ValueError("patient_history is empty — cannot compute features")
    df = df.sort_values(["PatientID", "ICULOS"]).reset_index(drop=True)

    # 1. Forward-fill vitals, then frozen-median fill remaining leading NaNs
    df = _impute_vitals(df)

    # 2. Lab missing indicators (on raw, pre-imputation values — done before
    #    imputation, but _impute_vitals only touches vitals so raw labs are intact)
    df = _add_lab_missing_indicators(df)

    # 3. Temporal features (delta1, delta6, roll6_std, baseline_dev)
    df = _add_temporal_features(df)

    # 4. Lab recent_test (labs only)
    df = _add_lab_recent_test(df)

    # 5. Clinical ratios and flags (on imputed vitals)
    df = _add_clinical_features(df)

    # 6. Assemble in frozen feature order
    df = _assemble_feature_row(df)

    return df


def compute_feature_row(
    patient_history: pd.DataFrame,
    target_iculos: int | None = None,
) -> pd.DataFrame:
    """Produce a single-row DataFrame with the 50 features for a given ICULOS.

    Parameters
    ----------
    patient_history : pd.DataFrame
        Full chronological observation history for one patient (training-contract
        column names).  Will be sorted internally.
    target_iculos : int or None
        The ICULOS hour to produce features for.  ``None`` → use the last
        (most recent) observation.

    Returns
    -------
    pd.DataFrame with exactly one row and ``FEATURE_NAMES`` columns.
    """
    transformed = transform_patient_history(patient_history)

    if target_iculos is None:
        row = transformed.iloc[[-1]]
    else:
        # ICULOS is preserved in FEATURE_NAMES (feature index 11), so we can
        # filter on it directly in the transformed DataFrame.
        mask = transformed["ICULOS"] == target_iculos
        if not mask.any():
            raise ValueError(f"ICULOS {target_iculos} not found in patient history")
        row = transformed.loc[mask]

    return row
