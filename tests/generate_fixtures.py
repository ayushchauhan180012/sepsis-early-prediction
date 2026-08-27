"""Generate committed parity fixtures for the Phase 7 test suite.

Usage:
    python tests/generate_fixtures.py \
        --dataset D:/sepsis_training_data/baseline_dataset.csv [-y]

Produces (overwrites):
    tests/fixtures/feature_parity.csv     raw inputs + expected 50 features
    tests/fixtures/alert_scenarios.json   probability streams + expected alert state

Reference semantics are reproduced verbatim from the training notebook
(notebooks/early_sepsis_alert_system.ipynb):

  * cell 6  — lab-missing indicators; per-patient vitals forward-fill; median
              fill using the frozen training medians (FROZEN_MEDIANS).
  * cell 8  — delta6/delta1, rolling-6 std, baseline_dev (first six stored
              observations), lab ``recent_test``, clinical ratios and flags.
  * cell 12 — uncertainty band (strict inequalities), filtered probability,
              high_risk ``>= threshold``, 2-hour persistence, 3-hour cooldown.

Before writing, the generator CROSS-CHECKS the notebook reference path against
the production path (``Backend.Services.feature_engineering.transform_patient_history``
and ``Backend.Services.alert_engine.evaluate_alert_state``) and refuses to emit
fixtures on any mismatch. Committed fixtures are therefore an independent
record of the notebook's output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from Backend.config import (
    FEATURE_NAMES,
    FROZEN_MEDIANS,
    LABS,
    VITALS,
    ALERT_PARAMS,
)
from Backend.Services.alert_engine import evaluate_alert_state
from Backend.Services.feature_engineering import transform_patient_history

FIXTURES_DIR = HERE / "fixtures"
FEATURE_PARITY_CSV = FIXTURES_DIR / "feature_parity.csv"
ALERT_SCENARIOS_JSON = FIXTURES_DIR / "alert_scenarios.json"

RAW_COLS = ["PatientID", "ICULOS", "Age"] + VITALS + LABS


def reference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 6 + cell 8 of the training notebook, on a single patient block."""
    df = df.sort_values(["PatientID", "ICULOS"]).reset_index(drop=True).copy()

    for col in LABS:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    df[VITALS] = df.groupby("PatientID")[VITALS].ffill()
    for col in VITALS:
        df[col] = df[col].fillna(FROZEN_MEDIANS[col])

    for col in VITALS:
        grouped = df.groupby("PatientID")[col]
        df[f"{col}_delta6"] = (grouped.shift(0) - grouped.shift(6)).fillna(0)
        df[f"{col}_delta1"] = (grouped.shift(0) - grouped.shift(1)).fillna(0)
        df[f"{col}_roll6_std"] = (
            grouped.rolling(window=6, min_periods=1).std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        baseline_mean = grouped.transform(
            lambda x: x.iloc[: min(6, len(x))].mean()
        )
        df[f"{col}_baseline_dev"] = df[col] - baseline_mean

    for col in LABS:
        df[f"{col}_recent_test"] = (
            df.groupby("PatientID")[col]
            .transform(
                lambda x: x.notnull().astype(int).rolling(6, min_periods=1).max()
            )
        )

    df["shock_index"] = df["HR"] / (df["SBP"] + 1)
    df["resp_o2_ratio"] = df["Resp"] / (df["O2Sat"] + 1)
    df["map_hr_ratio"] = df["MAP"] / (df["HR"] + 1)
    df["tachycardia"] = (df["HR"] > 100).astype(int)
    df["hypotension"] = (df["SBP"] < 90).astype(int)
    df["tachypnea"] = (df["Resp"] > 22).astype(int)

    return df[list(FEATURE_NAMES)].copy()


def reference_alert_state(iculos: np.ndarray, prob: np.ndarray) -> dict:
    """Cell 12 of the training notebook, single patient, no groupby overhead."""
    prob = np.asarray(prob, dtype=float)
    s = pd.DataFrame({"ICULOS": iculos, "prob": prob})
    s["uncertain"] = (s["prob"] > ALERT_PARAMS["uncertain_low"]) & (
        s["prob"] < ALERT_PARAMS["uncertain_high"]
    )
    s["prob_filtered"] = s["prob"]
    s.loc[s["uncertain"], "prob_filtered"] = 0.0
    s["high_risk"] = (s["prob_filtered"] >= ALERT_PARAMS["threshold"]).astype(int)
    s["high_risk_prev"] = s["high_risk"].shift(1).fillna(0).astype(int)
    s["alert_raw"] = ((s["high_risk"] == 1) & (s["high_risk_prev"] == 1)).astype(int)

    alert = np.zeros(len(s), dtype=int)
    last_alert_time = ALERT_PARAMS["last_alert_time_init"]
    for i, row in s.iterrows():
        if row["alert_raw"] == 1:
            if row["ICULOS"] - last_alert_time >= ALERT_PARAMS["cooldown_hours"]:
                alert[i] = 1
                last_alert_time = row["ICULOS"]

    return {
        "uncertain": s["uncertain"].astype(bool).tolist(),
        "filtered_probability": s["prob_filtered"].round(6).tolist(),
        "high_risk": (s["high_risk"] == 1).tolist(),
        "alert": (alert == 1).tolist(),
    }


def production_feature_check(history: pd.DataFrame) -> pd.DataFrame:
    """Produce production transform output and fail loudly on mismatch."""
    production = transform_patient_history(history[RAW_COLS].copy())
    reference = reference_features(history)
    if production.shape != reference.shape:
        raise SystemExit(
            f"shape mismatch: production={production.shape} reference={reference.shape}"
        )
    mismatch = ~np.isclose(
        production.values, reference.values, rtol=1e-9, atol=1e-12, equal_nan=True
    )
    if mismatch.any():
        rows, cols = np.nonzero(mismatch)
        ex = cols[0]
        raise SystemExit(
            f"reference/production feature mismatch for patient "
            f"{history['PatientID'].iloc[0]} at ICULOS "
            f"{history['ICULOS'].iloc[rows[0]]}, feature {reference.columns[ex]}"
        )
    return production


def pick_patients(df: pd.DataFrame, *, n: int, seed: int) -> list[str]:
    """Deterministically pick patients with a reasonable history length."""
    rng = np.random.default_rng(seed)
    lengths = df.groupby("PatientID").size()
    eligible = lengths[(lengths >= 12) & (lengths <= 300)].index.tolist()
    ordered = sorted(eligible)
    if len(ordered) < n:
        raise SystemExit(
            f"need at least {n} patients with 12-300 rows; got {len(ordered)}"
        )
    idx = rng.choice(len(ordered), size=n, replace=False)
    return [ordered[i] for i in idx]


def build_feature_parity_csv(df: pd.DataFrame, model) -> None:
    """Cross-check reference vs production and write expected features."""

    def classify(row) -> str:
        n_vitals_nan = row[VITALS].isna().sum()
        n_labs_nan = row[LABS].isna().sum()
        if n_vitals_nan > 0 and n_labs_nan > 0:
            return "both"
        if n_labs_nan > 0:
            return "labs"
        if n_vitals_nan > 0:
            return "vitals"
        return "complete"

    patients = pick_patients(df, n=4, seed=1234)
    blocks: list[pd.DataFrame] = []
    for pid in patients:
        block = (
            df[df["PatientID"] == pid]
            .sort_values("ICULOS")[RAW_COLS]
            .reset_index(drop=True)
        )
        production = production_feature_check(block)
        expected = production.rename(
            columns={c: f"expected_{c}" for c in production.columns}
        )
        blocks.append(pd.concat([block, expected.round(6)], axis=1))

    out = pd.concat(blocks, axis=0, ignore_index=True)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(FEATURE_PARITY_CSV, index=False)

    kinds = {}
    for _, row in out.iterrows():
        kinds[classify(row)] = kinds.get(classify(row), 0) + 1
    print(
        f"wrote {FEATURE_PARITY_CSV} ({len(out)} rows, "
        f"patients {sorted(out['PatientID'].unique())}, "
        + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        + ")"
    )


def real_patient_scenarios(df: pd.DataFrame, model) -> list[dict]:
    """Pick real patients exercising distinct alert behaviors, via the frozen model."""
    scenarios: list[dict] = []
    wanted = {"alert", "uncertain", "quiet"}  # ensure coverage breadth
    for pid in sorted(df["PatientID"].unique()):
        block = (
            df[df["PatientID"] == pid]
            .sort_values("ICULOS")[RAW_COLS]
            .reset_index(drop=True)
        )
        if not (12 <= len(block) <= 300):
            continue

        production = production_feature_check(block)
        prob = model.predict_proba(production.values)[:, 1]
        expected = reference_alert_state(block["ICULOS"].to_numpy(), prob)
        profile = set()
        if any(expected["alert"]):
            profile.add("alert")
        if any(expected["uncertain"]):
            profile.add("uncertain")
        if not any(expected["high_risk"]):
            profile.add("quiet")

        hit = wanted.intersection(profile)
        if not hit:
            continue

        scenarios.append(
            {
                "name": f"patient_{pid}",
                "source": "training",
                "patient_id": pid,
                "iculos": block["ICULOS"].tolist(),
                "prob": np.round(prob, 6).tolist(),
                "expected": _expected_list(block["ICULOS"], prob),
            }
        )
        print(f"  real scenario '{scenarios[-1]['name']}': {', '.join(sorted(hit))}")
        wanted -= hit
        if not wanted:
            break

    if wanted:
        raise SystemExit(f"could not cover alert profiles {sorted(wanted)} from training data")
    return scenarios


def synthetic_scenarios() -> list[dict]:
    """Crafted probability streams that hit boundary/persistence/cooldown logic."""
    cases = [
        {
            "name": "cooldown_spacing",
            "iculos": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "prob": [0.02, 0.02, 0.06, 0.06, 0.06, 0.07, 0.07, 0.07, 0.08, 0.08],
        },
        {
            "name": "uncertain_band_zeroing",
            "iculos": [1, 2, 3, 4, 5],
            "prob": [0.02, 0.04, 0.05, 0.06, 0.06],
        },
        {
            "name": "strict_boundaries",
            "iculos": [1, 2, 3, 4, 5, 6],
            "prob": [0.035, 0.055, 0.045, 0.045, 0.045, 0.045],
        },
    ]
    scenarios: list[dict] = []
    for case in cases:
        prob = np.round(np.asarray(case["prob"], dtype=float), 6)
        scenarios.append(
            {
                "name": case["name"],
                "source": "synthetic",
                "iculos": case["iculos"],
                "prob": prob.tolist(),
                "expected": _expected_list(case["iculos"], prob),
            }
        )
    return scenarios


def _expected_list(iculos, prob) -> list[dict]:
    state = reference_alert_state(np.asarray(iculos), np.asarray(prob))
    rows = []
    for i in range(len(iculos)):
        rows.append(
            {
                "uncertain": state["uncertain"][i],
                "filtered_probability": state["filtered_probability"][i],
                "high_risk": state["high_risk"][i],
                "alert": state["alert"][i],
            }
        )
    return rows


def build_alert_scenarios_json(df: pd.DataFrame, model) -> None:
    scenarios = synthetic_scenarios() + real_patient_scenarios(df, model)

    # Self-check: production evaluate_alert_state must reproduce every expected row.
    for sc in scenarios:
        produced = evaluate_alert_state(
            list(zip(sc["iculos"], sc["prob"]))
        )
        for a, b in zip(produced, sc["expected"]):
            assert a.high_risk == b["high_risk"], sc["name"]
            assert a.alert == b["alert"], sc["name"]
            assert np.isclose(
                a.filtered_probability, b["filtered_probability"], atol=1e-9
            ), sc["name"]

    payload = {
        "version": 1,
        "reference": "notebooks/early_sepsis_alert_system.ipynb cell 12",
        "parameters": {
            "threshold": ALERT_PARAMS["threshold"],
            "uncertain_low": ALERT_PARAMS["uncertain_low"],
            "uncertain_high": ALERT_PARAMS["uncertain_high"],
            "persistence": ALERT_PARAMS["persistence"],
            "cooldown_hours": ALERT_PARAMS["cooldown_hours"],
            "last_alert_time_init": ALERT_PARAMS["last_alert_time_init"],
        },
        "scenarios": scenarios,
    }
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(ALERT_SCENARIOS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {ALERT_SCENARIOS_JSON} ({len(scenarios)} scenarios)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to the training baseline dataset CSV (excluded from the repo).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Overwrite existing fixtures without prompting.",
    )
    args = parser.parse_args()

    if FEATURE_PARITY_CSV.exists() or ALERT_SCENARIOS_JSON.exists():
        if not args.yes:
            answer = input("Fixtures already exist. Overwrite? [y/N] ").strip().lower()
            if answer != "y":
                print("aborting (no changes written)")
                return

    if args.dataset is None:
        raise SystemExit(
            "no dataset given — pass --dataset PATH "
            "(e.g. D:/sepsis_training_data/baseline_dataset.csv, excluded from the repo)."
        )
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(
            f"training dataset not found: {dataset_path} — check the path."
        )

    model = joblib.load(ROOT / "Backend" / "Model" / "hgb_sepsis_model.joblib")

    df = pd.read_csv(dataset_path)
    for col in RAW_COLS:
        if col not in df.columns:
            raise SystemExit(f"dataset is missing required column {col!r}")

    build_feature_parity_csv(df, model)
    build_alert_scenarios_json(df, model)


if __name__ == "__main__":
    main()