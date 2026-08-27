"""Phase 7 — committed alert-state scenario fixture tests (O-8 resolution,
D-023).  ``tests/fixtures/alert_scenarios.json`` records probability streams
plus the notebook cell-12 alert output for them (uncertainty band, 2-hour
persistence, cooldown).  The production alert engine must reproduce the
committed record exactly.

Scenarios are a mix of:
  * synthetic streams crafted to hit boundary / persistence / cooldown logic;
  * real patients whose streams came from the frozen model on training data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from Backend.Services.alert_engine import evaluate_alert_state
from Backend.config import ALERT_PARAMS

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ALERT_SCENARIOS_JSON = FIXTURES / "alert_scenarios.json"


def _load_scenarios() -> list[dict]:
    if not ALERT_SCENARIOS_JSON.exists():
        pytest.fail(
            f"missing fixture {ALERT_SCENARIOS_JSON} — regenerate it with "
            "tests/generate_fixtures.py --dataset <training_csv>"
        )
    with open(ALERT_SCENARIOS_JSON, encoding="utf-8") as fh:
        return json.load(fh)["scenarios"]


SCENARIOS = _load_scenarios()


def test_fixture_parameters_match_contract():
    with open(ALERT_SCENARIOS_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    for key in ("threshold", "uncertain_low", "uncertain_high", "persistence",
                "cooldown_hours", "last_alert_time_init"):
        assert payload["parameters"][key] == ALERT_PARAMS[key], key


def test_scenarios_cover_distinct_behaviors():
    """The committed set should exercise alert, no-alert, and uncertainty."""
    behavior = {"alert": False, "quiet": False, "uncertain": False}
    for sc in SCENARIOS:
        if any(row["alert"] for row in sc["expected"]):
            behavior["alert"] = True
        if any(row["uncertain"] for row in sc["expected"]):
            behavior["uncertain"] = True
        if not any(row["high_risk"] for row in sc["expected"]):
            behavior["quiet"] = True
    assert all(behavior.values()), f"scenario coverage missing: {behavior}"
    assert any(s["source"] == "training" for s in SCENARIOS)
    assert any(s["source"] == "synthetic" for s in SCENARIOS)


def test_cooldown_scenario_spacing():
    """The committed cooldown scenario must space alerts by cooldown_hours."""
    sc = next(s for s in SCENARIOS if s["name"] == "cooldown_spacing")
    alert_hours = [iculos for iculos, row in zip(sc["iculos"], sc["expected"]) if row["alert"]]
    assert len(alert_hours) >= 3
    gaps = [b - a for a, b in zip(alert_hours, alert_hours[1:])]
    assert all(gap >= ALERT_PARAMS["cooldown_hours"] for gap in gaps)


@pytest.mark.parametrize("scenario", SCENARIOS,
                         ids=[s["name"] for s in SCENARIOS])
def test_evaluate_alert_state_reproduces_scenario(scenario):
    iculos = scenario["iculos"]
    prob = [float(p) for p in scenario["prob"]]
    assert len(iculos) == len(prob) == len(scenario["expected"])
    assert all(0.0 <= p <= 1.0 for p in prob)
    assert all(a < b for a, b in zip(iculos, iculos[1:]))

    produced = evaluate_alert_state(list(zip(iculos, prob)))
    assert len(produced) == len(scenario["expected"])
    for state, expected in zip(produced, scenario["expected"]):
        assert state.high_risk == expected["high_risk"]
        assert state.alert == expected["alert"]
        assert np.isclose(
            state.filtered_probability, expected["filtered_probability"], atol=1e-9
        )
        # filtered==0 exactly corresponds to the uncertainty band flipping on.
        assert expected["uncertain"] == (expected["filtered_probability"] == 0.0)