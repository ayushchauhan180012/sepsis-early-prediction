"""Stateless alert engine — recompute-from-history approach (D-013, O-7).

Given a patient's complete prediction history (ICULOS-ordered), derives:
  - filtered_probability (uncertainty band zeroing)
  - high_risk (threshold on filtered probability)
  - alert (persistence + cooldown on high_risk)

All alert parameters are frozen (TRAINING_CONTRACT §7, D-012) and loaded
from ``Backend.config.ALERT_PARAMS``.

The function is **pure/deterministic** — no mutable module-level state.
Running it on the same input always produces the same output.
"""

from __future__ import annotations

from dataclasses import dataclass

from Backend.config import ALERT_PARAMS


@dataclass(frozen=True, slots=True)
class AlertState:
    """Per-hour alert fields derived from raw probability."""
    iculos: int
    raw_probability: float
    filtered_probability: float
    high_risk: bool
    alert: bool


def evaluate_alert_state(
    predictions: list[tuple[int, float]],
    *,
    params: dict | None = None,
) -> list[AlertState]:
    """Recompute alert state for a patient's complete prediction history.

    Parameters
    ----------
    predictions : list of (iculos, raw_probability) tuples
        Must be in ICULOS ascending order (D-009).
    params : dict, optional
        Alert parameters override. Defaults to ``ALERT_PARAMS``.

    Returns
    -------
    list[AlertState]
        One ``AlertState`` per prediction, in the same order.
    """
    if params is None:
        params = ALERT_PARAMS

    uncertain_low: float = params["uncertain_low"]
    uncertain_high: float = params["uncertain_high"]
    threshold: float = params["threshold"]
    persistence: int = params["persistence"]
    cooldown_hours: int = params["cooldown_hours"]
    last_alert_time_init: int = params["last_alert_time_init"]

    result: list[AlertState] = []
    prev_high_risk: bool = False
    consecutive_high_risk: int = 0
    last_alert_time: int = last_alert_time_init

    for iculos, raw_p in predictions:
        # 1. Uncertainty filter (strict inequalities)
        uncertain = uncertain_low < raw_p < uncertain_high

        # 2. Filtering
        filtered_p = 0.0 if uncertain else raw_p

        # 3. Threshold
        high_risk = filtered_p >= threshold

        # 4. Persistence — track consecutive high-risk count
        if high_risk:
            consecutive_high_risk += 1
        else:
            consecutive_high_risk = 0

        alert_raw = high_risk and prev_high_risk

        # 5. Cooldown
        alert = alert_raw and (iculos - last_alert_time >= cooldown_hours)

        if alert:
            last_alert_time = iculos

        result.append(AlertState(
            iculos=iculos,
            raw_probability=raw_p,
            filtered_probability=filtered_p,
            high_risk=high_risk,
            alert=alert,
        ))

        prev_high_risk = high_risk

    return result
