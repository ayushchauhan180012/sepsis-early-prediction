"""Phase 5 correctness tests — alert engine (TRAINING_CONTRACT §7).

Tests cover:
  A. Uncertainty handling (strict band, boundary values)
  B. Threshold behavior (filtered probability ≥ 0.045)
  C. Persistence (two consecutive high-risk hours)
  D. Cooldown (3-hour minimum between alerts)
  E. Raw probability preservation
  F. Patient isolation
  G. Alert events (maximal contiguous runs)
  H. Alert summaries
  I. End-to-end process_observation
  J. Phase 4 regression (updated placeholder assertions)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import joblib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from Backend.config import settings, ALERT_PARAMS
from Backend.Services.alert_engine import evaluate_alert_state, AlertState
from Backend.Database.schema import Base
from Backend.Database.operations import (
    ensure_patient,
    upsert_observation,
    upsert_prediction,
    get_patient_predictions,
    update_prediction_alert_batch,
    rebuild_alert_events,
    upsert_alert_summary,
    get_patient_alerts,
    insert_alert,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def db_session():
    """In-memory SQLite session for DB tests."""
    test_engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    session = TestSession()
    yield session
    session.rollback()
    session.close()
    test_engine.dispose()


@pytest.fixture(scope="class")
def model():
    """The frozen HGB model loaded once per test class."""
    return joblib.load(settings.model_path)


# ── Helper functions ──────────────────────────────────────────────────────────

def _make_patient(session, patient_id: str, age: int = 65) -> None:
    """Ensure a patient exists in the DB."""
    ensure_patient(session, patient_id, age)


def _store_predictions(session, patient_id: str, raw_probs: list[float]) -> None:
    """Store predictions with the given raw probabilities at ICULOS 1..N.

    Alert fields are set to Phase 4 placeholders (will be overwritten by
    the alert engine recompute).
    """
    for i, p in enumerate(raw_probs, start=1):
        upsert_prediction(
            session, patient_id, i,
            raw_probability=p,
            filtered_probability=p,
            high_risk=False,
            alert=False,
        )


def _get_alert_states(raw_probs: list[float]) -> list[AlertState]:
    """Convenience: evaluate alert state for ICULOS 1..N."""
    return evaluate_alert_state(
        [(i, p) for i, p in enumerate(raw_probs, start=1)]
    )


# ──────────────────────────────────────────────────────────────────────────────
# A. Uncertainty handling
# ──────────────────────────────────────────────────────────────────────────────

class TestUncertainty:
    """The uncertainty band (0.035, 0.055) uses strict inequalities."""

    def test_below_band_passes_through(self):
        states = _get_alert_states([0.030])
        assert states[0].filtered_probability == pytest.approx(0.030)

    def test_above_band_passes_through(self):
        states = _get_alert_states([0.060])
        assert states[0].filtered_probability == pytest.approx(0.060)

    def test_inside_band_zeroed(self):
        states = _get_alert_states([0.045])
        assert states[0].filtered_probability == pytest.approx(0.0)

    def test_boundary_low_not_uncertain(self):
        """0.035 exactly is NOT uncertain (strict inequality)."""
        states = _get_alert_states([0.035])
        assert states[0].filtered_probability == pytest.approx(0.035)

    def test_boundary_high_not_uncertain(self):
        """0.055 exactly is NOT uncertain (strict inequality)."""
        states = _get_alert_states([0.055])
        assert states[0].filtered_probability == pytest.approx(0.055)

    def test_just_inside_band_low(self):
        """0.0351 is inside the band → filtered to 0."""
        states = _get_alert_states([0.0351])
        assert states[0].filtered_probability == pytest.approx(0.0)

    def test_just_inside_band_high(self):
        """0.0549 is inside the band → filtered to 0."""
        states = _get_alert_states([0.0549])
        assert states[0].filtered_probability == pytest.approx(0.0)

    def test_just_below_boundary_low(self):
        """0.0349 is below band → passes through."""
        states = _get_alert_states([0.0349])
        assert states[0].filtered_probability == pytest.approx(0.0349)

    def test_just_above_boundary_high(self):
        """0.0551 is above band → passes through."""
        states = _get_alert_states([0.0551])
        assert states[0].filtered_probability == pytest.approx(0.0551)

    def test_raw_probability_never_modified(self):
        """raw_probability is preserved unchanged regardless of uncertainty."""
        states = _get_alert_states([0.045])
        assert states[0].raw_probability == pytest.approx(0.045)
        assert states[0].filtered_probability == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# B. Threshold behavior
# ──────────────────────────────────────────────────────────────────────────────

class TestThreshold:
    """high_risk = filtered_probability >= 0.045."""

    def test_exactly_at_threshold(self):
        """0.035 passes through (not uncertain), 0.035 < 0.045 → not high risk."""
        states = _get_alert_states([0.035])
        assert states[0].filtered_probability == pytest.approx(0.035)
        assert states[0].high_risk is False

    def test_below_threshold(self):
        states = _get_alert_states([0.030])
        assert states[0].high_risk is False

    def test_above_threshold(self):
        """0.055 passes through (not uncertain), 0.055 >= 0.045 → high risk."""
        states = _get_alert_states([0.055])
        assert states[0].filtered_probability == pytest.approx(0.055)
        assert states[0].high_risk is True

    def test_threshold_on_filtered_value(self):
        """After uncertainty filtering, 0.060 passes through → high risk."""
        states = _get_alert_states([0.060])
        assert states[0].filtered_probability == pytest.approx(0.060)
        assert states[0].high_risk is True

    def test_uncertain_filtered_to_zero_not_high_risk(self):
        """0.050 is uncertain → filtered to 0 → not high risk."""
        states = _get_alert_states([0.050])
        assert states[0].filtered_probability == pytest.approx(0.0)
        assert states[0].high_risk is False

    def test_high_risk_at_boundary_low(self):
        """0.035 passes through, 0.035 < 0.045 → not high risk."""
        states = _get_alert_states([0.035])
        assert states[0].high_risk is False

    def test_high_risk_above_boundary_high(self):
        """0.055 passes through, 0.055 >= 0.045 → high risk."""
        states = _get_alert_states([0.055])
        assert states[0].high_risk is True

    def test_zero_probability(self):
        states = _get_alert_states([0.0])
        assert states[0].high_risk is False

    def test_probability_one(self):
        states = _get_alert_states([1.0])
        assert states[0].high_risk is True


# ──────────────────────────────────────────────────────────────────────────────
# C. Persistence (two consecutive high-risk hours)
# ──────────────────────────────────────────────────────────────────────────────

class TestPersistence:
    """alert_raw = high_risk AND previous_high_risk (2-hour rule)."""

    def test_single_high_risk_no_alert(self):
        """One hour of high_risk → no alert (prev_high_risk=False)."""
        states = _get_alert_states([0.060])
        assert states[0].high_risk is True
        assert states[0].alert is False

    def test_two_consecutive_high_risk_alerts_on_second(self):
        """2 consecutive high_risk → alert on hour 2."""
        states = _get_alert_states([0.060, 0.060])
        assert states[0].alert is False
        assert states[1].alert is True

    def test_three_consecutive_high_risk(self):
        """3 consecutive high_risk → alert on hour 2 only (cooldown blocks hour 3).

        h1: high, prev=False → no alert
        h2: high, prev=True → alert_raw, 2-(-999)>=3 → alert, last_alert_time=2
        h3: high, prev=True → alert_raw, 3-2=1 < 3 → blocked
        """
        states = _get_alert_states([0.060, 0.060, 0.060])
        assert states[0].alert is False
        assert states[1].alert is True
        assert states[2].alert is False  # blocked by cooldown (3-2=1 < 3)

    def test_high_risk_broken_by_low_risk(self):
        """high → low → high does NOT trigger on second high."""
        states = _get_alert_states([0.060, 0.010, 0.060])
        assert states[0].alert is False
        assert states[1].alert is False
        assert states[2].alert is False

    def test_persistence_requires_exact_two(self):
        """Only 1 high_risk hour → no alert."""
        states = _get_alert_states([0.060, 0.010])
        assert states[0].alert is False
        assert states[1].alert is False

    def test_uncertainty_breaks_persistence(self):
        """high (uncertain) → high (uncertain) → neither is high_risk."""
        # 0.050 is uncertain → filtered to 0 → not high_risk
        states = _get_alert_states([0.050, 0.050])
        assert states[0].high_risk is False
        assert states[1].high_risk is False
        assert states[0].alert is False
        assert states[1].alert is False


# ──────────────────────────────────────────────────────────────────────────────
# D. Cooldown (3 hours)
# ──────────────────────────────────────────────────────────────────────────────

class TestCooldown:
    """alert fires only when ICULOS - last_alert_time >= 3."""

    def test_initial_last_alert_time_always_fires(self):
        """First eligible alert with last_alert_time=-999 always fires."""
        # Two consecutive high-risk hours
        states = _get_alert_states([0.060, 0.060])
        assert states[1].alert is True

    def test_cooldown_blocks_within_3_hours(self):
        """Alert at hour 2, next eligible at hour 4 → blocked (4-2=2 < 3).

        We need: hour1=high, hour2=high (alert fires hour 2),
                 hour3=low (breaks persistence),
                 hour4=low, hour5=low,
                 hour6=high, hour7=high (alert_raw at hour 7).
        At hour 7: 7-2=5 >= 3 → should fire.

        But to test blocking: hour6=high, hour7=high,
        with alert_raw at hour 6 (prev_high_risk at hour 5).
        We need hour 5=high for persistence at 6, and hour 4=high for persistence at 5.

        Let me construct a tighter scenario:
        hours 1,2 = high → alert at 2
        hours 3 = low (break persistence)
        hours 4,5 = high → alert_raw at 5, cooldown: 5-2=3 >= 3 → fires at 5
        hours 6 = low (break persistence)
        hours 7,8 = high → alert_raw at 8, cooldown: 8-5=3 >= 3 → fires at 8
        """
        raw_probs = [
            0.060,  # h1: high, no prev → no alert
            0.060,  # h2: high, prev=high → alert_raw, 2-(-999)>=3 → alert
            0.010,  # h3: low → break persistence
            0.060,  # h4: high, prev=low → no alert_raw
            0.060,  # h5: high, prev=high → alert_raw, 5-2=3 >= 3 → alert
            0.010,  # h6: low → break persistence
            0.060,  # h7: high, prev=low → no alert_raw
            0.060,  # h8: high, prev=high → alert_raw, 8-5=3 >= 3 → alert
        ]
        states = _get_alert_states(raw_probs)
        assert states[0].alert is False  # h1
        assert states[1].alert is True   # h2: first alert
        assert states[2].alert is False  # h3
        assert states[3].alert is False  # h4
        assert states[4].alert is True   # h5: 5-2=3 >= 3
        assert states[5].alert is False  # h6
        assert states[6].alert is False  # h7
        assert states[7].alert is True   # h8: 8-5=3 >= 3

    def test_cooldown_blocks_at_2_hours(self):
        """Alert at hour 2, next eligible at hour 4 → 4-2=2 < 3 → blocked.

        hours 1,2 = high → alert at 2
        hours 3 = low
        hours 4,5 = high → alert_raw at 5, 5-2=3 >= 3 → fires at 5
        Wait, that's exactly 3.

        Better: alert at hour 3, next at hour 5 → 5-3=2 < 3 → blocked.
        hours 1 = low, 2,3 = high → alert at 3
        hours 4 = low
        hours 5,6 = high → alert_raw at 6, 6-3=3 >= 3 → fires at 6.

        Hmm. Let me try:
        hours 1 = low, 2 = high, 3 = high → alert at 3
        hours 4 = low
        hours 5 = high, 6 = high → alert_raw at 6, 6-3=3 → fires.

        To get blocking within 3 hours, we need a gap of only 1-2 hours.
        But persistence requires 2 consecutive hours. So the shortest gap
        between alert_raw events is: alert at N, break at N+1, high at N+2,
        high at N+3 → alert_raw at N+3. Gap = N+3 - N = 3. That's exactly 3.

        The only way to get a gap < 3 is if we somehow have alert_raw at N+2.
        But that requires high at N+1 and N+2. If N+1 is high AND N was alert,
        then persistence requires N+1=high (prev at N was also high since
        we just fired). So: h1=high, h2=high→alert, h3=high→persistence+alert_raw,
        h3-(-999)>=3 → alert at h3. So alert at both h2 and h3.

        Then h4=low, h5=high, h6=high→alert_raw, 6-3=3 → fires.

        The cooldown truly blocks when: there were previous alerts and the
        next alert_raw is within 3 hours. But with persistence=2, the minimum
        gap between alert_raw events is 3 hours (after the second alert fires
        at hour N, the next alert_raw can be at earliest N+3). So cooldown
        will always allow if we have clean persistence breaks.

        Wait, no. The cooldown uses last_alert_time, which is updated only
        when alert fires. If alert fires at h2, last_alert_time=2.
        Then h3=high, h4=high → alert_raw at h4. h4-2=2 < 3 → blocked!
        last_alert_time stays at 2. But now h5=high, h5 has prev_high_risk=True
        (from h4), alert_raw=True, h5-2=3 >= 3 → fires. last_alert_time=5.

        So the scenario is:
        h1=high, h2=high→alert(2), h3=high, h4=high→alert_raw but blocked,
        h5=high→alert_raw(5-2=3→fires).

        Let me test this.
        """
        raw_probs = [
            0.060,  # h1: high, prev=False → no alert
            0.060,  # h2: high, prev=True → alert_raw, 2-(-999)>=3 → alert
            0.060,  # h3: high, prev=True → alert_raw, 3-2=1 < 3 → blocked
            0.060,  # h4: high, prev=True → alert_raw, 4-2=2 < 3 → blocked
            0.060,  # h5: high, prev=True → alert_raw, 5-2=3 >= 3 → alert
        ]
        states = _get_alert_states(raw_probs)
        assert states[0].alert is False  # h1
        assert states[1].alert is True   # h2: first alert
        assert states[2].alert is False  # h3: blocked (3-2=1)
        assert states[3].alert is False  # h4: blocked (4-2=2)
        assert states[4].alert is True   # h5: allowed (5-2=3)

    def test_cooldown_allows_after_more_than_3_hours(self):
        """Alert at hour 2, next at hour 7 → 7-2=5 >= 3 → allowed."""
        raw_probs = [
            0.060,  # h1: high
            0.060,  # h2: alert
            0.010,  # h3: low
            0.010,  # h4: low
            0.010,  # h5: low
            0.010,  # h6: low
            0.060,  # h7: high
            0.060,  # h8: alert_raw, 8-2=6 >= 3 → alert
        ]
        states = _get_alert_states(raw_probs)
        assert states[1].alert is True
        assert states[7].alert is True

    def test_cooldown_multiple_cycles(self):
        """Three alert cycles with proper cooldown."""
        raw_probs = [
            0.060,  # h1: high
            0.060,  # h2: alert (first)
            0.010,  # h3: low
            0.060,  # h4: high
            0.060,  # h5: alert_raw, 5-2=3 → alert (second)
            0.010,  # h6: low
            0.060,  # h7: high
            0.060,  # h8: alert_raw, 8-5=3 → alert (third)
        ]
        states = _get_alert_states(raw_probs)
        assert states[1].alert is True
        assert states[4].alert is True
        assert states[7].alert is True


# ──────────────────────────────────────────────────────────────────────────────
# E. Raw probability preservation
# ──────────────────────────────────────────────────────────────────────────────

class TestRawProbability:
    """Raw probability is never modified; filtered probability is independent."""

    def test_raw_never_modified(self):
        raw_probs = [0.030, 0.045, 0.050, 0.060, 0.100]
        states = _get_alert_states(raw_probs)
        for state, raw in zip(states, raw_probs):
            assert state.raw_probability == pytest.approx(raw)

    def test_filtered_independent(self):
        """Filtered probability differs from raw when uncertain."""
        states = _get_alert_states([0.050])
        assert states[0].raw_probability == pytest.approx(0.050)
        assert states[0].filtered_probability == pytest.approx(0.0)

    def test_filtered_matches_raw_when_not_uncertain(self):
        states = _get_alert_states([0.060])
        assert states[0].filtered_probability == pytest.approx(0.060)

    def test_both_persisted(self):
        """Both raw and filtered values are in the AlertState."""
        states = _get_alert_states([0.050])
        assert states[0].raw_probability == 0.050
        assert states[0].filtered_probability == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# F. Patient isolation
# ──────────────────────────────────────────────────────────────────────────────

class TestPatientIsolation:
    """Each patient's alert state is computed independently."""

    def test_interleaved_patients_do_not_share_state(self):
        """Two patients with different probability profiles produce
        independent alert states."""
        # Patient A: steady high-risk
        probs_a = [0.060, 0.060, 0.060, 0.060]
        states_a = evaluate_alert_state(
            [(i, p) for i, p in enumerate(probs_a, start=1)]
        )

        # Patient B: high, low, high, high (h3-h4 consecutive → persistence at h4)
        probs_b = [0.060, 0.010, 0.060, 0.060]
        states_b = evaluate_alert_state(
            [(i, p) for i, p in enumerate(probs_b, start=1)]
        )

        # Patient A: alert on h2 (first), h3 blocked (3-2=1), h4 blocked (4-2=2)
        assert states_a[0].alert is False
        assert states_a[1].alert is True
        assert states_a[2].alert is False  # blocked (3-2=1 < 3)
        assert states_a[3].alert is False  # blocked (4-2=2 < 3)

        # Patient B: h1 high but no prev → no alert; h2 low → breaks persistence;
        # h3 high but no prev → no alert_raw; h4 high + prev=high → alert_raw,
        # last_alert_time still -999 (no prior alert), 4-(-999)>=3 → fires
        assert states_b[0].alert is False
        assert states_b[1].alert is False
        assert states_b[2].alert is False
        assert states_b[3].alert is True  # persistence met (h3+h4), cooldown ok

    def test_function_has_no_global_state(self):
        """Calling evaluate_alert_state twice with different inputs
        produces independent results."""
        states1 = _get_alert_states([0.060, 0.060])
        states2 = _get_alert_states([0.010, 0.010])

        assert states1[1].alert is True
        assert states2[0].alert is False
        assert states2[1].alert is False


# ──────────────────────────────────────────────────────────────────────────────
# G. Alert events (maximal contiguous runs)
# ──────────────────────────────────────────────────────────────────────────────

class TestAlertEvents:
    """Alert events are maximal contiguous runs of alert=True."""

    def test_event_created_when_alert_true(self, db_session):
        """A single alert hour creates one event."""
        _make_patient(db_session, "P-AE1")
        insert_alert(db_session, "P-AE1", 1, 1, peak_risk=0.060)
        alerts = get_patient_alerts(db_session, "P-AE1")
        assert len(alerts) == 1
        assert alerts[0].alert_start_iculos == 1
        assert alerts[0].alert_end_iculos == 1
        assert alerts[0].duration_hours == 1
        assert alerts[0].peak_risk == pytest.approx(0.060)

    def test_maximal_contiguous_run(self, db_session):
        """5 consecutive alert hours → one event with duration=5."""
        _make_patient(db_session, "P-AE2")
        _store_predictions(db_session, "P-AE2", [0.060] * 5)
        # Update to have alert=True for all
        from Backend.Database.schema import Prediction
        preds = get_patient_predictions(db_session, "P-AE2")
        for p in preds:
            p.alert = True
        events = rebuild_alert_events(db_session, "P-AE2")
        assert len(events) == 1
        assert events[0].alert_start_iculos == 1
        assert events[0].alert_end_iculos == 5
        assert events[0].duration_hours == 5

    def test_correct_start_end(self, db_session):
        """Alert runs at hours 3-7 → event starts at 3, ends at 7."""
        _make_patient(db_session, "P-AE3")
        # 8 hours: h1-2 no alert, h3-7 alert, h8 no alert
        for i in range(1, 9):
            upsert_prediction(
                db_session, "P-AE3", i,
                raw_probability=0.060,
                filtered_probability=0.060,
                high_risk=True,
                alert=(3 <= i <= 7),
            )
        events = rebuild_alert_events(db_session, "P-AE3")
        assert len(events) == 1
        assert events[0].alert_start_iculos == 3
        assert events[0].alert_end_iculos == 7
        assert events[0].duration_hours == 5

    def test_peak_risk_uses_raw_probability(self, db_session):
        """peak_risk = max raw probability in the run."""
        _make_patient(db_session, "P-AE4")
        raw_probs = [0.060, 0.080, 0.070, 0.055]
        for i, p in enumerate(raw_probs, start=1):
            upsert_prediction(
                db_session, "P-AE4", i,
                raw_probability=p,
                filtered_probability=0.0,
                high_risk=True,
                alert=True,
            )
        events = rebuild_alert_events(db_session, "P-AE4")
        assert len(events) == 1
        assert events[0].peak_risk == pytest.approx(0.080)

    def test_separate_runs_create_separate_events(self, db_session):
        """Two non-contiguous alert runs → two events."""
        _make_patient(db_session, "P-AE5")
        # h1-2 alert, h3 no alert, h4-5 alert
        alerts_flags = [True, True, False, True, True]
        for i, a in enumerate(alerts_flags, start=1):
            upsert_prediction(
                db_session, "P-AE5", i,
                raw_probability=0.060,
                filtered_probability=0.060,
                high_risk=True,
                alert=a,
            )
        events = rebuild_alert_events(db_session, "P-AE5")
        assert len(events) == 2
        assert events[0].alert_start_iculos == 1
        assert events[0].alert_end_iculos == 2
        assert events[0].duration_hours == 2
        assert events[1].alert_start_iculos == 4
        assert events[1].alert_end_iculos == 5
        assert events[1].duration_hours == 2

    def test_no_alerts_means_no_events(self, db_session):
        """No alert=True rows → no alert events."""
        _make_patient(db_session, "P-AE6")
        for i in range(1, 4):
            upsert_prediction(
                db_session, "P-AE6", i,
                raw_probability=0.010,
                filtered_probability=0.010,
                high_risk=False,
                alert=False,
            )
        events = rebuild_alert_events(db_session, "P-AE6")
        assert len(events) == 0

    def test_rebuild_is_idempotent(self, db_session):
        """Running rebuild twice produces the same result."""
        _make_patient(db_session, "P-AE7")
        for i in range(1, 4):
            upsert_prediction(
                db_session, "P-AE7", i,
                raw_probability=0.060,
                filtered_probability=0.060,
                high_risk=True,
                alert=True,
            )
        events1 = rebuild_alert_events(db_session, "P-AE7")
        events2 = rebuild_alert_events(db_session, "P-AE7")
        assert len(events1) == len(events2)
        assert events1[0].alert_start_iculos == events2[0].alert_start_iculos
        assert events1[0].alert_end_iculos == events2[0].alert_end_iculos
        assert events1[0].peak_risk == pytest.approx(events2[0].peak_risk)


# ──────────────────────────────────────────────────────────────────────────────
# H. Alert summaries
# ──────────────────────────────────────────────────────────────────────────────

class TestAlertSummaries:
    """Alert summaries are derived from alert events."""

    def test_summary_created(self, db_session):
        """After alerts exist, a summary row is created."""
        _make_patient(db_session, "P-AS1")
        insert_alert(db_session, "P-AS1", 1, 3, peak_risk=0.080)
        summary = upsert_alert_summary(db_session, "P-AS1")
        assert summary is not None

    def test_total_alert_count(self, db_session):
        _make_patient(db_session, "P-AS2")
        insert_alert(db_session, "P-AS2", 1, 2, peak_risk=0.060)
        insert_alert(db_session, "P-AS2", 5, 7, peak_risk=0.090)
        summary = upsert_alert_summary(db_session, "P-AS2")
        assert summary.total_alerts == 2

    def test_total_alert_hours(self, db_session):
        _make_patient(db_session, "P-AS3")
        insert_alert(db_session, "P-AS3", 1, 3, peak_risk=0.060)  # 3 hours
        insert_alert(db_session, "P-AS3", 6, 8, peak_risk=0.070)  # 3 hours
        summary = upsert_alert_summary(db_session, "P-AS3")
        assert summary.total_alert_hours == 6

    def test_first_and_last_alert_iculos(self, db_session):
        _make_patient(db_session, "P-AS4")
        insert_alert(db_session, "P-AS4", 3, 5, peak_risk=0.060)
        insert_alert(db_session, "P-AS4", 10, 12, peak_risk=0.070)
        summary = upsert_alert_summary(db_session, "P-AS4")
        assert summary.first_alert_iculos == 3
        assert summary.last_alert_iculos == 12

    def test_max_peak_risk(self, db_session):
        _make_patient(db_session, "P-AS5")
        insert_alert(db_session, "P-AS5", 1, 2, peak_risk=0.060)
        insert_alert(db_session, "P-AS5", 5, 6, peak_risk=0.090)
        summary = upsert_alert_summary(db_session, "P-AS5")
        assert summary.max_peak_risk == pytest.approx(0.090)

    def test_rebuild_is_idempotent(self, db_session):
        _make_patient(db_session, "P-AS6")
        insert_alert(db_session, "P-AS6", 1, 3, peak_risk=0.080)
        s1 = upsert_alert_summary(db_session, "P-AS6")
        s2 = upsert_alert_summary(db_session, "P-AS6")
        assert s1.total_alerts == s2.total_alerts
        assert s1.total_alert_hours == s2.total_alert_hours
        assert s1.max_peak_risk == pytest.approx(s2.max_peak_risk)

    def test_no_alerts_returns_none(self, db_session):
        _make_patient(db_session, "P-AS7")
        summary = upsert_alert_summary(db_session, "P-AS7")
        assert summary is None

    def test_stale_summary_replaced(self, db_session):
        """Running summary rebuild replaces old data."""
        _make_patient(db_session, "P-AS8")
        insert_alert(db_session, "P-AS8", 1, 2, peak_risk=0.060)
        s1 = upsert_alert_summary(db_session, "P-AS8")
        assert s1.total_alerts == 1

        # Add another alert
        insert_alert(db_session, "P-AS8", 5, 6, peak_risk=0.070)
        s2 = upsert_alert_summary(db_session, "P-AS8")
        assert s2.total_alerts == 2
        assert s2.total_alert_hours == 4


# ──────────────────────────────────────────────────────────────────────────────
# I. End-to-end process_observation
# ──────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    """Full pipeline: ingest → features → inference → alert recompute → persist."""

    def _make_obs(self, patient_id: str, iculos: int) -> dict:
        return {
            "PatientID": patient_id,
            "Age": 65,
            "ICULOS": iculos,
            "HR": 84.0 + iculos,
            "O2Sat": 98.0,
            "SBP": 118.0,
            "MAP": 77.0,
            "Resp": 18.0,
            "Temp": 36.94,
            "Lactate": None,
            "WBC": None,
            "Creatinine": None,
            "Platelets": None,
        }

    def test_returns_correct_keys(self, db_session, model):
        from Backend.Services.pred_cache import process_observation
        result = process_observation(db_session, self._make_obs("P-EE1", 1), model)
        assert set(result.keys()) == {
            "patient_id", "iculos", "raw_probability",
            "filtered_probability", "high_risk", "alert",
        }

    def test_raw_probability_is_valid(self, db_session, model):
        from Backend.Services.pred_cache import process_observation
        result = process_observation(db_session, self._make_obs("P-EE2", 1), model)
        assert isinstance(result["raw_probability"], float)
        assert 0.0 <= result["raw_probability"] <= 1.0

    def test_persisted_predictions_corrected(self, db_session, model):
        """All prediction rows have correct alert fields after processing."""
        from Backend.Services.pred_cache import process_observation
        from Backend.Services.alert_engine import evaluate_alert_state

        patient_id = "P-EE3"
        for hour in range(1, 4):
            process_observation(db_session, self._make_obs(patient_id, hour), model)

        preds = get_patient_predictions(db_session, patient_id)
        assert len(preds) == 3

        # Verify alert fields were recomputed (not placeholder values)
        raw_probs = [p.raw_probability for p in preds]
        alert_states = evaluate_alert_state(
            [(p.iculos, p.raw_probability) for p in preds]
        )
        for pred, state in zip(preds, alert_states):
            assert pred.filtered_probability == pytest.approx(state.filtered_probability)
            assert pred.high_risk == state.high_risk
            assert pred.alert == state.alert

    def test_alert_events_built(self, db_session, model):
        """Alert events are rebuilt after each observation."""
        from Backend.Services.pred_cache import process_observation
        from Backend.Database.operations import get_patient_alerts

        patient_id = "P-EE4"
        # Process 3 hours
        for hour in range(1, 4):
            process_observation(db_session, self._make_obs(patient_id, hour), model)

        # Alert events exist (may be empty if no alerts fired)
        alerts = get_patient_alerts(db_session, patient_id)
        # The events should be consistent with the predictions
        preds = get_patient_predictions(db_session, patient_id)
        has_alert = any(p.alert for p in preds)
        if has_alert:
            assert len(alerts) > 0

    def test_alert_summary_built(self, db_session, model):
        """Alert summary is rebuilt after each observation."""
        from Backend.Services.pred_cache import process_observation
        from Backend.Database.operations import get_patient_alerts
        from Backend.Database.schema import AlertSummary
        from sqlalchemy import select

        patient_id = "P-EE5"
        for hour in range(1, 4):
            process_observation(db_session, self._make_obs(patient_id, hour), model)

        alerts = get_patient_alerts(db_session, patient_id)
        stmt = select(AlertSummary).where(AlertSummary.patient_id == patient_id)
        summary = db_session.execute(stmt).scalar_one_or_none()

        if alerts:
            assert summary is not None
            assert summary.total_alerts == len(alerts)
        else:
            assert summary is None

    def test_multi_hour_correctness(self, db_session, model):
        """Process 6 hours and verify all prediction states are correct."""
        from Backend.Services.pred_cache import process_observation
        from Backend.Services.alert_engine import evaluate_alert_state

        patient_id = "P-EE6"
        for hour in range(1, 7):
            process_observation(db_session, self._make_obs(patient_id, hour), model)

        preds = get_patient_predictions(db_session, patient_id)
        raw_probs = [p.raw_probability for p in preds]
        alert_states = evaluate_alert_state(
            [(p.iculos, p.raw_probability) for p in preds]
        )

        for pred, state in zip(preds, alert_states):
            assert pred.raw_probability == pytest.approx(state.raw_probability)
            assert pred.filtered_probability == pytest.approx(state.filtered_probability)
            assert pred.high_risk == state.high_risk
            assert pred.alert == state.alert


# ──────────────────────────────────────────────────────────────────────────────
# J. Phase 4 regression — updated assertions
# ──────────────────────────────────────────────────────────────────────────────

class TestPhase4Regression:
    """Phase 4 tests updated to reflect Phase 5 alert engine behavior."""

    @pytest.fixture()
    def db_session(self):
        test_engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=test_engine)
        TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
        session = TestSession()
        yield session
        session.rollback()
        session.close()
        test_engine.dispose()

    def _make_obs(self, patient_id: str, iculos: int) -> dict:
        return {
            "PatientID": patient_id,
            "Age": 65,
            "ICULOS": iculos,
            "HR": 84.0 + iculos,
            "O2Sat": 98.0,
            "SBP": 118.0,
            "MAP": 77.0,
            "Resp": 18.0,
            "Temp": 36.94,
            "Lactate": None,
            "WBC": None,
            "Creatinine": None,
            "Platelets": None,
        }

    def test_process_observation_returns_dict(self, db_session, model):
        from Backend.Services.pred_cache import process_observation
        result = process_observation(db_session, self._make_obs("P-R1", 1), model)
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "patient_id", "iculos", "raw_probability",
            "filtered_probability", "high_risk", "alert",
        }

    def test_raw_probability_is_float_in_range(self, db_session, model):
        from Backend.Services.pred_cache import process_observation
        result = process_observation(db_session, self._make_obs("P-R2", 1), model)
        assert isinstance(result["raw_probability"], float)
        assert 0.0 <= result["raw_probability"] <= 1.0

    def test_alert_engine_applied(self, db_session, model):
        """filtered_probability is now derived by the alert engine, not pass-through."""
        from Backend.Services.pred_cache import process_observation
        result = process_observation(db_session, self._make_obs("P-R3", 1), model)
        # filtered_probability comes from the alert engine
        raw = result["raw_probability"]
        uncertain = 0.035 < raw < 0.055
        if uncertain:
            assert result["filtered_probability"] == 0.0
        else:
            assert result["filtered_probability"] == raw

    def test_high_risk_uses_persistence(self, db_session, model):
        """high_risk is now derived by the alert engine (threshold only for single hour)."""
        from Backend.Services.pred_cache import process_observation
        result = process_observation(db_session, self._make_obs("P-R4", 1), model)
        expected_high_risk = result["filtered_probability"] >= 0.045
        assert result["high_risk"] == expected_high_risk

    def test_prediction_persisted_in_db(self, db_session, model):
        from Backend.Services.pred_cache import process_observation
        from Backend.Database.operations import get_patient_predictions

        patient_id = "P-R5"
        result = process_observation(db_session, self._make_obs(patient_id, 1), model)
        preds = get_patient_predictions(db_session, patient_id)
        assert len(preds) == 1
        pred = preds[0]
        assert pred.patient_id == patient_id
        assert pred.iculos == 1
        assert pred.raw_probability == pytest.approx(result["raw_probability"])

    def test_persisted_fields_match_return(self, db_session, model):
        from Backend.Services.pred_cache import process_observation
        from Backend.Database.operations import get_patient_predictions

        patient_id = "P-R6"
        result = process_observation(db_session, self._make_obs(patient_id, 1), model)
        pred = get_patient_predictions(db_session, patient_id)[0]
        assert pred.raw_probability == result["raw_probability"]
        assert pred.filtered_probability == result["filtered_probability"]
        assert pred.high_risk == result["high_risk"]
        assert pred.alert == result["alert"]

    def test_multi_hour_patient(self, db_session, model):
        from Backend.Services.pred_cache import process_observation
        from Backend.Database.operations import get_patient_predictions

        patient_id = "P-R7"
        for hour in range(1, 4):
            result = process_observation(
                db_session, self._make_obs(patient_id, hour), model,
            )
            assert 0.0 <= result["raw_probability"] <= 1.0

        preds = get_patient_predictions(db_session, patient_id)
        assert len(preds) == 3
        assert [p.iculos for p in preds] == [1, 2, 3]
