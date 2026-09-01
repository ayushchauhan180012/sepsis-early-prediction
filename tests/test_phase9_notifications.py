"""Phase 9 Step 2 correctness tests — notification abstraction (D-027).

Tests cover:
  A. NotificationChannel ABC cannot be instantiated directly
  B. NoOpNotification.send succeeds and does nothing
  C. ConsoleNotification logs the alert through the logger
  D. Factory returns the correct channel type for "noop"/"console"
  E. Factory uses settings.notification_channel when channel_name is None
  F. Factory raises ValueError for unknown channel names
  G. Alert data is passed through correctly to channels
"""

from __future__ import annotations

import logging

import pytest

from Backend.Services.notifications import (
    ConsoleNotification,
    NoOpNotification,
    NotificationChannel,
    get_notification_channel,
)


def _sample_alert() -> dict:
    return {
        "iculos": 5,
        "raw_probability": 0.120,
        "filtered_probability": 0.120,
        "high_risk": True,
        "alert": True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# A. ABC not instantiable
# ──────────────────────────────────────────────────────────────────────────────

class TestABC:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            NotificationChannel()

    def test_noop_is_a_notification_channel(self):
        assert isinstance(NoOpNotification(), NotificationChannel)

    def test_console_is_a_notification_channel(self):
        assert isinstance(ConsoleNotification(), NotificationChannel)


# ──────────────────────────────────────────────────────────────────────────────
# B. NoOpNotification behavior
# ──────────────────────────────────────────────────────────────────────────────

class TestNoOp:
    def test_send_succeeds(self):
        NoOpNotification().send("P-1", _sample_alert())

    def test_send_does_nothing(self):
        """NoOp.send has no side effects and returns None."""
        assert NoOpNotification().send("P-1", _sample_alert()) is None


# ──────────────────────────────────────────────────────────────────────────────
# C. ConsoleNotification logs the alert
# ──────────────────────────────────────────────────────────────────────────────

class TestConsole:
    def test_send_logs_alert(self, caplog):
        caplog.set_level(logging.WARNING, logger="Backend.Services.notifications")
        ConsoleNotification().send("P-1", _sample_alert())
        assert any("SEPSIS ALERT" in rec.message for rec in caplog.records)
        assert any("P-1" in rec.message for rec in caplog.records)

    def test_send_returns_none(self, caplog):
        caplog.set_level(logging.WARNING, logger="Backend.Services.notifications")
        assert ConsoleNotification().send("P-1", _sample_alert()) is None


# ──────────────────────────────────────────────────────────────────────────────
# D. Factory returns correct types
# ──────────────────────────────────────────────────────────────────────────────

class TestFactoryTypes:
    def test_noop_channel(self):
        assert isinstance(get_notification_channel("noop"), NoOpNotification)

    def test_console_channel(self):
        assert isinstance(get_notification_channel("console"), ConsoleNotification)


# ──────────────────────────────────────────────────────────────────────────────
# E. Factory uses configured setting when omitted
# ──────────────────────────────────────────────────────────────────────────────

class TestFactoryConfig:
    def test_defaults_to_configured_setting(self, monkeypatch):
        from Backend.Services import notifications as notif_module
        monkeypatch.setattr(notif_module.settings, "notification_channel", "console")
        assert isinstance(get_notification_channel(), ConsoleNotification)

    def test_explicit_argument_overrides_setting(self, monkeypatch):
        from Backend.Services import notifications as notif_module
        monkeypatch.setattr(notif_module.settings, "notification_channel", "console")
        assert isinstance(get_notification_channel("noop"), NoOpNotification)


# ──────────────────────────────────────────────────────────────────────────────
# F. Unknown channel ValueError
# ──────────────────────────────────────────────────────────────────────────────

class TestFactoryUnknown:
    def test_unknown_channel_raises_valueerror(self):
        with pytest.raises(ValueError):
            get_notification_channel("email")

    def test_unknown_channel_from_config_raises(self, monkeypatch):
        from Backend.Services import notifications as notif_module
        monkeypatch.setattr(notif_module.settings, "notification_channel", "sms")
        with pytest.raises(ValueError):
            get_notification_channel()


# ──────────────────────────────────────────────────────────────────────────────
# G. Alert data passed through correctly
# ──────────────────────────────────────────────────────────────────────────────

class TestDataPassThrough:
    def test_console_logs_all_alert_fields(self, caplog):
        caplog.set_level(logging.WARNING, logger="Backend.Services.notifications")
        alert = _sample_alert()
        ConsoleNotification().send("P-42", alert)
        logged = " ".join(rec.message for rec in caplog.records)
        assert "P-42" in logged
        assert str(alert["iculos"]) in logged
        assert str(alert["raw_probability"]) in logged
        assert str(alert["filtered_probability"]) in logged
        assert str(alert["alert"]) in logged
