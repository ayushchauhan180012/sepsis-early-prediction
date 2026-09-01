"""Notification channel abstraction (Phase 9, D-027).

Provides a small pluggable abstraction for delivering sepsis alerts:

- ``NotificationChannel`` — abstract base class.
- ``NoOpNotification`` — default/safe backend; does nothing.
- ``ConsoleNotification`` — logs the alert through the application logger.
- ``get_notification_channel`` — environment/config-driven factory.

Design notes (D-027):
- Channel selection is environment/config driven (``NOTIFICATION_CHANNEL``).
- Only simulated backends are implemented; no real external providers.
- Notification delivery is **non-critical**: failures must never affect
  prediction or alert results.  Individual channels do not silently swallow
  every exception here — the dispatch boundary (in ``Backend/app.py``) is
  responsible for catching notification failures so they cannot affect
  prediction results.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from Backend.config import settings

log = logging.getLogger(__name__)


class NotificationChannel(ABC):
    """Abstract base for a sepsis-alert notification channel.

    Implementations must provide :meth:`send`, which delivers a single alert
    for a patient.
    """

    @abstractmethod
    def send(self, patient_id: str, alert_data: dict) -> None:
        """Deliver a sepsis alert for ``patient_id``.

        Parameters
        ----------
        patient_id : str
            The patient identifier associated with the alert.
        alert_data : dict
            The alert information to deliver (e.g. iculos, raw_probability,
            filtered_probability, high_risk, alert).
        """
        raise NotImplementedError


class NoOpNotification(NotificationChannel):
    """Safe default backend — performs no delivery action."""

    def send(self, patient_id: str, alert_data: dict) -> None:
        """Do nothing (D-027 default/no-op backend)."""


class ConsoleNotification(NotificationChannel):
    """Logs the sepsis alert through the application logger.

    Uses the existing Python ``logging`` infrastructure (matching the rest of
    the backend); does not use ``print()``.
    """

    def send(self, patient_id: str, alert_data: dict) -> None:
        iculos = alert_data.get("iculos")
        raw = alert_data.get("raw_probability")
        filtered = alert_data.get("filtered_probability")
        log.warning(
            "SEPSIS ALERT — patient_id=%s iculos=%s raw_probability=%s "
            "filtered_probability=%s alert=%s",
            patient_id, iculos, raw, filtered, alert_data.get("alert"),
        )


def get_notification_channel(
    channel_name: str | None = None,
) -> NotificationChannel:
    """Return the configured notification channel (factory, D-027).

    Parameters
    ----------
    channel_name : str or None
        The channel to instantiate.  If ``None``, reads
        ``settings.notification_channel``.

    Returns
    -------
    NotificationChannel
        ``NoOpNotification`` for ``"noop"``, ``ConsoleNotification`` for
        ``"console"``.

    Raises
    ------
    ValueError
        If ``channel_name`` is not a recognized channel.
    """
    if channel_name is None:
        channel_name = settings.notification_channel

    if channel_name == "noop":
        return NoOpNotification()
    if channel_name == "console":
        return ConsoleNotification()

    raise ValueError(
        f"Unknown notification channel: {channel_name!r}. "
        "Supported channels: 'noop', 'console'."
    )
