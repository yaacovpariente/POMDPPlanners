# SPDX-License-Identifier: MIT

"""Notification configuration dataclass for the run_progress module.

`NotificationConfig` bundles every parameter the run-progress
infrastructure needs into a single explicit object. Simulation classes
(`BaseSimulator`, `HyperParameterOptimizer`) accept one as a constructor
argument; the `SimulationsAPI` layer is the only thing that reads env
vars (via :meth:`NotificationConfig.from_env`).

This split keeps simulation classes pure DI — they no longer touch
:data:`os.environ` at construction time — while preserving the existing
zero-config UX through the API layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_SLACK_WEBHOOK_ENV_VAR = "SLACK_WEBHOOK_URL"
_DISABLE_ENV_VAR = "POMDPPLANNERS_DISABLE_NOTIFY"
_PYTEST_ENV_VAR = "PYTEST_CURRENT_TEST"
_TRIAL_INTERVAL_ENV_VAR = "HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL"
_PROGRESS_DB_ENV_VAR = "POMDP_PROGRESS_DB"
_DEFAULT_TRIAL_INTERVAL = 50


@dataclass
class NotificationConfig:
    """Configuration for Slack + progress-DB notifications.

    Construct directly for programmatic per-instance control, or via
    :meth:`from_env` to mirror the legacy env-var-driven behaviour at the
    API layer.

    Attributes:
        webhook_url: Slack incoming-webhook URL. ``None`` means "no Slack
            posts" (events still write to the progress DB if a notifier
            is constructed, but the SlackNotifier is replaced with a
            NullNotifier).
        trial_interval: Number of completed Optuna trials between
            milestone Slack messages during hyperparameter tuning. ``0``
            disables milestone messages. Only the
            :class:`HyperParameterOptimizer` reads this; the simulator
            ignores it.
        progress_db_path: Optional override for the SQLite progress DB
            path. ``None`` falls back to the path resolved by
            :class:`ProgressDB` (``POMDP_PROGRESS_DB`` env var, then
            ``~/.cache/POMDPPlanners/progress.db``).
        disable: Hard kill switch. When ``True``, the notifier built
            from this config is always a :class:`NullNotifier`, even if
            ``webhook_url`` is set. Equivalent to the legacy
            ``POMDPPLANNERS_DISABLE_NOTIFY=1`` env var.

    Example:
        Explicit construction (e.g. routing two simulations in one process
        to different Slack channels)::

            cfg_a = NotificationConfig(webhook_url="https://hooks.slack.com/A")
            cfg_b = NotificationConfig(webhook_url="https://hooks.slack.com/B")

        Environment-driven construction (used by ``LocalSimulationsAPI``)::

            cfg = NotificationConfig.from_env()
    """

    webhook_url: Optional[str] = None
    trial_interval: int = 0
    progress_db_path: Optional[Path] = None
    disable: bool = False

    @classmethod
    def from_env(cls) -> "NotificationConfig":
        """Build a config from the run-progress environment variables.

        Reads ``SLACK_WEBHOOK_URL``, ``POMDPPLANNERS_DISABLE_NOTIFY``,
        ``PYTEST_CURRENT_TEST``, ``HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL``,
        and ``POMDP_PROGRESS_DB``. Returns a config that is functionally
        disabled (``disable=True``) when running under pytest or when
        ``POMDPPLANNERS_DISABLE_NOTIFY=1`` is set; otherwise reflects the
        webhook URL (or ``None`` if unset) and the milestone interval
        (default 50).

        Returns:
            A new :class:`NotificationConfig` instance.
        """
        disable = bool(os.environ.get(_PYTEST_ENV_VAR)) or os.environ.get(_DISABLE_ENV_VAR) == "1"
        webhook_url = os.environ.get(_SLACK_WEBHOOK_ENV_VAR, "") or None
        trial_interval = _resolve_int_env(_TRIAL_INTERVAL_ENV_VAR, default=_DEFAULT_TRIAL_INTERVAL)
        progress_db_path_raw = os.environ.get(_PROGRESS_DB_ENV_VAR, "")
        progress_db_path = Path(progress_db_path_raw) if progress_db_path_raw else None
        return cls(
            webhook_url=webhook_url,
            trial_interval=trial_interval,
            progress_db_path=progress_db_path,
            disable=disable,
        )

    @classmethod
    def disabled(cls) -> "NotificationConfig":
        """Return a config with notifications hard-disabled.

        Used as the default for simulation classes constructed without an
        explicit config (e.g. unit tests, workflows called outside a
        :class:`SimulationsAPI`).

        Returns:
            A :class:`NotificationConfig` with ``disable=True``.
        """
        return cls(disable=True)

    def is_active(self) -> bool:
        """Return ``True`` iff this config should produce a real Slack notifier.

        A config is active when it is not hard-disabled and has a non-empty
        webhook URL. When ``False``, :func:`build_notifier` returns a
        :class:`NullNotifier`.
        """
        return (not self.disable) and bool(self.webhook_url)


def _resolve_int_env(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else 0
