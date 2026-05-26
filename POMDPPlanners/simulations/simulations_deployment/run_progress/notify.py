# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Notifier interface, implementations, and signal-handler installer.

The :class:`Notifier` Protocol defines four events that the simulator emits
during a run:

- ``run_started`` — fired at the entry point of the experiment loop.
- ``episode_completed`` — fired once per completed episode (heartbeat).
- ``run_finished`` — fired on a clean exit.
- ``run_failed`` — fired when an exception or signal aborts the run.

Two concrete implementations are provided:

- :class:`NullNotifier` — no-op; used when notifications are disabled.
- :class:`SlackNotifier` — writes events to the local progress DB and posts
  to a Slack incoming webhook (via stdlib ``urllib.request``).

The :func:`build_notifier` factory consumes a :class:`NotificationConfig`
and returns the appropriate implementation; it does **not** read any
environment variables itself — env-var resolution is the responsibility of
the caller (typically :meth:`NotificationConfig.from_env` invoked by the
``SimulationsAPI`` layer). The :func:`install_signal_handlers` helper
attaches SIGTERM/SIGINT handlers that emit ``run_failed`` before re-raising.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import types
import urllib.error
import urllib.request
from typing import Any, Callable, Protocol, runtime_checkable

from POMDPPlanners.simulations.simulations_deployment.run_progress.config import (
    NotificationConfig,
)
from POMDPPlanners.simulations.simulations_deployment.run_progress.db import ProgressDB

_SLACK_POST_TIMEOUT_SECONDS = 5

_SignalFrame = types.FrameType | None
_SignalHandler = Callable[[int, _SignalFrame], Any] | int | signal.Handlers | None


@runtime_checkable
class Notifier(Protocol):
    """Protocol describing the four run-lifecycle events a simulator emits.

    Implementations may write to a database, post to a chat tool, both, or
    neither. The Protocol is :func:`runtime_checkable` so tests can assert
    structural conformance without inheritance.
    """

    def run_started(self, run_id: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Record that a run has begun.

        Args:
            run_id: Unique id for the run.
            metadata: Optional metadata dict describing the run.
        """

    def episode_completed(self, run_id: str) -> None:
        """Record that a single episode finished (heartbeat).

        Args:
            run_id: The run to which the episode belongs.
        """

    def run_finished(self, run_id: str) -> None:
        """Record that a run finished cleanly.

        Args:
            run_id: The run to mark as finished.
        """

    def run_failed(self, run_id: str, error: str) -> None:
        """Record that a run aborted with an error.

        Args:
            run_id: The run that failed.
            error: Stringified exception or signal-name describing the cause.
        """


class NullNotifier:
    """No-op :class:`Notifier`. All methods discard their arguments.

    Used when notifications are disabled (``SLACK_WEBHOOK_URL`` unset,
    ``POMDPPLANNERS_DISABLE_NOTIFY=1`` set, or running under pytest).

    Example:
        Basic usage::

            notifier = NullNotifier()
            notifier.run_started("rid", metadata={"k": "v"})
            notifier.episode_completed("rid")
            notifier.run_finished("rid")
    """

    def run_started(self, run_id: str, *, metadata: dict[str, Any] | None = None) -> None:
        """No-op."""
        del run_id, metadata

    def episode_completed(self, run_id: str) -> None:
        """No-op."""
        del run_id

    def run_finished(self, run_id: str) -> None:
        """No-op."""
        del run_id

    def run_failed(self, run_id: str, error: str) -> None:
        """No-op."""
        del run_id, error


class SlackNotifier:
    """:class:`Notifier` that writes to :class:`ProgressDB` and Slack.

    ``run_started`` / ``run_finished`` / ``run_failed`` write to the DB
    **and** POST a short message to the Slack webhook. ``episode_completed``
    only writes a heartbeat to the DB — it does not post to Slack, to avoid
    spamming the channel during long runs.

    Slack-side failures are caught and logged at WARNING level via the
    supplied logger; they do not propagate to the caller.

    Example:
        Basic usage::

            notifier = SlackNotifier(
                webhook_url="https://hooks.slack.com/services/T/B/X",
                experiment_name="tiger_experiment",
            )
            notifier.run_started("rid-1", metadata={"episodes": 100})
            notifier.episode_completed("rid-1")
            notifier.run_finished("rid-1")
    """

    def __init__(
        self,
        webhook_url: str,
        experiment_name: str,
        *,
        db: ProgressDB | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize the Slack notifier.

        Args:
            webhook_url: A Slack incoming-webhook URL.
            experiment_name: Human-readable label recorded for each run
                started via this notifier.
            db: Optional :class:`ProgressDB` instance to write to. If
                ``None``, a default ``ProgressDB()`` is constructed.
            logger: Optional logger used to emit warnings when Slack posts
                fail. If ``None``, a logger named ``run_progress.notify``
                is used.
        """
        self.webhook_url = webhook_url
        self.experiment_name = experiment_name
        self.db = db if db is not None else ProgressDB()
        self.logger = logger if logger is not None else logging.getLogger("run_progress.notify")

    def run_started(self, run_id: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Record start in DB and POST a start message to Slack.

        Args:
            run_id: Unique id for the run.
            metadata: Optional metadata dict written to the DB.
        """
        self.db.start_run(run_id, self.experiment_name, metadata or {})
        self._post(f":rocket: Run *started*: `{self.experiment_name}` (id=`{run_id}`)")

    def episode_completed(self, run_id: str) -> None:
        """Update the heartbeat in the DB. Does not post to Slack.

        Args:
            run_id: The run to which the episode belongs.
        """
        self.db.heartbeat(run_id)

    def run_finished(self, run_id: str) -> None:
        """Mark the run finished in the DB and POST a success message.

        Args:
            run_id: The run to mark as finished.
        """
        self.db.finish_run(run_id, status="finished", error=None)
        self._post(f":white_check_mark: Run *finished*: `{self.experiment_name}` (id=`{run_id}`)")

    def run_failed(self, run_id: str, error: str) -> None:
        """Mark the run failed in the DB and POST a failure message.

        Args:
            run_id: The run that failed.
            error: Stringified exception or signal-name describing the cause.
        """
        self.db.finish_run(run_id, status="failed", error=error)
        self._post(f":x: Run *failed*: `{self.experiment_name}` (id=`{run_id}`): {error}")

    def _post(self, text: str) -> None:
        _post_slack_text(
            webhook_url=self.webhook_url,
            text=text,
            logger=self.logger,
        )


def post_trial_milestone(
    webhook_url: str,
    *,
    experiment_name: str,
    run_id: str,
    completed_trials: int,
    total_trials: int,
    config_name: str,
    best_score: float | None,
    logger: logging.Logger | None = None,
) -> None:
    """POST a Slack message reporting hyperparameter-tuning progress.

    Stateless helper called once per ``HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL``
    Optuna trials. ``webhook_url`` must be a non-empty Slack incoming-webhook
    URL — callers (typically the in-task Optuna callback) are responsible
    for skipping the call when notifications are disabled. Network errors
    are caught and logged at WARNING level; nothing propagates to the
    caller.

    Args:
        webhook_url: Slack incoming-webhook URL.
        experiment_name: Human-readable label for the tuning run, included
            in the message body.
        run_id: Unique id of the parent tuning run (same id used by
            ``run_started``/``run_finished``).
        completed_trials: Number of trials completed globally so far.
        total_trials: Total trials scheduled across all configs.
        config_name: Name of the config whose trial just completed; lets the
            reader see which leg of the sweep is currently running.
        best_score: Best objective value seen so far in the current config
            (e.g. ``study.best_value``). Pass ``None`` if no trial has
            produced a usable score yet — the message will note this.
        logger: Optional logger for swallowed-error WARNING. Defaults to
            a logger named ``run_progress.notify``.
    """
    progress_logger = logger if logger is not None else logging.getLogger("run_progress.notify")
    best_score_text = f"{best_score:.4f}" if best_score is not None else "n/a"
    text = (
        f":ladder: Tuning *milestone*: `{experiment_name}` (id=`{run_id}`) "
        f"— {completed_trials}/{total_trials} trials done, "
        f"current config `{config_name}`, best so far {best_score_text}"
    )
    _post_slack_text(webhook_url=webhook_url, text=text, logger=progress_logger)


def _post_slack_text(
    webhook_url: str,
    text: str,
    logger: logging.Logger,
) -> None:
    body = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_SLACK_POST_TIMEOUT_SECONDS) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Slack notification failed: %s", exc)


def build_notifier(
    experiment_name: str,
    *,
    config: NotificationConfig,
    logger: logging.Logger | None = None,
) -> Notifier:
    """Construct the appropriate :class:`Notifier` from a config.

    This function performs **no environment reads**. The caller passes an
    explicit :class:`NotificationConfig`; if that config is active
    (non-empty webhook URL AND ``disable=False``), a :class:`SlackNotifier`
    is returned. Otherwise a :class:`NullNotifier` is returned. Env-var
    resolution belongs in :meth:`NotificationConfig.from_env` — typically
    invoked by the ``SimulationsAPI`` layer.

    Args:
        experiment_name: Human-readable label used for run records and
            Slack messages.
        config: The :class:`NotificationConfig` controlling whether and
            where Slack messages are sent.
        logger: Optional logger forwarded to :class:`SlackNotifier`.

    Returns:
        A :class:`Notifier` instance — either :class:`NullNotifier` or
        :class:`SlackNotifier` depending on whether ``config.is_active()``.
    """
    if not config.is_active() or config.webhook_url is None:
        return NullNotifier()
    return SlackNotifier(
        webhook_url=config.webhook_url,
        experiment_name=experiment_name,
        logger=logger,
    )


def install_signal_handlers(
    notifier: Notifier,
    run_id: str,
) -> Callable[[], None]:
    """Install SIGTERM/SIGINT handlers that emit :meth:`Notifier.run_failed`.

    The installed handlers:

    1. Call ``notifier.run_failed(run_id, error="signal:<NAME>")``.
       Exceptions inside the notifier are swallowed.
    2. Chain to the previous handler if it is callable. Non-callable
       handlers (``signal.SIG_DFL`` / ``signal.SIG_IGN``, both integer
       sentinels) are skipped, and the signal is re-delivered to the
       process with default handling restored.

    Returns a callable that uninstalls both handlers (restoring whatever
    was previously installed). Calling the uninstall function multiple
    times is safe.

    Args:
        notifier: The :class:`Notifier` to emit ``run_failed`` on signal.
        run_id: The run identifier to pass to ``run_failed``.

    Returns:
        An idempotent uninstall callable. The caller should invoke it
        during normal teardown (e.g. ``BaseSimulator.__exit__``) so signal
        handling reverts to the prior configuration.
    """
    prev_handlers: dict[int, _SignalHandler] = {}
    handler = _make_signal_handler(notifier, run_id, prev_handlers)
    installed_signals: list[int] = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        prev_handlers[signum] = signal.getsignal(signum)
        try:
            signal.signal(signum, handler)
        except ValueError:
            # signal.signal raises ValueError if not on the main thread.
            del prev_handlers[signum]
            continue
        installed_signals.append(signum)

    def uninstall() -> None:
        while installed_signals:
            signum = installed_signals.pop()
            try:
                signal.signal(signum, prev_handlers[signum])
            except (ValueError, KeyError):
                pass

    return uninstall


def _make_signal_handler(
    notifier: Notifier,
    run_id: str,
    prev_handlers: dict[int, _SignalHandler],
) -> Callable[[int, _SignalFrame], None]:
    def handler(signum: int, frame: _SignalFrame) -> None:
        try:
            notifier.run_failed(run_id, f"signal:{signal.Signals(signum).name}")
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # never let notifier errors mask the signal
        prev = prev_handlers.get(signum)
        if callable(prev):
            prev(signum, frame)
        else:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    return handler
