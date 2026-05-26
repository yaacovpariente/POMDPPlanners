# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Progress tracking + notification module for long experiment runs.

Two layers:

- **Layer 1 (in-process)**: a :class:`Notifier` constructed automatically by
  :class:`~POMDPPlanners.simulations.simulator.base_simulator.BaseSimulator`.
  Emits ``run_started`` / ``episode_completed`` (heartbeat) / ``run_finished``
  / ``run_failed`` events to a local SQLite DB and (if a Slack webhook is
  configured) to Slack.

- **Layer 2 (external watcher)**: a CLI module
  (:mod:`POMDPPlanners.simulations.simulations_deployment.run_progress.watcher`)
  that reads the same DB and emits ``run_stalled`` Slack notifications for
  runs whose last heartbeat is older than a configurable threshold. Run from
  cron to catch hard process death (SIGKILL / OOM / reboot).

Whether notifications are sent is controlled by the deployment environment:
:func:`build_notifier` inspects ``SLACK_WEBHOOK_URL`` and a few opt-out env
vars. User-facing code constructs ``SimulationsAPI`` as usual; no
configuration is threaded through the public API.
"""

from POMDPPlanners.simulations.simulations_deployment.run_progress.config import (
    NotificationConfig,
)
from POMDPPlanners.simulations.simulations_deployment.run_progress.db import (
    ProgressDB,
    RunRow,
)
from POMDPPlanners.simulations.simulations_deployment.run_progress.notify import (
    NullNotifier,
    Notifier,
    SlackNotifier,
    build_notifier,
    install_signal_handlers,
    post_trial_milestone,
)

__all__ = [
    "Notifier",
    "NotificationConfig",
    "NullNotifier",
    "ProgressDB",
    "RunRow",
    "SlackNotifier",
    "build_notifier",
    "install_signal_handlers",
    "post_trial_milestone",
]
