# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""CLI watcher that posts Slack alerts for stalled experiment runs.

Intended to be invoked from cron (typically once per minute):

.. code-block:: text

    * * * * * /path/to/.venv/bin/python -m \
        POMDPPlanners.simulations.simulations_deployment.run_progress.watcher \
        --threshold-seconds 3600

For every run in the progress DB whose status is ``running`` AND whose
``last_heartbeat_at`` is older than ``--threshold-seconds`` AND which has
not previously been stall-notified, the watcher POSTs a short Slack
message and marks the row notified so subsequent ticks do not duplicate.

The watcher is **stateless** — it makes one pass through the DB and exits.
Cron handles "the watcher died" semantics so no separate daemon is needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

from POMDPPlanners.simulations.simulations_deployment.run_progress.db import (
    ProgressDB,
    RunRow,
)

_SLACK_WEBHOOK_ENV_VAR = "SLACK_WEBHOOK_URL"
_SLACK_POST_TIMEOUT_SECONDS = 5
_DEFAULT_THRESHOLD_SECONDS = 3600

_logger = logging.getLogger("run_progress.watcher")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one watcher pass.

    Args:
        argv: Optional argv override (for tests). When ``None``, falls
            back to :data:`sys.argv` after the program name.

    Returns:
        Exit status. ``0`` on success (including the "no stalled runs"
        case). Non-zero is reserved for hard configuration errors.
    """
    args = _parse_args(argv)
    webhook_url = args.webhook_url or os.environ.get(_SLACK_WEBHOOK_ENV_VAR, "")

    db = ProgressDB(path=args.db)
    stalled = db.list_stalled(threshold_seconds=args.threshold_seconds)
    if not stalled:
        _logger.info("No stalled runs found.")
        return 0

    if not webhook_url:
        _logger.warning(
            "Found %d stalled run(s) but SLACK_WEBHOOK_URL is unset; " "skipping notifications.",
            len(stalled),
        )
        return 0

    _logger.info("Notifying about %d stalled run(s).", len(stalled))
    for run in stalled:
        _post_to_slack(webhook_url, _format_stall_message(run))
        db.mark_stall_notified(run.run_id)
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_progress.watcher",
        description="Post Slack alerts for stalled POMDPPlanners experiment runs.",
    )
    parser.add_argument(
        "--threshold-seconds",
        type=int,
        default=_DEFAULT_THRESHOLD_SECONDS,
        help=(
            "Heartbeat age (in seconds) above which a still-running entry is "
            "considered stalled. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=(
            "Path to the SQLite progress DB. Defaults to "
            "$POMDP_PROGRESS_DB if set, else "
            "~/.cache/POMDPPlanners/progress.db."
        ),
    )
    parser.add_argument(
        "--webhook-url",
        type=str,
        default=None,
        help=(
            "Slack incoming-webhook URL. Defaults to $SLACK_WEBHOOK_URL. "
            "If neither is set, stalled runs are logged but not posted."
        ),
    )
    return parser.parse_args(argv)


def _format_stall_message(run: RunRow) -> str:
    return (
        f":hourglass_flowing_sand: Run *stalled*: `{run.experiment_name}` "
        f"(id=`{run.run_id}`, host=`{run.host}`, pid=`{run.pid}`, "
        f"last heartbeat at `{run.last_heartbeat_at}`)"
    )


def _post_to_slack(webhook_url: str, text: str) -> None:
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
        _logger.warning("Slack notification failed: %s", exc)


if __name__ == "__main__":  # pragma: no cover - exercised via tests
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
