# SPDX-License-Identifier: MIT

# pylint: disable=protected-access  # Tests need to inspect internal helpers
"""Tests for :mod:`POMDPPlanners.simulations.simulations_deployment.run_progress.watcher`."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from POMDPPlanners.simulations.simulations_deployment.run_progress import watcher
from POMDPPlanners.simulations.simulations_deployment.run_progress.db import ProgressDB


@pytest.fixture
def db_path(tmp_path) -> Path:
    """A fresh SQLite path inside the pytest tmp dir."""
    return tmp_path / "progress.db"


@pytest.fixture
def populated_db(db_path):
    """ProgressDB with a healthy run, a stalled run, and a finished old run."""
    db = ProgressDB(path=db_path)
    db.start_run("healthy", "exp_healthy", {})
    db.start_run("stalled", "exp_stalled", {"episodes": 100})
    db.start_run("done_old", "exp_done")
    db.finish_run("done_old", status="finished", error=None)
    _backdate_heartbeat(db_path, "stalled", seconds=7200)
    _backdate_heartbeat(db_path, "done_old", seconds=7200)
    return db


def test_main_no_stalled_runs_returns_zero(db_path):
    """Verify exit 0 when the DB has no stalled rows.

    Purpose: Validates the happy "everything is fine" tick.

    Given: A fresh DB with one fresh running row.
    When: main() is invoked with a short threshold and a webhook URL.
    Then: Exit status is 0 and urlopen is never called.

    Test type: integration
    """
    db = ProgressDB(path=db_path)
    db.start_run("fresh", "exp_fresh", {})

    with patch("urllib.request.urlopen") as mock_urlopen:
        exit_code = watcher.main(
            [
                "--db",
                str(db_path),
                "--threshold-seconds",
                "3600",
                "--webhook-url",
                "https://hooks.slack.com/test",
            ]
        )

    assert exit_code == 0
    assert mock_urlopen.call_count == 0


def test_main_posts_one_message_per_stalled_run(populated_db, db_path):
    """Verify exactly one Slack POST is sent per stalled run.

    Purpose: Validates the watcher's core notification path.

    Given: A populated DB with one stalled run and one finished old run.
    When: main() runs with a webhook URL and a 60-second threshold.
    Then: urlopen is called exactly once with the stalled run's
        experiment name in the JSON body.

    Test type: integration
    """
    del populated_db

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"ok"
        exit_code = watcher.main(
            [
                "--db",
                str(db_path),
                "--threshold-seconds",
                "60",
                "--webhook-url",
                "https://hooks.slack.com/test",
            ]
        )

    assert exit_code == 0
    assert mock_urlopen.call_count == 1
    request = mock_urlopen.call_args[0][0]
    body = json.loads(request.data.decode("utf-8"))
    assert "exp_stalled" in body["text"]


def test_main_is_idempotent_across_calls(populated_db, db_path):
    """Verify the watcher does not double-notify on a second tick.

    Purpose: Validates the stall_notified_at dedupe column. Cron may fire
    the watcher every minute; only the *first* tick after a stall should
    post.

    Given: A populated DB with one stalled run.
    When: main() runs twice in succession with the same threshold.
    Then: urlopen is called exactly once total (the second tick is silent).

    Test type: integration
    """
    del populated_db

    args = [
        "--db",
        str(db_path),
        "--threshold-seconds",
        "60",
        "--webhook-url",
        "https://hooks.slack.com/test",
    ]

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"ok"
        watcher.main(args)
        watcher.main(args)

    assert mock_urlopen.call_count == 1


def test_main_no_webhook_logs_and_returns_zero(populated_db, db_path, caplog):
    """Verify the watcher logs (does not crash) when no webhook is configured.

    Purpose: Validates the "configured for stall detection but no webhook"
    fallback path so the watcher is safe to install before Slack is set up.

    Given: A populated DB with one stalled run and *no* webhook configured.
    When: main() runs.
    Then: Exit status is 0, urlopen is never called, and a WARNING is
        logged that mentions the stalled count.

    Test type: integration
    """
    del populated_db

    with patch("urllib.request.urlopen") as mock_urlopen:
        with caplog.at_level("WARNING", logger="run_progress.watcher"):
            exit_code = watcher.main(["--db", str(db_path), "--threshold-seconds", "60"])

    assert exit_code == 0
    assert mock_urlopen.call_count == 0
    assert any("stalled" in r.message.lower() for r in caplog.records)


def test_main_db_arg_overrides_env(populated_db, db_path, monkeypatch):
    """Verify --db arg takes precedence over POMDP_PROGRESS_DB.

    Purpose: Validates CLI override semantics (testability for cron).

    Given: POMDP_PROGRESS_DB points to a *different* empty DB while a
        populated DB exists at a separate path.
    When: main() is invoked with --db pointing at the populated DB.
    Then: The populated DB is used (one Slack post for the stalled run).

    Test type: integration
    """
    del populated_db
    other_db = db_path.parent / "other.db"
    ProgressDB(path=other_db)  # empty
    monkeypatch.setenv("POMDP_PROGRESS_DB", str(other_db))

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"ok"
        watcher.main(
            [
                "--db",
                str(db_path),
                "--threshold-seconds",
                "60",
                "--webhook-url",
                "https://hooks.slack.com/test",
            ]
        )

    assert mock_urlopen.call_count == 1


def test_main_webhook_env_var_used_when_arg_absent(populated_db, db_path, monkeypatch):
    """Verify SLACK_WEBHOOK_URL env var is read when --webhook-url is absent.

    Purpose: Validates the cron-friendly env-only invocation path.

    Given: A populated DB and SLACK_WEBHOOK_URL set in the env, no CLI flag.
    When: main() runs.
    Then: Exactly one Slack POST is sent to the env-supplied URL.

    Test type: integration
    """
    del populated_db
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/env-url")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"ok"
        watcher.main(["--db", str(db_path), "--threshold-seconds", "60"])

    assert mock_urlopen.call_count == 1
    posted_url = mock_urlopen.call_args[0][0].full_url
    assert posted_url == "https://hooks.slack.com/env-url"


def test_main_slack_failure_does_not_propagate(populated_db, db_path):
    """Verify Slack outages do not cause the watcher to exit non-zero.

    Purpose: Validates resilience — cron should never see a flaky exit due
    to a transient Slack hiccup.

    Given: A populated DB with one stalled run and urlopen raising URLError.
    When: main() runs.
    Then: Exit code is 0 (no propagation), and the row is still marked as
        notified so the watcher does not retry on the next tick.

    Test type: integration
    """
    del populated_db
    import urllib.error  # pylint: disable=import-outside-toplevel

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no net")):
        exit_code = watcher.main(
            [
                "--db",
                str(db_path),
                "--threshold-seconds",
                "60",
                "--webhook-url",
                "https://hooks.slack.com/test",
            ]
        )

    assert exit_code == 0
    db = ProgressDB(path=db_path)
    row = db.get_run("stalled")
    assert row is not None
    assert row.stall_notified_at is not None


def test_format_stall_message_mentions_key_fields(populated_db):
    """Verify the formatted message includes ids the user needs to act on.

    Purpose: Validates the human-readable contract of stall messages.

    Given: A stalled RunRow fetched from the populated DB.
    When: _format_stall_message renders it.
    Then: The message includes experiment_name, run_id, host, pid, and
        the last_heartbeat_at timestamp.

    Test type: unit
    """
    stalled = populated_db.list_stalled(threshold_seconds=60)
    assert stalled, "test setup invariant violated"
    message = watcher._format_stall_message(stalled[0])

    assert stalled[0].experiment_name in message
    assert stalled[0].run_id in message
    assert stalled[0].host in message
    assert str(stalled[0].pid) in message
    assert stalled[0].last_heartbeat_at in message


def _backdate_heartbeat(db_path: Path, run_id: str, seconds: int) -> None:
    """Test helper: rewind a run's last_heartbeat_at by ``seconds``."""
    new_ts = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "UPDATE runs SET last_heartbeat_at = ? WHERE run_id = ?",
            (new_ts, run_id),
        )
        conn.commit()
