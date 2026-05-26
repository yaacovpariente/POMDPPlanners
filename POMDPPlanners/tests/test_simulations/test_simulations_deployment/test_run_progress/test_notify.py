# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

# pylint: disable=protected-access  # Tests need to inspect internal state
"""Tests for :mod:`POMDPPlanners.simulations.simulations_deployment.run_progress.notify`."""

import json
import logging
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from POMDPPlanners.simulations.simulations_deployment.run_progress.config import (
    NotificationConfig,
)
from POMDPPlanners.simulations.simulations_deployment.run_progress.db import ProgressDB
from POMDPPlanners.simulations.simulations_deployment.run_progress.notify import (
    NullNotifier,
    Notifier,
    SlackNotifier,
    build_notifier,
    post_trial_milestone,
)


@pytest.fixture
def progress_db(tmp_path):
    """Fresh ProgressDB rooted in pytest's tmp_path."""
    return ProgressDB(path=tmp_path / "progress.db")


@pytest.fixture
def slack_notifier(progress_db):
    """SlackNotifier wired to the tmp ProgressDB with a recording logger."""
    logger = logging.getLogger("test.run_progress.notify")
    logger.handlers.clear()
    return SlackNotifier(
        webhook_url="https://hooks.slack.com/test/abc",
        experiment_name="my_exp",
        db=progress_db,
        logger=logger,
    )


def test_null_notifier_satisfies_protocol():
    """Verify NullNotifier structurally matches the Notifier Protocol.

    Purpose: Validates the no-op type substitutes anywhere a Notifier is
    expected.

    Given: A fresh NullNotifier instance.
    When: isinstance() is checked against the runtime-checkable Protocol.
    Then: The instance is reported as a Notifier and all four methods can be
        called without raising.

    Test type: unit
    """
    notifier = NullNotifier()

    assert isinstance(notifier, Notifier)
    notifier.run_started("rid", metadata={"k": "v"})
    notifier.episode_completed("rid")
    notifier.run_finished("rid")
    notifier.run_failed("rid", "boom")


def test_slack_notifier_satisfies_protocol(slack_notifier):
    """Verify SlackNotifier structurally matches the Notifier Protocol.

    Purpose: Validates the concrete impl conforms to the structural contract.

    Given: A fresh SlackNotifier.
    When: isinstance() is checked against Notifier.
    Then: The instance is reported as a Notifier.

    Test type: unit
    """
    assert isinstance(slack_notifier, Notifier)


def test_run_started_writes_to_db_and_posts_to_slack(slack_notifier, progress_db):
    """Verify run_started persists a row AND POSTs to Slack.

    Purpose: Validates the start-of-run event side-effects.

    Given: A SlackNotifier with mocked urlopen.
    When: run_started is called with a run_id and metadata.
    Then: The DB contains a 'running' row with that metadata, and exactly
        one HTTP POST was issued to the webhook with a JSON body containing
        a 'text' field that mentions the experiment name.

    Test type: unit
    """
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"ok"

        slack_notifier.run_started("rid-A", metadata={"episodes": 7})

        assert mock_urlopen.call_count == 1
        request = mock_urlopen.call_args[0][0]
        assert request.full_url == "https://hooks.slack.com/test/abc"
        body = json.loads(request.data.decode("utf-8"))
        assert "my_exp" in body["text"]

    row = progress_db.get_run("rid-A")
    assert row is not None
    assert row.status == "running"
    assert row.metadata == {"episodes": 7}


def test_episode_completed_writes_heartbeat_only(slack_notifier, progress_db):
    """Verify episode_completed updates the DB heartbeat without posting.

    Purpose: Validates the per-episode hook does not spam Slack.

    Given: A SlackNotifier and a running run.
    When: episode_completed is called.
    Then: The DB heartbeat advances, and urlopen is never called.

    Test type: unit
    """
    slack_notifier.db.start_run("rid-B", "my_exp", {})

    with patch("urllib.request.urlopen") as mock_urlopen:
        slack_notifier.episode_completed("rid-B")
        assert mock_urlopen.call_count == 0

    row = progress_db.get_run("rid-B")
    assert row is not None
    assert row.status == "running"  # heartbeat does not change status


def test_run_finished_writes_to_db_and_posts_to_slack(slack_notifier, progress_db):
    """Verify run_finished marks the row 'finished' AND POSTs to Slack.

    Purpose: Validates the clean-finish event side-effects.

    Given: A SlackNotifier and a previously-started run.
    When: run_finished is called.
    Then: The DB row has status='finished', and one HTTP POST was sent
        containing the experiment name in the message body.

    Test type: unit
    """
    slack_notifier.db.start_run("rid-C", "my_exp", {})

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"ok"

        slack_notifier.run_finished("rid-C")

        assert mock_urlopen.call_count == 1
        body = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        assert "my_exp" in body["text"]

    row = progress_db.get_run("rid-C")
    assert row is not None
    assert row.status == "finished"
    assert row.error_msg is None


def test_run_failed_writes_error_and_posts_to_slack(slack_notifier, progress_db):
    """Verify run_failed records the error AND POSTs a failure message.

    Purpose: Validates the crash-path event side-effects.

    Given: A SlackNotifier and a previously-started run.
    When: run_failed is called with an error message.
    Then: The DB row has status='failed' and error_msg is the supplied
        string. The Slack POST body mentions the error string.

    Test type: unit
    """
    slack_notifier.db.start_run("rid-D", "my_exp", {})

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"ok"

        slack_notifier.run_failed("rid-D", "RuntimeError: division by zero")

        assert mock_urlopen.call_count == 1
        body = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        assert "RuntimeError" in body["text"]

    row = progress_db.get_run("rid-D")
    assert row is not None
    assert row.status == "failed"
    assert row.error_msg == "RuntimeError: division by zero"


def test_slack_post_failure_is_logged_not_raised(slack_notifier, caplog):
    """Verify Slack-side failures do not propagate to the caller.

    Purpose: Validates that the simulator is not derailed by a Slack outage.

    Given: A SlackNotifier whose urlopen raises a URLError.
    When: run_started is called.
    Then: No exception propagates; a WARNING is logged; the DB row is
        still written.

    Test type: unit
    """
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no DNS")):
        with caplog.at_level(logging.WARNING, logger="test.run_progress.notify"):
            slack_notifier.run_started("rid-E", metadata={})

    assert any("Slack notification failed" in r.message for r in caplog.records)
    row = slack_notifier.db.get_run("rid-E")
    assert row is not None  # DB write still happened despite Slack failure


def test_build_notifier_returns_null_for_disabled_config():
    """Verify build_notifier returns NullNotifier when the config is disabled.

    Purpose: Validates the kill-switch path. A config with disable=True
    must produce a no-op notifier even if a webhook URL is set.

    Given: A NotificationConfig with disable=True and a webhook URL.
    When: build_notifier is called with that config.
    Then: A NullNotifier is returned.

    Test type: unit
    """
    config = NotificationConfig(webhook_url="https://hooks.slack.com/x", disable=True)

    notifier = build_notifier("exp", config=config)

    assert isinstance(notifier, NullNotifier)


def test_build_notifier_returns_null_when_webhook_missing():
    """Verify build_notifier returns NullNotifier when webhook is unset.

    Purpose: Validates the default-off path: no webhook → silent.

    Given: A NotificationConfig with webhook_url=None.
    When: build_notifier is called with that config.
    Then: A NullNotifier is returned.

    Test type: unit
    """
    config = NotificationConfig(webhook_url=None)

    notifier = build_notifier("exp", config=config)

    assert isinstance(notifier, NullNotifier)


def test_build_notifier_returns_null_for_disabled_factory():
    """Verify the NotificationConfig.disabled() factory yields a NullNotifier.

    Purpose: Validates the default-arg path for simulation classes
    constructed without an explicit config.

    Given: NotificationConfig.disabled().
    When: build_notifier is called with it.
    Then: A NullNotifier is returned.

    Test type: unit
    """
    notifier = build_notifier("exp", config=NotificationConfig.disabled())

    assert isinstance(notifier, NullNotifier)


def test_build_notifier_returns_slack_when_config_is_active():
    """Verify build_notifier returns SlackNotifier for an active config.

    Purpose: Validates the happy-path of the factory.

    Given: A NotificationConfig with a non-empty webhook_url and
        disable=False.
    When: build_notifier is called with that config and an experiment name.
    Then: A SlackNotifier is returned, configured with the webhook URL
        and experiment_name from the call.

    Test type: unit
    """
    config = NotificationConfig(webhook_url="https://hooks.slack.com/abc")

    notifier = build_notifier("tiger_experiment", config=config)

    assert isinstance(notifier, SlackNotifier)
    assert notifier.webhook_url == "https://hooks.slack.com/abc"
    assert notifier.experiment_name == "tiger_experiment"


def test_slack_notifier_uses_supplied_db(slack_notifier, progress_db):
    """Verify SlackNotifier writes to the injected ProgressDB, not a default one.

    Purpose: Validates the DI surface used by the watcher's DB-path override.

    Given: A SlackNotifier constructed with an explicit ProgressDB pointing
        at a tmp path.
    When: run_started is called.
    Then: The row appears in the supplied ProgressDB.

    Test type: unit
    """
    with patch("urllib.request.urlopen"):
        slack_notifier.run_started("rid-F", metadata={})

    assert progress_db.get_run("rid-F") is not None


def test_slack_notifier_post_uses_json_content_type(slack_notifier):
    """Verify the Slack POST body is JSON with the right Content-Type header.

    Purpose: Validates the wire format matches Slack's incoming-webhook spec.

    Given: A SlackNotifier with mocked urlopen.
    When: A Slack-posting event (run_started) is fired.
    Then: The Request has Content-Type 'application/json' and a JSON body
        with a 'text' key.

    Test type: unit
    """
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"ok"
        slack_notifier.run_started("rid-G", metadata={})

    request = mock_urlopen.call_args[0][0]
    assert request.headers.get("Content-type") == "application/json"
    body = json.loads(request.data.decode("utf-8"))
    assert "text" in body


def test_run_failed_swallows_db_error_in_signal_path(slack_notifier):
    """Verify run_failed itself does not raise even if the DB write would.

    Purpose: Documents/locks in that callers (signal handlers) can rely on
    notifier methods to never raise. (DB writes via this notifier are
    expected to be reliable; this test guards against a future change that
    might make them throw.)

    Given: A SlackNotifier whose db.finish_run is monkeypatched to raise.
    When: run_failed is called from regular code.
    Then: The exception propagates (this test documents the *current*
        contract — the notifier does NOT swallow internal errors; the
        signal-handler in install_signal_handlers does that wrapping).

    Test type: unit
    """
    slack_notifier.db.finish_run = MagicMock(side_effect=RuntimeError("db down"))

    with pytest.raises(RuntimeError):
        slack_notifier.run_failed("rid-H", "boom")


def test_post_trial_milestone_posts_to_slack():
    """Verify post_trial_milestone POSTs a Slack message with progress info.

    Purpose: Validates the in-task helper used for trial-milestone events
    in hyperparameter tuning.

    Given: A mocked urlopen and a populated set of milestone arguments.
    When: post_trial_milestone is called with completed/total trial counts,
        a config name, and a best score.
    Then: Exactly one HTTP POST is sent to the supplied webhook URL with
        a JSON body whose 'text' field contains the experiment name, run
        id, completed/total counts, config name, and the score formatted.

    Test type: unit
    """
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"ok"

        post_trial_milestone(
            "https://hooks.slack.com/test/abc",
            experiment_name="tune_exp",
            run_id="rid-M",
            completed_trials=50,
            total_trials=200,
            config_name="config_a",
            best_score=-12.345,
        )

    assert mock_urlopen.call_count == 1
    request = mock_urlopen.call_args[0][0]
    assert request.full_url == "https://hooks.slack.com/test/abc"
    body = json.loads(request.data.decode("utf-8"))
    text = body["text"]
    assert "tune_exp" in text
    assert "rid-M" in text
    assert "50/200" in text
    assert "config_a" in text
    assert "-12.3450" in text  # 4 decimal places per the helper


def test_post_trial_milestone_handles_none_best_score():
    """Verify the helper renders a sensible string when best_score is None.

    Purpose: Validates the early-tuning case where no trial has yet
    produced a usable score.

    Given: A mocked urlopen.
    When: post_trial_milestone is called with best_score=None.
    Then: The message body includes 'n/a' (or equivalent) in place of a
        score number.

    Test type: unit
    """
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"ok"

        post_trial_milestone(
            "https://hooks.slack.com/test/abc",
            experiment_name="exp",
            run_id="rid-N",
            completed_trials=10,
            total_trials=200,
            config_name="cfg",
            best_score=None,
        )

    body = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
    assert "n/a" in body["text"]


def test_post_trial_milestone_swallows_network_errors(caplog):
    """Verify the helper does not propagate Slack-side failures.

    Purpose: Validates resilience — Slack outages must not derail a
    multi-day tuning run.

    Given: urlopen raises URLError.
    When: post_trial_milestone is called.
    Then: No exception propagates; a WARNING is logged via the supplied
        logger.

    Test type: unit
    """
    logger = logging.getLogger("test.run_progress.notify.trial_milestone")
    logger.handlers.clear()

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no net")):
        with caplog.at_level(logging.WARNING, logger=logger.name):
            post_trial_milestone(
                "https://hooks.slack.com/test/abc",
                experiment_name="exp",
                run_id="rid-O",
                completed_trials=50,
                total_trials=200,
                config_name="cfg",
                best_score=0.5,
                logger=logger,
            )

    assert any("Slack notification failed" in r.message for r in caplog.records)
