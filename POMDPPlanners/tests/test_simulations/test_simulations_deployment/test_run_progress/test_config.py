# SPDX-License-Identifier: MIT

"""Tests for :mod:`POMDPPlanners.simulations.simulations_deployment.run_progress.config`."""

from pathlib import Path

from POMDPPlanners.simulations.simulations_deployment.run_progress.config import (
    NotificationConfig,
)


def test_default_construction_is_inactive():
    """Verify the bare-default config is silent.

    Purpose: Validates the safe-by-default contract — constructing
    NotificationConfig() with no arguments produces a config that
    :func:`build_notifier` will turn into a NullNotifier.

    Given: No arguments.
    When: NotificationConfig() is constructed.
    Then: All fields hold their safe defaults; is_active() returns False.

    Test type: unit
    """
    cfg = NotificationConfig()

    assert cfg.webhook_url is None
    assert cfg.trial_interval == 0
    assert cfg.progress_db_path is None
    assert cfg.disable is False
    assert cfg.is_active() is False


def test_disabled_factory_returns_inactive_config():
    """Verify :meth:`NotificationConfig.disabled` is hard-disabled.

    Purpose: Validates the factory used as the default by simulation
    classes constructed without an explicit config (tests, workflows).

    Given: NotificationConfig.disabled().
    When: is_active() is checked.
    Then: disable is True and is_active() returns False — even if a
        webhook were also set (verified by manually constructing).

    Test type: unit
    """
    cfg = NotificationConfig.disabled()

    assert cfg.disable is True
    assert cfg.is_active() is False

    cfg_with_webhook_and_disabled = NotificationConfig(
        webhook_url="https://hooks.slack.com/test", disable=True
    )
    assert cfg_with_webhook_and_disabled.is_active() is False


def test_is_active_requires_webhook_and_not_disabled():
    """Verify is_active() returns True only when webhook is set and not disabled.

    Purpose: Documents the exact gate :func:`build_notifier` uses.

    Given: Various combinations of webhook_url and disable.
    When: is_active() is checked.
    Then: True iff webhook_url is non-empty AND disable is False.

    Test type: unit
    """
    assert NotificationConfig(webhook_url="https://x").is_active() is True
    assert NotificationConfig(webhook_url="https://x", disable=True).is_active() is False
    assert NotificationConfig(webhook_url=None).is_active() is False
    assert NotificationConfig(webhook_url="").is_active() is False


def test_from_env_returns_disabled_under_pytest(monkeypatch):
    """Verify from_env() returns disabled when PYTEST_CURRENT_TEST is set.

    Purpose: Validates the test-safety gate so test suites never page
    Slack even when SLACK_WEBHOOK_URL is set in the environment.

    Given: PYTEST_CURRENT_TEST is set, SLACK_WEBHOOK_URL is also set.
    When: NotificationConfig.from_env() is called.
    Then: disable is True; the webhook still appears in the config (so
        callers can introspect it), but is_active() returns False.

    Test type: unit
    """
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "fake")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

    cfg = NotificationConfig.from_env()

    assert cfg.disable is True
    assert cfg.is_active() is False


def test_from_env_returns_disabled_when_kill_switch_set(monkeypatch):
    """Verify POMDPPLANNERS_DISABLE_NOTIFY=1 produces a disabled config.

    Purpose: Validates the explicit per-run opt-out env var.

    Given: PYTEST_CURRENT_TEST is unset, SLACK_WEBHOOK_URL is set, and
        POMDPPLANNERS_DISABLE_NOTIFY=1 is set.
    When: NotificationConfig.from_env() is called.
    Then: disable is True; is_active() returns False.

    Test type: unit
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("POMDPPLANNERS_DISABLE_NOTIFY", "1")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

    cfg = NotificationConfig.from_env()

    assert cfg.disable is True
    assert cfg.is_active() is False


def test_from_env_returns_inactive_when_webhook_unset(monkeypatch):
    """Verify the no-webhook path yields an inactive (but not disabled) config.

    Purpose: Validates that absent SLACK_WEBHOOK_URL silently disables
    Slack posting without flipping the hard `disable` flag.

    Given: PYTEST_CURRENT_TEST and POMDPPLANNERS_DISABLE_NOTIFY are unset,
        SLACK_WEBHOOK_URL is unset.
    When: NotificationConfig.from_env() is called.
    Then: disable is False, webhook_url is None, is_active() is False.

    Test type: unit
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("POMDPPLANNERS_DISABLE_NOTIFY", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)

    cfg = NotificationConfig.from_env()

    assert cfg.disable is False
    assert cfg.webhook_url is None
    assert cfg.is_active() is False


def test_from_env_returns_active_config_when_webhook_set(monkeypatch):
    """Verify the happy-path env reading produces an active config.

    Purpose: Validates the zero-config UX through SimulationsAPI: just
    export SLACK_WEBHOOK_URL and notifications work.

    Given: PYTEST_CURRENT_TEST and POMDPPLANNERS_DISABLE_NOTIFY are unset,
        SLACK_WEBHOOK_URL is set.
    When: NotificationConfig.from_env() is called.
    Then: webhook_url matches; disable is False; is_active() is True.

    Test type: unit
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("POMDPPLANNERS_DISABLE_NOTIFY", raising=False)
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/abc")

    cfg = NotificationConfig.from_env()

    assert cfg.webhook_url == "https://hooks.slack.com/abc"
    assert cfg.disable is False
    assert cfg.is_active() is True


def test_from_env_reads_trial_interval(monkeypatch):
    """Verify HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL flows into the config.

    Purpose: Validates the milestone-cadence env var path.

    Given: HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL=10 in the env.
    When: NotificationConfig.from_env() is called.
    Then: trial_interval is 10.

    Test type: unit
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("POMDPPLANNERS_DISABLE_NOTIFY", raising=False)
    monkeypatch.setenv("HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL", "10")

    cfg = NotificationConfig.from_env()

    assert cfg.trial_interval == 10


def test_from_env_default_trial_interval_is_50(monkeypatch):
    """Verify the milestone-interval default when the env var is unset.

    Purpose: Locks in the documented default cadence.

    Given: HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL is unset.
    When: NotificationConfig.from_env() is called.
    Then: trial_interval is 50.

    Test type: unit
    """
    monkeypatch.delenv("HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL", raising=False)

    cfg = NotificationConfig.from_env()

    assert cfg.trial_interval == 50


def test_from_env_trial_interval_zero_when_non_positive(monkeypatch):
    """Verify a non-positive interval is normalised to 0 (disabled).

    Purpose: Defensive against bad env values; 0 means "no milestone
    posts" in the optimizer.

    Given: HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL is set to "0" or a
        negative number.
    When: NotificationConfig.from_env() is called.
    Then: trial_interval is 0.

    Test type: unit
    """
    monkeypatch.setenv("HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL", "0")
    assert NotificationConfig.from_env().trial_interval == 0

    monkeypatch.setenv("HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL", "-5")
    assert NotificationConfig.from_env().trial_interval == 0


def test_from_env_trial_interval_falls_back_on_malformed(monkeypatch):
    """Verify a non-integer env value falls back to the default of 50.

    Purpose: Avoid crashes if the env var is fat-fingered.

    Given: HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL is set to a non-integer
        string.
    When: NotificationConfig.from_env() is called.
    Then: trial_interval is the documented default of 50.

    Test type: unit
    """
    monkeypatch.setenv("HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL", "not-a-number")

    cfg = NotificationConfig.from_env()

    assert cfg.trial_interval == 50


def test_from_env_reads_progress_db_path(monkeypatch, tmp_path):
    """Verify POMDP_PROGRESS_DB is threaded into progress_db_path.

    Purpose: Validates the per-user DB-path override used for multi-tenant
    setups and explicit watcher targeting.

    Given: POMDP_PROGRESS_DB points to a path under tmp_path.
    When: NotificationConfig.from_env() is called.
    Then: progress_db_path equals that Path.

    Test type: unit
    """
    target = tmp_path / "my_progress.db"
    monkeypatch.setenv("POMDP_PROGRESS_DB", str(target))

    cfg = NotificationConfig.from_env()

    assert cfg.progress_db_path == target


def test_from_env_progress_db_path_is_none_when_unset(monkeypatch):
    """Verify progress_db_path is None when POMDP_PROGRESS_DB is unset.

    Purpose: Validates the "let ProgressDB pick the default" path.

    Given: POMDP_PROGRESS_DB is unset.
    When: NotificationConfig.from_env() is called.
    Then: progress_db_path is None.

    Test type: unit
    """
    monkeypatch.delenv("POMDP_PROGRESS_DB", raising=False)

    cfg = NotificationConfig.from_env()

    assert cfg.progress_db_path is None


def test_path_field_accepts_explicit_construction(tmp_path):
    """Verify NotificationConfig stores a Path object as-given.

    Purpose: Validates programmatic construction with a custom DB path.

    Given: An explicit progress_db_path argument.
    When: NotificationConfig is constructed.
    Then: The field stores the exact Path object.

    Test type: unit
    """
    target = tmp_path / "custom.db"

    cfg = NotificationConfig(progress_db_path=target)

    assert cfg.progress_db_path == target
    assert isinstance(cfg.progress_db_path, Path)
