# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

# pylint: disable=protected-access  # Tests need to inspect internal state
"""Tests for :mod:`POMDPPlanners.simulations.simulations_deployment.run_progress.db`."""

import json
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from POMDPPlanners.simulations.simulations_deployment.run_progress.db import (
    ProgressDB,
    RunRow,
)


@pytest.fixture
def db_path(tmp_path):
    """Fixture returning a fresh SQLite path inside the pytest tmp dir."""
    return tmp_path / "progress.db"


@pytest.fixture
def progress_db(db_path):
    """Fixture constructing a :class:`ProgressDB` on a fresh file."""
    return ProgressDB(path=db_path)


def test_initialization_creates_parent_directory(tmp_path):
    """Verify that ProgressDB auto-creates a missing parent directory.

    Purpose: Validates first-time setup on a host with no prior progress DB.

    Given: A path under a directory tree that does not yet exist on disk.
    When: ProgressDB is instantiated with that path.
    Then: The parent directory exists and the SQLite file is created.

    Test type: unit
    """
    nested_path = tmp_path / "a" / "b" / "c" / "progress.db"
    assert not nested_path.parent.exists()

    db = ProgressDB(path=nested_path)

    assert nested_path.parent.exists()
    assert db.path == nested_path


def test_initialization_uses_env_var_when_no_path_given(monkeypatch, tmp_path):
    """Verify that ProgressDB reads POMDP_PROGRESS_DB when no path is given.

    Purpose: Validates the env-var override path used by the watcher CLI.

    Given: POMDP_PROGRESS_DB is set to a path under tmp_path.
    When: ProgressDB is instantiated with no explicit path.
    Then: The DB is created at the env-var location.

    Test type: unit
    """
    env_path = tmp_path / "from_env.db"
    monkeypatch.setenv("POMDP_PROGRESS_DB", str(env_path))

    db = ProgressDB()

    assert db.path == env_path


def test_initialization_falls_back_to_default_path(monkeypatch, tmp_path):
    """Verify default path resolves under the user's cache directory.

    Purpose: Validates the no-config default for new installations.

    Given: POMDP_PROGRESS_DB is unset and Path.home() is monkeypatched.
    When: ProgressDB is instantiated with no path argument.
    Then: The path resolves to ``<home>/.cache/POMDPPlanners/progress.db``.

    Test type: unit
    """
    monkeypatch.delenv("POMDP_PROGRESS_DB", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db = ProgressDB()

    assert db.path == tmp_path / ".cache" / "POMDPPlanners" / "progress.db"
    assert db.path.parent.exists()


def test_schema_is_idempotent(db_path):
    """Verify that constructing ProgressDB twice on the same file is safe.

    Purpose: Validates the CREATE TABLE IF NOT EXISTS contract.

    Given: A ProgressDB has already been instantiated on a path.
    When: A second ProgressDB is instantiated on the same path.
    Then: No exception is raised, and existing data is preserved.

    Test type: unit
    """
    first = ProgressDB(path=db_path)
    first.start_run("rid-1", "exp-A", {"k": "v"})

    ProgressDB(path=db_path)  # second open should not raise or wipe rows

    fetched = first.get_run("rid-1")
    assert fetched is not None
    assert fetched.experiment_name == "exp-A"


def test_wal_mode_is_enabled(progress_db):
    """Verify that journal_mode=WAL is set on the SQLite file.

    Purpose: Validates the parent/watcher concurrency invariant.

    Given: A freshly initialized ProgressDB.
    When: PRAGMA journal_mode is queried on the file.
    Then: The reported mode is ``wal``.

    Test type: unit
    """
    with closing(sqlite3.connect(progress_db.path)) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_start_run_records_status_and_metadata(progress_db):
    """Verify that start_run inserts a row with the expected fields.

    Purpose: Validates the run-start hook writes correct initial state.

    Given: A fresh ProgressDB and a metadata dict.
    When: start_run is called with a run_id, experiment_name, and metadata.
    Then: get_run returns a RunRow with status='running', the metadata
        JSON-round-tripped, no error_msg, no finished_at, and a
        last_heartbeat_at equal to started_at.

    Test type: unit
    """
    progress_db.start_run("rid-2", "tiger_exp", {"episodes": 100, "seed": 42})

    row = progress_db.get_run("rid-2")
    assert isinstance(row, RunRow)
    assert row.run_id == "rid-2"
    assert row.experiment_name == "tiger_exp"
    assert row.status == "running"
    assert row.metadata == {"episodes": 100, "seed": 42}
    assert row.error_msg is None
    assert row.finished_at is None
    assert row.stall_notified_at is None
    assert row.last_heartbeat_at == row.started_at


def test_start_run_with_no_metadata_stores_empty_dict(progress_db):
    """Verify that omitting metadata results in an empty dict, not NULL.

    Purpose: Validates the metadata-default contract for callers that have
    nothing meaningful to record.

    Given: A fresh ProgressDB.
    When: start_run is called without a metadata argument.
    Then: get_run returns a row whose metadata is an empty dict.

    Test type: unit
    """
    progress_db.start_run("rid-3", "tiger_exp")

    row = progress_db.get_run("rid-3")
    assert row is not None
    assert row.metadata == {}


def test_heartbeat_updates_only_heartbeat_field(progress_db):
    """Verify that heartbeat advances last_heartbeat_at and leaves the rest.

    Purpose: Validates the per-episode update path is minimal and correct.

    Given: A running row whose last_heartbeat_at equals started_at.
    When: A short delay is awaited and then heartbeat is called.
    Then: last_heartbeat_at is strictly greater than the original; status,
        started_at, finished_at, and error_msg are unchanged.

    Test type: unit
    """
    progress_db.start_run("rid-4", "exp")
    before = progress_db.get_run("rid-4")
    assert before is not None
    time.sleep(0.01)  # ensure ISO timestamps differ

    progress_db.heartbeat("rid-4")

    after = progress_db.get_run("rid-4")
    assert after is not None
    assert after.last_heartbeat_at > before.last_heartbeat_at
    assert after.started_at == before.started_at
    assert after.status == "running"
    assert after.finished_at is None
    assert after.error_msg is None


def test_finish_run_marks_finished_with_timestamp(progress_db):
    """Verify the clean-finish update path.

    Purpose: Validates the run-finished hook writes the correct status.

    Given: A running row.
    When: finish_run is called with status='finished' and no error.
    Then: status becomes 'finished', finished_at is populated, error_msg
        remains None.

    Test type: unit
    """
    progress_db.start_run("rid-5", "exp")

    progress_db.finish_run("rid-5", status="finished", error=None)

    row = progress_db.get_run("rid-5")
    assert row is not None
    assert row.status == "finished"
    assert row.finished_at is not None
    assert row.error_msg is None


def test_finish_run_marks_failed_with_error_message(progress_db):
    """Verify the crash update path captures the error message.

    Purpose: Validates the run-failed hook records the cause.

    Given: A running row.
    When: finish_run is called with status='failed' and an error string.
    Then: status becomes 'failed', finished_at is populated, error_msg
        contains the supplied string.

    Test type: unit
    """
    progress_db.start_run("rid-6", "exp")

    progress_db.finish_run("rid-6", status="failed", error="RuntimeError: boom")

    row = progress_db.get_run("rid-6")
    assert row is not None
    assert row.status == "failed"
    assert row.finished_at is not None
    assert row.error_msg == "RuntimeError: boom"


def test_get_run_returns_none_for_unknown_id(progress_db):
    """Verify get_run does not raise when the id is absent.

    Purpose: Validates lookup safety so callers can use a simple None check.

    Given: A ProgressDB containing no rows.
    When: get_run is called with an unknown id.
    Then: ``None`` is returned (no exception raised).

    Test type: unit
    """
    assert progress_db.get_run("does-not-exist") is None


def test_list_stalled_returns_only_old_running_rows(progress_db, db_path):
    """Verify the stall query filters by status, age, and notification state.

    Purpose: Validates the watcher's primary query.

    Given: Four rows — running+old, running+fresh, finished+old, and
        running+old-but-already-notified.
    When: list_stalled is called with a threshold that classifies "old"
        heartbeats as stalled.
    Then: Only the first row is returned (and ordered oldest-first when
        multiple stalled rows are present).

    Test type: unit
    """
    progress_db.start_run("running_old", "exp")
    progress_db.start_run("running_fresh", "exp")
    progress_db.start_run("finished_old", "exp")
    progress_db.finish_run("finished_old", status="finished", error=None)
    progress_db.start_run("running_old_notified", "exp")
    progress_db.mark_stall_notified("running_old_notified")

    # Rewind two rows' heartbeats far into the past to make them "old".
    _backdate_heartbeat(db_path, "running_old", seconds=3600)
    _backdate_heartbeat(db_path, "running_old_notified", seconds=3600)

    stalled = progress_db.list_stalled(threshold_seconds=60)

    stalled_ids = [r.run_id for r in stalled]
    assert stalled_ids == ["running_old"]


def test_list_stalled_ordered_oldest_first(progress_db, db_path):
    """Verify ordering when multiple stalled runs exist.

    Purpose: Validates the watcher always notifies oldest stalls first.

    Given: Two stalled running rows with backdated heartbeats — one older
        than the other.
    When: list_stalled is called.
    Then: The returned list is ordered with the oldest heartbeat first.

    Test type: unit
    """
    progress_db.start_run("older", "exp")
    progress_db.start_run("newer", "exp")
    _backdate_heartbeat(db_path, "older", seconds=7200)
    _backdate_heartbeat(db_path, "newer", seconds=3600)

    stalled = progress_db.list_stalled(threshold_seconds=60)

    assert [r.run_id for r in stalled] == ["older", "newer"]


def test_mark_stall_notified_sets_timestamp(progress_db):
    """Verify mark_stall_notified records when the notification was sent.

    Purpose: Validates the watcher's dedupe-update path.

    Given: A running row with stall_notified_at IS NULL.
    When: mark_stall_notified is called.
    Then: The row's stall_notified_at becomes a non-empty ISO timestamp.

    Test type: unit
    """
    progress_db.start_run("rid-7", "exp")

    progress_db.mark_stall_notified("rid-7")

    row = progress_db.get_run("rid-7")
    assert row is not None
    assert row.stall_notified_at is not None
    assert "T" in row.stall_notified_at  # ISO-8601 format sanity


def test_metadata_round_trips_nested_structures(progress_db):
    """Verify that nested JSON metadata survives a write/read cycle.

    Purpose: Validates that arbitrary JSON-serializable structures are
    preserved end-to-end without flattening.

    Given: A metadata dict containing nested dicts and lists.
    When: start_run is called and the row is read back.
    Then: The metadata equals the original (no truncation, no type drift).

    Test type: unit
    """
    metadata = {
        "envs": ["tiger", "rocksample"],
        "planner": {"name": "POMCP", "params": {"c": 1.0, "iters": 500}},
    }
    progress_db.start_run("rid-8", "exp", metadata)

    row = progress_db.get_run("rid-8")
    assert row is not None
    assert row.metadata == metadata


def _backdate_heartbeat(db_path: Path, run_id: str, seconds: int) -> None:
    """Helper: rewind a row's last_heartbeat_at by ``seconds`` for stall tests."""
    new_ts = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "UPDATE runs SET last_heartbeat_at = ? WHERE run_id = ?",
            (new_ts, run_id),
        )
        conn.commit()


def test_helper_backdate_heartbeat_works(progress_db, db_path):
    """Smoke test for the backdate helper used by other tests.

    Purpose: Validates the test helper does what its callers depend on.

    Given: A running row.
    When: _backdate_heartbeat is called with a large delta.
    Then: The stored last_heartbeat_at is older than the original by at
        least that many seconds.

    Test type: unit
    """
    progress_db.start_run("rid-9", "exp")
    original = progress_db.get_run("rid-9")
    assert original is not None

    _backdate_heartbeat(db_path, "rid-9", seconds=3600)

    updated = progress_db.get_run("rid-9")
    assert updated is not None
    assert updated.last_heartbeat_at < original.last_heartbeat_at


def test_runrow_metadata_is_dict_not_string(progress_db):
    """Verify the metadata field is decoded JSON, not the raw JSON string.

    Purpose: Validates the row-to-RunRow conversion deserializes metadata.

    Given: A row stored with a non-empty metadata dict.
    When: get_run returns a RunRow.
    Then: row.metadata is a dict (not a string) and indexes work.

    Test type: unit
    """
    progress_db.start_run("rid-10", "exp", {"k": 1})

    row = progress_db.get_run("rid-10")
    assert row is not None
    assert isinstance(row.metadata, dict)
    assert row.metadata["k"] == 1
    assert not isinstance(row.metadata, str)
    # confirm it's not "just happen to" be JSON bytes
    json.dumps(row.metadata)  # raises if non-serializable
