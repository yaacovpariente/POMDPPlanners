# SPDX-License-Identifier: MIT

"""SQLite-backed progress storage for long experiment runs.

Stores per-run metadata, status transitions, and a heartbeat timestamp that
gets bumped once per completed episode. The DB is written to **only by the
parent process** of a run (joblib workers fork, and SQLite connections do not
survive fork). Reads from a separate watcher process are safe thanks to WAL
mode plus a 5-second busy timeout.

Classes:
    ProgressDB: SQLite progress writer / reader.
    RunRow: Frozen dataclass mirroring one row of the ``runs`` table.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

_DEFAULT_DB_REL_PATH = Path(".cache") / "POMDPPlanners" / "progress.db"
_PROGRESS_DB_ENV_VAR = "POMDP_PROGRESS_DB"

RunStatus = Literal["running", "finished", "failed"]


@dataclass(frozen=True)
class RunRow:
    """A snapshot of one row in the ``runs`` table.

    Attributes:
        run_id: Unique id for the run (typically ``uuid4().hex``).
        experiment_name: Human-readable label for the run (e.g. the simulator
            experiment name).
        host: Host the run started on, as returned by :func:`socket.gethostname`.
        pid: Process id of the parent simulator process.
        status: One of ``"running"``, ``"finished"``, or ``"failed"``.
        started_at: ISO-8601 UTC timestamp of run start.
        finished_at: ISO-8601 UTC timestamp of run finish, or ``None`` while
            running.
        last_heartbeat_at: ISO-8601 UTC timestamp of the most recent heartbeat.
        error_msg: Stringified exception if the run failed, else ``None``.
        metadata: JSON-decoded metadata dict supplied at run start.
        stall_notified_at: ISO-8601 UTC timestamp at which a watcher sent a
            stall notification for this run, or ``None`` if no stall
            notification has been sent.
    """

    run_id: str
    experiment_name: str
    host: str
    pid: int
    status: str
    started_at: str
    finished_at: str | None
    last_heartbeat_at: str
    error_msg: str | None
    metadata: dict[str, Any]
    stall_notified_at: str | None


class ProgressDB:
    """SQLite-backed progress writer / reader for experiment runs.

    Opens a fresh connection per call (no shared connection state), so each
    method is safe to call from any thread inside the parent process. WAL
    journaling is enabled on first connect so that a separate watcher process
    can read concurrently. The path can be supplied explicitly or resolved
    from the ``POMDP_PROGRESS_DB`` environment variable; if neither is set,
    the default is ``~/.cache/POMDPPlanners/progress.db``.

    Example:
        Basic usage::

            db = ProgressDB()
            db.start_run("abc123", "tiger_experiment", {"episodes": 100})
            db.heartbeat("abc123")
            db.finish_run("abc123", status="finished", error=None)
    """

    def __init__(self, path: Path | None = None):
        """Initialize the progress DB.

        Args:
            path: Optional explicit path to the SQLite file. If ``None``,
                the path is read from the ``POMDP_PROGRESS_DB`` env var, or
                defaults to ``~/.cache/POMDPPlanners/progress.db``. The
                parent directory is created if it does not already exist.
        """
        self.path = _resolve_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def start_run(
        self,
        run_id: str,
        experiment_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record the start of a new run.

        Args:
            run_id: Unique id for the run.
            experiment_name: Human-readable label for the run.
            metadata: Optional metadata dict, JSON-serialized into storage.
        """
        now = _now_iso()
        row = (
            run_id,
            experiment_name,
            socket.gethostname(),
            os.getpid(),
            "running",
            now,
            None,
            now,
            None,
            json.dumps(metadata or {}),
            None,
        )
        self._execute(
            "INSERT INTO runs (run_id, experiment_name, host, pid, status, "
            "started_at, finished_at, last_heartbeat_at, error_msg, "
            "metadata_json, stall_notified_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    def heartbeat(self, run_id: str) -> None:
        """Bump the heartbeat timestamp for a run.

        Args:
            run_id: The run to update.
        """
        self._execute(
            "UPDATE runs SET last_heartbeat_at = ? WHERE run_id = ?",
            (_now_iso(), run_id),
        )

    def finish_run(
        self,
        run_id: str,
        status: RunStatus,
        error: str | None = None,
    ) -> None:
        """Mark a run finished or failed.

        Args:
            run_id: The run to update.
            status: Either ``"finished"`` or ``"failed"``.
            error: Optional error message; stored verbatim.
        """
        self._execute(
            "UPDATE runs SET status = ?, finished_at = ?, error_msg = ? " "WHERE run_id = ?",
            (status, _now_iso(), error, run_id),
        )

    def list_stalled(self, threshold_seconds: int) -> list[RunRow]:
        """Return runs whose last heartbeat is older than ``threshold_seconds``.

        Only ``status='running'`` rows that have not yet been stall-notified
        are returned, so the watcher can use this list directly without
        additional filtering.

        Args:
            threshold_seconds: Heartbeat-age threshold in seconds. A run is
                considered stalled if ``now - last_heartbeat_at`` exceeds this.

        Returns:
            List of :class:`RunRow` entries, oldest heartbeat first.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=threshold_seconds)).isoformat()
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE status = 'running' "
                "AND last_heartbeat_at < ? AND stall_notified_at IS NULL "
                "ORDER BY last_heartbeat_at ASC",
                (cutoff,),
            ).fetchall()
        return [_row_to_runrow(row) for row in rows]

    def mark_stall_notified(self, run_id: str) -> None:
        """Record that a stall notification has been emitted for a run.

        Args:
            run_id: The run to mark.
        """
        self._execute(
            "UPDATE runs SET stall_notified_at = ? WHERE run_id = ?",
            (_now_iso(), run_id),
        )

    def get_run(self, run_id: str) -> RunRow | None:
        """Fetch a single run by id.

        Args:
            run_id: The run to look up.

        Returns:
            The matching :class:`RunRow`, or ``None`` if no such run exists.
        """
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _row_to_runrow(row) if row is not None else None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS runs ("
                "run_id TEXT PRIMARY KEY, "
                "experiment_name TEXT NOT NULL, "
                "host TEXT NOT NULL, "
                "pid INTEGER NOT NULL, "
                "status TEXT NOT NULL, "
                "started_at TEXT NOT NULL, "
                "finished_at TEXT, "
                "last_heartbeat_at TEXT NOT NULL, "
                "error_msg TEXT, "
                "metadata_json TEXT NOT NULL, "
                "stall_notified_at TEXT"
                ")"
            )
            conn.commit()

    def _execute(self, sql: str, params: tuple) -> None:
        with closing(self._connect()) as conn:
            conn.execute(sql, params)
            conn.commit()


def _resolve_path(path: Path | None) -> Path:
    if path is not None:
        return Path(path)
    env_path = os.environ.get(_PROGRESS_DB_ENV_VAR)
    if env_path:
        return Path(env_path)
    return Path.home() / _DEFAULT_DB_REL_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_runrow(row: sqlite3.Row) -> RunRow:
    return RunRow(
        run_id=row["run_id"],
        experiment_name=row["experiment_name"],
        host=row["host"],
        pid=row["pid"],
        status=row["status"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        last_heartbeat_at=row["last_heartbeat_at"],
        error_msg=row["error_msg"],
        metadata=json.loads(row["metadata_json"]),
        stall_notified_at=row["stall_notified_at"],
    )
