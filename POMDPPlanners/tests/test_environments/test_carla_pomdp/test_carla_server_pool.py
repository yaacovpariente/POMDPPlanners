# SPDX-License-Identifier: MIT

"""Unit tests for the headless CARLA server pool.

These tests never launch CARLA: the pool's ``command_factory`` seam substitutes tiny
``python -c`` subprocesses (a TCP listener standing in for a ready server, a sleeper
for a hung one, an immediate exit for a crashed one), and the lease tests operate on
hand-written pool directories with no processes at all.
"""

# pylint: disable=protected-access  # Tests exercise module internals directly

import fcntl
import json
import multiprocessing
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import pytest

from POMDPPlanners.environments.carla_pomdp import carla_server_pool
from POMDPPlanners.environments.carla_pomdp.carla_server_pool import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    DEFAULT_RPC_PORT_BASE,
    DEFAULT_TM_PORT_BASE,
    RPC_PORT_STRIDE,
    CarlaServerHandle,
    CarlaServerPool,
    acquire_pool_lease,
)

# Fake-server one-liners spawned in place of CarlaUE4.sh.
_LISTENER_CODE = (
    "import socket, sys, time\n"
    "server = socket.socket()\n"
    "server.bind(('127.0.0.1', int(sys.argv[1])))\n"
    "server.listen()\n"
    "time.sleep(60)\n"
)
_SLEEPER_CODE = "import time; time.sleep(60)"


def _listener_command(rpc_port: int, tm_port: int, gpu_index: Optional[int]) -> List[str]:
    """Command factory: a subprocess that listens on the RPC port (a 'ready' server)."""
    del tm_port, gpu_index
    return [sys.executable, "-c", _LISTENER_CODE, str(rpc_port)]


def _sleeper_command(rpc_port: int, tm_port: int, gpu_index: Optional[int]) -> List[str]:
    """Command factory: a subprocess that stays alive but never listens."""
    del rpc_port, tm_port, gpu_index
    return [sys.executable, "-c", _SLEEPER_CODE]


def _free_port_base() -> int:
    """Pick a port that is free right now (the small span above it is a good bet too)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _spawn(command: List[str], log_path: Path) -> "subprocess.Popen[bytes]":
    with open(log_path, "wb") as log_file:
        return subprocess.Popen(  # pylint: disable=consider-using-with
            command, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )


def _write_fake_pool(pool_dir: Path, n_servers: int, rpc_base: int = 3000) -> None:
    """Hand-write a pool spec + lease files (no server processes)."""
    servers = []
    for index in range(n_servers):
        lease_file = f"server_{index}.lease"
        (pool_dir / lease_file).touch()
        servers.append(
            {
                "index": index,
                "rpc_port": rpc_base + RPC_PORT_STRIDE * index,
                "traffic_manager_port": 9000 + index,
                "lease_file": lease_file,
            }
        )
    spec = {"host": "127.0.0.1", "servers": servers}
    (pool_dir / carla_server_pool.POOL_SPEC_FILENAME).write_text(json.dumps(spec))


def _hold_lease_in_child(pool_dir: str, port_queue: Any, release_event: Any) -> None:
    """Child-process target: acquire a lease, report its port, hold until released."""
    lease = acquire_pool_lease(pool_dir)
    port_queue.put(lease.rpc_port)
    release_event.wait(timeout=30)


class TestCarlaServerHandle:
    """Readiness polling and process-group termination of one server subprocess."""

    def test_server_handle_wait_until_ready_succeeds_for_listening_process(
        self, tmp_path: Path
    ) -> None:
        """Test that readiness returns once the server accepts TCP connections.

        Purpose: Validates the TCP readiness probe against a genuinely listening process

        Given: A subprocess that binds and listens on the handle's RPC port
        When: wait_until_ready is called with a generous timeout
        Then: It returns without raising, well before the timeout

        Test type: unit
        """
        port = _free_port_base()
        log_path = tmp_path / "server.log"
        process = _spawn([sys.executable, "-c", _LISTENER_CODE, str(port)], log_path)
        handle = CarlaServerHandle(
            process=process, rpc_port=port, traffic_manager_port=9000, log_path=log_path
        )
        try:
            handle.wait_until_ready(timeout=20.0)
        finally:
            handle.terminate()

    def test_server_handle_wait_until_ready_fails_fast_when_process_dies(
        self, tmp_path: Path
    ) -> None:
        """Test that a crashed server is reported immediately, not after the timeout.

        Purpose: Validates fail-fast diagnosis of a server that exits during startup

        Given: A subprocess that writes to its log and exits immediately with code 3
        When: wait_until_ready is called with a long timeout
        Then: RuntimeError naming the exit code and log path (with its content) is
            raised in a fraction of the timeout

        Test type: unit
        """
        port = _free_port_base()
        log_path = tmp_path / "server.log"
        process = _spawn(
            [sys.executable, "-c", "print('vulkan missing'); import sys; sys.exit(3)"],
            log_path,
        )
        handle = CarlaServerHandle(
            process=process, rpc_port=port, traffic_manager_port=9000, log_path=log_path
        )
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="exited with code 3") as exc_info:
            handle.wait_until_ready(timeout=60.0)
        assert time.monotonic() - start < 30.0
        assert str(log_path) in str(exc_info.value)
        assert "vulkan missing" in str(exc_info.value)

    def test_server_handle_wait_until_ready_times_out_for_silent_process(
        self, tmp_path: Path
    ) -> None:
        """Test that a live but never-listening server raises TimeoutError.

        Purpose: Validates the readiness timeout for a hung server process

        Given: A subprocess that stays alive but never opens the RPC port
        When: wait_until_ready is called with a ~2 second timeout
        Then: TimeoutError naming the port and log path is raised

        Test type: unit
        """
        port = _free_port_base()
        log_path = tmp_path / "server.log"
        process = _spawn([sys.executable, "-c", _SLEEPER_CODE], log_path)
        handle = CarlaServerHandle(
            process=process, rpc_port=port, traffic_manager_port=9000, log_path=log_path
        )
        try:
            with pytest.raises(TimeoutError, match=str(port)):
                handle.wait_until_ready(timeout=2.0)
        finally:
            handle.terminate()

    def test_server_handle_terminate_kills_process_group(self, tmp_path: Path) -> None:
        """Test that terminate stops the spawned process.

        Purpose: Validates that terminate signals the server's process group and reaps it

        Given: A running sleeper subprocess in its own session
        When: terminate is called
        Then: The process is no longer running and terminate is idempotent

        Test type: unit
        """
        log_path = tmp_path / "server.log"
        process = _spawn([sys.executable, "-c", _SLEEPER_CODE], log_path)
        handle = CarlaServerHandle(
            process=process, rpc_port=3000, traffic_manager_port=9000, log_path=log_path
        )
        assert handle.is_running
        handle.terminate(grace_seconds=5.0)
        assert not handle.is_running
        handle.terminate(grace_seconds=5.0)  # idempotent


class TestCarlaServerPool:
    """Launch, spec/lease bookkeeping, and shutdown of the pool."""

    def test_pool_assigns_stride_spaced_rpc_and_sequential_tm_ports(self, tmp_path: Path) -> None:
        """Test the pool's port assignment scheme.

        Purpose: Validates that RPC ports are stride-spaced (CARLA claims port..port+2)
            and Traffic Manager ports are sequential

        Given: A three-server pool with fake listener servers and known port bases
        When: The pool is started and its spec is read back
        Then: Server i has rpc_port base + 4*i and traffic_manager_port tm_base + i

        Test type: unit
        """
        rpc_base = _free_port_base()
        pool = CarlaServerPool(
            n_servers=3,
            pool_dir=tmp_path,
            rpc_port_base=rpc_base,
            tm_port_base=8100,
            ready_timeout=20.0,
            command_factory=_listener_command,
        )
        with pool:
            spec = carla_server_pool._read_pool_spec(tmp_path)
        assert [server["rpc_port"] for server in spec["servers"]] == [
            rpc_base + RPC_PORT_STRIDE * i for i in range(3)
        ]
        assert [server["traffic_manager_port"] for server in spec["servers"]] == [
            8100 + i for i in range(3)
        ]

    def test_pool_writes_spec_and_lease_files_and_cleans_up_servers(self, tmp_path: Path) -> None:
        """Test the pool's directory contents and shutdown behavior.

        Purpose: Validates that the pool writes pool.json plus one lease file per
            server, runs all servers, and terminates them on context exit

        Given: A two-server pool with fake listener servers
        When: The pool context is entered and then exited
        Then: Inside the context the spec and lease files exist and both servers run;
            after exit both server processes are terminated

        Test type: unit
        """
        rpc_base = _free_port_base()
        pool = CarlaServerPool(
            n_servers=2,
            pool_dir=tmp_path,
            rpc_port_base=rpc_base,
            ready_timeout=20.0,
            command_factory=_listener_command,
        )
        with pool:
            assert (tmp_path / carla_server_pool.POOL_SPEC_FILENAME).is_file()
            assert (tmp_path / "server_0.lease").is_file()
            assert (tmp_path / "server_1.lease").is_file()
            handles = list(pool.handles)
            assert all(handle.is_running for handle in handles)
        assert all(not handle.is_running for handle in handles)

    def test_pool_start_cleans_up_when_a_server_dies(self, tmp_path: Path) -> None:
        """Test that a failed start terminates the already-launched servers.

        Purpose: Validates that start() does not leak sibling server processes when
            one server never becomes ready

        Given: A two-server pool whose servers never listen, with a short timeout
        When: start() is invoked
        Then: TimeoutError propagates and no launched server process is left running

        Test type: unit
        """
        pool = CarlaServerPool(
            n_servers=2,
            pool_dir=tmp_path,
            rpc_port_base=_free_port_base(),
            ready_timeout=2.0,
            command_factory=_sleeper_command,
        )
        with pytest.raises(TimeoutError):
            pool.start()
        assert pool.handles == []

    def test_cli_parser_defaults(self) -> None:
        """Test the CLI argument parser's defaults.

        Purpose: Validates that the module CLI defaults match the module constants

        Given: The pool module's argument parser
        When: Only the required --n-servers flag is parsed
        Then: Port bases and ready timeout default to the module constants and the
            optional paths/GPUs default to None

        Test type: unit
        """
        args = carla_server_pool._build_arg_parser().parse_args(["--n-servers", "4"])
        assert args.n_servers == 4
        assert args.pool_dir is None
        assert args.carla_root is None
        assert args.rpc_port_base == DEFAULT_RPC_PORT_BASE
        assert args.tm_port_base == DEFAULT_TM_PORT_BASE
        assert args.gpus is None
        assert args.ready_timeout == DEFAULT_READY_TIMEOUT_SECONDS


class TestAcquirePoolLease:
    """The flock-based worker-side lease protocol."""

    def test_acquire_pool_lease_is_cached_within_process(self, tmp_path: Path) -> None:
        """Test that one process reuses its lease for a pool directory.

        Purpose: Validates the per-process lease cache so all episodes in a reused
            worker share one server

        Given: A hand-written two-server pool directory
        When: acquire_pool_lease is called twice with the same directory
        Then: Both calls return the same lease (same server), leaving the second
            server free for another process

        Test type: unit
        """
        _write_fake_pool(tmp_path, n_servers=2)
        first = acquire_pool_lease(tmp_path)
        second = acquire_pool_lease(tmp_path)
        assert first is second

    def test_acquire_pool_lease_exclusive_across_processes(self, tmp_path: Path) -> None:
        """Test that two processes lease two different servers.

        Purpose: Validates cross-process exclusivity of the flock lease

        Given: A hand-written two-server pool directory and a child process holding
            a lease on it
        When: The parent process acquires a lease on the same pool
        Then: The parent's server differs from the child's

        Test type: integration
        """
        _write_fake_pool(tmp_path, n_servers=2)
        context = multiprocessing.get_context("spawn")
        port_queue = context.Queue()
        release_event = context.Event()
        child = context.Process(
            target=_hold_lease_in_child, args=(str(tmp_path), port_queue, release_event)
        )
        child.start()
        try:
            child_port = port_queue.get(timeout=30)
            parent_lease = acquire_pool_lease(tmp_path)
            assert parent_lease.rpc_port != child_port
        finally:
            release_event.set()
            child.join(timeout=30)

    def test_acquire_pool_lease_raises_informative_error_when_exhausted(
        self, tmp_path: Path
    ) -> None:
        """Test the error raised when every server is already leased.

        Purpose: Validates that pool exhaustion produces an actionable error naming
            the n_jobs <= n_servers requirement

        Given: A hand-written one-server pool whose only lease file is already
            flock-ed through an independent file descriptor
        When: acquire_pool_lease is called
        Then: RuntimeError naming the pool directory and 'n_jobs' is raised

        Test type: unit
        """
        _write_fake_pool(tmp_path, n_servers=1)
        blocker_fd = os.open(tmp_path / "server_0.lease", os.O_RDWR)
        fcntl.flock(blocker_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(RuntimeError, match="n_jobs") as exc_info:
                acquire_pool_lease(tmp_path)
            assert str(tmp_path) in str(exc_info.value)
        finally:
            os.close(blocker_fd)

    def test_acquire_pool_lease_raises_for_missing_spec(self, tmp_path: Path) -> None:
        """Test the error for a directory that is not a pool.

        Purpose: Validates the guard against pointing server_pool_dir at a
            directory with no pool spec

        Given: An empty directory
        When: acquire_pool_lease is called on it
        Then: FileNotFoundError naming the expected spec path is raised

        Test type: unit
        """
        with pytest.raises(FileNotFoundError, match=carla_server_pool.POOL_SPEC_FILENAME):
            acquire_pool_lease(tmp_path)
