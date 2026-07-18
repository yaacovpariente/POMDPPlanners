# SPDX-License-Identifier: MIT

"""Headless CARLA server pool for parallel episode simulation.

A single CARLA server serves one client at a time, so parallel episode execution
(e.g. :class:`~POMDPPlanners.simulations.simulations_deployment.task_managers.JoblibTaskManager`
with ``n_jobs > 1``) needs one server per worker process. This module provides:

- :class:`CarlaServerPool` — a context manager that launches ``n_servers`` headless
  CARLA servers (``CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=<port>``),
  each on its own RPC port, and terminates them on exit.
- :func:`acquire_pool_lease` — the worker-side counterpart: a process claims exactly
  one server from the pool via an ``flock``-based lease and reuses it for the
  lifetime of the process.

Pool directory layout (written by :meth:`CarlaServerPool.start`):

- ``pool.json`` — the pool spec: host and per-server ``rpc_port`` /
  ``traffic_manager_port`` / lease-file name.
- ``server_<i>.lease`` — one lock file per server. A worker holds a server by
  holding an exclusive ``flock`` on its lease file; the kernel releases the lock
  automatically when the worker process dies, so a recycled joblib worker frees
  its server for the replacement worker.
- ``server_<i>.log`` — each server's combined stdout/stderr, for diagnosing
  startup failures.

Wiring into the episode loop is transparent: pass ``server_pool_dir=pool.pool_dir``
to :class:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp.CarlaPOMDP` and its
lazily-built session resolves its connection ports from the per-process lease
instead of the static ``host``/``port``/``traffic_manager_port``.

Limitation: the lease is per *process*, so all CARLA environments in one worker
process that share a pool directory share one server. This matches the simulator's
one-world-per-episode design.

Classes:
    CarlaServerLease: Connection endpoints of one leased pool server.
    CarlaServerHandle: One spawned headless CARLA server subprocess.
    CarlaServerPool: Context manager owning N headless CARLA servers.

Example:
    Launch four headless servers and run parallel episodes against them
    (illustrative — requires a CARLA installation at ``$CARLA_ROOT``)::

        from POMDPPlanners.environments.carla_pomdp import CarlaPOMDP, CarlaServerPool

        with CarlaServerPool(n_servers=4) as pool:
            env = CarlaPOMDP(discount_factor=0.95, server_pool_dir=pool.pool_dir)
            # Hand ``env`` to POMDPSimulator with JoblibConfig(n_jobs=4); each
            # joblib worker process leases its own server on first connection.

    Or manage a long-lived pool manually from the command line::

        python -m POMDPPlanners.environments.carla_pomdp.carla_server_pool --n-servers 4
"""

import argparse
import atexit
import fcntl
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Each CARLA server claims its RPC port plus the next two (streaming and secondary
# server), so consecutive servers must be spaced at least 3 ports apart.
RPC_PORT_STRIDE = 4
DEFAULT_RPC_PORT_BASE = 2000
DEFAULT_TM_PORT_BASE = 8000
DEFAULT_READY_TIMEOUT_SECONDS = 120.0
POOL_SPEC_FILENAME = "pool.json"

# Builds the server launch argv from (rpc_port, traffic_manager_port, gpu_index).
CommandFactory = Callable[[int, int, Optional[int]], List[str]]

# Per-process lease cache, keyed by resolved pool directory. The lock fd is kept
# open for the process lifetime; the kernel releases the flock on process death.
_PROCESS_LEASES: Dict[str, "CarlaServerLease"] = {}
_PROCESS_LEASE_FDS: Dict[str, int] = {}


@dataclass(frozen=True)
class CarlaServerLease:
    """Connection endpoints of one leased pool server.

    Attributes:
        host: Hostname the pool's servers listen on.
        rpc_port: CARLA RPC port of the leased server.
        traffic_manager_port: Client-side Traffic Manager port reserved for the
            lease holder (unique per server so parallel clients never collide).
    """

    host: str
    rpc_port: int
    traffic_manager_port: int


class CarlaServerHandle:
    """One spawned headless CARLA server subprocess.

    Owns the process for its lifetime: readiness polling on the RPC port and
    process-group termination. Instances are created by :class:`CarlaServerPool`.

    Attributes:
        process: The spawned server subprocess (its own session/process group).
        rpc_port: RPC port the server was asked to listen on.
        traffic_manager_port: Traffic Manager port reserved for this server's client.
        log_path: File receiving the server's combined stdout/stderr.
        gpu_index: GPU the server was pinned to, or ``None``.
    """

    def __init__(
        self,
        process: "subprocess.Popen[bytes]",
        rpc_port: int,
        traffic_manager_port: int,
        log_path: Path,
        gpu_index: Optional[int] = None,
    ) -> None:
        self.process = process
        self.rpc_port = rpc_port
        self.traffic_manager_port = traffic_manager_port
        self.log_path = log_path
        self.gpu_index = gpu_index

    @property
    def is_running(self) -> bool:
        """Whether the server process is still alive."""
        return self.process.poll() is None

    def wait_until_ready(self, timeout: float = DEFAULT_READY_TIMEOUT_SECONDS) -> None:
        """Block until the server accepts TCP connections on its RPC port.

        Args:
            timeout: Maximum seconds to wait.

        Raises:
            RuntimeError: If the server process exits before becoming ready.
            TimeoutError: If the port is not accepting connections within ``timeout``.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_running:
                raise RuntimeError(
                    f"CARLA server on port {self.rpc_port} exited with code "
                    f"{self.process.returncode} before becoming ready; see "
                    f"{self.log_path}:\n{self._log_tail()}"
                )
            if self._rpc_port_accepts_connection():
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"CARLA server on port {self.rpc_port} not ready after {timeout:.0f}s; "
            f"see {self.log_path}:\n{self._log_tail()}"
        )

    def terminate(self, grace_seconds: float = 10.0) -> None:
        """Terminate the server's process group (SIGTERM, then SIGKILL).

        Args:
            grace_seconds: Seconds to wait after SIGTERM before escalating.
        """
        if not self.is_running:
            return
        if not self._signal_process_group(signal.SIGTERM):
            return
        try:
            self.process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            self._signal_process_group(signal.SIGKILL)
            self.process.wait()

    def _rpc_port_accepts_connection(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", self.rpc_port), timeout=1.0):
                return True
        except OSError:
            return False

    def _signal_process_group(self, signum: int) -> bool:
        # The launch script spawns the UE4 binary as a child, so signal the whole
        # group (the server runs in its own session, see CarlaServerPool).
        try:
            os.killpg(self.process.pid, signum)
            return True
        except ProcessLookupError:
            return False

    def _log_tail(self, max_lines: int = 20) -> str:
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return "<log unavailable>"
        return "\n".join(lines[-max_lines:])


class CarlaServerPool:
    """Context manager owning N headless CARLA servers plus their lease directory.

    On :meth:`start` (or ``with`` entry) it spawns ``n_servers`` headless CARLA
    servers — RPC ports ``rpc_port_base + RPC_PORT_STRIDE * i``, Traffic Manager
    ports ``tm_port_base + i`` — writes the pool spec and lease files into
    ``pool_dir``, and waits for every server to accept connections. On
    :meth:`shutdown` (or ``with`` exit, or interpreter exit) it terminates them.

    Worker processes claim a server with :func:`acquire_pool_lease`, or
    transparently by constructing
    :class:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp.CarlaPOMDP` with
    ``server_pool_dir=pool.pool_dir``. Run at most ``n_servers`` workers (e.g.
    ``JoblibConfig(n_jobs=n_servers)`` — the default ``n_jobs=-1`` uses all cores
    and will exhaust the pool).

    Attributes:
        n_servers: Number of servers the pool launches.
        handles: Live :class:`CarlaServerHandle` objects (empty until started).

    Example:
        Illustrative — requires a CARLA installation at ``$CARLA_ROOT``::

            with CarlaServerPool(n_servers=2, gpu_indices=[0, 1]) as pool:
                env = CarlaPOMDP(discount_factor=0.95, server_pool_dir=pool.pool_dir)
    """

    def __init__(
        self,
        n_servers: int,
        pool_dir: Optional[Union[str, Path]] = None,
        carla_root: Optional[Union[str, Path]] = None,
        rpc_port_base: int = DEFAULT_RPC_PORT_BASE,
        tm_port_base: int = DEFAULT_TM_PORT_BASE,
        gpu_indices: Optional[Sequence[int]] = None,
        extra_args: Optional[Sequence[str]] = None,
        ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
        command_factory: Optional[CommandFactory] = None,
    ) -> None:
        """Configure a pool (no servers are launched until :meth:`start`).

        Args:
            n_servers: Number of headless servers to launch.
            pool_dir: Directory for the spec/lease/log files. Defaults to a fresh
                temporary directory.
            carla_root: CARLA installation directory containing ``CarlaUE4.sh``.
                Defaults to the ``CARLA_ROOT`` environment variable. Only used by
                the default launch command; ignored when ``command_factory`` is given.
            rpc_port_base: RPC port of server 0; server ``i`` gets
                ``rpc_port_base + RPC_PORT_STRIDE * i``.
            tm_port_base: Traffic Manager port of server 0; server ``i`` gets
                ``tm_port_base + i``.
            gpu_indices: GPUs to pin servers to (``-graphicsadapter=<gpu>``),
                cycled across servers. ``None`` leaves GPU selection to CARLA.
            extra_args: Extra CLI arguments appended to the default launch command.
            ready_timeout: Seconds to wait for each server to accept connections.
            command_factory: Override for the launch command; called with
                ``(rpc_port, traffic_manager_port, gpu_index)`` and must return the
                argv to spawn. Intended for tests and non-standard installs.

        Raises:
            ValueError: If ``n_servers`` is not positive.
        """
        if n_servers <= 0:
            raise ValueError(f"n_servers must be positive, got {n_servers}")
        self.n_servers = n_servers
        self.handles: List[CarlaServerHandle] = []
        self._pool_dir = Path(pool_dir) if pool_dir is not None else None
        self._carla_root = Path(carla_root) if carla_root is not None else None
        self._rpc_port_base = rpc_port_base
        self._tm_port_base = tm_port_base
        self._gpu_indices = list(gpu_indices) if gpu_indices is not None else None
        self._extra_args = list(extra_args) if extra_args is not None else []
        self._ready_timeout = ready_timeout
        self._command_factory = command_factory
        self._owner_pid: Optional[int] = None

    @property
    def pool_dir(self) -> Path:
        """The pool directory holding the spec, lease, and log files.

        Raises:
            RuntimeError: If accessed before :meth:`start` and no explicit
                ``pool_dir`` was configured.
        """
        if self._pool_dir is None:
            raise RuntimeError("Pool directory is allocated on start(); call start() first.")
        return self._pool_dir

    def start(self) -> "CarlaServerPool":
        """Launch all servers, write the pool spec, and wait until every one is ready.

        Launches every server first so their (slow) startups overlap, then blocks
        on readiness. On any failure the already-launched servers are terminated
        before the error propagates.

        Returns:
            This pool, for chaining.

        Raises:
            RuntimeError: If a server process exits before becoming ready.
            TimeoutError: If a server is not ready within ``ready_timeout``.
        """
        if self._pool_dir is None:
            self._pool_dir = Path(tempfile.mkdtemp(prefix="carla_pool_"))
        self._pool_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.handles = [self._launch_server(index) for index in range(self.n_servers)]
            self._write_pool_spec()
            for handle in self.handles:
                handle.wait_until_ready(timeout=self._ready_timeout)
        except BaseException:
            self.shutdown()
            raise
        self._owner_pid = os.getpid()
        atexit.register(self._atexit_shutdown)
        return self

    def shutdown(self) -> None:
        """Terminate every server in the pool. Idempotent."""
        for handle in self.handles:
            handle.terminate()
        self.handles = []

    def __enter__(self) -> "CarlaServerPool":
        return self.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        self.shutdown()

    def _rpc_port(self, index: int) -> int:
        return self._rpc_port_base + RPC_PORT_STRIDE * index

    def _tm_port(self, index: int) -> int:
        return self._tm_port_base + index

    def _gpu_index(self, index: int) -> Optional[int]:
        if not self._gpu_indices:
            return None
        return self._gpu_indices[index % len(self._gpu_indices)]

    def _build_command(self, index: int) -> List[str]:
        rpc_port, tm_port, gpu = self._rpc_port(index), self._tm_port(index), self._gpu_index(index)
        if self._command_factory is not None:
            return self._command_factory(rpc_port, tm_port, gpu)
        return self._default_command(rpc_port, gpu)

    def _default_command(self, rpc_port: int, gpu_index: Optional[int]) -> List[str]:
        carla_root = self._carla_root
        if carla_root is None:
            carla_root_env = os.environ.get("CARLA_ROOT")
            if carla_root_env is None:
                raise RuntimeError(
                    "CARLA installation not found: pass carla_root= or set the "
                    "CARLA_ROOT environment variable to the directory containing "
                    "CarlaUE4.sh."
                )
            carla_root = Path(carla_root_env)
        command = [
            str(carla_root / "CarlaUE4.sh"),
            "-RenderOffScreen",
            "-nosound",
            f"-carla-rpc-port={rpc_port}",
        ]
        if gpu_index is not None:
            command.append(f"-graphicsadapter={gpu_index}")
        command.extend(self._extra_args)
        return command

    def _launch_server(self, index: int) -> CarlaServerHandle:
        command = self._build_command(index)
        log_path = self.pool_dir / f"server_{index}.log"
        with open(log_path, "wb") as log_file:
            # The handle owns the process; start_new_session puts the launch
            # script and the UE4 binary it spawns into one killable group.
            process = subprocess.Popen(  # pylint: disable=consider-using-with
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return CarlaServerHandle(
            process=process,
            rpc_port=self._rpc_port(index),
            traffic_manager_port=self._tm_port(index),
            log_path=log_path,
            gpu_index=self._gpu_index(index),
        )

    def _write_pool_spec(self) -> None:
        servers = []
        for index, handle in enumerate(self.handles):
            lease_file = f"server_{index}.lease"
            (self.pool_dir / lease_file).touch()
            servers.append(
                {
                    "index": index,
                    "rpc_port": handle.rpc_port,
                    "traffic_manager_port": handle.traffic_manager_port,
                    "lease_file": lease_file,
                }
            )
        spec = {"host": "127.0.0.1", "servers": servers}
        (self.pool_dir / POOL_SPEC_FILENAME).write_text(json.dumps(spec, indent=2))

    def _atexit_shutdown(self) -> None:
        # Guard so forked/spawned children that inherited this object never kill
        # the launcher's servers.
        if self._owner_pid == os.getpid():
            self.shutdown()


def acquire_pool_lease(pool_dir: Union[str, Path]) -> CarlaServerLease:
    """Claim one server from a :class:`CarlaServerPool` for the current process.

    The first call locks a free server's lease file (exclusive non-blocking
    ``flock``) and caches the result; subsequent calls from the same process with
    the same pool directory return the cached lease. The lock is held for the
    process lifetime and released by the kernel when the process exits, so a
    recycled worker's server returns to the pool automatically.

    Args:
        pool_dir: Directory written by :meth:`CarlaServerPool.start`.

    Returns:
        The leased server's connection endpoints.

    Raises:
        FileNotFoundError: If ``pool_dir`` does not contain a pool spec.
        RuntimeError: If every server in the pool is already leased by another
            process (run at most ``n_servers`` workers).
    """
    key = str(Path(pool_dir).resolve())
    if key in _PROCESS_LEASES:
        return _PROCESS_LEASES[key]
    spec = _read_pool_spec(Path(key))
    lease_and_fd = _lease_first_free_server(Path(key), spec)
    if lease_and_fd is None:
        n_servers = len(spec["servers"])
        raise RuntimeError(
            f"All {n_servers} CARLA server(s) in the pool at {key} are already "
            f"leased by other processes. Run at most {n_servers} parallel workers "
            f"(e.g. JoblibConfig(n_jobs={n_servers}) — the default n_jobs=-1 uses "
            f"all cores), or start a larger CarlaServerPool."
        )
    lease, fd = lease_and_fd
    _PROCESS_LEASES[key] = lease
    _PROCESS_LEASE_FDS[key] = fd
    return lease


def _lease_first_free_server(
    pool_dir: Path, spec: Dict[str, Any]
) -> Optional[Tuple[CarlaServerLease, int]]:
    for server in spec["servers"]:
        fd = _try_acquire_lease_file(pool_dir / server["lease_file"])
        if fd is not None:
            lease = CarlaServerLease(
                host=spec["host"],
                rpc_port=server["rpc_port"],
                traffic_manager_port=server["traffic_manager_port"],
            )
            return lease, fd
    return None


def _try_acquire_lease_file(lease_path: Path) -> Optional[int]:
    fd = os.open(lease_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _read_pool_spec(pool_dir: Path) -> Dict[str, Any]:
    spec_path = pool_dir / POOL_SPEC_FILENAME
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"No CARLA pool spec at {spec_path}; is the CarlaServerPool started "
            f"and pointing at this directory?"
        )
    spec = json.loads(spec_path.read_text())
    if not isinstance(spec, dict):
        raise ValueError(f"Malformed pool spec at {spec_path}: expected a JSON object.")
    return spec


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a pool of headless CARLA servers and keep it up until "
        "SIGINT/SIGTERM."
    )
    parser.add_argument("--n-servers", type=int, required=True, help="Number of servers.")
    parser.add_argument("--pool-dir", type=Path, default=None, help="Pool directory.")
    parser.add_argument(
        "--carla-root", type=Path, default=None, help="CARLA install dir (default: $CARLA_ROOT)."
    )
    parser.add_argument("--rpc-port-base", type=int, default=DEFAULT_RPC_PORT_BASE)
    parser.add_argument("--tm-port-base", type=int, default=DEFAULT_TM_PORT_BASE)
    parser.add_argument(
        "--gpus", type=str, default=None, help="Comma-separated GPU indices to cycle over."
    )
    parser.add_argument("--ready-timeout", type=float, default=DEFAULT_READY_TIMEOUT_SECONDS)
    parser.add_argument(
        "--extra-args", nargs=argparse.REMAINDER, default=None, help="Extra server CLI args."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: start a pool, print its directory, run until interrupted.

    Args:
        argv: CLI arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on clean shutdown).
    """
    args = _build_arg_parser().parse_args(argv)
    gpu_indices = [int(gpu) for gpu in args.gpus.split(",")] if args.gpus else None
    pool = CarlaServerPool(
        n_servers=args.n_servers,
        pool_dir=args.pool_dir,
        carla_root=args.carla_root,
        rpc_port_base=args.rpc_port_base,
        tm_port_base=args.tm_port_base,
        gpu_indices=gpu_indices,
        extra_args=args.extra_args,
        ready_timeout=args.ready_timeout,
    )
    with pool:
        print(f"CARLA server pool ready: {pool.pool_dir}")
        print("Press Ctrl-C (or send SIGTERM) to shut the pool down.")
        signal.sigwait({signal.SIGINT, signal.SIGTERM})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
