# SPDX-License-Identifier: MIT

"""The local results server.

A server rather than a static export, for two reasons that are not style
choices. A browser refuses to let a ``file://`` page fetch a sibling JSON
file, so a trace viewer opened from disk can never load its trace. And a run
directory can hold gigabytes of video, which a static export would have to
copy; served over HTTP with range requests, it streams from where it already
is.

Built on :mod:`http.server` so the site needs no web framework: the package
already depends on enough, and a results viewer that cannot start because a
dependency drifted is worse than a plain one.
"""

from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import unquote, urlparse

import os

from POMDPPlanners.reporting import pages
from POMDPPlanners.reporting.artifacts import media_type_for
from POMDPPlanners.reporting.store import RunIndex

STATIC_ROOT = Path(__file__).resolve().parent / "static"


class Router:
    """Turns a URL path into a response, with no HTTP details in sight.

    Keeping routing separate from :class:`BaseHTTPRequestHandler` is what lets
    the tests drive every route as a function call instead of standing a
    socket up.
    """

    def __init__(self, index: RunIndex):
        """Bind the router to a run index.

        Args:
            index: The index of experiments and runs to serve.
        """
        self.index = index

    def resolve(self, path: str) -> Tuple[int, str, bytes]:
        """Resolve one request path to a complete response body.

        Convenient for callers that want the bytes — the tests, mostly. The
        server itself uses :meth:`resolve_source`, so that a gigabyte of
        episode video is not read into memory to answer one request.

        Args:
            path: The URL path, percent-encoded as the browser sent it.

        Returns:
            ``(status, media_type, body)``.
        """
        status, media_type, source = self.resolve_source(path)
        if isinstance(source, Path):
            return status, media_type, source.read_bytes()
        return status, media_type, source

    def resolve_source(self, path: str) -> Tuple[int, str, Union[bytes, Path]]:
        """Resolve one request path to a body or to the file that holds it.

        Returning a :class:`Path` rather than its contents is what lets the
        server stream a file and honour range requests. A CARLA or Isaac Lab
        episode is a video of arbitrary size, and reading it into memory to
        serve it would also mean a viewer could not seek within it.

        Args:
            path: The URL path, percent-encoded as the browser sent it.

        Returns:
            ``(status, media_type, body_or_path)``.
        """
        parts = [unquote(p) for p in urlparse(path).path.strip("/").split("/") if p]

        if not parts:
            return self._ok(pages.index_page(self.index.experiments, self.index.roots))

        head, rest = parts[0], parts[1:]
        handlers: Dict[str, Callable[[List[str]], Tuple[int, str, Union[bytes, Path]]]] = {
            "static": self._static,
            "experiment": self._experiment,
            "run": self._run,
            "artifact": self._artifact,
        }
        handler = handlers.get(head)
        if handler is None:
            return self._not_found(f"No route for /{head}")
        return handler(rest)

    # -- responses ---------------------------------------------------------

    @staticmethod
    def _ok(body: str) -> Tuple[int, str, bytes]:
        return HTTPStatus.OK, "text/html; charset=utf-8", body.encode("utf-8")

    @staticmethod
    def _not_found(message: str) -> Tuple[int, str, bytes]:
        return (
            HTTPStatus.NOT_FOUND,
            "text/html; charset=utf-8",
            pages.not_found(message).encode("utf-8"),
        )

    # -- routes ------------------------------------------------------------

    def _static(self, rest: Sequence[str]) -> Tuple[int, str, Union[bytes, Path]]:
        if not rest:
            return self._not_found("No static file named")
        target = (STATIC_ROOT / Path(*rest)).resolve()
        # A path that escapes the static root is a traversal attempt, not a
        # typo; it is refused rather than normalised.
        if not target.is_file() or STATIC_ROOT not in target.parents:
            return self._not_found("/".join(rest))
        return HTTPStatus.OK, media_type_for(target), target

    def _experiment(self, rest: Sequence[str]) -> Tuple[int, str, bytes]:
        if len(rest) != 2:
            return self._not_found("Malformed experiment URL")
        experiment = self._lookup_experiment(rest[0], rest[1])
        if experiment is None:
            return self._not_found(f"No experiment {rest[1]}")
        return self._ok(pages.experiment_page(experiment))

    def _lookup_experiment(self, store: str, experiment_id: str):
        try:
            store_index = int(store)
        except ValueError:
            return None
        return self.index.experiment(store_index, experiment_id)

    def _lookup_run(self, rest: Sequence[str]):
        if len(rest) < 3:
            return None
        try:
            store_index = int(rest[0])
        except ValueError:
            return None
        return self.index.run(store_index, rest[1], rest[2])

    # A flat chain of guard clauses, one per level of the URL. Folding it into
    # fewer exits would only hide which level of the hierarchy was missing.
    # pylint: disable-next=too-many-return-statements
    def _run(self, rest: Sequence[str]) -> Tuple[int, str, bytes]:
        run = self._lookup_run(rest)
        if run is None:
            return self._not_found("No such run")
        tail = list(rest[3:])

        if not tail:
            return self._ok(pages.run_page(run))

        if tail[0] != "env" or len(tail) < 2:
            return self._not_found("Malformed run URL")
        env = run.environment(tail[1])
        if env is None:
            return self._not_found(f"No environment {tail[1]} in this run")
        tail = tail[2:]

        if not tail:
            return self._ok(pages.environment_page(run, env))

        if tail == ["chart"]:
            return self._ok(pages.chart_builder_page(run, env))

        if tail[0] != "policy" or len(tail) < 2:
            return self._not_found("Malformed environment URL")
        policy = env.policy(tail[1])
        if policy is None:
            return self._not_found(f"No planner {tail[1]} on {env.name}")
        tail = tail[2:]

        if not tail:
            return self._ok(pages.policy_page(run, env, policy))

        if tail[0] != "episode" or len(tail) != 2:
            return self._not_found("Malformed planner URL")
        try:
            episode_index = int(tail[1])
        except ValueError:
            return self._not_found(f"Episode {tail[1]} is not a number")
        artifacts = policy.episodes.get(episode_index)
        if artifacts is None:
            return self._not_found(f"No episode {episode_index} for {policy.name}")
        return self._ok(pages.episode_page(run, env, policy, episode_index, artifacts))

    def _artifact(self, rest: Sequence[str]) -> Tuple[int, str, Union[bytes, Path]]:
        run = self._lookup_run(rest)
        if run is None or len(rest) < 4:
            return self._not_found("No such artifact")
        target = (run.artifact_root / Path(*rest[3:])).resolve()
        root = run.artifact_root.resolve()
        if not target.is_file() or root not in target.parents:
            return self._not_found("/".join(rest[3:]))
        return HTTPStatus.OK, media_type_for(target), target


def parse_range(header: Optional[str], size: int) -> Optional[Tuple[int, int]]:
    """Read a single byte range out of a ``Range`` header.

    Only one range is supported, which is all a media element ever asks for.
    Anything else — multiple ranges, a unit other than bytes, a malformed
    header — returns ``None``, and the caller answers with the whole file.
    That is a legal response to any range request, so an unusual client gets a
    working page rather than an error.

    Args:
        header: The request's ``Range`` header, or ``None``.
        size: The file's size in bytes.

    Returns:
        An inclusive ``(start, end)`` pair, or ``None``.
    """
    if not header or not header.startswith("bytes=") or "," in header:
        return None
    spec = header[len("bytes=") :].strip()
    start_text, _, end_text = spec.partition("-")
    try:
        if not start_text:
            # A suffix range: the last N bytes.
            length = int(end_text)
            if length <= 0:
                return None
            return max(0, size - length), size - 1
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    except ValueError:
        return None
    end = min(end, size - 1)
    if start > end or start >= size:
        return None
    return start, end


class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP glue around :class:`Router`."""

    server_version = "POMDPPlannersReport/1.0"
    protocol_version = "HTTP/1.1"

    # Streamed in chunks so a large episode video never sits in memory whole.
    CHUNK_SIZE = 256 * 1024

    def __init__(self, *args, router: Router, **kwargs):
        self.router = router
        super().__init__(*args, **kwargs)

    # pylint: disable-next=invalid-name
    def do_GET(self) -> None:
        """Serve one GET request."""
        status, media_type, source = self.router.resolve_source(self.path)
        if isinstance(source, Path):
            self._send_file(status, media_type, source)
        else:
            self._send_bytes(status, media_type, source)

    def _common_headers(self, media_type: str) -> None:
        self.send_header("Content-Type", media_type)
        # Results are immutable once written, but the server is restarted far
        # more often than a browser cache expires, so nothing is cached.
        self.send_header("Cache-Control", "no-store")

    def _send_bytes(self, status: int, media_type: str, body: bytes) -> None:
        self.send_response(status)
        self._common_headers(media_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, status: int, media_type: str, path: Path) -> None:
        """Send a file, honouring a byte range so video can be seeked.

        Without this a `<video>` element can only play an episode from the
        start: browsers seek by asking for a byte range, and a server that
        ignores the ask either replays from zero or refuses to scrub.
        """
        size = path.stat().st_size
        requested = parse_range(self.headers.get("Range"), size)

        with path.open("rb") as handle:
            if requested is None:
                self.send_response(status)
                self._common_headers(media_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(size))
                self.end_headers()
                remaining = size
            else:
                start, end = requested
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self._common_headers(media_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                handle.seek(start)
                remaining = end - start + 1

            while remaining > 0:
                chunk = handle.read(min(self.CHUNK_SIZE, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, format: str, *args) -> None:  # pylint: disable=redefined-builtin
        """Quieten the default per-request logging to one short line."""
        if os.environ.get("POMDP_REPORT_VERBOSE"):
            super().log_message(format, *args)


def build_server(
    roots: Sequence[Path], host: str = "127.0.0.1", port: int = 8765
) -> Tuple[ThreadingHTTPServer, RunIndex]:
    """Build the server and the index it serves.

    Args:
        roots: Directories to search for MLflow stores.
        host: Interface to bind. Defaults to loopback: these are local
            results, and binding every interface would publish them.
        port: TCP port. Pass ``0`` to let the OS choose one, which is what
            the tests do.

    Returns:
        The server, not yet started, and the index it will serve.
    """
    index = RunIndex(roots)
    handler = partial(_Handler, router=Router(index))
    return ThreadingHTTPServer((host, port), handler), index


def serve(roots: Sequence[Path], host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the results site until interrupted.

    Args:
        roots: Directories to search for MLflow stores.
        host: Interface to bind.
        port: TCP port.
    """
    server, index = build_server(roots, host=host, port=port)
    actual_host, actual_port = server.server_address[:2]
    print(f"Indexed {len(index.experiments)} experiment(s) in {len(index.stores)} store(s).")
    print(f"Serving on http://{actual_host}:{actual_port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
