# SPDX-License-Identifier: MIT

"""``pomdp-report`` — serve the results site for one or more run directories."""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from POMDPPlanners.reporting.server import serve


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser, with the ``serve`` subcommand registered.
    """
    parser = argparse.ArgumentParser(
        prog="pomdp-report",
        description="Browse POMDPPlanners simulation results from their MLflow stores.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Serve the results site for one or more run directories.",
        description=(
            "Searches each directory for MLflow stores (any 'mlruns' directory "
            "beneath it) and serves every experiment, run, environment, planner "
            "and episode it finds."
        ),
    )
    serve_parser.add_argument(
        "runs_dir",
        nargs="+",
        type=Path,
        help="Run directory, or any directory containing run directories.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="Interface to bind.")
    serve_parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv``.

    Returns:
        A process exit code.
    """
    args = build_parser().parse_args(argv)
    roots: List[Path] = list(args.runs_dir)
    missing = [str(r) for r in roots if not r.exists()]
    if missing:
        print("No such directory: " + ", ".join(missing))
        return 2
    serve(roots, host=args.host, port=args.port)
    return 0
