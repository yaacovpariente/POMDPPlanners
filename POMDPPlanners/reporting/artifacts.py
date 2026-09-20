# SPDX-License-Identifier: MIT

"""What an episode produced, classified by kind rather than by environment.

This is the site's load-bearing abstraction. Light-Dark writes a trace and a
GIF; CARLA, Isaac Lab and nuPlan write an MP4 and nothing else, because there
is no compact state to replay — the render *is* the episode. If the site
dispatched on the environment it would need a branch per environment and a
release to add one. It dispatches on the artifact's kind instead, so a new
environment that writes an MP4 is a first-class episode on the day it lands,
with no change here.

The kind is decided by file extension and, for JSON, by the trace envelope's
own fields — never by which environment produced it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json
import re

from POMDPPlanners.core.simulation.traces import ArtifactKind

# Extension to kind. A file whose extension is not here is not an episode
# artifact and is ignored rather than guessed at.
_EXTENSION_KINDS: Dict[str, ArtifactKind] = {
    ".json": ArtifactKind.TRACE,
    ".mp4": ArtifactKind.VIDEO,
    ".webm": ArtifactKind.VIDEO,
    ".mov": ArtifactKind.VIDEO,
    ".gif": ArtifactKind.GIF,
    ".png": ArtifactKind.PLOT,
    ".jpg": ArtifactKind.PLOT,
    ".jpeg": ArtifactKind.PLOT,
    ".svg": ArtifactKind.PLOT,
}

# The player each kind gets. The episode page reads this and nothing else.
PLAYER_FOR_KIND: Dict[ArtifactKind, str] = {
    ArtifactKind.TRACE: "trace-viewer",
    ArtifactKind.VIDEO: "video",
    ArtifactKind.GIF: "image",
    ArtifactKind.PLOT: "image",
}

# Which kind a page prefers when an episode produced several. A trace beats a
# video because it is interactive; a video beats a GIF because it seeks.
_KIND_PRIORITY: Sequence[ArtifactKind] = (
    ArtifactKind.TRACE,
    ArtifactKind.VIDEO,
    ArtifactKind.GIF,
    ArtifactKind.PLOT,
)

MEDIA_TYPES: Dict[str, str] = {
    ".json": "application/json",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".gif": "image/gif",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
}

# Environments name their per-episode files differently — agent_path_3.gif,
# battleship_board_3.gif, trace_3.json — so the index is taken from the
# trailing number rather than from a fixed prefix.
_EPISODE_INDEX = re.compile(r"(\d+)(?=\.[A-Za-z0-9]+$)")


def episode_index_from_name(name: str) -> Optional[int]:
    """Read the episode index out of an artifact's file name.

    Args:
        name: The file's base name, e.g. ``agent_path_3.gif``.

    Returns:
        The trailing integer, or ``None`` when the name carries none — which
        is how a per-policy plot is told apart from a per-episode artifact.
    """
    match = _EPISODE_INDEX.search(name)
    return int(match.group(1)) if match else None


def media_type_for(path: Path) -> str:
    """Return the media type to serve a file with.

    Args:
        path: The file being served.

    Returns:
        A media type string; ``application/octet-stream`` when unknown, so an
        unrecognised file is downloaded rather than mis-rendered.
    """
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


@dataclass(frozen=True)
class EpisodeArtifact:
    """One file an episode produced, and what the site can do with it.

    Attributes:
        kind: The classification the player is chosen from.
        path: Absolute path to the file on disk.
        relative_path: Path relative to its run's artifact root, which is also
            how the site addresses it in a URL.
        episode_index: Which episode it belongs to, or ``None`` for a
            per-policy artifact such as a returns histogram.
        payload_kind: For a trace, the payload identifier from its envelope;
            ``None`` otherwise. A viewer refuses a payload kind it does not
            know rather than drawing it wrong.
        summary: For a trace, the envelope's episode-level numbers — steps,
            discounted return, whether it reached a terminal state. Read while
            the file is already open for its payload kind, so that listing
            pages can show an episode's outcome without opening megabytes of
            particles a second time.
    """

    kind: ArtifactKind
    path: Path
    relative_path: str
    episode_index: Optional[int] = None
    payload_kind: Optional[str] = None
    summary: Optional["TraceSummary"] = None

    @property
    def player(self) -> str:
        """The player this artifact is rendered with."""
        return PLAYER_FOR_KIND[self.kind]


@dataclass(frozen=True)
class TraceSummary:
    """The episode-level numbers a trace envelope carries.

    Attributes:
        num_steps: Steps the episode ran.
        discounted_return: The episode's discounted return.
        reach_terminal_state: Whether it ended in a terminal state rather than
            by exhausting its step budget.
    """

    num_steps: Optional[int] = None
    discounted_return: Optional[float] = None
    reach_terminal_state: Optional[bool] = None


def _read_trace_envelope(path: Path) -> Optional[Tuple[str, TraceSummary]]:
    """Read a JSON file's payload kind and episode summary.

    The run directory also holds MLflow's own JSON tables, so a ``.json`` file
    is only a trace when it carries the envelope's identifying fields.

    Args:
        path: The JSON file to inspect.

    Returns:
        The ``payload_kind`` and its summary, or ``None`` when the file is not
        a trace.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if "schema_version" not in data or "payload_kind" not in data:
        return None

    def number(key: str):
        value = data.get(key)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    terminal = data.get("reach_terminal_state")
    summary = TraceSummary(
        num_steps=int(number("num_steps")) if number("num_steps") is not None else None,
        discounted_return=number("discounted_return"),
        reach_terminal_state=terminal if isinstance(terminal, bool) else None,
    )
    return str(data["payload_kind"]), summary


def classify(path: Path, relative_path: str) -> Optional[EpisodeArtifact]:
    """Classify one file, or return ``None`` when it is not an episode artifact.

    Args:
        path: Absolute path to the file.
        relative_path: Its path relative to the run's artifact root.

    Returns:
        The classified :class:`EpisodeArtifact`, or ``None``.
    """
    kind = _EXTENSION_KINDS.get(path.suffix.lower())
    if kind is None:
        return None

    payload_kind: Optional[str] = None
    summary: Optional[TraceSummary] = None
    if kind is ArtifactKind.TRACE:
        envelope = _read_trace_envelope(path)
        if envelope is None:
            return None
        payload_kind, summary = envelope

    return EpisodeArtifact(
        kind=kind,
        path=path,
        relative_path=relative_path,
        episode_index=episode_index_from_name(path.name),
        payload_kind=payload_kind,
        summary=summary,
    )


def preferred(artifacts: Sequence[EpisodeArtifact]) -> Optional[EpisodeArtifact]:
    """Pick the artifact an episode page should play.

    Args:
        artifacts: The episode's artifacts, in any order.

    Returns:
        The highest-priority artifact, or ``None`` when there are none.
    """
    for kind in _KIND_PRIORITY:
        for artifact in artifacts:
            if artifact.kind is kind:
                return artifact
    return None


def group_by_episode(
    artifacts: Sequence[EpisodeArtifact],
) -> Dict[int, List[EpisodeArtifact]]:
    """Group per-episode artifacts by their episode index.

    Artifacts with no index (per-policy plots) are left out: they belong to
    the policy page, not to any one episode.

    Args:
        artifacts: Artifacts discovered under one policy.

    Returns:
        A mapping from episode index to that episode's artifacts, in index
        order.
    """
    grouped: Dict[int, List[EpisodeArtifact]] = {}
    for artifact in artifacts:
        if artifact.episode_index is None:
            continue
        grouped.setdefault(artifact.episode_index, []).append(artifact)
    return {index: grouped[index] for index in sorted(grouped)}
