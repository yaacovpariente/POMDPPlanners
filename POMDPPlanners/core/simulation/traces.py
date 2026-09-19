# SPDX-License-Identifier: MIT

"""Machine-readable episode traces.

An environment already knows how to draw an episode: ``cache_visualization``
writes a GIF. That GIF is the end of the line — nothing downstream can read a
position, a reward or a belief back out of it. A *trace* is the same episode
written so a program can replay it: a browser viewer, a plotting script, a
regression check.

The file has two halves, and the split is the whole point:

* an **envelope** every environment fills in the same way — schema version,
  environment name, episode index, per-step action / reward / info, and the
  episode's outcome. Anything that indexes episodes reads only this half and
  therefore works for an environment it has never heard of.
* a **payload** the environment owns — Light-Dark writes rover positions,
  observations and belief clouds; a grid world would write a grid. The
  envelope names the payload's kind (``payload_kind``) so a reader can decide
  whether it knows how to draw it before it tries.

Writing a trace is opt-in. :meth:`Environment.build_episode_trace` returns
``None`` by default, and :meth:`Environment.cache_trace` then writes nothing,
so adding this module changes no existing environment's behaviour.

Everything here is plain JSON-compatible data: these objects are built inside
simulation worker processes, so they must pickle and must not hold live state.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import json

import numpy as np

# Bump the minor part when a field is added and old readers still work; bump
# the major part when an existing field changes meaning, so a reader can refuse
# a file it would silently misread.
TRACE_SCHEMA_VERSION = "1.0"


class ArtifactKind(str, Enum):
    """What an episode artifact *is*, independent of how it was produced.

    The reporting site picks a player from this and nothing else, so an
    environment that can only emit an MP4 (CARLA, Isaac Lab, nuPlan) is a
    first-class episode without the site knowing anything about it.
    """

    TRACE = "trace"
    VIDEO = "video"
    GIF = "gif"
    PLOT = "plot"


# One branch per type it has to handle, which is what makes it readable.
# pylint: disable-next=too-many-return-statements
def to_jsonable(value: Any) -> Any:
    """Convert numpy and other simulation values into JSON-compatible data.

    Traces are produced from live environment state, which is numpy almost
    everywhere. ``json`` refuses ``np.float64`` and ``np.ndarray``, and a
    silent ``str()`` fallback would write ``"[0. 5.]"`` — text a reader cannot
    parse back into numbers. So the conversion is explicit here, once.

    Args:
        value: Any value taken from an episode: numpy scalars and arrays,
            containers of them, or plain Python data.

    Returns:
        The same value expressed with ``dict``, ``list``, ``str``, ``float``,
        ``int``, ``bool`` and ``None`` only.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    return str(value)


@dataclass(frozen=True)
class TraceStep:
    """One step of an episode, in the environment-agnostic envelope.

    Only what every environment has: which step it was, what was done, what it
    paid, and whatever scalars the environment reported through ``step_info``.
    The state, the observation and the belief live in the payload, because
    their shape is the environment's business.

    Attributes:
        index: Zero-based step index within the episode.
        action: The action taken, as JSON data. ``None`` on the terminal
            bookkeeping step, which carries a state but no decision.
        reward: The realised reward, or ``None`` on the terminal step.
        info: The environment's per-step channels for this step, as a flat
            mapping of name to scalar. Empty when the environment reported
            nothing.
    """

    index: int
    action: Any = None
    reward: Optional[float] = None
    info: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this step to JSON-compatible data."""
        return {
            "index": int(self.index),
            "action": to_jsonable(self.action),
            "reward": None if self.reward is None else float(self.reward),
            "info": {str(k): to_jsonable(v) for k, v in (self.info or {}).items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceStep":
        """Rebuild a step from :meth:`to_dict` output.

        Args:
            data: A mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed :class:`TraceStep`.
        """
        reward = data.get("reward", None)
        return cls(
            index=int(data["index"]),
            action=data.get("action", None),
            reward=None if reward is None else float(reward),
            info=dict(data.get("info", {}) or {}),
        )


@dataclass(frozen=True)
class EpisodeTrace:
    """One episode, written so a program can replay it.

    Attributes:
        environment: The environment's ``name``, as the run directory and
            MLflow record use it.
        payload_kind: An identifier for the shape of ``payload``, versioned
            independently of the envelope (e.g. ``"light_dark.v1"``). A reader
            that does not recognise it must not try to draw the payload.
        episode_index: Zero-based index of the episode within its run.
        discount_factor: The discount the episode was run with, so a reader can
            recompute the discounted return without guessing.
        steps: The envelope's per-step records, in order.
        payload: Environment-specific episode data. Must be JSON-compatible.
        policy: Name of the policy that produced the episode, when known.
        reach_terminal_state: Whether the episode ended in a terminal state.
        metadata: Free-form envelope-level extras (seed, wall-clock, run id).
        schema_version: Version of the envelope itself.
    """

    environment: str
    payload_kind: str
    episode_index: int
    discount_factor: float
    steps: List[TraceStep] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    policy: Optional[str] = None
    reach_terminal_state: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRACE_SCHEMA_VERSION

    @property
    def num_steps(self) -> int:
        """Number of recorded steps, terminal bookkeeping step included."""
        return len(self.steps)

    @property
    def total_reward(self) -> float:
        """Undiscounted sum of the episode's realised rewards."""
        return float(sum(s.reward for s in self.steps if s.reward is not None))

    @property
    def discounted_return(self) -> float:
        """Discounted return, using the same convention as ``History``."""
        return float(
            sum(
                s.reward * self.discount_factor**index
                for index, s in enumerate(self.steps)
                if s.reward is not None
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this trace to JSON-compatible data.

        Returns:
            A dictionary that ``json.dump`` accepts and :meth:`from_dict`
            reads back.
        """
        return {
            "schema_version": self.schema_version,
            "environment": self.environment,
            "payload_kind": self.payload_kind,
            "episode_index": int(self.episode_index),
            "policy": self.policy,
            "discount_factor": float(self.discount_factor),
            "reach_terminal_state": bool(self.reach_terminal_state),
            "num_steps": self.num_steps,
            "total_reward": self.total_reward,
            "discounted_return": self.discounted_return,
            "metadata": to_jsonable(self.metadata),
            "steps": [step.to_dict() for step in self.steps],
            "payload": to_jsonable(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeTrace":
        """Rebuild a trace from :meth:`to_dict` output.

        ``num_steps``, ``total_reward`` and ``discounted_return`` are written
        for readers that only skim the header; they are recomputed from
        ``steps`` here rather than trusted, so a hand-edited file cannot make
        the object disagree with itself.

        Args:
            data: A mapping produced by :meth:`to_dict` or read from a
                ``trace.json``.

        Returns:
            The reconstructed :class:`EpisodeTrace`.

        Raises:
            ValueError: If the file's envelope major version is not one this
                code knows how to read.
        """
        version = str(data.get("schema_version", TRACE_SCHEMA_VERSION))
        if version.split(".", maxsplit=1)[0] != TRACE_SCHEMA_VERSION.split(".", maxsplit=1)[0]:
            raise ValueError(
                f"Unsupported trace schema version {version!r}; "
                f"this build reads {TRACE_SCHEMA_VERSION!r}"
            )
        return cls(
            environment=str(data["environment"]),
            payload_kind=str(data["payload_kind"]),
            episode_index=int(data["episode_index"]),
            discount_factor=float(data["discount_factor"]),
            steps=[TraceStep.from_dict(step) for step in data.get("steps", [])],
            payload=dict(data.get("payload", {}) or {}),
            policy=data.get("policy", None),
            reach_terminal_state=bool(data.get("reach_terminal_state", False)),
            metadata=dict(data.get("metadata", {}) or {}),
            schema_version=version,
        )

    def write(self, path: Union[str, Path]) -> Path:
        """Write this trace as JSON.

        Args:
            path: Destination file. Its parent directory is created if needed.

        Returns:
            The path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def read(cls, path: Union[str, Path]) -> "EpisodeTrace":
        """Read a trace written by :meth:`write`.

        Args:
            path: The ``trace.json`` to read.

        Returns:
            The reconstructed :class:`EpisodeTrace`.
        """
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def envelope_steps(
    history: Sequence[Any],
    include_terminal_step: bool = True,
) -> List[TraceStep]:
    """Build the envelope's step list from an episode ``History``'s step data.

    Every environment's trace shares this half, so it is written once here
    rather than copied into each environment's exporter.

    Args:
        history: The ``StepData`` records of one episode, in order.
        include_terminal_step: Keep the trailing bookkeeping step, which
            carries a state but no action or reward. Kept by default, because
            a viewer needs the final state to draw where the episode ended.

    Returns:
        One :class:`TraceStep` per retained record.
    """
    steps: List[TraceStep] = []
    for index, step in enumerate(history):
        action = getattr(step, "action", None)
        reward = getattr(step, "reward", None)
        if not include_terminal_step and action is None and reward is None:
            continue
        steps.append(
            TraceStep(
                index=index,
                action=to_jsonable(action),
                reward=None if reward is None else float(reward),
                info=dict(getattr(step, "info", None) or {}),
            )
        )
    return steps
