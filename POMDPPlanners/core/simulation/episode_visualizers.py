# SPDX-License-Identifier: MIT

"""How an environment shows one episode.

Every environment shows its episodes in exactly one way, and this module is
that one way's interface. An environment returns its visualizer from
:meth:`Environment.episode_visualizer`; the simulator calls
:meth:`EpisodeVisualizer.write` once per episode and knows nothing else.

There are two kinds, and the results site picks its player from the kind
alone:

* :class:`TraceVisualizer` — for environments implemented in this repo. It
  writes the episode as data (``trace_<i>.json``), and a three.js scene script
  that lives next to the visualizer replays it in the browser. The script is
  named after the payload kind: ``battleship.v1`` is drawn by
  ``battleship.scene.js`` in the same directory as the visualizer's module.
* :class:`VideoVisualizer` — for worlds that come from an external simulator
  (CARLA, Isaac Lab). The simulator already renders itself, so its camera
  footage is the visualization and is written as an ``.mp4``.

Visualizers are built inside simulation worker processes, so they hold the
environment and nothing live.
"""

import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

from POMDPPlanners.core.simulation.traces import ArtifactKind, EpisodeTrace, envelope_steps

if TYPE_CHECKING:
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.simulation.history import StepData

SCENE_SUFFIX = ".scene.js"


def scene_name(payload_kind: str) -> str:
    """Name of the scene that draws ``payload_kind``.

    The version suffix is dropped, so ``light_dark.v1`` is drawn by the scene
    named ``light_dark``, in the file ``light_dark.scene.js``.

    Args:
        payload_kind: A trace payload kind, such as ``"battleship.v1"``.

    Returns:
        The scene's name.
    """
    return payload_kind.split(".", 1)[0]


class EpisodeVisualizer(ABC):
    """Writes the one file that shows an episode.

    Attributes:
        kind: What the written file is. The results site chooses its player
            from this and nothing else.
        environment: The environment the episodes were run on.
    """

    kind: ClassVar[ArtifactKind]

    def __init__(self, environment: "Environment"):
        self.environment = environment

    @abstractmethod
    def write(
        self,
        history: "List[StepData]",
        output_dir: Path,
        episode_index: int,
        policy_name: Optional[str] = None,
    ) -> Optional[Path]:
        """Write one episode's visualization file into ``output_dir``.

        The visualizer owns the file name, so callers pass only the directory
        and the episode index. The index must end the file name, because the
        results site reads it from there.

        Args:
            history: The episode's step data, in order.
            output_dir: Directory to write into. Created if missing.
            episode_index: Zero-based index of the episode within its run.
            policy_name: Name of the policy that produced the episode.

        Returns:
            The path written, or ``None`` when there was nothing to write.
        """


class TraceVisualizer(EpisodeVisualizer):
    """Writes an episode as a trace that a three.js scene replays.

    Subclasses set :attr:`payload_kind` and implement :meth:`build_payload`.
    The envelope — actions, rewards, outcome — is the same for every
    environment and is filled in here.

    Attributes:
        payload_kind: The versioned name of the payload's shape, such as
            ``"battleship.v1"``. It also names the scene script.
    """

    kind = ArtifactKind.TRACE
    payload_kind: ClassVar[str]

    @abstractmethod
    def build_payload(self, history: "List[StepData]") -> Dict[str, Any]:
        """Build the environment-specific half of the trace.

        Args:
            history: The episode's step data, in order. Never empty.

        Returns:
            JSON-compatible data the scene script reads.
        """

    def reached_terminal_state(self, history: "List[StepData]") -> bool:
        """Whether the episode ended in a terminal state.

        Args:
            history: The episode's step data, in order. Never empty.

        Returns:
            ``True`` when the last recorded state is terminal.
        """
        return bool(self.environment.is_terminal(history[-1].state))

    def metadata(self) -> Dict[str, Any]:
        """Envelope-level extras written with every trace.

        Returns:
            The environment's class name; subclasses may add more.
        """
        return {"environment_class": type(self.environment).__name__}

    def build_trace(
        self,
        history: "List[StepData]",
        episode_index: int,
        policy_name: Optional[str] = None,
    ) -> EpisodeTrace:
        """Build the full trace of one episode.

        Args:
            history: The episode's step data, in order.
            episode_index: Zero-based index of the episode within its run.
            policy_name: Name of the policy that produced the episode.

        Returns:
            The episode's :class:`EpisodeTrace`.

        Raises:
            ValueError: If ``history`` is empty; there is no episode to write.
        """
        if not history:
            raise ValueError("Cannot export a trace for an empty history")
        return EpisodeTrace(
            environment=str(self.environment.name),
            payload_kind=self.payload_kind,
            episode_index=int(episode_index),
            discount_factor=float(self.environment.discount_factor),
            steps=envelope_steps(history),
            payload=self.build_payload(history),
            policy=policy_name,
            reach_terminal_state=self.reached_terminal_state(history),
            metadata=self.metadata(),
        )

    def write(
        self,
        history: "List[StepData]",
        output_dir: Path,
        episode_index: int,
        policy_name: Optional[str] = None,
    ) -> Path:
        """Write ``trace_<episode_index>.json`` into ``output_dir``."""
        trace = self.build_trace(history, episode_index, policy_name)
        return trace.write(Path(output_dir) / f"trace_{episode_index}.json")

    @classmethod
    def scene_script(cls) -> Path:
        """The scene script that draws this visualizer's traces.

        Returns:
            ``<scene name>.scene.js`` in the directory of the module that
            defines the visualizer. The file is not guaranteed to exist; a
            test checks that it does for every visualizer.
        """
        module_dir = Path(inspect.getfile(cls)).resolve().parent
        return module_dir / f"{scene_name(cls.payload_kind)}{SCENE_SUFFIX}"


class VideoVisualizer(EpisodeVisualizer):
    """Writes an external simulator's camera footage of an episode."""

    kind = ArtifactKind.VIDEO

    @abstractmethod
    def write_video(self, path: Path) -> bool:
        """Write the footage of the episode that just ran to ``path``.

        Args:
            path: Destination ``.mp4``.

        Returns:
            ``True`` when a video was written, ``False`` when the simulator
            recorded nothing.
        """

    def write(
        self,
        history: "List[StepData]",
        output_dir: Path,
        episode_index: int,
        policy_name: Optional[str] = None,
    ) -> Optional[Path]:
        """Write ``agent_path_<episode_index>.mp4`` into ``output_dir``."""
        del history, policy_name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"agent_path_{episode_index}.mp4"
        return path if self.write_video(path) else None
