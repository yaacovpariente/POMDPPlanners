# SPDX-License-Identifier: MIT

"""Tests for the episode visualizer interface.

Every environment shows its episodes through one visualizer, and the results
site picks its player from the visualizer's kind. These tests pin the parts
of that contract the base classes own: the envelope a trace visualizer fills
in, the file names the site reads the episode index from, and where a scene
script is expected to live.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.episode_visualizers import (
    TraceVisualizer,
    VideoVisualizer,
    scene_name,
)
from POMDPPlanners.core.simulation.traces import ArtifactKind, EpisodeTrace
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP


class _CountingVisualizer(TraceVisualizer):
    """Writes how many steps the episode had, and nothing else."""

    payload_kind = "counting.v1"

    def build_payload(self, history: List[StepData]) -> Dict[str, Any]:
        return {"steps": len(history)}


class _Footage(VideoVisualizer):
    """Pretends the simulator recorded footage when ``recorded`` is set."""

    def __init__(self, environment: Environment, recorded: bool):
        super().__init__(environment)
        self.recorded = recorded

    def write_video(self, path: Path) -> bool:
        if self.recorded:
            path.write_bytes(b"mp4-stand-in")
        return self.recorded


@pytest.fixture(name="env")
def env_fixture() -> TigerPOMDP:
    return TigerPOMDP(discount_factor=0.9)


def _step(action: Any, reward: Any) -> StepData:
    return StepData(
        state="tiger_left",
        action=action,
        next_state="tiger_left" if action else None,
        observation="hear_left" if action else None,
        reward=reward,
        belief=None,  # type: ignore[arg-type]  # the envelope never reads it
    )


def _history() -> List[StepData]:
    return [_step("listen", -1.0), _step("open_right", 10.0), _step(None, None)]


def test_scene_name_drops_the_version():
    """``light_dark.v1`` is drawn by the scene named ``light_dark``."""
    assert scene_name("light_dark.v1") == "light_dark"
    assert scene_name("battleship") == "battleship"


def test_trace_visualizer_fills_the_shared_envelope(env: TigerPOMDP):
    """The base class writes the half of the trace every environment shares.

    Purpose: A subclass supplies only its payload. The envelope — name,
    discount, steps, outcome, policy — must come from the environment and the
    episode, identically for every environment.

    Given: A three-step episode on Tiger.
    When: A trace is built for episode 5.
    Then: The envelope carries the environment's values, every step, the
        subclass's payload, and the environment's class name.
    """
    trace = _CountingVisualizer(env).build_trace(_history(), episode_index=5, policy_name="POMCP")

    assert trace.environment == env.name
    assert trace.payload_kind == "counting.v1"
    assert trace.episode_index == 5
    assert trace.discount_factor == 0.9
    assert trace.policy == "POMCP"
    assert [step.reward for step in trace.steps] == [-1.0, 10.0, None]
    assert trace.payload == {"steps": 3}
    assert trace.reach_terminal_state is False  # Tiger reports no terminal states
    assert trace.metadata == {"environment_class": "TigerPOMDP"}


def test_trace_visualizer_refuses_an_empty_episode(env: TigerPOMDP):
    """There is no trace for an episode that did not happen."""
    with pytest.raises(ValueError, match="empty history"):
        _CountingVisualizer(env).build_trace([], episode_index=0)


def test_trace_visualizer_writes_trace_named_by_episode(env: TigerPOMDP, tmp_path: Path):
    """The file name ends in the episode index, which is how the site reads it.

    Given: An output directory that does not exist yet.
    When: Episode 7 is written.
    Then: ``trace_7.json`` is created and reads back as the same trace.
    """
    out = tmp_path / "nested" / "visualizations"
    written = _CountingVisualizer(env).write(_history(), out, episode_index=7)

    assert written == out / "trace_7.json"
    assert EpisodeTrace.read(written).payload == {"steps": 3}
    assert _CountingVisualizer.kind is ArtifactKind.TRACE


def test_scene_script_lives_beside_the_visualizer_module():
    """The scene that draws ``counting.v1`` is ``counting.scene.js`` in this directory."""
    assert _CountingVisualizer.scene_script() == Path(__file__).resolve().parent / (
        "counting.scene.js"
    )


def test_video_visualizer_writes_agent_path_mp4(env: TigerPOMDP, tmp_path: Path):
    """Simulator footage is written as ``agent_path_<i>.mp4``.

    Given: A simulator that recorded footage.
    When: Episode 2 is written into a directory that does not exist yet.
    Then: The video is at ``agent_path_2.mp4`` and its kind is VIDEO.
    """
    written = _Footage(env, recorded=True).write(_history(), tmp_path / "viz", episode_index=2)

    assert written == tmp_path / "viz" / "agent_path_2.mp4"
    assert written.read_bytes() == b"mp4-stand-in"
    assert VideoVisualizer.kind is ArtifactKind.VIDEO


def test_video_visualizer_reports_nothing_written(env: TigerPOMDP, tmp_path: Path):
    """A simulator that recorded nothing yields no path rather than an empty file."""
    assert _Footage(env, recorded=False).write(_history(), tmp_path, episode_index=0) is None
    assert not list(tmp_path.iterdir())


def test_environment_has_no_visualizer_by_default():
    """An environment that does not override ``episode_visualizer`` shows nothing."""
    assert Environment.episode_visualizer(object()) is None  # type: ignore[arg-type]
