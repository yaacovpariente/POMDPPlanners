# SPDX-License-Identifier: MIT

"""Tests for the Maze episode trace exporter, discrete and continuous.

Two things are checked here that nothing else checks. The first is that the
payload survives a round trip through JSON, because that file is the whole
interface between Python and the browser viewer: a field that does not survive
is a field the viewer silently never sees.

The second is that adding the exporter left the GIF alone. The Maze's GIF bytes
are pinned by a golden hash in
``tests/test_environments/test_environment_visualizations_golden_files.py``, and
this migration must not move them. That test guards the renderer against its own
changes; this one guards it against the exporter, by rendering the same episode
before and after a trace is built from it and comparing the bytes.
"""

import hashlib
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.maze_pomdp import ContinuousMazePOMDP, DiscreteMazePOMDP
from POMDPPlanners.environments.maze_pomdp.maze_pomdp import (
    CUE_EMITTING,
    CUE_UNSEEN,
    GOAL_LEFT,
    GOAL_RIGHT,
    STATE_GOAL,
    create_maze_state,
)
from POMDPPlanners.environments.maze_pomdp.maze_visualizer import MazeVisualizer
from POMDPPlanners.environments.maze_pomdp.maze_visualization.trace_exporter import (
    MAZE_PAYLOAD_KIND,
    build_maze_trace,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    continuous_maze_pinned_kwargs,
    discrete_maze_pinned_kwargs,
)


@pytest.fixture(name="discrete_env")
def discrete_env_fixture() -> DiscreteMazePOMDP:
    """A discrete Maze with the suite's pinned configuration."""
    return DiscreteMazePOMDP(discount_factor=0.95, **discrete_maze_pinned_kwargs())


@pytest.fixture(name="continuous_env")
def continuous_env_fixture() -> ContinuousMazePOMDP:
    """A continuous Maze on the same pinned map."""
    return ContinuousMazePOMDP(discount_factor=0.95, **continuous_maze_pinned_kwargs())


def _split_belief(env: Any, position: Any, left_mass: float) -> WeightedParticleBelief:
    """A belief that puts ``left_mass`` on the left goal and the rest on the right.

    The particles all stand where the agent stands, which is what a real Maze
    belief looks like: movement is deterministic and fully observed, so the only
    thing the cloud disagrees about is the goal side.
    """
    del env
    particles = [
        create_maze_state(position, GOAL_LEFT, CUE_UNSEEN),
        create_maze_state(position, GOAL_RIGHT, CUE_UNSEEN),
    ]
    return WeightedParticleBelief(
        particles=particles,
        log_weights=np.log(np.array([left_mass, 1.0 - left_mass])),
    )


def _episode(env: Any, actions: List[Any]) -> List[StepData]:
    """Walk ``actions`` from the environment's own start state, recording each step."""
    state = create_maze_state(env.start_cell, GOAL_LEFT, CUE_UNSEEN)
    history: List[StepData] = []
    for index, action in enumerate(actions):
        next_state = env.sample_next_state(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation="left_cue" if index == 0 else "empty",
                reward=env.reward(state, action, next_state),
                belief=_split_belief(env, (state[0], state[1]), 0.9 if index else 0.5),
                info=env.step_info(state, action, next_state),
            )
        )
        state = next_state
    history.append(
        StepData(
            state=state,
            action=None,
            next_state=None,
            observation=None,
            reward=None,
            belief=_split_belief(env, (state[0], state[1]), 0.9),
            info=env.step_info(state, None, None),
        )
    )
    return history


def test_maze_trace_round_trips_through_json(discrete_env, tmp_path: Path):
    """A written trace reads back as the same trace, payload included.

    Purpose: The ``trace.json`` is the interface to the browser viewer. A field
    that does not survive the round trip is one the scene can never draw.

    Given: A discrete Maze episode.
    When: Its trace is written to disk and read back.
    Then: The envelope and the whole payload match the original.
    """
    trace = build_maze_trace(discrete_env, _episode(discrete_env, ["up", "up"]), 2, "PFT_DPW")
    restored = EpisodeTrace.read(trace.write(tmp_path / "trace_2.json"))

    assert restored.payload_kind == MAZE_PAYLOAD_KIND
    assert restored.environment == discrete_env.name
    assert restored.episode_index == 2
    assert restored.policy == "PFT_DPW"
    assert restored.discount_factor == pytest.approx(discrete_env.discount_factor)
    assert restored.payload == trace.to_dict()["payload"]
    assert [step.to_dict() for step in restored.steps] == [s.to_dict() for s in trace.steps]


def test_maze_trace_world_is_the_map_the_episode_ran_on(discrete_env):
    """The payload's world block is this instance's geometry, not the defaults.

    Purpose: The map is generated from a seed, so a viewer that drew the class
    defaults would draw a different maze from the one the agent walked — with
    the corridors in the wrong places and the trail running through walls.

    Given: A discrete Maze on the pinned seed.
    When: Its trace is built.
    Then: Size, seed, walkable set, cue and goals all match the environment.
    """
    world = build_maze_trace(discrete_env, _episode(discrete_env, ["up"]), 0).payload["world"]

    assert (world["width"], world["height"]) == (discrete_env.maze_width, discrete_env.maze_height)
    assert world["maze_seed"] == discrete_env.maze_seed
    assert {tuple(cell) for cell in world["walkable"]} == set(discrete_env.walkable_cells)
    assert tuple(world["start_cell"]) == discrete_env.start_cell
    assert tuple(world["cue_cell"]) == discrete_env.cue_cell
    assert tuple(world["left_goal_cell"]) == discrete_env.left_goal_cell
    assert tuple(world["right_goal_cell"]) == discrete_env.right_goal_cell
    assert world["cue_accuracy"] == pytest.approx(discrete_env.cue_accuracy)
    assert world["draws_cell_guides"] is True
    assert world["max_step_size"] is None


def test_continuous_maze_trace_records_its_movement_model(continuous_env):
    """The continuous variant says so in the payload.

    Purpose: The two variants share a map and a payload kind, so the only thing
    telling a scene not to rule cell guides over a real-valued trajectory is
    this pair of fields.
    """
    history = _episode(continuous_env, [np.array([0.0, 0.6])])
    world = build_maze_trace(continuous_env, history, 0).payload["world"]

    assert world["draws_cell_guides"] is False
    assert world["max_step_size"] == pytest.approx(continuous_env.max_step_size)
    assert {tuple(cell) for cell in world["walkable"]} == set(continuous_env.walkable_cells)


def test_maze_trace_delegates_belief_serialization_to_core(discrete_env):
    """The exporter writes no belief format of its own.

    Purpose: Belief is a core abstraction with a closed family of
    implementations. Serializing it per environment would mean a copy of the
    same dispatch in every environment and a new chance to miss a class.

    Given: An episode whose steps carry weighted particle beliefs.
    When: The trace is built.
    Then: Each step's belief payload equals ``belief_to_payload`` of that
        step's belief, field for field.
    """
    history = _episode(discrete_env, ["up", "up"])
    trace = build_maze_trace(discrete_env, history, 0)

    for step, written in zip(history, trace.payload["beliefs"]):
        assert written == belief_to_payload(step.belief)


def test_maze_belief_payload_keeps_the_goal_side_readable(discrete_env):
    """Each particle still carries the slot the bars are split on.

    Purpose: The belief here is a distribution over one bit — which corner pays
    — and the scene reads it by summing weights per goal side. A payload that
    wrote positions only would look complete and say nothing, and the failure
    would show up as two bars stuck at one half.

    Given: A belief holding 90% of its mass on the left goal.
    When: The trace is built.
    Then: The goal slot the world block names splits the written weights 0.9 to
        0.1.
    """
    history = _episode(discrete_env, ["up", "up"])
    payload = build_maze_trace(discrete_env, history, 0).payload
    slot = payload["world"]["state_goal_index"]
    assert slot == STATE_GOAL

    belief = payload["beliefs"][1]
    left = sum(
        weight
        for particle, weight in zip(belief["particles"], belief["weights"])
        if particle[slot] == payload["world"]["goal_left"]
    )
    assert left == pytest.approx(0.9)


def test_maze_trace_records_the_cue_phase_per_step(discrete_env):
    """The cue's single-use life is in the trace, not inferred by the viewer.

    Purpose: The cue fires once, on the step that crosses its cell, and a scene
    that guessed at which step that was would light the wrong one on any episode
    that revisits the cue cell.
    """
    payload = build_maze_trace(discrete_env, _episode(discrete_env, ["up", "up"]), 0).payload

    assert payload["cue_phases"][0] == CUE_UNSEEN
    assert CUE_EMITTING in payload["cue_phases"]
    assert payload["true_goal_side"] == GOAL_LEFT


def test_maze_trace_refuses_an_empty_history(discrete_env):
    """There is no episode to write, so nothing is invented."""
    with pytest.raises(ValueError):
        build_maze_trace(discrete_env, [], 0)


def test_building_a_trace_does_not_change_the_gif(discrete_env, tmp_path: Path):
    """The GIF bytes are the same before and after a trace is built.

    Purpose: This migration adds a trace beside the GIF, and the GIF's bytes are
    pinned by a golden hash. The exporter must therefore be a pure read of the
    recorded episode: if it consumed a random draw, rebuilt geometry or wrote
    anything onto the environment, the next render would differ and the golden
    file would move.

    Given: One deterministic episode.
    When: The GIF is rendered, a trace is built from the same history, and the
        GIF is rendered again.
    Then: The two files hash identically.
    """
    history = _episode(discrete_env, ["up", "up", "up"])

    before = tmp_path / "before.gif"
    MazeVisualizer(discrete_env).create_visualization(history, before)
    first = hashlib.sha256(before.read_bytes()).hexdigest()

    build_maze_trace(discrete_env, history, 0, "PFT_DPW")

    after = tmp_path / "after.gif"
    MazeVisualizer(discrete_env).create_visualization(history, after)
    assert hashlib.sha256(after.read_bytes()).hexdigest() == first
