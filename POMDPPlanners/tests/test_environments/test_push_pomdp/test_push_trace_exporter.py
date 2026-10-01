# SPDX-License-Identifier: MIT

"""Tests for the Push episode trace.

The payload has to survive the round trip through JSON, because that file is
the only interface between Python and the browser viewer: a field that does
not survive is a field the viewer silently never sees.
"""

from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.push_pomdp import (
    ContinuousPushPOMDP,
    ContinuousPushPOMDPDiscreteActions,
    PushPOMDP,
)
from POMDPPlanners.environments.push_pomdp.push_visualization import (
    PUSH_PAYLOAD_KIND,
    PushVisualizer,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    continuous_push_pinned_kwargs,
    push_pinned_kwargs,
)

DANGEROUS_AREAS = [(4.0, 5.0), (2.6, 6.4)]


def _belief(states: List[List[float]]) -> WeightedParticleBelief:
    particles = [np.asarray(s, dtype=float) for s in states]
    return WeightedParticleBelief(
        particles=particles,
        log_weights=np.log(np.ones(len(particles)) / len(particles)),
    )


def _episode(states: List[List[float]], actions: List[Any]) -> List[StepData]:
    """An episode over the given states, with a terminal bookkeeping step."""
    history = []
    for index, state in enumerate(states):
        is_last = index == len(states) - 1
        history.append(
            StepData(
                state=np.asarray(state, dtype=float),
                action=None if is_last else actions[index],
                next_state=None if is_last else np.asarray(states[index + 1], dtype=float),
                observation=None if is_last else np.asarray(state, dtype=float) + 0.05,
                reward=None if is_last else -1.5,
                belief=_belief([state, list(np.asarray(state, dtype=float) + 0.1)]),
            )
        )
    return history


@pytest.fixture(name="discrete_env")
def discrete_env_fixture() -> PushPOMDP:
    """A discrete Push environment with obstacles and dangerous areas."""
    return PushPOMDP(
        discount_factor=0.95,
        **push_pinned_kwargs(
            grid_size=10,
            obstacles=[(3.0, 3.0), (6.0, 6.0)],
            obstacle_radius=0.5,
            dangerous_areas=list(DANGEROUS_AREAS),
        ),
    )


@pytest.fixture(name="continuous_env")
def continuous_env_fixture() -> ContinuousPushPOMDP:
    """A continuous Push environment with square obstacles."""
    return ContinuousPushPOMDP(
        discount_factor=0.99,
        **continuous_push_pinned_kwargs(
            grid_size=10,
            obstacles=[(3.0, 3.0, 0.5), (6.0, 6.0, 0.5)],
            dangerous_areas=list(DANGEROUS_AREAS),
        ),
    )


def _discrete_episode() -> List[StepData]:
    return _episode(
        [
            [1.0, 1.0, 2.25, 1.0, 9.0, 9.0],
            [2.0, 1.0, 2.25, 1.0, 9.0, 9.0],
            [3.0, 1.0, 2.95, 1.0, 9.0, 9.0],
        ],
        ["right", "right"],
    )


def _continuous_episode() -> List[StepData]:
    return _episode(
        [
            [0.8, 1.2, 2.2, 1.4, 9.0, 9.0],
            [2.2, 1.2, 2.2, 1.4, 9.0, 9.0],
            [3.0, 1.5, 3.1, 1.4, 9.0, 9.0],
        ],
        [np.array([1.4, 0.0]), np.array([0.8, 0.3])],
    )


def test_discrete_payload_round_trips_through_json(discrete_env: PushPOMDP, tmp_path: Path):
    """A written trace reads back with every field the viewer needs.

    Purpose: The trace file is the only interface between Python and the
        browser viewer, so a field that does not survive the round trip is a
        field the viewer never sees and nobody is told about.

    Given: A three-step discrete Push episode.
    When: Its trace is written to disk and read back.
    Then: The kind, the world, the states, the observations and the beliefs
        come back unchanged.

    Test type: integration
    """
    trace = discrete_env.episode_visualizer().build_trace(_discrete_episode(), 3, "POMCPOW")
    restored = EpisodeTrace.read(trace.write(tmp_path / "trace_3.json"))

    assert restored.payload_kind == PUSH_PAYLOAD_KIND
    assert restored.episode_index == 3
    assert restored.policy == "POMCPOW"
    assert restored.discount_factor == pytest.approx(0.95)
    assert restored.payload == trace.to_dict()["payload"]
    assert restored.payload["states"][0] == [1.0, 1.0, 2.25, 1.0]
    assert restored.payload["next_states"][-1] is None
    assert len(restored.payload["beliefs"]) == 3
    assert restored.payload["beliefs"][0]["kind"] == "particles"


def test_discrete_world_describes_circular_obstacles(discrete_env: PushPOMDP):
    """The discrete world is written with the geometry that blocks movement.

    Purpose: The viewer draws an obstacle from this block alone, and the two
        variants disagree about what an obstacle is.

    Given: A discrete environment with two circular obstacles.
    When: Its trace is built.
    Then: The obstacles are centres with a radius, and the action labels carry
        the vectors the transition uses.

    Test type: unit
    """
    world = discrete_env.episode_visualizer().build_trace(_discrete_episode(), 0).payload["world"]

    assert world["variant"] == "discrete"
    assert world["obstacle_shape"] == "circle"
    assert world["obstacles"] == [[3.0, 3.0], [6.0, 6.0]]
    assert world["obstacle_radius"] == pytest.approx(0.5)
    assert world["robot_radius"] is None
    assert world["max_push"] is None
    assert world["dangerous_areas"] == [[4.0, 5.0], [2.6, 6.4]]
    assert world["action_vectors"]["up"] == [0.0, 1.0]
    assert world["target"] == [9.0, 9.0]
    assert world["push_threshold"] == pytest.approx(discrete_env.push_threshold)
    assert world["friction_coefficient"] == pytest.approx(discrete_env.friction_coefficient)


def test_continuous_world_describes_square_obstacles(continuous_env: ContinuousPushPOMDP):
    """The continuous world is written as the corner boxes it collides against.

    Purpose: The environment holds obstacles as an (N, 4) corner array, and a
        viewer that rebuilt them from the constructor's half-size tuples would
        draw a box the environment does not use.

    Given: A continuous environment with two square obstacles.
    When: Its trace is built.
    Then: Each obstacle is a centre with two half-extents, and the variant's
        own parameters are present.

    Test type: unit
    """
    world = (
        continuous_env.episode_visualizer().build_trace(_continuous_episode(), 0).payload["world"]
    )

    assert world["variant"] == "continuous"
    assert world["obstacle_shape"] == "square"
    assert world["obstacles"] == [[3.0, 3.0, 0.5, 0.5], [6.0, 6.0, 0.5, 0.5]]
    assert world["obstacle_radius"] is None
    assert world["robot_radius"] == pytest.approx(continuous_env.robot_radius)
    assert world["max_push"] == pytest.approx(continuous_env.max_push)
    assert world["obstacle_hit_terminal"] is False
    assert world["dangerous_area_hit_terminal"] is False


def test_continuous_actions_are_written_as_vectors(continuous_env: ContinuousPushPOMDP):
    """A continuous action survives as the vector it is.

    Purpose: The crate travels along the action, so an action written as a
        string or rounded to a label would make the viewer draw a push in a
        direction the environment never pushed.

    Given: A continuous episode whose actions are 2-D vectors.
    When: Its trace is built.
    Then: The envelope carries the vectors, and no action label table is
        invented for a world that has none.

    Test type: unit
    """
    trace = continuous_env.episode_visualizer().build_trace(_continuous_episode(), 0)

    assert trace.steps[0].action == [1.4, 0.0]
    assert trace.steps[-1].action is None
    assert trace.payload["world"]["action_vectors"] == {}


def test_discrete_action_wrapper_keeps_the_continuous_world():
    """The discrete-action wrapper is the continuous world with labels.

    Purpose: It inherits the visualizer, and a viewer that took its labels as
        proof of the discrete world would draw circular obstacles for a world
        whose obstacles are squares.

    Given: A ContinuousPushPOMDPDiscreteActions episode.
    When: Its trace is built.
    Then: The world is the continuous one and the labels carry their vectors.

    Test type: unit
    """
    env = ContinuousPushPOMDPDiscreteActions(
        discount_factor=0.99,
        **continuous_push_pinned_kwargs(grid_size=10, obstacles=[(3.0, 3.0, 0.5)]),
    )
    history = _episode([[0.8, 1.2, 2.2, 1.4, 9.0, 9.0], [1.8, 1.2, 2.2, 1.4, 9.0, 9.0]], ["right"])
    world = env.episode_visualizer().build_trace(history, 0).payload["world"]

    assert world["variant"] == "continuous"
    assert world["obstacle_shape"] == "square"
    assert world["action_vectors"]["right"] == [1.0, 0.0]


def test_hazard_terminal_state_keeps_only_the_two_positions():
    """A seven-number state is written as the four numbers a viewer draws.

    Purpose: The hazard-terminal variant appends an absorbing flag to the
        state, and a viewer that read states positionally would take that flag
        for a coordinate.

    Given: An environment whose obstacle hits are terminal.
    When: Its trace is built from seven-number states.
    Then: Each state is the robot and object positions only, and the flag is
        reported through the world block instead.

    Test type: unit
    """
    env = ContinuousPushPOMDP(
        discount_factor=0.99,
        **continuous_push_pinned_kwargs(
            grid_size=10, obstacles=[(3.0, 3.0, 0.5)], is_obstacle_hit_terminal=True
        ),
    )
    history = _episode(
        [[0.8, 1.2, 2.2, 1.4, 9.0, 9.0, 0.0], [1.8, 1.2, 2.2, 1.4, 9.0, 9.0, 1.0]],
        [np.array([1.0, 0.0])],
    )
    trace = env.episode_visualizer().build_trace(history, 0)

    assert trace.payload["states"] == [[0.8, 1.2, 2.2, 1.4], [1.8, 1.2, 2.2, 1.4]]
    assert trace.payload["world"]["obstacle_hit_terminal"] is True
    assert trace.reach_terminal_state is True


def test_empty_history_is_refused(discrete_env: PushPOMDP):
    """An episode with no steps is an error, not an empty trace.

    Purpose: A trace of nothing would be indexed and played as a real episode.

    Given: An empty history.
    When: The visualizer is asked to write it.
    Then: It raises.

    Test type: unit
    """
    with pytest.raises(ValueError):
        PushVisualizer(discrete_env).build_trace(history=[], episode_index=0)
