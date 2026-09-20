# SPDX-License-Identifier: MIT

"""Tests for the T-Maze episode trace exporter and its viewer wiring.

Two things are checked here that no other environment's trace has to check:

* the **cue phase** survives into the payload, named rather than left as the
  model's float. It is the one field the browser viewer cannot do without —
  T-Maze is a memory task, and a replay that does not say when the single
  reading happened shows an agent walking a corridor for no reason;
* the exporter does not disturb the GIF. The renderer is shared with the rest
  of the Maze family and its output is pinned by a golden hash, so the check is
  that rendering the same episode before and after a trace is built produces
  byte-identical bytes.
"""

import hashlib
from pathlib import Path
from typing import List

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.maze_pomdp.maze_visualizer import MazeVisualizer
from POMDPPlanners.environments.maze_pomdp.t_maze_pomdp import (
    CUE_EMITTING,
    GOAL_LEFT,
    GOAL_RIGHT,
    OBSERVATION_EMPTY,
    OBSERVATION_LEFT_CUE,
    OBSERVATION_RIGHT_CUE,
    STATE_CUE_PHASE,
    TMazePOMDP,
    create_t_maze_state,
)
from POMDPPlanners.environments.maze_pomdp.maze_visualization.t_maze_trace_exporter import (
    T_MAZE_PAYLOAD_KIND,
    build_t_maze_trace,
)
from POMDPPlanners.reporting.pages import scene_script_path
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import t_maze_pinned_kwargs

SCENE_MODULE = (
    Path(__file__).resolve().parents[1].parent
    / "reporting"
    / "static"
    / "viewer"
    / "scenes"
    / "t-maze.js"
)


def _belief(state: np.ndarray, left_weight: float) -> WeightedParticleBelief:
    """A two-particle belief on ``state``'s own cell, ``left_weight`` on the left goal.

    Both particles carry the agent's position and cue phase, because position is
    observable here: a filter's particles agree about where the agent is and
    disagree only about which arm pays.
    """
    position = (int(state[0]), int(state[1]))
    cue_phase = float(state[STATE_CUE_PHASE])
    return WeightedParticleBelief(
        particles=[
            create_t_maze_state(position, GOAL_LEFT, cue_phase),
            create_t_maze_state(position, GOAL_RIGHT, cue_phase),
        ],
        log_weights=np.log(np.array([left_weight, 1.0 - left_weight])),
    )


def _episode(
    environment: TMazePOMDP,
    goal_side: float = GOAL_LEFT,
    cue_observation: str = OBSERVATION_LEFT_CUE,
) -> List[StepData]:
    """Walk the stem, read the cue, and turn left at the junction.

    The episode is driven through the environment's own transition function, so
    the states and the cue phases are the ones the model produces rather than
    ones this test asserts into existence. Only the belief is hand-built, and
    deliberately not a point mass: a point mass would render and serialize the
    same picture whether the goal-side reading worked or not.
    """
    state = create_t_maze_state(environment.start_cell, goal_side)
    actions = ["up"] * environment.stem_length + ["left"]
    left_weights = [0.5] + [0.9] * environment.stem_length
    observations = [cue_observation] + [OBSERVATION_EMPTY] * environment.stem_length

    history: List[StepData] = []
    for action, left_weight, observation in zip(actions, left_weights, observations):
        next_state = environment.sample_next_state(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=environment.reward(state, action, next_state),
                belief=_belief(state, left_weight),
            )
        )
        state = next_state
        if environment.is_terminal(state):
            break

    history.append(
        StepData(
            state=state,
            action=None,
            next_state=state,
            observation=None,
            reward=0.0,
            belief=_belief(state, left_weights[-1]),
        )
    )
    return history


@pytest.fixture(name="env")
def env_fixture() -> TMazePOMDP:
    """A T-Maze with the suite's pinned configuration."""
    return TMazePOMDP(discount_factor=0.95, **t_maze_pinned_kwargs())


def test_trace_world_comes_from_the_environment_instance(env: TMazePOMDP):
    """The payload's world block is the configured environment's own geometry.

    Purpose: A viewer builds the corridor from this block alone, so a value
    copied from a class default would draw a maze the episode never ran in.

    Given: A T-Maze with a non-default stem and arm length.
    When: A trace is built for an episode on it.
    Then: The world block reports that instance's geometry and rewards.

    Test type: unit
    """
    configured = TMazePOMDP(
        discount_factor=0.95, **t_maze_pinned_kwargs(stem_length=5, arm_length=2)
    )
    trace = build_t_maze_trace(configured, _episode(configured), episode_index=0)

    world = trace.payload["world"]
    assert trace.payload_kind == T_MAZE_PAYLOAD_KIND
    assert world["stem_length"] == 5
    assert world["arm_length"] == 2
    assert world["junction"] == [0, 5]
    assert world["left_endpoint"] == [-2, 5]
    assert world["right_endpoint"] == [2, 5]
    assert sorted(map(tuple, world["cells"])) == sorted(configured.valid_cells)
    assert world["goal_reward"] == pytest.approx(configured.goal_reward)
    assert world["cue_accuracy"] == pytest.approx(configured.cue_accuracy)


def test_trace_carries_the_cue_phase_of_every_state(env: TMazePOMDP):
    """Every recorded state's cue phase is written, named rather than encoded.

    Purpose: The cue phase is what the T-Maze viewer exists to show. Without
    it the page cannot say when the one reading happened, and the memory
    stretch is indistinguishable from an ordinary corridor walk.

    Given: An episode that walks onto the cue cell and past it.
    When: The trace is built.
    Then: One phase name per state, agreeing with the state's own encoding,
        with exactly one "emitting" step.

    Test type: unit
    """
    history = _episode(env)
    trace = build_t_maze_trace(env, history, episode_index=0)

    phases = trace.payload["cue_phases"]
    assert len(phases) == len(history)
    assert phases.count("emitting") == 1
    assert phases[0] == "unseen"
    assert phases[-1] == "consumed"

    emitting = phases.index("emitting")
    assert float(history[emitting].state[STATE_CUE_PHASE]) == CUE_EMITTING
    # The states carry the phase too, in the slot the payload names, so a
    # reader can cross-check the label against the model's own value.
    slot = trace.payload["world"]["state_slots"]["cue_phase"]
    assert trace.payload["states"][emitting][slot] == CUE_EMITTING


def test_cue_reading_is_the_observation_received_not_the_true_side(env: TMazePOMDP):
    """A cue that lied is written as it was heard.

    Purpose: The cue is noisy, so the reading and the truth can disagree. A
    viewer that drew the true side on the cue's sign would hide exactly the
    episodes where the agent was misled, which are the ones worth looking at.

    Given: An episode whose goal is on the left but whose cue said "right".
    When: The trace is built.
    Then: cue_reading is "right" and goal_side is "left".

    Test type: unit
    """
    history = _episode(env, goal_side=GOAL_LEFT, cue_observation=OBSERVATION_RIGHT_CUE)
    trace = build_t_maze_trace(env, history, episode_index=0)

    assert trace.payload["cue_reading"] == "right"
    assert trace.payload["goal_side"] == "left"


def test_belief_is_serialized_by_core_and_sums_to_the_goal_side(env: TMazePOMDP):
    """The belief comes back as core's particle payload, keeping the goal side.

    Purpose: The exporter writes nothing about belief itself. What it must not
    do is lose the slot the viewer sums over: P(goal = left) is read out of the
    particles, so a payload without them would leave the page with no belief.

    Given: An episode whose belief swings to 0.9 on the step that reads the cue.
    When: The trace is built.
    Then: Each belief is a particle payload whose left-goal weights are the
        episode's own.

    Test type: unit
    """
    trace = build_t_maze_trace(env, _episode(env), episode_index=0)
    slot = trace.payload["world"]["state_slots"]["goal_side"]
    left_value = trace.payload["world"]["goal_left_value"]

    left_mass = []
    for belief in trace.payload["beliefs"]:
        assert belief["kind"] == "particles"
        left_mass.append(
            sum(
                weight
                for particle, weight in zip(belief["particles"], belief["weights"])
                if particle[slot] == left_value
            )
        )

    assert left_mass[0] == pytest.approx(0.5)
    assert all(mass == pytest.approx(0.9) for mass in left_mass[1:])


def test_environment_writes_the_trace_and_it_round_trips(env: TMazePOMDP, tmp_path: Path):
    """cache_trace writes a file the schema reads back unchanged.

    Purpose: The file is the interface between Python and the browser viewer,
    so a field that does not survive JSON is a field the viewer never sees.

    Given: An episode on a T-Maze.
    When: cache_trace writes it and EpisodeTrace reads it back.
    Then: The envelope and the T-Maze payload match what was built.

    Test type: integration
    """
    history = _episode(env)
    written = env.cache_trace(
        history=history, output_dir=tmp_path, episode_index=3, policy_name="a_policy"
    )

    assert written is not None
    reloaded = EpisodeTrace.read(written)
    original = build_t_maze_trace(env, history, episode_index=3, policy_name="a_policy")

    assert reloaded.payload_kind == T_MAZE_PAYLOAD_KIND
    assert reloaded.episode_index == 3
    assert reloaded.policy == "a_policy"
    assert reloaded.reach_terminal_state is True
    assert reloaded.num_steps == len(history)
    assert reloaded.payload == original.to_dict()["payload"]


def test_empty_history_is_refused(env: TMazePOMDP):
    """An episode with no steps is an error, not an empty trace.

    Purpose: An empty trace would reach the viewer as a page with a scrubber
    and nothing in it, which reads as a broken renderer rather than a missing
    episode.

    Given: An empty history.
    When: A trace is built from it.
    Then: ValueError.

    Test type: unit
    """
    with pytest.raises(ValueError):
        build_t_maze_trace(env, [], episode_index=0)


def test_building_a_trace_leaves_the_gif_bytes_unchanged(env: TMazePOMDP, tmp_path: Path):
    """The GIF renders byte-identically either side of a trace export.

    Purpose: The Maze renderer's output is pinned by a golden hash, and the
    exporter reads the same histories and the same belief objects. An exporter
    that normalised a belief in place, or consumed an iterator, would move that
    hash and nothing else would say so.

    Given: One deterministic T-Maze episode.
    When: The GIF is rendered, a trace is built from the same history, and the
        GIF is rendered again.
    Then: The two GIFs are byte-identical.

    Test type: integration
    """
    history = _episode(env)

    before = tmp_path / "before.gif"
    MazeVisualizer(env).create_visualization(history, before)
    first = hashlib.sha256(before.read_bytes()).hexdigest()

    build_t_maze_trace(env, history, episode_index=0)

    after = tmp_path / "after.gif"
    MazeVisualizer(env).create_visualization(history, after)
    assert hashlib.sha256(after.read_bytes()).hexdigest() == first


def test_the_scene_module_is_where_the_payload_kind_resolves():
    """The viewer's by-convention lookup finds this environment's scene module.

    Purpose: Scene modules are resolved from the payload kind by a name rule,
    not a table, so a kind and a filename that disagree fail only in a browser,
    at which point nothing in the suite has said a word.

    Given: The T-Maze payload kind.
    When: The reporting site derives the scene script path.
    Then: It points at a file that exists and registers itself under that kind.

    Test type: unit
    """
    assert scene_script_path(T_MAZE_PAYLOAD_KIND) == "/static/viewer/scenes/t-maze.js"
    assert SCENE_MODULE.is_file()
    source = SCENE_MODULE.read_text(encoding="utf-8")
    assert f'V.scenes["{T_MAZE_PAYLOAD_KIND}"]' in source
