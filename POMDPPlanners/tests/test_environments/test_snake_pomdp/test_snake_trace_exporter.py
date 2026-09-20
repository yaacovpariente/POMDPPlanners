# SPDX-License-Identifier: MIT

"""The Snake episode trace, and the GIF the move beside it must not have changed.

Two things are worth pinning here. The first is that the payload says what the
episode said: the body, the apple, the readings and the belief, aligned to the
step a reader will see them under. The second is that moving the renderer into
``visualizer/`` changed nothing about what it renders -- the golden hash covers
that inside the project's Docker image, and this covers the part of it that can
run anywhere.
"""

import hashlib
import random
from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.snake_pomdp import SnakeBelief, SnakePOMDP
from POMDPPlanners.environments.snake_pomdp.snake_pomdp import SnakeAction, SnakeTermination
from POMDPPlanners.environments.snake_pomdp.visualizer.snake_visualizer import SnakeVisualizer
from POMDPPlanners.environments.snake_pomdp.visualizer.trace_exporter import (
    SNAKE_PAYLOAD_KIND,
    build_snake_trace,
)


def build_env(**overrides) -> SnakePOMDP:
    """Return a small Snake environment, with overrides applied."""
    kwargs = {
        "grid_size": 7,
        "target_length": 6,
        "starvation_limit": 20,
        "discount_factor": 0.98,
    }
    kwargs.update(overrides)
    return SnakePOMDP(**kwargs)


def rollout(env: SnakePOMDP, actions, seed: int = 11):
    """Run a fixed action sequence and return its ``StepData`` history.

    The belief is the real :class:`SnakeBelief` rather than a stand-in: the
    belief is the field the trace exists to carry, and a mock one would leave
    the part most worth checking unchecked.

    Args:
        env: The environment to run in.
        actions: The actions to take, in order.
        seed: Seed pinning the apple draws, the readings and the particles.

    Returns:
        The episode's ``StepData`` records, terminal bookkeeping step included.
    """
    random.seed(seed)
    np.random.seed(seed)
    belief = SnakeBelief.from_environment(env, n_particles=16)
    state = env.initial_state_dist().sample()[0]
    history = []
    for action in actions:
        if env.is_terminal(state):
            break
        next_state, observation, reward = env.sample_next_step(state, int(action))
        history.append(
            StepData(
                state=state,
                action=int(action),
                next_state=next_state,
                observation=observation,
                reward=reward,
                belief=belief,
                info=env.step_info(state, int(action), next_state),
            )
        )
        belief = belief.update(action=int(action), observation=observation, pomdp=env)
        state = next_state
    history.append(
        StepData(
            state=state,
            action=None,
            next_state=None,
            observation=None,
            reward=None,
            belief=belief,
            info=env.step_info(state, None, None),
        )
    )
    return history


STRAIGHT_RUN = (
    [SnakeAction.GO_STRAIGHT] * 4 + [SnakeAction.TURN_RIGHT] + [SnakeAction.GO_STRAIGHT] * 3
)


@pytest.fixture(name="env")
def env_fixture() -> SnakePOMDP:
    """A small Snake environment."""
    return build_env()


@pytest.fixture(name="history")
def history_fixture(env: SnakePOMDP):
    """One fixed episode on that environment."""
    return rollout(env, STRAIGHT_RUN)


def test_snake_trace_round_trips_through_json(env, history, tmp_path: Path):
    """The trace survives being written and read back, field for field.

    Purpose: The viewer reads a file, not an object. A payload that held a
    numpy scalar would serialize and then come back as something the scene
    cannot index, and nothing in the exporter would have failed.

    Given: One recorded Snake episode.
    When: Its trace is written to disk and read back.
    Then: The reconstructed trace has the same kind, envelope and payload.

    Test type: unit
    """
    trace = build_snake_trace(env, history, episode_index=3, policy_name="POMCP")
    written = trace.write(tmp_path / "trace_3.json")
    restored = EpisodeTrace.read(written)

    assert restored.payload_kind == SNAKE_PAYLOAD_KIND == "snake.v1"
    assert restored.episode_index == 3
    assert restored.policy == "POMCP"
    assert restored.num_steps == len(history)
    assert restored.payload == trace.to_dict()["payload"]


def test_snake_trace_writes_one_entry_per_step(env, history):
    """Every per-step list is as long as the episode.

    Purpose: The scene indexes all of them with one step index. A list that was
    short by one would draw the previous step's apple against this step's body
    for the rest of the episode, and nothing would raise.

    Test type: unit
    """
    payload = build_snake_trace(env, history, 0).payload
    for field in (
        "bodies",
        "foods",
        "lengths",
        "steps_since_food",
        "terminations",
        "turn_outcomes",
        "sightings",
        "scents",
        "observations",
        "beliefs",
    ):
        assert len(payload[field]) == len(history), field


def test_snake_trace_bodies_and_apples_come_from_the_recorded_state(env, history):
    """Body, apple and length are read off the state the step was taken from.

    Purpose: A viewer that drew the successor's body would show the snake one
    cell ahead of the belief and the reading beside it, which is exactly the
    off-by-one the GIF renderer documents avoiding.

    Test type: unit
    """
    payload = build_snake_trace(env, history, 0).payload
    for index, step in enumerate(history):
        expected = [[int(r), int(c)] for r, c in env.body(step.state)]
        assert payload["bodies"][index] == expected
        food = env.food(step.state)
        assert payload["foods"][index] == (None if food is None else [food[0], food[1]])
        assert payload["lengths"][index] == env.snake_length(step.state)


def test_snake_trace_shifts_the_reading_by_one_step(env, history):
    """The sighting and scent are the reading the agent acted on, not the one it caused.

    Purpose: ``StepData.observation`` is what the step produced. Writing it
    beside the board the step was chosen from would credit the agent with a
    reading it had not received, which is the one way this payload could
    flatter a planner.

    Given: An episode whose steps each carry an observation.
    When: The trace is built.
    Then: Step 0 reports no reading, and step ``i`` reports what step ``i - 1``
        observed.

    Test type: unit
    """
    payload = build_snake_trace(env, history, 0).payload

    assert payload["sightings"][0] is None
    assert payload["scents"][0] is None

    for index in range(1, len(history)):
        previous = history[index - 1].observation
        flat = tuple(int(value) for value in previous)
        if flat[0] != 1:
            # The terminal reading carries neither, by design.
            assert payload["sightings"][index] is None
            assert payload["scents"][index] is None
            continue
        _, seen, scent = env.decode_observation(flat)
        assert payload["scents"][index] == scent
        assert payload["sightings"][index] == (None if seen is None else [seen[0], seen[1]])


def test_snake_trace_reports_what_each_turn_would_do(env, history):
    """The three turns carry the environment's own verdict, and none on a terminal state.

    Purpose: This is the danger the viewer marks. Recomputing it in JavaScript
    would be a second implementation of the transition rule, and the two would
    drift; taking it from ``transition_outcome`` means the marked cell is the
    one the environment would actually kill in.

    Test type: unit
    """
    payload = build_snake_trace(env, history, 0).payload
    for index, step in enumerate(history):
        outcomes = payload["turn_outcomes"][index]
        if env.is_terminal(step.state):
            assert outcomes is None
            continue
        expected = [
            int(env.transition_outcome(step.state, int(action))[3]) for action in SnakeAction
        ]
        assert outcomes == expected


def test_snake_trace_delegates_belief_serialization_to_core(env, history):
    """The exporter writes no belief format of its own.

    Purpose: Belief is a core abstraction with a closed family of
    implementations. An environment that serialized its own would be a
    sixteenth copy of the same dispatch and a place for the two to drift.
    Comparing against core's output field for field is what stops that.

    Test type: unit
    """
    payload = build_snake_trace(env, history, 0).payload
    for step, written in zip(history, payload["beliefs"]):
        assert written == belief_to_payload(step.belief)


def test_snake_belief_is_serialized_as_particles_not_as_unsupported(env):
    """``SnakeBelief`` reaches core's particle branch rather than its fallback.

    Purpose: ``SnakeBelief`` is this environment's own class, and core
    dispatches on type. If it ever stopped subclassing
    :class:`WeightedParticleBelief`, every trace would quietly carry
    ``unsupported`` and the viewer would draw no belief at all -- with nothing
    failing anywhere.

    Test type: unit
    """
    belief = SnakeBelief.from_environment(env, n_particles=8)
    assert isinstance(belief, WeightedParticleBelief)

    payload = belief_to_payload(belief)
    assert payload["kind"] == "particles"
    assert payload["belief_class"] == "SnakeBelief"
    # The viewer reads the apple cell out of these slots, so they have to be
    # the ones the payload's state_layout names.
    layout = build_snake_trace(env, rollout(env, [SnakeAction.GO_STRAIGHT]), 0).payload[
        "state_layout"
    ]
    row, col = layout["food_row"], layout["food_col"]
    for particle in payload["particles"]:
        assert env.in_grid((int(round(particle[row])), int(round(particle[col]))))


def test_snake_trace_takes_the_world_from_the_instance():
    """The world block reports the configured environment, not the class defaults.

    Purpose: A run may be configured well away from the defaults -- this one
    is -- and a viewer that drew the defaults would draw a board of the wrong
    size with a starvation clock the episode never ran under.

    Test type: unit
    """
    env = build_env(grid_size=9, target_length=5, starvation_limit=13, scent_accuracy=0.8)
    payload = build_snake_trace(env, rollout(env, STRAIGHT_RUN), 0).payload
    assert payload["world"] == {
        "grid_size": 9,
        "target_length": 5,
        "window_radius": 2,
        "detection_probability": 0.9,
        "scent_accuracy": 0.8,
        "starvation_limit": 13,
    }


def test_snake_trace_reports_the_episode_outcome(env):
    """An episode that ended reports its terminal state and its reason.

    Purpose: The site's status line and the viewer's outcome tag both read
    this. An episode that died against the wall but reported
    ``reach_terminal_state`` false would be described as having run out of
    steps.

    Test type: unit
    """
    # Straight west from the centre is the shortest way into a wall.
    history = rollout(env, [SnakeAction.TURN_LEFT] + [SnakeAction.GO_STRAIGHT] * 8)
    trace = build_snake_trace(env, history, 0)
    assert trace.reach_terminal_state is True
    assert trace.payload["terminations"][-1] != int(SnakeTermination.RUNNING)


def test_snake_environment_writes_a_trace_file(env, history, tmp_path: Path):
    """``cache_trace`` writes the file the reporting site looks for."""
    written = env.cache_trace(
        history=history, output_dir=tmp_path, episode_index=2, policy_name="POMCPOW"
    )
    assert written == tmp_path / "trace_2.json"
    assert EpisodeTrace.read(written).payload_kind == SNAKE_PAYLOAD_KIND


def test_snake_trace_rejects_an_empty_history(env):
    """There is no trace for an episode with no steps."""
    with pytest.raises(ValueError, match="empty history"):
        build_snake_trace(env, [], 0)


def test_moving_the_renderer_did_not_change_what_cache_visualization_renders(
    env, history, tmp_path: Path
):
    """The GIF ``cache_visualization`` writes is the moved renderer's own output.

    Purpose: The renderer moved into ``visualizer/`` and its bytes are pinned
    by a golden hash that only runs inside the project's Docker image. The one
    thing the move could have broken anywhere is the lazy import in
    ``cache_visualization``: pointed at a stale module it would still render a
    GIF, just not this one. Hashing both and comparing is what catches that
    without needing the pinned image.

    Given: One recorded Snake episode.
    When: It is rendered through ``cache_visualization`` and again through
        :class:`SnakeVisualizer` taken from its new home.
    Then: The two files are byte-identical.

    Test type: unit
    """
    env.cache_visualization(history=history, output_dir=tmp_path, episode_index=0)
    through_environment = tmp_path / "snake_board_0.gif"
    assert through_environment.exists()

    direct = tmp_path / "direct.gif"
    SnakeVisualizer(env).create_visualization(history, direct)

    assert (
        hashlib.sha256(through_environment.read_bytes()).hexdigest()
        == hashlib.sha256(direct.read_bytes()).hexdigest()
    )
