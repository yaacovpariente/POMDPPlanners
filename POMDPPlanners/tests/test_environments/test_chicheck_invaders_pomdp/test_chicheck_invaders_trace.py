# SPDX-License-Identifier: MIT

"""Tests for the Chicheck Invaders episode trace.

The payload must round-trip and mean what it says: a viewer decoding a state
vector with the layout the trace carries gets the flock the episode had, and
the shot record agrees with the environment's own ``fires`` and
``shot_target`` rather than with a second rule written in the visualizer.
"""

import random
from pathlib import Path
from typing import List

import numpy as np
import pytest

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import belief_to_payload
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.chicheck_invaders_pomdp import (
    CHICHECK_INVADERS_PAYLOAD_KIND,
    ChicheckInvadersPOMDP,
    ChicheckInvadersVisualizer,
    create_chicheck_invaders_belief,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import chicheck_invaders_pinned_kwargs


def build_chicheck_invaders_env() -> ChicheckInvadersPOMDP:
    """Build the environment the trace tests run on."""
    return ChicheckInvadersPOMDP(discount_factor=0.95, **chicheck_invaders_pinned_kwargs())


def create_deterministic_chicheck_invaders_episode(seed: int = 11) -> List[StepData]:
    """Create a deterministic Chicheck Invaders episode.

    The belief attached to each step is a real
    :class:`ChicheckInvadersBelief` rather than a mock, so the trace's belief
    payload is tested on what a run actually records. Both RNGs are seeded,
    not just NumPy, because ``conftest`` seeds the stdlib ``random`` once at
    import and the episode would otherwise depend on which tests ran first.

    Args:
        seed: Random seed pinning the flock, the dives, the sensor noise and the
            filter's resampling.

    Returns:
        List of StepData objects representing the episode history.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = build_chicheck_invaders_env()
    belief = create_chicheck_invaders_belief(env, n_particles=60)
    state = env.initial_state_dist().sample()[0]
    history: List[StepData] = []

    # Shoot, sidestep, shoot again. Indices follow ChicheckInvadersAction:
    # 0 stay, 1 left, 2 right, 3 fire.
    action_sequence = [1, 3, 0, 2, 3, 2, 0, 3, 1, 3, 0, 3]

    for action in action_sequence:
        if env.is_terminal(state):
            break
        next_state, observation, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=float(reward),
                belief=belief,
                info=env.step_info(state, action, next_state),
            )
        )
        belief = belief.update(action=action, observation=observation, pomdp=env)
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


@pytest.fixture(name="episode")
def episode_fixture():
    """A real, deterministic Chicheck Invaders episode."""
    return create_deterministic_chicheck_invaders_episode(seed=11)


@pytest.fixture(name="env")
def env_fixture():
    """The environment that episode ran on."""
    return build_chicheck_invaders_env()


def _decode_chicken(vector, layout, slot):
    """Decode one chicken slot out of a raw state vector, the way a viewer does.

    Written here rather than imported from the schema on purpose: the point of
    the layout block is that a reader who has only the trace can decode it, and
    a test that used the Python constants would pass even if the block were
    wrong.
    """
    base = layout["ship_width"] + slot * layout["chicken_width"]
    return {
        "column": vector[base + layout["chicken_column"]],
        "row": vector[base + layout["chicken_row"]],
        "direction": vector[base + layout["chicken_direction"]],
        "diving": vector[base + layout["chicken_mode"]] == layout["mode_dive"],
        "alive": vector[base + layout["chicken_alive"]] > 0,
    }


def test_trace_round_trips_through_json(env, episode, tmp_path: Path):
    """The written file reads back as the same trace.

    Purpose: The viewer reads JSON, not Python objects. A payload that only
    survives in memory is a payload nobody can draw.

    Given: A Chicheck Invaders episode.
    When: Its trace is written and read back.
    Then: The envelope and the whole payload compare equal.
    """
    trace = ChicheckInvadersVisualizer(env).build_trace(
        episode, episode_index=2, policy_name="PFT_DPW"
    )
    path = trace.write(tmp_path / "trace_2.json")

    reloaded = EpisodeTrace.read(path)

    assert reloaded.payload_kind == CHICHECK_INVADERS_PAYLOAD_KIND
    assert reloaded.episode_index == 2
    assert reloaded.policy == "PFT_DPW"
    assert reloaded.discount_factor == pytest.approx(env.discount_factor)
    assert len(reloaded.steps) == len(episode)
    assert reloaded.payload == trace.to_dict()["payload"]


def test_trace_layout_decodes_the_recorded_states(env, episode):
    """The layout block is enough to read a state vector back.

    Purpose: States are written as the raw vectors the run held, so the layout
    the trace carries is the only thing standing between a viewer and a flock
    drawn in the wrong cells. A belief particle is a state vector too, so this
    decoder is the one the belief uses as well.

    Given: A Chicheck Invaders episode.
    When: Each recorded state is decoded with the payload's own layout.
    Then: The ship's column and every chicken slot match the episode's states.
    """
    trace = ChicheckInvadersVisualizer(env).build_trace(episode, episode_index=0)
    layout = trace.payload["layout"]["state"]

    assert len(trace.payload["states"]) == len(episode)
    for step, vector in zip(episode, trace.payload["states"]):
        assert vector[layout["ship_column"]] == pytest.approx(env.ship_column(step.state))
        flock = env.chickens(step.state)
        for slot in range(env.num_chickens):
            decoded = _decode_chicken(vector, layout, slot)
            assert decoded["column"] == pytest.approx(flock[slot, 0])
            assert decoded["row"] == pytest.approx(flock[slot, 1])
            assert decoded["alive"] == bool(flock[slot, 4] > 0)


def test_trace_shots_come_from_the_environment(env, episode):
    """What the gun did is asked of the environment, not re-derived.

    Purpose: The beam is the whole hitscan event — there is no projectile to
    check it against — so a shot record that disagreed with the kill the episode
    scored would draw a lie nothing else would catch.

    Given: A Chicheck Invaders episode including steps that fire.
    When: The trace is built.
    Then: Each step's shot record equals the environment's own ``fires`` and
        ``shot_target`` for that step, and at least one step fired. The terminal
        bookkeeping step carries no action and therefore never fires.
    """
    trace = ChicheckInvadersVisualizer(env).build_trace(episode, episode_index=0)
    shots = trace.payload["shots"]

    assert len(shots) == len(episode)
    assert any(shot["fired"] for shot in shots), "fixture must exercise the gun"
    for step, shot in zip(episode, shots):
        if step.action is None:
            assert shot["fired"] is False
            continue
        assert shot["fired"] == env.fires(step.state, step.action)
        if not shot["fired"]:
            assert shot["target_slot"] == -1
            continue
        assert shot["target_slot"] == env.shot_target(step.state)


def test_trace_delegates_belief_serialization_to_core(env, episode):
    """The visualizer writes no belief format of its own.

    Purpose: Belief is a core abstraction with a closed family of
    implementations. Serializing it per environment would mean a copy of the
    same dispatch in every environment and a chance to miss a class in each.

    Given: An episode whose steps carry real particle beliefs.
    When: The trace is built.
    Then: Each step's belief payload equals ``belief_to_payload`` of that step's
        belief, field for field.
    """
    trace = ChicheckInvadersVisualizer(env).build_trace(episode, episode_index=0)

    for step, written in zip(episode, trace.payload["beliefs"]):
        assert written == belief_to_payload(step.belief)


def test_trace_takes_the_sensor_parameters_from_the_instance(episode):
    """The cone and the ring drawn are the ones this run was configured with.

    Purpose: The viewer builds both footprints straight from these two numbers.
    Reading them off the class defaults would draw a sensor nobody had the
    moment a run configured a wider cone.

    Given: An environment with a non-default camera slope and radar radius.
    When: The trace is built.
    Then: The payload's world block carries that run's values.
    """
    env = build_chicheck_invaders_env()
    env.camera_slope = 0.5
    env.radar_radius = 3.25

    world = ChicheckInvadersVisualizer(env).build_trace(episode, episode_index=0).payload["world"]

    assert world["camera_slope"] == pytest.approx(0.5)
    assert world["radar_radius"] == pytest.approx(3.25)


def test_trace_rejects_an_empty_history(env):
    """An empty history is an error, not an empty trace.

    Purpose: A zero-step trace would render as a blank viewer that looks like a
    working page, which is worse than a failure.

    Given: No steps.
    When: A trace is asked for.
    Then: ValueError.
    """
    with pytest.raises(ValueError):
        ChicheckInvadersVisualizer(env).build_trace([], episode_index=0)


def test_environment_writes_a_trace_file(env, episode, tmp_path: Path):
    """The environment writes the trace through the core hook.

    Purpose: The reporting site finds traces by a fixed file name across
    environments it has never heard of, so the environment must go through
    its visualizer's ``write`` rather than name its own file.

    Given: A Chicheck Invaders episode.
    When: The environment's visualizer writes episode 3.
    Then: ``trace_3.json`` appears and reads back with this payload kind.
    """
    written = env.episode_visualizer().write(history=episode, output_dir=tmp_path, episode_index=3)

    assert written == tmp_path / "trace_3.json"
    assert EpisodeTrace.read(written).payload_kind == CHICHECK_INVADERS_PAYLOAD_KIND
