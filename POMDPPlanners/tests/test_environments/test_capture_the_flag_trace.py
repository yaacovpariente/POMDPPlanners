# SPDX-License-Identifier: MIT

"""Tests for the CaptureTheFlag episode trace.

The trace is what the browser viewer reads, so these tests pin its contents.
"""

import random
from pathlib import Path
from typing import List, cast

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.belief_payloads import (
    MAX_PAYLOAD_PARTICLES,
    belief_to_payload,
)
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.capture_the_flag_pomdp import (
    CaptureTheFlagPOMDP,
    decode_joint_action,
)
from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_visualization import (
    CAPTURE_THE_FLAG_PAYLOAD_KIND,
    CaptureTheFlagVisualizer,
)


@pytest.fixture(name="env")
def env_fixture() -> CaptureTheFlagPOMDP:
    """The shipped default field, which is what the viewer is built around."""
    return CaptureTheFlagPOMDP(discount_factor=0.95)


def _episode(env: CaptureTheFlagPOMDP, length: int = 4, seed: int = 0) -> List[StepData]:
    """Roll a short episode carrying real weighted particle beliefs.

    The belief matters: the trace exists to carry it, and a history with a
    placeholder belief would exercise none of what the viewer draws.

    Args:
        env: The environment to roll on.
        length: How many decision steps to record.
        seed: Seed applied to both random streams.

    Returns:
        The recorded steps, with a trailing state-only bookkeeping step.
    """
    random.seed(seed)
    np.random.seed(seed)
    state = cast(DiscreteDistribution, env.initial_state_dist()).values[2]
    particles = [env.initial_state_dist().sample()[0] for _ in range(16)]
    history: List[StepData] = []
    for _ in range(length):
        action = int(np.random.randint(0, len(env.get_actions())))
        next_state = env.sample_next_state(state, action)
        observation = env.sample_observation(next_state, action)
        particles = [env.sample_next_state(p, action) for p in particles]
        log_weights = np.array(
            [env.observation_log_probability(p, action, [observation])[0] for p in particles]
        )
        log_weights = np.where(np.isfinite(log_weights), log_weights, -1e9)
        belief = WeightedParticleBelief(
            particles=particles, log_weights=log_weights, resampling=False
        )
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=env.reward(state, action, next_state),
                belief=belief,
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
            belief=history[-1].belief,
        )
    )
    return history


def test_trace_round_trips_through_json(env, tmp_path: Path):
    """A written trace reads back as the same trace.

    Purpose: The file is the interface between Python and the browser viewer,
    so a field that does not survive the round trip is a field the viewer
    silently never sees.

    Given: A four-step episode.
    When: Its trace is written and read back.
    Then: The envelope and the payload match the original.
    """
    history = _episode(env)
    trace = CaptureTheFlagVisualizer(env).build_trace(
        history, episode_index=3, policy_name="PFT_DPW"
    )
    restored = EpisodeTrace.read(trace.write(tmp_path / "trace.json"))

    assert restored.payload_kind == CAPTURE_THE_FLAG_PAYLOAD_KIND
    assert restored.episode_index == 3
    assert restored.policy == "PFT_DPW"
    assert restored.num_steps == len(history)
    assert restored.payload == trace.payload
    assert restored.discounted_return == pytest.approx(trace.discounted_return)


def test_trace_records_both_teams_from_the_recorded_states(env):
    """Every step's cells come from that step's state, for both teams.

    Purpose: The viewer draws four units. Red's cells are hidden from the
    planner but not from a reader of the episode, and reconstructing them from
    anything but the recorded state would draw a game nobody played.

    Given: A four-step episode.
    When: The trace is built.
    Then: Each step's blue and red cells equal the state's own, and the
        carrier, freeze and score fields follow the same states.
    """
    history = _episode(env)
    payload = CaptureTheFlagVisualizer(env).build_trace(history, episode_index=0).payload
    layout = env.layout

    assert len(payload["blue_cells"]) == len(history)
    for step, blue, red, carrier, scores in zip(
        history,
        payload["blue_cells"],
        payload["red_cells"],
        payload["carrier_red_flag"],
        payload["scores"],
    ):
        assert blue == [list(cell) for cell in layout.blue_cells(step.state)]
        assert red == [list(cell) for cell in layout.red_cells(step.state)]
        assert carrier == int(step.state[layout.carrier_red_flag])
        assert scores == [int(step.state[layout.score_blue]), int(step.state[layout.score_red])]


def test_trace_records_the_true_flag_cell_per_step(env):
    """The banner's cell is written, because the viewer has to stand it somewhere.

    Purpose: The flag's cell is the hidden variable the belief is over. The
    viewer draws both the belief and the truth, and the truth has to come from
    the episode rather than from a default.
    """
    history = _episode(env)
    payload = CaptureTheFlagVisualizer(env).build_trace(history, episode_index=0).payload
    for step, cell in zip(history, payload["red_flag_cell"]):
        assert tuple(cell) == env.red_flag_cell(step.state)


def test_trace_splits_the_joint_action_per_player(env):
    """Each step's per-player actions agree with the envelope's joint action.

    Purpose: The viewer labels each player's move and draws a scan ping, and
    decoding a base-6 odometer in JavaScript would be a second place for the
    team size to be assumed rather than read.
    """
    history = _episode(env)
    trace = CaptureTheFlagVisualizer(env).build_trace(history, episode_index=0)
    for step, actions in zip(trace.steps, trace.payload["player_actions"]):
        if step.action is None:
            assert actions is None
        else:
            assert actions == list(decode_joint_action(int(step.action), env.n_blue))


def test_trace_delegates_belief_serialization_to_core(env):
    """The exporter writes no belief format of its own.

    Purpose: Belief is a core abstraction with a closed family of
    implementations. Serializing it per environment would mean fifteen copies
    of one dispatch and fifteen chances to miss a class. Comparing against
    core's own output is what stops the two drifting.
    """
    history = _episode(env)
    trace = CaptureTheFlagVisualizer(env).build_trace(history, episode_index=0)
    for step, written in zip(history, trace.payload["beliefs"]):
        assert written == belief_to_payload(step.belief)


def test_trace_writes_the_state_layout_the_particles_need(env):
    """The state vector's offsets are written, because a particle is a state.

    Purpose: A CaptureTheFlag particle is a whole flat state vector. Without
    the layout a reader has a list of numbers and no way to find the red
    players or the flag index inside it, and would have to re-derive index
    arithmetic that breaks the first time a team size changes.

    Given: An environment with two blue and two red players.
    When: The trace is built.
    Then: The payload's layout matches the environment's own, and reading a
        particle at those offsets recovers cells inside the field.
    """
    payload = CaptureTheFlagVisualizer(env).build_trace(_episode(env), episode_index=0).payload
    layout = payload["state_layout"]

    assert layout["red_pos"] == env.layout.red_pos
    assert layout["flag_cell"] == env.layout.flag_cell
    assert layout["size"] == env.layout.size

    particle = payload["beliefs"][0]["particles"][0]
    assert len(particle) == layout["size"]
    for j in range(env.n_red):
        cell = (particle[layout["red_pos"] + 2 * j], particle[layout["red_pos"] + 2 * j + 1])
        assert 0 <= cell[0] < env.grid_size[0] and 0 <= cell[1] < env.grid_size[1]
    assert 0 <= particle[layout["flag_cell"]] < len(env.red_flag_candidates)


def test_trace_takes_the_world_from_the_instance():
    """The world block comes from the configured environment, not class defaults.

    Purpose: A run may be configured onto a different field, and a viewer that
    drew the defaults would replay the episode on a field the planner never
    played, with units walking through trees that are not there.
    """
    env = CaptureTheFlagPOMDP(
        discount_factor=0.95,
        grid_size=(7, 5),
        midline=3,
        trees=[(1, 1), (5, 3)],
        n_blue=1,
        n_red=1,
        blue_base=(0, 2),
        red_base=(6, 2),
        blue_flag_cell=(1, 2),
        red_flag_candidates=[(5, 1), (5, 2), (4, 4)],
    )
    world = (
        CaptureTheFlagVisualizer(env)
        .build_trace(_episode(env, length=2), episode_index=0)
        .payload["world"]
    )

    assert world["grid_size"] == [7, 5]
    assert world["midline"] == 3
    assert world["trees"] == [[1, 1], [5, 3]]
    assert world["n_blue"] == 1 and world["n_red"] == 1
    assert world["red_flag_candidates"] == [[5, 1], [5, 2], [4, 4]]
    assert world["red_roles"] == ["defend"]


def test_trace_inherits_the_subsampling_cap(env):
    """A belief larger than core's cap is trimmed in the exporter's output too.

    Purpose: A trace carrying tens of thousands of state vectors would be
    hundreds of megabytes and a browser scene nothing could draw.
    """
    count = MAX_PAYLOAD_PARTICLES * 2
    particles = [env.initial_state_dist().sample()[0] for _ in range(count)]
    history = [
        StepData(
            state=particles[0],
            action=0,
            next_state=particles[1],
            observation=None,
            reward=-2.0,
            belief=WeightedParticleBelief(
                particles=particles,
                log_weights=np.log(np.ones(count) / count),
                resampling=False,
            ),
        )
    ]
    belief = (
        CaptureTheFlagVisualizer(env).build_trace(history, episode_index=0).payload["beliefs"][0]
    )

    assert belief["num_particles"] == count
    assert belief["num_written"] == MAX_PAYLOAD_PARTICLES


def test_environment_writes_a_trace_file(env, tmp_path: Path):
    """The environment's visualizer writes a readable file under the episode's index.

    Purpose: This is the call the simulation path makes. If the environment
    does not override episode_visualizer, it silently writes nothing and the
    episode page has no player.
    """
    written = env.episode_visualizer().write(
        history=_episode(env), output_dir=tmp_path, episode_index=2, policy_name="PFT_DPW"
    )
    assert written == tmp_path / "trace_2.json"

    restored = EpisodeTrace.read(written)
    assert restored.payload_kind == CAPTURE_THE_FLAG_PAYLOAD_KIND
    assert restored.policy == "PFT_DPW"


def test_trace_rejects_an_empty_history(env):
    """There is no trace for an episode with no steps."""
    with pytest.raises(ValueError, match="empty history"):
        CaptureTheFlagVisualizer(env).build_trace([], episode_index=0)
