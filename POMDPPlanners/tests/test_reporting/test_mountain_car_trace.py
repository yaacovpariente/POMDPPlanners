# SPDX-License-Identifier: MIT

"""Tests for the Mountain Car episode trace exporter.

The exporter is the whole interface between a finished episode and the browser
viewer, so its payload is checked field by field: a field that does not survive
the round trip is a field the viewer silently never sees, and a field the
viewer reads by index is one a wrong-length row would quietly corrupt.

The hill is checked against the GIF renderer rather than against a literal.
Mountain Car's valley is the one part of this world that is not a constructor
argument, and the two drawings of one episode disagreeing about where the
ground is would be worse than either of them being wrong on its own.
"""

from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.mountain_car_pomdp import (
    MOUNTAIN_CAR_PAYLOAD_KIND,
    MountainCarPOMDP,
    MountainCarVisualizer,
    build_mountain_car_trace,
)
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_trace_exporter import (
    hill_height,
)

REVERSE, NEUTRAL, FORWARD = -1, 0, 1


@pytest.fixture(name="env")
def env_fixture():
    """A Mountain Car at the shipped defaults."""
    return MountainCarPOMDP(discount_factor=0.99)


def _belief(states, weights=None):
    """A particle belief over whole (position, velocity) states."""
    particles = [np.asarray(state, dtype=float) for state in states]
    if weights is None:
        weights = np.ones(len(particles)) / len(particles)
    return WeightedParticleBelief(particles=particles, log_weights=np.log(weights))


def _step(state, action, observation, reward, belief, next_state=None):
    return StepData(
        state=state,
        action=action,
        next_state=next_state,
        observation=observation,
        reward=reward,
        belief=belief,
    )


def _episode():
    """A short hand-built episode: reverse, coast, accelerate, arrive.

    Hand-built rather than planned, because these tests are about what the
    exporter writes and not about what a planner does. The shapes are the real
    ones: two-component states, a noisy reading of the same two numbers, and a
    terminal bookkeeping step that carries a state but no decision.
    """
    belief = _belief([(-0.5, 0.0), (-0.48, 0.001), (-0.52, -0.001)])
    return [
        _step((-0.50, 0.000), REVERSE, (-0.44, 0.004), -1.0, belief, (-0.51, -0.001)),
        _step((-0.51, -0.001), NEUTRAL, (-0.58, -0.008), -1.0, belief, (-0.52, -0.002)),
        _step((-0.52, -0.002), FORWARD, (-0.49, 0.003), -1.0, belief, (0.51, 0.030)),
        _step((0.51, 0.030), None, None, None, belief),
    ]


def test_trace_round_trips_through_json(env, tmp_path: Path):
    """A trace written and read back is the same trace.

    Purpose: The file is the interface between Python and the browser viewer,
    so a field that does not survive the round trip is a field the viewer
    silently never sees.

    Given: A Mountain Car episode.
    When: Its trace is written to disk and read back.
    Then: Envelope and payload match the original, field for field.

    Test type: unit
    """
    trace = build_mountain_car_trace(env, _episode(), episode_index=2, policy_name="PFT_DPW")
    restored = EpisodeTrace.read(trace.write(tmp_path / "trace_2.json"))

    assert restored.payload_kind == MOUNTAIN_CAR_PAYLOAD_KIND
    assert restored.episode_index == 2
    assert restored.policy == "PFT_DPW"
    assert restored.discount_factor == pytest.approx(env.discount_factor)
    assert restored.payload == trace.payload
    assert [step.to_dict() for step in restored.steps] == [step.to_dict() for step in trace.steps]


def test_trace_takes_the_world_from_the_instance(env):
    """The world block comes from the configured environment, not class defaults.

    Purpose: A run may be configured away from the defaults, and a viewer that
    drew the defaults would draw a different valley from the one the planner
    solved.

    Test type: unit
    """
    world = build_mountain_car_trace(env, _episode(), 0).payload["world"]

    assert world["min_position"] == pytest.approx(env.min_position)
    assert world["max_position"] == pytest.approx(env.max_position)
    assert world["max_speed"] == pytest.approx(env.max_speed)
    assert world["goal_position"] == pytest.approx(env.goal_position)
    assert world["power"] == pytest.approx(env.power)
    assert world["gravity"] == pytest.approx(env.gravity)
    assert world["actions"] == list(env.actions)
    assert np.allclose(world["observation_cov"], np.asarray(env.cov_matrix))
    assert np.allclose(world["state_transition_cov"], np.asarray(env.state_transition_cov))


def test_transition_noise_is_reported_at_the_configured_width(tmp_path: Path):
    """A run with non-default transition noise says so in its trace.

    Purpose: ``state_transition_cov`` is a constructor argument, so a viewer
    that quoted ``DEFAULT_STATE_TRANSITION_COV`` would misreport how noisy the
    episode it is drawing actually was.

    Test type: unit
    """
    noisy = MountainCarPOMDP(discount_factor=0.99, state_transition_cov=np.diag([1e-3, 1e-4]))

    world = build_mountain_car_trace(noisy, _episode(), 0).payload["world"]

    assert np.allclose(world["state_transition_cov"], [[1e-3, 0.0], [0.0, 1e-4]])


def test_hill_matches_the_gif_renderer(env):
    """The payload's valley is the one the GIF draws.

    Purpose: The hill is the only part of this world that is not a constructor
    argument — the GIF renderer carries it as a literal expression, and the
    payload carries it as three numbers. If those two drift apart, the browser
    and the GIF show the same episode on different ground, and neither one
    announces it.

    Test type: unit
    """
    hill = build_mountain_car_trace(env, _episode(), 0).payload["world"]["hill"]

    for position in np.linspace(env.min_position, env.max_position, 25):
        expected = MountainCarVisualizer.hill_height(float(position))
        assert hill_height(float(position)) == pytest.approx(expected)
        assert (
            hill["amplitude"] * np.sin(hill["frequency"] * position) + hill["offset"]
        ) == pytest.approx(expected)


def test_states_observations_and_beliefs_line_up_step_for_step(env):
    """Every recorded step contributes one entry to each payload list.

    Purpose: The viewer indexes these lists by step. A list that is one short
    would draw the wrong step's belief beside the right step's position for the
    whole rest of the episode, which looks like a planner mistake rather than
    an exporter one.

    Test type: unit
    """
    episode = _episode()

    payload = build_mountain_car_trace(env, episode, 0).payload

    assert len(payload["states"]) == len(episode)
    assert len(payload["next_states"]) == len(episode)
    assert len(payload["observations"]) == len(episode)
    assert len(payload["beliefs"]) == len(episode)
    assert payload["states"][0] == pytest.approx([-0.50, 0.0])
    assert payload["beliefs"][0]["kind"] == "particles"


def test_terminal_step_records_absence_rather_than_repetition(env):
    """The final step's missing observation is written as missing.

    Purpose: The terminal bookkeeping step carries a state but no action,
    reward or reading. Repeating the previous reading there would show the
    viewer a measurement the episode never took.

    Test type: unit
    """
    payload = build_mountain_car_trace(env, _episode(), 0).payload

    assert payload["observations"][-1] is None
    assert payload["next_states"][-1] is None
    assert payload["states"][-1] == pytest.approx([0.51, 0.030])


def test_outcome_is_read_from_the_final_state(env):
    """An episode that reached the goal is recorded as terminal.

    Purpose: The site counts outcomes across a run, so an episode that solved
    the problem and is filed as "ran out of steps" moves a number a reader
    will quote.

    Test type: unit
    """
    reached = build_mountain_car_trace(env, _episode(), 0)
    # Same episode, stopped short of the hilltop.
    short = _episode()
    short[-1] = _step((-0.2, 0.01), None, None, None, short[-1].belief)

    assert reached.reach_terminal_state is True
    assert build_mountain_car_trace(env, short, 0).reach_terminal_state is False


def test_a_state_of_the_wrong_width_is_refused(env):
    """A row that is not (position, velocity) raises rather than being drawn.

    Purpose: The viewer reads these rows by index. A three-component row would
    be written without complaint and drawn as a car at a position the episode
    never held, which is the one thing a recorded episode exists to rule out.

    Test type: unit
    """
    episode = _episode()
    episode[1] = _step((0.1, 0.2, 0.3), NEUTRAL, (0.1, 0.2), -1.0, episode[1].belief)

    with pytest.raises(ValueError, match="need 2 components"):
        build_mountain_car_trace(env, episode, 0)


def test_an_empty_history_is_refused(env):
    """There is no trace for an episode with no steps.

    Purpose: An empty payload would reach the viewer as a scene with nothing in
    it and no explanation, rather than as the recording failure it is.

    Test type: unit
    """
    with pytest.raises(ValueError, match="empty history"):
        build_mountain_car_trace(env, [], 0)


def test_the_environment_hook_produces_the_same_trace(env):
    """``build_episode_trace`` is the exporter, not a second implementation.

    Purpose: The episode loop calls the hook, and the tests above call the
    function. If those ever diverge, everything above is testing code nothing
    runs.

    Test type: unit
    """
    episode = _episode()

    hooked = env.build_episode_trace(episode, episode_index=5, policy_name="POMCPOW")
    direct = build_mountain_car_trace(env, episode, episode_index=5, policy_name="POMCPOW")

    assert hooked.to_dict() == direct.to_dict()
