# SPDX-License-Identifier: MIT

"""Tests for the Safety Ant Velocity trace exporter and its visualizer package.

Two things are checked here that nothing else checks:

* the payload a browser viewer reads survives a round trip through JSON, and
  carries the episode's own belief rather than a cloud rebuilt from the truth;
* moving the GIF renderer into ``visualizer/`` did not move a pixel. The golden
  hash comparison in the suite's golden-file module only runs inside the
  project's Docker image, so the check here is against the committed golden
  file itself, which is what that comparison would use.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.safety_ant_velocity_pomdp import SafeAntVelocityPOMDP
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_visualization.trace_exporter import (
    SAFETY_ANT_VELOCITY_PAYLOAD_KIND,
    build_safety_ant_velocity_trace,
)
from POMDPPlanners.tests.test_environments.test_environment_visualizations_golden_files import (
    GOLDEN_DIR,
    GOLDEN_VISUALIZATIONS,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import safety_ant_velocity_pinned_kwargs


@pytest.fixture(name="env")
def env_fixture():
    """A Safety Ant environment with the suite's pinned configuration."""
    return SafeAntVelocityPOMDP(discount_factor=0.95, **safety_ant_velocity_pinned_kwargs())


def _belief(states):
    particles = [np.asarray(s, dtype=float) for s in states]
    return WeightedParticleBelief(
        particles=particles,
        log_weights=np.log(np.ones(len(particles)) / len(particles)),
    )


def _episode(env, length: int = 4):
    """One episode built with the environment's own transition and observation."""
    np.random.seed(11)
    state = np.array([0.2, -0.3, 0.0, 0.0])
    history = []
    for step in range(length):
        is_last = step == length - 1
        if is_last:
            history.append(
                StepData(
                    state=state,
                    action=None,
                    next_state=None,
                    observation=None,
                    reward=None,
                    belief=_belief([state, state + 0.05]),
                )
            )
            break
        action = 3
        next_state, observation, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=reward,
                belief=_belief([state, state + 0.05]),
            )
        )
        state = next_state
    return history


def test_payload_round_trips_through_json(env, tmp_path: Path):
    """The payload a viewer reads survives being written and read back.

    Purpose: The trace file is the whole interface between this environment and
    the browser viewer, so a field that does not survive the round trip is a
    field the viewer silently never sees.

    Given: A trace built from a four-step episode.
    When: It is written to disk and read back.
    Then: The payload, the envelope and the payload kind all match.
    """
    original = build_safety_ant_velocity_trace(
        environment=env, history=_episode(env), episode_index=2, policy_name="PFT_DPW"
    )
    restored = EpisodeTrace.read(original.write(tmp_path / "trace.json"))

    assert restored.payload_kind == SAFETY_ANT_VELOCITY_PAYLOAD_KIND
    assert restored.payload == original.payload
    assert restored.steps == original.steps
    assert restored.policy == "PFT_DPW"
    assert restored.episode_index == 2


def test_world_block_comes_from_the_instance_the_episode_ran_on():
    """A configured run is described by its own constants, not the defaults.

    Purpose: The viewer draws the safe ring from this block. If it were the
    class defaults, a run with a different threshold would be drawn with a ring
    it never had, and the picture would silently disagree with the reward.

    Given: An environment configured well away from every default.
    When: Its trace is built.
    Then: The world block carries the configured values, and the critical
        threshold is the configured one scaled by the environment's margin.
    """
    env = SafeAntVelocityPOMDP(
        discount_factor=0.9,
        **safety_ant_velocity_pinned_kwargs(safe_velocity_threshold=0.75, max_force=2.5),
    )
    world = build_safety_ant_velocity_trace(
        environment=env, history=_episode(env), episode_index=0
    ).payload["world"]

    assert world["safe_velocity_threshold"] == pytest.approx(0.75)
    assert world["critical_velocity_threshold"] == pytest.approx(0.75 * 1.5)
    assert world["max_force"] == pytest.approx(2.5)
    # The viewer names the force an action commands from these two together.
    assert world["force_scales"] == [0.0, 0.33, 0.67, 1.0]
    assert world["actions"] == env.get_actions()
    # The ring the viewer draws is the speed is_terminal actually ends above.
    assert env.is_terminal(np.array([0.0, 0.0, world["critical_velocity_threshold"] + 1e-6, 0.0]))


def test_applied_force_matches_the_action_that_was_taken(env):
    """The force recovered from two recorded states is the one the action commands.

    Purpose: The environment samples the force direction inside its C++ kernel
    and records it nowhere, so the viewer's force arrow can only come from
    inverting the transition. If that inversion were wrong the arrow would
    point somewhere the episode never pushed, and nothing else would notice.

    Given: An episode whose every step takes action 3, the full-force action.
    When: The trace is built.
    Then: Every recovered force has magnitude ``max_force``, and the terminal
        step, which has no transition to invert, records none.
    """
    trace = build_safety_ant_velocity_trace(
        environment=env, history=_episode(env, length=5), episode_index=0
    )
    forces = trace.payload["applied_forces"]

    assert forces[-1] is None
    for force in forces[:-1]:
        assert np.hypot(force[0], force[1]) == pytest.approx(env.max_force, rel=1e-9)


def test_trace_carries_the_episodes_own_belief(env):
    """The belief written is the episode's own particles, not a rebuilt cloud.

    Purpose: The belief is the field this trace exists to carry and the one
    most easily faked — a cloud regenerated from the true state looks
    convincing and says nothing.

    Given: An episode whose beliefs are particles at known coordinates.
    When: The trace is built.
    Then: Those coordinates come back, as four-number states, with weights that
        sum to one.
    """
    history = _episode(env)
    trace = build_safety_ant_velocity_trace(environment=env, history=history, episode_index=0)
    belief = trace.payload["beliefs"][0]

    assert belief["kind"] == "particles"
    assert len(belief["particles"][0]) == 4
    assert belief["particles"][0] == pytest.approx(list(history[0].belief.particles[0]))
    assert sum(belief["weights"]) == pytest.approx(1.0)


def test_empty_history_is_refused(env):
    """There is no episode to write, so nothing is written."""
    with pytest.raises(ValueError, match="empty history"):
        build_safety_ant_velocity_trace(environment=env, history=[], episode_index=0)


def test_the_environment_writes_its_trace_through_cache_trace(env, tmp_path: Path):
    """The environment's own hook writes the file the reporting site looks for.

    Purpose: The exporter is only reachable in a real run through
    ``cache_trace``. A trace exporter nothing calls is a trace nobody gets.

    Given: A four-step episode.
    When: cache_trace is called with an output directory.
    Then: ``trace_<index>.json`` appears and reads back as this environment's
        trace.
    """
    written = env.cache_trace(
        history=_episode(env), output_dir=tmp_path, episode_index=1, policy_name="POMCPOW"
    )

    assert written == tmp_path / "trace_1.json"
    assert EpisodeTrace.read(written).payload_kind == SAFETY_ANT_VELOCITY_PAYLOAD_KIND


def test_moving_the_renderer_left_the_gif_byte_identical(tmp_path: Path):
    """The visualizer package is a move: the GIF's bytes did not change.

    Purpose: The renderer's output is pinned by a golden hash, and that
    comparison only runs inside the project's Docker image. This migration
    touched the renderer's imports, so the one thing that must be proved
    outside Docker is that the bytes are the same ones the golden file holds.

    Given: The golden-file suite's own deterministic Safety Ant episode.
    When: It is rendered by the moved visualizer.
    Then: The output hashes to the committed golden file.
    """
    spec = next(s for s in GOLDEN_VISUALIZATIONS if s.name == "safety_ant_velocity")
    output = tmp_path / "safety_ant_velocity.gif"
    spec.build_renderer()(spec.build_history(), output)

    golden = GOLDEN_DIR / spec.golden_file
    assert hashlib.sha256(output.read_bytes()).hexdigest() == (
        hashlib.sha256(golden.read_bytes()).hexdigest()
    )
