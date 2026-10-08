# SPDX-License-Identifier: MIT

"""Tests for the CartPole episode trace and its scene module.

A CartPole episode must survive the round trip to JSON and back with the
numbers a viewer reads by index still in the right places, and the scene module
must sit where the payload kind says.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from POMDPPlanners.core.belief import GaussianBelief, WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.traces import EpisodeTrace
from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp_gaussian_beliefs import (
    GaussianBeliefUpdaterType,
    create_cartpole_gaussian_belief,
)
from POMDPPlanners.environments.cartpole_pomdp.cartpole_visualization.cartpole_visualizer import (
    CARTPOLE_PAYLOAD_KIND,
    CartPoleVisualizer,
)
from POMDPPlanners.reporting.scenes import scene_script_url
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import cartpole_pinned_kwargs

SCENE_MODULE = CartPoleVisualizer.scene_script()


@pytest.fixture(name="env")
def env_fixture():
    """A CartPole environment with the suite's pinned configuration."""
    return CartPolePOMDP(
        discount_factor=0.95, noise_cov=np.eye(4) * 0.1, **cartpole_pinned_kwargs()
    )


def _particle_belief(states):
    particles = [np.asarray(s, dtype=float) for s in states]
    return WeightedParticleBelief(
        particles=particles,
        log_weights=np.log(np.ones(len(particles)) / len(particles)),
    )


def _episode(env, length: int = 4, belief=None):
    """A short episode whose last record is the terminal bookkeeping step."""
    history = []
    for step in range(length):
        state = np.array([0.1 * step, 0.2, 0.03 * step, -0.1])
        is_last = step == length - 1
        history.append(
            StepData(
                state=state,
                action=None if is_last else step % 2,
                next_state=None if is_last else state + 0.01,
                observation=None if is_last else state + 0.05,
                reward=None if is_last else 1.0,
                belief=belief if belief is not None else _particle_belief([state, state + 0.02]),
                info=None if is_last else {"upright": 1.0},
            )
        )
    return history


def test_states_survive_the_round_trip(env, tmp_path: Path):
    """Every recorded state comes back with its four components in order.

    Purpose: The scene module reads the state by index — position 0 places the
    cart, position 2 rotates the pole — so a reordered or truncated row would
    draw a pole at an angle the episode never held, and would look plausible.

    Given: A four-step episode.
    When: The trace is written and read back.
    Then: The states and observations match the recorded ones exactly.
    """
    history = _episode(env, 4)
    written = EpisodeTrace.read(
        CartPoleVisualizer(env).build_trace(history, episode_index=0).write(tmp_path / "trace.json")
    )

    assert written.payload_kind == CARTPOLE_PAYLOAD_KIND
    assert len(written.payload["states"]) == 4
    for step, row in zip(history, written.payload["states"]):
        assert row == pytest.approx(np.asarray(step.state, dtype=float).tolist())
    # The terminal step genuinely has no observation, and None says so rather
    # than repeating the previous reading.
    assert written.payload["observations"][-1] is None
    assert written.payload["next_states"][-1] is None


def test_world_block_comes_from_the_instance(env, tmp_path: Path):
    """The rig is described by the environment that ran, not by class defaults.

    Purpose: A configured run may move a threshold, and a viewer that drew the
    default would place the limit posts where the episode did not end.

    Given: An environment whose cart limit has been moved.
    When: The trace is built.
    Then: The payload's world reports the instance's value.
    """
    env.x_threshold = 1.75
    world = CartPoleVisualizer(env).build_trace(_episode(env, 2), 0).payload["world"]

    assert world["x_threshold"] == 1.75
    assert world["theta_threshold_radians"] == pytest.approx(env.theta_threshold_radians)
    assert world["length"] == pytest.approx(env.length)
    assert world["force_mag"] == pytest.approx(env.force_mag)
    assert np.asarray(world["noise_cov"]).shape == (4, 4)


def test_gaussian_belief_is_written_as_a_gaussian(env, tmp_path: Path):
    """A Gaussian belief reaches the viewer as a mean and a full covariance.

    Purpose: CartPole is filtered with an EKF or UKF, and the scene draws its
    fan of ghost poles from the [x, theta] marginal — the 2x2 sub-block at rows
    and columns 0 and 2. That needs the whole 4x4, not a diagonal summary.

    Given: An episode recorded with a Gaussian belief.
    When: The trace is built.
    Then: Each step carries a gaussian payload with a 4-vector mean and a 4x4
        covariance, and it survives JSON.
    """
    belief = create_cartpole_gaussian_belief(env, GaussianBeliefUpdaterType.UKF)
    assert isinstance(belief, GaussianBelief)

    trace = CartPoleVisualizer(env).build_trace(_episode(env, 3, belief=belief), 0)
    restored = EpisodeTrace.read(trace.write(tmp_path / "trace.json"))

    for payload in restored.payload["beliefs"]:
        assert payload["kind"] == "gaussian"
        assert len(payload["mean"]) == 4
        assert np.asarray(payload["covariance"]).shape == (4, 4)


def test_particle_belief_keeps_four_component_particles(env):
    """Particle beliefs arrive as four-component states, not flattened.

    Purpose: The scene reads each particle's cart position and pole angle by
    index, exactly as it reads the true state.

    Given: An episode recorded with a weighted particle belief.
    When: The trace is built.
    Then: The particles are four-component rows with normalized weights.
    """
    payload = CartPoleVisualizer(env).build_trace(_episode(env, 2), 0).payload["beliefs"][0]

    assert payload["kind"] == "particles"
    assert all(len(particle) == 4 for particle in payload["particles"])
    assert sum(payload["weights"]) == pytest.approx(1.0)


def test_environment_writes_a_trace_file(env, tmp_path: Path):
    """The environment's visualizer writes a readable file under the episode's index."""
    written = env.episode_visualizer().write(
        history=_episode(env, 3), output_dir=tmp_path, episode_index=2, policy_name="PFT_DPW"
    )
    assert written == tmp_path / "trace_2.json"

    restored = EpisodeTrace.read(written)
    assert restored.payload_kind == CARTPOLE_PAYLOAD_KIND
    assert restored.policy == "PFT_DPW"
    assert restored.episode_index == 2


def test_rejects_an_empty_history(env):
    """There is no trace for an episode with no steps."""
    with pytest.raises(ValueError, match="empty history"):
        CartPoleVisualizer(env).build_trace([], 0)


def test_rejects_a_state_of_the_wrong_width(env):
    """A row that is not four components is refused rather than written.

    Purpose: The viewer reads the pole angle from position 2. A three-component
    row would silently supply something else, and the drawing would look fine.
    """
    history = [
        StepData(
            state=np.array([0.0, 0.0, 0.0]),
            action=1,
            next_state=None,
            observation=None,
            reward=1.0,
            belief=cast(Any, None),
        )
    ]
    with pytest.raises(ValueError, match="4 components"):
        CartPoleVisualizer(env).build_trace(history, 0)


def test_scene_module_is_where_the_payload_kind_says(env):
    """The scene module sits at the path the kind resolves to, and claims it.

    Purpose: Scene modules are resolved by convention rather than a table, so
    a payload kind and a file name that disagree fail only in the browser,
    where nothing in the test suite would see it.
    """
    assert scene_script_url(CARTPOLE_PAYLOAD_KIND) == "/static/scenes/cartpole.js"
    assert SCENE_MODULE.is_file()
    assert 'V.scenes["cartpole.v1"]' in SCENE_MODULE.read_text(encoding="utf-8")
