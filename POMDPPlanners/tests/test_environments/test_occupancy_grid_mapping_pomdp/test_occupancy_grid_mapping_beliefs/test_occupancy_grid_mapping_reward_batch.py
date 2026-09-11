# SPDX-License-Identifier: MIT

"""Tests for the batched reward of the occupancy-grid mapping POMDP.

``reward_batch`` is what a belief-space planner calls for the expected reward
of every particle, so it must return the scalar ``reward`` values, with and
without a realised successor.
"""

import pickle

import numpy as np
import pytest

from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import RangeNoiseModel
from POMDPPlanners.tests.test_environments.test_occupancy_grid_mapping_pomdp.test_occupancy_grid_mapping_beliefs.test_occupancy_grid_mapping_vectorized_updater import (  # noqa: E501
    ACTIONS,
    RANGE_NOISE_MODELS,
    diverse_particles,
    make_env,
)


@pytest.mark.parametrize("range_noise_model", RANGE_NOISE_MODELS)
@pytest.mark.parametrize("move_failure_probability", [0.0, 0.25])
@pytest.mark.parametrize("action", ACTIONS)
def test_expected_reward_matches_scalar_reward(action, move_failure_probability, range_noise_model):
    """Purpose: the planner's expected reward must not change with the batching.

    Given: Particles at mixed poses with partly resolved maps, under each
        range law with noise wide enough for the truncation to matter.
    When: ``reward_batch`` without successors and the scalar ``reward`` are
        evaluated for the same action.
    Then: They agree to floating-point precision for every particle.

    Test type: unit
    """
    env = make_env(
        move_failure_probability=move_failure_probability,
        step_cost=0.1,
        range_noise_std_cells=1.0,
        range_noise_model=range_noise_model,
    )
    particles = diverse_particles(env)
    batched = env.reward_batch(particles, action)
    scalar = np.array([env.reward(particle, action) for particle in particles])
    np.testing.assert_allclose(batched, scalar, rtol=1e-12, atol=1e-9)
    assert batched.shape == (len(particles),)


def test_realised_reward_matches_scalar_reward():
    """Purpose: the runner's realised reward path must batch to the same values.

    Test type: unit
    """
    env = make_env(step_cost=0.1)
    particles = diverse_particles(env)
    successors = np.asarray([env.sample_next_state(particle, 0) for particle in particles])
    batched = env.reward_batch(particles, 0, successors)
    scalar = np.array(
        [env.reward(particle, 0, successor) for particle, successor in zip(particles, successors)]
    )
    np.testing.assert_allclose(batched, scalar, rtol=1e-12, atol=1e-9)


def test_reward_batch_accepts_a_list_of_states():
    """Purpose: the scalar filter hands a list of particles, not an array.

    Test type: unit
    """
    env = make_env()
    particles = list(diverse_particles(env, n_particles=5))
    np.testing.assert_allclose(
        env.reward_batch(particles, 1), [env.reward(particle, 1) for particle in particles]
    )


def test_batched_kernels_follow_setting_changes_and_survive_pickling():
    """Purpose: a setting edited after construction must not leave stale
    kernels behind, and a worker's unpickled copy must rebuild them.

    Test type: unit
    """
    env = make_env()
    particles = diverse_particles(env, n_particles=4)
    before = env.reward_batch(particles, 0)
    env.step_cost = 0.5
    np.testing.assert_allclose(env.reward_batch(particles, 0), before - 0.5)
    env.move_failure_probability = 0.5
    np.testing.assert_allclose(
        env.reward_batch(particles, 0), [env.reward(particle, 0) for particle in particles]
    )
    env.range_noise_std_cells = 1.0
    gaussian = env.reward_batch(particles, 0)
    env.range_noise_model = RangeNoiseModel.TRUNCATED_NORMAL
    truncated = env.reward_batch(particles, 0)
    np.testing.assert_allclose(truncated, [env.reward(particle, 0) for particle in particles])
    assert not np.allclose(truncated, gaussian)
    restored = pickle.loads(pickle.dumps(env))
    assert "_batched_kernels_cache" not in restored.__dict__
    np.testing.assert_allclose(restored.reward_batch(particles, 0), env.reward_batch(particles, 0))
