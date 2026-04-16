"""Tests for MountainCarVectorizedUpdater.

This module tests the vectorized batch transition and observation
log-likelihood methods, including an equivalence test that verifies
the vectorized results match the per-particle loop.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp_beliefs import (
    MountainCarVectorizedUpdater,
)
from POMDPPlanners.tests.test_core.test_belief.vectorized_updater_test_utils import (
    assert_batch_obs_log_likelihood_matches_loop,
    assert_batch_transition_matches_loop,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    return MountainCarPOMDP(discount_factor=0.99)


@pytest.fixture
def updater(env):
    return MountainCarVectorizedUpdater.from_environment(env)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFromEnvironment:
    def test_from_environment_creates_updater(self, env):
        """Test that from_environment constructs a valid updater.

        Purpose: Validates the factory classmethod.

        Given: A MountainCarPOMDP instance.
        When: from_environment is called.
        Then: A MountainCarVectorizedUpdater is returned with matching parameters.

        Test type: unit
        """
        updater = MountainCarVectorizedUpdater.from_environment(env)
        assert isinstance(updater, MountainCarVectorizedUpdater)
        assert updater.power == env.power
        assert updater.gravity == env.gravity
        assert updater.max_speed == env.max_speed
        assert updater.min_position == env.min_position
        assert updater.max_position == env.max_position


# ---------------------------------------------------------------------------
# batch_transition tests
# ---------------------------------------------------------------------------


class TestBatchTransition:
    def test_output_shape(self, updater):
        """Test that batch_transition returns correct shape.

        Purpose: Validates output shape of batch_transition.

        Given: 30 particles of dimension 2.
        When: batch_transition is called with action 1.
        Then: Result has shape (30, 2).

        Test type: unit
        """
        np.random.seed(0)
        particles = np.column_stack(
            [
                np.random.uniform(-0.6, -0.4, 30),
                np.zeros(30),
            ]
        )
        result = updater.batch_transition(particles, action=1)
        assert result.shape == (30, 2)

    def test_deterministic_transition(self, updater):
        """Test that batch_transition is deterministic.

        Purpose: Validates that repeated calls with the same input give
                 identical output (Mountain Car has no process noise).

        Given: A set of particles and an action.
        When: batch_transition is called twice with the same inputs.
        Then: Both results are identical.

        Test type: unit
        """
        particles = np.array([[-0.5, 0.0], [-0.6, 0.01]])
        result_a = updater.batch_transition(particles, action=1)
        result_b = updater.batch_transition(particles, action=1)
        np.testing.assert_array_equal(result_a, result_b)

    def test_physics_correctness_single_particle(self, updater):
        """Test that vectorized physics matches hand-computed values.

        Purpose: Validates the physics equations for a single particle.

        Given: A single particle at [-0.5, 0.0] and action=1 (forward).
        When: batch_transition is called.
        Then: The result matches the expected physics computation.

        Test type: unit
        """
        state = np.array([[-0.5, 0.0]])
        result = updater.batch_transition(state, action=1)

        # Hand-compute expected
        v = 0.0 + 1 * updater.power + np.cos(3.0 * (-0.5)) * (-updater.gravity)
        v = np.clip(v, -updater.max_speed, updater.max_speed)
        p = -0.5 + v
        p = np.clip(p, updater.min_position, updater.max_position)

        np.testing.assert_allclose(result[0], [p, v], atol=1e-12)

    def test_position_clipping(self, updater):
        """Test that positions are clipped at boundaries.

        Purpose: Validates that min/max position boundaries are enforced.

        Given: A particle near the left boundary with negative velocity.
        When: batch_transition is called.
        Then: Position does not go below min_position.

        Test type: unit
        """
        particles = np.array([[-1.19, -0.05]])
        result = updater.batch_transition(particles, action=-1)
        assert result[0, 0] >= updater.min_position

    def test_velocity_zeroed_at_left_wall(self, updater):
        """Test that velocity is zeroed when hitting the left wall.

        Purpose: Validates the boundary condition at min_position.

        Given: A particle that will end up at min_position with negative velocity.
        When: batch_transition is called.
        Then: The velocity is zeroed at the left wall if it is negative.

        Test type: unit
        """
        # Start at min_position with large negative velocity
        particles = np.array([[updater.min_position, -0.05]])
        result = updater.batch_transition(particles, action=-1)
        # The position clips to min_position; if velocity < 0, it gets zeroed
        if result[0, 0] == updater.min_position:
            assert result[0, 1] >= 0.0


# ---------------------------------------------------------------------------
# batch_observation_log_likelihood tests
# ---------------------------------------------------------------------------


class TestBatchObservationLogLikelihood:
    def test_output_shape(self, updater):
        """Test that batch_observation_log_likelihood returns correct shape.

        Purpose: Validates output shape.

        Given: 20 particles.
        When: batch_observation_log_likelihood is called.
        Then: Result has shape (20,).

        Test type: unit
        """
        np.random.seed(0)
        particles = np.column_stack(
            [
                np.random.uniform(-0.6, -0.4, 20),
                np.zeros(20),
            ]
        )
        obs = np.array([-0.5, 0.0])
        result = updater.batch_observation_log_likelihood(particles, action=1, observation=obs)
        assert result.shape == (20,)

    def test_values_are_finite(self, updater):
        """Test that log-likelihoods are finite.

        Purpose: Validates that all returned values are finite numbers.

        Given: A set of particles and an observation.
        When: batch_observation_log_likelihood is called.
        Then: All returned values are finite.

        Test type: unit
        """
        np.random.seed(0)
        particles = np.column_stack(
            [
                np.random.uniform(-0.6, -0.4, 20),
                np.zeros(20),
            ]
        )
        obs = np.array([-0.5, 0.0])
        result = updater.batch_observation_log_likelihood(particles, action=0, observation=obs)
        assert np.all(np.isfinite(result))

    def test_closer_particles_higher_likelihood(self, updater):
        """Test that particles closer to observation have higher log-likelihood.

        Purpose: Validates that the Gaussian observation model assigns higher
                 likelihood to particles nearer the observation.

        Given: One particle at the observation and one far away.
        When: batch_observation_log_likelihood is called.
        Then: The close particle has higher log-likelihood.

        Test type: unit
        """
        obs = np.array([-0.5, 0.0])
        close_particle = np.array([[-0.5, 0.001]])
        far_particle = np.array([[0.3, 0.05]])

        ll_close = updater.batch_observation_log_likelihood(
            close_particle, action=0, observation=obs
        )
        ll_far = updater.batch_observation_log_likelihood(far_particle, action=0, observation=obs)
        assert ll_close[0] > ll_far[0]


# ---------------------------------------------------------------------------
# config_id tests
# ---------------------------------------------------------------------------


class TestConfigId:
    def test_config_id_deterministic(self, updater):
        """Test that config_id is deterministic.

        Purpose: Validates reproducibility of config_id.

        Given: An updater.
        When: config_id is called twice.
        Then: The same ID is returned.

        Test type: unit
        """
        assert updater.config_id == updater.config_id

    def test_config_id_differs_for_different_params(self, env):
        """Test that config_id changes when parameters differ.

        Purpose: Validates that different configurations produce different IDs.

        Given: Two updaters with different power values.
        When: config_id is computed for both.
        Then: The IDs differ.

        Test type: unit
        """
        u1 = MountainCarVectorizedUpdater.from_environment(env)
        u2 = MountainCarVectorizedUpdater(
            obs_dist=env._obs_dist,
            power=env.power * 2,
            gravity=env.gravity,
            max_speed=env.max_speed,
            min_position=env.min_position,
            max_position=env.max_position,
        )
        assert u1.config_id != u2.config_id


# ---------------------------------------------------------------------------
# Equivalence test: vectorized vs per-particle loop
# ---------------------------------------------------------------------------


class TestEquivalenceWithPerParticleLoop:
    def test_batch_transition_matches_per_particle_loop(self, env, updater):
        """Test vectorized batch_transition matches per-particle deterministic physics.

        Purpose: Verifies that batch_transition produces the same deterministic
                 next states as the environment's state_transition_model physics.

        Given: A set of particles, an action, and a fixed random seed.
        When: batch_transition is called, and the same deterministic transitions
              are computed per-particle using the environment's state_transition_model.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(123)
        particles = np.column_stack(
            [
                np.random.uniform(-0.6, -0.4, 50),
                np.random.uniform(-0.02, 0.02, 50),
            ]
        )

        def per_particle_fn(particle, action):
            transition = env.state_transition_model(state=tuple(particle), action=action)
            return transition._compute_deterministic_next_state()

        assert_batch_transition_matches_loop(
            updater=updater,
            particles=particles,
            action=1,
            per_particle_transition_fn=per_particle_fn,
        )

    def test_batch_observation_log_likelihood_matches_per_particle_loop(self, env, updater):
        """Test vectorized log-likelihood matches per-particle observation_model.probability.

        Purpose: Verifies that batch_observation_log_likelihood matches the
                 per-particle observation probability from the environment.

        Given: A set of next-state particles and an observation.
        When: batch_observation_log_likelihood is called, and per-particle
              log(observation_model.probability) is computed.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(42)
        particles = np.column_stack(
            [
                np.random.uniform(-0.6, -0.4, 50),
                np.random.uniform(-0.02, 0.02, 50),
            ]
        )
        observation = np.array([-0.5, 0.0])

        def per_particle_ll_fn(particle, action, obs):
            obs_model = env.observation_model(next_state=tuple(particle), action=action)
            return np.log(obs_model.probability([obs])[0])

        assert_batch_obs_log_likelihood_matches_loop(
            updater=updater,
            particles=particles,
            action=0,
            observation=observation,
            per_particle_ll_fn=per_particle_ll_fn,
        )
