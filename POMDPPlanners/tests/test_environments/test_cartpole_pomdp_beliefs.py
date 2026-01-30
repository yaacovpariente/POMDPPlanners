"""Tests for CartPoleVectorizedUpdater.

This module tests the vectorized batch transition and observation
log-likelihood methods, including an equivalence test that verifies
the vectorized results match the per-particle loop.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.cartpole_pomdp_beliefs import (
    CartPoleVectorizedUpdater,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
    return CartPolePOMDP(discount_factor=0.99, noise_cov=noise_cov)


@pytest.fixture
def updater(env):
    return CartPoleVectorizedUpdater.from_environment(env)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFromEnvironment:
    def test_from_environment_creates_updater(self, env):
        """Test that from_environment constructs a valid updater.

        Purpose: Validates the factory classmethod.

        Given: A CartPolePOMDP instance.
        When: from_environment is called.
        Then: A CartPoleVectorizedUpdater is returned with matching parameters.

        Test type: unit
        """
        updater = CartPoleVectorizedUpdater.from_environment(env)
        assert isinstance(updater, CartPoleVectorizedUpdater)
        assert updater.force_mag == env.force_mag
        assert updater.gravity == env.gravity
        assert updater.masscart == env.masscart
        assert updater.masspole == env.masspole
        assert updater.total_mass == env.total_mass
        assert updater.length == env.length
        assert updater.polemass_length == env.polemass_length
        assert updater.tau == env.tau
        assert updater.kinematics_integrator == env.kinematics_integrator


# ---------------------------------------------------------------------------
# batch_transition tests
# ---------------------------------------------------------------------------


class TestBatchTransition:
    def test_output_shape(self, updater):
        """Test that batch_transition returns correct shape.

        Purpose: Validates output shape of batch_transition.

        Given: 30 particles of dimension 4.
        When: batch_transition is called with action 1.
        Then: Result has shape (30, 4).

        Test type: unit
        """
        np.random.seed(0)
        particles = np.random.uniform(-0.05, 0.05, (30, 4))
        result = updater.batch_transition(particles, action=1)
        assert result.shape == (30, 4)

    def test_deterministic_transition(self, updater):
        """Test that batch_transition is deterministic.

        Purpose: Validates that repeated calls with the same input give
                 identical output (CartPole has no process noise).

        Given: A set of particles and an action.
        When: batch_transition is called twice with the same inputs.
        Then: Both results are identical.

        Test type: unit
        """
        particles = np.array([[0.0, 0.0, 0.1, 0.0], [0.01, -0.02, 0.05, 0.1]])
        result_a = updater.batch_transition(particles, action=1)
        result_b = updater.batch_transition(particles, action=1)
        np.testing.assert_array_equal(result_a, result_b)

    def test_physics_correctness_single_particle(self, updater):
        """Test that vectorized physics matches hand-computed values.

        Purpose: Validates the physics equations for a single particle.

        Given: A single particle at [0, 0, 0.1, 0] and action=1 (right force).
        When: batch_transition is called.
        Then: The result matches the expected Euler integration result.

        Test type: unit
        """
        import math

        state = np.array([[0.0, 0.0, 0.1, 0.0]])
        result = updater.batch_transition(state, action=1)

        # Hand-compute expected values
        force = updater.force_mag  # 10.0
        theta = 0.1
        theta_dot = 0.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        temp = (force + updater.polemass_length * theta_dot**2 * sin_t) / updater.total_mass
        theta_acc = (updater.gravity * sin_t - cos_t * temp) / (
            updater.length * (4.0 / 3.0 - updater.masspole * cos_t**2 / updater.total_mass)
        )
        x_acc = temp - updater.polemass_length * theta_acc * cos_t / updater.total_mass

        expected = np.array(
            [
                0.0 + updater.tau * 0.0,  # x
                0.0 + updater.tau * x_acc,  # x_dot
                0.1 + updater.tau * 0.0,  # theta
                0.0 + updater.tau * theta_acc,  # theta_dot
            ]
        )
        np.testing.assert_allclose(result[0], expected, atol=1e-12)


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
        particles = np.random.uniform(-0.05, 0.05, (20, 4))
        obs = np.array([0.0, 0.0, 0.0, 0.0])
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
        particles = np.random.uniform(-0.05, 0.05, (20, 4))
        obs = np.array([0.01, 0.0, 0.02, 0.0])
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
        obs = np.array([0.0, 0.0, 0.0, 0.0])
        close_particle = np.array([[0.001, 0.0, 0.001, 0.0]])
        far_particle = np.array([[1.0, 1.0, 1.0, 1.0]])

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

        Given: Two updaters with different force_mag.
        When: config_id is computed for both.
        Then: The IDs differ.

        Test type: unit
        """
        u1 = CartPoleVectorizedUpdater.from_environment(env)
        u2 = CartPoleVectorizedUpdater(
            obs_dist=env._obs_dist,
            force_mag=env.force_mag * 2,
            gravity=env.gravity,
            masscart=env.masscart,
            masspole=env.masspole,
            total_mass=env.total_mass,
            length=env.length,
            polemass_length=env.polemass_length,
            tau=env.tau,
            kinematics_integrator=env.kinematics_integrator,
        )
        assert u1.config_id != u2.config_id


# ---------------------------------------------------------------------------
# Equivalence test: vectorized vs per-particle loop
# ---------------------------------------------------------------------------


class TestEquivalenceWithPerParticleLoop:
    def test_batch_transition_matches_per_particle_loop(self, env, updater):
        """Test vectorized batch_transition matches per-particle state_transition_model.

        Purpose: Verifies that batch_transition produces the same results as calling
                 the environment's state_transition_model per particle.

        Given: A set of particles, an action, and a fixed random seed.
        When: batch_transition is called, and the same transitions are computed
              per-particle using the environment's state_transition_model.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(123)
        n = 50
        particles = np.random.uniform(-0.05, 0.05, (n, 4))
        action = 1

        vectorized_result = updater.batch_transition(particles, action)

        per_particle_result = np.empty_like(particles)
        for i in range(n):
            next_state = env.state_transition_model(state=particles[i], action=action).sample()[0]
            per_particle_result[i] = next_state

        np.testing.assert_allclose(vectorized_result, per_particle_result, atol=1e-10)

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
        n = 50
        action = 1
        observation = np.array([0.01, 0.0, 0.02, 0.0])
        particles = np.random.uniform(-0.05, 0.05, (n, 4))

        vectorized_ll = updater.batch_observation_log_likelihood(particles, action, observation)

        per_particle_ll = np.empty(n)
        for i in range(n):
            obs_model = env.observation_model(next_state=particles[i], action=action)
            prob = obs_model.probability([observation])[0]
            per_particle_ll[i] = np.log(prob)

        np.testing.assert_allclose(vectorized_ll, per_particle_ll, atol=1e-10)
