"""Tests for SafetyAntVelocityVectorizedUpdater.

This module tests the vectorized batch transition and observation
log-likelihood methods, including an equivalence test that verifies
the vectorized results match the per-particle loop.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.safety_ant_velocity_pomdp import (
    SafeAntVelocityPOMDP,
)
from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp_beliefs import (
    SafetyAntVelocityVectorizedUpdater,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    return SafeAntVelocityPOMDP(discount_factor=0.99)


@pytest.fixture
def updater(env):
    return SafetyAntVelocityVectorizedUpdater.from_environment(env)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFromEnvironment:
    def test_from_environment_creates_updater(self, env):
        """Test that from_environment constructs a valid updater.

        Purpose: Validates the factory classmethod.

        Given: A SafeAntVelocityPOMDP instance.
        When: from_environment is called.
        Then: A SafetyAntVelocityVectorizedUpdater is returned with matching
              parameters.

        Test type: unit
        """
        updater = SafetyAntVelocityVectorizedUpdater.from_environment(env)
        assert isinstance(updater, SafetyAntVelocityVectorizedUpdater)
        assert updater.dt == env.dt
        assert updater.mass == env.mass
        assert updater.damping == env.damping
        assert updater.max_force == env.max_force

    def test_from_environment_obs_dist_covariance(self, env):
        """Test that from_environment builds the correct observation covariance.

        Purpose: Validates that the constructed observation distribution has the
                 expected diagonal covariance from position_noise and velocity_noise.

        Given: A SafeAntVelocityPOMDP with default noise parameters.
        When: from_environment is called.
        Then: The obs_dist covariance matches diag([pos², pos², vel², vel²]).

        Test type: unit
        """
        updater = SafetyAntVelocityVectorizedUpdater.from_environment(env)
        expected_cov = np.diag(
            [
                env.position_noise**2,
                env.position_noise**2,
                env.velocity_noise**2,
                env.velocity_noise**2,
            ]
        )
        np.testing.assert_allclose(updater.obs_dist.covariance, expected_cov)


# ---------------------------------------------------------------------------
# batch_transition tests
# ---------------------------------------------------------------------------


class TestBatchTransition:
    def test_output_shape(self, updater):
        """Test that batch_transition returns correct shape.

        Purpose: Validates output shape of batch_transition.

        Given: 30 particles of dimension 4.
        When: batch_transition is called with action 2.
        Then: Result has shape (30, 4).

        Test type: unit
        """
        np.random.seed(0)
        particles = np.column_stack(
            [
                np.random.uniform(-1, 1, (30, 2)),
                np.zeros((30, 2)),
            ]
        )
        result = updater.batch_transition(particles, action=2)
        assert result.shape == (30, 4)

    def test_zero_force_action_is_damping_only(self, updater):
        """Test that action=0 produces a damping-only transition.

        Purpose: Validates that when force magnitude is zero, only damping
                 affects the dynamics.

        Given: A particle with non-zero velocity and action=0.
        When: batch_transition is called.
        Then: The velocity is reduced by damping only, with no random force.

        Test type: unit
        """
        np.random.seed(0)
        velocity = np.array([1.0, 0.5])
        particles = np.array([[0.0, 0.0, velocity[0], velocity[1]]])
        result = updater.batch_transition(particles, action=0)

        # With zero force: acceleration = -damping * velocity / mass
        acc = -updater.damping * velocity / updater.mass
        expected_vel = velocity + acc * updater.dt
        expected_pos = particles[0, :2] + expected_vel * updater.dt

        np.testing.assert_allclose(result[0, :2], expected_pos, atol=1e-12)
        np.testing.assert_allclose(result[0, 2:4], expected_vel, atol=1e-12)

    def test_stochastic_transition_varies(self, updater):
        """Test that non-zero force actions produce varying transitions.

        Purpose: Validates that the random force direction leads to different
                 next states for identical particles.

        Given: Identical particles with a non-zero force action.
        When: batch_transition is called.
        Then: The resulting particles are not all identical.

        Test type: unit
        """
        np.random.seed(42)
        n = 100
        particles = np.tile([0.0, 0.0, 0.0, 0.0], (n, 1))
        result = updater.batch_transition(particles, action=3)
        # With random force directions, not all particles should be the same
        assert not np.allclose(result[0], result[1])


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
                np.random.uniform(-1, 1, (20, 2)),
                np.zeros((20, 2)),
            ]
        )
        obs = np.array([0.0, 0.0, 0.0, 0.0])
        result = updater.batch_observation_log_likelihood(particles, action=0, observation=obs)
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
                np.random.uniform(-1, 1, (20, 2)),
                np.random.uniform(-0.5, 0.5, (20, 2)),
            ]
        )
        obs = np.array([0.1, -0.1, 0.2, 0.0])
        result = updater.batch_observation_log_likelihood(particles, action=1, observation=obs)
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
        close_particle = np.array([[0.01, -0.01, 0.0, 0.0]])
        far_particle = np.array([[5.0, 5.0, 3.0, 3.0]])

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

        Given: Two updaters with different max_force.
        When: config_id is computed for both.
        Then: The IDs differ.

        Test type: unit
        """
        u1 = SafetyAntVelocityVectorizedUpdater.from_environment(env)
        cov = np.diag(
            [
                env.position_noise**2,
                env.position_noise**2,
                env.velocity_noise**2,
                env.velocity_noise**2,
            ]
        )
        from POMDPPlanners.utils.multivariate_normal import (
            CovarianceParameterizedMultivariateNormal,
        )

        obs_dist = CovarianceParameterizedMultivariateNormal(cov)
        u2 = SafetyAntVelocityVectorizedUpdater(
            obs_dist=obs_dist,
            dt=env.dt,
            mass=env.mass,
            damping=env.damping,
            max_force=env.max_force * 2,
            force_scales=np.array([0.0, 0.33, 0.67, 1.0]),
        )
        assert u1.config_id != u2.config_id


# ---------------------------------------------------------------------------
# Equivalence test: vectorized vs per-particle loop
# ---------------------------------------------------------------------------


class TestEquivalenceWithPerParticleLoop:
    def test_batch_transition_matches_per_particle_loop(self, env, updater):
        """Test vectorized batch_transition matches per-particle state_transition_model.

        Purpose: Verifies that batch_transition produces the same results as calling
                 the environment's state_transition_model per particle with the same
                 random seed.

        Given: A set of particles, an action, and a fixed random seed.
        When: batch_transition is called, and the same transitions are computed
              per-particle using the environment's state_transition_model.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(123)
        n = 50
        particles = np.column_stack(
            [
                np.random.uniform(-1, 1, (n, 2)),
                np.random.uniform(-0.5, 0.5, (n, 2)),
            ]
        )
        action = 2

        # Vectorized path
        np.random.seed(999)
        vectorized_result = updater.batch_transition(particles, action)

        # Per-particle path: same random seed, same noise sequence
        np.random.seed(999)
        per_particle_result = np.empty_like(particles)
        for i in range(n):
            next_state = env.state_transition_model(state=particles[i], action=action).sample()[0]
            per_particle_result[i] = next_state

        np.testing.assert_allclose(vectorized_result, per_particle_result, atol=1e-10)

    def test_batch_observation_log_likelihood_matches_per_particle_loop(self, env, updater):
        """Test vectorized log-likelihood matches per-particle observation probability up to constant.

        Purpose: Verifies that batch_observation_log_likelihood produces
                 log-likelihoods that differ from the per-particle
                 log(observation_model.probability) only by a constant offset.
                 The environment's probability() omits the Gaussian normalisation
                 constant, while the vectorized updater uses the fully normalised
                 log_pdf.  Relative differences between particles must still match.

        Given: A set of next-state particles and an observation.
        When: batch_observation_log_likelihood is called, and per-particle
              log(observation_model.probability) is computed.
        Then: The pairwise differences between log-likelihoods match within
              floating-point tolerance.

        Test type: integration
        """
        np.random.seed(42)
        n = 50
        action = 1
        observation = np.array([0.1, -0.1, 0.5, 0.2])
        particles = np.column_stack(
            [
                np.random.uniform(-1, 1, (n, 2)),
                np.random.uniform(-0.5, 0.5, (n, 2)),
            ]
        )

        vectorized_ll = updater.batch_observation_log_likelihood(particles, action, observation)

        per_particle_ll = np.empty(n)
        for i in range(n):
            obs_model = env.observation_model(next_state=particles[i], action=action)
            prob = obs_model.probability([observation])[0]
            per_particle_ll[i] = np.log(prob)

        # The environment's probability() omits the Gaussian normalisation
        # constant, so the two arrays differ by a constant offset.  Compare
        # pairwise differences to verify the relative structure is identical.
        vectorized_diff = vectorized_ll - vectorized_ll[0]
        per_particle_diff = per_particle_ll - per_particle_ll[0]
        np.testing.assert_allclose(vectorized_diff, per_particle_diff, atol=1e-10)
