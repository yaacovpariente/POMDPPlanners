"""Tests for CartPoleVectorizedUpdater.

This module tests the vectorized batch transition and observation
log-likelihood methods, including an equivalence test that verifies
the vectorized results match the per-particle loop.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp_beliefs import (
    CartPoleVectorizedUpdater,
)
from POMDPPlanners.tests.test_core.test_belief.belief_equivalence_utils import (
    assert_sample_distributions_match,
    assert_update_particles_match,
    assert_update_weights_match,
)
from POMDPPlanners.tests.test_core.test_belief.vectorized_updater_test_utils import (
    assert_batch_obs_log_likelihood_matches_loop,
    assert_batch_transition_matches_loop,
)


def _make_aligned_beliefs(updater, n_particles=60):
    """Create baseline + vectorized beliefs with identical initial particles."""
    np.random.seed(42)
    particles_array = np.random.uniform(-0.05, 0.05, (n_particles, 4))
    particles_list = [particles_array[i].copy() for i in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)

    base = WeightedParticleBelief(
        particles=particles_list,
        log_weights=log_weights.copy(),
        resampling=False,
    )
    vec = VectorizedWeightedParticleBelief(
        particles=particles_array.copy(),
        log_weights=log_weights.copy(),
        updater=updater,
        resampling=False,
    )
    return base, vec


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

    def test_deterministic_under_fixed_seed(self, updater):
        """Test that batch_transition is deterministic under a fixed seed.

        Purpose: Validates that the stochastic batch_transition produces
                 identical output when the global numpy RNG is seeded
                 identically before each call.

        Given: A set of particles and an action, with np.random.seed fixed
               before each call.
        When: batch_transition is called twice.
        Then: Both results are identical.

        Test type: unit
        """
        particles = np.array([[0.0, 0.0, 0.1, 0.0], [0.01, -0.02, 0.05, 0.1]])
        np.random.seed(7)
        result_a = updater.batch_transition(particles, action=1)
        np.random.seed(7)
        result_b = updater.batch_transition(particles, action=1)
        np.testing.assert_array_equal(result_a, result_b)

    def test_physics_deterministic_component(self, updater):
        """Test that the deterministic physics component matches hand-computed values.

        Purpose: Validates the deterministic physics equations for a single
                 particle. We inspect the internal deterministic helper so
                 stochastic noise does not enter the comparison.

        Given: A single particle at [0, 0, 0.1, 0] and action=1 (right force).
        When: The deterministic component of the transition is computed.
        Then: The result matches the expected Euler integration result.

        Test type: unit
        """
        import math

        state = np.array([[0.0, 0.0, 0.1, 0.0]])
        # pylint: disable=protected-access
        result = updater._deterministic_next_state(state, action=1)
        # pylint: enable=protected-access

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
            state_transition_dist=env._state_transition_dist,
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
        """Test vectorized batch_transition matches per-particle state_transition_model.sample.

        Purpose: Verifies that batch_transition produces the same stochastic
                 next states as the environment's state_transition_model.sample
                 when the global RNG is seeded identically on both paths.

        Given: A set of particles, an action, and a fixed random seed.
        When: batch_transition is called, and the same transitions are computed
              per-particle using the environment's state_transition_model.sample.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(123)
        particles = np.random.uniform(-0.05, 0.05, (50, 4))

        def per_particle_fn(particle, action):
            return env.state_transition_model(state=particle, action=action).sample()[0]

        assert_batch_transition_matches_loop(
            updater=updater,
            particles=particles,
            action=1,
            per_particle_transition_fn=per_particle_fn,
            seed=999,
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
        particles = np.random.uniform(-0.05, 0.05, (50, 4))
        observation = np.array([0.01, 0.0, 0.02, 0.0])

        def per_particle_ll_fn(particle, action, obs):
            obs_model = env.observation_model(next_state=particle, action=action)
            return np.log(obs_model.probability([obs])[0])

        assert_batch_obs_log_likelihood_matches_loop(
            updater=updater,
            particles=particles,
            action=1,
            observation=observation,
            per_particle_ll_fn=per_particle_ll_fn,
        )


# ---------------------------------------------------------------------------
# Belief-level equivalence against WeightedParticleBelief
# ---------------------------------------------------------------------------


class TestBeliefEquivalenceWithBaseline:
    def test_update_particles_match(self, env, updater):
        """Test vectorized belief update produces identical next particles.

        Purpose: Validates that VectorizedWeightedParticleBelief.update and
            WeightedParticleBelief.update agree on next-state particles once
            the vectorized updater mirrors the standard transition noise.

        Given: 60 aligned particles.
        When: Both beliefs are updated with action=1 under a shared seed.
        Then: Next particles agree within floating-point tolerance.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(updater)
        obs = np.array([0.01, 0.0, 0.02, 0.0])
        assert_update_particles_match(
            base=base, vec=vec, action=1, observation=obs, pomdp=env, seed=999
        )

    def test_update_weights_match(self, env, updater):
        """Test vectorized and baseline beliefs produce identical normalized weights.

        Purpose: Validates observation-reweighting consistency post-update.

        Given: 60 aligned particles.
        When: Both beliefs are updated with action=1 under a shared seed.
        Then: Normalized weights agree within 1e-6 L-infinity.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(updater)
        obs = np.array([0.01, 0.0, 0.02, 0.0])
        assert_update_weights_match(
            base=base,
            vec=vec,
            action=1,
            observation=obs,
            pomdp=env,
            atol=1e-6,
            seed=999,
        )

    def test_sample_distributions_match_post_update(self, env, updater):
        """Test sample() on both beliefs draws unbiased from normalized_weights.

        Purpose: Validates sample() unbiasedness and cross-belief agreement.

        Given: 60 aligned particles; one update step seeded identically.
        When: 20,000 samples are drawn from each belief.
        Then: Empirical histograms agree and each matches its normalized_weights.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(updater)
        obs = np.array([0.01, 0.0, 0.02, 0.0])
        np.random.seed(999)
        vec = vec.update(action=1, observation=obs, pomdp=env)
        np.random.seed(999)
        base = base.update(action=1, observation=obs, pomdp=env)

        assert_sample_distributions_match(
            base=base,
            vec=vec,
            n_samples=20_000,
            tol=0.02,
            atol_weights=0.02,
            seed=400,
        )
