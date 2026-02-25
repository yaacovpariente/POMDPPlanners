"""Tests for ContinuousLightDarkVectorizedUpdater.

This module tests the vectorized batch transition and observation
log-likelihood methods, including an equivalence test that verifies
the vectorized results match the per-particle loop in WeightedParticleBelief.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ContinuousLightDarkPOMDPDiscreteActions,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs import (
    ContinuousLightDarkVectorizedUpdater,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    return ContinuousLightDarkPOMDP(discount_factor=0.95)


@pytest.fixture
def updater(env):
    return ContinuousLightDarkVectorizedUpdater.from_environment(env)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFromEnvironment:
    def test_from_environment_creates_updater(self, env):
        """Test that from_environment constructs a valid updater.

        Purpose: Validates the factory classmethod.

        Given: A ContinuousLightDarkPOMDP instance.
        When: from_environment is called.
        Then: A ContinuousLightDarkVectorizedUpdater is returned with
              matching parameters.

        Test type: unit
        """
        updater = ContinuousLightDarkVectorizedUpdater.from_environment(env)
        assert isinstance(updater, ContinuousLightDarkVectorizedUpdater)
        assert updater.beacon_radius == env.beacon_radius
        assert updater.grid_size == env.grid_size
        np.testing.assert_array_equal(updater.beacons, env.beacons)


# ---------------------------------------------------------------------------
# batch_transition tests
# ---------------------------------------------------------------------------


class TestBatchTransition:
    def test_output_shape(self, updater):
        """Test that batch_transition returns correct shape.

        Purpose: Validates output shape of batch_transition.

        Given: 30 particles of dimension 2.
        When: batch_transition is called.
        Then: Result has shape (30, 2).

        Test type: unit
        """
        np.random.seed(0)
        particles = np.random.rand(30, 2) * 10
        action = np.array([1.0, 0.0])
        result = updater.batch_transition(particles, action)
        assert result.shape == (30, 2)

    def test_mean_shift(self, updater):
        """Test that mean transition is particles + action (plus noise).

        Purpose: Validates that the mean next state is particles + action.

        Given: Identical particles at [5, 5] and action [1, 0].
        When: batch_transition is called many times and results averaged.
        Then: The mean result is close to [6, 5].

        Test type: unit
        """
        np.random.seed(42)
        n = 5000
        particles = np.tile([5.0, 5.0], (n, 1))
        action = np.array([1.0, 0.0])
        results = updater.batch_transition(particles, action)
        mean_result = results.mean(axis=0)
        np.testing.assert_allclose(mean_result, [6.0, 5.0], atol=0.1)


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
        particles = np.random.rand(20, 2) * 10
        action = np.array([1.0, 0.0])
        obs = np.array([5.0, 5.0])
        result = updater.batch_observation_log_likelihood(particles, action, obs)
        assert result.shape == (20,)

    def test_near_beacon_higher_likelihood(self, updater):
        """Test that particles near a beacon with matching observation get higher likelihood.

        Purpose: Validates beacon-dependent observation likelihood.

        Given: One particle at [0, 0] (near beacon at origin) and one at
               [2.5, 2.5] (far from all beacons), with observation at their positions.
        When: Log-likelihoods are computed for observation matching their positions.
        Then: The near-beacon particle has higher log-likelihood because
              the near-beacon distribution has smaller covariance.

        Test type: unit
        """
        near_particle = np.array([[0.0, 0.0]])  # at beacon
        far_particle = np.array([[2.5, 2.5]])  # far from all beacons
        action = np.zeros(2)

        ll_near = updater.batch_observation_log_likelihood(near_particle, action, near_particle[0])
        ll_far = updater.batch_observation_log_likelihood(far_particle, action, far_particle[0])
        # Near-beacon has tighter covariance → higher peak density
        assert ll_near[0] > ll_far[0]

    def test_mixed_near_far_particles(self, updater):
        """Test that mixed near/far particles produce valid log-likelihoods.

        Purpose: Validates correct handling when particles span near and far regions.

        Given: Particles at [0,0] (near beacon) and [2.5,2.5] (far from beacons).
        When: batch_observation_log_likelihood is called.
        Then: All log-likelihoods are finite and non-positive.

        Test type: unit
        """
        particles = np.array([[0.0, 0.0], [2.5, 2.5]])
        action = np.zeros(2)
        obs = np.array([0.5, 0.5])
        ll = updater.batch_observation_log_likelihood(particles, action, obs)
        assert ll.shape == (2,)
        assert np.all(np.isfinite(ll))
        assert np.all(ll <= 0)  # log of probability density ≤ 0 for d≥2 Gaussian


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

        Given: Two updaters with different beacon_radius.
        When: config_id is computed for both.
        Then: The IDs differ.

        Test type: unit
        """
        u1 = ContinuousLightDarkVectorizedUpdater.from_environment(env)
        u2 = ContinuousLightDarkVectorizedUpdater(
            state_transition_dist=env._state_transition_dist,
            obs_dist_near_beacon=env._obs_dist_near_beacon,
            obs_dist_far_from_beacon=env._obs_dist_far_from_beacon,
            beacons=env.beacons,
            beacon_radius=env.beacon_radius * 2,
            grid_size=env.grid_size,
        )
        assert u1.config_id != u2.config_id


# ---------------------------------------------------------------------------
# Equivalence test: vectorized vs per-particle loop
# ---------------------------------------------------------------------------


class TestEquivalenceWithPerParticleLoop:
    def test_batch_transition_matches_per_particle_loop(self, env, updater):
        """Test vectorized batch_transition matches per-particle state_transition_model.

        Purpose: Verifies that batch_transition produces the same results as calling
                 the environment's state_transition_model per particle with the same noise.

        Given: A set of particles, an action, and a fixed random seed.
        When: batch_transition is called, and the same transitions are computed
              per-particle using the environment's state_transition_model.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        n = 50
        action = np.array([1.0, 0.5])

        np.random.seed(123)
        particles = np.random.rand(n, 2) * 10

        # Vectorized path
        np.random.seed(999)
        vectorized_result = updater.batch_transition(particles, action)

        # Per-particle path: use the same noise generation logic
        np.random.seed(999)
        per_particle_result = np.empty_like(particles)
        for i in range(n):
            next_state = env.state_transition_model(state=particles[i], action=action).sample()[0]
            per_particle_result[i] = next_state

        np.testing.assert_allclose(vectorized_result, per_particle_result, atol=1e-10)

    def test_batch_observation_log_likelihood_matches_per_particle_loop(self, env, updater):
        """Test vectorized log-likelihood matches per-particle log_pdf computation.

        Purpose: Verifies that batch_observation_log_likelihood matches the
                 per-particle log-PDF from the observation model's underlying
                 distribution, computed directly in log-space.

        Given: A set of next-state particles and an observation.
        When: batch_observation_log_likelihood is called, and per-particle
              log_pdf is computed via the observation model's active distribution.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(42)
        n = 50
        action = np.array([1.0, 0.0])
        observation = np.array([5.0, 5.0])
        particles = np.random.rand(n, 2) * 10

        # Vectorized path
        vectorized_ll = updater.batch_observation_log_likelihood(particles, action, observation)

        # Per-particle path: use log_pdf directly to avoid exp→log underflow
        per_particle_ll = np.empty(n)
        for i in range(n):
            obs_model = env.observation_model(next_state=particles[i], action=action)
            per_particle_ll[i] = obs_model._active_dist.log_pdf(
                np.array([observation]), particles[i]
            )[0]

        np.testing.assert_allclose(vectorized_ll, per_particle_ll, atol=1e-10)

    def test_full_update_equivalence(self, env, updater):
        """Test that a full vectorized update matches WeightedParticleBelief._update_weights.

        Purpose: End-to-end equivalence test comparing the vectorized belief update
                 path against the per-particle loop in WeightedParticleBelief.

        Given: Identical initial particles and weights.
        When: One step of update is performed via both paths with the same random seed.
        Then: Next particles match exactly, and both paths agree on which particles
              carry significant weight. The two paths differ in how they handle
              near-zero probabilities (eps floor vs exact log-space), so we compare
              normalized weights only for particles with non-negligible weight.

        Test type: integration
        """
        from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
        from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
            VectorizedWeightedParticleBelief,
        )

        np.random.seed(77)
        n = 40
        action = np.array([0.5, -0.5])
        observation = np.array([3.0, 4.0])

        particles_array = np.random.rand(n, 2) * 10
        particles_list = [particles_array[i].copy() for i in range(n)]
        log_w = np.full(n, -np.log(n))

        # Vectorized path
        np.random.seed(200)
        v_belief = VectorizedWeightedParticleBelief(
            particles=particles_array.copy(),
            log_weights=log_w.copy(),
            updater=updater,
            resampling=False,
        )
        v_updated = v_belief.update(action=action, observation=observation, pomdp=None)

        # Per-particle path (WeightedParticleBelief)
        np.random.seed(200)
        w_belief = WeightedParticleBelief(
            particles=particles_list,
            log_weights=log_w.copy(),
            resampling=False,
        )
        w_updated = w_belief.update(action=action, observation=observation, pomdp=env)

        # Compare next particles
        w_particles = np.array(w_updated.particles)
        np.testing.assert_allclose(v_updated.particles, w_particles, atol=1e-10)

        # Compare normalized weights for particles with non-negligible weight.
        # WeightedParticleBelief uses log(eps + prob), vectorized uses log_pdf directly.
        # When prob underflows to 0.0, the eps floor creates divergent normalized
        # weights for low-probability particles. We restrict comparison to particles
        # where the per-particle path assigns meaningful weight (above eps-floor level).
        v_nw = v_updated.normalized_weights
        w_nw = w_updated.normalized_weights
        significant = w_nw > 1e-4
        assert np.any(significant), "No particles have significant weight"
        np.testing.assert_allclose(v_nw[significant], w_nw[significant], atol=1e-3)

        # Verify both paths agree on the ranking of top particles
        v_top = np.argsort(v_nw)[::-1][:5]
        w_top = np.argsort(w_nw)[::-1][:5]
        np.testing.assert_array_equal(v_top, w_top)


# ---------------------------------------------------------------------------
# Discrete action support tests
# ---------------------------------------------------------------------------


@pytest.fixture
def discrete_env():
    return ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)


@pytest.fixture
def discrete_updater(discrete_env):
    return ContinuousLightDarkVectorizedUpdater.from_environment(discrete_env)


class TestDiscreteActionFromEnvironment:
    def test_from_environment_detects_action_mapping(self, discrete_env):
        """Test that from_environment picks up action_to_vector from discrete env.

        Purpose: Validates that the factory classmethod auto-detects the
                 action_to_vector attribute on the discrete variant.

        Given: A ContinuousLightDarkPOMDPDiscreteActions instance.
        When: from_environment is called.
        Then: The updater stores a non-None _action_to_vector mapping.

        Test type: unit
        """
        updater = ContinuousLightDarkVectorizedUpdater.from_environment(discrete_env)
        assert updater._action_to_vector is not None
        assert "right" in updater._action_to_vector

    def test_from_environment_continuous_has_no_mapping(self, env):
        """Test that from_environment leaves _action_to_vector as None for continuous env.

        Purpose: Validates that the continuous variant does not get an action mapping.

        Given: A ContinuousLightDarkPOMDP instance.
        When: from_environment is called.
        Then: The updater has _action_to_vector == None.

        Test type: unit
        """
        updater = ContinuousLightDarkVectorizedUpdater.from_environment(env)
        assert updater._action_to_vector is None


class TestDiscreteActionBatchTransition:
    def test_string_action_output_shape(self, discrete_updater):
        """Test that batch_transition works with string actions.

        Purpose: Validates that string actions are resolved via the mapping
                 and produce correct output shape.

        Given: 30 particles and a string action 'right'.
        When: batch_transition is called.
        Then: Result has shape (30, 2).

        Test type: unit
        """
        np.random.seed(0)
        particles = np.random.rand(30, 2) * 10
        result = discrete_updater.batch_transition(particles, "right")
        assert result.shape == (30, 2)

    def test_string_action_mean_shift(self, discrete_updater):
        """Test that string action 'right' shifts mean by [1, 0].

        Purpose: Validates that the resolved vector produces the expected
                 mean state shift.

        Given: Identical particles at [5, 5] and action 'right'.
        When: batch_transition is called on many particles.
        Then: The mean result is close to [6, 5].

        Test type: unit
        """
        np.random.seed(42)
        n = 5000
        particles = np.tile([5.0, 5.0], (n, 1))
        results = discrete_updater.batch_transition(particles, "right")
        mean_result = results.mean(axis=0)
        np.testing.assert_allclose(mean_result, [6.0, 5.0], atol=0.1)

    def test_string_action_matches_vector_action(self, discrete_updater):
        """Test that string 'right' produces same result as vector [1, 0].

        Purpose: Validates equivalence between string and vector action paths.

        Given: Same particles and random seed.
        When: batch_transition is called with 'right' and with np.array([1, 0]).
        Then: Results are identical.

        Test type: unit
        """
        np.random.seed(0)
        particles = np.random.rand(20, 2) * 10

        np.random.seed(42)
        result_string = discrete_updater.batch_transition(particles, "right")

        np.random.seed(42)
        result_vector = discrete_updater.batch_transition(particles, np.array([1.0, 0.0]))

        np.testing.assert_array_equal(result_string, result_vector)


class TestDiscreteActionFullUpdate:
    def test_vectorized_belief_update_with_string_action(self, discrete_updater):
        """Test full VectorizedWeightedParticleBelief.update() with a string action.

        Purpose: End-to-end test that the belief update path works when
                 the action is a string rather than a numeric array.

        Given: A VectorizedWeightedParticleBelief with the discrete updater.
        When: update() is called with action='right'.
        Then: A new belief is returned with valid particles and weights.

        Test type: integration
        """
        from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
            VectorizedWeightedParticleBelief,
        )

        np.random.seed(42)
        n = 30
        particles = np.random.rand(n, 2) * 10
        log_w = np.full(n, -np.log(n))

        belief = VectorizedWeightedParticleBelief(
            particles=particles,
            log_weights=log_w,
            updater=discrete_updater,
            resampling=False,
        )
        updated = belief.update(action="right", observation=np.array([5.0, 5.0]), pomdp=None)

        assert updated.particles.shape == (n, 2)
        assert np.all(np.isfinite(updated.log_weights))


class TestDiscreteActionConfigId:
    def test_config_id_differs_discrete_vs_continuous(self, env, discrete_env):
        """Test that config_id differs between discrete and continuous updaters.

        Purpose: Validates that the action mapping is reflected in config_id.

        Given: Updaters built from continuous and discrete environments.
        When: config_id is compared.
        Then: The IDs differ.

        Test type: unit
        """
        u_continuous = ContinuousLightDarkVectorizedUpdater.from_environment(env)
        u_discrete = ContinuousLightDarkVectorizedUpdater.from_environment(discrete_env)
        assert u_continuous.config_id != u_discrete.config_id
