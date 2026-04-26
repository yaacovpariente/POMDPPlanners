"""Tests for the Continuous Push vectorized belief updater.

This module tests the vectorized batch transition and observation
log-likelihood methods for the Continuous Push POMDP.
"""

# pylint: disable=protected-access,attribute-defined-outside-init,unsubscriptable-object
# ``unsubscriptable-object``: pylint can't infer that .sample() returns list
# and .probability() / batch_transition return ndarray; both support [i].

import numpy as np

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.push_pomdp import _native
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
    ContinuousPushPOMDP,
    ContinuousPushPOMDPDiscreteActions,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp_beliefs.continuous_push_vectorized_updater import (
    ContinuousPushVectorizedUpdater,
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
    particles_array = np.random.uniform(0.5, 3.5, (n_particles, 6))
    particles_array[:, 4:6] = [3.5, 3.5]
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


class TestContinuousPushVectorizedUpdater:
    """Test the vectorized belief updater."""

    def setup_method(self):
        """Set up shared test fixtures."""
        np.random.seed(42)
        self.env = ContinuousPushPOMDP(
            discount_factor=0.99,
            state_transition_cov_matrix=np.eye(2) * 0.01,
            robot_radius=0.3,
        )
        self.updater = ContinuousPushVectorizedUpdater.from_environment(self.env)

    def test_batch_transition_shape(self):
        """Test that batch_transition returns correct shape.

        Purpose: Validates output shape of batch transition.

        Given: 50 particles of shape (50, 6) and a continuous action.
        When: batch_transition is called.
        Then: Returns shape (50, 6).

        Test type: unit
        """
        particles = np.tile(self.env.initial_state_dist().sample()[0], (50, 1))
        result = self.updater.batch_transition(particles, np.array([1.0, 0.0]))
        assert result.shape == (50, 6)

    def test_batch_transition_target_preserved(self):
        """Test that target position is unchanged after transition.

        Purpose: Validates that target coordinates are preserved.

        Given: Particles with known target positions.
        When: batch_transition is called.
        Then: Target columns (4:6) are unchanged.

        Test type: unit
        """
        particles = np.tile(self.env.initial_state_dist().sample()[0], (20, 1))
        result = self.updater.batch_transition(particles, np.array([1.0, 0.0]))
        np.testing.assert_array_equal(result[:, 4:6], particles[:, 4:6])

    def test_batch_transition_with_string_action(self):
        """Test batch transition with string action (discrete wrapper).

        Purpose: Validates string action resolution.

        Given: Updater built from discrete-action environment.
        When: batch_transition is called with "right".
        Then: Returns shape (50, 6) with robot moved right.

        Test type: unit
        """
        env_d = ContinuousPushPOMDPDiscreteActions(
            discount_factor=0.99,
            state_transition_cov_matrix=np.eye(2) * 0.01,
            robot_radius=0.3,
        )
        updater = ContinuousPushVectorizedUpdater.from_environment(env_d)
        particles = np.tile(env_d.initial_state_dist().sample()[0], (50, 1))
        result = updater.batch_transition(particles, "right")  # type: ignore[arg-type]
        assert result.shape == (50, 6)

    def test_batch_observation_log_likelihood_shape(self):
        """Test that log-likelihood returns correct shape.

        Purpose: Validates log-likelihood output shape.

        Given: 50 particles and an observation.
        When: batch_observation_log_likelihood is called.
        Then: Returns shape (50,).

        Test type: unit
        """
        particles = np.tile(self.env.initial_state_dist().sample()[0], (50, 1))
        action = np.array([1.0, 0.0])
        next_p = self.updater.batch_transition(particles, action)
        obs = next_p[0].copy()
        ll = self.updater.batch_observation_log_likelihood(next_p, action, obs)
        assert ll.shape == (50,)

    def test_log_likelihood_finite(self):
        """Test that log-likelihoods are finite.

        Purpose: Validates numerical stability of log-likelihood.

        Given: Particles and a reasonable observation.
        When: batch_observation_log_likelihood is called.
        Then: All values are finite.

        Test type: unit
        """
        particles = np.tile(self.env.initial_state_dist().sample()[0], (30, 1))
        action = np.array([0.5, 0.0])
        next_p = self.updater.batch_transition(particles, action)
        obs = next_p[0].copy()
        ll = self.updater.batch_observation_log_likelihood(next_p, action, obs)
        assert np.all(np.isfinite(ll))

    def test_config_id_deterministic(self):
        """Test that config_id is deterministic.

        Purpose: Validates deterministic config identification.

        Given: Two updaters built from identical environments.
        When: config_id is accessed.
        Then: Both return the same string.

        Test type: unit
        """
        env2 = ContinuousPushPOMDP(
            discount_factor=0.99,
            state_transition_cov_matrix=np.eye(2) * 0.01,
            robot_radius=0.3,
        )
        updater2 = ContinuousPushVectorizedUpdater.from_environment(env2)
        assert self.updater.config_id == updater2.config_id

    def test_batch_transition_matches_per_particle_loop(self):
        """Test vectorized batch_transition matches per-particle env.sample_next_state.

        Purpose: Verifies that batch_transition produces the same results as
                 calling the environment's sample_next_state per particle
                 with the same random seed.

        Given: A set of particles with varied positions and a continuous action.
        When: batch_transition is called, and the same transitions are
              computed per-particle via env.sample_next_state.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(123)
        n = 30
        particles = np.random.uniform(0.5, 3.5, (n, 6))
        particles[:, 4:6] = [3.5, 3.5]
        action = np.array([0.5, 0.3])

        def per_particle_fn(particle, act):
            return self.env.sample_next_state(particle, act)

        assert_batch_transition_matches_loop(
            updater=self.updater,
            particles=particles,
            action=action,
            per_particle_transition_fn=per_particle_fn,
            seed=999,
            seed_fn=_native.set_seed,
        )

    def test_batch_obs_log_likelihood_matches_per_particle_loop(self):
        """Test vectorized log-likelihood matches per-particle env.observation_log_probability.

        Purpose: Verifies that batch_observation_log_likelihood matches the
                 per-particle env.observation_log_probability output for the
                 same (particle, action, observation) triples.

        Given: A set of particles with object positions near the observation.
        When: batch_observation_log_likelihood is called, and per-particle
              log-probabilities are computed.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(42)
        n = 30
        base_obj = np.array([2.0, 2.0])
        particles = np.empty((n, 6))
        particles[:, :2] = np.random.uniform(0.5, 3.5, (n, 2))
        particles[:, 2:4] = base_obj + np.random.normal(0, 0.05, (n, 2))
        particles[:, 4:6] = [3.5, 3.5]
        observation = particles[0].copy()
        observation[2:4] = base_obj + 0.02
        action = np.array([0.5, 0.0])

        def per_particle_ll_fn(particle, act, obs):
            return float(self.env.observation_log_probability(particle, act, [obs])[0])

        assert_batch_obs_log_likelihood_matches_loop(
            updater=self.updater,
            particles=particles,
            action=action,
            observation=observation,
            per_particle_ll_fn=per_particle_ll_fn,
        )

    def test_from_environment_discrete(self):
        """Test from_environment with discrete-action environment.

        Purpose: Validates factory method with discrete wrapper.

        Given: A ContinuousPushPOMDPDiscreteActions environment.
        When: from_environment is called.
        Then: Returns an updater with action_to_vector set.

        Test type: unit
        """
        env_d = ContinuousPushPOMDPDiscreteActions(
            discount_factor=0.99,
            state_transition_cov_matrix=np.eye(2) * 0.01,
            robot_radius=0.3,
        )
        updater = ContinuousPushVectorizedUpdater.from_environment(env_d)
        assert updater._action_to_vector is not None
        assert "right" in updater._action_to_vector


class TestBeliefEquivalenceWithBaseline:
    """Belief-level equivalence checks against WeightedParticleBelief."""

    def setup_method(self):
        """Set up shared test fixtures."""
        self.env = ContinuousPushPOMDP(
            discount_factor=0.99,
            state_transition_cov_matrix=np.eye(2) * 0.01,
            robot_radius=0.3,
        )
        self.updater = ContinuousPushVectorizedUpdater.from_environment(self.env)

    def test_update_particles_match(self):
        """Test vectorized belief update produces identical next particles.

        Purpose: Validates that VectorizedWeightedParticleBelief.update and
            WeightedParticleBelief.update agree on next-state particles for
            continuous Push when the RNG is seeded identically on both paths.

        Given: 60 aligned particles; both beliefs share the same updater.
        When: Both are updated with a 2D continuous action and fixed observation.
        Then: Next-particle arrays agree within floating-point tolerance.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(self.updater)
        action = np.array([0.5, 0.3])
        obs = np.array([2.0, 2.0, 2.0, 2.0, 3.5, 3.5])
        # Both paths now consume noise from the native C++ RNG (the baseline
        # auto-dispatches to ContinuousPushTransitionCpp.batch_sample via
        # hasattr, and the vectorized updater delegates there directly); seed
        # the native RNG to align the two sample sequences.
        assert_update_particles_match(
            base=base,
            vec=vec,
            action=action,
            observation=obs,
            pomdp=self.env,
            seed=999,
            seed_fn=_native.set_seed,
        )

    def test_update_weights_match(self):
        """Test vectorized and baseline beliefs produce identical normalized weights.

        Purpose: Validates that observation-reweighting is consistent between
            vectorized and per-particle update implementations.

        Given: 60 aligned particles.
        When: Both are updated with the same action, observation, and seed.
        Then: Normalized weights agree within 1e-6 L-infinity.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(self.updater)
        action = np.array([0.5, 0.3])
        obs = np.array([2.0, 2.0, 2.0, 2.0, 3.5, 3.5])
        assert_update_weights_match(
            base=base,
            vec=vec,
            action=action,
            observation=obs,
            pomdp=self.env,
            atol=1e-6,
            seed=999,
            seed_fn=_native.set_seed,
        )

    def test_sample_distributions_match_post_update(self):
        """Test sample() on both beliefs draws unbiased from normalized_weights.

        Purpose: Validates sample() unbiasedness and cross-belief agreement.

        Given: 60 aligned particles; one update step seeded identically.
        When: 20,000 samples are drawn from each belief.
        Then: Empirical histograms agree and each matches its normalized_weights.

        Test type: integration
        """
        base, vec = _make_aligned_beliefs(self.updater)
        action = np.array([0.5, 0.3])
        obs = np.array([2.0, 2.0, 2.0, 2.0, 3.5, 3.5])
        # Seed both the native C++ RNG (governs batch_sample) and numpy's
        # (belief.sample() still uses numpy); without the native seed the
        # two beliefs hold different post-update particle sets.
        _native.set_seed(999)
        np.random.seed(999)
        vec = vec.update(action=action, observation=obs, pomdp=self.env)
        _native.set_seed(999)
        np.random.seed(999)
        base = base.update(action=action, observation=obs, pomdp=self.env)

        assert_sample_distributions_match(
            base=base,
            vec=vec,
            n_samples=20_000,
            tol=0.02,
            atol_weights=0.02,
            seed=400,
        )
