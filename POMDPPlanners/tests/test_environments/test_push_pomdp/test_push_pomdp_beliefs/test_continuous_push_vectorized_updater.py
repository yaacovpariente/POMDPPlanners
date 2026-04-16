"""Tests for the Continuous Push vectorized belief updater.

This module tests the vectorized batch transition and observation
log-likelihood methods for the Continuous Push POMDP.
"""

# pylint: disable=protected-access

import numpy as np

from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
    ContinuousPushPOMDP,
    ContinuousPushPOMDPDiscreteActions,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp_beliefs.continuous_push_vectorized_updater import (
    ContinuousPushVectorizedUpdater,
)
from POMDPPlanners.tests.test_core.test_belief.vectorized_updater_test_utils import (
    assert_batch_obs_log_likelihood_matches_loop,
    assert_batch_transition_matches_loop,
)


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
        """Test vectorized batch_transition matches per-particle state_transition_model.

        Purpose: Verifies that batch_transition produces the same results as
                 calling the environment's state_transition_model per particle
                 with the same random seed.

        Given: A set of particles with varied positions and a continuous action.
        When: batch_transition is called, and the same transitions are
              computed per-particle via state_transition_model.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(123)
        n = 30
        particles = np.random.uniform(0.5, 3.5, (n, 6))
        particles[:, 4:6] = [3.5, 3.5]
        action = np.array([0.5, 0.3])

        def per_particle_fn(particle, act):
            return self.env.state_transition_model(state=particle, action=act).sample()[0]

        assert_batch_transition_matches_loop(
            updater=self.updater,
            particles=particles,
            action=action,
            per_particle_transition_fn=per_particle_fn,
            seed=999,
        )

    def test_batch_obs_log_likelihood_matches_per_particle_loop(self):
        """Test vectorized log-likelihood matches per-particle observation_model.probability.

        Purpose: Verifies that batch_observation_log_likelihood matches the
                 per-particle log(observation_model.probability) from the
                 environment.

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
            obs_model = self.env.observation_model(next_state=particle, action=act)
            prob = obs_model.probability([obs])[0]
            if prob > 0:
                return np.log(prob)
            return -np.inf

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
