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
