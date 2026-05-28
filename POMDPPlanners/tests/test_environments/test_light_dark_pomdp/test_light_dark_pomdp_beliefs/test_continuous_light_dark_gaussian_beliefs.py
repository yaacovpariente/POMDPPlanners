# SPDX-License-Identifier: MIT

"""Tests for the Continuous Light-Dark Gaussian belief factory.

This module tests the factory function and enum that produce pre-configured
GaussianBelief instances for the ContinuousLightDarkPOMDP environment.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.gaussian_belief import GaussianBelief
from POMDPPlanners.core.belief.gaussian_belief_updaters import (
    ExtendedKalmanFilterUpdater,
    LinearKalmanFilterUpdater,
    UnscentedKalmanFilterUpdater,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs import (
    GaussianBeliefUpdaterType,
    create_continuous_light_dark_gaussian_belief,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import continuous_light_dark_pinned_kwargs

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    return ContinuousLightDarkPOMDP(discount_factor=0.95, **continuous_light_dark_pinned_kwargs())


@pytest.fixture
def initial_cov():
    return np.eye(2) * 5.0


# ---------------------------------------------------------------------------
# Factory dispatch tests
# ---------------------------------------------------------------------------


class TestFactoryDispatch:
    def test_linear_kalman_returns_lkf_updater(self, env, initial_cov):
        """Test that LINEAR_KALMAN selects a LinearKalmanFilterUpdater.

        Purpose: Validates factory dispatch for the LINEAR_KALMAN variant.

        Given: A ContinuousLightDarkPOMDP and LINEAR_KALMAN updater type.
        When: create_continuous_light_dark_gaussian_belief is called.
        Then: The returned belief uses a LinearKalmanFilterUpdater.

        Test type: unit
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.LINEAR_KALMAN,
            initial_covariance=initial_cov,
        )
        assert isinstance(belief, GaussianBelief)
        assert isinstance(belief.updater, LinearKalmanFilterUpdater)

    def test_ekf_returns_ekf_updater(self, env, initial_cov):
        """Test that EKF selects an ExtendedKalmanFilterUpdater.

        Purpose: Validates factory dispatch for the EKF variant.

        Given: A ContinuousLightDarkPOMDP and EKF updater type.
        When: create_continuous_light_dark_gaussian_belief is called.
        Then: The returned belief uses an ExtendedKalmanFilterUpdater.

        Test type: unit
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=initial_cov,
        )
        assert isinstance(belief, GaussianBelief)
        assert isinstance(belief.updater, ExtendedKalmanFilterUpdater)

    def test_ukf_returns_ukf_updater(self, env, initial_cov):
        """Test that UKF selects an UnscentedKalmanFilterUpdater.

        Purpose: Validates factory dispatch for the UKF variant.

        Given: A ContinuousLightDarkPOMDP and UKF updater type.
        When: create_continuous_light_dark_gaussian_belief is called.
        Then: The returned belief uses an UnscentedKalmanFilterUpdater.

        Test type: unit
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.UKF,
            initial_covariance=initial_cov,
        )
        assert isinstance(belief, GaussianBelief)
        assert isinstance(belief.updater, UnscentedKalmanFilterUpdater)


# ---------------------------------------------------------------------------
# LKF parameter correctness tests
# ---------------------------------------------------------------------------


class TestLinearKalmanParameters:
    def test_lkf_matrices(self, env, initial_cov):
        """Test that the LKF updater has A=I, B=I, H=I, correct Q and R.

        Purpose: Validates that the factory maps environment dynamics to
                 the correct Kalman filter matrices.

        Given: A ContinuousLightDarkPOMDP.
        When: LINEAR_KALMAN belief is created.
        Then: A=I, B=I, H=I, Q=env.state_transition_cov_matrix,
              R=env.observation_cov_matrix.

        Test type: unit
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.LINEAR_KALMAN,
            initial_covariance=initial_cov,
        )
        updater = belief.updater
        assert isinstance(updater, LinearKalmanFilterUpdater)
        np.testing.assert_array_equal(updater.A, np.eye(2))
        np.testing.assert_array_equal(updater.B, np.eye(2))
        np.testing.assert_array_equal(updater.H, np.eye(2))
        np.testing.assert_array_equal(updater.Q, env.state_transition_cov_matrix)
        np.testing.assert_array_equal(updater.R, env.observation_cov_matrix)


# ---------------------------------------------------------------------------
# EKF function correctness tests
# ---------------------------------------------------------------------------


class TestEKFFunctions:
    def test_ekf_transition_fn(self, env, initial_cov):
        """Test that the EKF transition function implements x + u.

        Purpose: Validates the EKF transition function correctness.

        Given: An EKF-based belief from the factory.
        When: transition_fn is called with x=[1,2], u=[3,4].
        Then: Result is [4, 6].

        Test type: unit
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=initial_cov,
        )
        updater = belief.updater
        assert isinstance(updater, ExtendedKalmanFilterUpdater)
        result = updater.transition_fn(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        np.testing.assert_array_equal(result, [4.0, 6.0])

    def test_ekf_observation_fn(self, env, initial_cov):
        """Test that the EKF observation function is the identity.

        Purpose: Validates the EKF observation function correctness.

        Given: An EKF-based belief from the factory.
        When: observation_fn is called with x=[5, 7].
        Then: Result is [5, 7].

        Test type: unit
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=initial_cov,
        )
        updater = belief.updater
        assert isinstance(updater, ExtendedKalmanFilterUpdater)
        result = updater.observation_fn(np.array([5.0, 7.0]))
        np.testing.assert_array_equal(result, [5.0, 7.0])

    def test_ekf_jacobians_are_identity(self, env, initial_cov):
        """Test that the EKF Jacobians are identity matrices.

        Purpose: Validates the EKF Jacobians for linear dynamics.

        Given: An EKF-based belief from the factory.
        When: Jacobians are evaluated.
        Then: Both are np.eye(2).

        Test type: unit
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=initial_cov,
        )
        updater = belief.updater
        assert isinstance(updater, ExtendedKalmanFilterUpdater)
        F = updater.transition_jacobian(np.zeros(2), np.zeros(2))
        H = updater.observation_jacobian(np.zeros(2))
        np.testing.assert_array_equal(F, np.eye(2))
        np.testing.assert_array_equal(H, np.eye(2))


# ---------------------------------------------------------------------------
# UKF function correctness tests
# ---------------------------------------------------------------------------


class TestUKFFunctions:
    def test_ukf_transition_fn(self, env, initial_cov):
        """Test that the UKF transition function implements x + u.

        Purpose: Validates the UKF transition function correctness.

        Given: A UKF-based belief from the factory.
        When: transition_fn is called with x=[1,2], u=[3,4].
        Then: Result is [4, 6].

        Test type: unit
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.UKF,
            initial_covariance=initial_cov,
        )
        updater = belief.updater
        assert isinstance(updater, UnscentedKalmanFilterUpdater)
        result = updater.transition_fn(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        np.testing.assert_array_equal(result, [4.0, 6.0])

    def test_ukf_observation_fn(self, env, initial_cov):
        """Test that the UKF observation function is the identity.

        Purpose: Validates the UKF observation function correctness.

        Given: A UKF-based belief from the factory.
        When: observation_fn is called with x=[5, 7].
        Then: Result is [5, 7].

        Test type: unit
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.UKF,
            initial_covariance=initial_cov,
        )
        updater = belief.updater
        assert isinstance(updater, UnscentedKalmanFilterUpdater)
        result = updater.observation_fn(np.array([5.0, 7.0]))
        np.testing.assert_array_equal(result, [5.0, 7.0])


# ---------------------------------------------------------------------------
# Observation noise flag tests
# ---------------------------------------------------------------------------


class TestObservationNoiseFlag:
    def test_near_beacon_noise_halves_R(self, env, initial_cov):
        """Test that use_near_beacon_noise=True halves the observation covariance.

        Purpose: Validates the near-beacon noise flag.

        Given: A ContinuousLightDarkPOMDP.
        When: Factory is called with use_near_beacon_noise=True.
        Then: The updater's R equals env.observation_cov_matrix * 0.5.

        Test type: unit
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.LINEAR_KALMAN,
            initial_covariance=initial_cov,
            use_near_beacon_noise=True,
        )
        updater = belief.updater
        assert isinstance(updater, LinearKalmanFilterUpdater)
        np.testing.assert_array_equal(updater.R, env.observation_cov_matrix * 0.5)

    def test_far_beacon_noise_uses_full_R(self, env, initial_cov):
        """Test that use_near_beacon_noise=False uses the full observation covariance.

        Purpose: Validates the default observation noise behaviour.

        Given: A ContinuousLightDarkPOMDP.
        When: Factory is called with use_near_beacon_noise=False (default).
        Then: The updater's R equals env.observation_cov_matrix.

        Test type: unit
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.LINEAR_KALMAN,
            initial_covariance=initial_cov,
            use_near_beacon_noise=False,
        )
        updater = belief.updater
        assert isinstance(updater, LinearKalmanFilterUpdater)
        np.testing.assert_array_equal(updater.R, env.observation_cov_matrix)


# ---------------------------------------------------------------------------
# Belief attribute tests
# ---------------------------------------------------------------------------


class TestBeliefAttributes:
    def test_mean_is_start_state(self, env, initial_cov):
        """Test that the returned belief mean equals env.start_state.

        Purpose: Validates mean initialisation.

        Given: A ContinuousLightDarkPOMDP with start_state=[0, 5].
        When: Factory creates a belief.
        Then: belief.mean == env.start_state.

        Test type: unit
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.LINEAR_KALMAN,
            initial_covariance=initial_cov,
        )
        np.testing.assert_array_equal(belief.mean, env.start_state.astype(float))

    def test_covariance_matches_initial(self, env, initial_cov):
        """Test that the returned belief covariance equals initial_covariance.

        Purpose: Validates covariance initialisation.

        Given: initial_covariance = 5 * I.
        When: Factory creates a belief.
        Then: belief.covariance == initial_covariance.

        Test type: unit
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.LINEAR_KALMAN,
            initial_covariance=initial_cov,
        )
        np.testing.assert_array_equal(belief.covariance, initial_cov)


# ---------------------------------------------------------------------------
# End-to-end update tests
# ---------------------------------------------------------------------------


class TestEndToEndUpdate:
    @pytest.mark.parametrize(
        "updater_type",
        [
            GaussianBeliefUpdaterType.LINEAR_KALMAN,
            GaussianBeliefUpdaterType.EKF,
            GaussianBeliefUpdaterType.UKF,
        ],
    )
    def test_update_shrinks_covariance(self, env, initial_cov, updater_type):
        """Test that an observation update reduces uncertainty.

        Purpose: Validates that Bayesian update shrinks the covariance trace.

        Given: A high-uncertainty initial belief (cov = 5*I).
        When: update() is called with a plausible observation.
        Then: The trace of the posterior covariance is smaller.

        Test type: integration
        """
        belief = create_continuous_light_dark_gaussian_belief(
            env=env,
            updater_type=updater_type,
            initial_covariance=initial_cov,
        )
        action = np.array([1.0, 0.0])
        observation = belief.mean + action  # observation near predicted state
        new_belief = belief.update(action=action, observation=observation, pomdp=None)

        assert isinstance(new_belief, GaussianBelief)
        assert new_belief.mean.shape == (2,)
        assert new_belief.covariance.shape == (2, 2)
        assert np.trace(new_belief.covariance) < np.trace(initial_cov)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_invalid_enum_value_raises(self, env, initial_cov):
        """Test that an invalid updater type string raises ValueError.

        Purpose: Validates error handling for unknown updater types.

        Given: A string that is not a valid GaussianBeliefUpdaterType member.
        When: Factory is called with this string.
        Then: ValueError is raised by the Enum constructor.

        Test type: unit
        """
        with pytest.raises(ValueError):
            create_continuous_light_dark_gaussian_belief(
                env=env,
                updater_type=GaussianBeliefUpdaterType("invalid_type"),
                initial_covariance=initial_cov,
            )
