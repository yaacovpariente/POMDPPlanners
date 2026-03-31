"""Tests for the Mountain Car Gaussian belief factory.

This module tests the factory function and enum that produce pre-configured
GaussianBelief instances for the MountainCarPOMDP environment.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief.gaussian_belief import GaussianBelief
from POMDPPlanners.core.belief.gaussian_belief_updaters import (
    ExtendedKalmanFilterUpdater,
    UnscentedKalmanFilterUpdater,
)
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp_gaussian_beliefs import (
    GaussianBeliefUpdaterType,
    create_mountain_car_gaussian_belief,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    return MountainCarPOMDP(discount_factor=0.99)


@pytest.fixture
def initial_cov():
    return np.diag([0.01, 0.001])


# ---------------------------------------------------------------------------
# Factory dispatch tests
# ---------------------------------------------------------------------------


class TestFactoryDispatch:
    def test_ekf_returns_ekf_updater(self, env, initial_cov):
        """Test that EKF selects an ExtendedKalmanFilterUpdater.

        Purpose: Validates factory dispatch for the EKF variant.

        Given: A MountainCarPOMDP and EKF updater type.
        When: create_mountain_car_gaussian_belief is called.
        Then: The returned belief uses an ExtendedKalmanFilterUpdater.

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=initial_cov,
        )
        assert isinstance(belief, GaussianBelief)
        assert isinstance(belief.updater, ExtendedKalmanFilterUpdater)

    def test_ukf_returns_ukf_updater(self, env, initial_cov):
        """Test that UKF selects an UnscentedKalmanFilterUpdater.

        Purpose: Validates factory dispatch for the UKF variant.

        Given: A MountainCarPOMDP and UKF updater type.
        When: create_mountain_car_gaussian_belief is called.
        Then: The returned belief uses an UnscentedKalmanFilterUpdater.

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.UKF,
            initial_covariance=initial_cov,
        )
        assert isinstance(belief, GaussianBelief)
        assert isinstance(belief.updater, UnscentedKalmanFilterUpdater)


# ---------------------------------------------------------------------------
# EKF function correctness tests
# ---------------------------------------------------------------------------


class TestEKFFunctions:
    def test_ekf_transition_fn_matches_environment(self, env, initial_cov):
        """Test that the EKF transition function matches Mountain Car deterministic physics.

        Purpose: Validates that the EKF transition function produces the
                 same deterministic next state as the environment's physics.

        Given: An EKF-based belief and a known state-action pair.
        When: transition_fn is called and compared to deterministic environment transition.
        Then: Results match.

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=initial_cov,
        )
        updater = belief.updater
        assert isinstance(updater, ExtendedKalmanFilterUpdater)

        state = np.array([-0.5, 0.01])
        for action_val in [-1, 0, 1]:
            transition = env.state_transition_model(tuple(state), action_val)
            env_next = transition._compute_deterministic_next_state()
            ekf_next = updater.transition_fn(state, np.array([float(action_val)]))
            np.testing.assert_allclose(ekf_next, env_next, atol=1e-12)

    def test_ekf_observation_fn_is_identity(self, env, initial_cov):
        """Test that the EKF observation function is the identity.

        Purpose: Validates the observation function returns the state unchanged.

        Given: An EKF-based belief from the factory.
        When: observation_fn is called with an arbitrary state.
        Then: The result equals the input state.

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=initial_cov,
        )
        updater = belief.updater
        assert isinstance(updater, ExtendedKalmanFilterUpdater)
        state = np.array([-0.3, 0.02])
        np.testing.assert_array_equal(updater.observation_fn(state), state)

    def test_ekf_observation_jacobian_is_identity(self, env, initial_cov):
        """Test that the EKF observation Jacobian is I_2.

        Purpose: Validates the observation Jacobian for the identity observation.

        Given: An EKF-based belief from the factory.
        When: observation_jacobian is evaluated at an arbitrary state.
        Then: Result is np.eye(2).

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=initial_cov,
        )
        updater = belief.updater
        assert isinstance(updater, ExtendedKalmanFilterUpdater)
        H = updater.observation_jacobian(np.array([-0.5, 0.0]))
        np.testing.assert_array_equal(H, np.eye(2))

    def test_ekf_transition_jacobian_near_numerical(self, env, initial_cov):
        """Test that the analytical Jacobian matches numerical differentiation.

        Purpose: Validates the analytical Jacobian against finite differences.

        Given: An EKF-based belief and a state in the interior of the domain
               (away from clipping boundaries).
        When: The analytical Jacobian is computed and compared to numerical.
        Then: They agree within a reasonable tolerance.

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=initial_cov,
        )
        updater = belief.updater
        assert isinstance(updater, ExtendedKalmanFilterUpdater)

        # Use a state well inside boundaries (no clipping active)
        state = np.array([-0.5, 0.01])
        action = np.array([1.0])
        eps = 1e-6

        analytical_J = updater.transition_jacobian(state, action)

        numerical_J = np.zeros((2, 2))
        for i in range(2):
            state_plus = state.copy()
            state_minus = state.copy()
            state_plus[i] += eps
            state_minus[i] -= eps
            f_plus = updater.transition_fn(state_plus, action)
            f_minus = updater.transition_fn(state_minus, action)
            numerical_J[:, i] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(analytical_J, numerical_J, atol=1e-5)


# ---------------------------------------------------------------------------
# UKF function correctness tests
# ---------------------------------------------------------------------------


class TestUKFFunctions:
    def test_ukf_transition_fn_matches_environment(self, env, initial_cov):
        """Test that the UKF transition function matches Mountain Car deterministic physics.

        Purpose: Validates UKF transition function equivalence with environment
                 deterministic physics.

        Given: A UKF-based belief and a known state-action pair.
        When: transition_fn is called.
        Then: Result matches the environment's deterministic transition output.

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.UKF,
            initial_covariance=initial_cov,
        )
        updater = belief.updater
        assert isinstance(updater, UnscentedKalmanFilterUpdater)

        state = np.array([-0.5, 0.01])
        for action_val in [-1, 0, 1]:
            transition = env.state_transition_model(tuple(state), action_val)
            env_next = transition._compute_deterministic_next_state()
            ukf_next = updater.transition_fn(state, np.array([float(action_val)]))
            np.testing.assert_allclose(ukf_next, env_next, atol=1e-12)

    def test_ukf_observation_fn_is_identity(self, env, initial_cov):
        """Test that the UKF observation function is the identity.

        Purpose: Validates the UKF observation function returns state unchanged.

        Given: A UKF-based belief from the factory.
        When: observation_fn is called.
        Then: The result equals the input state.

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.UKF,
            initial_covariance=initial_cov,
        )
        updater = belief.updater
        assert isinstance(updater, UnscentedKalmanFilterUpdater)
        state = np.array([-0.3, 0.02])
        np.testing.assert_array_equal(updater.observation_fn(state), state)


# ---------------------------------------------------------------------------
# Belief attribute tests
# ---------------------------------------------------------------------------


class TestBeliefAttributes:
    def test_mean_is_initial_center(self, env, initial_cov):
        """Test that the returned belief mean is [-0.5, 0.0].

        Purpose: Validates mean initialisation to center of initial distribution.

        Given: A MountainCarPOMDP environment.
        When: Factory creates a belief.
        Then: belief.mean == [-0.5, 0.0].

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=initial_cov,
        )
        np.testing.assert_array_equal(belief.mean, np.array([-0.5, 0.0]))

    def test_covariance_matches_initial(self, env, initial_cov):
        """Test that the returned belief covariance equals initial_covariance.

        Purpose: Validates covariance initialisation.

        Given: initial_covariance = diag([0.01, 0.001]).
        When: Factory creates a belief.
        Then: belief.covariance == initial_covariance.

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=initial_cov,
        )
        np.testing.assert_array_equal(belief.covariance, initial_cov)

    def test_default_covariance(self, env):
        """Test that the default covariance matches expected values.

        Purpose: Validates default covariance when none is provided.

        Given: No initial_covariance argument.
        When: Factory creates a belief with default covariance.
        Then: belief.covariance == np.diag([0.2**2/12, 1e-4]).

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
        )
        expected = np.diag([0.2**2 / 12.0, 1e-4])
        np.testing.assert_allclose(belief.covariance, expected)

    def test_mean_shape_is_2d(self, env, initial_cov):
        """Test that belief mean has shape (2,).

        Purpose: Validates output dimensionality.

        Given: A MountainCarPOMDP (2D state).
        When: Factory creates a belief.
        Then: belief.mean.shape == (2,) and belief.dim == 2.

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.UKF,
            initial_covariance=initial_cov,
        )
        assert belief.mean.shape == (2,)
        assert belief.dim == 2


# ---------------------------------------------------------------------------
# Noise parameter tests
# ---------------------------------------------------------------------------


class TestNoiseParameters:
    def test_ekf_uses_env_cov_matrix_as_R(self, env, initial_cov):
        """Test that the EKF updater R matches env.cov_matrix.

        Purpose: Validates that observation noise is taken from environment.

        Given: A MountainCarPOMDP with known cov_matrix.
        When: EKF belief is created.
        Then: updater.R == env.cov_matrix.

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=initial_cov,
        )
        updater = belief.updater
        assert isinstance(updater, ExtendedKalmanFilterUpdater)
        np.testing.assert_array_equal(updater.R, env.cov_matrix)

    def test_process_noise_scale_controls_Q(self, env, initial_cov):
        """Test that process_noise_scale controls the Q matrix diagonal.

        Purpose: Validates that Q = process_noise_scale * I_2.

        Given: process_noise_scale=0.01.
        When: EKF belief is created.
        Then: updater.Q == 0.01 * np.eye(2).

        Test type: unit
        """
        scale = 0.01
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=initial_cov,
            process_noise_scale=scale,
        )
        updater = belief.updater
        assert isinstance(updater, ExtendedKalmanFilterUpdater)
        np.testing.assert_array_equal(updater.Q, np.eye(2) * scale)


# ---------------------------------------------------------------------------
# End-to-end update tests
# ---------------------------------------------------------------------------


class TestEndToEndUpdate:
    @pytest.mark.parametrize(
        "updater_type",
        [
            GaussianBeliefUpdaterType.EKF,
            GaussianBeliefUpdaterType.UKF,
        ],
    )
    def test_update_produces_valid_gaussian_belief(self, env, initial_cov, updater_type):
        """Test that belief.update returns a valid GaussianBelief with correct shapes.

        Purpose: Validates that a single update step produces well-formed output.

        Given: A Mountain Car GaussianBelief with the given updater type.
        When: update() is called with a plausible action-observation pair.
        Then: The returned belief has correct types and shapes.

        Test type: integration
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=updater_type,
            initial_covariance=initial_cov,
        )
        action = np.array([1.0])
        next_state = env.state_transition_model(tuple(belief.mean), 1).sample()[0]
        observation = next_state + np.random.randn(2) * 0.01

        new_belief = belief.update(action=action, observation=observation, pomdp=None)

        assert isinstance(new_belief, GaussianBelief)
        assert new_belief.mean.shape == (2,)
        assert new_belief.covariance.shape == (2, 2)
        assert np.all(np.isfinite(new_belief.mean))
        assert np.all(np.isfinite(new_belief.covariance))

    @pytest.mark.parametrize(
        "updater_type",
        [
            GaussianBeliefUpdaterType.EKF,
            GaussianBeliefUpdaterType.UKF,
        ],
    )
    def test_update_shrinks_covariance(self, env, updater_type):
        """Test that an observation update reduces uncertainty.

        Purpose: Validates that Bayesian update shrinks the covariance trace.

        Given: A high-uncertainty initial belief.
        When: update() is called with a plausible observation.
        Then: The trace of the posterior covariance is smaller.

        Test type: integration
        """
        large_cov = np.eye(2) * 1.0
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=updater_type,
            initial_covariance=large_cov,
        )
        action = np.array([1.0])
        next_state = env.state_transition_model(tuple(belief.mean), 1).sample()[0]
        observation = next_state

        new_belief = belief.update(action=action, observation=observation, pomdp=None)

        assert np.trace(new_belief.covariance) < np.trace(large_cov)

    def test_ekf_ukf_produce_similar_results(self, env):
        """Test that EKF and UKF produce similar posterior estimates.

        Purpose: Validates consistency between the two nonlinear filters.

        Given: The same initial belief and observation for both EKF and UKF.
        When: Both are updated with the same action-observation pair.
        Then: Posterior means and covariance traces are reasonably close.

        Test type: integration
        """
        cov = np.diag([0.01, 0.001])
        belief_ekf = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=cov,
        )
        belief_ukf = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.UKF,
            initial_covariance=cov,
        )

        action = np.array([1.0])
        next_state = env.state_transition_model((-0.5, 0.0), 1).sample()[0]
        observation = next_state

        new_ekf = belief_ekf.update(action=action, observation=observation, pomdp=None)
        new_ukf = belief_ukf.update(action=action, observation=observation, pomdp=None)

        np.testing.assert_allclose(new_ekf.mean, new_ukf.mean, atol=0.1)
        assert abs(np.trace(new_ekf.covariance) - np.trace(new_ukf.covariance)) < 0.5

    def test_multi_step_update(self, env):
        """Test that multiple sequential belief updates produce valid results.

        Purpose: Validates multi-step belief tracking with environment observations.

        Given: A Mountain Car GaussianBelief with EKF updater.
        When: 5 sequential update steps are performed with environment observations.
        Then: All intermediate beliefs are valid and finite.

        Test type: integration
        """
        np.random.seed(42)
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=np.diag([0.01, 0.001]),
        )

        state = env.initial_state_dist().sample()[0]
        actions = env.get_actions()
        for _ in range(5):
            action_val = np.random.choice(actions)
            next_state, observation, _ = env.sample_next_step(tuple(state), action_val)

            belief = belief.update(
                action=np.array([float(action_val)]),
                observation=observation,
                pomdp=None,
            )
            assert isinstance(belief, GaussianBelief)
            assert np.all(np.isfinite(belief.mean))
            assert np.all(np.isfinite(belief.covariance))
            state = next_state


# ---------------------------------------------------------------------------
# Sampling tests
# ---------------------------------------------------------------------------


class TestSampling:
    def test_sample_returns_correct_shape(self, env, initial_cov):
        """Test that sampling from the Gaussian belief returns shape (2,).

        Purpose: Validates sample output dimensionality.

        Given: A Mountain Car GaussianBelief.
        When: belief.sample() is called.
        Then: The returned state has shape (2,).

        Test type: unit
        """
        belief = create_mountain_car_gaussian_belief(
            env=env,
            updater_type=GaussianBeliefUpdaterType.UKF,
            initial_covariance=initial_cov,
        )
        sample = belief.sample()
        assert sample.shape == (2,)
        assert np.all(np.isfinite(sample))


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
            create_mountain_car_gaussian_belief(
                env=env,
                updater_type=GaussianBeliefUpdaterType("invalid_type"),
                initial_covariance=initial_cov,
            )


# ---------------------------------------------------------------------------
# Belief factory integration tests
# ---------------------------------------------------------------------------


class TestBeliefFactoryIntegration:
    def test_create_mountain_car_belief_gaussian_type(self, env):
        """Test that create_mountain_car_belief dispatches to Gaussian factory.

        Purpose: Validates that the main belief factory supports GAUSSIAN type.

        Given: A MountainCarPOMDP environment.
        When: create_mountain_car_belief is called with BeliefType.GAUSSIAN.
        Then: A GaussianBelief is returned.

        Test type: integration
        """
        from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp_beliefs import (
            create_mountain_car_belief,
        )
        from POMDPPlanners.utils.belief_factory import BeliefType

        belief = create_mountain_car_belief(env, belief_type=BeliefType.GAUSSIAN)
        assert isinstance(belief, GaussianBelief)
        assert belief.mean.shape == (2,)
        assert belief.covariance.shape == (2, 2)

    def test_create_mountain_car_belief_gaussian_with_kwargs(self, env):
        """Test that create_mountain_car_belief forwards kwargs to Gaussian factory.

        Purpose: Validates kwargs forwarding for Gaussian belief creation.

        Given: A MountainCarPOMDP environment.
        When: create_mountain_car_belief is called with GAUSSIAN and custom kwargs.
        Then: The returned belief uses the specified parameters.

        Test type: integration
        """
        from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp_beliefs import (
            create_mountain_car_belief,
        )
        from POMDPPlanners.utils.belief_factory import BeliefType

        custom_cov = np.eye(2) * 2.0
        belief = create_mountain_car_belief(
            env,
            belief_type=BeliefType.GAUSSIAN,
            updater_type=GaussianBeliefUpdaterType.EKF,
            initial_covariance=custom_cov,
        )
        assert isinstance(belief, GaussianBelief)
        assert isinstance(belief.updater, ExtendedKalmanFilterUpdater)
        np.testing.assert_array_equal(belief.covariance, custom_cov)
