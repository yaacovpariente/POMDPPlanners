# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for Gaussian belief updater classes.

This module tests the concrete updater implementations:
- LinearKalmanFilterUpdater: linear-Gaussian systems
- ExtendedKalmanFilterUpdater: nonlinear systems with Jacobians
- UnscentedKalmanFilterUpdater: nonlinear systems without Jacobians
"""

import numpy as np

from POMDPPlanners.core.belief import GaussianBelief
from POMDPPlanners.core.belief.gaussian_belief_updaters import (
    LinearKalmanFilterUpdater,
    ExtendedKalmanFilterUpdater,
    UnscentedKalmanFilterUpdater,
)


# ---------------------------------------------------------------------------
# Linear Kalman Filter tests
# ---------------------------------------------------------------------------


class TestLinearKalmanFilterUpdater:
    def test_1d_static_system(self):
        """Test 1D Kalman filter on a static system with perfect observation.

        Purpose: Validates that the KF converges toward the observation for a static system.

        Given: A=1, B=0, H=1, Q=0.1, R=0.5 (1D static system with noisy observation).
        When: A single predict-correct cycle is performed with observation z=3.0.
        Then: The posterior mean moves toward 3.0 and covariance decreases.

        Test type: unit
        """
        updater = LinearKalmanFilterUpdater(
            A=np.array([[1.0]]),
            B=np.array([[0.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.1]]),
            R=np.array([[0.5]]),
        )
        mean = np.array([0.0])
        cov = np.array([[1.0]])
        action = np.array([0.0])
        observation = np.array([3.0])

        new_mean, new_cov = updater.update(mean, cov, action, observation)

        # Mean should move toward observation
        assert new_mean[0] > 0.0
        assert new_mean[0] < 3.0
        # Covariance should decrease (information gained)
        assert new_cov[0, 0] < cov[0, 0] + 0.1  # prior + process noise

    def test_1d_analytical_values(self):
        """Test 1D Kalman filter against hand-computed analytical solution.

        Purpose: Validates KF predict-correct matches textbook formula.

        Given: A=1, B=0, H=1, Q=0, R=1, prior mean=0, prior cov=1.
        When: Observation z=2.0 is incorporated.
        Then: Posterior mean = 1.0, posterior variance = 0.5.

        Test type: unit
        """
        updater = LinearKalmanFilterUpdater(
            A=np.array([[1.0]]),
            B=np.array([[0.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.0]]),
            R=np.array([[1.0]]),
        )
        mean = np.array([0.0])
        cov = np.array([[1.0]])

        new_mean, new_cov = updater.update(mean, cov, np.array([0.0]), np.array([2.0]))

        # Predicted: mean_pred=0, cov_pred=1 (Q=0)
        # S = 1 + 1 = 2, K = 1/2
        # new_mean = 0 + 0.5*2 = 1.0
        # new_cov = (1 - 0.5)*1 = 0.5
        np.testing.assert_allclose(new_mean, [1.0], atol=1e-12)
        np.testing.assert_allclose(new_cov, [[0.5]], atol=1e-12)

    def test_2d_tracking_covariance_decreases(self):
        """Test 2D Kalman filter covariance reduction over multiple steps.

        Purpose: Validates that repeated observations reduce uncertainty.

        Given: 2D identity system with Q=0.01*I and R=0.5*I.
        When: 10 observations at [1, 1] are incorporated sequentially.
        Then: Covariance trace decreases monotonically.

        Test type: unit
        """
        updater = LinearKalmanFilterUpdater(
            A=np.eye(2),
            B=np.zeros((2, 1)),
            H=np.eye(2),
            Q=0.01 * np.eye(2),
            R=0.5 * np.eye(2),
        )
        mean = np.zeros(2)
        cov = np.eye(2)
        prev_trace = np.trace(cov)

        for _ in range(10):
            mean, cov = updater.update(mean, cov, np.array([0.0]), np.array([1.0, 1.0]))
            curr_trace = np.trace(cov)
            assert curr_trace < prev_trace + 0.02 + 1e-10  # allow for process noise
            prev_trace = curr_trace

    def test_with_control_input(self):
        """Test Kalman filter with a non-zero control input.

        Purpose: Validates that the control matrix B shifts the predicted mean.

        Given: A=1, B=1, H=1, Q=0, R=1, prior mean=0, prior cov=1.
        When: Action u=5 and observation z=5 are applied.
        Then: Posterior mean is close to 5.

        Test type: unit
        """
        updater = LinearKalmanFilterUpdater(
            A=np.array([[1.0]]),
            B=np.array([[1.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.0]]),
            R=np.array([[1.0]]),
        )
        mean = np.array([0.0])
        cov = np.array([[1.0]])

        new_mean, _ = updater.update(mean, cov, np.array([5.0]), np.array([5.0]))

        # Predicted mean = 0 + 1*5 = 5, S=1+1=2, K=0.5
        # new_mean = 5 + 0.5*(5-5) = 5.0
        np.testing.assert_allclose(new_mean, [5.0], atol=1e-12)

    def test_integration_with_gaussian_belief(self):
        """Test that the linear KF updater works with GaussianBelief.update().

        Purpose: Validates end-to-end integration of the updater class with GaussianBelief.

        Given: A GaussianBelief using a linear KF updater.
        When: belief.update() is called.
        Then: Returns a new GaussianBelief with updated mean and covariance.

        Test type: integration
        """
        updater = LinearKalmanFilterUpdater(
            A=np.eye(2),
            B=np.zeros((2, 1)),
            H=np.eye(2),
            Q=0.1 * np.eye(2),
            R=0.5 * np.eye(2),
        )
        belief = GaussianBelief(mean=np.zeros(2), covariance=np.eye(2), updater=updater)
        new_belief = belief.update(
            action=np.array([0.0]),
            observation=np.array([1.0, 2.0]),
        )
        assert isinstance(new_belief, GaussianBelief)
        assert new_belief.dim == 2
        assert new_belief.mean[0] > 0.0
        assert new_belief.mean[1] > 0.0

    def test_config_id_deterministic(self):
        """Test that config_id is deterministic for identical updaters.

        Purpose: Validates that the same parameters produce the same config_id.

        Given: Two LinearKalmanFilterUpdater instances with identical parameters.
        When: config_id is computed for both.
        Then: Both config_ids are equal.

        Test type: unit
        """
        u1 = LinearKalmanFilterUpdater(
            A=np.eye(2),
            B=np.zeros((2, 1)),
            H=np.eye(2),
            Q=0.1 * np.eye(2),
            R=0.5 * np.eye(2),
        )
        u2 = LinearKalmanFilterUpdater(
            A=np.eye(2),
            B=np.zeros((2, 1)),
            H=np.eye(2),
            Q=0.1 * np.eye(2),
            R=0.5 * np.eye(2),
        )
        assert u1.config_id == u2.config_id

    def test_config_id_sensitive_to_parameters(self):
        """Test that config_id differs when parameters differ.

        Purpose: Validates config_id sensitivity to matrix values.

        Given: Two LinearKalmanFilterUpdater instances with different Q matrices.
        When: config_id is computed for both.
        Then: The config_ids are different.

        Test type: unit
        """
        u1 = LinearKalmanFilterUpdater(
            A=np.eye(2),
            B=np.zeros((2, 1)),
            H=np.eye(2),
            Q=0.1 * np.eye(2),
            R=0.5 * np.eye(2),
        )
        u2 = LinearKalmanFilterUpdater(
            A=np.eye(2),
            B=np.zeros((2, 1)),
            H=np.eye(2),
            Q=0.2 * np.eye(2),
            R=0.5 * np.eye(2),
        )
        assert u1.config_id != u2.config_id


# ---------------------------------------------------------------------------
# Extended Kalman Filter tests
# ---------------------------------------------------------------------------


class TestExtendedKalmanFilterUpdater:
    def test_linear_system_matches_kf(self):
        """Test that the EKF matches the linear KF on a linear system.

        Purpose: Validates that the EKF reduces to the standard KF for linear models.

        Given: A linear system represented as both KF and EKF.
        When: Both are updated with the same observation.
        Then: Posterior means and covariances match within numerical tolerance.

        Test type: unit
        """
        A = np.eye(2)
        B = np.zeros((2, 1))
        H = np.eye(2)
        Q = 0.1 * np.eye(2)
        R = 0.5 * np.eye(2)

        kf_updater = LinearKalmanFilterUpdater(A=A, B=B, H=H, Q=Q, R=R)
        ekf_updater = ExtendedKalmanFilterUpdater(
            transition_fn=lambda x, u: A @ x + B @ u,
            observation_fn=lambda x: H @ x,
            transition_jacobian=lambda x, u: A,
            observation_jacobian=lambda x: H,
            Q=Q,
            R=R,
        )

        mean = np.array([1.0, -1.0])
        cov = 2.0 * np.eye(2)
        action = np.array([0.0])
        obs = np.array([2.0, 3.0])

        kf_mean, kf_cov = kf_updater.update(mean, cov, action, obs)
        ekf_mean, ekf_cov = ekf_updater.update(mean, cov, action, obs)

        np.testing.assert_allclose(ekf_mean, kf_mean, atol=1e-12)
        np.testing.assert_allclose(ekf_cov, kf_cov, atol=1e-12)

    def test_nonlinear_system_reduces_covariance(self):
        """Test EKF on a nonlinear system reduces covariance after observation.

        Purpose: Validates that the EKF incorporates observations to reduce uncertainty.

        Given: A nonlinear system f(x,u)=x, h(x)=x^3 (cube observation).
        When: An observation is incorporated.
        Then: Posterior covariance trace is smaller than predicted covariance trace.

        Test type: unit
        """
        ekf_updater = ExtendedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x**3,
            transition_jacobian=lambda x, u: np.eye(len(x)),
            observation_jacobian=lambda x: 3.0 * np.diag(x**2),
            Q=0.01 * np.eye(1),
            R=0.1 * np.eye(1),
        )
        mean = np.array([1.0])
        cov = np.array([[0.5]])

        _, new_cov = ekf_updater.update(mean, cov, np.array([0.0]), np.array([1.0]))

        # Predicted cov = cov + Q = 0.51
        # After correction, cov should be less than predicted
        assert new_cov[0, 0] < 0.51

    def test_integration_with_gaussian_belief(self):
        """Test that the EKF updater works with GaussianBelief.update().

        Purpose: Validates end-to-end integration of the EKF class with GaussianBelief.

        Given: A GaussianBelief using an EKF updater for a linear system.
        When: belief.update() is called.
        Then: Returns a new GaussianBelief with updated parameters.

        Test type: integration
        """
        ekf_updater = ExtendedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x,
            transition_jacobian=lambda x, u: np.eye(len(x)),
            observation_jacobian=lambda x: np.eye(len(x)),
            Q=0.1 * np.eye(2),
            R=0.5 * np.eye(2),
        )
        belief = GaussianBelief(mean=np.zeros(2), covariance=np.eye(2), updater=ekf_updater)
        new_belief = belief.update(
            action=np.array([0.0]),
            observation=np.array([1.0, 2.0]),
        )
        assert isinstance(new_belief, GaussianBelief)
        assert new_belief.dim == 2

    def test_ekf_covariance_is_symmetric(self):
        """Test that the EKF posterior covariance is symmetric.

        Purpose: Validates the symmetrization step in the EKF implementation.

        Given: A nonlinear EKF system.
        When: An update is performed.
        Then: The resulting covariance is symmetric.

        Test type: unit
        """
        ekf_updater = ExtendedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x**2,
            transition_jacobian=lambda x, u: np.eye(len(x)),
            observation_jacobian=lambda x: 2.0 * np.diag(x),
            Q=0.1 * np.eye(2),
            R=0.5 * np.eye(2),
        )
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])

        _, new_cov = ekf_updater.update(mean, cov, np.array([0.0]), np.array([1.0, 4.0]))

        np.testing.assert_allclose(new_cov, new_cov.T, atol=1e-12)

    def test_config_id_deterministic(self):
        """Test that config_id is deterministic for identical EKF updaters.

        Purpose: Validates that the same Q and R produce the same config_id.

        Given: Two ExtendedKalmanFilterUpdater instances with identical Q and R.
        When: config_id is computed for both.
        Then: Both config_ids are equal.

        Test type: unit
        """
        Q = 0.1 * np.eye(2)
        R = 0.5 * np.eye(2)
        u1 = ExtendedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x,
            transition_jacobian=lambda x, u: np.eye(len(x)),
            observation_jacobian=lambda x: np.eye(len(x)),
            Q=Q,
            R=R,
        )
        u2 = ExtendedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x,
            transition_jacobian=lambda x, u: np.eye(len(x)),
            observation_jacobian=lambda x: np.eye(len(x)),
            Q=Q,
            R=R,
        )
        assert u1.config_id == u2.config_id

    def test_config_id_sensitive_to_parameters(self):
        """Test that config_id differs when Q differs.

        Purpose: Validates config_id sensitivity to noise parameters.

        Given: Two ExtendedKalmanFilterUpdater instances with different Q matrices.
        When: config_id is computed for both.
        Then: The config_ids are different.

        Test type: unit
        """
        R = 0.5 * np.eye(2)
        u1 = ExtendedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x,
            transition_jacobian=lambda x, u: np.eye(len(x)),
            observation_jacobian=lambda x: np.eye(len(x)),
            Q=0.1 * np.eye(2),
            R=R,
        )
        u2 = ExtendedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x,
            transition_jacobian=lambda x, u: np.eye(len(x)),
            observation_jacobian=lambda x: np.eye(len(x)),
            Q=0.2 * np.eye(2),
            R=R,
        )
        assert u1.config_id != u2.config_id


# ---------------------------------------------------------------------------
# Unscented Kalman Filter tests
# ---------------------------------------------------------------------------


class TestUnscentedKalmanFilterUpdater:
    def test_linear_system_matches_kf(self):
        """Test that the UKF matches the linear KF on a linear system.

        Purpose: Validates that the UKF reduces to the standard KF for linear models.

        Given: A linear system represented as both KF and UKF.
        When: Both are updated with the same observation.
        Then: Posterior means and covariances match within numerical tolerance.

        Test type: unit
        """
        A = np.eye(2)
        B = np.zeros((2, 1))
        H = np.eye(2)
        Q = 0.1 * np.eye(2)
        R = 0.5 * np.eye(2)

        kf_updater = LinearKalmanFilterUpdater(A=A, B=B, H=H, Q=Q, R=R)
        ukf_updater = UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: A @ x + B @ u,
            observation_fn=lambda x: H @ x,
            Q=Q,
            R=R,
        )

        mean = np.array([1.0, -1.0])
        cov = 2.0 * np.eye(2)
        action = np.array([0.0])
        obs = np.array([2.0, 3.0])

        kf_mean, kf_cov = kf_updater.update(mean, cov, action, obs)
        ukf_mean, ukf_cov = ukf_updater.update(mean, cov, action, obs)

        np.testing.assert_allclose(ukf_mean, kf_mean, atol=1e-6)
        np.testing.assert_allclose(ukf_cov, kf_cov, atol=1e-6)

    def test_nonlinear_system_reduces_covariance(self):
        """Test UKF on a nonlinear system reduces covariance after observation.

        Purpose: Validates that the UKF incorporates observations to reduce uncertainty.

        Given: A nonlinear system f(x,u)=x, h(x)=x^3 (cube observation).
        When: An observation is incorporated.
        Then: Posterior covariance trace is smaller than predicted covariance trace.

        Test type: unit
        """
        ukf_updater = UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x**3,
            Q=0.01 * np.eye(1),
            R=0.1 * np.eye(1),
        )
        mean = np.array([1.0])
        cov = np.array([[0.5]])

        new_mean, new_cov = ukf_updater.update(mean, cov, np.array([0.0]), np.array([1.0]))

        # Predicted cov = cov + Q = 0.51
        # After correction, cov should be less than predicted
        assert new_cov[0, 0] < 0.51
        assert np.isfinite(new_mean[0])

    def test_integration_with_gaussian_belief(self):
        """Test that the UKF updater works with GaussianBelief.update().

        Purpose: Validates end-to-end integration of the UKF class with GaussianBelief.

        Given: A GaussianBelief using a UKF updater for a linear system.
        When: belief.update() is called.
        Then: Returns a new GaussianBelief with updated parameters.

        Test type: integration
        """
        ukf_updater = UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x,
            Q=0.1 * np.eye(2),
            R=0.5 * np.eye(2),
        )
        belief = GaussianBelief(mean=np.zeros(2), covariance=np.eye(2), updater=ukf_updater)
        new_belief = belief.update(
            action=np.array([0.0]),
            observation=np.array([1.0, 2.0]),
        )
        assert isinstance(new_belief, GaussianBelief)
        assert new_belief.dim == 2

    def test_ukf_covariance_is_symmetric(self):
        """Test that the UKF posterior covariance is symmetric.

        Purpose: Validates the symmetrization step in the UKF implementation.

        Given: A nonlinear UKF system with off-diagonal covariance.
        When: An update is performed.
        Then: The resulting covariance is symmetric.

        Test type: unit
        """
        ukf_updater = UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x**2,
            Q=0.1 * np.eye(2),
            R=0.5 * np.eye(2),
        )
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])

        _, new_cov = ukf_updater.update(mean, cov, np.array([0.0]), np.array([1.0, 4.0]))

        np.testing.assert_allclose(new_cov, new_cov.T, atol=1e-12)

    def test_ukf_matches_ekf_on_linear_system(self):
        """Test that UKF and EKF produce identical results on a linear system.

        Purpose: Validates UKF-EKF equivalence when the system is linear.

        Given: The same linear system represented as both EKF and UKF.
        When: Both are updated with identical inputs.
        Then: Posterior means and covariances match within tolerance.

        Test type: unit
        """
        Q = 0.1 * np.eye(2)
        R = 0.5 * np.eye(2)

        ekf_updater = ExtendedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x,
            transition_jacobian=lambda x, u: np.eye(len(x)),
            observation_jacobian=lambda x: np.eye(len(x)),
            Q=Q,
            R=R,
        )
        ukf_updater = UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x,
            Q=Q,
            R=R,
        )

        mean = np.array([1.0, -1.0])
        cov = 2.0 * np.eye(2)
        action = np.array([0.0])
        obs = np.array([2.0, 3.0])

        ekf_mean, ekf_cov = ekf_updater.update(mean, cov, action, obs)
        ukf_mean, ukf_cov = ukf_updater.update(mean, cov, action, obs)

        np.testing.assert_allclose(ukf_mean, ekf_mean, atol=1e-6)
        np.testing.assert_allclose(ukf_cov, ekf_cov, atol=1e-6)

    def test_sigma_point_scaling_parameters(self):
        """Test UKF with custom alpha, beta, kappa scaling parameters.

        Purpose: Validates that non-default sigma point parameters produce valid results.

        Given: A UKF with alpha=0.5, beta=2.0, kappa=1.0.
        When: An update is performed.
        Then: Output mean has correct shape and covariance is symmetric PSD.

        Test type: unit
        """
        ukf_updater = UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x,
            Q=0.1 * np.eye(2),
            R=0.5 * np.eye(2),
            alpha=0.5,
            beta=2.0,
            kappa=1.0,
        )
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)

        new_mean, new_cov = ukf_updater.update(mean, cov, np.array([0.0]), np.array([1.0, 1.0]))

        assert new_mean.shape == (2,)
        np.testing.assert_allclose(new_cov, new_cov.T, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(new_cov)
        assert np.all(eigenvalues > 0)

    def test_higher_dimensional_system(self):
        """Test UKF on a higher-dimensional system (d=5 state, p=3 observation).

        Purpose: Validates that the UKF handles dimension mismatches between
            state and observation spaces correctly.

        Given: A 5D state system with 3D observations via a projection matrix.
        When: An update is performed.
        Then: State mean is 5D, covariance is 5x5, and covariance is symmetric PSD.

        Test type: unit
        """
        d, p = 5, 3
        H = np.random.RandomState(42).randn(p, d)

        ukf_updater = UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: H @ x,
            Q=0.01 * np.eye(d),
            R=0.1 * np.eye(p),
        )
        mean = np.zeros(d)
        cov = np.eye(d)

        obs = np.ones(p)
        new_mean, new_cov = ukf_updater.update(mean, cov, np.zeros(1), obs)

        assert new_mean.shape == (d,)
        assert new_cov.shape == (d, d)
        np.testing.assert_allclose(new_cov, new_cov.T, atol=1e-10)
        eigenvalues = np.linalg.eigvalsh(new_cov)
        assert np.all(eigenvalues > 0)

    def test_config_id_deterministic(self):
        """Test that config_id is deterministic for identical UKF updaters.

        Purpose: Validates that the same parameters produce the same config_id.

        Given: Two UnscentedKalmanFilterUpdater instances with identical parameters.
        When: config_id is computed for both.
        Then: Both config_ids are equal.

        Test type: unit
        """
        Q = 0.1 * np.eye(2)
        R = 0.5 * np.eye(2)
        u1 = UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x,
            Q=Q,
            R=R,
            alpha=1e-3,
            beta=2.0,
            kappa=0.0,
        )
        u2 = UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x,
            Q=Q,
            R=R,
            alpha=1e-3,
            beta=2.0,
            kappa=0.0,
        )
        assert u1.config_id == u2.config_id

    def test_config_id_sensitive_to_parameters(self):
        """Test that config_id differs when scaling parameters differ.

        Purpose: Validates config_id sensitivity to alpha/beta/kappa.

        Given: Two UnscentedKalmanFilterUpdater instances with different alpha.
        When: config_id is computed for both.
        Then: The config_ids are different.

        Test type: unit
        """
        Q = 0.1 * np.eye(2)
        R = 0.5 * np.eye(2)
        u1 = UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x,
            Q=Q,
            R=R,
            alpha=1e-3,
        )
        u2 = UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: x,
            observation_fn=lambda x: x,
            Q=Q,
            R=R,
            alpha=0.5,
        )
        assert u1.config_id != u2.config_id
