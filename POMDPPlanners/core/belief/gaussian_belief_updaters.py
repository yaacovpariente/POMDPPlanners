"""Pre-built Gaussian belief updater factories.

This module provides factory functions that return ``GaussianBeliefUpdater``
callables for common Bayesian filtering algorithms. Each factory captures
the system model parameters in a closure so that the returned callable has
the ``(mean, covariance, action, observation, pomdp)`` signature expected
by :class:`~POMDPPlanners.core.belief.GaussianBelief`.

Functions:
    linear_kalman_filter_updater: Factory for linear-Gaussian systems.
    extended_kalman_filter_updater: Factory for nonlinear systems with known Jacobians.
"""

from typing import Any, Callable, Tuple

import numpy as np


def linear_kalman_filter_updater(
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray, Any, Any, Any], Tuple[np.ndarray, np.ndarray]]:
    """Create a Kalman filter updater for a linear-Gaussian system.

    The system model is:

        x_{t+1} = A x_t + B u_t + w,   w ~ N(0, Q)
        z_t     = H x_{t+1} + v,        v ~ N(0, R)

    Args:
        A: State transition matrix of shape (d, d).
        B: Control input matrix of shape (d, m).
        H: Observation matrix of shape (p, d).
        Q: Process noise covariance of shape (d, d).
        R: Observation noise covariance of shape (p, p).

    Returns:
        A callable suitable for ``GaussianBelief(updater=...)``.

    Example:
        >>> import numpy as np
        >>> A = np.eye(2)
        >>> B = np.zeros((2, 1))
        >>> H = np.eye(2)
        >>> Q = 0.1 * np.eye(2)
        >>> R = 0.5 * np.eye(2)
        >>> updater = linear_kalman_filter_updater(A, B, H, Q, R)
        >>> mean = np.zeros(2)
        >>> cov = np.eye(2)
        >>> new_mean, new_cov = updater(mean, cov, np.zeros(1), np.array([1.0, 0.0]), None)
        >>> new_mean.shape
        (2,)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    H = np.asarray(H, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    def _update(mean, covariance, action, observation, _pomdp):
        action = np.asarray(action, dtype=float).ravel()
        observation = np.asarray(observation, dtype=float).ravel()

        predicted_mean, predicted_cov = _predict(mean, covariance, action)
        return _correct(predicted_mean, predicted_cov, observation)

    def _predict(mean, covariance, action):
        predicted_mean = A @ mean + B @ action
        predicted_cov = A @ covariance @ A.T + Q
        return predicted_mean, predicted_cov

    def _correct(predicted_mean, predicted_cov, observation):
        innovation = observation - H @ predicted_mean
        S = H @ predicted_cov @ H.T + R
        K = predicted_cov @ H.T @ np.linalg.inv(S)
        new_mean = predicted_mean + K @ innovation
        new_cov = (np.eye(len(predicted_mean)) - K @ H) @ predicted_cov
        new_cov = 0.5 * (new_cov + new_cov.T)
        return new_mean, new_cov

    return _update


def extended_kalman_filter_updater(
    transition_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    observation_fn: Callable[[np.ndarray], np.ndarray],
    transition_jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray],
    observation_jacobian: Callable[[np.ndarray], np.ndarray],
    Q: np.ndarray,
    R: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray, Any, Any, Any], Tuple[np.ndarray, np.ndarray]]:
    """Create an Extended Kalman Filter updater for nonlinear systems.

    The system model is:

        x_{t+1} = f(x_t, u_t) + w,   w ~ N(0, Q)
        z_t     = h(x_{t+1}) + v,     v ~ N(0, R)

    The EKF linearises around the current estimate using the provided Jacobians.

    Args:
        transition_fn: State transition function ``f(state, action) -> next_state``.
        observation_fn: Observation function ``h(state) -> observation``.
        transition_jacobian: Jacobian of ``f`` w.r.t. state, ``F(state, action) -> (d, d)``.
        observation_jacobian: Jacobian of ``h`` w.r.t. state, ``H(state) -> (p, d)``.
        Q: Process noise covariance of shape (d, d).
        R: Observation noise covariance of shape (p, p).

    Returns:
        A callable suitable for ``GaussianBelief(updater=...)``.

    Example:
        >>> import numpy as np
        >>> f = lambda x, u: x
        >>> h = lambda x: x
        >>> F = lambda x, u: np.eye(len(x))
        >>> H = lambda x: np.eye(len(x))
        >>> Q = 0.1 * np.eye(2)
        >>> R = 0.5 * np.eye(2)
        >>> updater = extended_kalman_filter_updater(f, h, F, H, Q, R)
        >>> mean = np.zeros(2)
        >>> cov = np.eye(2)
        >>> new_mean, new_cov = updater(mean, cov, np.zeros(1), np.array([1.0, 0.0]), None)
        >>> new_mean.shape
        (2,)
    """
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    def _update(mean, covariance, action, observation, _pomdp):
        action = np.asarray(action, dtype=float).ravel()
        observation = np.asarray(observation, dtype=float).ravel()

        predicted_mean, predicted_cov = _predict(mean, covariance, action)
        return _correct(predicted_mean, predicted_cov, observation)

    def _predict(mean, covariance, action):
        predicted_mean = transition_fn(mean, action)
        F = transition_jacobian(mean, action)
        predicted_cov = F @ covariance @ F.T + Q
        return predicted_mean, predicted_cov

    def _correct(predicted_mean, predicted_cov, observation):
        H = observation_jacobian(predicted_mean)
        innovation = observation - observation_fn(predicted_mean)
        S = H @ predicted_cov @ H.T + R
        K = predicted_cov @ H.T @ np.linalg.inv(S)
        new_mean = predicted_mean + K @ innovation
        new_cov = (np.eye(len(predicted_mean)) - K @ H) @ predicted_cov
        new_cov = 0.5 * (new_cov + new_cov.T)
        return new_mean, new_cov

    return _update
