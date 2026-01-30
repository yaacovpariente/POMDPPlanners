"""Gaussian belief updater abstract base class and concrete implementations.

This module provides the ``GaussianBeliefUpdater`` ABC and three concrete
implementations for common Bayesian filtering algorithms. Each updater
captures its system model parameters at construction time, so the
``update`` method only requires the current belief statistics and the
latest action-observation pair.

Classes:
    GaussianBeliefUpdater: Abstract base class for Gaussian belief updaters.
    LinearKalmanFilterUpdater: Updater for linear-Gaussian systems.
    ExtendedKalmanFilterUpdater: Updater for nonlinear systems with known Jacobians.
    UnscentedKalmanFilterUpdater: Updater for nonlinear systems without Jacobians.
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np

from POMDPPlanners.utils.config_to_id import config_to_id


class GaussianBeliefUpdater(ABC):
    """Abstract base class for Gaussian belief updaters.

    Subclasses implement a Bayesian predict-correct cycle that maps
    ``(mean, covariance, action, observation)`` to an updated
    ``(new_mean, new_covariance)`` pair.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    @abstractmethod
    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a single predict-correct belief update.

        Args:
            mean: Prior mean vector of shape (d,).
            covariance: Prior covariance matrix of shape (d, d).
            action: Action that was executed.
            observation: Observation that was received.

        Returns:
            A tuple ``(new_mean, new_covariance)`` representing the
            posterior Gaussian.
        """

    @property
    @abstractmethod
    def config_id(self) -> str:
        """Return a deterministic identifier for this updater configuration."""


class LinearKalmanFilterUpdater(GaussianBeliefUpdater):
    """Kalman filter updater for a linear-Gaussian system.

    The system model is:

        x_{t+1} = A x_t + B u_t + w,   w ~ N(0, Q)
        z_t     = H x_{t+1} + v,        v ~ N(0, R)

    Attributes:
        A: State transition matrix of shape (d, d).
        B: Control input matrix of shape (d, m).
        H: Observation matrix of shape (p, d).
        Q: Process noise covariance of shape (d, d).
        R: Observation noise covariance of shape (p, p).

    Example:
        >>> import numpy as np
        >>> A = np.eye(2)
        >>> B = np.zeros((2, 1))
        >>> H = np.eye(2)
        >>> Q = 0.1 * np.eye(2)
        >>> R = 0.5 * np.eye(2)
        >>> updater = LinearKalmanFilterUpdater(A=A, B=B, H=H, Q=Q, R=R)
        >>> mean = np.zeros(2)
        >>> cov = np.eye(2)
        >>> new_mean, new_cov = updater.update(mean, cov, np.zeros(1), np.array([1.0, 0.0]))
        >>> new_mean.shape
        (2,)
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

    def update(self, mean, covariance, action, observation):
        action = np.asarray(action, dtype=float).ravel()
        observation = np.asarray(observation, dtype=float).ravel()
        predicted_mean, predicted_cov = self._predict(mean, covariance, action)
        return self._correct(predicted_mean, predicted_cov, observation)

    def _predict(self, mean, covariance, action):
        predicted_mean = self.A @ mean + self.B @ action
        predicted_cov = self.A @ covariance @ self.A.T + self.Q
        return predicted_mean, predicted_cov

    def _correct(self, predicted_mean, predicted_cov, observation):
        innovation = observation - self.H @ predicted_mean
        S = self.H @ predicted_cov @ self.H.T + self.R
        K = predicted_cov @ self.H.T @ np.linalg.inv(S)
        new_mean = predicted_mean + K @ innovation
        new_cov = (np.eye(len(predicted_mean)) - K @ self.H) @ predicted_cov
        new_cov = 0.5 * (new_cov + new_cov.T)
        return new_mean, new_cov

    @property
    def config_id(self) -> str:
        config_dict = {
            "class": "LinearKalmanFilterUpdater",
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "H": self.H.tolist(),
            "Q": self.Q.tolist(),
            "R": self.R.tolist(),
        }
        return config_to_id(config_dict)


class ExtendedKalmanFilterUpdater(GaussianBeliefUpdater):
    """Extended Kalman Filter updater for nonlinear systems.

    The system model is:

        x_{t+1} = f(x_t, u_t) + w,   w ~ N(0, Q)
        z_t     = h(x_{t+1}) + v,     v ~ N(0, R)

    The EKF linearises around the current estimate using the provided
    Jacobians.

    Attributes:
        transition_fn: State transition function ``f(state, action) -> next_state``.
        observation_fn: Observation function ``h(state) -> observation``.
        transition_jacobian: Jacobian of ``f`` w.r.t. state, ``F(state, action) -> (d, d)``.
        observation_jacobian: Jacobian of ``h`` w.r.t. state, ``H(state) -> (p, d)``.
        Q: Process noise covariance of shape (d, d).
        R: Observation noise covariance of shape (p, p).

    Example:
        >>> import numpy as np
        >>> f = lambda x, u: x
        >>> h = lambda x: x
        >>> F = lambda x, u: np.eye(len(x))
        >>> H = lambda x: np.eye(len(x))
        >>> Q = 0.1 * np.eye(2)
        >>> R = 0.5 * np.eye(2)
        >>> updater = ExtendedKalmanFilterUpdater(
        ...     transition_fn=f, observation_fn=h,
        ...     transition_jacobian=F, observation_jacobian=H, Q=Q, R=R,
        ... )
        >>> mean = np.zeros(2)
        >>> cov = np.eye(2)
        >>> new_mean, new_cov = updater.update(mean, cov, np.zeros(1), np.array([1.0, 0.0]))
        >>> new_mean.shape
        (2,)
    """

    def __init__(
        self,
        transition_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        observation_fn: Callable[[np.ndarray], np.ndarray],
        transition_jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray],
        observation_jacobian: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
    ):
        self.transition_fn = transition_fn
        self.observation_fn = observation_fn
        self.transition_jacobian = transition_jacobian
        self.observation_jacobian = observation_jacobian
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

    def update(self, mean, covariance, action, observation):
        action = np.asarray(action, dtype=float).ravel()
        observation = np.asarray(observation, dtype=float).ravel()
        predicted_mean, predicted_cov = self._predict(mean, covariance, action)
        return self._correct(predicted_mean, predicted_cov, observation)

    def _predict(self, mean, covariance, action):
        predicted_mean = self.transition_fn(mean, action)
        F = self.transition_jacobian(mean, action)
        predicted_cov = F @ covariance @ F.T + self.Q
        return predicted_mean, predicted_cov

    def _correct(self, predicted_mean, predicted_cov, observation):
        H = self.observation_jacobian(predicted_mean)
        innovation = observation - self.observation_fn(predicted_mean)
        S = H @ predicted_cov @ H.T + self.R
        K = predicted_cov @ H.T @ np.linalg.inv(S)
        new_mean = predicted_mean + K @ innovation
        new_cov = (np.eye(len(predicted_mean)) - K @ H) @ predicted_cov
        new_cov = 0.5 * (new_cov + new_cov.T)
        return new_mean, new_cov

    @property
    def config_id(self) -> str:
        config_dict = {
            "class": "ExtendedKalmanFilterUpdater",
            "Q": self.Q.tolist(),
            "R": self.R.tolist(),
        }
        return config_to_id(config_dict)


class UnscentedKalmanFilterUpdater(GaussianBeliefUpdater):
    """Unscented Kalman Filter updater for nonlinear systems.

    The system model is:

        x_{t+1} = f(x_t, u_t) + w,   w ~ N(0, Q)
        z_t     = h(x_{t+1}) + v,     v ~ N(0, R)

    Unlike the EKF, the UKF does not require Jacobians. Instead, it
    propagates deterministic sigma points through the nonlinear functions
    to estimate the posterior statistics.

    Attributes:
        transition_fn: State transition function ``f(state, action) -> next_state``.
        observation_fn: Observation function ``h(state) -> observation``.
        Q: Process noise covariance of shape (d, d).
        R: Observation noise covariance of shape (p, p).
        alpha: Spread of sigma points around the mean.
        beta: Prior knowledge about the distribution (2.0 is optimal for Gaussian).
        kappa: Secondary scaling parameter.

    Example:
        >>> import numpy as np
        >>> f = lambda x, u: x
        >>> h = lambda x: x
        >>> Q = 0.1 * np.eye(2)
        >>> R = 0.5 * np.eye(2)
        >>> updater = UnscentedKalmanFilterUpdater(
        ...     transition_fn=f, observation_fn=h, Q=Q, R=R,
        ... )
        >>> mean = np.zeros(2)
        >>> cov = np.eye(2)
        >>> new_mean, new_cov = updater.update(mean, cov, np.zeros(1), np.array([1.0, 0.0]))
        >>> new_mean.shape
        (2,)
    """

    def __init__(
        self,
        transition_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        observation_fn: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.transition_fn = transition_fn
        self.observation_fn = observation_fn
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def update(self, mean, covariance, action, observation):
        action = np.asarray(action, dtype=float).ravel()
        observation = np.asarray(observation, dtype=float).ravel()
        predicted_mean, predicted_cov = self._predict(mean, covariance, action)
        return self._correct(predicted_mean, predicted_cov, observation)

    def _compute_sigma_points(self, mean, covariance):
        d = len(mean)
        lam = self.alpha**2 * (d + self.kappa) - d
        scaling = d + lam

        W_m = np.full(2 * d + 1, 1.0 / (2.0 * scaling))
        W_c = np.full(2 * d + 1, 1.0 / (2.0 * scaling))
        W_m[0] = lam / scaling
        W_c[0] = lam / scaling + (1.0 - self.alpha**2 + self.beta)

        sqrt_matrix = np.linalg.cholesky(scaling * covariance)

        sigma_points = np.empty((2 * d + 1, d))
        sigma_points[0] = mean
        for i in range(d):
            sigma_points[i + 1] = mean + sqrt_matrix[:, i]
            sigma_points[d + i + 1] = mean - sqrt_matrix[:, i]

        return sigma_points, W_m, W_c

    def _predict(self, mean, covariance, action):
        sigma_points, W_m, W_c = self._compute_sigma_points(mean, covariance)
        propagated = np.array(
            [self.transition_fn(sigma_points[i], action) for i in range(len(sigma_points))]
        )

        predicted_mean = W_m @ propagated
        diff = propagated - predicted_mean
        predicted_cov = (diff * W_c[:, None]).T @ diff + self.Q
        predicted_cov = 0.5 * (predicted_cov + predicted_cov.T)

        return predicted_mean, predicted_cov

    def _correct(self, predicted_mean, predicted_cov, observation):
        sigma_points, W_m, W_c = self._compute_sigma_points(predicted_mean, predicted_cov)
        obs_sigmas = np.array(
            [self.observation_fn(sigma_points[i]) for i in range(len(sigma_points))]
        )

        z_mean = W_m @ obs_sigmas
        z_diff = obs_sigmas - z_mean
        S = (z_diff * W_c[:, None]).T @ z_diff + self.R
        x_diff = sigma_points - predicted_mean
        Pxz = (x_diff * W_c[:, None]).T @ z_diff

        K = Pxz @ np.linalg.inv(S)
        new_mean = predicted_mean + K @ (observation - z_mean)
        new_cov = predicted_cov - K @ S @ K.T
        new_cov = 0.5 * (new_cov + new_cov.T)
        return new_mean, new_cov

    @property
    def config_id(self) -> str:
        config_dict = {
            "class": "UnscentedKalmanFilterUpdater",
            "Q": self.Q.tolist(),
            "R": self.R.tolist(),
            "alpha": self.alpha,
            "beta": self.beta,
            "kappa": self.kappa,
        }
        return config_to_id(config_dict)
