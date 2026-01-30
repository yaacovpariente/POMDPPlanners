"""Gaussian belief state representation for POMDP environments.

This module provides a multivariate Gaussian belief state that delegates
updates to a user-provided callable, allowing compatibility with EKF, UKF,
or any custom Gaussian update rule.

Classes:
    GaussianBelief: Multivariate Gaussian belief with callable updater

Type Aliases:
    GaussianBeliefUpdater: Callable signature for Gaussian belief update functions
"""

from typing import Any, Callable, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief.base_belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.multivariate_normal import (
    CovarianceParameterizedMultivariateNormal,
)

GaussianBeliefUpdater = Callable[
    [np.ndarray, np.ndarray, Any, Any, Optional["Environment"]],
    Tuple[np.ndarray, np.ndarray],
]


class GaussianBelief(Belief):
    """Multivariate Gaussian belief state representation.

    Represents the belief as a multivariate normal distribution N(mean, covariance).
    The update mechanism delegates to a user-provided callable, allowing the same
    class to work with EKF, UKF, or any custom Gaussian update rule without
    requiring environment modifications.

    This belief type is compatible with PFT_DPW, Sparse-PFT, and SparseSampling
    planners. It is NOT compatible with POMCP/POMCP_DPW planners because it does
    not support incremental particle accumulation via ``inplace_update()``.

    Attributes:
        mean: Mean vector of the Gaussian distribution.
        covariance: Covariance matrix of the Gaussian distribution.
        updater: Callable that computes the Bayesian belief update.
        n_terminal_check_samples: Number of Monte Carlo samples for terminal checks.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>>
        >>> # Define a simple identity updater
        >>> def identity_updater(mean, cov, action, obs, pomdp):
        ...     return obs, cov * 0.9
        >>>
        >>> # Create 2D Gaussian belief
        >>> mean = np.array([0.0, 0.0])
        >>> cov = np.eye(2)
        >>> belief = GaussianBelief(mean=mean, covariance=cov, updater=identity_updater)
        >>>
        >>> # Sample a state
        >>> state = belief.sample()
        >>> len(state) == 2
        True
        >>>
        >>> # Update belief with observation
        >>> new_belief = belief.update(
        ...     action=0, observation=np.array([1.0, 1.0]), pomdp=None
        ... )
        >>> np.allclose(new_belief.mean, [1.0, 1.0])
        True
    """

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        updater: GaussianBeliefUpdater,
        n_terminal_check_samples: int = 50,
    ):
        """Initialize Gaussian belief.

        Args:
            mean: Mean vector of shape (d,).
            covariance: Positive definite covariance matrix of shape (d, d).
            updater: Callable with signature
                ``(mean, covariance, action, observation, pomdp) -> (new_mean, new_covariance)``.
            n_terminal_check_samples: Number of Monte Carlo samples drawn for
                terminal state checks. Defaults to 50.

        Raises:
            ValueError: If mean/covariance dimensions are inconsistent or
                covariance is not a valid positive definite matrix.
        """
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        self._validate_inputs(mean, covariance)

        self.mean = mean
        self.covariance = covariance
        self.updater = updater
        self.n_terminal_check_samples = n_terminal_check_samples
        self._mvn = CovarianceParameterizedMultivariateNormal(covariance)

    @staticmethod
    def _validate_inputs(mean: np.ndarray, covariance: np.ndarray) -> None:
        if mean.ndim != 1:
            raise ValueError(f"mean must be a 1D array, got {mean.ndim}D")
        if covariance.ndim != 2:
            raise ValueError(f"covariance must be a 2D array, got {covariance.ndim}D")
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError(f"covariance must be square, got shape {covariance.shape}")
        if mean.shape[0] != covariance.shape[0]:
            raise ValueError(
                f"mean length {mean.shape[0]} does not match "
                f"covariance dimension {covariance.shape[0]}"
            )

    def sample(self) -> np.ndarray:
        """Sample a state from the Gaussian belief.

        Returns:
            A state vector of shape (d,) sampled from N(mean, covariance).
        """
        return self._mvn.sample(self.mean, n_samples=1)[0]

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Optional[Environment] = None,
        state: Optional[Any] = None,
    ) -> "GaussianBelief":
        """Update belief using the provided updater callable.

        Args:
            action: Action that was executed.
            observation: Observation that was received.
            pomdp: Environment instance passed to the updater.
            state: Ignored for Gaussian beliefs.

        Returns:
            New GaussianBelief with updated mean and covariance.
        """
        new_mean, new_cov = self.updater(self.mean, self.covariance, action, observation, pomdp)
        return GaussianBelief(
            mean=new_mean,
            covariance=new_cov,
            updater=self.updater,
            n_terminal_check_samples=self.n_terminal_check_samples,
        )

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on belief configuration."""
        config_dict = {
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "n_terminal_check_samples": self.n_terminal_check_samples,
        }
        return config_to_id(config_dict)

    @property
    def dim(self) -> int:
        """Return the dimensionality of the Gaussian belief."""
        return len(self.mean)

    def entropy(self) -> float:
        """Compute the differential entropy of the Gaussian distribution.

        Uses the closed-form expression:
            H = 0.5 * (d * ln(2 * pi * e) + ln(det(Sigma)))

        Returns:
            Differential entropy in nats.
        """
        d = self.dim
        log_det = np.linalg.slogdet(self.covariance)[1]
        return 0.5 * (d * np.log(2.0 * np.pi * np.e) + log_det)
