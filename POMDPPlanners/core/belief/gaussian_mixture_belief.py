# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Gaussian Mixture belief state representation for POMDP environments.

This module provides a Gaussian Mixture Model (GMM) belief state that
represents the posterior as a weighted mixture of multivariate Gaussians.
Updates are delegated to a :class:`GaussianMixtureBeliefUpdater` instance,
following the same dependency injection pattern as
:class:`~POMDPPlanners.core.belief.GaussianBelief`.

Classes:
    GaussianMixtureBeliefUpdater: ABC for GMM belief update strategies.
    GaussianMixtureBelief: GMM belief with pluggable updater.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.special import logsumexp

from POMDPPlanners.core.belief.base_belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.multivariate_normal import (
    CovarianceParameterizedMultivariateNormal,
)


class GaussianMixtureBeliefUpdater(ABC):
    """Abstract base class for Gaussian mixture belief updaters.

    Subclasses implement an update cycle that maps
    ``(means, covariances, weights, action, observation)`` to an updated
    ``(new_means, new_covariances, new_weights)`` tuple.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    @abstractmethod
    def update(
        self,
        means: List[np.ndarray],
        covariances: List[np.ndarray],
        weights: np.ndarray,
        action: Any,
        observation: Any,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Perform a belief update for the Gaussian mixture.

        Args:
            means: List of k mean vectors, each of shape (d,).
            covariances: List of k covariance matrices, each of shape (d, d).
            weights: Mixture weights of shape (k,).
            action: Action that was executed.
            observation: Observation that was received.

        Returns:
            A tuple ``(new_means, new_covariances, new_weights)``.
        """

    @property
    @abstractmethod
    def config_id(self) -> str:
        """Return a deterministic identifier for this updater configuration."""


class GaussianMixtureBelief(Belief):
    """Gaussian Mixture Model belief state representation.

    Represents the belief as a weighted mixture of multivariate normal
    distributions: p(x) = sum_k w_k * N(x; mu_k, Sigma_k). The update
    mechanism delegates to a :class:`GaussianMixtureBeliefUpdater` instance,
    allowing flexibility in how mixture components are updated, pruned, or
    merged.

    This belief type is compatible with PFT_DPW, Sparse-PFT, and
    SparseSampling planners. It is NOT compatible with POMCP/POMCP_DPW
    planners because it does not support incremental particle accumulation
    via ``inplace_update()``.

    Attributes:
        means: List of mean vectors, one per component.
        covariances: List of covariance matrices, one per component.
        weights: Array of mixture weights summing to 1.
        updater: GaussianMixtureBeliefUpdater that computes the Bayesian belief update.
        n_terminal_check_samples: Number of Monte Carlo samples for terminal checks.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>>
        >>> # Define a simple updater that shrinks covariances
        >>> from POMDPPlanners.core.belief.gaussian_mixture_belief import (
        ...     GaussianMixtureBeliefUpdater,
        ... )
        >>> class ShrinkUpdater(GaussianMixtureBeliefUpdater):
        ...     def update(self, means, covs, weights, action, obs):
        ...         return means, [c * 0.9 for c in covs], weights
        ...     @property
        ...     def config_id(self):
        ...         return "shrink"
        >>>
        >>> # Create a 2-component GMM belief in 2D
        >>> means = [np.array([0.0, 0.0]), np.array([3.0, 3.0])]
        >>> covs = [np.eye(2), np.eye(2)]
        >>> weights = np.array([0.5, 0.5])
        >>> belief = GaussianMixtureBelief(
        ...     means=means, covariances=covs, weights=weights, updater=ShrinkUpdater(),
        ... )
        >>>
        >>> # Sample a state
        >>> state = belief.sample()
        >>> len(state) == 2
        True
        >>>
        >>> # Update belief
        >>> new_belief = belief.update(
        ...     action=0, observation=np.array([1.0, 1.0]), pomdp=None
        ... )
        >>> new_belief.n_components == 2
        True
    """

    def __init__(
        self,
        means: List[np.ndarray],
        covariances: List[np.ndarray],
        weights: np.ndarray,
        updater: GaussianMixtureBeliefUpdater,
        n_terminal_check_samples: int = 50,
    ):
        """Initialize Gaussian Mixture belief.

        Args:
            means: List of k mean vectors, each of shape (d,).
            covariances: List of k positive definite covariance matrices,
                each of shape (d, d).
            weights: Mixture weights of shape (k,) that must sum to 1.
            updater: A :class:`GaussianMixtureBeliefUpdater` instance whose
                ``update(means, covariances, weights, action, observation)``
                method returns ``(new_means, new_covariances, new_weights)``.
            n_terminal_check_samples: Number of Monte Carlo samples drawn for
                terminal state checks. Defaults to 50.

        Raises:
            ValueError: If inputs are inconsistent (mismatched counts,
                dimensions, or invalid weights).
        """
        means = [np.asarray(m, dtype=float) for m in means]
        covariances = [np.asarray(c, dtype=float) for c in covariances]
        weights = np.asarray(weights, dtype=float)
        self._validate_inputs(means, covariances, weights)

        self.means = means
        self.covariances = covariances
        self.weights = weights
        self._cum_weights = np.cumsum(weights)
        self.updater = updater
        self.n_terminal_check_samples = n_terminal_check_samples
        self._mvns = [CovarianceParameterizedMultivariateNormal(c) for c in covariances]

    @staticmethod
    def _validate_inputs(
        means: List[np.ndarray],
        covariances: List[np.ndarray],
        weights: np.ndarray,
    ) -> None:
        if len(means) == 0:
            raise ValueError("Must have at least one component")
        if len(means) != len(covariances):
            raise ValueError(
                f"Number of means ({len(means)}) does not match "
                f"number of covariances ({len(covariances)})"
            )
        if weights.ndim != 1 or len(weights) != len(means):
            raise ValueError(
                f"weights must be a 1D array of length {len(means)}, " f"got shape {weights.shape}"
            )
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"weights must sum to 1, got {weights.sum()}")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")

        d = len(means[0])
        for i, (m, c) in enumerate(zip(means, covariances)):
            if m.ndim != 1:
                raise ValueError(f"means[{i}] must be 1D, got {m.ndim}D")
            if len(m) != d:
                raise ValueError(f"means[{i}] has length {len(m)}, expected {d}")
            if c.ndim != 2 or c.shape != (d, d):
                raise ValueError(f"covariances[{i}] must have shape ({d}, {d}), " f"got {c.shape}")

    def sample(self) -> np.ndarray:
        """Sample a state from the Gaussian mixture belief.

        Selects a component according to the mixture weights, then draws
        a sample from that component's Gaussian distribution.

        Returns:
            A state vector of shape (d,).
        """
        k = int(np.searchsorted(self._cum_weights, np.random.random()))
        if k >= len(self.weights):
            k = len(self.weights) - 1
        return self._mvns[k].sample(self.means[k], n_samples=1)[0]

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Optional[Environment] = None,
        state: Optional[Any] = None,
    ) -> "GaussianMixtureBelief":
        """Update belief using the provided updater.

        Args:
            action: Action that was executed.
            observation: Observation that was received.
            pomdp: Unused. Kept for interface compatibility with
                :class:`~POMDPPlanners.core.belief.base_belief.Belief`.
            state: Ignored for Gaussian mixture beliefs.

        Returns:
            New GaussianMixtureBelief with updated components and weights.
        """
        new_means, new_covs, new_weights = self.updater.update(
            self.means, self.covariances, self.weights, action, observation
        )
        return GaussianMixtureBelief(
            means=new_means,
            covariances=new_covs,
            weights=new_weights,
            updater=self.updater,
            n_terminal_check_samples=self.n_terminal_check_samples,
        )

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on belief configuration."""
        sorted_data = sorted(
            zip(
                [m.tolist() for m in self.means],
                [c.tolist() for c in self.covariances],
                self.weights.tolist(),
            )
        )
        config_dict = {
            "components": sorted_data,
            "n_terminal_check_samples": self.n_terminal_check_samples,
            "updater": self.updater.config_id,
        }
        return config_to_id(config_dict)

    @property
    def dim(self) -> int:
        """Return the dimensionality of the belief state."""
        return len(self.means[0])

    @property
    def n_components(self) -> int:
        """Return the number of mixture components."""
        return len(self.means)

    def entropy(self, n_samples: int = 1000) -> float:
        """Estimate the differential entropy via Monte Carlo sampling.

        There is no closed-form expression for the entropy of a Gaussian
        mixture, so this method uses the approximation:
            H ~ -mean(log p(x_i)),  x_i ~ p(x)

        Args:
            n_samples: Number of Monte Carlo samples. Defaults to 1000.

        Returns:
            Estimated differential entropy in nats.
        """
        samples = np.array([self.sample() for _ in range(n_samples)])
        log_pdf_values = self._log_pdf(samples)
        return -float(np.mean(log_pdf_values))

    def _log_pdf(self, values: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(values)
        n = values.shape[0]
        k = self.n_components
        log_components = np.empty((n, k))
        for j in range(k):
            log_components[:, j] = np.log(self.weights[j]) + self._mvns[j].log_pdf(
                values, self.means[j]
            )
        return np.asarray(logsumexp(log_components, axis=1))
