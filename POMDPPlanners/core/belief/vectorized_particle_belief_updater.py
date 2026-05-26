# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Abstract base class for vectorized particle belief updaters.

This module provides the ``VectorizedParticleBeliefUpdater`` ABC that defines
a batched interface for particle belief updates. Concrete implementations
perform all-particle transitions and observation log-likelihood evaluations
using vectorized NumPy operations, eliminating Python-level loops over
individual particles.

Classes:
    VectorizedParticleBeliefUpdater: ABC for batched particle belief updates.
"""

from abc import ABC, abstractmethod

import numpy as np


class VectorizedParticleBeliefUpdater(ABC):
    """Abstract base class for vectorized particle belief updaters.

    Subclasses implement batched transition and observation log-likelihood
    methods that operate on the full particle array at once, enabling
    NumPy-level vectorization instead of Python loops.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    @abstractmethod
    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Transition all particles in a single batched operation.

        Args:
            particles: Current particle states of shape (N, d).
            action: Action vector.

        Returns:
            Next-state particles of shape (N, d).
        """

    @abstractmethod
    def batch_observation_log_likelihood(
        self, next_particles: np.ndarray, action: np.ndarray, observation: np.ndarray
    ) -> np.ndarray:
        """Compute observation log-likelihoods for all particles at once.

        Args:
            next_particles: Transitioned particle states of shape (N, d).
            action: Action vector.
            observation: Observed value.

        Returns:
            Log-likelihoods of shape (N,).
        """

    @property
    @abstractmethod
    def config_id(self) -> str:
        """Return a deterministic identifier for this updater configuration."""
