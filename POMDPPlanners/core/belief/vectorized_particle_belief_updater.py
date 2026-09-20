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
from typing import Optional

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

    def ruled_out_by_a_running_episode(
        self, next_particles: np.ndarray
    ) -> Optional[np.ndarray]:
        """Which particles the episode's continuation has ruled out.

        This is the opt-in half of the conditioning described in
        :mod:`POMDPPlanners.core.belief.running_episode_conditioning`. The
        default returns ``None``, meaning "do not condition": an environment
        whose terminal states are observable, or are not absorbing, loses
        nothing by leaving them in the population, and a blanket change here
        would alter every environment's filter at once.

        An environment overrides this only when its terminal state is both
        absorbing and invisible to its own observation model, because that is
        the combination the likelihood cannot undo on its own.

        Args:
            next_particles: The transitioned particles, shape (N, d).

        Returns:
            A boolean mask over ``next_particles``, ``True`` where the
            particle is terminal for a reason that varies between particles;
            or ``None`` to leave the weights alone. Terminality every particle
            shares -- a step limit read off a counter they all carry -- must
            be left out: it cancels in the posterior and flooring on it would
            empty the belief on the final step.
        """
        del next_particles
        return None

    @property
    @abstractmethod
    def config_id(self) -> str:
        """Return a deterministic identifier for this updater configuration."""
