"""Module for probability distribution implementations.

This module provides abstract and concrete implementations of probability
distributions used throughout the POMDP planning framework.

Classes:
    Distribution: Abstract base class for all probability distributions
    DiscreteDistribution: Implementation for discrete probability distributions
    Numpy2DDistribution: Specialized distribution for 2D numpy array values
"""

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


class Distribution(ABC):
    """Abstract base class for probability distributions.

    This class defines the interface that all probability distributions
    must implement, providing methods for sampling and probability calculation.

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement the sample() method.
    """

    @abstractmethod
    def sample(self, n_samples: int = 1) -> List[Any]:
        """Sample values from the distribution.

        Args:
            n_samples: Number of samples to return. Defaults to 1.

        Returns:
            List of n_samples independent samples from the distribution

        Note:
            Subclasses must implement this method according to their
            specific distribution type and parameters.
        """

    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate probabilities for given values.

        Args:
            values: List of values to calculate probabilities for

        Returns:
            Numpy array of probabilities corresponding to input values

        Raises:
            NotImplementedError: This method is not implemented by default.
                Subclasses should override if probability calculation is needed.
        """
        raise NotImplementedError("The method is not implemented for this distribution.")


class DiscreteDistribution(Distribution):
    """Implementation of discrete probability distributions.

    This class represents a discrete probability distribution over a finite
    set of values, with associated probabilities that sum to 1.

    Attributes:
        values: List of possible values in the distribution support
        probs: Numpy array of probabilities corresponding to each value

    Example:
        >>> import numpy as np
        >>> # Create a distribution over actions
        >>> actions = ["up", "down", "left", "right"]
        >>> probs = np.array([0.4, 0.3, 0.2, 0.1])
        >>> dist = DiscreteDistribution(actions, probs)

        >>> # Sample actions
        >>> samples = dist.sample(5)
        >>> len(samples) == 5
        True
        >>> all(sample in actions for sample in samples)
        True

        >>> # Get probability of specific action
        >>> prob_up = dist.probability(["up"])[0]
        >>> bool(prob_up == 0.4)
        True
    """

    def __init__(self, values: list, probs: np.ndarray):
        """Initialize the discrete distribution.

        Args:
            values: List of distinct values that can be sampled
            probs: Numpy array of probabilities for each value (must sum to 1)

        Raises:
            TypeError: If values is not a list or probs is not a numpy array
            ValueError: If values and probs have different lengths or probs don't sum to 1
        """
        if not isinstance(values, list):
            raise TypeError("values must be a list")
        if not isinstance(probs, np.ndarray):
            raise TypeError("probs must be a numpy.ndarray")
        if len(values) != len(probs):
            raise ValueError("values and probs must have the same length")
        if abs(float(probs.sum()) - 1.0) > 1e-6:
            raise ValueError("probs must sum to 1.0 (within tolerance)")

        self.values = values
        self.probs = probs
        self._cumprobs = np.cumsum(probs)

    def sample(self, n_samples: int = 1) -> List[Any]:
        if n_samples == 1:
            idx = int(np.searchsorted(self._cumprobs, np.random.rand()))
            if idx >= len(self.values):
                idx = len(self.values) - 1
            return [self.values[idx]]
        indices = np.searchsorted(self._cumprobs, np.random.rand(n_samples))
        np.clip(indices, 0, len(self.values) - 1, out=indices)
        return [self.values[idx] for idx in indices]

    def probability(self, values: List[Any]) -> np.ndarray:
        # Vectorized probability calculation
        result = np.zeros(len(values))
        for i, value in enumerate(values):
            for j, dist_value in enumerate(self.values):
                if np.array_equal(value, dist_value):
                    result[i] = self.probs[j]
                    break
        return result


class Numpy2DDistribution(Distribution):
    def __init__(self, values: np.ndarray, probs: np.ndarray):
        if values.shape[0] != 2:
            raise ValueError("values must have shape (2, N)")
        if values.shape[1] != len(probs):
            raise ValueError("values second dimension must match length of probs")

        self.values = values
        self.probs = probs
        self._cumprobs = np.cumsum(probs)

    def sample(self, n_samples: int = 1) -> List[Any]:
        indices = np.searchsorted(self._cumprobs, np.random.rand(n_samples))
        np.clip(indices, 0, self.values.shape[1] - 1, out=indices)
        return [self.values[:, idx] for idx in indices]

    def probability(self, values: List[Any]) -> np.ndarray:
        # Vectorized probability calculation for 2D numpy arrays
        result = np.zeros(len(values))
        for i, value in enumerate(values):
            for j in range(self.values.shape[1]):
                if np.array_equal(value, self.values[:, j]):
                    result[i] = self.probs[j]
                    break
        return result
