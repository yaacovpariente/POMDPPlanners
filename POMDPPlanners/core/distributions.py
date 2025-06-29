from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np


class Distribution(ABC):
    @abstractmethod
    def sample(self, n_samples: int = 1) -> List[Any]:
        """Sample n_samples from the distribution.
        
        Args:
            n_samples: Number of samples to return (default: 1)
            
        Returns:
            List of n_samples independent samples
        """
        pass

    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate probabilities for a list of values.
        
        Args:
            values: List of values to calculate probabilities for
            
        Returns:
            numpy array of probabilities corresponding to input values
        """
        raise NotImplementedError(
            "The method is not implemented for this distribution."
        )


class DiscreteDistribution(Distribution):
    def __init__(self, values: list, probs: np.array):
        assert isinstance(values, list)
        assert isinstance(probs, np.ndarray)

        assert len(values) == len(probs)
        assert np.isclose(np.sum(probs), 1.0, rtol=1e-10)

        self.values = values
        self.probs = probs

    def sample(self, n_samples: int = 1) -> List[Any]:
        indices = np.random.choice(len(self.values), size=n_samples, p=self.probs)
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
    def __init__(self, values: np.ndarray, probs: np.array):
        assert values.shape[0] == 2
        assert values.shape[1] == len(probs)

        self.values = values
        self.probs = probs

    def sample(self, n_samples: int = 1) -> List[Any]:
        indices = np.random.choice(len(self.probs), size=n_samples, p=self.probs)
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
