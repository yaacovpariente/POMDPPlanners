from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass

    def probability(self, value):
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

    def sample(self):
        idx = np.random.choice(len(self.values), p=self.probs)
        return self.values[idx]

    def probability(self, value):
        idx = np.array(
            [i if np.array_equal(v, value) else 0 for i, v in enumerate(self.values)]
        )
        if len(idx) == 0:
            return 0.0

        return sum(self.probs * idx)


class Numpy2DDistribution(Distribution):
    def __init__(self, values: np.ndarray, probs: np.array):
        assert values.shape[0] == 2
        assert values.shape[1] == len(probs)

        self.values = values
        self.probs = probs

    def sample(self):
        idx = np.random.choice(len(self.probs), p=self.probs)
        return self.values[:, idx]
