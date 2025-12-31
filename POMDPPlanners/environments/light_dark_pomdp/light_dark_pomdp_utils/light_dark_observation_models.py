from abc import abstractmethod
from typing import Any, List, Union

import numpy as np
from scipy.stats import multivariate_normal

from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import ObservationModel


class BaseContinuousLightDarkObservationModel(ObservationModel):
    def __init__(
        self,
        next_state: np.ndarray,
        action: np.ndarray,
        observation_cov_matrix: np.ndarray,
        grid_size: int,
        beacons: np.ndarray,
        beacon_radius: float,
    ):
        super().__init__(next_state=next_state, action=action)
        self.observation_cov_matrix = observation_cov_matrix.copy()
        self.grid_size = grid_size
        self.beacons = beacons
        self.beacon_radius = beacon_radius
        self.near_beacon = self._near_beacon(next_state)

    def _near_beacon(self, next_state: np.ndarray) -> bool:
        next_state = next_state.reshape(2, 1)
        distances = np.linalg.norm(next_state - self.beacons, axis=0)
        # Cast to builtins.bool for mypy compatibility (np.bool_ -> bool)
        return bool(np.any(distances <= self.beacon_radius))

    @abstractmethod
    def sample(self, n_samples: int = 1) -> List[Any]:
        pass


class ContinuousLightDarkNormalNoiseObservationModel(BaseContinuousLightDarkObservationModel):
    def __init__(
        self,
        next_state: np.ndarray,
        action: np.ndarray,
        observation_cov_matrix: np.ndarray,
        grid_size: int,
        beacons: np.ndarray,
        beacon_radius: float,
    ):
        super().__init__(
            next_state=next_state,
            action=action,
            observation_cov_matrix=observation_cov_matrix,
            grid_size=grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )
        if self.near_beacon:
            self.observation_cov_matrix *= 0.5

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        # Vectorized sampling: generate all noise samples at once
        noise = np.random.multivariate_normal(
            mean=np.zeros(2), cov=self.observation_cov_matrix, size=n_samples
        )

        # Vectorized observation calculation
        observations = self.next_state + noise
        observations = np.clip(observations, 0, self.grid_size)

        # Convert to list of arrays
        return [obs for obs in observations]

    def probability(self, values: List[np.ndarray]) -> np.ndarray:
        # Convert list to numpy array for vectorized computation
        values_array = np.array(values)
        res = multivariate_normal.pdf(
            values_array, mean=self.next_state, cov=self.observation_cov_matrix  # type: ignore
        )
        if not isinstance(res, np.ndarray):
            res = np.array([res])

        return res


class ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
    BaseContinuousLightDarkObservationModel
):
    def __init__(
        self,
        next_state: np.ndarray,
        action: np.ndarray,
        observation_cov_matrix: np.ndarray,
        grid_size: int,
        beacons: np.ndarray,
        beacon_radius: float,
    ):
        super().__init__(
            next_state=next_state,
            action=action,
            observation_cov_matrix=observation_cov_matrix,
            grid_size=grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

    def sample(self, n_samples: int = 1) -> List[Union[np.ndarray, None]]:
        noise = np.random.multivariate_normal(
            mean=np.zeros(2), cov=self.observation_cov_matrix, size=n_samples
        )

        if self._near_beacon(self.next_state):
            # Vectorized observation calculation
            observations = self.next_state + noise
            observations = np.clip(observations, 0, self.grid_size)
            return [obs for obs in observations]
        else:
            return [None] * n_samples

    def probability(self, values: List[Union[np.ndarray, None]]) -> np.ndarray:
        res = np.zeros(len(values))
        for i, value in enumerate(values):
            if value is None:
                if self.near_beacon:
                    res[i] = 0
                else:
                    res[i] = 1
            else:
                res[i] = multivariate_normal.pdf(
                    value, mean=self.next_state, cov=self.observation_cov_matrix  # type: ignore
                )

        return res


class BaseDiscreteLightDarkObservationModel(ObservationModel):
    """Base class for discrete Light-Dark observation models.

    This base class provides common functionality for discrete observation models,
    including beacon proximity detection, action-to-vector mapping, and distribution
    creation logic.

    Attributes:
        beacons: Array of beacon positions
        obstacles: Array of obstacle positions
        beacon_radius: Radius within which a beacon is considered "near"
        observation_error_prob: Base probability of observation error
        actions: List of possible actions
        action_to_vector: Mapping from action names to direction vectors
        near_beacon: Boolean indicating if next_state is near a beacon
    """

    def __init__(
        self,
        next_state: np.ndarray,
        action: Any,
        beacons: np.ndarray,
        obstacles: np.ndarray,
        beacon_radius: float,
        observation_error_prob: float,
    ):
        super().__init__(next_state=next_state, action=action)

        self.beacons = beacons
        self.obstacles = obstacles
        self.beacon_radius = beacon_radius
        self.observation_error_prob = observation_error_prob
        self.actions = ["up", "down", "right", "left"]
        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }

        # Check if near beacon
        self.near_beacon = self._near_beacon(next_state)

    def _near_beacon(self, next_state: np.ndarray) -> bool:
        """Check if next_state is near any beacon.

        Args:
            next_state: State to check for beacon proximity

        Returns:
            True if next_state is within beacon_radius of any beacon, False otherwise
        """
        distances = np.linalg.norm(self.beacons - next_state[:, np.newaxis], axis=0)
        min_distance: float = float(np.min(distances))
        return min_distance < self.beacon_radius

    def _create_distribution(self, next_state: np.ndarray) -> DiscreteDistribution:
        """Create a discrete distribution for observations.

        Args:
            next_state: Current state for creating observation distribution

        Returns:
            DiscreteDistribution with observation values and probabilities
        """
        if self.near_beacon:
            beacon_error_factor = 0.2
        else:
            beacon_error_factor = 1.0

        values = [next_state + self.action_to_vector[action] for action in self.actions]
        values.append(next_state)

        observation_error_prob = self.observation_error_prob * beacon_error_factor
        probs = np.ones(len(values)) * (observation_error_prob / (len(values) - 1))
        probs[-1] = 1 - observation_error_prob

        return DiscreteDistribution(values=values, probs=probs)

    @abstractmethod
    def sample(self, n_samples: int = 1) -> List[Any]:
        """Sample observations from the distribution.

        Args:
            n_samples: Number of samples to generate

        Returns:
            List of sampled observations
        """
        pass

    @abstractmethod
    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate probability of given observation values.

        Args:
            values: List of observation values to calculate probabilities for

        Returns:
            Array of probabilities for each value
        """
        pass


class DiscreteLDObservationModel(BaseDiscreteLightDarkObservationModel):
    """Discrete Light-Dark observation model with distance-dependent error probability.

    This observation model provides discrete observations based on the robot's position
    relative to beacons. When near beacons, the observation error probability is reduced,
    making observations more accurate.

    Attributes:
        distribution: DiscreteDistribution for sampling observations
    """

    def __init__(
        self,
        next_state: np.ndarray,
        action: Any,
        beacons: np.ndarray,
        obstacles: np.ndarray,
        beacon_radius: float,
        observation_error_prob: float,
    ):
        super().__init__(
            next_state=next_state,
            action=action,
            beacons=beacons,
            obstacles=obstacles,
            beacon_radius=beacon_radius,
            observation_error_prob=observation_error_prob,
        )

        # Always create distribution
        self.distribution = self._create_distribution(next_state)

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        """Sample observations from the discrete distribution.

        Args:
            n_samples: Number of samples to generate

        Returns:
            List of sampled observation states
        """
        return self.distribution.sample(n_samples)

    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate probability of given observation values.

        Args:
            values: List of observation values to calculate probabilities for

        Returns:
            Array of probabilities for each value
        """
        return self.distribution.probability(values)


class DiscreteLDObservationModelNoObsInDark(BaseDiscreteLightDarkObservationModel):
    """Discrete Light-Dark observation model that returns None when not near beacons.

    This observation model provides discrete observations based on the robot's position
    relative to beacons. When near beacons, observations are sampled from a discrete
    distribution. When far from beacons, observations are None (no observation available).

    Similar to ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel but for discrete
    observations using DiscreteDistribution instead of continuous multivariate normal.

    Attributes:
        distribution: DiscreteDistribution for sampling observations (only used when near beacon),
                      None when far from beacon
    """

    def __init__(
        self,
        next_state: np.ndarray,
        action: Any,
        beacons: np.ndarray,
        obstacles: np.ndarray,
        beacon_radius: float,
        observation_error_prob: float,
    ):
        super().__init__(
            next_state=next_state,
            action=action,
            beacons=beacons,
            obstacles=obstacles,
            beacon_radius=beacon_radius,
            observation_error_prob=observation_error_prob,
        )

        # Only create distribution if near beacon
        if self.near_beacon:
            self.distribution = self._create_distribution(next_state)
        else:
            # Distribution not needed when far from beacon (will return None)
            self.distribution = None

    def sample(self, n_samples: int = 1) -> List[Union[np.ndarray, None]]:
        """Sample observations from the discrete distribution or return None.

        Args:
            n_samples: Number of samples to generate

        Returns:
            List of sampled observation states when near beacon, or list of None when far from beacon
        """
        if self.near_beacon:
            return self.distribution.sample(n_samples)  # type: ignore[union-attr]
        else:
            return [None] * n_samples

    def probability(self, values: List[Union[Any, None]]) -> np.ndarray:
        """Calculate probability of given observation values.

        Args:
            values: List of observation values to calculate probabilities for.
                   Can include None values.

        Returns:
            Array of probabilities for each value:
            - If value is None and near beacon: probability is 0
            - If value is None and far from beacon: probability is 1
            - If value is actual observation: probability from distribution (if near beacon) or 0 (if far)
        """
        res = np.zeros(len(values))
        for i, value in enumerate(values):
            if value is None:
                if self.near_beacon:
                    res[i] = 0
                else:
                    res[i] = 1
            else:
                if self.near_beacon:
                    # Calculate probability from distribution
                    res[i] = self.distribution.probability([value])[0]  # type: ignore[union-attr]
                else:
                    # Far from beacon, actual observations have probability 0
                    res[i] = 0

        return res
