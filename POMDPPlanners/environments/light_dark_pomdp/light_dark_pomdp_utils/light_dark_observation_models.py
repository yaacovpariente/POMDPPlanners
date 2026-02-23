from abc import abstractmethod
from typing import Any, List, Union

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import ObservationModel
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal

__all__ = [
    "BaseContinuousLightDarkObservationModel",
    "BaseDiscreteLightDarkObservationModel",
    "ContinuousLightDarkDistanceBasedObservationModel",
    "ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel",
    "ContinuousLightDarkNormalNoiseObservationModel",
    "DiscreteLDDistanceBasedObservationModel",
    "DiscreteLDObservationModel",
    "DiscreteLDObservationModelNoObsInDark",
]


class BaseContinuousLightDarkObservationModel(ObservationModel):
    def __init__(
        self,
        next_state: np.ndarray,
        action: np.ndarray,
        obs_dist_near_beacon: CovarianceParameterizedMultivariateNormal,
        obs_dist_far_from_beacon: CovarianceParameterizedMultivariateNormal,
        grid_size: int,
        beacons: np.ndarray,
        beacon_radius: float,
    ):
        super().__init__(next_state=next_state, action=action)
        self.obs_dist_near_beacon = obs_dist_near_beacon
        self.obs_dist_far_from_beacon = obs_dist_far_from_beacon
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
        obs_dist_near_beacon: CovarianceParameterizedMultivariateNormal,
        obs_dist_far_from_beacon: CovarianceParameterizedMultivariateNormal,
        grid_size: int,
        beacons: np.ndarray,
        beacon_radius: float,
    ):
        super().__init__(
            next_state=next_state,
            action=action,
            obs_dist_near_beacon=obs_dist_near_beacon,
            obs_dist_far_from_beacon=obs_dist_far_from_beacon,
            grid_size=grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )
        # Select distribution based on beacon proximity
        self._active_dist = (
            self.obs_dist_near_beacon if self.near_beacon else self.obs_dist_far_from_beacon
        )

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        # Sample using pre-computed Cholesky decomposition
        observations = self._active_dist.sample(self.next_state, n_samples)
        observations = np.clip(observations, 0, self.grid_size)

        # Convert to list of arrays
        return list(observations)

    def probability(self, values: List[np.ndarray]) -> np.ndarray:
        # Convert list to numpy array for vectorized computation
        values_array = np.array(values)
        res = self._active_dist.pdf(values_array, self.next_state)
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
        obs_dist_near_beacon: CovarianceParameterizedMultivariateNormal,
        obs_dist_far_from_beacon: CovarianceParameterizedMultivariateNormal,
        grid_size: int,
        beacons: np.ndarray,
        beacon_radius: float,
    ):
        super().__init__(
            next_state=next_state,
            action=action,
            obs_dist_near_beacon=obs_dist_near_beacon,
            obs_dist_far_from_beacon=obs_dist_far_from_beacon,
            grid_size=grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )
        # Select distribution based on beacon proximity
        self._active_dist = (
            self.obs_dist_near_beacon if self.near_beacon else self.obs_dist_far_from_beacon
        )

    def sample(self, n_samples: int = 1) -> List[Union[np.ndarray, str]]:
        if self._near_beacon(self.next_state):
            # Sample using pre-computed Cholesky decomposition
            observations = self._active_dist.sample(self.next_state, n_samples)
            observations = np.clip(observations, 0, self.grid_size)
            return list(observations)
        return ["None"] * n_samples

    def probability(self, values: List[Union[np.ndarray, str]]) -> np.ndarray:
        res = np.zeros(len(values))
        for i, value in enumerate(values):
            if isinstance(value, str) and value == "None":
                if self.near_beacon:
                    res[i] = 0
                else:
                    res[i] = 1
            else:
                res[i] = self._active_dist.pdf(np.array([value]), self.next_state)[0]

        return res


class ContinuousLightDarkDistanceBasedObservationModel(BaseContinuousLightDarkObservationModel):
    """Continuous Light-Dark observation model with binary near/far beacon noise levels.

    This observation model uses a binary near/far approach based on the distance to the
    nearest beacon. When within beacon_radius, observations are sampled from the near-beacon
    distribution. When the distance exceeds beacon_radius, observations are "None"
    (no observation available).

    Attributes:
        min_distance_to_beacon: Distance to the nearest beacon
    """

    def __init__(
        self,
        next_state: np.ndarray,
        action: np.ndarray,
        obs_dist_near_beacon: CovarianceParameterizedMultivariateNormal,
        obs_dist_far_from_beacon: CovarianceParameterizedMultivariateNormal,
        grid_size: int,
        beacons: np.ndarray,
        beacon_radius: float,
    ):
        super().__init__(
            next_state=next_state,
            action=action,
            obs_dist_near_beacon=obs_dist_near_beacon,
            obs_dist_far_from_beacon=obs_dist_far_from_beacon,
            grid_size=grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )
        # Compute distance to nearest beacon
        next_state_reshaped = next_state.reshape(2, 1)
        distances = np.linalg.norm(next_state_reshaped - self.beacons, axis=0)
        self.min_distance_to_beacon: float = float(np.min(distances))

        # Select distribution based on beacon proximity
        self._active_dist = (
            self.obs_dist_near_beacon if self.near_beacon else self.obs_dist_far_from_beacon
        )

    def sample(self, n_samples: int = 1) -> List[Union[np.ndarray, str]]:
        # If beyond beacon_radius, return "None" observations
        if self.min_distance_to_beacon > self.beacon_radius:
            return ["None"] * n_samples

        # Sample using pre-computed Cholesky decomposition
        observations = self._active_dist.sample(self.next_state, n_samples)
        observations = np.clip(observations, 0, self.grid_size)

        # Convert to list of arrays
        return list(observations)

    def probability(self, values: List[Union[np.ndarray, str]]) -> np.ndarray:
        res = np.zeros(len(values))
        for i, value in enumerate(values):
            if isinstance(value, str) and value == "None":
                if self.min_distance_to_beacon > self.beacon_radius:
                    res[i] = 1.0  # "None" observation is certain when far from beacon
                else:
                    res[i] = 0.0  # "None" observation is impossible when near beacon
            else:
                if self.min_distance_to_beacon > self.beacon_radius:
                    res[i] = 0.0  # Actual observations have probability 0 when far from beacon
                else:
                    # Calculate probability using pre-computed Cholesky decomposition
                    res[i] = self._active_dist.pdf(np.array([value]), self.next_state)[0]

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

    @abstractmethod
    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate probability of given observation values.

        Args:
            values: List of observation values to calculate probabilities for

        Returns:
            Array of probabilities for each value
        """


class DiscreteLDDistanceBasedObservationModel(BaseDiscreteLightDarkObservationModel):
    """Discrete Light-Dark observation model with continuous distance-based error probability.

    This observation model scales the observation error probability continuously based on
    the distance to the nearest beacon, rather than using a binary threshold. The error
    probability scales linearly from a minimum value (when at beacon) to the base value
    (when at beacon_radius distance). When the distance exceeds beacon_radius, observations
    are "None" (no observation available).

    The scaling formula is:
        error_factor = min_factor + (1 - min_factor) * (distance / beacon_radius)
        error_prob(d) = base_error_prob * error_factor  (only when distance <= beacon_radius)

    Where:
        - min_factor = 0.2 (error probability is reduced to 20% when at beacon)
        - distance = distance to nearest beacon
        - At distance 0: error_prob = 0.2 * base_error_prob
        - At distance beacon_radius: error_prob = 1.0 * base_error_prob
        - Beyond beacon_radius: observation = "None"

    Attributes:
        distribution: DiscreteDistribution for sampling observations (only used when near beacon),
                      None when far from beacon
        min_distance_to_beacon: Distance to the nearest beacon
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

        # Compute distance to nearest beacon
        distances = np.linalg.norm(self.beacons - next_state[:, np.newaxis], axis=0)
        self.min_distance_to_beacon: float = float(np.min(distances))

        # Only create distribution if within beacon_radius
        if self.min_distance_to_beacon <= self.beacon_radius:
            self.distribution = self._create_distribution(next_state)
        else:
            # Distribution not needed when far from beacon (will return "None")
            self.distribution = None

    def _create_distribution(self, next_state: np.ndarray) -> DiscreteDistribution:
        """Create a discrete distribution for observations with continuous distance-based scaling.

        Args:
            next_state: Current state for creating observation distribution

        Returns:
            DiscreteDistribution with observation values and probabilities
        """
        # Continuous distance-based scaling
        # Scale from 0.0001 (at beacon) to 1.0 (at beacon_radius)
        min_factor = 0.0001
        if self.beacon_radius > 0:
            # Linear interpolation: factor = min_factor + (1 - min_factor) * (d / beacon_radius)
            # This gives 0.0001 at d=0, 1.0 at d=beacon_radius
            distance_ratio = self.min_distance_to_beacon / self.beacon_radius
            beacon_error_factor = min_factor + (1.0 - min_factor) * distance_ratio
        else:
            # If beacon_radius is 0, use base error probability
            beacon_error_factor = 1.0

        values = [next_state + self.action_to_vector[action] for action in self.actions]
        values.append(next_state)

        observation_error_prob = self.observation_error_prob * beacon_error_factor
        probs = np.ones(len(values)) * (observation_error_prob / (len(values) - 1))
        probs[-1] = 1 - observation_error_prob

        return DiscreteDistribution(values=values, probs=probs)

    def sample(self, n_samples: int = 1) -> List[Union[np.ndarray, str]]:
        """Sample observations from the discrete distribution or return "None".

        Args:
            n_samples: Number of samples to generate

        Returns:
            List of sampled observation states when near beacon, or list of "None" when far from beacon
        """
        if self.min_distance_to_beacon <= self.beacon_radius:
            return self.distribution.sample(n_samples)  # type: ignore[union-attr]
        return ["None"] * n_samples

    def probability(self, values: List[Union[Any, str]]) -> np.ndarray:
        """Calculate probability of given observation values.

        Args:
            values: List of observation values to calculate probabilities for.
                   Can include "None" values.

        Returns:
            Array of probabilities for each value:
            - If value is "None" and near beacon: probability is 0
            - If value is "None" and far from beacon: probability is 1
            - If value is actual observation: probability from distribution (if near beacon) or 0 (if far)
        """
        res = np.zeros(len(values))
        for i, value in enumerate(values):
            if isinstance(value, str) and value == "None":
                if self.min_distance_to_beacon <= self.beacon_radius:
                    res[i] = 0.0  # "None" observation is impossible when near beacon
                else:
                    res[i] = 1.0  # "None" observation is certain when far from beacon
            else:
                if self.min_distance_to_beacon <= self.beacon_radius:
                    # Calculate probability from distribution
                    res[i] = self.distribution.probability([value])[0]  # type: ignore[union-attr]
                else:
                    # Far from beacon, actual observations have probability 0
                    res[i] = 0.0

        return res


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
    """Discrete Light-Dark observation model that returns "None" when not near beacons.

    This observation model provides discrete observations based on the robot's position
    relative to beacons. When near beacons, observations are sampled from a discrete
    distribution. When far from beacons, observations are "None" (no observation available).

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
            # Distribution not needed when far from beacon (will return "None")
            self.distribution = None

    def sample(self, n_samples: int = 1) -> List[Union[np.ndarray, str]]:
        """Sample observations from the discrete distribution or return "None".

        Args:
            n_samples: Number of samples to generate

        Returns:
            List of sampled observation states when near beacon, or list of "None" when far from beacon
        """
        if self.near_beacon:
            return self.distribution.sample(n_samples)  # type: ignore[union-attr]
        return ["None"] * n_samples

    def probability(self, values: List[Union[Any, str]]) -> np.ndarray:
        """Calculate probability of given observation values.

        Args:
            values: List of observation values to calculate probabilities for.
                   Can include "None" values.

        Returns:
            Array of probabilities for each value:
            - If value is "None" and near beacon: probability is 0
            - If value is "None" and far from beacon: probability is 1
            - If value is actual observation: probability from distribution (if near beacon) or 0 (if far)
        """
        res = np.zeros(len(values))
        for i, value in enumerate(values):
            if isinstance(value, str) and value == "None":
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
