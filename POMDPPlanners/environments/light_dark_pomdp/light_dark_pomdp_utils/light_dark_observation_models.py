from typing import Any, List

import numpy as np
from scipy.stats import multivariate_normal

from POMDPPlanners.core.environment import ObservationModel


class ContinuousLightDarkNormalNoiseObservationModel(ObservationModel):
    def __init__(
        self,
        next_state: np.ndarray,
        action: np.ndarray,
        observation_cov_matrix: np.ndarray,
        grid_size: int,
        beacons: np.ndarray,
        beacon_radius: float,
    ):
        super().__init__(next_state, action)
        self.observation_cov_matrix = observation_cov_matrix.copy()
        self.grid_size = grid_size
        self.beacons = beacons
        self.beacon_radius = beacon_radius

        self.near_beacon = self._near_beacon(next_state)
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
            values_array, mean=self.next_state, cov=self.observation_cov_matrix
        )
        if not isinstance(res, np.ndarray):
            res = np.array([res])

        return res

    def _near_beacon(self, next_state: np.ndarray) -> bool:
        next_state = next_state.reshape(2, 1)
        distances = np.linalg.norm(next_state - self.beacons, axis=0)
        return np.any(distances <= self.beacon_radius)
