import numpy as np
from POMDPPlanners.core.environment import ObservationModel


class ContinuousLightDarkNormalNoiseObservationModel(ObservationModel):
    def __init__(self, next_state: np.ndarray, action: np.ndarray, observation_cov_matrix: np.ndarray, grid_size: int, beacons: np.ndarray, beacon_radius: float):
        super().__init__(next_state, action)
        self.observation_cov_matrix = observation_cov_matrix.copy()
        self.grid_size = grid_size
        self.beacons = beacons
        self.beacon_radius = beacon_radius
        
        self.near_beacon = self._near_beacon(next_state)
        if self.near_beacon:
            self.observation_cov_matrix *= 0.5

    def sample(self) -> np.ndarray:
        noise = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=self.observation_cov_matrix
        )
        
        observation = self.next_state + noise
        observation = np.clip(observation, 0, self.grid_size)
        
        return observation

    def probability(self, next_observation: np.ndarray) -> float:
        # For continuous observations, we return the probability density
        # of the multivariate normal distribution
        noise = next_observation - self.next_state
        return np.exp(-0.5 * noise.T @ np.linalg.inv(self.observation_cov_matrix) @ noise) / \
               (2 * np.pi * np.sqrt(np.linalg.det(self.observation_cov_matrix)))

    def _near_beacon(self, next_state: np.ndarray) -> bool:
        next_state = next_state.reshape(2, 1)
        distances = np.linalg.norm(next_state - self.beacons, axis=0)
        return np.any(distances <= self.beacon_radius)

