import numpy as np
from POMDPPlanners.core.environment import ObservationModel


class ContinuousLightDarkNormalNoiseObservationModel(ObservationModel):
    def __init__(self, next_state: np.ndarray, action: np.ndarray, observation_cov_matrix: np.ndarray, grid_size: int):
        super().__init__(next_state, action)
        self.observation_cov_matrix = observation_cov_matrix
        self.grid_size = grid_size

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


