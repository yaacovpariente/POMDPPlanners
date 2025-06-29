from typing import List, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    StateTransitionModel,
    ObservationModel,
    SpaceInfo,
    SpaceType
)
from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import StepData
import scipy.stats


class MountainCarTransition(StateTransitionModel):
    def __init__(
        self,
        state: Tuple[float, float],
        action: int,
        power: float,
        gravity: float,
        max_speed: float,
        min_position: float,
        max_position: float,
    ):
        super().__init__(state, action)

        self.power = power
        self.gravity = gravity
        self.max_speed = max_speed
        self.min_position = min_position
        self.max_position = max_position

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        position, velocity = self.state
        v = velocity + self.action * self.power + np.cos(3 * position) * (-self.gravity)
        v = np.clip(v, -self.max_speed, self.max_speed)
        p = position + v
        p = np.clip(p, self.min_position, self.max_position)
        if p == self.min_position and v < 0:
            v = 0
        next_state = np.array([p, v])
        return [next_state] * n_samples

    def probability(self, values: List[np.ndarray]) -> np.ndarray:
        result = np.zeros(len(values))
        expected_next_state = self.sample()[0]
        for i, value in enumerate(values):
            if np.array_equal(value, expected_next_state):
                result[i] = 1.0
        return result


class MountainCarObservation(ObservationModel):
    def __init__(
        self, next_state: Tuple[float, float], action: int, cov_matrix: np.ndarray
    ):
        super().__init__(next_state=next_state, action=action)
        self.cov_matrix = cov_matrix
        self.mean = np.array(next_state)

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        samples = []
        for _ in range(n_samples):
            obs = np.random.multivariate_normal(self.mean, self.cov_matrix)
            samples.append(np.array(obs))
        return samples

    def probability(self, values: List[np.ndarray]) -> np.ndarray:
        # Vectorized probability for a batch of observations
        values_array = np.array(values)
        return scipy.stats.multivariate_normal(self.mean, self.cov_matrix).pdf(values_array)


class MountainCarPOMDP(DiscreteActionsEnvironment):
    def __init__(self, discount_factor: float, name: str = "MountainCarPOMDP", output_dir: Optional[Path] = None, debug: bool = False):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.power = 0.001
        self.gravity = 0.0025

        # Define actions: -1 (left), 0 (no acceleration), 1 (right)
        self.actions = [-1, 0, 1]

        # Define observation noise parameters
        self.position_noise = 0.1
        self.velocity_noise = 0.01

        # Observation noise matrix
        self.cov_matrix = np.array(
            [[self.position_noise**2, 0], [0, self.velocity_noise**2]]
        )

        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # Action space is [-1, 0, 1]
            observation_space=SpaceType.CONTINUOUS  # Observation space is position and velocity
        )
        super().__init__(discount_factor=discount_factor, name=name, space_info=space_info, output_dir=output_dir, debug=debug)

    def state_transition_model(
        self, state: Tuple[float, float], action: int
    ) -> Distribution:
        return MountainCarTransition(
            state=state,
            action=action,
            power=self.power,
            gravity=self.gravity,
            max_speed=self.max_speed,
            min_position=self.min_position,
            max_position=self.max_position,
        )

    def observation_model(
        self, next_state: Tuple[float, float], action: int
    ) -> Distribution:
        return MountainCarObservation(
            next_state=next_state, action=action, cov_matrix=self.cov_matrix
        )

    def reward(self, state: Tuple[float, float], action: int) -> float:
        position, _ = state

        # Reward for reaching the goal
        if position >= self.goal_position:
            return 0.0

        # Small negative reward for each step to encourage reaching the goal quickly
        return -1.0

    def is_terminal(self, state: Tuple[float, float]) -> bool:
        position, _ = state
        return position >= self.goal_position

    def initial_state_dist(self) -> Distribution:
        class InitialState(Distribution):
            def sample(self, n_samples: int = 1) -> List[np.ndarray]:
                samples = []
                for _ in range(n_samples):
                    position = np.random.uniform(-0.6, -0.4)
                    velocity = 0.0
                    samples.append(np.array([position, velocity]))
                return samples
        return InitialState()

    def initial_observation_dist(self) -> Distribution:
        class InitialObservation(Distribution):
            def sample(self, n_samples: int = 1) -> List[np.ndarray]:
                return [np.array([0.0, 0.0])] * n_samples
        return InitialObservation()

    def get_actions(self) -> List[Any]:
        return self.actions

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        # Create a figure and axis
        pass

    def is_equal_observation(self, observation1: Tuple[float, float], observation2: Tuple[float, float]) -> bool:
        return np.array_equal(observation1, observation2)
