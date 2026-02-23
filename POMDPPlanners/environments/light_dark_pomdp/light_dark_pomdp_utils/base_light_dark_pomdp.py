from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple

import logging
import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    Environment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_visualizer import (
    LightDarkPOMDPVisualizer,
)
from POMDPPlanners.utils.config_to_id import config_to_id


class BaseLightDarkPOMDP(Environment, ABC):
    def __init__(
        self,
        discount_factor: float,
        name: str,
        space_info: SpaceInfo,
        reward_range: Optional[Tuple[float, float]] = None,
        beacons: Optional[List[Tuple[float, float]]] = None,
        goal_state: np.ndarray = np.array([10, 5]),
        start_state: np.ndarray = np.array([0, 5]),
        obstacles: Optional[List[Tuple[float, float]]] = None,
        obstacle_hit_probability: float = 0.2,
        obstacle_reward: float = -10.0,
        obstacle_radius: float = 1.0,
        goal_reward: float = 10.0,
        beacon_radius: float = 1.0,
        fuel_cost: float = 2.0,
        grid_size: int = 11,
    ):
        if beacons is None:
            beacons = [(0, 0), (0, 5), (0, 10), (5, 0), (5, 5), (5, 10), (10, 0), (10, 5), (10, 10)]
        if obstacles is None:
            obstacles = [(3, 7), (5, 5)]
        self.__type_check(
            discount_factor=discount_factor,
            name=name,
            beacons=beacons,
            goal_state=goal_state,
            start_state=start_state,
            obstacles=obstacles,
            obstacle_hit_probability=obstacle_hit_probability,
            obstacle_reward=obstacle_reward,
            obstacle_radius=obstacle_radius,
            goal_reward=goal_reward,
            beacon_radius=beacon_radius,
            fuel_cost=fuel_cost,
            grid_size=grid_size,
        )

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=reward_range,
        )

        # Convert lists of tuples to numpy arrays (maintaining internal representation)
        self.beacons = self._convert_beacons_to_array(beacons)
        self.goal_state = goal_state
        self.start_state = start_state
        self.obstacles = self._convert_obstacles_to_array(obstacles)
        self.obstacle_hit_probability = obstacle_hit_probability
        self.obstacle_reward = obstacle_reward
        self.obstacle_radius = obstacle_radius
        self.goal_reward = goal_reward
        self.beacon_radius = beacon_radius
        self.fuel_cost = fuel_cost
        self.grid_size = grid_size

        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }

    def _convert_beacons_to_array(self, beacons_list: List[Tuple[float, float]]) -> np.ndarray:
        """Convert list of (x, y) tuples to 2xN numpy array format for beacons.

        Args:
            beacons_list: List of (x, y) coordinate tuples

        Returns:
            2xN numpy array where first row is x coordinates, second row is y coordinates
        """
        if not beacons_list:
            return np.empty((2, 0))

        # Convert list of tuples to numpy array and transpose to get 2xN format
        coords_array = np.array(beacons_list).T  # Shape: (2, N)
        return coords_array

    def _convert_obstacles_to_array(self, obstacles_list: List[Tuple[float, float]]) -> np.ndarray:
        """Convert list of (x, y) tuples to 2xN numpy array format for obstacles.

        Args:
            obstacles_list: List of (x, y) coordinate tuples

        Returns:
            2xN numpy array where first row is x coordinates, second row is y coordinates
        """
        if not obstacles_list:
            return np.empty((2, 0))

        # Convert list of tuples to numpy array and transpose to get 2xN format (same as beacons)
        coords_array = np.array(obstacles_list).T  # Shape: (2, N)
        return coords_array

    def __validate_types(
        self,
        discount_factor: float,
        name: str,
        beacons: List[Tuple[float, float]],
        goal_state: np.ndarray,
        start_state: np.ndarray,
        obstacles: List[Tuple[float, float]],
        obstacle_hit_probability: float,
        obstacle_reward: float,
        goal_reward: float,
        beacon_radius: float,
        fuel_cost: float,
        grid_size: int,
    ):
        """Validate parameter types."""
        type_checks = [
            (isinstance(discount_factor, float), "discount_factor must be a float"),
            (isinstance(name, str), "name must be a string"),
            (isinstance(beacons, list), "beacons must be a list of tuples"),
            (isinstance(goal_state, np.ndarray), "goal_state must be a numpy array"),
            (isinstance(start_state, np.ndarray), "start_state must be a numpy array"),
            (isinstance(obstacles, list), "obstacles must be a list of tuples"),
            (
                isinstance(obstacle_hit_probability, float),
                "obstacle_hit_probability must be a float",
            ),
            (isinstance(obstacle_reward, float), "obstacle_reward must be a float"),
            (isinstance(goal_reward, float), "goal_reward must be a float"),
            (isinstance(beacon_radius, float), "beacon_radius must be a float"),
            (isinstance(fuel_cost, float), "fuel_cost must be a float"),
            (isinstance(grid_size, int), "grid_size must be an integer"),
        ]

        for condition, error_msg in type_checks:
            if not condition:
                raise TypeError(error_msg)

        # Validate coordinate tuple structures
        if beacons and not all(isinstance(b, tuple) and len(b) == 2 for b in beacons):
            raise TypeError("beacons must be a list of (x, y) coordinate tuples")
        if obstacles and not all(isinstance(o, tuple) and len(o) == 2 for o in obstacles):
            raise TypeError("obstacles must be a list of (x, y) coordinate tuples")

    def __validate_value_ranges(
        self,
        discount_factor: float,
        obstacle_hit_probability: float,
        grid_size: int,
        beacon_radius: float,
        obstacle_radius: float,
        goal_state: np.ndarray,
        start_state: np.ndarray,
    ):
        """Validate parameter value ranges."""
        # Probability ranges
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be between 0 and 1")
        if not 0 <= obstacle_hit_probability <= 1:
            raise ValueError("obstacle_hit_probability must be between 0 and 1")

        # Positive value checks
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if beacon_radius <= 0:
            raise ValueError("beacon_radius must be positive")
        if obstacle_radius <= 0:
            raise ValueError("obstacle_radius must be positive")

        # Shape checks
        if goal_state.shape != (2,):
            raise ValueError("goal_state must be a 2D vector")
        if start_state.shape != (2,):
            raise ValueError("start_state must be a 2D vector")

    def __validate_coordinates_within_grid(
        self,
        beacons: List[Tuple[float, float]],
        goal_state: np.ndarray,
        start_state: np.ndarray,
        obstacles: List[Tuple[float, float]],
        grid_size: int,
    ):
        """Validate that all coordinates are within grid bounds."""
        for beacon in beacons:
            if not (0 <= beacon[0] <= grid_size and 0 <= beacon[1] <= grid_size):
                raise ValueError("beacons coordinates must be within grid")

        if not (np.all(goal_state >= 0) and np.all(goal_state <= grid_size)):
            raise ValueError("goal_state must be within grid")

        if not (np.all(start_state >= 0) and np.all(start_state <= grid_size)):
            raise ValueError("start_state must be within grid")

        for obstacle in obstacles:
            if not (0 <= obstacle[0] <= grid_size and 0 <= obstacle[1] <= grid_size):
                raise ValueError("obstacles coordinates must be within grid")

    def __type_check(
        self,
        discount_factor: float,
        name: str,
        beacons: List[Tuple[float, float]],
        goal_state: np.ndarray,
        start_state: np.ndarray,
        obstacles: List[Tuple[float, float]],
        obstacle_hit_probability: float,
        obstacle_reward: float,
        obstacle_radius: float,
        goal_reward: float,
        beacon_radius: float,
        fuel_cost: float,
        grid_size: int,
    ):
        """Perform comprehensive type and value validation."""
        self.__validate_types(
            discount_factor,
            name,
            beacons,
            goal_state,
            start_state,
            obstacles,
            obstacle_hit_probability,
            obstacle_reward,
            goal_reward,
            beacon_radius,
            fuel_cost,
            grid_size,
        )
        self.__validate_value_ranges(
            discount_factor,
            obstacle_hit_probability,
            grid_size,
            beacon_radius,
            obstacle_radius,
            goal_state,
            start_state,
        )
        self.__validate_coordinates_within_grid(
            beacons, goal_state, start_state, obstacles, grid_size
        )

    @abstractmethod
    def state_transition_model(self, state: np.ndarray, action: Any) -> StateTransitionModel:
        pass

    @abstractmethod
    def observation_model(self, next_state: np.ndarray, action: Any) -> ObservationModel:
        pass

    @abstractmethod
    def reward(self, state: np.ndarray, action: Any) -> float:
        pass

    @abstractmethod
    def is_terminal(self, state: np.ndarray) -> bool:
        pass

    @abstractmethod
    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        pass

    def initial_state_dist(self) -> Distribution:
        return DiscreteDistribution(values=[self.start_state], probs=np.array([1.0]))

    def initial_observation_dist(self) -> Distribution:
        return DiscreteDistribution(values=[1.0], probs=np.array([1.0]))

    def visualize_path(
        self,
        path: List[np.ndarray],
        agent_belief_path: List[DiscreteDistribution],
        actions: List[str],
        cache_path: Path,
    ) -> None:
        """Create and save an animated visualization of the agent's path.

        Args:
            path: List of state positions (2D numpy arrays) along the agent's trajectory.
            agent_belief_path: List of belief distributions at each step.
            actions: List of actions taken at each step.
            cache_path: Path where to save the visualization (must end with .gif).

        Raises:
            TypeError: If cache_path is not a Path object.
            ValueError: If cache_path doesn't end with .gif.
        """
        visualizer = LightDarkPOMDPVisualizer(self)
        visualizer.visualize_path(path, agent_belief_path, actions, cache_path)

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache visualization of agent's path and belief.

        Args:
            history: List of step data from an episode.
            cache_path: Path where to save the visualization.

        Raises:
            TypeError: If history is not a List or contains non-StepData objects,
                or if cache_path is not a Path object.
            ValueError: If history is empty or contains invalid data.
        """
        visualizer = LightDarkPOMDPVisualizer(self)
        visualizer.cache_visualization(history, cache_path)

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return np.array_equal(observation1, observation2)

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on environment configuration.
        This implementation ensures that the config_id is invariant to the order of beacons and obstacles.
        """

        def serialize_value(value, key=None):  # pylint: disable=too-many-return-statements
            if isinstance(value, np.ndarray):
                if key in ["beacons", "obstacles"] and value.shape[0] == 2:
                    # This is beacons or obstacles in 2xN format
                    # Transpose to get Nx2 format, sort by rows (coordinate pairs), then transpose back
                    transposed = value.T  # Nx2 format
                    sorted_indices = np.lexsort(
                        (transposed[:, 1], transposed[:, 0])
                    )  # Sort by x, then y
                    sorted_array = transposed[sorted_indices].T  # Back to 2xN format
                    # Convert to float to ensure consistent data types
                    return sorted_array.astype(float).tolist()
                return value.tolist()
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            if isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            if isinstance(value, SpaceInfo):
                return {
                    "action_space": serialize_value(value.action_space),
                    "observation_space": serialize_value(value.observation_space),
                }
            if isinstance(value, Enum):
                return value.value
            if hasattr(value, "__dict__"):
                # Skip logger objects
                if isinstance(value, logging.Logger):
                    return None
                return serialize_value(value.__dict__)
            return str(value)

        config_dict = {}
        for key, value in self.__dict__.items():
            # Skip logger and private attributes
            if key.startswith("_") or callable(value) or isinstance(value, logging.Logger):
                continue
            serialized_value = serialize_value(value, key)
            if serialized_value is not None:  # Skip None values (like logger)
                config_dict[key] = serialized_value
        config_dict = dict(sorted(config_dict.items()))
        return config_to_id(config_dict)


class BaseLightDarkPOMDPDiscreteActions(BaseLightDarkPOMDP):
    def __init__(
        self,
        discount_factor: float,
        name: str,
        is_discrete_observations: bool,
        reward_range: Optional[Tuple[float, float]] = None,
        beacons: Optional[List[Tuple[float, float]]] = None,
        goal_state: np.ndarray = np.array([10, 5]),
        start_state: np.ndarray = np.array([0, 5]),
        obstacles: Optional[List[Tuple[float, float]]] = None,
        obstacle_hit_probability: float = 0.2,
        obstacle_reward: float = -10.0,
        goal_reward: float = 10.0,
        beacon_radius: float = 1.0,
        fuel_cost: float = 2.0,
        grid_size: int = 11,
    ):
        if beacons is None:
            beacons = [(0, 0), (0, 5), (0, 10), (5, 0), (5, 5), (5, 10), (10, 0), (10, 5), (10, 10)]
        if obstacles is None:
            obstacles = [(3, 7), (5, 5)]
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=(
                SpaceType.DISCRETE if is_discrete_observations else SpaceType.CONTINUOUS
            ),
        )
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=reward_range,
            beacons=beacons,
            goal_state=goal_state,
            start_state=start_state,
            obstacles=obstacles,
            obstacle_hit_probability=obstacle_hit_probability,
            obstacle_reward=obstacle_reward,
            goal_reward=goal_reward,
            beacon_radius=beacon_radius,
            fuel_cost=fuel_cost,
            grid_size=grid_size,
        )

        self.actions = ["up", "down", "right", "left"]

    def get_actions(self) -> List[Any]:
        return self.actions
