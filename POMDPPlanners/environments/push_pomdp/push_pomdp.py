"""Push POMDP Environment Implementation.

This module implements a robotic push task as a POMDP, where a robot must
push an object to a target location on a 2D grid. The robot can move in
four directions and pushes objects when within range, with noisy observations
of the object's position.

The Push POMDP features:
- Continuous 2D state space: [robot_x, robot_y, object_x, object_y, target_x, target_y]
- Discrete action space: ["up", "down", "left", "right"]
- Noisy observations of object position (robot and target positions are known)
- Physics-based pushing mechanics with friction
- Distance-based rewards encouraging object movement toward target

Key mechanics:
- Robot must be within push_threshold distance to move objects
- Friction reduces the effectiveness of pushes
- Object position observations include Gaussian noise
- Episode terminates when object reaches target

Classes:
    PushStateTransition: Physics-based pushing dynamics
    PushObservation: Noisy object position observations
    PushPOMDP: Main push task environment with POMDP formulation
"""

from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.push_pomdp.push_pomdp_visualizer import PushPOMDPVisualizer
from POMDPPlanners.utils.statistics_utils import confidence_interval


class PushPOMDPMetrics(Enum):
    """Metric names for Push POMDP environment."""

    ROBOT_OBSTACLE_COLLISION_RATE = "robot_obstacle_collision_rate"
    OBJECT_OBSTACLE_COLLISION_RATE = "object_obstacle_collision_rate"
    TOTAL_OBSTACLE_COLLISION_RATE = "total_obstacle_collision_rate"
    TOTAL_ROBOT_OBSTACLE_COLLISIONS = "total_robot_obstacle_collisions"
    TOTAL_OBJECT_OBSTACLE_COLLISIONS = "total_object_obstacle_collisions"
    TOTAL_ALL_OBSTACLE_COLLISIONS = "total_all_obstacle_collisions"


class PushStateTransition(StateTransitionModel):
    """State transition model for Push POMDP with physics-based pushing.

    This model implements robot movement and object pushing dynamics on a 2D grid.
    The robot moves according to discrete actions, and can push objects when
    within the push threshold distance. Friction reduces push effectiveness.

    State representation: [robot_x, robot_y, object_x, object_y, target_x, target_y]

    Attributes:
        state: Current state vector containing all entity positions
        action: Movement action ("up", "down", "left", "right")
        grid_size: Size of the grid environment
        push_threshold: Maximum distance for robot to push object
        friction_coefficient: Friction that reduces push force (0=no friction, 1=max friction)
        robot_pos: Current robot position [x, y]
        object_pos: Current object position [x, y]
        target_pos: Target position [x, y] (fixed)

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> # Define state: [robot_x, robot_y, object_x, object_y, target_x, target_y]
        >>> state = np.array([2.0, 3.0, 2.5, 3.1, 8.0, 8.0])
        >>> action = "right"  # Move robot right

        >>> # Create transition model
        >>> transition = PushStateTransition(
        ...     state=state,
        ...     action=action,
        ...     grid_size=10,
        ...     push_threshold=1.0,
        ...     friction_coefficient=0.3
        ... )

        >>> # Simulate step
        >>> next_state = transition.sample()[0]
        >>> len(next_state) == 6  # [robot_x, robot_y, object_x, object_y, target_x, target_y]
        True
        >>> isinstance(next_state, np.ndarray)
        True
        >>> bool(next_state[0] > state[0])  # Robot moved right
        True
    """

    def __init__(
        self,
        state: np.ndarray,
        action: str,
        grid_size: int,
        push_threshold: float,
        friction_coefficient: float,
        obstacles: Optional[List[Tuple[float, float]]] = None,
        obstacle_radius: float = 0.5,
    ):
        super().__init__(state, action)
        self.grid_size = grid_size
        self.push_threshold = push_threshold
        self.friction_coefficient = friction_coefficient
        self.obstacles = obstacles if obstacles is not None else []
        self.obstacle_radius = obstacle_radius

        # State components: [robot_x, robot_y, object_x, object_y, target_x, target_y]
        self.robot_pos = state[:2]
        self.object_pos = state[2:4]
        self.target_pos = state[4:6]

        # Action to movement mapping
        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }

    def sample(self, n_samples: int = 1) -> List[Any]:
        # Get movement vector for the action
        movement = self.action_to_vector[self.action]

        # Calculate intended new robot position
        intended_robot_pos = self.robot_pos + movement

        # Check for collision with obstacles - if colliding, robot doesn't move
        if self._is_colliding_with_obstacle(intended_robot_pos):
            new_robot_pos = self.robot_pos  # Robot stays in place
        else:
            new_robot_pos = intended_robot_pos

        # Check if robot is close enough to push object
        distance_to_object = np.linalg.norm(new_robot_pos - self.object_pos)

        if distance_to_object < self.push_threshold:
            # Calculate intended object position after push
            push_force = movement * (1 - self.friction_coefficient)
            intended_object_pos = self.object_pos + push_force

            # Check for obstacle collision for object - if colliding, object doesn't move
            if self._is_colliding_with_obstacle(intended_object_pos):
                new_object_pos = self.object_pos  # Object stays in place
            else:
                new_object_pos = intended_object_pos
        else:
            new_object_pos = self.object_pos

        # Ensure positions stay within grid bounds
        new_robot_pos = np.clip(new_robot_pos, 0, self.grid_size - 1)
        new_object_pos = np.clip(new_object_pos, 0, self.grid_size - 1)

        # Combine all state components
        next_state = np.concatenate([new_robot_pos, new_object_pos, self.target_pos])
        return [next_state] * n_samples

    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate probability of transitioning to given next states.

        Since the push dynamics are deterministic, the probability is 1.0
        for the correct next state and 0.0 for all other states.

        Args:
            values: List of potential next states to evaluate

        Returns:
            Array of probabilities for each state in values
        """
        probabilities = []
        expected_next_state = self.sample()[0]

        for state in values:
            # Check if this state matches the expected deterministic next state
            if np.array_equal(state, expected_next_state):
                probabilities.append(1.0)
            else:
                probabilities.append(0.0)

        return np.array(probabilities)

    def _is_colliding_with_obstacle(self, position: np.ndarray) -> bool:
        """Check if a position collides with any obstacle."""
        if not self.obstacles:
            return False

        pos_x, pos_y = position

        for obs_x, obs_y in self.obstacles:
            # Calculate Euclidean distance
            distance = np.sqrt((pos_x - obs_x) ** 2 + (pos_y - obs_y) ** 2)
            if distance <= self.obstacle_radius:
                return True

        return False


class PushObservation(ObservationModel):
    """Noisy observation model for Push POMDP.

    This model provides partial observability by adding Gaussian noise to
    the object's position while keeping robot and target positions fully observable.
    This creates uncertainty about the exact object location, making planning challenging.

    Observation format: [robot_x, robot_y, noisy_object_x, noisy_object_y, target_x, target_y]

    Attributes:
        next_state: True state after action execution
        action: Action that was taken (not used in observation generation)
        observation_noise: Standard deviation of Gaussian noise for object position
        grid_size: Size of the grid environment
        robot_pos: Robot position (observed exactly)
        object_pos: True object position (observed with noise)
        target_pos: Target position (observed exactly)

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> # True state after robot movement
        >>> true_state = np.array([3.0, 3.0, 2.8, 3.2, 8.0, 8.0])
        >>> action = "right"

        >>> # Create observation model
        >>> obs_model = PushObservation(
        ...     next_state=true_state,
        ...     action=action,
        ...     observation_noise=0.1,
        ...     grid_size=10
        ... )

        >>> # Sample noisy observation
        >>> observation = obs_model.sample()[0]
        >>> len(observation) == 6  # [robot_x, robot_y, noisy_obj_x, noisy_obj_y, target_x, target_y]
        True
        >>> bool(observation[0] == 3.0)  # Robot position exact
        True
        >>> bool(observation[1] == 3.0)  # Robot position exact
        True
        >>> bool(observation[4] == 8.0)  # Target position exact
        True

        >>> # Calculate observation probability
        >>> prob = obs_model.probability([observation])
        >>> len(prob) == 1
        True
    """

    def __init__(
        self,
        next_state: np.ndarray,
        action: str,
        observation_noise: float,
        grid_size: int,
    ):
        super().__init__(next_state=next_state, action=action)
        self.observation_noise = observation_noise
        self.grid_size = grid_size

        # State components: [robot_x, robot_y, object_x, object_y, target_x, target_y]
        self.robot_pos = next_state[:2]
        self.object_pos = next_state[2:4]
        self.target_pos = next_state[4:6]

    def sample(self, n_samples: int = 1) -> List[Any]:
        observations = []
        for _ in range(n_samples):
            # Add noise to object position observation
            noisy_object_pos = self.object_pos + np.random.normal(0, self.observation_noise, size=2)
            noisy_object_pos = np.clip(noisy_object_pos, 0, self.grid_size - 1)

            # Combine observations (robot position is known exactly, target position is known)
            observation = np.concatenate([self.robot_pos, noisy_object_pos, self.target_pos])
            observations.append(observation)
        return observations

    def probability(self, values: List[Any]) -> np.ndarray:
        # Calculate probabilities based on Gaussian noise model for list of observations
        # Using 2D Gaussian PDF: (1/(2*pi*sigma^2)) * exp(-0.5 * ||diff||^2 / sigma^2)
        probabilities = []
        variance = self.observation_noise**2
        normalization = 1.0 / (2.0 * np.pi * variance)

        for observation in values:
            # Ensure observation is numpy array with correct shape
            if not isinstance(observation, np.ndarray) or observation.size == 0:
                raise ValueError(
                    f"Expected non-empty numpy array observation, got {type(observation)} with shape {getattr(observation, 'shape', 'unknown')}"
                )

            if observation.shape != (6,):
                raise ValueError(f"Expected observation shape (6,), got {observation.shape}")

            object_pos_diff = observation[2:4] - self.object_pos
            log_prob = -0.5 * np.sum(object_pos_diff**2) / variance
            prob = normalization * np.exp(log_prob)
            probabilities.append(prob)

        return np.array(probabilities)


class FixedStateDistribution(Distribution):
    """Deterministic distribution that always returns the same fixed state."""

    def __init__(self, state: np.ndarray):
        self.state = state.copy()

    def sample(self, n_samples: int = 1) -> List[Any]:
        return [self.state.copy() for _ in range(n_samples)]


class RandomInitialStateDistribution(Distribution):
    """Random initial state distribution for Push POMDP."""

    def __init__(
        self,
        grid_size: int,
        target_pos: np.ndarray,
        obstacles: List[Tuple[float, float]],
        obstacle_radius: float,
        parent: "PushPOMDP",
    ):
        self.grid_size = grid_size
        self.target_pos = target_pos
        self.obstacles = obstacles
        self.obstacle_radius = obstacle_radius
        self.parent = parent

    def sample(self, n_samples: int = 1) -> List[Any]:
        initial_states = []
        for _ in range(n_samples):
            robot_pos = self._generate_robot_position()
            object_pos = self._generate_object_position()
            initial_state = np.concatenate([robot_pos, object_pos, self.target_pos])
            initial_states.append(initial_state)
        return initial_states

    def _generate_robot_position(self) -> np.ndarray:
        max_attempts = 100
        for _ in range(max_attempts):
            robot_pos = np.random.uniform(0, self.grid_size - 1, size=2)
            if not self.parent._is_colliding_with_obstacle(robot_pos):
                return robot_pos
        return np.random.uniform(0, self.grid_size - 1, size=2)

    def _generate_object_position(self) -> np.ndarray:
        max_attempts = 100
        for _ in range(max_attempts):
            object_pos = np.random.uniform(0, self.grid_size - 1, size=2)
            if np.linalg.norm(
                object_pos - self.target_pos
            ) >= 2.0 and not self.parent._is_colliding_with_obstacle(object_pos):
                return object_pos
        return np.random.uniform(0, self.grid_size - 1, size=2)


class PushPOMDP(DiscreteActionsEnvironment):
    """Robotic push task formulated as a POMDP.

    This environment simulates a robot that must push an object to a target location
    on a 2D grid. The robot can move in four directions and pushes objects when close
    enough, with partial observability through noisy object position measurements.

    Problem Structure:
    - State: [robot_x, robot_y, object_x, object_y, target_x, target_y] (continuous)
    - Actions: ["up", "down", "left", "right"] (discrete)
    - Observations: [robot_x, robot_y, noisy_object_x, noisy_object_y, target_x, target_y]
    - Rewards: -distance_to_target + 100 (when object reaches target)
    - Termination: Object within 0.5 units of target position

    Key Features:
    - Physics-based pushing with configurable friction
    - Distance-based pushing threshold
    - Noisy observations of object position only
    - Dense reward signal based on object-target distance
    - Obstacle collision detection with configurable penalties
    - Obstacles prevent robot and object movement through them

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> env = PushPOMDP(discount_factor=0.99)
        >>>
        >>> # Get initial state and actions
        >>> initial_state = env.initial_state_dist().sample()[0]
        >>> actions = env.get_actions()
        >>>
        >>> # Sample complete step using convenience method
        >>> action = actions[0]
        >>> next_state, observation, reward = env.sample_next_step(initial_state, action)
        >>>
        >>> # Check terminal condition
        >>> env.is_terminal(initial_state)
        False
    """

    def __init__(
        self,
        discount_factor: float,
        grid_size: int = 10,
        push_threshold: float = 1.0,
        friction_coefficient: float = 0.3,
        observation_noise: float = 0.1,
        obstacles: Optional[List[Tuple[float, float]]] = None,
        obstacle_radius: float = 0.5,
        obstacle_penalty: float = -10.0,
        initial_state: Optional[np.ndarray] = None,
        name: str = "PushPOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        self.grid_size = grid_size
        self.push_threshold = push_threshold
        self.friction_coefficient = friction_coefficient
        self.observation_noise = observation_noise
        self.obstacles: List[Tuple[float, float]] = obstacles if obstacles is not None else []
        self.obstacle_radius = obstacle_radius
        self.obstacle_penalty = obstacle_penalty
        self._initial_state = initial_state

        # Define actions
        self.actions = ["up", "down", "right", "left"]

        # Initialize target position (fixed)
        self.target_pos = np.array([grid_size - 1, grid_size - 1])

        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # Action space is discrete positions
            observation_space=SpaceType.CONTINUOUS,  # Observation space is positions with noise
        )
        # Calculate reward range based on maximum distance to target
        # Maximum distance is diagonal from corner to corner: sqrt(2) * (grid_size - 1)
        max_distance = np.sqrt(2) * (grid_size - 1)
        min_reward = -max_distance  # Worst case: maximum distance to target
        max_reward = 100.0  # Best case: at target with bonus reward

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=(min_reward, max_reward),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }

    def _is_colliding_with_obstacle(
        self, position: np.ndarray, action: Optional[str] = None
    ) -> bool:
        """Check if a position collides with any obstacle.

        Args:
            position: Position to check as [x, y] array
            action: Optional action to check collision after movement. If None, checks current position.

        Returns:
            True if position is within obstacle_radius of any obstacle center
        """
        if not self.obstacles:
            return False

        if action is not None:
            # Check collision at next position after taking action
            check_pos_x, check_pos_y = position + self.action_to_vector[action]
        else:
            # Check collision at current position
            check_pos_x, check_pos_y = position

        for obs_x, obs_y in self.obstacles:
            # Calculate Euclidean distance
            distance = np.sqrt((check_pos_x - obs_x) ** 2 + (check_pos_y - obs_y) ** 2)
            if distance <= self.obstacle_radius:
                return True

        return False

    def state_transition_model(self, state: np.ndarray, action: str) -> StateTransitionModel:
        return PushStateTransition(
            state=state,
            action=action,
            grid_size=self.grid_size,
            push_threshold=self.push_threshold,
            friction_coefficient=self.friction_coefficient,
            obstacles=self.obstacles,
            obstacle_radius=self.obstacle_radius,
        )

    def observation_model(self, next_state: np.ndarray, action: str) -> ObservationModel:
        return PushObservation(
            next_state=next_state,
            action=action,
            observation_noise=self.observation_noise,
            grid_size=self.grid_size,
        )

    def reward(self, state: np.ndarray, action: str) -> float:
        # State components: [robot_x, robot_y, object_x, object_y, target_x, target_y]
        robot_pos = state[:2]
        object_pos = state[2:4]
        target_pos = state[4:6]

        # Calculate distance to target
        distance_to_target = np.linalg.norm(object_pos - target_pos)

        # Base reward is negative distance to encourage moving closer to target
        reward = -distance_to_target

        # Additional reward for reaching target
        if distance_to_target < 0.5:
            reward += 100.0

        if self._is_colliding_with_obstacle(robot_pos):
            reward += self.obstacle_penalty

        return float(reward)

    def is_terminal(self, state: np.ndarray) -> bool:
        object_pos = state[2:4]
        target_pos = state[4:6]

        # Episode ends when object is close to target
        # Ensure builtins.bool return (mypy compatibility)
        return bool(np.linalg.norm(object_pos - target_pos) < 0.5)

    def initial_state_dist(self) -> Distribution:
        # If a fixed initial state is provided, return a deterministic distribution
        if self._initial_state is not None:
            return FixedStateDistribution(self._initial_state)

        return RandomInitialStateDistribution(
            self.grid_size, self.target_pos, self.obstacles, self.obstacle_radius, self
        )

    def initial_observation_dist(self) -> Distribution:
        return self.initial_state_dist()

    def get_actions(self) -> List[str]:
        return self.actions

    def is_equal_observation(self, observation1: np.ndarray, observation2: np.ndarray) -> bool:
        return np.array_equal(observation1, observation2)

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache animated visualization of the push episode.

        Creates an animated GIF showing the robot pushing the object toward the target,
        with obstacles, collision detection, distance indicators, and success feedback.

        Args:
            history: Episode history containing states, actions, and rewards
            cache_path: Path where to save the visualization (must end with .gif)

        Raises:
            ValueError: If history is empty or cache_path doesn't end with .gif
            TypeError: If cache_path is not a Path object
        """
        visualizer = PushPOMDPVisualizer(self)
        visualizer.create_visualization(history, cache_path)

    def get_metric_names(self) -> List[str]:
        """Get names of Push POMDP specific metrics.

        Returns:
            List containing collision-related metric names
        """
        return [metric.value for metric in PushPOMDPMetrics]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        robot_collisions = []
        object_collisions = []
        total_collisions = []

        for history in histories:
            history_robot_collisions = 0
            history_object_collisions = 0
            total_steps = len(history.history)

            for step in history.history:
                robot_pos = step.state[:2]  # [robot_x, robot_y]
                object_pos = step.state[2:4]  # [object_x, object_y]

                if self._is_colliding_with_obstacle(robot_pos):
                    history_robot_collisions += 1

                if self._is_colliding_with_obstacle(object_pos):
                    history_object_collisions += 1

            if total_steps > 0:
                robot_collisions.append(history_robot_collisions)
                object_collisions.append(history_object_collisions)
                total_collisions.append(history_robot_collisions + history_object_collisions)

        total_steps_all = sum(len(history.history) for history in histories)
        avg_robot_collisions = sum(robot_collisions) / total_steps_all if total_steps_all > 0 else 0
        avg_object_collisions = (
            sum(object_collisions) / total_steps_all if total_steps_all > 0 else 0
        )
        avg_total_collisions = sum(total_collisions) / total_steps_all if total_steps_all > 0 else 0

        robot_collision_rates = [
            c / len(history.history) for c, history in zip(robot_collisions, histories)
        ]
        object_collision_rates = [
            c / len(history.history) for c, history in zip(object_collisions, histories)
        ]
        total_collision_rates = [
            c / len(history.history) for c, history in zip(total_collisions, histories)
        ]

        robot_collisions_ci = confidence_interval(data=robot_collision_rates, confidence=0.95)
        object_collisions_ci = confidence_interval(data=object_collision_rates, confidence=0.95)
        total_collisions_ci = confidence_interval(data=total_collision_rates, confidence=0.95)

        total_robot_collisions_ci = confidence_interval(data=robot_collisions, confidence=0.95)
        total_object_collisions_ci = confidence_interval(data=object_collisions, confidence=0.95)
        total_all_collisions_ci = confidence_interval(data=total_collisions, confidence=0.95)

        return [
            MetricValue(
                name=PushPOMDPMetrics.ROBOT_OBSTACLE_COLLISION_RATE.value,
                value=avg_robot_collisions,
                lower_confidence_bound=robot_collisions_ci[0],
                upper_confidence_bound=robot_collisions_ci[1],
            ),
            MetricValue(
                name=PushPOMDPMetrics.OBJECT_OBSTACLE_COLLISION_RATE.value,
                value=avg_object_collisions,
                lower_confidence_bound=object_collisions_ci[0],
                upper_confidence_bound=object_collisions_ci[1],
            ),
            MetricValue(
                name=PushPOMDPMetrics.TOTAL_OBSTACLE_COLLISION_RATE.value,
                value=avg_total_collisions,
                lower_confidence_bound=total_collisions_ci[0],
                upper_confidence_bound=total_collisions_ci[1],
            ),
            MetricValue(
                name=PushPOMDPMetrics.TOTAL_ROBOT_OBSTACLE_COLLISIONS.value,
                value=sum(robot_collisions),
                lower_confidence_bound=total_robot_collisions_ci[0],
                upper_confidence_bound=total_robot_collisions_ci[1],
            ),
            MetricValue(
                name=PushPOMDPMetrics.TOTAL_OBJECT_OBSTACLE_COLLISIONS.value,
                value=sum(object_collisions),
                lower_confidence_bound=total_object_collisions_ci[0],
                upper_confidence_bound=total_object_collisions_ci[1],
            ),
            MetricValue(
                name=PushPOMDPMetrics.TOTAL_ALL_OBSTACLE_COLLISIONS.value,
                value=sum(total_collisions),
                lower_confidence_bound=total_all_collisions_ci[0],
                upper_confidence_bound=total_all_collisions_ci[1],
            ),
        ]
