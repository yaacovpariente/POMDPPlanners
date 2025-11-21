"""Module for RockSample POMDP environment.

This module provides the RockSample POMDP environment implementation based on the
classic rock sampling problem.

The environment involves a robot navigating a grid world with rocks that are either
good or bad. The robot must use a noisy sensor to determine rock quality and decide
whether to sample them, balancing exploration and exploitation.

Classes:
    RockSampleState: Represents the state of the environment
    RockSamplePOMDP: The main POMDP environment implementation
"""

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_visualizer import (
    RockSampleVisualizer,
)
from POMDPPlanners.utils.statistics_utils import confidence_interval


class RockSamplePOMDPMetrics(Enum):
    """Metric names for RockSample POMDP environment."""

    AVG_ROCKS_SAMPLED = "avg_rocks_sampled"
    EXIT_SUCCESS_RATE = "exit_success_rate"
    AVERAGE_DANGEROUS_AREA_STEPS = "average_dangerous_area_steps"


@dataclass(frozen=True)
class RockSampleState:
    """State representation for RockSample POMDP.

    Attributes:
        robot_pos: Robot position as (row, col) tuple
        rocks: Boolean array indicating rock quality (True=good, False=bad)
    """

    robot_pos: Tuple[int, int]
    rocks: Tuple[bool, ...]  # Tuple for immutability

    def __post_init__(self):
        """Validate state components."""
        if not isinstance(self.robot_pos, tuple) or len(self.robot_pos) != 2:
            raise ValueError("robot_pos must be a tuple of two integers")
        if not isinstance(self.rocks, tuple):
            raise ValueError("rocks must be a tuple of booleans")


class RockSampleStateTransitionModel(StateTransitionModel):
    """State transition model for RockSample POMDP."""

    def __init__(self, state: RockSampleState, action: int, pomdp: "RockSamplePOMDP"):
        """Initialize transition model.

        Args:
            state: Current state
            action: Action to execute
            pomdp: Reference to the POMDP environment
        """
        super().__init__(state=state, action=action)
        self.pomdp = pomdp

    def sample(self, n_samples: int = 1) -> List[RockSampleState]:
        """Sample next states (deterministic transitions)."""
        next_state = self._compute_next_state()
        return [next_state] * n_samples

    def probability(self, values: List[RockSampleState]) -> np.ndarray:
        """Calculate transition probabilities for given next states.

        Since RockSample has deterministic transitions, the probability is 1.0
        for the correct next state and 0.0 for all others.

        Args:
            values: List of next state values to calculate probabilities for

        Returns:
            Array of transition probabilities (1.0 for correct state, 0.0 otherwise)
        """
        # Compute the deterministic next state
        expected_next_state = self._compute_next_state()

        # Check which states match the expected next state
        probs = np.array([1.0 if state == expected_next_state else 0.0 for state in values])

        return probs

    def _compute_next_state(self) -> RockSampleState:
        """Compute the deterministic next state."""
        robot_row, robot_col = self.state.robot_pos
        rocks = list(self.state.rocks)

        # Handle terminal state
        if robot_col >= self.pomdp.map_size[1]:
            return RockSampleState((-1, -1), tuple(rocks))

        # Movement actions
        if self.action == 1:  # North
            new_pos = (max(0, robot_row - 1), robot_col)
        elif self.action == 2:  # East
            new_pos = (
                robot_row,
                robot_col + 1,
            )  # Allow moving beyond boundary for exit
        elif self.action == 3:  # South
            new_pos = (min(self.pomdp.map_size[0] - 1, robot_row + 1), robot_col)
        elif self.action == 4:  # West
            new_pos = (robot_row, max(0, robot_col - 1))
        elif self.action == 0:  # Sample
            new_pos = (robot_row, robot_col)
            # Check if robot is at a rock position and sample it
            for i, rock_pos in enumerate(self.pomdp.rock_positions):
                if (robot_row, robot_col) == rock_pos:
                    rocks[i] = False  # Rock becomes bad after sampling
                    break
        else:  # Check actions (5 and above)
            new_pos = (robot_row, robot_col)  # Stay in place for checking

        # Handle exit condition - robot must move beyond right boundary to exit
        if new_pos[1] >= self.pomdp.map_size[1]:
            return RockSampleState((-1, -1), tuple(rocks))

        return RockSampleState(new_pos, tuple(rocks))


class RockSampleObservationModel(ObservationModel):
    """Observation model for RockSample POMDP."""

    def __init__(self, next_state: RockSampleState, action: int, pomdp: "RockSamplePOMDP"):
        """Initialize observation model.

        Args:
            next_state: Next state after transition
            action: Action that was executed
            pomdp: Reference to the POMDP environment
        """
        super().__init__(next_state=next_state, action=action)
        self.pomdp = pomdp

    def sample(self, n_samples: int = 1) -> List[str]:
        """Sample observations."""
        if self.action <= 4:  # Movement or sample actions
            return ["none"] * n_samples

        # Check actions (5 and above)
        rock_idx = self.action - 5
        if rock_idx >= len(self.pomdp.rock_positions):
            return ["none"] * n_samples

        # Calculate observation probabilities based on distance and rock quality
        robot_pos = self.next_state.robot_pos
        rock_pos = self.pomdp.rock_positions[rock_idx]
        rock_quality = self.next_state.rocks[rock_idx]

        # Calculate Euclidean distance
        distance = math.sqrt((robot_pos[0] - rock_pos[0]) ** 2 + (robot_pos[1] - rock_pos[1]) ** 2)

        # Sensor efficiency decreases exponentially with distance
        efficiency = math.exp(-distance / self.pomdp.sensor_efficiency)

        observations = []
        for _ in range(n_samples):
            if np.random.random() < efficiency:
                # Correct observation with high probability
                obs = "good" if rock_quality else "bad"
            else:
                # Incorrect observation
                obs = "bad" if rock_quality else "good"
            observations.append(obs)

        return observations

    def probability(self, values: List[str]) -> np.ndarray:
        """Calculate observation probabilities."""
        if self.action <= 4:  # Movement or sample actions
            probs = np.array([1.0 if obs == "none" else 0.0 for obs in values])
            return probs

        # Check actions
        rock_idx = self.action - 5
        if rock_idx >= len(self.pomdp.rock_positions):
            probs = np.array([1.0 if obs == "none" else 0.0 for obs in values])
            return probs

        robot_pos = self.next_state.robot_pos
        rock_pos = self.pomdp.rock_positions[rock_idx]
        rock_quality = self.next_state.rocks[rock_idx]

        distance = math.sqrt((robot_pos[0] - rock_pos[0]) ** 2 + (robot_pos[1] - rock_pos[1]) ** 2)

        efficiency = math.exp(-distance / self.pomdp.sensor_efficiency)

        probs = []
        for obs in values:
            if obs == "none":
                prob = 0.0
            elif obs == "good":
                prob = efficiency if rock_quality else (1.0 - efficiency)
            elif obs == "bad":
                prob = (1.0 - efficiency) if rock_quality else efficiency
            else:
                prob = 0.0
            probs.append(prob)

        return np.array(probs)


class RockSamplePOMDP(DiscreteActionsEnvironment):
    """RockSample POMDP environment

    This environment implements the classic rock sampling problem where a robot
    must navigate a grid, use sensors to evaluate rocks, and decide which ones
    to sample while balancing exploration costs and sampling rewards.

    Attributes:
        map_size: Grid dimensions as (rows, cols)
        rock_positions: List of rock positions as (row, col) tuples
        init_pos: Initial robot position
        sensor_efficiency: Sensor noise parameter (higher = less noise)
        bad_rock_penalty: Penalty for sampling a bad rock
        good_rock_reward: Reward for sampling a good rock
        step_penalty: Cost for each action
        sensor_use_penalty: Additional cost for using sensor
        exit_reward: Reward for reaching the exit

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> env = RockSamplePOMDP(map_size=(5, 5), rock_positions=[(0, 0), (2, 2), (3, 3)])
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
        map_size: Tuple[int, int] = (5, 5),
        rock_positions: Optional[List[Tuple[int, int]]] = None,
        init_pos: Tuple[int, int] = (0, 0),
        sensor_efficiency: float = 10.0,
        bad_rock_penalty: float = -10.0,
        good_rock_reward: float = 10.0,
        step_penalty: float = 0.0,
        sensor_use_penalty: float = 0.0,
        exit_reward: float = 10.0,
        dangerous_areas: Optional[List[Tuple[int, int]]] = None,
        dangerous_area_radius: float = 1.0,
        dangerous_area_penalty: float = 5.0,
        discount_factor: float = 0.95,
        name: str = "RockSample",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize RockSample POMDP.

        Args:
            map_size: Grid dimensions (rows, cols). Defaults to (5, 5).
            rock_positions: Rock locations. Defaults to [(0,0), (2,2), (3,3)].
            init_pos: Initial robot position. Defaults to (0, 0).
            sensor_efficiency: Sensor parameter. Defaults to 20.0.
            bad_rock_penalty: Bad rock penalty. Defaults to -10.0.
            good_rock_reward: Good rock reward. Defaults to 10.0.
            step_penalty: Action cost. Defaults to 0.0.
            sensor_use_penalty: Sensor cost. Defaults to 0.0.
            exit_reward: Exit reward. Defaults to 10.0.
            dangerous_areas: List of dangerous area center positions as (row, col) tuples. Defaults to None.
            dangerous_area_radius: Radius around dangerous area centers. Defaults to 1.0.
            dangerous_area_penalty: Penalty magnitude applied randomly when in dangerous areas. Defaults to 5.0.
            discount_factor: Discount factor. Defaults to 0.95.
            name: Environment name. Defaults to "RockSample".
            output_dir: Output directory for logging. Defaults to None.
            debug: Enable debug logging. Defaults to False.
        """
        # Calculate reward range based on parameters
        min_reward = step_penalty + bad_rock_penalty + sensor_use_penalty
        max_reward = step_penalty + exit_reward

        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=(min_reward, max_reward),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.map_size = map_size
        self.rock_positions = (
            rock_positions if rock_positions is not None else [(0, 0), (2, 2), (3, 3)]
        )
        self.init_pos = init_pos
        self.sensor_efficiency = sensor_efficiency
        self.bad_rock_penalty = bad_rock_penalty
        self.good_rock_reward = good_rock_reward
        self.step_penalty = step_penalty
        self.sensor_use_penalty = sensor_use_penalty
        self.exit_reward = exit_reward
        self.dangerous_areas: List[Tuple[int, int]] = (
            dangerous_areas if dangerous_areas is not None else []
        )
        self.dangerous_area_radius = dangerous_area_radius
        self.dangerous_area_penalty = dangerous_area_penalty

        # Validate parameters
        self._validate_parameters()

        # Define actions: 0=sample, 1=north, 2=east, 3=south, 4=west, 5+=check_rock_i
        self.action_names = ["sample", "north", "east", "south", "west"]
        self.action_names.extend([f"check_rock_{i}" for i in range(len(self.rock_positions))])

        # Action to direction vector mapping for visualization
        self.action_to_vector = {
            0: (0, 0),  # sample - no movement
            1: (0, -1),  # north - up (negative row)
            2: (1, 0),  # east - right (positive col)
            3: (0, 1),  # south - down (positive row)
            4: (-1, 0),  # west - left (negative col)
        }
        # Check actions don't involve movement
        for i in range(5, len(self.action_names)):
            self.action_to_vector[i] = (0, 0)

    def _validate_parameters(self):
        """Validate environment parameters."""
        if self.map_size[0] <= 0 or self.map_size[1] <= 0:
            raise ValueError("Map size must be positive")
        for pos in self.rock_positions:
            if not (0 <= pos[0] < self.map_size[0] and 0 <= pos[1] < self.map_size[1]):
                raise ValueError(f"Rock position {pos} is outside map bounds {self.map_size}")
        if not (
            0 <= self.init_pos[0] < self.map_size[0] and 0 <= self.init_pos[1] < self.map_size[1]
        ):
            raise ValueError(
                f"Initial position {self.init_pos} is outside map bounds {self.map_size}"
            )

    def _is_in_dangerous_area(self, position: Tuple[int, int]) -> bool:
        """Check if a position is within any dangerous area.

        Args:
            position: Position to check as (row, col) tuple

        Returns:
            True if position is within radius of any dangerous area center
        """
        if not self.dangerous_areas:
            return False

        pos_row, pos_col = position

        for danger_row, danger_col in self.dangerous_areas:
            # Calculate Euclidean distance
            distance = math.sqrt((pos_row - danger_row) ** 2 + (pos_col - danger_col) ** 2)
            if distance <= self.dangerous_area_radius:
                return True

        return False

    def get_actions(self) -> List[int]:
        """Get all available actions."""
        return list(range(len(self.action_names)))

    def state_transition_model(
        self, state: RockSampleState, action: int
    ) -> RockSampleStateTransitionModel:
        """Get state transition model."""
        return RockSampleStateTransitionModel(state, action, self)

    def observation_model(
        self, next_state: RockSampleState, action: int
    ) -> RockSampleObservationModel:
        """Get observation model."""
        return RockSampleObservationModel(next_state, action, self)

    def reward(self, state: RockSampleState, action: int) -> float:
        """Calculate immediate reward."""
        total_reward = self.step_penalty

        # Check if robot exits the grid
        robot_row, robot_col = state.robot_pos
        if action == 2 and robot_col == self.map_size[1] - 1:  # East at right edge
            total_reward += self.exit_reward
            return total_reward

        # Sample action rewards
        if action == 0:  # Sample
            for i, rock_pos in enumerate(self.rock_positions):
                if (robot_row, robot_col) == rock_pos:
                    if state.rocks[i]:  # Good rock
                        total_reward += self.good_rock_reward
                    else:  # Bad rock
                        total_reward += self.bad_rock_penalty
                    break

        # Sensor use penalty
        if action >= 5:  # Check actions
            total_reward += self.sensor_use_penalty

        # Add dangerous area penalty/bonus with 50% probability
        if self._is_in_dangerous_area(state.robot_pos):
            # Random penalty or bonus with equal probability
            danger_modifier = (
                self.dangerous_area_penalty
                if np.random.random() < 0.5
                else -self.dangerous_area_penalty
            )
            total_reward += danger_modifier

        return total_reward

    def is_terminal(self, state: RockSampleState) -> bool:
        """Check if state is terminal."""
        return state.robot_pos == (-1, -1)

    def initial_state_dist(self) -> DiscreteDistribution:
        """Get initial state distribution."""
        # All rocks start as good with equal probability
        num_rocks = len(self.rock_positions)
        possible_rock_states = []

        # Generate all possible rock configurations
        for i in range(2**num_rocks):
            rock_config = tuple(bool(i & (1 << j)) for j in range(num_rocks))
            initial_state = RockSampleState(self.init_pos, rock_config)
            possible_rock_states.append(initial_state)

        # Equal probability for all configurations
        probs = np.ones(len(possible_rock_states)) / len(possible_rock_states)

        return DiscreteDistribution(values=possible_rock_states, probs=probs)

    def initial_observation_dist(self) -> DiscreteDistribution:
        """Get initial observation distribution."""
        return DiscreteDistribution(values=["none"], probs=np.array([1.0]))

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Check if two observations are equal."""
        return observation1 == observation2

    def get_metric_names(self) -> List[str]:
        """Get names of RockSample POMDP specific metrics.

        Returns:
            List containing metric names: avg_rocks_sampled, exit_success_rate,
            and average_dangerous_area_steps
        """
        return [metric.value for metric in RockSamplePOMDPMetrics]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute environment-specific metrics."""
        if not histories:
            return []

        metrics = []

        # Calculate average number of rocks sampled
        rocks_sampled = []
        for history in histories:
            sampled_count = 0
            for step in history.history:
                if hasattr(step, "action") and step.action == 0:  # Sample action
                    sampled_count += 1
            rocks_sampled.append(sampled_count)

        if rocks_sampled:
            mean_rocks = float(np.mean(rocks_sampled))
            ci_low, ci_high = confidence_interval(rocks_sampled)
            metrics.append(
                MetricValue(
                    name=RockSamplePOMDPMetrics.AVG_ROCKS_SAMPLED.value,
                    value=mean_rocks,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )

        # Calculate exit success rate
        exits = [
            1 if any(self.is_terminal(step.state) for step in history.history) else 0
            for history in histories
        ]

        if exits:
            exit_rate = float(np.mean(exits))
            ci_low, ci_high = confidence_interval(exits)
            metrics.append(
                MetricValue(
                    name=RockSamplePOMDPMetrics.EXIT_SUCCESS_RATE.value,
                    value=exit_rate,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )

        # Calculate dangerous area metrics
        dangerous_area_steps = []
        for history in histories:
            steps_in_danger = 0
            for step in history.history:
                if self._is_in_dangerous_area(step.state.robot_pos):
                    steps_in_danger += 1
            dangerous_area_steps.append(steps_in_danger)

        if dangerous_area_steps:
            avg_dangerous_steps = float(np.mean(dangerous_area_steps))
            ci_low, ci_high = confidence_interval(dangerous_area_steps)

            metrics.append(
                MetricValue(
                    name=RockSamplePOMDPMetrics.AVERAGE_DANGEROUS_AREA_STEPS.value,
                    value=avg_dangerous_steps,
                    lower_confidence_bound=ci_low,
                    upper_confidence_bound=ci_high,
                )
            )

        return metrics

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache visualization of episode history.

        Args:
            history: Episode history containing states, actions, and rewards
            cache_path: Path where to save the visualization (must end with .gif)
        """
        visualizer = RockSampleVisualizer(self)
        visualizer.create_visualization(history, cache_path)

    def visualize_path(
        self, path: List["RockSampleState"], actions: List[int], cache_path: Path
    ) -> None:
        """Visualize robot path through the environment.

        Args:
            path: List of states representing the path
            actions: List of actions taken at each state
            cache_path: Path where to save the animation (must end with .gif)
        """
        visualizer = RockSampleVisualizer(self)
        visualizer.visualize_path(path, actions, cache_path)


def create_random_rock_sample(
    map_size: int = 7, num_rocks: int = 8, seed: Optional[int] = None
) -> RockSamplePOMDP:
    """Create a random RockSample instance.

    Args:
        map_size: Size of square grid. Defaults to 7.
        num_rocks: Number of rocks to place. Defaults to 8.
        seed: Random seed. Defaults to None.

    Returns:
        Randomly configured RockSample POMDP
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random rock positions
    all_positions = [(r, c) for r in range(map_size) for c in range(map_size)]
    rock_positions = list(
        np.random.choice(len(all_positions), size=min(num_rocks, len(all_positions)), replace=False)
    )
    rock_positions = [all_positions[i] for i in rock_positions]

    return RockSamplePOMDP(
        map_size=(map_size, map_size), rock_positions=rock_positions, init_pos=(0, 0)
    )
