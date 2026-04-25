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
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
    ObservationModel,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.rock_sample_pomdp import _native
from POMDPPlanners.utils.statistics_utils import confidence_interval


class RockSamplePOMDPMetrics(Enum):
    """Metric names for RockSample POMDP environment."""

    AVG_ROCKS_SAMPLED = "avg_rocks_sampled"
    EXIT_SUCCESS_RATE = "exit_success_rate"
    AVERAGE_DANGEROUS_AREA_STEPS = "average_dangerous_area_steps"


# Type alias for RockSampleState
RockSampleState = np.ndarray


def create_rock_sample_state(
    robot_pos: Tuple[int, int], rocks: Tuple[bool, ...]
) -> RockSampleState:
    """Create a RockSample state as a numpy array.

    Args:
        robot_pos: Robot position as (row, col) tuple
        rocks: Tuple of booleans indicating rock quality (True=good, False=bad)

    Returns:
        State as numpy array: [robot_row, robot_col, rock_0, rock_1, ..., rock_n]
        where rock values are 1.0 for good (True) and 0.0 for bad (False)
    """
    state = np.zeros(2 + len(rocks), dtype=np.float32)
    state[0] = robot_pos[0]
    state[1] = robot_pos[1]
    state[2:] = np.array([1.0 if r else 0.0 for r in rocks], dtype=np.float32)
    return state


def get_robot_pos(state: RockSampleState) -> Tuple[int, int]:
    """Extract robot position from state array.

    Args:
        state: State array

    Returns:
        Robot position as (row, col) tuple
    """
    return (int(state[0]), int(state[1]))


def get_rocks(state: RockSampleState) -> Tuple[bool, ...]:
    """Extract rock qualities from state array.

    Args:
        state: State array

    Returns:
        Tuple of booleans indicating rock quality
    """
    return tuple(bool(r > 0.5) for r in state[2:])


def states_equal(state1: RockSampleState, state2: RockSampleState) -> bool:
    """Check if two states are equal.

    Args:
        state1: First state
        state2: Second state

    Returns:
        True if states are equal
    """
    return np.array_equal(state1, state2)


_OBS_CODE_TO_STR = ("none", "good", "bad")
_OBS_STR_TO_CODE = {"none": 0, "good": 1, "bad": 2}


class RockSampleStateTransitionModel(_native.RockSampleTransitionCpp):
    """State transition model for RockSample POMDP.

    Thin Python wrapper around the native C++ class. Deterministic dynamics:
    movement clamps to grid boundaries (East is unclamped to allow exit),
    sample flips the colocated rock to bad, and check actions leave the
    state unchanged. Terminal sentinel ``[-1, -1, ...]`` is absorbing.
    """

    def __init__(self, state: RockSampleState, action: int, pomdp: "RockSamplePOMDP"):
        """Initialize transition model.

        Args:
            state: Current state
            action: Action to execute
            pomdp: Reference to the POMDP environment
        """
        super().__init__(
            state=np.asarray(state, dtype=float),
            action=int(action),
            map_rows=pomdp.map_size[0],
            map_cols=pomdp.map_size[1],
            num_rocks=len(pomdp.rock_positions),
            rock_positions=np.asarray(pomdp.rock_positions, dtype=np.int32),
            sensor_efficiency=pomdp.sensor_efficiency,
        )
        self.pomdp = pomdp


StateTransitionModel.register(RockSampleStateTransitionModel)


class RockSampleObservationModel(ObservationModel):
    """Observation model for RockSample POMDP.

    Uses the native C++ sampler via composition (not inheritance) so the
    string-typed public observation API stays decoupled from the
    integer-coded C++ batch interface: callers exchange ``"none" / "good"
    / "bad"`` with this class, which translates to/from the C++ integer
    codes (0=none, 1=good, 2=bad) and delegates ``sample`` / ``probability``
    to the native implementation. The core belief update's batch path
    (which assumes numeric observations) therefore does not pick this
    class up via ``hasattr(..., "batch_log_likelihood")``; the vectorized
    updater calls the native extension directly for its batched path.
    """

    def __init__(self, next_state: RockSampleState, action: int, pomdp: "RockSamplePOMDP"):
        """Initialize observation model.

        Args:
            next_state: Next state after transition
            action: Action that was executed
            pomdp: Reference to the POMDP environment
        """
        super().__init__(next_state=next_state, action=action)
        self.pomdp = pomdp
        self._native = _native.RockSampleObservationCpp(
            next_state=np.asarray(next_state, dtype=float),
            action=int(action),
            map_rows=pomdp.map_size[0],
            map_cols=pomdp.map_size[1],
            num_rocks=len(pomdp.rock_positions),
            rock_positions=np.asarray(pomdp.rock_positions, dtype=np.int32),
            sensor_efficiency=pomdp.sensor_efficiency,
        )

    def sample(self, n_samples: int = 1) -> List[str]:
        """Sample observations as string labels."""
        return [_OBS_CODE_TO_STR[c] for c in self._native.sample(n_samples)]

    def probability(self, values: List[str]) -> np.ndarray:
        """Calculate observation probabilities for string observations."""
        codes = np.array([_OBS_STR_TO_CODE.get(v, -1) for v in values], dtype=np.int32)
        return self._native.probability(codes)


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

        # Cached int32 rock positions array; identical to what the per-call
        # wrappers build via ``np.asarray(...)``. Reused on the hot-path
        # native-kernel sample overrides to skip the per-call allocation.
        self._rock_positions_int32 = np.asarray(self.rock_positions, dtype=np.int32)

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

    def state_transition_model(self, state: RockSampleState, action: int) -> StateTransitionModel:
        """Get state transition model."""
        return RockSampleStateTransitionModel(  # pyright: ignore[reportReturnType]
            state, action, self
        )

    def observation_model(self, next_state: RockSampleState, action: int) -> ObservationModel:
        """Get observation model."""
        return RockSampleObservationModel(  # pyright: ignore[reportReturnType]
            next_state, action, self
        )

    def reward(self, state: RockSampleState, action: int) -> float:
        """Calculate immediate reward."""
        next_state = self.state_transition_model(state, action).sample()[0]
        return self._reward_from_next_state(state, action, next_state)

    def _reward_from_next_state(
        self, state: RockSampleState, action: int, next_state: RockSampleState
    ) -> float:
        total_reward = self.step_penalty

        robot_row, robot_col = get_robot_pos(state)
        if action == 2 and robot_col == self.map_size[1] - 1:
            total_reward += self.exit_reward
            return total_reward

        if action == 0:
            rocks = get_rocks(state)
            for i, rock_pos in enumerate(self.rock_positions):
                if (robot_row, robot_col) == rock_pos:
                    if rocks[i]:
                        total_reward += self.good_rock_reward
                    else:
                        total_reward += self.bad_rock_penalty
                    break

        if action >= 5:
            total_reward += self.sensor_use_penalty

        if self._is_in_dangerous_area(get_robot_pos(next_state)):
            total_reward += self.dangerous_area_penalty

        return total_reward

    # ── Hot-path sampling overrides ─────────────────────────────────
    # The default base-class implementations build a Python wrapper
    # (``RockSampleStateTransitionModel`` / ``RockSampleObservationModel``)
    # per call, which forwards to the native C++ kernel. The overrides
    # below construct the native kernel directly, skipping the wrapper
    # allocation while preserving the identical kernel-construction
    # sequence and arguments.

    def sample_next_state(self, state: RockSampleState, action: int, n_samples: int = 1) -> Any:
        kernel = _native.RockSampleTransitionCpp(
            state=np.asarray(state, dtype=float),
            action=int(action),
            map_rows=self.map_size[0],
            map_cols=self.map_size[1],
            num_rocks=len(self.rock_positions),
            rock_positions=self._rock_positions_int32,
            sensor_efficiency=self.sensor_efficiency,
        )
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def sample_observation(
        self, next_state: RockSampleState, action: int, n_samples: int = 1
    ) -> Any:
        kernel = _native.RockSampleObservationCpp(
            next_state=np.asarray(next_state, dtype=float),
            action=int(action),
            map_rows=self.map_size[0],
            map_cols=self.map_size[1],
            num_rocks=len(self.rock_positions),
            rock_positions=self._rock_positions_int32,
            sensor_efficiency=self.sensor_efficiency,
        )
        codes = kernel.sample(n_samples)
        if n_samples == 1:
            return _OBS_CODE_TO_STR[codes[0]]
        return [_OBS_CODE_TO_STR[c] for c in codes]

    def transition_log_probability(
        self, state: RockSampleState, action: int, next_states: Any
    ) -> np.ndarray:
        kernel = _native.RockSampleTransitionCpp(
            state=np.asarray(state, dtype=float),
            action=int(action),
            map_rows=self.map_size[0],
            map_cols=self.map_size[1],
            num_rocks=len(self.rock_positions),
            rock_positions=self._rock_positions_int32,
            sensor_efficiency=self.sensor_efficiency,
        )
        probs = np.asarray(kernel.probability(next_states))
        return np.log(probs + 1e-300)

    def observation_log_probability(
        self, next_state: RockSampleState, action: int, observations: Any
    ) -> np.ndarray:
        kernel = _native.RockSampleObservationCpp(
            next_state=np.asarray(next_state, dtype=float),
            action=int(action),
            map_rows=self.map_size[0],
            map_cols=self.map_size[1],
            num_rocks=len(self.rock_positions),
            rock_positions=self._rock_positions_int32,
            sensor_efficiency=self.sensor_efficiency,
        )
        codes = np.array([_OBS_STR_TO_CODE.get(v, -1) for v in observations], dtype=np.int32)
        probs = np.asarray(kernel.probability(codes))
        return np.log(probs + 1e-300)

    def sample_next_state_batch(self, states: Any, action: int) -> np.ndarray:
        states_array = np.ascontiguousarray(np.asarray(states, dtype=np.float64))
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        kernel = _native.RockSampleTransitionCpp(
            state=states_array[0],
            action=int(action),
            map_rows=self.map_size[0],
            map_cols=self.map_size[1],
            num_rocks=len(self.rock_positions),
            rock_positions=self._rock_positions_int32,
            sensor_efficiency=self.sensor_efficiency,
        )
        return np.asarray(kernel.batch_sample(states_array), dtype=np.float64)

    def observation_log_probability_per_state(
        self, next_states: Any, action: int, observation: Any
    ) -> np.ndarray:
        next_states_array = np.ascontiguousarray(np.asarray(next_states, dtype=np.float64))
        if next_states_array.ndim == 1:
            next_states_array = next_states_array.reshape(1, -1)
        kernel = _native.RockSampleObservationCpp(
            next_state=next_states_array[0],
            action=int(action),
            map_rows=self.map_size[0],
            map_cols=self.map_size[1],
            num_rocks=len(self.rock_positions),
            rock_positions=self._rock_positions_int32,
            sensor_efficiency=self.sensor_efficiency,
        )
        observation_code = _OBS_STR_TO_CODE.get(observation, -1)
        return np.asarray(
            kernel.batch_log_likelihood(
                next_particles=next_states_array,
                observation=int(observation_code),
            ),
            dtype=np.float64,
        )

    def sample_next_step(
        self, state: RockSampleState, action: int
    ) -> Tuple[RockSampleState, str, float]:
        """Override to avoid reward() recomputing next state."""
        next_state = self.sample_next_state(state=state, action=action)
        observation = self.sample_observation(next_state=next_state, action=action)
        reward = self._reward_from_next_state(state, action, next_state)
        return next_state, observation, reward

    def is_terminal(self, state: RockSampleState) -> bool:
        """Check if state is terminal."""
        return int(state[0]) == -1 and int(state[1]) == -1

    def initial_state_dist(self) -> DiscreteDistribution:
        """Get initial state distribution."""
        # All rocks start as good with equal probability
        num_rocks = len(self.rock_positions)
        possible_rock_states = []

        # Generate all possible rock configurations
        for i in range(2**num_rocks):
            rock_config = tuple(bool(i & (1 << j)) for j in range(num_rocks))
            initial_state = create_rock_sample_state(self.init_pos, rock_config)
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
                robot_pos = get_robot_pos(step.state)
                if self._is_in_dangerous_area(robot_pos):
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
        from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_visualizer import (  # pylint: disable=import-outside-toplevel
            RockSampleVisualizer,
        )

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
        from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_visualizer import (  # pylint: disable=import-outside-toplevel
            RockSampleVisualizer,
        )

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
