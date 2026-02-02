"""LaserTag POMDP Environment Implementation.

This module implements the LaserTag problem, a pursuit-evasion POMDP environment
where an agent must navigate a grid to tag an opponent that moves stochastically.
The agent has noisy observations of the opponent's location.

The LaserTag problem features:
- A grid-based environment (default 7x11) with optional walls
- Robot and opponent moving on discrete grid cells
- 5 possible actions: North, South, East, West, Tag
- 8-directional laser range measurements with Gaussian noise
- Positive reward for successful tagging, negative reward for failed tag attempts
- Step cost for each movement action
- Opponent moves with 0.4 prob toward robot in x-dir, 0.4 prob toward robot in y-dir, 0.2 prob stay

Classes:
    LaserTagState: State representation with robot and opponent positions
    LaserTagStateTransition: State transition model for robot and opponent movement
    LaserTagObservation: Observation model with noisy opponent position measurements
    LaserTagPOMDP: Main environment class implementing the LaserTag problem
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.utils.statistics_utils import confidence_interval


class LaserTagPOMDPMetrics(Enum):
    """Metric names for LaserTag POMDP environment."""

    TAG_SUCCESS_RATE = "tag_success_rate"
    GOAL_REACHING_RATE = "goal_reaching_rate"
    AVERAGE_EPISODE_LENGTH = "average_episode_length"
    AVERAGE_FAILED_TAG_ATTEMPTS = "average_failed_tag_attempts"
    AVERAGE_OBSTACLE_COLLISIONS = "average_obstacle_collisions"
    AVERAGE_DANGEROUS_AREA_STEPS = "average_dangerous_area_steps"
    AVERAGE_ALL_DANGEROUS_ENCOUNTERS = "average_all_dangerous_encounters"


# State representation for LaserTag POMDP as numpy array
# LaserTagState: np.ndarray with shape (5,) and dtype float64
#
# State vector structure:
#   Index 0: Robot row position (int stored as float)
#   Index 1: Robot column position (int stored as float)
#   Index 2: Opponent row position (int stored as float)
#   Index 3: Opponent column position (int stored as float)
#   Index 4: Terminal flag (0.0 = non-terminal, 1.0 = terminal)
#
# Example:
#   state = np.array([0.0, 0.0, 6.0, 10.0, 0.0])
#   # Robot at (0, 0), opponent at (6, 10), non-terminal
#
# Access patterns:
#   robot_row = int(state[0])
#   robot_col = int(state[1])
#   robot_pos = (int(state[0]), int(state[1]))
#   opponent_row = int(state[2])
#   opponent_col = int(state[3])
#   opponent_pos = (int(state[2]), int(state[3]))
#   is_terminal = bool(state[4])


class LaserTagStateTransition(StateTransitionModel):
    """State transition model for LaserTag POMDP.

    Handles robot movement (deterministic based on action) and opponent movement
    (probabilistic, with tendency to move toward robot's position).

    Attributes:
        state: Current state as numpy array (shape (5,))
        action: Action to be executed (0=North, 1=South, 2=East, 3=West, 4=Tag)
        floor_shape: Tuple of (rows, cols) for grid dimensions
        walls: Set of wall positions

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> state = np.array([3.0, 5.0, 2.0, 4.0, 0.0])  # Robot at (3,5), opponent at (2,4)
        >>> action_directions = {
        ...     0: (-1, 0),  # North (up)
        ...     1: (1, 0),   # South (down)
        ...     2: (0, 1),   # East (right)
        ...     3: (0, -1),  # West (left)
        ...     4: (0, 0),   # Tag (no movement)
        ... }
        >>> transition = LaserTagStateTransition(
        ...     state=state,
        ...     action=0,  # North
        ...     action_directions=action_directions,
        ...     floor_shape=(7, 11),
        ...     walls=set()
        ... )
        >>> next_states = transition.sample(n_samples=5)
        >>> probabilities = transition.probability(next_states)
    """

    def __init__(
        self,
        state: np.ndarray,
        action: int,
        action_directions: Dict[int, Tuple[int, int]],
        floor_shape: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        transition_error_prob: float = 0.0,
    ):
        """Initialize the state transition model.

        Args:
            state: Current state as numpy array with shape (5,)
            action: Action to execute (0=North, 1=South, 2=East, 3=West, 4=Tag)
            floor_shape: Grid dimensions as (rows, cols)
            walls: Set of wall positions as (row, col) tuples
            transition_error_prob: Probability that the robot executes a random movement action
                instead of the intended one. Only applies to movement actions (0-3), not Tag (4).
                Defaults to 0.0 (deterministic transitions).
        """
        super().__init__(state, action)
        self.floor_shape: Tuple[int, int] = floor_shape
        self.walls: Set[Tuple[int, int]] = walls
        self.action_directions: Dict[int, Tuple[int, int]] = action_directions
        self.transition_error_prob = transition_error_prob

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a wall."""
        row, col = pos
        return (
            0 <= row < self.floor_shape[0]
            and 0 <= col < self.floor_shape[1]
            and pos not in self.walls
        )

    def _get_robot_next_position(self, action: Optional[int] = None) -> Tuple[int, int]:
        """Get robot's next position based on action.

        Args:
            action: Action to execute. If None, uses self.action. Defaults to None.

        Returns:
            Robot's next position as (row, col) tuple.
        """
        actual_action = self.action if action is None else action

        if actual_action == 4:  # Tag action
            return (int(self.state[0]), int(self.state[1]))

        dr, dc = self.action_directions[actual_action]
        new_pos = (int(self.state[0]) + dr, int(self.state[1]) + dc)

        # If new position is invalid, stay at current position
        if self._is_valid_position(new_pos):
            return new_pos
        else:
            return (int(self.state[0]), int(self.state[1]))

    def _create_position(
        self, fixed_coord: int, moving_coord: int, is_horizontal: bool
    ) -> Tuple[int, int]:
        """Create a position tuple based on axis orientation."""
        if is_horizontal:
            return (fixed_coord, moving_coord)  # (row, col)
        else:
            return (moving_coord, fixed_coord)  # (row, col)

    def _calculate_directional_moves(
        self, opponent_coord: int, robot_coord: int, fixed_coord: int, is_horizontal: bool
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Calculate movement probabilities for one axis (horizontal or vertical)."""
        moves = []

        if robot_coord > opponent_coord:  # Robot is ahead
            toward_pos = self._create_position(fixed_coord, opponent_coord + 1, is_horizontal)
            away_pos = self._create_position(fixed_coord, opponent_coord - 1, is_horizontal)
            toward_prob, away_prob = 0.4, 0.0
        elif robot_coord < opponent_coord:  # Robot is behind
            toward_pos = self._create_position(fixed_coord, opponent_coord - 1, is_horizontal)
            away_pos = self._create_position(fixed_coord, opponent_coord + 1, is_horizontal)
            toward_prob, away_prob = 0.4, 0.0
        else:  # Same position on this axis
            toward_pos = self._create_position(fixed_coord, opponent_coord + 1, is_horizontal)
            away_pos = self._create_position(fixed_coord, opponent_coord - 1, is_horizontal)
            toward_prob, away_prob = 0.0, 0.0

        if self._is_valid_position(toward_pos):
            moves.append((toward_pos, toward_prob))
        if self._is_valid_position(away_pos):
            moves.append((away_pos, away_prob))

        return moves

    def _get_horizontal_moves(
        self, robot_col: int, opp_row: int, opp_col: int
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Get opponent's horizontal (column) movement options."""
        return self._calculate_directional_moves(
            opponent_coord=opp_col,
            robot_coord=robot_col,
            fixed_coord=opp_row,
            is_horizontal=True,
        )

    def _get_vertical_moves(
        self, robot_row: int, opp_row: int, opp_col: int
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Get opponent's vertical (row) movement options."""
        return self._calculate_directional_moves(
            opponent_coord=opp_row,
            robot_coord=robot_row,
            fixed_coord=opp_col,
            is_horizontal=False,
        )

    def _normalize_move_probabilities(
        self,
        directional_moves: List[Tuple[Tuple[int, int], float]],
        stay_position: Tuple[int, int],
        stay_prob: float,
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Normalize movement probabilities and handle blocked moves."""
        # Add stay option
        move_probs = directional_moves + [(stay_position, stay_prob)]

        # Calculate actual total probability
        actual_total = sum(prob for _, prob in move_probs if prob > 0)

        # Redistribute remaining probability to staying if moves are blocked
        if actual_total < 1.0:
            stay_index = len(move_probs) - 1
            current_pos, current_stay_prob = move_probs[stay_index]
            move_probs[stay_index] = (current_pos, current_stay_prob + (1.0 - actual_total))

        # Filter out zero probability moves
        return [(pos, prob) for pos, prob in move_probs if prob > 0]

    def _get_opponent_move_probabilities(
        self, robot_pos: Tuple[int, int]
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Get opponent's movement probabilities based on robot position.

        Uses the following movement model:
        - 0.4 probability to move in x-direction (toward/away from robot)
        - 0.4 probability to move in y-direction (toward/away from robot)
        - 0.2 probability to stay in place
        """
        current_opp = (int(self.state[2]), int(self.state[3]))
        robot_row, robot_col = robot_pos
        opp_row, opp_col = current_opp

        # Get movement options for each axis
        x_moves = self._get_horizontal_moves(robot_col, opp_row, opp_col)
        y_moves = self._get_vertical_moves(robot_row, opp_row, opp_col)

        # Combine and normalize
        return self._normalize_move_probabilities(x_moves + y_moves, current_opp, stay_prob=0.2)

    def _get_actual_action(self) -> int:
        """Get the actual action to execute, accounting for transition errors.

        Returns:
            The action that will actually be executed. For Tag action (4), always returns
            the intended action. For movement actions (0-3), with probability (1-p)
            returns the intended action, with probability p returns a uniformly random
            action from {0,1,2,3} excluding the intended action.
        """
        # Tag action always executes correctly
        if self.action == 4:
            return self.action

        # For movement actions, apply error probability
        if np.random.random() < self.transition_error_prob:
            # Select uniformly from {0,1,2,3} excluding the intended action
            available_actions = [a for a in [0, 1, 2, 3] if a != self.action]
            return np.random.choice(available_actions)
        else:
            return self.action

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        """Sample next states from the transition model."""
        samples = []
        # Get actual action to execute (may differ from intended due to errors)
        actual_action = self._get_actual_action()
        robot_next = self._get_robot_next_position(action=actual_action)
        robot_current = (int(self.state[0]), int(self.state[1]))
        opponent_current = (int(self.state[2]), int(self.state[3]))

        # Check if tagging occurred (using actual action)
        if actual_action == 4 and robot_current == opponent_current:
            # Successful tag - terminal state
            for _ in range(n_samples):
                samples.append(
                    np.array(
                        [
                            float(robot_next[0]),
                            float(robot_next[1]),
                            float(opponent_current[0]),
                            float(opponent_current[1]),
                            1.0,
                        ]
                    )
                )
        else:
            # Regular transition
            opp_moves = self._get_opponent_move_probabilities(robot_next)
            positions, probabilities = zip(*opp_moves)

            for _ in range(n_samples):
                opp_next = np.random.choice(len(positions), p=probabilities)
                opp_next_pos = positions[opp_next]
                samples.append(
                    np.array(
                        [
                            float(robot_next[0]),
                            float(robot_next[1]),
                            float(opp_next_pos[0]),
                            float(opp_next_pos[1]),
                            0.0,
                        ]
                    )
                )

        return samples

    def _compute_transition_probability_for_action(
        self, next_state: np.ndarray, action: int
    ) -> float:
        """Compute transition probability for a given next state and action.

        Args:
            next_state: The next state as numpy array with shape (5,)
            action: The action to consider

        Returns:
            Probability of transitioning to next_state given action
        """
        if not isinstance(next_state, np.ndarray) or len(next_state) != 5:
            return 0.0

        robot_next = self._get_robot_next_position(action=action)
        robot_current = (int(self.state[0]), int(self.state[1]))
        opponent_current = (int(self.state[2]), int(self.state[3]))

        next_robot = (int(next_state[0]), int(next_state[1]))
        next_opponent = (int(next_state[2]), int(next_state[3]))
        next_terminal = bool(next_state[4])

        # Check if tagging occurred
        if action == 4 and robot_current == opponent_current:
            # Successful tag case
            if next_robot == robot_next and next_opponent == opponent_current and next_terminal:
                return 1.0
            else:
                return 0.0
        else:
            # Regular transition case
            if next_robot == robot_next and not next_terminal:
                opp_moves = self._get_opponent_move_probabilities(robot_next)
                # Find probability for this opponent position
                for opp_pos, prob in opp_moves:
                    if next_opponent == opp_pos:
                        return prob
            return 0.0

    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate transition probabilities for given next states."""
        result = np.zeros(len(values))

        # Tag action always executes correctly (no errors)
        if self.action == 4:
            for i, next_state in enumerate(values):
                result[i] = self._compute_transition_probability_for_action(next_state, self.action)
        else:
            # Movement action: account for error probability
            # With probability (1-p): execute intended action
            # With probability p: execute one of 3 error actions uniformly
            for i, next_state in enumerate(values):
                # Probability from intended action
                prob_intended = (1.0 - self.transition_error_prob) * (
                    self._compute_transition_probability_for_action(next_state, self.action)
                )

                # Probability from error actions
                error_actions = [a for a in [0, 1, 2, 3] if a != self.action]
                prob_error = 0.0
                if self.transition_error_prob > 0.0 and len(error_actions) > 0:
                    error_prob_sum = sum(
                        self._compute_transition_probability_for_action(next_state, error_action)
                        for error_action in error_actions
                    )
                    prob_error = (
                        self.transition_error_prob * (1.0 / len(error_actions)) * error_prob_sum
                    )

                result[i] = prob_intended + prob_error

        return result


class LaserTagObservation(ObservationModel):
    """Observation model for LaserTag POMDP.

    Provides 8-directional laser range measurements from the robot's position.
    Each measurement represents the number of clear cells in that direction
    before hitting a wall or boundary, with Gaussian noise.

    Attributes:
        next_state: The state after action execution as numpy array (shape (5,))
        action: The action that was taken
        measurement_noise: Standard deviation of Gaussian measurement noise
        floor_shape: Grid dimensions as (rows, cols)
        walls: Set of wall positions

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> state = np.array([3.0, 5.0, 2.0, 4.0, 0.0])  # Robot at (3,5), opponent at (2,4)
        >>> obs_model = LaserTagObservation(
        ...     next_state=state,
        ...     action=0,
        ...     measurement_noise=1.0,
        ...     floor_shape=(7, 11),
        ...     walls=set()
        ... )
        >>> observations = obs_model.sample(n_samples=3)  # doctest: +SKIP
        >>> probabilities = obs_model.probability(observations)  # doctest: +SKIP
    """

    def __init__(
        self,
        next_state: np.ndarray,
        action: int,
        measurement_noise: float = 1.0,
        floor_shape: Tuple[int, int] = (7, 11),
        walls: Optional[Set[Tuple[int, int]]] = None,
    ):
        """Initialize the observation model.

        Args:
            next_state: State after taking the action as numpy array with shape (5,)
            action: Action that was executed
            measurement_noise: Standard deviation of Gaussian measurement noise
            floor_shape: Grid dimensions as (rows, cols)
            walls: Set of wall positions as (row, col) tuples
        """
        super().__init__(next_state, action)
        self.measurement_noise = measurement_noise
        self.floor_shape = floor_shape
        self.walls = walls if walls is not None else set()

        # 8-directional laser measurements: N, NE, E, SE, S, SW, W, NW
        self._laser_directions = [
            (-1, 0),  # North
            (-1, 1),  # Northeast
            (0, 1),  # East
            (1, 1),  # Southeast
            (1, 0),  # South
            (1, -1),  # Southwest
            (0, -1),  # West
            (-1, -1),  # Northwest
        ]

    def _get_laser_measurement(
        self, robot_pos: Tuple[int, int], direction: Tuple[int, int]
    ) -> float:
        """Get laser range measurement in a specific direction.

        Args:
            robot_pos: Robot's position as (row, col)
            direction: Direction vector as (row_delta, col_delta)

        Returns:
            Number of clear cells in that direction before hitting obstacle/boundary
        """
        row, col = robot_pos
        dr, dc = direction
        distance = 0.0

        # Cast ray in direction until hitting obstacle or boundary
        while True:
            row += dr
            col += dc
            distance += 1.0

            # Check if hit boundary
            if row < 0 or row >= self.floor_shape[0] or col < 0 or col >= self.floor_shape[1]:
                break

            # Check if hit wall
            if (row, col) in self.walls or (row, col) == (
                int(self.next_state[2]),
                int(self.next_state[3]),
            ):
                break

        return distance - 1.0  # Don't count the wall/boundary cell

    def sample(self, n_samples: int = 1) -> List[Tuple[float, ...]]:
        """Sample observations from the observation model.

        Returns:
            List of 8-tuple observations representing laser measurements in 8 directions
        """
        samples: List[Tuple[float, ...]] = []

        if bool(self.next_state[4]):
            # Terminal state - return special terminal observation
            for _ in range(n_samples):
                samples.append((-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
        else:
            # Get true laser measurements from robot position
            robot_pos = (int(self.next_state[0]), int(self.next_state[1]))
            true_measurements = []

            for direction in self._laser_directions:
                measurement = self._get_laser_measurement(robot_pos, direction)
                true_measurements.append(measurement)

            # Add Gaussian noise to each measurement
            for _ in range(n_samples):
                noisy_measurements = []
                for true_measure in true_measurements:
                    noise = np.random.normal(0, self.measurement_noise)
                    noisy_measure = max(0.0, true_measure + noise)  # Clamp to non-negative
                    noisy_measurements.append(noisy_measure)

                samples.append(tuple(noisy_measurements))

        return samples

    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate observation probabilities for given values."""
        result = np.zeros(len(values))

        if bool(self.next_state[4]):
            # Terminal state case
            terminal_obs = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
            for i, obs in enumerate(values):
                if np.array_equal(obs, terminal_obs):
                    result[i] = 1.0
        else:
            # Get true laser measurements
            robot_pos = (int(self.next_state[0]), int(self.next_state[1]))
            true_measurements = []

            for direction in self._laser_directions:
                measurement = self._get_laser_measurement(robot_pos, direction)
                true_measurements.append(measurement)

            # Calculate Gaussian probability density for each observation
            variance = self.measurement_noise**2

            for i, obs in enumerate(values):
                if isinstance(obs, (tuple, list, np.ndarray)) and len(obs) == 8:
                    # Product of independent Gaussian PDFs for each direction
                    prob = 1.0
                    for j, (true_measure, observed_measure) in enumerate(
                        zip(true_measurements, obs)
                    ):
                        if observed_measure >= 0:  # Valid measurement
                            diff = observed_measure - true_measure
                            prob *= np.exp(-0.5 * diff**2 / variance) / np.sqrt(
                                2 * np.pi * variance
                            )
                        else:
                            prob = 0.0  # Invalid negative measurement
                            break
                    result[i] = prob

        return result


class LaserTagPOMDP(DiscreteActionsEnvironment):
    """LaserTag POMDP environment implementation.

    This is a pursuit-evasion problem where a robot must navigate a grid to tag
    an opponent. The robot receives noisy observations of the opponent's position
    and must decide when and where to attempt tagging.

    Problem Structure:
    - States: numpy array [robot_row, robot_col, opp_row, opp_col, terminal]
    - Actions: North(0), South(1), East(2), West(3), Tag(4)
    - Observations: 8-directional laser measurements (N,NE,E,SE,S,SW,W,NW)
    - Rewards: Tag success(+10), Tag failure(-10), Movement(-1)

    Attributes:
        floor_shape: Grid dimensions as (rows, cols)
        walls: Set of wall positions as (row, col) tuples
        tag_reward: Reward for successful tagging
        tag_penalty: Penalty for unsuccessful tagging
        step_cost: Cost per movement action
        measurement_noise: Standard deviation of observation noise

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> env = LaserTagPOMDP(discount_factor=0.95)
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
        name: str = "LaserTagPOMDP",
        floor_shape: Tuple[int, int] = (11, 7),
        walls: Optional[Set[Tuple[int, int]]] = {
            (1, 2),
            (3, 0),
            (3, 4),
            (5, 0),
            (6, 4),
            (9, 1),
            (9, 4),
            (10, 6),
        },
        tag_reward: float = 10.0,
        tag_penalty: float = 10.0,
        step_cost: float = 1.0,
        measurement_noise: float = 1.0,
        dangerous_areas: Optional[Set[Tuple[int, int]]] = {(5, 3), (7, 1), (2, 5)},
        dangerous_area_radius: float = 1.0,
        dangerous_area_penalty: float = 5.0,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
        initial_state: Optional[np.ndarray] = None,
        transition_error_prob: float = 0.0,
    ):
        """Initialize the LaserTag POMDP environment.

        Args:
            discount_factor: Discount factor for future rewards (0 < discount_factor <= 1)
            name: Name identifier for this environment instance
            floor_shape: Grid dimensions as (rows, cols). Defaults to (11, 7).
            walls: Set of wall positions as (row, col) tuples. Each tuple represents
                the (row, col) coordinates of a wall on the grid. Defaults to empty set.
            tag_reward: Reward for successful tagging. Defaults to 10.0.
            tag_penalty: Penalty for unsuccessful tagging. Defaults to 10.0.
            step_cost: Cost per movement action. Defaults to 1.0.
            measurement_noise: Standard deviation of observation noise. Defaults to 1.0.
            dangerous_areas: List of dangerous area center positions as (row, col) tuples. Defaults to None.
            dangerous_area_radius: Radius around dangerous area centers. Defaults to 1.0.
            dangerous_area_penalty: Penalty magnitude applied randomly when in dangerous areas. Defaults to 2.0.
            output_dir: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
            initial_state: Optional initial state as numpy array with shape (5,). If provided,
                the initial state distribution will return this state with probability 1.0.
                If None, returns uniform distribution over all valid initial states. Defaults to None.
            transition_error_prob: Probability that the robot executes a random movement action
                instead of the intended one. Only applies to movement actions (0-3), not Tag (4).
                With probability (1-p), the intended action is executed. With probability p, a random
                action is selected uniformly from {0,1,2,3} excluding the intended action.
                Defaults to 0.0 (deterministic transitions).

        Raises:
            ValueError: If discount_factor is not in valid range [0, 1] or if transition_error_prob
                is not in valid range [0, 1]
        """
        if not (0.0 <= discount_factor <= 1.0):
            raise ValueError("discount_factor must be between 0 and 1 (inclusive)")
        if not (0.0 <= transition_error_prob <= 1.0):
            raise ValueError("transition_error_prob must be between 0 and 1 (inclusive)")

        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # 5 discrete actions
            observation_space=SpaceType.CONTINUOUS,  # Continuous 8-dimensional laser measurements with noise
        )

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=(-tag_penalty, tag_reward),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.floor_shape: Tuple[int, int] = floor_shape
        self.walls: Set[Tuple[int, int]] = walls if walls is not None else set()
        self.tag_reward = tag_reward
        self.tag_penalty = tag_penalty
        self.step_cost = step_cost
        self.measurement_noise = measurement_noise
        self.dangerous_areas: List[Tuple[int, int]] = (
            list(dangerous_areas) if dangerous_areas is not None else []
        )
        self.dangerous_area_radius = dangerous_area_radius
        self.dangerous_area_penalty = dangerous_area_penalty
        self.initial_state = initial_state
        self.transition_error_prob = transition_error_prob

        # Action definitions
        self.actions = [0, 1, 2, 3, 4]  # North, South, East, West, Tag
        self.action_names = ["North", "South", "East", "West", "Tag"]
        self._action_directions: Dict[int, Tuple[int, int]] = {
            0: (-1, 0),  # North (up)
            1: (1, 0),  # South (down)
            2: (0, 1),  # East (right)
            3: (0, -1),  # West (left)
            4: (0, 0),  # Tag (no movement)
        }

    def state_transition_model(self, state: np.ndarray, action: int) -> StateTransitionModel:
        """Get the state transition model for a given state-action pair."""
        return LaserTagStateTransition(
            state=state,
            action=action,
            action_directions=self._action_directions,
            floor_shape=self.floor_shape,
            walls=self.walls,
            transition_error_prob=self.transition_error_prob,
        )

    def observation_model(self, next_state: np.ndarray, action: int) -> ObservationModel:
        """Get the observation model for a given next state and action."""
        return LaserTagObservation(
            next_state=next_state,
            action=action,
            measurement_noise=self.measurement_noise,
            floor_shape=self.floor_shape,
            walls=self.walls,
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
            distance = np.sqrt((pos_row - danger_row) ** 2 + (pos_col - danger_col) ** 2)
            if distance <= self.dangerous_area_radius:
                return True

        return False

    def reward(self, state: np.ndarray, action: int) -> float:
        """Calculate the immediate reward for a state-action pair."""
        if bool(state[4]):
            return 0.0  # No reward in terminal state

        base_reward = 0.0
        robot_pos = (int(state[0]), int(state[1]))
        opponent_pos = (int(state[2]), int(state[3]))

        if action == 4:  # Tag action
            if robot_pos == opponent_pos:
                base_reward = self.tag_reward  # Successful tag
            else:
                base_reward = -self.tag_penalty  # Failed tag attempt
        else:
            base_reward = -self.step_cost  # Movement cost

        intended_pos = (robot_pos[0], robot_pos[1])
        # Check for wall collision and apply dangerous area penalty
        if action in [0, 1, 2, 3]:  # Movement actions
            # Calculate intended position based on action
            dr, dc = self._action_directions[action]
            intended_pos = (robot_pos[0] + dr, robot_pos[1] + dc)

        if intended_pos in self.walls or self._is_in_dangerous_area(intended_pos):
            # Apply dangerous area penalty for wall collision and danerous area
            base_reward -= self.dangerous_area_penalty

        return base_reward

    def is_terminal(self, state: np.ndarray) -> bool:
        """Check if a state is terminal."""
        return bool(state[4])

    def initial_state_dist(self) -> Distribution:
        """Get the initial state distribution."""
        # If initial_state is provided, return distribution with that state at probability 1
        if self.initial_state is not None:
            return DiscreteDistribution(values=[self.initial_state], probs=np.array([1.0]))

        # Generate all valid robot and opponent positions
        valid_positions = []
        for row in range(self.floor_shape[0]):
            for col in range(self.floor_shape[1]):
                if (row, col) not in self.walls:
                    valid_positions.append((row, col))

        # Create all possible initial states (robot and opponent at different positions)
        initial_states = []
        for robot_pos in valid_positions:
            for opp_pos in valid_positions:
                if robot_pos != opp_pos:  # Robot and opponent start at different positions
                    initial_states.append(
                        np.array(
                            [
                                float(robot_pos[0]),
                                float(robot_pos[1]),
                                float(opp_pos[0]),
                                float(opp_pos[1]),
                                0.0,
                            ]
                        )
                    )

        # Uniform distribution over all initial states
        num_states = len(initial_states)
        probs = np.ones(num_states) / num_states

        return DiscreteDistribution(values=initial_states, probs=probs)

    def initial_observation_dist(self) -> Distribution:
        """Get the initial observation distribution."""
        # Return distribution over possible initial laser observations
        # For simplicity, return a uniform distribution over typical laser readings
        # This would normally be computed from the initial state distribution
        typical_readings = (
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
        )  # Mid-range readings
        return DiscreteDistribution(values=[typical_readings], probs=np.array([1.0]))

    def get_actions(self) -> List[int]:
        """Get all possible actions in the discrete action space."""
        return self.actions

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Check if two observations are equal.

        Observations are 8-dimensional laser measurements or terminal observations.
        """
        return np.array_equal(observation1, observation2)

    def _count_episode_metrics(
        self, history: History, action_dirs: Dict[int, Tuple[int, int]]
    ) -> Tuple[int, int, int, int]:
        episode_failed_tags = 0
        episode_obstacle_collisions = 0
        episode_dangerous_area_steps = 0

        for step in history.history:
            if step.action == 4 and step.reward is not None and step.reward < 0:
                episode_failed_tags += 1

            if isinstance(step.state, np.ndarray) and len(step.state) == 5:
                robot_pos = (int(step.state[0]), int(step.state[1]))
                if self._is_in_dangerous_area(robot_pos):
                    episode_dangerous_area_steps += 1

            if step.action in [0, 1, 2, 3]:
                if (
                    isinstance(step.state, np.ndarray)
                    and len(step.state) == 5
                    and hasattr(step, "next_state")
                    and isinstance(step.next_state, np.ndarray)
                    and len(step.next_state) == 5
                ):
                    if step.action in action_dirs:
                        dr, dc = action_dirs[step.action]
                        robot_pos = (int(step.state[0]), int(step.state[1]))
                        next_robot_pos = (int(step.next_state[0]), int(step.next_state[1]))
                        intended_pos = (robot_pos[0] + dr, robot_pos[1] + dc)

                        if intended_pos in self.walls and next_robot_pos == robot_pos:
                            episode_obstacle_collisions += 1

        return (
            episode_failed_tags,
            episode_obstacle_collisions,
            episode_dangerous_area_steps,
            episode_obstacle_collisions + episode_dangerous_area_steps,
        )

    def _collect_episode_data(self, histories: List[History]) -> Tuple:
        episode_lengths = []
        success_indicators = []
        goal_reached_indicators = []
        failed_tags_per_episode = []
        obstacle_collisions_per_episode = []
        dangerous_area_steps_per_episode = []
        all_dangerous_encounters_per_episode = []

        action_dirs = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}

        for history in histories:
            episode_length = len(history.history)
            episode_lengths.append(episode_length)

            episode_successful = (
                history.history
                and history.history[-1].reward is not None
                and history.history[-1].reward > 0
            )
            success_indicators.append(1 if episode_successful else 0)

            # Check if goal was reached (opponent was tagged) by checking if any step reached terminal state
            goal_reached = False
            for step in history.history:
                if isinstance(step.state, np.ndarray) and len(step.state) == 5:
                    if bool(step.state[4]):  # Terminal flag is set when tag is successful
                        goal_reached = True
                        break
            goal_reached_indicators.append(1 if goal_reached else 0)

            (
                episode_failed_tags,
                episode_obstacle_collisions,
                episode_dangerous_area_steps,
                episode_all_dangerous_encounters,
            ) = self._count_episode_metrics(history, action_dirs)

            failed_tags_per_episode.append(episode_failed_tags)
            obstacle_collisions_per_episode.append(episode_obstacle_collisions)
            dangerous_area_steps_per_episode.append(episode_dangerous_area_steps)
            all_dangerous_encounters_per_episode.append(episode_all_dangerous_encounters)

        return (
            episode_lengths,
            success_indicators,
            goal_reached_indicators,
            failed_tags_per_episode,
            obstacle_collisions_per_episode,
            dangerous_area_steps_per_episode,
            all_dangerous_encounters_per_episode,
        )

    def _calculate_confidence_intervals(
        self,
        total_episodes: int,
        success_indicators: List[int],
        goal_reached_indicators: List[int],
        episode_lengths: List[int],
        failed_tags_per_episode: List[int],
        obstacle_collisions_per_episode: List[int],
        dangerous_area_steps_per_episode: List[int],
        all_dangerous_encounters_per_episode: List[int],
    ) -> Tuple:
        if total_episodes >= 2:
            success_ci = confidence_interval(data=success_indicators, confidence=0.95)
            goal_reached_ci = confidence_interval(data=goal_reached_indicators, confidence=0.95)
            episode_length_ci = confidence_interval(data=episode_lengths, confidence=0.95)
            failed_tags_ci = confidence_interval(data=failed_tags_per_episode, confidence=0.95)
            obstacle_collisions_ci = confidence_interval(
                data=obstacle_collisions_per_episode, confidence=0.95
            )
            dangerous_area_steps_ci = confidence_interval(
                data=dangerous_area_steps_per_episode, confidence=0.95
            )
            all_dangerous_encounters_ci = confidence_interval(
                data=all_dangerous_encounters_per_episode, confidence=0.95
            )
        else:
            success_ci = (-np.inf, np.inf)
            goal_reached_ci = (-np.inf, np.inf)
            episode_length_ci = (-np.inf, np.inf)
            failed_tags_ci = (-np.inf, np.inf)
            obstacle_collisions_ci = (-np.inf, np.inf)
            dangerous_area_steps_ci = (-np.inf, np.inf)
            all_dangerous_encounters_ci = (-np.inf, np.inf)

        return (
            success_ci,
            goal_reached_ci,
            episode_length_ci,
            failed_tags_ci,
            obstacle_collisions_ci,
            dangerous_area_steps_ci,
            all_dangerous_encounters_ci,
        )

    def get_metric_names(self) -> List[str]:
        """Get names of LaserTag POMDP specific metrics.

        Returns:
            List containing metric names: tag_success_rate, average_episode_length,
            average_failed_tag_attempts, average_obstacle_collisions,
            average_dangerous_area_steps, and average_all_dangerous_encounters
        """
        return [metric.value for metric in LaserTagPOMDPMetrics]

    def _build_metric_values(
        self,
        success_rate: float,
        goal_reaching_rate: float,
        avg_episode_length: float,
        avg_failed_tags: float,
        avg_obstacle_collisions: float,
        avg_dangerous_area_steps: float,
        avg_all_dangerous_encounters: float,
        success_ci: Tuple[float, float],
        goal_reached_ci: Tuple[float, float],
        episode_length_ci: Tuple[float, float],
        failed_tags_ci: Tuple[float, float],
        obstacle_collisions_ci: Tuple[float, float],
        dangerous_area_steps_ci: Tuple[float, float],
        all_dangerous_encounters_ci: Tuple[float, float],
    ) -> List[MetricValue]:
        return [
            MetricValue(
                name=LaserTagPOMDPMetrics.TAG_SUCCESS_RATE.value,
                value=success_rate,
                lower_confidence_bound=success_ci[0],
                upper_confidence_bound=success_ci[1],
            ),
            MetricValue(
                name=LaserTagPOMDPMetrics.GOAL_REACHING_RATE.value,
                value=goal_reaching_rate,
                lower_confidence_bound=goal_reached_ci[0],
                upper_confidence_bound=goal_reached_ci[1],
            ),
            MetricValue(
                name=LaserTagPOMDPMetrics.AVERAGE_EPISODE_LENGTH.value,
                value=avg_episode_length,
                lower_confidence_bound=episode_length_ci[0],
                upper_confidence_bound=episode_length_ci[1],
            ),
            MetricValue(
                name=LaserTagPOMDPMetrics.AVERAGE_FAILED_TAG_ATTEMPTS.value,
                value=avg_failed_tags,
                lower_confidence_bound=failed_tags_ci[0],
                upper_confidence_bound=failed_tags_ci[1],
            ),
            MetricValue(
                name=LaserTagPOMDPMetrics.AVERAGE_OBSTACLE_COLLISIONS.value,
                value=avg_obstacle_collisions,
                lower_confidence_bound=obstacle_collisions_ci[0],
                upper_confidence_bound=obstacle_collisions_ci[1],
            ),
            MetricValue(
                name=LaserTagPOMDPMetrics.AVERAGE_DANGEROUS_AREA_STEPS.value,
                value=avg_dangerous_area_steps,
                lower_confidence_bound=dangerous_area_steps_ci[0],
                upper_confidence_bound=dangerous_area_steps_ci[1],
            ),
            MetricValue(
                name=LaserTagPOMDPMetrics.AVERAGE_ALL_DANGEROUS_ENCOUNTERS.value,
                value=avg_all_dangerous_encounters,
                lower_confidence_bound=all_dangerous_encounters_ci[0],
                upper_confidence_bound=all_dangerous_encounters_ci[1],
            ),
        ]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute LaserTag POMDP specific metrics from simulation histories."""
        total_episodes = len(histories)
        if total_episodes == 0:
            return []

        (
            episode_lengths,
            success_indicators,
            goal_reached_indicators,
            failed_tags_per_episode,
            obstacle_collisions_per_episode,
            dangerous_area_steps_per_episode,
            all_dangerous_encounters_per_episode,
        ) = self._collect_episode_data(histories)

        successful_tags = sum(success_indicators)
        success_rate = successful_tags / total_episodes
        goals_reached = sum(goal_reached_indicators)
        goal_reaching_rate = goals_reached / total_episodes
        avg_episode_length = float(np.mean(episode_lengths))
        avg_failed_tags = float(np.mean(failed_tags_per_episode))
        avg_obstacle_collisions = float(np.mean(obstacle_collisions_per_episode))
        avg_dangerous_area_steps = float(np.mean(dangerous_area_steps_per_episode))
        avg_all_dangerous_encounters = float(np.mean(all_dangerous_encounters_per_episode))

        (
            success_ci,
            goal_reached_ci,
            episode_length_ci,
            failed_tags_ci,
            obstacle_collisions_ci,
            dangerous_area_steps_ci,
            all_dangerous_encounters_ci,
        ) = self._calculate_confidence_intervals(
            total_episodes,
            success_indicators,
            goal_reached_indicators,
            episode_lengths,
            failed_tags_per_episode,
            obstacle_collisions_per_episode,
            dangerous_area_steps_per_episode,
            all_dangerous_encounters_per_episode,
        )

        return self._build_metric_values(
            success_rate,
            goal_reaching_rate,
            avg_episode_length,
            avg_failed_tags,
            avg_obstacle_collisions,
            avg_dangerous_area_steps,
            avg_all_dangerous_encounters,
            success_ci,
            goal_reached_ci,
            episode_length_ci,
            failed_tags_ci,
            obstacle_collisions_ci,
            dangerous_area_steps_ci,
            all_dangerous_encounters_ci,
        )

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache visualization of the LaserTag episode as an animated GIF.

        Creates an animated visualization showing:
        - Robot movement (red circle with path trail)
        - Opponent movement (blue circle)
        - Walls (black squares)
        - Dangerous areas (red circles)
        - Action arrows showing robot's intended movement
        - Laser measurements (green rays from robot position)
        - Belief particles (if available) showing robot's belief about opponent location
        - Grid boundaries and coordinate system

        Args:
            history: The history of states, actions, and observations from an episode
            cache_path: Path where to save the visualization GIF

        Raises:
            ValueError: If history is empty or contains invalid data
            TypeError: If cache_path is not a Path object or doesn't end with .gif
        """
        # Lazy import to avoid circular dependency
        from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_visualizer import (
            LaserTagVisualizer,
        )

        visualizer = LaserTagVisualizer(
            floor_shape=self.floor_shape,
            walls=self.walls,
            dangerous_areas=self.dangerous_areas,
            dangerous_area_radius=self.dangerous_area_radius,
        )
        visualizer.create_visualization(history, cache_path)
        self.logger.info("Saved LaserTag visualization to %s", cache_path)
