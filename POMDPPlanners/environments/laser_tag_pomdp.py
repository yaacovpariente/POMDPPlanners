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

Based on the LaserTag.jl implementation from: https://github.com/JuliaPOMDP/LaserTag.jl

Classes:
    LaserTagState: State representation with robot and opponent positions
    LaserTagStateTransition: State transition model for robot and opponent movement
    LaserTagObservation: Observation model with noisy opponent position measurements
    LaserTagPOMDP: Main environment class implementing the LaserTag problem
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
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
from POMDPPlanners.utils.statistics import confidence_interval


@dataclass
class LaserTagState:
    """State representation for LaserTag POMDP.

    Attributes:
        robot: Robot's position as (row, col) tuple
        opponent: Opponent's position as (row, col) tuple
        terminal: Whether the episode has terminated

    Example:
        >>> state = LaserTagState(robot=(0, 0), opponent=(6, 10), terminal=False)
        >>> state.robot
        (0, 0)
        >>> state.opponent
        (6, 10)
        >>> state.terminal
        False
    """

    robot: Tuple[int, int]
    opponent: Tuple[int, int]
    terminal: bool = False

    def __eq__(self, other):
        if not isinstance(other, LaserTagState):
            return False
        return (
            self.robot == other.robot
            and self.opponent == other.opponent
            and self.terminal == other.terminal
        )

    def __hash__(self):
        return hash((self.robot, self.opponent, self.terminal))


class LaserTagStateTransition(StateTransitionModel):
    """State transition model for LaserTag POMDP.

    Handles robot movement (deterministic based on action) and opponent movement
    (probabilistic, with tendency to move toward robot's position).

    Attributes:
        state: Current LaserTagState
        action: Action to be executed (0=North, 1=South, 2=East, 3=West, 4=Tag)
        floor_shape: Tuple of (rows, cols) for grid dimensions
        walls: Set of wall positions

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> state = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        >>> transition = LaserTagStateTransition(
        ...     state=state,
        ...     action=0,  # North
        ...     floor_shape=(7, 11),
        ...     walls=set()
        ... )
        >>> next_states = transition.sample(n_samples=5)  # doctest: +SKIP
        >>> probabilities = transition.probability(next_states)  # doctest: +SKIP
    """

    def __init__(
        self,
        state: LaserTagState,
        action: int,
        floor_shape: Tuple[int, int],
        walls: Set[Tuple[int, int]],
    ):
        """Initialize the state transition model.

        Args:
            state: Current LaserTagState
            action: Action to execute (0=North, 1=South, 2=East, 3=West, 4=Tag)
            floor_shape: Grid dimensions as (rows, cols)
            walls: Set of wall positions as (row, col) tuples
        """
        super().__init__(state, action)
        self.floor_shape: Tuple[int, int] = floor_shape
        self.walls: Set[Tuple[int, int]] = walls
        self._action_directions: Dict[int, Tuple[int, int]] = {
            0: (-1, 0),  # North (up)
            1: (1, 0),  # South (down)
            2: (0, 1),  # East (right)
            3: (0, -1),  # West (left)
            4: (0, 0),  # Tag (no movement)
        }

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a wall."""
        row, col = pos
        return (
            0 <= row < self.floor_shape[0]
            and 0 <= col < self.floor_shape[1]
            and pos not in self.walls
        )

    def _get_robot_next_position(self) -> Tuple[int, int]:
        """Get robot's next position based on action."""
        if self.action == 4:  # Tag action
            return self.state.robot

        dr, dc = self._action_directions[self.action]
        new_pos = (self.state.robot[0] + dr, self.state.robot[1] + dc)

        # If new position is invalid, stay at current position
        if self._is_valid_position(new_pos):
            return new_pos
        else:
            return self.state.robot

    def _get_opponent_move_probabilities(
        self, robot_pos: Tuple[int, int]
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Get opponent's movement probabilities based on robot position.

        Uses Julia LaserTag.jl movement model:
        - 0.4 probability to move in x-direction (toward/away from robot)
        - 0.4 probability to move in y-direction (toward/away from robot)
        - 0.2 probability to stay in place
        """
        current_opp = self.state.opponent
        robot_row, robot_col = robot_pos
        opp_row, opp_col = current_opp

        # Calculate movement preferences
        x_moves = []  # Column movements
        y_moves = []  # Row movements

        # X-direction (column) movements
        if robot_col > opp_col:  # Robot is east, opponent should move east
            east_pos = (opp_row, opp_col + 1)
            if self._is_valid_position(east_pos):
                x_moves.append((east_pos, 0.4))  # Move toward robot
            west_pos = (opp_row, opp_col - 1)
            if self._is_valid_position(west_pos):
                x_moves.append((west_pos, 0.0))  # Move away from robot gets remaining prob
        elif robot_col < opp_col:  # Robot is west, opponent should move west
            west_pos = (opp_row, opp_col - 1)
            if self._is_valid_position(west_pos):
                x_moves.append((west_pos, 0.4))  # Move toward robot
            east_pos = (opp_row, opp_col + 1)
            if self._is_valid_position(east_pos):
                x_moves.append((east_pos, 0.0))  # Move away from robot gets remaining prob
        else:  # Same column
            east_pos = (opp_row, opp_col + 1)
            if self._is_valid_position(east_pos):
                x_moves.append((east_pos, 0.0))
            west_pos = (opp_row, opp_col - 1)
            if self._is_valid_position(west_pos):
                x_moves.append((west_pos, 0.0))

        # Y-direction (row) movements
        if robot_row > opp_row:  # Robot is south, opponent should move south
            south_pos = (opp_row + 1, opp_col)
            if self._is_valid_position(south_pos):
                y_moves.append((south_pos, 0.4))  # Move toward robot
            north_pos = (opp_row - 1, opp_col)
            if self._is_valid_position(north_pos):
                y_moves.append((north_pos, 0.0))  # Move away from robot gets remaining prob
        elif robot_row < opp_row:  # Robot is north, opponent should move north
            north_pos = (opp_row - 1, opp_col)
            if self._is_valid_position(north_pos):
                y_moves.append((north_pos, 0.4))  # Move toward robot
            south_pos = (opp_row + 1, opp_col)
            if self._is_valid_position(south_pos):
                y_moves.append((south_pos, 0.0))  # Move away from robot gets remaining prob
        else:  # Same row
            north_pos = (opp_row - 1, opp_col)
            if self._is_valid_position(north_pos):
                y_moves.append((north_pos, 0.0))
            south_pos = (opp_row + 1, opp_col)
            if self._is_valid_position(south_pos):
                y_moves.append((south_pos, 0.0))

        # Combine moves and normalize probabilities
        move_probs = []
        total_x_prob = 0.4 if any(prob > 0 for _, prob in x_moves) else 0.0
        total_y_prob = 0.4 if any(prob > 0 for _, prob in y_moves) else 0.0
        stay_prob = 0.2

        # Add x-direction moves
        for pos, prob in x_moves:
            if prob > 0:  # Toward robot
                move_probs.append((pos, prob))
            elif total_x_prob > 0:  # Away from robot gets remaining x-prob
                remaining_x_prob = 0.0  # No probability for away moves in Julia model
                move_probs.append((pos, remaining_x_prob))

        # Add y-direction moves
        for pos, prob in y_moves:
            if prob > 0:  # Toward robot
                move_probs.append((pos, prob))
            elif total_y_prob > 0:  # Away from robot gets remaining y-prob
                remaining_y_prob = 0.0  # No probability for away moves in Julia model
                move_probs.append((pos, remaining_y_prob))

        # Add stay probability
        move_probs.append((current_opp, stay_prob))

        # Normalize probabilities to handle blocked moves
        actual_total = sum(prob for _, prob in move_probs if prob > 0)
        if actual_total < 1.0:
            # Redistribute remaining probability to staying
            stay_index = len(move_probs) - 1
            move_probs[stay_index] = (current_opp, stay_prob + (1.0 - actual_total))

        # Filter out zero probability moves
        move_probs = [(pos, prob) for pos, prob in move_probs if prob > 0]

        return move_probs

    def sample(self, n_samples: int = 1) -> List[LaserTagState]:
        """Sample next states from the transition model."""
        samples = []
        robot_next = self._get_robot_next_position()

        # Check if tagging occurred
        if self.action == 4 and self.state.robot == self.state.opponent:
            # Successful tag - terminal state
            for _ in range(n_samples):
                samples.append(LaserTagState(robot_next, self.state.opponent, terminal=True))
        else:
            # Regular transition
            opp_moves = self._get_opponent_move_probabilities(robot_next)
            positions, probabilities = zip(*opp_moves)

            for _ in range(n_samples):
                opp_next = np.random.choice(len(positions), p=probabilities)
                samples.append(LaserTagState(robot_next, positions[opp_next], terminal=False))

        return samples

    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate transition probabilities for given next states."""
        result = np.zeros(len(values))
        robot_next = self._get_robot_next_position()

        # Check if tagging occurred
        if self.action == 4 and self.state.robot == self.state.opponent:
            # Successful tag case
            for i, next_state in enumerate(values):
                if (
                    isinstance(next_state, LaserTagState)
                    and next_state.robot == robot_next
                    and next_state.opponent == self.state.opponent
                    and next_state.terminal
                ):
                    result[i] = 1.0
        else:
            # Regular transition case
            opp_moves = self._get_opponent_move_probabilities(robot_next)

            for i, next_state in enumerate(values):
                if (
                    isinstance(next_state, LaserTagState)
                    and next_state.robot == robot_next
                    and not next_state.terminal
                ):
                    # Find probability for this opponent position
                    for opp_pos, prob in opp_moves:
                        if next_state.opponent == opp_pos:
                            result[i] = prob
                            break

        return result


class LaserTagObservation(ObservationModel):
    """Observation model for LaserTag POMDP.

    Provides 8-directional laser range measurements from the robot's position.
    Each measurement represents the number of clear cells in that direction
    before hitting a wall or boundary, with Gaussian noise.

    Attributes:
        next_state: The state after action execution
        action: The action that was taken
        measurement_noise: Standard deviation of Gaussian measurement noise
        floor_shape: Grid dimensions as (rows, cols)
        walls: Set of wall positions

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> state = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
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
        next_state: LaserTagState,
        action: int,
        measurement_noise: float = 1.0,
        floor_shape: Tuple[int, int] = (7, 11),
        walls: Optional[Set[Tuple[int, int]]] = None,
    ):
        """Initialize the observation model.

        Args:
            next_state: State after taking the action
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
                self.next_state.opponent[0],
                self.next_state.opponent[1],
            ):
                break

        return distance - 1.0  # Don't count the wall/boundary cell

    def sample(self, n_samples: int = 1) -> List[Tuple[float, ...]]:
        """Sample observations from the observation model.

        Returns:
            List of 8-tuple observations representing laser measurements in 8 directions
        """
        samples = []

        if self.next_state.terminal:
            # Terminal state - return special terminal observation
            for _ in range(n_samples):
                samples.append((-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
        else:
            # Get true laser measurements from robot position
            robot_pos = self.next_state.robot
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

        if self.next_state.terminal:
            # Terminal state case
            terminal_obs = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
            for i, obs in enumerate(values):
                if obs == terminal_obs:
                    result[i] = 1.0
        else:
            # Get true laser measurements
            robot_pos = self.next_state.robot
            true_measurements = []

            for direction in self._laser_directions:
                measurement = self._get_laser_measurement(robot_pos, direction)
                true_measurements.append(measurement)

            # Calculate Gaussian probability density for each observation
            variance = self.measurement_noise**2

            for i, obs in enumerate(values):
                if isinstance(obs, (tuple, list)) and len(obs) == 8:
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
    - States: Robot position, opponent position, terminal flag
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
        >>> # Create LaserTag environment
        >>> env = LaserTagPOMDP(
        ...     discount_factor=0.95,
        ...     floor_shape=(7, 11),
        ...     tag_reward=10.0,
        ...     step_cost=1.0
        ... )
        >>>
        >>> # Sample initial state and get actions
        >>> initial_state = env.initial_state_dist().sample()[0]  # doctest: +SKIP
        >>> actions = env.get_actions()
        >>> len(actions) == 5
        True
        >>>
        >>> # Create test state for reward calculation
        >>> test_state = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
        >>> reward = env.reward(test_state, action=0)  # Move north  # doctest: +SKIP
        >>>
        >>> # Check terminal condition
        >>> is_done = env.is_terminal(test_state)
        >>> is_done
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

        Raises:
            ValueError: If discount_factor is not in valid range [0, 1]
        """
        if not (0.0 <= discount_factor <= 1.0):
            raise ValueError("discount_factor must be between 0 and 1 (inclusive)")

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

        # Action definitions
        self.actions = [0, 1, 2, 3, 4]  # North, South, East, West, Tag
        self.action_names = ["North", "South", "East", "West", "Tag"]

    def state_transition_model(self, state: LaserTagState, action: int) -> StateTransitionModel:
        """Get the state transition model for a given state-action pair."""
        return LaserTagStateTransition(
            state=state, action=action, floor_shape=self.floor_shape, walls=self.walls
        )

    def observation_model(self, next_state: LaserTagState, action: int) -> ObservationModel:
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

    def reward(self, state: LaserTagState, action: int) -> float:
        """Calculate the immediate reward for a state-action pair."""
        if state.terminal:
            return 0.0  # No reward in terminal state

        base_reward = 0.0

        if action == 4:  # Tag action
            if state.robot == state.opponent:
                base_reward = self.tag_reward  # Successful tag
            else:
                base_reward = -self.tag_penalty  # Failed tag attempt
        else:
            base_reward = -self.step_cost  # Movement cost

        # Check for wall collision and apply dangerous area penalty
        if action in [0, 1, 2, 3]:  # Movement actions
            # Calculate intended position based on action
            action_directions = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}
            dr, dc = action_directions[action]
            intended_pos = (state.robot[0] + dr, state.robot[1] + dc)

            # Check if intended position is a wall (collision)
            if intended_pos in self.walls:
                # Apply dangerous area penalty for wall collision
                base_reward -= self.dangerous_area_penalty

        # Add dangerous area penalty/bonus with 50% probability
        if self._is_in_dangerous_area(state.robot):
            # Random penalty or bonus with equal probability
            danger_modifier = (
                self.dangerous_area_penalty
                if np.random.random() < 0.5
                else -self.dangerous_area_penalty
            )
            base_reward += danger_modifier

        return base_reward

    def is_terminal(self, state: LaserTagState) -> bool:
        """Check if a state is terminal."""
        return state.terminal

    def initial_state_dist(self) -> Distribution:
        """Get the initial state distribution."""
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
                    initial_states.append(LaserTagState(robot_pos, opp_pos, terminal=False))

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
        if isinstance(observation1, (tuple, list)) and isinstance(observation2, (tuple, list)):
            if len(observation1) == len(observation2) == 8:
                # Compare 8-dimensional laser measurements with tolerance
                return all(
                    abs(obs1 - obs2) < 1e-10 for obs1, obs2 in zip(observation1, observation2)
                )
        return observation1 == observation2

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute LaserTag POMDP specific metrics from simulation histories."""
        total_episodes = len(histories)
        if total_episodes == 0:
            # Return empty metrics for no data
            return []

        # Initialize per-episode lists for confidence intervals
        episode_lengths = []
        success_indicators = []
        failed_tags_per_episode = []
        obstacle_collisions_per_episode = []
        dangerous_area_steps_per_episode = []

        # Action direction mappings (used multiple times)
        action_dirs = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}

        # Single loop to collect all metrics
        for history in histories:
            episode_length = len(history.history)
            episode_lengths.append(episode_length)

            # Check if episode ended with successful tag
            episode_successful = (
                history.history
                and history.history[-1].reward is not None
                and history.history[-1].reward > 0
            )
            success_indicators.append(1 if episode_successful else 0)

            # Initialize episode-specific counters
            episode_failed_tags = 0
            episode_obstacle_collisions = 0
            episode_dangerous_area_steps = 0

            # Count per-step metrics
            for step in history.history:
                # Count failed tag attempts
                if (
                    step.action == 4 and step.reward is not None and step.reward < 0
                ):  # Tag action with negative reward
                    episode_failed_tags += 1

                # Count dangerous area steps
                if isinstance(step.state, LaserTagState):
                    if self._is_in_dangerous_area(step.state.robot):
                        episode_dangerous_area_steps += 1

                # Count obstacle collisions
                if step.action in [0, 1, 2, 3]:  # Movement actions
                    if (
                        isinstance(step.state, LaserTagState)
                        and hasattr(step, "next_state")
                        and isinstance(step.next_state, LaserTagState)
                    ):
                        if step.action in action_dirs:
                            dr, dc = action_dirs[step.action]
                            intended_pos = (
                                step.state.robot[0] + dr,
                                step.state.robot[1] + dc,
                            )

                            # Check if intended position was a wall and robot didn't move
                            if (
                                intended_pos in self.walls
                                and step.next_state.robot == step.state.robot
                            ):
                                episode_obstacle_collisions += 1

            # Store episode counts for confidence intervals
            failed_tags_per_episode.append(episode_failed_tags)
            obstacle_collisions_per_episode.append(episode_obstacle_collisions)
            dangerous_area_steps_per_episode.append(episode_dangerous_area_steps)

        # Calculate aggregate statistics
        successful_tags = sum(success_indicators)
        success_rate = successful_tags / total_episodes
        avg_episode_length = np.mean(episode_lengths)
        avg_failed_tags = np.mean(failed_tags_per_episode)
        avg_obstacle_collisions = np.mean(obstacle_collisions_per_episode)
        avg_dangerous_area_steps = np.mean(dangerous_area_steps_per_episode)

        # Calculate confidence intervals (handle single episode case)
        if total_episodes >= 2:
            success_ci = confidence_interval(data=success_indicators, confidence=0.95)
            episode_length_ci = confidence_interval(data=episode_lengths, confidence=0.95)
            failed_tags_ci = confidence_interval(data=failed_tags_per_episode, confidence=0.95)
            obstacle_collisions_ci = confidence_interval(
                data=obstacle_collisions_per_episode, confidence=0.95
            )
            dangerous_area_steps_ci = confidence_interval(
                data=dangerous_area_steps_per_episode, confidence=0.95
            )

        else:
            # For single episode, confidence bounds equal the value (no statistical inference)
            success_ci = (-np.inf, np.inf)
            episode_length_ci = (-np.inf, np.inf)
            failed_tags_ci = (-np.inf, np.inf)
            obstacle_collisions_ci = (-np.inf, np.inf)
            dangerous_area_steps_ci = (-np.inf, np.inf)

        return [
            MetricValue(
                name="tag_success_rate",
                value=success_rate,
                lower_confidence_bound=success_ci[0],
                upper_confidence_bound=success_ci[1],
            ),
            MetricValue(
                name="average_episode_length",
                value=avg_episode_length,
                lower_confidence_bound=episode_length_ci[0],
                upper_confidence_bound=episode_length_ci[1],
            ),
            MetricValue(
                name="average_failed_tag_attempts",
                value=avg_failed_tags,
                lower_confidence_bound=failed_tags_ci[0],
                upper_confidence_bound=failed_tags_ci[1],
            ),
            MetricValue(
                name="average_obstacle_collisions",
                value=avg_obstacle_collisions,
                lower_confidence_bound=obstacle_collisions_ci[0],
                upper_confidence_bound=obstacle_collisions_ci[1],
            ),
            MetricValue(
                name="average_dangerous_area_steps",
                value=avg_dangerous_area_steps,
                lower_confidence_bound=dangerous_area_steps_ci[0],
                upper_confidence_bound=dangerous_area_steps_ci[1],
            ),
        ]

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache visualization of the LaserTag episode as an animated GIF.

        Creates an animated visualization showing:
        - Robot movement (red circle with path trail)
        - Opponent movement (blue circle)
        - Walls (black squares)
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
        if not isinstance(history, List):
            raise TypeError("history must be a List object")
        if not history:
            raise ValueError("Cannot visualize empty history")
        for step in history:
            if not isinstance(step, StepData):
                raise TypeError("history must be a List of StepData objects")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

        # Create directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract robot and opponent paths
        robot_path = []
        opponent_path = []
        actions = []
        beliefs = []

        for step in history:
            if not isinstance(step.state, LaserTagState):
                raise ValueError(f"Expected LaserTagState, got {type(step.state)}")

            robot_path.append(step.state.robot)
            opponent_path.append(step.state.opponent)
            actions.append(step.action)

            # Try to extract belief if available
            if hasattr(step, "belief") and step.belief is not None:
                beliefs.append(step.belief)
            else:
                beliefs.append(None)

        # Set up the figure and axis with extra space for legend
        fig, ax = plt.subplots(figsize=(14, 8))
        rows, cols = self.floor_shape
        ax.set_xlim(-0.5, rows - 0.5)
        ax.set_ylim(-0.5, cols - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Invert y-axis so (0,0) is top-left like matrix indexing

        # Set grid
        ax.set_xticks(range(rows))
        ax.set_yticks(range(cols))
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Row")
        ax.set_ylabel("Column")
        ax.set_title("LaserTag POMDP Episode Visualization")

        # Draw walls as black squares
        wall_patches = []
        for i, wall in enumerate(self.walls):
            row, col = wall
            square = plt.Rectangle(
                (row - 0.4, col - 0.4),
                0.8,
                0.8,
                facecolor="black",
                edgecolor="black",
                alpha=0.7,
                label="Wall" if i == 0 else "",
            )  # Only label first wall
            ax.add_patch(square)
            if i == 0:  # Keep reference for legend
                wall_patches.append(square)

        # Draw dangerous areas as red circles (like in light_dark_pomdp)
        danger_patches = []
        for i, danger_center in enumerate(self.dangerous_areas):
            row, col = danger_center
            circle = plt.Circle(
                (row, col),
                self.dangerous_area_radius,
                facecolor="red",
                edgecolor="none",
                alpha=0.3,
                label="Dangerous Areas" if i == 0 else "",
            )  # Only label first area
            ax.add_patch(circle)
            if i == 0:  # Keep reference for legend
                danger_patches.append(circle)

        # Initialize animated elements
        (robot_agent,) = ax.plot([], [], "ro", markersize=12, label="Robot")
        (opponent_agent,) = ax.plot([], [], "bo", markersize=12, label="Opponent")
        (robot_path_line,) = ax.plot([], [], "r-", alpha=0.5, linewidth=2, label="Robot Path")
        (opponent_path_line,) = ax.plot([], [], "b-", alpha=0.5, linewidth=2, label="Opponent Path")

        # Action arrow
        action_arrow = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )

        # Step counter
        step_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Action text
        action_text = ax.text(
            0.02,
            0.90,
            "",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        # Tag success indicator
        tag_text = ax.text(
            0.02,
            0.02,
            "",
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="gold",
                edgecolor="red",
                linewidth=3,
                alpha=0.9,
            ),
            color="red",
            visible=False,
        )

        # Belief particles (if available) - opponent beliefs
        opponent_belief_scatter = ax.scatter(
            [], [], c="lightblue", alpha=0.6, s=30, label="Opponent Belief Particles"
        )
        # Robot belief particles
        robot_belief_scatter = ax.scatter(
            [], [], c="lightcoral", alpha=0.6, s=30, label="Robot Belief Particles"
        )

        # Laser rays (8 directions from robot)
        laser_lines = []
        laser_directions = [
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ]
        for i in range(8):
            (line,) = ax.plot(
                [],
                [],
                "g-",
                alpha=0.4,
                linewidth=1,
                label="Laser Rays" if i == 0 else "",
            )
            laser_lines.append(line)

        # Legend - position it inside the plot area to avoid truncation
        ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), framealpha=0.9)

        def init():
            robot_agent.set_data([], [])
            opponent_agent.set_data([], [])
            robot_path_line.set_data([], [])
            opponent_path_line.set_data([], [])
            action_arrow.set_position((0, 0))
            action_arrow.xy = (0, 0)
            step_text.set_text("")
            action_text.set_text("")
            tag_text.set_visible(False)
            opponent_belief_scatter.set_offsets(np.empty((0, 2)))
            robot_belief_scatter.set_offsets(np.empty((0, 2)))
            # Clear laser rays
            for line in laser_lines:
                line.set_data([], [])
            return [
                robot_agent,
                opponent_agent,
                robot_path_line,
                opponent_path_line,
                action_arrow,
                step_text,
                action_text,
                tag_text,
                opponent_belief_scatter,
                robot_belief_scatter,
            ] + laser_lines

        def update(frame):
            # Current positions
            robot_pos = robot_path[frame]
            opponent_pos = opponent_path[frame]

            # Update agent positions (transpose: row,col to x,y for plotting)
            robot_agent.set_data([robot_pos[0]], [robot_pos[1]])  # row, col
            opponent_agent.set_data([opponent_pos[0]], [opponent_pos[1]])  # row, col

            # Update path lines up to current frame
            robot_rows = [pos[0] for pos in robot_path[: frame + 1]]
            robot_cols = [pos[1] for pos in robot_path[: frame + 1]]
            opponent_rows = [pos[0] for pos in opponent_path[: frame + 1]]
            opponent_cols = [pos[1] for pos in opponent_path[: frame + 1]]

            robot_path_line.set_data(robot_rows, robot_cols)
            opponent_path_line.set_data(opponent_rows, opponent_cols)

            # Update action arrow
            if frame < len(actions):
                action = actions[frame]
                action_dirs = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1), 4: (0, 0)}
                action_names = {0: "North", 1: "South", 2: "East", 3: "West", 4: "Tag"}

                if action in action_dirs:
                    dr, dc = action_dirs[action]
                    # Arrow from robot position in direction of action
                    action_arrow.set_position((robot_pos[0], robot_pos[1]))
                    action_arrow.xy = (robot_pos[0] + dr * 0.3, robot_pos[1] + dc * 0.3)

                    # Update text displays
                    step_text.set_text(f"Step: {frame + 1}/{len(robot_path)}")
                    action_text.set_text(f'Action: {action_names.get(action, "Unknown")}')

                    # Show tag indicator for tag actions
                    if action == 4:  # Tag action
                        if robot_pos == opponent_pos:  # Successful tag
                            tag_text.set_text("🏷️ TAGGED! 🏷️")
                            tag_text.set_bbox(
                                dict(
                                    boxstyle="round,pad=0.5",
                                    facecolor="gold",
                                    edgecolor="green",
                                    linewidth=3,
                                    alpha=0.9,
                                )
                            )
                            tag_text.set_color("green")
                            tag_text.set_visible(True)
                        else:  # Failed tag attempt
                            tag_text.set_text("❌ MISSED! ❌")
                            tag_text.set_bbox(
                                dict(
                                    boxstyle="round,pad=0.5",
                                    facecolor="lightcoral",
                                    edgecolor="red",
                                    linewidth=3,
                                    alpha=0.9,
                                )
                            )
                            tag_text.set_color("darkred")
                            tag_text.set_visible(True)
                    else:
                        tag_text.set_visible(False)

            # Update belief particles if available
            if frame < len(beliefs) and beliefs[frame] is not None:
                try:
                    belief = beliefs[frame]
                    if hasattr(belief, "to_unique_support_distribution"):
                        unique_belief = belief.to_unique_support_distribution()
                        if len(unique_belief.values) > 0:
                            # Extract opponent and robot positions from belief states
                            opponent_belief_positions = []
                            opponent_belief_weights = []
                            robot_belief_positions = []
                            robot_belief_weights = []

                            for i, state in enumerate(unique_belief.values):
                                if isinstance(state, LaserTagState):
                                    # Convert row,col to x,y for plotting (transposed mapping)
                                    opponent_belief_positions.append(
                                        [state.opponent[0], state.opponent[1]]
                                    )
                                    opponent_belief_weights.append(
                                        unique_belief.probs[i] * 100
                                    )  # Scale for visibility

                                    robot_belief_positions.append([state.robot[0], state.robot[1]])
                                    robot_belief_weights.append(
                                        unique_belief.probs[i] * 100
                                    )  # Scale for visibility

                            if opponent_belief_positions:
                                opponent_belief_scatter.set_offsets(
                                    np.array(opponent_belief_positions)
                                )
                                opponent_belief_scatter.set_sizes(np.array(opponent_belief_weights))
                            else:
                                opponent_belief_scatter.set_offsets(np.empty((0, 2)))

                            if robot_belief_positions:
                                robot_belief_scatter.set_offsets(np.array(robot_belief_positions))
                                robot_belief_scatter.set_sizes(np.array(robot_belief_weights))
                            else:
                                robot_belief_scatter.set_offsets(np.empty((0, 2)))
                        else:
                            opponent_belief_scatter.set_offsets(np.empty((0, 2)))
                            robot_belief_scatter.set_offsets(np.empty((0, 2)))
                except:
                    opponent_belief_scatter.set_offsets(np.empty((0, 2)))
                    robot_belief_scatter.set_offsets(np.empty((0, 2)))
            else:
                opponent_belief_scatter.set_offsets(np.empty((0, 2)))
                robot_belief_scatter.set_offsets(np.empty((0, 2)))

            # Update laser rays visualization
            for i, (line, direction) in enumerate(zip(laser_lines, laser_directions)):
                dr, dc = direction
                # Show laser ray from robot position
                start_x, start_y = robot_pos[0], robot_pos[1]  # row, col

                # Calculate laser measurement for visualization
                robot_r, robot_c = robot_pos
                distance = 0
                ray_x, ray_y = robot_r, robot_c

                # Cast ray to find end point
                while True:
                    ray_r = robot_r + dr * (distance + 1)
                    ray_c = robot_c + dc * (distance + 1)

                    # Check bounds and walls
                    if (
                        ray_r < 0
                        or ray_r >= self.floor_shape[0]
                        or ray_c < 0
                        or ray_c >= self.floor_shape[1]
                        or (ray_r, ray_c) in self.walls
                    ):
                        break
                    distance += 1

                # Draw laser ray
                end_x = robot_r + dr * distance
                end_y = robot_c + dc * distance
                line.set_data([start_x, end_x], [start_y, end_y])

            return [
                robot_agent,
                opponent_agent,
                robot_path_line,
                opponent_path_line,
                action_arrow,
                step_text,
                action_text,
                tag_text,
                opponent_belief_scatter,
                robot_belief_scatter,
            ] + laser_lines

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(robot_path),
            init_func=init,
            blit=True,
            repeat=False,
            interval=1000,  # 1 second per frame
        )

        # Save animation with proper layout
        plt.tight_layout()
        anim.save(cache_path, writer="pillow", fps=1)
        plt.close(fig)

        self.logger.info(f"Saved LaserTag visualization to {cache_path}")
