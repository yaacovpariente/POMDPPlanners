# pylint: disable=too-many-lines
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
- When aligned on an axis, the 0.4 budget is split equally (0.2/0.2) between both directions

Classes:
    LaserTagState: State representation with robot and opponent positions
    LaserTagPOMDP: Main environment class implementing the LaserTag problem
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.utils.statistics_utils import confidence_interval
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_visualizer import (  # pylint: disable=import-outside-toplevel
    LaserTagVisualizer,
)


# 8-directional laser measurements: N, NE, E, SE, S, SW, W, NW (matches LaserTagObservation)
_LASER_DIRECTIONS: List[Tuple[int, int]] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]


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

    def __init__(  # pylint: disable=dangerous-default-value
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
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("discount_factor must be between 0 and 1 (inclusive)")
        if not 0.0 <= transition_error_prob <= 1.0:
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

        if walls is None:
            walls = {(1, 2), (3, 0), (3, 4), (5, 0), (6, 4), (9, 1), (9, 4), (10, 6)}
        if dangerous_areas is None:
            dangerous_areas = {(5, 3), (7, 1), (2, 5)}
        self.floor_shape: Tuple[int, int] = floor_shape
        self.walls: Set[Tuple[int, int]] = walls
        self.tag_reward = tag_reward
        self.tag_penalty = tag_penalty
        self.step_cost = step_cost
        self.measurement_noise = measurement_noise
        self.dangerous_areas: List[Tuple[int, int]] = list(dangerous_areas)
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
        # Pre-built C-contiguous int64 (4, 2) array of (dr, dc) for actions 0..3,
        # consumed by the native ``lasertag_discrete_reward_batch`` kernel so the
        # hot path doesn't repack a Python dict on every call.
        self._action_directions_arr: np.ndarray = np.ascontiguousarray(
            np.array(
                [self._action_directions[a] for a in (0, 1, 2, 3)],
                dtype=np.int64,
            )
        )
        # Pre-built (D, 2) C-contiguous float64 array of dangerous-area centres
        # (or empty (0, 2) when none are configured) so the native kernel can
        # consume it directly without per-call reallocation.
        if self.dangerous_areas:
            self._dangerous_areas_arr: np.ndarray = np.ascontiguousarray(
                np.asarray(self.dangerous_areas, dtype=np.float64).reshape(-1, 2)
            )
        else:
            self._dangerous_areas_arr = np.empty((0, 2), dtype=np.float64)
        # Flattened int64 walls buffer (length 2 * n_walls) for the native
        # kernel; pairs are (row, col). Sorted for deterministic ordering.
        walls_list = sorted(self.walls)
        self._reward_walls_flat: np.ndarray = np.array(
            [coord for pair in walls_list for coord in pair],
            dtype=np.int64,
        )
        self._reward_n_walls: int = len(walls_list)

    def _is_valid_position_inline(self, pos: Tuple[int, int]) -> bool:
        row, col = pos
        return (
            0 <= row < self.floor_shape[0]
            and 0 <= col < self.floor_shape[1]
            and pos not in self.walls
        )

    def _get_native_step_params(self) -> Optional[Any]:
        """Return cached static params for the native single-step kernels."""
        cached = getattr(self, "_cached_native_step_params", None)
        if cached is not None:
            return cached
        try:
            from POMDPPlanners.environments.laser_tag_pomdp import (  # pylint: disable=import-outside-toplevel
                _native,
            )
        except ImportError:
            return None
        if not hasattr(_native, "sample_next_state_step"):
            return None
        walls_list = sorted(self.walls)
        walls_flat = np.array([c for pair in walls_list for c in pair], dtype=np.int64)
        params = (_native, int(self.floor_shape[0]), int(self.floor_shape[1]), walls_flat)
        # pylint: disable=attribute-defined-outside-init
        self._cached_native_step_params = params
        return params

    def sample_next_state(self, state: np.ndarray, action: int, n_samples: int = 1) -> Any:
        # Fast path: native single-step C++ kernel for the n_samples == 1 case
        # (the POMCPOW hot path). RNG draws are issued from numpy in the same
        # order and quantity as the original Python implementation, then
        # forwarded to C++ to preserve byte-identical reproducibility.
        if n_samples == 1:
            params = self._get_native_step_params()
            if params is not None:
                return self._native_sample_next_state_one(state, action, params)

        # Slow / batch path: original numpy implementation.
        return self._python_sample_next_state(state, action, n_samples)

    def _native_sample_next_state_one(
        self,
        state: np.ndarray,
        action: int,
        params: Any,
    ) -> np.ndarray:
        # Resolve actual_action via the same numpy RNG draws as the original
        # Python path (one np.random.random() coin for action != 4, plus an
        # np.random.choice for the error branch when triggered).
        actual_action = self._resolve_actual_action(action)

        # Successful tag short-circuit: no opponent draw needed.
        robot_current = (int(state[0]), int(state[1]))
        opponent_current = (int(state[2]), int(state[3]))
        if actual_action == 4 and robot_current == opponent_current:
            return np.array(
                [
                    float(robot_current[0]),
                    float(robot_current[1]),
                    float(opponent_current[0]),
                    float(opponent_current[1]),
                    1.0,
                ]
            )

        # Otherwise draw the opponent uniform via numpy and forward to C++.
        opp_uniform = float(np.random.random())
        native, rows, cols, walls_flat = params
        return native.sample_next_state_step(
            state=np.ascontiguousarray(np.asarray(state, dtype=np.float64)),
            actual_action=int(actual_action),
            opp_uniform=opp_uniform,
            rows=rows,
            cols=cols,
            walls_flat=walls_flat,
        )

    def _resolve_actual_action(self, action: int) -> int:
        # Mirrors the action-error coin used by the original Python path.
        if action == 4:
            return 4
        if np.random.random() < self.transition_error_prob:
            available_actions = [a for a in (0, 1, 2, 3) if a != action]
            return int(np.random.choice(available_actions))
        return action

    def _python_sample_next_state(self, state: np.ndarray, action: int, n_samples: int) -> Any:
        # _get_actual_action: matches LaserTagStateTransition._get_actual_action
        if action == 4:
            actual_action = action
        else:
            if np.random.random() < self.transition_error_prob:
                available_actions = [a for a in [0, 1, 2, 3] if a != action]
                actual_action = int(np.random.choice(available_actions))
            else:
                actual_action = action

        # _get_robot_next_position(actual_action)
        robot_current = (int(state[0]), int(state[1]))
        if actual_action == 4:
            robot_next = robot_current
        else:
            dr, dc = self._action_directions[actual_action]
            cand = (robot_current[0] + dr, robot_current[1] + dc)
            robot_next = cand if self._is_valid_position_inline(cand) else robot_current

        opponent_current = (int(state[2]), int(state[3]))

        # Tag at same cell → terminal: no extra RNG draws regardless of n_samples
        if actual_action == 4 and robot_current == opponent_current:
            terminal_array = np.array(
                [
                    float(robot_next[0]),
                    float(robot_next[1]),
                    float(opponent_current[0]),
                    float(opponent_current[1]),
                    1.0,
                ]
            )
            if n_samples == 1:
                return terminal_array
            return [terminal_array.copy() for _ in range(n_samples)]

        # Regular transition: build opponent move distribution then draw indices
        # in a single np.random.choice call (matches the wrapper's RNG draw order
        # for any n_samples).
        opp_moves = self._opponent_move_probabilities_inline(state, robot_next)
        positions, probabilities = zip(*opp_moves)
        opp_indices = np.random.choice(len(positions), size=n_samples, p=probabilities)
        if n_samples == 1:
            opp_next_pos = positions[opp_indices[0]]
            return np.array(
                [
                    float(robot_next[0]),
                    float(robot_next[1]),
                    float(opp_next_pos[0]),
                    float(opp_next_pos[1]),
                    0.0,
                ]
            )
        samples: List[np.ndarray] = []
        for idx in opp_indices:
            opp_next_pos = positions[idx]
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

    def _opponent_move_probabilities_inline(
        self, state: np.ndarray, robot_pos: Tuple[int, int]
    ) -> List[Tuple[Tuple[int, int], float]]:
        # Mirror of LaserTagStateTransition._get_opponent_move_probabilities,
        # but operating on a state ndarray rather than self.state.
        current_opp = (int(state[2]), int(state[3]))
        robot_row, robot_col = robot_pos
        opp_row, opp_col = current_opp

        x_moves = self._directional_moves_inline(opp_col, robot_col, opp_row, True)
        y_moves = self._directional_moves_inline(opp_row, robot_row, opp_col, False)

        move_probs = x_moves + y_moves + [(current_opp, 0.2)]
        actual_total = sum(prob for _, prob in move_probs if prob > 0)
        if actual_total < 1.0:
            stay_index = len(move_probs) - 1
            current_pos, current_stay_prob = move_probs[stay_index]
            move_probs[stay_index] = (current_pos, current_stay_prob + (1.0 - actual_total))
        return [(pos, prob) for pos, prob in move_probs if prob > 0]

    def _directional_moves_inline(
        self, opponent_coord: int, robot_coord: int, fixed_coord: int, is_horizontal: bool
    ) -> List[Tuple[Tuple[int, int], float]]:
        if robot_coord > opponent_coord:
            toward_pos = (
                (fixed_coord, opponent_coord + 1)
                if is_horizontal
                else (opponent_coord + 1, fixed_coord)
            )
            away_pos = (
                (fixed_coord, opponent_coord - 1)
                if is_horizontal
                else (opponent_coord - 1, fixed_coord)
            )
            toward_prob, away_prob = 0.4, 0.0
        elif robot_coord < opponent_coord:
            toward_pos = (
                (fixed_coord, opponent_coord - 1)
                if is_horizontal
                else (opponent_coord - 1, fixed_coord)
            )
            away_pos = (
                (fixed_coord, opponent_coord + 1)
                if is_horizontal
                else (opponent_coord + 1, fixed_coord)
            )
            toward_prob, away_prob = 0.4, 0.0
        else:
            toward_pos = (
                (fixed_coord, opponent_coord + 1)
                if is_horizontal
                else (opponent_coord + 1, fixed_coord)
            )
            away_pos = (
                (fixed_coord, opponent_coord - 1)
                if is_horizontal
                else (opponent_coord - 1, fixed_coord)
            )
            toward_prob, away_prob = 0.2, 0.2

        moves: List[Tuple[Tuple[int, int], float]] = []
        if self._is_valid_position_inline(toward_pos):
            moves.append((toward_pos, toward_prob))
        if self._is_valid_position_inline(away_pos):
            moves.append((away_pos, away_prob))
        return moves

    def sample_observation(self, next_state: np.ndarray, action: int, n_samples: int = 1) -> Any:
        if bool(next_state[4]):
            terminal_obs = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
            if n_samples == 1:
                return terminal_obs
            return [terminal_obs] * n_samples

        # Fast path: native single-step C++ kernel for n_samples == 1 (the
        # POMCPOW hot path). The 8 noise samples are pre-drawn from numpy in
        # the same order as the original Python path so byte-identical numpy
        # RNG state is preserved across both paths.
        if n_samples == 1:
            params = self._get_native_step_params()
            if params is not None:
                native, rows, cols, walls_flat = params
                noise = np.random.normal(0, self.measurement_noise, size=8)
                obs_arr = native.sample_observation_step(
                    next_state=np.ascontiguousarray(np.asarray(next_state, dtype=np.float64)),
                    noise=np.ascontiguousarray(noise, dtype=np.float64),
                    rows=rows,
                    cols=cols,
                    walls_flat=walls_flat,
                )
                # tolist() is ~8x faster than a per-element float() genexpr and
                # produces a Python list of floats; tuple() wraps to match the
                # historical return type.
                return tuple(obs_arr.tolist())

        robot_pos = (int(next_state[0]), int(next_state[1]))
        opp_pos = (int(next_state[2]), int(next_state[3]))
        # Compute true 8-direction laser measurements (no RNG)
        true_measurements = [
            self._laser_distance_inline(robot_pos, direction, opp_pos)
            for direction in _LASER_DIRECTIONS
        ]
        # Add Gaussian noise to each measurement (8 np.random.normal draws per
        # sample, in dir order — matches wrapper's RNG draw sequence).
        if n_samples == 1:
            noisy: List[float] = []
            for true_measure in true_measurements:
                noise_value = np.random.normal(0, self.measurement_noise)
                noisy.append(max(0.0, true_measure + noise_value))
            return tuple(noisy)

        samples: List[Tuple[float, ...]] = []
        for _ in range(n_samples):
            noisy_inner: List[float] = []
            for true_measure in true_measurements:
                noise_value = np.random.normal(0, self.measurement_noise)
                noisy_inner.append(max(0.0, true_measure + noise_value))
            samples.append(tuple(noisy_inner))
        return samples

    def transition_log_probability(
        self, state: np.ndarray, action: int, next_states: Any
    ) -> np.ndarray:
        # Inlined from the deleted LaserTagStateTransition.probability(): for Tag
        # action (4), probability is deterministic; for movement actions, mix the
        # intended-action probability with uniformly distributed error actions.
        result = np.zeros(len(next_states))
        if action == 4:
            for i, next_state in enumerate(next_states):
                result[i] = self._transition_probability_for_action(state, next_state, 4)
        else:
            error_actions = [a for a in (0, 1, 2, 3) if a != action]
            error_weight = (
                self.transition_error_prob / len(error_actions)
                if (self.transition_error_prob > 0.0 and len(error_actions) > 0)
                else 0.0
            )
            for i, next_state in enumerate(next_states):
                prob_intended = (1.0 - self.transition_error_prob) * (
                    self._transition_probability_for_action(state, next_state, action)
                )
                prob_error = 0.0
                if error_weight > 0.0:
                    prob_error = error_weight * sum(
                        self._transition_probability_for_action(state, next_state, error_action)
                        for error_action in error_actions
                    )
                result[i] = prob_intended + prob_error
        with np.errstate(divide="ignore"):
            return np.log(result)

    def _transition_probability_for_action(
        self, state: np.ndarray, next_state: Any, action: int
    ) -> float:
        # Inlined from the deleted LaserTagStateTransition._compute_transition_probability_for_action.
        if not isinstance(next_state, np.ndarray) or len(next_state) != 5:
            return 0.0

        robot_current = (int(state[0]), int(state[1]))
        opponent_current = (int(state[2]), int(state[3]))

        if action == 4:
            robot_next = robot_current
        else:
            dr, dc = self._action_directions[action]
            cand = (robot_current[0] + dr, robot_current[1] + dc)
            robot_next = cand if self._is_valid_position_inline(cand) else robot_current

        next_robot = (int(next_state[0]), int(next_state[1]))
        next_opponent = (int(next_state[2]), int(next_state[3]))
        next_terminal = bool(next_state[4])

        # Successful tag: deterministic transition into terminal state.
        if action == 4 and robot_current == opponent_current:
            if next_robot == robot_next and next_opponent == opponent_current and next_terminal:
                return 1.0
            return 0.0

        # Regular transition: opponent moves stochastically, terminal flag stays 0.
        if next_robot == robot_next and not next_terminal:
            opp_moves = self._opponent_move_probabilities_inline(state, robot_next)
            for opp_pos, prob in opp_moves:
                if next_opponent == opp_pos:
                    return prob
        return 0.0

    def observation_log_probability(
        self, next_state: np.ndarray, action: int, observations: Any
    ) -> np.ndarray:
        # Inlined from the deleted LaserTagObservation.probability(): terminal
        # states emit a sentinel observation deterministically; non-terminal states
        # emit independent Gaussian-noise laser ranges in 8 directions.
        del action  # observation distribution does not depend on action in LaserTag

        # Fast path: native C++ kernel that mirrors the Python loop bit-for-bit
        # but skips Python-level per-direction overhead (laser ray-casting,
        # exp/sqrt, tuple iteration). The native entry handles the terminal
        # sentinel branch and returns log-probabilities directly.
        params = self._get_native_step_params()
        if params is not None:
            obs_arr = self._coerce_observations_array(observations)
            if obs_arr is not None:
                native, rows, cols, walls_flat = params
                return np.asarray(
                    native.observation_log_probability_step(
                        next_state=np.ascontiguousarray(np.asarray(next_state, dtype=np.float64)),
                        observations=obs_arr,
                        measurement_noise=float(self.measurement_noise),
                        rows=rows,
                        cols=cols,
                        walls_flat=walls_flat,
                    )
                )
        return self._python_observation_log_probability(next_state, observations)

    @staticmethod
    def _coerce_observations_array(observations: Any) -> Optional[np.ndarray]:
        # Convert a heterogeneous observation collection (tuple of tuples,
        # list of ndarrays, etc.) into a contiguous (N, 8) float64 array. Any
        # row that is not exactly length 8 disqualifies the fast path so the
        # native kernel never sees malformed inputs.
        if isinstance(observations, np.ndarray):
            arr = observations
        else:
            try:
                arr = np.asarray(observations, dtype=np.float64)
            except (ValueError, TypeError):
                return None
        if arr.ndim == 1:
            if arr.shape[0] != 8:
                return None
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != 8:
            return None
        return np.ascontiguousarray(arr, dtype=np.float64)

    def _python_observation_log_probability(
        self, next_state: np.ndarray, observations: Any
    ) -> np.ndarray:
        # Pure-Python fallback retained for parity testing and for unusual
        # observation shapes the native fast path declines to handle.
        result = np.zeros(len(observations))

        if bool(next_state[4]):
            terminal_obs = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
            for i, obs in enumerate(observations):
                if np.array_equal(obs, terminal_obs):
                    result[i] = 1.0
            with np.errstate(divide="ignore"):
                return np.log(result)

        robot_pos = (int(next_state[0]), int(next_state[1]))
        opp_pos = (int(next_state[2]), int(next_state[3]))
        true_measurements = [
            self._laser_distance_inline(robot_pos, direction, opp_pos)
            for direction in _LASER_DIRECTIONS
        ]
        variance = self.measurement_noise**2
        norm_const = 1.0 / np.sqrt(2 * np.pi * variance)

        for i, obs in enumerate(observations):
            if isinstance(obs, (tuple, list, np.ndarray)) and len(obs) == 8:
                prob = 1.0
                for true_measure, observed_measure in zip(true_measurements, obs):
                    if observed_measure >= 0:
                        diff = observed_measure - true_measure
                        prob *= np.exp(-0.5 * diff**2 / variance) * norm_const
                    else:
                        prob = 0.0
                        break
                result[i] = prob
        with np.errstate(divide="ignore"):
            return np.log(result)

    def _laser_distance_inline(
        self,
        robot_pos: Tuple[int, int],
        direction: Tuple[int, int],
        opp_pos: Tuple[int, int],
    ) -> float:
        row, col = robot_pos
        dr, dc = direction
        distance = 0.0
        while True:
            row += dr
            col += dc
            distance += 1.0
            if row < 0 or row >= self.floor_shape[0] or col < 0 or col >= self.floor_shape[1]:
                break
            if (row, col) in self.walls or (row, col) == opp_pos:
                break
        return distance - 1.0

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

    def _get_native_rollout_params(
        self,
    ) -> Optional[Any]:
        """Return cached static params for the native discrete rollout, or None."""
        cached = getattr(self, "_cached_native_rollout_params", None)
        if cached is not None:
            return cached
        try:
            # pylint: disable=import-outside-toplevel
            from POMDPPlanners.environments.laser_tag_pomdp import (
                _native,
            )
        except ImportError:
            return None
        if not hasattr(_native, "simulate_rollout_discrete"):
            return None
        walls_list = sorted(self.walls)
        walls_flat = np.array([coord for pair in walls_list for coord in pair], dtype=np.int64)
        if self.dangerous_areas:
            dangerous_areas_arr = np.array(self.dangerous_areas, dtype=np.float64)
        else:
            dangerous_areas_arr = np.empty((0,), dtype=np.float64)
        params = (
            _native,
            self.floor_shape[0],
            self.floor_shape[1],
            walls_flat,
            dangerous_areas_arr,
            float(self.dangerous_area_radius),
            float(self.dangerous_area_penalty),
            float(self.tag_reward),
            float(self.tag_penalty),
            float(self.step_cost),
            float(self.transition_error_prob),
        )
        # pylint: disable=attribute-defined-outside-init
        self._cached_native_rollout_params = params
        return params

    def simulate_random_rollout(
        self,
        state: Any,
        action_sampler: Any,
        max_depth: int,
        discount_factor: float,
        depth: int = 0,
    ) -> float:
        # Attempt the native C++ rollout.  The C++ kernel draws actions
        # uniformly from {0,1,2,3,4} using the module-level mt19937_64 RNG,
        # which differs from the Python path's numpy mt19937 RNG; the two paths
        # are therefore only equivalent in distribution, not bit-by-bit.
        # If the action_sampler is not a uniform sampler over all 5 actions, fall
        # back to the Python loop so planner-specific rollout policies still work.
        params = self._get_native_rollout_params()
        if params is not None:
            state_arr = np.ascontiguousarray(np.asarray(state, dtype=np.float64))
            (
                _native,
                rows,
                cols,
                walls_flat,
                dangerous_areas_arr,
                dangerous_area_radius,
                dangerous_area_penalty,
                tag_reward,
                tag_penalty,
                step_cost,
                transition_error_prob,
            ) = params
            return float(
                _native.simulate_rollout_discrete(
                    initial_state=state_arr,
                    max_depth=max_depth,
                    discount=discount_factor,
                    initial_depth=depth,
                    rows=rows,
                    cols=cols,
                    walls_flat=walls_flat,
                    dangerous_areas=dangerous_areas_arr,
                    dangerous_area_radius=dangerous_area_radius,
                    dangerous_area_penalty=dangerous_area_penalty,
                    tag_reward=tag_reward,
                    tag_penalty=tag_penalty,
                    step_cost=step_cost,
                    transition_error_prob=transition_error_prob,
                )
            )

        # Python fallback (also used in equivalence tests via super()).
        sample_next = self.sample_next_state
        reward_fn = self.reward
        action_sample = action_sampler.sample

        total = 0.0
        gamma_power = 1.0
        current = state
        while depth < max_depth and current[4] != 1.0:
            action = action_sample()
            r = reward_fn(state=current, action=action)
            total += gamma_power * r
            current = sample_next(state=current, action=action)
            gamma_power *= discount_factor
            depth += 1
        return total

    # ── Vectorized batch overrides ─────────────────────────────────
    # PFT-DPW belief updates and any caller of the batch API otherwise hit
    # the per-state Python fallback in ``Environment``. Delegate to the
    # vectorized updater (which already exists for explicit belief
    # filtering) so all-particle work happens inside NumPy, not a Python
    # loop. The updater is built lazily on first call and cached.

    def _get_vectorized_updater(self) -> Any:
        cached = getattr(self, "_cached_vectorized_updater", None)
        if cached is not None:
            return cached
        # pylint: disable=import-outside-toplevel
        from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_beliefs.laser_tag_vectorized_updater import (
            LaserTagVectorizedUpdater,
        )

        cached = LaserTagVectorizedUpdater.from_environment(self)
        # pylint: disable=attribute-defined-outside-init
        self._cached_vectorized_updater = cached
        return cached

    def sample_next_state_batch(self, states: Any, action: int) -> np.ndarray:
        states_array = np.ascontiguousarray(np.asarray(states, dtype=float))
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        return self._get_vectorized_updater().batch_transition(states_array, np.asarray(action))

    def observation_log_probability_per_state(
        self, next_states: Any, action: int, observation: Any
    ) -> np.ndarray:
        next_states_arr = np.ascontiguousarray(np.asarray(next_states, dtype=float))
        if next_states_arr.ndim == 1:
            next_states_arr = next_states_arr.reshape(1, -1)
        return self._get_vectorized_updater().batch_observation_log_likelihood(
            next_states_arr, np.asarray(action), np.asarray(observation, dtype=float)
        )

    def _get_wall_grid(self) -> np.ndarray:
        cached = getattr(self, "_wall_grid", None)
        if cached is not None:
            return cached
        grid = np.zeros(self.floor_shape, dtype=bool)
        for row, col in self.walls:
            grid[row, col] = True
        # pylint: disable=attribute-defined-outside-init
        self._wall_grid = grid
        return grid

    def reward_batch(self, states: Any, action: int) -> np.ndarray:
        """Vectorised reward for a batch of states under a single action.

        Args:
            states: Array-like of shape (N, 5) or (5,) representing particle states.
                Each state is [robot_row, robot_col, opp_row, opp_col, terminal_flag].
            action: Integer action in {0, 1, 2, 3, 4}.

        Returns:
            Float64 array of shape (N,) with per-particle rewards.
        """
        states_arr = np.asarray(states, dtype=np.float64)
        if states_arr.ndim == 1:
            states_arr = states_arr.reshape(1, -1)
        states_arr = np.ascontiguousarray(states_arr)
        native_fn = self._get_native_reward_batch()
        if native_fn is not None:
            return np.asarray(
                native_fn(
                    states=states_arr,
                    action=int(action),
                    rows=int(self.floor_shape[0]),
                    cols=int(self.floor_shape[1]),
                    walls_flat=self._reward_walls_flat,
                    n_walls=self._reward_n_walls,
                    dangerous_areas=self._dangerous_areas_arr,
                    n_dangerous=int(self._dangerous_areas_arr.shape[0]),
                    dangerous_area_radius=float(self.dangerous_area_radius),
                    dangerous_area_penalty=float(self.dangerous_area_penalty),
                    tag_reward=float(self.tag_reward),
                    tag_penalty=float(self.tag_penalty),
                    step_cost=float(self.step_cost),
                    action_directions=self._action_directions_arr,
                )
            )
        return self._compute_reward_batch_python(states_arr, action)

    def _get_native_reward_batch(self) -> Optional[Any]:
        cached = getattr(self, "_cached_native_reward_batch", None)
        if cached is not None:
            return cached if cached is not False else None
        try:
            from POMDPPlanners.environments.laser_tag_pomdp import (  # pylint: disable=import-outside-toplevel
                _native,
            )
        except ImportError:
            # pylint: disable=attribute-defined-outside-init
            self._cached_native_reward_batch = False
            return None
        fn = getattr(_native, "lasertag_discrete_reward_batch", None)
        # pylint: disable=attribute-defined-outside-init
        self._cached_native_reward_batch = fn if fn is not None else False
        return fn

    def _compute_reward_batch_python(self, states: np.ndarray, action: int) -> np.ndarray:
        n = states.shape[0]
        terminal_mask = states[:, 4].astype(bool)

        robot_r = states[:, 0].astype(np.int64)
        robot_c = states[:, 1].astype(np.int64)
        opp_r = states[:, 2].astype(np.int64)
        opp_c = states[:, 3].astype(np.int64)

        rewards = self._compute_base_rewards(action, robot_r, robot_c, opp_r, opp_c, n)
        self._apply_area_penalty_batch(rewards, action, robot_r, robot_c)

        rewards[terminal_mask] = 0.0
        return rewards

    def _compute_base_rewards(
        self,
        action: int,
        robot_r: np.ndarray,
        robot_c: np.ndarray,
        opp_r: np.ndarray,
        opp_c: np.ndarray,
        n: int,
    ) -> np.ndarray:
        if action == 4:
            at_same_pos = (robot_r == opp_r) & (robot_c == opp_c)
            rewards = np.where(at_same_pos, float(self.tag_reward), float(-self.tag_penalty))
        else:
            rewards = np.full(n, float(-self.step_cost), dtype=np.float64)
        return rewards

    def _compute_intended_positions(
        self, action: int, robot_r: np.ndarray, robot_c: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if action in (0, 1, 2, 3):
            dr, dc = self._action_directions[action]
            return robot_r + dr, robot_c + dc
        return robot_r.copy(), robot_c.copy()

    def _apply_area_penalty_batch(
        self, rewards: np.ndarray, action: int, robot_r: np.ndarray, robot_c: np.ndarray
    ) -> None:
        int_r, int_c = self._compute_intended_positions(action, robot_r, robot_c)

        wall_mask = self._compute_wall_hit_mask(int_r, int_c)
        danger_mask = self._compute_danger_mask(int_r, int_c)

        penalty_mask = wall_mask | danger_mask
        rewards[penalty_mask] -= self.dangerous_area_penalty

    def _compute_wall_hit_mask(self, int_r: np.ndarray, int_c: np.ndarray) -> np.ndarray:
        rows, cols = self.floor_shape
        wall_grid = self._get_wall_grid()

        # Out-of-bounds positions are NOT in self.walls, so they return False here.
        # We clip to grid bounds for safe indexing, then zero-out OOB entries via the mask.
        clipped_r = np.clip(int_r, 0, rows - 1)
        clipped_c = np.clip(int_c, 0, cols - 1)
        in_bounds = (int_r >= 0) & (int_r < rows) & (int_c >= 0) & (int_c < cols)
        return in_bounds & wall_grid[clipped_r, clipped_c]

    def _compute_danger_mask(self, int_r: np.ndarray, int_c: np.ndarray) -> np.ndarray:
        if not self.dangerous_areas:
            return np.zeros(len(int_r), dtype=bool)
        centers = np.array(self.dangerous_areas, dtype=np.float64)  # shape (D, 2)
        # Euclidean distance from each intended position to each danger center: (N, D)
        dr = int_r[:, np.newaxis] - centers[:, 0]
        dc = int_c[:, np.newaxis] - centers[:, 1]
        dist = np.sqrt(dr**2 + dc**2)
        return np.asarray((dist <= self.dangerous_area_radius).any(axis=1), dtype=bool)

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
        - Robot movement (red circle)
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
        visualizer = LaserTagVisualizer(
            floor_shape=self.floor_shape,
            walls=self.walls,
            dangerous_areas=self.dangerous_areas,
            dangerous_area_radius=self.dangerous_area_radius,
        )
        visualizer.create_visualization(history, cache_path)
        self.logger.info("Saved LaserTag visualization to %s", cache_path)
