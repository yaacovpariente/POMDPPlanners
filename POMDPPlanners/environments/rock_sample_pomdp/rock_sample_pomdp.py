# SPDX-License-Identifier: MIT

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
import warnings
from enum import Enum
from pathlib import Path
from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.rock_sample_pomdp import _native
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_utils.rock_sample_reward_models import (
    BaseRockSampleRewardModel,
    RockSampleDistanceDecayedHazardPenaltyRewardModel,
    RockSampleZeroMeanHazardShockRewardModel,
    RockSampleRewardModel,
)
from POMDPPlanners.utils.statistics_utils import confidence_interval


class RockSamplePOMDPMetrics(Enum):
    """Metric names for RockSample POMDP environment."""

    AVG_ROCKS_SAMPLED = "avg_rocks_sampled"
    EXIT_SUCCESS_RATE = "exit_success_rate"
    AVERAGE_DANGEROUS_AREA_STEPS = "average_dangerous_area_steps"


class RewardModelType(Enum):
    """Reward-model variants for :class:`RockSamplePOMDP`.

    Variants differ only in how the dangerous-area penalty is applied —
    base scoring (exit / sample / sense / step) is identical across all
    three. See
    :mod:`POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_utils.rock_sample_reward_models`
    for the per-variant semantics.
    """

    CONSTANT_HAZARD_PENALTY = "constant_hazard_penalty"
    DISTANCE_DECAYED_HAZARD_PENALTY = "distance_decayed_hazard_penalty"
    ZERO_MEAN_HAZARD_SHOCK = "zero_mean_hazard_shock"


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


class RockSamplePOMDP(DiscreteActionsEnvironment):  # pylint: disable=too-many-public-methods
    """RockSample POMDP environment

    This environment implements the classic rock sampling problem where a robot
    must navigate a grid, use sensors to evaluate rocks, and decide which ones
    to sample while balancing exploration costs and sampling rewards.

    Stochasticity:
        The dangerous-area penalty can be applied either deterministically
        (the default) or stochastically. When
        ``dangerous_area_hit_probability == 1.0`` (default), the penalty is
        applied every time the robot's next position lies inside a
        dangerous area, matching legacy behavior. When
        ``dangerous_area_hit_probability < 1.0``, the penalty is applied
        only with that probability per ``reward()`` / ``reward_batch()``
        call (one Bernoulli draw per state), producing a heavy-tailed
        return distribution suitable for benchmarking risk-sensitive
        planners (e.g. ICVaR-aware MCTS) against expected-value MCTS on
        the same env. Note that this makes ``reward(state, action)``
        non-deterministic given a state-action pair, so any external
        caching that assumes deterministic rewards must be aware of this.
        ``transition_log_probability`` is unaffected.

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
        dangerous_area_penalty: float = -5.0,
        dangerous_area_hit_probability: float = 1.0,
        reward_model_type: RewardModelType = RewardModelType.CONSTANT_HAZARD_PENALTY,
        penalty_decay: float = 1.0,
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
            dangerous_area_penalty: Reward delta added to ``total_reward`` when the
                robot's next position lies inside a dangerous area (per-step,
                gated by ``dangerous_area_hit_probability``). Use a negative
                value to penalize danger entry; positive values reward it (a
                ``UserWarning`` is emitted in that case). Defaults to ``-5.0``.
            dangerous_area_hit_probability: Probability that the dangerous-area
                penalty is actually applied to the reward when the robot's
                next position is inside a dangerous area. Must lie in
                ``[0, 1]``. Defaults to ``1.0`` (deterministic penalty,
                matching legacy behavior). Values below ``1.0`` make the
                reward stochastic (per-call Bernoulli draw), useful for
                risk-sensitive planning benchmarks. Note that this makes
                ``reward(state, action)`` non-deterministic given a
                state-action pair. Ignored by ``ZERO_MEAN_HAZARD_SHOCK``
                and ``DISTANCE_DECAYED_HAZARD_PENALTY`` reward models.
            reward_model_type: Which dangerous-area penalty model to use.
                Defaults to ``RewardModelType.CONSTANT_HAZARD_PENALTY`` (legacy
                constant-probability behaviour). ``ZERO_MEAN_HAZARD_SHOCK``
                applies ``±dangerous_area_penalty`` 50/50 in-zone (zero
                expected contribution, high variance — useful for
                risk-sensitive planner benchmarks).
                ``DISTANCE_DECAYED_HAZARD_PENALTY`` applies the penalty with
                probability ``exp(-min_dist / penalty_decay)`` based on
                distance to the closest dangerous-area centre (no radius
                cutoff). See
                :class:`~POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp.RewardModelType`.
            penalty_decay: Distance-decay constant for the
                ``DISTANCE_DECAYED_HAZARD_PENALTY`` reward model. Must be
                positive. Ignored by other reward models. Defaults to
                ``1.0``.
            discount_factor: Discount factor. Defaults to 0.95.
            name: Environment name. Defaults to "RockSample".
            output_dir: Output directory for logging. Defaults to None.
            debug: Enable debug logging. Defaults to False.
        """
        if not 0.0 <= dangerous_area_hit_probability <= 1.0:
            raise ValueError("dangerous_area_hit_probability must be between 0 and 1 (inclusive)")

        if dangerous_area_penalty > 0:
            warnings.warn(
                "dangerous_area_penalty is positive; with the additive reward "
                "convention this rewards danger entry. Pass a negative value "
                "to penalize.",
                UserWarning,
                stacklevel=2,
            )

        # Calculate reward range based on parameters. ZERO_MEAN_HAZARD_SHOCK
        # can flip the sign of the dangerous-area contribution, so its
        # effective danger term spans ``[-|penalty|, +|penalty|]``.
        if dangerous_areas:
            if reward_model_type == RewardModelType.ZERO_MEAN_HAZARD_SHOCK:
                danger_term_min = -abs(dangerous_area_penalty)
                danger_term_max = abs(dangerous_area_penalty)
            else:
                danger_term_min = min(0.0, dangerous_area_penalty)
                danger_term_max = max(0.0, dangerous_area_penalty)
        else:
            danger_term_min = 0.0
            danger_term_max = 0.0
        min_reward = step_penalty + bad_rock_penalty + sensor_use_penalty + danger_term_min
        max_reward = step_penalty + exit_reward + danger_term_max

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
        self.dangerous_area_hit_probability = float(dangerous_area_hit_probability)
        self.reward_model_type = reward_model_type
        self.penalty_decay = float(penalty_decay)

        self.reward_model: BaseRockSampleRewardModel = self._build_reward_model()

        # Validate parameters
        self._validate_parameters()

        # Cached int32 rock positions array; identical to what the per-call
        # wrappers build via ``np.asarray(...)``. Reused on the hot-path
        # native-kernel sample overrides to skip the per-call allocation.
        self._rock_positions_int32 = np.asarray(self.rock_positions, dtype=np.int32)

        # Flat interleaved [row0, col0, row1, col1, ...] version for the
        # native simulate_rollout_discrete kernel.
        self._rock_positions_flat: np.ndarray = np.asarray(
            [coord for rp in self.rock_positions for coord in rp], dtype=np.int32
        )

        # Cached (K, 2) float64 dangerous-area centres for the native
        # reward / rollout kernels. Empty (0, 2) array when no danger
        # zones are configured.
        if self.dangerous_areas:
            self._dangerous_areas_arr: np.ndarray = np.ascontiguousarray(
                np.asarray(self.dangerous_areas, dtype=np.float64)
            )
        else:
            self._dangerous_areas_arr = np.empty((0, 2), dtype=np.float64)
        self._reward_variant_code: int = {
            RewardModelType.CONSTANT_HAZARD_PENALTY: 0,
            RewardModelType.ZERO_MEAN_HAZARD_SHOCK: 1,
            RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY: 2,
        }[self.reward_model_type]

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

        # Per-action C++ kernel caches: actions are ``int`` so a plain
        # ``Dict[int, Any]`` suffices. Lazily built by ``_get_trans_kernel``
        # / ``_get_obs_kernel`` and reset on unpickle.
        self._trans_kernel_cache: Dict[int, Any] = {}
        self._obs_kernel_cache: Dict[int, Any] = {}

    def _build_reward_model(self) -> BaseRockSampleRewardModel:
        common_kwargs = {
            "map_size": self.map_size,
            "rock_positions": self.rock_positions,
            "step_penalty": self.step_penalty,
            "bad_rock_penalty": self.bad_rock_penalty,
            "good_rock_reward": self.good_rock_reward,
            "sensor_use_penalty": self.sensor_use_penalty,
            "exit_reward": self.exit_reward,
            "dangerous_areas": self.dangerous_areas,
            "dangerous_area_radius": self.dangerous_area_radius,
            "dangerous_area_penalty": self.dangerous_area_penalty,
            "dangerous_area_hit_probability": self.dangerous_area_hit_probability,
        }
        if self.reward_model_type == RewardModelType.CONSTANT_HAZARD_PENALTY:
            return RockSampleRewardModel(**common_kwargs)
        if self.reward_model_type == RewardModelType.ZERO_MEAN_HAZARD_SHOCK:
            return RockSampleZeroMeanHazardShockRewardModel(**common_kwargs)
        if self.reward_model_type == RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY:
            return RockSampleDistanceDecayedHazardPenaltyRewardModel(
                **common_kwargs, penalty_decay=self.penalty_decay
            )
        raise ValueError(f"Unknown reward model type: {self.reward_model_type}")

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
        """Check if a position is within any dangerous area (metrics helper)."""
        if not self.dangerous_areas:
            return False
        pos_row, pos_col = position
        for danger_row, danger_col in self.dangerous_areas:
            distance = math.sqrt((pos_row - danger_row) ** 2 + (pos_col - danger_col) ** 2)
            if distance <= self.dangerous_area_radius:
                return True
        return False

    def get_actions(self) -> List[int]:
        """Get all available actions."""
        return list(range(len(self.action_names)))

    def reward(
        self,
        state: RockSampleState,
        action: int,
        next_state: Any = None,
    ) -> float:
        """Calculate immediate reward.

        Uses the realised ``next_state`` when supplied (e.g. by
        :meth:`Environment.sample_next_step`) so the dangerous-area
        penalty fires against the same outcome as the trajectory
        instead of a fresh draw.
        """
        if next_state is None:
            next_state = self.sample_next_state(state=state, action=action)
        return self._reward_from_next_state(state, action, next_state)

    def _reward_from_next_state(
        self, state: RockSampleState, action: int, next_state: RockSampleState
    ) -> float:
        return self.reward_model.compute_reward(state, action, next_state=next_state)

    def reward_batch(
        self,
        states: Any,
        action: int,
        next_states: Any = None,
    ) -> np.ndarray:
        """Calculate rewards for a batch of states given a single action.

        Threads caller-supplied ``next_states`` through to the
        dangerous-area position check so the batch path agrees with
        the scalar :meth:`reward` whenever ``Environment.sample_next_step``
        (or any other caller) pre-samples next states. When
        ``next_states is None``, we fall back to closed-form
        reconstruction of the next robot position from
        ``(state, action)``; RockSample transitions are deterministic,
        so this fallback matches a fresh draw from
        :meth:`sample_next_state`. The per-call Bernoulli refund for
        the dangerous-area penalty is preserved in both branches.
        """
        states_array = np.ascontiguousarray(np.asarray(states, dtype=np.float64))
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        next_states_array: Optional[np.ndarray] = None
        if next_states is not None:
            next_states_array = np.ascontiguousarray(np.asarray(next_states, dtype=np.float64))
            if next_states_array.ndim == 1:
                next_states_array = next_states_array.reshape(1, -1)
        return self._reward_batch_vectorized(states_array, int(action), next_states_array)

    def _reward_batch_vectorized(
        self,
        states: np.ndarray,
        action: int,
        next_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Fallback to the Python reward model when caller did not pre-sample
        # next states (closed-form reconstruction lives there).
        if next_states is None:
            return self.reward_model.compute_reward_batch(states, action, next_states=None)
        return np.asarray(
            _native.reward_batch(
                states=np.ascontiguousarray(states, dtype=np.float64),
                action=int(action),
                next_states=np.ascontiguousarray(next_states, dtype=np.float64),
                map_rows=int(self.map_size[0]),
                map_cols=int(self.map_size[1]),
                rock_positions=self._rock_positions_int32,
                step_penalty=float(self.step_penalty),
                bad_rock_penalty=float(self.bad_rock_penalty),
                good_rock_reward=float(self.good_rock_reward),
                sensor_use_penalty=float(self.sensor_use_penalty),
                exit_reward=float(self.exit_reward),
                dangerous_areas=self._dangerous_areas_arr,
                dangerous_area_radius=float(self.dangerous_area_radius),
                dangerous_area_penalty=float(self.dangerous_area_penalty),
                dangerous_area_hit_probability=float(self.dangerous_area_hit_probability),
                reward_variant_code=int(self._reward_variant_code),
                penalty_decay=float(self.penalty_decay),
            ),
            dtype=np.float64,
        )

    # ── Native-backed env-API implementations ────────────────────────
    # Each method fetches a cached per-action C++ kernel, mutates its
    # stored state via ``set_state`` / ``set_next_state`` (when needed),
    # and dispatches to the same native sample / probability /
    # batch_sample / batch_log_likelihood entry points as before. The
    # kernel itself caches frozen env geometry (rock positions, grid
    # size, sensor efficiency) so we no longer rebuild those per call.

    def _get_trans_kernel(self, action: int) -> Any:
        kernel = self._trans_kernel_cache.get(action)
        if kernel is None:
            placeholder = np.zeros(2 + len(self.rock_positions), dtype=np.float64)
            kernel = _native.RockSampleTransitionCpp(
                state=placeholder,
                action=int(action),
                map_rows=self.map_size[0],
                map_cols=self.map_size[1],
                num_rocks=len(self.rock_positions),
                rock_positions=self._rock_positions_int32,
                sensor_efficiency=self.sensor_efficiency,
            )
            self._trans_kernel_cache[action] = kernel
        return kernel

    def _get_obs_kernel(self, action: int) -> Any:
        kernel = self._obs_kernel_cache.get(action)
        if kernel is None:
            placeholder = np.zeros(2 + len(self.rock_positions), dtype=np.float64)
            kernel = _native.RockSampleObservationCpp(
                next_state=placeholder,
                action=int(action),
                map_rows=self.map_size[0],
                map_cols=self.map_size[1],
                num_rocks=len(self.rock_positions),
                rock_positions=self._rock_positions_int32,
                sensor_efficiency=self.sensor_efficiency,
            )
            self._obs_kernel_cache[action] = kernel
        return kernel

    def sample_next_state(self, state: RockSampleState, action: int, n_samples: int = 1) -> Any:
        kernel = self._get_trans_kernel(int(action))
        kernel.set_state(np.asarray(state, dtype=float))
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def sample_observation(
        self, next_state: RockSampleState, action: int, n_samples: int = 1
    ) -> Any:
        kernel = self._get_obs_kernel(int(action))
        kernel.set_next_state(np.asarray(next_state, dtype=float))
        codes = kernel.sample(n_samples)
        if n_samples == 1:
            return _OBS_CODE_TO_STR[codes[0]]
        return [_OBS_CODE_TO_STR[c] for c in codes]

    def transition_log_probability(
        self, state: RockSampleState, action: int, next_states: Any
    ) -> np.ndarray:
        kernel = self._get_trans_kernel(int(action))
        kernel.set_state(np.asarray(state, dtype=float))
        probs = np.asarray(kernel.probability(next_states))
        with np.errstate(divide="ignore"):
            return np.log(probs)

    def observation_log_probability(
        self, next_state: RockSampleState, action: int, observations: Any
    ) -> np.ndarray:
        kernel = self._get_obs_kernel(int(action))
        kernel.set_next_state(np.asarray(next_state, dtype=float))
        # Length-1 fast path: skip the list comprehension + np.array allocation.
        if hasattr(observations, "__len__") and len(observations) == 1:
            codes = np.array([_OBS_STR_TO_CODE.get(observations[0], -1)], dtype=np.int32)
        else:
            codes = np.fromiter(
                (_OBS_STR_TO_CODE.get(v, -1) for v in observations),
                dtype=np.int32,
                count=len(observations) if hasattr(observations, "__len__") else -1,
            )
        probs = np.asarray(kernel.probability(codes))
        with np.errstate(divide="ignore"):
            return np.log(probs)

    def sample_next_state_batch(self, states: Any, action: int) -> np.ndarray:
        states_array = np.ascontiguousarray(np.asarray(states, dtype=np.float64))
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        # ``batch_sample`` reads each row's state from the input array;
        # the kernel's stored ``state_`` is not consulted, so we skip
        # ``set_state`` on this hot path.
        kernel = self._get_trans_kernel(int(action))
        return np.asarray(kernel.batch_sample(states_array), dtype=np.float64)

    def observation_log_probability_per_state(
        self, next_states: Any, action: int, observation: Any
    ) -> np.ndarray:
        next_states_array = np.ascontiguousarray(np.asarray(next_states, dtype=np.float64))
        if next_states_array.ndim == 1:
            next_states_array = next_states_array.reshape(1, -1)
        # ``batch_log_likelihood`` reads each row's next-state from the
        # input array; ``next_state_`` on the kernel is unused, so we
        # skip ``set_next_state`` on this hot path.
        kernel = self._get_obs_kernel(int(action))
        observation_code = _OBS_STR_TO_CODE.get(observation, -1)
        return np.asarray(
            kernel.batch_log_likelihood(
                next_particles=next_states_array,
                observation=int(observation_code),
            ),
            dtype=np.float64,
        )

    def simulate_random_rollout(
        self,
        state: Any,
        action_sampler: Any,  # pylint: disable=unused-argument
        max_depth: int,
        discount_factor: float,
        depth: int = 0,
    ) -> float:
        """Random rollout via native C++ deterministic transition and reward kernel.

        The C++ kernel applies the variant-aware dangerous-area reward term
        directly, so no Python fallback is required when danger zones are
        configured.

        Args:
            state: Current RockSample state array.
            action_sampler: Object with a ``sample()`` method returning an
                integer action. Currently unused — actions are drawn
                uniformly by the native kernel.
            max_depth: Maximum rollout depth.
            discount_factor: Per-step discount factor.
            depth: Depth already consumed by the search tree. Defaults to 0.

        Returns:
            Discounted sum of immediate rewards along the sampled trajectory.
        """
        steps_left = max_depth - depth
        if steps_left <= 0:
            return 0.0

        state_arr = np.ascontiguousarray(np.asarray(state, dtype=np.float64).ravel())
        n_actions = len(self.action_names)
        action_indices = np.random.randint(0, n_actions, size=steps_left, dtype=np.int32)

        return _native.simulate_rollout_discrete(
            initial_state=state_arr,
            action_indices=action_indices,
            rock_positions_flat=self._rock_positions_flat,
            max_depth=max_depth,
            start_depth=depth,
            discount_factor=discount_factor,
            map_rows=int(self.map_size[0]),
            map_cols=int(self.map_size[1]),
            n_actions=n_actions,
            step_penalty=float(self.step_penalty),
            exit_reward=float(self.exit_reward),
            good_rock_reward=float(self.good_rock_reward),
            bad_rock_penalty=float(self.bad_rock_penalty),
            sensor_use_penalty=float(self.sensor_use_penalty),
            dangerous_areas=self._dangerous_areas_arr,
            dangerous_area_radius=float(self.dangerous_area_radius),
            dangerous_area_penalty=float(self.dangerous_area_penalty),
            dangerous_area_hit_probability=float(self.dangerous_area_hit_probability),
            reward_variant_code=int(self._reward_variant_code),
            penalty_decay=float(self.penalty_decay),
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

    def __getstate__(self) -> Dict[str, Any]:
        # Per-action C++ kernel caches hold pybind11 objects that aren't
        # picklable. Drop them at serialization time; ``__setstate__``
        # rebuilds empty caches so the env works after unpickling.
        state = self.__dict__.copy()
        state["_trans_kernel_cache"] = {}
        state["_obs_kernel_cache"] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)
        self._trans_kernel_cache = {}
        self._obs_kernel_cache = {}

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

    def hash_action(self, action: Any) -> Hashable:
        # Discrete int actions; already hashable.
        return action

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
