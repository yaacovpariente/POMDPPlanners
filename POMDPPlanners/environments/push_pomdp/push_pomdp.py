# SPDX-License-Identifier: MIT

# pylint: disable=too-many-lines
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
    PushPOMDP: Main push task environment with POMDP formulation
"""

import math
from enum import Enum
from pathlib import Path
from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.core.simulation.step_info_metrics import (
    EpisodeReduction,
    StepInfoMetric,
    extract_episode_step_infos,
    order_and_fill_metrics,
)
from POMDPPlanners.environments.push_pomdp import _native
from POMDPPlanners.environments.push_pomdp.push_pomdp_utils.push_reward_models import (
    BasePushRewardModel,
    DiscretePushDistanceDecayedHazardPenaltyRewardModel,
    DiscretePushZeroMeanHazardShockRewardModel,
    DiscretePushRewardModel,
    RewardModelType,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp_visualizer import PushPOMDPVisualizer
from POMDPPlanners.utils.statistics_utils import confidence_interval


class PushStepChannel(Enum):
    """Per-step measurement channels reported by :meth:`PushPOMDP.step_info`."""

    GOAL_REACHED = "goal_reached"
    ROBOT_OBSTACLE_COLLISION = "robot_obstacle_collision"
    OBJECT_OBSTACLE_COLLISION = "object_obstacle_collision"
    ANY_OBSTACLE_COLLISION = "any_obstacle_collision"
    IN_DANGEROUS_AREA = "in_dangerous_area"


class PushPOMDPMetrics(Enum):
    """Metric names for Push POMDP environment."""

    GOAL_REACHING_RATE = "goal_reaching_rate"
    ROBOT_OBSTACLE_COLLISION_RATE = "robot_obstacle_collision_rate"
    OBJECT_OBSTACLE_COLLISION_RATE = "object_obstacle_collision_rate"
    TOTAL_OBSTACLE_COLLISION_RATE = "total_obstacle_collision_rate"
    TOTAL_ROBOT_OBSTACLE_COLLISIONS = "total_robot_obstacle_collisions"
    TOTAL_OBJECT_OBSTACLE_COLLISIONS = "total_object_obstacle_collisions"
    TOTAL_ALL_OBSTACLE_COLLISIONS = "total_all_obstacle_collisions"
    DANGEROUS_AREA_RATE = "dangerous_area_rate"
    TOTAL_DANGEROUS_AREA_STEPS = "total_dangerous_area_steps"


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
            parts = [robot_pos, object_pos, self.target_pos]
            if self.parent.hazard_terminal_enabled:
                # Hazard-terminal envs carry a trailing live (0.0) slot.
                parts.append(np.zeros(1))
            initial_state = np.concatenate(parts)
            initial_states.append(initial_state)
        return initial_states

    def _generate_robot_position(self) -> np.ndarray:
        max_attempts = 100
        for _ in range(max_attempts):
            robot_pos = np.random.uniform(0, self.grid_size - 1, size=2)
            # pylint: disable-next=protected-access
            if not self.parent._is_colliding_with_obstacle(robot_pos):
                return robot_pos
        return np.random.uniform(0, self.grid_size - 1, size=2)

    def _generate_object_position(self) -> np.ndarray:
        max_attempts = 100
        for _ in range(max_attempts):
            object_pos = np.random.uniform(0, self.grid_size - 1, size=2)
            # pylint: disable-next=protected-access
            colliding = self.parent._is_colliding_with_obstacle(object_pos)
            if np.linalg.norm(object_pos - self.target_pos) >= 2.0 and not colliding:
                return object_pos
        return np.random.uniform(0, self.grid_size - 1, size=2)


class PushPOMDP(DiscreteActionsEnvironment):  # pylint: disable=too-many-public-methods
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

    Stochasticity:
        The obstacle-collision penalty can be applied either
        deterministically (the default) or stochastically. When
        ``obstacle_hit_probability == 1.0`` (default), the penalty is
        applied every time the robot's intended next position lies inside
        an obstacle, matching legacy behavior. When
        ``obstacle_hit_probability < 1.0``, the penalty is applied only
        with that probability per ``reward()`` / ``reward_batch()`` call
        (one Bernoulli draw per state), producing a heavy-tailed return
        distribution suitable for benchmarking risk-sensitive planners
        (e.g. ICVaR-aware MCTS) against expected-value MCTS on the same
        env. Note that this makes ``reward(state, action)`` non-
        deterministic given a state-action pair, so any external caching
        that assumes deterministic rewards must be aware of this.
        ``transition_log_probability`` is unaffected; the obstacle still
        deterministically blocks movement. The native C++ rollout applies
        the Bernoulli ``obstacle_hit_probability`` draw internally, so
        ``simulate_random_rollout`` always routes through the native
        kernel.

    Dangerous areas:
        ``dangerous_areas`` is a separate, additive concept from
        ``obstacles``. Each entry is a circular region centred at
        ``(x, y)`` with radius ``dangerous_area_radius``. Entering a
        dangerous area applies ``dangerous_area_penalty`` (a negative
        number, added to reward — same sign convention as
        ``obstacle_penalty``) but does NOT block movement. Penalty fires
        when the robot's intended next position lies inside any
        dangerous area; the object position is ignored. At most one
        ``dangerous_area_penalty`` is applied per step even when
        multiple zones overlap. Like obstacles, the penalty supports a
        Bernoulli ``dangerous_area_hit_probability`` (default 1.0) for
        risk-sensitive planning. The native C++ rollout applies the
        Bernoulli draw internally, so all rollouts route through the
        native kernel regardless of the configured probability.

    Hazard termination (draw-coupled):
        With ``is_dangerous_area_hit_terminal=True`` the dangerous-area
        hazard both terminates the episode and pays its penalty on the
        SAME draw. The state gains a trailing terminal slot, becoming
        7-D ``[rx, ry, ox, oy, tx, ty, terminal]`` with ``terminal``
        ``0.0`` (live) / ``1.0`` (terminated); an already-terminal state
        is absorbing (echoed verbatim by the transition). The
        ``dangerous_area_hit_probability`` Bernoulli is drawn inside the
        C++ transition kernel against the realised post-action robot
        position, so ``reward`` becomes DETERMINISTIC given the realised
        ``next_state`` — hence :attr:`reward_requires_next_state` is
        ``True`` and drivers must sample the transition first and pass
        ``next_state`` to :meth:`reward`. Reaching the goal wins over the
        hazard (the slot stays ``0.0`` at the goal). Obstacles are NOT
        coupled to termination and keep their legacy uncoupled Bernoulli
        penalty. The flag is incompatible with
        ``RewardModelType.ZERO_MEAN_HAZARD_SHOCK`` (that variant has no
        hit probability to couple termination to). With the flag off
        (the default) the state stays 6-D and every code path is
        unchanged.

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

    # Class-level action -> (dx, dy) offset table. Shared across instances
    # and referenced by both the deterministic next-state helper and the
    # obstacle-collision check.
    _ACTION_TO_DXY = {
        "up": (0, 1),
        "down": (0, -1),
        "right": (1, 0),
        "left": (-1, 0),
    }

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        discount_factor: float,
        grid_size: int = 10,
        push_threshold: float = 1.0,
        friction_coefficient: float = 0.3,
        observation_noise: float = 0.1,
        obstacles: Optional[List[Tuple[float, float]]] = None,
        obstacle_radius: float = 0.5,
        obstacle_penalty: float = -10.0,
        obstacle_hit_probability: float = 1.0,
        dangerous_areas: Optional[List[Tuple[float, float]]] = None,
        dangerous_area_radius: float = 0.5,
        dangerous_area_penalty: float = -10.0,
        dangerous_area_hit_probability: float = 1.0,
        reward_model_type: RewardModelType = RewardModelType.CONSTANT_HAZARD_PENALTY,
        penalty_decay: float = 1.0,
        initial_state: Optional[np.ndarray] = None,
        transition_error_prob: float = 0.0,
        is_dangerous_area_hit_terminal: bool = False,
        name: str = "PushPOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        if not 0.0 <= obstacle_hit_probability <= 1.0:
            raise ValueError("obstacle_hit_probability must be between 0 and 1 (inclusive)")
        if not 0.0 <= dangerous_area_hit_probability <= 1.0:
            raise ValueError("dangerous_area_hit_probability must be between 0 and 1 (inclusive)")
        if (
            reward_model_type == RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY
            and penalty_decay <= 0.0
        ):
            raise ValueError("penalty_decay must be strictly positive")
        self._configure_hazard_terminal(is_dangerous_area_hit_terminal, reward_model_type)

        self.grid_size = grid_size
        self.push_threshold = push_threshold
        self.friction_coefficient = friction_coefficient
        self.observation_noise = observation_noise
        self.obstacles: List[Tuple[float, float]] = obstacles if obstacles is not None else []
        self.obstacle_radius = obstacle_radius
        self.obstacle_penalty = obstacle_penalty
        self.obstacle_hit_probability = float(obstacle_hit_probability)
        self.dangerous_areas: List[Tuple[float, float]] = (
            list(dangerous_areas) if dangerous_areas is not None else []
        )
        self.dangerous_area_radius = float(dangerous_area_radius)
        self.dangerous_area_penalty = float(dangerous_area_penalty)
        self.dangerous_area_hit_probability = float(dangerous_area_hit_probability)
        self.reward_model_type = reward_model_type
        self.penalty_decay = float(penalty_decay)
        self._initial_state = initial_state
        self.transition_error_prob = transition_error_prob

        # Cached constants for the scalar observation log-probability fast-path
        # used by POMCPOW's WeightedParticleBeliefStateUpdate.inplace_update.
        self._obs_variance = float(observation_noise) * float(observation_noise)
        self._obs_log_norm = -math.log(2.0 * math.pi * self._obs_variance)

        # Define actions
        self.actions = ["up", "down", "right", "left"]

        # Initialize target position (fixed)
        self.target_pos = np.array([grid_size - 1, grid_size - 1])

        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # Action space is discrete positions
            observation_space=SpaceType.CONTINUOUS,  # Observation space is positions with noise
        )
        # Calculate reward range based on maximum distance to target plus the
        # additive obstacle and dangerous-area penalties when configured.
        # Without those terms, any robot action that drives the robot into a
        # penalised region produces a reward strictly more negative than the
        # advertised lower bound. Maximum distance is diagonal from corner to
        # corner: sqrt(2) * (grid_size - 1).
        max_distance = np.sqrt(2) * (grid_size - 1)
        if self.dangerous_areas:
            if reward_model_type == RewardModelType.ZERO_MEAN_HAZARD_SHOCK:
                danger_term_min = -abs(dangerous_area_penalty)
                danger_term_max = abs(dangerous_area_penalty)
            else:
                danger_term_min = min(0.0, dangerous_area_penalty)
                danger_term_max = max(0.0, dangerous_area_penalty)
        else:
            danger_term_min = 0.0
            danger_term_max = 0.0
        min_reward = (
            -max_distance + min(0.0, obstacle_penalty if self.obstacles else 0.0) + danger_term_min
        )
        max_reward = 100.0 + danger_term_max

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

        # Pre-built per-action (dx, dy) C-contiguous float64 buffers; one
        # array per label, shared across all transition kernel calls so the
        # cached C++ kernel doesn't need to repack on every dispatch.
        self._action_dxdy_map: Dict[str, np.ndarray] = {
            label: np.array([float(dx), float(dy)], dtype=np.float64)
            for label, (dx, dy) in self._ACTION_TO_DXY.items()
        }

        # Precomputed error-action list per action label. Avoids per-call
        # list comprehension + np.random.choice over a Python list (~5 µs)
        # in _sample_one_next_state's error branch.
        self._error_actions_for: Dict[str, List[str]] = {
            a: [b for b in self.actions if b != a] for a in self.actions
        }

        # Flat (M*2,) float64 buffer of obstacle centres; shared across all
        # cached kernels (they each copy it at construction).
        if self.obstacles:
            self._obstacles_flat_arr: np.ndarray = np.asarray(
                self.obstacles, dtype=np.float64
            ).ravel()
        else:
            self._obstacles_flat_arr = np.empty((0,), dtype=np.float64)

        # (K, 2) float64 buffer of dangerous-area centres for the native
        # rollout kernel. Empty (0, 2) when no dangerous areas configured.
        if self.dangerous_areas:
            self._dangerous_areas_arr: np.ndarray = np.ascontiguousarray(
                np.asarray(self.dangerous_areas, dtype=np.float64).reshape(-1, 2)
            )
        else:
            self._dangerous_areas_arr = np.empty((0, 2), dtype=np.float64)

        # Per-action C++ transition kernel cache. Built lazily on first
        # dispatch so the env survives pickling without bundling pybind11
        # objects.
        self._trans_kernel_cache: Dict[str, Any] = {}

        self.reward_model: BasePushRewardModel = self._build_reward_model()
        # Cache the bound methods so the hot path skips ``self.reward_model``
        # attribute lookup on each call (~50–100 ns saved per call). Bound
        # methods pickle correctly via ``(class, name, instance)``, so
        # __setstate__ doesn't need to rebuild them.
        # Hazard-terminal envs route the scalar reward through the native
        # batch kernel too: with the flag on the dangerous term is read off
        # the realised terminal slot (deterministic, no RNG), which the
        # Python reward model does not model. Flag-off keeps the historical
        # bound-method fast path byte-for-byte.
        self._compute_reward = (
            self._native_reward_scalar
            if self._hazard_terminal_enabled
            else self.reward_model.compute_reward
        )
        self._compute_reward_batch = self.reward_model.compute_reward_batch

    def _configure_hazard_terminal(
        self,
        is_dangerous_area_hit_terminal: bool,
        reward_model_type: RewardModelType,
    ) -> None:
        if (
            is_dangerous_area_hit_terminal
            and reward_model_type == RewardModelType.ZERO_MEAN_HAZARD_SHOCK
        ):
            raise ValueError(
                "is_dangerous_area_hit_terminal is incompatible with the "
                "ZERO_MEAN_HAZARD_SHOCK reward model (the shock hazard has no "
                "hit probability to couple termination to)"
            )
        self.is_dangerous_area_hit_terminal = bool(is_dangerous_area_hit_terminal)
        # When the hazard-terminal flag is enabled the state gains a LAST
        # terminal-flag slot (0.0 live / 1.0 terminal); otherwise the state
        # keeps the historical 6-D layout untouched.
        self._hazard_terminal_enabled = self.is_dangerous_area_hit_terminal
        self._state_dim = 7 if self._hazard_terminal_enabled else 6

    @property
    def hazard_terminal_enabled(self) -> bool:
        """Whether the hazard-terminal flag is on (state carries a terminal slot)."""
        return self._hazard_terminal_enabled

    @property
    def reward_requires_next_state(self) -> bool:
        """Reward depends on the realised ``next_state`` iff the hazard flag is on.

        When enabled, the hazard penalty is deterministic given the terminal
        slot set during the transition, so drivers must sample the transition
        first and pass the realised ``next_state`` to :meth:`reward`.
        """
        return self._hazard_terminal_enabled

    def _native_reward_scalar(self, state: Any, action: str, next_state: Any) -> float:
        # Scalar entry point for hazard-terminal envs: reshape to (1, D) and
        # reuse the native batch kernel, which reads the terminal slot.
        states_array = np.ascontiguousarray(state, dtype=np.float64).reshape(1, -1)
        next_states_arr = np.ascontiguousarray(next_state, dtype=np.float64).reshape(1, -1)
        return float(self._native_reward_batch(states_array, action, next_states_arr)[0])

    def _build_reward_model(self) -> BasePushRewardModel:
        # ``Dict[str, Any]`` opt-out is required so pyright doesn't narrow
        # the value type to the lub of obstacles + scalars and reject the
        # ``**`` unpack as incompatible with each reward-model __init__.
        common_kwargs: Dict[str, Any] = {
            "obstacles": self.obstacles,
            "obstacle_radius": self.obstacle_radius,
            "obstacle_penalty": self.obstacle_penalty,
            "obstacle_hit_probability": self.obstacle_hit_probability,
            "dangerous_areas": self.dangerous_areas,
            "dangerous_areas_arr": self._dangerous_areas_arr,
            "dangerous_area_radius": self.dangerous_area_radius,
            "dangerous_area_penalty": self.dangerous_area_penalty,
            "dangerous_area_hit_probability": self.dangerous_area_hit_probability,
        }
        if self.reward_model_type == RewardModelType.CONSTANT_HAZARD_PENALTY:
            return DiscretePushRewardModel(**common_kwargs)
        if self.reward_model_type == RewardModelType.ZERO_MEAN_HAZARD_SHOCK:
            return DiscretePushZeroMeanHazardShockRewardModel(**common_kwargs)
        if self.reward_model_type == RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY:
            return DiscretePushDistanceDecayedHazardPenaltyRewardModel(
                penalty_decay=self.penalty_decay, **common_kwargs
            )
        raise ValueError(f"Unknown reward model type: {self.reward_model_type}")

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
            dx, dy = self._ACTION_TO_DXY[action]
            check_x = float(position[0]) + dx
            check_y = float(position[1]) + dy
        else:
            check_x = float(position[0])
            check_y = float(position[1])

        obs_r_sq = self.obstacle_radius * self.obstacle_radius
        for obs_x, obs_y in self.obstacles:
            ddx = check_x - obs_x
            ddy = check_y - obs_y
            if ddx * ddx + ddy * ddy <= obs_r_sq:
                return True

        return False

    def sample_next_step(self, state: Any, action: Any) -> Tuple[Any, Any, float]:
        next_state = self.sample_next_state(state=state, action=action)
        next_observation = self.sample_observation(next_state=next_state, action=action)
        r = self._compute_reward(state, action, next_state)
        return next_state, next_observation, r

    # ── Env-API sampling implementations ────────────────────────────
    # These methods inline the per-call physics (push, friction, obstacle
    # collision, grid clipping) and the per-call RNG draws so callers
    # never need to allocate a per-(state, action) wrapper object.

    def sample_next_state(self, state: np.ndarray, action: str, n_samples: int = 1) -> Any:
        if n_samples == 1:
            return self._sample_one_next_state(state, action)
        samples: List[np.ndarray] = []
        for _ in range(n_samples):
            samples.append(self._sample_one_next_state(state, action))
        return samples

    def _sample_one_next_state(self, state: np.ndarray, action: str) -> np.ndarray:
        # RNG order: one np.random.random() call, optionally one
        # np.random.randint() call indexing into the precomputed error list.
        # randint+index is ~3.5x faster than np.random.choice over a Python list.
        if self.transition_error_prob > 0.0 and np.random.random() < self.transition_error_prob:
            error_actions = self._error_actions_for[action]
            actual_action = error_actions[np.random.randint(len(error_actions))]
        else:
            actual_action = action
        return self._compute_next_state_for_action(state, actual_action)

    def _get_trans_kernel(self, action: str) -> Any:
        # Per-action native transition kernel cache. Each kernel freezes the
        # (dx, dy) for ``action`` and copies obstacles_flat once at
        # construction; per-call work is just set_state + compute_next_state.
        cached = self._trans_kernel_cache.get(action)
        if cached is not None:
            return cached
        action_dxdy = self._action_dxdy_map[action]
        if self._hazard_terminal_enabled:
            # Hazard-terminal: 7-D kernel that also draws the terminal slot
            # for the realised next robot position on the C++ RNG.
            variant_code, penalty_decay = self._reward_variant_native_params()
            kernel = _native.PushDiscreteTransitionCpp(
                state=np.zeros(7, dtype=np.float64),
                action_dxdy=action_dxdy,
                grid_size=float(self.grid_size),
                push_threshold=float(self.push_threshold),
                friction_coefficient=float(self.friction_coefficient),
                obstacles_flat=self._obstacles_flat_arr,
                n_obstacles=len(self.obstacles),
                obstacle_radius=float(self.obstacle_radius),
                dangerous_areas=self._dangerous_areas_arr,
                dangerous_area_radius=float(self.dangerous_area_radius),
                dangerous_area_hit_probability=float(self.dangerous_area_hit_probability),
                reward_variant_code=variant_code,
                penalty_decay=penalty_decay,
                is_dangerous_area_hit_terminal=True,
            )
        else:
            kernel = _native.PushDiscreteTransitionCpp(
                state=np.zeros(6, dtype=np.float64),
                action_dxdy=action_dxdy,
                grid_size=float(self.grid_size),
                push_threshold=float(self.push_threshold),
                friction_coefficient=float(self.friction_coefficient),
                obstacles_flat=self._obstacles_flat_arr,
                n_obstacles=len(self.obstacles),
                obstacle_radius=float(self.obstacle_radius),
            )
        self._trans_kernel_cache[action] = kernel
        return kernel

    def _compute_next_state_for_action(self, state: np.ndarray, action: str) -> np.ndarray:
        # Native dispatch path: route the closed-form deterministic transition
        # through the cached PushDiscreteTransitionCpp kernel. Used by both
        # the sampling path (after error-action selection) and the closed-form
        # transition_log_probability path.
        kernel = self._get_trans_kernel(action)
        kernel.set_state(np.ascontiguousarray(state, dtype=np.float64))
        return np.asarray(kernel.compute_next_state())

    def _compute_next_state_for_action_python(self, state: np.ndarray, action: str) -> np.ndarray:
        # Pure-Python reference implementation kept for the parity test.
        dx, dy = self._ACTION_TO_DXY[action]

        # Extract scalar positions from the 6-D state.
        rx, ry = float(state[0]), float(state[1])
        ox, oy = float(state[2]), float(state[3])
        tx, ty = float(state[4]), float(state[5])

        # Intended new robot position; obstacle collision blocks movement.
        irx, iry = rx + dx, ry + dy
        if self._is_colliding_with_obstacle_scalar(irx, iry):
            nrx, nry = rx, ry
        else:
            nrx, nry = irx, iry

        # Push if robot is within push_threshold of the object.
        ddx, ddy = nrx - ox, nry - oy
        dist_sq = ddx * ddx + ddy * ddy
        push_threshold_sq = self.push_threshold * self.push_threshold

        if dist_sq < push_threshold_sq:
            push_scale = 1.0 - self.friction_coefficient
            iox = ox + dx * push_scale
            ioy = oy + dy * push_scale
            if self._is_colliding_with_obstacle_scalar(iox, ioy):
                nox, noy = ox, oy
            else:
                nox, noy = iox, ioy
        else:
            nox, noy = ox, oy

        # Clip to grid bounds.
        gmax = self.grid_size - 1
        nrx = max(0.0, min(nrx, gmax))
        nry = max(0.0, min(nry, gmax))
        nox = max(0.0, min(nox, gmax))
        noy = max(0.0, min(noy, gmax))

        result = np.empty(6)
        result[0] = nrx
        result[1] = nry
        result[2] = nox
        result[3] = noy
        result[4] = tx
        result[5] = ty
        return result

    def _is_colliding_with_obstacle_scalar(self, pos_x: float, pos_y: float) -> bool:
        if not self.obstacles:
            return False
        obs_r_sq = self.obstacle_radius * self.obstacle_radius
        for obs_x, obs_y in self.obstacles:
            ddx = pos_x - obs_x
            ddy = pos_y - obs_y
            if ddx * ddx + ddy * ddy <= obs_r_sq:
                return True
        return False

    def _is_in_dangerous_area(self, position: np.ndarray, action: Optional[str] = None) -> bool:
        """Check if ``position`` (optionally after ``action``) lies in any dangerous area."""
        if not self.dangerous_areas:
            return False
        if action is not None:
            dx, dy = self._ACTION_TO_DXY[action]
            check_x = float(position[0]) + dx
            check_y = float(position[1]) + dy
        else:
            check_x = float(position[0])
            check_y = float(position[1])
        return self._is_in_dangerous_area_scalar(check_x, check_y)

    def _is_in_dangerous_area_scalar(self, pos_x: float, pos_y: float) -> bool:
        if not self.dangerous_areas:
            return False
        r_sq = self.dangerous_area_radius * self.dangerous_area_radius
        for danger_x, danger_y in self.dangerous_areas:
            ddx = pos_x - danger_x
            ddy = pos_y - danger_y
            if ddx * ddx + ddy * ddy <= r_sq:
                return True
        return False

    def sample_observation(self, next_state: np.ndarray, action: str, n_samples: int = 1) -> Any:
        if n_samples == 1:
            return self._sample_one_observation(next_state)
        samples: List[np.ndarray] = []
        for _ in range(n_samples):
            samples.append(self._sample_one_observation(next_state))
        return samples

    def _sample_one_observation(self, next_state: np.ndarray) -> np.ndarray:
        # Two Gaussian draws (object-x, object-y), clamped to [0, grid_size - 1].
        # Robot and target slices are observed exactly. Single size=2 draw beats
        # two scalar np.random.normal calls by avoiding dispatch overhead.
        gmax = self.grid_size - 1
        rx, ry = float(next_state[0]), float(next_state[1])
        ox, oy = float(next_state[2]), float(next_state[3])
        tx, ty = float(next_state[4]), float(next_state[5])

        noise = np.random.normal(0.0, self.observation_noise, size=2)
        nox = max(0.0, min(ox + float(noise[0]), gmax))
        noy = max(0.0, min(oy + float(noise[1]), gmax))

        observation = np.empty(next_state.shape[0])
        observation[0] = rx
        observation[1] = ry
        observation[2] = nox
        observation[3] = noy
        observation[4] = tx
        observation[5] = ty
        if next_state.shape[0] > 6:
            # Hazard-terminal slot is observed exactly (copied verbatim).
            observation[6] = float(next_state[6])
        return observation

    def transition_log_probability(
        self, state: np.ndarray, action: str, next_states: Any
    ) -> np.ndarray:
        # Closed-form discrete probability:
        #   P(next | s, a) = (1 - p_err) * 1[next == intended(s, a)]
        #                  + p_err * (1 / n_err) * sum_a' 1[next == intended(s, a')]
        # where the second sum runs over error_actions = actions \ {a}.
        # Single cached kernel handles every action label: set_state once,
        # compute_next_state for the intended action and
        # compute_next_state_for_action(other_dxdy) for the error branch.
        kernel = self._get_trans_kernel(action)
        kernel.set_state(np.ascontiguousarray(state, dtype=np.float64))
        if self._hazard_terminal_enabled:
            # Flag-on: ``compute_next_state`` would also draw the hazard slot
            # on the C++ RNG. A probability query must not perturb the RNG
            # stream, so use the deterministic 6-D geometry entry point.
            intended_next = np.asarray(
                kernel.compute_next_state_for_action(self._action_dxdy_map[action])
            )
        else:
            intended_next = np.asarray(kernel.compute_next_state())
        error_actions = [a for a in self.actions if a != action]
        num_error_actions = len(error_actions)
        error_results: List[np.ndarray] = []
        if self.transition_error_prob > 0.0 and num_error_actions > 0:
            error_results = [
                np.asarray(
                    kernel.compute_next_state_for_action(self._action_dxdy_map[error_action])
                )
                for error_action in error_actions
            ]

        # Hazard-terminal envs carry a 7th terminal slot whose draw probability
        # is intentionally NOT modelled here (same as ``ContinuousPushPOMDP``):
        # the closed form covers only the deterministic movement geometry, so
        # the slot is marginalised out by comparing the first six entries.
        matches = (
            (lambda c, ref: np.array_equal(np.asarray(c)[:6], np.asarray(ref)[:6]))
            if self._hazard_terminal_enabled
            else np.array_equal
        )

        probabilities = np.empty(len(next_states), dtype=float)
        for i, candidate in enumerate(next_states):
            prob_intended = 1.0 if matches(candidate, intended_next) else 0.0
            prob_error = 0.0
            if error_results:
                error_match_count = sum(1 for er in error_results if matches(candidate, er))
                prob_error = (
                    self.transition_error_prob
                    * (1.0 / num_error_actions)
                    * float(error_match_count)
                )
            total = (1.0 - self.transition_error_prob) * prob_intended + prob_error
            probabilities[i] = total

        with np.errstate(divide="ignore"):
            return np.log(probabilities)

    def observation_log_probability(
        self, next_state: np.ndarray, action: str, observations: Any
    ) -> np.ndarray:
        # Closed-form 2-D Gaussian log-pdf on the object-position slice
        # (cols 2:4) against next_state[2:4]. Robot/target dims are observed
        # exactly so they don't affect the likelihood.
        variance = self.observation_noise * self.observation_noise
        log_norm = -float(np.log(2.0 * np.pi * variance))
        obs_arr = np.asarray(observations, dtype=float)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        diffs = obs_arr[:, 2:4] - np.asarray(next_state, dtype=float)[2:4]
        sq = np.sum(diffs * diffs, axis=1)
        return log_norm - 0.5 * sq / variance

    def observation_log_probability_single(
        self, next_state: Any, action: Any, observation: Any
    ) -> float:
        # Scalar fast-path used by POMCPOW's incremental belief update.
        # Same 2-D Gaussian on object position (cols 2:4) as the batched path
        # above, but skips numpy array allocation per call. Cached
        # ``_obs_variance`` and ``_obs_log_norm`` are set in __init__.
        del action  # unused; obs noise is action-independent for this env
        dx = float(observation[2]) - float(next_state[2])
        dy = float(observation[3]) - float(next_state[3])
        return self._obs_log_norm - 0.5 * (dx * dx + dy * dy) / self._obs_variance

    def reward(self, state: np.ndarray, action: str, next_state: Any = None) -> float:
        # Use the realised ``next_state`` when supplied (e.g. by
        # :meth:`Environment.sample_next_step`) so the obstacle/danger
        # penalty fires against the same draw as the trajectory; otherwise
        # sample a fresh next state to evaluate the action result.
        if next_state is None:
            next_state = self.sample_next_state(state, action)
        return self._compute_reward(state, action, next_state)

    def is_terminal(self, state: np.ndarray) -> bool:
        # Absorbing terminal slot (hazard-terminal envs only), then the
        # historical goal check.
        if len(state) > 6 and float(state[6]) > 0.5:
            return True
        return self._object_at_goal(state)

    @staticmethod
    def _object_at_goal(state: Any) -> bool:
        # Episode ends when object is close to target. Goal-only predicate:
        # deliberately ignores the hazard-terminal slot so metrics can tell a
        # goal arrival from a hazard death.
        dx = float(state[2] - state[4])
        dy = float(state[3] - state[5])
        return dx * dx + dy * dy < 0.25  # 0.5^2 = 0.25

    def initial_state_dist(self) -> Distribution:
        # If a fixed initial state is provided, return a deterministic distribution
        if self._initial_state is not None:
            return FixedStateDistribution(self._pad_terminal_slot(self._initial_state))

        return RandomInitialStateDistribution(
            self.grid_size, self.target_pos, self.obstacles, self.obstacle_radius, self
        )

    def _pad_terminal_slot(self, state: np.ndarray) -> np.ndarray:
        # Append a live (0.0) terminal slot to a caller-provided 6-D state
        # when the env carries the slot; pass a 7-D state through unchanged.
        arr = np.asarray(state, dtype=np.float64)
        if self._hazard_terminal_enabled and arr.shape[0] == 6:
            return np.concatenate([arr, np.zeros(1)])
        return arr

    def initial_observation_dist(self) -> Distribution:
        return self.initial_state_dist()

    def get_actions(self) -> List[str]:
        return self.actions

    def is_equal_observation(self, observation1: np.ndarray, observation2: np.ndarray) -> bool:
        return np.array_equal(observation1, observation2)

    def hash_observation(self, observation: Any) -> Hashable:
        # ndarray observations are unhashable; ``tobytes()`` is consistent with
        # ``np.array_equal`` for fixed-shape/dtype observations of this env.
        return np.ascontiguousarray(observation).tobytes()

    def hash_action(self, action: Any) -> Hashable:
        # Discrete-action env: actions are str labels (e.g. "up").
        return action

    def __getstate__(self) -> Dict[str, Any]:
        # Per-action C++ kernel cache holds pybind11 objects that are not
        # picklable. Drop them at serialization time; ``__setstate__``
        # rebuilds an empty cache on the receiving end.
        state = self.__dict__.copy()
        state["_trans_kernel_cache"] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)
        self._trans_kernel_cache = {}

    def _get_native_rollout_obstacles(self) -> np.ndarray:
        # Returns (M, 2) float64 array of obstacle centres; cached on first call.
        cached = getattr(self, "_cached_native_rollout_obstacles", None)
        if cached is not None:
            return cached  # type: ignore[return-value]
        if self.obstacles:
            obs_arr: np.ndarray = np.array(self.obstacles, dtype=np.float64)
        else:
            obs_arr = np.empty((0,), dtype=np.float64)
        # pylint: disable=attribute-defined-outside-init
        self._cached_native_rollout_obstacles: np.ndarray = obs_arr
        return obs_arr

    def simulate_random_rollout(
        self,
        state: Any,
        action_sampler: Any,
        max_depth: int,
        discount_factor: float,
        depth: int = 0,
    ) -> float:
        # Fast path: delegate the entire rollout to the C++ kernel.
        # Pre-draw action indices in Python so np's random stream (used by
        # belief updates etc.) remains the caller's random stream. ``_native``
        # is imported at module top.
        del action_sampler  # rollouts use the module-level np RNG directly
        remaining = max_depth - depth
        if remaining <= 0 or self.is_terminal(state=state):
            return 0.0

        # Native kernel is variant-aware and scores obstacle / dangerous-area
        # penalties against the REALISED post-action robot position
        # (``next_state[:2]``), matching ``DiscretePushRewardModel``. Pass the
        # variant code + penalty decay so the C++ rollout dispatches
        # CONSTANT_HAZARD_PENALTY / ZERO_MEAN_HAZARD_SHOCK / DISTANCE_DECAYED_HAZARD_PENALTY exactly
        # like the standalone ``push_reward_batch`` kernel.
        action_indices = np.random.randint(0, 4, size=remaining, dtype=np.int64)
        obs_arr = self._get_native_rollout_obstacles()
        state_arr = np.asarray(state, dtype=np.float64)
        variant_code, penalty_decay = self._reward_variant_native_params()

        return float(
            _native.simulate_rollout_discrete(
                state=state_arr,
                action_indices=action_indices,
                max_depth=max_depth,
                depth=depth,
                discount=discount_factor,
                grid_size=float(self.grid_size),
                push_threshold=float(self.push_threshold),
                friction_coefficient=float(self.friction_coefficient),
                obstacles=obs_arr,
                obstacle_radius=float(self.obstacle_radius),
                obstacle_penalty=float(self.obstacle_penalty),
                dangerous_areas=self._dangerous_areas_arr,
                dangerous_area_radius=float(self.dangerous_area_radius),
                dangerous_area_penalty=float(self.dangerous_area_penalty),
                transition_error_prob=float(self.transition_error_prob),
                obstacle_hit_probability=float(self.obstacle_hit_probability),
                dangerous_area_hit_probability=float(self.dangerous_area_hit_probability),
                reward_variant_code=variant_code,
                penalty_decay=penalty_decay,
                is_dangerous_area_hit_terminal=self.is_dangerous_area_hit_terminal,
            )
        )

    def _python_simulate_random_rollout(
        self,
        state: Any,
        actions: List[str],
        max_depth: int,
        discount_factor: float,
        depth: int = 0,
    ) -> float:
        # Pure-Python reference rollout: mirrors the pre-native logic but
        # accepts a pre-drawn action list (to allow exact comparison with the
        # C++ path when transition_error_prob=0 and the actions are held fixed).
        sample_one = self._sample_one_next_state
        reward_from_next = self._compute_reward
        is_terminal = self.is_terminal

        total = 0.0
        gamma_power = 1.0
        current = state
        cursor = 0
        while depth < max_depth and not is_terminal(state=current):
            if cursor >= len(actions):
                break
            action = actions[cursor]
            cursor += 1
            next_state = sample_one(current, action)
            r = reward_from_next(current, action, next_state)
            total += gamma_power * r
            current = next_state
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
        from POMDPPlanners.environments.push_pomdp.push_pomdp_beliefs.push_vectorized_updater import (
            PushVectorizedUpdater,
        )

        cached = PushVectorizedUpdater.from_environment(self)
        # pylint: disable=attribute-defined-outside-init
        self._cached_vectorized_updater = cached
        return cached

    def sample_next_state_batch(self, states: Any, action: str) -> np.ndarray:
        states_array = np.ascontiguousarray(np.asarray(states, dtype=float))
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        return self._get_vectorized_updater().batch_transition(states_array, action)

    def reward_batch(
        self,
        states: Any,
        action: str,
        next_states: Any = None,
    ) -> np.ndarray:
        """Calculate rewards for a batch of states given a single action.

        When ``next_states`` is supplied (e.g. by a caller that has
        already sampled the realised batch transition), it is used
        directly; otherwise N next states are drawn here via the cached
        ``PushVectorizedUpdater``. Per-particle rewards are computed in
        the C++ ``push_reward_batch`` kernel (variant-aware: CONSTANT_HAZARD_PENALTY,
        ZERO_MEAN_HAZARD_SHOCK, DISTANCE_DECAYED_HAZARD_PENALTY) so the batch
        cost is a single round-trip into native code.
        """
        states_array = np.ascontiguousarray(np.asarray(states, dtype=float))
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        if next_states is None:
            next_states_arr = self._get_vectorized_updater().batch_transition(states_array, action)
        else:
            next_states_arr = np.ascontiguousarray(np.asarray(next_states, dtype=float))
            if next_states_arr.ndim == 1:
                next_states_arr = next_states_arr.reshape(1, -1)
        return self._native_reward_batch(states_array, action, next_states_arr)

    def _native_reward_batch(
        self,
        states_array: np.ndarray,
        action: str,
        next_states_arr: np.ndarray,
    ) -> np.ndarray:
        obstacles_arr = self._get_native_rollout_obstacles()
        if obstacles_arr.ndim == 1 and obstacles_arr.shape[0] == 0:
            obstacles_native = np.empty((0, 2), dtype=np.float64)
        else:
            obstacles_native = np.ascontiguousarray(obstacles_arr, dtype=np.float64)
        variant_code, penalty_decay = self._reward_variant_native_params()
        return np.asarray(
            _native.push_reward_batch(
                states=states_array,
                action_idx=int(self.actions.index(action)),
                next_states=next_states_arr,
                obstacles=obstacles_native,
                obstacle_radius=float(self.obstacle_radius),
                obstacle_penalty=float(self.obstacle_penalty),
                obstacle_hit_probability=float(self.obstacle_hit_probability),
                dangerous_areas=self._dangerous_areas_arr,
                dangerous_area_radius=float(self.dangerous_area_radius),
                dangerous_area_penalty=float(self.dangerous_area_penalty),
                dangerous_area_hit_probability=float(self.dangerous_area_hit_probability),
                reward_variant_code=variant_code,
                penalty_decay=penalty_decay,
                is_dangerous_area_hit_terminal=self.is_dangerous_area_hit_terminal,
            ),
            dtype=np.float64,
        )

    def _reward_variant_native_params(self) -> Tuple[int, float]:
        if self.reward_model_type == RewardModelType.ZERO_MEAN_HAZARD_SHOCK:
            return 1, 0.0
        if self.reward_model_type == RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY:
            return 2, float(getattr(self.reward_model, "penalty_decay", 1.0))
        return 0, 0.0

    def observation_log_probability_per_state(
        self, next_states: Any, action: str, observation: Any
    ) -> np.ndarray:
        next_states_arr = np.ascontiguousarray(np.asarray(next_states, dtype=float))
        if next_states_arr.ndim == 1:
            next_states_arr = next_states_arr.reshape(1, -1)
        # The closed-form 2-D Gaussian on cols 2:4 doesn't depend on action,
        # so we inline it directly rather than going through the updater
        # (which uses CovarianceParameterizedMultivariateNormal). Keeps a
        # single source of truth for the noise model.
        variance = self.observation_noise * self.observation_noise
        log_norm = -float(np.log(2.0 * np.pi * variance))
        obs_obj = np.asarray(observation, dtype=float).ravel()[2:4]
        diffs = next_states_arr[:, 2:4] - obs_obj[None, :]
        sq = np.sum(diffs * diffs, axis=1)
        return log_norm - 0.5 * sq / variance

    def cache_visualization(
        self, history: List[StepData], output_dir: Path, episode_index: int
    ) -> None:
        """Cache animated visualization of the push episode.

        Creates an animated GIF showing the robot pushing the object toward the target,
        with obstacles, collision detection, distance indicators, and success feedback.

        Args:
            history: Episode history containing states, actions, and rewards
            output_dir: Directory into which the ``.gif`` visualization is written
            episode_index: Zero-based episode index, used to name the file

        Raises:
            ValueError: If history is empty
        """
        cache_path = output_dir / f"agent_path_{episode_index}.gif"
        visualizer = PushPOMDPVisualizer(self)
        visualizer.create_visualization(history, cache_path)

    def get_metric_names(self) -> List[str]:
        """Get names of Push POMDP specific metrics.

        Returns:
            List containing collision-related metric names
        """
        return [metric.value for metric in PushPOMDPMetrics]

    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Report obstacle contact and hazard exposure for this step's state.

        Args:
            state: The state the step was taken from, or the final state on the
                terminal step.
            action: Unused; every channel here is a property of the state.
            next_state: Unused; each state is scored when it is recorded.

        Returns:
            The robot / object collision indicators, their sum, and the
            dangerous-area indicator.
        """
        del action, next_state
        robot_pos = state[:2]
        object_pos = state[2:4]
        robot_collision = float(self._is_colliding_with_obstacle(robot_pos))
        object_collision = float(self._is_colliding_with_obstacle(object_pos))
        return {
            # Goal-only: a hazard-terminated state (7-D slot set) is NOT a goal
            # arrival and must not inflate GOAL_REACHING_RATE.
            PushStepChannel.GOAL_REACHED.value: float(self._object_at_goal(state)),
            PushStepChannel.ROBOT_OBSTACLE_COLLISION.value: robot_collision,
            PushStepChannel.OBJECT_OBSTACLE_COLLISION.value: object_collision,
            # Own channel: the aggregator reduces one channel at a time and
            # cannot add two. A step where both collide counts twice, as before.
            PushStepChannel.ANY_OBSTACLE_COLLISION.value: robot_collision + object_collision,
            PushStepChannel.IN_DANGEROUS_AREA.value: float(self._is_in_dangerous_area(robot_pos)),
        }

    def get_metric_specs(self) -> List[StepInfoMetric]:
        """Declare the Push metrics derived from the per-step channels.

        Returns:
            Specs for the goal rate and the four per-episode totals. The four
            ``*_rate`` metrics are pooled across episodes and cannot be expressed
            as a per-episode reduction, so :meth:`compute_metrics` produces them.
        """
        return [
            StepInfoMetric(
                name=PushPOMDPMetrics.GOAL_REACHING_RATE.value,
                channel=PushStepChannel.GOAL_REACHED.value,
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=PushPOMDPMetrics.TOTAL_ROBOT_OBSTACLE_COLLISIONS.value,
                channel=PushStepChannel.ROBOT_OBSTACLE_COLLISION.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=PushPOMDPMetrics.TOTAL_OBJECT_OBSTACLE_COLLISIONS.value,
                channel=PushStepChannel.OBJECT_OBSTACLE_COLLISION.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=PushPOMDPMetrics.TOTAL_ALL_OBSTACLE_COLLISIONS.value,
                channel=PushStepChannel.ANY_OBSTACLE_COLLISION.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=PushPOMDPMetrics.TOTAL_DANGEROUS_AREA_STEPS.value,
                channel=PushStepChannel.IN_DANGEROUS_AREA.value,
                per_episode=EpisodeReduction.SUM,
            ),
        ]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute all nine Push metrics, in declaration order.

        Args:
            histories: List of simulation histories.

        Returns:
            One MetricValue per declared metric name, in declaration order.

        Raises:
            ValueError: If ``histories`` is empty, or if an episode ran without
                being measured. See
                :meth:`~POMDPPlanners.core.environment.environment.Environment.compute_metrics`.
        """
        computed = list(super().compute_metrics(histories)) + self._pooled_rate_metrics(histories)
        return order_and_fill_metrics(self.get_metric_names(), computed)

    def _pooled_rate_metrics(self, histories: List[History]) -> List[MetricValue]:
        # These four are pooled over all steps of all episodes -- sum(counts) /
        # sum(episode lengths) -- which is not the mean of per-episode values
        # unless every episode has the same length, so the shared aggregator
        # cannot express them. Their confidence intervals are computed from a
        # *different* statistic than their point estimate (per-episode rates vs.
        # a pooled ratio); that inconsistency predates this migration and is
        # preserved rather than quietly fixed.
        counts = self._per_episode_counts(histories)
        total_steps_all = sum(len(history.history) for history in histories)

        metrics: List[MetricValue] = []
        for name, key in (
            (PushPOMDPMetrics.ROBOT_OBSTACLE_COLLISION_RATE.value, "robot"),
            (PushPOMDPMetrics.OBJECT_OBSTACLE_COLLISION_RATE.value, "object"),
            (PushPOMDPMetrics.TOTAL_OBSTACLE_COLLISION_RATE.value, "total"),
            (PushPOMDPMetrics.DANGEROUS_AREA_RATE.value, "dangerous"),
        ):
            episode_counts = counts[key]
            value = sum(episode_counts) / total_steps_all if total_steps_all > 0 else 0
            # zip() against the full history list, not against the episodes that
            # actually contributed a count: a zero-step history is excluded from
            # episode_counts but not from histories, so the pairing shifts by one
            # from that point on. That is a pre-existing defect, kept verbatim
            # because fixing it here would change reported values.
            per_episode_rates = [
                count / len(history.history) for count, history in zip(episode_counts, histories)
            ]
            lower, upper = self._rate_interval(per_episode_rates)
            metrics.append(
                MetricValue(
                    name=name,
                    value=value,
                    lower_confidence_bound=lower,
                    upper_confidence_bound=upper,
                )
            )
        return metrics

    def _per_episode_counts(self, histories: List[History]) -> Dict[str, List[float]]:
        # Read back the same channels the specs consume, so a pooled rate and its
        # matching total can never be computed from different measurements.
        counts: Dict[str, List[float]] = {"robot": [], "object": [], "total": [], "dangerous": []}
        channels = {
            "robot": PushStepChannel.ROBOT_OBSTACLE_COLLISION.value,
            "object": PushStepChannel.OBJECT_OBSTACLE_COLLISION.value,
            "total": PushStepChannel.ANY_OBSTACLE_COLLISION.value,
            "dangerous": PushStepChannel.IN_DANGEROUS_AREA.value,
        }
        for episode in extract_episode_step_infos(histories):
            if not episode:
                # Historically a zero-step episode contributed no entry at all.
                continue
            for key, channel in channels.items():
                counts[key].append(sum(step_info.get(channel, 0.0) for step_info in episode))
        return counts

    @staticmethod
    def _rate_interval(rates: List[float]) -> Tuple[float, float]:
        if not rates:
            return (0, 0)
        return confidence_interval(data=rates, confidence=0.95)
