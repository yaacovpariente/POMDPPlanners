"""Continuous Push POMDP Environment Implementation.

This module implements a continuous-action variant of the Push POMDP where
the robot moves via 2D action vectors with Gaussian noise, has a configurable
radius, and obstacles are axis-aligned squares.

The Continuous Push POMDP features:
- Continuous 2D state space: [robot_x, robot_y, object_x, object_y, target_x, target_y]
- Continuous action space (2D movement vectors)
- Robot modelled as a circle with configurable radius
- Object modelled as a point
- Square obstacles defined as axis-aligned bounding boxes
- Gaussian transition noise on robot movement
- Capped push force with friction
- Noisy observations of object position

Classes:
    ContinuousPushPOMDP: Main environment with continuous actions.
    ContinuousPushPOMDPDiscreteActions: Discrete action wrapper.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    Environment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.push_pomdp import _native
from POMDPPlanners.environments.push_pomdp.continuous_push_geometry import (
    circle_aabb_overlap,
    point_inside_aabb,
)
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal
from POMDPPlanners.utils.statistics_utils import confidence_interval
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp_visualizer import (  # pylint: disable=import-outside-toplevel
    ContinuousPushPOMDPVisualizer,
)


class ContinuousPushPOMDPMetrics(Enum):
    """Metric names for Continuous Push POMDP environment."""

    GOAL_REACHING_RATE = "goal_reaching_rate"
    ROBOT_OBSTACLE_COLLISION_RATE = "robot_obstacle_collision_rate"
    OBJECT_OBSTACLE_COLLISION_RATE = "object_obstacle_collision_rate"
    TOTAL_OBSTACLE_COLLISION_RATE = "total_obstacle_collision_rate"
    TOTAL_ROBOT_OBSTACLE_COLLISIONS = "total_robot_obstacle_collisions"
    TOTAL_OBJECT_OBSTACLE_COLLISIONS = "total_object_obstacle_collisions"
    TOTAL_ALL_OBSTACLE_COLLISIONS = "total_all_obstacle_collisions"


class _FixedStateDistribution(Distribution):
    def __init__(self, state: np.ndarray):
        self._state = state.copy()

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        return [self._state.copy() for _ in range(n_samples)]


class _RandomInitialStateDistribution(Distribution):
    def __init__(self, parent: "ContinuousPushPOMDP"):
        self._parent = parent

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        states = []
        for _ in range(n_samples):
            robot = self._generate_robot_position()
            obj = self._generate_object_position()
            states.append(np.concatenate([robot, obj, self._parent.target_pos]))
        return states

    # pylint: disable=protected-access
    def _generate_robot_position(self) -> np.ndarray:
        p = self._parent
        for _ in range(100):
            pos = np.random.uniform(p.robot_radius, p.grid_size - 1 - p.robot_radius, size=2)
            if not p._is_circle_colliding_with_obstacle(pos, p.robot_radius):
                return pos
        return np.random.uniform(p.robot_radius, p.grid_size - 1 - p.robot_radius, size=2)

    def _generate_object_position(self) -> np.ndarray:
        p = self._parent
        for _ in range(100):
            pos = np.random.uniform(0, p.grid_size - 1, size=2)
            far_from_target = np.linalg.norm(pos - p.target_pos) >= 2.0
            if far_from_target and not p._is_point_colliding_with_obstacle(pos):
                return pos
        return np.random.uniform(0, p.grid_size - 1, size=2)

    # pylint: enable=protected-access


class ContinuousPushPOMDP(Environment):
    """Continuous-action Push POMDP environment.

    A robot (circle) must push an object (point) to a target location on
    a 2D grid.  The robot moves via continuous 2D action vectors with
    Gaussian noise; obstacles are axis-aligned squares.

    State: [robot_x, robot_y, object_x, object_y, target_x, target_y]
    Actions: 2D numpy vectors
    Observations: [robot_x, robot_y, noisy_obj_x, noisy_obj_y, target_x, target_y]

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>>
        >>> env = ContinuousPushPOMDP(discount_factor=0.99)
        >>>
        >>> initial_state = env.initial_state_dist().sample()[0]
        >>>
        >>> action = np.array([1.0, 0.0])
        >>> next_state, observation, reward = env.sample_next_step(initial_state, action)
        >>>
        >>> env.is_terminal(initial_state)
        False
    """

    def __init__(
        self,
        discount_factor: float,
        grid_size: int = 10,
        push_threshold: float = 1.0,
        friction_coefficient: float = 0.3,
        max_push: float = 2.0,
        observation_noise: float = 0.1,
        obstacles: Optional[List[Tuple[float, float, float]]] = None,
        obstacle_penalty: float = -10.0,
        robot_radius: float = 0.3,
        state_transition_cov_matrix: np.ndarray = np.eye(2) * 0.1,
        initial_state: Optional[np.ndarray] = None,
        name: str = "ContinuousPushPOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        self.grid_size = grid_size
        self.push_threshold = push_threshold
        self.friction_coefficient = friction_coefficient
        self.max_push = max_push
        self.observation_noise = observation_noise
        self.obstacle_penalty = obstacle_penalty
        self.robot_radius = robot_radius
        self.state_transition_cov_matrix = state_transition_cov_matrix
        self._initial_state = initial_state

        self._obstacle_tuples: List[Tuple[float, float, float]] = (
            obstacles if obstacles is not None else []
        )
        self.obstacles = self._build_obstacle_array(self._obstacle_tuples)

        self.target_pos = np.array([grid_size - 1.0, grid_size - 1.0])

        self._state_transition_dist = CovarianceParameterizedMultivariateNormal(
            state_transition_cov_matrix
        )

        # Snapshot covariance buffer without copying for the per-action
        # kernel cache below — the Cholesky lives inside the C++ kernel
        # and never sees this array again after construction.
        self._trans_cov_view = self._state_transition_dist.covariance_view()

        # Per-action kernel caches: one C++ kernel per distinct action
        # vector. Hot-path overrides flip the (next_)state field via
        # set_state / set_next_state instead of rebuilding (skips the
        # per-call Cholesky and obstacle-buffer copy).
        self._trans_kernel_cache: Dict[bytes, Any] = {}
        self._obs_kernel_cache: Dict[bytes, Any] = {}

        space_info = SpaceInfo(
            action_space=SpaceType.CONTINUOUS,
            observation_space=SpaceType.CONTINUOUS,
        )
        max_distance = np.sqrt(2) * (grid_size - 1)
        min_reward = -max_distance + self.obstacle_penalty
        max_reward = 100.0

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=(min_reward, max_reward),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

    def _build_obstacle_array(
        self, obstacle_tuples: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        if not obstacle_tuples:
            return np.empty((0, 4))
        rows = []
        for cx, cy, half_size in obstacle_tuples:
            rows.append([cx, cy, half_size, half_size])
        return np.array(rows, dtype=float)

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    # ── Hot-path sampling overrides ─────────────────────────────────
    # The default base-class implementations build a Python-wrapper
    # subclass per call (np.asarray(...).ravel() x2, side-attribute
    # storage). The actual RNG draw lives entirely inside the C++
    # _native.ContinuousPush{Transition,Observation}Cpp.sample() method.
    # These overrides skip the Python subclass, fetch a cached per-action
    # C++ kernel (Cholesky factored once, obstacle buffer copied once),
    # and rewrite only the (next_)state field per call via set_state /
    # set_next_state. The C++ RNG state lives on the kernel, so each
    # cached kernel maintains a single RNG stream per (env, action).

    def _get_trans_kernel(self, action: np.ndarray) -> Any:
        action_arr = np.ascontiguousarray(np.asarray(action, dtype=np.float64))
        key = action_arr.tobytes()
        kernel = self._trans_kernel_cache.get(key)
        if kernel is None:
            kernel = _native.ContinuousPushTransitionCpp(
                state=np.zeros(6, dtype=np.float64),
                action=action_arr,
                grid_size=float(self.grid_size),
                push_threshold=float(self.push_threshold),
                friction_coefficient=float(self.friction_coefficient),
                max_push=float(self.max_push),
                robot_radius=float(self.robot_radius),
                obstacles=self.obstacles,
                covariance=self._trans_cov_view,
            )
            self._trans_kernel_cache[key] = kernel
        return kernel

    def _get_obs_kernel(self, action: np.ndarray) -> Any:
        action_arr = np.ascontiguousarray(np.asarray(action, dtype=np.float64))
        key = action_arr.tobytes()
        kernel = self._obs_kernel_cache.get(key)
        if kernel is None:
            kernel = _native.ContinuousPushObservationCpp(
                next_state=np.zeros(6, dtype=np.float64),
                action=action_arr,
                observation_noise=float(self.observation_noise),
                grid_size=float(self.grid_size),
            )
            self._obs_kernel_cache[key] = kernel
        return kernel

    def sample_next_state(self, state: np.ndarray, action: np.ndarray, n_samples: int = 1) -> Any:
        kernel = self._get_trans_kernel(action)
        kernel.set_state(state)
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def sample_observation(
        self, next_state: np.ndarray, action: np.ndarray, n_samples: int = 1
    ) -> Any:
        kernel = self._get_obs_kernel(action)
        kernel.set_next_state(next_state)
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def transition_log_probability(
        self, state: np.ndarray, action: np.ndarray, next_states: Any
    ) -> np.ndarray:
        kernel = self._get_trans_kernel(action)
        kernel.set_state(state)
        probs = np.asarray(kernel.probability(next_states))
        with np.errstate(divide="ignore"):
            return np.log(probs)

    def observation_log_probability(
        self, next_state: np.ndarray, action: np.ndarray, observations: Any
    ) -> np.ndarray:
        kernel = self._get_obs_kernel(action)
        kernel.set_next_state(next_state)
        # batch_log_likelihood is symmetric in the object-position slice:
        # passing the candidate observations as ``next_particles`` and the
        # fixed next_state as ``observation`` yields the per-row log-pdf of
        # each observation given next_state.
        obs_arr = np.asarray(observations, dtype=float)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        return np.asarray(
            kernel.batch_log_likelihood(
                next_particles=np.ascontiguousarray(obs_arr, dtype=np.float64),
                observation=np.ascontiguousarray(np.asarray(next_state, dtype=np.float64)),
            )
        )

    def sample_next_state_batch(self, states: Any, action: np.ndarray) -> np.ndarray:
        states_array = np.ascontiguousarray(np.asarray(states, dtype=np.float64))
        if states_array.ndim == 1:
            states_array = states_array.reshape(1, -1)
        kernel = self._get_trans_kernel(action)
        # batch_sample reads the per-row state from the input, not the
        # kernel's stored state, so no set_state is needed here.
        return np.asarray(kernel.batch_sample(states_array), dtype=np.float64)

    def observation_log_probability_per_state(
        self, next_states: Any, action: np.ndarray, observation: Any
    ) -> np.ndarray:
        next_states_array = np.ascontiguousarray(np.asarray(next_states, dtype=np.float64))
        if next_states_array.ndim == 1:
            next_states_array = next_states_array.reshape(1, -1)
        observation_array = np.ascontiguousarray(np.asarray(observation, dtype=np.float64))
        kernel = self._get_obs_kernel(action)
        # batch_log_likelihood reads next_state per row from the input;
        # no set_next_state needed.
        return np.asarray(
            kernel.batch_log_likelihood(
                next_particles=next_states_array,
                observation=observation_array,
            ),
            dtype=np.float64,
        )

    def reward(self, state: np.ndarray, action: np.ndarray) -> float:
        # Single-state reward routes through the same vectorised path used
        # by reward_batch — wrap the state as a (1, 6) row, reuse the
        # cached kernel, and unpack the scalar.
        state_arr = np.ascontiguousarray(np.asarray(state, dtype=np.float64)).reshape(1, -1)
        action_arr = np.ascontiguousarray(np.asarray(action, dtype=np.float64)).ravel()
        rewards = self._reward_batch_array(state_arr, action_arr)
        return float(rewards[0])

    def reward_batch(
        self, states: Union[np.ndarray, Sequence[Any]], action: np.ndarray
    ) -> np.ndarray:
        states_arr = np.ascontiguousarray(np.asarray(states, dtype=np.float64))
        if states_arr.ndim == 1:
            states_arr = states_arr.reshape(1, -1)
        action_arr = np.ascontiguousarray(np.asarray(action, dtype=np.float64)).ravel()
        return self._reward_batch_array(states_arr, action_arr)

    def _reward_batch_array(self, states: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Vectorised reward computation reusing the cached transition kernel.

        ``states`` must be ``(N, 6)`` C-contiguous float64 and ``action``
        must be a 1-D float64 array; callers (``reward`` / ``reward_batch``)
        normalise the inputs.

        For each row this draws a fresh next state via the cached kernel
        (same RNG stream as ``sample_next_state_batch``) and computes:
        - distance penalty to target,
        - +100 goal bonus when within 0.5 of the target,
        - obstacle penalty when ``state[:2] + action`` would push the
          robot into an obstacle.
        """
        kernel = self._get_trans_kernel(action)
        # batch_sample reads per-row state from the input — no set_state
        # required. C++ kernel returns shape (N, 6) float64.
        next_states = np.asarray(kernel.batch_sample(states), dtype=np.float64)

        # Distance-to-target penalty + goal bonus, computed in one pass on
        # the (N, 2) object/target slices.
        delta = next_states[:, 2:4] - next_states[:, 4:6]
        dist_to_target = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        rewards = -dist_to_target
        rewards[dist_to_target < 0.5] += 100.0

        # Obstacle penalty: post-action robot position vs. obstacle AABBs.
        if self.obstacles.shape[0] > 0:
            robot_after = states[:, :2] + action  # (N, 2)
            collide = self._batch_circle_obstacle_overlap(robot_after, self.robot_radius)
            if np.any(collide):
                rewards[collide] += self.obstacle_penalty

        return rewards

    def _batch_circle_obstacle_overlap(self, positions: np.ndarray, radius: float) -> np.ndarray:
        """Vectorised circle-vs-AABB overlap test against ``self.obstacles``.

        ``positions`` is shape ``(N, 2)``. Returns an ``(N,)`` boolean mask
        flagging rows that overlap any obstacle. Mirrors the per-wall logic
        of :func:`continuous_push_geometry.circle_aabb_overlap` but
        broadcast across all walls in one shot.
        """
        walls = self.obstacles
        # Broadcast positions (N, 1, 2) against walls (M, 4): closest point
        # on each AABB to each position, then squared distance < r**2.
        cx = walls[:, 0]
        cy = walls[:, 1]
        hx = walls[:, 2]
        hy = walls[:, 3]
        px = positions[:, 0:1]
        py = positions[:, 1:2]
        closest_x = np.clip(px, cx - hx, cx + hx)
        closest_y = np.clip(py, cy - hy, cy + hy)
        dx = px - closest_x
        dy = py - closest_y
        dist_sq = dx * dx + dy * dy
        return np.any(dist_sq < radius * radius, axis=1)

    def __getstate__(self):
        # Per-action C++ kernel cache holds pybind11 objects that aren't
        # picklable. Drop them at serialization time; __setstate__ rebuilds
        # empty caches so the env works after unpickling.
        state = self.__dict__.copy()
        state["_trans_kernel_cache"] = {}
        state["_obs_kernel_cache"] = {}
        return state

    def __setstate__(self, state):
        vars(self).update(state)
        self._trans_kernel_cache = {}
        self._obs_kernel_cache = {}

    def is_terminal(self, state: np.ndarray) -> bool:
        obj = state[2:4]
        target = state[4:6]
        return bool(np.linalg.norm(obj - target) < 0.5)

    def initial_state_dist(self) -> Distribution:
        if self._initial_state is not None:
            return _FixedStateDistribution(self._initial_state)
        return _RandomInitialStateDistribution(self)

    def initial_observation_dist(self) -> Distribution:
        return self.initial_state_dist()

    def is_equal_observation(self, observation1: np.ndarray, observation2: np.ndarray) -> bool:
        return bool(np.array_equal(observation1, observation2))

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache animated visualization of the continuous push episode.

        Creates an animated GIF showing the robot pushing the object toward
        the target, with rectangular obstacles, collision detection, distance
        indicators, and success feedback.

        Args:
            history: Episode history containing states, actions, and rewards.
            cache_path: Path where to save the visualization (must end with .gif).

        Raises:
            ValueError: If history is empty or cache_path doesn't end with .gif.
            TypeError: If cache_path is not a Path object.
        """

        visualizer = ContinuousPushPOMDPVisualizer(self)
        visualizer.create_visualization(history, cache_path)

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    def _is_circle_colliding_with_obstacle(self, pos: np.ndarray, radius: float) -> bool:
        if self.obstacles.shape[0] == 0:
            return False
        for i in range(self.obstacles.shape[0]):
            if circle_aabb_overlap(pos, radius, self.obstacles[i]):
                return True
        return False

    def _is_point_colliding_with_obstacle(self, pos: np.ndarray) -> bool:
        if self.obstacles.shape[0] == 0:
            return False
        for i in range(self.obstacles.shape[0]):
            if point_inside_aabb(pos, self.obstacles[i]):
                return True
        return False

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metric_names(self) -> List[str]:
        """Get names of Continuous Push POMDP specific metrics.

        Returns:
            List of metric name strings.
        """
        return [m.value for m in ContinuousPushPOMDPMetrics]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        goal_reached_list: List[int] = []
        robot_col_list: List[int] = []
        obj_col_list: List[int] = []
        total_col_list: List[int] = []

        for history in histories:
            goal_hit = False
            r_cols = 0
            o_cols = 0

            for step in history.history:
                if self.is_terminal(step.state):
                    goal_hit = True
                if self._is_circle_colliding_with_obstacle(step.state[:2], self.robot_radius):
                    r_cols += 1
                if self._is_point_colliding_with_obstacle(step.state[2:4]):
                    o_cols += 1

            goal_reached_list.append(1 if goal_hit else 0)
            robot_col_list.append(r_cols)
            obj_col_list.append(o_cols)
            total_col_list.append(r_cols + o_cols)

        total_steps = sum(len(h.history) for h in histories)
        avg_r = sum(robot_col_list) / total_steps if total_steps > 0 else 0
        avg_o = sum(obj_col_list) / total_steps if total_steps > 0 else 0
        avg_t = sum(total_col_list) / total_steps if total_steps > 0 else 0

        r_rates = [c / len(h.history) for c, h in zip(robot_col_list, histories) if len(h.history)]
        o_rates = [c / len(h.history) for c, h in zip(obj_col_list, histories) if len(h.history)]
        t_rates = [c / len(h.history) for c, h in zip(total_col_list, histories) if len(h.history)]

        r_ci = confidence_interval(data=r_rates, confidence=0.95) if r_rates else (0, 0)
        o_ci = confidence_interval(data=o_rates, confidence=0.95) if o_rates else (0, 0)
        t_ci = confidence_interval(data=t_rates, confidence=0.95) if t_rates else (0, 0)

        tr_ci = (
            confidence_interval(data=robot_col_list, confidence=0.95) if robot_col_list else (0, 0)
        )
        to_ci = confidence_interval(data=obj_col_list, confidence=0.95) if obj_col_list else (0, 0)
        ta_ci = (
            confidence_interval(data=total_col_list, confidence=0.95) if total_col_list else (0, 0)
        )

        avg_goal = float(np.mean(goal_reached_list)) if goal_reached_list else 0.0
        g_ci = (
            confidence_interval(data=goal_reached_list, confidence=0.95)
            if goal_reached_list
            else (0, 0)
        )

        return [
            MetricValue(
                ContinuousPushPOMDPMetrics.GOAL_REACHING_RATE.value, avg_goal, g_ci[0], g_ci[1]
            ),
            MetricValue(
                ContinuousPushPOMDPMetrics.ROBOT_OBSTACLE_COLLISION_RATE.value,
                avg_r,
                r_ci[0],
                r_ci[1],
            ),
            MetricValue(
                ContinuousPushPOMDPMetrics.OBJECT_OBSTACLE_COLLISION_RATE.value,
                avg_o,
                o_ci[0],
                o_ci[1],
            ),
            MetricValue(
                ContinuousPushPOMDPMetrics.TOTAL_OBSTACLE_COLLISION_RATE.value,
                avg_t,
                t_ci[0],
                t_ci[1],
            ),
            MetricValue(
                ContinuousPushPOMDPMetrics.TOTAL_ROBOT_OBSTACLE_COLLISIONS.value,
                sum(robot_col_list),
                tr_ci[0],
                tr_ci[1],
            ),
            MetricValue(
                ContinuousPushPOMDPMetrics.TOTAL_OBJECT_OBSTACLE_COLLISIONS.value,
                sum(obj_col_list),
                to_ci[0],
                to_ci[1],
            ),
            MetricValue(
                ContinuousPushPOMDPMetrics.TOTAL_ALL_OBSTACLE_COLLISIONS.value,
                sum(total_col_list),
                ta_ci[0],
                ta_ci[1],
            ),
        ]


class ContinuousPushPOMDPDiscreteActions(ContinuousPushPOMDP, DiscreteActionsEnvironment):
    """Discrete-action wrapper for the Continuous Push POMDP.

    Maps string actions ``["up", "down", "right", "left"]`` to unit
    vectors and delegates to the continuous parent.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>>
        >>> env = ContinuousPushPOMDPDiscreteActions(discount_factor=0.99)
        >>>
        >>> initial_state = env.initial_state_dist().sample()[0]
        >>> actions = env.get_actions()
        >>>
        >>> action = actions[0]
        >>> next_state, observation, reward = env.sample_next_step(initial_state, action)
        >>>
        >>> env.is_terminal(initial_state)
        False
    """

    def __init__(
        self,
        discount_factor: float,
        grid_size: int = 10,
        push_threshold: float = 1.0,
        friction_coefficient: float = 0.3,
        max_push: float = 2.0,
        observation_noise: float = 0.1,
        obstacles: Optional[List[Tuple[float, float, float]]] = None,
        obstacle_penalty: float = -10.0,
        robot_radius: float = 0.3,
        state_transition_cov_matrix: np.ndarray = np.eye(2) * 0.1,
        initial_state: Optional[np.ndarray] = None,
        name: str = "ContinuousPushPOMDPDiscreteActions",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        super().__init__(
            discount_factor=discount_factor,
            grid_size=grid_size,
            push_threshold=push_threshold,
            friction_coefficient=friction_coefficient,
            max_push=max_push,
            observation_noise=observation_noise,
            obstacles=obstacles,
            obstacle_penalty=obstacle_penalty,
            robot_radius=robot_radius,
            state_transition_cov_matrix=state_transition_cov_matrix,
            initial_state=initial_state,
            name=name,
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.CONTINUOUS,
        )

        self.actions = ["up", "down", "right", "left"]
        # Cache action vectors as contiguous float64 ndarrays so the hot
        # path can skip np.asarray(...).ravel() conversions.
        self.action_to_vector = {
            "up": np.ascontiguousarray([0.0, 1.0], dtype=np.float64),
            "down": np.ascontiguousarray([0.0, -1.0], dtype=np.float64),
            "right": np.ascontiguousarray([1.0, 0.0], dtype=np.float64),
            "left": np.ascontiguousarray([-1.0, 0.0], dtype=np.float64),
        }

    def get_actions(self) -> List[str]:
        return self.actions

    def reward(self, state: np.ndarray, action: Any) -> float:
        if isinstance(action, str):
            action = self.action_to_vector[action]
        return super().reward(state, action)

    def reward_batch(self, states: Union[np.ndarray, Sequence[Any]], action: Any) -> np.ndarray:
        if isinstance(action, str):
            action = self.action_to_vector[action]
        return super().reward_batch(states, action)

    def sample_next_state(self, state: np.ndarray, action: Any, n_samples: int = 1) -> Any:
        return super().sample_next_state(state, self.action_to_vector[action], n_samples=n_samples)

    def sample_observation(self, next_state: np.ndarray, action: Any, n_samples: int = 1) -> Any:
        return super().sample_observation(
            next_state, self.action_to_vector[action], n_samples=n_samples
        )

    def transition_log_probability(
        self, state: np.ndarray, action: Any, next_states: Any
    ) -> np.ndarray:
        return super().transition_log_probability(state, self.action_to_vector[action], next_states)

    def observation_log_probability(
        self, next_state: np.ndarray, action: Any, observations: Any
    ) -> np.ndarray:
        return super().observation_log_probability(
            next_state, self.action_to_vector[action], observations
        )

    def sample_next_state_batch(self, states: Any, action: Any) -> np.ndarray:
        return super().sample_next_state_batch(states, self.action_to_vector[action])

    def observation_log_probability_per_state(
        self, next_states: Any, action: Any, observation: Any
    ) -> np.ndarray:
        return super().observation_log_probability_per_state(
            next_states, self.action_to_vector[action], observation
        )
