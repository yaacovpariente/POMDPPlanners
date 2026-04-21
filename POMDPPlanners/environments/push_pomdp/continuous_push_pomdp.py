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
    ContinuousPushStateTransitionModel: Continuous movement with noise and push.
    ContinuousPushObservationModel: Noisy object position observations.
    ContinuousPushPOMDP: Main environment with continuous actions.
    ContinuousPushPOMDPDiscreteActions: Discrete action wrapper.
"""

from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    Environment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
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


class ContinuousPushStateTransitionModel(_native.ContinuousPushTransitionCpp):
    """State transition model for Continuous Push POMDP.

    Implements continuous robot movement with Gaussian noise and capped
    push mechanics.  The robot is a circle; the object is a point.  Both
    interact with axis-aligned square obstacles.

    State representation: [robot_x, robot_y, object_x, object_y, target_x, target_y]

    The ``sample()``, ``probability()`` and ``batch_sample()`` methods
    execute entirely in C++ via the ``_native`` extension; this Python
    subclass only wraps the constructor so existing call sites that pass a
    :class:`CovarianceParameterizedMultivariateNormal` keep working.

    Attributes:
        state: Current state vector.
        action: 2D movement vector.
        grid_size: Grid dimension.
        push_threshold: Maximum robot-object distance for a push.
        friction_coefficient: Friction reducing push force (0-1).
        max_push: Maximum push magnitude.
        obstacles: Shape ``(M, 4)`` AABB array.
        robot_radius: Robot body radius.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> state = np.array([2.0, 3.0, 2.5, 3.1, 8.0, 8.0])
        >>> action = np.array([1.0, 0.0])
        >>> cov = np.eye(2) * 0.01
        >>> dist = CovarianceParameterizedMultivariateNormal(cov)
        >>> transition = ContinuousPushStateTransitionModel(
        ...     state=state, action=action,
        ...     state_transition_dist=dist, grid_size=10,
        ...     push_threshold=1.0, friction_coefficient=0.3,
        ...     max_push=2.0, obstacles=np.empty((0, 4)),
        ...     robot_radius=0.3,
        ... )
        >>> next_state = transition.sample()[0]
        >>> len(next_state) == 6
        True
    """

    def __init__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        state_transition_dist: CovarianceParameterizedMultivariateNormal,
        grid_size: float,
        push_threshold: float,
        friction_coefficient: float,
        max_push: float,
        obstacles: np.ndarray,
        robot_radius: float,
    ):
        action_arr = np.asarray(action, dtype=float).ravel()
        super().__init__(
            state=np.asarray(state, dtype=float).ravel(),
            action=action_arr,
            grid_size=float(grid_size),
            push_threshold=float(push_threshold),
            friction_coefficient=float(friction_coefficient),
            max_push=float(max_push),
            robot_radius=float(robot_radius),
            obstacles=np.asarray(obstacles, dtype=float),
            covariance=state_transition_dist.covariance,
        )
        # Python-visible attributes match the pre-port contract. The C++
        # base exposes ``state`` / ``action`` read-only properties; other
        # attributes live here only for downstream introspection.
        self._state_transition_dist = state_transition_dist
        self.grid_size = float(grid_size)
        self.push_threshold = float(push_threshold)
        self.friction_coefficient = float(friction_coefficient)
        self.max_push = float(max_push)
        self.obstacles = np.asarray(obstacles, dtype=float)
        self.robot_radius = float(robot_radius)

        self.robot_pos = np.asarray(state, dtype=float)[:2].copy()
        self.object_pos = np.asarray(state, dtype=float)[2:4].copy()
        self.target_pos = np.asarray(state, dtype=float)[4:6].copy()


StateTransitionModel.register(ContinuousPushStateTransitionModel)


class ContinuousPushObservationModel(_native.ContinuousPushObservationCpp):
    """Noisy observation model for Continuous Push POMDP.

    Robot and target positions are observed exactly; object position is
    observed with additive Gaussian noise.

    Observation format: [robot_x, robot_y, noisy_obj_x, noisy_obj_y, target_x, target_y]

    The ``sample()``, ``probability()`` and ``batch_log_likelihood()``
    methods execute entirely in C++ via the ``_native`` extension.

    Attributes:
        next_state: True state after action.
        action: Action taken (2D vector).
        observation_noise: Standard deviation of object position noise.
        grid_size: Grid dimension.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> state = np.array([3.0, 3.0, 2.8, 3.2, 8.0, 8.0])
        >>> action = np.array([1.0, 0.0])
        >>> obs_model = ContinuousPushObservationModel(
        ...     next_state=state, action=action,
        ...     observation_noise=0.1, grid_size=10,
        ... )
        >>> obs = obs_model.sample()[0]
        >>> len(obs) == 6
        True
        >>> bool(obs[0] == 3.0)  # Robot exact
        True
    """

    def __init__(
        self,
        next_state: np.ndarray,
        action: np.ndarray,
        observation_noise: float,
        grid_size: float,
    ):
        super().__init__(
            next_state=np.asarray(next_state, dtype=float).ravel(),
            action=np.asarray(action, dtype=float).ravel(),
            observation_noise=float(observation_noise),
            grid_size=float(grid_size),
        )
        self.observation_noise = float(observation_noise)
        self.grid_size = float(grid_size)
        self.robot_pos = np.asarray(next_state, dtype=float)[:2]
        self.object_pos = np.asarray(next_state, dtype=float)[2:4]
        self.target_pos = np.asarray(next_state, dtype=float)[4:6]


ObservationModel.register(ContinuousPushObservationModel)


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

    def state_transition_model(self, state: np.ndarray, action: np.ndarray) -> StateTransitionModel:
        return ContinuousPushStateTransitionModel(  # pyright: ignore[reportReturnType]
            state=state,
            action=np.asarray(action, dtype=float),
            state_transition_dist=self._state_transition_dist,
            grid_size=self.grid_size,
            push_threshold=self.push_threshold,
            friction_coefficient=self.friction_coefficient,
            max_push=self.max_push,
            obstacles=self.obstacles,
            robot_radius=self.robot_radius,
        )

    def observation_model(self, next_state: np.ndarray, action: np.ndarray) -> ObservationModel:
        return ContinuousPushObservationModel(  # pyright: ignore[reportReturnType]
            next_state=next_state,
            action=np.asarray(action, dtype=float),
            observation_noise=self.observation_noise,
            grid_size=self.grid_size,
        )

    def reward(self, state: np.ndarray, action: np.ndarray) -> float:
        action = np.asarray(action, dtype=float)
        next_state = self._sample_transition(state, action)
        return self._compute_reward_from_next_state(state, next_state, action)

    def reward_batch(
        self, states: Union[np.ndarray, Sequence[Any]], action: np.ndarray
    ) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        states_arr = np.asarray(states, dtype=float)
        rewards = np.empty(states_arr.shape[0])
        for i in range(states_arr.shape[0]):
            rewards[i] = self.reward(states_arr[i], action)
        return rewards

    def _sample_transition(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return ContinuousPushStateTransitionModel(
            state=state,
            action=action,
            state_transition_dist=self._state_transition_dist,
            grid_size=self.grid_size,
            push_threshold=self.push_threshold,
            friction_coefficient=self.friction_coefficient,
            max_push=self.max_push,
            obstacles=self.obstacles,
            robot_radius=self.robot_radius,
        ).sample()[0]

    def _compute_reward_from_next_state(
        self, state: np.ndarray, next_state: np.ndarray, action: np.ndarray
    ) -> float:
        next_obj = next_state[2:4]
        target = next_state[4:6]
        dist_to_target = float(np.linalg.norm(next_obj - target))
        rew = -dist_to_target
        if dist_to_target < 0.5:
            rew += 100.0
        if self._is_circle_colliding_with_obstacle(state[:2] + action, self.robot_radius):
            rew += self.obstacle_penalty
        return rew

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
        self.action_to_vector = {
            "up": np.array([0.0, 1.0]),
            "down": np.array([0.0, -1.0]),
            "right": np.array([1.0, 0.0]),
            "left": np.array([-1.0, 0.0]),
        }

    def get_actions(self) -> List[str]:
        return self.actions

    def state_transition_model(self, state: np.ndarray, action: Any) -> StateTransitionModel:
        return super().state_transition_model(state, self.action_to_vector[action])

    def observation_model(self, next_state: np.ndarray, action: Any) -> ObservationModel:
        return super().observation_model(next_state, self.action_to_vector[action])

    def reward(self, state: np.ndarray, action: Any) -> float:
        if isinstance(action, str):
            action = self.action_to_vector[action]
        return super().reward(state, action)

    def reward_batch(self, states: Union[np.ndarray, Sequence[Any]], action: Any) -> np.ndarray:
        if isinstance(action, str):
            action = self.action_to_vector[action]
        return super().reward_batch(np.asarray(states), action)
