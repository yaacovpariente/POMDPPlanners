"""Continuous LaserTag POMDP Environment Implementation.

This module implements a continuous-space variant of the LaserTag
pursuit-evasion POMDP where a robot must navigate to tag an opponent that
moves stochastically through continuous 2-D space.

Two environment classes are provided:

* :class:`ContinuousLaserTagPOMDP` – continuous actions ``[dx, dy, tag_flag]``
* :class:`ContinuousLaserTagPOMDPDiscreteActions` – five string actions
  ``"up"``, ``"down"``, ``"right"``, ``"left"``, ``"tag"``

State representation:
    ``np.ndarray`` shape ``(5,)`` –
    ``[robot_x, robot_y, opponent_x, opponent_y, terminal_flag]``

Observation:
    ``np.ndarray`` shape ``(8,)`` – noisy 8-direction laser range
    measurements.  Terminal observation is ``np.full(8, -1.0)``.

Classes:
    ContinuousLaserTagStateTransitionModel: State transition model.
    ContinuousLaserTagObservationModel: Observation model.
    ContinuousLaserTagPOMDP: Continuous-action environment.
    ContinuousLaserTagPOMDPDiscreteActions: Discrete-action variant.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    Environment,
    SpaceInfo,
    SpaceType,
    ObservationModel,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.laser_tag_pomdp import _native
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_visualizer import (
    ContinuousLaserTagVisualizer,
)
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal
from POMDPPlanners.utils.statistics_utils import confidence_interval


# Default walls matching the discrete LaserTag grid, converted to AABBs
# Original wall cells (row, col) on an 11×7 grid with half-size 0.5
_DEFAULT_WALL_HALF_SIZE = 0.5
_DEFAULT_WALLS_CELLS = [
    (1, 2),
    (3, 0),
    (3, 4),
    (5, 0),
    (6, 4),
    (9, 1),
    (9, 4),
    (10, 6),
]


def _cells_to_aabbs(
    cells: Sequence[Tuple[float, ...]], half_size: float
) -> List[Tuple[float, float, float, float]]:
    return [(float(r), float(c), half_size, half_size) for r, c in cells]


_DEFAULT_WALLS = _cells_to_aabbs(_DEFAULT_WALLS_CELLS, _DEFAULT_WALL_HALF_SIZE)

_DEFAULT_DANGEROUS_AREAS: List[Tuple[float, float]] = [
    (5.0, 3.0),
    (7.0, 1.0),
    (2.0, 5.0),
]


class ContinuousLaserTagPOMDPMetrics(Enum):
    """Metric names for Continuous LaserTag POMDP."""

    TAG_SUCCESS_RATE = "tag_success_rate"
    GOAL_REACHING_RATE = "goal_reaching_rate"
    AVERAGE_EPISODE_LENGTH = "average_episode_length"
    AVERAGE_FAILED_TAG_ATTEMPTS = "average_failed_tag_attempts"
    AVERAGE_WALL_COLLISIONS = "average_wall_collisions"
    AVERAGE_DANGEROUS_AREA_STEPS = "average_dangerous_area_steps"
    AVERAGE_ALL_DANGEROUS_ENCOUNTERS = "average_all_dangerous_encounters"


class ContinuousLaserTagStateTransitionModel(_native.ContinuousLaserTagTransitionCpp):
    """State transition model for the Continuous LaserTag POMDP.

    Robot movement: ``next_pos = pos + action[:2] + noise`` where noise is
    sampled from a 2-D Gaussian. Opponent pursues the robot stochastically
    (mean step of ``pursuit_speed`` along ``(robot - opponent)`` unit vector,
    plus a separate 2-D Gaussian). Wall collisions are resolved via
    circle-AABB minimum translation vector and the final position is
    clamped to the grid.

    The ``sample()``, ``probability()`` and ``batch_sample()`` methods
    execute entirely in C++ via the ``_native`` extension; this Python
    subclass only wraps the constructor so existing call sites that pass
    :class:`CovarianceParameterizedMultivariateNormal` keep working.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.utils.multivariate_normal import (
        ...     CovarianceParameterizedMultivariateNormal,
        ... )
        >>> from POMDPPlanners.environments.laser_tag_pomdp import _native
        >>> _native.set_seed(42)
        >>> state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
        >>> action = np.array([1.0, 0.0, 0.0])
        >>> robot_dist = CovarianceParameterizedMultivariateNormal(np.eye(2) * 0.1)
        >>> opponent_dist = CovarianceParameterizedMultivariateNormal(np.eye(2) * 0.05)
        >>> walls = np.empty((0, 4))
        >>> grid_size = np.array([11.0, 7.0])
        >>> model = ContinuousLaserTagStateTransitionModel(
        ...     state=state, action=action,
        ...     robot_transition_dist=robot_dist,
        ...     opponent_transition_dist=opponent_dist,
        ...     pursuit_speed=0.6,
        ...     walls=walls, grid_size=grid_size,
        ...     robot_radius=0.3, opponent_radius=0.3, tag_radius=0.5,
        ... )
        >>> samples = model.sample(n_samples=3)
        >>> len(samples)
        3
    """

    def __init__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        robot_transition_dist: CovarianceParameterizedMultivariateNormal,
        opponent_transition_dist: CovarianceParameterizedMultivariateNormal,
        pursuit_speed: float,
        walls: np.ndarray,
        grid_size: np.ndarray,
        robot_radius: float,
        opponent_radius: float,
        tag_radius: float,
    ):
        super().__init__(
            state=state,
            action=action,
            robot_covariance=robot_transition_dist.covariance,
            opponent_covariance=opponent_transition_dist.covariance,
            pursuit_speed=pursuit_speed,
            walls=np.asarray(walls, dtype=float).reshape(-1, 4),
            grid_size=np.asarray(grid_size, dtype=float),
            robot_radius=robot_radius,
            opponent_radius=opponent_radius,
            tag_radius=tag_radius,
        )
        self._robot_dist = robot_transition_dist
        self._opponent_dist = opponent_transition_dist


StateTransitionModel.register(ContinuousLaserTagStateTransitionModel)


class ContinuousLaserTagObservationModel(_native.ContinuousLaserTagObservationCpp):
    """Observation model for the Continuous LaserTag POMDP.

    Provides 8-direction laser range measurements with Gaussian noise. The
    ``sample()``, ``probability()`` and ``batch_log_likelihood()`` methods
    execute entirely in C++ via the ``_native`` extension; this Python
    subclass only wraps the constructor.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.laser_tag_pomdp import _native
        >>> _native.set_seed(42)
        >>> state = np.array([3.0, 3.0, 8.0, 5.0, 0.0])
        >>> walls = np.empty((0, 4))
        >>> grid_size = np.array([11.0, 7.0])
        >>> model = ContinuousLaserTagObservationModel(
        ...     next_state=state, action=np.array([1.0, 0.0, 0.0]),
        ...     measurement_noise=1.0, walls=walls,
        ...     grid_size=grid_size, opponent_radius=0.3,
        ... )
        >>> obs = model.sample(n_samples=2)
        >>> len(obs)
        2
    """

    def __init__(
        self,
        next_state: np.ndarray,
        action: np.ndarray,
        measurement_noise: float,
        walls: np.ndarray,
        grid_size: np.ndarray,
        opponent_radius: float,
    ):
        super().__init__(
            next_state=next_state,
            action=action,
            measurement_noise=measurement_noise,
            walls=np.asarray(walls, dtype=float).reshape(-1, 4),
            grid_size=np.asarray(grid_size, dtype=float),
            opponent_radius=opponent_radius,
        )
        self._measurement_noise = measurement_noise


ObservationModel.register(ContinuousLaserTagObservationModel)


class ContinuousLaserTagPOMDP(Environment):
    """Continuous LaserTag POMDP with continuous ``[dx, dy, tag_flag]`` actions.

    A pursuit-evasion problem in continuous 2-D space where a robot must
    navigate to tag an opponent.  The robot receives noisy 8-direction
    laser range observations.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>>
        >>> # Initialize environment
        >>> env = ContinuousLaserTagPOMDP(discount_factor=0.95)
        >>>
        >>> # Get initial state
        >>> initial_state = env.initial_state_dist().sample()[0]
        >>>
        >>> # Sample complete step
        >>> action = np.array([1.0, 0.0, 0.0])
        >>> next_state, observation, reward = env.sample_next_step(initial_state, action)
        >>>
        >>> # Check terminal condition
        >>> env.is_terminal(initial_state)
        False
    """

    def __init__(
        self,
        discount_factor: float,
        name: str = "ContinuousLaserTagPOMDP",
        grid_size: Tuple[float, float] = (11.0, 7.0),
        walls: Optional[List[Tuple[float, float, float, float]]] = None,
        robot_radius: float = 0.3,
        opponent_radius: float = 0.3,
        tag_radius: float = 0.5,
        tag_reward: float = 10.0,
        tag_penalty: float = 10.0,
        step_cost: float = 1.0,
        measurement_noise: float = 1.0,
        robot_transition_cov_matrix: np.ndarray = np.eye(2) * 0.1,
        opponent_transition_cov_matrix: np.ndarray = np.eye(2) * 0.05,
        pursuit_speed: float = 0.6,
        dangerous_areas: Optional[List[Tuple[float, float]]] = None,
        dangerous_area_radius: float = 1.0,
        dangerous_area_penalty: float = 5.0,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
        initial_state: Optional[np.ndarray] = None,
    ):
        """Initialize the Continuous LaserTag POMDP.

        Args:
            discount_factor: Discount factor for future rewards.
            name: Name identifier for this environment.
            grid_size: Arena dimensions ``(width, height)``.
            walls: Wall AABBs as ``(cx, cy, hx, hy)`` tuples.
            robot_radius: Robot body radius.
            opponent_radius: Opponent body radius.
            tag_radius: Maximum distance for a tag to succeed.
            tag_reward: Reward for successful tagging.
            tag_penalty: Penalty for failed tag attempt.
            step_cost: Cost per action.
            measurement_noise: Std of Gaussian laser noise.
            robot_transition_cov_matrix: 2x2 covariance for robot noise.
            opponent_transition_cov_matrix: 2x2 covariance for opponent noise.
            pursuit_speed: Mean opponent step magnitude toward robot.
            dangerous_areas: Dangerous area centers as ``(x, y)`` tuples.
            dangerous_area_radius: Radius of dangerous areas.
            dangerous_area_penalty: Penalty for being in a dangerous area.
            output_dir: Optional logging directory.
            debug: Enable debug logging.
            use_queue_logger: Use queue-based logger.
            initial_state: Fixed initial state (if provided).
        """
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("discount_factor must be between 0 and 1 (inclusive)")

        space_info = SpaceInfo(
            action_space=SpaceType.CONTINUOUS,
            observation_space=SpaceType.CONTINUOUS,
        )
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=(-tag_penalty - step_cost, tag_reward),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.grid_size_tuple = grid_size
        self._grid_size = np.array(grid_size, dtype=float)
        wall_list = walls if walls is not None else _DEFAULT_WALLS
        self._walls = np.array(wall_list, dtype=float).reshape(-1, 4)
        self.robot_radius = robot_radius
        self.opponent_radius = opponent_radius
        self.tag_radius = tag_radius
        self.tag_reward = tag_reward
        self.tag_penalty = tag_penalty
        self.step_cost = step_cost
        self.measurement_noise = measurement_noise
        self.pursuit_speed = pursuit_speed
        self.dangerous_areas: List[Tuple[float, float]] = (
            list(dangerous_areas) if dangerous_areas is not None else list(_DEFAULT_DANGEROUS_AREAS)
        )
        self.dangerous_area_radius = dangerous_area_radius
        self.dangerous_area_penalty = dangerous_area_penalty
        self.initial_state_value = initial_state

        self.robot_transition_cov_matrix = np.asarray(robot_transition_cov_matrix)
        self.opponent_transition_cov_matrix = np.asarray(opponent_transition_cov_matrix)

        self._robot_transition_dist = CovarianceParameterizedMultivariateNormal(
            self.robot_transition_cov_matrix
        )
        self._opponent_transition_dist = CovarianceParameterizedMultivariateNormal(
            self.opponent_transition_cov_matrix
        )

        # Per-action kernel caches: one C++ kernel per distinct action vector.
        # Hot-path overrides flip the (next_)state field via set_state /
        # set_next_state instead of rebuilding (skips the per-call Cholesky
        # build and wall-array repacking). Keys are action.tobytes(); values
        # are the long-lived C++ kernels.
        self._trans_kernel_cache: Dict[bytes, Any] = {}
        self._obs_kernel_cache: Dict[bytes, Any] = {}

    # ------------------------------------------------------------------
    # Core Environment interface
    # ------------------------------------------------------------------

    def state_transition_model(self, state: np.ndarray, action: np.ndarray) -> StateTransitionModel:
        return ContinuousLaserTagStateTransitionModel(  # pyright: ignore[reportReturnType]
            state=state,
            action=np.asarray(action, dtype=float),
            robot_transition_dist=self._robot_transition_dist,
            opponent_transition_dist=self._opponent_transition_dist,
            pursuit_speed=self.pursuit_speed,
            walls=self._walls,
            grid_size=self._grid_size,
            robot_radius=self.robot_radius,
            opponent_radius=self.opponent_radius,
            tag_radius=self.tag_radius,
        )

    def observation_model(self, next_state: np.ndarray, action: np.ndarray) -> ObservationModel:
        return ContinuousLaserTagObservationModel(  # pyright: ignore[reportReturnType]
            next_state=next_state,
            action=np.asarray(action, dtype=float),
            measurement_noise=self.measurement_noise,
            walls=self._walls,
            grid_size=self._grid_size,
            opponent_radius=self.opponent_radius,
        )

    # ── Hot-path sampling overrides ─────────────────────────────────
    # The default base-class implementations build a Python wrapper subclass
    # per call which only stores a few attributes; the actual RNG draw lives
    # in the C++ _native kernel. These overrides skip the Python subclass,
    # fetch a cached per-action C++ kernel (Cholesky factored once per action,
    # walls repacked once per action), and rewrite only the (next_)state
    # field per call via set_state / set_next_state. The C++ RNG state lives
    # on the kernel, so each cached kernel maintains a single RNG stream per
    # (env, action).

    def _get_trans_kernel(self, action: np.ndarray) -> Any:
        action_arr = np.ascontiguousarray(np.asarray(action, dtype=np.float64))
        key = action_arr.tobytes()
        kernel = self._trans_kernel_cache.get(key)
        if kernel is None:
            # Use a zero placeholder state — set_state will overwrite it.
            kernel = _native.ContinuousLaserTagTransitionCpp(
                state=np.zeros(5, dtype=np.float64),
                action=action_arr,
                robot_covariance=self._robot_transition_dist.covariance,
                opponent_covariance=self._opponent_transition_dist.covariance,
                pursuit_speed=self.pursuit_speed,
                walls=self._walls,
                grid_size=self._grid_size,
                robot_radius=self.robot_radius,
                opponent_radius=self.opponent_radius,
                tag_radius=self.tag_radius,
            )
            self._trans_kernel_cache[key] = kernel
        return kernel

    def _get_obs_kernel(self, action: np.ndarray) -> Any:
        action_arr = np.ascontiguousarray(np.asarray(action, dtype=np.float64))
        key = action_arr.tobytes()
        kernel = self._obs_kernel_cache.get(key)
        if kernel is None:
            kernel = _native.ContinuousLaserTagObservationCpp(
                next_state=np.zeros(5, dtype=np.float64),
                action=action_arr,
                measurement_noise=self.measurement_noise,
                walls=self._walls,
                grid_size=self._grid_size,
                opponent_radius=self.opponent_radius,
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
        probs = np.asarray(kernel.probability(observations))
        with np.errstate(divide="ignore"):
            return np.log(probs)

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
        log_probs = np.asarray(
            kernel.batch_log_likelihood(
                next_particles=next_states_array,
                observation=observation_array,
            ),
            dtype=np.float64,
        )
        return log_probs

    def reward(self, state: np.ndarray, action: np.ndarray) -> float:
        if bool(state[4]):
            return 0.0

        action = np.asarray(action, dtype=float)
        robot_pos = state[:2]
        opp_pos = state[2:4]
        tag_flag = action[2] if len(action) > 2 else 0.0

        base_reward = 0.0
        if tag_flag > 0.5:
            dist = float(np.linalg.norm(robot_pos - opp_pos))
            if dist <= self.tag_radius:
                base_reward = self.tag_reward
            else:
                base_reward = -self.tag_penalty
        base_reward -= self.step_cost

        # Dangerous area penalty
        if self._is_in_dangerous_area(robot_pos):
            base_reward -= self.dangerous_area_penalty

        return base_reward

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: np.ndarray,
    ) -> np.ndarray:
        states_arr = np.asarray(states, dtype=float)
        if states_arr.ndim == 1:
            states_arr = states_arr.reshape(1, -1)
        n = states_arr.shape[0]
        action = np.asarray(action, dtype=float)
        tag_flag = action[2] if len(action) > 2 else 0.0

        rewards = np.full(n, -self.step_cost)

        terminal_mask = states_arr[:, 4] == 1.0
        rewards[terminal_mask] = 0.0

        live = ~terminal_mask
        if tag_flag > 0.5:
            dists = np.linalg.norm(states_arr[live, :2] - states_arr[live, 2:4], axis=1)
            tag_success = dists <= self.tag_radius
            live_idx = np.where(live)[0]
            rewards[live_idx[tag_success]] += self.tag_reward
            rewards[live_idx[~tag_success]] -= self.tag_penalty

        # Dangerous area penalty
        if self.dangerous_areas:
            for dx, dy in self.dangerous_areas:
                d = np.sqrt((states_arr[live, 0] - dx) ** 2 + (states_arr[live, 1] - dy) ** 2)
                live_idx = np.where(live)[0]
                in_danger = d <= self.dangerous_area_radius
                rewards[live_idx[in_danger]] -= self.dangerous_area_penalty

        return rewards

    def is_terminal(self, state: np.ndarray) -> bool:
        return bool(state[4])

    def initial_state_dist(self) -> Distribution:
        if self.initial_state_value is not None:
            return DiscreteDistribution(values=[self.initial_state_value], probs=np.array([1.0]))
        return _ContinuousLaserTagInitialDist(
            self._grid_size,
            self._walls,
            self.robot_radius,
            self.opponent_radius,
            self.tag_radius,
        )

    def initial_observation_dist(self) -> Distribution:
        mid = np.full(8, min(self._grid_size) / 2.0)
        return DiscreteDistribution(values=[mid], probs=np.array([1.0]))

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return np.array_equal(
            np.asarray(observation1, dtype=float),
            np.asarray(observation2, dtype=float),
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metric_names(self) -> List[str]:
        return [m.value for m in ContinuousLaserTagPOMDPMetrics]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        if not histories:
            return []

        episode_data = self._collect_episode_data(histories)
        cis = self._compute_confidence_intervals(episode_data)
        return self._build_metrics(episode_data, cis)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        visualizer = ContinuousLaserTagVisualizer(
            grid_size=self._grid_size,
            walls=self._walls,
            robot_radius=self.robot_radius,
            opponent_radius=self.opponent_radius,
            dangerous_areas=self.dangerous_areas,
            dangerous_area_radius=self.dangerous_area_radius,
        )
        visualizer.create_visualization(history, cache_path)
        self.logger.info("Saved ContinuousLaserTag visualization to %s", cache_path)

    # ------------------------------------------------------------------
    # Accessors used by the vectorized updater
    # ------------------------------------------------------------------

    @property
    def walls(self) -> np.ndarray:
        return self._walls

    @property
    def grid_size(self) -> np.ndarray:
        return self._grid_size

    # ------------------------------------------------------------------
    # Pickling
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_in_dangerous_area(self, position: np.ndarray) -> bool:
        if not self.dangerous_areas:
            return False
        for dx, dy in self.dangerous_areas:
            if (
                np.sqrt((position[0] - dx) ** 2 + (position[1] - dy) ** 2)
                <= self.dangerous_area_radius
            ):
                return True
        return False

    def _collect_episode_data(self, histories: List[History]) -> Dict[str, list]:
        data: Dict[str, list] = {
            "lengths": [],
            "success": [],
            "goal_reached": [],
            "failed_tags": [],
            "wall_collisions": [],
            "dangerous_steps": [],
            "all_dangerous": [],
        }
        for history in histories:
            steps = history.history
            data["lengths"].append(len(steps))
            data["success"].append(
                1 if steps and steps[-1].reward is not None and steps[-1].reward > 0 else 0
            )
            goal = any(
                isinstance(s.state, np.ndarray) and len(s.state) == 5 and bool(s.state[4])
                for s in steps
            )
            data["goal_reached"].append(1 if goal else 0)

            failed, wall_col, danger_steps = self._count_episode_metrics(steps)
            data["failed_tags"].append(failed)
            data["wall_collisions"].append(wall_col)
            data["dangerous_steps"].append(danger_steps)
            data["all_dangerous"].append(wall_col + danger_steps)
        return data

    def _count_episode_metrics(self, steps: List[StepData]) -> Tuple[int, int, int]:
        failed_tags = 0
        wall_collisions = 0
        dangerous_steps = 0

        for step in steps:
            action = np.asarray(step.action, dtype=float) if step.action is not None else None
            if action is not None and len(action) >= 3 and action[2] > 0.5:
                if step.reward is not None and step.reward < 0:
                    failed_tags += 1

            if isinstance(step.state, np.ndarray) and len(step.state) == 5:
                if self._is_in_dangerous_area(step.state[:2]):
                    dangerous_steps += 1

        return failed_tags, wall_collisions, dangerous_steps

    def _compute_confidence_intervals(
        self, data: Dict[str, list]
    ) -> Dict[str, Tuple[float, float]]:
        n = len(data["lengths"])
        if n < 2:
            return {k: (-np.inf, np.inf) for k in data}
        return {k: confidence_interval(data=v, confidence=0.95) for k, v in data.items()}

    def _build_metrics(
        self,
        data: Dict[str, list],
        cis: Dict[str, Tuple[float, float]],
    ) -> List[MetricValue]:
        n = len(data["lengths"])
        metric_map = [
            (ContinuousLaserTagPOMDPMetrics.TAG_SUCCESS_RATE, "success"),
            (ContinuousLaserTagPOMDPMetrics.GOAL_REACHING_RATE, "goal_reached"),
            (ContinuousLaserTagPOMDPMetrics.AVERAGE_EPISODE_LENGTH, "lengths"),
            (ContinuousLaserTagPOMDPMetrics.AVERAGE_FAILED_TAG_ATTEMPTS, "failed_tags"),
            (ContinuousLaserTagPOMDPMetrics.AVERAGE_WALL_COLLISIONS, "wall_collisions"),
            (ContinuousLaserTagPOMDPMetrics.AVERAGE_DANGEROUS_AREA_STEPS, "dangerous_steps"),
            (ContinuousLaserTagPOMDPMetrics.AVERAGE_ALL_DANGEROUS_ENCOUNTERS, "all_dangerous"),
        ]
        metrics = []
        for metric_enum, key in metric_map:
            val = float(np.mean(data[key])) if n > 0 else 0.0
            ci = cis[key]
            metrics.append(
                MetricValue(
                    name=metric_enum.value,
                    value=val,
                    lower_confidence_bound=ci[0],
                    upper_confidence_bound=ci[1],
                )
            )
        return metrics


class ContinuousLaserTagPOMDPDiscreteActions(ContinuousLaserTagPOMDP, DiscreteActionsEnvironment):
    """Continuous LaserTag POMDP with discrete string actions.

    Actions: ``"up"``, ``"down"``, ``"right"``, ``"left"``, ``"tag"``.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>>
        >>> env = ContinuousLaserTagPOMDPDiscreteActions(discount_factor=0.95)
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
        name: str = "ContinuousLaserTagPOMDPDiscreteActions",
        grid_size: Tuple[float, float] = (11.0, 7.0),
        walls: Optional[List[Tuple[float, float, float, float]]] = None,
        robot_radius: float = 0.3,
        opponent_radius: float = 0.3,
        tag_radius: float = 0.5,
        tag_reward: float = 10.0,
        tag_penalty: float = 10.0,
        step_cost: float = 1.0,
        measurement_noise: float = 1.0,
        robot_transition_cov_matrix: np.ndarray = np.eye(2) * 0.1,
        opponent_transition_cov_matrix: np.ndarray = np.eye(2) * 0.05,
        pursuit_speed: float = 0.6,
        dangerous_areas: Optional[List[Tuple[float, float]]] = None,
        dangerous_area_radius: float = 1.0,
        dangerous_area_penalty: float = 5.0,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
        initial_state: Optional[np.ndarray] = None,
    ):
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            grid_size=grid_size,
            walls=walls,
            robot_radius=robot_radius,
            opponent_radius=opponent_radius,
            tag_radius=tag_radius,
            tag_reward=tag_reward,
            tag_penalty=tag_penalty,
            step_cost=step_cost,
            measurement_noise=measurement_noise,
            robot_transition_cov_matrix=robot_transition_cov_matrix,
            opponent_transition_cov_matrix=opponent_transition_cov_matrix,
            pursuit_speed=pursuit_speed,
            dangerous_areas=dangerous_areas,
            dangerous_area_radius=dangerous_area_radius,
            dangerous_area_penalty=dangerous_area_penalty,
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
            initial_state=initial_state,
        )

        # Override space info to discrete actions
        self.space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.CONTINUOUS,
        )

        self.actions: List[str] = ["up", "down", "right", "left", "tag"]
        self.action_to_vector: Dict[str, np.ndarray] = {
            "up": np.array([0.0, 1.0, 0.0]),
            "down": np.array([0.0, -1.0, 0.0]),
            "right": np.array([1.0, 0.0, 0.0]),
            "left": np.array([-1.0, 0.0, 0.0]),
            "tag": np.array([0.0, 0.0, 1.0]),
        }

    def get_actions(self) -> List[str]:
        return self.actions

    def state_transition_model(self, state: np.ndarray, action: Any) -> StateTransitionModel:
        return super().state_transition_model(state, self.action_to_vector[action])

    def observation_model(self, next_state: np.ndarray, action: Any) -> ObservationModel:
        return super().observation_model(next_state, self.action_to_vector[action])

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

    def reward(self, state: np.ndarray, action: Any) -> float:
        return super().reward(state, self.action_to_vector[action])

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: Any,
    ) -> np.ndarray:
        return super().reward_batch(np.asarray(states), self.action_to_vector[action])

    def _count_episode_metrics(self, steps: List[StepData]) -> Tuple[int, int, int]:
        converted = []
        for step in steps:
            if step.action is not None and isinstance(step.action, str):
                converted.append(step._replace(action=self.action_to_vector[step.action]))
            else:
                converted.append(step)
        return super()._count_episode_metrics(converted)


class _ContinuousLaserTagInitialDist(Distribution):
    """Rejection-sampling initial state distribution."""

    def __init__(
        self,
        grid_size: np.ndarray,
        walls: np.ndarray,
        robot_radius: float,
        opponent_radius: float,
        tag_radius: float,
    ):
        self._grid_size = grid_size
        self._walls = walls
        self._robot_radius = robot_radius
        self._opponent_radius = opponent_radius
        self._tag_radius = tag_radius

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        samples: List[np.ndarray] = []
        for _ in range(n_samples):
            samples.append(self._sample_single())
        return samples

    def probability(self, values: List[Any]) -> np.ndarray:
        return np.zeros(len(values))

    def _sample_single(self) -> np.ndarray:
        max_attempts = 10000
        for _ in range(max_attempts):
            robot = self._sample_valid_position(self._robot_radius)
            opp = self._sample_valid_position(self._opponent_radius)
            if np.linalg.norm(robot - opp) > self._tag_radius:
                return np.array([robot[0], robot[1], opp[0], opp[1], 0.0])
        # Fallback
        return np.array([1.0, 1.0, self._grid_size[0] - 1.0, self._grid_size[1] - 1.0, 0.0])

    def _sample_valid_position(self, radius: float) -> np.ndarray:
        for _ in range(1000):
            x = np.random.uniform(radius, self._grid_size[0] - radius)
            y = np.random.uniform(radius, self._grid_size[1] - radius)
            pos = np.array([x, y])
            if not self._overlaps_wall(pos, radius):
                return pos
        return np.array([self._grid_size[0] / 2, self._grid_size[1] / 2])

    def _overlaps_wall(self, pos: np.ndarray, radius: float) -> bool:
        if self._walls.shape[0] == 0:
            return False
        for i in range(self._walls.shape[0]):
            cx, cy, hx, hy = self._walls[i]
            closest_x = np.clip(pos[0], cx - hx, cx + hx)
            closest_y = np.clip(pos[1], cy - hy, cy + hy)
            dx = pos[0] - closest_x
            dy = pos[1] - closest_y
            if dx * dx + dy * dy < radius * radius:
                return True
        return False
