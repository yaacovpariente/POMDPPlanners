# SPDX-License-Identifier: MIT

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

Opponent behaviour is selectable via ``opponent_policy`` (see
:class:`~POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils.OpponentPolicy`):
``EVADE`` (default) flees the robot's pre-move position at ``evasion_speed``;
``PURSUE`` chases the robot's post-move position. ``EVADE_WHEN_SPOTTED`` flees
only while the robot has line of sight to it and otherwise **holds its position**
(in this continuous env; the discrete grid env moves randomly instead).
``evasion_speed`` is a direction-neutral step magnitude under all policies.

Classes:
    ContinuousLaserTagPOMDP: Continuous-action environment.
    ContinuousLaserTagPOMDPDiscreteActions: Discrete-action variant.
"""

# pylint: disable=too-many-lines  # Module size exceeds 1000 lines due to native rollout addition.

from __future__ import annotations

from enum import Enum
from pathlib import Path
from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    Environment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.environments.laser_tag_pomdp import _native
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils import (
    OpponentPolicy,
)
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_visualizer import (
    ContinuousLaserTagVisualizer,
)
from POMDPPlanners.planners.planners_utils.rollout import python_random_rollout
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


class ContinuousLaserTagPOMDP(Environment):
    """Continuous LaserTag POMDP with continuous ``[dx, dy, tag_flag]`` actions.

    A pursuit-evasion problem in continuous 2-D space where a robot must
    navigate to tag an opponent.  The robot receives noisy 8-direction
    laser range observations.

    Stochasticity:
        The dangerous-area penalty can be applied either deterministically
        (the default) or stochastically.  When
        ``dangerous_area_hit_probability == 1.0`` (default), the kernel's
        deterministic deduction is preserved verbatim, matching legacy
        behavior.  When ``dangerous_area_hit_probability < 1.0``, the
        accumulated dangerous-area deduction is applied to the reward only
        with that probability per ``reward()`` call, producing a
        heavy-tailed return distribution suitable for benchmarking
        risk-sensitive planners (e.g. ICVaR-aware MCTS) against
        expected-value MCTS on the same env.  Note that this makes
        ``reward(state, action)`` non-deterministic given a state-action
        pair, so any external caching that assumes deterministic rewards
        must be aware of this.  ``transition_log_probability`` is
        unaffected.

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

    Example:
        Risk-sensitive evaluation -- a 10%-tail-risk environment suitable
        for benchmarking ICVaR-aware planners against expected-value MCTS::

            >>> env = ContinuousLaserTagPOMDP(
            ...     discount_factor=0.95,
            ...     dangerous_area_penalty=150.0,
            ...     dangerous_area_hit_probability=0.1,
            ... )
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
        evasion_speed: float = 0.6,
        dangerous_areas: Optional[List[Tuple[float, float]]] = None,
        dangerous_area_radius: float = 1.0,
        dangerous_area_penalty: float = 5.0,
        dangerous_area_hit_probability: float = 1.0,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
        initial_state: Optional[np.ndarray] = None,
        opponent_policy: OpponentPolicy = OpponentPolicy.EVADE,
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
            evasion_speed: Mean opponent step magnitude away from robot.
            dangerous_areas: Dangerous area centers as ``(x, y)`` tuples.
            dangerous_area_radius: Radius of dangerous areas.
            dangerous_area_penalty: Penalty for being in a dangerous area.
            dangerous_area_hit_probability: Probability that the
                dangerous-area penalty is actually applied to the reward
                when the robot is inside a dangerous area.  Must lie in
                ``[0, 1]``.  Defaults to ``1.0`` (deterministic penalty,
                matching legacy behavior).  Values below ``1.0`` make the
                reward stochastic, useful for risk-sensitive planning
                benchmarks.
            output_dir: Optional logging directory.
            debug: Enable debug logging.
            use_queue_logger: Use queue-based logger.
            initial_state: Fixed initial state (if provided).
            opponent_policy: Selects the opponent transition behaviour.
                ``EVADE`` (default) flees the robot at ``evasion_speed`` away from
                its pre-move position; ``PURSUE`` chases toward its post-move
                position. ``evasion_speed`` is a direction-neutral step magnitude
                under both policies. See
                :class:`~POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils.OpponentPolicy`.
        """
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("discount_factor must be between 0 and 1 (inclusive)")
        if not 0.0 <= dangerous_area_hit_probability <= 1.0:
            raise ValueError("dangerous_area_hit_probability must be between 0 and 1 (inclusive)")

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
        self.evasion_speed = evasion_speed
        self.opponent_policy = opponent_policy
        self.dangerous_areas: List[Tuple[float, float]] = (
            list(dangerous_areas) if dangerous_areas is not None else list(_DEFAULT_DANGEROUS_AREAS)
        )
        self.dangerous_area_radius = dangerous_area_radius
        self.dangerous_area_penalty = dangerous_area_penalty
        self.dangerous_area_hit_probability = float(dangerous_area_hit_probability)
        # Packed (K, 2) float64 dangerous-area array; reused by the C++ reward kernel.
        self._dangerous_areas_arr = (
            np.ascontiguousarray(np.asarray(self.dangerous_areas, dtype=np.float64).reshape(-1, 2))
            if self.dangerous_areas
            else np.empty(0, dtype=np.float64)
        )
        self.initial_state_value = initial_state

        self.robot_transition_cov_matrix = np.asarray(robot_transition_cov_matrix)
        self.opponent_transition_cov_matrix = np.asarray(opponent_transition_cov_matrix)

        self._robot_transition_dist = CovarianceParameterizedMultivariateNormal(
            self.robot_transition_cov_matrix
        )
        self._opponent_transition_dist = CovarianceParameterizedMultivariateNormal(
            self.opponent_transition_cov_matrix
        )

        # Per-action C++ kernel caches (Cholesky factored once per action).
        # The bytes-keyed cache is the source of truth; the id-keyed shortcut
        # skips ``tobytes()`` when the same Python action object is passed
        # repeatedly (typical for the discrete-action wrapper, which maps each
        # label to a single cached ndarray).
        self._trans_kernel_cache: Dict[bytes, Any] = {}
        self._obs_kernel_cache: Dict[bytes, Any] = {}
        self._trans_kernel_id_cache: Dict[int, Any] = {}
        self._obs_kernel_id_cache: Dict[int, Any] = {}
        # Static params for cont_simulate_rollout: built once, unpacked per call.
        self._rollout_static_params: Dict[str, Any] = {
            "robot_covariance": self._robot_transition_dist.covariance,
            "opponent_covariance": self._opponent_transition_dist.covariance,
            "evasion_speed": self.evasion_speed,
            "walls": self._walls,
            "grid_size": self._grid_size,
            "robot_radius": self.robot_radius,
            "opponent_radius": self.opponent_radius,
            "tag_radius": self.tag_radius,
            "tag_reward": self.tag_reward,
            "tag_penalty": self.tag_penalty,
            "step_cost": self.step_cost,
            "dangerous_areas": self._dangerous_areas_arr,
            "dangerous_area_radius": self.dangerous_area_radius,
            "dangerous_area_penalty": self.dangerous_area_penalty,
            "opponent_policy_code": self.opponent_policy.native_code,
        }

    # ------------------------------------------------------------------
    # Core Environment interface
    # ------------------------------------------------------------------

    # ── Hot-path sampling overrides ─────────────────────────────────
    # Skip the Python wrapper subclass; fetch/reuse a cached per-action C++
    # kernel (Cholesky factored once, walls repacked once) and mutate only
    # the stored state via set_state / set_next_state.

    def _get_trans_kernel(self, action: np.ndarray) -> Any:
        cached = self._trans_kernel_id_cache.get(id(action))
        if cached is not None:
            return cached
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
                evasion_speed=self.evasion_speed,
                walls=self._walls,
                grid_size=self._grid_size,
                robot_radius=self.robot_radius,
                opponent_radius=self.opponent_radius,
                tag_radius=self.tag_radius,
                opponent_policy_code=self.opponent_policy.native_code,
            )
            self._trans_kernel_cache[key] = kernel
        if isinstance(action, np.ndarray):
            self._trans_kernel_id_cache[id(action)] = kernel
        return kernel

    def _get_obs_kernel(self, action: np.ndarray) -> Any:
        cached = self._obs_kernel_id_cache.get(id(action))
        if cached is not None:
            return cached
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
        if isinstance(action, np.ndarray):
            self._obs_kernel_id_cache[id(action)] = kernel
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
        # kernel.probability returns a C-contiguous float64 ndarray; skip
        # the redundant np.asarray wrap.
        probs = kernel.probability(next_states)
        with np.errstate(divide="ignore"):
            return np.log(probs)

    def observation_log_probability(
        self, next_state: np.ndarray, action: np.ndarray, observations: Any
    ) -> np.ndarray:
        kernel = self._get_obs_kernel(action)
        kernel.set_next_state(next_state)
        # B1 fix: call kernel.log_probability(...) directly instead of
        # np.log(kernel.probability(...)). The legacy probability path
        # round-trips log_pdf through std::exp, which underflows to 0.0
        # for low-density observations and makes np.log return -inf,
        # disagreeing with the batched
        # ``observation_log_probability_per_state`` path that goes
        # through ``batch_log_likelihood``. log_probability returns
        # log_pdf directly, eliminating the round-trip.
        return kernel.log_probability(observations)

    def observation_log_probability_single(
        self, next_state: Any, action: Any, observation: Any
    ) -> float:
        # Scalar fast-path for POMCPOW's WeightedParticleBeliefStateUpdate.
        # B1 fix: use kernel.log_probability so low-density observations
        # whose log_pdf is below the IEEE-754 ``exp`` underflow boundary
        # still return a finite, large-negative log-likelihood instead of
        # collapsing to -inf via the legacy ``np.log(probability(...))``
        # round-trip.
        kernel = self._get_obs_kernel(action)
        kernel.set_next_state(next_state)
        return float(kernel.log_probability([observation])[0])

    def sample_next_state_batch(self, states: Any, action: np.ndarray) -> np.ndarray:
        # Short-circuit when the caller already hands us a C-contiguous
        # float64 (N, 5) buffer; otherwise normalise.
        if (
            isinstance(states, np.ndarray)
            and states.dtype == np.float64
            and states.ndim == 2
            and states.flags["C_CONTIGUOUS"]
        ):
            states_array = states
        else:
            states_array = np.ascontiguousarray(np.asarray(states, dtype=np.float64))
            if states_array.ndim == 1:
                states_array = states_array.reshape(1, -1)
        kernel = self._get_trans_kernel(action)
        # batch_sample reads the per-row state from the input, not the
        # kernel's stored state, so no set_state is needed here.
        # The native kernel returns a C-contiguous float64 ndarray
        # (py::array_t<double>) so the np.asarray re-wrap is a no-op — drop it.
        return kernel.batch_sample(states_array)

    def observation_log_probability_per_state(
        self, next_states: Any, action: np.ndarray, observation: Any
    ) -> np.ndarray:
        if (
            isinstance(next_states, np.ndarray)
            and next_states.dtype == np.float64
            and next_states.ndim == 2
            and next_states.flags["C_CONTIGUOUS"]
        ):
            next_states_array = next_states
        else:
            next_states_array = np.ascontiguousarray(np.asarray(next_states, dtype=np.float64))
            if next_states_array.ndim == 1:
                next_states_array = next_states_array.reshape(1, -1)
        if (
            isinstance(observation, np.ndarray)
            and observation.dtype == np.float64
            and observation.ndim == 1
            and observation.flags["C_CONTIGUOUS"]
        ):
            observation_array = observation
        else:
            observation_array = np.ascontiguousarray(np.asarray(observation, dtype=np.float64))
        kernel = self._get_obs_kernel(action)
        # batch_log_likelihood reads next_state per row from the input;
        # no set_next_state needed. Native kernel returns a C-contiguous
        # float64 ndarray (py::array_t<double>) so the np.asarray re-wrap is
        # a no-op — drop it.
        return kernel.batch_log_likelihood(
            next_particles=next_states_array,
            observation=observation_array,
        )

    def simulate_random_rollout(
        self,
        state: np.ndarray,
        action_sampler: Any,
        max_depth: int,
        discount_factor: float,
        depth: int = 0,
    ) -> float:
        """Random rollout dispatched to native C++ via ``cont_simulate_rollout``.

        Pre-samples actions from ``action_sampler``, packs them into a ``(N, 3)``
        buffer, and runs the full discounted-return loop inside C++. Results are
        numerically identical to the :meth:`Environment.simulate_random_rollout`
        Python fallback.

        When ``dangerous_area_hit_probability < 1.0``, falls back to the
        Python rollout: the native kernel applies the dangerous-area
        penalty deterministically per step, which contradicts the
        stochastic semantics; routing through Python ``reward()`` keeps
        the per-step Bernoulli intact.

        Also falls back when ``dangerous_areas`` is non-empty: the C++
        ``cont_simulate_rollout`` kernel scores the danger penalty
        against the *pre-transition* robot position, while the Python
        ``reward()`` path (post-fix) consumes the realised post-transition
        position. Until the C++ kernel is rebuilt this is the only
        correctness-preserving path for configs with danger areas.
        """
        has_dangerous_areas = self._dangerous_areas_arr.shape[0] > 0
        if self.dangerous_area_hit_probability < 1.0 or has_dangerous_areas:
            return python_random_rollout(
                state=state,
                depth=depth,
                action_sampler=action_sampler,
                environment=self,
                discount_factor=discount_factor,
                max_depth=max_depth,
            )
        steps_left = max_depth - depth
        if steps_left <= 0 or bool(state[4]):
            return 0.0
        actions_buffer = self._sample_action_buffer(action_sampler, steps_left)
        return self._native_rollout(state, actions_buffer, depth, max_depth, discount_factor)

    def _native_rollout(
        self,
        state: np.ndarray,
        actions_buffer: np.ndarray,
        depth: int,
        max_depth: int,
        discount_factor: float,
    ) -> float:
        state_arr = np.ascontiguousarray(np.asarray(state, dtype=np.float64).ravel())
        return _native.cont_simulate_rollout(
            initial_state=state_arr,
            actions_buffer=actions_buffer,
            start_depth=depth,
            max_depth=max_depth,
            discount_factor=discount_factor,
            **self._rollout_static_params,
        )

    def _sample_action_buffer(self, action_sampler: Any, n_steps: int) -> np.ndarray:
        action_sample = action_sampler.sample
        rows = []
        for _ in range(n_steps):
            act = np.asarray(action_sample(), dtype=np.float64).ravel()
            if act.shape[0] == 2:
                act = np.concatenate([act, [0.0]])
            rows.append(act)
        return np.ascontiguousarray(np.stack(rows, axis=0))

    def reward(self, state: np.ndarray, action: np.ndarray, next_state: Any = None) -> float:
        # Single-state reward routes through the same C++ kernel used by
        # reward_batch — wrap the state as a (1, 5) row and unpack the
        # scalar. Keeps ``reward`` and ``reward_batch`` semantically and
        # numerically equivalent.
        state_arr = np.ascontiguousarray(np.asarray(state, dtype=np.float64)).reshape(1, -1)
        action_arr = np.ascontiguousarray(np.asarray(action, dtype=np.float64)).ravel()
        # When the caller threads ``next_state`` (e.g. via
        # :meth:`Environment.sample_next_step`), the dangerous-area check
        # must use the realised post-transition robot position rather
        # than the pre-transition state. We disable the kernel's
        # dangerous-area term and reapply it in Python against
        # ``next_state[:2]`` so trajectory and reward agree on the same
        # transition draw.
        if next_state is None:
            danger_arr = self._dangerous_areas_arr
            danger_state_arr = state_arr
        else:
            danger_arr = np.empty(0, dtype=np.float64)
            danger_state_arr = np.ascontiguousarray(
                np.asarray(next_state, dtype=np.float64)
            ).reshape(1, -1)
        rewards = _native.reward_batch(
            state_arr,
            action_arr,
            self.tag_radius,
            self.tag_reward,
            self.tag_penalty,
            self.step_cost,
            danger_arr,
            self.dangerous_area_radius,
            self.dangerous_area_penalty,
        )
        base = float(rewards[0])
        if next_state is None:
            return self._apply_single_state_danger_refund(base, state_arr)
        return self._apply_single_state_danger_penalty(base, state_arr, danger_state_arr)

    def _apply_single_state_danger_refund(self, base: float, state_arr: np.ndarray) -> float:
        # Legacy path (no threaded next_state): the kernel already
        # deducted the deterministic danger penalty; when
        # ``dangerous_area_hit_probability < 1.0`` we cancel the
        # deduction with probability ``1 - p`` per call.
        if self.dangerous_area_hit_probability >= 1.0:
            return base
        if state_arr[0, 4] != 0.0:
            return base
        n_matches = self._count_dangerous_area_matches(state_arr[0, :2])
        if n_matches == 0:
            return base
        if np.random.random() >= self.dangerous_area_hit_probability:
            return base + n_matches * self.dangerous_area_penalty
        return base

    def _apply_single_state_danger_penalty(
        self,
        base: float,
        state_arr: np.ndarray,
        next_state_arr: np.ndarray,
    ) -> float:
        # Threaded-next_state path: the kernel ran with no danger areas,
        # so we apply the deduction here against the *realised*
        # post-transition robot position. Mirrors the
        # ``reward_batch`` semantics (one penalty per matching area)
        # while keeping the per-call Bernoulli draw intentional.
        # Terminal handling matches the kernel: if the pre-transition
        # state was already terminal the row already returned 0.0 from
        # the kernel and we add nothing.
        if state_arr[0, 4] != 0.0:
            return base
        if next_state_arr[0, 4] != 0.0:
            return base
        n_matches = self._count_dangerous_area_matches(next_state_arr[0, :2])
        if n_matches == 0:
            return base
        if (
            self.dangerous_area_hit_probability < 1.0
            and np.random.random() >= self.dangerous_area_hit_probability
        ):
            return base
        return base - n_matches * self.dangerous_area_penalty

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: np.ndarray,
        next_states: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    ) -> np.ndarray:
        # Skip np.asarray re-allocation when the caller already passes a
        # C-contiguous float64 array of the right shape (the planners hot
        # path; matches the same short-circuit used by sample_next_state_batch
        # in PR-D follow-ups).
        if isinstance(states, np.ndarray):
            states_nd: np.ndarray = states
            if (
                states_nd.dtype == np.float64
                and states_nd.ndim == 2
                and states_nd.flags["C_CONTIGUOUS"]
            ):
                states_arr = states_nd
            else:
                states_arr = np.ascontiguousarray(np.asarray(states_nd, dtype=np.float64))
                if states_arr.ndim == 1:
                    states_arr = states_arr.reshape(1, -1)
        else:
            states_arr = np.ascontiguousarray(np.asarray(states, dtype=np.float64))
            if states_arr.ndim == 1:
                states_arr = states_arr.reshape(1, -1)
        if (
            isinstance(action, np.ndarray)
            and action.dtype == np.float64
            and action.ndim == 1
            and action.flags["C_CONTIGUOUS"]
        ):
            action_arr = action
        else:
            action_arr = np.ascontiguousarray(np.asarray(action, dtype=np.float64)).ravel()
        # Honour a threaded ``next_states`` argument: when present, the
        # dangerous-area check (a *post-action* check) must use the
        # realised next-state robot positions rather than re-using
        # ``states_arr``. We disable the kernel's danger term and
        # reapply it in Python against ``next_states_arr[:, :2]``.
        if next_states is None:
            next_states_arr: Optional[np.ndarray] = None
            danger_arr = self._dangerous_areas_arr
        else:
            next_states_arr = np.ascontiguousarray(np.asarray(next_states, dtype=np.float64))
            if next_states_arr.ndim == 1:
                next_states_arr = next_states_arr.reshape(1, -1)
            danger_arr = np.empty(0, dtype=np.float64)
        rewards = _native.reward_batch(
            states_arr,
            action_arr,
            self.tag_radius,
            self.tag_reward,
            self.tag_penalty,
            self.step_cost,
            danger_arr,
            self.dangerous_area_radius,
            self.dangerous_area_penalty,
        )
        if next_states_arr is None:
            return self._apply_stochastic_dangerous_refund(rewards, states_arr)
        return self._apply_dangerous_penalty_to_next_states(rewards, states_arr, next_states_arr)

    def _apply_stochastic_dangerous_refund(
        self, rewards: np.ndarray, states_arr: np.ndarray
    ) -> np.ndarray:
        if self.dangerous_area_hit_probability >= 1.0 or self._dangerous_areas_arr.size == 0:
            return rewards
        positions = states_arr[:, :2]
        centers = self._dangerous_areas_arr.reshape(-1, 2)
        deltas = positions[:, None, :] - centers[None, :, :]
        in_zone = np.sum(deltas * deltas, axis=2) <= (
            self.dangerous_area_radius * self.dangerous_area_radius
        )
        # Terminal rows contribute zero reward and never get a deduction.
        terminal_mask = states_arr[:, 4] != 0.0
        if terminal_mask.any():
            in_zone = in_zone.copy()
            in_zone[terminal_mask, :] = False
        match_counts = np.sum(in_zone, axis=1)
        any_match = match_counts > 0
        # One Bernoulli per state: when miss, refund the full deterministic
        # deduction (n_matches * penalty). Mirrors the semantics of
        # ``reward()`` so single- and batch-paths agree statistically.
        miss = np.random.random(states_arr.shape[0]) >= self.dangerous_area_hit_probability
        refund_mask = any_match & miss
        if not refund_mask.any():
            return rewards
        refunds = match_counts.astype(np.float64) * float(self.dangerous_area_penalty)
        return rewards + np.where(refund_mask, refunds, 0.0)

    def _apply_dangerous_penalty_to_next_states(
        self,
        rewards: np.ndarray,
        states_arr: np.ndarray,
        next_states_arr: np.ndarray,
    ) -> np.ndarray:
        # Threaded-next_states path: the kernel ran with no danger
        # areas, so apply the deduction here against the *realised*
        # post-transition robot positions. Mirrors the
        # ``_apply_stochastic_dangerous_refund`` semantics — one penalty
        # per matching area, one Bernoulli per row when
        # ``hit_probability < 1.0`` — but uses ``next_states_arr[:, :2]``
        # for the position check so the deduction matches the same
        # transition draw consumed by the trajectory.
        if self._dangerous_areas_arr.size == 0:
            return rewards
        positions = next_states_arr[:, :2]
        centers = self._dangerous_areas_arr.reshape(-1, 2)
        deltas = positions[:, None, :] - centers[None, :, :]
        in_zone = np.sum(deltas * deltas, axis=2) <= (
            self.dangerous_area_radius * self.dangerous_area_radius
        )
        # The kernel returns 0.0 for rows whose pre-transition state was
        # already terminal — those rows skip the penalty too. Rows whose
        # *next_state* is terminal also skip it (no deduction after the
        # episode ends).
        pre_terminal = states_arr[:, 4] != 0.0
        post_terminal = next_states_arr[:, 4] != 0.0
        skip_mask = pre_terminal | post_terminal
        if skip_mask.any():
            in_zone = in_zone.copy()
            in_zone[skip_mask, :] = False
        match_counts = np.sum(in_zone, axis=1)
        any_match = match_counts > 0
        if not any_match.any():
            return rewards
        if self.dangerous_area_hit_probability < 1.0:
            applied = (
                np.random.random(next_states_arr.shape[0]) < self.dangerous_area_hit_probability
            )
            apply_mask = any_match & applied
        else:
            apply_mask = any_match
        if not apply_mask.any():
            return rewards
        deductions = match_counts.astype(np.float64) * float(self.dangerous_area_penalty)
        return rewards - np.where(apply_mask, deductions, 0.0)

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

    def hash_observation(self, observation: Any) -> Hashable:
        # is_equal_observation casts to float arrays before comparing;
        # match that contract by hashing the float64 byte representation.
        return np.ascontiguousarray(np.asarray(observation, dtype=float)).tobytes()

    def hash_action(self, action: Any) -> Hashable:
        # Continuous actions are ndarray of shape (3,); bytes match
        # np.array_equal semantics for arrays of identical shape and dtype.
        return np.ascontiguousarray(action, dtype=np.float64).tobytes()

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
        # empty caches so the env works after unpickling. The id-keyed
        # shortcut caches must also be dropped — id() values are not stable
        # across processes.
        state = self.__dict__.copy()
        state["_trans_kernel_cache"] = {}
        state["_obs_kernel_cache"] = {}
        state["_trans_kernel_id_cache"] = {}
        state["_obs_kernel_id_cache"] = {}
        return state

    def __setstate__(self, state):
        vars(self).update(state)
        self._trans_kernel_cache = {}
        self._obs_kernel_cache = {}
        self._trans_kernel_id_cache = {}
        self._obs_kernel_id_cache = {}

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

    def _count_dangerous_area_matches(self, position: np.ndarray) -> int:
        if self._dangerous_areas_arr.size == 0:
            return 0
        centers = self._dangerous_areas_arr.reshape(-1, 2)
        dx = centers[:, 0] - float(position[0])
        dy = centers[:, 1] - float(position[1])
        return int(np.sum(dx * dx + dy * dy <= self.dangerous_area_radius**2))

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
        evasion_speed: float = 0.6,
        dangerous_areas: Optional[List[Tuple[float, float]]] = None,
        dangerous_area_radius: float = 1.0,
        dangerous_area_penalty: float = 5.0,
        dangerous_area_hit_probability: float = 1.0,
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
        initial_state: Optional[np.ndarray] = None,
        opponent_policy: OpponentPolicy = OpponentPolicy.EVADE,
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
            evasion_speed=evasion_speed,
            dangerous_areas=dangerous_areas,
            dangerous_area_radius=dangerous_area_radius,
            dangerous_area_penalty=dangerous_area_penalty,
            dangerous_area_hit_probability=dangerous_area_hit_probability,
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
            initial_state=initial_state,
            opponent_policy=opponent_policy,
        )

        # Override space info to discrete actions
        self.space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.CONTINUOUS,
        )

        self.actions: List[str] = ["up", "down", "right", "left", "tag"]
        # Cache action vectors as C-contiguous float64 arrays so the hot
        # path can hand them straight to the C++ kernels with no
        # ``np.asarray(...).ravel()`` repacking.
        self.action_to_vector: Dict[str, np.ndarray] = {
            "up": np.ascontiguousarray([0.0, 1.0, 0.0], dtype=np.float64),
            "down": np.ascontiguousarray([0.0, -1.0, 0.0], dtype=np.float64),
            "right": np.ascontiguousarray([1.0, 0.0, 0.0], dtype=np.float64),
            "left": np.ascontiguousarray([-1.0, 0.0, 0.0], dtype=np.float64),
            "tag": np.ascontiguousarray([0.0, 0.0, 1.0], dtype=np.float64),
        }

    def get_actions(self) -> List[str]:
        return self.actions

    # Hot-path overrides: the discrete wrapper used to delegate every
    # method to its parent via ``super().method(self.action_to_vector[action])``.
    # That ``super()`` frame plus the dict lookup added ~50ms of pure
    # attribute-traversal overhead per second on the PFT-DPW profile. We
    # inline the cached-kernel calls directly instead, dropping the
    # ``super()`` frame and skipping any input renormalisation since the
    # cached action vectors are already C-contiguous float64.

    def sample_next_state(self, state: np.ndarray, action: Any, n_samples: int = 1) -> Any:
        action_arr = self.action_to_vector[action]
        kernel = self._get_trans_kernel(action_arr)
        kernel.set_state(state)
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def sample_observation(self, next_state: np.ndarray, action: Any, n_samples: int = 1) -> Any:
        action_arr = self.action_to_vector[action]
        kernel = self._get_obs_kernel(action_arr)
        kernel.set_next_state(next_state)
        samples = kernel.sample(n_samples)
        if n_samples == 1:
            return samples[0]
        return samples

    def transition_log_probability(
        self, state: np.ndarray, action: Any, next_states: Any
    ) -> np.ndarray:
        return ContinuousLaserTagPOMDP.transition_log_probability(
            self, state, self.action_to_vector[action], next_states
        )

    def observation_log_probability(
        self, next_state: np.ndarray, action: Any, observations: Any
    ) -> np.ndarray:
        return ContinuousLaserTagPOMDP.observation_log_probability(
            self, next_state, self.action_to_vector[action], observations
        )

    def observation_log_probability_single(
        self, next_state: Any, action: Any, observation: Any
    ) -> float:
        return ContinuousLaserTagPOMDP.observation_log_probability_single(
            self, next_state, self.action_to_vector[action], observation
        )

    def sample_next_state_batch(self, states: Any, action: Any) -> np.ndarray:
        return ContinuousLaserTagPOMDP.sample_next_state_batch(
            self, states, self.action_to_vector[action]
        )

    def observation_log_probability_per_state(
        self, next_states: Any, action: Any, observation: Any
    ) -> np.ndarray:
        return ContinuousLaserTagPOMDP.observation_log_probability_per_state(
            self, next_states, self.action_to_vector[action], observation
        )

    def reward(self, state: np.ndarray, action: Any, next_state: Any = None) -> float:
        return ContinuousLaserTagPOMDP.reward(
            self, state, self.action_to_vector[action], next_state=next_state
        )

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: Any,
        next_states: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    ) -> np.ndarray:
        return ContinuousLaserTagPOMDP.reward_batch(
            self, states, self.action_to_vector[action], next_states=next_states
        )

    def simulate_random_rollout(
        self,
        state: np.ndarray,
        action_sampler: Any,
        max_depth: int,
        discount_factor: float,
        depth: int = 0,
    ) -> float:
        if self.dangerous_area_hit_probability < 1.0:
            return python_random_rollout(
                state=state,
                depth=depth,
                action_sampler=action_sampler,
                environment=self,
                discount_factor=discount_factor,
                max_depth=max_depth,
            )
        steps_left = max_depth - depth
        if steps_left <= 0 or bool(state[4]):
            return 0.0
        actions_buffer = self._sample_discrete_action_buffer(action_sampler, steps_left)
        return self._native_rollout(state, actions_buffer, depth, max_depth, discount_factor)

    def _sample_discrete_action_buffer(self, action_sampler: Any, n_steps: int) -> np.ndarray:
        action_sample = action_sampler.sample
        rows = []
        for _ in range(n_steps):
            action_str = action_sample()
            rows.append(self.action_to_vector[action_str])
        return np.ascontiguousarray(np.stack(rows, axis=0))

    def _count_episode_metrics(self, steps: List[StepData]) -> Tuple[int, int, int]:
        converted = []
        for step in steps:
            if step.action is not None and isinstance(step.action, str):
                converted.append(step._replace(action=self.action_to_vector[step.action]))
            else:
                converted.append(step)
        return super()._count_episode_metrics(converted)

    def hash_action(self, action: Any) -> Hashable:
        # Discrete-action variant: actions are str labels (e.g. "up").
        return action


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
