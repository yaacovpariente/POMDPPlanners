# SPDX-License-Identifier: MIT

import random
from bisect import bisect
from collections.abc import Hashable
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.environment import DiscreteActionsEnvironment
from POMDPPlanners.core.simulation import History, MetricValue
from POMDPPlanners.environments.light_dark_pomdp import (
    _native,  # pylint: disable=no-name-in-module
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.base_light_dark_pomdp import (
    BaseLightDarkPOMDPDiscreteActions,
)
from POMDPPlanners.planners.planners_utils.rollout import python_random_rollout
from POMDPPlanners.utils.statistics_utils import confidence_interval


class DiscreteLightDarkPOMDPMetrics(Enum):
    """Metric names for Discrete Light-Dark POMDP environment."""

    GOAL_REACHING_RATE = "goal_reaching_rate"
    OBSTACLE_HIT_RATE = "obstacle_hit_rate"
    AVG_OBSTACLE_HIT_COUNTER = "avg_obstacle_hit_counter"
    OUT_OF_GRID_RATE = "out_of_grid_rate"
    AVG_ZERO_MEAN_HAZARD_SHOCK_COUNTER = "avg_high_variance_states_counter"


class ObservationModelType(Enum):
    NORMAL = "normal"
    NO_OBS_IN_DARK = "no_obs_in_dark"
    DISTANCE_BASED = "distance_based"


class DiscreteLightDarkPOMDP(BaseLightDarkPOMDPDiscreteActions, DiscreteActionsEnvironment):
    """Discrete Light-Dark POMDP Environment for Robot Navigation with Observation Uncertainty.

    This environment implements a discretized version of the classic Light-Dark POMDP problem,
    where a robot must navigate from a start position to a goal position in a grid world
    with beacons and obstacles. The key challenge is that the robot's observation quality
    depends on its distance from beacons - closer to beacons means more accurate observations.

    Problem Description:
    The robot operates in a discrete grid world where it can move in four cardinal directions.
    The environment includes:
    - Beacons: Fixed positions that provide location reference with varying accuracy
    - Obstacles: Grid cells that incur penalties when hit
    - Goal: Target position that provides high reward when reached
    - Observation uncertainty: Decreases with proximity to beacons (light areas)

    Key Features:
    - Discrete state space: Robot positions are restricted to grid cells
    - Discrete action space: North, South, East, West movements
    - Multiple observation models available (normal, no observation in dark)
    - Distance-dependent observation accuracy: Closer to beacons = better observations
    - Stochastic transitions: Actions may fail with configurable probability
    - Obstacle avoidance: Penalties for hitting obstacles during navigation
    - Configurable environment parameters: Grid size, beacon positions, obstacles

    State Space:
    - 2D grid coordinates (x, y) representing robot position
    - Bounded by grid_size parameter (default: 11x11 grid)

    Action Space:
    - Discrete actions: ['North', 'South', 'East', 'West']
    - Each action moves robot one grid cell in the corresponding direction
    - Boundary conditions: Actions that would move outside grid are blocked

    Observation Space:
    - Discrete observations based on beacon proximity and noise
    - Observation accuracy improves with proximity to beacons
    - Stochastic observation errors controlled by observation_error_prob

    Reward Structure:
    - Goal reward: Large positive reward for reaching the goal state
    - Obstacle penalty: Negative reward for hitting obstacles
    - Fuel cost: Small negative reward for each movement action
    - Distance-based penalties: Encourage efficient navigation

    Attributes:
        transition_error_prob: Probability that an action fails (results in different movement)
        observation_error_prob: Probability of observation noise/error
        is_stochastic_reward: Whether rewards include stochastic components
        beacons: List of (x, y) beacon positions that provide navigation references
        goal_state: Target position (x, y) that robot should reach
        start_state: Initial robot position (x, y)
        obstacles: List of (x, y) obstacle positions to avoid
        grid_size: Dimension of the square grid world

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> env = DiscreteLightDarkPOMDP(
        ...     discount_factor=0.95,
        ...     transition_error_prob=0.1,
        ...     observation_error_prob=0.15,
        ...     beacons=[(1, 1), (2, 2)],
        ...     grid_size=11
        ... )
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

    References:
    - Platt, R., et al. "Belief space planning assuming maximum likelihood observations." (2010)
    - Kurniawati, H., et al. "SARSOP: Efficient point-based POMDP planning by approximating optimally reachable belief spaces." (2008)
    - Light-Dark domain: Classic POMDP benchmark for testing observation uncertainty
    """

    def __init__(
        self,
        discount_factor: float,
        name: str = "DiscreteLightDarkPOMDP",
        transition_error_prob: float = 0.05,
        observation_error_prob: float = 0.05,
        beacons: List[Tuple[float, float]] = [
            (0, 0),
            (0, 5),
            (0, 10),
            (5, 0),
            (5, 5),
            (5, 10),
            (10, 0),
            (10, 5),
            (10, 10),
        ],
        goal_state: np.ndarray = np.array([10, 5]),
        start_state: np.ndarray = np.array([0, 5]),
        obstacles: List[Tuple[float, float]] = [(3, 7), (5, 5)],
        obstacle_hit_probability: float = 0.2,
        obstacle_reward: float = -10.0,
        goal_reward: float = 10.0,
        beacon_radius: float = 1.0,
        fuel_cost: float = 2.0,
        grid_size: int = 11,
        is_stochastic_reward: bool = True,
        observation_model_type: ObservationModelType = ObservationModelType.NORMAL,
    ):
        self.transition_error_prob = transition_error_prob
        self.observation_error_prob = observation_error_prob
        self.is_stochastic_reward = is_stochastic_reward
        self.observation_model_type = observation_model_type

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            is_discrete_observations=True,
            beacons=beacons,
            goal_state=goal_state,
            start_state=start_state,
            obstacles=obstacles,
            obstacle_hit_probability=obstacle_hit_probability,
            obstacle_reward=obstacle_reward,
            goal_reward=goal_reward,
            beacon_radius=beacon_radius,
            fuel_cost=fuel_cost,
            grid_size=grid_size,
        )

        self._precompute_sampling_tables()

    def _precompute_sampling_tables(self) -> None:
        n_actions = len(self.actions)
        # Precompute transition cumulative sums per action for bisect sampling
        self._transition_probs = {}
        self._transition_cum: dict[str, list[float]] = {}
        # Wrapper-equivalent cumulative-probability arrays for np.searchsorted
        # (matches DiscreteDistribution(values, probs)._cumprobs = np.cumsum(probs))
        self._transition_cumprobs_np: dict[str, np.ndarray] = {}
        for i, act in enumerate(self.actions):
            probs = np.ones(n_actions) * (self.transition_error_prob / (n_actions - 1))
            probs[i] = 1 - self.transition_error_prob
            probs[0] += 1 - probs.sum()
            self._transition_probs[act] = probs
            cum = []
            running = 0.0
            for p in probs:
                running += float(p)
                cum.append(running)
            self._transition_cum[act] = cum
            self._transition_cumprobs_np[act] = np.cumsum(probs)

        # Precompute action vectors as a list for index-based access
        self._action_vectors = [self.action_to_vector[a] for a in self.actions]

        # Precompute observation cumulative sums (near vs far beacon)
        n_obs = n_actions + 1
        self._n_obs_values = n_obs

        near_error = self.observation_error_prob * 0.2
        obs_probs_near = np.ones(n_obs) * (near_error / (n_obs - 1))
        obs_probs_near[-1] = 1 - near_error
        self._obs_probs_near = obs_probs_near

        far_error = self.observation_error_prob * 1.0
        obs_probs_far = np.ones(n_obs) * (far_error / (n_obs - 1))
        obs_probs_far[-1] = 1 - far_error
        self._obs_probs_far = obs_probs_far

        self._obs_cum_near = list(np.cumsum(obs_probs_near))
        self._obs_cum_far = list(np.cumsum(obs_probs_far))
        # Wrapper-equivalent np.cumsum arrays for np.searchsorted RNG path
        self._obs_cumprobs_near_np = np.cumsum(obs_probs_near)
        self._obs_cumprobs_far_np = np.cumsum(obs_probs_far)

        # Precompute reward helpers: obstacle positions as set of tuples for O(1) lookup
        self._obstacle_tuples = {
            (int(self.obstacles[0, j]), int(self.obstacles[1, j]))
            for j in range(self.obstacles.shape[1])
        }
        self._goal_x = int(self.goal_state[0])
        self._goal_y = int(self.goal_state[1])
        self._beacon_radius_sq = self.beacon_radius * self.beacon_radius

        # Precompute beacon positions as list of (x, y) tuples for pure Python loop
        self._beacon_tuples = [
            (float(self.beacons[0, j]), float(self.beacons[1, j]))
            for j in range(self.beacons.shape[1])
        ]
        self._n_actions = n_actions

        # Pre-built buffers for native discrete kernels.
        # _beacons_flat: shape (2 * n_beacons,) interleaved [x0,y0,x1,y1,...].
        # _obstacles_flat: shape (2 * n_obstacles,) interleaved.
        # _action_offsets_array: shape (n_actions, 2) stacked offset vectors.
        # _goal_state_f64: shape (2,) goal as float64.
        # _actions_array_f64: shape (n_actions, 2) — float64 view of action vectors,
        #   suitable for native discrete_simulate_rollout.
        self._beacons_flat: np.ndarray = np.ascontiguousarray(
            self.beacons.T.ravel(), dtype=np.float64
        )
        self._obstacles_flat: np.ndarray = np.ascontiguousarray(
            self.obstacles.T.ravel(), dtype=np.float64
        )
        self._action_offsets_array: np.ndarray = np.ascontiguousarray(
            np.stack(
                [np.asarray(self._action_vectors[i], dtype=np.float64) for i in range(n_actions)],
                axis=0,
            ),
            dtype=np.float64,
        )
        self._goal_state_f64: np.ndarray = np.ascontiguousarray(self.goal_state, dtype=np.float64)
        self._actions_array_f64: np.ndarray = self._action_offsets_array
        # Native log-prob fast path requires float64 prob tables.
        self._obs_probs_near_f64: np.ndarray = np.ascontiguousarray(
            self._obs_probs_near, dtype=np.float64
        )
        self._obs_probs_far_f64: np.ndarray = np.ascontiguousarray(
            self._obs_probs_far, dtype=np.float64
        )

    def sample_next_step(self, state: np.ndarray, action: Any) -> Tuple[Any, Any, float]:
        if self.observation_model_type != ObservationModelType.NORMAL:
            return super().sample_next_step(state, action)

        # Inline state transition — bisect on precomputed cumsums
        chosen_idx = bisect(self._transition_cum[action], random.random())
        next_state = state + self._action_vectors[chosen_idx]

        # Inline observation — pure Python beacon check + bisect sampling
        sx = float(next_state[0])
        sy = float(next_state[1])
        br_sq = self._beacon_radius_sq
        near_beacon = any(
            (sx - bx) * (sx - bx) + (sy - by) * (sy - by) < br_sq for bx, by in self._beacon_tuples
        )
        obs_cum = self._obs_cum_near if near_beacon else self._obs_cum_far
        obs_idx = bisect(obs_cum, random.random())
        if obs_idx < self._n_actions:
            observation = next_state + self._action_vectors[obs_idx]
        else:
            observation = next_state

        # Inline reward — pure Python math avoids numpy overhead on 2-element
        # arrays. Score against the realised ``next_state`` rather than the
        # intended ``state + action`` so transition-error draws are reflected
        # in the obstacle / goal / out-of-grid checks.
        reward = self._compute_reward_fast(state, next_state)
        return next_state, observation, reward

    # ── Helpers for non-NORMAL observation models (inlined post-PR-D) ───────
    # These replace the previously-routed-through factory the env used to
    # build per-call wrappers (DiscreteLDObservationModelNoObsInDark and
    # DiscreteLDDistanceBasedObservationModel). The math is preserved
    # bit-for-bit with the deleted wrapper implementations.

    def _is_near_beacon(self, next_state: np.ndarray) -> bool:
        # Strict-less-than match for parity with the deleted wrapper helper
        # ``BaseDiscreteLightDarkObservationModel._near_beacon``.
        distances = np.linalg.norm(self.beacons - next_state[:, np.newaxis], axis=0)
        return float(np.min(distances)) < self.beacon_radius

    def _min_distance_to_beacon(self, next_state: np.ndarray) -> float:
        distances = np.linalg.norm(self.beacons - next_state[:, np.newaxis], axis=0)
        return float(np.min(distances))

    def _distance_based_error_factor(self, min_distance: float) -> float:
        # Continuous scaling: factor = min_factor + (1 - min_factor) * (d / beacon_radius).
        # Mirrors DiscreteLDDistanceBasedObservationModel._create_distribution.
        min_factor = 0.0001
        if self.beacon_radius > 0:
            distance_ratio = min_distance / self.beacon_radius
            return min_factor + (1.0 - min_factor) * distance_ratio
        return 1.0

    def _scaled_obs_probs(self, error_factor: float) -> np.ndarray:
        n_obs = self._n_obs_values
        error_prob = self.observation_error_prob * error_factor
        probs = np.ones(n_obs) * (error_prob / (n_obs - 1))
        probs[-1] = 1.0 - error_prob
        return probs

    def _sample_from_obs_probs(self, next_state: np.ndarray, probs: np.ndarray, n_samples: int):
        # Mirrors DiscreteDistribution.sample(): draw uniform, np.searchsorted
        # on np.cumsum(probs), clip to last index, map index < n_actions to
        # next_state + dir, else next_state.
        cumprobs = np.cumsum(probs)
        n_obs = self._n_obs_values
        if n_samples == 1:
            idx = int(np.searchsorted(cumprobs, np.random.rand()))
            if idx >= n_obs:
                idx = n_obs - 1
            if idx < self._n_actions:
                return [next_state + self._action_vectors[idx]]
            return [next_state]
        draws = np.random.rand(n_samples)
        idxs = np.searchsorted(cumprobs, draws)
        idxs = np.clip(idxs, 0, n_obs - 1)
        out: List[Any] = []
        for idx in idxs:
            if idx < self._n_actions:
                out.append(next_state + self._action_vectors[idx])
            else:
                out.append(next_state)
        return out

    def _sample_observation_non_normal(self, next_state: np.ndarray, n_samples: int):
        # Returns None for NORMAL (caller proceeds with the inlined path) or
        # the unwrapped sample value(s) for NO_OBS_IN_DARK / DISTANCE_BASED.
        if self.observation_model_type == ObservationModelType.NO_OBS_IN_DARK:
            samples = self._sample_no_obs_in_dark(next_state, n_samples)
        elif self.observation_model_type == ObservationModelType.DISTANCE_BASED:
            samples = self._sample_distance_based(next_state, n_samples)
        else:
            return None
        return samples[0] if n_samples == 1 else samples

    def _sample_no_obs_in_dark(self, next_state: np.ndarray, n_samples: int) -> List[Any]:
        # Mirrors DiscreteLDObservationModelNoObsInDark.sample().
        if self._is_near_beacon(next_state):
            # Same probabilities as NORMAL near-beacon path
            # (beacon_error_factor=0.2 baked into self._obs_probs_near).
            return self._sample_from_obs_probs(next_state, self._obs_probs_near, n_samples)
        return ["None"] * n_samples

    def _sample_distance_based(self, next_state: np.ndarray, n_samples: int) -> List[Any]:
        # Mirrors DiscreteLDDistanceBasedObservationModel.sample().
        min_distance = self._min_distance_to_beacon(next_state)
        if min_distance > self.beacon_radius:
            return ["None"] * n_samples
        error_factor = self._distance_based_error_factor(min_distance)
        probs = self._scaled_obs_probs(error_factor)
        return self._sample_from_obs_probs(next_state, probs, n_samples)

    def _log_prob_from_obs_probs(
        self, next_state: np.ndarray, observations: Any, probs_vec: np.ndarray
    ) -> np.ndarray:
        # Mirrors DiscreteDistribution.probability(): exact-match candidate
        # search; out is np.log(p) for matched candidates, -inf otherwise.
        candidates = [next_state + self._action_vectors[i] for i in range(self._n_actions)]
        candidates.append(next_state)
        candidates_array = np.stack(candidates, axis=0)
        observations_array = np.asarray(observations)
        if observations_array.ndim == 1:
            observations_array = observations_array.reshape(1, -1)
        out = np.full(len(observations_array), -np.inf, dtype=np.float64)
        for i, obs in enumerate(observations_array):
            for j, candidate in enumerate(candidates_array):
                if np.array_equal(obs, candidate):
                    p = float(probs_vec[j])
                    out[i] = np.log(p) if p > 0.0 else -np.inf
                    break
        return out

    def _log_prob_no_obs_in_dark(self, next_state: np.ndarray, observations: Any) -> np.ndarray:
        # Mirrors DiscreteLDObservationModelNoObsInDark.probability() with
        # log conversion applied at the end.
        observations_list = self._normalize_observations_list(observations)
        if self._is_near_beacon(next_state):
            # Map "None" → -inf, real obs → log(p) from near-beacon dist.
            probs_vec = self._obs_probs_near
            return self._log_prob_dispatch(next_state, observations_list, probs_vec, near=True)
        # Far: "None" has probability 1 (log 0), real obs probability 0 (-inf).
        return self._log_prob_dispatch(next_state, observations_list, None, near=False)

    def _log_prob_distance_based(self, next_state: np.ndarray, observations: Any) -> np.ndarray:
        # Mirrors DiscreteLDDistanceBasedObservationModel.probability() with
        # log conversion applied at the end.
        observations_list = self._normalize_observations_list(observations)
        min_distance = self._min_distance_to_beacon(next_state)
        if min_distance > self.beacon_radius:
            return self._log_prob_dispatch(next_state, observations_list, None, near=False)
        error_factor = self._distance_based_error_factor(min_distance)
        probs_vec = self._scaled_obs_probs(error_factor)
        return self._log_prob_dispatch(next_state, observations_list, probs_vec, near=True)

    def _normalize_observations_list(self, observations: Any) -> list:
        # The non-NORMAL paths must accept lists with mixed "None" strings
        # and ndarray observations, matching the wrapper's
        # ``probability(values: List[Union[Any, str]])`` signature.
        if isinstance(observations, np.ndarray):
            if observations.ndim == 1:
                return [observations]
            return list(observations)
        return list(observations)

    def _log_prob_dispatch(
        self,
        next_state: np.ndarray,
        observations_list: list,
        probs_vec: Any,
        near: bool,
    ) -> np.ndarray:
        out = np.full(len(observations_list), -np.inf, dtype=np.float64)
        if near:
            # probs_vec is non-None here.
            candidates = [next_state + self._action_vectors[i] for i in range(self._n_actions)]
            candidates.append(next_state)
            for i, value in enumerate(observations_list):
                if isinstance(value, str) and value == "None":
                    # "None" has probability 0 → -inf.
                    continue
                value_arr = np.asarray(value)
                for j, cand in enumerate(candidates):
                    if np.array_equal(value_arr, cand):
                        p = float(probs_vec[j])
                        out[i] = np.log(p) if p > 0.0 else -np.inf
                        break
            return out
        # Far branch: only "None" is possible (log 1 = 0), all else -inf.
        for i, value in enumerate(observations_list):
            if isinstance(value, str) and value == "None":
                out[i] = 0.0
        return out

    def _compute_reward_fast(self, state: np.ndarray, next_state: np.ndarray) -> float:
        # Score the obstacle / goal / out-of-grid checks against the realised
        # ``next_state`` (threaded from :meth:`sample_next_step`) so the
        # reward and trajectory agree on the same draw under transition
        # stochasticity.
        del state  # state is unused on this fast path; kept for signature parity
        nx = int(next_state[0])
        ny = int(next_state[1])

        dx = nx - self._goal_x
        dy = ny - self._goal_y
        dist_to_goal = (dx * dx + dy * dy) ** 0.5

        reward = -self.fuel_cost - dist_to_goal

        if dx == 0 and dy == 0:
            reward += self.goal_reward
        elif (nx, ny) in self._obstacle_tuples:
            # Use numpy RNG to stay on the same stream as ``reward()`` —
            # mixing ``random`` and ``np.random`` would split the obstacle
            # Bernoulli into two independent draws, so ``sample_next_step``
            # and ``reward(state, action, next_state)`` would disagree on
            # the same triple.
            if np.random.rand() < self.obstacle_hit_probability:
                reward += self.obstacle_reward
        elif nx < 0 or ny < 0 or nx > self.grid_size or ny > self.grid_size:
            reward += self.obstacle_reward

        return reward

    def sample_next_state(self, state: np.ndarray, action: Any, n_samples: int = 1) -> Any:
        # Inlined wrapper-equivalent of
        # state_transition_model(state, action).sample()[0].
        # Wrapper builds DiscreteDistribution(values, probs) with
        # _cumprobs = np.cumsum(probs); .sample() does
        # idx = int(np.searchsorted(_cumprobs, np.random.rand())) clamped
        # to len(values)-1. Same RNG draw is preserved here.
        cumprobs = self._transition_cumprobs_np[action]
        if n_samples == 1:
            uniform_draw = float(np.random.rand())
            if hasattr(_native, "discrete_sample_next_state_step"):
                return _native.discrete_sample_next_state_step(
                    state=np.ascontiguousarray(state, dtype=np.float64),
                    cumprobs_for_action=np.ascontiguousarray(cumprobs, dtype=np.float64),
                    action_vectors=self._action_offsets_array,
                    uniform_draw=uniform_draw,
                    n_actions=self._n_actions,
                )
            idx = int(np.searchsorted(cumprobs, uniform_draw))
            if idx >= self._n_actions:
                idx = self._n_actions - 1
            return state + self._action_vectors[idx]
        # Vectorize: draw N uniforms and N searchsorted calls, then
        # gather the per-row offset vectors. Preserves RNG draw count
        # (one np.random.rand() per sample, in order).
        draws = np.random.rand(n_samples)
        idxs = np.searchsorted(cumprobs, draws)
        idxs = np.clip(idxs, 0, self._n_actions - 1)
        offsets = np.stack([self._action_vectors[i] for i in idxs], axis=0)
        return state + offsets

    def sample_observation(self, next_state: np.ndarray, action: Any, n_samples: int = 1) -> Any:
        # Dispatch on observation_model_type. NORMAL stays inlined here for
        # the original hot-path; NO_OBS_IN_DARK and DISTANCE_BASED route
        # through helpers that mirror the deleted wrapper sampling logic
        # bit-for-bit. ``action`` is unused (reads only next_state) but
        # part of the env-API signature.
        del action
        non_normal = self._sample_observation_non_normal(next_state, n_samples)
        if non_normal is not None:
            return non_normal

        # NORMAL path — wrapper-equivalent near-beacon check (strict-less-than):
        # distances = np.linalg.norm(beacons - next_state[:, None], axis=0)
        # near_beacon = float(np.min(distances)) < beacon_radius.
        if n_samples == 1 and hasattr(_native, "discrete_sample_observation_step_normal"):
            uniform_draw = float(np.random.rand())
            return _native.discrete_sample_observation_step_normal(
                next_state=np.ascontiguousarray(next_state, dtype=np.float64),
                beacons=self._beacons_flat,
                cumprobs_near=self._obs_cumprobs_near_np,
                cumprobs_far=self._obs_cumprobs_far_np,
                action_vectors=self._action_offsets_array,
                beacon_radius=float(self.beacon_radius),
                uniform_draw=uniform_draw,
                n_actions=self._n_actions,
                n_obs=self._n_obs_values,
            )

        distances = np.linalg.norm(self.beacons - next_state[:, np.newaxis], axis=0)
        near_beacon = float(np.min(distances)) < self.beacon_radius

        cumprobs = self._obs_cumprobs_near_np if near_beacon else self._obs_cumprobs_far_np
        n_obs = self._n_obs_values

        if n_samples == 1:
            # Wrapper builds values = [next_state + dir for dir in dirs] + [next_state],
            # then DiscreteDistribution.sample() draws np.random.rand() and
            # np.searchsorted; same draw replicated here.
            idx = int(np.searchsorted(cumprobs, np.random.rand()))
            if idx >= n_obs:
                idx = n_obs - 1
            if idx < self._n_actions:
                return next_state + self._action_vectors[idx]
            return next_state

        # Vectorized path: draw N uniforms in order so RNG sequence is
        # identical to N successive single-sample draws.
        draws = np.random.rand(n_samples)
        idxs = np.searchsorted(cumprobs, draws)
        idxs = np.clip(idxs, 0, n_obs - 1)
        observations = []
        for idx in idxs:
            if idx < self._n_actions:
                observations.append(next_state + self._action_vectors[idx])
            else:
                observations.append(next_state)
        return np.stack(observations, axis=0)

    def transition_log_probability(
        self, state: np.ndarray, action: Any, next_states: Any
    ) -> np.ndarray:
        action_index = self.actions.index(action)
        probs = self._transition_probs[action]
        # Wrapper-equivalent values list: state + action_to_vector[a] for
        # each action, in self.actions order. Match by exact equality.
        candidates = np.stack(
            [state + self._action_vectors[i] for i in range(self._n_actions)], axis=0
        )
        next_states_array = np.asarray(next_states)
        if next_states_array.ndim == 1:
            next_states_array = next_states_array.reshape(1, -1)
        out = np.full(len(next_states_array), -np.inf, dtype=np.float64)
        for i, ns in enumerate(next_states_array):
            for j, candidate in enumerate(candidates):
                if np.array_equal(ns, candidate):
                    p = float(probs[j])
                    out[i] = np.log(p) if p > 0.0 else -np.inf
                    break
        # action_index is referenced for documentation parity with
        # state_transition_model; the per-candidate match above already
        # uses the action-specific probs vector.
        del action_index
        return out

    def observation_log_probability(
        self, next_state: np.ndarray, action: Any, observations: Any
    ) -> np.ndarray:
        del action  # unused; observation log-prob depends only on next_state
        if self.observation_model_type == ObservationModelType.NO_OBS_IN_DARK:
            return self._log_prob_no_obs_in_dark(next_state, observations)
        if self.observation_model_type == ObservationModelType.DISTANCE_BASED:
            return self._log_prob_distance_based(next_state, observations)

        # NORMAL — delegate to native kernel. The kernel applies the same
        # strict-less-than near-beacon predicate, exact-equality candidate
        # match, and ``log(p) for p>0 else -inf`` rules as the Python path.
        observations_array = np.ascontiguousarray(np.asarray(observations, dtype=np.float64))
        next_state_f64 = np.ascontiguousarray(np.asarray(next_state, dtype=np.float64))
        return _native.discrete_observation_log_prob(
            next_state=next_state_f64,
            observations=observations_array,
            beacons=self._beacons_flat,
            beacon_radius=float(self.beacon_radius),
            obs_probs_near=self._obs_probs_near_f64,
            obs_probs_far=self._obs_probs_far_f64,
            action_offsets=self._action_offsets_array,
        )

    def reward(self, state: np.ndarray, action: Any, next_state: Any = None) -> float:
        if state.shape != (2,):
            raise ValueError("state must be a 2D vector")

        # Honour the realised ``next_state`` threaded by
        # :meth:`Environment.sample_next_step` so the obstacle / goal /
        # out-of-grid checks score against the same draw as the trajectory.
        # Only resample locally when no realised next_state was supplied.
        if next_state is None:
            next_state_local = state + self.action_to_vector[action]
        else:
            next_state_local = np.asarray(next_state)

        is_goal_state = np.all(next_state_local == self.goal_state)
        is_obstacle_hit = np.any(np.all(next_state_local.reshape(-1, 1) == self.obstacles, axis=0))
        is_out_of_grid = np.any(next_state_local < 0) or np.any(next_state_local > self.grid_size)

        # Start with base reward (fuel cost)
        reward = -self.fuel_cost - np.linalg.norm(next_state_local - self.goal_state)

        if is_goal_state:
            reward += self.goal_reward
        elif is_obstacle_hit:
            if np.random.rand() < self.obstacle_hit_probability:
                reward += self.obstacle_reward
        elif is_out_of_grid:
            reward += self.obstacle_reward

        return float(reward)

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: str,
        next_states: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    ) -> np.ndarray:
        states = np.asarray(states)
        # Honour per-row realised ``next_states`` threaded by callers so
        # obstacle / goal / out-of-grid checks score against the realised
        # trajectory. Only synthesise the deterministic-action next_states
        # when nothing was supplied (legacy ``reward_batch(states, action)``
        # call site contract).
        if next_states is None:
            next_states_arr = states + self.action_to_vector[action]
        else:
            next_states_arr = np.asarray(next_states)
        dists_to_goal = np.linalg.norm(next_states_arr - self.goal_state, axis=1)
        rewards = -self.fuel_cost - dists_to_goal

        goal_mask = np.all(next_states_arr == self.goal_state, axis=1)
        rewards[goal_mask] += self.goal_reward

        obs_match = np.all(
            next_states_arr[:, :, np.newaxis] == self.obstacles[np.newaxis, :, :],
            axis=1,
        )
        in_obstacle = np.any(obs_match, axis=1)
        obstacle_mask = in_obstacle & ~goal_mask
        n_obs = int(np.sum(obstacle_mask))
        if n_obs > 0:
            hits = np.random.rand(n_obs) < self.obstacle_hit_probability
            rewards[obstacle_mask] += np.where(hits, self.obstacle_reward, 0.0)

        oob = np.any(next_states_arr < 0, axis=1) | np.any(next_states_arr > self.grid_size, axis=1)
        rewards[oob & ~goal_mask & ~in_obstacle] += self.obstacle_reward
        return rewards

    def sample_next_state_batch(
        self, states: Union[np.ndarray, Sequence[Any]], action: str
    ) -> np.ndarray:
        states_arr = np.asarray(states)
        n = len(states_arr)
        cumprobs = self._transition_cumprobs_np[action]
        draws = np.random.rand(n)
        idxs = np.searchsorted(cumprobs, draws)
        idxs = np.clip(idxs, 0, self._n_actions - 1)
        offsets = np.stack([self._action_vectors[i] for i in idxs], axis=0)
        return states_arr + offsets

    def observation_log_probability_per_state(
        self,
        next_states: Union[np.ndarray, Sequence[Any]],
        action: Any,
        observation: np.ndarray,
    ) -> np.ndarray:
        if self.observation_model_type != ObservationModelType.NORMAL:
            return super().observation_log_probability_per_state(
                next_states=next_states, action=action, observation=observation
            )
        del action  # unused; per-state log-prob reads only next_states + obs
        next_states_arr = np.ascontiguousarray(np.asarray(next_states, dtype=np.float64))
        if next_states_arr.ndim == 1:
            next_states_arr = next_states_arr.reshape(1, -1)
        observation_arr = np.ascontiguousarray(np.asarray(observation, dtype=np.float64))
        return _native.discrete_observation_log_prob_per_state(
            next_states=next_states_arr,
            observation=observation_arr,
            beacons=self._beacons_flat,
            beacon_radius=float(self.beacon_radius),
            obs_probs_near=self._obs_probs_near_f64,
            obs_probs_far=self._obs_probs_far_f64,
            action_offsets=self._action_offsets_array,
        )

    def is_terminal(self, state: np.ndarray) -> bool:
        if state.shape != (2,):
            raise ValueError("state must be a 2D vector")
        # Native fast-path: state-equals-goal OR state-in-any-obstacle (exact
        # float equality, matching the Python np.all/np.any-on-equality rules).
        return _native.discrete_is_terminal(
            state=np.ascontiguousarray(state, dtype=np.float64),
            goal_state=self._goal_state_f64,
            obstacles=self._obstacles_flat,
        )

    def simulate_random_rollout(
        self,
        state: Any,
        action_sampler: Any,
        max_depth: int,
        discount_factor: float,
        depth: int = 0,
    ) -> float:
        """Random rollout via native C++.

        Pre-draws the per-step action indices on the Python side (so the
        ``action_sampler`` interaction stays observable for tests / hooks)
        and forwards to the native discrete rollout kernel. The kernel uses
        the module-level C++ RNG for the per-step obstacle-hit and
        transition-error draws.

        Falls back to the base-class Python loop when the env is configured
        for a non-NORMAL observation model only if the rollout would
        otherwise short-circuit at the wrong place — actually rollout reward
        and dynamics are independent of the observation model, so the native
        path is safe for all observation models.

        Args:
            state: Current 2-D position ``[x, y]``.
            action_sampler: Object with a ``sample()`` method; used only for
                the Python fallback path. On the native path, action indices
                are pre-drawn via ``np.random.randint``.
            max_depth: Maximum rollout depth.
            discount_factor: Per-step discount factor.
            depth: Depth already consumed by the search tree. Defaults to 0.

        Returns:
            Discounted sum of immediate rewards along the sampled trajectory.
        """
        steps_left = max_depth - depth
        if steps_left <= 0:
            return 0.0

        # Stochastic-reward semantics: the native kernel draws the per-step
        # obstacle-hit Bernoulli with ``obstacle_hit_probability``. When
        # ``is_stochastic_reward`` is False the Python ``reward()`` path
        # would deterministically apply the obstacle penalty (rather than
        # drawing a Bernoulli) — fall back to Python in that case.
        # Also bypass when obstacles are configured AND transitions are
        # stochastic: the C++ kernel scores the obstacle penalty against
        # the *intended* action offset (state + action_vector), while the
        # Python ``reward()`` path now consults the realised next_state.
        # Until ``discrete_simulate_rollout`` is rebuilt to consume the
        # realised position, route around it whenever the divergence can
        # actually fire (``transition_error_prob > 0`` makes the realised
        # action differ from the intended one).
        bypass_native_for_realised_pos = (
            self.transition_error_prob > 0.0 and self._obstacles_flat.shape[0] > 0
        )
        if not self.is_stochastic_reward or bypass_native_for_realised_pos:
            return python_random_rollout(
                state=state,
                depth=depth,
                action_sampler=action_sampler,
                environment=self,
                discount_factor=discount_factor,
                max_depth=max_depth,
            )

        state_arr = np.ascontiguousarray(np.asarray(state, dtype=np.float64).ravel())
        action_indices = np.random.randint(0, len(self.actions), size=steps_left, dtype=np.int32)
        return _native.discrete_simulate_rollout(
            initial_state=state_arr,
            action_array=self._actions_array_f64,
            action_indices=action_indices,
            max_depth=max_depth,
            start_depth=depth,
            discount_factor=discount_factor,
            goal_state=self._goal_state_f64,
            obstacles=self._obstacles_flat,
            grid_size=float(self.grid_size),
            fuel_cost=float(self.fuel_cost),
            goal_reward=float(self.goal_reward),
            obstacle_reward=float(self.obstacle_reward),
            obstacle_hit_probability=float(self.obstacle_hit_probability),
            transition_error_prob=float(self.transition_error_prob),
        )

    def get_metric_names(self) -> List[str]:
        """Get names of Discrete Light-Dark POMDP specific metrics.

        Returns:
            List containing metric names: goal_reaching_rate, obstacle_hit_rate,
            avg_obstacle_hit_counter, out_of_grid_rate, and avg_high_variance_states_counter
        """
        return [metric.value for metric in DiscreteLightDarkPOMDPMetrics]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        goal_reached = []
        obstacle_hits = []
        obstacle_hit_counter = []
        out_of_grid = []
        high_variance_states_counter = []
        for history in histories:
            goal_reached_in_history = False
            obstacle_hit_in_history = False
            obstacle_hit_counter_in_history = 0
            out_of_grid_in_history = False
            out_of_grid_counter_in_history = 0

            for _, step in enumerate(history.history):
                if np.array_equal(step.state, self.goal_state):
                    goal_reached_in_history = True
                    break

                # Check if step hits an obstacle
                if np.any(np.all(step.state.reshape(-1, 1) == self.obstacles, axis=0)):
                    obstacle_hit_in_history = True
                    obstacle_hit_counter_in_history += 1

                # Check if step is out of grid
                is_out_of_grid = np.any(step.state < 0) or np.any(step.state > self.grid_size)
                if is_out_of_grid:
                    out_of_grid_in_history = True
                    out_of_grid_counter_in_history += 1

            goal_reached.append(1 if goal_reached_in_history else 0)
            obstacle_hits.append(1 if obstacle_hit_in_history else 0)
            obstacle_hit_counter.append(obstacle_hit_counter_in_history)
            out_of_grid.append(1 if out_of_grid_in_history else 0)
            # Sum obstacle hits and out-of-grid occurrences as high-variance states
            high_variance_states_counter.append(
                obstacle_hit_counter_in_history + out_of_grid_counter_in_history
            )

        avg_goal_reached = float(np.mean(goal_reached))
        avg_obstacle_hits = float(np.mean(obstacle_hits))
        avg_obstacle_hit_counter = float(np.mean(obstacle_hit_counter))
        avg_out_of_grid = float(np.mean(out_of_grid))
        avg_high_variance_states_counter = float(np.mean(high_variance_states_counter))
        goal_reached_ci = confidence_interval(data=goal_reached, confidence=0.95)
        obstacle_hits_ci = confidence_interval(data=obstacle_hits, confidence=0.95)
        obstacle_hit_counter_ci = confidence_interval(data=obstacle_hit_counter, confidence=0.95)
        out_of_grid_ci = confidence_interval(data=out_of_grid, confidence=0.95)
        high_variance_states_counter_ci = confidence_interval(
            data=high_variance_states_counter, confidence=0.95
        )

        return [
            MetricValue(
                name=DiscreteLightDarkPOMDPMetrics.GOAL_REACHING_RATE.value,
                value=avg_goal_reached,
                lower_confidence_bound=goal_reached_ci[0],
                upper_confidence_bound=goal_reached_ci[1],
            ),
            MetricValue(
                name=DiscreteLightDarkPOMDPMetrics.OBSTACLE_HIT_RATE.value,
                value=avg_obstacle_hits,
                lower_confidence_bound=obstacle_hits_ci[0],
                upper_confidence_bound=obstacle_hits_ci[1],
            ),
            MetricValue(
                name=DiscreteLightDarkPOMDPMetrics.AVG_OBSTACLE_HIT_COUNTER.value,
                value=avg_obstacle_hit_counter,
                lower_confidence_bound=obstacle_hit_counter_ci[0],
                upper_confidence_bound=obstacle_hit_counter_ci[1],
            ),
            MetricValue(
                name=DiscreteLightDarkPOMDPMetrics.OUT_OF_GRID_RATE.value,
                value=avg_out_of_grid,
                lower_confidence_bound=out_of_grid_ci[0],
                upper_confidence_bound=out_of_grid_ci[1],
            ),
            MetricValue(
                name=DiscreteLightDarkPOMDPMetrics.AVG_ZERO_MEAN_HAZARD_SHOCK_COUNTER.value,
                value=avg_high_variance_states_counter,
                lower_confidence_bound=high_variance_states_counter_ci[0],
                upper_confidence_bound=high_variance_states_counter_ci[1],
            ),
        ]

    def hash_action(self, action: Any) -> Hashable:
        # Discrete-action env: actions are str labels (e.g. "up").
        return action
