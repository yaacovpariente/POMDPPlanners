"""Continuous Light-Dark POMDP Environment Implementation.

This module implements the continuous Light-Dark domain, a classic POMDP benchmark
where an agent must navigate to a goal position in a continuous 2D space while
dealing with position-dependent observation noise.

The Continuous Light-Dark POMDP features:
- Continuous 2D state space representing agent position
- Discrete or continuous action space for movement
- Light source at a specific location that affects observation quality
- Observation noise that decreases closer to the light source
- Goal region that agent must reach to maximize reward
- Optional obstacles that cause negative rewards when hit

Key characteristics:
- State: [x, y] position in continuous 2D space
- Actions: Movement vectors or discrete directions
- Observations: Noisy position estimates (noise depends on distance from light)
- Rewards: Goal reaching bonus, movement costs, obstacle penalties
- Multiple reward model variants available

Classes:
    RewardModelType: Enumeration of available reward model types
    ContinuousLightDarkPOMDP: Main environment class
    ContinuousLightDarkPOMDPDiscreteActions: Discrete action variant
"""

import math
from collections.abc import Hashable
from enum import Enum
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import History, MetricValue
from POMDPPlanners.environments.light_dark_pomdp import (
    _native,  # pylint: disable=no-name-in-module
)
from POMDPPlanners.planners.planners_utils.rollout import python_random_rollout
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.base_light_dark_pomdp import (
    BaseLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.numba_kernels import (
    is_terminal_kernel,
)
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models import (
    BaseLightDarkRewardModel,
    ContinuousLDDangerousStatesRewardModel,
    ContinuousLightDarkDecayingHitProbabilityRewardModel,
    ContinuousLightDarkRewardModel,
)
from POMDPPlanners.utils.numba_kernels import (
    any_point_within_radius_kernel,
    min_distance_to_points_kernel,
    mvn_sample_2d_kernel,
)
from POMDPPlanners.utils.statistics_utils import confidence_interval


class ContinuousLightDarkPOMDPMetrics(Enum):
    """Metric names for Continuous Light-Dark POMDP environment."""

    GOAL_REACHING_RATE = "goal_reaching_rate"
    OBSTACLE_HIT_RATE = "obstacle_hit_rate"
    AVG_OBSTACLE_HIT_COUNTER = "avg_obstacle_hit_counter"
    OUT_OF_GRID_RATE = "out_of_grid_rate"
    AVG_DANGEROUS_STATES_COUNTER = "avg_dangerous_states_counter"


class RewardModelType(Enum):
    STANDARD = "standard"
    DECAYING_HIT_PROBABILITY = "decaying_hit_probability"
    DANGEROUS_STATES = "dangerous_states"


class ObservationModelType(Enum):
    NORMAL_NOISE = "normal_noise"
    NORMAL_NOISE_NO_OBS_IN_DARK = "normal_noise_no_obs_in_dark"
    DISTANCE_BASED = "distance_based"


class ContinuousLightDarkPOMDP(BaseLightDarkPOMDP):
    """Continuous Light-Dark POMDP environment with continuous actions.

    This environment extends the base Light-Dark problem to continuous 2D space
    with continuous action vectors. The agent navigates toward a goal while
    dealing with position-dependent observation noise and optional obstacles.

    Key features:
    - Continuous 2D state and action spaces
    - Light beacons reduce observation noise when nearby
    - Multiple observation models available (normal noise, normal noise with no observation in dark)
    - Multiple reward models available (standard, decaying hit probability, dangerous states)
    - Optional obstacles with configurable hit penalties
    - Terminal conditions for goal reaching, obstacle hits, and boundary violations

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> env = ContinuousLightDarkPOMDP(
        ...     discount_factor=0.95,
        ...     goal_state=np.array([10, 5]),
        ...     start_state=np.array([0, 5])
        ... )
        >>>
        >>> # Get initial state
        >>> initial_state = env.initial_state_dist().sample()[0]
        >>>
        >>> # Sample complete step (action must be provided based on environment type)
        >>> action = np.array([1.0, 0.0])  # Move right
        >>> next_state, observation, reward = env.sample_next_step(initial_state, action)
        >>>
        >>> # Check terminal condition
        >>> env.is_terminal(initial_state)
        False
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        discount_factor: float,
        name: str = "ContinuousLightDarkPOMDP",
        state_transition_cov_matrix: np.ndarray = np.eye(2) * 0.05,
        observation_cov_matrix: np.ndarray = np.eye(2) * 0.05,
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
        fuel_cost: float = 2.0,
        grid_size: int = 11,
        goal_state_radius: float = 1.5,
        beacon_radius: float = 1.0,
        obstacle_radius: float = 1.5,
        reward_model_type: RewardModelType = RewardModelType.STANDARD,
        observation_model_type: ObservationModelType = ObservationModelType.NORMAL_NOISE,
        penalty_decay: float = 1.0,
        is_obstacle_hit_terminal: bool = True,
    ):
        space_info = SpaceInfo(
            action_space=SpaceType.CONTINUOUS, observation_space=SpaceType.CONTINUOUS
        )
        # Calculate reward range based on reward model type
        # Maximum distance to goal is diagonal of grid: sqrt(2) * grid_size
        max_distance_to_goal = np.sqrt(2) * grid_size

        if reward_model_type == RewardModelType.STANDARD:
            # Min: -fuel_cost - max_distance + obstacle_reward (always negative)
            # Max: -fuel_cost - 0 + goal_reward (at goal)
            min_reward = -fuel_cost - max_distance_to_goal + obstacle_reward
            max_reward = -fuel_cost + goal_reward
        elif reward_model_type == RewardModelType.DECAYING_HIT_PROBABILITY:
            # Similar to standard but with distance-based penalties
            # Min: -fuel_cost - max_distance + obstacle_reward (max penalty)
            # Max: -fuel_cost - 0 + goal_reward (at goal, no penalty)
            min_reward = -fuel_cost - max_distance_to_goal + obstacle_reward
            max_reward = -fuel_cost + goal_reward
        elif reward_model_type == RewardModelType.DANGEROUS_STATES:
            # Min: -fuel_cost - max_distance + obstacle_reward (negative)
            # Max: -fuel_cost - 0 + goal_reward OR -fuel_cost - distance - obstacle_reward (if obstacle_reward is negative)
            min_reward = -fuel_cost - max_distance_to_goal + obstacle_reward
            max_reward = max(-fuel_cost + goal_reward, -fuel_cost - obstacle_reward)
        else:
            # Default fallback
            min_reward = -fuel_cost - max_distance_to_goal + obstacle_reward
            max_reward = -fuel_cost + goal_reward

        calculated_reward_range = (min_reward, max_reward)

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=calculated_reward_range,
            beacons=beacons,
            goal_state=goal_state,
            start_state=start_state,
            obstacles=obstacles,
            obstacle_hit_probability=obstacle_hit_probability,
            obstacle_reward=obstacle_reward,
            goal_reward=goal_reward,
            beacon_radius=beacon_radius,
            obstacle_radius=obstacle_radius,
            fuel_cost=fuel_cost,
            grid_size=grid_size,
        )

        self.__type_check(
            state_transition_cov_matrix=state_transition_cov_matrix,
            observation_cov_matrix=observation_cov_matrix,
            goal_state_radius=goal_state_radius,
            beacon_radius=beacon_radius,
            obstacle_radius=obstacle_radius,
        )

        self.state_transition_cov_matrix = state_transition_cov_matrix
        self.observation_cov_matrix = observation_cov_matrix
        self.goal_state_radius = goal_state_radius
        self.beacon_radius = beacon_radius
        self.observation_model_type = observation_model_type
        self.penalty_decay = penalty_decay
        self.is_obstacle_hit_terminal = is_obstacle_hit_terminal

        # Create distributions with pre-computed Cholesky decomposition
        self._state_transition_dist = CovarianceParameterizedMultivariateNormal(
            state_transition_cov_matrix
        )
        self._obs_dist_far_from_beacon = CovarianceParameterizedMultivariateNormal(
            observation_cov_matrix
        )
        self._obs_dist_near_beacon = CovarianceParameterizedMultivariateNormal(
            observation_cov_matrix * 0.5
        )

        # Snapshot covariance buffers without copying for the per-action
        # kernel cache below — the Cholesky lives inside the C++ kernel
        # and never sees these arrays again after construction.
        self._trans_cov_view = self._state_transition_dist.covariance_view()
        self._obs_cov_near_view = self._obs_dist_near_beacon.covariance_view()
        self._obs_cov_far_view = self._obs_dist_far_from_beacon.covariance_view()

        # Per-action kernel caches: one C++ kernel per distinct action vector.
        # Hot-path overrides flip the (next_)state field via set_state /
        # set_next_state instead of rebuilding (skips the per-call Cholesky).
        # Keys are action.tobytes(); values are the long-lived C++ kernels.
        self._trans_kernel_cache: Dict[bytes, Any] = {}
        self._obs_kernel_cache: Dict[bytes, Any] = {}

        # Cached scalar constants for the observation_log_probability_single
        # fast-path used by POMCPOW's WeightedParticleBeliefStateUpdate.
        # Each near/far covariance is 2x2; we store the inverse and the
        # log normalizer so each per-state call avoids array allocation.
        self._cls_obs_inv_cov_far = self._build_inverse_2x2(observation_cov_matrix)
        self._cls_obs_inv_cov_near = self._build_inverse_2x2(observation_cov_matrix * 0.5)
        self._cls_obs_log_norm_far = self._build_log_norm_2d(observation_cov_matrix)
        self._cls_obs_log_norm_near = self._build_log_norm_2d(observation_cov_matrix * 0.5)
        self._cls_beacon_radius_sq = float(beacon_radius) * float(beacon_radius)
        # Beacons stored as (2, N) float ndarray; precompute (N, 2) for the
        # singleton near-beacon scan.
        self._cls_beacons_t = np.ascontiguousarray(self.beacons.T, dtype=np.float64)
        # Initialize reward model based on type
        self.reward_model: BaseLightDarkRewardModel
        if reward_model_type == RewardModelType.STANDARD:
            self.reward_model = ContinuousLightDarkRewardModel(
                goal_state=self.goal_state,
                obstacles=self.obstacles,
                goal_state_radius=self.goal_state_radius,
                obstacle_radius=self.obstacle_radius,
                grid_size=self.grid_size,
                obstacle_hit_probability=self.obstacle_hit_probability,
                obstacle_reward=self.obstacle_reward,
                goal_reward=self.goal_reward,
                fuel_cost=self.fuel_cost,
            )
        elif reward_model_type == RewardModelType.DECAYING_HIT_PROBABILITY:
            self.reward_model = ContinuousLightDarkDecayingHitProbabilityRewardModel(
                goal_state=self.goal_state,
                obstacles=self.obstacles,
                goal_state_radius=self.goal_state_radius,
                obstacle_radius=self.obstacle_radius,
                grid_size=self.grid_size,
                obstacle_hit_probability=self.obstacle_hit_probability,
                obstacle_reward=self.obstacle_reward,
                goal_reward=self.goal_reward,
                fuel_cost=self.fuel_cost,
                penalty_decay=self.penalty_decay,
            )
        elif reward_model_type == RewardModelType.DANGEROUS_STATES:
            self.reward_model = ContinuousLDDangerousStatesRewardModel(
                goal_state=self.goal_state,
                obstacles=self.obstacles,
                goal_state_radius=self.goal_state_radius,
                obstacle_radius=self.obstacle_radius,
                grid_size=self.grid_size,
                obstacle_hit_probability=self.obstacle_hit_probability,
                obstacle_reward=self.obstacle_reward,
                goal_reward=self.goal_reward,
                fuel_cost=self.fuel_cost,
            )
        else:
            raise ValueError(f"Unknown reward model type: {reward_model_type}")

    def __type_check(
        self,
        state_transition_cov_matrix: np.ndarray,
        observation_cov_matrix: np.ndarray,
        goal_state_radius: float,
        beacon_radius: float,
        obstacle_radius: float,
    ):
        if state_transition_cov_matrix.shape != (2, 2):
            raise ValueError("state_transition_cov_matrix must be a 2x2 matrix")
        if observation_cov_matrix.shape != (2, 2):
            raise ValueError("observation_cov_matrix must be a 2x2 matrix")
        if goal_state_radius <= 0:
            raise ValueError("goal_state_radius must be greater than 0")
        if beacon_radius <= 0:
            raise ValueError("beacon_radius must be greater than 0")
        if obstacle_radius <= 0:
            raise ValueError("obstacle_radius must be greater than 0")

    # ── Helpers for non-NORMAL_NOISE observation models (inlined post-PR-D) ──
    # These replace the deleted Python wrapper classes
    # ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel and
    # ContinuousLightDarkDistanceBasedObservationModel. The math is preserved
    # bit-for-bit (RNG draw count, near-beacon predicate strictness, clipping).

    def _near_beacon_continuous(self, next_state: np.ndarray) -> bool:
        return bool(
            any_point_within_radius_kernel(
                np.asarray(next_state, dtype=float), self.beacons, self.beacon_radius
            )
        )

    def _sample_mvn_clipped(
        self,
        next_state: np.ndarray,
        dist: CovarianceParameterizedMultivariateNormal,
        n_samples: int,
    ) -> np.ndarray:
        # Mirrors the wrapper sample path:
        #   z = np.random.standard_normal((n_samples, 2))
        #   obs = next_state + z @ dist._cholesky_L_T
        #   obs = clip(obs, 0, grid_size)
        z = np.random.standard_normal((n_samples, 2))
        chol_L_T = dist._cholesky_L_T  # pylint: disable=protected-access
        observations = mvn_sample_2d_kernel(next_state, z, chol_L_T)
        return np.clip(observations, 0, self.grid_size)

    def _sample_no_obs_in_dark(self, next_state: np.ndarray, n_samples: int) -> List[Any]:
        # Mirrors ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel.sample():
        # near_beacon (strict <) → MVN samples from near-beacon dist, else "None".
        if self._near_beacon_continuous(next_state):
            observations = self._sample_mvn_clipped(
                next_state, self._obs_dist_near_beacon, n_samples
            )
            return list(observations)
        return ["None"] * n_samples

    def _sample_distance_based(self, next_state: np.ndarray, n_samples: int) -> List[Any]:
        # Mirrors ContinuousLightDarkDistanceBasedObservationModel.sample():
        # if min_distance > beacon_radius → "None", else MVN from active_dist
        # (near if strict-near, else far). The strictness mismatch between
        # `near_beacon` (strict <) and `> beacon_radius` is preserved.
        min_distance = float(min_distance_to_points_kernel(next_state, self.beacons))
        if min_distance > self.beacon_radius:
            return ["None"] * n_samples
        active_dist = (
            self._obs_dist_near_beacon
            if self._near_beacon_continuous(next_state)
            else self._obs_dist_far_from_beacon
        )
        observations = self._sample_mvn_clipped(next_state, active_dist, n_samples)
        return list(observations)

    def _log_prob_no_obs_in_dark(self, next_state: np.ndarray, observations: Any) -> np.ndarray:
        # Mirrors ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel.probability().
        # near_beacon → "None" probability is 0 (-inf); real obs uses the
        # active (near-beacon) distribution's PDF.
        # far_beacon → "None" probability is 1 (log 0); real obs uses the
        # active (far-from-beacon) distribution's PDF (note: the deleted
        # wrapper computed pdf for any non-"None" value regardless of
        # beacon proximity, so the env API mirrors that asymmetry).
        observations_list = self._normalize_observations_list(observations)
        near = self._near_beacon_continuous(next_state)
        active_dist = self._obs_dist_near_beacon if near else self._obs_dist_far_from_beacon
        return self._mixed_log_probs(
            next_state,
            observations_list,
            none_log_mass=-np.inf if near else 0.0,
            real_obs_dist=active_dist,
        )

    def _log_prob_distance_based(self, next_state: np.ndarray, observations: Any) -> np.ndarray:
        # Mirrors ContinuousLightDarkDistanceBasedObservationModel.probability().
        # if min_distance > beacon_radius → "None" probability is 1 (log 0),
        # real obs probability 0 (-inf); else "None" probability is 0 (-inf)
        # and real obs uses the active distribution's PDF (active dist is
        # near-beacon when strict-near, else far-from-beacon).
        observations_list = self._normalize_observations_list(observations)
        min_distance = float(min_distance_to_points_kernel(next_state, self.beacons))
        far = min_distance > self.beacon_radius
        if far:
            return self._mixed_log_probs(
                next_state,
                observations_list,
                none_log_mass=0.0,
                real_obs_dist=None,
            )
        active_dist = (
            self._obs_dist_near_beacon
            if self._near_beacon_continuous(next_state)
            else self._obs_dist_far_from_beacon
        )
        return self._mixed_log_probs(
            next_state,
            observations_list,
            none_log_mass=-np.inf,
            real_obs_dist=active_dist,
        )

    def _normalize_observations_list(self, observations: Any) -> list:
        if isinstance(observations, np.ndarray):
            if observations.ndim == 1:
                return [observations]
            return list(observations)
        return list(observations)

    def _mixed_log_probs(
        self,
        next_state: np.ndarray,
        observations_list: list,
        none_log_mass: float,
        real_obs_dist: Any,
    ) -> np.ndarray:
        # ``none_log_mass`` is the log-probability assigned to the "None"
        # sentinel; ``real_obs_dist`` (or None if real observations are
        # impossible) provides the PDF for non-"None" array observations.
        out = np.full(len(observations_list), -np.inf, dtype=np.float64)
        for i, value in enumerate(observations_list):
            if isinstance(value, str) and value == "None":
                out[i] = none_log_mass
                continue
            if real_obs_dist is None:
                continue  # real observations carry probability 0 → -inf
            value_arr = np.asarray(value, dtype=float)
            pdf = float(real_obs_dist.pdf(np.array([value_arr]), next_state)[0])
            out[i] = np.log(pdf) if pdf > 0.0 else -np.inf
        return out

    # ── Hot-path sampling overrides ─────────────────────────────────
    # The default base-class implementations build a Python-wrapper
    # subclass per call (np.asarray(...).ravel() x2, side-attribute
    # storage). The actual RNG draw lives entirely inside the C++
    # _native.ContinuousLightDark{Transition,Observation}Cpp.sample()
    # method. These overrides skip the Python subclass, fetch a cached
    # per-action C++ kernel (Cholesky factored once per action), and
    # rewrite only the (next_)state field per call via set_state /
    # set_next_state. The C++ RNG state lives on the kernel, so each
    # cached kernel maintains a single RNG stream per (env, action).

    def _get_trans_kernel(self, action: np.ndarray) -> Any:
        action_arr = np.ascontiguousarray(np.asarray(action, dtype=np.float64))
        key = action_arr.tobytes()
        kernel = self._trans_kernel_cache.get(key)
        if kernel is None:
            # Use a zero placeholder state — set_state will overwrite it.
            kernel = _native.ContinuousLightDarkTransitionCpp(
                state=np.zeros(2, dtype=np.float64),
                action=action_arr,
                covariance=self._trans_cov_view,
            )
            self._trans_kernel_cache[key] = kernel
        return kernel

    def _get_obs_kernel(self, action: np.ndarray) -> Any:
        action_arr = np.ascontiguousarray(np.asarray(action, dtype=np.float64))
        key = action_arr.tobytes()
        kernel = self._obs_kernel_cache.get(key)
        if kernel is None:
            kernel = _native.ContinuousLightDarkObservationCpp(
                next_state=np.zeros(2, dtype=np.float64),
                action=action_arr,
                covariance_near=self._obs_cov_near_view,
                covariance_far=self._obs_cov_far_view,
                beacons=self.beacons,
                beacon_radius=float(self.beacon_radius),
                grid_size=float(self.grid_size),
            )
            self._obs_kernel_cache[key] = kernel
        return kernel

    def sample_next_state(
        self, state: np.ndarray, action: np.ndarray, n_samples: int = 1
    ) -> np.ndarray:
        kernel = self._get_trans_kernel(action)
        kernel.set_state(state)
        if n_samples == 1:
            return kernel.sample()[0]
        return np.asarray(kernel.sample(n_samples), dtype=np.float64)

    def sample_observation(
        self, next_state: np.ndarray, action: np.ndarray, n_samples: int = 1
    ) -> Any:
        if self.observation_model_type == ObservationModelType.NORMAL_NOISE:
            kernel = self._get_obs_kernel(action)
            kernel.set_next_state(next_state)
            if n_samples == 1:
                return kernel.sample()[0]
            return np.asarray(kernel.sample(n_samples), dtype=np.float64)
        if self.observation_model_type == ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK:
            samples = self._sample_no_obs_in_dark(next_state, n_samples)
            if n_samples == 1:
                return samples[0]
            return samples
        if self.observation_model_type == ObservationModelType.DISTANCE_BASED:
            samples = self._sample_distance_based(next_state, n_samples)
            if n_samples == 1:
                return samples[0]
            return samples
        raise ValueError(f"Unknown observation model type: {self.observation_model_type}")

    def transition_log_probability(
        self, state: np.ndarray, action: np.ndarray, next_states: Any
    ) -> np.ndarray:
        kernel = self._get_trans_kernel(action)
        kernel.set_state(state)
        next_states_array = np.asarray(next_states, dtype=np.float64)
        if next_states_array.ndim == 1:
            next_states_array = next_states_array.reshape(1, -1)
        probs = np.asarray(kernel.probability(next_states_array), dtype=np.float64)
        return np.log(probs + 1e-300)

    def observation_log_probability(
        self, next_state: np.ndarray, action: np.ndarray, observations: Any
    ) -> np.ndarray:
        if self.observation_model_type == ObservationModelType.NORMAL_NOISE:
            kernel = self._get_obs_kernel(action)
            kernel.set_next_state(next_state)
            obs_array = np.asarray(observations, dtype=np.float64)
            if obs_array.ndim == 1:
                obs_array = obs_array.reshape(1, -1)
            probs = np.asarray(kernel.probability(obs_array), dtype=np.float64)
            return np.log(probs + 1e-300)
        del action  # unused; non-NORMAL paths read only next_state and obs
        if self.observation_model_type == ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK:
            return self._log_prob_no_obs_in_dark(next_state, observations)
        if self.observation_model_type == ObservationModelType.DISTANCE_BASED:
            return self._log_prob_distance_based(next_state, observations)
        raise ValueError(f"Unknown observation model type: {self.observation_model_type}")

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
        if self.observation_model_type != ObservationModelType.NORMAL_NOISE:
            # NoObsInDark and DistanceBased models lack a native batch kernel;
            # fall back to the base-class per-state Python loop.
            return super().observation_log_probability_per_state(
                next_states=next_states, action=action, observation=observation
            )
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

    @staticmethod
    def _build_inverse_2x2(cov: np.ndarray) -> Tuple[float, float, float]:
        # Returns (inv00, inv01, inv11) for a 2x2 symmetric positive-definite matrix.
        a = float(cov[0, 0])
        b = float(cov[0, 1])
        d = float(cov[1, 1])
        det = a * d - b * b
        if det <= 0.0:
            raise ValueError("Observation covariance must be positive definite")
        inv_det = 1.0 / det
        return (d * inv_det, -b * inv_det, a * inv_det)

    @staticmethod
    def _build_log_norm_2d(cov: np.ndarray) -> float:
        # log(1 / (2*pi*sqrt(det))) for a 2x2 covariance matrix.
        a = float(cov[0, 0])
        b = float(cov[0, 1])
        d = float(cov[1, 1])
        det = a * d - b * b
        if det <= 0.0:
            raise ValueError("Observation covariance must be positive definite")
        return -math.log(2.0 * math.pi) - 0.5 * math.log(det)

    def _is_near_beacon_scalar(self, x: float, y: float) -> bool:
        # Scalar near-beacon scan; mirrors the broadcasting check in
        # BaseContinuousLightDarkObservationModel._near_beacon but without
        # numpy allocation. Returns True if within beacon_radius of any beacon.
        beacons = self._cls_beacons_t
        if beacons.shape[0] == 0:
            return False
        radius_sq = self._cls_beacon_radius_sq
        for i in range(beacons.shape[0]):
            dx = x - beacons[i, 0]
            dy = y - beacons[i, 1]
            if dx * dx + dy * dy <= radius_sq:
                return True
        return False

    def observation_log_probability_single(
        self, next_state: Any, action: Any, observation: Any
    ) -> float:
        # Scalar fast-path used by POMCPOW's incremental belief update for
        # the NORMAL_NOISE observation model. Other model types fall back to
        # the base-class default that wraps the batched probability call.
        if self.observation_model_type != ObservationModelType.NORMAL_NOISE:
            return super().observation_log_probability_single(
                next_state=next_state, action=action, observation=observation
            )

        nx = float(next_state[0])
        ny = float(next_state[1])
        if self._is_near_beacon_scalar(nx, ny):
            inv00, inv01, inv11 = self._cls_obs_inv_cov_near
            log_norm = self._cls_obs_log_norm_near
        else:
            inv00, inv01, inv11 = self._cls_obs_inv_cov_far
            log_norm = self._cls_obs_log_norm_far

        dx = float(observation[0]) - nx
        dy = float(observation[1]) - ny
        # Mahalanobis squared for a 2D symmetric inverse covariance.
        m_sq = inv00 * dx * dx + 2.0 * inv01 * dx * dy + inv11 * dy * dy
        return log_norm - 0.5 * m_sq

    def reward(self, state: np.ndarray, action: np.ndarray) -> float:
        return self.reward_model.compute_reward(state, action)

    def reward_batch(
        self, states: Union[np.ndarray, Sequence[Any]], action: np.ndarray
    ) -> np.ndarray:
        return self.reward_model.compute_reward_batch(np.asarray(states), action)

    def is_terminal(self, state: np.ndarray) -> bool:
        if state.shape != (2,):
            raise ValueError("state must be a 2D vector")
        return is_terminal_kernel(
            state,
            self.goal_state,
            self.obstacles,
            self.goal_state_radius,
            self.obstacle_radius,
            self.is_obstacle_hit_terminal,
        )

    def get_metric_names(self) -> List[str]:
        """Get names of Continuous Light-Dark POMDP specific metrics.

        Returns:
            List containing metric names: goal_reaching_rate, obstacle_hit_rate,
            avg_obstacle_hit_counter, out_of_grid_rate, and avg_dangerous_states_counter
        """
        return [metric.value for metric in ContinuousLightDarkPOMDPMetrics]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        goal_reached = []
        obstacle_hits = []
        obstacle_hit_counter = []
        out_of_grid = []
        dangerous_states_counter = []
        for history in histories:
            goal_reached_in_history = False
            obstacle_hit_in_history = False
            obstacle_hit_counter_in_history = 0
            out_of_grid_in_history = False
            out_of_grid_counter_in_history = 0

            for _, step in enumerate(history.history):
                if np.linalg.norm(step.state - self.goal_state) <= self.goal_state_radius:
                    goal_reached_in_history = True
                    break

                # Calculate distance to each obstacle (obstacles are 2xN format)
                distances = np.linalg.norm(step.state.reshape(-1, 1) - self.obstacles, axis=0)
                if np.any(distances <= self.obstacle_radius):
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
            # Sum obstacle hits and out-of-grid occurrences as dangerous states
            dangerous_states_counter.append(
                obstacle_hit_counter_in_history + out_of_grid_counter_in_history
            )

        avg_goal_reached = float(np.mean(goal_reached))
        avg_obstacle_hits = float(np.mean(obstacle_hits))
        avg_obstacle_hit_counter = float(np.mean(obstacle_hit_counter))
        avg_out_of_grid = float(np.mean(out_of_grid))
        avg_dangerous_states_counter = float(np.mean(dangerous_states_counter))
        goal_reached_ci = confidence_interval(data=goal_reached, confidence=0.95)
        obstacle_hits_ci = confidence_interval(data=obstacle_hits, confidence=0.95)
        obstacle_hit_counter_ci = confidence_interval(data=obstacle_hit_counter, confidence=0.95)
        out_of_grid_ci = confidence_interval(data=out_of_grid, confidence=0.95)
        dangerous_states_counter_ci = confidence_interval(
            data=dangerous_states_counter, confidence=0.95
        )

        return [
            MetricValue(
                name=ContinuousLightDarkPOMDPMetrics.GOAL_REACHING_RATE.value,
                value=avg_goal_reached,
                lower_confidence_bound=goal_reached_ci[0],
                upper_confidence_bound=goal_reached_ci[1],
            ),
            MetricValue(
                name=ContinuousLightDarkPOMDPMetrics.OBSTACLE_HIT_RATE.value,
                value=avg_obstacle_hits,
                lower_confidence_bound=obstacle_hits_ci[0],
                upper_confidence_bound=obstacle_hits_ci[1],
            ),
            MetricValue(
                name=ContinuousLightDarkPOMDPMetrics.AVG_OBSTACLE_HIT_COUNTER.value,
                value=avg_obstacle_hit_counter,
                lower_confidence_bound=obstacle_hit_counter_ci[0],
                upper_confidence_bound=obstacle_hit_counter_ci[1],
            ),
            MetricValue(
                name=ContinuousLightDarkPOMDPMetrics.OUT_OF_GRID_RATE.value,
                value=avg_out_of_grid,
                lower_confidence_bound=out_of_grid_ci[0],
                upper_confidence_bound=out_of_grid_ci[1],
            ),
            MetricValue(
                name=ContinuousLightDarkPOMDPMetrics.AVG_DANGEROUS_STATES_COUNTER.value,
                value=avg_dangerous_states_counter,
                lower_confidence_bound=dangerous_states_counter_ci[0],
                upper_confidence_bound=dangerous_states_counter_ci[1],
            ),
        ]

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

    def __eq__(self, other):
        if not isinstance(other, ContinuousLightDarkPOMDP):
            return False

        if not super().__eq__(other):
            return False

        return (
            np.array_equal(self.state_transition_cov_matrix, other.state_transition_cov_matrix)
            and np.array_equal(self.observation_cov_matrix, other.observation_cov_matrix)
            and self.goal_state_radius == other.goal_state_radius
            and self.beacon_radius == other.beacon_radius
            and self.obstacle_radius == other.obstacle_radius
            and self.observation_model_type == other.observation_model_type
        )

    def hash_action(self, action: Any) -> Hashable:
        # Continuous actions are ndarray; bytes match np.array_equal semantics
        # for arrays of identical shape and dtype.
        return np.ascontiguousarray(action, dtype=np.float64).tobytes()


class ContinuousLightDarkPOMDPDiscreteActions(ContinuousLightDarkPOMDP, DiscreteActionsEnvironment):
    """Continuous Light-Dark POMDP environment with discrete actions.

    This variant of the Continuous Light-Dark POMDP uses discrete directional actions
    (up, down, left, right) instead of continuous action vectors. The continuous
    state space and observation model are preserved.

    Actions are mapped to unit vectors:
    - "up": [0, 1]
    - "down": [0, -1]
    - "right": [1, 0]
    - "left": [-1, 0]

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> env = ContinuousLightDarkPOMDPDiscreteActions(
        ...     discount_factor=0.95,
        ...     goal_state=np.array([10, 5]),
        ...     start_state=np.array([0, 5])
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
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        discount_factor: float,
        state_transition_cov_matrix: np.ndarray = np.eye(2),
        observation_cov_matrix: np.ndarray = np.eye(2),
        obstacle_hit_probability: float = 0.2,
        obstacle_reward: float = -10.0,
        goal_reward: float = 10.0,
        fuel_cost: float = 2.0,
        grid_size: int = 11,
        goal_state_radius: float = 1.5,
        beacon_radius: float = 1.0,
        obstacle_radius: float = 1.5,
        name: str = "ContinuousLightDarkPOMDPDiscreteActions",
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
        reward_model_type: RewardModelType = RewardModelType.STANDARD,
        observation_model_type: ObservationModelType = ObservationModelType.NORMAL_NOISE,
        penalty_decay: float = 1.0,
        is_obstacle_hit_terminal: bool = True,
    ):
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            state_transition_cov_matrix=state_transition_cov_matrix,
            observation_cov_matrix=observation_cov_matrix,
            beacons=beacons,
            goal_state=goal_state,
            start_state=start_state,
            obstacles=obstacles,
            obstacle_hit_probability=obstacle_hit_probability,
            obstacle_reward=obstacle_reward,
            goal_reward=goal_reward,
            fuel_cost=fuel_cost,
            grid_size=grid_size,
            goal_state_radius=goal_state_radius,
            beacon_radius=beacon_radius,
            obstacle_radius=obstacle_radius,
            reward_model_type=reward_model_type,
            observation_model_type=observation_model_type,
            penalty_decay=penalty_decay,
            is_obstacle_hit_terminal=is_obstacle_hit_terminal,
        )

        # Override space info
        self.space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.CONTINUOUS
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

        # Pre-built arrays for native simulate_rollout.
        # _actions_array: shape (n_actions, 2) — action vectors stacked row-wise.
        # _obstacles_flat: shape (2*n_obstacles,) — interleaved [x0,y0,x1,y1,...].
        self._actions_array: np.ndarray = np.ascontiguousarray(
            np.stack(list(self.action_to_vector.values()), axis=0), dtype=np.float64
        )
        # self.obstacles is stored as (2, N); flatten to [x0,y0,x1,y1,...].
        self._obstacles_flat: np.ndarray = np.ascontiguousarray(
            self.obstacles.T.ravel(), dtype=np.float64
        )
        self._goal_state_f64: np.ndarray = np.ascontiguousarray(self.goal_state, dtype=np.float64)

    def get_actions(self) -> List[Any]:
        return self.actions

    def reward(self, state: np.ndarray, action: Any) -> float:
        action_vector = self.action_to_vector[action]
        return super().reward(state, action_vector)

    def reward_batch(self, states: Union[np.ndarray, Sequence[Any]], action: Any) -> np.ndarray:
        return super().reward_batch(np.asarray(states), self.action_to_vector[action])

    def sample_next_state(self, state: np.ndarray, action: Any, n_samples: int = 1) -> np.ndarray:
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

    def simulate_random_rollout(
        self,
        state: Any,
        action_sampler: Any,
        max_depth: int,
        discount_factor: float,
        depth: int = 0,
    ) -> float:
        """Random rollout via native C++ for the STANDARD reward model.

        Falls back to the base-class Python loop for non-STANDARD reward
        models (DECAYING_HIT_PROBABILITY, DANGEROUS_STATES) which have
        different stochastic semantics not yet ported to C++.

        Args:
            state: Current 2-D position ``[x, y]``.
            action_sampler: Object with a ``sample()`` method; used only for
                the Python fallback path. On the native path, action indices
                are pre-drawn inside this method.
            max_depth: Maximum rollout depth.
            discount_factor: Per-step discount factor.
            depth: Depth already consumed by the search tree. Defaults to 0.

        Returns:
            Discounted sum of immediate rewards along the sampled trajectory.
        """
        if not isinstance(self.reward_model, ContinuousLightDarkRewardModel) or isinstance(
            self.reward_model, (ContinuousLDDangerousStatesRewardModel,)
        ):
            return python_random_rollout(
                state=state,
                depth=depth,
                action_sampler=action_sampler,
                environment=self,
                discount_factor=discount_factor,
                max_depth=max_depth,
            )

        steps_left = max_depth - depth
        if steps_left <= 0:
            return 0.0

        state_arr = np.ascontiguousarray(np.asarray(state, dtype=np.float64).ravel())
        action_indices = np.random.randint(0, len(self.actions), size=steps_left, dtype=np.int32)

        return _native.simulate_rollout(
            initial_state=state_arr,
            action_array=self._actions_array,
            action_indices=action_indices,
            max_depth=max_depth,
            start_depth=depth,
            discount_factor=discount_factor,
            goal_state=self._goal_state_f64,
            obstacles=self._obstacles_flat,
            goal_state_radius=float(self.goal_state_radius),
            obstacle_radius=float(self.obstacle_radius),
            grid_size=float(self.grid_size),
            fuel_cost=float(self.fuel_cost),
            goal_reward=float(self.goal_reward),
            obstacle_reward=float(self.obstacle_reward),
            obstacle_hit_probability=float(self.obstacle_hit_probability),
            is_obstacle_hit_terminal=bool(self.is_obstacle_hit_terminal),
            covariance=self._trans_cov_view,
        )

    def __eq__(self, other):
        if not isinstance(other, ContinuousLightDarkPOMDPDiscreteActions):
            return False
        # Compare only configuration parameters, ignoring internal objects like reward_model
        return (
            self.discount_factor == other.discount_factor
            and np.array_equal(self.state_transition_cov_matrix, other.state_transition_cov_matrix)
            and np.array_equal(self.observation_cov_matrix, other.observation_cov_matrix)
            and np.array_equal(self.beacons, other.beacons)
            and np.array_equal(self.goal_state, other.goal_state)
            and np.array_equal(self.start_state, other.start_state)
            and np.array_equal(self.obstacles, other.obstacles)
            and self.obstacle_hit_probability == other.obstacle_hit_probability
            and self.obstacle_reward == other.obstacle_reward
            and self.goal_reward == other.goal_reward
            and self.fuel_cost == other.fuel_cost
            and self.grid_size == other.grid_size
            and self.goal_state_radius == other.goal_state_radius
            and self.beacon_radius == other.beacon_radius
            and self.obstacle_radius == other.obstacle_radius
            and self.observation_model_type == other.observation_model_type
            and self.penalty_decay == other.penalty_decay
            and self.actions == other.actions
            and all(
                np.array_equal(value, other.action_to_vector[k])
                for k, value in self.action_to_vector.items()
            )
        )

    def hash_action(self, action: Any) -> Hashable:
        # Discrete-action variant: actions are str labels (e.g. "up").
        return action
