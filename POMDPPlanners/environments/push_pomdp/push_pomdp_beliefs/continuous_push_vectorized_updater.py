# SPDX-License-Identifier: MIT

"""Vectorized particle belief updater for the Continuous Push POMDP.

This module implements a concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
that performs batched state transitions and observation log-likelihood
evaluations for the Continuous Push environment, replacing per-particle
Python loops with NumPy array operations.

Classes:
    ContinuousPushVectorizedUpdater: Batched updater for the Continuous
        Push POMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.environments.push_pomdp import _native
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.multivariate_normal import (
    CovarianceParameterizedMultivariateNormal,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
        ContinuousPushPOMDP,
    )


class ContinuousPushVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Vectorized particle belief updater for the Continuous Push POMDP.

    Performs all-particle transitions and observation log-likelihood
    evaluations using vectorized NumPy operations, replacing per-particle
    Python loops with batched array operations.

    Attributes:
        obs_dist: 2-D observation noise distribution for object position.
        grid_size: Size of the square grid.
        push_threshold: Maximum robot-object distance for a push to occur.
        friction_coefficient: Friction reducing push force (0-1).
        max_push: Maximum push magnitude.
        obstacles: Shape ``(M, 4)`` AABB array.
        robot_radius: Robot body radius.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.push_pomdp import ContinuousPushPOMDP
        >>> env = ContinuousPushPOMDP(discount_factor=0.99)
        >>> updater = ContinuousPushVectorizedUpdater.from_environment(env)
        >>> particles = np.tile(env.initial_state_dist().sample()[0], (50, 1))
        >>> action = np.array([1.0, 0.0])
        >>> next_p = updater.batch_transition(particles, action)
        >>> next_p.shape[1]
        6
        >>> obs = particles[0].copy()
        >>> ll = updater.batch_observation_log_likelihood(next_p, action, obs)
        >>> ll.shape[0]
        50
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_dist: CovarianceParameterizedMultivariateNormal,
        state_transition_dist: CovarianceParameterizedMultivariateNormal,
        grid_size: float,
        push_threshold: float,
        friction_coefficient: float,
        max_push: float,
        obstacles: np.ndarray,
        robot_radius: float,
        action_to_vector: Optional[Dict[str, np.ndarray]] = None,
        dangerous_areas_arr: Optional[np.ndarray] = None,
        obstacle_hit_probability: float = 1.0,
        dangerous_area_radius: float = 0.5,
        dangerous_area_penalty: float = 0.0,
        dangerous_area_hit_probability: float = 1.0,
        reward_variant_code: int = 0,
        penalty_decay: float = 1.0,
        is_obstacle_hit_terminal: bool = False,
        is_dangerous_area_hit_terminal: bool = False,
    ):
        self.obs_dist = obs_dist
        self.state_transition_dist = state_transition_dist
        self.grid_size = float(grid_size)
        self.push_threshold = float(push_threshold)
        self.friction_coefficient = float(friction_coefficient)
        self.max_push = float(max_push)
        self.obstacles = obstacles
        self.robot_radius = float(robot_radius)
        self._action_to_vector = action_to_vector
        # Hazard-terminal parameters (only consumed when a flag is enabled;
        # the flag-off path builds the legacy 9-arg native kernel).
        self._dangerous_areas_arr = (
            np.empty((0, 2), dtype=np.float64)
            if dangerous_areas_arr is None
            else np.ascontiguousarray(dangerous_areas_arr, dtype=np.float64)
        )
        self._obstacle_hit_probability = float(obstacle_hit_probability)
        self._dangerous_area_radius = float(dangerous_area_radius)
        self._dangerous_area_penalty = float(dangerous_area_penalty)
        self._dangerous_area_hit_probability = float(dangerous_area_hit_probability)
        self._reward_variant_code = int(reward_variant_code)
        self._penalty_decay = float(penalty_decay)
        self._is_obstacle_hit_terminal = bool(is_obstacle_hit_terminal)
        self._is_dangerous_area_hit_terminal = bool(is_dangerous_area_hit_terminal)
        self._hazard_terminal_enabled = (
            self._is_obstacle_hit_terminal or self._is_dangerous_area_hit_terminal
        )
        # Cache the object-position marginal noise (scalar sigma) used by
        # the native observation model. The full obs_dist is retained for
        # back-compat access via the public attribute.
        self._observation_noise = float(np.sqrt(obs_dist.covariance[0, 0]))
        # Cached covariance arrays for the native batch entry points.
        self._state_transition_cov: np.ndarray = state_transition_dist.covariance
        self._obstacles_arr: np.ndarray = np.asarray(obstacles, dtype=float)

    @classmethod
    def from_environment(cls, env: "ContinuousPushPOMDP") -> "ContinuousPushVectorizedUpdater":
        """Construct an updater from a ContinuousPushPOMDP instance.

        Args:
            env: Environment to extract parameters from.

        Returns:
            A new ``ContinuousPushVectorizedUpdater`` instance.
        """
        # pylint: disable=protected-access
        cov = np.diag([env.observation_noise**2, env.observation_noise**2])
        obs_dist = CovarianceParameterizedMultivariateNormal(cov)

        has_discrete = (
            hasattr(env, "space_info") and env.space_info.action_space == SpaceType.DISCRETE
        )
        action_map = getattr(env, "action_to_vector", None) if has_discrete else None

        variant_code, penalty_decay = env._reward_variant_native_params()
        return cls(
            obs_dist=obs_dist,
            state_transition_dist=env._state_transition_dist,
            grid_size=env.grid_size,
            push_threshold=env.push_threshold,
            friction_coefficient=env.friction_coefficient,
            max_push=env.max_push,
            obstacles=env.obstacles,
            robot_radius=env.robot_radius,
            action_to_vector=action_map,
            dangerous_areas_arr=env._dangerous_areas_arr,
            obstacle_hit_probability=env.obstacle_hit_probability,
            dangerous_area_radius=env.dangerous_area_radius,
            dangerous_area_penalty=env.dangerous_area_penalty,
            dangerous_area_hit_probability=env.dangerous_area_hit_probability,
            reward_variant_code=variant_code,
            penalty_decay=penalty_decay,
            is_obstacle_hit_terminal=env.is_obstacle_hit_terminal,
            is_dangerous_area_hit_terminal=env.is_dangerous_area_hit_terminal,
        )
        # pylint: enable=protected-access

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        action_vec = self._resolve_action(action)
        # Delegate to the native C++ batch sampler so both this path and
        # the per-particle transition share the same C++ RNG. The
        # ``state=particles[0]`` passed to the ctor is unused on the batch
        # path; only the ctor signature requires it.
        if not self._hazard_terminal_enabled:
            transition = _native.ContinuousPushTransitionCpp(
                state=particles[0],
                action=action_vec,
                grid_size=self.grid_size,
                push_threshold=self.push_threshold,
                friction_coefficient=self.friction_coefficient,
                max_push=self.max_push,
                robot_radius=self.robot_radius,
                obstacles=self._obstacles_arr,
                covariance=self._state_transition_cov,
            )
            return transition.batch_sample(particles)
        transition = _native.ContinuousPushTransitionCpp(
            state=particles[0],
            action=action_vec,
            grid_size=self.grid_size,
            push_threshold=self.push_threshold,
            friction_coefficient=self.friction_coefficient,
            max_push=self.max_push,
            robot_radius=self.robot_radius,
            obstacles=self._obstacles_arr,
            covariance=self._state_transition_cov,
            dangerous_areas=self._dangerous_areas_arr,
            obstacle_hit_probability=self._obstacle_hit_probability,
            dangerous_area_radius=self._dangerous_area_radius,
            dangerous_area_penalty=self._dangerous_area_penalty,
            dangerous_area_hit_probability=self._dangerous_area_hit_probability,
            reward_variant_code=self._reward_variant_code,
            penalty_decay=self._penalty_decay,
            is_obstacle_hit_terminal=self._is_obstacle_hit_terminal,
            is_dangerous_area_hit_terminal=self._is_dangerous_area_hit_terminal,
        )
        return transition.batch_sample(particles)

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        obs = np.asarray(observation, dtype=float).ravel()
        action_vec = self._resolve_action(action)
        # Delegate to the native C++ observation log-likelihood. The
        # ``next_state=next_particles[0]`` passed to the ctor is unused on
        # the batch path.
        obs_model = _native.ContinuousPushObservationCpp(
            next_state=next_particles[0],
            action=action_vec,
            observation_noise=self._observation_noise,
            grid_size=self.grid_size,
        )
        return obs_model.batch_log_likelihood(next_particles, obs)

    @property
    def config_id(self) -> str:
        config_dict = {
            "class": "ContinuousPushVectorizedUpdater",
            "obs_cov": self.obs_dist.covariance.tolist(),
            "state_transition_cov": self.state_transition_dist.covariance.tolist(),
            "grid_size": self.grid_size,
            "push_threshold": self.push_threshold,
            "friction_coefficient": self.friction_coefficient,
            "max_push": self.max_push,
            "obstacles": self.obstacles.tolist(),
            "robot_radius": self.robot_radius,
            "is_obstacle_hit_terminal": self._is_obstacle_hit_terminal,
            "is_dangerous_area_hit_terminal": self._is_dangerous_area_hit_terminal,
        }
        if self._hazard_terminal_enabled:
            config_dict["dangerous_areas"] = self._dangerous_areas_arr.tolist()
            config_dict["dangerous_area_radius"] = self._dangerous_area_radius
            config_dict["dangerous_area_penalty"] = self._dangerous_area_penalty
            config_dict["dangerous_area_hit_probability"] = self._dangerous_area_hit_probability
            config_dict["obstacle_hit_probability"] = self._obstacle_hit_probability
            config_dict["reward_variant_code"] = self._reward_variant_code
            config_dict["penalty_decay"] = self._penalty_decay
        if self._action_to_vector is not None:
            config_dict["action_to_vector"] = {
                k: v.tolist() for k, v in self._action_to_vector.items()
            }
        return config_to_id(config_dict)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_action(self, action) -> np.ndarray:
        if self._action_to_vector is not None and isinstance(action, str):
            return np.asarray(self._action_to_vector[action], dtype=float).ravel()
        return np.asarray(action, dtype=float).ravel()
