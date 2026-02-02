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
from POMDPPlanners.environments.push_pomdp.continuous_push_geometry import (
    batch_clamp_circle_to_grid,
    batch_clamp_point_to_grid,
    batch_point_inside_aabb,
    batch_resolve_circle_wall_collision,
)
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

    def __init__(
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
    ):
        self.obs_dist = obs_dist
        self.state_transition_dist = state_transition_dist
        self.grid_size = grid_size
        self.push_threshold = push_threshold
        self.friction_coefficient = friction_coefficient
        self.max_push = max_push
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self._action_to_vector = action_to_vector

        self._precompute_cholesky_scalars()
        self._precompute_obs_precision_scalars()

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
        )
        # pylint: enable=protected-access

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        action_vec = self._resolve_action(action)
        n = particles.shape[0]

        # Robot movement: pos + action + noise
        z = np.random.standard_normal((n, 2))
        noise_x = z[:, 0] * self._trans_L00
        noise_y = z[:, 0] * self._trans_L10 + z[:, 1] * self._trans_L11

        new_robot = particles[:, :2] + action_vec
        new_robot[:, 0] += noise_x
        new_robot[:, 1] += noise_y

        new_robot = batch_resolve_circle_wall_collision(
            new_robot, self.robot_radius, self.obstacles
        )
        new_robot = batch_clamp_circle_to_grid(new_robot, self.robot_radius, self.grid_size)

        # Push mechanics
        new_obj = self._batch_apply_push(new_robot, particles[:, 2:4], action_vec)

        return np.column_stack([new_robot, new_obj, particles[:, 4:6]])

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        obs = np.asarray(observation, dtype=float).ravel()
        obs_obj = obs[2:4]
        particle_obj = next_particles[:, 2:4]
        return self.obs_dist.log_pdf(particle_obj, obs_obj)

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
        }
        if self._action_to_vector is not None:
            config_dict["action_to_vector"] = {
                k: v.tolist() for k, v in self._action_to_vector.items()
            }
        return config_to_id(config_dict)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _precompute_cholesky_scalars(self):
        L = self.state_transition_dist._cholesky_L  # pylint: disable=protected-access
        self._trans_L00 = float(L[0, 0])
        self._trans_L10 = float(L[1, 0])
        self._trans_L11 = float(L[1, 1])

    def _precompute_obs_precision_scalars(self):
        P = np.linalg.inv(self.obs_dist.covariance)
        self._obs_P00 = float(P[0, 0])
        self._obs_2P01 = float(2.0 * P[0, 1])
        self._obs_P11 = float(P[1, 1])
        self._obs_log_norm = float(
            self.obs_dist._log_normalization  # pylint: disable=protected-access
        )

    def _resolve_action(self, action) -> np.ndarray:
        if self._action_to_vector is not None and isinstance(action, str):
            return np.asarray(self._action_to_vector[action], dtype=float).ravel()
        return np.asarray(action, dtype=float).ravel()

    def _batch_apply_push(
        self,
        robot_pos: np.ndarray,
        obj_pos: np.ndarray,
        action_vec: np.ndarray,
    ) -> np.ndarray:
        new_obj = obj_pos.copy()

        dist_to_obj = np.linalg.norm(robot_pos - obj_pos, axis=1)
        can_push = dist_to_obj < self.push_threshold

        action_norm = float(np.linalg.norm(action_vec))
        if action_norm < 1e-12 or not np.any(can_push):
            return new_obj

        direction = action_vec / action_norm
        force_mag = min(action_norm, self.max_push) * (1.0 - self.friction_coefficient)
        push_delta = direction * force_mag

        intended_obj = obj_pos + push_delta
        blocked = batch_point_inside_aabb(intended_obj, self.obstacles)
        actually_push = can_push & ~blocked

        pushed_obj = batch_clamp_point_to_grid(intended_obj, self.grid_size)
        new_obj = np.where(actually_push[:, None], pushed_obj, new_obj)
        return new_obj
