"""Vectorized particle belief updater for the Push POMDP.

This module implements a concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
that performs batched state transitions and observation log-likelihood
evaluations for the Push environment, replacing per-particle Python
loops with NumPy array operations.

Classes:
    PushVectorizedUpdater: Batched updater for the Push POMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.multivariate_normal import (
    CovarianceParameterizedMultivariateNormal,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP


class PushVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Vectorized particle belief updater for the Push POMDP.

    Performs all-particle transitions and observation log-likelihood
    evaluations using vectorized NumPy operations, replacing per-particle
    Python loops with batched array operations.

    The Push transition is deterministic when ``transition_error_prob`` is
    zero.  When nonzero, each particle independently samples which action
    actually executes.  Observations follow a 2-D Gaussian centred on the
    true object position (only indices 2:4 of the state carry noise).

    Attributes:
        obs_dist: 2-D observation noise distribution for object position.
        grid_size: Size of the square grid.
        push_threshold: Maximum robot-object distance for a push to occur.
        friction_coefficient: Friction reducing push force (0-1).
        obstacles: Array of obstacle centres, shape (M, 2) or (0, 2).
        obstacle_radius: Collision radius for each obstacle.
        transition_error_prob: Probability of executing a random other action.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.push_pomdp import PushPOMDP
        >>> env = PushPOMDP(discount_factor=0.99)
        >>> updater = PushVectorizedUpdater.from_environment(env)
        >>> particles = np.tile(env.initial_state_dist().sample()[0], (50, 1))
        >>> action = 0  # "up"
        >>> next_p = updater.batch_transition(particles, action)
        >>> next_p.shape[1]
        6
        >>> obs = particles[0].copy()
        >>> ll = updater.batch_observation_log_likelihood(next_p, action, obs)
        >>> ll.shape[0]
        50
    """

    ACTION_VECTORS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=float)

    def __init__(
        self,
        obs_dist: CovarianceParameterizedMultivariateNormal,
        grid_size: int,
        push_threshold: float,
        friction_coefficient: float,
        obstacles: np.ndarray,
        obstacle_radius: float,
        transition_error_prob: float,
    ):
        """Initialize the vectorized updater.

        Args:
            obs_dist: 2-D MVN for observation noise on object position.
            grid_size: Size of the square grid.
            push_threshold: Maximum robot-object distance for pushing.
            friction_coefficient: Friction reducing push force.
            obstacles: Obstacle centres, shape (M, 2).
            obstacle_radius: Collision radius for each obstacle.
            transition_error_prob: Probability of executing a random action.
        """
        self.obs_dist = obs_dist
        self.grid_size = grid_size
        self.push_threshold = push_threshold
        self.friction_coefficient = friction_coefficient
        self.obstacles = obstacles
        self.obstacle_radius = obstacle_radius
        self.transition_error_prob = transition_error_prob

    @classmethod
    def from_environment(cls, env: "PushPOMDP") -> "PushVectorizedUpdater":
        """Construct an updater from a PushPOMDP instance.

        Args:
            env: Environment to extract parameters from.

        Returns:
            A new ``PushVectorizedUpdater`` instance.
        """
        cov = np.diag([env.observation_noise**2, env.observation_noise**2])
        obs_dist = CovarianceParameterizedMultivariateNormal(cov)
        obstacles = np.array(env.obstacles, dtype=float) if env.obstacles else np.empty((0, 2))
        return cls(
            obs_dist=obs_dist,
            grid_size=env.grid_size,
            push_threshold=env.push_threshold,
            friction_coefficient=env.friction_coefficient,
            obstacles=obstacles,
            obstacle_radius=env.obstacle_radius,
            transition_error_prob=env.transition_error_prob,
        )

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        # particles: (N, 6) = [robot_x, robot_y, obj_x, obj_y, target_x, target_y]
        action_idx = int(action)
        if self.transition_error_prob > 0:
            return self._batch_transition_with_error(particles, action_idx)
        return self._transition_for_action(particles, action_idx)

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        observation = np.asarray(observation, dtype=float).ravel()
        obs_obj = observation[2:4]
        particle_obj = next_particles[:, 2:4]
        return self.obs_dist.log_pdf(particle_obj, obs_obj)

    @property
    def config_id(self) -> str:
        config_dict = {
            "class": "PushVectorizedUpdater",
            "obs_cov": self.obs_dist.covariance.tolist(),
            "grid_size": self.grid_size,
            "push_threshold": self.push_threshold,
            "friction_coefficient": self.friction_coefficient,
            "obstacles": self.obstacles.tolist(),
            "obstacle_radius": self.obstacle_radius,
            "transition_error_prob": self.transition_error_prob,
        }
        return config_to_id(config_dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition_for_action(self, particles: np.ndarray, action_idx: int) -> np.ndarray:
        movement = self.ACTION_VECTORS[action_idx]

        # Robot movement
        intended_robot = particles[:, :2] + movement
        robot_colliding = self._batch_obstacle_collision(intended_robot)
        new_robot = np.where(robot_colliding[:, None], particles[:, :2], intended_robot)

        # Object pushing
        dist_to_obj = np.linalg.norm(new_robot - particles[:, 2:4], axis=1)
        can_push = dist_to_obj < self.push_threshold

        push_force = movement * (1 - self.friction_coefficient)
        intended_obj = particles[:, 2:4] + push_force
        obj_colliding = self._batch_obstacle_collision(intended_obj)
        pushed_obj = np.where(obj_colliding[:, None], particles[:, 2:4], intended_obj)
        new_obj = np.where(can_push[:, None], pushed_obj, particles[:, 2:4])

        # Grid clipping
        new_robot = np.clip(new_robot, 0, self.grid_size - 1)
        new_obj = np.clip(new_obj, 0, self.grid_size - 1)

        return np.column_stack([new_robot, new_obj, particles[:, 4:6]])

    def _batch_transition_with_error(
        self, particles: np.ndarray, intended_action: int
    ) -> np.ndarray:
        n = particles.shape[0]

        # Compute next state for all 4 possible actions
        all_results = np.empty((4, n, 6))
        for a in range(4):
            all_results[a] = self._transition_for_action(particles, a)

        # Sample which action each particle actually executes
        error_mask = np.random.random(n) < self.transition_error_prob
        chosen_actions = np.full(n, intended_action, dtype=int)

        error_count = int(error_mask.sum())
        if error_count > 0:
            other_actions = [a for a in range(4) if a != intended_action]
            chosen_actions[error_mask] = np.random.choice(other_actions, size=error_count)

        # Gather per-particle results using fancy indexing
        return all_results[chosen_actions, np.arange(n)]

    def _batch_obstacle_collision(self, positions: np.ndarray) -> np.ndarray:
        if self.obstacles.shape[0] == 0:
            return np.zeros(len(positions), dtype=bool)
        # positions: (N, 2), obstacles: (M, 2)
        diff = positions[:, None, :] - self.obstacles[None, :, :]  # (N, M, 2)
        distances = np.linalg.norm(diff, axis=2)  # (N, M)
        return np.any(distances <= self.obstacle_radius, axis=1)  # (N,)
