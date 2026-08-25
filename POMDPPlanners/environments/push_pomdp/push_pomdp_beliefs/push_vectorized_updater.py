# SPDX-License-Identifier: MIT

"""Vectorized particle belief updater for the Push POMDP.

This module implements a concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
that performs batched state transitions and observation log-likelihood
evaluations for the Push environment.

The hot inner kernels (per-particle deterministic transition with optional
action-error coin flips, and per-particle 2-D Gaussian log-pdf on the
object-position slice) are implemented in C++ via the
``POMDPPlanners.environments.push_pomdp._native`` extension; the Python
class is a thin wrapper that pre-flattens obstacles and dispatches to
the native kernels.

Classes:
    PushVectorizedUpdater: Batched updater for the Push POMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.environments.push_pomdp import _native
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

    When the environment sets ``is_dangerous_area_hit_terminal`` the particles
    carry a trailing absorbing slot (7-D states) and the hazard geometry is
    threaded through to the native batch transition, which draws that slot on
    the C++ RNG. With the flag off the updater behaves exactly as before: 6-D
    states and no hazard kwargs on the native call.

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
    ACTION_NAME_TO_INDEX = {"up": 0, "down": 1, "right": 2, "left": 3}

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_dist: CovarianceParameterizedMultivariateNormal,
        grid_size: int,
        push_threshold: float,
        friction_coefficient: float,
        obstacles: np.ndarray,
        obstacle_radius: float,
        transition_error_prob: float,
        dangerous_areas_arr: Optional[np.ndarray] = None,
        dangerous_area_radius: float = 0.5,
        dangerous_area_hit_probability: float = 1.0,
        reward_variant_code: int = 0,
        penalty_decay: float = 1.0,
        is_dangerous_area_hit_terminal: bool = False,
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
            dangerous_areas_arr: Dangerous-area centres, shape (K, 2); ``None``
                means no dangerous areas.
            dangerous_area_radius: Hit radius around each dangerous area.
            dangerous_area_hit_probability: Probability a dangerous area is
                actually hit when the robot is inside its radius.
            reward_variant_code: Native code for the environment's reward
                variant (selects the hazard-draw semantics).
            penalty_decay: Decay constant for the distance-decayed variant.
            is_dangerous_area_hit_terminal: When ``True`` the particles carry a
                trailing absorbing slot and hazard hits are terminal.
        """
        self.obs_dist = obs_dist
        self.grid_size = grid_size
        self.push_threshold = push_threshold
        self.friction_coefficient = friction_coefficient
        self.obstacles = obstacles
        self.obstacle_radius = obstacle_radius
        self.transition_error_prob = transition_error_prob

        # Recover sigma from the diagonal isotropic covariance once at
        # construction time so each batch_observation_log_likelihood call
        # avoids redundant attribute lookups.
        self._observation_noise = float(np.sqrt(obs_dist.covariance[0, 0]))
        # Pre-flatten obstacles for the C++ kernel (one allocation here,
        # reused on every batch_transition call).
        self._obstacles_arr = np.ascontiguousarray(obstacles, dtype=np.float64)
        # Hazard-terminal parameters (only consumed when the flag is enabled;
        # the flag-off path builds the legacy 8-kwarg native call).
        self._dangerous_areas_arr = (
            np.empty((0, 2), dtype=np.float64)
            if dangerous_areas_arr is None
            else np.ascontiguousarray(dangerous_areas_arr, dtype=np.float64)
        )
        self._dangerous_area_radius = float(dangerous_area_radius)
        self._dangerous_area_hit_probability = float(dangerous_area_hit_probability)
        self._reward_variant_code = int(reward_variant_code)
        self._penalty_decay = float(penalty_decay)
        self._is_dangerous_area_hit_terminal = bool(is_dangerous_area_hit_terminal)
        self._hazard_terminal_enabled = self._is_dangerous_area_hit_terminal

    @classmethod
    def from_environment(cls, env: "PushPOMDP") -> "PushVectorizedUpdater":
        """Construct an updater from a PushPOMDP instance.

        Args:
            env: Environment to extract parameters from.

        Returns:
            A new ``PushVectorizedUpdater`` instance.
        """
        # pylint: disable=protected-access
        cov = np.diag([env.observation_noise**2, env.observation_noise**2])
        obs_dist = CovarianceParameterizedMultivariateNormal(cov)
        obstacles = np.array(env.obstacles, dtype=float) if env.obstacles else np.empty((0, 2))
        variant_code, penalty_decay = env._reward_variant_native_params()
        return cls(
            obs_dist=obs_dist,
            grid_size=env.grid_size,
            push_threshold=env.push_threshold,
            friction_coefficient=env.friction_coefficient,
            obstacles=obstacles,
            obstacle_radius=env.obstacle_radius,
            transition_error_prob=env.transition_error_prob,
            dangerous_areas_arr=env._dangerous_areas_arr,
            dangerous_area_radius=env.dangerous_area_radius,
            dangerous_area_hit_probability=env.dangerous_area_hit_probability,
            reward_variant_code=variant_code,
            penalty_decay=penalty_decay,
            is_dangerous_area_hit_terminal=env.is_dangerous_area_hit_terminal,
        )
        # pylint: enable=protected-access

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        # particles: (N, 6) = [robot_x, robot_y, obj_x, obj_y, target_x, target_y],
        # or (N, 7) with a trailing absorbing slot when the hazard-terminal flag
        # is set. In that case already-absorbing rows are echoed verbatim and the
        # slot for the remaining rows is drawn on the C++ RNG.
        action_idx = self.ACTION_NAME_TO_INDEX[action] if isinstance(action, str) else int(action)
        particles_arr = np.ascontiguousarray(particles, dtype=np.float64)
        if not self._hazard_terminal_enabled:
            return _native.belief_batch_transition_discrete(
                particles=particles_arr,
                action_idx=action_idx,
                transition_error_prob=float(self.transition_error_prob),
                obstacles=self._obstacles_arr,
                obstacle_radius=float(self.obstacle_radius),
                grid_size=float(self.grid_size),
                push_threshold=float(self.push_threshold),
                friction_coefficient=float(self.friction_coefficient),
            )
        return _native.belief_batch_transition_discrete(
            particles=particles_arr,
            action_idx=action_idx,
            transition_error_prob=float(self.transition_error_prob),
            obstacles=self._obstacles_arr,
            obstacle_radius=float(self.obstacle_radius),
            grid_size=float(self.grid_size),
            push_threshold=float(self.push_threshold),
            friction_coefficient=float(self.friction_coefficient),
            dangerous_areas=self._dangerous_areas_arr,
            dangerous_area_radius=float(self._dangerous_area_radius),
            dangerous_area_hit_probability=float(self._dangerous_area_hit_probability),
            reward_variant_code=int(self._reward_variant_code),
            penalty_decay=float(self._penalty_decay),
            is_dangerous_area_hit_terminal=True,
        )

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        # Accepts 6- or 7-wide particles and observations; with the
        # hazard-terminal flag the trailing absorbing slot is ignored, since
        # only the object-position slice carries observation noise.
        del action  # unused: observation model depends only on next_state
        observation_arr = np.ascontiguousarray(np.asarray(observation, dtype=np.float64).ravel())
        next_particles_arr = np.ascontiguousarray(next_particles, dtype=np.float64)
        return _native.belief_batch_obs_log_likelihood_discrete(
            next_particles=next_particles_arr,
            observation=observation_arr,
            observation_noise=self._observation_noise,
        )

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
        if self._hazard_terminal_enabled:
            config_dict["is_dangerous_area_hit_terminal"] = self._is_dangerous_area_hit_terminal
            config_dict["dangerous_areas"] = self._dangerous_areas_arr.tolist()
            config_dict["dangerous_area_radius"] = self._dangerous_area_radius
            config_dict["dangerous_area_hit_probability"] = self._dangerous_area_hit_probability
            config_dict["reward_variant_code"] = self._reward_variant_code
            config_dict["penalty_decay"] = self._penalty_decay
        return config_to_id(config_dict)
