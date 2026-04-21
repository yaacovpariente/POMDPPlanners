"""Vectorized particle belief updater for the Continuous LaserTag POMDP.

This module implements a concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
that performs batched state transitions and observation log-likelihood
evaluations for the Continuous LaserTag environment.

``batch_transition`` delegates to the native ``_native`` C++ extension's
``ContinuousLaserTagTransitionCpp.batch_sample`` so both the explicit
vectorized path and the per-particle :class:`ContinuousLaserTagStateTransitionModel`
path share the same C++ RNG (the Python-side numpy loop used before was
what forced the cross-path skip in the baseline equivalence tests).
``batch_observation_log_likelihood`` delegates to
``ContinuousLaserTagObservationCpp.batch_log_likelihood``.

Classes:
    ContinuousLaserTagVectorizedUpdater: Batched updater for the
        Continuous LaserTag POMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.environments.laser_tag_pomdp import _native
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
        ContinuousLaserTagPOMDP,
    )


class ContinuousLaserTagVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Vectorized particle belief updater for the Continuous LaserTag POMDP.

    Performs all-particle transitions and observation log-likelihood
    evaluations by delegating to the native ``_native`` C++ extension's
    batch entry points. The explicit vectorized path and the per-particle
    :class:`ContinuousLaserTagStateTransitionModel` path now share the same
    C++ RNG, closing the cross-path divergence that previously forced the
    equivalence tests to seed each particle individually.

    Attributes:
        walls: Shape ``(M, 4)`` wall AABB array.
        grid_size: Shape ``(2,)`` arena dimensions.
        robot_radius: Robot body radius.
        opponent_radius: Opponent body radius.
        tag_radius: Maximum tag distance.
        pursuit_speed: Mean opponent pursuit step magnitude.
        measurement_noise: Laser measurement noise std.
        robot_covariance: Shape ``(2, 2)`` robot transition noise covariance.
        opponent_covariance: Shape ``(2, 2)`` opponent transition noise
            covariance.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
        ...     ContinuousLaserTagPOMDP,
        ... )
        >>> env = ContinuousLaserTagPOMDP(discount_factor=0.95)
        >>> updater = ContinuousLaserTagVectorizedUpdater.from_environment(env)
        >>> state = env.initial_state_dist().sample()[0]
        >>> particles = np.tile(state, (50, 1))
        >>> action = np.array([1.0, 0.0, 0.0])
        >>> next_p = updater.batch_transition(particles, action)
        >>> next_p.shape[1]
        5
        >>> obs = env.observation_model(state, action).sample()[0]
        >>> ll = updater.batch_observation_log_likelihood(next_p, action, obs)
        >>> ll.shape[0]
        50
    """

    def __init__(
        self,
        walls: np.ndarray,
        grid_size: np.ndarray,
        robot_radius: float,
        opponent_radius: float,
        tag_radius: float,
        pursuit_speed: float,
        measurement_noise: float,
        robot_covariance: np.ndarray,
        opponent_covariance: np.ndarray,
        action_to_vector: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Initialize the vectorized updater.

        Args:
            walls: Wall AABB array of shape ``(M, 4)``.
            grid_size: Arena dimensions ``(width, height)``.
            robot_radius: Robot body radius.
            opponent_radius: Opponent body radius.
            tag_radius: Maximum tag distance.
            pursuit_speed: Mean opponent pursuit step magnitude.
            measurement_noise: Laser measurement noise std.
            robot_covariance: Robot transition noise covariance ``(2, 2)``.
            opponent_covariance: Opponent transition noise covariance
                ``(2, 2)``.
            action_to_vector: Optional mapping from string actions to
                3-D vectors (for discrete action variant).
        """
        self.walls = np.asarray(walls, dtype=float).reshape(-1, 4)
        self.grid_size = np.asarray(grid_size, dtype=float)
        self.robot_radius = robot_radius
        self.opponent_radius = opponent_radius
        self.tag_radius = tag_radius
        self.pursuit_speed = pursuit_speed
        self.measurement_noise = measurement_noise
        self.robot_covariance = np.asarray(robot_covariance, dtype=float)
        self.opponent_covariance = np.asarray(opponent_covariance, dtype=float)
        self._action_to_vector = action_to_vector

    @classmethod
    def from_environment(
        cls, env: "ContinuousLaserTagPOMDP"
    ) -> "ContinuousLaserTagVectorizedUpdater":
        """Construct an updater from a ContinuousLaserTagPOMDP instance."""
        has_discrete_actions = (
            hasattr(env, "space_info") and env.space_info.action_space == SpaceType.DISCRETE
        )
        action_map = getattr(env, "action_to_vector", None) if has_discrete_actions else None

        return cls(
            walls=env.walls,
            grid_size=env.grid_size,
            robot_radius=env.robot_radius,
            opponent_radius=env.opponent_radius,
            tag_radius=env.tag_radius,
            pursuit_speed=env.pursuit_speed,
            measurement_noise=env.measurement_noise,
            robot_covariance=env.robot_transition_cov_matrix,
            opponent_covariance=env.opponent_transition_cov_matrix,
            action_to_vector=action_map,
        )

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        action_vec = self._resolve_action(action)
        particles_arr = np.asarray(particles, dtype=float)
        # The state passed to the ctor is unused on the batch path; only the
        # ctor signature requires a reference row. The first particle row is
        # valid whether or not particles is empty (an empty input is a
        # pathological case that should not reach the belief update).
        ref_state = particles_arr[0] if particles_arr.shape[0] > 0 else np.zeros(5)
        transition = _native.ContinuousLaserTagTransitionCpp(
            state=ref_state,
            action=action_vec,
            robot_covariance=self.robot_covariance,
            opponent_covariance=self.opponent_covariance,
            pursuit_speed=self.pursuit_speed,
            walls=self.walls,
            grid_size=self.grid_size,
            robot_radius=self.robot_radius,
            opponent_radius=self.opponent_radius,
            tag_radius=self.tag_radius,
        )
        return transition.batch_sample(particles_arr)

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        action_vec = self._resolve_action(action)
        next_arr = np.asarray(next_particles, dtype=float)
        observation_arr = np.asarray(observation, dtype=float).ravel()
        ref_state = next_arr[0] if next_arr.shape[0] > 0 else np.zeros(5)
        obs_model = _native.ContinuousLaserTagObservationCpp(
            next_state=ref_state,
            action=action_vec,
            measurement_noise=self.measurement_noise,
            walls=self.walls,
            grid_size=self.grid_size,
            opponent_radius=self.opponent_radius,
        )
        return obs_model.batch_log_likelihood(next_arr, observation_arr)

    @property
    def config_id(self) -> str:
        config_dict = {
            "class": "ContinuousLaserTagVectorizedUpdater",
            "walls": self.walls.tolist(),
            "grid_size": self.grid_size.tolist(),
            "robot_radius": self.robot_radius,
            "opponent_radius": self.opponent_radius,
            "tag_radius": self.tag_radius,
            "pursuit_speed": self.pursuit_speed,
            "measurement_noise": self.measurement_noise,
            "robot_covariance": self.robot_covariance.tolist(),
            "opponent_covariance": self.opponent_covariance.tolist(),
        }
        if self._action_to_vector is not None:
            config_dict["action_to_vector"] = {
                k: v.tolist() for k, v in self._action_to_vector.items()
            }
        return config_to_id(config_dict)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_action(self, action: Any) -> np.ndarray:
        if self._action_to_vector is not None and isinstance(action, str):
            return np.asarray(self._action_to_vector[action], dtype=float).ravel()
        return np.asarray(action, dtype=float).ravel()
