"""Vectorized particle belief updater for the Continuous LaserTag POMDP.

This module implements a concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
that performs batched state transitions and observation log-likelihood
evaluations for the Continuous LaserTag environment, replacing per-particle
Python loops with NumPy array operations where possible.

Pre-computed Cholesky scalars for the 2-D robot and opponent transition
noise enable fast vectorized sampling, mirroring the approach used by
the Continuous Light-Dark vectorized updater.

Classes:
    ContinuousLaserTagVectorizedUpdater: Batched updater for the
        Continuous LaserTag POMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_geometry import (
    batch_clamp_to_grid,
    batch_laser_measurements,
    batch_resolve_wall_collision,
)
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
        ContinuousLaserTagPOMDP,
    )


class ContinuousLaserTagVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Vectorized particle belief updater for the Continuous LaserTag POMDP.

    Performs all-particle transitions and observation log-likelihood
    evaluations using vectorized NumPy operations.  Pre-computed Cholesky
    scalars for 2-D noise enable fast sampling without per-particle calls.

    Attributes:
        walls: Shape ``(M, 4)`` wall AABB array.
        grid_size: Shape ``(2,)`` arena dimensions.
        robot_radius: Robot body radius.
        opponent_radius: Opponent body radius.
        tag_radius: Maximum tag distance.
        pursuit_speed: Mean opponent pursuit step magnitude.
        measurement_noise: Laser measurement noise std.

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
        robot_L00: float,
        robot_L10: float,
        robot_L11: float,
        opponent_L00: float,
        opponent_L10: float,
        opponent_L11: float,
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
            robot_L00: Cholesky L[0,0] for robot transition noise.
            robot_L10: Cholesky L[1,0] for robot transition noise.
            robot_L11: Cholesky L[1,1] for robot transition noise.
            opponent_L00: Cholesky L[0,0] for opponent transition noise.
            opponent_L10: Cholesky L[1,0] for opponent transition noise.
            opponent_L11: Cholesky L[1,1] for opponent transition noise.
            action_to_vector: Optional mapping from string actions to
                3-D vectors (for discrete action variant).
        """
        self.walls = np.asarray(walls, dtype=float)
        self.grid_size = np.asarray(grid_size, dtype=float)
        self.robot_radius = robot_radius
        self.opponent_radius = opponent_radius
        self.tag_radius = tag_radius
        self.pursuit_speed = pursuit_speed
        self.measurement_noise = measurement_noise
        self._action_to_vector = action_to_vector

        # Robot transition Cholesky scalars
        self._robot_L00 = robot_L00
        self._robot_L10 = robot_L10
        self._robot_L11 = robot_L11

        # Opponent transition Cholesky scalars
        self._opp_L00 = opponent_L00
        self._opp_L10 = opponent_L10
        self._opp_L11 = opponent_L11

        # Observation log-likelihood constants
        self._variance = measurement_noise**2
        self._inv_2var = 0.5 / self._variance
        self._log_norm_1d = -0.5 * np.log(2.0 * np.pi * self._variance)

    @classmethod
    def from_environment(
        cls, env: "ContinuousLaserTagPOMDP"
    ) -> "ContinuousLaserTagVectorizedUpdater":
        """Construct an updater from a ContinuousLaserTagPOMDP instance.

        Args:
            env: Environment to extract parameters from.

        Returns:
            A new ``ContinuousLaserTagVectorizedUpdater`` instance.
        """
        # pylint: disable=protected-access
        robot_L = env._robot_transition_dist._cholesky_L
        opp_L = env._opponent_transition_dist._cholesky_L

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
            robot_L00=float(robot_L[0, 0]),
            robot_L10=float(robot_L[1, 0]),
            robot_L11=float(robot_L[1, 1]),
            opponent_L00=float(opp_L[0, 0]),
            opponent_L10=float(opp_L[1, 0]),
            opponent_L11=float(opp_L[1, 1]),
            action_to_vector=action_map,
        )
        # pylint: enable=protected-access

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        action_vec = self._resolve_action(action)
        result = particles.copy()

        terminal_mask = particles[:, 4] == 1.0
        live_idx = np.where(~terminal_mask)[0]
        if live_idx.size == 0:
            return result

        live = particles[live_idx]
        robot_pos = live[:, :2]
        opp_pos = live[:, 2:4]

        tag_flag = action_vec[2] if len(action_vec) > 2 else 0.0

        # Tag action: no robot movement or noise (matches discrete LaserTag)
        if tag_flag > 0.5:
            dists = np.linalg.norm(robot_pos - opp_pos, axis=1)
            tagged = dists <= self.tag_radius
            tag_idx = live_idx[tagged]
            non_tag_idx = live_idx[~tagged]

            result[tag_idx, 4] = 1.0

            if non_tag_idx.size > 0:
                new_opp = self._batch_opponent_move(robot_pos[~tagged], opp_pos[~tagged])
                result[non_tag_idx, 2:4] = new_opp
        else:
            new_robot = self._batch_robot_move(robot_pos, action_vec)
            new_opp = self._batch_opponent_move(new_robot, opp_pos)
            result[live_idx, :2] = new_robot
            result[live_idx, 2:4] = new_opp

        return result

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        observation = np.asarray(observation, dtype=float).ravel()
        n = next_particles.shape[0]
        log_ll = np.full(n, -np.inf)

        terminal_mask = next_particles[:, 4] == 1.0
        obs_is_terminal = np.allclose(observation, -1.0)

        if obs_is_terminal:
            log_ll[terminal_mask] = 0.0
            return log_ll

        non_term_idx = np.where(~terminal_mask)[0]
        if non_term_idx.size == 0:
            return log_ll

        measurements = batch_laser_measurements(
            next_particles[non_term_idx, :2],
            next_particles[non_term_idx, 2:4],
            self.opponent_radius,
            self.walls,
            self.grid_size,
        )
        diff = observation[None, :] - measurements
        log_ll[non_term_idx] = 8.0 * self._log_norm_1d - np.sum(diff**2, axis=1) * self._inv_2var
        return log_ll

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
        }
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

    def _batch_robot_move(self, robot_pos: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
        n = robot_pos.shape[0]
        movement = action_vec[:2]
        intended = robot_pos + movement

        z = np.random.standard_normal((n, 2))
        noise_x = z[:, 0] * self._robot_L00
        noise_y = z[:, 0] * self._robot_L10 + z[:, 1] * self._robot_L11

        new_pos = intended.copy()
        new_pos[:, 0] += noise_x
        new_pos[:, 1] += noise_y

        new_pos = batch_resolve_wall_collision(new_pos, self.robot_radius, self.walls)
        new_pos = batch_clamp_to_grid(new_pos, self.robot_radius, self.grid_size)
        return new_pos

    def _batch_opponent_move(self, robot_pos: np.ndarray, opp_pos: np.ndarray) -> np.ndarray:
        n = robot_pos.shape[0]
        diff = robot_pos - opp_pos
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-9)
        direction = diff / dist
        mean_opp = opp_pos + self.pursuit_speed * direction

        z = np.random.standard_normal((n, 2))
        noise_x = z[:, 0] * self._opp_L00
        noise_y = z[:, 0] * self._opp_L10 + z[:, 1] * self._opp_L11

        new_opp = mean_opp.copy()
        new_opp[:, 0] += noise_x
        new_opp[:, 1] += noise_y

        new_opp = batch_resolve_wall_collision(new_opp, self.opponent_radius, self.walls)
        new_opp = batch_clamp_to_grid(new_opp, self.opponent_radius, self.grid_size)
        return new_opp
