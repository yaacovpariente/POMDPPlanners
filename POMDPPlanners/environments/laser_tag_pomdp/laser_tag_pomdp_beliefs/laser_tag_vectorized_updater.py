"""Vectorized particle belief updater for the LaserTag POMDP.

This module implements a concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
that performs batched state transitions and observation log-likelihood
evaluations for the LaserTag environment.

The hot inner kernels (per-particle transition with opponent-move sampling
and per-particle 8-direction laser measurement / Gaussian log-likelihood)
are implemented in C++ via the
``POMDPPlanners.environments.laser_tag_pomdp._native`` extension; the
Python class is a thin wrapper that pre-flattens the lookup tables and
dispatches to the native kernels.

Wall-distance lookup tables are precomputed once at construction time so
that laser measurements and collision checks use fast array indexing rather
than per-ray Python loops at update time.

Classes:
    LaserTagVectorizedUpdater: Batched updater for the LaserTag POMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.environments.laser_tag_pomdp import _native
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import (
        LaserTagPOMDP,
    )

# 8 laser directions: N, NE, E, SE, S, SW, W, NW
_LASER_DIRECTIONS = np.array(
    [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]],
    dtype=int,
)


def _precompute_wall_distances(rows: int, cols: int, valid_cell: np.ndarray) -> np.ndarray:
    """Build wall distance table for every cell and laser direction.

    For each cell (r, c) and each of the 8 directions, counts how many
    clear cells lie before the first wall or grid boundary.

    Args:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        valid_cell: Boolean array of shape (rows, cols).

    Returns:
        Integer array of shape (rows, cols, 8) with wall distances.
    """
    table = np.zeros((rows, cols, 8), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if not valid_cell[r, c]:
                continue
            for d in range(8):
                dr, dc = int(_LASER_DIRECTIONS[d, 0]), int(_LASER_DIRECTIONS[d, 1])
                dist = 0
                nr, nc = r + dr, c + dc
                while 0 <= nr < rows and 0 <= nc < cols:
                    if not valid_cell[nr, nc]:
                        break
                    dist += 1
                    nr += dr
                    nc += dc
                table[r, c, d] = dist
    return table


class LaserTagVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Vectorized particle belief updater for the LaserTag POMDP.

    Performs all-particle transitions and observation log-likelihood
    evaluations using vectorized NumPy operations.  Precomputed lookup
    tables for wall distances and valid cells replace per-ray Python
    loops at update time.

    Attributes:
        floor_shape: Grid dimensions as (rows, cols).
        valid_cell: Boolean array of shape (rows, cols).
        wall_dist_table: Integer array of shape (rows, cols, 8).
        measurement_noise: Standard deviation of laser measurement noise.
        transition_error_prob: Probability of executing a random movement.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP
        >>> env = LaserTagPOMDP(discount_factor=0.95)
        >>> updater = LaserTagVectorizedUpdater.from_environment(env)
        >>> state = env.initial_state_dist().sample()[0]
        >>> particles = np.tile(state, (50, 1))
        >>> action = 0  # North
        >>> next_p = updater.batch_transition(particles, action)
        >>> next_p.shape[1]
        5
        >>> obs = env.sample_observation(next_state=state, action=action)
        >>> obs_arr = np.array(obs, dtype=float)
        >>> ll = updater.batch_observation_log_likelihood(next_p, action, obs_arr)
        >>> ll.shape[0]
        50
    """

    def __init__(
        self,
        floor_shape: Tuple[int, int],
        valid_cell: np.ndarray,
        wall_dist_table: np.ndarray,
        measurement_noise: float,
        transition_error_prob: float,
    ):
        """Initialize the vectorized updater.

        Args:
            floor_shape: Grid dimensions as (rows, cols).
            valid_cell: Boolean array of shape (rows, cols).
            wall_dist_table: Integer array of shape (rows, cols, 8).
            measurement_noise: Standard deviation of laser noise.
            transition_error_prob: Probability of executing a random action.
        """
        self.floor_shape = floor_shape
        self.valid_cell = valid_cell
        self.wall_dist_table = wall_dist_table
        self.measurement_noise = measurement_noise
        self.transition_error_prob = transition_error_prob

        # Precompute observation constants
        self._variance = measurement_noise**2
        self._inv_2var = 0.5 / self._variance
        self._log_norm_1d = -0.5 * np.log(2.0 * np.pi * self._variance)

        # Pre-flatten the lookup tables for the C++ kernels (one allocation
        # at construction time, reused on every batch_transition /
        # batch_observation_log_likelihood call).
        self._valid_cell_flat = np.ascontiguousarray(valid_cell.astype(np.uint8).ravel())
        self._wall_dist_table_flat = np.ascontiguousarray(wall_dist_table.astype(np.int32).ravel())
        self._rows = int(floor_shape[0])
        self._cols = int(floor_shape[1])

    @classmethod
    def from_environment(cls, env: "LaserTagPOMDP") -> "LaserTagVectorizedUpdater":
        """Construct an updater from a LaserTagPOMDP instance.

        Args:
            env: Environment to extract parameters from.

        Returns:
            A new ``LaserTagVectorizedUpdater`` instance.
        """
        rows, cols = env.floor_shape
        valid_cell = np.ones((rows, cols), dtype=bool)
        for wr, wc in env.walls:
            valid_cell[wr, wc] = False
        wall_dist_table = _precompute_wall_distances(rows, cols, valid_cell)
        return cls(
            floor_shape=env.floor_shape,
            valid_cell=valid_cell,
            wall_dist_table=wall_dist_table,
            measurement_noise=env.measurement_noise,
            transition_error_prob=env.transition_error_prob,
        )

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        # particles: (N, 5) = [robot_row, robot_col, opp_row, opp_col, terminal]
        action_idx = int(action)
        particles_arr = np.ascontiguousarray(particles, dtype=np.float64)
        return _native.belief_batch_transition_discrete(
            particles=particles_arr,
            action_idx=action_idx,
            transition_error_prob=float(self.transition_error_prob),
            valid_cell_flat=self._valid_cell_flat,
            rows=self._rows,
            cols=self._cols,
        )

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        del action  # unused: observation model depends only on next_state
        observation_arr = np.ascontiguousarray(np.asarray(observation, dtype=np.float64).ravel())
        next_particles_arr = np.ascontiguousarray(next_particles, dtype=np.float64)
        return _native.belief_batch_obs_log_likelihood_discrete(
            next_particles=next_particles_arr,
            observation=observation_arr,
            wall_dist_table_flat=self._wall_dist_table_flat,
            rows=self._rows,
            cols=self._cols,
            log_norm_1d=float(self._log_norm_1d),
            inv_2var=float(self._inv_2var),
        )

    @property
    def config_id(self) -> str:
        config_dict = {
            "class": "LaserTagVectorizedUpdater",
            "floor_shape": list(self.floor_shape),
            "valid_cell": self.valid_cell.tolist(),
            "measurement_noise": self.measurement_noise,
            "transition_error_prob": self.transition_error_prob,
        }
        return config_to_id(config_dict)
