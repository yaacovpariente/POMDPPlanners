"""Vectorized particle belief updater for the LaserTag POMDP.

This module implements a concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
that performs batched state transitions and observation log-likelihood
evaluations for the LaserTag environment, replacing per-particle Python
loops with NumPy array operations.

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

# 5 action movement vectors: N, S, E, W, Tag
_ACTION_DIRECTIONS = np.array(
    [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]],
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
        if self.transition_error_prob > 0 and action_idx != 4:
            return self._batch_transition_with_error(particles, action_idx)
        return self._transition_for_action(particles, action_idx)

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
        obs_is_terminal = np.all(observation == -1.0)

        if obs_is_terminal:
            log_ll[terminal_mask] = 0.0
            return log_ll

        # Non-terminal observation: terminal particles get -inf (already set)
        non_term_idx = np.where(~terminal_mask)[0]
        if non_term_idx.size == 0:
            return log_ll

        log_ll[non_term_idx] = self._batch_non_terminal_log_likelihood(
            next_particles[non_term_idx], observation
        )
        return log_ll

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

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _transition_for_action(self, particles: np.ndarray, action_idx: int) -> np.ndarray:
        result = particles.copy()

        terminal_mask = particles[:, 4] == 1.0
        live_idx = np.where(~terminal_mask)[0]
        if live_idx.size == 0:
            return result

        live = particles[live_idx]
        robot_pos = live[:, :2].astype(int)
        opp_pos = live[:, 2:4].astype(int)

        # Robot movement
        new_robot = self._batch_robot_move(robot_pos, action_idx)

        # Tag check
        if action_idx == 4:
            tagged = np.all(new_robot == opp_pos, axis=1)
            tag_idx = live_idx[tagged]
            non_tag_idx = live_idx[~tagged]

            result[tag_idx, :2] = new_robot[tagged].astype(float)
            result[tag_idx, 4] = 1.0

            if non_tag_idx.size > 0:
                result[non_tag_idx, :2] = new_robot[~tagged].astype(float)
                new_opp = self._batch_opponent_move(new_robot[~tagged], opp_pos[~tagged])
                result[non_tag_idx, 2:4] = new_opp.astype(float)
        else:
            result[live_idx, :2] = new_robot.astype(float)
            new_opp = self._batch_opponent_move(new_robot, opp_pos)
            result[live_idx, 2:4] = new_opp.astype(float)

        return result

    def _batch_robot_move(self, robot_pos: np.ndarray, action_idx: int) -> np.ndarray:
        if action_idx == 4:
            return robot_pos.copy()
        movement = _ACTION_DIRECTIONS[action_idx]
        intended = robot_pos + movement
        valid = self._batch_is_valid(intended)
        return np.where(valid[:, None], intended, robot_pos)

    def _batch_is_valid(self, positions: np.ndarray) -> np.ndarray:
        rows, cols = self.floor_shape
        in_bounds = (
            (positions[:, 0] >= 0)
            & (positions[:, 0] < rows)
            & (positions[:, 1] >= 0)
            & (positions[:, 1] < cols)
        )
        result = np.zeros(len(positions), dtype=bool)
        ib_idx = np.where(in_bounds)[0]
        if ib_idx.size > 0:
            r = positions[ib_idx, 0]
            c = positions[ib_idx, 1]
            result[ib_idx] = self.valid_cell[r, c]
        return result

    def _batch_opponent_move(self, robot_pos: np.ndarray, opp_pos: np.ndarray) -> np.ndarray:
        k = robot_pos.shape[0]
        new_opp = opp_pos.copy()

        # All 4 cardinal direction targets
        right_target = np.column_stack([opp_pos[:, 0], opp_pos[:, 1] + 1])
        left_target = np.column_stack([opp_pos[:, 0], opp_pos[:, 1] - 1])
        up_target = np.column_stack([opp_pos[:, 0] - 1, opp_pos[:, 1]])
        down_target = np.column_stack([opp_pos[:, 0] + 1, opp_pos[:, 1]])

        right_valid = self._batch_is_valid(right_target)
        left_valid = self._batch_is_valid(left_target)
        up_valid = self._batch_is_valid(up_target)
        down_valid = self._batch_is_valid(down_target)

        # Horizontal: 0.4 toward robot, or 0.2/0.2 split when same column
        same_col = robot_pos[:, 1] == opp_pos[:, 1]
        right_prob = np.where(
            same_col & right_valid,
            0.2,
            np.where((robot_pos[:, 1] > opp_pos[:, 1]) & right_valid, 0.4, 0.0),
        )
        left_prob = np.where(
            same_col & left_valid,
            0.2,
            np.where((robot_pos[:, 1] < opp_pos[:, 1]) & left_valid, 0.4, 0.0),
        )

        # Vertical: 0.4 toward robot, or 0.2/0.2 split when same row
        same_row = robot_pos[:, 0] == opp_pos[:, 0]
        up_prob = np.where(
            same_row & up_valid,
            0.2,
            np.where((robot_pos[:, 0] < opp_pos[:, 0]) & up_valid, 0.4, 0.0),
        )
        down_prob = np.where(
            same_row & down_valid,
            0.2,
            np.where((robot_pos[:, 0] > opp_pos[:, 0]) & down_valid, 0.4, 0.0),
        )

        # 5-way categorical sampling (stay absorbs blocked mass)
        cum1 = right_prob
        cum2 = cum1 + left_prob
        cum3 = cum2 + up_prob
        cum4 = cum3 + down_prob

        u = np.random.random(k)
        choose_right = u < cum1
        choose_left = ~choose_right & (u < cum2)
        choose_up = ~choose_right & ~choose_left & (u < cum3)
        choose_down = ~choose_right & ~choose_left & ~choose_up & (u < cum4)

        new_opp[choose_right] = right_target[choose_right]
        new_opp[choose_left] = left_target[choose_left]
        new_opp[choose_up] = up_target[choose_up]
        new_opp[choose_down] = down_target[choose_down]

        return new_opp

    def _batch_transition_with_error(
        self, particles: np.ndarray, intended_action: int
    ) -> np.ndarray:
        n = particles.shape[0]

        # Compute results for all 4 movement actions
        all_results = np.empty((4, n, 5))
        for a in range(4):
            # Each call samples fresh opponent moves, which is fine:
            # we only keep the one selected per particle.
            all_results[a] = self._transition_for_action(particles, a)

        # Sample which action each particle actually executes
        error_mask = np.random.random(n) < self.transition_error_prob
        chosen_actions = np.full(n, intended_action, dtype=int)

        error_count = int(error_mask.sum())
        if error_count > 0:
            other_actions = [a for a in range(4) if a != intended_action]
            chosen_actions[error_mask] = np.random.choice(other_actions, size=error_count)

        return all_results[chosen_actions, np.arange(n)]

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _batch_non_terminal_log_likelihood(
        self, particles: np.ndarray, observation: np.ndarray
    ) -> np.ndarray:
        true_measurements = self._batch_laser_measurements(particles)
        diff = observation[None, :] - true_measurements  # (K, 8)
        log_ll = 8.0 * self._log_norm_1d - np.sum(diff**2, axis=1) * self._inv_2var
        return log_ll

    def _batch_laser_measurements(self, particles: np.ndarray) -> np.ndarray:
        robot_r = particles[:, 0].astype(int)
        robot_c = particles[:, 1].astype(int)
        opp_r = particles[:, 2].astype(int)
        opp_c = particles[:, 3].astype(int)

        # Wall distances from lookup table: (K, 8)
        wall_dist = self.wall_dist_table[robot_r, robot_c, :]

        # Compute opponent distance on each ray
        diff_r = opp_r - robot_r  # (K,)
        diff_c = opp_c - robot_c  # (K,)

        measurements = wall_dist.astype(float)

        for d in range(8):
            opp_dist = self._compute_opponent_distance_on_ray(diff_r, diff_c, d, wall_dist[:, d])
            valid = opp_dist >= 0
            measurements[valid, d] = np.minimum(
                measurements[valid, d], opp_dist[valid].astype(float)
            )

        return measurements

    def _compute_opponent_distance_on_ray(
        self,
        diff_r: np.ndarray,
        diff_c: np.ndarray,
        direction_idx: int,
        wall_dist: np.ndarray,
    ) -> np.ndarray:
        """Compute opponent distance along a ray direction for all particles.

        Returns -1 for particles where opponent is not on the ray.
        """
        k = diff_r.shape[0]
        dr = int(_LASER_DIRECTIONS[direction_idx, 0])
        dc = int(_LASER_DIRECTIONS[direction_idx, 1])

        result = np.full(k, -1, dtype=int)

        if dr != 0 and dc != 0:
            # Diagonal: both coordinates must give same positive integer k
            k_r = np.where(dr != 0, diff_r / dr, 0.0)
            k_c = np.where(dc != 0, diff_c / dc, 0.0)
            is_int_r = (k_r == np.floor(k_r)) & (k_r >= 1)
            is_int_c = (k_c == np.floor(k_c)) & (k_c >= 1)
            same_k = np.abs(k_r - k_c) < 0.5
            valid = is_int_r & is_int_c & same_k
            step = k_r.astype(int)
            on_ray = valid & (step >= 1) & ((step - 1) <= wall_dist)
            result[on_ray] = step[on_ray] - 1
        elif dr != 0:
            # Vertical ray: dc==0, so diff_c must be 0
            k_r = np.where(dr != 0, diff_r / dr, 0.0)
            is_int = (k_r == np.floor(k_r)) & (k_r >= 1)
            same_col = diff_c == 0
            step = k_r.astype(int)
            on_ray = is_int & same_col & (step >= 1) & ((step - 1) <= wall_dist)
            result[on_ray] = step[on_ray] - 1
        elif dc != 0:
            # Horizontal ray: dr==0, so diff_r must be 0
            k_c = np.where(dc != 0, diff_c / dc, 0.0)
            is_int = (k_c == np.floor(k_c)) & (k_c >= 1)
            same_row = diff_r == 0
            step = k_c.astype(int)
            on_ray = is_int & same_row & (step >= 1) & ((step - 1) <= wall_dist)
            result[on_ray] = step[on_ray] - 1

        return result
