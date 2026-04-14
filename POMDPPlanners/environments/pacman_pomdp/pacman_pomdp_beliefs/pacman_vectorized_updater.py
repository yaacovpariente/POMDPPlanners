"""Vectorized particle belief updater for the PacMan POMDP.

This module implements a concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
that performs batched state transitions and observation log-likelihood
evaluations for the PacMan environment, replacing per-particle Python loops
with NumPy array operations.

Classes:
    PacManVectorizedUpdater: Batched updater for the PacMan POMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_grid_utils import (
    precompute_neighbor_table,
    precompute_neighbor_validity,
    precompute_valid_cell_mask,
)
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP


class PacManVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Vectorized particle belief updater for PacMan POMDP.

    Performs all-particle transitions and observation log-likelihood
    evaluations using vectorized NumPy operations. Ghost movement uses
    batched softmax sampling, and collision/pellet logic operates on
    the full particle array at once.

    Attributes:
        maze_size: Grid dimensions (rows, cols).
        num_ghosts: Number of ghosts.
        num_pellets: Number of initial pellets.
        state_dim: Dimensionality of the array state.
        ghost_aggressiveness: Softmax temperature for ghost pursuit.
        ghost_coordination: Ghost coordination mode.
        ghost_strategies: Per-ghost strategy list.
        observation_noise_factor: Multiplier for observation noise.
        max_observation_noise: Maximum observation noise std.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        maze_size: Tuple[int, int],
        num_ghosts: int,
        num_pellets: int,
        state_dim: int,
        neighbor_table: np.ndarray,
        neighbor_validity: np.ndarray,
        pellet_positions: np.ndarray,
        ghost_aggressiveness: float,
        ghost_coordination: str,
        ghost_strategies: List[str],
        observation_noise_factor: float,
        max_observation_noise: float,
        idx_pac_row: int,
        idx_pac_col: int,
        idx_ghosts_start: int,
        idx_pellets_start: int,
        idx_pellets_end: int,
        idx_score: int,
        idx_terminal: int,
    ):
        self.maze_size = maze_size
        self.num_ghosts = num_ghosts
        self.num_pellets = num_pellets
        self.state_dim = state_dim
        self.neighbor_table = neighbor_table
        self.neighbor_validity = neighbor_validity
        self.pellet_positions = pellet_positions
        self.ghost_aggressiveness = ghost_aggressiveness
        self.ghost_coordination = ghost_coordination
        self.ghost_strategies = ghost_strategies
        self.observation_noise_factor = observation_noise_factor
        self.max_observation_noise = max_observation_noise
        self._idx_pac_row = idx_pac_row
        self._idx_pac_col = idx_pac_col
        self._idx_ghosts_start = idx_ghosts_start
        self._idx_pellets_start = idx_pellets_start
        self._idx_pellets_end = idx_pellets_end
        self._idx_score = idx_score
        self._idx_terminal = idx_terminal

    @classmethod
    def from_environment(cls, env: "PacManPOMDP") -> "PacManVectorizedUpdater":
        """Construct an updater from a PacManPOMDP instance.

        Args:
            env: Environment to extract parameters from.

        Returns:
            A new ``PacManVectorizedUpdater`` instance.
        """
        valid_mask = precompute_valid_cell_mask(env.maze_size, env.walls)
        neighbor_table = precompute_neighbor_table(env.maze_size, valid_mask)
        neighbor_valid = precompute_neighbor_validity(env.maze_size, valid_mask)
        pellet_pos = np.array(
            env._all_pellet_positions, dtype=np.int32  # pylint: disable=protected-access
        ).reshape(-1, 2)

        return cls(
            maze_size=env.maze_size,
            num_ghosts=env.num_ghosts,
            num_pellets=env._num_initial_pellets,  # pylint: disable=protected-access
            state_dim=env._state_dim,  # pylint: disable=protected-access
            neighbor_table=neighbor_table,
            neighbor_validity=neighbor_valid,
            pellet_positions=pellet_pos,
            ghost_aggressiveness=env.ghost_aggressiveness,
            ghost_coordination=env.ghost_coordination,
            ghost_strategies=env.ghost_strategies,
            observation_noise_factor=env.observation_noise_factor,
            max_observation_noise=env.max_observation_noise,
            idx_pac_row=env._idx_pac_row,  # pylint: disable=protected-access
            idx_pac_col=env._idx_pac_col,  # pylint: disable=protected-access
            idx_ghosts_start=env._idx_ghosts_start,  # pylint: disable=protected-access
            idx_pellets_start=env._idx_pellets_start,  # pylint: disable=protected-access
            idx_pellets_end=env._idx_pellets_end,  # pylint: disable=protected-access
            idx_score=env._idx_score,  # pylint: disable=protected-access
            idx_terminal=env._idx_terminal,  # pylint: disable=protected-access
        )

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(  # pylint: disable=arguments-renamed
        self, particles: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        action_idx = int(action)
        result = particles.copy()
        is_terminal = result[:, self._idx_terminal] > 0.5

        if np.all(is_terminal):
            return result

        live = ~is_terminal
        self._apply_pacman_movement(result, action_idx, live)
        self._apply_ghost_movement(result, live)
        self._apply_collisions_and_pellets(result, live)
        return result

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,  # pylint: disable=unused-argument
        observation: np.ndarray,
    ) -> np.ndarray:
        del action  # observation likelihood is action-independent for PacMan
        observation = np.asarray(observation, dtype=np.float64).ravel()
        n = next_particles.shape[0]
        is_terminal = next_particles[:, self._idx_terminal] > 0.5
        obs_is_terminal = np.all(observation < -0.5)

        log_ll = np.full(n, -np.inf)

        if obs_is_terminal:
            log_ll[is_terminal] = 0.0
            return log_ll

        log_ll[is_terminal] = -np.inf
        live = ~is_terminal
        if not np.any(live):
            return log_ll

        log_ll[live] = self._compute_live_observation_log_ll(next_particles[live], observation)
        return log_ll

    @property
    def config_id(self) -> str:
        config_dict = {
            "class": "PacManVectorizedUpdater",
            "maze_size": list(self.maze_size),
            "num_ghosts": self.num_ghosts,
            "num_pellets": self.num_pellets,
            "ghost_aggressiveness": self.ghost_aggressiveness,
            "ghost_coordination": self.ghost_coordination,
            "ghost_strategies": self.ghost_strategies,
            "observation_noise_factor": self.observation_noise_factor,
            "max_observation_noise": self.max_observation_noise,
        }
        return config_to_id(config_dict)

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _apply_pacman_movement(self, result: np.ndarray, action_idx: int, live: np.ndarray) -> None:
        pac_r = result[live, self._idx_pac_row].astype(np.int32)
        pac_c = result[live, self._idx_pac_col].astype(np.int32)
        new_pos = self.neighbor_table[pac_r, pac_c, action_idx]
        result[live, self._idx_pac_row] = new_pos[:, 0]
        result[live, self._idx_pac_col] = new_pos[:, 1]

    def _apply_ghost_movement(self, result: np.ndarray, live: np.ndarray) -> None:
        for g in range(self.num_ghosts):
            g_row_idx = self._idx_ghosts_start + 2 * g
            g_col_idx = self._idx_ghosts_start + 2 * g + 1
            g_rows = result[live, g_row_idx].astype(np.int32)
            g_cols = result[live, g_col_idx].astype(np.int32)
            pac_r = result[live, self._idx_pac_row].astype(np.int32)
            pac_c = result[live, self._idx_pac_col].astype(np.int32)

            strategy = self._get_ghost_strategy(g)
            if strategy == "aggressive":
                new_r, new_c = self._batch_move_aggressive(g_rows, g_cols, pac_r, pac_c)
            elif strategy == "patrol":
                new_r, new_c = self._batch_move_patrol(g_rows, g_cols)
            elif strategy == "ambush":
                new_r, new_c = self._batch_move_ambush(g_rows, g_cols, pac_r, pac_c)
            else:
                new_r, new_c = self._batch_move_aggressive(g_rows, g_cols, pac_r, pac_c)

            result[live, g_row_idx] = new_r
            result[live, g_col_idx] = new_c

    def _get_ghost_strategy(self, ghost_id: int) -> str:
        if self.ghost_coordination == "coordinated":
            return "coordinated"
        if self.ghost_coordination == "mixed":
            return "aggressive" if ghost_id % 2 == 0 else "patrol"
        if ghost_id < len(self.ghost_strategies):
            return self.ghost_strategies[ghost_id]
        return "aggressive"

    def _batch_move_aggressive(
        self,
        g_rows: np.ndarray,
        g_cols: np.ndarray,
        pac_r: np.ndarray,
        pac_c: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._batch_softmax_move_toward(
            g_rows, g_cols, pac_r, pac_c, self.ghost_aggressiveness
        )

    def _batch_softmax_move_toward(
        self,
        g_rows: np.ndarray,
        g_cols: np.ndarray,
        target_r: np.ndarray,
        target_c: np.ndarray,
        temperature: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(g_rows)
        # For each of 5 moves, compute resulting positions
        # neighbor_table shape: (rows, cols, 5, 2)
        all_next = self.neighbor_table[g_rows, g_cols]  # (n, 5, 2)
        valid = self.neighbor_validity[g_rows, g_cols]  # (n, 5)

        # Manhattan distance to target for each move
        dist: np.ndarray = np.abs(
            all_next[:, :, 0].astype(np.float64) - target_r[:, None].astype(np.float64)
        ) + np.abs(
            all_next[:, :, 1].astype(np.float64) - target_c[:, None].astype(np.float64)
        )  # (n, 5)

        # Softmax scores: negative distance / temperature
        scores: np.ndarray = np.where(valid, -dist / temperature, -1e9)  # type: ignore[operator]
        scores_max = scores.max(axis=1, keepdims=True)  # pylint: disable=unexpected-keyword-arg
        exp_scores = np.exp(scores - scores_max)
        exp_scores = np.where(valid, exp_scores, 0.0)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        # Sample using cumulative probability
        cum_probs = np.cumsum(probs, axis=1)
        u = np.random.random(n)[:, None]
        chosen = (u < cum_probs).argmax(axis=1)

        new_r = all_next[np.arange(n), chosen, 0]
        new_c = all_next[np.arange(n), chosen, 1]
        return new_r.astype(np.float64), new_c.astype(np.float64)

    def _batch_move_patrol(
        self, g_rows: np.ndarray, g_cols: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(g_rows)
        valid = self.neighbor_validity[g_rows, g_cols]  # (n, 5)
        # Uniform random over valid moves
        probs = np.where(valid, 1.0, 0.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        cum_probs = np.cumsum(probs, axis=1)
        u = np.random.random(n)[:, None]
        chosen = (u < cum_probs).argmax(axis=1)

        all_next = self.neighbor_table[g_rows, g_cols]  # (n, 5, 2)
        new_r = all_next[np.arange(n), chosen, 0]
        new_c = all_next[np.arange(n), chosen, 1]
        return new_r.astype(np.float64), new_c.astype(np.float64)

    def _batch_move_ambush(
        self,
        g_rows: np.ndarray,
        g_cols: np.ndarray,
        pac_r: np.ndarray,
        pac_c: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(g_rows)
        all_next = self.neighbor_table[g_rows, g_cols]  # (n, 5, 2)
        valid = self.neighbor_validity[g_rows, g_cols]  # (n, 5)

        # Prefer positions 2-4 tiles from pacman
        dist: np.ndarray = np.abs(
            all_next[:, :, 0].astype(np.float64) - pac_r[:, None].astype(np.float64)
        ) + np.abs(all_next[:, :, 1].astype(np.float64) - pac_c[:, None].astype(np.float64))
        in_range = (dist >= 2) & (dist <= 4)
        scores: np.ndarray = np.where(in_range, -dist, -(dist + 10))  # type: ignore[operator]
        scores = np.where(valid, scores, -1e9)

        # Deterministic: pick best move
        chosen = scores.argmax(axis=1)
        new_r = all_next[np.arange(n), chosen, 0]
        new_c = all_next[np.arange(n), chosen, 1]
        return new_r.astype(np.float64), new_c.astype(np.float64)

    def _apply_collisions_and_pellets(self, result: np.ndarray, live: np.ndarray) -> None:
        live_idx = np.where(live)[0]
        pac_r = result[live_idx, self._idx_pac_row].astype(np.int32)
        pac_c = result[live_idx, self._idx_pac_col].astype(np.int32)

        # Check ghost collisions
        collision = self._check_ghost_collisions(result, live_idx, pac_r, pac_c)
        result[live_idx[collision], self._idx_terminal] = 1.0

        # Check pellet collection (only for non-collision particles)
        not_collided = ~collision
        self._collect_pellets(
            result, live_idx[not_collided], pac_r[not_collided], pac_c[not_collided]
        )

        # Check win condition
        self._check_win_condition(result, live_idx[not_collided])

    def _check_ghost_collisions(
        self,
        result: np.ndarray,
        live_idx: np.ndarray,
        pac_r: np.ndarray,
        pac_c: np.ndarray,
    ) -> np.ndarray:
        collision = np.zeros(len(live_idx), dtype=bool)
        for g in range(self.num_ghosts):
            g_row_idx = self._idx_ghosts_start + 2 * g
            g_col_idx = self._idx_ghosts_start + 2 * g + 1
            g_r = result[live_idx, g_row_idx].astype(np.int32)
            g_c = result[live_idx, g_col_idx].astype(np.int32)
            collision |= (pac_r == g_r) & (pac_c == g_c)
        return collision

    def _collect_pellets(
        self,
        result: np.ndarray,
        idx: np.ndarray,
        pac_r: np.ndarray,
        pac_c: np.ndarray,
    ) -> None:
        if len(idx) == 0 or self.num_pellets == 0:
            return
        pellet_pos = self.pellet_positions  # (P, 2)
        for p in range(self.num_pellets):
            p_col_idx = self._idx_pellets_start + p
            on_pellet = (pac_r == pellet_pos[p, 0]) & (pac_c == pellet_pos[p, 1])
            active = result[idx, p_col_idx] > 0.5
            collected = on_pellet & active
            result[idx[collected], p_col_idx] = 0.0
            result[idx[collected], self._idx_score] += 1.0

    def _check_win_condition(self, result: np.ndarray, idx: np.ndarray) -> None:
        if len(idx) == 0 or self.num_pellets == 0:
            return
        pellet_mask = result[idx, self._idx_pellets_start : self._idx_pellets_end]
        all_collected = pellet_mask.sum(axis=1) < 0.5
        result[idx[all_collected], self._idx_terminal] = 1.0

    # ------------------------------------------------------------------
    # Observation log-likelihood helpers
    # ------------------------------------------------------------------

    def _compute_live_observation_log_ll(
        self, particles: np.ndarray, observation: np.ndarray
    ) -> np.ndarray:
        pac_r = particles[:, self._idx_pac_row]
        pac_c = particles[:, self._idx_pac_col]
        total_log_ll = np.zeros(particles.shape[0])

        for g in range(self.num_ghosts):
            g_row_idx = self._idx_ghosts_start + 2 * g
            g_col_idx = self._idx_ghosts_start + 2 * g + 1
            true_g_r = particles[:, g_row_idx]
            true_g_c = particles[:, g_col_idx]

            obs_g_r = observation[2 * g]
            obs_g_c = observation[2 * g + 1]

            log_ll_g = self._ghost_observation_log_likelihood(
                true_g_r, true_g_c, pac_r, pac_c, obs_g_r, obs_g_c
            )
            total_log_ll += log_ll_g

        return total_log_ll

    def _ghost_observation_log_likelihood(
        self,
        true_g_r: np.ndarray,
        true_g_c: np.ndarray,
        pac_r: np.ndarray,
        pac_c: np.ndarray,
        obs_g_r: float,
        obs_g_c: float,
    ) -> np.ndarray:
        manhattan_dist = np.abs(true_g_r - pac_r) + np.abs(true_g_c - pac_c)
        noise_std = np.minimum(
            manhattan_dist * self.observation_noise_factor,
            self.max_observation_noise,
        )
        noise_std = np.maximum(noise_std, 1e-6)
        variance = noise_std**2

        row_diff = obs_g_r - true_g_r
        col_diff = obs_g_c - true_g_c
        dist_sq = row_diff**2 + col_diff**2

        # Isotropic 2D Gaussian log-PDF
        log_norm = -np.log(2.0 * np.pi * variance)
        log_prob = log_norm - dist_sq / (2.0 * variance)
        return log_prob
