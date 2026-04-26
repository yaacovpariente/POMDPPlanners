"""Type stubs for the PacMan POMDP native C++ extension."""

# pylint: disable=unused-argument,unnecessary-ellipsis

from typing import List, Sequence, Union

import numpy as np
from numpy.typing import NDArray

def set_seed(seed: int) -> None:
    """Seed the module-local RNG used by ``sample()`` / batch entry points."""

def simulate_rollout(
    state: NDArray[np.float64],
    action_indices: NDArray[np.int32],
    maze_rows: int,
    maze_cols: int,
    neighbor_table: NDArray[np.int32],
    neighbor_validity: NDArray[np.uint8],
    pellet_positions: NDArray[np.int32],
    ghost_aggressiveness: float,
    ghost_coordination_code: int,
    ghost_strategy_codes: NDArray[np.int32],
    num_ghosts: int,
    num_pellets: int,
    pellet_reward: float,
    idx_pac_row: int,
    idx_pac_col: int,
    idx_ghosts_start: int,
    idx_pellets_start: int,
    idx_pellets_end: int,
    idx_score: int,
    idx_terminal: int,
    patrol_dir_state: NDArray[np.int32],
    ghost_collision_penalty: float,
    step_penalty: float,
    win_reward: float,
    discount_factor: float,
    depth: int,
    max_depth: int,
) -> float:
    """Run a random rollout from state using pre-drawn action_indices.

    Returns the discounted cumulative reward accumulated until terminal or
    max_depth is reached. action_indices must have length >= (max_depth - depth).
    """

class PacManTransitionCpp:
    """Native transition kernel for PacMan POMDP (pybind11-backed)."""

    def __init__(
        self,
        state: NDArray[np.float64],
        action: int,
        maze_rows: int,
        maze_cols: int,
        neighbor_table: NDArray[np.int32],
        neighbor_validity: NDArray[np.uint8],
        pellet_positions: NDArray[np.int32],
        ghost_aggressiveness: float,
        ghost_coordination_code: int,
        ghost_strategy_codes: NDArray[np.int32],
        num_ghosts: int,
        num_pellets: int,
        pellet_reward: float,
        idx_pac_row: int,
        idx_pac_col: int,
        idx_ghosts_start: int,
        idx_pellets_start: int,
        idx_pellets_end: int,
        idx_score: int,
        idx_terminal: int,
        patrol_dir_state: NDArray[np.int32],
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def probability(
        self, values: Union[NDArray[np.float64], Sequence[NDArray[np.float64]]]
    ) -> NDArray[np.float64]: ...
    def batch_sample(self, particles: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @property
    def state(self) -> NDArray[np.float64]: ...
    @property
    def action(self) -> int: ...

class PacManObservationCpp:
    """Native observation kernel for PacMan POMDP (pybind11-backed)."""

    def __init__(
        self,
        next_state: NDArray[np.float64],
        action: int,
        num_ghosts: int,
        maze_rows: int,
        maze_cols: int,
        observation_noise_factor: float,
        max_observation_noise: float,
        idx_pac_row: int,
        idx_pac_col: int,
        idx_ghosts_start: int,
        idx_terminal: int,
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def probability(
        self, values: Union[NDArray[np.float64], Sequence[NDArray[np.float64]]]
    ) -> NDArray[np.float64]: ...
    def batch_log_likelihood(
        self,
        next_particles: NDArray[np.float64],
        observation: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...
    @property
    def next_state(self) -> NDArray[np.float64]: ...
    @property
    def action(self) -> int: ...
