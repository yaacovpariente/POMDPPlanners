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
    dangerous_areas: NDArray[np.float64],
    dangerous_area_radius: float,
    dangerous_area_penalty: float,
    reward_variant_code: int,
    penalty_decay: float,
    is_dangerous_area_hit_terminal: bool = False,
) -> float:
    """Run a random rollout from state using pre-drawn action_indices.

    Returns the discounted cumulative reward accumulated until terminal or
    max_depth is reached. action_indices must have length >= (max_depth - depth).
    Reward per step includes the variant-aware dangerous-area contribution.

    When ``is_dangerous_area_hit_terminal`` is set the dangerous-area hazard is
    draw-coupled: the transition sets the (absorbing) terminal slot on a hazard
    hit and the per-step penalty becomes deterministic given that slot.
    """

def reward_batch(
    states: NDArray[np.float64],
    action: int,
    next_states: NDArray[np.float64],
    reward_variant_code: int,
    penalty_decay: float,
    dangerous_areas: NDArray[np.float64],
    dangerous_area_radius: float,
    dangerous_area_penalty: float,
    num_ghosts: int,
    step_penalty: float,
    ghost_collision_penalty: float,
    pellet_reward: float,
    win_reward: float,
    idx_pac_row: int,
    idx_pac_col: int,
    idx_ghosts_start: int,
    idx_pellets_start: int,
    idx_pellets_end: int,
    idx_score: int,
    idx_terminal: int,
    is_dangerous_area_hit_terminal: bool = False,
) -> NDArray[np.float64]:
    """Variant-aware standalone batch reward kernel. Returns (N,) float64.

    When ``is_dangerous_area_hit_terminal`` is set the dangerous-area penalty is
    deterministic: ``-dangerous_area_penalty`` is applied iff the row's realised
    ``next_states`` terminal slot is set and PacMan is in a zone (the decayed
    variant has no radius cutoff, so it is always in-zone).
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
        dangerous_areas: NDArray[np.float64] = ...,
        dangerous_area_radius: float = 0.0,
        reward_variant_code: int = 0,
        penalty_decay: float = 1.0,
        is_dangerous_area_hit_terminal: bool = False,
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def probability(
        self, values: Union[NDArray[np.float64], Sequence[NDArray[np.float64]]]
    ) -> NDArray[np.float64]: ...
    def batch_sample(self, particles: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def set_state(self, state: NDArray[np.float64]) -> None: ...
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
    def set_next_state(self, next_state: NDArray[np.float64]) -> None: ...
    @property
    def next_state(self) -> NDArray[np.float64]: ...
    @property
    def action(self) -> int: ...
