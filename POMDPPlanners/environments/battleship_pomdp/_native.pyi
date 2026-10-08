"""Type stubs for the native C++ Battleship extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/battleship.cpp``. Nothing in the module draws randomness, so it has
no ``set_seed``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis

from typing import Any, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

_StateLike = Union[Sequence[float], NDArray[Any]]

def sample_next_state(state: _StateLike, action: int, num_cells: int) -> NDArray[np.float64]:
    """Copy of ``state`` with the probe flag of ``action`` set."""
    ...

def sample_next_state_batch(
    states: NDArray[Any], action: int, num_cells: int
) -> NDArray[np.float64]:
    """Apply one probe to every row of an ``(N, D)`` array."""
    ...

def transition_log_probability(
    state: _StateLike, action: int, num_cells: int, next_states: Any
) -> NDArray[np.float64]:
    """``0.0`` for the realised successor, ``-inf`` for any other candidate."""
    ...

def sample_observation(next_state: _StateLike, action: int, num_cells: int) -> int:
    """``1`` (hit) if the probed cell is occupied, else ``0`` (miss)."""
    ...

def observation_log_probability(
    next_state: _StateLike, action: int, num_cells: int, observations: Any
) -> NDArray[np.float64]:
    """``0.0`` for the observation the sensor emits, ``-inf`` for any other."""
    ...

def reward(
    state: _StateLike, action: int, num_cells: int, hit_reward: float, miss_penalty: float
) -> float:
    """Reward of one probe."""
    ...

def reward_batch(
    states: NDArray[Any], action: int, num_cells: int, hit_reward: float, miss_penalty: float
) -> NDArray[np.float64]:
    """Reward of one probe for every row of an ``(N, D)`` array."""
    ...

def is_terminal(state: _StateLike, num_cells: int) -> bool:
    """True when every occupied cell has been probed."""
    ...

def sample_next_step(
    state: _StateLike, action: int, num_cells: int, hit_reward: float, miss_penalty: float
) -> Tuple[NDArray[np.float64], int, float]:
    """``(next_state, observation, reward)`` in one call."""
    ...

def simulate_rollout_discrete(
    initial_state: _StateLike,
    action_indices: NDArray[np.int32],
    max_depth: int,
    start_depth: int,
    discount_factor: float,
    num_cells: int,
    hit_reward: float,
    miss_penalty: float,
) -> float:
    """Discounted return of a rollout with pre-drawn actions.

    Stops at a terminal state or at ``max_depth``.
    """
    ...
