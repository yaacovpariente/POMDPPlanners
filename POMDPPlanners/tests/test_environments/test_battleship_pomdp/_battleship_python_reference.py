# SPDX-License-Identifier: MIT

"""Pure-Python reference for the Battleship kernels that moved to C++.

These are the bodies ``BattleshipPOMDP`` used before its hot path was ported to
``battleship_pomdp/_cpp/battleship.cpp``, copied verbatim apart from taking
``num_cells`` and the reward parameters as arguments instead of reading them
off ``self``. The native-equivalence tests compare the environment against
them, so they must not be edited to follow a change in the C++ code.

The module name is underscore-prefixed so pytest does not collect it.
"""

from typing import Any, Callable, Sequence

import numpy as np

HIT = 1
MISS = 0


def sample_next_state(state: Any, action: int, num_cells: int, n_samples: int = 1) -> Any:
    next_state = np.array(state, dtype=np.float64, copy=True)
    next_state[num_cells + int(action)] = 1.0
    if n_samples == 1:
        return next_state
    return np.tile(next_state, (int(n_samples), 1))


def sample_next_state_batch(states: Any, action: int, num_cells: int) -> np.ndarray:
    next_states = np.array(states, dtype=np.float64, copy=True)
    if next_states.ndim == 1:
        next_states = next_states.reshape(1, -1)
    next_states[:, num_cells + int(action)] = 1.0
    return next_states


def transition_log_probability(
    state: Any, action: int, next_states: Any, num_cells: int
) -> np.ndarray:
    expected = sample_next_state(state=state, action=action, num_cells=num_cells)
    candidates = np.asarray(next_states, dtype=np.float64)
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)
    matches = np.all(np.abs(candidates - expected) < 0.5, axis=1)
    return np.where(matches, 0.0, -np.inf)


def _occupancy(state: Any, num_cells: int) -> np.ndarray:
    return np.asarray(state, dtype=np.float64)[:num_cells] > 0.5


def sample_observation(next_state: Any, action: int, num_cells: int, n_samples: int = 1) -> Any:
    observation = HIT if bool(_occupancy(next_state, num_cells)[int(action)]) else MISS
    if n_samples == 1:
        return observation
    return [observation] * int(n_samples)


def observation_log_probability(
    next_state: Any, action: int, observations: Any, num_cells: int
) -> np.ndarray:
    truth = HIT if bool(_occupancy(next_state, num_cells)[int(action)]) else MISS
    candidates = np.asarray(observations).ravel()
    return np.where(candidates == truth, 0.0, -np.inf).astype(np.float64)


def reward(
    state: Any, action: int, num_cells: int, hit_reward: float, miss_penalty: float
) -> float:
    state_arr = np.asarray(state, dtype=np.float64)
    cell = int(action)
    holds_ship = bool(state_arr[cell] > 0.5)
    already_probed = bool(state_arr[num_cells + cell] > 0.5)
    return hit_reward if holds_ship and not already_probed else -miss_penalty


def reward_batch(
    states: Any, action: int, num_cells: int, hit_reward: float, miss_penalty: float
) -> np.ndarray:
    states_arr = np.asarray(states, dtype=np.float64)
    if states_arr.ndim == 1:
        states_arr = states_arr.reshape(1, -1)
    cell = int(action)
    is_new_hit = (states_arr[:, cell] > 0.5) & (states_arr[:, num_cells + cell] <= 0.5)
    return np.where(is_new_hit, hit_reward, -miss_penalty).astype(np.float64)


def is_terminal(state: Any, num_cells: int) -> bool:
    state_arr = np.asarray(state, dtype=np.float64)
    return not bool(np.any((state_arr[:num_cells] > 0.5) & (state_arr[num_cells:] <= 0.5)))


def random_rollout(
    state: Any,
    depth: int,
    next_action: Callable[[], int],
    num_cells: int,
    hit_reward: float,
    miss_penalty: float,
    discount_factor: float,
    max_depth: int,
) -> float:
    """``python_random_rollout`` specialised to the reference kernels above."""
    if depth >= max_depth or is_terminal(state, num_cells):
        return 0.0
    action = next_action()
    next_state = sample_next_state(state, action, num_cells)
    r = reward(state, action, num_cells, hit_reward, miss_penalty)
    return r + discount_factor * random_rollout(
        next_state,
        depth + 1,
        next_action,
        num_cells,
        hit_reward,
        miss_penalty,
        discount_factor,
        max_depth,
    )


def replay(actions: Sequence[int]) -> Callable[[], int]:
    """Return a callable that hands out ``actions`` in order."""
    iterator = iter(int(a) for a in actions)
    return lambda: next(iterator)
