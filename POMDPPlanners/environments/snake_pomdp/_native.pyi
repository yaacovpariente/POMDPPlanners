"""Type stubs for the native C++ Snake extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/snake.cpp``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis

from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

_StateLike = Union[Sequence[float], NDArray[np.floating]]

def set_seed(seed: int) -> None:
    """Seed the module-local RNG.

    Every draw in this module comes from it: the food respawn cell, the
    window detection and the scent quadrant.
    """
    ...

def reward(
    state: _StateLike,
    action: int,
    grid_size: int,
    target_length: int,
    starvation_limit: int,
) -> float:
    """``+1`` for eating, ``-1`` for a wall, self or starvation death, ``0`` otherwise."""
    ...

def reward_batch(
    states: NDArray[np.floating],
    action: int,
    grid_size: int,
    target_length: int,
    starvation_limit: int,
) -> NDArray[np.float64]:
    """:func:`reward` for each row of an ``(N, state_size)`` array."""
    ...

def sample_next_step(
    state: _StateLike,
    action: int,
    grid_size: int,
    target_length: int,
    starvation_limit: int,
    window_radius: int,
    detection_probability: float,
    scent_accuracy: float,
) -> Tuple[NDArray[np.float64], Tuple[int, ...], float]:
    """One ``(next_state, observation, reward)`` draw.

    Makes the draws a transition sample and then an observation sample make,
    in that order.
    """
    ...

def simulate_rollout_discrete(
    initial_state: _StateLike,
    action_indices: NDArray[np.int32],
    max_depth: int,
    start_depth: int,
    discount_factor: float,
    grid_size: int,
    target_length: int,
    starvation_limit: int,
) -> float:
    """Discounted reward sum of a rollout with pre-drawn actions.

    ``action_indices`` holds at least ``max_depth - start_depth`` actions. The
    rollout stops at ``max_depth`` or at the first terminal state.
    """
    ...

class SnakeTransitionCpp:
    """Native transition sampler for one action.

    The body update is deterministic; the food respawns uniformly over the
    free cells after an eat that does not end the episode.
    """

    state: NDArray[np.float64]
    action: int

    def __init__(
        self,
        state: _StateLike,
        action: int,
        grid_size: int,
        target_length: int,
        starvation_limit: int,
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def probability(self, values: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def log_probability(self, values: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def batch_sample(self, particles: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def set_state(self, state: _StateLike) -> None: ...

class SnakeObservationCpp:
    """Native sampler and likelihood for the body, window and scent reading.

    Log-likelihoods of readings the sensor cannot produce are ``-inf``.
    """

    next_state: NDArray[np.float64]
    action: int

    def __init__(
        self,
        next_state: _StateLike,
        action: int,
        grid_size: int,
        target_length: int,
        window_radius: int,
        detection_probability: float,
        scent_accuracy: float,
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[Tuple[int, ...]]: ...
    def probability(self, values: Sequence[Any]) -> NDArray[np.float64]: ...
    def log_probability(self, values: Sequence[Any]) -> NDArray[np.float64]: ...
    def batch_log_likelihood(
        self,
        next_particles: NDArray[np.floating],
        observation: Sequence[int],
    ) -> NDArray[np.float64]: ...
    def set_next_state(self, next_state: _StateLike) -> None: ...
