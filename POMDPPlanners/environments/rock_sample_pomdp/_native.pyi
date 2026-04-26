"""Type stubs for the native C++ RockSample sampling extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/rock_sample.cpp``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis

from typing import List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

def set_seed(seed: int) -> None:
    """Seed the module-level RNG used by ``sample()``.

    RockSample transitions are deterministic; only the observation sampler
    consumes RNG (for the Bernoulli correct/flipped sensor flip).
    """
    ...

def simulate_rollout_discrete(
    initial_state: NDArray[np.floating],
    action_indices: NDArray[np.int32],
    rock_positions_flat: NDArray[np.int32],
    max_depth: int,
    start_depth: int,
    discount_factor: float,
    map_rows: int,
    map_cols: int,
    n_actions: int,
    step_penalty: float,
    exit_reward: float,
    good_rock_reward: float,
    bad_rock_penalty: float,
    sensor_use_penalty: float,
) -> float:
    """Native random rollout for RockSamplePOMDP (no dangerous-area term).

    Walks the deterministic transition model and computes rewards in C++.
    Returns the discounted sum of rewards. ``action_indices`` must be a
    pre-drawn integer array of length ``max_depth - start_depth``.
    ``rock_positions_flat`` is a 1-D int32 array ``[row0, col0, row1, col1, …]``.
    """
    ...

class RockSampleTransitionCpp:
    """Native state transition sampler (deterministic RockSample dynamics)."""

    state: Tuple[float, ...]
    action: int

    def __init__(
        self,
        state: Union[Sequence[float], NDArray[np.floating]],
        action: int,
        map_rows: int,
        map_cols: int,
        num_rocks: int,
        rock_positions: NDArray[np.int32],
        sensor_efficiency: float,
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def probability(
        self,
        values: Union[Sequence[NDArray[np.floating]], NDArray[np.floating]],
    ) -> NDArray[np.float64]: ...
    def batch_sample(self, particles: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def set_state(self, state: Union[Sequence[float], NDArray[np.floating]]) -> None: ...

class RockSampleObservationCpp:
    """Native categorical observation sampler for RockSample.

    Observation codes: ``0=none``, ``1=good``, ``2=bad``. Movement / sample
    actions deterministically produce ``none``; check actions produce a
    noisy Bernoulli flip whose probability depends on Euclidean distance to
    the queried rock via ``exp(-distance / sensor_efficiency)``.
    """

    next_state: Tuple[float, ...]
    action: int

    def __init__(
        self,
        next_state: Union[Sequence[float], NDArray[np.floating]],
        action: int,
        map_rows: int,
        map_cols: int,
        num_rocks: int,
        rock_positions: NDArray[np.int32],
        sensor_efficiency: float,
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[int]: ...
    def probability(self, values: NDArray[np.int32]) -> NDArray[np.float64]: ...
    def batch_log_likelihood(
        self,
        next_particles: NDArray[np.floating],
        observation: int,
    ) -> NDArray[np.float64]: ...
    def set_next_state(self, next_state: Union[Sequence[float], NDArray[np.floating]]) -> None: ...
