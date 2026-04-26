"""Type stubs for the native C++ MountainCar sampling extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/mountain_car.cpp``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis

from typing import List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

def set_seed(seed: int) -> None:
    """Seed the module-level RNG used by ``sample()`` calls."""
    ...

def simulate_rollout(
    initial_state: NDArray[np.float64],
    actions: NDArray[np.int32],
    action_indices: NDArray[np.int32],
    max_depth: int,
    start_depth: int,
    discount_factor: float,
    power: float,
    gravity: float,
    max_speed: float,
    min_position: float,
    max_position: float,
    goal_position: float,
    covariance: NDArray[np.float64],
) -> float:
    """Native random rollout for MountainCar.

    Returns the discounted return from ``initial_state``.
    ``actions`` must be a 1-D int32 array of action values (e.g. [-1, 0, 1]).
    ``action_indices`` must be a pre-drawn 1-D int32 array of indices into ``actions``.
    """
    ...

class MountainCarTransitionCpp:
    """Native physics + Gaussian-noise transition sampler."""

    state: Tuple[float, float]
    action: int
    power: float
    gravity: float
    max_speed: float
    min_position: float
    max_position: float

    def __init__(
        self,
        state: Union[Tuple[float, float], Sequence[float], NDArray[np.floating]],
        action: int,
        power: float,
        gravity: float,
        max_speed: float,
        min_position: float,
        max_position: float,
        covariance: NDArray[np.floating],
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def probability(
        self,
        values: Union[Sequence[NDArray[np.floating]], NDArray[np.floating]],
    ) -> NDArray[np.float64]: ...
    def batch_sample(self, particles: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def set_state(self, state: Union[Sequence[float], NDArray[np.floating]]) -> None: ...
    def _compute_deterministic_next_state(self) -> NDArray[np.float64]: ...

class MountainCarObservationCpp:
    """Native Gaussian-noise observation sampler."""

    next_state: Tuple[float, float]
    action: int
    mean: NDArray[np.float64]

    def __init__(
        self,
        next_state: Union[Tuple[float, float], Sequence[float], NDArray[np.floating]],
        action: int,
        covariance: NDArray[np.floating],
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def probability(
        self,
        values: Union[Sequence[NDArray[np.floating]], NDArray[np.floating]],
    ) -> NDArray[np.float64]: ...
    def batch_log_likelihood(
        self,
        next_particles: NDArray[np.floating],
        observation: NDArray[np.floating],
    ) -> NDArray[np.float64]: ...
    def set_next_state(self, next_state: Union[Sequence[float], NDArray[np.floating]]) -> None: ...
