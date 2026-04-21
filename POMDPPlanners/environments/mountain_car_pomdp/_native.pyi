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
