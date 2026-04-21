"""Type stubs for the native C++ Continuous LaserTag sampling extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/continuous_laser_tag.cpp``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis

from typing import List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

def set_seed(seed: int) -> None:
    """Seed the module-level RNG used by ``sample()`` / batch entry points."""
    ...

class ContinuousLaserTagTransitionCpp:
    """Native state transition sampler (robot + opponent Gaussian steps)."""

    state: Tuple[float, float, float, float, float]
    action: Tuple[float, float, float]

    def __init__(
        self,
        state: Union[Sequence[float], NDArray[np.floating]],
        action: Union[Sequence[float], NDArray[np.floating]],
        robot_covariance: NDArray[np.floating],
        opponent_covariance: NDArray[np.floating],
        pursuit_speed: float,
        walls: NDArray[np.floating],
        grid_size: NDArray[np.floating],
        robot_radius: float,
        opponent_radius: float,
        tag_radius: float,
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def probability(
        self,
        values: Union[Sequence[NDArray[np.floating]], NDArray[np.floating]],
    ) -> NDArray[np.float64]: ...
    def batch_sample(self, particles: NDArray[np.floating]) -> NDArray[np.float64]: ...

class ContinuousLaserTagObservationCpp:
    """Native 8-direction laser observation sampler."""

    next_state: Tuple[float, float, float, float, float]
    action: Tuple[float, float, float]
    mean: NDArray[np.float64]

    def __init__(
        self,
        next_state: Union[Sequence[float], NDArray[np.floating]],
        action: Union[Sequence[float], NDArray[np.floating]],
        measurement_noise: float,
        walls: NDArray[np.floating],
        grid_size: NDArray[np.floating],
        opponent_radius: float,
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
