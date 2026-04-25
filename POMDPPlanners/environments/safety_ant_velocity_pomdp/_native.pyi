"""Type stubs for the native C++ Safety Ant Velocity sampling extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/safety_ant_velocity.cpp``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis

from typing import List, Sequence, Union

import numpy as np
from numpy.typing import NDArray

def set_seed(seed: int) -> None:
    """Seed the module-level RNG used by ``sample()`` / ``batch_sample()`` calls."""
    ...

class SafeAntVelocityTransitionCpp:
    """Native damped-force + uniform-angle transition sampler."""

    state: NDArray[np.float64]
    action: int
    dt: float
    mass: float
    damping: float
    max_force: float
    force_scales: NDArray[np.float64]

    def __init__(
        self,
        state: Union[Sequence[float], NDArray[np.floating]],
        action: int,
        dt: float,
        mass: float,
        damping: float,
        max_force: float,
        force_scales: NDArray[np.floating],
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def batch_sample(self, particles: NDArray[np.floating]) -> NDArray[np.float64]: ...

class SafeAntVelocityObservationCpp:
    """Native identity-mean diagonal-Gaussian observation sampler."""

    next_state: NDArray[np.float64]
    action: int
    mean: NDArray[np.float64]

    def __init__(
        self,
        next_state: Union[Sequence[float], NDArray[np.floating]],
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
