"""Type stubs for the native C++ CartPole sampling extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/cartpole.cpp``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis

from typing import List, Sequence, Union

import numpy as np
from numpy.typing import NDArray

def set_seed(seed: int) -> None:
    """Seed the module-level RNG used by ``sample()`` calls."""
    ...

class CartPoleTransitionCpp:
    """Native physics + Gaussian-noise transition sampler."""

    state: NDArray[np.float64]
    action: int
    force_mag: float
    total_mass: float
    polemass_length: float
    gravity: float
    length: float
    kinematics_integrator: str
    tau: float
    masspole: float

    def __init__(
        self,
        state: Union[Sequence[float], NDArray[np.floating]],
        action: int,
        force_mag: float,
        total_mass: float,
        polemass_length: float,
        gravity: float,
        length: float,
        kinematics_integrator: str,
        tau: float,
        masspole: float,
        covariance: NDArray[np.floating],
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def probability(
        self,
        values: Union[Sequence[NDArray[np.floating]], NDArray[np.floating]],
    ) -> NDArray[np.float64]: ...
    def batch_sample(self, particles: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def _compute_deterministic_next_state(self) -> NDArray[np.float64]: ...

class CartPoleObservationCpp:
    """Native Gaussian-noise observation sampler."""

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
