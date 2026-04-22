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
