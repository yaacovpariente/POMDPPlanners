"""Type stubs for the native C++ Continuous Light-Dark sampling extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/continuous_light_dark.cpp``.

Only the ``NORMAL_NOISE`` observation model variant is backed by the
native extension; the ``NORMAL_NOISE_NO_OBS_IN_DARK`` and
``DISTANCE_BASED`` variants return string ``"None"`` observations and
continue to execute on the Python path.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis

from typing import List, Sequence, Union

import numpy as np
from numpy.typing import NDArray

def set_seed(seed: int) -> None:
    """Seed the module-level RNG used by ``sample()`` calls."""
    ...

def simulate_rollout(
    initial_state: NDArray[np.float64],
    action_array: NDArray[np.float64],
    action_indices: NDArray[np.int32],
    max_depth: int,
    start_depth: int,
    discount_factor: float,
    goal_state: NDArray[np.float64],
    obstacles: NDArray[np.float64],
    goal_state_radius: float,
    obstacle_radius: float,
    grid_size: float,
    fuel_cost: float,
    goal_reward: float,
    obstacle_reward: float,
    obstacle_hit_probability: float,
    is_obstacle_hit_terminal: bool,
    covariance: NDArray[np.float64],
) -> float:
    """Native random rollout for the STANDARD reward model.

    Returns the discounted return from ``initial_state`` using the STANDARD
    reward model. ``action_indices`` must be a pre-drawn 1-D int array with
    at least ``max_depth - start_depth`` entries. ``obstacles`` must be a flat
    1-D array ``[x0, y0, x1, y1, ...]`` (row-major interleaved).
    """
    ...

class ContinuousLightDarkTransitionCpp:
    """Native additive-Gaussian transition sampler for Continuous Light-Dark."""

    state: NDArray[np.float64]
    action: NDArray[np.float64]

    def __init__(
        self,
        state: Union[Sequence[float], NDArray[np.floating]],
        action: NDArray[np.floating],
        covariance: NDArray[np.floating],
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def probability(
        self,
        values: Union[Sequence[NDArray[np.floating]], NDArray[np.floating]],
    ) -> NDArray[np.float64]: ...
    def batch_sample(self, particles: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def set_state(self, state: Union[Sequence[float], NDArray[np.floating]]) -> None: ...

class ContinuousLightDarkObservationCpp:
    """Native state-dependent Gaussian observation sampler for Continuous Light-Dark.

    Picks between a near-beacon and a far-from-beacon covariance based on
    whether the next_state falls within ``beacon_radius`` of any beacon.
    Samples are clipped to ``[0, grid_size]`` after Gaussian draw.
    """

    next_state: NDArray[np.float64]
    mean: NDArray[np.float64]
    action: NDArray[np.float64]

    def __init__(
        self,
        next_state: Union[Sequence[float], NDArray[np.floating]],
        action: NDArray[np.floating],
        covariance_near: NDArray[np.floating],
        covariance_far: NDArray[np.floating],
        beacons: NDArray[np.floating],
        beacon_radius: float,
        grid_size: float,
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
