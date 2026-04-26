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

def reward_batch(
    states: NDArray[np.floating],
    action: NDArray[np.floating],
    tag_radius: float,
    tag_reward: float,
    tag_penalty: float,
    step_cost: float,
    dangerous_areas: NDArray[np.floating],
    dangerous_area_radius: float,
    dangerous_area_penalty: float,
) -> NDArray[np.float64]:
    """Vectorised reward kernel; see ContinuousLaserTagPOMDP.reward_batch."""
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
    def set_state(
        self,
        state: Union[Sequence[float], NDArray[np.floating]],
    ) -> None: ...

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
    def set_next_state(
        self,
        next_state: Union[Sequence[float], NDArray[np.floating]],
    ) -> None: ...

# ── Discrete LaserTag native rollout ─────────────────────────────────────────

def simulate_rollout_discrete(
    initial_state: NDArray[np.floating],
    max_depth: int,
    discount: float,
    initial_depth: int,
    rows: int,
    cols: int,
    walls_flat: NDArray[np.integer],
    dangerous_areas: NDArray[np.floating],
    dangerous_area_radius: float,
    dangerous_area_penalty: float,
    tag_reward: float,
    tag_penalty: float,
    step_cost: float,
    transition_error_prob: float,
) -> float:
    """Run a full random-action rollout for the discrete LaserTagPOMDP in one C++ frame.

    Actions are drawn uniformly from {0,1,2,3,4} using the module-level mt19937_64 RNG.
    Seed via set_seed() before calling for reproducible results.
    Returns the discounted sum of immediate rewards along the sampled trajectory.
    """
    ...

# ── ContinuousLaserTag native rollout (added by perf agent) ──────────────────

def cont_simulate_rollout(
    initial_state: NDArray[np.floating],
    actions_buffer: NDArray[np.floating],
    start_depth: int,
    max_depth: int,
    discount_factor: float,
    robot_covariance: NDArray[np.floating],
    opponent_covariance: NDArray[np.floating],
    pursuit_speed: float,
    walls: NDArray[np.floating],
    grid_size: NDArray[np.floating],
    robot_radius: float,
    opponent_radius: float,
    tag_radius: float,
    tag_reward: float,
    tag_penalty: float,
    step_cost: float,
    dangerous_areas: NDArray[np.floating],
    dangerous_area_radius: float,
    dangerous_area_penalty: float,
) -> float:
    """Run a full random rollout for ContinuousLaserTagPOMDP in one C++ frame.

    ``actions_buffer`` must be shape (N, 3) float64 with N >= max_depth - start_depth.
    Returns the discounted sum of immediate rewards along the sampled trajectory.
    """
    ...
