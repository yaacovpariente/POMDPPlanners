"""Type stubs for the native C++ Continuous Push sampling extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/continuous_push.cpp``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis

from typing import List, Sequence, Union

import numpy as np
from numpy.typing import NDArray

def set_seed(seed: int) -> None:
    """Seed the module-level RNG used by ``sample()`` calls."""
    ...

def simulate_rollout_discrete(
    state: NDArray[np.float64],
    action_indices: NDArray[np.int64],
    max_depth: int,
    depth: int,
    discount: float,
    grid_size: float,
    push_threshold: float,
    friction_coefficient: float,
    obstacles: NDArray[np.float64],
    obstacle_radius: float,
    obstacle_penalty: float,
    transition_error_prob: float,
) -> float:
    """Run a full discrete Push POMDP rollout in C++.

    Returns the discounted sum of immediate rewards along the sampled
    trajectory, using pre-drawn action indices supplied by the caller.
    """
    ...

def cont_simulate_rollout(
    initial_state: NDArray[np.floating],
    action_array: NDArray[np.floating],
    action_indices: NDArray[np.int32],
    max_depth: int,
    start_depth: int,
    discount_factor: float,
    grid_size: float,
    push_threshold: float,
    friction_coefficient: float,
    max_push: float,
    robot_radius: float,
    obstacle_penalty: float,
    obstacles: NDArray[np.floating],
    covariance: NDArray[np.floating],
) -> float:
    """Native random rollout for ContinuousPushPOMDP.

    Returns the discounted sum of immediate rewards from ``initial_state``.
    ``action_indices`` must be a pre-drawn int32 array of shape
    ``(steps_left,)``.  ``obstacles`` must have shape ``(M, 4)`` with
    rows ``(cx, cy, hx, hy)``.
    """
    ...

class ContinuousPushTransitionCpp:
    """Native 6-D transition sampler for Continuous Push POMDP."""

    state: NDArray[np.float64]
    action: NDArray[np.float64]

    def __init__(
        self,
        state: Union[Sequence[float], NDArray[np.floating]],
        action: Union[Sequence[float], NDArray[np.floating]],
        grid_size: float,
        push_threshold: float,
        friction_coefficient: float,
        max_push: float,
        robot_radius: float,
        obstacles: NDArray[np.floating],
        covariance: NDArray[np.floating],
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def sample_one(self, state: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def probability(
        self,
        values: Union[Sequence[NDArray[np.floating]], NDArray[np.floating]],
    ) -> NDArray[np.float64]: ...
    def batch_sample(self, particles: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def set_state(self, state: Union[Sequence[float], NDArray[np.floating]]) -> None: ...

class ContinuousPushObservationCpp:
    """Native observation sampler for Continuous Push POMDP."""

    next_state: NDArray[np.float64]
    action: NDArray[np.float64]

    def __init__(
        self,
        next_state: Union[Sequence[float], NDArray[np.floating]],
        action: Union[Sequence[float], NDArray[np.floating]],
        observation_noise: float,
        grid_size: float,
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def sample_one(self, next_state: NDArray[np.floating]) -> NDArray[np.float64]: ...
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

def observation_log_probability_step(
    next_state: NDArray[np.floating],
    observations: NDArray[np.floating],
    observation_noise: float,
) -> NDArray[np.float64]:
    """Per-observation log-probability for ContinuousPushPOMDP.

    Lean single-step entry that mirrors
    ContinuousPushObservationCpp.batch_log_likelihood for one fixed
    next_state but skips kernel-cache lookup and set_next_state overhead.
    ``observations`` must be shape (N, 6) float64.
    """
    ...

def belief_batch_transition_discrete(
    particles: NDArray[np.floating],
    action_idx: int,
    transition_error_prob: float,
    obstacles: NDArray[np.floating],
    obstacle_radius: float,
    grid_size: float,
    push_threshold: float,
    friction_coefficient: float,
) -> NDArray[np.float64]:
    """Native batch transition for the discrete Push belief updater.

    Applies ``action_idx`` to all (N, 6) particles in one C++ call.
    When ``transition_error_prob > 0`` an independent C++ RNG decides
    per-particle which action actually executes (matches the Python
    ``PushVectorizedUpdater._batch_transition_with_error`` semantics).
    """
    ...

def belief_batch_obs_log_likelihood_discrete(
    next_particles: NDArray[np.floating],
    observation: NDArray[np.floating],
    observation_noise: float,
) -> NDArray[np.float64]:
    """Native batch observation log-likelihood for the discrete Push updater.

    Returns the per-particle log N(obs[2:4] | particle[2:4], sigma**2 * I_2)
    over all (N, 6) particles. Bit-for-bit equivalent to the Python
    ``PushVectorizedUpdater.batch_observation_log_likelihood`` (no RNG).
    """
    ...

class PushDiscreteTransitionCpp:
    """Native deterministic transition kernel for the discrete Push POMDP.

    One kernel per cached action label: the resolved (dx, dy) for that label
    is frozen at construction; ``set_state`` flips the input state per call,
    ``compute_next_state`` returns the closed-form next state for the cached
    action, and ``compute_next_state_for_action`` evaluates an alternative
    (dx, dy) without rebuilding (used by the error-action branch in
    ``transition_log_probability``).
    """

    def __init__(
        self,
        state: NDArray[np.floating],
        action_dxdy: NDArray[np.floating],
        grid_size: float,
        push_threshold: float,
        friction_coefficient: float,
        obstacles_flat: NDArray[np.floating],
        n_obstacles: int,
        obstacle_radius: float,
    ) -> None: ...
    def set_state(self, state: NDArray[np.floating]) -> None: ...
    def compute_next_state(self) -> NDArray[np.float64]: ...
    def compute_next_state_for_action(
        self, action_dxdy: NDArray[np.floating]
    ) -> NDArray[np.float64]: ...
