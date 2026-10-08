"""Type stubs for the native C++ Chicheck Invaders extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/chicheck_invaders.cpp``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis,too-many-arguments

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

def set_seed(seed: int) -> None:
    """Seed the module-level RNG that every sampler in this module draws from.

    ``np.random.seed`` does not reach this RNG.
    """
    ...

class ChicheckInvadersConfigCpp:
    """The constructor parameters the dynamics, sensors and reward read, copied once."""

    state_size: int
    observation_size: int

    def __init__(
        self,
        num_columns: int,
        num_rows: int,
        num_chickens: int,
        fire_cooldown: int,
        dive_probability: float,
        camera_detection_probability: float,
        radar_detection_probability: float,
        ship_column_noise_std: float,
        camera_offset_noise_std: float,
        radar_range_noise_std: float,
        drop_flag_error_probability: float,
        camera_slope: float,
        radar_radius: float,
        full_observation: bool,
        kill_reward: float,
        shot_cost: float,
        step_cost: float,
        ship_hit_penalty: float,
        clear_reward: float,
        max_steps: int,
    ) -> None: ...

def reward(
    config: ChicheckInvadersConfigCpp,
    state: NDArray[np.floating],
    action: int,
    next_state: Optional[NDArray[np.floating]] = ...,
) -> float:
    """Scalar reward; ``next_state=None`` scores the no-successor branch."""
    ...

def reward_batch(
    config: ChicheckInvadersConfigCpp,
    states: NDArray[np.floating],
    action: int,
    next_states: Optional[NDArray[np.floating]] = ...,
) -> NDArray[np.float64]:
    """Reward of each row of ``states`` under one action, shape ``(N,)``."""
    ...

def is_terminal(config: ChicheckInvadersConfigCpp, state: NDArray[np.floating]) -> bool:
    """Flock cleared, ship hit, or step budget spent."""
    ...

def sample_next_step(
    config: ChicheckInvadersConfigCpp, state: NDArray[np.floating], action: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """``(next_state, observation, reward)`` for one sampled step."""
    ...

def simulate_rollout(
    config: ChicheckInvadersConfigCpp,
    state: NDArray[np.floating],
    action_indices: NDArray[np.int32],
    discount_factor: float,
) -> float:
    """Discounted return along pre-drawn actions, stopping at a terminal state."""
    ...

class ChicheckInvadersTransitionCpp:
    """Transition kernel for one action index in ``[0, 4)``."""

    state: NDArray[np.float64]
    action: int

    def __init__(
        self, state: NDArray[np.floating], action: int, config: ChicheckInvadersConfigCpp
    ) -> None: ...
    def set_state(self, state: NDArray[np.floating]) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def sample_from(self, state: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def log_probability(self, next_states: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def batch_sample(self, particles: NDArray[np.floating]) -> NDArray[np.float64]: ...

class ChicheckInvadersObservationCpp:
    """Observation kernel; the reading does not depend on the action."""

    next_state: NDArray[np.float64]

    def __init__(
        self, next_state: NDArray[np.floating], config: ChicheckInvadersConfigCpp
    ) -> None: ...
    def set_next_state(self, next_state: NDArray[np.floating]) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def sample_from(self, next_state: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def log_probability(self, observations: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def batch_log_likelihood(
        self, next_particles: NDArray[np.floating], observation: NDArray[np.floating]
    ) -> NDArray[np.float64]: ...
