"""Type stubs for the native C++ CaptureTheFlag extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/capture_the_flag.cpp``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis

from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

MAX_PLAYERS_PER_TEAM: int

def set_seed(seed: int) -> None:
    """Seed the module-level RNG every sampling method draws from."""
    ...

class CaptureTheFlagModelCpp:
    """The CaptureTheFlag model for one configuration.

    Holds the field, the teams and every coefficient; each method takes the
    state and the joint action id. Probabilities, likelihoods and rewards
    match the Python reference in ``CaptureTheFlagPOMDP`` bit for bit.
    Likelihood methods return probabilities, not logs: the caller applies
    ``np.log`` as the reference does.
    """

    state_size: int
    observation_size: int
    n_actions: int

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        width: int,
        height: int,
        midline: int,
        trees: Sequence[Tuple[int, int]],
        n_blue: int,
        n_red: int,
        n_red_defenders: int,
        blue_base: Tuple[int, int],
        red_base: Tuple[int, int],
        blue_flag_cell: Tuple[int, int],
        red_flag_candidates: Sequence[Tuple[int, int]],
        slip_probability: float,
        range_error_probability: float,
        red_pursuit_probability: float,
        red_alert_radius: int,
        freeze_steps: int,
        tagger_cooldown_steps: int,
        detector_half_distance_move: float,
        detector_half_distance_scan: float,
        score_to_win: int,
        capture_reward: float,
        concede_penalty: float,
        tagged_penalty: float,
        tag_reward: float,
        pickup_reward: float,
        move_cost: float,
        scan_cost: float,
    ) -> None: ...
    def is_terminal(self, state: NDArray[np.floating]) -> bool: ...
    def sample_next_state(
        self, state: NDArray[np.floating], action: int, n_samples: int = 1
    ) -> Union[NDArray[np.float64], List[NDArray[np.float64]]]: ...
    def batch_sample(self, states: NDArray[np.floating], action: int) -> NDArray[np.float64]: ...
    def transition_probability(
        self, state: NDArray[np.floating], action: int, next_states: Any
    ) -> NDArray[np.float64]: ...
    def successor_distribution(
        self, state: NDArray[np.floating], action: int
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def sample_observation(
        self, next_state: NDArray[np.floating], action: int, n_samples: int = 1
    ) -> Union[Tuple[float, ...], List[Tuple[float, ...]]]: ...
    def observation_probability(
        self, next_state: NDArray[np.floating], action: int, observations: Any
    ) -> NDArray[np.float64]: ...
    def batch_observation_probability(
        self, next_states: NDArray[np.floating], action: int, observation: Any
    ) -> NDArray[np.float64]: ...
    def reward(
        self, state: NDArray[np.floating], action: int, next_state: NDArray[np.floating]
    ) -> float: ...
    def reward_batch(
        self, states: NDArray[np.floating], action: int, next_states: NDArray[np.floating]
    ) -> NDArray[np.float64]: ...
    def sample_next_step(
        self, state: NDArray[np.floating], action: int
    ) -> Tuple[NDArray[np.float64], Tuple[float, ...], float]: ...
    def simulate_rollout(
        self,
        state: NDArray[np.floating],
        action_indices: NDArray[np.integer],
        discount_factor: float,
    ) -> float: ...
