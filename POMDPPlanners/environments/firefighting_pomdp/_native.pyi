"""Type stubs for the native C++ firefighting extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/firefighting.cpp``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis,too-many-arguments

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

def set_seed(seed: int) -> None:
    """Seed the module-level RNG every sampling method of this module draws from."""
    ...

class FirefightingModelCpp:
    """One firefighting world: grid, obstacles, depot, probabilities and costs.

    Each method mirrors a ``FirefightingPOMDP`` method. The action is a call
    argument, the joint action integer in ``[0, 5 ** num_firefighters)``; an
    action outside that range raises ``ValueError``.
    """

    state_size: int
    observation_size: int
    num_actions: int

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        num_firefighters: int,
        obstacle_mask: NDArray[np.uint8],
        depot_row: int,
        depot_col: int,
        max_tank: int,
        sensing_radius: int,
        observation_error_probability: float,
        slip_probability: float,
        spread_probability: float,
        wind_gain_low: float,
        wind_gain_high: float,
        crosswind_attenuation: float,
        growth_probability: float,
        burnout_probability: float,
        suppression_probability_unburnt: float,
        suppression_probability_smoldering: float,
        suppression_probability_burning: float,
        max_steps: int,
        success_reward: float,
        step_cost: float,
        smoldering_cell_cost: float,
        burning_cell_cost: float,
        burnt_cell_cost: float,
        damage_cost: float,
        water_cost: float,
        is_all_firefighters_disabled_terminal: bool,
    ) -> None: ...
    def sample_next_state(self, state: NDArray[np.float64], action: int) -> NDArray[np.float64]: ...
    def sample_next_states(
        self, state: NDArray[np.float64], action: int, n_samples: int
    ) -> NDArray[np.float64]: ...
    def sample_next_state_batch(
        self, states: NDArray[np.float64], action: int
    ) -> NDArray[np.float64]: ...
    def transition_log_probability(
        self, state: NDArray[np.float64], action: int, next_states: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
    def sample_observation(self, next_state: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def sample_observations(
        self, next_state: NDArray[np.float64], n_samples: int
    ) -> NDArray[np.float64]: ...
    def observation_log_probability(
        self, next_state: NDArray[np.float64], observations: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
    def reward(
        self, state: NDArray[np.float64], action: int, next_state: NDArray[np.float64]
    ) -> float: ...
    def reward_batch(
        self, states: NDArray[np.float64], action: int, next_states: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
    def is_terminal(self, state: NDArray[np.float64]) -> bool: ...
    def sample_next_step(
        self, state: NDArray[np.float64], action: int
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]: ...
    def simulate_rollout(
        self,
        initial_state: NDArray[np.float64],
        actions: NDArray[np.int64],
        max_depth: int,
        start_depth: int,
        discount_factor: float,
    ) -> float:
        """Discounted return of a rollout under the pre-drawn joint ``actions``.

        Stops at ``max_depth`` or at the first terminal state; ``actions`` must
        hold at least ``max_depth - start_depth`` entries.
        """
        ...
