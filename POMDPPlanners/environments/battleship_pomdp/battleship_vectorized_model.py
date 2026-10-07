# SPDX-License-Identifier: MIT

"""Torch vectorized generative model for Battleship, for VOPP.

This module provides :class:`BattleshipVectorizedModel`, an implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.battleship_pomdp.battleship_pomdp.BattleshipPOMDP`.
VOPP needs one to search on a batch of particles in a single torch call; the
scalar environment steps one state at a time.

The state row is the environment's own layout,
``[occupancy_0 .. occupancy_{C-1} | probed_0 .. probed_{C-1}]`` with
``C = board_size ** 2``. Action ``a`` is the flat index of the probed cell, so
``num_actions == C``. The observation is a width-1 row holding
:data:`~POMDPPlanners.environments.battleship_pomdp.battleship_pomdp.HIT` (``1``)
or :data:`~POMDPPlanners.environments.battleship_pomdp.battleship_pomdp.MISS`
(``0``).

Every kernel reads only ``board_size``, ``hit_reward`` and ``miss_penalty``
from the environment. The fleet (``ship_lengths``, ``allow_adjacent_ships``)
enters only through the initial-state distribution, which the belief supplies
as particles, so every environment configuration is supported and nothing is
declined.

Like the scalar environment, a terminal state is not special-cased: stepping
from one still sets the probe flag and scores the probe. VOPP discounts
finished rows through :meth:`BattleshipVectorizedModel.terminal_mask`.
"""

from typing import Optional

import torch
from torch import Tensor

from POMDPPlanners.environments.battleship_pomdp.battleship_pomdp import (
    HIT,
    MISS,
    BattleshipPOMDP,
)

# The scalar observation_log_probability returns -inf for an impossible
# reading. A finite floor keeps particle weights free of NaN after
# normalisation; it is the floor the other vectorized models use.
_LOG_PROB_FLOOR = -690.7755278982137  # == log(1e-300)


class BattleshipVectorizedModel:
    """Batched torch model of the Battleship transition, sensor and reward.

    All five kernels are deterministic: a probe sets one flag, the sensor
    reports that cell's occupancy without noise, and the reward is decided by
    the pre-probe state. ``sample_next_states`` and ``sample_observations``
    therefore draw no random numbers.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype of state, observation and reward tensors.
        num_actions: Number of probe actions, ``board_size ** 2``.
        num_cells: Number of board cells, ``board_size ** 2``.

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.battleship_pomdp.battleship_pomdp import (
        ...     BattleshipPOMDP,
        ...     create_battleship_state,
        ... )
        >>> from POMDPPlanners.environments.battleship_pomdp.battleship_vectorized_model import (
        ...     BattleshipVectorizedModel,
        ... )
        >>> env = BattleshipPOMDP(board_size=3, ship_lengths=(2,))
        >>> model = BattleshipVectorizedModel(env, device=torch.device("cpu"))
        >>> state = create_battleship_state([1, 1, 0, 0, 0, 0, 0, 0, 0])
        >>> states = torch.as_tensor(state, dtype=torch.float32).repeat(2, 1)
        >>> actions = torch.tensor([0, 4])
        >>> next_states = model.sample_next_states(states, actions)
        >>> model.sample_observations(next_states, actions)[:, 0].tolist()
        [1.0, 0.0]
        >>> [round(r, 3) for r in model.rewards(states, actions, next_states).tolist()]
        [1.0, -0.1]
    """

    def __init__(
        self,
        env: BattleshipPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Read the board size and the reward constants from ``env``.

        Args:
            env: The environment whose kernels are mirrored.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for state, observation and reward tensors.
        """
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self.num_cells = int(env.num_cells)
        self.num_actions = self.num_cells
        self._hit_reward = float(env.hit_reward)
        self._miss_reward = -float(env.miss_penalty)

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        """Set the probe flag of each row's action cell; nothing else changes."""
        flag_columns = (actions.to(torch.int64) + self.num_cells).unsqueeze(1)
        return states.clone().scatter_(1, flag_columns, 1.0)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        """Report ``HIT`` where the probed cell is occupied, else ``MISS``."""
        occupied = self._probed_cell_occupied(next_states, actions)
        observation = torch.where(occupied, float(HIT), float(MISS)).to(self.dtype)
        return observation.unsqueeze(1)

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        """Score ``hit_reward`` for an occupied, unprobed cell, else ``-miss_penalty``.

        ``next_states`` is unused: as in the scalar ``reward``, whether a probe
        is a new hit is decided by the state it is taken from.
        """
        del next_states
        cells = actions.to(torch.int64).unsqueeze(1)
        occupied = states.gather(1, cells).squeeze(1) > 0.5
        probed = states.gather(1, cells + self.num_cells).squeeze(1) > 0.5
        new_hit = occupied & ~probed
        hit = torch.full_like(new_hit, self._hit_reward, dtype=self.dtype)
        miss = torch.full_like(new_hit, self._miss_reward, dtype=self.dtype)
        return torch.where(new_hit, hit, miss)

    def terminal_mask(self, states: Tensor) -> Tensor:
        """Flag rows in which every occupied cell has been probed."""
        occupied = states[:, : self.num_cells] > 0.5
        unprobed = states[:, self.num_cells :] <= 0.5
        return ~torch.any(occupied & unprobed, dim=1)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        """Return ``0`` for the reading the sensor emits, else the floor ``log(1e-300)``."""
        occupied = self._probed_cell_occupied(next_states, actions)
        truth = torch.where(occupied, float(HIT), float(MISS)).to(observations.dtype)
        matches = observations[:, 0] == truth
        zero = torch.zeros(matches.shape[0], dtype=self.dtype, device=self.device)
        return torch.where(matches, zero, torch.full_like(zero, _LOG_PROB_FLOOR))

    def action_keys(self, actions: Tensor) -> Tensor:
        """Use the action index as its tree key."""
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        """Use the ``HIT``/``MISS`` code as the observation's tree key."""
        return observations[:, 0].round().to(torch.int64)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _probed_cell_occupied(self, states: Tensor, actions: Tensor) -> Tensor:
        cells = actions.to(torch.int64).unsqueeze(1)
        return states.gather(1, cells).squeeze(1) > 0.5
