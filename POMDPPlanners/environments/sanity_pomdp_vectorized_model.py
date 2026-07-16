# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for the Sanity POMDP.

This module provides :class:`SanityVectorizedModel`, a fully batched
implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.sanity_pomdp.SanityPOMDP`.

The Sanity POMDP is deterministic and perfectly observable: two states
(``0`` good, ``1`` bad), two actions (``0`` -> good, ``1`` -> bad), the
observation equals the state, the reward is ``1.0`` in the good state and
``0.0`` otherwise, and no state is terminal. Every one of these numeric
constants is read from a live environment instance in ``__init__`` (a
next-state table, a reward table, and a terminal table), so the environment
stays the single source of truth; only the lookups are re-expressed as
batched torch gathers. The accompanying parity test pins these kernels to
the environment's native scalar kernels.

Only the standard two-action ``[0, 1]`` configuration is supported; any other
action set raises :class:`NotImplementedError` at construction, because the
transition and reward tables assume that ``action == next_state`` structure.
"""

from typing import List, Optional

import torch
from torch import Tensor

from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP

_SUPPORTED_ACTIONS: List[int] = [0, 1]
_SUPPORTED_STATES: List[int] = [0, 1]


class SanityVectorizedModel:
    """Fully vectorized torch generative model for the Sanity POMDP.

    The model batches the transition, observation, reward, terminal, and
    observation-likelihood kernels over a leading particle dimension ``N`` and
    keeps every tensor on a single device. States and observations are the
    scalar state index carried in a ``[N, 1]`` tensor; actions are integer
    indices into the environment's two-action set.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of actions in the environment's action set.

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
        >>> from POMDPPlanners.environments.sanity_pomdp_vectorized_model import (
        ...     SanityVectorizedModel,
        ... )
        >>> env = SanityPOMDP(discount_factor=0.95)
        >>> model = SanityVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.tensor([[0.0], [1.0], [0.0]])
        >>> actions = torch.tensor([0, 1, 1])
        >>> next_states = model.sample_next_states(states, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(observations.shape), tuple(rewards.shape)
        ((3, 1), (3, 1), (3,))
        >>> rewards.tolist()
        [1.0, 0.0, 0.0]
    """

    def __init__(
        self,
        env: SanityPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The environment whose transition, reward, and terminal
                kernels are mirrored into lookup tables.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.

        Raises:
            NotImplementedError: If ``env`` exposes an action set other than
                the standard ``[0, 1]`` two-action set.
        """
        self._require_supported_actions(env)
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self.num_actions = len(env.get_actions())
        self._build_tables(env)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_supported_actions(env: SanityPOMDP) -> None:
        if list(env.get_actions()) != _SUPPORTED_ACTIONS:
            raise NotImplementedError(
                "vectorized model supports only the standard two-action "
                f"{_SUPPORTED_ACTIONS} Sanity POMDP action set"
            )

    def _build_tables(self, env: SanityPOMDP) -> None:
        # The transition is state-independent, so a dummy state suffices.
        next_states = [int(env.sample_next_state(0, action)) for action in _SUPPORTED_ACTIONS]
        rewards = [float(env.reward(0, action)) for action in _SUPPORTED_ACTIONS]
        terminals = [bool(env.is_terminal(state)) for state in _SUPPORTED_STATES]
        self._next_state_table = torch.tensor(next_states, dtype=self.dtype, device=self.device)
        self._reward_table = torch.tensor(rewards, dtype=self.dtype, device=self.device)
        self._terminal_table = torch.tensor(terminals, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        del states  # Transition depends only on the action in the Sanity POMDP.
        return self._next_state_table[actions].unsqueeze(-1)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # Perfect observability: the observation equals the state.
        return next_states.clone()

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del states, next_states  # Reward is a function of the action alone.
        return self._reward_table[actions]

    def terminal_mask(self, states: Tensor) -> Tensor:
        return self._terminal_table[states[:, 0].to(torch.int64)]

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # Perfect observability does not depend on the action.
        match = observations[:, 0] == next_states[:, 0]
        return torch.where(
            match,
            torch.zeros(match.shape[0], dtype=self.dtype, device=self.device),
            torch.full((match.shape[0],), -torch.inf, dtype=self.dtype, device=self.device),
        )

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        return observations[:, 0].to(torch.int64)
