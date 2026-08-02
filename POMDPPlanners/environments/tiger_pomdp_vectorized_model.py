# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for the Tiger POMDP.

This module provides :class:`TigerVectorizedModel`, a fully batched,
GPU-friendly implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.tiger_pomdp.TigerPOMDP`.

The Tiger problem is fully discrete: two states (``tiger_left``,
``tiger_right``), three actions (``listen``, ``open_left``, ``open_right``),
and three observations (``hear_left``, ``hear_right``, ``hear_nothing``). The
model re-expresses the environment's transition, observation, and reward
kernels as small torch lookup tables so a vectorized planner (VOPP) can run
tens of thousands of parallel simulations on the GPU without a host/device
sync.

Every table is populated by querying a live environment instance in
``__init__`` (``transition_log_probability``, ``observation_log_probability``,
``reward``), so the environment stays the single source of truth for the
transition/observation/reward semantics; only the batched lookups are
re-expressed in torch. The accompanying parity test pins these kernels to the
environment's native scalar implementations.

States and observations are represented as integer-coded ``[N, 1]`` tensors
whose single column holds the index into ``STATES`` / ``OBSERVATIONS``.
Actions are integer indices into the fixed three-element action set. Only the
standard Tiger label sets are supported; a reconfigured subclass with
different labels raises :class:`NotImplementedError`.
"""

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.tiger_pomdp import (
    ACTIONS,
    OBSERVATIONS,
    STATES,
    TigerPOMDP,
)


class TigerVectorizedModel:
    """Fully vectorized torch generative model for the Tiger POMDP.

    The model batches the transition, observation, reward, terminal, and
    observation-likelihood kernels over a leading particle dimension ``N`` and
    keeps every tensor on a single device. States and observations are
    integer-coded ``[N, 1]`` tensors; actions are integer indices into the
    fixed three-element action set (``listen``, ``open_left``, ``open_right``).
    All kernels reduce to gathers into small probability / reward tables read
    from the environment at construction.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for reward and log-probability tensors.
        num_actions: Number of actions in the fixed action set (three).
        num_states: Number of discrete states (two).
        num_observations: Number of discrete observations (three).

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.environments.tiger_pomdp_vectorized_model import (
        ...     TigerVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> model = TigerVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.tensor([[0], [1], [0]])
        >>> actions = torch.tensor([0, 1, 2])  # listen, open_left, open_right
        >>> next_states = model.sample_next_states(states, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(observations.shape), tuple(rewards.shape)
        ((3, 1), (3, 1), (3,))
    """

    def __init__(
        self,
        env: TigerPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The environment whose transition, observation, and reward
                kernels are mirrored into torch lookup tables.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for reward and log-probability tensors.

        Raises:
            NotImplementedError: If ``env`` uses non-standard state, action, or
                observation label sets that this model does not represent.
        """
        self._require_supported_labels(env)
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self.num_states = len(STATES)
        self.num_actions = len(ACTIONS)
        self.num_observations = len(OBSERVATIONS)
        self._transition_probs = self._build_transition_probs(env)
        self._observation_log_probs, self._observation_probs = self._build_observation_tables(env)
        self._reward_table = self._build_reward_table(env)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_supported_labels(env: TigerPOMDP) -> None:
        if (
            list(env.states) != STATES
            or list(env.actions) != ACTIONS
            or list(env.observations) != OBSERVATIONS
        ):
            raise NotImplementedError(
                "vectorized model supports only the standard Tiger label sets "
                "(states/actions/observations)"
            )

    def _build_transition_probs(self, env: TigerPOMDP) -> Tensor:
        table = np.zeros((self.num_actions, self.num_states, self.num_states))
        for a_idx, action in enumerate(ACTIONS):
            for s_idx, state in enumerate(STATES):
                table[a_idx, s_idx] = np.exp(env.transition_log_probability(state, action, STATES))
        return self._to_float_tensor(table)

    def _build_observation_tables(self, env: TigerPOMDP) -> tuple[Tensor, Tensor]:
        log_table = np.zeros((self.num_actions, self.num_states, self.num_observations))
        for a_idx, action in enumerate(ACTIONS):
            for ns_idx, next_state in enumerate(STATES):
                log_table[a_idx, ns_idx] = env.observation_log_probability(
                    next_state, action, OBSERVATIONS
                )
        return self._to_float_tensor(log_table), self._to_float_tensor(np.exp(log_table))

    def _build_reward_table(self, env: TigerPOMDP) -> Tensor:
        table = np.zeros((self.num_actions, self.num_states))
        for a_idx, action in enumerate(ACTIONS):
            for s_idx, state in enumerate(STATES):
                table[a_idx, s_idx] = env.reward(state, action)
        return self._to_float_tensor(table)

    def _to_float_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        probs = self._transition_probs[actions.to(torch.int64), self._indices(states)]
        return torch.multinomial(probs, num_samples=1)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        probs = self._observation_probs[actions.to(torch.int64), self._indices(next_states)]
        return torch.multinomial(probs, num_samples=1)

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del next_states  # Tiger rewards depend only on the current state/action.
        return self._reward_table[actions.to(torch.int64), self._indices(states)]

    def terminal_mask(self, states: Tensor) -> Tensor:
        # TigerPOMDP.is_terminal is always False (doors reset via the transition).
        return torch.zeros(states.shape[0], dtype=torch.bool, device=self.device)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        return self._observation_log_probs[
            actions.to(torch.int64), self._indices(next_states), self._indices(observations)
        ]

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        return self._indices(observations)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _indices(coded: Tensor) -> Tensor:
        return coded.reshape(coded.shape[0], -1)[:, 0].to(torch.int64)
