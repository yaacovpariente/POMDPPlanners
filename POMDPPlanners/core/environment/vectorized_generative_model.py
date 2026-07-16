# SPDX-License-Identifier: MIT

"""Protocol for a fully vectorized POMDP generative model ``G(S, A)``.

A :class:`VectorizedGenerativeModel` is the batched, on-device counterpart of
the scalar generative interface on
:class:`~POMDPPlanners.core.environment.environment.Environment`. It is the
external model a GPU-vectorized planner such as VOPP drives during its
forward search: given a tensor of states and a tensor of per-row actions it
produces tensors of next states, observations, and rewards in a single
vectorized call, with no Python per-row loop and no host/device sync.

This protocol is deliberately narrow and *opt-in*. The base ``Environment``
ABC is untouched: an environment need not provide a vectorized model, and
every existing (scalar / small-batch) planner keeps working unchanged. Only
the handful of environments a vectorized planner is actually run on implement
this protocol, in a small module beside the environment. The environment
remains the single source of truth for configuration (spaces, reward
semantics, terminal rules); only the tight numeric kernels are re-expressed in
torch, and a parity test pins the two together.

Conventions
-----------
* Every argument and return value is a :class:`torch.Tensor` on
  :attr:`~VectorizedGenerativeModel.device`.
* ``N`` is the batch (number of parallel particles / episodes); ``ds`` and
  ``do`` are the state and observation dimensions.
* Actions are integer indices into a fixed, finite action set (the VOPP
  representative-action assumption), so an actions tensor has shape ``[N]``
  and integer dtype.
* Observation and action *keys* are the integer tree keys consumed by
  :class:`~POMDPPlanners.core.tree.vectorized_belief_tree.vectorized_belief_tree.VectorizedBeliefTree`;
  turning continuous observations into integer keys (binning / hashing) is the
  model's responsibility.
"""

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class VectorizedGenerativeModel(Protocol):
    """Batched generative model consumed by a vectorized online POMDP planner.

    Implementations expose the transition, observation, reward, terminal, and
    observation-likelihood kernels a planner needs, all as vectorized torch
    operations over a batch dimension. See the module docstring for the shape
    and dtype conventions shared by every method.
    """

    @property
    def device(self) -> torch.device:
        """Device every tensor argument and return value must live on."""
        ...  # pylint: disable=unnecessary-ellipsis

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        """Sample one next state per row under the transition model.

        Args:
            states: ``[N, ds]`` current states.
            actions: ``[N]`` integer action indices.

        Returns:
            ``[N, ds]`` sampled next states.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        """Sample one observation per row under the observation model.

        Args:
            next_states: ``[N, ds]`` post-transition states.
            actions: ``[N]`` integer action indices.

        Returns:
            ``[N, do]`` sampled observations.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        """Return one immediate reward per row.

        Args:
            states: ``[N, ds]`` current states.
            actions: ``[N]`` integer action indices.
            next_states: ``[N, ds]`` realised next states from
                :meth:`sample_next_states`.

        Returns:
            ``[N]`` immediate rewards.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    def terminal_mask(self, states: Tensor) -> Tensor:
        """Return a boolean terminal flag per row.

        Args:
            states: ``[N, ds]`` states to test.

        Returns:
            ``[N]`` boolean tensor, ``True`` where the state is terminal.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        """Return the observation log-likelihood per row (for belief updates).

        Args:
            next_states: ``[N, ds]`` post-transition states.
            actions: ``[N]`` integer action indices.
            observations: ``[N, do]`` observations to score.

        Returns:
            ``[N]`` log-likelihoods ``log Z(o | s', a)``.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    def action_keys(self, actions: Tensor) -> Tensor:
        """Map action indices to the integer keys used by the belief tree.

        Args:
            actions: ``[N]`` integer action indices.

        Returns:
            ``[N]`` integer action keys.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    def observation_keys(self, observations: Tensor) -> Tensor:
        """Map observations to the integer keys used by the belief tree.

        Args:
            observations: ``[N, do]`` observations.

        Returns:
            ``[N]`` integer observation keys.
        """
        ...  # pylint: disable=unnecessary-ellipsis
