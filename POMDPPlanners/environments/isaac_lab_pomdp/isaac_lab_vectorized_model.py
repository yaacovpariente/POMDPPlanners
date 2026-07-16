# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for the Isaac Lab planner model.

This module provides :class:`IsaacLabVectorizedModel`, a fully batched,
GPU-friendly implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
built from the *fitted* planner-side model of
:mod:`POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp`.

The Isaac Lab world is a physics simulator, so the model VOPP searches is not
the simulator itself but the linear-Gaussian surrogate fit from a warm-up of
random transitions: a :class:`LinearGaussianTransition`, a
:class:`GaussianObservationModel`, and a :class:`LinearRewardModel`. Because
those three are already linear / Gaussian, they re-express exactly as batched
torch kernels:

* transition ``s' = A s + B a + b + N(0, Sigma_tr)``,
* observation ``o = s' + N(0, Sigma_obs)`` (state and observation share one
  space, so ``ds == do``),
* reward ``r = w_s . s + w_a . a + w_n . s' + b_r``, and
* a terminal test that is always ``False`` (the Isaac velocity tasks never
  terminate the model).

Actions are integer indices into a fixed preset table of continuous action
vectors (the VOPP representative-action assumption); the ground-truth Isaac
action for index ``i`` is ``action_presets[i]``. An accompanying parity test
pins every kernel to the fitted numpy models.
"""

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
    GaussianObservationModel,
    LinearGaussianTransition,
    LinearRewardModel,
)


class IsaacLabVectorizedModel:
    """Fully vectorized torch generative model for the Isaac Lab surrogate model.

    The model batches the linear-Gaussian transition, additive-Gaussian
    observation, linear reward, always-false terminal, and observation
    log-likelihood kernels over a leading particle dimension, keeping every
    tensor on one device. Actions are integer indices into a fixed preset table
    of continuous Isaac actions.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of preset actions (rows of the action table).

    Example:
        >>> import numpy as np
        >>> import torch
        >>> from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
        ...     GaussianObservationModel,
        ...     LinearGaussianTransition,
        ...     LinearRewardModel,
        ... )
        >>> transition = LinearGaussianTransition(
        ...     np.eye(3), np.zeros((3, 2)), np.zeros(3), 0.01 * np.eye(3)
        ... )
        >>> observation = GaussianObservationModel(3, noise_std=0.1)
        >>> reward = LinearRewardModel(np.zeros(3), np.zeros(2), np.ones(3), 0.0)
        >>> presets = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        >>> model = IsaacLabVectorizedModel(
        ...     transition, observation, reward, presets, device=torch.device("cpu")
        ... )
        >>> states = torch.zeros(4, 3)
        >>> actions = torch.tensor([0, 1, 2, 1])
        >>> next_states = model.sample_next_states(states, actions)
        >>> tuple(next_states.shape), model.num_actions
        ((4, 3), 3)
    """

    def __init__(
        self,
        transition: LinearGaussianTransition,
        observation_model: GaussianObservationModel,
        reward_model: LinearRewardModel,
        action_presets: np.ndarray,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_resolution: float = 0.5,
    ) -> None:
        """Build the vectorized model from the fitted planner-side models.

        Args:
            transition: Fitted linear-Gaussian transition surrogate.
            observation_model: Fitted additive-Gaussian observation model.
            reward_model: Fitted linear reward model.
            action_presets: ``[num_actions, action_dim]`` table of continuous
                Isaac action vectors; action index ``i`` selects row ``i``.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_resolution: Grid spacing used to quantize observations
                into integer tree keys.

        Raises:
            ValueError: If the model dimensions are inconsistent or
                ``observation_resolution`` is not positive.
        """
        self._validate_dimensions(transition, observation_model, action_presets)
        if observation_resolution <= 0.0:
            raise ValueError("observation_resolution must be positive")
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self._obs_resolution = float(observation_resolution)
        self._action_table = self._to_tensor(np.asarray(action_presets, dtype=np.float64))
        self.num_actions = int(self._action_table.shape[0])
        self._build_transition(transition)
        self._build_observation(observation_model)
        self._build_reward(reward_model)
        self._hash_primes = self._build_hash_primes(transition.dim)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_dimensions(
        transition: LinearGaussianTransition,
        observation_model: GaussianObservationModel,
        action_presets: np.ndarray,
    ) -> None:
        if observation_model.dim != transition.dim:
            raise ValueError("observation dim must equal state dim (obs == state space)")
        presets = np.asarray(action_presets)
        if presets.ndim != 2 or presets.shape[0] == 0:
            raise ValueError("action_presets must be a non-empty [num_actions, action_dim] array")
        if presets.shape[1] != transition.action_dim:
            raise ValueError("action_presets action_dim must match the transition action_dim")

    def _build_transition(self, transition: LinearGaussianTransition) -> None:
        self._weight_state = self._to_tensor(
            transition._weight_state
        )  # pylint: disable=protected-access
        self._weight_action = self._to_tensor(
            transition._weight_action
        )  # pylint: disable=protected-access
        self._bias = self._to_tensor(transition._bias)  # pylint: disable=protected-access
        covariance = self._to_tensor(
            transition._normal.covariance
        )  # pylint: disable=protected-access
        self._trans_chol_t = torch.linalg.cholesky(covariance).mT.contiguous()

    def _build_observation(self, observation_model: GaussianObservationModel) -> None:
        covariance = self._to_tensor(
            observation_model._normal.covariance
        )  # pylint: disable=protected-access
        self._obs_chol_t = torch.linalg.cholesky(covariance).mT.contiguous()
        self._obs_inv = torch.linalg.inv(covariance).contiguous()
        logdet = torch.linalg.slogdet(covariance)[1]
        dim = observation_model.dim
        self._obs_lognorm = float(-0.5 * dim * math.log(2.0 * math.pi) - 0.5 * logdet.item())

    def _build_reward(self, reward_model: LinearRewardModel) -> None:
        self._reward_state = self._to_tensor(
            reward_model._weight_state
        )  # pylint: disable=protected-access
        self._reward_action = self._to_tensor(
            reward_model._weight_action
        )  # pylint: disable=protected-access
        self._reward_next = self._to_tensor(
            reward_model._weight_next_state
        )  # pylint: disable=protected-access
        self._reward_bias = float(reward_model._bias)  # pylint: disable=protected-access

    def _build_hash_primes(self, dim: int) -> Tensor:
        primes = self._first_primes(dim)
        return torch.as_tensor(primes, dtype=torch.int64, device=self.device)

    @staticmethod
    def _first_primes(count: int) -> np.ndarray:
        primes = []
        candidate = 73856093
        while len(primes) < count:
            if all(candidate % p != 0 for p in range(2, int(candidate**0.5) + 1)):
                primes.append(candidate)
            candidate += 2
        return np.asarray(primes, dtype=np.int64)

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    @property
    def action_vectors(self) -> Tensor:
        """The ``[num_actions, action_dim]`` table of continuous action vectors."""
        return self._action_table

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        mean = self._transition_mean(states, actions)
        noise = torch.randn(states.shape[0], states.shape[1], dtype=self.dtype, device=self.device)
        return mean + noise @ self._trans_chol_t

    def _transition_mean(self, states: Tensor, actions: Tensor) -> Tensor:
        action_vectors = self._action_table[actions]
        return states @ self._weight_state.mT + action_vectors @ self._weight_action.mT + self._bias

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # observation noise is additive and action-independent.
        noise = torch.randn(
            next_states.shape[0], next_states.shape[1], dtype=self.dtype, device=self.device
        )
        return next_states + noise @ self._obs_chol_t

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        action_vectors = self._action_table[actions]
        return (
            states @ self._reward_state
            + action_vectors @ self._reward_action
            + next_states @ self._reward_next
            + self._reward_bias
        )

    def terminal_mask(self, states: Tensor) -> Tensor:
        return torch.zeros(states.shape[0], dtype=torch.bool, device=self.device)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # additive-Gaussian likelihood does not depend on the action.
        diff = observations - next_states
        maha = torch.einsum("ni,ij,nj->n", diff, self._obs_inv, diff)
        return self._obs_lognorm - 0.5 * maha

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        quantized = torch.floor(observations / self._obs_resolution).to(torch.int64)
        return (quantized * self._hash_primes).sum(dim=1)
