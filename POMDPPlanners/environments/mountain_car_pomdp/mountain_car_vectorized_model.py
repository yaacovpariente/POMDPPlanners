# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for the Mountain Car POMDP.

This module provides :class:`MountainCarVectorizedModel`, a fully batched,
GPU-friendly implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp.MountainCarPOMDP`.

It re-expresses the environment's physics transition (the ``cos(3*position)``
hill, gravity, force, and the velocity / position clamping with the
min-position corner rule), its Gaussian observation model, and its sparse
step reward as torch tensor kernels so a vectorized planner (VOPP) can run
tens of thousands of parallel simulations on the GPU without a host/device
sync. Every constant (power, gravity, speed / position bounds, goal, the
discrete action set, and the transition / observation covariances) is read
from a live environment instance, so the environment stays the single source
of truth for configuration; only the numeric kernels are duplicated in torch.
The accompanying parity test pins these kernels to the environment's native
(C++) implementations.

The Mountain Car action set is already discrete (``[-1, 0, 1]``); the model
uses it directly, so action index ``i`` maps to force multiplier
``env.get_actions()[i]``. The model requires the standard scalar action set;
a non-scalar action set raises :class:`NotImplementedError`.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp import (
    MountainCarPOMDP,
)

# Spatial-hash primes for turning quantized 2-D observations into integer keys.
_HASH_PRIME_X = 73856093
_HASH_PRIME_Y = 19349663


class MountainCarVectorizedModel:
    """Fully vectorized torch generative model for the Mountain Car POMDP.

    The model batches the transition, observation, reward, terminal, and
    observation-likelihood kernels over a leading particle dimension and keeps
    every tensor on a single device. Actions are integer indices into the
    environment's discrete action set (``[-1, 0, 1]`` by default); observations
    are the noisy 2-D ``[position, velocity]`` measurements of the Gaussian
    observation model.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of actions in the discrete action set.

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp import (
        ...     MountainCarPOMDP,
        ... )
        >>> from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_vectorized_model import (
        ...     MountainCarVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> env = MountainCarPOMDP(discount_factor=0.99)
        >>> model = MountainCarVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.tensor([[-0.5, 0.0], [-0.4, 0.01], [-0.6, -0.01]])
        >>> actions = torch.tensor([2, 0, 1])  # forward, reverse, neutral
        >>> next_states = model.sample_next_states(states, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(observations.shape), tuple(rewards.shape)
        ((3, 2), (3, 2), (3,))
    """

    def __init__(
        self,
        env: MountainCarPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_resolution: float = 0.01,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The environment whose parameters and action set are mirrored.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_resolution: Grid spacing used to quantize continuous
                observations into integer tree keys.

        Raises:
            NotImplementedError: If ``env`` exposes a non-scalar action set that
                the scalar force-multiplier kernel cannot model.
            ValueError: If ``observation_resolution`` is not positive.
        """
        self._require_supported_action_set(env)
        if observation_resolution <= 0.0:
            raise ValueError("observation_resolution must be positive")
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self._obs_resolution = float(observation_resolution)
        self._build_physics(env)
        self._build_noise_kernels(env)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_supported_action_set(env: MountainCarPOMDP) -> None:
        actions = env.get_actions()
        if not actions or not all(np.isscalar(action) for action in actions):
            raise NotImplementedError(
                "vectorized model supports only a non-empty scalar discrete "
                "action set (force multipliers such as [-1, 0, 1])"
            )

    def _build_physics(self, env: MountainCarPOMDP) -> None:
        self.num_actions = len(env.get_actions())
        self._action_forces = self._to_tensor(
            np.asarray(env.get_actions(), dtype=np.float64) * env.power
        )
        self._gravity = float(env.gravity)
        self._max_speed = float(env.max_speed)
        self._min_position = float(env.min_position)
        self._max_position = float(env.max_position)
        self._goal_position = float(env.goal_position)

    def _build_noise_kernels(self, env: MountainCarPOMDP) -> None:
        trans_cov = self._to_tensor(np.asarray(env.state_transition_cov, dtype=np.float64))
        obs_cov = self._to_tensor(np.asarray(env.cov_matrix, dtype=np.float64))
        self._trans_chol_t = torch.linalg.cholesky(trans_cov).mT.contiguous()
        self._obs_chol_t = torch.linalg.cholesky(obs_cov).mT.contiguous()
        self._obs_inv, self._obs_lognorm = self._inverse_and_lognorm(obs_cov)

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    def _inverse_and_lognorm(self, cov: Tensor) -> Tuple[Tensor, Tensor]:
        inverse = torch.linalg.inv(cov).contiguous()
        logdet = torch.linalg.slogdet(cov)[1]
        lognorm = -math.log(2.0 * math.pi) - 0.5 * logdet
        return inverse, lognorm

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        mean = self._deterministic_next(states, actions)
        noise = torch.randn(states.shape[0], 2, dtype=self.dtype, device=self.device)
        return self._apply_bounds(mean + noise @ self._trans_chol_t)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # Observations do not depend on the action.
        noise = torch.randn(next_states.shape[0], 2, dtype=self.dtype, device=self.device)
        return next_states + noise @ self._obs_chol_t

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del actions, next_states  # Reward scores the current state's position only.
        reached_goal = states[:, 0] >= self._goal_position
        return torch.where(
            reached_goal,
            torch.zeros((), dtype=self.dtype, device=self.device),
            -torch.ones((), dtype=self.dtype, device=self.device),
        )

    def terminal_mask(self, states: Tensor) -> Tensor:
        return states[:, 0] >= self._goal_position

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # Observation likelihood does not depend on the action.
        diff = observations - next_states
        maha = torch.einsum("ni,ij,nj->n", diff, self._obs_inv, diff)
        return self._obs_lognorm - 0.5 * maha

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        quantized = torch.floor(observations / self._obs_resolution).to(torch.int64)
        return quantized[:, 0] * _HASH_PRIME_X + quantized[:, 1] * _HASH_PRIME_Y

    # ------------------------------------------------------------------ #
    # Internal physics helpers
    # ------------------------------------------------------------------ #

    def _deterministic_next(self, states: Tensor, actions: Tensor) -> Tensor:
        position = states[:, 0]
        velocity = states[:, 1] + self._action_forces[actions]
        velocity = velocity - self._gravity * torch.cos(3.0 * position)
        velocity = torch.clamp(velocity, -self._max_speed, self._max_speed)
        next_position = torch.clamp(position + velocity, self._min_position, self._max_position)
        return self._resolve_min_position_corner(next_position, velocity)

    def _apply_bounds(self, sample: Tensor) -> Tensor:
        velocity = torch.clamp(sample[:, 1], -self._max_speed, self._max_speed)
        position = torch.clamp(sample[:, 0], self._min_position, self._max_position)
        return self._resolve_min_position_corner(position, velocity)

    def _resolve_min_position_corner(self, position: Tensor, velocity: Tensor) -> Tensor:
        at_floor = (position == self._min_position) & (velocity < 0.0)
        velocity = torch.where(at_floor, torch.zeros_like(velocity), velocity)
        return torch.stack((position, velocity), dim=1)
