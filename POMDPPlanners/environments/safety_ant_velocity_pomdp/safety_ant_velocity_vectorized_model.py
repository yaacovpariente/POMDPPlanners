# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for Safety Ant Velocity.

This module provides :class:`SafetyAntVelocityVectorizedModel`, a fully
batched, GPU-friendly implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp.SafeAntVelocityPOMDP`.

It re-expresses the environment's damped-force transition (a deterministic
integration step driven by a uniformly-sampled force direction), its
identity-mean diagonal-Gaussian observation model, and its speed-based reward
and terminal rules as torch tensor kernels so a vectorized planner (VOPP) can
run tens of thousands of parallel simulations on the GPU without a host/device
sync. Every constant (time step, mass, damping, max force, force scales, noise
standard deviations, safety threshold, penalty, movement scale) is read from a
live environment instance, so the environment stays the single source of truth
for configuration; only the numeric kernels are duplicated in torch. The
accompanying parity test pins these kernels to the environment's native
(C++) implementations.

The Safety Ant Velocity action space is already discrete -- four force levels
indexed ``0..3`` -- so actions pass straight through as integer indices into
the environment's ``DEFAULT_FORCE_SCALES`` table; no preset representative
action table is synthesized. The only supported configuration is one whose
discrete action set is exactly ``range(len(force_scales))`` (so that action
index ``i`` selects ``force_scales[i]``); any other action set raises
:class:`NotImplementedError`.
"""

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp import (
    DEFAULT_FORCE_SCALES,
    SafeAntVelocityPOMDP,
)

# Spatial-hash primes for turning quantized 4-D observations into integer keys.
_HASH_PRIMES = (73856093, 19349663, 83492791, 39916801)


class SafetyAntVelocityVectorizedModel:
    """Fully vectorized torch generative model for the Safety Ant Velocity POMDP.

    The model batches the transition, observation, reward, terminal, and
    observation-likelihood kernels over a leading particle dimension and keeps
    every tensor on a single device. States and observations are the 4-D vector
    ``[position_x, position_y, velocity_x, velocity_y]``. Actions are integer
    indices ``0..num_actions-1`` into the environment's force-scale table.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of discrete force levels (rows in the force table).

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp import (
        ...     SafeAntVelocityPOMDP,
        ... )
        >>> from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_vectorized_model import (
        ...     SafetyAntVelocityVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> env = SafeAntVelocityPOMDP(discount_factor=0.99)
        >>> model = SafetyAntVelocityVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.tensor([[0.0, 0.0, 0.5, 0.0], [1.0, 0.0, 0.0, 0.3]])
        >>> actions = torch.tensor([2, 2])
        >>> next_states = model.sample_next_states(states, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(observations.shape), tuple(rewards.shape)
        ((2, 4), (2, 4), (2,))
    """

    def __init__(
        self,
        env: SafeAntVelocityPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_resolution: float = 0.1,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The environment whose physics, noise, and reward parameters are
                mirrored.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_resolution: Grid spacing used to quantize continuous
                observations into integer tree keys.

        Raises:
            NotImplementedError: If ``env``'s discrete action set is not exactly
                ``range(len(force_scales))`` (the index-into-force-table
                assumption the transition kernel makes).
            ValueError: If ``observation_resolution`` or the environment mass is
                not positive.
        """
        self._require_supported_action_set(env)
        if observation_resolution <= 0.0:
            raise ValueError("observation_resolution must be positive")
        if env.mass <= 0.0:
            raise ValueError("environment mass must be positive")
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self._obs_resolution = float(observation_resolution)
        self._hash_primes = torch.tensor(_HASH_PRIMES, dtype=torch.int64, device=self.device)
        self._build_physics(env)
        self._build_noise_kernels(env)
        self._build_reward_geometry(env)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_supported_action_set(env: SafeAntVelocityPOMDP) -> None:
        expected = list(range(len(DEFAULT_FORCE_SCALES)))
        if list(env.get_actions()) != expected:
            raise NotImplementedError(
                "vectorized model supports only the default force-level action "
                "set range(len(force_scales)); action index i must select "
                "force_scales[i]"
            )

    def _build_physics(self, env: SafeAntVelocityPOMDP) -> None:
        force_scales = self._to_tensor(np.asarray(DEFAULT_FORCE_SCALES, dtype=np.float64))
        self.num_actions = int(force_scales.shape[0])
        self._dt = float(env.dt)
        self._mass = float(env.mass)
        self._damping = float(env.damping)
        # Per-action force magnitude: force_scales[action] * max_force.
        self._force_magnitudes = force_scales * float(env.max_force)

    def _build_noise_kernels(self, env: SafeAntVelocityPOMDP) -> None:
        std = np.array(
            [env.position_noise, env.position_noise, env.velocity_noise, env.velocity_noise],
            dtype=np.float64,
        )
        self._obs_std = self._to_tensor(std)
        self._obs_inv_var = self._to_tensor(1.0 / (std * std))
        self._obs_lognorm = float(
            -0.5 * std.shape[0] * math.log(2.0 * math.pi) - np.sum(np.log(std))
        )

    def _build_reward_geometry(self, env: SafeAntVelocityPOMDP) -> None:
        self._safe_threshold = float(env.safe_velocity_threshold)
        self._terminal_threshold = float(env.safe_velocity_threshold) * 1.5
        self._violation_penalty = float(env.safety_violation_penalty)
        self._movement_scale = float(env.movement_reward_scale)

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        force_magnitude = self._force_magnitudes[actions]
        theta = torch.rand(states.shape[0], dtype=self.dtype, device=self.device)
        theta = theta * (2.0 * math.pi) - math.pi
        force_x = force_magnitude * torch.cos(theta)
        force_y = force_magnitude * torch.sin(theta)
        return self._integrate_step(states, force_x, force_y)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # Observation noise does not depend on the action.
        noise = torch.randn(next_states.shape, dtype=self.dtype, device=self.device)
        return next_states + noise * self._obs_std

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del states, actions  # Reward scores the realised next-state speed only.
        speed = torch.linalg.vector_norm(next_states[:, 2:4], dim=1)
        reward = speed * self._movement_scale
        unsafe = speed > self._safe_threshold
        return reward + self._violation_penalty * unsafe.to(self.dtype)

    def terminal_mask(self, states: Tensor) -> Tensor:
        speed = torch.linalg.vector_norm(states[:, 2:4], dim=1)
        return speed > self._terminal_threshold

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # Diagonal-Gaussian likelihood does not depend on the action.
        diff = observations - next_states
        maha = (diff * diff * self._obs_inv_var).sum(dim=1)
        return self._obs_lognorm - 0.5 * maha

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        quantized = torch.floor(observations / self._obs_resolution).to(torch.int64)
        return (quantized * self._hash_primes).sum(dim=1)

    # ------------------------------------------------------------------ #
    # Internal physics helper
    # ------------------------------------------------------------------ #

    def _integrate_step(self, states: Tensor, force_x: Tensor, force_y: Tensor) -> Tensor:
        velocity = states[:, 2:4]
        accel_x = (force_x - self._damping * velocity[:, 0]) / self._mass
        accel_y = (force_y - self._damping * velocity[:, 1]) / self._mass
        next_vx = velocity[:, 0] + accel_x * self._dt
        next_vy = velocity[:, 1] + accel_y * self._dt
        next_px = states[:, 0] + next_vx * self._dt
        next_py = states[:, 1] + next_vy * self._dt
        return torch.stack([next_px, next_py, next_vx, next_vy], dim=1)
