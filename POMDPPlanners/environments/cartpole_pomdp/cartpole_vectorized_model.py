# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for CartPole POMDP.

This module provides :class:`CartPoleVectorizedModel`, a fully batched,
GPU-friendly implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp.CartPolePOMDP`.

It re-expresses the environment's classical cart-pole physics (the same
``temp`` / ``thetaacc`` / ``xacc`` update the native C++ kernel integrates),
its zero-mean Gaussian transition and observation noise, its per-step
"+1 while alive" reward, and its out-of-bounds terminal rule as torch tensor
kernels so a vectorized planner (VOPP) can run tens of thousands of parallel
simulations on the GPU without a host/device sync. Every constant (masses,
lengths, force magnitude, integration step, thresholds, covariances) is read
from a live environment instance, so the environment stays the single source
of truth for configuration; only the numeric kernels are duplicated in torch.
The accompanying parity test pins these kernels to the environment's native
(C++) implementations.

Only the default ``"euler"`` kinematics integrator is supported. The
semi-implicit Euler variant is a different (velocity-first) update and is
checked at construction; any mismatch raises :class:`NotImplementedError`.
"""

import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp import CartPolePOMDP

# Spatial-hash primes for turning a quantized 4-D observation into an int key.
_HASH_PRIMES = (73856093, 19349663, 83492791, 50331653)


class CartPoleVectorizedModel:
    """Fully vectorized torch generative model for the CartPole POMDP.

    The model batches the transition, observation, reward, terminal, and
    observation-likelihood kernels over a leading particle dimension and keeps
    every tensor on a single device. The 4-D state is
    ``[x, x_dot, theta, theta_dot]``; actions are the two discrete pushes
    (``0`` = push left, ``1`` = push right) used directly as their own tree
    keys; observations are the noisy 4-D state measurements of the environment.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of discrete actions (always two for CartPole).

    Example:
        >>> import numpy as np
        >>> import torch
        >>> from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp import (
        ...     CartPolePOMDP,
        ... )
        >>> from POMDPPlanners.environments.cartpole_pomdp.cartpole_vectorized_model import (
        ...     CartPoleVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> env = CartPolePOMDP(discount_factor=0.99, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
        >>> model = CartPoleVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.zeros(3, 4)
        >>> actions = torch.tensor([1, 0, 1])  # push right, left, right
        >>> next_states = model.sample_next_states(states, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(observations.shape), tuple(rewards.shape)
        ((3, 4), (3, 4), (3,))
    """

    def __init__(
        self,
        env: CartPolePOMDP,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        observation_resolution: float = 0.05,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The environment whose physics, covariances, and thresholds are
                mirrored.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_resolution: Grid spacing used to quantize continuous
                observations into integer tree keys.

        Raises:
            NotImplementedError: If ``env`` uses a kinematics integrator other
                than the supported plain ``"euler"`` update.
            ValueError: If ``observation_resolution`` is not positive.
        """
        self._require_supported_config(env)
        if observation_resolution <= 0.0:
            raise ValueError("observation_resolution must be positive")
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self._obs_resolution = float(observation_resolution)
        self.num_actions = 2
        self._build_physics(env)
        self._build_noise_kernels(env)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_supported_config(env: CartPolePOMDP) -> None:
        if env.kinematics_integrator != "euler":
            raise NotImplementedError(
                "vectorized model supports only the 'euler' kinematics integrator, "
                f"got {env.kinematics_integrator!r}"
            )

    def _build_physics(self, env: CartPolePOMDP) -> None:
        self._force_mag = float(env.force_mag)
        self._total_mass = float(env.total_mass)
        self._polemass_length = float(env.polemass_length)
        self._gravity = float(env.gravity)
        self._length = float(env.length)
        self._masspole = float(env.masspole)
        self._tau = float(env.tau)
        self._x_threshold = float(env.x_threshold)
        self._theta_threshold = float(env.theta_threshold_radians)

    def _build_noise_kernels(self, env: CartPolePOMDP) -> None:
        trans_cov = self._to_tensor(np.asarray(env.state_transition_cov, dtype=np.float64))
        obs_cov = self._to_tensor(np.asarray(env.noise_cov, dtype=np.float64))
        self._trans_chol_t = torch.linalg.cholesky(trans_cov).mT.contiguous()
        self._obs_chol_t = torch.linalg.cholesky(obs_cov).mT.contiguous()
        self._obs_inv, self._obs_lognorm = self._inverse_and_lognorm(obs_cov)

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    def _inverse_and_lognorm(self, cov: Tensor) -> Tuple[Tensor, float]:
        inverse = torch.linalg.inv(cov).contiguous()
        logdet = float(torch.linalg.slogdet(cov)[1])
        lognorm = -0.5 * (cov.shape[0] * math.log(2.0 * math.pi) + logdet)
        return inverse, lognorm

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        mean = self._transition_mean(states, actions)
        noise = torch.randn(states.shape[0], 4, dtype=self.dtype, device=self.device)
        return mean + noise @ self._trans_chol_t

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # Observation noise does not depend on the action.
        noise = torch.randn(next_states.shape[0], 4, dtype=self.dtype, device=self.device)
        return next_states + noise @ self._obs_chol_t

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del actions, next_states  # Reward is +1 while the current state is alive.
        alive = ~self.terminal_mask(states)
        return alive.to(self.dtype)

    def terminal_mask(self, states: Tensor) -> Tensor:
        x = states[:, 0]
        theta = states[:, 2]
        return (
            (x < -self._x_threshold)
            | (x > self._x_threshold)
            | (theta < -self._theta_threshold)
            | (theta > self._theta_threshold)
        )

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
        primes = torch.tensor(_HASH_PRIMES, dtype=torch.int64, device=self.device)
        return (quantized * primes).sum(dim=1)

    # ------------------------------------------------------------------ #
    # Internal physics helpers
    # ------------------------------------------------------------------ #

    def _transition_mean(self, states: Tensor, actions: Tensor) -> Tensor:
        x, x_dot, theta, theta_dot = states.unbind(dim=1)
        force = (actions.to(self.dtype) * 2.0 - 1.0) * self._force_mag
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        temp = (force + self._polemass_length * theta_dot * theta_dot * sintheta) / self._total_mass
        thetaacc = (self._gravity * sintheta - costheta * temp) / (
            self._length * (4.0 / 3.0 - self._masspole * costheta * costheta / self._total_mass)
        )
        xacc = temp - self._polemass_length * thetaacc * costheta / self._total_mass
        next_x = x + self._tau * x_dot
        next_x_dot = x_dot + self._tau * xacc
        next_theta = theta + self._tau * theta_dot
        next_theta_dot = theta_dot + self._tau * thetaacc
        return torch.stack([next_x, next_x_dot, next_theta, next_theta_dot], dim=1)
