# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for Continuous Light-Dark.

This module provides :class:`ContinuousLightDarkVectorizedModel`, a fully
batched, GPU-friendly implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp.ContinuousLightDarkPOMDP`.

It re-expresses the environment's ``NORMAL_NOISE`` observation model and
``CONSTANT_HAZARD_PENALTY`` reward model as torch tensor kernels so a
vectorized planner (VOPP) can run tens of thousands of parallel simulations
on the GPU without a host/device sync. Every constant (covariances, beacons,
obstacles, goal, radii, costs) is read from a live environment instance, so
the environment stays the single source of truth for configuration; only the
numeric kernels are duplicated in torch. The accompanying parity test pins
these kernels to the environment's native (C++/numba) implementations.

Only one environment configuration is supported: the ``CONSTANT_HAZARD_PENALTY``
reward model, the ``NORMAL_NOISE`` observation model, and
``is_obstacle_hit_terminal=False``. The default draw-coupled hazard-terminal
path appends a dynamic absorbing state slot to the transition and is not
modeled here; all three conditions are checked at construction and any
mismatch raises :class:`NotImplementedError`.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ContinuousLightDarkRewardModel,
    ObservationModelType,
)

_DEFAULT_ACTION_VECTORS = np.array(
    [[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float64
)
# Spatial-hash primes for turning quantized 2-D observations into integer keys.
_HASH_PRIME_X = 73856093
_HASH_PRIME_Y = 19349663


class ContinuousLightDarkVectorizedModel:
    """Fully vectorized torch generative model for the Continuous Light-Dark POMDP.

    The model batches the transition, observation, reward, terminal, and
    observation-likelihood kernels over a leading particle dimension and keeps
    every tensor on a single device. Actions are integer indices into a fixed
    action table (defaulting to the four unit-direction moves); observations
    are the noisy 2-D positions of the ``NORMAL_NOISE`` model.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of rows in the action table.

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ...     ContinuousLightDarkPOMDP,
        ... )
        >>> from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_vectorized_model import (
        ...     ContinuousLightDarkVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> env = ContinuousLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=False)
        >>> model = ContinuousLightDarkVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.tensor([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]])
        >>> actions = torch.tensor([2, 2, 2])  # move right
        >>> next_states = model.sample_next_states(states, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(observations.shape), tuple(rewards.shape)
        ((3, 2), (3, 2), (3,))
    """

    def __init__(
        self,
        env: ContinuousLightDarkPOMDP,
        action_vectors: Optional[np.ndarray] = None,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_resolution: float = 0.1,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The environment whose parameters and model types are mirrored.
            action_vectors: Optional ``[num_actions, 2]`` array of continuous
                action vectors. Defaults to the four unit-direction moves.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_resolution: Grid spacing used to quantize continuous
                observations into integer tree keys.

        Raises:
            NotImplementedError: If ``env`` uses a reward or observation model
                other than the supported ``CONSTANT_HAZARD_PENALTY`` /
                ``NORMAL_NOISE`` pair.
            ValueError: If ``observation_resolution`` is not positive.
        """
        self._require_supported_models(env)
        if observation_resolution <= 0.0:
            raise ValueError("observation_resolution must be positive")
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self._obs_resolution = float(observation_resolution)
        vectors = _DEFAULT_ACTION_VECTORS if action_vectors is None else np.asarray(action_vectors)
        self._action_table = self._to_tensor(vectors)
        self.num_actions = int(self._action_table.shape[0])
        self._build_noise_kernels(env)
        self._build_reward_geometry(env)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_supported_models(env: ContinuousLightDarkPOMDP) -> None:
        # Exact-type check on purpose: the zero-mean-hazard model subclasses
        # ContinuousLightDarkRewardModel, so isinstance would wrongly accept it.
        reward_model_type = type(env.reward_model)  # pylint: disable=unidiomatic-typecheck
        if reward_model_type is not ContinuousLightDarkRewardModel:
            raise NotImplementedError(
                "vectorized model supports only the CONSTANT_HAZARD_PENALTY reward model"
            )
        if env.observation_model_type is not ObservationModelType.NORMAL_NOISE:
            raise NotImplementedError(
                "vectorized model supports only the NORMAL_NOISE observation model"
            )
        if env.is_obstacle_hit_terminal:
            raise NotImplementedError(
                "vectorized model requires is_obstacle_hit_terminal=False "
                "(the draw-coupled hazard-terminal absorbing slot is not modeled)"
            )

    def _build_noise_kernels(self, env: ContinuousLightDarkPOMDP) -> None:
        trans_cov = self._to_tensor(env.state_transition_cov_matrix)
        obs_cov_far = self._to_tensor(env.observation_cov_matrix)
        obs_cov_near = obs_cov_far * 0.5
        self._trans_chol_t = torch.linalg.cholesky(trans_cov).mT.contiguous()
        self._obs_chol_t = torch.stack(
            [torch.linalg.cholesky(obs_cov_near).mT, torch.linalg.cholesky(obs_cov_far).mT]
        )
        self._trans_inv, self._trans_lognorm = self._inverse_and_lognorm(trans_cov)
        obs_inv_near, obs_lognorm_near = self._inverse_and_lognorm(obs_cov_near)
        obs_inv_far, obs_lognorm_far = self._inverse_and_lognorm(obs_cov_far)
        self._obs_inv = torch.stack([obs_inv_near, obs_inv_far])
        self._obs_lognorm = torch.stack([obs_lognorm_near, obs_lognorm_far])
        self._beacons = self._to_tensor(np.asarray(env.beacons, dtype=np.float64).T)
        self._beacon_radius_sq = float(env.beacon_radius) ** 2

    def _build_reward_geometry(self, env: ContinuousLightDarkPOMDP) -> None:
        self._goal = self._to_tensor(np.asarray(env.goal_state, dtype=np.float64))
        self._obstacles = self._to_tensor(np.asarray(env.obstacles, dtype=np.float64).T)
        self._goal_radius = float(env.goal_state_radius)
        self._obstacle_radius = float(env.obstacle_radius)
        self._grid_size = float(env.grid_size)
        self._fuel_cost = float(env.fuel_cost)
        self._goal_reward = float(env.goal_reward)
        self._obstacle_reward = float(env.obstacle_reward)
        self._hit_probability = float(env.obstacle_hit_probability)

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    def _inverse_and_lognorm(self, cov: Tensor) -> Tuple[Tensor, Tensor]:
        inverse = torch.linalg.inv(cov).contiguous()
        logdet = torch.linalg.slogdet(cov)[1]
        lognorm = -math.log(2.0 * math.pi) - 0.5 * logdet
        return inverse, lognorm

    @property
    def action_vectors(self) -> Tensor:
        """The ``[num_actions, 2]`` table of continuous action displacements.

        Row ``i`` is the ``(dx, dy)`` displacement applied by action index
        ``i`` (used, for example, to draw action arrows in a visualization).
        """
        return self._action_table

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        mean = states + self._action_table[actions]
        noise = torch.randn(states.shape[0], 2, dtype=self.dtype, device=self.device)
        return mean + noise @ self._trans_chol_t

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # NORMAL_NOISE observations do not depend on the action.
        cov_index = self._near_beacon_index(next_states)
        chol_t = self._obs_chol_t[cov_index]
        noise = torch.randn(next_states.shape[0], 2, dtype=self.dtype, device=self.device)
        return next_states + torch.einsum("nj,nji->ni", noise, chol_t)

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del states, actions  # Reward scores the realised next state only.
        dist_to_goal = torch.linalg.vector_norm(next_states - self._goal, dim=1)
        reward = -self._fuel_cost - dist_to_goal
        is_goal = dist_to_goal <= self._goal_radius
        in_obstacle = self._in_obstacle_range(next_states) & ~is_goal
        out_of_grid = self._out_of_grid(next_states) & ~is_goal & ~in_obstacle
        reward = reward + self._goal_reward * is_goal.to(self.dtype)
        reward = reward + self._obstacle_reward * out_of_grid.to(self.dtype)
        reward = reward + self._sample_hazard(in_obstacle)
        return reward

    def terminal_mask(self, states: Tensor) -> Tensor:
        dist_to_goal = torch.linalg.vector_norm(states[:, :2] - self._goal, dim=1)
        return dist_to_goal <= self._goal_radius

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # NORMAL_NOISE likelihood does not depend on the action.
        cov_index = self._near_beacon_index(next_states)
        inverse = self._obs_inv[cov_index]
        diff = observations - next_states
        maha = torch.einsum("ni,nij,nj->n", diff, inverse, diff)
        return self._obs_lognorm[cov_index] - 0.5 * maha

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        quantized = torch.floor(observations / self._obs_resolution).to(torch.int64)
        return quantized[:, 0] * _HASH_PRIME_X + quantized[:, 1] * _HASH_PRIME_Y

    # ------------------------------------------------------------------ #
    # Internal reward / geometry helpers
    # ------------------------------------------------------------------ #

    def _near_beacon_index(self, points: Tensor) -> Tensor:
        # 0 -> near a beacon (use the near covariance), 1 -> far from all.
        diff = points[:, None, :] - self._beacons[None, :, :]
        min_sq = (diff * diff).sum(dim=-1).min(dim=1).values
        return (min_sq >= self._beacon_radius_sq).to(torch.long)

    def _in_obstacle_range(self, points: Tensor) -> Tensor:
        if self._obstacles.shape[0] == 0:
            return torch.zeros(points.shape[0], dtype=torch.bool, device=self.device)
        diff = points[:, None, :] - self._obstacles[None, :, :]
        min_sq = (diff * diff).sum(dim=-1).min(dim=1).values
        return min_sq <= self._obstacle_radius * self._obstacle_radius

    def _out_of_grid(self, points: Tensor) -> Tensor:
        below = (points < 0.0).any(dim=1)
        above = (points > self._grid_size).any(dim=1)
        return below | above

    def _sample_hazard(self, in_obstacle: Tensor) -> Tensor:
        draws = torch.rand(in_obstacle.shape[0], dtype=self.dtype, device=self.device)
        hit = in_obstacle & (draws < self._hit_probability)
        return self._obstacle_reward * hit.to(self.dtype)
