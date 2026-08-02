# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for the (discrete) Push POMDP.

This module provides :class:`PushVectorizedModel`, a fully batched,
GPU-friendly implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.push_pomdp.push_pomdp.PushPOMDP`.

It re-expresses the environment's closed-form deterministic push transition
(robot move with obstacle blocking, friction-scaled object push, grid
clipping) and its ``CONSTANT_HAZARD_PENALTY`` reward / Gaussian
object-position observation model as torch tensor kernels, so a vectorized
planner (VOPP) can run tens of thousands of parallel simulations on the GPU
without a host/device sync. Every constant (grid size, push threshold,
friction, obstacle / dangerous-area geometry, penalties, observation noise,
action-error probability) is read from a live environment instance, so the
environment stays the single source of truth for configuration; only the
numeric kernels are duplicated in torch. The accompanying parity test pins
these kernels to the environment's native (C++/numpy) implementations.

The discrete Push POMDP is a natural fit for VOPP: its four moves
(``up``/``down``/``right``/``left``) form the fixed, finite representative
action set indexed ``0..3`` directly, with no preset-table synthesis. Only
the standard ``CONSTANT_HAZARD_PENALTY`` reward model is supported; the
zero-mean-shock and distance-decayed reward variants are not modeled and are
rejected at construction with :class:`NotImplementedError`.
"""

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.environments.push_pomdp.push_pomdp_utils.push_reward_models import (
    RewardModelType,
)

# Action index -> (dx, dy) offset, matching PushPOMDP.actions ordering
# ("up", "down", "right", "left").
_ACTION_VECTORS = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
# Spatial-hash primes for turning quantized 2-D object observations into keys.
_HASH_PRIME_X = 73856093
_HASH_PRIME_Y = 19349663


class PushVectorizedModel:
    """Fully vectorized torch generative model for the discrete Push POMDP.

    The model batches the transition, observation, reward, terminal, and
    observation-likelihood kernels over a leading particle dimension and keeps
    every tensor on a single device. State and observation are the 6-D vector
    ``[robot_x, robot_y, object_x, object_y, target_x, target_y]``; only the
    object-position slice carries observation noise. Actions are the four
    integer move indices (``0`` up, ``1`` down, ``2`` right, ``3`` left);
    observation keys hash the quantized object position.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of moves in the action set (always 4).

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
        >>> from POMDPPlanners.environments.push_pomdp.push_vectorized_model import (
        ...     PushVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> env = PushPOMDP(discount_factor=0.99)
        >>> model = PushVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.tensor([[1.0, 1.0, 1.5, 1.0, 9.0, 9.0]])
        >>> actions = torch.tensor([2])  # move right
        >>> next_states = model.sample_next_states(states, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(observations.shape), tuple(rewards.shape)
        ((1, 6), (1, 6), (1,))
    """

    def __init__(
        self,
        env: PushPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_resolution: float = 0.1,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The environment whose parameters and reward model are mirrored.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_resolution: Grid spacing used to quantize continuous
                object observations into integer tree keys.

        Raises:
            NotImplementedError: If ``env`` uses a reward model other than the
                supported ``CONSTANT_HAZARD_PENALTY`` variant.
            ValueError: If ``observation_resolution`` is not positive.
        """
        self._require_supported_models(env)
        if observation_resolution <= 0.0:
            raise ValueError("observation_resolution must be positive")
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self._obs_resolution = float(observation_resolution)
        self._action_table = self._to_tensor(_ACTION_VECTORS)
        self.num_actions = int(self._action_table.shape[0])
        self._build_transition_geometry(env)
        self._build_reward_geometry(env)
        self._build_observation_kernel(env)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_supported_models(env: PushPOMDP) -> None:
        if env.reward_model_type is not RewardModelType.CONSTANT_HAZARD_PENALTY:
            raise NotImplementedError(
                "vectorized model supports only the CONSTANT_HAZARD_PENALTY reward model"
            )

    def _build_transition_geometry(self, env: PushPOMDP) -> None:
        self._grid_max = float(env.grid_size - 1)
        self._push_threshold_sq = float(env.push_threshold) ** 2
        self._push_scale = 1.0 - float(env.friction_coefficient)
        self._transition_error_prob = float(env.transition_error_prob)
        self._obstacles = self._centres_tensor(env.obstacles)
        self._obstacle_radius_sq = float(env.obstacle_radius) ** 2

    def _build_reward_geometry(self, env: PushPOMDP) -> None:
        self._obstacle_penalty = float(env.obstacle_penalty)
        self._obstacle_hit_prob = float(env.obstacle_hit_probability)
        self._dangerous_areas = self._centres_tensor(env.dangerous_areas)
        self._danger_radius_sq = float(env.dangerous_area_radius) ** 2
        self._danger_penalty = float(env.dangerous_area_penalty)
        self._danger_hit_prob = float(env.dangerous_area_hit_probability)

    def _build_observation_kernel(self, env: PushPOMDP) -> None:
        self._obs_noise = float(env.observation_noise)
        self._obs_variance = self._obs_noise * self._obs_noise
        self._obs_log_norm = -math.log(2.0 * math.pi * self._obs_variance)

    def _centres_tensor(self, centres: object) -> Tensor:
        array = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
        return self._to_tensor(array)

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        # The env clips only after the push distance is measured, so the robot
        # and object positions are resolved unclamped and clamped together last.
        offsets = self._action_table[self._resolve_actions(actions)]
        robot = self._resolve_robot(states[:, 0:2], offsets)
        obj = self._resolve_object(robot, states[:, 2:4], offsets)
        clipped = torch.clamp(torch.cat([robot, obj], dim=1), 0.0, self._grid_max)
        return torch.cat([clipped, states[:, 4:6]], dim=1)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # Observation noise is action-independent for this env.
        noise = torch.randn(next_states.shape[0], 2, dtype=self.dtype, device=self.device)
        obj_obs = torch.clamp(next_states[:, 2:4] + noise * self._obs_noise, 0.0, self._grid_max)
        observations = next_states.clone()
        observations[:, 2:4] = obj_obs
        return observations

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del states, actions  # Reward scores the realised next state only.
        delta = next_states[:, 2:4] - next_states[:, 4:6]
        dist = torch.linalg.vector_norm(delta, dim=1)
        reward = -dist + 100.0 * (dist < 0.5).to(self.dtype)
        robot = next_states[:, 0:2]
        reward = reward + self._obstacle_contribution(robot)
        reward = reward + self._dangerous_area_contribution(robot)
        return reward

    def terminal_mask(self, states: Tensor) -> Tensor:
        delta = states[:, 2:4] - states[:, 4:6]
        return (delta * delta).sum(dim=1) < 0.25

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # Object-position Gaussian likelihood ignores the action.
        diff = observations[:, 2:4] - next_states[:, 2:4]
        sq = (diff * diff).sum(dim=1)
        return self._obs_log_norm - 0.5 * sq / self._obs_variance

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        quantized = torch.floor(observations[:, 2:4] / self._obs_resolution).to(torch.int64)
        return quantized[:, 0] * _HASH_PRIME_X + quantized[:, 1] * _HASH_PRIME_Y

    # ------------------------------------------------------------------ #
    # Internal transition helpers
    # ------------------------------------------------------------------ #

    def _resolve_actions(self, actions: Tensor) -> Tensor:
        indices = actions.to(torch.int64)
        if self._transition_error_prob <= 0.0:
            return indices
        draws = torch.rand(indices.shape[0], dtype=self.dtype, device=self.device)
        errored = draws < self._transition_error_prob
        offset = torch.randint(1, self.num_actions, indices.shape, device=self.device)
        return torch.where(errored, (indices + offset) % self.num_actions, indices)

    def _resolve_robot(self, robot: Tensor, offsets: Tensor) -> Tensor:
        # Unclamped: obstacle blocking keeps the old position, else moves.
        intended = robot + offsets
        blocked = self._in_obstacle(intended).unsqueeze(1)
        return torch.where(blocked, robot, intended)

    def _resolve_object(self, robot: Tensor, obj: Tensor, offsets: Tensor) -> Tensor:
        # ``robot`` is the unclamped new robot position (matches the env's push
        # distance check); the object is clamped by the caller afterwards.
        diff = robot - obj
        pushed_mask = ((diff * diff).sum(dim=1) < self._push_threshold_sq).unsqueeze(1)
        intended = obj + offsets * self._push_scale
        blocked = self._in_obstacle(intended).unsqueeze(1)
        moved = torch.where(blocked, obj, intended)
        return torch.where(pushed_mask, moved, obj)

    def _in_obstacle(self, points: Tensor) -> Tensor:
        return self._within_radius(points, self._obstacles, self._obstacle_radius_sq)

    # ------------------------------------------------------------------ #
    # Internal reward helpers
    # ------------------------------------------------------------------ #

    def _obstacle_contribution(self, robot: Tensor) -> Tensor:
        collide = self._in_obstacle(robot)
        applied = self._apply_hit_probability(collide, self._obstacle_hit_prob)
        return self._obstacle_penalty * applied.to(self.dtype)

    def _dangerous_area_contribution(self, robot: Tensor) -> Tensor:
        in_zone = self._within_radius(robot, self._dangerous_areas, self._danger_radius_sq)
        applied = self._apply_hit_probability(in_zone, self._danger_hit_prob)
        return self._danger_penalty * applied.to(self.dtype)

    def _apply_hit_probability(self, hit: Tensor, probability: float) -> Tensor:
        if probability >= 1.0:
            return hit
        draws = torch.rand(hit.shape[0], dtype=self.dtype, device=self.device)
        return hit & (draws < probability)

    def _within_radius(self, points: Tensor, centres: Tensor, radius_sq: float) -> Tensor:
        if centres.shape[0] == 0:
            return torch.zeros(points.shape[0], dtype=torch.bool, device=self.device)
        diff = points[:, None, :] - centres[None, :, :]
        min_sq = (diff * diff).sum(dim=-1).min(dim=1).values
        return min_sq <= radius_sq
