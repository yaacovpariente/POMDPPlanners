# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for RockSample.

This module provides :class:`RockSampleVectorizedModel`, a fully batched,
GPU-friendly implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp.RockSamplePOMDP`.

It re-expresses the environment's deterministic grid transition, its noisy
long-range Bernoulli sensor observation, and its ``CONSTANT_HAZARD_PENALTY``
reward as torch tensor kernels so a vectorized planner (VOPP) can run tens of
thousands of parallel simulations on the GPU without a host/device sync. Every
constant (grid size, rock positions, sensor efficiency, costs, dangerous-area
geometry) is read from a live environment instance, so the environment stays
the single source of truth for configuration; only the numeric kernels are
duplicated in torch. The accompanying parity test pins these kernels to the
environment's native (C++) implementations.

State layout per particle mirrors the environment:
``[robot_row, robot_col, rock_0_quality, ..., rock_{R-1}_quality]`` with the
terminal sentinel ``[-1, -1, ...]``. Observations are the categorical sensor
codes ``0=none``, ``1=good``, ``2=bad`` carried in a width-1 tensor.

Only one environment configuration is supported: the
``CONSTANT_HAZARD_PENALTY`` reward model with
``is_dangerous_area_hit_terminal=False``. The zero-mean and distance-decayed
hazard variants and the draw-coupled hazard-terminal state slot are not
modeled here; each is checked at construction and any mismatch raises
:class:`NotImplementedError`.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    RewardModelType,
    RockSamplePOMDP,
)

# Observation codes must match the native kernel / vectorized updater.
_OBS_NONE = 0
_OBS_GOOD = 1
_OBS_BAD = 2

# Defensive flooring constants mirroring the native kernel so impossible
# events score the same finite log-likelihood in both implementations.
_PROB_FLOOR = 1e-300
_LOG_PROB_FLOOR = -690.7755278982137  # == log(_PROB_FLOOR)

# Number of non-check actions (sample, north, east, south, west).
_NUM_MOVE_ACTIONS = 5


class RockSampleVectorizedModel:
    """Fully vectorized torch generative model for the RockSample POMDP.

    The model batches the transition, observation, reward, terminal, and
    observation-likelihood kernels over a leading particle dimension and keeps
    every tensor on a single device. Actions are the environment's discrete
    action ids used directly as integer indices (``0=sample``, ``1..4`` moves,
    ``5..4+R`` check-rock), so ``num_actions == 5 + R``; observations are the
    categorical sensor codes ``0/1/2``.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of discrete actions (``5 + num_rocks``).

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
        ...     RockSamplePOMDP,
        ... )
        >>> from POMDPPlanners.environments.rock_sample_pomdp.rocksample_vectorized_model import (
        ...     RockSampleVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> env = RockSamplePOMDP(map_size=(5, 5), rock_positions=[(0, 0), (2, 2), (3, 3)])
        >>> model = RockSampleVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0], [2.0, 2.0, 1.0, 1.0, 0.0]])
        >>> actions = torch.tensor([2, 5])  # move east, check rock 0
        >>> next_states = model.sample_next_states(states, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(observations.shape), tuple(rewards.shape)
        ((2, 5), (2, 1), (2,))
    """

    def __init__(
        self,
        env: RockSamplePOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The environment whose parameters and models are mirrored.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.

        Raises:
            NotImplementedError: If ``env`` uses a reward model other than
                ``CONSTANT_HAZARD_PENALTY`` or enables the draw-coupled
                hazard-terminal state slot.
        """
        self._require_supported_config(env)
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self._num_rocks = len(env.rock_positions)
        self.num_actions = _NUM_MOVE_ACTIONS + self._num_rocks
        self._build_geometry(env)
        self._build_reward(env)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_supported_config(env: RockSamplePOMDP) -> None:
        if env.reward_model_type is not RewardModelType.CONSTANT_HAZARD_PENALTY:
            raise NotImplementedError(
                "vectorized model supports only the CONSTANT_HAZARD_PENALTY reward model"
            )
        if env.is_dangerous_area_hit_terminal:
            raise NotImplementedError(
                "vectorized model requires is_dangerous_area_hit_terminal=False "
                "(the draw-coupled hazard-terminal absorbing slot is not modeled)"
            )

    def _build_geometry(self, env: RockSamplePOMDP) -> None:
        self._map_rows = int(env.map_size[0])
        self._map_cols = int(env.map_size[1])
        rocks = np.asarray(env.rock_positions, dtype=np.float64).reshape(-1, 2)
        self._rock_rows = self._to_tensor(rocks[:, 0])
        self._rock_cols = self._to_tensor(rocks[:, 1])
        self._sensor_efficiency = float(env.sensor_efficiency)
        self._inv_sigma = 1.0 / self._sensor_efficiency

    def _build_reward(self, env: RockSamplePOMDP) -> None:
        self._step_penalty = float(env.step_penalty)
        self._exit_reward = float(env.exit_reward)
        self._good_rock_reward = float(env.good_rock_reward)
        self._bad_rock_penalty = float(env.bad_rock_penalty)
        self._sensor_use_penalty = float(env.sensor_use_penalty)
        dangers = np.asarray(env.dangerous_areas, dtype=np.float64).reshape(-1, 2)
        self._danger_centres = self._to_tensor(dangers)
        self._num_dangers = int(dangers.shape[0])
        self._danger_radius_sq = float(env.dangerous_area_radius) ** 2
        self._danger_penalty = float(env.dangerous_area_penalty)
        self._danger_hit_prob = float(env.dangerous_area_hit_probability)

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        row, col = states[:, 0], states[:, 1]
        new_row = torch.where(actions == 1, torch.clamp(row - 1.0, min=0.0), row)
        new_row = torch.where(
            actions == 3, torch.clamp(row + 1.0, max=float(self._map_rows - 1)), new_row
        )
        east_col = col + 1.0
        new_col = torch.where(actions == 4, torch.clamp(col - 1.0, min=0.0), col)
        new_col = torch.where(actions == 2, east_col, new_col)
        nxt = states.clone()
        nxt[:, 0], nxt[:, 1] = new_row, new_col
        self._apply_sample_flip(nxt, states, actions)
        exit_mask = (actions == 2) & (east_col >= float(self._map_cols))
        neg = torch.full_like(row, -1.0)
        nxt[:, 0] = torch.where(exit_mask, neg, nxt[:, 0])
        nxt[:, 1] = torch.where(exit_mask, neg, nxt[:, 1])
        terminal_in = self._terminal_rows(states)
        return torch.where(terminal_in.unsqueeze(1), states, nxt)

    def _apply_sample_flip(self, nxt: Tensor, states: Tensor, actions: Tensor) -> None:
        row, col = states[:, 0], states[:, 1]
        sample_mask = actions == 0
        for k in range(self._num_rocks):
            hit = sample_mask & (row == self._rock_rows[k]) & (col == self._rock_cols[k])
            nxt[:, 2 + k] = torch.where(hit, torch.zeros_like(nxt[:, 2 + k]), nxt[:, 2 + k])

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        is_check = actions >= _NUM_MOVE_ACTIONS
        efficiency, rock_good = self._sensor_efficiency_and_quality(next_states, actions)
        correct = torch.where(rock_good, float(_OBS_GOOD), float(_OBS_BAD))
        flipped = torch.where(rock_good, float(_OBS_BAD), float(_OBS_GOOD))
        draws = torch.rand(next_states.shape[0], dtype=self.dtype, device=self.device)
        check_obs = torch.where(draws < efficiency, correct, flipped)
        obs = torch.where(is_check, check_obs, torch.full_like(check_obs, float(_OBS_NONE)))
        return obs.unsqueeze(1)

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        reward, is_exit = self._base_reward(states, actions)
        return reward + self._danger_reward(next_states, is_exit)

    def terminal_mask(self, states: Tensor) -> Tensor:
        return self._terminal_rows(states)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        obs = observations[:, 0]
        is_check = actions >= _NUM_MOVE_ACTIONS
        efficiency, rock_good = self._sensor_efficiency_and_quality(next_states, actions)
        prob_good = torch.where(rock_good, efficiency, 1.0 - efficiency)
        prob_bad = torch.where(rock_good, 1.0 - efficiency, efficiency)
        prob = torch.where(obs == _OBS_GOOD, prob_good, prob_bad)
        logp = torch.clamp(torch.log(torch.clamp(prob, min=_PROB_FLOOR)), min=_LOG_PROB_FLOOR)
        terminal = self._terminal_rows(next_states)
        floor = torch.full_like(logp, _LOG_PROB_FLOOR)
        check_result = torch.where((obs == _OBS_NONE) | terminal, floor, logp)
        noncheck_result = torch.where(obs == _OBS_NONE, torch.zeros_like(logp), floor)
        return torch.where(is_check, check_result, noncheck_result)

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        return observations[:, 0].to(torch.int64)

    # ------------------------------------------------------------------ #
    # Internal reward / sensor helpers
    # ------------------------------------------------------------------ #

    def _terminal_rows(self, states: Tensor) -> Tensor:
        return (states[:, 0] < 0.0) & (states[:, 1] < 0.0)

    def _sensor_efficiency_and_quality(
        self, next_states: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor]:
        rock_idx = torch.clamp(actions - _NUM_MOVE_ACTIONS, min=0)
        if self._num_rocks == 0:
            zeros = torch.zeros(next_states.shape[0], dtype=self.dtype, device=self.device)
            return zeros, zeros > 0.5
        rock_r = self._rock_rows[rock_idx]
        rock_c = self._rock_cols[rock_idx]
        dist = torch.sqrt((next_states[:, 0] - rock_r) ** 2 + (next_states[:, 1] - rock_c) ** 2)
        efficiency = torch.exp(-dist * self._inv_sigma)
        quality = next_states.gather(1, (rock_idx + 2).unsqueeze(1)).squeeze(1)
        return efficiency, quality > 0.5

    def _base_reward(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        col = states[:, 1]
        reward = torch.full(
            (states.shape[0],), self._step_penalty, dtype=self.dtype, device=self.device
        )
        reward = reward + self._sample_reward(states, actions)
        reward = reward + (actions >= _NUM_MOVE_ACTIONS).to(self.dtype) * self._sensor_use_penalty
        is_exit = (actions == 2) & (col == float(self._map_cols - 1))
        reward = reward + is_exit.to(self.dtype) * self._exit_reward
        return reward, is_exit

    def _sample_reward(self, states: Tensor, actions: Tensor) -> Tensor:
        row, col = states[:, 0], states[:, 1]
        sample_mask = actions == 0
        reward = torch.zeros(states.shape[0], dtype=self.dtype, device=self.device)
        matched = torch.zeros(states.shape[0], dtype=torch.bool, device=self.device)
        for k in range(self._num_rocks):
            at_rock = (
                sample_mask & ~matched & (row == self._rock_rows[k]) & (col == self._rock_cols[k])
            )
            good = states[:, 2 + k] > 0.5
            delta = good.to(self.dtype) * self._good_rock_reward
            delta = delta + (~good).to(self.dtype) * self._bad_rock_penalty
            reward = reward + at_rock.to(self.dtype) * delta
            matched = matched | at_rock
        return reward

    def _danger_reward(self, next_states: Tensor, is_exit: Tensor) -> Tensor:
        if self._num_dangers == 0:
            return torch.zeros(next_states.shape[0], dtype=self.dtype, device=self.device)
        diff = next_states[:, None, :2] - self._danger_centres[None, :, :]
        min_sq = (diff * diff).sum(dim=-1).min(dim=1).values
        in_zone = (min_sq <= self._danger_radius_sq) & ~is_exit
        draws = torch.rand(next_states.shape[0], dtype=self.dtype, device=self.device)
        hit = in_zone & (draws < self._danger_hit_prob)
        return hit.to(self.dtype) * self._danger_penalty
