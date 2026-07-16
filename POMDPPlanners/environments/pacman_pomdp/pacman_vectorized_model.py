# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for the PacMan POMDP.

This module provides :class:`PacManVectorizedModel`, a fully batched,
GPU-friendly implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.pacman_pomdp.pacman_pomdp.PacManPOMDP`.

It re-expresses the environment's native C++ transition, observation, and
reward kernels as torch tensor operations so a vectorized planner (VOPP) can
run tens of thousands of parallel simulations on the GPU without a host/device
sync. Every constant (maze neighbor table, pellet positions, ghost
aggressiveness, observation-noise parameters, reward scalars, hazard geometry)
is read from a live environment instance, so the environment stays the single
source of truth for configuration; only the numeric kernels are duplicated in
torch. The accompanying parity test pins these kernels to the environment's
native implementations.

Only the standard PacMan configuration is supported: ``independent`` ghost
coordination, all-``aggressive`` ghost strategies, the
``CONSTANT_HAZARD_PENALTY`` reward model, and
``is_dangerous_area_hit_terminal=False``. The patrol/ambush strategies and the
coordinated/mixed modes carry hidden or non-per-particle state (a ghost patrol
direction lives on the environment, not the particle), and the draw-coupled
hazard-terminal path couples an absorbing slot to the reward draw; none is
modeled here. All four conditions are checked at construction and any mismatch
raises :class:`NotImplementedError`.
"""

import math
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP

# log(1e-300); the native ``batch_log_likelihood`` floors impossible-event
# log-likelihoods to this value, so the torch kernel matches it exactly.
_LOG_PROB_FLOOR = -690.7755278982137
# Odd multiplier for the rolling polynomial hash that turns a discrete
# observation (small non-negative integer coordinates) into one int64 key.
_OBS_HASH_MULTIPLIER = 2654435761


class PacManVectorizedModel:
    """Fully vectorized torch generative model for the PacMan POMDP.

    The model batches the transition, observation, reward, terminal, and
    observation-likelihood kernels over a leading particle dimension and keeps
    every tensor on a single device. Actions are the five discrete moves
    (north, east, south, west, stay) indexed ``0..4``; observations are the
    per-ghost noisy integer grid positions of the native observation model.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of discrete actions (always 5 for PacMan).

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import (
        ...     create_simple_maze_pacman,
        ... )
        >>> from POMDPPlanners.environments.pacman_pomdp.pacman_vectorized_model import (
        ...     PacManVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> env = create_simple_maze_pacman()
        >>> model = PacManVectorizedModel(env, device=torch.device("cpu"))
        >>> state = torch.as_tensor(env.initial_state_dist().sample()[0]).unsqueeze(0)
        >>> actions = torch.tensor([1])  # east
        >>> next_states = model.sample_next_states(state, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(state, actions, next_states)
        >>> tuple(next_states.shape[1:]), tuple(observations.shape), tuple(rewards.shape)
        ((10,), (1, 2), (1,))
    """

    def __init__(
        self,
        env: "PacManPOMDP",
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The environment whose parameters and kernels are mirrored.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.

        Raises:
            NotImplementedError: If ``env`` uses ghost coordination other than
                ``independent``, any non-``aggressive`` ghost strategy, a reward
                model other than ``CONSTANT_HAZARD_PENALTY``, or the draw-coupled
                ``is_dangerous_area_hit_terminal`` path.
        """
        self._require_supported_config(env)
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self.num_actions = 5
        self._build_layout(env)
        self._build_maze_tables()
        self._build_scalars(env)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_supported_config(env: "PacManPOMDP") -> None:
        from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import (  # pylint: disable=import-outside-toplevel
            RewardModelType,
        )

        if env.ghost_coordination != "independent":
            raise NotImplementedError(
                "vectorized model supports only 'independent' ghost coordination"
            )
        if any(strategy != "aggressive" for strategy in env.ghost_strategies):
            raise NotImplementedError(
                "vectorized model supports only the 'aggressive' ghost strategy"
            )
        if env.reward_model_type is not RewardModelType.CONSTANT_HAZARD_PENALTY:
            raise NotImplementedError(
                "vectorized model supports only the CONSTANT_HAZARD_PENALTY reward model"
            )
        if env.is_dangerous_area_hit_terminal:
            raise NotImplementedError(
                "vectorized model requires is_dangerous_area_hit_terminal=False "
                "(the draw-coupled hazard-terminal absorbing slot is not modeled)"
            )

    def _build_layout(self, env: "PacManPOMDP") -> None:
        trans = env.get_transition_cpp_ctor_kwargs()
        self._num_ghosts = int(trans["num_ghosts"])
        self._idx_pac_row = int(trans["idx_pac_row"])
        self._idx_pac_col = int(trans["idx_pac_col"])
        self._idx_ghosts_start = int(trans["idx_ghosts_start"])
        self._idx_ghosts_end = self._idx_ghosts_start + 2 * self._num_ghosts
        self._idx_pellets_start = int(trans["idx_pellets_start"])
        self._idx_pellets_end = int(trans["idx_pellets_end"])
        self._idx_score = int(trans["idx_score"])
        self._idx_terminal = int(trans["idx_terminal"])
        self._num_pellets = self._idx_pellets_end - self._idx_pellets_start
        self._trans_kwargs = trans

    def _build_maze_tables(self) -> None:
        trans = self._trans_kwargs
        self._maze_rows = int(trans["maze_rows"])
        self._maze_cols = int(trans["maze_cols"])
        self._neighbor_table = torch.as_tensor(
            np.asarray(trans["neighbor_table"], dtype=np.int64), device=self.device
        )
        self._neighbor_validity = torch.as_tensor(
            np.asarray(trans["neighbor_validity"], dtype=np.bool_), device=self.device
        )
        pellets = np.asarray(trans["pellet_positions"], dtype=np.int64).reshape(-1, 2)
        self._pellet_positions = torch.as_tensor(pellets, device=self.device)

    def _build_scalars(self, env: "PacManPOMDP") -> None:
        obs = env.get_observation_cpp_ctor_kwargs()
        self._ghost_aggressiveness = float(self._trans_kwargs["ghost_aggressiveness"])
        self._pellet_reward = float(self._trans_kwargs["pellet_reward"])
        self._obs_noise_factor = float(obs["observation_noise_factor"])
        self._max_obs_noise = float(obs["max_observation_noise"])
        self._step_penalty = float(env.step_penalty)
        self._collision_penalty = float(env.ghost_collision_penalty)
        self._win_reward = float(env.win_reward)
        self._danger_penalty = float(env.dangerous_area_penalty)
        self._danger_radius_sq = float(env.dangerous_area_radius) ** 2
        danger = np.asarray(self._trans_kwargs["dangerous_areas"], dtype=np.float64).reshape(-1, 2)
        self._dangerous_areas = torch.as_tensor(danger, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        out = states.clone()
        active = states[:, self._idx_terminal] <= 0.5
        old_pac = states[:, self._idx_pac_row : self._idx_pac_col + 1].round().long()
        new_pac = self._move_pacman(old_pac, actions)
        old_ghosts = states[:, self._idx_ghosts_start : self._idx_ghosts_end].round().long()
        new_ghosts = self._move_ghosts(old_ghosts, old_pac)
        collided = self._transition_collision(old_pac, new_pac, old_ghosts, new_ghosts)
        new_pellets, collected = self._collect_pellets(states, new_pac, collided)
        new_score = states[:, self._idx_score] + collected.to(self.dtype) * self._pellet_reward
        won = (~collided) & (~self._any_pellet_active(new_pellets))
        new_terminal = (collided | won).to(self.dtype)
        self._write_active(out, active, new_pac, new_ghosts, new_pellets, new_score, new_terminal)
        return out

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # Observations do not depend on the action.
        num_rows = next_states.shape[0]
        pac = next_states[:, self._idx_pac_row : self._idx_pac_col + 1]
        ghosts = next_states[:, self._idx_ghosts_start : self._idx_ghosts_end].view(
            num_rows, self._num_ghosts, 2
        )
        std = self._observation_std(ghosts, pac)
        noise = torch.randn(num_rows, self._num_ghosts, 2, dtype=self.dtype, device=self.device)
        obs = torch.round(ghosts + noise * std.unsqueeze(-1))
        obs[..., 0] = torch.clamp(obs[..., 0], 0, self._maze_rows - 1)
        obs[..., 1] = torch.clamp(obs[..., 1], 0, self._maze_cols - 1)
        terminal = next_states[:, self._idx_terminal] > 0.5
        obs[terminal] = -1.0
        return obs.reshape(num_rows, 2 * self._num_ghosts)

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del actions  # Reward is action-independent given (state, next_state).
        num_rows = states.shape[0]
        reward = torch.full((num_rows,), self._step_penalty, dtype=self.dtype, device=self.device)
        reward = reward + self._reward_collision(next_states) * self._collision_penalty
        got_pellet = next_states[:, self._idx_score] > states[:, self._idx_score]
        reward = reward + got_pellet.to(self.dtype) * self._pellet_reward
        won = (next_states[:, self._idx_terminal] > 0.5) & (
            ~self._any_pellet_active(
                next_states[:, self._idx_pellets_start : self._idx_pellets_end]
            )
        )
        reward = reward + won.to(self.dtype) * self._win_reward
        reward = reward - self._danger_contribution(next_states) * self._danger_penalty
        active = states[:, self._idx_terminal] <= 0.5
        return torch.where(active, reward, torch.zeros_like(reward))

    def terminal_mask(self, states: Tensor) -> Tensor:
        return states[:, self._idx_terminal] > 0.5

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # Observation likelihood is action-independent.
        num_rows = next_states.shape[0]
        state_terminal = next_states[:, self._idx_terminal] > 0.5
        obs = observations.view(num_rows, self._num_ghosts, 2)
        obs_terminal_ghost = (obs < -0.5).all(dim=-1)
        obs_all_terminal = obs_terminal_ghost.all(dim=1)
        per_ghost = self._per_ghost_log_prob(next_states, obs)
        per_ghost = torch.where(obs_terminal_ghost, self._neg_inf(obs_terminal_ghost), per_ghost)
        total = per_ghost.sum(dim=1)
        neg_inf = torch.full_like(total, float("-inf"))
        zeros = torch.zeros_like(total)
        terminal_branch = torch.where(obs_all_terminal, zeros, neg_inf)
        non_terminal_branch = torch.where(obs_all_terminal, neg_inf, total)
        result = torch.where(state_terminal, terminal_branch, non_terminal_branch)
        return torch.clamp(result, min=_LOG_PROB_FLOOR)

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        coords = observations.round().to(torch.int64) + 1  # shift -1 sentinel to 0
        key = torch.zeros(observations.shape[0], dtype=torch.int64, device=observations.device)
        for column in range(coords.shape[1]):
            key = key * _OBS_HASH_MULTIPLIER + coords[:, column]
        return key

    # ------------------------------------------------------------------ #
    # Transition internals
    # ------------------------------------------------------------------ #

    def _move_pacman(self, pac: Tensor, actions: Tensor) -> Tensor:
        return self._neighbor_table[pac[:, 0], pac[:, 1], actions.to(torch.int64)]

    def _move_ghosts(self, ghosts: Tensor, pac: Tensor) -> Tensor:
        num_rows = ghosts.shape[0]
        moved = torch.empty(num_rows, 2 * self._num_ghosts, dtype=self.dtype, device=self.device)
        for ghost in range(self._num_ghosts):
            gpos = ghosts[:, 2 * ghost : 2 * ghost + 2]
            neighbors = self._neighbor_table[gpos[:, 0], gpos[:, 1]]  # [N, 5, 2]
            validity = self._neighbor_validity[gpos[:, 0], gpos[:, 1]]  # [N, 5]
            dist = (neighbors - pac.unsqueeze(1)).abs().sum(dim=-1).to(self.dtype)
            scores = -dist / self._ghost_aggressiveness
            scores = torch.where(validity, scores, torch.full_like(scores, float("-inf")))
            probs = torch.softmax(scores, dim=1)
            choice = torch.multinomial(probs, num_samples=1).squeeze(1)
            picked = neighbors[torch.arange(num_rows, device=self.device), choice]
            moved[:, 2 * ghost : 2 * ghost + 2] = picked.to(self.dtype)
        return moved

    def _transition_collision(
        self, old_pac: Tensor, new_pac: Tensor, old_ghosts: Tensor, new_ghosts: Tensor
    ) -> Tensor:
        num_rows = new_pac.shape[0]
        new_pac_f = new_pac.to(self.dtype)
        collided = torch.zeros(num_rows, dtype=torch.bool, device=self.device)
        old_pac_f = old_pac.to(self.dtype)
        for ghost in range(self._num_ghosts):
            new_g = new_ghosts[:, 2 * ghost : 2 * ghost + 2]
            old_g = old_ghosts[:, 2 * ghost : 2 * ghost + 2].to(self.dtype)
            same_cell = (new_g == new_pac_f).all(dim=1)
            swap = (old_g == new_pac_f).all(dim=1) & (new_g == old_pac_f).all(dim=1)
            collided = collided | same_cell | swap
        return collided

    def _collect_pellets(
        self, states: Tensor, new_pac: Tensor, collided: Tensor
    ) -> Tuple[Tensor, Tensor]:
        block = states[:, self._idx_pellets_start : self._idx_pellets_end]
        if self._num_pellets == 0:
            return block, torch.zeros(states.shape[0], dtype=torch.bool, device=self.device)
        new_pac_f = new_pac.to(self.dtype)
        pellets = self._pellet_positions.to(self.dtype)
        match = (pellets.unsqueeze(0) == new_pac_f.unsqueeze(1)).all(dim=-1)  # [N, P]
        can_collect = match & (block > 0.5) & (~collided).unsqueeze(1)
        new_block = torch.where(can_collect, torch.zeros_like(block), block)
        return new_block, can_collect.any(dim=1)

    def _any_pellet_active(self, block: Tensor) -> Tensor:
        if self._num_pellets == 0:
            return torch.zeros(block.shape[0], dtype=torch.bool, device=self.device)
        return (block > 0.5).any(dim=1)

    def _write_active(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        out: Tensor,
        active: Tensor,
        new_pac: Tensor,
        new_ghosts: Tensor,
        new_pellets: Tensor,
        new_score: Tensor,
        new_terminal: Tensor,
    ) -> None:
        active_col = active.unsqueeze(1)
        pac_block = out[:, self._idx_pac_row : self._idx_pac_col + 1]
        out[:, self._idx_pac_row : self._idx_pac_col + 1] = torch.where(
            active_col, new_pac.to(self.dtype), pac_block
        )
        ghost_block = out[:, self._idx_ghosts_start : self._idx_ghosts_end]
        out[:, self._idx_ghosts_start : self._idx_ghosts_end] = torch.where(
            active_col, new_ghosts, ghost_block
        )
        pellet_block = out[:, self._idx_pellets_start : self._idx_pellets_end]
        out[:, self._idx_pellets_start : self._idx_pellets_end] = torch.where(
            active_col, new_pellets, pellet_block
        )
        out[:, self._idx_score] = torch.where(active, new_score, out[:, self._idx_score])
        out[:, self._idx_terminal] = torch.where(active, new_terminal, out[:, self._idx_terminal])

    # ------------------------------------------------------------------ #
    # Observation / reward internals
    # ------------------------------------------------------------------ #

    def _observation_std(self, ghosts: Tensor, pac: Tensor) -> Tensor:
        dist = (ghosts - pac.unsqueeze(1)).abs().sum(dim=-1).to(self.dtype)
        std = dist * self._obs_noise_factor
        std = torch.clamp(std, max=self._max_obs_noise)
        return torch.clamp(std, min=1e-6)

    def _reward_collision(self, next_states: Tensor) -> Tensor:
        num_rows = next_states.shape[0]
        pac = next_states[:, self._idx_pac_row : self._idx_pac_col + 1]
        ghosts = next_states[:, self._idx_ghosts_start : self._idx_ghosts_end].view(
            num_rows, self._num_ghosts, 2
        )
        same_cell = (ghosts == pac.unsqueeze(1)).all(dim=-1)
        return same_cell.any(dim=1).to(self.dtype)

    def _danger_contribution(self, next_states: Tensor) -> Tensor:
        num_rows = next_states.shape[0]
        if self._dangerous_areas.shape[0] == 0:
            return torch.zeros(num_rows, dtype=self.dtype, device=self.device)
        pac = next_states[:, self._idx_pac_row : self._idx_pac_col + 1]
        diff = pac.unsqueeze(1) - self._dangerous_areas.unsqueeze(0)
        min_sq = (diff * diff).sum(dim=-1).min(dim=1).values
        return (min_sq <= self._danger_radius_sq).to(self.dtype)

    def _per_ghost_log_prob(self, next_states: Tensor, obs: Tensor) -> Tensor:
        num_rows = next_states.shape[0]
        pac = next_states[:, self._idx_pac_row : self._idx_pac_col + 1]
        ghosts = next_states[:, self._idx_ghosts_start : self._idx_ghosts_end].view(
            num_rows, self._num_ghosts, 2
        )
        std = self._observation_std(ghosts, pac)
        log_row = self._bin_log_prob(obs[..., 0], ghosts[..., 0], std, self._maze_rows - 1)
        log_col = self._bin_log_prob(obs[..., 1], ghosts[..., 1], std, self._maze_cols - 1)
        return log_row + log_col

    def _bin_log_prob(self, obs: Tensor, mean: Tensor, std: Tensor, max_coord: int) -> Tensor:
        coord = obs.to(torch.int64)
        coord_f = coord.to(self.dtype)
        inv = 1.0 / (std * math.sqrt(2.0))

        def cdf(value: Tensor) -> Tensor:
            return 0.5 * (1.0 + torch.erf((value - mean) * inv))

        interior = cdf(coord_f + 0.5) - cdf(coord_f - 0.5)
        low = cdf(torch.full_like(coord_f, 0.5))
        high = 1.0 - cdf(torch.full_like(coord_f, max_coord - 0.5))
        mass = torch.where(coord == 0, low, torch.where(coord == max_coord, high, interior))
        log_prob = torch.log(mass)
        in_range = (coord >= 0) & (coord <= max_coord) & (mass > 0.0)
        return torch.where(in_range, log_prob, self._neg_inf(log_prob))

    def _neg_inf(self, template: Tensor) -> Tensor:
        return torch.full(template.shape, float("-inf"), dtype=self.dtype, device=self.device)
