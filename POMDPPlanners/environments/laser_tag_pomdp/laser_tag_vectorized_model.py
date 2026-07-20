# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for the discrete LaserTag POMDP.

This module provides :class:`LaserTagVectorizedModel`, a fully batched,
GPU-friendly implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for :class:`~POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp.LaserTagPOMDP`.

It re-expresses the environment's opponent-motion kernel (all of ``EVADE``,
``PURSUE``, and ``EVADE_WHEN_SPOTTED``), its 8-direction laser observation
kernel, and its ``CONSTANT_HAZARD_PENALTY`` reward kernel as torch tensor
operations so a vectorized planner (VOPP) can
run tens of thousands of parallel simulations on the GPU without a
host/device sync. Every constant (grid geometry, walls, dangerous areas,
measurement noise, costs, action directions) is read from a live environment
instance, so the environment stays the single source of truth for
configuration; only the numeric kernels are duplicated in torch. The
accompanying parity test pins these kernels to the environment's native
scalar kernels.

Only the ``CONSTANT_HAZARD_PENALTY`` reward model, deterministic robot motion
(``transition_error_prob == 0.0``), and ``is_dangerous_area_hit_terminal=False``
are supported; all three opponent policies (``EVADE``, ``PURSUE``,
``EVADE_WHEN_SPOTTED``) are modeled. Any other configuration is checked at
construction and raises :class:`NotImplementedError`.
"""

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import (
    LaserTagPOMDP,
    RewardModelType,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils import (
    OpponentPolicy,
)

# 5 discrete actions: North, South, East, West, Tag (matches LaserTagPOMDP).
_ACTION_DIRECTIONS = np.array([[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]], dtype=np.float64)
# 8 laser directions: N, NE, E, SE, S, SW, W, NW (matches LaserTagPOMDP).
_LASER_DIRECTIONS = np.array(
    [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]],
    dtype=np.float64,
)
# Opponent candidate cells in categorical order: stay, north, south, east, west.
_OPPONENT_OFFSETS = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=np.float64)
# Distinct primes for hashing quantized 8-D observations into integer keys.
_HASH_PRIMES = np.array(
    [73856093, 19349663, 83492791, 39916801, 51539607, 15485863, 32452843, 49979687],
    dtype=np.int64,
)
_TAG_ACTION = 4


class LaserTagVectorizedModel:
    """Fully vectorized torch generative model for the discrete LaserTag POMDP.

    The model batches the transition, observation, reward, terminal, and
    observation-likelihood kernels over a leading particle dimension and keeps
    every tensor on a single device. States are ``[robot_row, robot_col,
    opp_row, opp_col, terminal]`` rows; actions are the integer indices
    ``0..4`` of the environment's discrete action set; observations are the
    8-direction laser ranges (with an all ``-1`` sentinel for terminal states).

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of discrete actions (always ``5``).

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import (
        ...     LaserTagPOMDP,
        ... )
        >>> from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_vectorized_model import (
        ...     LaserTagVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> env = LaserTagPOMDP(discount_factor=0.95)
        >>> model = LaserTagVectorizedModel(env, device=torch.device("cpu"))
        >>> states = torch.tensor([[0.0, 0.0, 6.0, 5.0, 0.0], [2.0, 3.0, 2.0, 3.0, 0.0]])
        >>> actions = torch.tensor([2, 4])  # move east, tag
        >>> next_states = model.sample_next_states(states, actions)
        >>> observations = model.sample_observations(next_states, actions)
        >>> rewards = model.rewards(states, actions, next_states)
        >>> tuple(next_states.shape), tuple(observations.shape), tuple(rewards.shape)
        ((2, 5), (2, 8), (2,))
    """

    def __init__(
        self,
        env: LaserTagPOMDP,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_resolution: float = 0.1,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The environment whose parameters and model types are mirrored.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_resolution: Grid spacing used to quantize continuous
                laser observations into integer tree keys.

        Raises:
            NotImplementedError: If ``env`` uses a reward model,
                transition-error, or hazard-terminal configuration the torch
                kernels do not model.
            ValueError: If ``observation_resolution`` is not positive.
        """
        self._require_supported_config(env)
        if observation_resolution <= 0.0:
            raise ValueError("observation_resolution must be positive")
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self._obs_resolution = float(observation_resolution)
        self.num_actions = int(len(env.get_actions()))
        self._opponent_policy = env.opponent_policy
        self._build_geometry(env)
        self._build_noise_and_reward(env)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_supported_config(env: LaserTagPOMDP) -> None:
        if env.reward_model_type is not RewardModelType.CONSTANT_HAZARD_PENALTY:
            raise NotImplementedError(
                "vectorized model supports only the CONSTANT_HAZARD_PENALTY reward model"
            )
        if float(env.transition_error_prob) != 0.0:
            raise NotImplementedError(
                "vectorized model supports only deterministic robot motion "
                "(transition_error_prob == 0.0)"
            )
        if env.is_dangerous_area_hit_terminal:
            raise NotImplementedError(
                "vectorized model requires is_dangerous_area_hit_terminal=False "
                "(the draw-coupled hazard-terminal absorbing slot is not modeled)"
            )

    def _build_geometry(self, env: LaserTagPOMDP) -> None:
        self._rows, self._cols = int(env.floor_shape[0]), int(env.floor_shape[1])
        self._max_ray = max(self._rows, self._cols)
        wall_grid = np.zeros((self._rows, self._cols), dtype=bool)
        for wall_row, wall_col in env.walls:
            if 0 <= wall_row < self._rows and 0 <= wall_col < self._cols:
                wall_grid[wall_row, wall_col] = True
        self._wall_grid = torch.as_tensor(wall_grid, device=self.device)
        self._action_dirs = self._to_tensor(_ACTION_DIRECTIONS)
        self._laser_dirs = self._to_tensor(_LASER_DIRECTIONS)
        self._opp_offsets = self._to_tensor(_OPPONENT_OFFSETS)
        self._hash_primes = torch.as_tensor(_HASH_PRIMES, device=self.device)

    def _build_noise_and_reward(self, env: LaserTagPOMDP) -> None:
        self._sigma = float(env.measurement_noise)
        self._variance = self._sigma * self._sigma
        self._log_norm_1d = -0.5 * float(np.log(2.0 * np.pi * self._variance))
        self._tag_reward = float(env.tag_reward)
        self._tag_penalty = float(env.tag_penalty)
        self._step_cost = float(env.step_cost)
        self._area_penalty = float(env.dangerous_area_penalty)
        self._danger_radius_sq = float(env.dangerous_area_radius) ** 2
        if env.dangerous_areas:
            centers = np.asarray(env.dangerous_areas, dtype=np.float64).reshape(-1, 2)
        else:
            centers = np.empty((0, 2), dtype=np.float64)
        self._danger_centers = self._to_tensor(centers)

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        robot, opp = states[:, 0:2], states[:, 2:4]
        robot_next = self._robot_next(robot, actions)
        reference = self._opponent_robot_reference(robot, robot_next)
        probs = self._opponent_move_probs(reference, opp)
        choice = torch.multinomial(probs, 1).squeeze(1)
        opp_next = opp + self._opp_offsets[choice]
        tag_success = (actions == _TAG_ACTION) & self._same_cell(robot, opp)
        opp_next = torch.where(tag_success[:, None], opp, opp_next)
        terminal = tag_success.to(self.dtype)[:, None]
        return torch.cat([robot_next, opp_next, terminal], dim=1)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # observations do not depend on the action in LaserTag.
        true_dist = self._laser_distances(next_states[:, 0:2], next_states[:, 2:4])
        noise = torch.randn(true_dist.shape, dtype=self.dtype, device=self.device)
        observations = torch.clamp(true_dist + noise * self._sigma, min=0.0)
        sentinel = torch.full_like(observations, -1.0)
        terminal = next_states[:, 4] > 0.5
        return torch.where(terminal[:, None], sentinel, observations)

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        robot, opp = states[:, 0:2], states[:, 2:4]
        tag_base = torch.where(
            self._same_cell(robot, opp),
            torch.tensor(self._tag_reward, dtype=self.dtype, device=self.device),
            torch.tensor(-self._tag_penalty, dtype=self.dtype, device=self.device),
        )
        base = torch.where(
            actions == _TAG_ACTION, tag_base, -self._step_cost * torch.ones_like(tag_base)
        )
        realised = next_states[:, 0:2]
        penalised = self._is_wall_cell(realised) | self._in_danger(realised)
        base = base - penalised.to(self.dtype) * self._area_penalty
        state_terminal = states[:, 4] > 0.5
        return torch.where(state_terminal, torch.zeros_like(base), base)

    def terminal_mask(self, states: Tensor) -> Tensor:
        return states[:, 4] > 0.5

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # likelihood does not depend on the action in LaserTag.
        true_dist = self._laser_distances(next_states[:, 0:2], next_states[:, 2:4])
        diff = observations - true_dist
        gaussian = 8.0 * self._log_norm_1d - 0.5 * (diff * diff).sum(dim=1) / self._variance
        neg_inf = torch.full_like(gaussian, float("-inf"))
        non_terminal = torch.where((observations < 0.0).any(dim=1), neg_inf, gaussian)
        is_sentinel = (observations == -1.0).all(dim=1)
        terminal_logp = torch.where(is_sentinel, torch.zeros_like(gaussian), neg_inf)
        return torch.where(next_states[:, 4] > 0.5, terminal_logp, non_terminal)

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        quantized = torch.floor(observations / self._obs_resolution).to(torch.int64)
        return (quantized * self._hash_primes).sum(dim=1)

    # ------------------------------------------------------------------ #
    # Internal transition / geometry helpers
    # ------------------------------------------------------------------ #

    def _robot_next(self, robot: Tensor, actions: Tensor) -> Tensor:
        candidate = robot + self._action_dirs[actions]
        valid = self._cell_valid(candidate)
        return torch.where(valid[:, None], candidate, robot)

    def _opponent_robot_reference(self, robot: Tensor, robot_next: Tensor) -> Tensor:
        # PURSUE conditions the opponent move on the robot's post-move cell;
        # EVADE and EVADE_WHEN_SPOTTED on its pre-move cell. Mirrors the scalar
        # LaserTagPOMDP._opponent_robot_reference.
        if self._opponent_policy is OpponentPolicy.PURSUE:
            return robot_next
        return robot

    def _opponent_move_probs(self, robot: Tensor, opp: Tensor) -> Tensor:
        if self._opponent_policy is OpponentPolicy.EVADE_WHEN_SPOTTED:
            flee = self._directional_move_probs(robot, opp)
            wander = self._random_move_probs(opp)
            spotted = self._opponent_spotted(robot, opp)
            return torch.where(spotted[:, None], flee, wander)
        return self._directional_move_probs(robot, opp)

    def _directional_move_probs(self, robot: Tensor, opp: Tensor) -> Tensor:
        # EVADE puts the 0.4 directional mass on the distance-increasing cell,
        # PURSUE on the distance-decreasing cell; the flags flip accordingly.
        robot_r, robot_c = robot[:, 0], robot[:, 1]
        opp_r, opp_c = opp[:, 0], opp[:, 1]
        if self._opponent_policy is OpponentPolicy.PURSUE:
            north_hot, south_hot = robot_r < opp_r, robot_r > opp_r
            east_hot, west_hot = robot_c > opp_c, robot_c < opp_c
        else:
            north_hot, south_hot = robot_r > opp_r, robot_r < opp_r
            east_hot, west_hot = robot_c < opp_c, robot_c > opp_c
        row_aligned, col_aligned = robot_r == opp_r, robot_c == opp_c
        north = self._directional_prob(north_hot, row_aligned)
        south = self._directional_prob(south_hot, row_aligned)
        east = self._directional_prob(east_hot, col_aligned)
        west = self._directional_prob(west_hot, col_aligned)
        return self._assemble_move_probs(opp, north, south, east, west)

    def _random_move_probs(self, opp: Tensor) -> Tensor:
        # EVADE_WHEN_SPOTTED's unspotted branch: uniform 0.2 on each cardinal
        # neighbour, with invalid neighbours falling through to stay.
        weight = torch.full((opp.shape[0],), 0.2, dtype=self.dtype, device=self.device)
        return self._assemble_move_probs(opp, weight, weight, weight, weight)

    def _assemble_move_probs(
        self, opp: Tensor, north: Tensor, south: Tensor, east: Tensor, west: Tensor
    ) -> Tensor:
        # Zero out moves into invalid cells (their mass falls through to stay,
        # matching the scalar slack redistribution) and derive the stay mass.
        north = north * self._cell_valid(opp + self._opp_offsets[1]).to(self.dtype)
        south = south * self._cell_valid(opp + self._opp_offsets[2]).to(self.dtype)
        east = east * self._cell_valid(opp + self._opp_offsets[3]).to(self.dtype)
        west = west * self._cell_valid(opp + self._opp_offsets[4]).to(self.dtype)
        stay = 1.0 - (north + south + east + west)
        return torch.stack([stay, north, south, east, west], dim=1)

    def _directional_prob(self, hot: Tensor, aligned: Tensor) -> Tensor:
        # 0.4 on the hot (toward/away) cell, a symmetric 0.2 when the robot and
        # opponent share the axis coordinate, otherwise 0.0.
        return torch.where(
            hot,
            torch.full_like(hot, 0.4, dtype=self.dtype),
            torch.where(
                aligned,
                torch.full_like(aligned, 0.2, dtype=self.dtype),
                torch.zeros(hot.shape, dtype=self.dtype, device=self.device),
            ),
        )

    def _opponent_spotted(self, robot: Tensor, opp: Tensor) -> Tensor:
        # Batched mirror of LaserTagPOMDP._is_opponent_spotted: True iff the
        # opponent lies on one of the robot's 8 laser rays, unoccluded by a wall
        # or the grid boundary. Opponent detection takes priority over a wall in
        # the same cell (they can never coincide) exactly as the scalar ray walk.
        batch = robot.shape[0]
        pos = robot[:, None, :].expand(batch, 8, 2).clone()
        opp_cell = opp[:, None, :]
        active = torch.ones(batch, 8, dtype=torch.bool, device=self.device)
        spotted = torch.zeros(batch, 8, dtype=torch.bool, device=self.device)
        for _ in range(self._max_ray):
            pos = pos + self._laser_dirs
            in_bounds = self._in_bounds(pos)
            is_wall = self._wall_at(pos) & in_bounds
            is_opp = (pos == opp_cell).all(dim=2)
            spotted = spotted | (active & in_bounds & is_opp)
            active = active & ~(~in_bounds | is_wall | is_opp)
        return spotted.any(dim=1)

    def _laser_distances(self, robot: Tensor, opp: Tensor) -> Tensor:
        batch = robot.shape[0]
        pos = robot[:, None, :].expand(batch, 8, 2).clone()
        opp_cell = opp[:, None, :]
        count = torch.zeros(batch, 8, dtype=self.dtype, device=self.device)
        active = torch.ones(batch, 8, dtype=torch.bool, device=self.device)
        for _ in range(self._max_ray):
            pos = pos + self._laser_dirs
            in_bounds = self._in_bounds(pos)
            is_wall = self._wall_at(pos) & in_bounds
            is_opp = (pos == opp_cell).all(dim=2)
            free = active & in_bounds & ~is_wall & ~is_opp
            count = count + free.to(self.dtype)
            active = active & ~(~in_bounds | is_wall | is_opp)
        return count

    def _cell_valid(self, cells: Tensor) -> Tensor:
        in_bounds = self._in_bounds(cells)
        return in_bounds & ~(self._wall_at(cells) & in_bounds)

    def _is_wall_cell(self, cells: Tensor) -> Tensor:
        return self._in_bounds(cells) & self._wall_at(cells)

    def _in_bounds(self, cells: Tensor) -> Tensor:
        rows = cells[..., 0]
        cols = cells[..., 1]
        return (rows >= 0) & (rows < self._rows) & (cols >= 0) & (cols < self._cols)

    def _wall_at(self, cells: Tensor) -> Tensor:
        rows = cells[..., 0].round().long().clamp(0, self._rows - 1)
        cols = cells[..., 1].round().long().clamp(0, self._cols - 1)
        return self._wall_grid[rows, cols]

    def _in_danger(self, points: Tensor) -> Tensor:
        if self._danger_centers.shape[0] == 0:
            return torch.zeros(points.shape[0], dtype=torch.bool, device=self.device)
        diff = points[:, None, :] - self._danger_centers[None, :, :]
        min_sq = (diff * diff).sum(dim=2).min(dim=1).values
        return min_sq <= self._danger_radius_sq

    @staticmethod
    def _same_cell(first: Tensor, second: Tensor) -> Tensor:
        return (first == second).all(dim=1)
