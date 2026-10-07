# SPDX-License-Identifier: MIT

"""Torch, on-device vectorized generative model for Continuous Push.

This module provides :class:`ContinuousPushVectorizedModel`, a batched
implementation of
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
for
:class:`~POMDPPlanners.environments.push_pomdp.continuous_push_pomdp.ContinuousPushPOMDPDiscreteActions`.

It re-expresses the environment's native C++ kernels in torch:

* the transition -- a Gaussian robot step pushed out of the obstacle AABBs
  (one obstacle at a time, in obstacle order) and clamped to the grid, then a
  friction-scaled push of the object when the moved robot is within
  ``push_threshold`` of it, blocked by obstacles and clamped to the grid; with
  a hazard-terminal flag on, the draw that sets the terminal slot;
* the observation -- the next state with Gaussian noise on the object
  position, clamped to the grid, and its likelihood on that slice alone;
* the reward the scalar ``reward(state, action, next_state)`` returns, for all
  three reward models.

Every constant is read from a live environment instance, so the environment
stays the single source of truth for configuration. The registry conformance
test compares each kernel with the scalar environment.

The discrete-action variant is required: VOPP plans over a fixed, finite action
set, and its four moves are the model's action indices ``0..3`` in
``env.get_actions()`` order. Every configuration of that variant is modeled:
the state width follows the environment (six columns, or seven when a
hazard-terminal flag adds the terminal slot), so no configuration needs a
shape the protocol cannot carry.
"""

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
    ContinuousPushPOMDPDiscreteActions,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp_utils.push_reward_models import (
    RewardModelType,
)

# Spatial-hash primes for turning the quantized 2-D object observation into a key.
_HASH_PRIME_X = 73856093
_HASH_PRIME_Y = 19349663
# The object is on the target when its squared distance is below 0.5 ** 2.
_GOAL_RADIUS = 0.5
_GOAL_REWARD = 100.0


class ContinuousPushVectorizedModel:
    """Batched torch generative model for the discrete-action Continuous Push POMDP.

    States and observations are ``[robot_x, robot_y, object_x, object_y,
    target_x, target_y]`` rows, with a seventh terminal-slot column when the
    environment has a hazard-terminal flag on. Actions are the integer indices
    of ``env.get_actions()`` (``up``, ``down``, ``right``, ``left``).
    Observation keys hash the quantized object position, the only part of the
    observation the likelihood scores.

    Attributes:
        device: Device every tensor argument and return value lives on.
        dtype: Floating dtype used for state / observation / reward tensors.
        num_actions: Number of discrete actions (four).

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
        ...     ContinuousPushPOMDPDiscreteActions,
        ... )
        >>> from POMDPPlanners.environments.push_pomdp.continuous_push_vectorized_model import (
        ...     ContinuousPushVectorizedModel,
        ... )
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> env = ContinuousPushPOMDPDiscreteActions(discount_factor=0.99)
        >>> model = ContinuousPushVectorizedModel(env, device=torch.device("cpu"))
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
        env: ContinuousPushPOMDPDiscreteActions,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        observation_resolution: float = 0.1,
    ) -> None:
        """Build the model from a live environment instance.

        Args:
            env: The discrete-action environment whose parameters are mirrored.
            device: Target device; defaults to CPU.
            dtype: Floating dtype for real-valued tensors.
            observation_resolution: Bin width used to quantize the observed
                object position into an integer tree key.

        Raises:
            NotImplementedError: If ``env`` is not the discrete-action variant;
                the continuous-action environment has no finite action set.
            ValueError: If ``observation_resolution`` is not positive.
        """
        if not isinstance(env, ContinuousPushPOMDPDiscreteActions):
            raise NotImplementedError(
                "vectorized model needs ContinuousPushPOMDPDiscreteActions: VOPP plans "
                "over a finite action set"
            )
        if observation_resolution <= 0.0:
            raise ValueError("observation_resolution must be positive")
        self.device = torch.empty(0, device=device).device
        self.dtype = dtype
        self._obs_resolution = float(observation_resolution)
        vectors = np.stack([env.action_to_vector[name] for name in env.get_actions()])
        self._action_table = self._to_tensor(vectors)
        self.num_actions = int(vectors.shape[0])
        self._has_slot = bool(env.hazard_terminal_enabled)
        self._build_transition(env)
        self._build_reward(env)
        sigma = float(env.observation_noise)
        self._obs_noise = sigma
        self._obs_inv_2var = 0.5 / (sigma * sigma)
        self._obs_log_norm = -math.log(2.0 * math.pi * sigma * sigma)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _build_transition(self, env: ContinuousPushPOMDPDiscreteActions) -> None:
        obstacles = np.asarray(env.obstacles, dtype=np.float64).reshape(-1, 4)
        self._obstacle_rows = [tuple(float(v) for v in row) for row in obstacles]
        self._obstacle_min = self._to_tensor(obstacles[:, 0:2] - obstacles[:, 2:4])
        self._obstacle_max = self._to_tensor(obstacles[:, 0:2] + obstacles[:, 2:4])
        cov = np.asarray(env.state_transition_cov_matrix, dtype=np.float64)
        self._chol_t = self._to_tensor(np.linalg.cholesky(cov).T)
        self._grid_max = float(env.grid_size) - 1.0
        self._robot_radius = float(env.robot_radius)
        self._push_threshold = float(env.push_threshold)
        self._friction = float(env.friction_coefficient)
        self._max_push = float(env.max_push)

    def _build_reward(self, env: ContinuousPushPOMDPDiscreteActions) -> None:
        self._obstacle_penalty = float(env.obstacle_penalty)
        self._obstacle_hit_prob = float(env.obstacle_hit_probability)
        self._obstacle_terminal = bool(env.is_obstacle_hit_terminal)
        self._danger_centres = self._to_tensor(
            np.asarray(env.dangerous_areas, dtype=np.float64).reshape(-1, 2)
        )
        self._danger_radius_sq = float(env.dangerous_area_radius) ** 2
        self._danger_penalty = float(env.dangerous_area_penalty)
        self._danger_hit_prob = float(env.dangerous_area_hit_probability)
        self._danger_terminal = bool(env.is_dangerous_area_hit_terminal)
        self._reward_model = env.reward_model_type
        self._penalty_decay = float(env.penalty_decay)

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # Generative kernels
    # ------------------------------------------------------------------ #

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        action = self._action_table[actions.to(torch.int64)]
        noise = torch.randn(states.shape[0], 2, dtype=self.dtype, device=self.device)
        robot = self._resolve_obstacles(states[:, 0:2] + action + noise @ self._chol_t)
        robot = torch.clamp(robot, self._robot_radius, self._grid_max - self._robot_radius)
        obj = self._push_object(robot, states[:, 2:4], action)
        next_states = torch.cat([robot, obj, states[:, 4:6]], dim=1)
        if not self._has_slot:
            return next_states
        terminal = self._hazard_terminal_draw(next_states).to(self.dtype)
        next_states = torch.cat([next_states, terminal[:, None]], dim=1)
        # A row whose terminal slot is already set is absorbing.
        return torch.where((states[:, 6] > 0.5)[:, None], states, next_states)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions  # Observation noise does not depend on the action.
        noise = torch.randn(next_states.shape[0], 2, dtype=self.dtype, device=self.device)
        observed = torch.clamp(next_states[:, 2:4] + noise * self._obs_noise, 0.0, self._grid_max)
        observations = next_states.clone()
        observations[:, 2:4] = observed
        return observations

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del states, actions  # The reward scores the realised next state only.
        dist = torch.linalg.vector_norm(next_states[:, 2:4] - next_states[:, 4:6], dim=1)
        goal = dist < _GOAL_RADIUS
        reward = -dist + _GOAL_REWARD * goal.to(self.dtype)
        robot = next_states[:, 0:2]
        if self._has_slot:
            # With a hazard-terminal flag on, a terminal-flagged hazard pays its
            # penalty exactly when the step terminated away from the goal.
            ended_on_hazard = (next_states[:, 6] > 0.5) & ~goal
        else:
            ended_on_hazard = torch.zeros_like(goal)
        reward = reward + self._obstacle_reward(robot, ended_on_hazard)
        return reward + self._danger_reward(robot, ended_on_hazard)

    def terminal_mask(self, states: Tensor) -> Tensor:
        delta = states[:, 2:4] - states[:, 4:6]
        on_target = (delta * delta).sum(dim=1) < _GOAL_RADIUS**2
        if self._has_slot:
            return on_target | (states[:, 6] > 0.5)
        return on_target

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions  # The object-position likelihood ignores the action.
        diff = observations[:, 2:4] - next_states[:, 2:4]
        return self._obs_log_norm - (diff * diff).sum(dim=1) * self._obs_inv_2var

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        quantized = torch.floor(observations[:, 2:4] / self._obs_resolution).to(torch.int64)
        return quantized[:, 0] * _HASH_PRIME_X + quantized[:, 1] * _HASH_PRIME_Y

    # ------------------------------------------------------------------ #
    # Transition helpers
    # ------------------------------------------------------------------ #

    def _resolve_obstacles(self, position: Tensor) -> Tensor:
        # The native kernel resolves the obstacles one after another, each
        # against the position the previous one left, so this loop is over
        # obstacles only.
        x, y = position[:, 0], position[:, 1]
        radius = self._robot_radius
        for cx, cy, hx, hy in self._obstacle_rows:
            dx = x - torch.clamp(x, cx - hx, cx + hx)
            dy = y - torch.clamp(y, cy - hy, cy + hy)
            dist_sq = dx * dx + dy * dy
            overlap = dist_sq < radius * radius
            dist = torch.where(dist_sq > 1e-12, torch.sqrt(dist_sq), torch.zeros_like(dist_sq))
            inside = overlap & (dist < 1e-12)
            outside = overlap & ~inside
            safe = torch.where(outside, dist, torch.ones_like(dist))
            push_x = x + dx / safe * (radius - dist)
            push_y = y + dy / safe * (radius - dist)
            # Centre inside the box: move along the axis of least penetration;
            # ties go to the first of left, right, down, up.
            pen_left = x - (cx - hx)
            pen_right = (cx + hx) - x
            pen_down = y - (cy - hy)
            pen_up = (cy + hy) - y
            min_pen = torch.minimum(
                torch.minimum(pen_left, pen_right), torch.minimum(pen_down, pen_up)
            )
            left = min_pen == pen_left
            right = ~left & (min_pen == pen_right)
            down = ~left & ~right & (min_pen == pen_down)
            up = ~left & ~right & ~down
            inside_x = torch.where(left, cx - hx - radius, torch.where(right, cx + hx + radius, x))
            inside_y = torch.where(down, cy - hy - radius, torch.where(up, cy + hy + radius, y))
            x = torch.where(outside, push_x, torch.where(inside, inside_x, x))
            y = torch.where(outside, push_y, torch.where(inside, inside_y, y))
        return torch.stack([x, y], dim=1)

    def _push_object(self, robot: Tensor, obj: Tensor, action: Tensor) -> Tensor:
        # The push is measured from the moved robot and applied along the
        # action direction, scaled by friction and capped at max_push.
        in_reach = torch.linalg.vector_norm(robot - obj, dim=1) < self._push_threshold
        norm = torch.linalg.vector_norm(action, dim=1)
        moving = norm >= 1e-12
        safe_norm = torch.where(moving, norm, torch.ones_like(norm))
        force = torch.clamp(norm, max=self._max_push) * (1.0 - self._friction)
        intended = obj + action / safe_norm[:, None] * force[:, None]
        blocked = self._point_in_obstacle(intended)
        pushed = in_reach & moving & ~blocked
        moved = torch.clamp(intended, 0.0, self._grid_max)
        return torch.where(pushed[:, None], moved, obj)

    def _hazard_terminal_draw(self, next_states: Tensor) -> Tensor:
        # Mirrors cont_hazard_terminal_draw: no hazard ends a step that reached
        # the goal; otherwise each enabled hazard the robot is in ends it with
        # its hit probability.
        delta = next_states[:, 2:4] - next_states[:, 4:6]
        goal = (delta * delta).sum(dim=1) < _GOAL_RADIUS**2
        robot = next_states[:, 0:2]
        hit = torch.zeros_like(goal)
        if self._obstacle_terminal and self._obstacle_rows:
            hit = hit | (
                self._robot_in_obstacle(robot) & self._draw(self._obstacle_hit_prob, robot)
            )
        if self._danger_terminal and self._danger_centres.shape[0] > 0:
            hit = hit | self._danger_fires(robot)
        return hit & ~goal

    # ------------------------------------------------------------------ #
    # Reward helpers
    # ------------------------------------------------------------------ #

    def _obstacle_reward(self, robot: Tensor, ended_on_hazard: Tensor) -> Tensor:
        if not self._obstacle_rows:
            return torch.zeros(robot.shape[0], dtype=self.dtype, device=self.device)
        collide = self._robot_in_obstacle(robot)
        if self._obstacle_terminal:
            charged = collide & ended_on_hazard
        else:
            charged = collide & self._draw(self._obstacle_hit_prob, robot)
        return self._obstacle_penalty * charged.to(self.dtype)

    def _danger_reward(self, robot: Tensor, ended_on_hazard: Tensor) -> Tensor:
        zeros = torch.zeros(robot.shape[0], dtype=self.dtype, device=self.device)
        if self._danger_centres.shape[0] == 0:
            return zeros
        decayed = self._reward_model is RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY
        if self._danger_terminal:
            # The decayed model has no radius cutoff, so every terminal step
            # away from the goal counts as in a zone.
            in_zone = self._in_danger_zone(robot) | decayed
            return self._danger_penalty * (in_zone & ended_on_hazard).to(self.dtype)
        if self._reward_model is RewardModelType.ZERO_MEAN_HAZARD_SHOCK:
            sign = torch.where(self._draw(0.5, robot), 1.0, -1.0).to(self.dtype)
            return torch.where(self._in_danger_zone(robot), self._danger_penalty * sign, zeros)
        return self._danger_penalty * self._danger_fires(robot).to(self.dtype)

    def _danger_fires(self, robot: Tensor) -> Tensor:
        """Whether the dangerous-area hazard fires at each robot position."""
        if self._reward_model is RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY:
            diff = robot[:, None, :] - self._danger_centres[None, :, :]
            min_dist = torch.sqrt((diff * diff).sum(dim=-1).min(dim=1).values)
            uniforms = torch.rand(robot.shape[0], dtype=self.dtype, device=self.device)
            return uniforms < torch.exp(-min_dist / self._penalty_decay)
        return self._in_danger_zone(robot) & self._draw(self._danger_hit_prob, robot)

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #

    def _robot_in_obstacle(self, robot: Tensor) -> Tensor:
        """Whether the robot disc overlaps any obstacle AABB (strictly)."""
        if not self._obstacle_rows:
            return torch.zeros(robot.shape[0], dtype=torch.bool, device=self.device)
        point = robot[:, None, :]
        closest = torch.minimum(torch.maximum(point, self._obstacle_min), self._obstacle_max)
        gap = point - closest
        return ((gap * gap).sum(dim=-1) < self._robot_radius**2).any(dim=1)

    def _point_in_obstacle(self, point: Tensor) -> Tensor:
        """Whether a point lies in any obstacle AABB, boundary included."""
        if not self._obstacle_rows:
            return torch.zeros(point.shape[0], dtype=torch.bool, device=self.device)
        p = point[:, None, :]
        inside = (p >= self._obstacle_min) & (p <= self._obstacle_max)
        return inside.all(dim=-1).any(dim=1)

    def _in_danger_zone(self, robot: Tensor) -> Tensor:
        diff = robot[:, None, :] - self._danger_centres[None, :, :]
        return ((diff * diff).sum(dim=-1) <= self._danger_radius_sq).any(dim=1)

    def _draw(self, probability: float, like: Tensor) -> Tensor:
        """One Bernoulli(``probability``) per row."""
        if probability >= 1.0:
            return torch.ones(like.shape[0], dtype=torch.bool, device=self.device)
        return torch.rand(like.shape[0], dtype=self.dtype, device=self.device) < probability
