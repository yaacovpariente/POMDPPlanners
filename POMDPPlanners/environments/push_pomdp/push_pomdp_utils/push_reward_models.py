"""Reward models for the Push POMDP family.

Mirrors the abstract-base / concrete-subclass layout used by
:mod:`POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models`
and
:mod:`POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils.laser_tag_reward_models`,
so further Push reward variants can be added without growing the env
class.

The reward model owns the parameters and pre-built buffers the reward
computation needs (obstacle geometry, dangerous areas, penalty
probabilities). The environment retains its own copies for transition /
observation paths and delegates ``reward()`` / ``reward_batch()`` to the
model.
"""

import math
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np


class BasePushRewardModel(ABC):
    """Abstract reward model for Push POMDP variants."""

    @abstractmethod
    def compute_reward(self, state: np.ndarray, action: Any, next_state: np.ndarray) -> float:
        """Return the scalar reward for ``(state, action, next_state)``."""

    @abstractmethod
    def compute_reward_batch(
        self, states: np.ndarray, action: Any, next_states: np.ndarray
    ) -> np.ndarray:
        """Return the per-row reward for a batch of states under a single action."""

    def __eq__(self, other: object) -> bool:
        # The Push envs expose ``reward_model`` as a public attribute, so the
        # base ``Environment.__eq__`` walks into it when comparing two envs.
        # Compare by configuration (all stored attributes) so two envs built
        # from identical params still test equal.
        if not isinstance(other, type(self)):
            return False
        self_attrs = self.__dict__
        other_attrs = other.__dict__
        if self_attrs.keys() != other_attrs.keys():
            return False
        for key, value in self_attrs.items():
            other_value = other_attrs[key]
            if isinstance(value, np.ndarray) or isinstance(other_value, np.ndarray):
                if not (isinstance(value, np.ndarray) and isinstance(other_value, np.ndarray)):
                    return False
                if not np.array_equal(value, other_value):
                    return False
            elif value != other_value:
                return False
        return True

    def __hash__(self) -> int:
        # Reward models are mutable (params can be reassigned post-init) and
        # are not placed in hash-keyed containers, so identity hashing keeps
        # ``__eq__`` consistent without imposing a deep hash contract.
        return id(self)


class DiscretePushRewardModel(BasePushRewardModel):
    """Standard reward model for :class:`PushPOMDP` (discrete actions).

    Reward structure:
        * Base term: ``-distance(object, target)`` where positions are read
          from the realised ``next_state``.
        * Goal bonus: ``+100.0`` when the object lies within ``0.5`` of the
          target.
        * Obstacle penalty: ``obstacle_penalty`` is added when the realised
          post-action robot position (``next_state[:2]``) lies within
          ``obstacle_radius`` of any circular obstacle. When
          ``obstacle_hit_probability < 1.0`` the penalty fires with that
          probability per call (one Bernoulli draw per state).
        * Dangerous-area penalty: ``dangerous_area_penalty`` is added when
          the realised robot position lies within ``dangerous_area_radius``
          of any dangerous-area centre. Stochastic via
          ``dangerous_area_hit_probability`` like obstacles. At most one
          penalty per row even when zones overlap.
    """

    def __init__(
        self,
        obstacles: List[Tuple[float, float]],
        obstacle_radius: float,
        obstacle_penalty: float,
        obstacle_hit_probability: float,
        dangerous_areas_arr: np.ndarray,
        dangerous_area_radius: float,
        dangerous_area_penalty: float,
        dangerous_area_hit_probability: float,
    ):
        self.obstacles = obstacles
        self.obstacle_radius = float(obstacle_radius)
        self.obstacle_penalty = float(obstacle_penalty)
        self.obstacle_hit_probability = float(obstacle_hit_probability)
        self.dangerous_areas_arr = dangerous_areas_arr
        self.dangerous_area_radius = float(dangerous_area_radius)
        self.dangerous_area_penalty = float(dangerous_area_penalty)
        self.dangerous_area_hit_probability = float(dangerous_area_hit_probability)

    def compute_reward(self, state: np.ndarray, action: Any, next_state: np.ndarray) -> float:
        del state, action  # penalties consume the realised post-transition robot pos
        dx = next_state[2] - next_state[4]
        dy = next_state[3] - next_state[5]
        distance_to_target = math.sqrt(dx * dx + dy * dy)

        reward = -distance_to_target
        if distance_to_target < 0.5:
            reward += 100.0

        if self._is_colliding_with_obstacle_scalar(float(next_state[0]), float(next_state[1])):
            if (
                self.obstacle_hit_probability >= 1.0
                or np.random.random() < self.obstacle_hit_probability
            ):
                reward += self.obstacle_penalty

        if self._is_in_dangerous_area_scalar(float(next_state[0]), float(next_state[1])):
            if (
                self.dangerous_area_hit_probability >= 1.0
                or np.random.random() < self.dangerous_area_hit_probability
            ):
                reward += self.dangerous_area_penalty

        return float(reward)

    def compute_reward_batch(
        self, states: np.ndarray, action: Any, next_states: np.ndarray
    ) -> np.ndarray:
        del states, action
        dx = next_states[:, 2] - next_states[:, 4]
        dy = next_states[:, 3] - next_states[:, 5]
        dist = np.sqrt(dx * dx + dy * dy)
        rewards = -dist
        rewards = np.where(dist < 0.5, rewards + 100.0, rewards)
        rewards += self._obstacle_penalty_batch(next_states)
        rewards += self._dangerous_area_penalty_batch(next_states)
        return rewards.astype(np.float64)

    def _is_colliding_with_obstacle_scalar(self, pos_x: float, pos_y: float) -> bool:
        if not self.obstacles:
            return False
        obs_r_sq = self.obstacle_radius * self.obstacle_radius
        for obs_x, obs_y in self.obstacles:
            ddx = pos_x - obs_x
            ddy = pos_y - obs_y
            if ddx * ddx + ddy * ddy <= obs_r_sq:
                return True
        return False

    def _is_in_dangerous_area_scalar(self, pos_x: float, pos_y: float) -> bool:
        if self.dangerous_areas_arr.shape[0] == 0:
            return False
        r_sq = self.dangerous_area_radius * self.dangerous_area_radius
        # dangerous_areas_arr is (K, 2) float64; small K, scalar loop wins.
        for danger_x, danger_y in self.dangerous_areas_arr:
            ddx = pos_x - float(danger_x)
            ddy = pos_y - float(danger_y)
            if ddx * ddx + ddy * ddy <= r_sq:
                return True
        return False

    def _obstacle_penalty_batch(self, next_states: np.ndarray) -> np.ndarray:
        if not self.obstacles:
            return np.zeros(len(next_states), dtype=np.float64)
        positions = next_states[:, :2]
        obs_arr = np.asarray(self.obstacles, dtype=float)  # (M, 2)
        diff = positions[:, None, :] - obs_arr[None, :, :]  # (N, M, 2)
        dist_sq = np.sum(diff * diff, axis=2)  # (N, M)
        colliding = np.any(dist_sq <= self.obstacle_radius * self.obstacle_radius, axis=1)
        if self.obstacle_hit_probability < 1.0:
            applied = np.random.random(len(next_states)) < self.obstacle_hit_probability
            colliding = colliding & applied
        return np.where(colliding, self.obstacle_penalty, 0.0).astype(np.float64)

    def _dangerous_area_penalty_batch(self, next_states: np.ndarray) -> np.ndarray:
        if self.dangerous_areas_arr.shape[0] == 0:
            return np.zeros(len(next_states), dtype=np.float64)
        positions = next_states[:, :2]
        diff = positions[:, None, :] - self.dangerous_areas_arr[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        in_zone = np.any(dist_sq <= self.dangerous_area_radius * self.dangerous_area_radius, axis=1)
        if self.dangerous_area_hit_probability < 1.0:
            applied = np.random.random(len(next_states)) < self.dangerous_area_hit_probability
            in_zone = in_zone & applied
        return np.where(in_zone, self.dangerous_area_penalty, 0.0).astype(np.float64)


class ContinuousPushRewardModel(BasePushRewardModel):
    """Standard reward model for :class:`ContinuousPushPOMDP`.

    Reward structure mirrors :class:`DiscretePushRewardModel` (distance
    penalty, goal bonus, obstacle penalty, dangerous-area penalty) but
    obstacles are axis-aligned bounding boxes (``(cx, cy, half_x, half_y)``
    rows) tested against a circular robot footprint of radius
    ``robot_radius``. Dangerous areas remain circular point-vs-circle
    checks. Both penalties are optionally stochastic via per-call Bernoulli
    draws.
    """

    def __init__(
        self,
        obstacles: np.ndarray,
        robot_radius: float,
        obstacle_penalty: float,
        obstacle_hit_probability: float,
        dangerous_areas_arr: np.ndarray,
        dangerous_area_radius: float,
        dangerous_area_penalty: float,
        dangerous_area_hit_probability: float,
    ):
        self.obstacles = obstacles
        self.robot_radius = float(robot_radius)
        self.obstacle_penalty = float(obstacle_penalty)
        self.obstacle_hit_probability = float(obstacle_hit_probability)
        self.dangerous_areas_arr = dangerous_areas_arr
        self.dangerous_area_radius = float(dangerous_area_radius)
        self.dangerous_area_penalty = float(dangerous_area_penalty)
        self.dangerous_area_hit_probability = float(dangerous_area_hit_probability)

    def compute_reward(self, state: np.ndarray, action: Any, next_state: np.ndarray) -> float:
        state_arr = np.ascontiguousarray(np.asarray(state, dtype=np.float64)).reshape(1, -1)
        next_states_arr = np.ascontiguousarray(np.asarray(next_state, dtype=np.float64)).reshape(
            1, -1
        )
        rewards = self.compute_reward_batch(state_arr, action, next_states_arr)
        return float(rewards[0])

    def compute_reward_batch(
        self, states: np.ndarray, action: Any, next_states: np.ndarray
    ) -> np.ndarray:
        del states, action

        delta = next_states[:, 2:4] - next_states[:, 4:6]
        dist_to_target = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        rewards = -dist_to_target
        rewards[dist_to_target < 0.5] += 100.0

        if self.obstacles.shape[0] > 0:
            robot_after = next_states[:, :2]
            collide = self._batch_circle_obstacle_overlap(robot_after, self.robot_radius)
            if self.obstacle_hit_probability < 1.0:
                applied = np.random.random(collide.shape[0]) < self.obstacle_hit_probability
                collide = collide & applied
            if np.any(collide):
                rewards[collide] += self.obstacle_penalty

        if self.dangerous_areas_arr.shape[0] > 0:
            robot_after = next_states[:, :2]
            in_zone = self._batch_robot_in_dangerous_areas(robot_after)
            if self.dangerous_area_hit_probability < 1.0:
                applied = np.random.random(in_zone.shape[0]) < self.dangerous_area_hit_probability
                in_zone = in_zone & applied
            if np.any(in_zone):
                rewards[in_zone] += self.dangerous_area_penalty

        return rewards

    def _batch_robot_in_dangerous_areas(self, positions: np.ndarray) -> np.ndarray:
        diff = positions[:, None, :] - self.dangerous_areas_arr[None, :, :]
        dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
        return np.any(dist_sq <= self.dangerous_area_radius * self.dangerous_area_radius, axis=1)

    def _batch_circle_obstacle_overlap(self, positions: np.ndarray, radius: float) -> np.ndarray:
        walls = self.obstacles
        cx = walls[:, 0]
        cy = walls[:, 1]
        hx = walls[:, 2]
        hy = walls[:, 3]
        px = positions[:, 0:1]
        py = positions[:, 1:2]
        closest_x = np.clip(px, cx - hx, cx + hx)
        closest_y = np.clip(py, cy - hy, cy + hy)
        dx = px - closest_x
        dy = py - closest_y
        dist_sq = dx * dx + dy * dy
        return np.any(dist_sq < radius * radius, axis=1)
