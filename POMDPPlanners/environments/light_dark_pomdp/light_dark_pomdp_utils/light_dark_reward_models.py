from abc import ABC, abstractmethod

import numpy as np

from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.numba_kernels import (
    compute_reward_base_kernel,
    compute_reward_decaying_hit_prob_kernel,
)


class BaseLightDarkRewardModel(ABC):
    @abstractmethod
    def _compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        pass

    def compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        if state.shape != (2,):
            raise ValueError("state must be a 2D vector")
        if action.shape != (2,):
            raise ValueError("action must be a 2D vector")

        return self._compute_reward(state, action)

    def compute_reward_batch(self, states: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.array([self.compute_reward(states[i], action) for i in range(len(states))])


class ContinuousLightDarkRewardModel(BaseLightDarkRewardModel):
    def __init__(
        self,
        goal_state: np.ndarray,
        obstacles: np.ndarray,
        goal_state_radius: float,
        obstacle_radius: float,
        grid_size: int,
        obstacle_hit_probability: float,
        obstacle_reward: float,
        goal_reward: float,
        fuel_cost: float,
    ):
        self.goal_state = goal_state
        self.obstacles = obstacles
        self.goal_state_radius = goal_state_radius
        self.obstacle_radius = obstacle_radius
        self.grid_size = grid_size
        self.obstacle_hit_probability = obstacle_hit_probability
        self.obstacle_reward = obstacle_reward
        self.goal_reward = goal_reward
        self.fuel_cost = fuel_cost
        # Cached scalars/arrays for compute_reward_batch hot path:
        # squared radii skip per-call sqrts in mask checks, and obstacle
        # rows are pre-broadcast as (1, M) views to avoid the (N, 2, M)
        # intermediate that np.linalg.norm forces.
        self._goal_radius_sq = float(goal_state_radius) * float(goal_state_radius)
        self._obstacle_radius_sq = float(obstacle_radius) * float(obstacle_radius)
        self._obs_x_row = np.ascontiguousarray(obstacles[0]).reshape(1, -1)
        self._obs_y_row = np.ascontiguousarray(obstacles[1]).reshape(1, -1)

    def _is_goal_state(self, state: np.ndarray) -> bool:
        """Check if state is within goal state radius."""
        return bool(np.linalg.norm(state - self.goal_state) <= self.goal_state_radius)

    def _is_in_obstacle_range(self, state: np.ndarray) -> bool:
        """Check if state is within obstacle radius of any obstacle."""
        return bool(
            (
                np.linalg.norm(state.reshape(-1, 1) - self.obstacles, axis=0)
                <= self.obstacle_radius
            ).any()
        )

    def _is_out_of_grid(self, state: np.ndarray) -> bool:
        """Check if state is outside the grid boundaries."""
        return bool(np.any(state < 0) or np.any(state > self.grid_size))

    def _compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        reward, in_obstacle_region = compute_reward_base_kernel(
            state,
            action,
            self.goal_state,
            self.obstacles,
            self.goal_state_radius,
            self.obstacle_radius,
            float(self.grid_size),
            self.fuel_cost,
            self.goal_reward,
            self.obstacle_reward,
        )
        if in_obstacle_region:
            reward += self._obstacle_reward_scalar()
        return float(reward)

    def _obstacle_reward_scalar(self) -> float:
        """Stochastic obstacle-hit contribution.

        Called by ``_compute_reward`` only when the Numba kernel reports the
        next state landed in an obstacle region but was not at goal / out-of-grid,
        preserving the pre-refactor RNG call pattern.
        """
        return self.obstacle_reward if np.random.rand() < self.obstacle_hit_probability else 0.0

    def _obstacle_reward(self, state: np.ndarray) -> float:  # pylint: disable=unused-argument
        return self.obstacle_reward if np.random.rand() < self.obstacle_hit_probability else 0.0

    def compute_reward_batch(self, states: np.ndarray, action: np.ndarray) -> np.ndarray:
        next_states = states + action
        nx = next_states[:, 0]
        ny = next_states[:, 1]

        # Goal distance from raw components — cheaper than np.linalg.norm.
        g_dx = nx - self.goal_state[0]
        g_dy = ny - self.goal_state[1]
        sq_dist_to_goal = g_dx * g_dx + g_dy * g_dy
        rewards = -self.fuel_cost - np.sqrt(sq_dist_to_goal)
        goal_mask = sq_dist_to_goal <= self._goal_radius_sq
        rewards[goal_mask] += self.goal_reward

        # Obstacle membership — squared distances on (N, M), no (N, 2, M)
        # intermediate. self._obs_{x,y}_row are cached (1, M) views.
        o_dx = nx[:, None] - self._obs_x_row
        o_dy = ny[:, None] - self._obs_y_row
        sq_obs_dists = o_dx * o_dx + o_dy * o_dy
        in_range = np.any(sq_obs_dists <= self._obstacle_radius_sq, axis=1)
        obstacle_mask = in_range & ~goal_mask
        n_obs = int(np.count_nonzero(obstacle_mask))
        if n_obs > 0:
            rewards[obstacle_mask] += self._obstacle_reward_batch(n_obs)

        oob = (nx < 0.0) | (ny < 0.0) | (nx > self.grid_size) | (ny > self.grid_size)
        rewards[oob & ~goal_mask & ~in_range] += self.obstacle_reward
        return rewards

    def _obstacle_reward_batch(self, n: int) -> np.ndarray:
        hits = np.random.rand(n) < self.obstacle_hit_probability
        return np.where(hits, self.obstacle_reward, 0.0)


class ContinuousLDDangerousStatesRewardModel(ContinuousLightDarkRewardModel):
    def _obstacle_reward_scalar(self) -> float:
        """The expected reward is 0.0, but the variance is high."""
        return self.obstacle_reward if np.random.rand() < 0.5 else -self.obstacle_reward

    def _obstacle_reward(self, state: np.ndarray) -> float:
        """The expected reward is 0.0, but the variance is high."""
        return self.obstacle_reward if np.random.rand() < 0.5 else -self.obstacle_reward

    def _obstacle_reward_batch(self, n: int) -> np.ndarray:
        signs = np.where(np.random.rand(n) < 0.5, 1.0, -1.0)
        return self.obstacle_reward * signs


class ContinuousLightDarkDecayingHitProbabilityRewardModel(BaseLightDarkRewardModel):
    def __init__(
        self,
        goal_state: np.ndarray,
        obstacles: np.ndarray,
        goal_state_radius: float,
        obstacle_radius: float,
        grid_size: int,
        obstacle_hit_probability: float,
        obstacle_reward: float,
        goal_reward: float,
        fuel_cost: float,
        penalty_decay: float,
    ):
        self.goal_state = goal_state
        self.obstacles = obstacles
        self.goal_state_radius = goal_state_radius
        self.obstacle_radius = obstacle_radius
        self.grid_size = grid_size
        self.obstacle_hit_probability = obstacle_hit_probability
        self.obstacle_reward = obstacle_reward
        self.goal_reward = goal_reward
        self.fuel_cost = fuel_cost
        self.penalty_decay = penalty_decay

    def _compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        uniform = float(np.random.rand())
        return float(
            compute_reward_decaying_hit_prob_kernel(
                state,
                action,
                self.goal_state,
                self.obstacles,
                self.goal_state_radius,
                float(self.grid_size),
                self.fuel_cost,
                self.goal_reward,
                self.obstacle_reward,
                self.penalty_decay,
                uniform,
            )
        )

    def _obstacle_reward(self, state: np.ndarray) -> float:
        # Calculate distance to nearest obstacle
        distances = np.linalg.norm(state.reshape(-1, 1) - self.obstacles, axis=0)
        d: float = np.min(distances)

        # Calculate probability based on distance and decay factor
        p = np.exp(-d / self.penalty_decay)

        # Return obstacle reward if random value is less than probability
        return self.obstacle_reward if np.random.rand() < p else 0.0

    def compute_reward_batch(self, states: np.ndarray, action: np.ndarray) -> np.ndarray:
        next_states = states + action
        dists_to_goal = np.linalg.norm(next_states - self.goal_state, axis=1)
        rewards = -self.fuel_cost - dists_to_goal

        goal_mask = dists_to_goal <= self.goal_state_radius
        rewards[goal_mask] += self.goal_reward

        oob = np.any(next_states < 0, axis=1) | np.any(next_states > self.grid_size, axis=1)
        rewards[oob & ~goal_mask] += self.obstacle_reward

        diffs = next_states[:, :, np.newaxis] - self.obstacles[np.newaxis, :, :]
        min_dists = np.min(np.linalg.norm(diffs, axis=1), axis=1)
        probs = np.exp(-min_dists / self.penalty_decay)
        rewards[np.random.rand(len(next_states)) < probs] += self.obstacle_reward
        return rewards
