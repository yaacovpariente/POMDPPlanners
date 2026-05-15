from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.numba_kernels import (
    compute_reward_base_batch_kernel,
    compute_reward_base_kernel,
    compute_reward_decaying_hit_prob_kernel,
)


class BaseLightDarkRewardModel(ABC):
    @abstractmethod
    def _compute_reward(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray
    ) -> float:
        pass

    def compute_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: Optional[np.ndarray] = None,
    ) -> float:
        if state.shape != (2,):
            raise ValueError("state must be a 2D vector")
        if action.shape != (2,):
            raise ValueError("action must be a 2D vector")

        # Honour the realised ``next_state`` threaded by
        # :meth:`Environment.sample_next_step` so obstacle / goal /
        # out-of-grid checks score against the same draw as the trajectory.
        # Fall back to the deterministic ``state + action`` only when no
        # realised draw was supplied (legacy ``compute_reward(state, action)``
        # callers).
        next_state_local = state + action if next_state is None else np.asarray(next_state)
        return self._compute_reward(state, action, next_state_local)

    def compute_reward_batch(
        self,
        states: np.ndarray,
        action: np.ndarray,
        next_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if next_states is None:
            return np.array([self.compute_reward(states[i], action) for i in range(len(states))])
        next_states_arr = np.asarray(next_states)
        return np.array(
            [
                self.compute_reward(states[i], action, next_state=next_states_arr[i])
                for i in range(len(states))
            ]
        )


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

    def _compute_reward(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray
    ) -> float:
        del state, action  # Realised position threaded via ``next_state``.
        reward, in_obstacle_region = compute_reward_base_kernel(
            next_state,
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

    def compute_reward_batch(
        self,
        states: np.ndarray,
        action: np.ndarray,
        next_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Deterministic part runs in a Numba kernel. The mask flags the rows
        # where the next state landed in an obstacle region but was not at
        # goal / out-of-grid; those rows still need the stochastic obstacle
        # contribution applied in Python so seeded RNG stays reproducible.
        # Honour the realised ``next_states`` when threaded; otherwise fall
        # back to the deterministic ``states + action`` pre-noise position.
        if next_states is None:
            next_states_arr = np.asarray(states) + np.asarray(action)
        else:
            next_states_arr = np.ascontiguousarray(np.asarray(next_states, dtype=np.float64))
        rewards, obstacle_mask = compute_reward_base_batch_kernel(
            next_states_arr,
            self.goal_state,
            self.obstacles,
            self.goal_state_radius,
            self.obstacle_radius,
            float(self.grid_size),
            self.fuel_cost,
            self.goal_reward,
            self.obstacle_reward,
        )
        n_obs = int(np.count_nonzero(obstacle_mask))
        if n_obs > 0:
            rewards[obstacle_mask] += self._obstacle_reward_batch(n_obs)
        return rewards

    def _obstacle_reward_batch(self, n: int) -> np.ndarray:
        hits = np.random.rand(n) < self.obstacle_hit_probability
        return np.where(hits, self.obstacle_reward, 0.0)


class ContinuousLDHighVarianceStatesRewardModel(ContinuousLightDarkRewardModel):
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

    def _compute_reward(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray
    ) -> float:
        del state, action  # Realised position threaded via ``next_state``.
        uniform = float(np.random.rand())
        return float(
            compute_reward_decaying_hit_prob_kernel(
                next_state,
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

    def compute_reward_batch(
        self,
        states: np.ndarray,
        action: np.ndarray,
        next_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Honour the realised ``next_states`` threaded by callers so the
        # distance-to-goal, out-of-grid, and decaying-hit-probability
        # checks score against the same draw as the trajectory. Fall back
        # to the deterministic ``states + action`` only when no realised
        # batch was supplied.
        if next_states is None:
            next_states_arr = np.asarray(states) + np.asarray(action)
        else:
            next_states_arr = np.asarray(next_states)
        dists_to_goal = np.linalg.norm(next_states_arr - self.goal_state, axis=1)
        rewards = -self.fuel_cost - dists_to_goal

        goal_mask = dists_to_goal <= self.goal_state_radius
        rewards[goal_mask] += self.goal_reward

        oob = np.any(next_states_arr < 0, axis=1) | np.any(next_states_arr > self.grid_size, axis=1)
        rewards[oob & ~goal_mask] += self.obstacle_reward

        diffs = next_states_arr[:, :, np.newaxis] - self.obstacles[np.newaxis, :, :]
        min_dists = np.min(np.linalg.norm(diffs, axis=1), axis=1)
        probs = np.exp(-min_dists / self.penalty_decay)
        rewards[np.random.rand(len(next_states_arr)) < probs] += self.obstacle_reward
        return rewards
