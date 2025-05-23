from abc import ABC, abstractmethod
import numpy as np

class BaseLightDarkRewardModel(ABC):
    @abstractmethod
    def _compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        pass
    
    def compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        assert state.shape == (2,), "state must be a 2D vector"
        assert action.shape == (2,), "action must be a 2D vector"
        
        return self._compute_reward(state, action)
    

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

    def _compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        next_state = state + action

        is_goal_state = np.linalg.norm(next_state - self.goal_state) <= self.goal_state_radius
        
        is_in_obstacle_range = np.any(
            np.linalg.norm(next_state.reshape(-1, 1) - self.obstacles, axis=0) <= self.obstacle_radius
        )
        
        is_out_of_grid = np.any(next_state < 0) or np.any(next_state > self.grid_size)

        reward = -self.fuel_cost - np.linalg.norm(next_state - self.goal_state)

        if is_goal_state:
            reward += self.goal_reward
        elif is_in_obstacle_range:
            reward += self._obstacle_reward(next_state)
        elif is_out_of_grid:
            reward += self.obstacle_reward

        return reward
    
    def _obstacle_reward(self, state: np.ndarray) -> float:
        return self.obstacle_reward if np.random.rand() < self.obstacle_hit_probability else 0.0
    

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
        penalty_decay: float
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
        next_state = state + action

        is_goal_state = np.linalg.norm(next_state - self.goal_state) <= self.goal_state_radius
        
        is_out_of_grid = np.any(next_state < 0) or np.any(next_state > self.grid_size)

        reward = -self.fuel_cost - np.linalg.norm(next_state - self.goal_state)

        if is_goal_state:
            reward += self.goal_reward
        elif is_out_of_grid:
            reward += self.obstacle_reward

        reward += self._obstacle_reward(next_state)
        
        return reward
    
    def _obstacle_reward(self, state: np.ndarray) -> float:
        # Calculate distance to nearest obstacle
        distances = np.linalg.norm(state.reshape(-1, 1) - self.obstacles, axis=0)
        d = np.min(distances)
        
        # Calculate probability based on distance and decay factor
        p = np.exp(-d/self.penalty_decay)
        
        # Return obstacle reward if random value is less than probability
        return self.obstacle_reward if np.random.rand() < p else 0.0
    
