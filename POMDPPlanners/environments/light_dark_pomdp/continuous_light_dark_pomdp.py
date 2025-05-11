from typing import List, Any
from pathlib import Path
import json
import hashlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from POMDPPlanners.core.environment import DiscreteActionsEnvironment, ObservationModel
from POMDPPlanners.core.distributions import Distribution, DiscreteDistribution
from POMDPPlanners.core.simulation import History
from POMDPPlanners.core.simulation import MetricValue
from POMDPPlanners.utils.statistics import confidence_interval
from POMDPPlanners.core.belief import Belief

from POMDPPlanners.environments.light_dark_pomdp.base_light_dark_pomdp import BaseLightDarkPOMDP


class ContinuousLightDarkPOMDP(BaseLightDarkPOMDP):
    def __init__(
        self, 
        discount_factor: float, 
        name: str = "ContinuousLightDarkPOMDP",
        state_transition_cov_matrix: np.ndarray = np.eye(2),
        observation_cov_matrix: np.ndarray = np.eye(2),
        beacons: np.ndarray = np.array([[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]]),
        goal_state: np.ndarray = np.array([10, 5]),
        start_state: np.ndarray = np.array([0, 5]),
        obstacles: np.ndarray = np.array([[3, 7], [5, 5]]),
        obstacle_hit_probability: float = 0.2,
        obstacle_reward: float = -10.0,
        goal_reward: float = 10.0,
        fuel_cost: float = 2.0,
        grid_size: int = 11,
        goal_state_radius: float = 1.5,
        beacon_radius: float = 1.0,
        obstacle_radius: float = 1.5,
    ):
        super().__init__(
            discount_factor=discount_factor, 
            name=name, 
            beacons=beacons, 
            goal_state=goal_state, 
            start_state=start_state, 
            obstacles=obstacles, 
            obstacle_hit_probability=obstacle_hit_probability, 
            obstacle_reward=obstacle_reward, 
            goal_reward=goal_reward, 
            fuel_cost=fuel_cost, 
            grid_size=grid_size
        )
        
        self.__type_check(
            state_transition_cov_matrix=state_transition_cov_matrix,
            observation_cov_matrix=observation_cov_matrix,
            goal_state_radius=goal_state_radius,
            beacon_radius=beacon_radius,
            obstacle_radius=obstacle_radius,
        )
        
        self.state_transition_cov_matrix = state_transition_cov_matrix
        self.observation_cov_matrix = observation_cov_matrix
        self.goal_state_radius = goal_state_radius
        self.beacon_radius = beacon_radius
        self.obstacle_radius = obstacle_radius

    def __type_check(
        self,
        state_transition_cov_matrix: np.ndarray,
        observation_cov_matrix: np.ndarray,
        goal_state_radius: float,
        beacon_radius: float,
        obstacle_radius: float,
    ):
        assert state_transition_cov_matrix.shape == (2, 2), "state_transition_cov_matrix must be a 2x2 matrix"
        assert observation_cov_matrix.shape == (2, 2), "observation_cov_matrix must be a 2x2 matrix"
        assert goal_state_radius > 0, "goal_state_radius must be greater than 0"
        assert beacon_radius > 0, "beacon_radius must be greater than 0"
        assert obstacle_radius > 0, "obstacle_radius must be greater than 0"

    def state_transition_model(self, state: np.ndarray, action: Any) -> Distribution:
        assert state.shape == (2,), "state must be a 2D vector"
        
        action_vector = self.action_to_vector[action]
        deterministic_next_state = state + action_vector
        noise = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=self.state_transition_cov_matrix
        )
        
        next_state = deterministic_next_state + noise
        next_state = np.clip(next_state, 0, self.grid_size)
        
        return DiscreteDistribution(values=[next_state], probs=np.array([1.0]))

    def observation_model(self, next_state: np.ndarray, action: Any) -> Distribution:
        assert next_state.shape == (2,), "next_state must be a 2D vector"
        
        noise = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=self.observation_cov_matrix
        )
        
        observation = next_state + noise
        observation = np.clip(observation, 0, self.grid_size)
        
        return DiscreteDistribution(values=[observation], probs=np.array([1.0]))

    def reward(self, state: np.ndarray, action: Any) -> float:
        assert state.shape == (2,), "state must be a 2D vector"

        action_vector = self.action_to_vector[action]
        next_state = state + action_vector

        is_goal_state = np.linalg.norm(next_state - self.goal_state) <= self.goal_state_radius
        
        is_obstacle_hit = np.any(
            np.linalg.norm(next_state.reshape(-1, 1) - self.obstacles, axis=0) <= self.obstacle_radius
        )
        
        is_out_of_grid = np.any(next_state < 0) or np.any(next_state > self.grid_size)

        reward = -self.fuel_cost - np.linalg.norm(next_state - self.goal_state)

        if is_goal_state:
            reward += self.goal_reward
        elif is_obstacle_hit:
            if np.random.rand() < self.obstacle_hit_probability:
                reward += self.obstacle_reward
        elif is_out_of_grid:
            reward += self.obstacle_reward

        return reward

    def is_terminal(self, state: np.ndarray) -> bool:
        assert state.shape == (2,), "state must be a 2D vector"

        is_goal_state = np.linalg.norm(state - self.goal_state) <= self.goal_state_radius

        is_obstacle_hit = np.any(
            np.linalg.norm(state.reshape(-1, 1) - self.obstacles, axis=0) <= self.obstacle_radius
        )

        is_out_of_grid = np.any(state < 0) or np.any(state > self.grid_size)

        is_terminal = is_goal_state or is_obstacle_hit or is_out_of_grid

        return is_terminal

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        goal_reached = []
        obstacle_hits = []
        for history in histories:
            goal_reached_in_history = False
            obstacle_hit_in_history = False
            
            for i, step in enumerate(history.history):
                if np.linalg.norm(step.state - self.goal_state) <= self.goal_state_radius:
                    goal_reached_in_history = True
                    break
                
                if np.any(
                    np.linalg.norm(step.state.reshape(-1, 1) - self.obstacles, axis=0) <= self.obstacle_radius
                ):
                    obstacle_hit_in_history = True
            
            goal_reached.append(1 if goal_reached_in_history else 0)
            obstacle_hits.append(1 if obstacle_hit_in_history else 0)

        avg_goal_reached = np.mean(goal_reached)
        avg_obstacle_hits = np.mean(obstacle_hits)

        goal_reached_ci = confidence_interval(data=goal_reached, confidence=0.95)
        obstacle_hits_ci = confidence_interval(data=obstacle_hits, confidence=0.95)

        return [
            MetricValue(
                name="goal_reaching_rate",
                value=avg_goal_reached,
                lower_confidence_bound=goal_reached_ci[0],
                upper_confidence_bound=goal_reached_ci[1],
            ),
            MetricValue(
                name="obstacle_hit_rate",
                value=avg_obstacle_hits,
                lower_confidence_bound=obstacle_hits_ci[0],
                upper_confidence_bound=obstacle_hits_ci[1],
            )
        ]

    def __eq__(self, other):
        if not isinstance(other, ContinuousLightDarkPOMDP):
            return False
        
        if not super().__eq__(other):
            return False
        
        return (
            np.array_equal(self.state_transition_cov_matrix, other.state_transition_cov_matrix) and
            np.array_equal(self.observation_cov_matrix, other.observation_cov_matrix) and
            self.goal_state_radius == other.goal_state_radius and
            self.beacon_radius == other.beacon_radius and
            self.obstacle_radius == other.obstacle_radius
        )

