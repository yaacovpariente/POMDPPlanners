from typing import List, Any
from pathlib import Path
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from POMDPPlanners.core.environment import DiscreteActionsEnvironment, ObservationModel, SpaceInfo, SpaceType
from POMDPPlanners.core.distributions import Distribution, DiscreteDistribution
from POMDPPlanners.core.simulation import History
from POMDPPlanners.core.simulation import MetricValue
from POMDPPlanners.utils.statistics import confidence_interval
from POMDPPlanners.core.belief import Belief

from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.base_light_dark_pomdp import BaseLightDarkPOMDP
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_observation_models import ContinuousLightDarkNormalNoiseObservationModel
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models import (
    ContinuousLightDarkRewardModel,
    ContinuousLightDarkDecayingHitProbabilityRewardModel
)

class RewardModelType(Enum):
    STANDARD = "standard"
    DECAYING_HIT_PROBABILITY = "decaying_hit_probability"

class StateTransitionModel(Distribution):
    def __init__(self, state: np.ndarray, action: np.ndarray, state_transition_cov_matrix: np.ndarray):
        self.state = state
        self.action = action
        self.state_transition_cov_matrix = state_transition_cov_matrix
        self.mean = state + action
        
    def sample(self) -> np.ndarray:
        return np.random.multivariate_normal(
            mean=self.mean,
            cov=self.state_transition_cov_matrix
        )
    
    def probability(self, value: np.ndarray) -> float:
        # Calculate the probability density of the multivariate normal distribution
        n = len(self.mean)
        diff = value - self.mean
        exponent = -0.5 * np.dot(np.dot(diff, np.linalg.inv(self.state_transition_cov_matrix)), diff)
        normalization = 1.0 / (np.sqrt((2 * np.pi) ** n * np.linalg.det(self.state_transition_cov_matrix)))
        return normalization * np.exp(exponent)
        

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
        reward_model_type: RewardModelType = RewardModelType.STANDARD,
        penalty_decay: float = 1.0,
    ):
        space_info = SpaceInfo(
            action_space=SpaceType.CONTINUOUS,
            observation_space=SpaceType.CONTINUOUS
        )
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            beacons=beacons,
            goal_state=goal_state,
            start_state=start_state,
            obstacles=obstacles,
            obstacle_hit_probability=obstacle_hit_probability,
            obstacle_reward=obstacle_reward,
            goal_reward=goal_reward,
            beacon_radius=beacon_radius,
            fuel_cost=fuel_cost,
            grid_size=grid_size,
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
        self.penalty_decay = penalty_decay
        
        # Initialize reward model based on type
        if reward_model_type == RewardModelType.STANDARD:
            self.reward_model = ContinuousLightDarkRewardModel(
                goal_state=goal_state,
                obstacles=obstacles,
                goal_state_radius=goal_state_radius,
                obstacle_radius=obstacle_radius,
                grid_size=grid_size,
                obstacle_hit_probability=obstacle_hit_probability,
                obstacle_reward=obstacle_reward,
                goal_reward=goal_reward,
                fuel_cost=fuel_cost,
            )
        elif reward_model_type == RewardModelType.DECAYING_HIT_PROBABILITY:
            self.reward_model = ContinuousLightDarkDecayingHitProbabilityRewardModel(
                goal_state=goal_state,
                obstacles=obstacles,
                goal_state_radius=goal_state_radius,
                obstacle_radius=obstacle_radius,
                grid_size=grid_size,
                obstacle_hit_probability=obstacle_hit_probability,
                obstacle_reward=obstacle_reward,
                goal_reward=goal_reward,
                fuel_cost=fuel_cost,
                penalty_decay=penalty_decay,
            )
        else:
            raise ValueError(f"Unknown reward model type: {reward_model_type}")

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
    
    def state_transition_model(self, state: np.ndarray, action: np.ndarray) -> Distribution:
        assert state.shape == (2,), "state must be a 2D vector"
        assert action.shape == (2,), "action must be a 2D vector"
        
        return StateTransitionModel(
            state=state,
            action=action,
            state_transition_cov_matrix=self.state_transition_cov_matrix
        )

    def observation_model(self, next_state: np.ndarray, action: np.ndarray) -> Distribution:
        assert next_state.shape == (2,), "next_state must be a 2D vector"
        assert action.shape == (2,), "action must be a 2D vector"
        
        return ContinuousLightDarkNormalNoiseObservationModel(
            next_state=next_state,
            action=action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size
        )

    def reward(self, state: np.ndarray, action: np.ndarray) -> float:
        return self.reward_model.compute_reward(state, action)

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

class ContinuousLightDarkPOMDPDiscreteActions(ContinuousLightDarkPOMDP):
    def __init__(
        self,
        discount_factor: float,
        state_transition_cov_matrix: np.ndarray = np.eye(2),
        observation_cov_matrix: np.ndarray = np.eye(2),
        obstacle_hit_probability: float = 0.2,
        obstacle_reward: float = -10.0,
        goal_reward: float = 10.0,
        fuel_cost: float = 2.0,
        grid_size: int = 11,
        goal_state_radius: float = 1.5,
        beacon_radius: float = 1.0,
        obstacle_radius: float = 1.5,
        name: str = "ContinuousLightDarkPOMDPDiscreteActions",
        beacons: np.ndarray = np.array([[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]]),
        goal_state: np.ndarray = np.array([10, 5]),
        start_state: np.ndarray = np.array([0, 5]),
        obstacles: np.ndarray = np.array([[3, 7], [5, 5]]),
        reward_model_type: RewardModelType = RewardModelType.STANDARD,
        penalty_decay: float = 1.0,
    ):
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            state_transition_cov_matrix=state_transition_cov_matrix,
            observation_cov_matrix=observation_cov_matrix,
            beacons=beacons,
            goal_state=goal_state,
            start_state=start_state,
            obstacles=obstacles,
            obstacle_hit_probability=obstacle_hit_probability,
            obstacle_reward=obstacle_reward,
            goal_reward=goal_reward,
            fuel_cost=fuel_cost,
            grid_size=grid_size,
            goal_state_radius=goal_state_radius,
            beacon_radius=beacon_radius,
            obstacle_radius=obstacle_radius,
            reward_model_type=reward_model_type,
            penalty_decay=penalty_decay,
        )

        # Override space info
        self.space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.CONTINUOUS
        )
        
        self.actions = ["up", "down", "right", "left"]
        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }

    def get_actions(self) -> List[Any]:
        return self.actions

    def state_transition_model(self, state: str, action: Any) -> Distribution:
        action_vector = self.action_to_vector[action]
        return super().state_transition_model(state, action_vector)

    def observation_model(self, next_state: str, action: Any) -> Distribution:
        action_vector = self.action_to_vector[action]
        return super().observation_model(next_state, action_vector)

    def reward(self, state: str, action: Any) -> float:
        action_vector = self.action_to_vector[action]
        return super().reward(state, action_vector)

    def __eq__(self, other):
        if not isinstance(other, ContinuousLightDarkPOMDPDiscreteActions):
            return False
        # Compare only configuration parameters, ignoring internal objects like reward_model
        return (
            self.discount_factor == other.discount_factor and
            np.array_equal(self.state_transition_cov_matrix, other.state_transition_cov_matrix) and
            np.array_equal(self.observation_cov_matrix, other.observation_cov_matrix) and
            np.array_equal(self.beacons, other.beacons) and
            np.array_equal(self.goal_state, other.goal_state) and
            np.array_equal(self.start_state, other.start_state) and
            np.array_equal(self.obstacles, other.obstacles) and
            self.obstacle_hit_probability == other.obstacle_hit_probability and
            self.obstacle_reward == other.obstacle_reward and
            self.goal_reward == other.goal_reward and
            self.fuel_cost == other.fuel_cost and
            self.grid_size == other.grid_size and
            self.goal_state_radius == other.goal_state_radius and
            self.beacon_radius == other.beacon_radius and
            self.obstacle_radius == other.obstacle_radius and
            self.penalty_decay == other.penalty_decay and
            self.actions == other.actions and
            all(np.array_equal(self.action_to_vector[k], other.action_to_vector[k]) for k in self.action_to_vector)
        )
