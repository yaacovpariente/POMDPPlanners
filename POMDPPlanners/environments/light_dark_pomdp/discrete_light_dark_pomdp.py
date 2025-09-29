import hashlib
import json
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.base_light_dark_pomdp import (
    BaseLightDarkPOMDP,
    BaseLightDarkPOMDPDiscreteActions,
)
from POMDPPlanners.utils.statistics_utils import confidence_interval


class DiscreteLDObservationModel(ObservationModel):
    def __init__(
        self,
        next_state: np.ndarray,
        action: Any,
        beacons: np.ndarray,
        obstacles: np.ndarray,
        beacon_radius: float,
        observation_error_prob: float,
    ):
        self.next_state = next_state
        self.action = action
        self.beacons = beacons
        self.obstacles = obstacles
        self.beacon_radius = beacon_radius
        self.observation_error_prob = observation_error_prob
        self.actions = ["up", "down", "right", "left"]
        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }

        distances = np.linalg.norm(self.beacons - next_state[:, np.newaxis], axis=0)
        min_distance: float = float(np.min(distances))
        if min_distance < self.beacon_radius:
            beacon_error_factor = 0.2
        else:
            beacon_error_factor = 1.0

        values = [next_state + self.action_to_vector[action] for action in self.actions]
        values.append(next_state)

        observation_error_prob = self.observation_error_prob * beacon_error_factor
        probs = np.ones(len(values)) * (observation_error_prob / (len(values) - 1))
        probs[-1] = 1 - observation_error_prob

        self.distribution = DiscreteDistribution(values=values, probs=probs)

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        return self.distribution.sample(n_samples)

    def probability(self, values: List[Any]) -> np.ndarray:
        return self.distribution.probability(values)


class DiscreteLightDarkPOMDP(BaseLightDarkPOMDPDiscreteActions, DiscreteActionsEnvironment):
    """Discrete Light-Dark POMDP Environment for Robot Navigation with Observation Uncertainty.

    This environment implements a discretized version of the classic Light-Dark POMDP problem,
    where a robot must navigate from a start position to a goal position in a grid world
    with beacons and obstacles. The key challenge is that the robot's observation quality
    depends on its distance from beacons - closer to beacons means more accurate observations.

    Problem Description:
    The robot operates in a discrete grid world where it can move in four cardinal directions.
    The environment includes:
    - Beacons: Fixed positions that provide location reference with varying accuracy
    - Obstacles: Grid cells that incur penalties when hit
    - Goal: Target position that provides high reward when reached
    - Observation uncertainty: Decreases with proximity to beacons (light areas)

    Key Features:
    - Discrete state space: Robot positions are restricted to grid cells
    - Discrete action space: North, South, East, West movements
    - Distance-dependent observation accuracy: Closer to beacons = better observations
    - Stochastic transitions: Actions may fail with configurable probability
    - Obstacle avoidance: Penalties for hitting obstacles during navigation
    - Configurable environment parameters: Grid size, beacon positions, obstacles

    State Space:
    - 2D grid coordinates (x, y) representing robot position
    - Bounded by grid_size parameter (default: 11x11 grid)

    Action Space:
    - Discrete actions: ['North', 'South', 'East', 'West']
    - Each action moves robot one grid cell in the corresponding direction
    - Boundary conditions: Actions that would move outside grid are blocked

    Observation Space:
    - Discrete observations based on beacon proximity and noise
    - Observation accuracy improves with proximity to beacons
    - Stochastic observation errors controlled by observation_error_prob

    Reward Structure:
    - Goal reward: Large positive reward for reaching the goal state
    - Obstacle penalty: Negative reward for hitting obstacles
    - Fuel cost: Small negative reward for each movement action
    - Distance-based penalties: Encourage efficient navigation

    Attributes:
        transition_error_prob: Probability that an action fails (results in different movement)
        observation_error_prob: Probability of observation noise/error
        is_stochastic_reward: Whether rewards include stochastic components
        beacons: List of (x, y) beacon positions that provide navigation references
        goal_state: Target position (x, y) that robot should reach
        start_state: Initial robot position (x, y)
        obstacles: List of (x, y) obstacle positions to avoid
        grid_size: Dimension of the square grid world

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> env = DiscreteLightDarkPOMDP(
        ...     discount_factor=0.95,
        ...     transition_error_prob=0.1,
        ...     observation_error_prob=0.15,
        ...     beacons=[(1, 1), (2, 2)],
        ...     grid_size=11
        ... )
        >>>
        >>> # Get initial state and actions
        >>> initial_state = env.initial_state_dist().sample()[0]
        >>> actions = env.get_actions()
        >>>
        >>> # Sample complete step using convenience method
        >>> action = actions[0]
        >>> next_state, observation, reward = env.sample_next_step(initial_state, action)
        >>>
        >>> # Check terminal condition
        >>> env.is_terminal(initial_state)
        False

    References:
    - Platt, R., et al. "Belief space planning assuming maximum likelihood observations." (2010)
    - Kurniawati, H., et al. "SARSOP: Efficient point-based POMDP planning by approximating optimally reachable belief spaces." (2008)
    - Light-Dark domain: Classic POMDP benchmark for testing observation uncertainty
    """

    def __init__(
        self,
        discount_factor: float,
        name: str = "DiscreteLightDarkPOMDP",
        transition_error_prob: float = 0.05,
        observation_error_prob: float = 0.05,
        beacons: List[Tuple[float, float]] = [
            (0, 0),
            (0, 5),
            (0, 10),
            (5, 0),
            (5, 5),
            (5, 10),
            (10, 0),
            (10, 5),
            (10, 10),
        ],
        goal_state: np.ndarray = np.array([10, 5]),
        start_state: np.ndarray = np.array([0, 5]),
        obstacles: List[Tuple[float, float]] = [(3, 7), (5, 5)],
        obstacle_hit_probability: float = 0.2,
        obstacle_reward: float = -10.0,
        goal_reward: float = 10.0,
        beacon_radius: float = 1.0,
        fuel_cost: float = 2.0,
        grid_size: int = 11,
        is_stochastic_reward: bool = True,
    ):
        self.transition_error_prob = transition_error_prob
        self.observation_error_prob = observation_error_prob
        self.is_stochastic_reward = is_stochastic_reward

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            is_discrete_observations=True,
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

    def state_transition_model(self, state: np.ndarray, action: Any) -> StateTransitionModel:
        action_index = self.actions.index(action)
        values = [state + self.action_to_vector[action] for action in self.actions]

        # Distribute error probability equally among other actions
        probs = np.ones(len(values)) * (self.transition_error_prob / (len(self.actions) - 1))
        probs[action_index] = 1 - self.transition_error_prob
        s = sum(probs)
        probs[0] += 1 - s

        return DiscreteDistribution(values, probs)  # type: ignore[return-value]

    def observation_model(self, next_state: np.ndarray, action: Any) -> ObservationModel:
        return DiscreteLDObservationModel(
            next_state=next_state,
            action=action,
            beacons=self.beacons,
            obstacles=self.obstacles,
            beacon_radius=self.beacon_radius,
            observation_error_prob=self.observation_error_prob,
        )

    def reward(self, state: np.ndarray, action: Any) -> float:
        if state.shape != (2,):
            raise ValueError("state must be a 2D vector")

        next_state = state + self.action_to_vector[action]

        is_goal_state = np.all(next_state == self.goal_state)
        is_obstacle_hit = np.any(np.all(next_state.reshape(-1, 1) == self.obstacles, axis=0))
        is_out_of_grid = np.any(next_state < 0) or np.any(next_state > self.grid_size)

        # Start with base reward (fuel cost)
        reward = -self.fuel_cost - np.linalg.norm(next_state - self.goal_state)

        if is_goal_state:
            reward += self.goal_reward
        elif is_obstacle_hit:
            if np.random.rand() < self.obstacle_hit_probability:
                reward += self.obstacle_reward
        elif is_out_of_grid:
            reward += self.obstacle_reward

        return float(reward)

    def is_terminal(self, state: np.ndarray) -> bool:
        if state.shape != (2,):
            raise ValueError("state must be a 2D vector")

        is_goal_state = np.all(state == self.goal_state)
        is_obstacle_hit = np.any(np.all(state.reshape(-1, 1) == self.obstacles, axis=0))
        is_out_of_grid = np.any(state < 0) or np.any(state > self.grid_size)

        is_terminal = is_goal_state or is_obstacle_hit or is_out_of_grid

        return bool(is_terminal)

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        # Calculate time to reach goal for each history
        goal_reached = []
        obstacle_hits = []
        for history in histories:
            goal_reached_in_history = False
            obstacle_hit_in_history = False

            for i, step in enumerate(history.history):
                if np.array_equal(step.state, self.goal_state):
                    goal_reached_in_history = True
                    break

                if np.any(np.all(step.state.reshape(-1, 1) == self.obstacles, axis=0)):
                    obstacle_hit_in_history = True

            goal_reached.append(1 if goal_reached_in_history else 0)
            obstacle_hits.append(1 if obstacle_hit_in_history else 0)

        avg_goal_reached = float(np.mean(goal_reached))
        avg_obstacle_hits = float(np.mean(obstacle_hits))

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
            ),
        ]
