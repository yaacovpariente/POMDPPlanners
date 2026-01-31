from enum import Enum
from typing import Any, List, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    ObservationModel,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.base_light_dark_pomdp import (
    BaseLightDarkPOMDPDiscreteActions,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_observation_models import (
    DiscreteLDDistanceBasedObservationModel,
    DiscreteLDObservationModel,
    DiscreteLDObservationModelNoObsInDark,
)
from POMDPPlanners.utils.statistics_utils import confidence_interval


class DiscreteLightDarkPOMDPMetrics(Enum):
    """Metric names for Discrete Light-Dark POMDP environment."""

    GOAL_REACHING_RATE = "goal_reaching_rate"
    OBSTACLE_HIT_RATE = "obstacle_hit_rate"
    AVG_OBSTACLE_HIT_COUNTER = "avg_obstacle_hit_counter"
    OUT_OF_GRID_RATE = "out_of_grid_rate"
    AVG_DANGEROUS_STATES_COUNTER = "avg_dangerous_states_counter"


class ObservationModelType(Enum):
    NORMAL = "normal"
    NO_OBS_IN_DARK = "no_obs_in_dark"
    DISTANCE_BASED = "distance_based"


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
    - Multiple observation models available (normal, no observation in dark)
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
        observation_model_type: ObservationModelType = ObservationModelType.NORMAL,
    ):
        self.transition_error_prob = transition_error_prob
        self.observation_error_prob = observation_error_prob
        self.is_stochastic_reward = is_stochastic_reward
        self.observation_model_type = observation_model_type

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
        if self.observation_model_type == ObservationModelType.NORMAL:
            return DiscreteLDObservationModel(
                next_state=next_state,
                action=action,
                beacons=self.beacons,
                obstacles=self.obstacles,
                beacon_radius=self.beacon_radius,
                observation_error_prob=self.observation_error_prob,
            )
        elif self.observation_model_type == ObservationModelType.NO_OBS_IN_DARK:
            return DiscreteLDObservationModelNoObsInDark(
                next_state=next_state,
                action=action,
                beacons=self.beacons,
                obstacles=self.obstacles,
                beacon_radius=self.beacon_radius,
                observation_error_prob=self.observation_error_prob,
            )
        elif self.observation_model_type == ObservationModelType.DISTANCE_BASED:
            return DiscreteLDDistanceBasedObservationModel(
                next_state=next_state,
                action=action,
                beacons=self.beacons,
                obstacles=self.obstacles,
                beacon_radius=self.beacon_radius,
                observation_error_prob=self.observation_error_prob,
            )
        else:
            raise ValueError(f"Unknown observation model type: {self.observation_model_type}")

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

    def reward_batch(self, states: Union[np.ndarray, Sequence[Any]], action: str) -> np.ndarray:
        states = np.asarray(states)
        next_states = states + self.action_to_vector[action]
        dists_to_goal = np.linalg.norm(next_states - self.goal_state, axis=1)
        rewards = -self.fuel_cost - dists_to_goal

        goal_mask = np.all(next_states == self.goal_state, axis=1)
        rewards[goal_mask] += self.goal_reward

        obs_match = np.all(
            next_states[:, :, np.newaxis] == self.obstacles[np.newaxis, :, :],
            axis=1,
        )
        in_obstacle = np.any(obs_match, axis=1)
        obstacle_mask = in_obstacle & ~goal_mask
        n_obs = int(np.sum(obstacle_mask))
        if n_obs > 0:
            hits = np.random.rand(n_obs) < self.obstacle_hit_probability
            rewards[obstacle_mask] += np.where(hits, self.obstacle_reward, 0.0)

        oob = np.any(next_states < 0, axis=1) | np.any(next_states > self.grid_size, axis=1)
        rewards[oob & ~goal_mask & ~in_obstacle] += self.obstacle_reward
        return rewards

    def is_terminal(self, state: np.ndarray) -> bool:
        if state.shape != (2,):
            raise ValueError("state must be a 2D vector")

        is_goal_state = np.all(state == self.goal_state)
        is_obstacle_hit = np.any(np.all(state.reshape(-1, 1) == self.obstacles, axis=0))

        is_terminal = is_goal_state or is_obstacle_hit

        return bool(is_terminal)

    def get_metric_names(self) -> List[str]:
        """Get names of Discrete Light-Dark POMDP specific metrics.

        Returns:
            List containing metric names: goal_reaching_rate, obstacle_hit_rate,
            avg_obstacle_hit_counter, out_of_grid_rate, and avg_dangerous_states_counter
        """
        return [metric.value for metric in DiscreteLightDarkPOMDPMetrics]

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        goal_reached = []
        obstacle_hits = []
        obstacle_hit_counter = []
        out_of_grid = []
        dangerous_states_counter = []
        for history in histories:
            goal_reached_in_history = False
            obstacle_hit_in_history = False
            obstacle_hit_counter_in_history = 0
            out_of_grid_in_history = False
            out_of_grid_counter_in_history = 0

            for i, step in enumerate(history.history):
                if np.array_equal(step.state, self.goal_state):
                    goal_reached_in_history = True
                    break

                # Check if step hits an obstacle
                if np.any(np.all(step.state.reshape(-1, 1) == self.obstacles, axis=0)):
                    obstacle_hit_in_history = True
                    obstacle_hit_counter_in_history += 1

                # Check if step is out of grid
                is_out_of_grid = np.any(step.state < 0) or np.any(step.state > self.grid_size)
                if is_out_of_grid:
                    out_of_grid_in_history = True
                    out_of_grid_counter_in_history += 1

            goal_reached.append(1 if goal_reached_in_history else 0)
            obstacle_hits.append(1 if obstacle_hit_in_history else 0)
            obstacle_hit_counter.append(obstacle_hit_counter_in_history)
            out_of_grid.append(1 if out_of_grid_in_history else 0)
            # Sum obstacle hits and out-of-grid occurrences as dangerous states
            dangerous_states_counter.append(
                obstacle_hit_counter_in_history + out_of_grid_counter_in_history
            )

        avg_goal_reached = float(np.mean(goal_reached))
        avg_obstacle_hits = float(np.mean(obstacle_hits))
        avg_obstacle_hit_counter = float(np.mean(obstacle_hit_counter))
        avg_out_of_grid = float(np.mean(out_of_grid))
        avg_dangerous_states_counter = float(np.mean(dangerous_states_counter))
        goal_reached_ci = confidence_interval(data=goal_reached, confidence=0.95)
        obstacle_hits_ci = confidence_interval(data=obstacle_hits, confidence=0.95)
        obstacle_hit_counter_ci = confidence_interval(data=obstacle_hit_counter, confidence=0.95)
        out_of_grid_ci = confidence_interval(data=out_of_grid, confidence=0.95)
        dangerous_states_counter_ci = confidence_interval(
            data=dangerous_states_counter, confidence=0.95
        )

        return [
            MetricValue(
                name=DiscreteLightDarkPOMDPMetrics.GOAL_REACHING_RATE.value,
                value=avg_goal_reached,
                lower_confidence_bound=goal_reached_ci[0],
                upper_confidence_bound=goal_reached_ci[1],
            ),
            MetricValue(
                name=DiscreteLightDarkPOMDPMetrics.OBSTACLE_HIT_RATE.value,
                value=avg_obstacle_hits,
                lower_confidence_bound=obstacle_hits_ci[0],
                upper_confidence_bound=obstacle_hits_ci[1],
            ),
            MetricValue(
                name=DiscreteLightDarkPOMDPMetrics.AVG_OBSTACLE_HIT_COUNTER.value,
                value=avg_obstacle_hit_counter,
                lower_confidence_bound=obstacle_hit_counter_ci[0],
                upper_confidence_bound=obstacle_hit_counter_ci[1],
            ),
            MetricValue(
                name=DiscreteLightDarkPOMDPMetrics.OUT_OF_GRID_RATE.value,
                value=avg_out_of_grid,
                lower_confidence_bound=out_of_grid_ci[0],
                upper_confidence_bound=out_of_grid_ci[1],
            ),
            MetricValue(
                name=DiscreteLightDarkPOMDPMetrics.AVG_DANGEROUS_STATES_COUNTER.value,
                value=avg_dangerous_states_counter,
                lower_confidence_bound=dangerous_states_counter_ci[0],
                upper_confidence_bound=dangerous_states_counter_ci[1],
            ),
        ]
