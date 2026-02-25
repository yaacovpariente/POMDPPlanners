"""Continuous Light-Dark POMDP Environment Implementation.

This module implements the continuous Light-Dark domain, a classic POMDP benchmark
where an agent must navigate to a goal position in a continuous 2D space while
dealing with position-dependent observation noise.

The Continuous Light-Dark POMDP features:
- Continuous 2D state space representing agent position
- Discrete or continuous action space for movement
- Light source at a specific location that affects observation quality
- Observation noise that decreases closer to the light source
- Goal region that agent must reach to maximize reward
- Optional obstacles that cause negative rewards when hit

Key characteristics:
- State: [x, y] position in continuous 2D space
- Actions: Movement vectors or discrete directions
- Observations: Noisy position estimates (noise depends on distance from light)
- Rewards: Goal reaching bonus, movement costs, obstacle penalties
- Multiple reward model variants available

Classes:
    RewardModelType: Enumeration of available reward model types
    StateTransitionModel: Continuous movement with Gaussian noise
    ContinuousLightDarkPOMDP: Main environment class
    ContinuousLightDarkPOMDPDiscreteActions: Discrete action variant
"""

from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

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
)
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal

# pylint: disable=no-name-in-module
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_observation_models import (
    # Type checkers have trouble resolving this import despite the class being properly defined
    # and listed in __all__. The import works correctly at runtime.
    ContinuousLightDarkDistanceBasedObservationModel,  # pyright: ignore[reportAttributeAccessIssue]
    ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel,
    ContinuousLightDarkNormalNoiseObservationModel,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_reward_models import (
    BaseLightDarkRewardModel,
    ContinuousLDDangerousStatesRewardModel,
    ContinuousLightDarkDecayingHitProbabilityRewardModel,
    ContinuousLightDarkRewardModel,
)
from POMDPPlanners.utils.statistics_utils import confidence_interval


class ContinuousLightDarkPOMDPMetrics(Enum):
    """Metric names for Continuous Light-Dark POMDP environment."""

    GOAL_REACHING_RATE = "goal_reaching_rate"
    OBSTACLE_HIT_RATE = "obstacle_hit_rate"
    AVG_OBSTACLE_HIT_COUNTER = "avg_obstacle_hit_counter"
    OUT_OF_GRID_RATE = "out_of_grid_rate"
    AVG_DANGEROUS_STATES_COUNTER = "avg_dangerous_states_counter"


class RewardModelType(Enum):
    STANDARD = "standard"
    DECAYING_HIT_PROBABILITY = "decaying_hit_probability"
    DANGEROUS_STATES = "dangerous_states"


class ObservationModelType(Enum):
    NORMAL_NOISE = "normal_noise"
    NORMAL_NOISE_NO_OBS_IN_DARK = "normal_noise_no_obs_in_dark"
    DISTANCE_BASED = "distance_based"


class ContinuousLightDarkStateTransitionModel(StateTransitionModel):
    """State transition model for Continuous Light-Dark POMDP.

    This model implements continuous movement in 2D space with Gaussian noise.
    The agent's next position is determined by adding the action vector to the
    current position, with additional Gaussian noise to model uncertainty.

    Attributes:
        state: Current 2D position [x, y]
        action: Movement vector [dx, dy]
        mean: Expected next position (state + action)

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> # Define current position and movement action
        >>> state = np.array([3.0, 4.0])  # Current position
        >>> action = np.array([1.0, 0.5])  # Move right and slightly up
        >>>
        >>> # Define movement noise
        >>> cov_matrix = np.eye(2) * 0.1  # Small movement noise
        >>> state_dist = CovarianceParameterizedMultivariateNormal(cov_matrix)
        >>>
        >>> # Create transition model
        >>> transition = ContinuousLightDarkStateTransitionModel(
        ...     state=state,
        ...     action=action,
        ...     state_dist=state_dist
        ... )
        >>>
        >>> # Sample next position with noise
        >>> next_position = transition.sample()[0]  # doctest: +SKIP
        >>> # Returns position around [4.0, 4.5] ± noise
        >>>
        >>> # Calculate probability of specific next position
        >>> prob = transition.probability([next_position])  # doctest: +SKIP
    """

    def __init__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        state_dist: CovarianceParameterizedMultivariateNormal,
    ):
        super().__init__(state=state, action=action)
        self._state_dist = state_dist
        self.mean = state + action

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        samples = self._state_dist.sample(self.mean, n_samples)
        return list(samples)

    def probability(self, values: List[np.ndarray]) -> np.ndarray:
        values_array = np.array(values)
        return self._state_dist.pdf(values_array, self.mean)


class ContinuousLightDarkPOMDP(BaseLightDarkPOMDP):
    """Continuous Light-Dark POMDP environment with continuous actions.

    This environment extends the base Light-Dark problem to continuous 2D space
    with continuous action vectors. The agent navigates toward a goal while
    dealing with position-dependent observation noise and optional obstacles.

    Key features:
    - Continuous 2D state and action spaces
    - Light beacons reduce observation noise when nearby
    - Multiple observation models available (normal noise, normal noise with no observation in dark)
    - Multiple reward models available (standard, decaying hit probability, dangerous states)
    - Optional obstacles with configurable hit penalties
    - Terminal conditions for goal reaching, obstacle hits, and boundary violations

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> env = ContinuousLightDarkPOMDP(
        ...     discount_factor=0.95,
        ...     goal_state=np.array([10, 5]),
        ...     start_state=np.array([0, 5])
        ... )
        >>>
        >>> # Get initial state
        >>> initial_state = env.initial_state_dist().sample()[0]
        >>>
        >>> # Sample complete step (action must be provided based on environment type)
        >>> action = np.array([1.0, 0.0])  # Move right
        >>> next_state, observation, reward = env.sample_next_step(initial_state, action)
        >>>
        >>> # Check terminal condition
        >>> env.is_terminal(initial_state)
        False
    """

    def __init__(
        self,
        discount_factor: float,
        name: str = "ContinuousLightDarkPOMDP",
        state_transition_cov_matrix: np.ndarray = np.eye(2) * 0.05,
        observation_cov_matrix: np.ndarray = np.eye(2) * 0.05,
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
        fuel_cost: float = 2.0,
        grid_size: int = 11,
        goal_state_radius: float = 1.5,
        beacon_radius: float = 1.0,
        obstacle_radius: float = 1.5,
        reward_model_type: RewardModelType = RewardModelType.STANDARD,
        observation_model_type: ObservationModelType = ObservationModelType.NORMAL_NOISE,
        penalty_decay: float = 1.0,
        is_obstacle_hit_terminal: bool = True,
    ):
        space_info = SpaceInfo(
            action_space=SpaceType.CONTINUOUS, observation_space=SpaceType.CONTINUOUS
        )
        # Calculate reward range based on reward model type
        # Maximum distance to goal is diagonal of grid: sqrt(2) * grid_size
        max_distance_to_goal = np.sqrt(2) * grid_size

        if reward_model_type == RewardModelType.STANDARD:
            # Min: -fuel_cost - max_distance + obstacle_reward (always negative)
            # Max: -fuel_cost - 0 + goal_reward (at goal)
            min_reward = -fuel_cost - max_distance_to_goal + obstacle_reward
            max_reward = -fuel_cost + goal_reward
        elif reward_model_type == RewardModelType.DECAYING_HIT_PROBABILITY:
            # Similar to standard but with distance-based penalties
            # Min: -fuel_cost - max_distance + obstacle_reward (max penalty)
            # Max: -fuel_cost - 0 + goal_reward (at goal, no penalty)
            min_reward = -fuel_cost - max_distance_to_goal + obstacle_reward
            max_reward = -fuel_cost + goal_reward
        elif reward_model_type == RewardModelType.DANGEROUS_STATES:
            # Min: -fuel_cost - max_distance + obstacle_reward (negative)
            # Max: -fuel_cost - 0 + goal_reward OR -fuel_cost - distance - obstacle_reward (if obstacle_reward is negative)
            min_reward = -fuel_cost - max_distance_to_goal + obstacle_reward
            max_reward = max(-fuel_cost + goal_reward, -fuel_cost - obstacle_reward)
        else:
            # Default fallback
            min_reward = -fuel_cost - max_distance_to_goal + obstacle_reward
            max_reward = -fuel_cost + goal_reward

        calculated_reward_range = (min_reward, max_reward)

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=calculated_reward_range,
            beacons=beacons,
            goal_state=goal_state,
            start_state=start_state,
            obstacles=obstacles,
            obstacle_hit_probability=obstacle_hit_probability,
            obstacle_reward=obstacle_reward,
            goal_reward=goal_reward,
            beacon_radius=beacon_radius,
            obstacle_radius=obstacle_radius,
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
        self.observation_model_type = observation_model_type
        self.penalty_decay = penalty_decay
        self.is_obstacle_hit_terminal = is_obstacle_hit_terminal

        # Create distributions with pre-computed Cholesky decomposition
        self._state_transition_dist = CovarianceParameterizedMultivariateNormal(
            state_transition_cov_matrix
        )
        self._obs_dist_far_from_beacon = CovarianceParameterizedMultivariateNormal(
            observation_cov_matrix
        )
        self._obs_dist_near_beacon = CovarianceParameterizedMultivariateNormal(
            observation_cov_matrix * 0.5
        )

        # Initialize reward model based on type
        self.reward_model: BaseLightDarkRewardModel
        if reward_model_type == RewardModelType.STANDARD:
            self.reward_model = ContinuousLightDarkRewardModel(
                goal_state=self.goal_state,
                obstacles=self.obstacles,
                goal_state_radius=self.goal_state_radius,
                obstacle_radius=self.obstacle_radius,
                grid_size=self.grid_size,
                obstacle_hit_probability=self.obstacle_hit_probability,
                obstacle_reward=self.obstacle_reward,
                goal_reward=self.goal_reward,
                fuel_cost=self.fuel_cost,
            )
        elif reward_model_type == RewardModelType.DECAYING_HIT_PROBABILITY:
            self.reward_model = ContinuousLightDarkDecayingHitProbabilityRewardModel(
                goal_state=self.goal_state,
                obstacles=self.obstacles,
                goal_state_radius=self.goal_state_radius,
                obstacle_radius=self.obstacle_radius,
                grid_size=self.grid_size,
                obstacle_hit_probability=self.obstacle_hit_probability,
                obstacle_reward=self.obstacle_reward,
                goal_reward=self.goal_reward,
                fuel_cost=self.fuel_cost,
                penalty_decay=self.penalty_decay,
            )
        elif reward_model_type == RewardModelType.DANGEROUS_STATES:
            self.reward_model = ContinuousLDDangerousStatesRewardModel(
                goal_state=self.goal_state,
                obstacles=self.obstacles,
                goal_state_radius=self.goal_state_radius,
                obstacle_radius=self.obstacle_radius,
                grid_size=self.grid_size,
                obstacle_hit_probability=self.obstacle_hit_probability,
                obstacle_reward=self.obstacle_reward,
                goal_reward=self.goal_reward,
                fuel_cost=self.fuel_cost,
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
        if state_transition_cov_matrix.shape != (2, 2):
            raise ValueError("state_transition_cov_matrix must be a 2x2 matrix")
        if observation_cov_matrix.shape != (2, 2):
            raise ValueError("observation_cov_matrix must be a 2x2 matrix")
        if goal_state_radius <= 0:
            raise ValueError("goal_state_radius must be greater than 0")
        if beacon_radius <= 0:
            raise ValueError("beacon_radius must be greater than 0")
        if obstacle_radius <= 0:
            raise ValueError("obstacle_radius must be greater than 0")

    def state_transition_model(self, state: np.ndarray, action: np.ndarray) -> StateTransitionModel:
        if state.shape != (2,):
            raise ValueError("state must be a 2D vector")
        if action.shape != (2,):
            raise ValueError("action must be a 2D vector")

        return ContinuousLightDarkStateTransitionModel(  # type: ignore[return-value]
            state=state,
            action=action,
            state_dist=self._state_transition_dist,
        )

    def observation_model(self, next_state: np.ndarray, action: np.ndarray) -> ObservationModel:
        if next_state.shape != (2,):
            raise ValueError("next_state must be a 2D vector")
        if action.shape != (2,):
            raise ValueError("action must be a 2D vector")

        if self.observation_model_type == ObservationModelType.NORMAL_NOISE:
            return ContinuousLightDarkNormalNoiseObservationModel(
                next_state=next_state,
                action=action,
                obs_dist_near_beacon=self._obs_dist_near_beacon,
                obs_dist_far_from_beacon=self._obs_dist_far_from_beacon,
                grid_size=self.grid_size,
                beacons=self.beacons,
                beacon_radius=self.beacon_radius,
            )
        if self.observation_model_type == ObservationModelType.NORMAL_NOISE_NO_OBS_IN_DARK:
            return ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
                next_state=next_state,
                action=action,
                obs_dist_near_beacon=self._obs_dist_near_beacon,
                obs_dist_far_from_beacon=self._obs_dist_far_from_beacon,
                grid_size=self.grid_size,
                beacons=self.beacons,
                beacon_radius=self.beacon_radius,
            )
        if self.observation_model_type == ObservationModelType.DISTANCE_BASED:
            return ContinuousLightDarkDistanceBasedObservationModel(
                next_state=next_state,
                action=action,
                obs_dist_near_beacon=self._obs_dist_near_beacon,
                obs_dist_far_from_beacon=self._obs_dist_far_from_beacon,
                grid_size=self.grid_size,
                beacons=self.beacons,
                beacon_radius=self.beacon_radius,
            )
        raise ValueError(f"Unknown observation model type: {self.observation_model_type}")

    def reward(self, state: np.ndarray, action: np.ndarray) -> float:
        return self.reward_model.compute_reward(state, action)

    def reward_batch(
        self, states: Union[np.ndarray, Sequence[Any]], action: np.ndarray
    ) -> np.ndarray:
        return self.reward_model.compute_reward_batch(np.asarray(states), action)

    def is_terminal(self, state: np.ndarray) -> bool:
        if state.shape != (2,):
            raise ValueError("state must be a 2D vector")

        is_goal_state = np.linalg.norm(state - self.goal_state) <= self.goal_state_radius

        if self.is_obstacle_hit_terminal:
            # Calculate distance to each obstacle (obstacles are 2xN format)
            distances = np.linalg.norm(state.reshape(-1, 1) - self.obstacles, axis=0)
            is_obstacle_hit = bool(np.any(distances <= self.obstacle_radius))
        else:
            is_obstacle_hit = False

        is_terminal = is_goal_state or is_obstacle_hit

        return bool(is_terminal)

    def get_metric_names(self) -> List[str]:
        """Get names of Continuous Light-Dark POMDP specific metrics.

        Returns:
            List containing metric names: goal_reaching_rate, obstacle_hit_rate,
            avg_obstacle_hit_counter, out_of_grid_rate, and avg_dangerous_states_counter
        """
        return [metric.value for metric in ContinuousLightDarkPOMDPMetrics]

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

            for _, step in enumerate(history.history):
                if np.linalg.norm(step.state - self.goal_state) <= self.goal_state_radius:
                    goal_reached_in_history = True
                    break

                # Calculate distance to each obstacle (obstacles are 2xN format)
                distances = np.linalg.norm(step.state.reshape(-1, 1) - self.obstacles, axis=0)
                if np.any(distances <= self.obstacle_radius):
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
                name=ContinuousLightDarkPOMDPMetrics.GOAL_REACHING_RATE.value,
                value=avg_goal_reached,
                lower_confidence_bound=goal_reached_ci[0],
                upper_confidence_bound=goal_reached_ci[1],
            ),
            MetricValue(
                name=ContinuousLightDarkPOMDPMetrics.OBSTACLE_HIT_RATE.value,
                value=avg_obstacle_hits,
                lower_confidence_bound=obstacle_hits_ci[0],
                upper_confidence_bound=obstacle_hits_ci[1],
            ),
            MetricValue(
                name=ContinuousLightDarkPOMDPMetrics.AVG_OBSTACLE_HIT_COUNTER.value,
                value=avg_obstacle_hit_counter,
                lower_confidence_bound=obstacle_hit_counter_ci[0],
                upper_confidence_bound=obstacle_hit_counter_ci[1],
            ),
            MetricValue(
                name=ContinuousLightDarkPOMDPMetrics.OUT_OF_GRID_RATE.value,
                value=avg_out_of_grid,
                lower_confidence_bound=out_of_grid_ci[0],
                upper_confidence_bound=out_of_grid_ci[1],
            ),
            MetricValue(
                name=ContinuousLightDarkPOMDPMetrics.AVG_DANGEROUS_STATES_COUNTER.value,
                value=avg_dangerous_states_counter,
                lower_confidence_bound=dangerous_states_counter_ci[0],
                upper_confidence_bound=dangerous_states_counter_ci[1],
            ),
        ]

    def __eq__(self, other):
        if not isinstance(other, ContinuousLightDarkPOMDP):
            return False

        if not super().__eq__(other):
            return False

        return (
            np.array_equal(self.state_transition_cov_matrix, other.state_transition_cov_matrix)
            and np.array_equal(self.observation_cov_matrix, other.observation_cov_matrix)
            and self.goal_state_radius == other.goal_state_radius
            and self.beacon_radius == other.beacon_radius
            and self.obstacle_radius == other.obstacle_radius
            and self.observation_model_type == other.observation_model_type
        )


class ContinuousLightDarkPOMDPDiscreteActions(ContinuousLightDarkPOMDP, DiscreteActionsEnvironment):
    """Continuous Light-Dark POMDP environment with discrete actions.

    This variant of the Continuous Light-Dark POMDP uses discrete directional actions
    (up, down, left, right) instead of continuous action vectors. The continuous
    state space and observation model are preserved.

    Actions are mapped to unit vectors:
    - "up": [0, 1]
    - "down": [0, -1]
    - "right": [1, 0]
    - "left": [-1, 0]

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> # Initialize environment
        >>> env = ContinuousLightDarkPOMDPDiscreteActions(
        ...     discount_factor=0.95,
        ...     goal_state=np.array([10, 5]),
        ...     start_state=np.array([0, 5])
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
    """

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
        reward_model_type: RewardModelType = RewardModelType.STANDARD,
        observation_model_type: ObservationModelType = ObservationModelType.NORMAL_NOISE,
        penalty_decay: float = 1.0,
        is_obstacle_hit_terminal: bool = True,
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
            observation_model_type=observation_model_type,
            penalty_decay=penalty_decay,
            is_obstacle_hit_terminal=is_obstacle_hit_terminal,
        )

        # Override space info
        self.space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.CONTINUOUS
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

    def state_transition_model(self, state: np.ndarray, action: Any) -> StateTransitionModel:
        action_vector = self.action_to_vector[action]
        return super().state_transition_model(state, action_vector)

    def observation_model(self, next_state: np.ndarray, action: Any) -> ObservationModel:
        action_vector = self.action_to_vector[action]
        return super().observation_model(next_state, action_vector)

    def reward(self, state: np.ndarray, action: Any) -> float:
        action_vector = self.action_to_vector[action]
        return super().reward(state, action_vector)

    def reward_batch(self, states: Union[np.ndarray, Sequence[Any]], action: Any) -> np.ndarray:
        return super().reward_batch(np.asarray(states), self.action_to_vector[action])

    def __eq__(self, other):
        if not isinstance(other, ContinuousLightDarkPOMDPDiscreteActions):
            return False
        # Compare only configuration parameters, ignoring internal objects like reward_model
        return (
            self.discount_factor == other.discount_factor
            and np.array_equal(self.state_transition_cov_matrix, other.state_transition_cov_matrix)
            and np.array_equal(self.observation_cov_matrix, other.observation_cov_matrix)
            and np.array_equal(self.beacons, other.beacons)
            and np.array_equal(self.goal_state, other.goal_state)
            and np.array_equal(self.start_state, other.start_state)
            and np.array_equal(self.obstacles, other.obstacles)
            and self.obstacle_hit_probability == other.obstacle_hit_probability
            and self.obstacle_reward == other.obstacle_reward
            and self.goal_reward == other.goal_reward
            and self.fuel_cost == other.fuel_cost
            and self.grid_size == other.grid_size
            and self.goal_state_radius == other.goal_state_radius
            and self.beacon_radius == other.beacon_radius
            and self.obstacle_radius == other.obstacle_radius
            and self.observation_model_type == other.observation_model_type
            and self.penalty_decay == other.penalty_decay
            and self.actions == other.actions
            and all(
                np.array_equal(value, other.action_to_vector[k])
                for k, value in self.action_to_vector.items()
            )
        )
