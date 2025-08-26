"""Push POMDP Environment Implementation.

This module implements a robotic push task as a POMDP, where a robot must
push an object to a target location on a 2D grid. The robot can move in
four directions and pushes objects when within range, with noisy observations
of the object's position.

The Push POMDP features:
- Continuous 2D state space: [robot_x, robot_y, object_x, object_y, target_x, target_y]
- Discrete action space: ["up", "down", "left", "right"]
- Noisy observations of object position (robot and target positions are known)
- Physics-based pushing mechanics with friction
- Distance-based rewards encouraging object movement toward target

Key mechanics:
- Robot must be within push_threshold distance to move objects
- Friction reduces the effectiveness of pushes
- Object position observations include Gaussian noise
- Episode terminates when object reaches target

Classes:
    PushStateTransition: Physics-based pushing dynamics
    PushObservation: Noisy object position observations
    PushPOMDP: Main push task environment with POMDP formulation
"""

from typing import List, Any, Tuple, Optional
import numpy as np
from pathlib import Path

from POMDPPlanners.core.environment import (
    ObservationModel,
    StateTransitionModel,
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType
)
from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.simulation import History, StepData


class PushStateTransition(StateTransitionModel):
    """State transition model for Push POMDP with physics-based pushing.
    
    This model implements robot movement and object pushing dynamics on a 2D grid.
    The robot moves according to discrete actions, and can push objects when
    within the push threshold distance. Friction reduces push effectiveness.
    
    State representation: [robot_x, robot_y, object_x, object_y, target_x, target_y]
    
    Attributes:
        state: Current state vector containing all entity positions
        action: Movement action ("up", "down", "left", "right")
        grid_size: Size of the grid environment
        push_threshold: Maximum distance for robot to push object
        friction_coefficient: Friction that reduces push force (0=no friction, 1=max friction)
        robot_pos: Current robot position [x, y]
        object_pos: Current object position [x, y]
        target_pos: Target position [x, y] (fixed)
        
    Example:
        Using the Push state transition model::
        
            import numpy as np
            
            # Define state: [robot_x, robot_y, object_x, object_y, target_x, target_y]
            state = np.array([2.0, 3.0, 2.5, 3.1, 8.0, 8.0])
            action = "right"  # Move robot right
            
            # Create transition model
            transition = PushStateTransition(
                state=state,
                action=action,
                grid_size=10,
                push_threshold=1.0,
                friction_coefficient=0.3
            )
            
            # Simulate step
            next_state = transition.sample()[0]
            # Robot moves right, might push object if close enough
            # Returns new [robot_x, robot_y, object_x, object_y, target_x, target_y]
    """
    
    def __init__(
        self,
        state: np.ndarray,
        action: str,
        grid_size: int,
        push_threshold: float,
        friction_coefficient: float,
    ):
        super().__init__(state, action)
        self.grid_size = grid_size
        self.push_threshold = push_threshold
        self.friction_coefficient = friction_coefficient
        
        # State components: [robot_x, robot_y, object_x, object_y, target_x, target_y]
        self.robot_pos = state[:2]
        self.object_pos = state[2:4]
        self.target_pos = state[4:6]
        
        # Action to movement mapping
        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }

    def sample(self, n_samples: int = 1) -> List[Any]:
        # Get movement vector for the action
        movement = self.action_to_vector[self.action]
        
        # Calculate new robot position
        new_robot_pos = self.robot_pos + movement
        
        # Check if robot is close enough to push object
        distance_to_object = np.linalg.norm(new_robot_pos - self.object_pos)
        
        if distance_to_object < self.push_threshold:
            # Calculate push force based on friction
            push_force = movement * (1 - self.friction_coefficient)
            new_object_pos = self.object_pos + push_force
        else:
            new_object_pos = self.object_pos
            
        # Ensure positions stay within grid bounds
        new_robot_pos = np.clip(new_robot_pos, 0, self.grid_size - 1)
        new_object_pos = np.clip(new_object_pos, 0, self.grid_size - 1)
        
        # Combine all state components
        next_state = np.concatenate([new_robot_pos, new_object_pos, self.target_pos])
        return [next_state] * n_samples


class PushObservation(ObservationModel):
    """Noisy observation model for Push POMDP.
    
    This model provides partial observability by adding Gaussian noise to
    the object's position while keeping robot and target positions fully observable.
    This creates uncertainty about the exact object location, making planning challenging.
    
    Observation format: [robot_x, robot_y, noisy_object_x, noisy_object_y, target_x, target_y]
    
    Attributes:
        next_state: True state after action execution
        action: Action that was taken (not used in observation generation)
        observation_noise: Standard deviation of Gaussian noise for object position
        grid_size: Size of the grid environment
        robot_pos: Robot position (observed exactly)
        object_pos: True object position (observed with noise)
        target_pos: Target position (observed exactly)
        
    Example:
        Using the Push observation model::
        
            import numpy as np
            
            # True state after robot movement
            true_state = np.array([3.0, 3.0, 2.8, 3.2, 8.0, 8.0])
            action = "right"
            
            # Create observation model
            obs_model = PushObservation(
                next_state=true_state,
                action=action,
                observation_noise=0.1,
                grid_size=10
            )
            
            # Sample noisy observation
            observation = obs_model.sample()[0]
            # Robot and target positions exact: [3.0, 3.0, ?, ?, 8.0, 8.0]
            # Object position noisy: around [2.8, 3.2] ± 0.1
            
            # Calculate observation probability
            prob = obs_model.probability(observation)
    """
    
    def __init__(
        self,
        next_state: np.ndarray,
        action: str,
        observation_noise: float,
        grid_size: int,
    ):
        super().__init__(next_state=next_state, action=action)
        self.observation_noise = observation_noise
        self.grid_size = grid_size
        
        # State components: [robot_x, robot_y, object_x, object_y, target_x, target_y]
        self.robot_pos = next_state[:2]
        self.object_pos = next_state[2:4]
        self.target_pos = next_state[4:6]

    def sample(self, n_samples: int = 1) -> List[Any]:
        observations = []
        for _ in range(n_samples):
            # Add noise to object position observation
            noisy_object_pos = self.object_pos + np.random.normal(0, self.observation_noise, size=2)
            noisy_object_pos = np.clip(noisy_object_pos, 0, self.grid_size - 1)
            
            # Combine observations (robot position is known exactly, target position is known)
            observation = np.concatenate([self.robot_pos, noisy_object_pos, self.target_pos])
            observations.append(observation)
        return observations

    def probability(self, observation: np.ndarray) -> float:
        # Calculate probability based on Gaussian noise model
        object_pos_diff = observation[2:4] - self.object_pos
        log_prob = -0.5 * np.sum(object_pos_diff**2) / (self.observation_noise**2)
        return np.exp(log_prob)


class PushPOMDP(DiscreteActionsEnvironment):
    """Robotic push task formulated as a POMDP.
    
    This environment simulates a robot that must push an object to a target location
    on a 2D grid. The robot can move in four directions and pushes objects when close
    enough, with partial observability through noisy object position measurements.
    
    Problem Structure:
    - State: [robot_x, robot_y, object_x, object_y, target_x, target_y] (continuous)
    - Actions: ["up", "down", "left", "right"] (discrete)
    - Observations: [robot_x, robot_y, noisy_object_x, noisy_object_y, target_x, target_y]
    - Rewards: -distance_to_target + 100 (when object reaches target)
    - Termination: Object within 0.5 units of target position
    
    Key Features:
    - Physics-based pushing with configurable friction
    - Distance-based pushing threshold
    - Noisy observations of object position only
    - Dense reward signal based on object-target distance
    
    Example:
        Creating and using a Push POMDP::
        
            # Create push environment
            push_env = PushPOMDP(
                discount_factor=0.99,
                grid_size=10,
                push_threshold=1.0,
                friction_coefficient=0.3,
                observation_noise=0.1
            )
            
            # Get initial state
            initial_state_dist = push_env.initial_state_dist()
            state = initial_state_dist.sample()[0]
            
            # Move robot and potentially push object
            actions = push_env.get_actions()  # ["up", "down", "left", "right"]
            action = "right"
            reward = push_env.reward(state, action)
            
            # Check if object reached target
            is_done = push_env.is_terminal(state)
    """
    
    def __init__(
        self,
        discount_factor: float,
        grid_size: int = 10,
        push_threshold: float = 1.0,
        friction_coefficient: float = 0.3,
        observation_noise: float = 0.1,
        name: str = "PushPOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
    ):
        self.grid_size = grid_size
        self.push_threshold = push_threshold
        self.friction_coefficient = friction_coefficient
        self.observation_noise = observation_noise
        
        # Define actions
        self.actions = ["up", "down", "right", "left"]
        
        # Initialize target position (fixed)
        self.target_pos = np.array([grid_size - 1, grid_size - 1])

        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # Action space is discrete positions
            observation_space=SpaceType.CONTINUOUS  # Observation space is positions with noise
        )
        super().__init__(discount_factor=discount_factor, name=name, space_info=space_info, 
                        reward_range=(-2 * np.sqrt(self.grid_size), 100.0), output_dir=output_dir, debug=debug)

    def state_transition_model(self, state: np.ndarray, action: str) -> StateTransitionModel:
        return PushStateTransition(
            state=state,
            action=action,
            grid_size=self.grid_size,
            push_threshold=self.push_threshold,
            friction_coefficient=self.friction_coefficient,
        )

    def observation_model(self, next_state: np.ndarray, action: str) -> ObservationModel:
        return PushObservation(
            next_state=next_state,
            action=action,
            observation_noise=self.observation_noise,
            grid_size=self.grid_size,
        )

    def reward(self, state: np.ndarray, action: str) -> float:
        # State components: [robot_x, robot_y, object_x, object_y, target_x, target_y]
        object_pos = state[2:4]
        target_pos = state[4:6]
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(object_pos - target_pos)
        
        # Base reward is negative distance to encourage moving closer to target
        reward = -distance_to_target
        
        # Additional reward for reaching target
        if distance_to_target < 0.5:
            reward += 100.0
            
        return reward

    def is_terminal(self, state: np.ndarray) -> bool:
        object_pos = state[2:4]
        target_pos = state[4:6]
        
        # Episode ends when object is close to target
        return np.linalg.norm(object_pos - target_pos) < 0.5

    def initial_state_dist(self) -> Distribution:
        class InitialState(Distribution):
            def __init__(self, grid_size: int, target_pos: np.ndarray):
                self.grid_size = grid_size
                self.target_pos = target_pos
                
            def sample(self, n_samples: int = 1) -> List[Any]:
                initial_states = []
                for _ in range(n_samples):
                    # Random initial positions for robot and object
                    robot_pos = np.random.uniform(0, self.grid_size - 1, size=2)
                    object_pos = np.random.uniform(0, self.grid_size - 1, size=2)
                    
                    # Ensure object is not too close to target initially
                    while np.linalg.norm(object_pos - self.target_pos) < 2.0:
                        object_pos = np.random.uniform(0, self.grid_size - 1, size=2)
                    
                    initial_state = np.concatenate([robot_pos, object_pos, self.target_pos])
                    initial_states.append(initial_state)
                return initial_states
        
        return InitialState(self.grid_size, self.target_pos)

    def initial_observation_dist(self) -> Distribution:
        return self.initial_state_dist()

    def get_actions(self) -> List[str]:
        return self.actions

    def is_equal_observation(self, observation1: np.ndarray, observation2: np.ndarray) -> bool:
        return np.array_equal(observation1, observation2)

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        # TODO: Implement visualization
        pass 