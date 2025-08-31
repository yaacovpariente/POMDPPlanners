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
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from POMDPPlanners.core.environment import (
    ObservationModel,
    StateTransitionModel,
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType
)
from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.simulation import History, StepData, MetricValue
from POMDPPlanners.utils.statistics import confidence_interval


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
        obstacles: List[Tuple[float, float]] = None,
        obstacle_radius: float = 0.5,
    ):
        super().__init__(state, action)
        self.grid_size = grid_size
        self.push_threshold = push_threshold
        self.friction_coefficient = friction_coefficient
        self.obstacles = obstacles if obstacles is not None else []
        self.obstacle_radius = obstacle_radius
        
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
        
        # Calculate intended new robot position
        intended_robot_pos = self.robot_pos + movement
        
        # Check for collision with obstacles - if colliding, robot doesn't move
        if self._is_colliding_with_obstacle(intended_robot_pos):
            new_robot_pos = self.robot_pos  # Robot stays in place
        else:
            new_robot_pos = intended_robot_pos
        
        # Check if robot is close enough to push object
        distance_to_object = np.linalg.norm(new_robot_pos - self.object_pos)
        
        if distance_to_object < self.push_threshold:
            # Calculate intended object position after push
            push_force = movement * (1 - self.friction_coefficient)
            intended_object_pos = self.object_pos + push_force
            
            # Check for obstacle collision for object - if colliding, object doesn't move
            if self._is_colliding_with_obstacle(intended_object_pos):
                new_object_pos = self.object_pos  # Object stays in place
            else:
                new_object_pos = intended_object_pos
        else:
            new_object_pos = self.object_pos
            
        # Ensure positions stay within grid bounds
        new_robot_pos = np.clip(new_robot_pos, 0, self.grid_size - 1)
        new_object_pos = np.clip(new_object_pos, 0, self.grid_size - 1)
        
        # Combine all state components
        next_state = np.concatenate([new_robot_pos, new_object_pos, self.target_pos])
        return [next_state] * n_samples

    def _is_colliding_with_obstacle(self, position: np.ndarray) -> bool:
        """Check if a position collides with any obstacle."""
        if not self.obstacles:
            return False
            
        pos_x, pos_y = position
        
        for obs_x, obs_y in self.obstacles:
            # Calculate Euclidean distance
            distance = np.sqrt((pos_x - obs_x)**2 + (pos_y - obs_y)**2)
            if distance <= self.obstacle_radius:
                return True
        
        return False


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

    def probability(self, observations: List[Any]) -> np.ndarray:
        # Calculate probabilities based on Gaussian noise model for list of observations
        probabilities = []
        for observation in observations:
            # Ensure observation is numpy array with correct shape
            if not isinstance(observation, np.ndarray) or observation.size == 0:
                raise ValueError(f"Expected non-empty numpy array observation, got {type(observation)} with shape {getattr(observation, 'shape', 'unknown')}")
            
            if observation.shape != (6,):
                raise ValueError(f"Expected observation shape (6,), got {observation.shape}")
            
            object_pos_diff = observation[2:4] - self.object_pos
            log_prob = -0.5 * np.sum(object_pos_diff**2) / (self.observation_noise**2)
            prob = np.exp(log_prob)
            probabilities.append(prob)
        
        return np.array(probabilities)


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
    - Obstacle collision detection with configurable penalties
    - Obstacles prevent robot and object movement through them
    
    Example:
        Creating and using a Push POMDP::
        
            # Create push environment with obstacles
            push_env = PushPOMDP(
                discount_factor=0.99,
                grid_size=10,
                push_threshold=1.0,
                friction_coefficient=0.3,
                observation_noise=0.1,
                obstacles=[(3.0, 4.0), (6.0, 7.0)],  # Obstacle positions
                obstacle_radius=0.5,
                obstacle_penalty=-10.0
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
        obstacles: Optional[List[Tuple[float, float]]] = None,
        obstacle_radius: float = 0.5,
        obstacle_penalty: float = -10.0,
        name: str = "PushPOMDP",
        output_dir: Optional[Path] = None,
        debug: bool = False,
    ):
        self.grid_size = grid_size
        self.push_threshold = push_threshold
        self.friction_coefficient = friction_coefficient
        self.observation_noise = observation_noise
        self.obstacles: List[Tuple[float, float]] = obstacles if obstacles is not None else []
        self.obstacle_radius = obstacle_radius
        self.obstacle_penalty = obstacle_penalty
        
        # Define actions
        self.actions = ["up", "down", "right", "left"]
        
        # Initialize target position (fixed)
        self.target_pos = np.array([grid_size - 1, grid_size - 1])

        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # Action space is discrete positions
            observation_space=SpaceType.CONTINUOUS  # Observation space is positions with noise
        )
        # Calculate reward range based on maximum distance to target
        # Maximum distance is diagonal from corner to corner: sqrt(2) * (grid_size - 1)
        max_distance = np.sqrt(2) * (grid_size - 1)
        min_reward = -max_distance  # Worst case: maximum distance to target
        max_reward = 100.0  # Best case: at target with bonus reward
        
        super().__init__(discount_factor=discount_factor, name=name, space_info=space_info, 
                        reward_range=(min_reward, max_reward), output_dir=output_dir, debug=debug)

    def _is_colliding_with_obstacle(self, position: np.ndarray) -> bool:
        """Check if a position collides with any obstacle.
        
        Args:
            position: Position to check as [x, y] array
            
        Returns:
            True if position is within obstacle_radius of any obstacle center
        """
        if not self.obstacles:
            return False
            
        pos_x, pos_y = position
        
        for obs_x, obs_y in self.obstacles:
            # Calculate Euclidean distance
            distance = np.sqrt((pos_x - obs_x)**2 + (pos_y - obs_y)**2)
            if distance <= self.obstacle_radius:
                return True
        
        return False

    def state_transition_model(self, state: np.ndarray, action: str) -> StateTransitionModel:
        return PushStateTransition(
            state=state,
            action=action,
            grid_size=self.grid_size,
            push_threshold=self.push_threshold,
            friction_coefficient=self.friction_coefficient,
            obstacles=self.obstacles,
            obstacle_radius=self.obstacle_radius,
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
        robot_pos = state[:2]
        object_pos = state[2:4]
        target_pos = state[4:6]
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(object_pos - target_pos)
        
        # Base reward is negative distance to encourage moving closer to target
        reward = -distance_to_target
        
        # Additional reward for reaching target
        if distance_to_target < 0.5:
            reward += 100.0
        
        # Apply obstacle penalty if robot or object is colliding with obstacles
        if self._is_colliding_with_obstacle(robot_pos):
            reward += self.obstacle_penalty
        
        if self._is_colliding_with_obstacle(object_pos):
            reward += self.obstacle_penalty
            
        return reward

    def is_terminal(self, state: np.ndarray) -> bool:
        object_pos = state[2:4]
        target_pos = state[4:6]
        
        # Episode ends when object is close to target
        return np.linalg.norm(object_pos - target_pos) < 0.5

    def initial_state_dist(self) -> Distribution:
        class InitialState(Distribution):
            def __init__(self, grid_size: int, target_pos: np.ndarray, obstacles: List[Tuple[float, float]], 
                        obstacle_radius: float, parent: 'PushPOMDP'):
                self.grid_size = grid_size
                self.target_pos = target_pos
                self.obstacles = obstacles
                self.obstacle_radius = obstacle_radius
                self.parent = parent
                
            def sample(self, n_samples: int = 1) -> List[Any]:
                initial_states = []
                for _ in range(n_samples):
                    # Random initial positions for robot and object, avoiding obstacles
                    max_attempts = 100
                    
                    # Generate robot position
                    attempts = 0
                    while attempts < max_attempts:
                        robot_pos = np.random.uniform(0, self.grid_size - 1, size=2)
                        if not self.parent._is_colliding_with_obstacle(robot_pos):
                            break
                        attempts += 1
                    
                    # Generate object position
                    attempts = 0
                    while attempts < max_attempts:
                        object_pos = np.random.uniform(0, self.grid_size - 1, size=2)
                        # Ensure object is not too close to target initially and not in obstacles
                        if (np.linalg.norm(object_pos - self.target_pos) >= 2.0 and 
                            not self.parent._is_colliding_with_obstacle(object_pos)):
                            break
                        attempts += 1
                    
                    initial_state = np.concatenate([robot_pos, object_pos, self.target_pos])
                    initial_states.append(initial_state)
                return initial_states
        
        return InitialState(self.grid_size, self.target_pos, self.obstacles, self.obstacle_radius, self)

    def initial_observation_dist(self) -> Distribution:
        return self.initial_state_dist()

    def get_actions(self) -> List[str]:
        return self.actions

    def is_equal_observation(self, observation1: np.ndarray, observation2: np.ndarray) -> bool:
        return np.array_equal(observation1, observation2)

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache animated visualization of the push episode.
        
        Creates an animated GIF showing the robot pushing the object toward the target,
        with obstacles, collision detection, distance indicators, and success feedback.
        
        Args:
            history: Episode history containing states, actions, and rewards
            cache_path: Path where to save the visualization (must end with .gif)
            
        Raises:
            ValueError: If history is empty or cache_path doesn't end with .gif
            TypeError: If cache_path is not a Path object
        """
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")
        if not history:
            raise ValueError("Cannot visualize empty history")
            
        # Extract episode data
        states = [step.state for step in history]
        actions = [step.action for step in history[:-1]]  # Last step has no action
        rewards = [step.reward for step in history]
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(-0.5, self.grid_size + 0.5)
        ax.set_ylim(-0.5, self.grid_size + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title('Push POMDP Episode Visualization', fontsize=14, fontweight='bold')
        
        # Initialize visual elements
        robot_scatter = ax.scatter([], [], s=200, c='blue', marker='o', 
                                 edgecolor='darkblue', linewidth=2, zorder=5, label='Robot')
        object_scatter = ax.scatter([], [], s=180, c='orange', marker='s', 
                                  edgecolor='darkorange', linewidth=2, zorder=4, label='Object')
        target_scatter = ax.scatter([], [], s=250, c='gold', marker='*', 
                                  edgecolor='darkgoldenrod', linewidth=2, zorder=3, label='Target')
        
        # Plot obstacles as permanent features
        obstacle_scatters = []
        for i, (obs_x, obs_y) in enumerate(self.obstacles):
            obstacle_circle = plt.Circle((obs_x, obs_y), self.obstacle_radius, 
                                       facecolor='red', edgecolor='darkred', 
                                       alpha=0.6, linewidth=2, zorder=2)
            ax.add_patch(obstacle_circle)
            if i == 0:  # Label only the first obstacle for legend
                ax.scatter(obs_x, obs_y, s=1, c='red', label='Obstacles')
        
        # Initialize path traces
        robot_path_line, = ax.plot([], [], 'b-', alpha=0.4, linewidth=2, label='Robot Path')
        object_path_line, = ax.plot([], [], 'orange', linestyle='--', alpha=0.4, linewidth=2, label='Object Path')
        
        # Initialize push vector arrow
        push_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                               arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.8),
                               zorder=6, visible=False)
        
        # Initialize connection line between robot and object during pushes
        connection_line, = ax.plot([], [], 'r-', alpha=0.6, linewidth=2, zorder=1)
        
        # Text displays
        step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                          verticalalignment='top', horizontalalignment='left')
        
        distance_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=11,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                              verticalalignment='top', horizontalalignment='left')
        
        reward_text = ax.text(0.02, 0.78, '', transform=ax.transAxes, fontsize=11,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                            verticalalignment='top', horizontalalignment='left')
        
        # Success banner (initially hidden)
        success_text = ax.text(0.5, 0.5, '', transform=ax.transAxes, fontsize=20, 
                             fontweight='bold', color='darkgreen',
                             bbox=dict(boxstyle='round,pad=1.0', facecolor='lightgreen', 
                                     edgecolor='darkgreen', linewidth=3, alpha=0.9),
                             horizontalalignment='center', verticalalignment='center',
                             visible=False, zorder=10)
        
        # Collision indicator (initially hidden)
        collision_text = ax.text(0.5, 0.3, '', transform=ax.transAxes, fontsize=16,
                               fontweight='bold', color='darkred',
                               bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcoral',
                                       edgecolor='darkred', linewidth=2, alpha=0.9),
                               horizontalalignment='center', verticalalignment='center',
                               visible=False, zorder=10)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        def animate(frame):
            if frame >= len(states):
                return (robot_scatter, object_scatter, target_scatter, robot_path_line, 
                       object_path_line, push_arrow, connection_line, step_text, 
                       distance_text, reward_text, success_text, collision_text)
            
            state = states[frame]
            robot_pos = state[:2]
            object_pos = state[2:4]
            target_pos = state[4:6]
            
            # Update entity positions
            robot_scatter.set_offsets([robot_pos])
            object_scatter.set_offsets([object_pos])
            target_scatter.set_offsets([target_pos])
            
            # Update path traces
            robot_path = [s[:2] for s in states[:frame+1]]
            object_path = [s[2:4] for s in states[:frame+1]]
            
            if len(robot_path) > 1:
                robot_x = [pos[0] for pos in robot_path]
                robot_y = [pos[1] for pos in robot_path]
                robot_path_line.set_data(robot_x, robot_y)
                
                object_x = [pos[0] for pos in object_path]
                object_y = [pos[1] for pos in object_path]
                object_path_line.set_data(object_x, object_y)
            
            # Calculate metrics for this frame
            distance_to_target = np.linalg.norm(object_pos - target_pos)
            robot_to_object_dist = np.linalg.norm(robot_pos - object_pos)
            
            # Check for collisions
            robot_collision = self._is_colliding_with_obstacle(robot_pos)
            object_collision = self._is_colliding_with_obstacle(object_pos)
            
            # Update push visualization
            if frame < len(actions):
                action = actions[frame]
                action_vector = {
                    "up": np.array([0, 1]),
                    "down": np.array([0, -1]),
                    "right": np.array([1, 0]),
                    "left": np.array([-1, 0])
                }.get(action, np.array([0, 0]))
                
                # Show push arrow and connection if robot is close enough to push
                is_pushing = robot_to_object_dist < self.push_threshold
                if is_pushing and np.any(action_vector != 0):
                    # Show push arrow from robot toward object
                    arrow_scale = 0.6
                    push_arrow.set_position(robot_pos)
                    push_arrow.xy = robot_pos + action_vector * arrow_scale
                    push_arrow.set_visible(True)
                    
                    # Show connection line
                    connection_line.set_data([robot_pos[0], object_pos[0]], 
                                           [robot_pos[1], object_pos[1]])
                else:
                    push_arrow.set_visible(False)
                    connection_line.set_data([], [])
            else:
                push_arrow.set_visible(False)
                connection_line.set_data([], [])
            
            # Update text displays
            action_name = actions[frame] if frame < len(actions) else 'Terminal'
            step_text.set_text(f'Step: {frame+1}/{len(states)}\nAction: {action_name}')
            
            distance_text.set_text(f'Object ↔ Target: {distance_to_target:.2f}\n' +
                                 f'Robot ↔ Object: {robot_to_object_dist:.2f}')
            
            current_reward = rewards[frame] if frame < len(rewards) else 0.0
            total_reward = sum(rewards[:frame+1])
            reward_text.set_text(f'Step Reward: {current_reward:.1f}\nTotal Reward: {total_reward:.1f}')
            
            # Show success message if target reached
            if distance_to_target < 0.5:
                success_text.set_text('★ TARGET REACHED! ★\nEpisode Complete')
                success_text.set_visible(True)
            else:
                success_text.set_visible(False)
            
            # Show collision warning
            if robot_collision or object_collision:
                collision_parts = []
                if robot_collision:
                    collision_parts.append('Robot')
                if object_collision:
                    collision_parts.append('Object')
                collision_text.set_text(f'⚠ {" & ".join(collision_parts)} Collision! ⚠')
                collision_text.set_visible(True)
            else:
                collision_text.set_visible(False)
            
            return (robot_scatter, object_scatter, target_scatter, robot_path_line, 
                   object_path_line, push_arrow, connection_line, step_text, 
                   distance_text, reward_text, success_text, collision_text)
        
        # Create animation
        ani = animation.FuncAnimation(fig, animate, frames=len(states), 
                                    interval=1200, blit=False, repeat=True)
        
        plt.tight_layout()
        
        # Save animation
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(cache_path, writer="pillow", fps=0.8)
        plt.close(fig)  # Free memory

    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        robot_collisions = []
        object_collisions = []
        total_collisions = []
        
        for history in histories:
            history_robot_collisions = 0
            history_object_collisions = 0
            total_steps = len(history.history)
            
            for step in history.history:
                robot_pos = step.state[:2]  # [robot_x, robot_y]
                object_pos = step.state[2:4]  # [object_x, object_y]
                
                if self._is_colliding_with_obstacle(robot_pos):
                    history_robot_collisions += 1
                
                if self._is_colliding_with_obstacle(object_pos):
                    history_object_collisions += 1
            
            if total_steps > 0:
                robot_collisions.append(history_robot_collisions)
                object_collisions.append(history_object_collisions)
                total_collisions.append(history_robot_collisions + history_object_collisions)
        
        total_steps_all = sum(len(history.history) for history in histories)
        avg_robot_collisions = sum(robot_collisions) / total_steps_all if total_steps_all > 0 else 0
        avg_object_collisions = sum(object_collisions) / total_steps_all if total_steps_all > 0 else 0
        avg_total_collisions = sum(total_collisions) / total_steps_all if total_steps_all > 0 else 0
        
        robot_collision_rates = [c / len(history.history) for c, history in zip(robot_collisions, histories)]
        object_collision_rates = [c / len(history.history) for c, history in zip(object_collisions, histories)]
        total_collision_rates = [c / len(history.history) for c, history in zip(total_collisions, histories)]
        
        robot_collisions_ci = confidence_interval(data=robot_collision_rates, confidence=0.95)
        object_collisions_ci = confidence_interval(data=object_collision_rates, confidence=0.95)
        total_collisions_ci = confidence_interval(data=total_collision_rates, confidence=0.95)
        
        total_robot_collisions_ci = confidence_interval(data=robot_collisions, confidence=0.95)
        total_object_collisions_ci = confidence_interval(data=object_collisions, confidence=0.95)
        total_all_collisions_ci = confidence_interval(data=total_collisions, confidence=0.95)
        
        return [
            MetricValue(
                name="robot_obstacle_collision_rate",
                value=avg_robot_collisions,
                lower_confidence_bound=robot_collisions_ci[0],
                upper_confidence_bound=robot_collisions_ci[1]
            ),
            MetricValue(
                name="object_obstacle_collision_rate", 
                value=avg_object_collisions,
                lower_confidence_bound=object_collisions_ci[0],
                upper_confidence_bound=object_collisions_ci[1]
            ),
            MetricValue(
                name="total_obstacle_collision_rate",
                value=avg_total_collisions,
                lower_confidence_bound=total_collisions_ci[0],
                upper_confidence_bound=total_collisions_ci[1]
            ),
            MetricValue(
                name="total_robot_obstacle_collisions",
                value=sum(robot_collisions),
                lower_confidence_bound=total_robot_collisions_ci[0],
                upper_confidence_bound=total_robot_collisions_ci[1]
            ),
            MetricValue(
                name="total_object_obstacle_collisions",
                value=sum(object_collisions),
                lower_confidence_bound=total_object_collisions_ci[0],
                upper_confidence_bound=total_object_collisions_ci[1]
            ),
            MetricValue(
                name="total_all_obstacle_collisions",
                value=sum(total_collisions),
                lower_confidence_bound=total_all_collisions_ci[0],
                upper_confidence_bound=total_all_collisions_ci[1]
            )
        ] 