"""LaserTag POMDP Environment Implementation.

This module implements the LaserTag problem, a pursuit-evasion POMDP environment
where an agent must navigate a grid to tag an opponent that moves stochastically.
The agent has noisy observations of the opponent's location.

The LaserTag problem features:
- A grid-based environment (default 7x11) with optional obstacles
- Robot and opponent moving on discrete grid cells
- 5 possible actions: North, South, East, West, Tag
- Noisy observations of opponent location (with measurement error)
- Positive reward for successful tagging, negative reward for failed tag attempts
- Step cost for each movement action
- Opponent moves probabilistically toward robot's position

Based on the LaserTag.jl implementation from: https://github.com/JuliaPOMDP/LaserTag.jl

Classes:
    LaserTagState: State representation with robot and opponent positions
    LaserTagStateTransition: State transition model for robot and opponent movement
    LaserTagObservation: Observation model with noisy opponent position measurements
    LaserTagPOMDP: Main environment class implementing the LaserTag problem
"""

from typing import List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    ObservationModel,
    StateTransitionModel,
    SpaceInfo,
    SpaceType
)
from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.simulation import History, MetricValue, StepData


@dataclass
class LaserTagState:
    """State representation for LaserTag POMDP.
    
    Attributes:
        robot: Robot's position as (row, col) tuple
        opponent: Opponent's position as (row, col) tuple  
        terminal: Whether the episode has terminated
        
    Example:
        Creating a LaserTag state::
        
            state = LaserTagState(robot=(0, 0), opponent=(6, 10), terminal=False)
            print(state.robot)     # (0, 0)
            print(state.opponent)  # (6, 10)
            print(state.terminal)  # False
    """
    robot: Tuple[int, int]
    opponent: Tuple[int, int]
    terminal: bool = False
    
    def __eq__(self, other):
        if not isinstance(other, LaserTagState):
            return False
        return (self.robot == other.robot and 
                self.opponent == other.opponent and
                self.terminal == other.terminal)
    
    def __hash__(self):
        return hash((self.robot, self.opponent, self.terminal))


class LaserTagStateTransition(StateTransitionModel):
    """State transition model for LaserTag POMDP.
    
    Handles robot movement (deterministic based on action) and opponent movement
    (probabilistic, with tendency to move toward robot's position).
    
    Attributes:
        state: Current LaserTagState
        action: Action to be executed (0=North, 1=South, 2=East, 3=West, 4=Tag)
        floor_shape: Tuple of (rows, cols) for grid dimensions
        obstacles: Set of obstacle positions
        
    Example:
        Creating and using the state transition model::
        
            state = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
            transition = LaserTagStateTransition(
                state=state, 
                action=0,  # North
                floor_shape=(7, 11),
                obstacles=set()
            )
            next_states = transition.sample(n_samples=5)
            probabilities = transition.probability(next_states)
    """
    
    def __init__(self, state: LaserTagState, action: int, floor_shape: Tuple[int, int], 
                 obstacles: set):
        """Initialize the state transition model.
        
        Args:
            state: Current LaserTagState
            action: Action to execute (0=North, 1=South, 2=East, 3=West, 4=Tag)
            floor_shape: Grid dimensions as (rows, cols)
            obstacles: Set of obstacle positions as (row, col) tuples
        """
        super().__init__(state, action)
        self.floor_shape = floor_shape
        self.obstacles = obstacles
        self._action_directions = {
            0: (-1, 0),  # North (up)
            1: (1, 0),   # South (down)
            2: (0, 1),   # East (right)  
            3: (0, -1),  # West (left)
            4: (0, 0)    # Tag (no movement)
        }
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not an obstacle."""
        row, col = pos
        return (0 <= row < self.floor_shape[0] and 
                0 <= col < self.floor_shape[1] and
                pos not in self.obstacles)
    
    def _get_robot_next_position(self) -> Tuple[int, int]:
        """Get robot's next position based on action."""
        if self.action == 4:  # Tag action
            return self.state.robot
        
        dr, dc = self._action_directions[self.action]
        new_pos = (self.state.robot[0] + dr, self.state.robot[1] + dc)
        
        # If new position is invalid, stay at current position
        if self._is_valid_position(new_pos):
            return new_pos
        else:
            return self.state.robot
    
    def _get_opponent_move_probabilities(self, robot_pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """Get opponent's movement probabilities based on robot position."""
        current_opp = self.state.opponent
        
        # Possible movement directions for opponent
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]  # N, S, E, W, Stay
        valid_moves = []
        
        for dr, dc in directions:
            new_pos = (current_opp[0] + dr, current_opp[1] + dc)
            if self._is_valid_position(new_pos):
                valid_moves.append(new_pos)
        
        # If no valid moves, stay in place
        if not valid_moves:
            return [(current_opp, 1.0)]
        
        # Calculate probabilities - opponent has 0.4 chance to move toward robot
        move_probs = []
        toward_robot_moves = []
        
        # Find moves that get closer to robot
        for pos in valid_moves:
            old_dist = abs(current_opp[0] - robot_pos[0]) + abs(current_opp[1] - robot_pos[1])
            new_dist = abs(pos[0] - robot_pos[0]) + abs(pos[1] - robot_pos[1])
            
            if new_dist < old_dist:
                toward_robot_moves.append(pos)
        
        # Distribute probabilities
        if toward_robot_moves:
            # 0.4 total probability for moves toward robot, distributed equally
            toward_prob = 0.4 / len(toward_robot_moves)
            # Remaining 0.6 for other moves
            other_prob = 0.6 / (len(valid_moves) - len(toward_robot_moves)) if len(valid_moves) > len(toward_robot_moves) else 0
            
            for pos in valid_moves:
                if pos in toward_robot_moves:
                    move_probs.append((pos, toward_prob))
                else:
                    move_probs.append((pos, other_prob))
        else:
            # No beneficial moves, distribute equally
            prob = 1.0 / len(valid_moves)
            for pos in valid_moves:
                move_probs.append((pos, prob))
        
        return move_probs
    
    def sample(self, n_samples: int = 1) -> List[LaserTagState]:
        """Sample next states from the transition model."""
        samples = []
        robot_next = self._get_robot_next_position()
        
        # Check if tagging occurred
        if self.action == 4 and self.state.robot == self.state.opponent:
            # Successful tag - terminal state
            for _ in range(n_samples):
                samples.append(LaserTagState(robot_next, self.state.opponent, terminal=True))
        else:
            # Regular transition
            opp_moves = self._get_opponent_move_probabilities(robot_next)
            positions, probabilities = zip(*opp_moves)
            
            for _ in range(n_samples):
                opp_next = np.random.choice(len(positions), p=probabilities)
                samples.append(LaserTagState(robot_next, positions[opp_next], terminal=False))
        
        return samples
    
    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate transition probabilities for given next states."""
        result = np.zeros(len(values))
        robot_next = self._get_robot_next_position()
        
        # Check if tagging occurred
        if self.action == 4 and self.state.robot == self.state.opponent:
            # Successful tag case
            for i, next_state in enumerate(values):
                if (isinstance(next_state, LaserTagState) and 
                    next_state.robot == robot_next and 
                    next_state.opponent == self.state.opponent and
                    next_state.terminal):
                    result[i] = 1.0
        else:
            # Regular transition case
            opp_moves = self._get_opponent_move_probabilities(robot_next)
            
            for i, next_state in enumerate(values):
                if (isinstance(next_state, LaserTagState) and 
                    next_state.robot == robot_next and
                    not next_state.terminal):
                    
                    # Find probability for this opponent position
                    for opp_pos, prob in opp_moves:
                        if next_state.opponent == opp_pos:
                            result[i] = prob
                            break
        
        return result


class LaserTagObservation(ObservationModel):
    """Observation model for LaserTag POMDP.
    
    Provides noisy observations of the opponent's position. The observation
    is the opponent's true position plus Gaussian noise.
    
    Attributes:
        next_state: The state after action execution
        action: The action that was taken
        measurement_noise: Standard deviation of measurement noise
        
    Example:
        Creating and using the observation model::
        
            state = LaserTagState(robot=(3, 5), opponent=(2, 4), terminal=False)
            obs_model = LaserTagObservation(
                next_state=state,
                action=0,
                measurement_noise=1.0
            )
            observations = obs_model.sample(n_samples=3)
            probabilities = obs_model.probability(observations)
    """
    
    def __init__(self, next_state: LaserTagState, action: int, measurement_noise: float = 1.0):
        """Initialize the observation model.
        
        Args:
            next_state: State after taking the action
            action: Action that was executed
            measurement_noise: Standard deviation of Gaussian measurement noise
        """
        super().__init__(next_state, action)
        self.measurement_noise = measurement_noise
    
    def sample(self, n_samples: int = 1) -> List[Tuple[float, float]]:
        """Sample observations from the observation model."""
        samples = []
        
        if self.next_state.terminal:
            # Terminal state - return terminal observation
            for _ in range(n_samples):
                samples.append((-1.0, -1.0))  # Special terminal observation
        else:
            # Add Gaussian noise to opponent's true position
            true_pos = np.array(self.next_state.opponent, dtype=float)
            for _ in range(n_samples):
                noise = np.random.normal(0, self.measurement_noise, size=2)
                noisy_obs = true_pos + noise
                samples.append((noisy_obs[0], noisy_obs[1]))
        
        return samples
    
    def probability(self, values: List[Any]) -> np.ndarray:
        """Calculate observation probabilities for given values."""
        result = np.zeros(len(values))
        
        if self.next_state.terminal:
            # Terminal state case
            for i, obs in enumerate(values):
                if obs == (-1.0, -1.0):
                    result[i] = 1.0
        else:
            # Calculate Gaussian probability density
            true_pos = np.array(self.next_state.opponent, dtype=float)
            variance = self.measurement_noise ** 2
            
            for i, obs in enumerate(values):
                if isinstance(obs, (tuple, list)) and len(obs) == 2:
                    obs_pos = np.array(obs, dtype=float)
                    # Bivariate Gaussian PDF
                    diff = obs_pos - true_pos
                    exp_term = np.exp(-0.5 * np.sum(diff ** 2) / variance)
                    result[i] = exp_term / (2 * np.pi * variance)
        
        return result


class LaserTagPOMDP(DiscreteActionsEnvironment):
    """LaserTag POMDP environment implementation.
    
    This is a pursuit-evasion problem where a robot must navigate a grid to tag
    an opponent. The robot receives noisy observations of the opponent's position
    and must decide when and where to attempt tagging.
    
    Problem Structure:
    - States: Robot position, opponent position, terminal flag
    - Actions: North(0), South(1), East(2), West(3), Tag(4)
    - Observations: Noisy (x, y) coordinates of opponent position
    - Rewards: Tag success(+10), Tag failure(-10), Movement(-1)
    
    Attributes:
        floor_shape: Grid dimensions as (rows, cols)
        obstacles: Set of obstacle positions
        tag_reward: Reward for successful tagging
        tag_penalty: Penalty for unsuccessful tagging  
        step_cost: Cost per movement action
        measurement_noise: Standard deviation of observation noise
        
    Example:
        Creating and using a LaserTag POMDP::
        
            # Create LaserTag environment
            env = LaserTagPOMDP(
                discount_factor=0.95,
                floor_shape=(7, 11),
                tag_reward=10.0,
                step_cost=1.0
            )
            
            # Sample initial state and get actions
            initial_state = env.initial_state_dist().sample()[0]
            actions = env.get_actions()
            
            # Execute action and get reward
            reward = env.reward(initial_state, action=0)  # Move north
            
            # Check terminal condition
            is_done = env.is_terminal(initial_state)
    """
    
    def __init__(self, 
                 discount_factor: float,
                 name: str = "LaserTagPOMDP",
                 floor_shape: Tuple[int, int] = (7, 11),
                 obstacles: Optional[set] = None,
                 tag_reward: float = 10.0,
                 tag_penalty: float = 10.0,
                 step_cost: float = 1.0,
                 measurement_noise: float = 1.0,
                 output_dir: Optional[Path] = None,
                 debug: bool = False):
        """Initialize the LaserTag POMDP environment.
        
        Args:
            discount_factor: Discount factor for future rewards (0 < discount_factor <= 1)
            name: Name identifier for this environment instance
            floor_shape: Grid dimensions as (rows, cols). Defaults to (7, 11).
            obstacles: Set of obstacle positions as (row, col) tuples. Defaults to empty set.
            tag_reward: Reward for successful tagging. Defaults to 10.0.
            tag_penalty: Penalty for unsuccessful tagging. Defaults to 10.0.
            step_cost: Cost per movement action. Defaults to 1.0.
            measurement_noise: Standard deviation of observation noise. Defaults to 1.0.
            output_dir: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
            
        Raises:
            ValueError: If discount_factor is not in valid range [0, 1]
        """
        if not (0.0 <= discount_factor <= 1.0):
            raise ValueError("discount_factor must be between 0 and 1 (inclusive)")
        
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,  # 5 discrete actions
            observation_space=SpaceType.CONTINUOUS  # Continuous (x,y) coordinates with noise
        )
        
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            output_dir=output_dir,
            debug=debug
        )
        
        self.floor_shape = floor_shape
        self.obstacles = obstacles if obstacles is not None else set()
        self.tag_reward = tag_reward
        self.tag_penalty = tag_penalty
        self.step_cost = step_cost
        self.measurement_noise = measurement_noise
        
        # Action definitions
        self.actions = [0, 1, 2, 3, 4]  # North, South, East, West, Tag
        self.action_names = ["North", "South", "East", "West", "Tag"]
    
    def state_transition_model(self, state: LaserTagState, action: int) -> StateTransitionModel:
        """Get the state transition model for a given state-action pair."""
        return LaserTagStateTransition(
            state=state,
            action=action,
            floor_shape=self.floor_shape,
            obstacles=self.obstacles
        )
    
    def observation_model(self, next_state: LaserTagState, action: int) -> ObservationModel:
        """Get the observation model for a given next state and action."""
        return LaserTagObservation(
            next_state=next_state,
            action=action,
            measurement_noise=self.measurement_noise
        )
    
    def reward(self, state: LaserTagState, action: int) -> float:
        """Calculate the immediate reward for a state-action pair."""
        if state.terminal:
            return 0.0  # No reward in terminal state
        
        if action == 4:  # Tag action
            if state.robot == state.opponent:
                return self.tag_reward  # Successful tag
            else:
                return -self.tag_penalty  # Failed tag attempt
        else:
            return -self.step_cost  # Movement cost
    
    def is_terminal(self, state: LaserTagState) -> bool:
        """Check if a state is terminal."""
        return state.terminal
    
    def initial_state_dist(self) -> Distribution:
        """Get the initial state distribution."""
        # Generate all valid robot and opponent positions
        valid_positions = []
        for row in range(self.floor_shape[0]):
            for col in range(self.floor_shape[1]):
                if (row, col) not in self.obstacles:
                    valid_positions.append((row, col))
        
        # Create all possible initial states (robot and opponent at different positions)
        initial_states = []
        for robot_pos in valid_positions:
            for opp_pos in valid_positions:
                if robot_pos != opp_pos:  # Robot and opponent start at different positions
                    initial_states.append(LaserTagState(robot_pos, opp_pos, terminal=False))
        
        # Uniform distribution over all initial states
        num_states = len(initial_states)
        probs = np.ones(num_states) / num_states
        
        return DiscreteDistribution(values=initial_states, probs=probs)
    
    def initial_observation_dist(self) -> Distribution:
        """Get the initial observation distribution."""
        # Return distribution over possible noisy observations at start
        # For simplicity, return a distribution centered at (0, 0) with noise
        return DiscreteDistribution(
            values=[(0.0, 0.0)], 
            probs=np.array([1.0])
        )
    
    def get_actions(self) -> List[int]:
        """Get all possible actions in the discrete action space."""
        return self.actions
    
    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Check if two observations are equal."""
        if isinstance(observation1, (tuple, list)) and isinstance(observation2, (tuple, list)):
            if len(observation1) == len(observation2) == 2:
                return (abs(observation1[0] - observation2[0]) < 1e-10 and 
                        abs(observation1[1] - observation2[1]) < 1e-10)
        return observation1 == observation2
    
    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        """Compute LaserTag POMDP specific metrics from simulation histories."""
        # Calculate success rate (successful tags)
        successful_tags = 0
        failed_tag_attempts = 0
        obstacle_collisions = 0
        total_episodes = len(histories)
        episode_lengths = []
        
        for history in histories:
            episode_length = len(history.history)
            episode_lengths.append(episode_length)
            
            # Check if episode ended with successful tag
            if history.history and history.history[-1].reward > 0:
                successful_tags += 1
            
            # Count failed tag attempts and obstacle collisions
            for i, step in enumerate(history.history):
                if step.action == 4 and step.reward < 0:  # Tag action with negative reward
                    failed_tag_attempts += 1
                
                # Check for obstacle collision by comparing robot position before and after movement
                if step.action in [0, 1, 2, 3]:  # Movement actions
                    # Check if robot tried to move but stayed in same position due to obstacle
                    if (isinstance(step.state, LaserTagState) and 
                        hasattr(step, 'next_state') and isinstance(step.next_state, LaserTagState)):
                        
                        # Calculate intended position based on current state and action
                        action_dirs = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}
                        if step.action in action_dirs:
                            dr, dc = action_dirs[step.action]
                            intended_pos = (step.state.robot[0] + dr, step.state.robot[1] + dc)
                            
                            # Check if intended position was an obstacle and robot didn't move
                            if (intended_pos in self.obstacles and 
                                step.next_state.robot == step.state.robot):
                                obstacle_collisions += 1
        
        success_rate = successful_tags / total_episodes if total_episodes > 0 else 0.0
        avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
        avg_failed_tags = failed_tag_attempts / total_episodes if total_episodes > 0 else 0.0
        avg_obstacle_collisions = obstacle_collisions / total_episodes if total_episodes > 0 else 0.0
        
        return [
            MetricValue(
                name="tag_success_rate",
                value=success_rate,
                lower_confidence_bound=success_rate,
                upper_confidence_bound=success_rate
            ),
            MetricValue(
                name="average_episode_length", 
                value=avg_episode_length,
                lower_confidence_bound=avg_episode_length,
                upper_confidence_bound=avg_episode_length
            ),
            MetricValue(
                name="average_failed_tag_attempts",
                value=avg_failed_tags,
                lower_confidence_bound=avg_failed_tags,
                upper_confidence_bound=avg_failed_tags
            ),
            MetricValue(
                name="average_obstacle_collisions",
                value=avg_obstacle_collisions,
                lower_confidence_bound=avg_obstacle_collisions,
                upper_confidence_bound=avg_obstacle_collisions
            )
        ]
    
    def cache_visualization(self, history: History, cache_path: Path) -> None:
        """Cache visualization of the LaserTag episode as an animated GIF.
        
        Creates an animated visualization showing:
        - Robot movement (red circle with path trail)
        - Opponent movement (blue circle)
        - Obstacles (black squares with red borders)
        - Action arrows showing robot's intended movement
        - Belief particles (if available) showing robot's belief about opponent location
        - Grid boundaries and coordinate system
        
        Args:
            history: The history of states, actions, and observations from an episode
            cache_path: Path where to save the visualization GIF
            
        Raises:
            ValueError: If history is empty or contains invalid data
            TypeError: If cache_path is not a Path object or doesn't end with .gif
        """
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")
        if not history.history:
            raise ValueError("Cannot visualize empty history")
        
        # Create directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract robot and opponent paths
        robot_path = []
        opponent_path = []
        actions = []
        beliefs = []
        
        for step in history.history:
            if not isinstance(step.state, LaserTagState):
                raise ValueError(f"Expected LaserTagState, got {type(step.state)}")
            
            robot_path.append(step.state.robot)
            opponent_path.append(step.state.opponent)
            actions.append(step.action)
            
            # Try to extract belief if available
            if hasattr(step, 'belief') and step.belief is not None:
                beliefs.append(step.belief)
            else:
                beliefs.append(None)
        
        # Set up the figure and axis with extra space for legend
        fig, ax = plt.subplots(figsize=(14, 8))
        rows, cols = self.floor_shape
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert y-axis so (0,0) is top-left like matrix indexing
        
        # Set grid
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title('LaserTag POMDP Episode Visualization')
        
        # Draw obstacles as red squares
        obstacle_patches = []
        for i, obstacle in enumerate(self.obstacles):
            row, col = obstacle
            square = plt.Rectangle((col - 0.4, row - 0.4), 0.8, 0.8, 
                                 facecolor='red', edgecolor='black', alpha=0.7,
                                 label='Obstacles' if i == 0 else "")  # Only label first obstacle
            ax.add_patch(square)
            if i == 0:  # Keep reference for legend
                obstacle_patches.append(square)
        
        # Initialize animated elements
        robot_agent, = ax.plot([], [], 'ro', markersize=12, label='Robot')
        opponent_agent, = ax.plot([], [], 'bo', markersize=12, label='Opponent')
        robot_path_line, = ax.plot([], [], 'r-', alpha=0.5, linewidth=2, label='Robot Path')
        opponent_path_line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=2, label='Opponent Path')
        
        # Action arrow
        action_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Step counter
        step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Action text
        action_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Belief particles (if available)
        belief_scatter = ax.scatter([], [], c='yellow', alpha=0.6, s=30, label='Belief Particles')
        
        # Legend - position it inside the plot area to avoid truncation
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
        
        def init():
            robot_agent.set_data([], [])
            opponent_agent.set_data([], [])
            robot_path_line.set_data([], [])
            opponent_path_line.set_data([], [])
            action_arrow.set_position((0, 0))
            action_arrow.xy = (0, 0)
            step_text.set_text('')
            action_text.set_text('')
            belief_scatter.set_offsets(np.empty((0, 2)))
            return [robot_agent, opponent_agent, robot_path_line, opponent_path_line, 
                   action_arrow, step_text, action_text, belief_scatter]
        
        def update(frame):
            # Current positions
            robot_pos = robot_path[frame]
            opponent_pos = opponent_path[frame]
            
            # Update agent positions (convert row,col to x,y for plotting)
            robot_agent.set_data([robot_pos[1]], [robot_pos[0]])  # col, row
            opponent_agent.set_data([opponent_pos[1]], [opponent_pos[0]])  # col, row
            
            # Update path lines up to current frame
            robot_cols = [pos[1] for pos in robot_path[:frame+1]]
            robot_rows = [pos[0] for pos in robot_path[:frame+1]]
            opponent_cols = [pos[1] for pos in opponent_path[:frame+1]]
            opponent_rows = [pos[0] for pos in opponent_path[:frame+1]]
            
            robot_path_line.set_data(robot_cols, robot_rows)
            opponent_path_line.set_data(opponent_cols, opponent_rows)
            
            # Update action arrow
            if frame < len(actions):
                action = actions[frame]
                action_dirs = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1), 4: (0, 0)}
                action_names = {0: "North", 1: "South", 2: "East", 3: "West", 4: "Tag"}
                
                if action in action_dirs:
                    dr, dc = action_dirs[action]
                    # Arrow from robot position in direction of action
                    action_arrow.set_position((robot_pos[1], robot_pos[0]))
                    action_arrow.xy = (robot_pos[1] + dc * 0.3, robot_pos[0] + dr * 0.3)
                    
                    # Update text displays
                    step_text.set_text(f'Step: {frame + 1}/{len(robot_path)}')
                    action_text.set_text(f'Action: {action_names.get(action, "Unknown")}')
            
            # Update belief particles if available
            if frame < len(beliefs) and beliefs[frame] is not None:
                try:
                    belief = beliefs[frame]
                    if hasattr(belief, 'to_unique_support_distribution'):
                        unique_belief = belief.to_unique_support_distribution()
                        if len(unique_belief.values) > 0:
                            # Extract opponent positions from belief states
                            belief_positions = []
                            belief_weights = []
                            for i, state in enumerate(unique_belief.values):
                                if isinstance(state, LaserTagState):
                                    # Convert row,col to x,y for plotting
                                    belief_positions.append([state.opponent[1], state.opponent[0]])
                                    belief_weights.append(unique_belief.probs[i] * 100)  # Scale for visibility
                            
                            if belief_positions:
                                belief_scatter.set_offsets(np.array(belief_positions))
                                belief_scatter.set_sizes(np.array(belief_weights))
                            else:
                                belief_scatter.set_offsets(np.empty((0, 2)))
                        else:
                            belief_scatter.set_offsets(np.empty((0, 2)))
                except:
                    belief_scatter.set_offsets(np.empty((0, 2)))
            else:
                belief_scatter.set_offsets(np.empty((0, 2)))
            
            return [robot_agent, opponent_agent, robot_path_line, opponent_path_line,
                   action_arrow, step_text, action_text, belief_scatter]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(robot_path), init_func=init,
            blit=True, repeat=False, interval=1000  # 1 second per frame
        )
        
        # Save animation with proper layout
        plt.tight_layout()
        anim.save(cache_path, writer='pillow', fps=1)
        plt.close(fig)
        
        self.logger.info(f"Saved LaserTag visualization to {cache_path}")