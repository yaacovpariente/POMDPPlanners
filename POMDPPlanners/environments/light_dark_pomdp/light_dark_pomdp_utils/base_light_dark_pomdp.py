from typing import List, Any
from pathlib import Path
import json
import hashlib
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from POMDPPlanners.core.environment import (
    Environment,
    SpaceInfo,
    SpaceType
)
from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.simulation import History
from POMDPPlanners.core.simulation import MetricValue
from POMDPPlanners.utils.statistics import confidence_interval
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.utils.config_to_id import config_to_id


class BaseLightDarkPOMDP(Environment, ABC):
    def __init__(
        self,
        discount_factor: float,
        name: str,
        space_info: SpaceInfo,
        beacons: np.ndarray = np.array(
            [[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]]
        ),
        goal_state: np.ndarray = np.array([10, 5]),
        start_state: np.ndarray = np.array([0, 5]),
        obstacles: np.ndarray = np.array([[3, 7], [5, 5]]),
        obstacle_hit_probability: float = 0.2,
        obstacle_reward: float = -10.0,
        obstacle_radius: float = 1.0,
        goal_reward: float = 10.0,
        beacon_radius: float = 1.0,
        fuel_cost: float = 2.0,
        grid_size: int = 11,
    ):
        self.__type_check(
            discount_factor=discount_factor,
            name=name,
            beacons=beacons,
            goal_state=goal_state,
            start_state=start_state,
            obstacles=obstacles,
            obstacle_hit_probability=obstacle_hit_probability,
            obstacle_reward=obstacle_reward,
            obstacle_radius=obstacle_radius,
            goal_reward=goal_reward,
            beacon_radius=beacon_radius,
            fuel_cost=fuel_cost,
            grid_size=grid_size,
        )
        
        super().__init__(discount_factor=discount_factor, name=name, space_info=space_info)
        
        self.beacons = beacons
        self.goal_state = goal_state
        self.start_state = start_state
        self.obstacles = obstacles
        self.obstacle_hit_probability = obstacle_hit_probability
        self.obstacle_reward = obstacle_reward
        self.obstacle_radius = obstacle_radius
        self.goal_reward = goal_reward
        self.beacon_radius = beacon_radius
        self.fuel_cost = fuel_cost
        self.grid_size = grid_size
        
    def __type_check(
        self,
        discount_factor: float,
        name: str,
        beacons: np.ndarray,
        goal_state: np.ndarray,
        start_state: np.ndarray,
        obstacles: np.ndarray,
        obstacle_hit_probability: float,
        obstacle_reward: float,
        obstacle_radius: float,
        goal_reward: float,
        beacon_radius: float,
        fuel_cost: float,
        grid_size: int,
    ):
        # Type checks
        if not isinstance(discount_factor, float):
            raise TypeError("discount_factor must be a float")
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(beacons, np.ndarray):
            raise TypeError("beacons must be a numpy array")
        if not isinstance(goal_state, np.ndarray):
            raise TypeError("goal_state must be a numpy array")
        if not isinstance(start_state, np.ndarray):
            raise TypeError("start_state must be a numpy array")
        if not isinstance(obstacles, np.ndarray):
            raise TypeError("obstacles must be a numpy array")
        if not isinstance(obstacle_hit_probability, float):
            raise TypeError("obstacle_hit_probability must be a float")
        if not isinstance(obstacle_reward, float):
            raise TypeError("obstacle_reward must be a float")
        if not isinstance(goal_reward, float):
            raise TypeError("goal_reward must be a float")
        if not isinstance(beacon_radius, float):
            raise TypeError("beacon_radius must be a float")
        if not isinstance(fuel_cost, float):
            raise TypeError("fuel_cost must be a float")
        if not isinstance(grid_size, int):
            raise TypeError("grid_size must be an integer")

        # Value range checks
        if not (0 <= discount_factor <= 1):
            raise ValueError("discount_factor must be between 0 and 1")
        if not (0 <= obstacle_hit_probability <= 1):
            raise ValueError("obstacle_hit_probability must be between 0 and 1")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if beacon_radius <= 0:
            raise ValueError("beacon_radius must be positive")
        if obstacle_radius <= 0:
            raise ValueError("obstacle_radius must be positive")
        # Shape checks
        if beacons.shape[0] != 2:
            raise ValueError("beacons must be a 2xN array")
        if goal_state.shape != (2,):
            raise ValueError("goal_state must be a 2D vector")
        if start_state.shape != (2,):
            raise ValueError("start_state must be a 2D vector")
        if obstacles.shape[0] != 2:
            raise ValueError("obstacles must be a 2xN array")

        # Range checks for states
        if not (np.all(beacons >= 0) and np.all(beacons <= grid_size)):
            raise ValueError("beacons coordinates must be within grid")
        if not (np.all(goal_state >= 0) and np.all(goal_state <= grid_size)):
            raise ValueError("goal_state must be within grid")
        if not (np.all(start_state >= 0) and np.all(start_state <= grid_size)):
            raise ValueError("start_state must be within grid")
        if not (np.all(obstacles >= 0) and np.all(obstacles <= grid_size)):
            raise ValueError("obstacles must be within grid")
        
    @abstractmethod
    def state_transition_model(self, state: np.ndarray, action: Any) -> Distribution:
        pass
    
    @abstractmethod
    def observation_model(self, next_state: np.ndarray, action: Any) -> Distribution:
        pass
    
    @abstractmethod
    def reward(self, state: np.ndarray, action: Any) -> float:
        pass
    
    @abstractmethod
    def is_terminal(self, state: np.ndarray) -> bool:
        pass
    
    @abstractmethod
    def compute_metrics(self, histories: List[History]) -> List[MetricValue]:
        pass

    def initial_state_dist(self) -> Distribution:
        return DiscreteDistribution(values=[self.start_state], probs=np.array([1.0]))

    def initial_observation_dist(self) -> Distribution:
        return DiscreteDistribution(values=[1.0], probs=np.array([1.0]))
    
    def visualize_path(self, path: List[np.ndarray], agent_belief_path: List[DiscreteDistribution], actions: List[str], cache_path: Path):
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

        # Control belief particles color
        belief_particles_color = '#FFFF00'  # Yellow color

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(-1, self.grid_size + 1)
        ax.set_ylim(-1, self.grid_size + 1)
        ax.set_xticks(np.arange(-1, self.grid_size + 1, 1))
        ax.set_yticks(np.arange(-1, self.grid_size + 1, 1))
        ax.set_facecolor('#696969')  # Darker grey
        ax.grid(False)  # Remove the grid from the background

        # Plot circles around beacons with white background and light fade effect
        for i in range(self.beacons.shape[1]):
            beacon_x, beacon_y = self.beacons[0, i], self.beacons[1, i]
            # Create multiple circles with exponential decay for realistic light fade
            for j in range(15):  # More circles for smoother gradient
                radius = self.beacon_radius + j * 0.05  # Smaller radius increments
                # Exponential decay for more realistic light fade
                alpha = 0.9 * np.exp(-j * 0.3)  # Exponential decay from 0.9 to near 0
                circle = plt.Circle((beacon_x, beacon_y), radius, 
                                  facecolor='white', edgecolor='none', alpha=alpha)
                ax.add_patch(circle)
        
        # Plot the beacons
        ax.scatter(self.beacons[0], self.beacons[1], color="blue", marker='^', s=100, label="Beacons")
        # Plot the goal state
        ax.scatter(
            self.goal_state[0], self.goal_state[1], color="green", marker='*', s=200, label="Goal State"
        )
        # Plot the start state
        ax.scatter(
            self.start_state[0], self.start_state[1], color="red", label="Start State"
        )
        # Plot circles around obstacles with transparent red background
        for i in range(self.obstacles.shape[1]):
            obstacle_x, obstacle_y = self.obstacles[0, i], self.obstacles[1, i]
            circle = plt.Circle((obstacle_x, obstacle_y), self.obstacle_radius, 
                              facecolor='red', edgecolor='none', alpha=0.3)
            ax.add_patch(circle)
        
        # Plot the obstacles
        ax.scatter(
            self.obstacles[0], self.obstacles[1], color="black", label="Obstacles"
        )

        # Initialize the agent's position and path line
        (agent,) = ax.plot([], [], "ro", markersize=10)
        (path_line,) = ax.plot([], [], "r-", alpha=0.5, linewidth=2)
        # Initialize the action arrow
        arrow = plt.arrow(0, 0, 0, 0, color='red', width=0.1, head_width=0.3, head_length=0.3, length_includes_head=True)
        # Initialize belief particles scatter plots for different time steps
        belief_scatters = []
        max_history = min(len(path), 10)  # Show up to 10 previous time steps
        # Create gradient from yellow to red
        yellow_color = np.array([1.0, 1.0, 0.0])  # Yellow RGB
        red_color = np.array([1.0, 0.0, 0.0])     # Red RGB
        colors = []
        for i in range(max_history):
            # Interpolate between yellow (current) and red (old)
            t = (max_history - 1 - i) / max_history  # 0 for current, 1 for oldest
            color = yellow_color * (1 - t) + red_color * t
            # Apply additional intensity decay
            alpha_factor = 0.1 + 0.9 * (i / max_history)  # 0.1 to 1.0 - more extreme decay
            color = color * alpha_factor
            colors.append(color)
        
        for i in range(max_history):
            scatter = ax.scatter([], [], c=[colors[i]], alpha=0.6 + 0.3 * (i / max_history), 
                               s=50, label="")
            belief_scatters.append(scatter)
        
        # Create a proper legend entry for belief particles
        from matplotlib.lines import Line2D
        legend_element = Line2D([0], [0], marker='o', color=belief_particles_color, 
                               markersize=8, alpha=0.7, linestyle='', label='Belief Particles')
        ax.add_artist(legend_element)

        def init():
            agent.set_data([], [])
            path_line.set_data([], [])
            arrow.set_data(x=0, y=0, dx=0, dy=0)
            for scatter in belief_scatters:
                scatter.set_offsets(np.empty((0, 2)))
                scatter.set_sizes([])
            return [agent, path_line, arrow] + belief_scatters

        def update(frame):
            # Update current position
            x = float(path[frame][0])
            y = float(path[frame][1])
            agent.set_data([x], [y])

            # Update path line up to current position
            path_x = [float(p[0]) for p in path[: frame + 1]]
            path_y = [float(p[1]) for p in path[: frame + 1]]
            path_line.set_data(path_x, path_y)

            # Update action arrow based on the action vector
            if frame < len(actions):
                action = actions[frame]
                if isinstance(action, str):
                    dx, dy = self.action_to_vector[action]
                else:
                    dx, dy = action
                arrow.set_data(x=x, y=y, dx=dx, dy=dy)
            else:
                arrow.set_data(x=x, y=y, dx=0, dy=0)

            # Update belief particles with history
            for i, scatter in enumerate(belief_scatters):
                history_frame = frame - (len(belief_scatters) - 1 - i)
                if history_frame >= 0 and history_frame < len(agent_belief_path):
                    belief = agent_belief_path[history_frame]
                    if len(belief.values) > 0:
                        # Convert belief values to array of positions
                        positions = np.array(belief.values)
                        # Scale probabilities to reasonable sizes for visualization (multiply by 200)
                        sizes = np.array(belief.probs) * 600
                        scatter.set_offsets(positions)
                        scatter.set_sizes(sizes)
                    else:
                        scatter.set_offsets(np.empty((0, 2)))
                        scatter.set_sizes([])
                else:
                    scatter.set_offsets(np.empty((0, 2)))
                    scatter.set_sizes([])

            return [agent, path_line, arrow] + belief_scatters

        ani = animation.FuncAnimation(
            fig, update, frames=len(path), init_func=init, blit=True, repeat=False
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("Agent Path")
        plt.tight_layout()

        # Save the animation
        if cache_path is not None:
            ani.save(cache_path, writer="pillow", fps=2)  # Use pillow instead of ffmpeg

        plt.close()

    def cache_visualization(self, history: History, cache_path: Path) -> None:
        """Cache visualization of agent's path and belief.
        
        Args:
            history: The history of states, actions, and observations
            cache_path: Path where to save the visualization
            
        Raises:
            ValueError: If history is empty or contains invalid data
        """
        if not history.history:
            raise ValueError("Cannot visualize empty history")
            
        # Extract data with validation
        agent_path = []
        agent_belief_path = []
        actions = []
        
        for step in history.history:
            if not hasattr(step, 'state') or not hasattr(step, 'belief') or not hasattr(step, 'action'):
                raise ValueError(f"History step missing required attributes: {step}")
                
            agent_path.append(step.state)
            agent_belief_path.append(step.belief.to_unique_support_distribution())
            actions.append(step.action)
            
        # Validate all lists have same length
        if not (len(agent_path) == len(agent_belief_path) == len(actions)):
            raise ValueError(f"Mismatched lengths: path={len(agent_path)}, belief={len(agent_belief_path)}, actions={len(actions)}")
            
        # Create directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.visualize_path(path=agent_path, agent_belief_path=agent_belief_path, actions=actions, cache_path=cache_path)

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return np.array_equal(observation1, observation2)
    
    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on environment configuration.
        This implementation ensures that the config_id is invariant to the order of beacons and obstacles.
        """
        config_dict = {}
        
        # Include all public attributes that aren't callables
        for key, value in self.__dict__.items():
            if key.startswith('_') or callable(value):
                continue
                
            # Handle numpy arrays
            if isinstance(value, np.ndarray):
                if key in ['beacons', 'obstacles']:
                    # For beacons and obstacles, sort columns to ensure order invariance
                    # First, transpose to get columns as rows
                    # Then sort rows lexicographically
                    # Finally, transpose back
                    sorted_array = np.sort(value.T, axis=0).T
                    config_dict[key] = sorted_array.tolist()
                else:
                    config_dict[key] = value.tolist()
            # Handle basic Python types
            elif isinstance(value, (str, int, float, bool, list, tuple)):
                config_dict[key] = value
            # Handle dictionaries
            elif isinstance(value, dict):
                # Only include serializable values
                serializable_dict = {}
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool, list, tuple)):
                        serializable_dict[k] = v
                    elif isinstance(v, np.ndarray):
                        serializable_dict[k] = v.tolist()
                config_dict[key] = serializable_dict
                
        # Sort dictionary to ensure consistent ordering
        config_dict = dict(sorted(config_dict.items()))
        
        return config_to_id(config_dict)


class BaseLightDarkPOMDPDiscreteActions(BaseLightDarkPOMDP):
    def __init__(
        self,
        discount_factor: float,
        name: str,
        is_discrete_observations: bool,
        beacons: np.ndarray = np.array(
            [[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]]
        ),
        goal_state: np.ndarray = np.array([10, 5]),
        start_state: np.ndarray = np.array([0, 5]),
        obstacles: np.ndarray = np.array([[3, 7], [5, 5]]),
        obstacle_hit_probability: float = 0.2,
        obstacle_reward: float = -10.0,
        goal_reward: float = 10.0,
        beacon_radius: float = 1.0,
        fuel_cost: float = 2.0,
        grid_size: int = 11,
    ):
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.DISCRETE if is_discrete_observations else SpaceType.CONTINUOUS
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
        
        self.actions = ["up", "down", "right", "left"]
        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }

    def get_actions(self) -> List[Any]:
        return self.actions

