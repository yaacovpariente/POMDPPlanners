from typing import List, Any
from pathlib import Path
import json
import hashlib
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.distributions import Distribution, DiscreteDistribution
from POMDPPlanners.core.simulation import History
from POMDPPlanners.core.simulation import MetricValue
from POMDPPlanners.utils.statistics import confidence_interval
from POMDPPlanners.core.belief import Belief


class BaseLightDarkPOMDP(Environment, ABC):
    def __init__(
        self,
        discount_factor: float,
        name,
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
        self.__type_check(
            discount_factor=discount_factor,
            name=name,
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
        
        super().__init__(discount_factor=discount_factor, name=name)
        
        self.beacons = beacons
        self.goal_state = goal_state
        self.start_state = start_state
        self.obstacles = obstacles
        self.obstacle_hit_probability = obstacle_hit_probability
        self.obstacle_reward = obstacle_reward
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
        goal_reward: float,
        beacon_radius: float,
        fuel_cost: float,
        grid_size: int,
    ):
        # Type checks
        assert isinstance(discount_factor, float), "discount_factor must be a float"
        assert isinstance(name, str), "name must be a string"
        assert isinstance(beacons, np.ndarray), "beacons must be a numpy array"
        assert isinstance(goal_state, np.ndarray), "goal_state must be a numpy array"
        assert isinstance(start_state, np.ndarray), "start_state must be a numpy array"
        assert isinstance(obstacles, np.ndarray), "obstacles must be a numpy array"
        assert isinstance(obstacle_hit_probability, float), "obstacle_hit_probability must be a float"
        assert isinstance(obstacle_reward, float), "obstacle_reward must be a float"
        assert isinstance(goal_reward, float), "goal_reward must be a float"
        assert isinstance(beacon_radius, float), "beacon_radius must be a float"
        assert isinstance(fuel_cost, float), "fuel_cost must be a float"
        assert isinstance(grid_size, int), "grid_size must be an integer"

        # Value range checks
        assert 0 <= discount_factor <= 1, "discount_factor must be between 0 and 1"
        assert 0 <= obstacle_hit_probability <= 1, "obstacle_hit_probability must be between 0 and 1"
        assert grid_size > 0, "grid_size must be positive"
        assert beacon_radius > 0, "beacon_radius must be positive"

        # Shape checks
        assert beacons.shape[0] == 2, "beacons must be a 2xN array"
        assert goal_state.shape == (2,), "goal_state must be a 2D vector"
        assert start_state.shape == (2,), "start_state must be a 2D vector"
        assert obstacles.shape[0] == 2, "obstacles must be a 2xN array"

        # Range checks for states
        assert np.all(beacons >= 0) and np.all(beacons <= grid_size), "beacons coordinates must be within grid"
        assert np.all(goal_state >= 0) and np.all(goal_state <= grid_size), "goal_state must be within grid"
        assert np.all(start_state >= 0) and np.all(start_state <= grid_size), "start_state must be within grid"
        assert np.all(obstacles >= 0) and np.all(obstacles <= grid_size), "obstacles must be within grid"
        
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
        assert isinstance(cache_path, Path)
        assert str(cache_path).endswith(".gif")

        fig, ax = plt.subplots()
        ax.set_xlim(-1, self.grid_size + 1)
        ax.set_ylim(-1, self.grid_size + 1)
        ax.set_xticks(np.arange(-1, self.grid_size + 1, 1))
        ax.set_yticks(np.arange(-1, self.grid_size + 1, 1))
        ax.grid()

        # Plot the beacons
        ax.scatter(self.beacons[0], self.beacons[1], color="blue", label="Beacons")
        # Plot the goal state
        ax.scatter(
            self.goal_state[0], self.goal_state[1], color="green", label="Goal State"
        )
        # Plot the start state
        ax.scatter(
            self.start_state[0], self.start_state[1], color="red", label="Start State"
        )
        # Plot the obstacles
        ax.scatter(
            self.obstacles[0], self.obstacles[1], color="black", label="Obstacles"
        )

        # Initialize the agent's position and path line
        (agent,) = ax.plot([], [], "ro", markersize=10)
        (path_line,) = ax.plot([], [], "r-", alpha=0.5, linewidth=2)
        # Initialize the action arrow
        arrow = plt.arrow(0, 0, 0, 0, color='red', width=0.1, head_width=0.3, head_length=0.3, length_includes_head=True)
        # Initialize belief particles scatter plot
        belief_scatter = ax.scatter([], [], c='purple', alpha=0.5, label='Belief Particles')

        def init():
            agent.set_data([], [])
            path_line.set_data([], [])
            arrow.set_data(x=0, y=0, dx=0, dy=0)
            belief_scatter.set_offsets(np.empty((0, 2)))
            belief_scatter.set_sizes([])
            return agent, path_line, arrow, belief_scatter

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
                dx, dy = self.action_to_vector[action]
                arrow.set_data(x=x, y=y, dx=dx, dy=dy)
            else:
                arrow.set_data(x=x, y=y, dx=0, dy=0)

            # Update belief particles
            belief = agent_belief_path[frame]
            if len(belief.values) > 0:
                # Convert belief values to array of positions
                positions = np.array(belief.values)
                # Scale probabilities to reasonable sizes for visualization (multiply by 1000)
                sizes = np.array(belief.probs) * 1000
                belief_scatter.set_offsets(positions)
                belief_scatter.set_sizes(sizes)
            else:
                belief_scatter.set_offsets(np.empty((0, 2)))
                belief_scatter.set_sizes([])

            return agent, path_line, arrow, belief_scatter

        ani = animation.FuncAnimation(
            fig, update, frames=len(path), init_func=init, blit=True, repeat=False
        )
        plt.legend()
        plt.title("Agent Path and Belief Visualization")

        # Save the animation
        if cache_path is not None:
            ani.save(cache_path, writer="pillow", fps=2)  # Use pillow instead of ffmpeg

        plt.close()

    def cache_visualization(self, history: History, cache_path: Path) -> None:
        agent_path = [step.state for step in history.history]
        agent_belief_path = [step.belief.to_unique_support_distribution() for step in history.history]
        actions = [step.action for step in history.history]
        
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
        
        # Create a deterministic string representation and hash it
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()


class BaseLightDarkPOMDPDiscreteActions(BaseLightDarkPOMDP):
    def __init__(
        self,
        discount_factor: float,
        name,
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

