from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import logging
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    Environment,
    ObservationModel,
    SpaceInfo,
    SpaceType,
    StateTransitionModel,
)
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.utils.config_to_id import config_to_id


class BaseLightDarkPOMDP(Environment, ABC):
    def __init__(
        self,
        discount_factor: float,
        name: str,
        space_info: SpaceInfo,
        reward_range: Optional[Tuple[float, float]] = None,
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

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=reward_range,
        )

        # Convert lists of tuples to numpy arrays (maintaining internal representation)
        self.beacons = self._convert_beacons_to_array(beacons)
        self.goal_state = goal_state
        self.start_state = start_state
        self.obstacles = self._convert_obstacles_to_array(obstacles)
        self.obstacle_hit_probability = obstacle_hit_probability
        self.obstacle_reward = obstacle_reward
        self.obstacle_radius = obstacle_radius
        self.goal_reward = goal_reward
        self.beacon_radius = beacon_radius
        self.fuel_cost = fuel_cost
        self.grid_size = grid_size

        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }

    def _convert_beacons_to_array(self, beacons_list: List[Tuple[float, float]]) -> np.ndarray:
        """Convert list of (x, y) tuples to 2xN numpy array format for beacons.

        Args:
            beacons_list: List of (x, y) coordinate tuples

        Returns:
            2xN numpy array where first row is x coordinates, second row is y coordinates
        """
        if not beacons_list:
            return np.empty((2, 0))

        # Convert list of tuples to numpy array and transpose to get 2xN format
        coords_array = np.array(beacons_list).T  # Shape: (2, N)
        return coords_array

    def _convert_obstacles_to_array(self, obstacles_list: List[Tuple[float, float]]) -> np.ndarray:
        """Convert list of (x, y) tuples to 2xN numpy array format for obstacles.

        Args:
            obstacles_list: List of (x, y) coordinate tuples

        Returns:
            2xN numpy array where first row is x coordinates, second row is y coordinates
        """
        if not obstacles_list:
            return np.empty((2, 0))

        # Convert list of tuples to numpy array and transpose to get 2xN format (same as beacons)
        coords_array = np.array(obstacles_list).T  # Shape: (2, N)
        return coords_array

    def __type_check(
        self,
        discount_factor: float,
        name: str,
        beacons: List[Tuple[float, float]],
        goal_state: np.ndarray,
        start_state: np.ndarray,
        obstacles: List[Tuple[float, float]],
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
        if not isinstance(beacons, list):
            raise TypeError("beacons must be a list of tuples")
        if beacons and not all(
            isinstance(beacon, tuple) and len(beacon) == 2 for beacon in beacons
        ):
            raise TypeError("beacons must be a list of (x, y) coordinate tuples")
        if not isinstance(goal_state, np.ndarray):
            raise TypeError("goal_state must be a numpy array")
        if not isinstance(start_state, np.ndarray):
            raise TypeError("start_state must be a numpy array")
        if not isinstance(obstacles, list):
            raise TypeError("obstacles must be a list of tuples")
        if obstacles and not all(
            isinstance(obstacle, tuple) and len(obstacle) == 2 for obstacle in obstacles
        ):
            raise TypeError("obstacles must be a list of (x, y) coordinate tuples")
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
        if goal_state.shape != (2,):
            raise ValueError("goal_state must be a 2D vector")
        if start_state.shape != (2,):
            raise ValueError("start_state must be a 2D vector")

        # Range checks for states
        for beacon in beacons:
            if not (0 <= beacon[0] <= grid_size and 0 <= beacon[1] <= grid_size):
                raise ValueError("beacons coordinates must be within grid")
        if not (np.all(goal_state >= 0) and np.all(goal_state <= grid_size)):
            raise ValueError("goal_state must be within grid")
        if not (np.all(start_state >= 0) and np.all(start_state <= grid_size)):
            raise ValueError("start_state must be within grid")
        for obstacle in obstacles:
            if not (0 <= obstacle[0] <= grid_size and 0 <= obstacle[1] <= grid_size):
                raise ValueError("obstacles coordinates must be within grid")

    @abstractmethod
    def state_transition_model(self, state: np.ndarray, action: Any) -> StateTransitionModel:
        pass

    @abstractmethod
    def observation_model(self, next_state: np.ndarray, action: Any) -> ObservationModel:
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

    def visualize_path(
        self,
        path: List[np.ndarray],
        agent_belief_path: List[DiscreteDistribution],
        actions: List[str],
        cache_path: Path,
    ):
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

        # Control belief particles color
        belief_particles_color = "#FFFF00"  # Yellow color

        fig: Figure
        ax: Axes
        fig_temp, ax_temp = plt.subplots(figsize=(10, 8))
        fig = cast(Figure, fig_temp)
        ax = cast(Axes, ax_temp)
        ax.set_xlim(-1, self.grid_size + 1)
        ax.set_ylim(-1, self.grid_size + 1)
        ax.set_xticks(np.arange(-1, self.grid_size + 1, 1))  # type: ignore[arg-type]
        ax.set_yticks(np.arange(-1, self.grid_size + 1, 1))  # type: ignore[arg-type]
        ax.set_facecolor("#696969")  # Darker grey
        ax.grid(False)  # Remove the grid from the background

        # Plot circles around beacons with white background and light fade effect
        for i in range(self.beacons.shape[1]):
            beacon_x, beacon_y = self.beacons[0, i], self.beacons[1, i]
            # Create multiple circles with exponential decay for realistic light fade
            for j in range(15):  # More circles for smoother gradient
                radius = self.beacon_radius + j * 0.05  # Smaller radius increments
                # Exponential decay for more realistic light fade
                alpha = 0.9 * np.exp(-j * 0.3)  # Exponential decay from 0.9 to near 0
                circle = plt.Circle(  # type: ignore[attr-defined]
                    (beacon_x, beacon_y),
                    float(radius),  # type: ignore[arg-type]
                    facecolor="white",
                    edgecolor="none",
                    alpha=float(alpha),
                )
                ax.add_patch(circle)

        # Plot the beacons
        ax.scatter(
            self.beacons[0],
            self.beacons[1],
            color="blue",
            marker="^",
            s=100,
            label="Beacons",
        )
        # Plot the goal state
        ax.scatter(
            self.goal_state[0],
            self.goal_state[1],
            color="green",
            marker="*",
            s=200,
            label="Goal State",
        )
        # Plot the start state
        ax.scatter(self.start_state[0], self.start_state[1], color="red", label="Start State")
        # Plot circles around obstacles with transparent red background
        for i in range(self.obstacles.shape[1]):  # obstacles.shape[1] is number of obstacles
            obstacle_x, obstacle_y = (
                self.obstacles[0, i],
                self.obstacles[1, i],
            )  # obstacles[0,i] is x, obstacles[1,i] is y
            circle = plt.Circle(  # type: ignore[attr-defined]
                (obstacle_x, obstacle_y),
                float(self.obstacle_radius),  # type: ignore[arg-type]
                facecolor="red",
                edgecolor="none",
                alpha=0.3,
            )
            ax.add_patch(circle)

        # Plot the obstacles
        if self.obstacles.size > 0:
            ax.scatter(self.obstacles[0], self.obstacles[1], color="black", label="Obstacles")

        # Initialize the agent's position and path line
        agent = cast(Line2D, ax.plot([], [], "ro", markersize=10)[0])
        path_line = cast(Line2D, ax.plot([], [], "r-", alpha=0.5, linewidth=2)[0])
        # Initialize the action arrow
        arrow = plt.arrow(
            0,
            0,
            0,
            0,
            color="red",
            width=0.1,
            head_width=0.3,
            head_length=0.3,
            length_includes_head=True,
        )
        # Initialize belief particles scatter plots for different time steps
        belief_scatters = []
        max_history = min(len(path), 10)  # Show up to 10 previous time steps
        # Create gradient from yellow to red
        yellow_color = np.array([1.0, 1.0, 0.0])  # Yellow RGB
        red_color = np.array([1.0, 0.0, 0.0])  # Red RGB
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
            scatter = ax.scatter(
                [],
                [],
                c=[colors[i]],
                alpha=0.6 + 0.3 * (i / max_history),
                s=50,
                label="",
            )
            belief_scatters.append(scatter)

        # Create a proper legend entry for belief particles
        legend_element = Line2D(
            [],
            [],
            marker="o",
            color=belief_particles_color,
            markersize=8,
            alpha=0.7,
            linestyle="",
            label="Belief Particles",
        )
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
                if action is None:
                    # Handle None actions (e.g., terminal step)
                    dx, dy = 0, 0
                elif isinstance(action, str):
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
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title("Agent Path")
        plt.tight_layout()

        # Save the animation
        if cache_path is not None:
            ani.save(cache_path, writer="pillow", fps=2)  # Use pillow instead of ffmpeg

        plt.close()

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache visualization of agent's path and belief.

        Args:
            history: List of step data from an episode
            cache_path: Path where to save the visualization

        Raises:
            ValueError: If history is empty or contains invalid data
        """
        if not isinstance(history, List):
            raise TypeError("history must be a List object")
        if not history:
            raise ValueError("Cannot visualize empty history")
        for step in history:
            if not isinstance(step, StepData):
                raise TypeError("history must be a List of StepData objects")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")

        # Extract data with validation
        agent_path = []
        agent_belief_path: List[DiscreteDistribution] = []
        actions = []

        for step in history:
            if (
                not hasattr(step, "state")
                or not hasattr(step, "belief")
                or not hasattr(step, "action")
            ):
                raise ValueError(f"History step missing required attributes: {step}")

            agent_path.append(step.state)
            if isinstance(step.belief, WeightedParticleBelief):
                agent_belief_path.append(step.belief.to_unique_support_distribution())
            else:
                particles = [step.belief.sample() for _ in range(20)]
                weights = np.ones(len(particles)) / len(particles)
                discrete_distribution = DiscreteDistribution(values=particles, probs=weights)
                agent_belief_path.append(discrete_distribution)

            actions.append(step.action)

        # Validate all lists have same length
        if not (len(agent_path) == len(agent_belief_path) == len(actions)):
            raise ValueError(
                f"Mismatched lengths: path={len(agent_path)}, belief={len(agent_belief_path)}, actions={len(actions)}"
            )

        # Create directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.visualize_path(
            path=agent_path,
            agent_belief_path=agent_belief_path,
            actions=actions,
            cache_path=cache_path,
        )

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return np.array_equal(observation1, observation2)

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on environment configuration.
        This implementation ensures that the config_id is invariant to the order of beacons and obstacles.
        """

        def serialize_value(value, key=None):
            if isinstance(value, np.ndarray):
                if key in ["beacons", "obstacles"] and value.shape[0] == 2:
                    # This is beacons or obstacles in 2xN format
                    # Transpose to get Nx2 format, sort by rows (coordinate pairs), then transpose back
                    transposed = value.T  # Nx2 format
                    sorted_indices = np.lexsort(
                        (transposed[:, 1], transposed[:, 0])
                    )  # Sort by x, then y
                    sorted_array = transposed[sorted_indices].T  # Back to 2xN format
                    # Convert to float to ensure consistent data types
                    return sorted_array.astype(float).tolist()
                else:
                    return value.tolist()
            elif isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            elif isinstance(value, SpaceInfo):
                return {
                    "action_space": serialize_value(value.action_space),
                    "observation_space": serialize_value(value.observation_space),
                }
            elif isinstance(value, Enum):
                return value.value
            elif hasattr(value, "__dict__"):
                # Skip logger objects
                if isinstance(value, logging.Logger):
                    return None
                return serialize_value(value.__dict__)
            else:
                return str(value)

        config_dict = {}
        for key, value in self.__dict__.items():
            # Skip logger and private attributes
            if key.startswith("_") or callable(value) or isinstance(value, logging.Logger):
                continue
            serialized_value = serialize_value(value, key)
            if serialized_value is not None:  # Skip None values (like logger)
                config_dict[key] = serialized_value
        config_dict = dict(sorted(config_dict.items()))
        return config_to_id(config_dict)


class BaseLightDarkPOMDPDiscreteActions(BaseLightDarkPOMDP):
    def __init__(
        self,
        discount_factor: float,
        name: str,
        is_discrete_observations: bool,
        reward_range: Optional[Tuple[float, float]] = None,
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
    ):
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=(
                SpaceType.DISCRETE if is_discrete_observations else SpaceType.CONTINUOUS
            ),
        )
        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=space_info,
            reward_range=reward_range,
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

    def get_actions(self) -> List[Any]:
        return self.actions
