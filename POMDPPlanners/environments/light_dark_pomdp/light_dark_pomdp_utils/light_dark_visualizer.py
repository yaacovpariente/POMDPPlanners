# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any, List, Tuple, cast

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.simulation import StepData


class LightDarkPOMDPVisualizer:
    """Visualizer for Light-Dark POMDP environments.

    Handles all visualization and animation logic for Light-Dark POMDP environments,
    including path visualization, belief particle rendering, and animation generation.

    Attributes:
        environment: The Light-Dark POMDP environment instance to visualize.
    """

    def __init__(self, environment: Any):
        """Initialize the visualizer.

        Args:
            environment: The Light-Dark POMDP environment instance to visualize.
                Must have attributes: beacons, goal_state, start_state, obstacles,
                obstacle_radius, beacon_radius, grid_size, action_to_vector.
        """
        self.environment = environment

    def _setup_figure_and_axes(self) -> Tuple[Figure, Axes]:
        fig_temp, ax_temp = plt.subplots(figsize=(10, 8))
        fig = cast(Figure, fig_temp)
        ax = cast(Axes, ax_temp)
        ax.set_xlim(-1, self.environment.grid_size + 1)
        ax.set_ylim(-1, self.environment.grid_size + 1)
        ax.set_xticks(np.arange(-1, self.environment.grid_size + 1, 1))  # type: ignore[arg-type]
        ax.set_yticks(np.arange(-1, self.environment.grid_size + 1, 1))  # type: ignore[arg-type]
        ax.set_facecolor("#696969")
        ax.grid(False)
        return fig, ax

    def _draw_beacons_with_light_fade(self, ax: Axes):
        for i in range(self.environment.beacons.shape[1]):
            beacon_x, beacon_y = self.environment.beacons[0, i], self.environment.beacons[1, i]
            for j in range(15):
                radius = self.environment.beacon_radius + j * 0.05
                alpha = 0.9 * np.exp(-j * 0.3)
                circle = plt.Circle(  # type: ignore[attr-defined]
                    (beacon_x, beacon_y),
                    float(radius),  # type: ignore[arg-type]
                    facecolor="white",
                    edgecolor="none",
                    alpha=float(alpha),
                )
                ax.add_patch(circle)

    def _draw_environment_elements(self, ax: Axes):
        ax.scatter(
            self.environment.beacons[0],
            self.environment.beacons[1],
            color="blue",
            marker="^",
            s=100,
            label="Beacons",
        )
        ax.scatter(
            self.environment.goal_state[0],
            self.environment.goal_state[1],
            color="green",
            marker="*",
            s=200,
            label="Goal State",
        )
        ax.scatter(
            self.environment.start_state[0],
            self.environment.start_state[1],
            color="red",
            label="Start State",
        )
        for i in range(self.environment.obstacles.shape[1]):
            obstacle_x, obstacle_y = (
                self.environment.obstacles[0, i],
                self.environment.obstacles[1, i],
            )
            circle = plt.Circle(  # type: ignore[attr-defined]
                (obstacle_x, obstacle_y),
                float(self.environment.obstacle_radius),  # type: ignore[arg-type]
                facecolor="red",
                edgecolor="none",
                alpha=0.3,
            )
            ax.add_patch(circle)
        if self.environment.obstacles.size > 0:
            ax.scatter(
                self.environment.obstacles[0],
                self.environment.obstacles[1],
                color="black",
                label="Obstacles",
            )

    def _create_belief_colors(self, max_history: int) -> List[np.ndarray]:
        yellow_color = np.array([1.0, 1.0, 0.0])
        red_color = np.array([1.0, 0.0, 0.0])
        colors = []
        for i in range(max_history):
            t = (max_history - 1 - i) / max_history
            color = yellow_color * (1 - t) + red_color * t
            alpha_factor = 0.1 + 0.9 * (i / max_history)
            color = color * alpha_factor
            colors.append(color)
        return colors

    def _initialize_animation_elements(
        self, ax: Axes, path: List[np.ndarray], belief_particles_color: str
    ) -> Tuple[Line2D, Line2D, Any, List[Any]]:
        agent = cast(Line2D, ax.plot([], [], "ro", markersize=10)[0])
        path_line = cast(Line2D, ax.plot([], [], "r-", alpha=0.5, linewidth=2)[0])
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
        belief_scatters = []
        max_history = min(len(path), 10)
        colors = self._create_belief_colors(max_history)
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
        return agent, path_line, arrow, belief_scatters

    def _create_init_function(
        self, agent: Line2D, path_line: Line2D, arrow: Any, belief_scatters: List[Any]
    ):
        def init():
            agent.set_data([], [])
            path_line.set_data([], [])
            arrow.set_data(x=0, y=0, dx=0, dy=0)
            for scatter in belief_scatters:
                scatter.set_offsets(np.empty((0, 2)))
                scatter.set_sizes([])
            return [agent, path_line, arrow] + belief_scatters

        return init

    def _create_update_function(
        self,
        path: List[np.ndarray],
        actions: List[str],
        agent_belief_path: List[DiscreteDistribution],
        agent: Line2D,
        path_line: Line2D,
        arrow: Any,
        belief_scatters: List[Any],
    ):
        def update(frame):
            x = float(path[frame][0])
            y = float(path[frame][1])
            agent.set_data([x], [y])
            path_x = [float(p[0]) for p in path[: frame + 1]]
            path_y = [float(p[1]) for p in path[: frame + 1]]
            path_line.set_data(path_x, path_y)
            if frame < len(actions):
                action = actions[frame]
                if action is None:
                    dx, dy = 0, 0
                elif isinstance(action, str):
                    dx, dy = self.environment.action_to_vector[action]
                else:
                    dx, dy = action
                arrow.set_data(x=x, y=y, dx=dx, dy=dy)
            else:
                arrow.set_data(x=x, y=y, dx=0, dy=0)
            for i, scatter in enumerate(belief_scatters):
                history_frame = frame - (len(belief_scatters) - 1 - i)
                if 0 <= history_frame < len(agent_belief_path):
                    belief = agent_belief_path[history_frame]
                    if len(belief.values) > 0:
                        positions = np.array(belief.values)
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

        return update

    def visualize_path(
        self,
        path: List[np.ndarray],
        agent_belief_path: List[DiscreteDistribution],
        actions: List[str],
        cache_path: Path,
    ):
        """Create and save an animated visualization of the agent's path.

        Args:
            path: List of state positions (2D numpy arrays) along the agent's trajectory.
            agent_belief_path: List of belief distributions at each step.
            actions: List of actions taken at each step.
            cache_path: Path where to save the visualization (must end with .gif).

        Raises:
            TypeError: If cache_path is not a Path object.
            ValueError: If cache_path doesn't end with .gif.
        """
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

        belief_particles_color = "#FFFF00"
        fig, ax = self._setup_figure_and_axes()
        self._draw_beacons_with_light_fade(ax)
        self._draw_environment_elements(ax)
        agent, path_line, arrow, belief_scatters = self._initialize_animation_elements(
            ax, path, belief_particles_color
        )
        init = self._create_init_function(agent, path_line, arrow, belief_scatters)
        update = self._create_update_function(
            path, actions, agent_belief_path, agent, path_line, arrow, belief_scatters
        )
        ani = animation.FuncAnimation(
            fig, update, frames=len(path), init_func=init, blit=True, repeat=False
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title("Agent Path")
        plt.tight_layout()
        if cache_path is not None:
            ani.save(cache_path, writer="pillow", fps=2)
        plt.close()

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache visualization of agent's path and belief.

        Args:
            history: List of step data from an episode.
            cache_path: Path where to save the visualization.

        Raises:
            TypeError: If history is not a List or contains non-StepData objects,
                or if cache_path is not a Path object.
            ValueError: If history is empty or contains invalid data.
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
        if not len(agent_path) == len(agent_belief_path) == len(actions):
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
