"""LaserTag POMDP Visualization Module.

This module provides visualization functionality for LaserTag POMDP environments,
creating animated GIF visualizations of episodes.
"""

from pathlib import Path
from typing import List, Set, Tuple, cast

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import LaserTagState

from POMDPPlanners.core.simulation import StepData


class LaserTagVisualizer:
    """Handles visualization for LaserTag POMDP environments.

    Creates animated GIF visualizations showing robot movement, opponent movement,
    walls, laser measurements, belief particles, and action indicators.

    Attributes:
        floor_shape: Grid dimensions as (rows, cols)
        walls: Set of wall positions as (row, col) tuples
        dangerous_areas: List of dangerous area center positions
        dangerous_area_radius: Radius around dangerous area centers
    """

    def __init__(
        self,
        floor_shape: Tuple[int, int],
        walls: Set[Tuple[int, int]],
        dangerous_areas: List[Tuple[int, int]],
        dangerous_area_radius: float,
    ):
        """Initialize the LaserTag visualizer.

        Args:
            floor_shape: Grid dimensions as (rows, cols)
            walls: Set of wall positions as (row, col) tuples
            dangerous_areas: List of dangerous area center positions
            dangerous_area_radius: Radius around dangerous area centers
        """
        self.floor_shape = floor_shape
        self.walls = walls
        self.dangerous_areas = dangerous_areas
        self.dangerous_area_radius = dangerous_area_radius

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Create animated GIF visualization of a LaserTag episode.

        Creates an animated visualization showing:
        - Robot movement (red circle with path trail)
        - Opponent movement (blue circle with path trail)
        - Walls (black squares)
        - Dangerous areas (red circles)
        - Action arrows showing robot's intended movement
        - Laser measurements (green rays from robot position)
        - Belief particles (if available) showing robot's belief about opponent location
        - Grid boundaries and coordinate system
        - Step counter and action labels

        Args:
            history: The history of states, actions, and observations from an episode
            cache_path: Path where to save the visualization GIF

        Raises:
            ValueError: If history is empty or contains invalid data
            TypeError: If cache_path is not a Path object or doesn't end with .gif
        """
        self._validate_visualization_inputs(history, cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        robot_path, opponent_path, actions, beliefs = self._extract_history_data(history)

        fig, ax = self._setup_visualization_figure()
        self._draw_static_elements(ax)

        (
            robot_agent,
            opponent_agent,
            robot_path_line,
            opponent_path_line,
            action_arrow,
            step_text,
            action_text,
            tag_text,
            opponent_belief_scatter,
            robot_belief_scatter,
            laser_lines,
        ) = self._create_animated_elements(ax)

        laser_directions = [
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        ]

        def init():
            robot_agent.set_data([], [])
            opponent_agent.set_data([], [])
            robot_path_line.set_data([], [])
            opponent_path_line.set_data([], [])
            action_arrow.set_position((0, 0))
            action_arrow.xy = (0, 0)
            step_text.set_text("")
            action_text.set_text("")
            tag_text.set_visible(False)
            opponent_belief_scatter.set_offsets(np.empty((0, 2)))
            robot_belief_scatter.set_offsets(np.empty((0, 2)))
            for line in laser_lines:
                line.set_data([], [])
            return [
                robot_agent,
                opponent_agent,
                robot_path_line,
                opponent_path_line,
                action_arrow,
                step_text,
                action_text,
                tag_text,
                opponent_belief_scatter,
                robot_belief_scatter,
            ] + laser_lines

        def update(frame):
            robot_pos = robot_path[frame]
            opponent_pos = opponent_path[frame]

            robot_agent.set_data([robot_pos[0]], [robot_pos[1]])
            opponent_agent.set_data([opponent_pos[0]], [opponent_pos[1]])

            robot_rows = [pos[0] for pos in robot_path[: frame + 1]]
            robot_cols = [pos[1] for pos in robot_path[: frame + 1]]
            opponent_rows = [pos[0] for pos in opponent_path[: frame + 1]]
            opponent_cols = [pos[1] for pos in opponent_path[: frame + 1]]

            robot_path_line.set_data(robot_rows, robot_cols)
            opponent_path_line.set_data(opponent_rows, opponent_cols)

            self._update_action_visualization(
                frame,
                actions,
                robot_path,
                robot_pos,
                opponent_pos,
                action_arrow,
                step_text,
                action_text,
                tag_text,
            )

            self._update_belief_visualization(
                frame, beliefs, opponent_belief_scatter, robot_belief_scatter
            )

            self._update_laser_rays(robot_pos, laser_lines, laser_directions)

            return [
                robot_agent,
                opponent_agent,
                robot_path_line,
                opponent_path_line,
                action_arrow,
                step_text,
                action_text,
                tag_text,
                opponent_belief_scatter,
                robot_belief_scatter,
            ] + laser_lines

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(robot_path),
            init_func=init,
            blit=True,
            repeat=False,
            interval=1000,
        )

        plt.tight_layout()
        anim.save(cache_path, writer="pillow", fps=1)
        plt.close(fig)

    def _validate_visualization_inputs(self, history: List[StepData], cache_path: Path) -> None:
        if not isinstance(history, List):
            raise TypeError("history must be a List object")
        if not history:
            raise ValueError("Cannot visualize empty history")
        for step in history:
            if not isinstance(step, StepData):
                raise TypeError("history must be a List of StepData objects")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

    def _extract_history_data(self, history: List[StepData]) -> Tuple:
        robot_path = []
        opponent_path = []
        actions = []
        beliefs = []

        for step in history:
            if not isinstance(step.state, LaserTagState):
                raise ValueError(f"Expected LaserTagState, got {type(step.state)}")

            robot_path.append(step.state.robot)
            opponent_path.append(step.state.opponent)
            actions.append(step.action)

            if hasattr(step, "belief") and step.belief is not None:
                beliefs.append(step.belief)
            else:
                beliefs.append(None)

        return robot_path, opponent_path, actions, beliefs

    def _setup_visualization_figure(self) -> Tuple[Figure, Axes]:
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=(14, 8))  # type: ignore[assignment]
        rows, cols = self.floor_shape
        ax.set_xlim(-0.5, rows - 0.5)
        ax.set_ylim(-0.5, cols - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()

        ax.set_xticks(range(rows))  # type: ignore[arg-type]
        ax.set_yticks(range(cols))  # type: ignore[arg-type]
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Row")
        ax.set_ylabel("Column")
        ax.set_title("LaserTag POMDP Episode Visualization")

        return fig, ax

    def _draw_static_elements(self, ax: Axes) -> None:
        for i, wall in enumerate(self.walls):
            row, col = wall
            square = plt.Rectangle(  # type: ignore
                (row - 0.4, col - 0.4),
                0.8,
                0.8,
                facecolor="black",
                edgecolor="black",
                alpha=0.7,
                label="Wall" if i == 0 else "",
            )
            ax.add_patch(square)

        for i, danger_center in enumerate(self.dangerous_areas):
            row, col = danger_center
            circle = plt.Circle(  # type: ignore[attr-defined]
                (row, col),
                float(self.dangerous_area_radius),  # type: ignore[arg-type]
                facecolor="red",
                edgecolor="none",
                alpha=0.3,
                label="Dangerous Areas" if i == 0 else "",
            )
            ax.add_patch(circle)

    def _create_animated_elements(self, ax: Axes) -> Tuple:

        robot_agent = cast(Line2D, ax.plot([], [], "ro", markersize=12, label="Robot")[0])
        opponent_agent = cast(Line2D, ax.plot([], [], "bo", markersize=12, label="Opponent")[0])
        robot_path_line = cast(
            Line2D, ax.plot([], [], "r-", alpha=0.5, linewidth=2, label="Robot Path")[0]
        )
        opponent_path_line = cast(
            Line2D, ax.plot([], [], "b-", alpha=0.5, linewidth=2, label="Opponent Path")[0]
        )

        action_arrow = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
        )

        step_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

        action_text = ax.text(
            0.02,
            0.90,
            "",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.8},
        )

        tag_text = ax.text(
            0.02,
            0.02,
            "",
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="bottom",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "gold",
                "edgecolor": "red",
                "linewidth": 3,
                "alpha": 0.9,
            },
            color="red",
            visible=False,
        )

        opponent_belief_scatter = ax.scatter(
            [], [], c="lightblue", alpha=0.6, s=30, label="Opponent Belief Particles"
        )
        robot_belief_scatter = ax.scatter(
            [], [], c="lightcoral", alpha=0.6, s=30, label="Robot Belief Particles"
        )

        laser_lines = []
        for i in range(8):
            (line,) = ax.plot(
                [],
                [],
                "g-",
                alpha=0.4,
                linewidth=1,
                label="Laser Rays" if i == 0 else "",
            )
            laser_lines.append(line)

        ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), framealpha=0.9)

        return (
            robot_agent,
            opponent_agent,
            robot_path_line,
            opponent_path_line,
            action_arrow,
            step_text,
            action_text,
            tag_text,
            opponent_belief_scatter,
            robot_belief_scatter,
            laser_lines,
        )

    def _update_belief_particles(self, belief, opponent_belief_scatter, robot_belief_scatter):

        if hasattr(belief, "to_unique_support_distribution"):
            unique_belief = belief.to_unique_support_distribution()
            if len(unique_belief.values) > 0:
                opponent_belief_positions = []
                opponent_belief_weights = []
                robot_belief_positions = []
                robot_belief_weights = []

                for i, state in enumerate(unique_belief.values):
                    if isinstance(state, LaserTagState):
                        opponent_belief_positions.append([state.opponent[0], state.opponent[1]])
                        opponent_belief_weights.append(unique_belief.probs[i] * 100)
                        robot_belief_positions.append([state.robot[0], state.robot[1]])
                        robot_belief_weights.append(unique_belief.probs[i] * 100)

                if opponent_belief_positions:
                    opponent_belief_scatter.set_offsets(np.array(opponent_belief_positions))
                    opponent_belief_scatter.set_sizes(np.array(opponent_belief_weights))
                else:
                    opponent_belief_scatter.set_offsets(np.empty((0, 2)))

                if robot_belief_positions:
                    robot_belief_scatter.set_offsets(np.array(robot_belief_positions))
                    robot_belief_scatter.set_sizes(np.array(robot_belief_weights))
                else:
                    robot_belief_scatter.set_offsets(np.empty((0, 2)))
            else:
                opponent_belief_scatter.set_offsets(np.empty((0, 2)))
                robot_belief_scatter.set_offsets(np.empty((0, 2)))
        else:
            opponent_belief_scatter.set_offsets(np.empty((0, 2)))
            robot_belief_scatter.set_offsets(np.empty((0, 2)))

    def _update_laser_rays(
        self, robot_pos: Tuple[int, int], laser_lines: List, laser_directions: List
    ):
        for line, direction in zip(laser_lines, laser_directions):
            dr, dc = direction
            robot_r, robot_c = robot_pos
            distance = 0

            while True:
                ray_r = robot_r + dr * (distance + 1)
                ray_c = robot_c + dc * (distance + 1)

                if (
                    ray_r < 0
                    or ray_r >= self.floor_shape[0]
                    or ray_c < 0
                    or ray_c >= self.floor_shape[1]
                    or (ray_r, ray_c) in self.walls
                ):
                    break
                distance += 1

            end_x = robot_r + dr * distance
            end_y = robot_c + dc * distance
            line.set_data([robot_pos[0], end_x], [robot_pos[1], end_y])

    def _update_tag_text_success(self, tag_text):
        tag_text.set_text("🏷️ TAGGED! 🏷️")
        tag_text.set_bbox(
            {
                "boxstyle": "round,pad=0.5",
                "facecolor": "gold",
                "edgecolor": "green",
                "linewidth": 3,
                "alpha": 0.9,
            }
        )
        tag_text.set_color("green")
        tag_text.set_visible(True)

    def _update_tag_text_failure(self, tag_text):
        tag_text.set_text("❌ MISSED! ❌")
        tag_text.set_bbox(
            {
                "boxstyle": "round,pad=0.5",
                "facecolor": "lightcoral",
                "edgecolor": "red",
                "linewidth": 3,
                "alpha": 0.9,
            }
        )
        tag_text.set_color("darkred")
        tag_text.set_visible(True)

    def _update_action_visualization(
        self,
        frame: int,
        actions: List,
        robot_path: List,
        robot_pos: Tuple[int, int],
        opponent_pos: Tuple[int, int],
        action_arrow,
        step_text,
        action_text,
        tag_text,
    ):
        if frame >= len(actions):
            return

        action = actions[frame]
        action_dirs = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1), 4: (0, 0)}
        action_names = {0: "North", 1: "South", 2: "East", 3: "West", 4: "Tag"}

        if action not in action_dirs:
            return

        dr, dc = action_dirs[action]
        action_arrow.set_position((robot_pos[0], robot_pos[1]))
        action_arrow.xy = (robot_pos[0] + dr * 0.3, robot_pos[1] + dc * 0.3)

        step_text.set_text(f"Step: {frame + 1}/{len(robot_path)}")
        action_text.set_text(f'Action: {action_names.get(action, "Unknown")}')

        if action == 4:
            if robot_pos == opponent_pos:
                self._update_tag_text_success(tag_text)
            else:
                self._update_tag_text_failure(tag_text)
        else:
            tag_text.set_visible(False)

    def _update_belief_visualization(
        self, frame: int, beliefs: List, opponent_belief_scatter, robot_belief_scatter
    ):
        if frame < len(beliefs) and beliefs[frame] is not None:
            try:
                self._update_belief_particles(
                    beliefs[frame], opponent_belief_scatter, robot_belief_scatter
                )
            except Exception:
                opponent_belief_scatter.set_offsets(np.empty((0, 2)))
                robot_belief_scatter.set_offsets(np.empty((0, 2)))
        else:
            opponent_belief_scatter.set_offsets(np.empty((0, 2)))
            robot_belief_scatter.set_offsets(np.empty((0, 2)))
