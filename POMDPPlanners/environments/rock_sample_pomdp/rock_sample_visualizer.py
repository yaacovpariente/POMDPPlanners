"""Visualization module for RockSample POMDP Environment.

This module provides visualization capabilities for RockSample POMDP episodes,
creating animated GIFs showing robot movement, rock sampling, sensor usage,
and exit behavior.

Classes:
    RockSampleVisualizer: Handles all visualization logic for RockSample POMDP
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast

from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    get_robot_pos,
    get_rocks,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
        RockSamplePOMDP,
        RockSampleState,
    )


class RockSampleVisualizer:
    """Handles visualization and animation for RockSample POMDP environments.

    This class encapsulates all visualization logic for RockSample POMDP episodes,
    creating animated GIFs showing robot movement, rock sampling, sensor checks,
    dangerous areas, and exit behavior.

    Attributes:
        env: Reference to the RockSamplePOMDP environment instance
        map_size: Grid dimensions as (rows, cols)
        rock_positions: List of rock positions
        action_names: Names of available actions
        action_to_vector: Mapping from action indices to direction vectors
        dangerous_areas: List of dangerous area center positions
        dangerous_area_radius: Radius around dangerous area centers
    """

    def __init__(self, env: "RockSamplePOMDP"):
        """Initialize visualizer with environment reference.

        Args:
            env: RockSamplePOMDP environment instance to visualize
        """
        self.env = env
        self.map_size = env.map_size
        self.rock_positions = env.rock_positions
        self.action_names = env.action_names
        self.action_to_vector = env.action_to_vector
        self.dangerous_areas = env.dangerous_areas
        self.dangerous_area_radius = env.dangerous_area_radius

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Create animated visualization of a RockSample POMDP episode.

        Creates an animated GIF showing the robot navigating, sampling rocks,
        using sensors, and exiting the grid.

        Args:
            history: Episode history containing states, actions, and rewards
            cache_path: Path where to save the visualization (must end with .gif)

        Raises:
            ValueError: If history is empty or cache_path doesn't end with .gif
            TypeError: If cache_path is not a Path object or history is invalid
        """
        self._validate_visualization_inputs(history, cache_path)
        path, actions = self._extract_path_and_actions(history)
        self.visualize_path(path, actions, cache_path)

    def visualize_path(
        self, path: List["RockSampleState"], actions: List[int], cache_path: Path
    ) -> None:
        """Visualize robot path through the environment.

        Args:
            path: List of states representing the path
            actions: List of actions taken at each state
            cache_path: Path where to save the animation
        """
        self._validate_path_cache_inputs(cache_path)
        fig, ax = self._setup_path_visualization_plot()
        visual_elements = self._initialize_path_visual_elements(ax)
        animate_fn = self._create_path_animation_function(path, actions, visual_elements)
        self._save_path_animation(fig, animate_fn, len(path), cache_path)

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

    def _extract_path_and_actions(
        self, history: List[StepData]
    ) -> Tuple[List["RockSampleState"], List[int]]:
        path = [step.state for step in history]
        actions = [step.action for step in history[:-1]]  # Last step has no action
        return path, actions

    def _validate_path_cache_inputs(self, cache_path: Path) -> None:
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

    def _setup_path_visualization_plot(self) -> Tuple[Figure, Axes]:
        fig_temp, ax_temp = plt.subplots(figsize=(10, 8))
        fig = cast(Figure, fig_temp)
        ax = cast(Axes, ax_temp)
        ax.set_xlim(-0.5, self.map_size[1] - 0.5)
        ax.set_ylim(self.map_size[0] - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title("RockSample POMDP Episode Visualization")
        return fig, ax

    def _initialize_path_visual_elements(self, ax: Axes) -> Dict[str, Any]:
        elements: Dict[str, Any] = {}
        elements["rock_scatters"] = self._create_rock_scatters(ax)
        self._add_danger_areas_to_plot(ax)
        self._add_exit_zone_to_plot(ax)
        elements["robot_scatter"] = ax.scatter(
            [], [], s=150, c="blue", marker="o", zorder=5, label="Robot"
        )
        elements["path_line"] = cast(
            Line2D, ax.plot([], [], "b-", alpha=0.5, linewidth=2, label="Path")[0]
        )
        elements["arrow"] = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
            zorder=6,
        )
        elements["action_text"] = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            verticalalignment="top",
        )
        elements["sample_text"] = ax.text(
            0.02,
            0.02,
            "",
            transform=ax.transAxes,
            fontsize=20,
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
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        return elements

    def _create_rock_scatters(self, ax: Axes) -> List[Any]:
        rock_scatters = []
        for i in range(len(self.rock_positions)):
            scatter = ax.scatter([], [], s=200, marker="s", alpha=0.7, label=f"Rock {i}")
            rock_scatters.append(scatter)
        return rock_scatters

    def _add_danger_areas_to_plot(self, ax: Axes) -> None:
        for i, danger_center in enumerate(self.dangerous_areas):
            row, col = danger_center
            circle = plt.Circle(  # type: ignore[attr-defined]
                (col, row),
                float(self.dangerous_area_radius),  # type: ignore[arg-type]
                facecolor="red",
                edgecolor="none",
                alpha=0.3,
                label="Dangerous Areas" if i == 0 else "",
            )
            ax.add_patch(circle)

    def _add_exit_zone_to_plot(self, ax: Axes) -> None:
        exit_x = self.map_size[1] - 0.5
        ax.axvline(
            x=float(exit_x), color="gold", linewidth=3, alpha=0.7, label="Exit"  # type: ignore[arg-type]
        )

    def _create_path_animation_function(
        self,
        path: List["RockSampleState"],
        actions: List[int],
        elements: Dict[str, Any],
    ) -> Any:
        def animate(frame: int) -> Tuple[Any, ...]:
            if frame >= len(path):
                return tuple(
                    [
                        elements["robot_scatter"],
                        elements["path_line"],
                        elements["arrow"],
                        elements["action_text"],
                        elements["sample_text"],
                    ]
                    + elements["rock_scatters"]
                )

            state = path[frame]
            self._update_rock_displays(state, elements["rock_scatters"])
            self._update_robot_and_path(state, path, frame, actions, elements)
            self._update_action_display(frame, state, path, actions, elements)

            return tuple(
                [
                    elements["robot_scatter"],
                    elements["path_line"],
                    elements["arrow"],
                    elements["action_text"],
                    elements["sample_text"],
                ]
                + elements["rock_scatters"]
            )

        return animate

    def _update_rock_displays(self, state: "RockSampleState", rock_scatters: List[Any]) -> None:
        rocks = get_rocks(state)
        for i, rock_pos in enumerate(self.rock_positions):
            if i < len(rocks):
                color = "green" if rocks[i] else "red"
                rock_scatters[i].set_offsets([[rock_pos[1], rock_pos[0]]])
                rock_scatters[i].set_color(color)

    def _update_robot_and_path(
        self,
        state: "RockSampleState",
        path: List["RockSampleState"],
        frame: int,
        actions: List[int],
        elements: Dict[str, Any],
    ) -> None:
        robot_pos = get_robot_pos(state)
        if robot_pos == (-1, -1):
            elements["robot_scatter"].set_offsets(np.empty((0, 2)))
            elements["arrow"].set_visible(False)
            elements["sample_text"].set_visible(False)
        else:
            elements["robot_scatter"].set_offsets([[robot_pos[1], robot_pos[0]]])
            self._update_action_arrow(robot_pos, frame, actions, elements)

        self._update_path_line(path, frame, elements)

    def _update_action_arrow(
        self,
        robot_pos: Tuple[int, int],
        frame: int,
        actions: List[int],
        elements: Dict[str, Any],
    ) -> None:
        if frame < len(actions):
            action = actions[frame]
            dx, dy = self.action_to_vector.get(action, (0, 0))
            if dx != 0 or dy != 0:
                x, y = robot_pos[1], robot_pos[0]
                arrow_scale = 0.4
                elements["arrow"].set_position((x, y))
                elements["arrow"].xy = (x + dx * arrow_scale, y + dy * arrow_scale)
                elements["arrow"].set_visible(True)
            else:
                elements["arrow"].set_visible(False)
        else:
            elements["arrow"].set_visible(False)

    def _update_path_line(
        self, path: List["RockSampleState"], frame: int, elements: Dict[str, Any]
    ) -> None:
        valid_positions = [
            pos for pos in [get_robot_pos(p) for p in path[: frame + 1]] if pos != (-1, -1)
        ]
        if valid_positions:
            path_x = [pos[1] for pos in valid_positions]
            path_y = [pos[0] for pos in valid_positions]
            elements["path_line"].set_data(path_x, path_y)

    def _update_action_display(
        self,
        frame: int,
        state: "RockSampleState",
        path: List["RockSampleState"],
        actions: List[int],
        elements: Dict[str, Any],
    ) -> None:
        if frame < len(actions):
            action = actions[frame]
            action_name = self.action_names[action]
            elements["action_text"].set_text(f"Step: {frame+1}/{len(path)}\nAction: {action_name}")
            self._handle_sample_action_display(action, state, elements)
        else:
            elements["action_text"].set_text(f"Step: {frame+1}/{len(path)}\nAction: Terminal")
            elements["sample_text"].set_visible(False)

    def _handle_sample_action_display(
        self, action: int, state: "RockSampleState", elements: Dict[str, Any]
    ) -> None:
        if action == 0:
            sample_success = self._check_sample_success(state)
            if sample_success:
                self._show_sample_success(elements)
            else:
                self._show_sample_failure(elements)
        else:
            elements["sample_text"].set_visible(False)

    def _check_sample_success(self, state: "RockSampleState") -> bool:
        robot_row, robot_col = get_robot_pos(state)
        rocks = get_rocks(state)
        for i, rock_pos in enumerate(self.rock_positions):
            if (robot_row, robot_col) == rock_pos:
                return rocks[i]
        return False

    def _show_sample_success(self, elements: Dict[str, Any]) -> None:
        elements["sample_text"].set_text("★ VALUABLE! ★")
        elements["sample_text"].set_bbox(
            {
                "boxstyle": "round,pad=0.5",
                "facecolor": "lightgreen",
                "edgecolor": "green",
                "linewidth": 3,
                "alpha": 0.9,
            }
        )
        elements["sample_text"].set_color("darkgreen")
        elements["sample_text"].set_visible(True)

    def _show_sample_failure(self, elements: Dict[str, Any]) -> None:
        elements["sample_text"].set_text("✗ WORTHLESS! ✗")
        elements["sample_text"].set_bbox(
            {
                "boxstyle": "round,pad=0.5",
                "facecolor": "lightcoral",
                "edgecolor": "red",
                "linewidth": 3,
                "alpha": 0.9,
            }
        )
        elements["sample_text"].set_color("darkred")
        elements["sample_text"].set_visible(True)

    def _save_path_animation(
        self, fig: Figure, animate_fn: Any, num_frames: int, cache_path: Path
    ) -> None:
        ani = animation.FuncAnimation(
            fig, animate_fn, frames=num_frames, interval=1000, blit=False, repeat=False
        )
        plt.tight_layout()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ani.save(cache_path, writer="pillow", fps=1)
        plt.close()
