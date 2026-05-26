# SPDX-License-Identifier: MIT

"""Visualization module for Continuous Push POMDP Environment.

This module provides visualization capabilities for Continuous Push POMDP
episodes, creating animated GIFs showing robot movement, object pushing,
obstacle collisions, and task completion.

Obstacles are axis-aligned bounding boxes rendered as rectangles.  The robot
is drawn as a circle with its configured radius.  Actions are displayed as
formatted 2D vectors.

Classes:
    ContinuousPushPOMDPVisualizer: Handles all visualization logic for
        Continuous Push POMDP
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple, cast

from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np

from POMDPPlanners.core.simulation import StepData

if TYPE_CHECKING:
    from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
        ContinuousPushPOMDP,
    )


class ContinuousPushPOMDPVisualizer:
    """Handles visualization and animation for Continuous Push POMDP environments.

    This class encapsulates all visualization logic for Continuous Push POMDP
    episodes, creating animated GIFs showing robot movement (with circular body),
    object pushing, rectangular obstacle collisions, and task completion.

    Attributes:
        env: Reference to the ContinuousPushPOMDP environment instance.
        grid_size: Size of the grid environment.
        push_threshold: Distance threshold for robot to push object.
        obstacles: Shape ``(M, 4)`` AABB array ``(cx, cy, hx, hy)``.
        robot_radius: Radius of the robot body.
    """

    def __init__(self, env: "ContinuousPushPOMDP"):
        """Initialize visualizer with environment reference.

        Args:
            env: ContinuousPushPOMDP environment instance to visualize.
        """
        self.env = env
        self.grid_size = env.grid_size
        self.push_threshold = env.push_threshold
        self.obstacles = env.obstacles
        self.robot_radius = env.robot_radius
        self.dangerous_areas = env.dangerous_areas
        self.dangerous_area_radius = env.dangerous_area_radius

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Create animated visualization of a Continuous Push POMDP episode.

        Creates an animated GIF showing the robot pushing the object toward
        the target, with rectangular obstacles, collision detection, distance
        indicators, and success feedback.

        Args:
            history: Episode history containing states, actions, and rewards.
            cache_path: Path where to save the visualization (must end with .gif).

        Raises:
            ValueError: If history is empty or cache_path doesn't end with .gif.
            TypeError: If cache_path is not a Path object.
        """
        self._validate_visualization_inputs(history, cache_path)
        states, actions, rewards = self._extract_episode_data(history)
        fig, ax = self._setup_visualization_figure()
        robot_scatter, object_scatter, target_scatter = self._initialize_entity_scatters(ax)
        robot_circle_patch = self._initialize_robot_circle(ax)
        self._initialize_obstacles(ax)
        self._initialize_dangerous_areas(ax)
        push_arrow, connection_line = self._initialize_push_visuals(ax)
        action_arrow = self._initialize_action_arrow(ax)
        step_text, distance_text, reward_text, success_text, collision_text = (
            self._initialize_text_displays(ax)
        )
        proxy_action = Line2D(
            [], [], color="red", linewidth=2, marker=">", markersize=8, label="Action"
        )
        ax.legend(
            handles=ax.get_legend_handles_labels()[0] + [proxy_action],
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        animate = self._create_animate_function(
            states,
            actions,
            rewards,
            robot_scatter,
            object_scatter,
            target_scatter,
            robot_circle_patch,
            push_arrow,
            connection_line,
            action_arrow,
            step_text,
            distance_text,
            reward_text,
            success_text,
            collision_text,
        )
        ani = animation.FuncAnimation(
            fig, animate, frames=len(states), interval=1200, blit=False, repeat=True
        )
        plt.tight_layout()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(cache_path, writer="pillow", fps=0.8)  # type: ignore[arg-type]
        plt.close(fig)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_episode_data(
        self, history: List[StepData]
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        states = [step.state for step in history]
        actions = [step.action for step in history[:-1]]
        rewards = [step.reward for step in history]
        return states, actions, rewards

    # ------------------------------------------------------------------
    # Figure and artist setup
    # ------------------------------------------------------------------

    def _setup_visualization_figure(self) -> Tuple[Figure, Axes]:
        fig_temp, ax_temp = plt.subplots(figsize=(12, 10))
        fig = cast(Figure, fig_temp)
        ax = cast(Axes, ax_temp)
        ax.set_xlim(-0.5, self.grid_size + 0.5)
        ax.set_ylim(-0.5, self.grid_size + 0.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Y Position", fontsize=12)
        ax.set_title("Continuous Push POMDP Episode Visualization", fontsize=14, fontweight="bold")
        return fig, ax

    def _initialize_entity_scatters(self, ax: Axes) -> Tuple[Any, Any, Any]:
        robot_scatter = ax.scatter(
            [],
            [],
            s=200,
            c="blue",
            marker="o",
            edgecolor="darkblue",
            linewidth=2,
            zorder=5,
            label="Robot",
        )
        object_scatter = ax.scatter(
            [],
            [],
            s=180,
            c="orange",
            marker="s",
            edgecolor="darkorange",
            linewidth=2,
            zorder=4,
            label="Object",
        )
        target_scatter = ax.scatter(
            [],
            [],
            s=250,
            c="gold",
            marker="*",
            edgecolor="darkgoldenrod",
            linewidth=2,
            zorder=3,
            label="Target",
        )
        return robot_scatter, object_scatter, target_scatter

    def _initialize_robot_circle(self, ax: Axes) -> Circle:
        robot_circle = Circle(
            (0, 0),
            self.robot_radius,
            facecolor="blue",
            edgecolor="darkblue",
            alpha=0.25,
            linewidth=1.5,
            zorder=4,
        )
        ax.add_patch(robot_circle)
        robot_circle.set_visible(False)
        # Proxy marker for legend (Circle patches don't render in legends)
        ax.plot(
            [],
            [],
            "o",
            markersize=12,
            markerfacecolor="blue",
            markeredgecolor="darkblue",
            alpha=0.25,
            label="Robot Radius",
        )
        return robot_circle

    def _initialize_obstacles(self, ax: Axes) -> None:
        for i in range(self.obstacles.shape[0]):
            cx, cy, hx, hy = self.obstacles[i]
            rect = Rectangle(
                (cx - hx, cy - hy),
                2 * hx,
                2 * hy,
                facecolor="red",
                edgecolor="darkred",
                alpha=0.6,
                linewidth=2,
                zorder=2,
            )
            ax.add_patch(rect)
            if i == 0:
                ax.scatter(cx, cy, s=1, c="red", label="Obstacles")

    def _initialize_dangerous_areas(self, ax: Axes) -> None:
        for i, (cx, cy) in enumerate(self.dangerous_areas):
            danger_circle = Circle(
                (cx, cy),
                float(self.dangerous_area_radius),  # type: ignore[arg-type]
                facecolor="red",
                edgecolor="none",
                alpha=0.3,
                zorder=1,
                label="Dangerous Areas" if i == 0 else "",
            )
            ax.add_patch(danger_circle)

    def _initialize_push_visuals(self, ax: Axes) -> Tuple[Any, Line2D]:
        push_arrow = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 3, "alpha": 0.8},
            zorder=6,
            visible=False,
        )
        connection_line = cast(Line2D, ax.plot([], [], "r-", alpha=0.6, linewidth=2, zorder=1)[0])
        return push_arrow, connection_line

    def _initialize_action_arrow(self, ax: Axes) -> Any:
        action_arrow = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
            zorder=7,
        )
        return action_arrow

    def _update_action_arrow(
        self, action_arrow: Any, actions: List[Any], frame: int, robot_pos: np.ndarray
    ) -> None:
        if frame >= len(actions):
            action_arrow.set_visible(False)
            return
        direction = self._action_to_vector(actions[frame])
        mag = float(np.linalg.norm(direction))
        if mag > 1e-12:
            unit = direction / mag
            arrow_scale = 0.6
            action_arrow.set_position((robot_pos[0], robot_pos[1]))
            action_arrow.xy = (
                robot_pos[0] + unit[0] * arrow_scale,
                robot_pos[1] + unit[1] * arrow_scale,
            )
            action_arrow.set_visible(True)
        else:
            action_arrow.set_visible(False)

    def _initialize_text_displays(self, ax: Axes) -> Tuple[Any, Any, Any, Any, Any]:
        step_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightblue", "alpha": 0.8},
            verticalalignment="top",
            horizontalalignment="left",
        )
        distance_text = ax.text(
            0.02,
            0.88,
            "",
            transform=ax.transAxes,
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8},
            verticalalignment="top",
            horizontalalignment="left",
        )
        reward_text = ax.text(
            0.02,
            0.78,
            "",
            transform=ax.transAxes,
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgreen", "alpha": 0.8},
            verticalalignment="top",
            horizontalalignment="left",
        )
        success_text = ax.text(
            0.5,
            0.5,
            "",
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            color="darkgreen",
            bbox={
                "boxstyle": "round,pad=1.0",
                "facecolor": "lightgreen",
                "edgecolor": "darkgreen",
                "linewidth": 3,
                "alpha": 0.9,
            },
            horizontalalignment="center",
            verticalalignment="center",
            visible=False,
            zorder=10,
        )
        collision_text = ax.text(
            0.5,
            0.3,
            "",
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            color="darkred",
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": "lightcoral",
                "edgecolor": "darkred",
                "linewidth": 2,
                "alpha": 0.9,
            },
            horizontalalignment="center",
            verticalalignment="center",
            visible=False,
            zorder=10,
        )
        return step_text, distance_text, reward_text, success_text, collision_text

    # ------------------------------------------------------------------
    # Per-frame update helpers
    # ------------------------------------------------------------------

    def _update_entity_positions(
        self,
        robot_scatter: Any,
        object_scatter: Any,
        target_scatter: Any,
        robot_circle_patch: Circle,
        state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        robot_pos = state[:2]
        object_pos = state[2:4]
        target_pos = state[4:6]
        robot_scatter.set_offsets([robot_pos])
        object_scatter.set_offsets([object_pos])
        target_scatter.set_offsets([target_pos])
        robot_circle_patch.set_center(tuple(robot_pos))
        robot_circle_patch.set_visible(True)
        return robot_pos, object_pos, target_pos

    _STRING_ACTION_TO_VECTOR = {
        "up": np.array([0.0, 1.0]),
        "down": np.array([0.0, -1.0]),
        "right": np.array([1.0, 0.0]),
        "left": np.array([-1.0, 0.0]),
    }

    def _action_to_vector(self, action: Any) -> np.ndarray:
        if isinstance(action, str):
            return self._STRING_ACTION_TO_VECTOR.get(action, np.array([0.0, 0.0]))
        return np.asarray(action, dtype=float)

    def _update_push_visualization(
        self,
        push_arrow: Any,
        connection_line: Line2D,
        actions: List[Any],
        frame: int,
        robot_pos: np.ndarray,
        object_pos: np.ndarray,
        robot_to_object_dist: float,
    ) -> None:
        if frame < len(actions):
            action_vector = self._action_to_vector(actions[frame])
            action_norm = float(np.linalg.norm(action_vector))
            is_pushing = robot_to_object_dist < self.push_threshold
            if is_pushing and action_norm > 1e-12:
                direction = action_vector / action_norm
                arrow_scale = 0.6
                push_arrow.set_position(robot_pos)
                push_arrow.xy = robot_pos + direction * arrow_scale
                push_arrow.set_visible(True)
                connection_line.set_data(
                    [robot_pos[0], object_pos[0]], [robot_pos[1], object_pos[1]]
                )
            else:
                push_arrow.set_visible(False)
                connection_line.set_data([], [])
        else:
            push_arrow.set_visible(False)
            connection_line.set_data([], [])

    def _format_action_label(self, action: Any) -> str:
        if isinstance(action, str):
            return action
        vec = np.asarray(action, dtype=float)
        return f"({vec[0]:.2f}, {vec[1]:.2f})"

    def _update_text_displays(
        self,
        step_text: Any,
        distance_text: Any,
        reward_text: Any,
        success_text: Any,
        collision_text: Any,
        frame: int,
        states: List[Any],
        actions: List[Any],
        rewards: List[Any],
        distance_to_target: float,
        robot_to_object_dist: float,
        robot_collision: bool,
        object_collision: bool,
    ) -> None:
        if frame < len(actions):
            action_name = self._format_action_label(actions[frame])
        else:
            action_name = "Terminal"
        step_text.set_text(f"Step: {frame+1}/{len(states)}\nAction: {action_name}")
        distance_text.set_text(
            f"Object -> Target: {distance_to_target:.2f}\n"
            + f"Robot -> Object: {robot_to_object_dist:.2f}"
        )
        raw_reward = rewards[frame] if frame < len(rewards) else None
        current_reward = raw_reward if raw_reward is not None else 0.0
        total_reward = sum(r for r in rewards[: frame + 1] if r is not None)
        reward_text.set_text(f"Step Reward: {current_reward:.1f}\nTotal Reward: {total_reward:.1f}")
        if distance_to_target < 0.5:
            success_text.set_text("TARGET REACHED!\nEpisode Complete")
            success_text.set_visible(True)
        else:
            success_text.set_visible(False)
        if robot_collision or object_collision:
            collision_parts = []
            if robot_collision:
                collision_parts.append("Robot")
            if object_collision:
                collision_parts.append("Object")
            collision_text.set_text(f'{" & ".join(collision_parts)} Collision!')
            collision_text.set_visible(True)
        else:
            collision_text.set_visible(False)

    # ------------------------------------------------------------------
    # Animation factory
    # ------------------------------------------------------------------

    def _create_animate_function(
        self,
        states: List[Any],
        actions: List[Any],
        rewards: List[Any],
        robot_scatter: Any,
        object_scatter: Any,
        target_scatter: Any,
        robot_circle_patch: Circle,
        push_arrow: Any,
        connection_line: Line2D,
        action_arrow: Any,
        step_text: Any,
        distance_text: Any,
        reward_text: Any,
        success_text: Any,
        collision_text: Any,
    ) -> Any:
        def animate(frame: int) -> Tuple[Any, ...]:
            if frame >= len(states):
                return (
                    robot_scatter,
                    object_scatter,
                    target_scatter,
                    robot_circle_patch,
                    push_arrow,
                    connection_line,
                    action_arrow,
                    step_text,
                    distance_text,
                    reward_text,
                    success_text,
                    collision_text,
                )
            state = states[frame]
            robot_pos, object_pos, target_pos = self._update_entity_positions(
                robot_scatter, object_scatter, target_scatter, robot_circle_patch, state
            )
            distance_to_target = float(np.linalg.norm(object_pos - target_pos))
            robot_to_object_dist = float(np.linalg.norm(robot_pos - object_pos))
            robot_collision = (
                self.env._is_circle_colliding_with_obstacle(  # pylint: disable=protected-access
                    robot_pos, self.robot_radius
                )
            )
            object_collision = self.env._is_point_colliding_with_obstacle(
                object_pos
            )  # pylint: disable=protected-access
            self._update_push_visualization(
                push_arrow,
                connection_line,
                actions,
                frame,
                robot_pos,
                object_pos,
                robot_to_object_dist,
            )
            self._update_action_arrow(action_arrow, actions, frame, robot_pos)
            self._update_text_displays(
                step_text,
                distance_text,
                reward_text,
                success_text,
                collision_text,
                frame,
                states,
                actions,
                rewards,
                distance_to_target,
                robot_to_object_dist,
                robot_collision,
                object_collision,
            )
            return (
                robot_scatter,
                object_scatter,
                target_scatter,
                robot_circle_patch,
                push_arrow,
                connection_line,
                action_arrow,
                step_text,
                distance_text,
                reward_text,
                success_text,
                collision_text,
            )

        return animate
