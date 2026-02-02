"""Visualization module for Push POMDP Environment.

This module provides visualization capabilities for Push POMDP episodes,
creating animated GIFs showing robot movement, object pushing, obstacle
collisions, and task completion.

Classes:
    PushPOMDPVisualizer: Handles all visualization logic for Push POMDP
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple, cast

from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from POMDPPlanners.core.simulation import StepData

if TYPE_CHECKING:
    from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP


class PushPOMDPVisualizer:
    """Handles visualization and animation for Push POMDP environments.

    This class encapsulates all visualization logic for Push POMDP episodes,
    creating animated GIFs showing robot movement, object pushing, obstacle
    collisions, and task completion.

    Attributes:
        env: Reference to the PushPOMDP environment instance
        grid_size: Size of the grid environment
        push_threshold: Distance threshold for robot to push object
        obstacles: List of obstacle positions
        obstacle_radius: Radius of obstacles for collision detection
    """

    def __init__(self, env: "PushPOMDP"):
        """Initialize visualizer with environment reference.

        Args:
            env: PushPOMDP environment instance to visualize
        """
        self.env = env
        self.grid_size = env.grid_size
        self.push_threshold = env.push_threshold
        self.obstacles = env.obstacles
        self.obstacle_radius = env.obstacle_radius

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Create animated visualization of a Push POMDP episode.

        Creates an animated GIF showing the robot pushing the object toward the target,
        with obstacles, collision detection, distance indicators, and success feedback.

        Args:
            history: Episode history containing states, actions, and rewards
            cache_path: Path where to save the visualization (must end with .gif)

        Raises:
            ValueError: If history is empty or cache_path doesn't end with .gif
            TypeError: If cache_path is not a Path object
        """
        self._validate_visualization_inputs(history, cache_path)
        states, actions, rewards = self._extract_episode_data(history)
        fig, ax = self._setup_visualization_figure()
        robot_scatter, object_scatter, target_scatter = self._initialize_entity_scatters(ax)
        self._initialize_obstacles(ax)
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

    def _extract_episode_data(
        self, history: List[StepData]
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        states = [step.state for step in history]
        actions = [step.action for step in history[:-1]]  # Last step has no action
        rewards = [step.reward for step in history]
        return states, actions, rewards

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
        ax.set_title("Push POMDP Episode Visualization", fontsize=14, fontweight="bold")
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

    def _initialize_obstacles(self, ax: Axes) -> None:
        for i, (obs_x, obs_y) in enumerate(self.obstacles):
            obstacle_circle = plt.Circle(  # type: ignore[attr-defined]
                (obs_x, obs_y),
                float(self.obstacle_radius),  # type: ignore[arg-type]
                facecolor="red",
                edgecolor="darkred",
                alpha=0.6,
                linewidth=2,
                zorder=2,
            )
            ax.add_patch(obstacle_circle)
            if i == 0:  # Label only the first obstacle for legend
                ax.scatter(obs_x, obs_y, s=1, c="red", label="Obstacles")

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

    _ACTION_DIRS = {
        "up": np.array([0.0, 1.0]),
        "down": np.array([0.0, -1.0]),
        "right": np.array([1.0, 0.0]),
        "left": np.array([-1.0, 0.0]),
    }

    def _update_action_arrow(
        self, action_arrow: Any, actions: List[Any], frame: int, robot_pos: np.ndarray
    ) -> None:
        if frame >= len(actions):
            action_arrow.set_visible(False)
            return
        action = actions[frame]
        direction = self._ACTION_DIRS.get(action, np.array([0.0, 0.0]))
        mag = float(np.linalg.norm(direction))
        if mag > 1e-12:
            arrow_scale = 0.6
            action_arrow.set_position((robot_pos[0], robot_pos[1]))
            action_arrow.xy = (
                robot_pos[0] + direction[0] * arrow_scale,
                robot_pos[1] + direction[1] * arrow_scale,
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

    def _update_entity_positions(
        self, robot_scatter: Any, object_scatter: Any, target_scatter: Any, state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        robot_pos = state[:2]
        object_pos = state[2:4]
        target_pos = state[4:6]
        robot_scatter.set_offsets([robot_pos])
        object_scatter.set_offsets([object_pos])
        target_scatter.set_offsets([target_pos])
        return robot_pos, object_pos, target_pos

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
            action = actions[frame]
            action_vector = {
                "up": np.array([0, 1]),
                "down": np.array([0, -1]),
                "right": np.array([1, 0]),
                "left": np.array([-1, 0]),
            }.get(action, np.array([0, 0]))
            is_pushing = robot_to_object_dist < self.push_threshold
            if is_pushing and np.any(action_vector != 0):
                arrow_scale = 0.6
                push_arrow.set_position(robot_pos)
                push_arrow.xy = robot_pos + action_vector * arrow_scale
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
        action_name = actions[frame] if frame < len(actions) else "Terminal"
        step_text.set_text(f"Step: {frame+1}/{len(states)}\nAction: {action_name}")
        distance_text.set_text(
            f"Object ↔ Target: {distance_to_target:.2f}\n"
            + f"Robot ↔ Object: {robot_to_object_dist:.2f}"
        )
        current_reward = rewards[frame] if frame < len(rewards) else 0.0
        total_reward = sum(rewards[: frame + 1])  # type: ignore
        reward_text.set_text(f"Step Reward: {current_reward:.1f}\nTotal Reward: {total_reward:.1f}")
        if distance_to_target < 0.5:
            success_text.set_text("★ TARGET REACHED! ★\nEpisode Complete")
            success_text.set_visible(True)
        else:
            success_text.set_visible(False)
        if robot_collision or object_collision:
            collision_parts = []
            if robot_collision:
                collision_parts.append("Robot")
            if object_collision:
                collision_parts.append("Object")
            collision_text.set_text(f'⚠ {" & ".join(collision_parts)} Collision! ⚠')
            collision_text.set_visible(True)
        else:
            collision_text.set_visible(False)

    def _create_animate_function(
        self,
        states: List[Any],
        actions: List[Any],
        rewards: List[Any],
        robot_scatter: Any,
        object_scatter: Any,
        target_scatter: Any,
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
                robot_scatter, object_scatter, target_scatter, state
            )
            distance_to_target = float(np.linalg.norm(object_pos - target_pos))
            robot_to_object_dist = float(np.linalg.norm(robot_pos - object_pos))
            robot_collision = self.env._is_colliding_with_obstacle(robot_pos)
            object_collision = self.env._is_colliding_with_obstacle(object_pos)
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
