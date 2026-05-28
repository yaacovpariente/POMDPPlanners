# SPDX-License-Identifier: MIT

"""Visualization utilities for Safety Ant Velocity POMDP Environment.

This module provides visualization capabilities for the Safety Ant Velocity POMDP,
creating animated GIF visualizations of episode trajectories with safety zones,
velocity vectors, and safety constraint violations.

Classes:
    SafeAntVelocityVisualizer: Creates animated visualizations of episodes
"""

from pathlib import Path
from typing import List

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

from POMDPPlanners.core.simulation import StepData


class SafeAntVelocityVisualizer:
    """Visualizer for Safety Ant Velocity POMDP episodes.

    This class creates animated visualizations showing the ant's movement trajectory,
    velocity vectors, force applications, safety zones, and safety constraint violations.

    Attributes:
        env: The SafeAntVelocityPOMDP environment instance
        safe_velocity_threshold: Maximum safe velocity magnitude
        max_force: Maximum force that can be applied
    """

    def __init__(self, env):
        """Initialize the visualizer.

        Args:
            env: SafeAntVelocityPOMDP environment instance
        """
        self.env = env
        self.safe_velocity_threshold = env.safe_velocity_threshold
        self.max_force = env.max_force

    def create_animation(self, history: List[StepData], cache_path: Path) -> None:
        """Create animated visualization of the safety ant velocity episode.

        Creates an animated GIF showing the ant's movement trajectory with velocity vectors,
        safety zones, force applications, and safety constraint violations.

        Args:
            history: Episode history containing states, actions, and rewards
            cache_path: Path where to save the visualization (must end with .gif)

        Raises:
            ValueError: If history is empty or cache_path doesn't end with .gif
            TypeError: If cache_path is not a Path object
        """
        self._validate_visualization_inputs(history, cache_path)
        states, actions, rewards = self._extract_episode_data(history)
        fig, (ax_main, ax_speed) = self._setup_visualization_plots(states)
        visual_elements = self._initialize_visual_elements(ax_main, ax_speed)
        animate_fn = self._create_animation_function(states, actions, rewards, visual_elements)
        self._save_animation(fig, animate_fn, len(states), cache_path)

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

    def _extract_episode_data(self, history: List[StepData]) -> tuple:
        states = [step.state for step in history]
        actions = [step.action for step in history[:-1]]
        rewards = [step.reward for step in history]
        return states, actions, rewards

    def _setup_visualization_plots(self, states):
        all_positions = np.array([state[:2] for state in states])
        x_min, x_max = all_positions[:, 0].min() - 1, all_positions[:, 0].max() + 1
        y_min, y_max = all_positions[:, 1].min() - 1, all_positions[:, 1].max() + 1

        fig, (ax_main, ax_speed) = plt.subplots(1, 2, figsize=(16, 8))

        self._configure_main_plot(ax_main, x_min, x_max, y_min, y_max)
        self._configure_speed_plot(ax_speed, states)

        return fig, (ax_main, ax_speed)

    def _configure_main_plot(self, ax_main, x_min, x_max, y_min, y_max):
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)
        ax_main.set_aspect("equal")
        ax_main.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax_main.set_xlabel("X Position", fontsize=12)
        ax_main.set_ylabel("Y Position", fontsize=12)
        ax_main.set_title(
            "Safety Ant Velocity POMDP: Trajectory & Safety Zones",
            fontsize=14,
            fontweight="bold",
        )

    def _configure_speed_plot(self, ax_speed, states):
        ax_speed.set_xlim(0, len(states))
        max_speed = max(np.linalg.norm(state[2:4]) for state in states) if states else 0  # type: ignore
        ax_speed.set_ylim(0, max(max_speed * 1.1, self.safe_velocity_threshold * 2))
        ax_speed.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax_speed.set_xlabel("Time Step", fontsize=12)
        ax_speed.set_ylabel("Speed (Velocity Magnitude)", fontsize=12)
        ax_speed.set_title("Speed Over Time", fontsize=14, fontweight="bold")

        ax_speed.axhline(
            y=self.safe_velocity_threshold,
            color="orange",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Safety Threshold ({self.safe_velocity_threshold:.1f})",
        )
        ax_speed.axhline(
            y=self.safe_velocity_threshold * 1.5,
            color="red",
            linestyle="-",
            linewidth=2,
            alpha=0.8,
            label=f"Critical Threshold ({self.safe_velocity_threshold * 1.5:.1f})",
        )

    def _initialize_visual_elements(self, ax_main, ax_speed):
        elements = {}
        elements["ant_scatter"] = ax_main.scatter(
            [],
            [],
            s=200,
            c="blue",
            marker="o",
            edgecolor="darkblue",
            linewidth=2,
            zorder=5,
            label="Ant",
        )
        (elements["path_line"],) = ax_main.plot(
            [], [], "b-", alpha=0.6, linewidth=2, label="Trajectory"
        )
        elements["velocity_arrow"] = ax_main.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "green", "lw": 3, "alpha": 0.8},
            zorder=6,
            visible=False,
        )
        elements["force_arrow"] = ax_main.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "purple", "lw": 3, "alpha": 0.8},
            zorder=6,
            visible=False,
        )
        elements["safety_circle"] = plt.Circle(  # type: ignore
            (0, 0),
            0,
            fill=False,
            edgecolor="orange",
            linewidth=2,
            alpha=0.6,
            linestyle="--",
            visible=False,
        )
        elements["critical_circle"] = plt.Circle(  # type: ignore
            (0, 0),
            0,
            fill=False,
            edgecolor="red",
            linewidth=2,
            alpha=0.6,
            linestyle="-",
            visible=False,
        )
        ax_main.add_patch(elements["safety_circle"])
        ax_main.add_patch(elements["critical_circle"])

        (elements["speed_line"],) = ax_speed.plot([], [], "b-", linewidth=2, label="Speed")
        elements["speed_points"] = ax_speed.scatter(
            [],
            [],
            s=50,
            c=[],
            cmap="RdYlGn_r",
            vmin=0,
            vmax=self.safe_velocity_threshold * 1.5,
            edgecolor="black",
            linewidth=1,
            zorder=5,
        )

        elements["step_text"] = self._create_text_box(ax_main, 0.02, 0.98, "lightblue", fontsize=12)
        elements["velocity_text"] = self._create_text_box(
            ax_main, 0.02, 0.88, "lightyellow", fontsize=11
        )
        elements["reward_text"] = self._create_text_box(
            ax_main, 0.02, 0.78, "lightgreen", fontsize=11
        )
        elements["safety_text"] = ax_main.text(
            0.5,
            0.95,
            "",
            transform=ax_main.transAxes,
            fontsize=14,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "green", "alpha": 0.9},
            horizontalalignment="center",
            verticalalignment="top",
            visible=False,
            zorder=10,
        )

        ax_main.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax_speed.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        return elements

    def _create_text_box(self, ax, x, y, color, fontsize):
        return ax.text(
            x,
            y,
            "",
            transform=ax.transAxes,
            fontsize=fontsize,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "alpha": 0.8},
            verticalalignment="top",
            horizontalalignment="left",
        )

    def _create_animation_function(self, states, actions, rewards, elements):
        def animate(frame):
            if frame >= len(states):
                return tuple(elements.values())

            state = states[frame]
            position, velocity = state[:2], state[2:4]
            speed = np.linalg.norm(velocity)

            self._update_trajectory_elements(elements, position, velocity, states, frame)
            self._update_force_vector(elements, position, actions, frame)
            self._update_safety_circles(elements, position, speed)
            self._update_speed_plot(elements, states, frame)
            self._update_text_displays(elements, frame, states, actions, rewards, velocity, speed)
            self._update_safety_status(elements, speed)

            return tuple(elements.values())

        return animate

    def _update_trajectory_elements(self, elements, position, velocity, states, frame):
        elements["ant_scatter"].set_offsets([position])

        path_positions = [s[:2] for s in states[: frame + 1]]
        if len(path_positions) > 1:
            path_x = [pos[0] for pos in path_positions]
            path_y = [pos[1] for pos in path_positions]
            elements["path_line"].set_data(path_x, path_y)

        if np.linalg.norm(velocity) > 0.01:
            elements["velocity_arrow"].set_position(position)
            elements["velocity_arrow"].xy = position + velocity * 0.5
            elements["velocity_arrow"].set_visible(True)
        else:
            elements["velocity_arrow"].set_visible(False)

    def _update_force_vector(self, elements, position, actions, frame):
        if frame < len(actions):
            action = actions[frame]
            force_scales = [0.0, 0.33, 0.67, 1.0]
            force_magnitude = force_scales[action] * self.max_force
            if force_magnitude > 0:
                force_direction = np.array([1.0, 0.5])
                force_direction = force_direction / np.linalg.norm(force_direction)
                elements["force_arrow"].set_position(position)
                elements["force_arrow"].xy = position + force_direction * force_magnitude * 0.8
                elements["force_arrow"].set_visible(True)
            else:
                elements["force_arrow"].set_visible(False)
        else:
            elements["force_arrow"].set_visible(False)

    def _update_safety_circles(self, elements, position, speed):
        safety_radius = (
            speed / self.safe_velocity_threshold * 1.0 if self.safe_velocity_threshold > 0 else 0.1
        )
        critical_radius = (
            speed / (self.safe_velocity_threshold * 1.5) * 1.5
            if self.safe_velocity_threshold > 0
            else 0.1
        )

        elements["safety_circle"].center = position
        elements["safety_circle"].set_radius(max(0.1, float(safety_radius)))
        elements["safety_circle"].set_visible(True)

        elements["critical_circle"].center = position
        elements["critical_circle"].set_radius(max(0.1, float(critical_radius)))
        elements["critical_circle"].set_visible(bool(speed > self.safe_velocity_threshold))

    def _update_speed_plot(self, elements, states, frame):
        speeds = [np.linalg.norm(s[2:4]) for s in states[: frame + 1]]
        time_steps = list(range(len(speeds)))
        elements["speed_line"].set_data(time_steps, speeds)

        colors = [
            (
                "green"
                if s <= self.safe_velocity_threshold
                else "orange" if s <= self.safe_velocity_threshold * 1.5 else "red"
            )
            for s in speeds
        ]
        if speeds:
            elements["speed_points"].set_offsets(list(zip(time_steps, speeds)))
            elements["speed_points"].set_color(colors)

    def _update_text_displays(self, elements, frame, states, actions, rewards, velocity, speed):
        action_name = f"Force Level {actions[frame]}" if frame < len(actions) else "Terminal"
        elements["step_text"].set_text(f"Step: {frame+1}/{len(states)}\nAction: {action_name}")

        elements["velocity_text"].set_text(
            f"Velocity: [{velocity[0]:.2f}, {velocity[1]:.2f}]\nSpeed: {speed:.2f}"
        )

        current_reward = rewards[frame] if frame < len(rewards) else 0.0
        total_reward = sum(rewards[: frame + 1])  # type: ignore
        elements["reward_text"].set_text(
            f"Step Reward: {current_reward:.1f}\nTotal Reward: {total_reward:.1f}"
        )

    def _update_safety_status(self, elements, speed):
        if speed > self.safe_velocity_threshold * 1.5:
            elements["safety_text"].set_text("⚠ CRITICAL VIOLATION ⚠\nTerminal State!")
            elements["safety_text"].get_bbox_patch().set_facecolor("red")
            elements["safety_text"].set_visible(True)
        elif speed > self.safe_velocity_threshold:
            elements["safety_text"].set_text("⚠ SAFETY VIOLATION ⚠")
            elements["safety_text"].get_bbox_patch().set_facecolor("orange")
            elements["safety_text"].set_visible(True)
        else:
            elements["safety_text"].set_text("✓ SAFE OPERATION ✓")
            elements["safety_text"].get_bbox_patch().set_facecolor("green")
            elements["safety_text"].set_visible(True)

    def _save_animation(self, fig, animate_fn, num_frames, cache_path):
        ani = animation.FuncAnimation(
            fig, animate_fn, frames=num_frames, interval=1200, blit=False, repeat=True
        )
        plt.tight_layout()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(cache_path, writer="pillow", fps=0.8)  # type: ignore[arg-type]
        plt.close(fig)
