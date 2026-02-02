"""Continuous LaserTag POMDP Visualization Module.

This module provides visualization for the continuous-space LaserTag
environment, creating animated GIF visualizations of episodes.
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
import numpy as np

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_geometry import (
    LASER_DIRECTIONS,
    compute_laser_measurements,
)


def _render_robot_image(size_px: int = 64) -> np.ndarray:
    """Render a red humanoid robot icon to an RGBA numpy array."""
    fig, ax = _create_icon_canvas(size_px)
    _draw_humanoid_body(ax, facecolor="#D32F2F", edgecolor="#B71C1C")
    return _figure_to_rgba(fig, size_px)


def _render_opponent_image(size_px: int = 64) -> np.ndarray:
    """Render a blue wheeled rover icon to an RGBA numpy array."""
    fig, ax = _create_icon_canvas(size_px)
    _draw_rover_body(ax, facecolor="#1976D2", edgecolor="#0D47A1")
    return _figure_to_rgba(fig, size_px)


def _create_icon_canvas(size_px: int):
    dpi = 100
    fig_size = size_px / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig, ax


def _draw_humanoid_body(ax, facecolor: str, edgecolor: str) -> None:
    # Head
    head = Circle((0.5, 0.82), 0.13, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(head)
    # Eye visor
    visor = FancyBboxPatch(
        (0.35, 0.79),
        0.30,
        0.07,
        boxstyle="round,pad=0.02",
        facecolor="#FFCDD2",
        edgecolor=edgecolor,
        linewidth=0.8,
    )
    ax.add_patch(visor)
    # Torso
    torso = FancyBboxPatch(
        (0.32, 0.40),
        0.36,
        0.30,
        boxstyle="round,pad=0.04",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.5,
    )
    ax.add_patch(torso)
    # Arms
    left_arm = FancyBboxPatch(
        (0.18, 0.42),
        0.12,
        0.25,
        boxstyle="round,pad=0.03",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
    )
    right_arm = FancyBboxPatch(
        (0.70, 0.42),
        0.12,
        0.25,
        boxstyle="round,pad=0.03",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
    )
    ax.add_patch(left_arm)
    ax.add_patch(right_arm)
    # Legs
    left_leg = FancyBboxPatch(
        (0.33, 0.10),
        0.13,
        0.28,
        boxstyle="round,pad=0.03",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
    )
    right_leg = FancyBboxPatch(
        (0.54, 0.10),
        0.13,
        0.28,
        boxstyle="round,pad=0.03",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
    )
    ax.add_patch(left_leg)
    ax.add_patch(right_leg)
    # Antenna
    ax.plot([0.5, 0.5], [0.95, 1.0], color=edgecolor, linewidth=1.5)
    ax.plot(0.5, 1.0, "o", color="#FF5252", markersize=3)


def _draw_rover_body(ax, facecolor: str, edgecolor: str) -> None:
    # Main chassis
    chassis = FancyBboxPatch(
        (0.15, 0.30),
        0.70,
        0.35,
        boxstyle="round,pad=0.05",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.5,
    )
    ax.add_patch(chassis)
    # Sensor dome
    dome = Circle((0.5, 0.72), 0.12, facecolor="#90CAF9", edgecolor=edgecolor, linewidth=1.2)
    ax.add_patch(dome)
    # Sensor eye
    eye = Circle((0.5, 0.72), 0.05, facecolor="white", edgecolor=edgecolor, linewidth=0.8)
    ax.add_patch(eye)
    # Wheels
    for wx in (0.25, 0.50, 0.75):
        wheel = Circle((wx, 0.22), 0.09, facecolor="#455A64", edgecolor="#263238", linewidth=1.2)
        ax.add_patch(wheel)
        # Wheel hub
        hub = Circle((wx, 0.22), 0.03, facecolor="#90A4AE", edgecolor="#263238", linewidth=0.6)
        ax.add_patch(hub)
    # Antenna mast
    ax.plot([0.70, 0.78], [0.65, 0.88], color=edgecolor, linewidth=1.2)
    antenna_tip = Polygon(
        [[0.75, 0.90], [0.81, 0.90], [0.78, 0.95]],
        closed=True,
        facecolor="#FF8F00",
        edgecolor=edgecolor,
        linewidth=0.8,
    )
    ax.add_patch(antenna_tip)


def _figure_to_rgba(fig, size_px: int) -> np.ndarray:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    rgba = np.asarray(buf).copy()
    plt.close(fig)
    h, w = rgba.shape[:2]
    if h != size_px or w != size_px:
        from PIL import Image

        img = Image.fromarray(rgba).resize((size_px, size_px), Image.Resampling.LANCZOS)
        rgba = np.asarray(img)
    return rgba


class ContinuousLaserTagVisualizer:
    """Handles visualization for the Continuous LaserTag POMDP.

    Creates animated GIF visualizations showing robot and opponent
    movement as rendered icons, rectangular walls, laser rays, belief
    particles, and tag indicators. The robot is shown as a red humanoid
    and the opponent as a blue wheeled rover.

    Attributes:
        grid_size: Arena dimensions ``(width, height)`` as ndarray.
        walls: Shape ``(M, 4)`` wall AABB array.
        robot_radius: Robot body radius.
        opponent_radius: Opponent body radius.
        dangerous_areas: Dangerous area centers as ``(x, y)`` tuples.
        dangerous_area_radius: Radius of dangerous areas.
    """

    def __init__(
        self,
        grid_size: np.ndarray,
        walls: np.ndarray,
        robot_radius: float,
        opponent_radius: float,
        dangerous_areas: List[Tuple[float, float]],
        dangerous_area_radius: float,
    ):
        self.grid_size = np.asarray(grid_size, dtype=float)
        self.walls = np.asarray(walls, dtype=float).reshape(-1, 4)
        self.robot_radius = robot_radius
        self.opponent_radius = opponent_radius
        self.dangerous_areas = dangerous_areas
        self.dangerous_area_radius = dangerous_area_radius
        self._robot_img = _render_robot_image()
        self._opponent_img = _render_opponent_image()

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Create animated GIF visualization of a Continuous LaserTag episode.

        Args:
            history: Episode step data list.
            cache_path: Path to save the GIF.

        Raises:
            ValueError: If history is empty.
            TypeError: If inputs have wrong type.
        """
        self._validate_inputs(history, cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        robot_path, opponent_path, actions, beliefs = self._extract_history(history)
        fig, ax = self._setup_figure()
        self._draw_static(ax)
        elements = self._create_animated_elements(ax)

        def init():
            return self._init_frame(elements)

        def update(frame):
            return self._update_frame(frame, robot_path, opponent_path, actions, beliefs, elements)

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(robot_path),
            init_func=init,
            blit=False,
            repeat=False,
            interval=1000,
        )
        plt.tight_layout()
        anim.save(cache_path, writer="pillow", fps=1)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_inputs(self, history: List[StepData], cache_path: Path) -> None:
        if not isinstance(history, list):
            raise TypeError("history must be a list")
        if not history:
            raise ValueError("Cannot visualize empty history")
        for step in history:
            if not isinstance(step, StepData):
                raise TypeError("history must contain StepData objects")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_history(self, history: List[StepData]) -> Tuple:
        robot_path, opponent_path, actions, beliefs = [], [], [], []
        for step in history:
            if not isinstance(step.state, np.ndarray) or len(step.state) != 5:
                raise ValueError("Expected numpy state with shape (5,)")
            robot_path.append(step.state[:2].copy())
            opponent_path.append(step.state[2:4].copy())
            actions.append(step.action)
            beliefs.append(getattr(step, "belief", None))
        return robot_path, opponent_path, actions, beliefs

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------

    def _setup_figure(self) -> Tuple[Figure, Axes]:
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=(14, 8))  # type: ignore[assignment]
        w, h = self.grid_size
        margin = 0.5
        ax.set_xlim(-margin, w + margin)
        ax.set_ylim(-margin, h + margin)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Continuous LaserTag POMDP Episode")
        ax.grid(True, alpha=0.3)
        return fig, ax

    def _draw_static(self, ax: Axes) -> None:
        # Walls as filled rectangles
        for i in range(self.walls.shape[0]):
            cx, cy, hx, hy = self.walls[i]
            rect = plt.Rectangle(  # type: ignore
                (cx - hx, cy - hy),
                2 * hx,
                2 * hy,
                facecolor="black",
                edgecolor="black",
                alpha=0.7,
                label="Wall" if i == 0 else "",
            )
            ax.add_patch(rect)

        # Dangerous areas
        for i, (dx, dy) in enumerate(self.dangerous_areas):
            circle = plt.Circle(  # type: ignore[attr-defined]
                (dx, dy),
                self.dangerous_area_radius,
                facecolor="red",
                edgecolor="none",
                alpha=0.3,
                label="Dangerous Area" if i == 0 else "",
            )
            ax.add_patch(circle)

    # ------------------------------------------------------------------
    # Animated elements
    # ------------------------------------------------------------------

    def _create_animated_elements(self, ax: Axes) -> dict:
        robot_box = self._make_annotation_box(self._robot_img, ax, zoom=1.0)
        opponent_box = self._make_annotation_box(self._opponent_img, ax, zoom=1.0)
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
            visible=False,
        )

        action_arrow = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
        )

        opp_belief = ax.scatter([], [], c="lightblue", alpha=0.6, s=30)
        robot_belief = ax.scatter([], [], c="lightcoral", alpha=0.6, s=30)

        laser_lines = []
        for _ in range(8):
            (line,) = ax.plot([], [], "g-", alpha=0.4, linewidth=1)
            laser_lines.append(line)

        self._add_legend_with_proxies(ax)

        return {
            "robot": robot_box,
            "opponent": opponent_box,
            "step_text": step_text,
            "action_text": action_text,
            "tag_text": tag_text,
            "action_arrow": action_arrow,
            "opp_belief": opp_belief,
            "robot_belief": robot_belief,
            "laser_lines": laser_lines,
        }

    def _make_annotation_box(self, img: np.ndarray, ax: Axes, zoom: float = 0.35) -> AnnotationBbox:
        offset_img = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(
            offset_img,
            (0, 0),
            frameon=False,
            pad=0,
            zorder=5,
        )
        ax.add_artist(ab)
        return ab

    def _add_legend_with_proxies(self, ax: Axes) -> None:
        proxy_robot = Line2D(
            [], [], marker="s", color="w", markerfacecolor="#D32F2F", markersize=10, label="Robot"
        )
        proxy_opponent = Line2D(
            [],
            [],
            marker="s",
            color="w",
            markerfacecolor="#1976D2",
            markersize=10,
            label="Opponent",
        )
        proxy_belief = Line2D(
            [], [], marker="o", color="w", markerfacecolor="lightblue", markersize=8, label="Belief"
        )
        proxy_laser = Line2D([], [], color="g", alpha=0.4, linewidth=1, label="Laser")
        proxy_action = Line2D(
            [], [], color="red", linewidth=2, marker=">", markersize=8, label="Action"
        )
        ax.legend(
            handles=[proxy_robot, proxy_opponent, proxy_belief, proxy_laser, proxy_action],
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            framealpha=0.9,
        )

    def _init_frame(self, e: dict) -> list:
        e["robot"].xyann = (0, 0)
        e["robot"].set_visible(False)
        e["opponent"].xyann = (0, 0)
        e["opponent"].set_visible(False)
        e["step_text"].set_text("")
        e["action_text"].set_text("")
        e["tag_text"].set_visible(False)
        e["action_arrow"].set_position((0, 0))
        e["action_arrow"].xy = (0, 0)
        e["action_arrow"].set_visible(False)
        e["opp_belief"].set_offsets(np.empty((0, 2)))
        e["robot_belief"].set_offsets(np.empty((0, 2)))
        for line in e["laser_lines"]:
            line.set_data([], [])
        return self._all_artists(e)

    def _update_frame(self, frame, robot_path, opponent_path, actions, beliefs, e) -> list:
        rp = robot_path[frame]
        op = opponent_path[frame]

        e["robot"].xyann = (rp[0], rp[1])
        e["robot"].set_visible(True)
        e["opponent"].xyann = (op[0], op[1])
        e["opponent"].set_visible(True)

        e["step_text"].set_text(f"Step: {frame + 1}/{len(robot_path)}")

        self._update_action_info(frame, actions, rp, op, e)
        self._update_belief_scatter(frame, beliefs, e)
        self._update_laser_rays(rp, op, e)

        return self._all_artists(e)

    _STRING_ACTION_DIRS = {
        "up": np.array([0.0, 1.0]),
        "down": np.array([0.0, -1.0]),
        "right": np.array([1.0, 0.0]),
        "left": np.array([-1.0, 0.0]),
        "tag": np.array([0.0, 0.0]),
    }

    def _update_action_info(self, frame, actions, rp, op, e) -> None:
        if frame >= len(actions):
            return
        action = actions[frame]
        if isinstance(action, str):
            e["action_text"].set_text(f"Action: {action}")
            is_tag = action == "tag"
            direction = self._STRING_ACTION_DIRS.get(action, np.array([0.0, 0.0]))
        else:
            a = np.asarray(action, dtype=float).ravel()
            is_tag = a.size > 2 and a[2] > 0.5
            e["action_text"].set_text(
                f"Action: [{a[0]:.1f}, {a[1]:.1f}, {a[2]:.1f}]" if a.size > 2 else ""
            )
            direction = a[:2] if a.size >= 2 else np.array([0.0, 0.0])

        self._update_action_arrow(e["action_arrow"], rp, direction)

        if is_tag:
            dist = np.linalg.norm(rp - op)
            if dist <= self.robot_radius + self.opponent_radius + 0.5:
                e["tag_text"].set_text("TAGGED!")
                e["tag_text"].set_color("green")
            else:
                e["tag_text"].set_text("MISSED!")
                e["tag_text"].set_color("red")
            e["tag_text"].set_visible(True)
        else:
            e["tag_text"].set_visible(False)

    def _update_action_arrow(self, action_arrow, position, direction) -> None:
        mag = float(np.linalg.norm(direction))
        if mag > 1e-12:
            unit = direction / mag
            arrow_scale = 0.6
            action_arrow.set_position((position[0], position[1]))
            action_arrow.xy = (
                position[0] + unit[0] * arrow_scale,
                position[1] + unit[1] * arrow_scale,
            )
            action_arrow.set_visible(True)
        else:
            action_arrow.set_visible(False)

    def _update_belief_scatter(self, frame, beliefs, e) -> None:
        if frame < len(beliefs) and beliefs[frame] is not None:
            try:
                belief = beliefs[frame]
                if hasattr(belief, "to_unique_support_distribution"):
                    unique = belief.to_unique_support_distribution()
                    if len(unique.values) > 0:
                        opp_pts = []
                        rob_pts = []
                        for st in unique.values:
                            if isinstance(st, np.ndarray) and len(st) == 5:
                                opp_pts.append([st[2], st[3]])
                                rob_pts.append([st[0], st[1]])
                        if opp_pts:
                            e["opp_belief"].set_offsets(np.array(opp_pts))
                        if rob_pts:
                            e["robot_belief"].set_offsets(np.array(rob_pts))
                        return
            except Exception:
                pass
        e["opp_belief"].set_offsets(np.empty((0, 2)))
        e["robot_belief"].set_offsets(np.empty((0, 2)))

    def _update_laser_rays(self, rp, op, e) -> None:
        measurements = compute_laser_measurements(
            rp,
            op,
            self.opponent_radius,
            self.walls,
            self.grid_size,
        )
        for i, line in enumerate(e["laser_lines"]):
            d = measurements[i]
            end = rp + LASER_DIRECTIONS[i] * d
            line.set_data([rp[0], end[0]], [rp[1], end[1]])

    def _all_artists(self, e: dict) -> list:
        return [
            e["robot"],
            e["opponent"],
            e["step_text"],
            e["action_text"],
            e["tag_text"],
            e["action_arrow"],
            e["opp_belief"],
            e["robot_belief"],
        ] + e["laser_lines"]
