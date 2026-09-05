# SPDX-License-Identifier: MIT

"""Maze POMDP visualization.

Renders one episode as an animated GIF: the T-shaped corridor, the agent's path,
the cue cell, and — the part a POMDP visualization exists for — the belief.

The frame is split in two, and the split is the point. The **left two thirds** are
the playable map and nothing else is ever drawn there, so the corridor never
competes with a status line for the same pixels. The **right third** carries the
legend, the step / action / observation readout and the goal-side belief bars.

Layout and colour follow the two visualizers this repository already has for
grid-world POMDPs:

* ``light_dark_visualizer`` — the unreachable field is filled a dark grey so the
  walkable region reads as the bright object in the picture, the grid is off, the
  legend sits outside the map, the agent and its path are red, the belief
  particles are yellow and sized by weight, and the goal is a green star. The cue
  cell borrows the beacons' white radial glow, because it plays the same role: it
  is the one place in the world that emits information.
* ``laser_tag_visualizer`` — the readout lines are rounded, filled text boxes, and
  the legend is built from explicit proxy handles rather than from label strings
  scattered over the drawing calls.

Two belief views are drawn, because the T-Maze hides two different things:

* **Position particles.** Drawn as unfilled rings, sized by weight. Position is in
  principle observable here, so this cloud is normally a single ring sitting
  exactly on the agent — a filled marker would simply disappear under it. When the
  ring is *not* on the agent the belief has gone wrong, and that is worth seeing.
* **Goal-side posterior.** The hidden variable the task actually turns on. It is
  shown twice: as a faint tint over each arm endpoint, so it is visible where the
  decision is made, and as two labelled bars in the side panel, so it is readable
  as a number. The true goal side is marked separately with a green star; that is
  observer information the planner does not have.

Nothing here draws from an RNG and nothing is iterated out of order, so two
renders of one history are byte-identical — which is what the golden-file test
compares. The whole render also runs inside an isolated matplotlib style context,
so a caller that has set a seaborn theme (``returns_plots`` sets ``whitegrid``
globally before this runs in a real simulation) neither changes the output nor
finds its own theme disturbed afterwards.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

from POMDPPlanners.core.simulation import StepData  # noqa: E402
from POMDPPlanners.environments.t_maze_pomdp.maze_pomdp import (  # noqa: E402
    GOAL_LEFT,
    OBSERVATION_LEFT_CUE,
    OBSERVATION_RIGHT_CUE,
    STATE_GOAL,
    STATE_X,
    STATE_Y,
)

# Palette. The field / corridor pair is the Light-Dark contrast; the agent red,
# belief yellow and goal green are that visualizer's marker colours.
_FIELD = "#3d4148"
_CORRIDOR = "#e9edf2"
_CORRIDOR_EDGE = "#b9c2cc"
_OUTLINE = "#11151a"
_AGENT = "#d62728"
_PATH = "#d62728"
_BELIEF = "#ffe11a"
_GOAL = "#2ca02c"
_CUE = "#1f77b4"
_START = "#8c8c8c"
_PANEL_TEXT = "#1b1f24"

# Observation alphabet as a reader sees it. The state's cue-delivery phase is a
# program value and is deliberately never shown: it is not something the agent
# observed.
_OBSERVATION_LABELS: Dict[Any, str] = {
    OBSERVATION_LEFT_CUE: "left cue",
    OBSERVATION_RIGHT_CUE: "right cue",
}
_NOTHING = "—"

# rcParams the render pins. Applied on top of matplotlib's own defaults inside a
# style context, so whatever theme the caller had set is both ignored here and
# restored afterwards.
_STYLE: Dict[str, Any] = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "font.size": 10.0,
    "axes.titlesize": 13.0,
    "text.color": _PANEL_TEXT,
}


class MazeVisualizer:
    """Render a generated Maze or a legacy T-shaped Maze episode.

    Attributes:
        environment: The maze whose geometry is drawn.
    """

    def __init__(self, environment: Any) -> None:
        """Initialize the visualizer.

        Args:
            environment: The environment supplying the corridor geometry.
        """
        self.environment = environment

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Write an animated GIF of ``history``.

        Args:
            history: The episode's step records, in order.
            cache_path: Destination path; must end in ``.gif``.

        Raises:
            ValueError: If ``history`` is empty or ``cache_path`` is not a ``.gif``.
        """
        if not history:
            raise ValueError("Cannot visualize an empty episode history.")
        if cache_path.suffix != ".gif":
            raise ValueError(f"cache_path must end in .gif, got {cache_path}.")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        states = [np.asarray(step.state, dtype=np.float64) for step in history]
        actions = [step.action for step in history]
        beliefs = [getattr(step, "belief", None) for step in history]
        # ``StepData.observation`` is what the world returned *after* the step's
        # action, so the reading the agent holds while standing on frame ``f`` is
        # the one recorded on frame ``f - 1``.
        observations = [None] + [getattr(step, "observation", None) for step in history[:-1]]

        # "default" resets every rcParam to matplotlib's own, discarding any
        # seaborn theme the caller installed; the context manager puts the
        # caller's rcParams back on exit.
        with plt.style.context(["default", _STYLE]):
            figure, map_axes, panel_axes, bar_axes = self._setup_figure()
            self._draw_corridor(map_axes)
            artists = self._create_animated_artists(map_axes, panel_axes, bar_axes)

            def draw_frame(frame: int):
                return self._draw_frame(
                    frame, states, actions, observations, beliefs, artists
                )

            anim = animation.FuncAnimation(
                figure, draw_frame, frames=len(states), blit=False, repeat=False
            )
            anim.save(cache_path, writer="pillow", fps=1)
            plt.close(figure)

    # Static scene
    def _setup_figure(self):
        """Build the figure: map on the left, legend / readout / bars on the right."""
        env = self.environment
        figure = plt.figure(figsize=(10.0, 6.0))
        grid = figure.add_gridspec(
            2, 2, width_ratios=[1.9, 1.0], height_ratios=[3.0, 1.0], wspace=0.06, hspace=0.18
        )
        map_axes = figure.add_subplot(grid[:, 0])
        panel_axes = figure.add_subplot(grid[0, 1])
        bar_axes = figure.add_subplot(grid[1, 1])

        cells = self._walkable_cells()
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        margin = 0.6
        map_axes.set_xlim(min(xs) - margin, max(xs) + margin)
        map_axes.set_ylim(min(ys) - margin, max(ys) + margin)
        map_axes.set_aspect("equal")
        map_axes.set_facecolor(_FIELD)
        map_axes.grid(False)
        map_axes.set_xticks([])
        map_axes.set_yticks([])
        for spine in map_axes.spines.values():
            spine.set_color(_OUTLINE)
            spine.set_linewidth(1.2)
        mode = "continuous" if not self._draws_cell_guides() else "discrete"
        map_axes.set_title(f"Maze ({mode}) — cue accuracy {env.cue_accuracy:.2f}", pad=10)

        panel_axes.set_axis_off()

        bar_axes.set_xlim(0.0, 1.0)
        bar_axes.set_ylim(-0.6, 1.6)
        bar_axes.set_yticks([0, 1])
        bar_axes.set_yticklabels(["right", "left"])
        bar_axes.set_xticks([0.0, 0.5, 1.0])
        bar_axes.tick_params(labelsize=8, length=2)
        bar_axes.grid(False)
        for side in ("top", "right"):
            bar_axes.spines[side].set_visible(False)
        bar_axes.set_title("belief over goal side", fontsize=9, pad=4)

        return figure, map_axes, panel_axes, bar_axes

    def _draw_corridor(self, axes) -> None:
        """Fill the walkable cells and trace one heavy outline around them."""
        env = self.environment
        guide_color = _CORRIDOR_EDGE if self._draws_cell_guides() else _CORRIDOR
        for x, y in sorted(self._walkable_cells()):
            axes.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    1.0,
                    1.0,
                    facecolor=_CORRIDOR,
                    edgecolor=guide_color,
                    linewidth=0.6,
                    zorder=0,
                )
            )
        for x0, y0, x1, y1 in self._boundary_segments():
            axes.plot(
                [x0, x1], [y0, y1], "-", color=_OUTLINE, linewidth=2.6, solid_capstyle="round",
                zorder=1,
            )
        self._draw_cue_glow(axes, env.cue_cell)

    def _boundary_segments(self) -> List[Tuple[float, float, float, float]]:
        """Cell edges with a walkable cell on one side and the field on the other.

        Returns:
            One ``(x0, y0, x1, y1)`` per wall edge, in a sorted, RNG-free order so
            the drawing is reproducible.
        """
        cells = self._walkable_cells()
        segments: List[Tuple[float, float, float, float]] = []
        for x, y in sorted(cells):
            if (x, y + 1) not in cells:
                segments.append((x - 0.5, y + 0.5, x + 0.5, y + 0.5))
            if (x, y - 1) not in cells:
                segments.append((x - 0.5, y - 0.5, x + 0.5, y - 0.5))
            if (x - 1, y) not in cells:
                segments.append((x - 0.5, y - 0.5, x - 0.5, y + 0.5))
            if (x + 1, y) not in cells:
                segments.append((x + 0.5, y - 0.5, x + 0.5, y + 0.5))
        return segments

    @staticmethod
    def _draw_cue_glow(axes, cell: Tuple[int, int]) -> None:
        """Halo the cue cell the way Light-Dark halos a beacon: it emits information."""
        for index in range(10):
            axes.add_patch(
                Circle(
                    (float(cell[0]), float(cell[1])),
                    0.30 + index * 0.035,
                    facecolor="white",
                    edgecolor="none",
                    alpha=float(0.30 * np.exp(-index * 0.35)),
                    zorder=1,
                )
            )

    def _create_animated_artists(self, map_axes, panel_axes, bar_axes) -> dict:
        """Create every artist a frame updates, plus the static legend."""
        env = self.environment
        left_goal, right_goal = self._goal_cells()

        left_tint = Rectangle(
            (left_goal[0] - 0.5, left_goal[1] - 0.5),
            1.0,
            1.0,
            facecolor=_BELIEF,
            edgecolor="none",
            alpha=0.0,
            zorder=2,
        )
        right_tint = Rectangle(
            (right_goal[0] - 0.5, right_goal[1] - 0.5),
            1.0,
            1.0,
            facecolor=_BELIEF,
            edgecolor="none",
            alpha=0.0,
            zorder=2,
        )
        map_axes.add_patch(left_tint)
        map_axes.add_patch(right_tint)

        # Fixed scene markers. Drawn once; the frame loop never touches them.
        map_axes.plot(
            [env.start_cell[0]],
            [env.start_cell[1]],
            "s",
            color=_START,
            markersize=11,
            markerfacecolor="none",
            markeredgewidth=2.0,
            zorder=3,
        )
        map_axes.plot(
            [env.cue_cell[0]],
            [env.cue_cell[1]],
            "^",
            color=_CUE,
            markersize=12,
            zorder=4,
        )
        for endpoint in (left_goal, right_goal):
            map_axes.plot(
                [endpoint[0]],
                [endpoint[1]],
                "s",
                color=_OUTLINE,
                markersize=15,
                markerfacecolor="none",
                markeredgewidth=1.6,
                zorder=3,
            )

        self._add_legend(panel_axes)

        # Both bars share the goal-side belief colour so the panel reads as one
        # quantity; the true goal keeps green to itself.
        left_bar, right_bar = bar_axes.barh(
            [1, 0],
            [0.0, 0.0],
            height=0.55,
            color=_BELIEF,
            edgecolor=_OUTLINE,
            linewidth=0.8,
            zorder=2,
        )
        left_value = bar_axes.text(0.0, 1.0, "", va="center", ha="left", fontsize=9)
        right_value = bar_axes.text(0.0, 0.0, "", va="center", ha="left", fontsize=9)

        return {
            "trail": map_axes.plot([], [], "-", color=_PATH, alpha=0.55, linewidth=2.5, zorder=5)[
                0
            ],
            "agent": map_axes.plot(
                [], [], "o", color=_AGENT, markersize=13, zorder=6
            )[0],
            # Unfilled so it survives sitting exactly on top of the agent.
            "belief_ring": map_axes.scatter(
                [],
                [],
                s=[],
                facecolors="none",
                edgecolors=_BELIEF,
                linewidths=2.2,
                zorder=7,
            ),
            "true_goal": map_axes.plot(
                [], [], "*", color=_GOAL, markersize=20, zorder=8
            )[0],
            "left_tint": left_tint,
            "right_tint": right_tint,
            "left_bar": left_bar,
            "right_bar": right_bar,
            "left_value": left_value,
            "right_value": right_value,
            "readout": panel_axes.text(
                0.0,
                0.42,
                "",
                transform=panel_axes.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                linespacing=1.7,
                bbox={
                    "boxstyle": "round,pad=0.5",
                    "facecolor": "#f2f4f7",
                    "edgecolor": "#c3cad3",
                },
            ),
        }

    @staticmethod
    def _add_legend(panel_axes) -> None:
        """Legend built from proxy handles, as ``laser_tag_visualizer`` does."""
        handles = [
            Line2D([], [], marker="o", color="none", markerfacecolor=_AGENT, markersize=10,
                   label="agent"),
            Line2D([], [], color=_PATH, alpha=0.55, linewidth=2.5, label="path so far"),
            Line2D([], [], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor=_BELIEF, markeredgewidth=2.2, markersize=12,
                   label="belief particles"),
            Line2D([], [], marker="*", color="none", markerfacecolor=_GOAL,
                   markeredgecolor=_GOAL, markersize=14, label="true goal (observer only)"),
            Line2D([], [], marker="^", color="none", markerfacecolor=_CUE,
                   markeredgecolor=_CUE, markersize=10, label="cue cell"),
            Line2D([], [], marker="s", color="none", markerfacecolor="none",
                   markeredgecolor=_START, markeredgewidth=2.0, markersize=10, label="start"),
            Line2D([], [], marker="s", color="none", markerfacecolor="none",
                   markeredgecolor=_OUTLINE, markeredgewidth=1.6, markersize=11,
                   label="candidate goals"),
            Line2D([], [], marker="s", color="none", markerfacecolor=_BELIEF, alpha=0.6,
                   markersize=11, label="goal-side belief"),
        ]
        panel_axes.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(-0.02, 1.06),
            frameon=True,
            framealpha=0.95,
            fontsize=9,
            handletextpad=0.8,
            borderpad=0.6,
        )

    # Per-frame
    def _draw_frame(
        self,
        frame: int,
        states: List[np.ndarray],
        actions: List[Any],
        observations: List[Any],
        beliefs: List[Any],
        artists: dict,
    ):
        state = states[frame]
        artists["agent"].set_data([state[STATE_X]], [state[STATE_Y]])
        artists["trail"].set_data(
            [row[STATE_X] for row in states[: frame + 1]],
            [row[STATE_Y] for row in states[: frame + 1]],
        )
        goal_cell = self._goal_cell(float(state[STATE_GOAL]))
        artists["true_goal"].set_data([goal_cell[0]], [goal_cell[1]])

        positions, weights, left_probability = self._read_belief(beliefs[frame])
        if positions.size:
            artists["belief_ring"].set_offsets(positions)
            artists["belief_ring"].set_sizes(120.0 + weights * 700.0)
        else:
            artists["belief_ring"].set_offsets(np.empty((0, 2)))
            artists["belief_ring"].set_sizes(np.empty(0))

        self._update_goal_belief(left_probability, artists)

        action = actions[frame]
        observation = observations[frame]
        artists["readout"].set_text(
            f"Step {frame + 1} / {len(states)}\n"
            f"Action: {action if action is not None else _NOTHING}\n"
            f"Last observation: {self._observation_label(observation)}\n"
            f"True goal: {'left' if float(state[STATE_GOAL]) == GOAL_LEFT else 'right'}"
        )
        return list(artists.values())

    @staticmethod
    def _observation_label(observation: Any) -> str:
        """Render an observation for a human, without leaking program identifiers."""
        if observation is None:
            return _NOTHING
        return _OBSERVATION_LABELS.get(observation, "no cue")

    @staticmethod
    def _update_goal_belief(left_probability: Optional[float], artists: dict) -> None:
        """Drive the arm tints and the two panel bars from the goal-side posterior."""
        if left_probability is None:
            artists["left_tint"].set_alpha(0.0)
            artists["right_tint"].set_alpha(0.0)
            artists["left_bar"].set_width(0.0)
            artists["right_bar"].set_width(0.0)
            artists["left_value"].set_position((0.02, 1.0))
            artists["left_value"].set_horizontalalignment("left")
            artists["left_value"].set_text("unavailable")
            artists["right_value"].set_text("")
            return

        left = float(left_probability)
        right = 1.0 - left
        artists["left_tint"].set_alpha(0.55 * left)
        artists["right_tint"].set_alpha(0.55 * right)
        artists["left_bar"].set_width(left)
        artists["right_bar"].set_width(right)
        MazeVisualizer._place_bar_label(artists["left_value"], left, 1.0)
        MazeVisualizer._place_bar_label(artists["right_value"], right, 0.0)

    def _walkable_cells(self):
        """Return corridor cells for either public API generation."""
        if hasattr(self.environment, "walkable_cells"):
            return self.environment.walkable_cells
        return self.environment.valid_cells

    def _goal_cells(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return candidate goals for generated and compatibility environments."""
        if hasattr(self.environment, "left_goal_cell"):
            return self.environment.left_goal_cell, self.environment.right_goal_cell
        return self.environment.left_endpoint, self.environment.right_endpoint

    def _goal_cell(self, goal_side: float) -> Tuple[int, int]:
        """Return the true goal without exposing it as planner information."""
        if hasattr(self.environment, "goal_cell"):
            return self.environment.goal_cell(goal_side)
        return self.environment.goal_endpoint(goal_side)

    def _draws_cell_guides(self) -> bool:
        """Draw guides only when positions are discrete cells."""
        return bool(getattr(self.environment, "draws_cell_guides", True))

    @staticmethod
    def _place_bar_label(label, value: float, row: float) -> None:
        """Print a bar's number beside it, or inside it when it would run off the axis.

        Args:
            label: The text artist for this bar.
            value: The probability the bar shows, in ``[0, 1]``.
            row: The bar's y position.
        """
        if value > 0.72:
            label.set_position((value - 0.03, row))
            label.set_horizontalalignment("right")
        else:
            label.set_position((value + 0.03, row))
            label.set_horizontalalignment("left")
        label.set_text(f"{value:.2f}")

    @staticmethod
    def _read_belief(belief: Any) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """Return belief positions, their weights, and the mass on the left goal.

        Reads the unique-support view when the belief offers one — that is the
        collapsed distribution the other visualizers draw — and falls back to raw
        particles and normalized weights otherwise.

        Args:
            belief: The belief recorded on this step, or ``None``.

        Returns:
            An ``(N, 2)`` array of positions, an ``(N,)`` array of weights, and the
            probability the goal is on the left, or ``None`` when no belief is
            available.
        """
        if belief is None:
            return np.empty((0, 2)), np.empty(0), None

        values: List[Any]
        probs: np.ndarray
        if hasattr(belief, "to_unique_support_distribution"):
            support = belief.to_unique_support_distribution()
            values = list(support.values)
            probs = np.asarray(support.probs, dtype=np.float64)
        else:
            values = list(getattr(belief, "particles", []))
            probs = np.asarray(getattr(belief, "normalized_weights", []), dtype=np.float64)
        if not values or probs.size != len(values):
            return np.empty((0, 2)), np.empty(0), None

        total = float(np.sum(probs))
        if total <= 0.0:
            return np.empty((0, 2)), np.empty(0), None
        probs = probs / total

        positions = np.array(
            [[float(np.asarray(v)[STATE_X]), float(np.asarray(v)[STATE_Y])] for v in values],
            dtype=np.float64,
        )
        left_mass = float(
            np.sum(
                [
                    probs[index]
                    for index, value in enumerate(values)
                    if float(np.asarray(value)[STATE_GOAL]) == GOAL_LEFT
                ]
            )
        )
        return positions, probs, left_mass


# The old class name remains an import alias for saved code. Public titles and new
# documentation use Maze.
TMazeVisualizer = MazeVisualizer

__all__ = ["MazeVisualizer", "TMazeVisualizer"]
