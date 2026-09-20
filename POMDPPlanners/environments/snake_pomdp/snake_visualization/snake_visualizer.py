# SPDX-License-Identifier: MIT

"""Visualization for the Snake POMDP.

Renders an episode as an animated GIF: a game-style board on the left and a
heads-up display on the right.

The board shows four things at once, which is the point of it -- separating what
the agent knows from what is true is the only way to tell a planner that reasoned
about the hidden food from one that walked into it:

* the **snake**, which the agent observes exactly, drawn head first with a
  gradient down the body and eyes pointing along the heading;
* the **belief**, as an amber glow whose brightness is the posterior probability
  that the food is in that cell. This is the belief itself -- a categorical
  distribution over one hidden cell -- and not a fit to it. It is drawn as a
  heatmap rather than as scattered particles because the particles here all sit
  on grid cells and would simply stack up on top of each other;
* the **vision window**, outlined in cyan, with everything outside it fogged.
  The fog is about the *food*: the agent sees its own body everywhere, so the
  body is drawn at full strength through the fog;
* the **food**, drawn only because a reviewer needs it, and labelled as hidden
  from the agent.

Every frame is drawn *before* its own action resolves.
:class:`~POMDPPlanners.core.simulation.StepData` records the state a step was
taken *from*, so the board shows the situation the agent chose from, and the HUD
reports the reading it chose on -- the previous step's observation -- rather than
the one its action is about to produce. Showing the next reading beside the
current board would credit the agent with information it did not have.

Classes:
    SnakeVisualizer: Renders Snake episodes.
"""

# pylint: disable=too-many-lines  # one renderer, its board and its HUD

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from PIL import Image
import matplotlib.pyplot as plt

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.snake_pomdp.snake_pomdp import (
    OBSERVATION_LIVE,
    SnakeQuadrant,
    SnakeTermination,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.snake_pomdp.snake_pomdp import SnakePOMDP


# The palette. Named rather than inlined because several of them are used in
# both the board and the HUD, and a board tile that drifts from its legend key
# is the one kind of error a reviewer cannot see.
_BACKGROUND = "#071426"
_PANEL = "#0d2240"
_PANEL_EDGE = "#1d4570"
_TILE_DARK = "#182c45"
_TILE_LIGHT = "#1e3651"
_WALL = "#2d4463"
_WALL_EDGE = "#44608a"
_INK = "#e8f4ff"
_MUTED = "#7fa8cc"
_CYAN = "#2fe3f2"
_AMBER = (1.0, 0.69, 0.18)
#: Opacity the most probable food cell is drawn at. Kept below 1 so the tiles,
#: the vision window and the snake stay legible through a wide posterior.
_BELIEF_MAX_ALPHA = 0.58
#: Palette size each GIF frame is quantized to. The board is flat colour apart
#: from the belief ramp, so 192 entries carry the belief ramp and keep the legend keys the exact colours
#: the board uses; at 96 the red food key drifted towards the amber belief key.
_GIF_COLORS = 192
#: Frames sampled to build that shared palette. Eight is enough to cover the
#: board, the HUD and the belief ramp, and caps the cost for a long episode.
_PALETTE_SAMPLE_FRAMES = 8
#: Rendering resolution. 72 gives a 1123x706 frame -- readable at the width the
#: docs embed it at, and a quarter the pixels of the matplotlib default, which
#: is what keeps a hundred-frame episode to a few hundred kilobytes.
_FIGURE_DPI = 72
_FOG = (0.02, 0.06, 0.13)
_FOG_ALPHA = 0.62
_APPLE = "#c8322c"
_APPLE_DARK = "#8d1f1b"
_DEATH = "#ff4d5a"
_STARVE = "#f5c451"
_STARVE_EDGE = "#8a6210"

#: Snake colours, head first: a bright green head fading to cyan at the tail.
_HEAD_COLOR = (0.35, 0.92, 0.33)
_TAIL_COLOR = (0.16, 0.85, 0.93)

#: Quadrant arrow directions in axis coordinates, ``(dx, dy)``. The board is
#: drawn with row 0 at the top, so "north" points up on the screen and down in
#: row numbers; these are screen directions.
_QUADRANT_ARROWS: Dict[int, Tuple[float, float]] = {
    int(SnakeQuadrant.NORTH_EAST): (0.7, 0.7),
    int(SnakeQuadrant.NORTH_WEST): (-0.7, 0.7),
    int(SnakeQuadrant.SOUTH_EAST): (0.7, -0.7),
    int(SnakeQuadrant.SOUTH_WEST): (-0.7, -0.7),
}

_QUADRANT_NAMES: Dict[int, str] = {
    int(SnakeQuadrant.NORTH_EAST): "NE",
    int(SnakeQuadrant.NORTH_WEST): "NW",
    int(SnakeQuadrant.SOUTH_EAST): "SE",
    int(SnakeQuadrant.SOUTH_WEST): "SW",
}

_TERMINATION_TEXT: Dict[int, str] = {
    int(SnakeTermination.RUNNING): "",
    int(SnakeTermination.WALL): "DEATH - hit the wall",
    int(SnakeTermination.SELF): "DEATH - hit itself",
    int(SnakeTermination.STARVATION): "STARVATION - no food in time",
    int(SnakeTermination.WIN): "WIN - target length reached",
}


class SnakeVisualizer:
    """Renders a Snake episode as an animated GIF.

    Attributes:
        env: The environment the episode was run in, used for grid geometry and
            for the vision window the board outlines.
    """

    def __init__(self, env: "SnakePOMDP"):
        """Initialize the visualizer.

        Args:
            env: The Snake environment instance to visualize.
        """
        self.env = env
        self.grid_size = env.grid_size

    # -- public API -----------------------------------------------------

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Render ``history`` to ``cache_path``.

        Args:
            history: Episode history. One frame is drawn per recorded step, so
                the frame count equals the recorded episode length.
            cache_path: Destination path, which must end in ``.gif``.

        Raises:
            TypeError: If ``history`` is not a list of :class:`StepData`, or
                ``cache_path`` is not a :class:`Path`.
            ValueError: If ``history`` is empty or ``cache_path`` is not a GIF.
        """
        self._validate(history, cache_path)
        frames = self._build_frames(history)
        figure, artists = self._setup_figure()
        self._save(figure, artists, frames, cache_path)

    # -- frame extraction -----------------------------------------------

    def _validate(self, history: List[StepData], cache_path: Path) -> None:
        if not isinstance(history, list):
            raise TypeError("history must be a list object")
        if not history:
            raise ValueError("Cannot visualize empty history")
        for step in history:
            if not isinstance(step, StepData):
                raise TypeError("history must be a list of StepData objects")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

    def _belief_grid(self, belief: Any) -> np.ndarray:
        """Posterior over the food cell as a ``(grid, grid)`` array.

        Uses the exact marginal when the belief can supply one, and otherwise
        the weighted particle histogram, so an episode recorded with a generic
        particle filter still renders. Both are deterministic functions of the
        belief, which the golden-file hash depends on.

        Args:
            belief: The belief recorded on the step.

        Returns:
            ``(grid_size, grid_size)`` probabilities, all zero when the belief
            carries nothing usable.
        """
        blank = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        if belief is None:
            return blank
        if hasattr(belief, "marginal"):
            return np.asarray(belief.marginal(self.env), dtype=np.float64)
        particles = getattr(belief, "particles", None)
        if particles is None or len(particles) == 0:
            return blank
        weights = np.asarray(
            getattr(belief, "normalized_weights", np.full(len(particles), 1.0 / len(particles))),
            dtype=np.float64,
        )
        for particle, weight in zip(particles, weights):
            food = self.env.food(particle)
            if food is not None:
                blank[food[0], food[1]] += float(weight)
        total = float(blank.sum())
        return blank / total if total > 0.0 else blank

    def _build_frames(self, history: List[StepData]) -> List[Dict[str, Any]]:
        """Turn the history into one self-contained drawing record per step."""
        frames: List[Dict[str, Any]] = []
        for index, step in enumerate(history):
            state = step.state
            # The reading the agent had when it chose this step's action is the
            # *previous* step's observation. The step's own observation is the
            # consequence of the action it has not taken yet on this frame.
            previous = history[index - 1].observation if index > 0 else None
            seen, scent = self._decode(previous)
            frames.append(
                {
                    "index": index,
                    "total": len(history),
                    "body": self.env.body(state),
                    "food": self.env.food(state),
                    "belief": self._belief_grid(step.belief),
                    "length": self.env.snake_length(state),
                    "steps_since_food": self.env.steps_since_food(state),
                    "termination": int(self.env.termination(state)),
                    "seen": seen,
                    "scent": scent,
                    "action": step.action,
                    "reward": step.reward,
                }
            )
        return frames

    def _decode(self, observation: Any) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """Return ``(seen, scent)`` of a recorded reading, or ``(None, None)``."""
        if observation is None:
            return None, None
        flat = tuple(int(value) for value in observation)
        if not flat or flat[0] != OBSERVATION_LIVE:
            return None, None
        _, seen, scent = self.env.decode_observation(flat)
        return seen, scent

    # -- figure construction --------------------------------------------

    def _setup_figure(self) -> Tuple[Figure, Dict[str, Any]]:
        """Build the board, the HUD and every artist the animation mutates."""
        figure = plt.figure(figsize=(15.6, 9.8), dpi=_FIGURE_DPI, facecolor=_BACKGROUND)
        artists: Dict[str, Any] = {}
        board = figure.add_axes((0.035, 0.05, 0.58, 0.88))
        self._style_board(board)
        self._draw_walls(board)
        self._draw_tiles(board)
        artists["board"] = board
        artists.update(self._board_artists(board))
        artists.update(self._hud_artists(figure))
        return figure, artists

    def _style_board(self, board: Any) -> None:
        """Set the board's limits, ticks and frame."""
        size = self.grid_size
        board.set_facecolor(_BACKGROUND)
        board.set_xlim(-1.6, size + 0.6)
        board.set_ylim(size + 0.6, -1.6)
        board.set_aspect("equal")
        board.set_xticks(range(size))
        board.set_yticks(range(size))
        board.tick_params(length=0, labelsize=8.5, colors=_MUTED, pad=4)
        board.xaxis.set_ticks_position("top")
        for spine in board.spines.values():
            spine.set_visible(False)

    def _draw_walls(self, board: Any) -> None:
        """Draw the wall ring that sits just outside the playable grid.

        The spec places the walls outside the ``grid_size`` playable cells
        rather than taking a ring of cells away from them, so the ring is drawn
        around the board instead of inside it. A head that steps into one of
        these tiles has left the grid.
        """
        size = self.grid_size
        for row in range(-1, size + 1):
            for col in range(-1, size + 1):
                if 0 <= row < size and 0 <= col < size:
                    continue
                board.add_patch(
                    FancyBboxPatch(
                        (col - 0.45, row - 0.45),
                        0.9,
                        0.9,
                        boxstyle="round,pad=0,rounding_size=0.13",
                        facecolor=_WALL,
                        edgecolor=_WALL_EDGE,
                        linewidth=1.0,
                        zorder=1,
                    )
                )

    def _draw_tiles(self, board: Any) -> None:
        """Draw the checkered playable tiles."""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                board.add_patch(
                    FancyBboxPatch(
                        (col - 0.46, row - 0.46),
                        0.92,
                        0.92,
                        boxstyle="round,pad=0,rounding_size=0.1",
                        facecolor=_TILE_LIGHT if (row + col) % 2 else _TILE_DARK,
                        edgecolor="#223d5c",
                        linewidth=0.8,
                        zorder=1,
                    )
                )

    def _board_artists(self, board: Any) -> Dict[str, Any]:
        """Create the mutable board artists: fog, belief, window, food, snake."""
        size = self.grid_size
        extent = (-0.5, size - 0.5, size - 0.5, -0.5)
        artists: Dict[str, Any] = {}
        artists["fog"] = board.imshow(
            np.zeros((size, size, 4)), extent=extent, zorder=2, interpolation="nearest"
        )
        artists["belief"] = board.imshow(
            np.zeros((size, size, 4)), extent=extent, zorder=3, interpolation="nearest"
        )
        artists["window"] = Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor=_CYAN, linewidth=2.4, zorder=4, visible=False
        )
        board.add_patch(artists["window"])

        artists["apple"] = Circle(
            (0, 0),
            0.31,
            facecolor=_APPLE,
            edgecolor=_APPLE_DARK,
            linewidth=1.4,
            zorder=5,
            visible=False,
        )
        board.add_patch(artists["apple"])
        artists["apple_stem"] = Rectangle(
            (0, 0), 0.05, 0.16, facecolor="#4d7a2a", edgecolor="none", zorder=6, visible=False
        )
        board.add_patch(artists["apple_stem"])

        artists["segments"] = []
        for _ in range(self.env.target_length):
            patch = FancyBboxPatch(
                (0, 0),
                0.78,
                0.78,
                boxstyle="round,pad=0,rounding_size=0.17",
                facecolor="#2fe36a",
                edgecolor="#0a3d1f",
                linewidth=1.1,
                zorder=7,
                visible=False,
            )
            board.add_patch(patch)
            artists["segments"].append(patch)

        artists["eyes"] = []
        artists["pupils"] = []
        for _ in range(2):
            eye = Circle(
                (0, 0), 0.098, facecolor="white", edgecolor="none", zorder=9, visible=False
            )
            pupil = Circle(
                (0, 0), 0.05, facecolor="#0b2218", edgecolor="none", zorder=10, visible=False
            )
            board.add_patch(eye)
            board.add_patch(pupil)
            artists["eyes"].append(eye)
            artists["pupils"].append(pupil)

        artists["death"] = [
            board.plot([], [], color=_DEATH, linewidth=4.0, zorder=11, solid_capstyle="round")[0]
            for _ in range(2)
        ]
        artists["starve"] = [
            Polygon(
                np.zeros((3, 2)),
                closed=True,
                facecolor=_STARVE,
                edgecolor=_STARVE_EDGE,
                linewidth=1.2,
                zorder=11,
                visible=False,
            )
            for _ in range(2)
        ]
        for polygon in artists["starve"]:
            board.add_patch(polygon)

        artists["food_label"] = board.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            color="#f5c451",
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=12,
            visible=False,
            arrowprops={"arrowstyle": "->", "color": "#f5c451", "linewidth": 1.2},
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": _BACKGROUND,
                "edgecolor": "#f5c451",
                "linewidth": 0.8,
                "alpha": 0.88,
            },
        )
        return artists

    # -- HUD -------------------------------------------------------------

    def _panel(
        self, figure: Figure, left: float, bottom: float, width: float, height: float
    ) -> None:
        """Draw one rounded HUD card."""
        figure.add_artist(
            FancyBboxPatch(
                (left, bottom),
                width,
                height,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                transform=figure.transFigure,
                facecolor=_PANEL,
                edgecolor=_PANEL_EDGE,
                linewidth=1.2,
                zorder=0,
            )
        )

    # pylint: disable-next=too-many-locals,too-many-statements
    def _hud_artists(self, figure: Figure) -> Dict[str, Any]:
        """Create the heads-up display and every text artist the animation sets."""
        artists: Dict[str, Any] = {}
        figure.text(0.645, 0.945, "SNAKE", fontsize=27, fontweight="bold", color=_INK)
        figure.text(
            0.752, 0.948, "/  PARTIAL OBSERVABILITY", fontsize=15, fontweight="bold", color=_CYAN
        )
        figure.text(
            0.035,
            0.018,
            "Body known everywhere. Food remains hidden.",
            fontsize=11,
            color=_MUTED,
        )

        self._panel(figure, 0.640, 0.740, 0.335, 0.165)
        figure.text(0.652, 0.878, "AGENT STATE", fontsize=12, fontweight="bold", color=_CYAN)
        for label, key, left in (
            ("SCORE", "score", 0.652),
            ("LENGTH", "length", 0.762),
            ("STEP", "step", 0.888),
        ):
            figure.text(left, 0.838, label, fontsize=9.5, fontweight="bold", color=_MUTED)
            artists[key] = figure.text(left, 0.782, "", fontsize=21, fontweight="bold", color=_INK)
        figure.text(0.888, 0.768, "STEPS SINCE FOOD", fontsize=8, fontweight="bold", color=_MUTED)
        artists["since_food"] = figure.text(
            0.888, 0.748, "", fontsize=12.5, fontweight="bold", color=_INK
        )
        artists["pips"] = []
        for index in range(self.env.target_length):
            pip = Circle(
                (0.766 + 0.0132 * index, 0.766),
                0.0046,
                transform=figure.transFigure,
                facecolor="#2b4a6d",
                edgecolor="none",
                zorder=1,
            )
            figure.add_artist(pip)
            artists["pips"].append(pip)

        self._panel(figure, 0.640, 0.545, 0.163, 0.175)
        figure.text(0.651, 0.688, "LAST SCENT", fontsize=12, fontweight="bold", color=_CYAN)
        artists["scent_name"] = figure.text(
            0.746, 0.688, "", fontsize=12, fontweight="bold", color=_INK
        )
        arrow_axes = figure.add_axes((0.660, 0.575, 0.12, 0.092))
        arrow_axes.set_xlim(-1, 1)
        arrow_axes.set_ylim(-1, 1)
        arrow_axes.axis("off")
        arrow_axes.set_facecolor("none")
        artists["arrow"] = arrow_axes.annotate(
            "",
            xy=(0, 0),
            xytext=(0, 0),
            arrowprops={
                "arrowstyle": "-|>,head_width=0.32,head_length=0.62",
                "color": "#f5c451",
                "linewidth": 3.0,
            },
            visible=False,
        )
        figure.text(
            0.651,
            0.556,
            f"Correct quadrant: {self.env.scent_accuracy:.2f}",
            fontsize=9.5,
            color=_MUTED,
        )

        self._panel(figure, 0.812, 0.545, 0.163, 0.175)
        figure.text(0.823, 0.688, "FOOD DETECTION", fontsize=12, fontweight="bold", color=_CYAN)
        figure.text(
            0.823,
            0.616,
            f"Inside window: {self.env.detection_probability:.2f}",
            fontsize=9.5,
            color=_INK,
        )
        artists["detection"] = figure.text(0.823, 0.585, "", fontsize=9.5, color=_MUTED)
        figure.text(
            0.823,
            0.556,
            f"Window: {2 * self.env.window_radius + 1} x {2 * self.env.window_radius + 1}",
            fontsize=9.5,
            color=_MUTED,
        )

        self._panel(figure, 0.640, 0.245, 0.335, 0.278)
        figure.text(0.652, 0.492, "LEGEND", fontsize=12, fontweight="bold", color=_CYAN)
        for offset, (color, edge, label) in enumerate(
            (
                (_TILE_DARK, "#223d5c", "Fog / unknown for the food"),
                ("none", _CYAN, "Vision window / food may be reported"),
                ("#d68a1e", "#8a5a10", "Belief / food probability"),
                ("#2fe36a", "#0a3d1f", "Body / always known"),
                (_APPLE, _APPLE_DARK, "Food / hidden from the agent"),
                (_WALL, _WALL_EDGE, "Wall / outside the playable grid"),
            )
        ):
            bottom = 0.442 - 0.038 * offset
            figure.add_artist(
                FancyBboxPatch(
                    (0.654, bottom),
                    0.022,
                    0.026,
                    boxstyle="round,pad=0,rounding_size=0.006",
                    transform=figure.transFigure,
                    facecolor=color,
                    edgecolor=edge,
                    linewidth=1.6,
                    zorder=1,
                )
            )
            figure.text(0.686, bottom + 0.006, label, fontsize=10, color=_INK)

        self._panel(figure, 0.640, 0.125, 0.335, 0.100)
        figure.text(0.652, 0.192, "EPISODE", fontsize=12, fontweight="bold", color=_CYAN)
        artists["outcome"] = figure.text(
            0.652, 0.158, "", fontsize=11.5, fontweight="bold", color=_INK
        )
        artists["caption"] = figure.text(0.652, 0.134, "", fontsize=10, color=_MUTED)
        return artists

    # -- per-frame drawing ------------------------------------------------

    def _set_belief(self, artists: Dict[str, Any], belief: np.ndarray) -> None:
        """Paint the amber belief glow, scaled to the frame's own maximum.

        The scale is relative rather than absolute because the posterior's
        magnitude changes by two orders of magnitude over an episode: it starts
        at 1/141 spread over the whole grid and collapses onto one cell the
        moment the window reports a sighting. An absolute scale would render the
        prior as invisible and every later frame as one dot. The gamma and the
        ceiling on opacity keep the tiles readable underneath, which matters
        because a wide, flat posterior really does cover most of the board.
        """
        rgba = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float64)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = _AMBER
        peak = float(belief.max())
        if peak > 0.0:
            rgba[..., 3] = np.clip(belief / peak, 0.0, 1.0) ** 0.7 * _BELIEF_MAX_ALPHA
        artists["belief"].set_data(rgba)

    def _set_fog(self, artists: Dict[str, Any], head: Tuple[int, int]) -> None:
        """Fog every cell the vision window does not cover, and outline it."""
        rgba = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float64)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = _FOG
        rgba[..., 3] = _FOG_ALPHA
        for row, col in self.env.window_cells(head):
            rgba[row, col, 3] = 0.0
        artists["fog"].set_data(rgba)

        radius = self.env.window_radius
        low_row = max(head[0] - radius, 0)
        low_col = max(head[1] - radius, 0)
        high_row = min(head[0] + radius, self.grid_size - 1)
        high_col = min(head[1] + radius, self.grid_size - 1)
        artists["window"].set_bounds(
            low_col - 0.5, low_row - 0.5, high_col - low_col + 1.0, high_row - low_row + 1.0
        )
        artists["window"].set_visible(True)

    def _set_snake(self, artists: Dict[str, Any], body: Tuple[Tuple[int, int], ...]) -> None:
        """Place the body segments, their colour gradient and the head's eyes."""
        count = max(len(body), 1)
        for index, patch in enumerate(artists["segments"]):
            if index >= len(body):
                patch.set_visible(False)
                continue
            row, col = body[index]
            fraction = index / max(count - 1, 1)
            colour = tuple(
                head * (1.0 - fraction) + tail * fraction
                for head, tail in zip(_HEAD_COLOR, _TAIL_COLOR)
            )
            patch.set_bounds(col - 0.39, row - 0.39, 0.78, 0.78)
            patch.set_facecolor(colour)
            patch.set_visible(True)

        head_row, head_col = body[0]
        if len(body) >= 2:
            step = (head_row - body[1][0], head_col - body[1][1])
        else:
            step = (0, 1)
        # Eyes sit forward of the head's centre, side by side across the
        # heading, so the snake visibly faces where it is about to move.
        forward = (float(step[1]) * 0.15, float(step[0]) * 0.15)
        across = (-float(step[0]) * 0.16, float(step[1]) * 0.16)
        for sign, eye, pupil in zip((1.0, -1.0), artists["eyes"], artists["pupils"]):
            centre = (
                head_col + forward[0] + sign * across[0],
                head_row + forward[1] + sign * across[1],
            )
            eye.set_center(centre)
            eye.set_visible(True)
            pupil.set_center((centre[0] + forward[0] * 0.6, centre[1] + forward[1] * 0.6))
            pupil.set_visible(True)

    def _set_food(self, artists: Dict[str, Any], food: Optional[Tuple[int, int]]) -> None:
        """Place the apple, or hide it when the episode ended with no food."""
        visible = food is not None and self.env.in_grid(food)
        artists["apple"].set_visible(visible)
        artists["apple_stem"].set_visible(visible)
        artists["food_label"].set_visible(visible)
        if not visible or food is None:
            return
        row, col = food
        artists["apple"].set_center((col, row + 0.03))
        artists["apple_stem"].set_bounds(col - 0.025, row - 0.34, 0.05, 0.16)
        # The label is pushed towards whichever side of the board has room, so
        # it does not sit on top of the snake or run off the figure when the
        # food happens to be in a corner.
        towards_left = col > self.grid_size / 2
        artists["food_label"].set_position((col - 1.6 if towards_left else col + 1.6, row - 1.5))
        artists["food_label"].set_horizontalalignment("right" if towards_left else "left")
        artists["food_label"].xy = (col, row - 0.35)
        artists["food_label"].set_text("HIDDEN FOOD\nviewer only, not observed")

    def _set_markers(self, artists: Dict[str, Any], frame: Dict[str, Any]) -> None:
        """Draw the death cross or the starvation hourglass over the head."""
        termination = frame["termination"]
        row, col = frame["body"][0]
        died = termination in (int(SnakeTermination.WALL), int(SnakeTermination.SELF))
        starved = termination == int(SnakeTermination.STARVATION)
        for line, (dx, dy) in zip(artists["death"], ((0.3, 0.3), (0.3, -0.3))):
            if died:
                line.set_data([col - dx, col + dx], [row - dy, row + dy])
            else:
                line.set_data([], [])
        for polygon, upward in zip(artists["starve"], (True, False)):
            polygon.set_visible(starved)
            if not starved:
                continue
            # Both triangles meet at the head's own row: that shared apex is the
            # hourglass waist, so the tip never depends on `upward`. Only the
            # direction the bulb opens in does. This used to read
            # `row if upward else row`, a dead conditional with the same value
            # on both branches; the geometry below is unchanged.
            tip = row
            sign = -1.0 if upward else 1.0
            polygon.set_xy(
                np.array(
                    [
                        [col - 0.26, tip + sign * 0.32],
                        [col + 0.26, tip + sign * 0.32],
                        [col, tip],
                    ]
                )
            )

    def _set_hud(self, artists: Dict[str, Any], frame: Dict[str, Any]) -> None:
        """Fill in the heads-up display for one frame."""
        target = self.env.target_length
        score = max(frame["length"] - 3, 0)
        artists["score"].set_text(f"{score:02d}")
        artists["length"].set_text(f"{frame['length']} / {target}")
        artists["step"].set_text(f"{frame['index'] + 1:03d}")
        artists["since_food"].set_text(f"{frame['steps_since_food']} / {self.env.starvation_limit}")
        for index, pip in enumerate(artists["pips"]):
            pip.set_facecolor("#39e07a" if index < frame["length"] else "#2b4a6d")

        scent = frame["scent"]
        if scent is None:
            # No reading yet on the opening frame; on any later one the previous
            # step produced the terminal reading, which carries no scent.
            artists["scent_name"].set_text("none yet" if frame["index"] == 0 else "episode over")
            artists["arrow"].set_visible(False)
        else:
            artists["scent_name"].set_text(_QUADRANT_NAMES[int(scent)])
            dx, dy = _QUADRANT_ARROWS[int(scent)]
            artists["arrow"].set_position((-dx, -dy))
            artists["arrow"].xy = (dx, dy)
            artists["arrow"].set_visible(True)

        if frame["seen"] is None:
            artists["detection"].set_text("This frame: not observed")
        else:
            artists["detection"].set_text(
                f"This frame: seen at {frame['seen'][0]}, {frame['seen'][1]}"
            )

        outcome = _TERMINATION_TEXT[frame["termination"]]
        artists["outcome"].set_text(outcome if outcome else "Running")
        artists["outcome"].set_color(
            _DEATH
            if frame["termination"] in (int(SnakeTermination.WALL), int(SnakeTermination.SELF))
            else "#f5c451"
            if frame["termination"] == int(SnakeTermination.STARVATION)
            else "#39e07a"
            if frame["termination"] == int(SnakeTermination.WIN)
            else _INK
        )
        if frame["action"] is None:
            artists["caption"].set_text(
                f"Step {frame['index'] + 1} of {frame['total']}  -  final recorded state"
            )
        else:
            reward = "-" if frame["reward"] is None else f"{float(frame['reward']):+.0f}"
            artists["caption"].set_text(
                f"Step {frame['index'] + 1} of {frame['total']}  -  about to act "
                f"({int(frame['action'])}), reward {reward}"
            )

    def _draw_frame(self, artists: Dict[str, Any], frame: Dict[str, Any]) -> None:
        """Set every mutable artist for one frame."""
        self._set_fog(artists, frame["body"][0])
        self._set_belief(artists, frame["belief"])
        self._set_food(artists, frame["food"])
        self._set_snake(artists, frame["body"])
        self._set_markers(artists, frame)
        self._set_hud(artists, frame)

    def _save(
        self,
        figure: Figure,
        artists: Dict[str, Any],
        frames: List[Dict[str, Any]],
        cache_path: Path,
    ) -> None:
        """Render every frame and write the GIF."""
        rendered = []
        canvas = FigureCanvasAgg(figure)
        try:
            for frame in frames:
                self._draw_frame(artists, frame)
                canvas.draw()
                rendered.append(
                    Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert("RGB")
                )
            images = _quantize(rendered)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(
                cache_path,
                save_all=True,
                append_images=images[1:],
                duration=[260] * (len(images) - 1) + [2200],
                loop=0,
                # Disposal 1 leaves the previous frame in place, so the writer
                # can store only the pixels that changed. Most of this board is
                # static from step to step, and with disposal 2 every frame was
                # written in full.
                disposal=1,
                optimize=True,
            )
        finally:
            plt.close(figure)


def _quantize(rendered: List[Image.Image]) -> List[Image.Image]:
    """Map every rendered frame onto one shared palette.

    Two reasons, and the second is the one that matters. A truecolour frame of
    this board is about 120 kB, so a long episode would write a GIF an order of
    magnitude larger than any other in this repository. And a palette chosen per
    frame is chosen from *that frame's* colours, so the legend swatches shift
    hue from frame to frame -- on the final frame, where the belief has gone and
    almost no amber is left, the "belief" and "food" keys came out washed out
    and no longer matched the board they explain.

    The shared palette is built from a fixed, evenly spaced sample of the
    frames rather than from all of them, so its cost does not grow with the
    episode. Median-cut quantization is deterministic for identical input
    pixels, so the golden hash is unaffected.

    Args:
        rendered: The RGB frames, in order.

    Returns:
        The same frames in palette mode, all sharing one palette.
    """
    step = max(len(rendered) // _PALETTE_SAMPLE_FRAMES, 1)
    sample = rendered[::step][:_PALETTE_SAMPLE_FRAMES]
    width, height = rendered[0].size
    strip = Image.new("RGB", (width, height * len(sample)))
    for index, frame in enumerate(sample):
        strip.paste(frame, (0, index * height))
    palette = strip.quantize(colors=_GIF_COLORS, method=Image.Quantize.MEDIANCUT, dither=0)
    return [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in rendered]
