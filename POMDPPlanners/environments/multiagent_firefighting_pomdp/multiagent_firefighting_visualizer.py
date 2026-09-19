# SPDX-License-Identifier: MIT

"""Visualization for the multi-agent firefighting POMDP.

Renders an episode as an animated GIF with three panels, Pillow and numpy only,
in the style the other renderers here use. The three panels exist because a
firefighting run is unreadable without all three -- you cannot tell a planner
that inferred the wind from one that guessed:

* **Ground truth** -- the real grid: the five cell categories, the obstacles,
  the depot, each live robot with its health and tank badge, and the Chebyshev
  sensing footprint drawn around it so a reader can see what was and was not
  visible when the action was chosen. The true wind is printed here and
  labelled hidden, because it is the one thing on this panel the robots never
  get.
* **Belief: P(alight)** -- the per-cell probability that the cell is smoldering
  or burning, as the weighted mean over the belief's particles. This is the
  exact marginal of the particle belief rather than a summary of it. Drawing
  the particles themselves is the usual choice here, but one particle is a
  whole hundred-cell world plus a wind, and a hundred overlaid worlds are not a
  picture of anything; the marginal is the projection the task turns on.
* **Belief: wind** -- the total particle weight on each of the eight wind
  values, with the true one marked. This is the other half of the belief, and
  it is the half the environment is *about*: the fire map is partly observed
  every step, while the wind is never observed at all and can only be inferred
  from where the fire grew.

Every frame is drawn *before* its own action resolves, because
:class:`~POMDPPlanners.core.simulation.StepData` records the state a step was
taken *from*: each frame shows what the robots chose from, and the caption says
what they are about to do.

The output is deterministic by construction. Particles are read in the belief's
own order, there is no jitter anywhere, the palette is a fixed table rather than
a median cut over the frames, and nothing embeds a timestamp -- rendering the
same history twice produces byte-identical files, which is what the golden test
depends on.

Classes:
    MultiAgentFirefightingVisualizer: Renders firefighting episodes.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as PIL_VERSION

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.multiagent_firefighting_pomdp.multiagent_firefighting_world import (
    NUM_WIND_VALUES,
    FireCategory,
    FirefightingAction,
    WindDirection,
    WindStrength,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from POMDPPlanners.environments.multiagent_firefighting_pomdp.multiagent_firefighting_pomdp import (  # noqa: E501
        MultiAgentFirefightingPOMDP,
    )

#: Milliseconds per frame, and for the last frame, which is held so a reader
#: can see how the episode ended before the loop restarts.
FRAME_MS = 420
FINAL_FRAME_MS = 1600

#: Canvas geometry. Fixed, so every episode of every size renders to the same
#: shape and a contact sheet of several episodes lines up.
CANVAS_SIZE = (1120, 500)
PANEL_TOP = 74
PANEL_SIZE = 320
TRUTH_LEFT = 32
BELIEF_LEFT = 392
WIND_LEFT = 760
WIND_WIDTH = 328

COLOR_PAGE = (22, 24, 30)
COLOR_PANEL = (32, 35, 43)
COLOR_GRID_LINE = (48, 52, 62)
COLOR_TEXT = (226, 230, 238)
COLOR_TEXT_DIM = (140, 148, 162)
COLOR_TRUE_WIND = (232, 78, 62)
COLOR_ROBOT = (236, 240, 248)
COLOR_ROBOT_DEAD = (104, 108, 118)
COLOR_HEALTH = (92, 208, 132)
COLOR_TANK = (86, 160, 232)
COLOR_BAR_BACK = (58, 62, 72)
COLOR_FOOTPRINT = (150, 200, 240)
COLOR_DEPOT = (240, 214, 96)
COLOR_OBSTACLE = (104, 110, 122)
COLOR_WIND_BAR = (120, 176, 232)

#: Fill colour of each cell category, indexed by :class:`FireCategory`.
CATEGORY_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (46, 84, 52),  # UNBURNT: fuel, dark green
    (196, 134, 44),  # SMOLDERING: amber
    (214, 66, 34),  # BURNING: red
    (52, 46, 44),  # BURNT: spent, near black
    (48, 96, 156),  # WET: blue
)

#: Short labels for the key, in category order.
CATEGORY_LABELS: Tuple[str, ...] = ("Unburnt", "Smoldering", "Burning", "Burnt", "Wet")

#: How many steps the P(alight) ramp is quantized to. A discrete ramp rather
#: than a continuous one is deliberate: it keeps the GIF palette small and
#: exact, and it makes "about a third" and "about a half" distinguishable at a
#: glance, which a smooth gradient does not.
BELIEF_RAMP_STEPS = 24

#: Per-robot action labels, in the order :class:`FirefightingAction` declares.
ACTION_LABELS: Tuple[str, ...] = ("N", "E", "S", "W", "SUP")

#: Wind labels, indexed by ``direction * 2 + strength``.
WIND_LABELS: Tuple[str, ...] = (
    "N low",
    "N high",
    "E low",
    "E high",
    "S low",
    "S high",
    "W low",
    "W high",
)


def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Return a font of roughly ``size`` pixels.

    Pillow 10.1 and later can scale its bundled face, which keeps text
    identical on every machine carrying the same Pillow. Older Pillow ignores
    the argument and returns one fixed bitmap face, which would silently
    collapse the title, the headings and the tick labels into one size. A
    render that quietly looks wrong is worse than one that stops, so this
    raises rather than falling back.

    Args:
        size: Requested height in pixels.

    Returns:
        A font suitable for :meth:`PIL.ImageDraw.ImageDraw.text`.

    Raises:
        RuntimeError: If Pillow is too old to scale its default font, or was
            built without FreeType and can only return the bitmap face.
    """
    try:
        font = ImageFont.load_default(size=size)
    except TypeError as exc:  # pragma: no cover - only on Pillow < 10.1
        raise RuntimeError(
            "The multi-agent firefighting renderer needs Pillow >= 10.1 for "
            f"scalable default fonts; this environment has Pillow {PIL_VERSION}."
        ) from exc
    if not isinstance(font, ImageFont.FreeTypeFont):  # pragma: no cover
        raise RuntimeError(
            "The multi-agent firefighting renderer needs a FreeType-enabled Pillow; "
            f"this one ({PIL_VERSION}) returns only its fixed bitmap face."
        )
    return font


def belief_ramp(level: int) -> Tuple[int, int, int]:
    """Return the colour of one step of the P(alight) ramp.

    Runs from the panel's own background at zero through deep violet and
    magenta to a hot yellow at one, so that a cell the belief is sure is
    burning reads at a glance and a cell it is unsure about does not pretend to
    be either extreme.

    Args:
        level: Ramp step in ``[0, BELIEF_RAMP_STEPS]``.

    Returns:
        An RGB triple.
    """
    fraction = float(level) / float(BELIEF_RAMP_STEPS)
    stops = (
        (0.0, (26, 28, 38)),
        (0.35, (78, 40, 114)),
        (0.7, (198, 62, 106)),
        (1.0, (248, 214, 96)),
    )
    for index in range(len(stops) - 1):
        low_position, low_color = stops[index]
        high_position, high_color = stops[index + 1]
        if fraction <= high_position:
            span = high_position - low_position
            weight = 0.0 if span == 0.0 else (fraction - low_position) / span
            return tuple(  # type: ignore[return-value]
                int(round(low + (high - low) * weight)) for low, high in zip(low_color, high_color)
            )
    return stops[-1][1]


def _build_master_palette() -> Image.Image:
    """Return the fixed GIF palette every frame is quantized against.

    A fixed table rather than a median cut over the frames. Median cut weights
    a colour by how many pixels carry it, so the large flat panels would take
    nearly every slot and the robots, the footprints and the true-wind marker
    would quantize to mud -- and, worse for a golden test, the chosen palette
    would depend on the frames, so two episodes would not be comparable and a
    Pillow upgrade could silently change the bytes.

    Returns:
        A one-pixel ``P`` image carrying the palette.
    """
    entries: List[Tuple[int, int, int]] = [
        COLOR_PAGE,
        COLOR_PANEL,
        COLOR_GRID_LINE,
        COLOR_TEXT,
        COLOR_TEXT_DIM,
        COLOR_TRUE_WIND,
        COLOR_ROBOT,
        COLOR_ROBOT_DEAD,
        COLOR_HEALTH,
        COLOR_TANK,
        COLOR_BAR_BACK,
        COLOR_FOOTPRINT,
        COLOR_DEPOT,
        COLOR_OBSTACLE,
        COLOR_WIND_BAR,
    ]
    entries.extend(CATEGORY_COLORS)
    entries.extend(belief_ramp(level) for level in range(BELIEF_RAMP_STEPS + 1))
    # A grey ramp for anti-aliased text, then a coarse colour cube so that any
    # blend the drawing produces lands on something reasonable.
    entries.extend((value, value, value) for value in range(0, 256, 8))
    for red in (0, 64, 128, 192, 255):
        for green in (0, 64, 128, 192, 255):
            for blue in (0, 64, 128, 192, 255):
                entries.append((red, green, blue))

    flat: List[int] = []
    for colour in entries[:256]:
        flat.extend(colour)
    flat.extend([0] * (768 - len(flat)))
    master = Image.new("P", (1, 1))
    master.putpalette(flat)
    return master


class MultiAgentFirefightingVisualizer:
    """Renders a multi-agent firefighting episode as an animated GIF.

    Attributes:
        env: The environment the episode was run in, read for grid geometry,
            the obstacle set, the depot and the state layout.
        num_rows: Grid rows, copied from the environment.
        num_cols: Grid columns, copied from the environment.
    """

    def __init__(self, env: "MultiAgentFirefightingPOMDP"):
        """Initialize the visualizer.

        Args:
            env: The environment instance to visualize.
        """
        self.env = env
        self.num_rows = env.num_rows
        self.num_cols = env.num_cols
        self._cell_width = PANEL_SIZE / float(self.num_cols)
        self._cell_height = PANEL_SIZE / float(self.num_rows)
        self._title_font = get_font(22)
        self._heading_font = get_font(15)
        self._label_font = get_font(12)
        self._background: Optional[Image.Image] = None

    # -- public API ------------------------------------------------------

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Render ``history`` to ``cache_path``.

        Args:
            history: Episode history. One frame per recorded step, so the frame
                count equals the recorded episode length.
            cache_path: Destination path, which must end in ``.gif``.

        Raises:
            TypeError: If ``history`` is not a list of :class:`StepData`, or
                ``cache_path`` is not a :class:`Path`.
            ValueError: If ``history`` is empty or ``cache_path`` is not a GIF.
        """
        self._validate(history, cache_path)
        images = self.render_frames(history)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._save(images, cache_path)

    def render_frames(self, history: List[StepData]) -> List[Image.Image]:
        """Render one RGB frame per recorded step.

        Exposed so tests and contact sheets can inspect a single frame without
        decoding a GIF.

        Args:
            history: Episode history.

        Returns:
            One RGB image per step, all :data:`CANVAS_SIZE`.
        """
        background = self._background_image()
        return [
            self._render_frame(background, step, index, len(history))
            for index, step in enumerate(history)
        ]

    # -- validation ------------------------------------------------------

    @staticmethod
    def _validate(history: List[StepData], cache_path: Path) -> None:
        """Reject inputs the renderer cannot draw.

        Args:
            history: Episode history.
            cache_path: Destination path.

        Raises:
            TypeError: If the types are wrong.
            ValueError: If the history is empty or the path is not a GIF.
        """
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

    # -- belief projections ----------------------------------------------

    def _belief_weights(self, belief: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the belief's particles and normalized weights, or ``None``.

        Args:
            belief: The belief recorded on the step.

        Returns:
            ``(particles, weights)`` with the weights summing to one, or
            ``None`` when the belief carries nothing usable. Particles are
            read in the belief's own order, which is what keeps two renders of
            one history byte-identical.
        """
        if belief is None:
            return None
        particles = getattr(belief, "particles", None)
        if particles is None or len(particles) == 0:
            return None
        array = np.asarray(particles, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.env.state_size:
            return None
        weights = np.asarray(
            getattr(belief, "normalized_weights", np.full(len(array), 1.0 / len(array))),
            dtype=np.float64,
        )
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return None
        return array, weights / total

    def _alight_marginal(self, belief: Any) -> Optional[np.ndarray]:
        """Per-cell probability that the cell is alight under ``belief``.

        Args:
            belief: The belief recorded on the step.

        Returns:
            ``(num_rows, num_cols)`` probabilities, or ``None`` when the
            belief carries nothing usable, which the renderer draws blank.
        """
        resolved = self._belief_weights(belief)
        if resolved is None:
            return None
        particles, weights = resolved
        categories = np.rint(particles[:, self.env.fire_offset :]).astype(np.int64)
        alight = (categories == int(FireCategory.SMOLDERING)) | (
            categories == int(FireCategory.BURNING)
        )
        return (weights[:, None] * alight).sum(axis=0).reshape(self.num_rows, self.num_cols)

    def _wind_marginal(self, belief: Any) -> Optional[np.ndarray]:
        """Total particle weight on each of the eight wind values.

        Args:
            belief: The belief recorded on the step.

        Returns:
            ``(8,)`` weights indexed by ``direction * 2 + strength``, or
            ``None`` when the belief carries nothing usable.
        """
        resolved = self._belief_weights(belief)
        if resolved is None:
            return None
        particles, weights = resolved
        directions = np.rint(particles[:, self.env.wind_direction_index]).astype(np.int64)
        strengths = np.rint(particles[:, self.env.wind_strength_index]).astype(np.int64)
        valid = (
            (directions >= 0)
            & (directions < len(WindDirection))
            & (strengths >= 0)
            & (strengths < len(WindStrength))
        )
        marginal = np.zeros(NUM_WIND_VALUES, dtype=np.float64)
        np.add.at(
            marginal,
            (directions[valid] * len(WindStrength) + strengths[valid]),
            weights[valid],
        )
        return marginal

    # -- geometry --------------------------------------------------------

    def _cell_box(self, left: int, row: int, col: int) -> Tuple[float, float, float, float]:
        """Return the pixel box of one cell inside the panel at ``left``.

        Args:
            left: Panel's left edge.
            row: Cell row.
            col: Cell column.

        Returns:
            ``(x0, y0, x1, y1)``.
        """
        x0 = left + col * self._cell_width
        y0 = PANEL_TOP + row * self._cell_height
        return (x0, y0, x0 + self._cell_width, y0 + self._cell_height)

    # -- background ------------------------------------------------------

    def _background_image(self) -> Image.Image:
        """Return the page, panel frames, headings and key, drawn once.

        Everything that never changes is rasterised once per episode and
        copied, so the per-frame cost is only what actually moves.

        Returns:
            The background image.
        """
        if self._background is not None:
            return self._background
        canvas = Image.new("RGB", CANVAS_SIZE, COLOR_PAGE)
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (TRUTH_LEFT, 18),
            "Multi-Agent Firefighting POMDP",
            font=self._title_font,
            fill=COLOR_TEXT,
        )
        for left, width, heading in (
            (TRUTH_LEFT, PANEL_SIZE, "Ground truth (wind hidden from robots)"),
            (BELIEF_LEFT, PANEL_SIZE, "Belief: P(cell alight)"),
            (WIND_LEFT, WIND_WIDTH, "Belief: hidden wind"),
        ):
            draw.rectangle(
                (left - 2, PANEL_TOP - 2, left + width + 1, PANEL_TOP + PANEL_SIZE + 1),
                fill=COLOR_PANEL,
                outline=COLOR_GRID_LINE,
            )
            draw.text((left, PANEL_TOP - 22), heading, font=self._heading_font, fill=COLOR_TEXT_DIM)
        self._draw_key(draw)
        self._background = canvas
        return canvas

    def _draw_key(self, draw: ImageDraw.ImageDraw) -> None:
        """Draw the category key under the truth panel.

        Args:
            draw: The background's draw handle.
        """
        top = PANEL_TOP + PANEL_SIZE + 44
        x = TRUTH_LEFT
        for category, label in enumerate(CATEGORY_LABELS):
            draw.rectangle((x, top, x + 11, top + 11), fill=CATEGORY_COLORS[category])
            draw.text((x + 16, top - 1), label, font=self._label_font, fill=COLOR_TEXT_DIM)
            x += 26 + 7 * len(label)
        draw.rectangle((x, top, x + 11, top + 11), fill=COLOR_OBSTACLE)
        draw.text((x + 16, top - 1), "Obstacle", font=self._label_font, fill=COLOR_TEXT_DIM)
        x += 26 + 7 * len("Obstacle")
        draw.rectangle((x, top, x + 11, top + 11), outline=COLOR_DEPOT)
        draw.text((x + 16, top - 1), "Depot", font=self._label_font, fill=COLOR_TEXT_DIM)

    # -- frame -----------------------------------------------------------

    def _render_frame(
        self, background: Image.Image, step: StepData, index: int, total: int
    ) -> Image.Image:
        """Draw one frame.

        Args:
            background: The shared background.
            step: The recorded step this frame shows.
            index: Zero-based frame index.
            total: How many frames the episode has.

        Returns:
            An RGB image of :data:`CANVAS_SIZE`.
        """
        canvas = background.copy()
        draw = ImageDraw.Draw(canvas)
        state = np.asarray(step.state, dtype=np.float64)
        self._draw_truth(draw, state)
        self._draw_belief(draw, self._alight_marginal(step.belief))
        self._draw_wind(draw, self._wind_marginal(step.belief), self.env.wind(state))
        self._draw_status(draw, state, step, index, total)
        return canvas

    def _draw_truth(self, draw: ImageDraw.ImageDraw, state: np.ndarray) -> None:
        """Draw the true grid, the depot, the footprints and the robots.

        Args:
            draw: The frame's draw handle.
            state: The state this frame shows.
        """
        fire = self.env.fire_map(state)
        obstacles = self.env.obstacle_mask
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                box = self._cell_box(TRUTH_LEFT, row, col)
                colour = (
                    COLOR_OBSTACLE if obstacles[row, col] else CATEGORY_COLORS[int(fire[row, col])]
                )
                draw.rectangle(box, fill=colour, outline=COLOR_GRID_LINE)

        depot_row, depot_col = self.env.depot_cell
        depot_box = self._cell_box(TRUTH_LEFT, depot_row, depot_col)
        draw.rectangle(depot_box, outline=COLOR_DEPOT, width=2)
        draw.text(
            (depot_box[0] + 0.30 * self._cell_width, depot_box[1] + 0.16 * self._cell_height),
            "D",
            font=self._label_font,
            fill=COLOR_DEPOT,
        )

        robots = self.env.robots(state)
        radius = self.env.sensing_radius
        for robot in range(self.env.num_robots):
            row, col, _, health = (int(value) for value in robots[robot])
            if health > 0:
                top_left = self._cell_box(TRUTH_LEFT, max(0, row - radius), max(0, col - radius))
                bottom_right = self._cell_box(
                    TRUTH_LEFT,
                    min(self.num_rows - 1, row + radius),
                    min(self.num_cols - 1, col + radius),
                )
                draw.rectangle(
                    (top_left[0], top_left[1], bottom_right[2] - 1, bottom_right[3] - 1),
                    outline=COLOR_FOOTPRINT,
                )
            box = self._cell_box(TRUTH_LEFT, row, col)
            inset_x = 0.22 * self._cell_width
            inset_y = 0.22 * self._cell_height
            draw.ellipse(
                (box[0] + inset_x, box[1] + inset_y, box[2] - inset_x, box[3] - inset_y),
                fill=COLOR_ROBOT if health > 0 else COLOR_ROBOT_DEAD,
                outline=COLOR_PAGE,
            )
            draw.text(
                (box[0] + 0.33 * self._cell_width, box[1] + 0.18 * self._cell_height),
                str(robot),
                font=self._label_font,
                fill=COLOR_PAGE,
            )

    def _draw_belief(self, draw: ImageDraw.ImageDraw, marginal: Optional[np.ndarray]) -> None:
        """Draw the per-cell P(alight) panel.

        No robots and no footprints are drawn here, which keeps the panel
        unmistakably a belief rather than a second copy of the map.

        Args:
            draw: The frame's draw handle.
            marginal: Per-cell probabilities, or ``None`` for a blank panel.
        """
        if marginal is None:
            draw.text(
                (BELIEF_LEFT + 10, PANEL_TOP + PANEL_SIZE // 2 - 8),
                "no belief recorded",
                font=self._label_font,
                fill=COLOR_TEXT_DIM,
            )
            return
        levels = np.clip(np.rint(marginal * BELIEF_RAMP_STEPS).astype(int), 0, BELIEF_RAMP_STEPS)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                draw.rectangle(
                    self._cell_box(BELIEF_LEFT, row, col),
                    fill=belief_ramp(int(levels[row, col])),
                    outline=COLOR_GRID_LINE,
                )
        for position, label in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
            x = BELIEF_LEFT + position * (PANEL_SIZE - 40)
            draw.rectangle(
                (x, PANEL_TOP + PANEL_SIZE + 8, x + 40, PANEL_TOP + PANEL_SIZE + 18),
                fill=belief_ramp(int(position * BELIEF_RAMP_STEPS)),
            )
            draw.text(
                (x, PANEL_TOP + PANEL_SIZE + 20), label, font=self._label_font, fill=COLOR_TEXT_DIM
            )

    def _draw_wind(
        self,
        draw: ImageDraw.ImageDraw,
        marginal: Optional[np.ndarray],
        true_wind: Tuple[int, int],
    ) -> None:
        """Draw the particle histogram over the eight wind values.

        Args:
            draw: The frame's draw handle.
            marginal: Weight per wind value, or ``None`` for a blank panel.
            true_wind: ``(direction, strength)`` of the wind actually blowing,
                marked so a reader can see whether the belief found it.
        """
        true_index = true_wind[0] * len(WindStrength) + true_wind[1]
        bar_left = WIND_LEFT + 62
        bar_span = WIND_WIDTH - 86
        row_height = PANEL_SIZE / float(NUM_WIND_VALUES)
        for value in range(NUM_WIND_VALUES):
            top = PANEL_TOP + value * row_height + 0.22 * row_height
            bottom = PANEL_TOP + (value + 1) * row_height - 0.22 * row_height
            is_true = value == true_index
            draw.text(
                (WIND_LEFT + 8, top - 1),
                WIND_LABELS[value],
                font=self._label_font,
                fill=COLOR_TRUE_WIND if is_true else COLOR_TEXT_DIM,
            )
            draw.rectangle((bar_left, top, bar_left + bar_span, bottom), fill=COLOR_BAR_BACK)
            weight = 0.0 if marginal is None else float(marginal[value])
            width = max(0.0, min(1.0, weight)) * bar_span
            if width > 0.0:
                draw.rectangle(
                    (bar_left, top, bar_left + width, bottom),
                    fill=COLOR_TRUE_WIND if is_true else COLOR_WIND_BAR,
                )
            if is_true:
                draw.polygon(
                    [
                        (bar_left + bar_span + 6, (top + bottom) / 2),
                        (bar_left + bar_span + 16, top),
                        (bar_left + bar_span + 16, bottom),
                    ],
                    fill=COLOR_TRUE_WIND,
                )
        draw.text(
            (WIND_LEFT, PANEL_TOP + PANEL_SIZE + 8),
            "red = true wind, never observed",
            font=self._label_font,
            fill=COLOR_TEXT_DIM,
        )

    def _draw_status(
        self,
        draw: ImageDraw.ImageDraw,
        state: np.ndarray,
        step: StepData,
        index: int,
        total: int,
    ) -> None:
        """Draw the caption line and the per-robot health and tank badges.

        Args:
            draw: The frame's draw handle.
            state: The state this frame shows.
            step: The recorded step.
            index: Zero-based frame index.
            total: How many frames the episode has.
        """
        robots = self.env.robots(state)
        fire = self.env.fire_map(state)
        alight = int(np.count_nonzero(self.env.alight_mask(fire)))
        burnt = int(np.count_nonzero(fire == int(FireCategory.BURNT)))
        direction, strength = self.env.wind(state)
        wind_label = WIND_LABELS[direction * len(WindStrength) + strength]

        draw.text(
            (WIND_LEFT - 320, 24),
            f"step {self.env.step_count(state)}   frame {index + 1}/{total}   "
            f"alight {alight}   burnt {burnt}   wind {wind_label}",
            font=self._heading_font,
            fill=COLOR_TEXT_DIM,
        )

        top = PANEL_TOP + PANEL_SIZE + 10
        for robot in range(self.env.num_robots):
            _, _, tank, health = (int(value) for value in robots[robot])
            x = TRUTH_LEFT + robot * 170
            draw.text(
                (x, top),
                f"R{robot}",
                font=self._label_font,
                fill=COLOR_ROBOT if health > 0 else COLOR_ROBOT_DEAD,
            )
            self._draw_bar(draw, x + 24, top + 2, health, self.env.max_health, COLOR_HEALTH)
            self._draw_bar(draw, x + 92, top + 2, tank, self.env.max_tank, COLOR_TANK)
            draw.text((x + 24, top + 12), "hp", font=self._label_font, fill=COLOR_TEXT_DIM)
            draw.text((x + 92, top + 12), "tank", font=self._label_font, fill=COLOR_TEXT_DIM)

        draw.text(
            (TRUTH_LEFT, CANVAS_SIZE[1] - 26),
            self._caption(step),
            font=self._heading_font,
            fill=COLOR_TEXT,
        )

    @staticmethod
    def _draw_bar(
        draw: ImageDraw.ImageDraw,
        left: int,
        top: int,
        value: int,
        capacity: int,
        colour: Tuple[int, int, int],
    ) -> None:
        """Draw one small capacity bar.

        Args:
            draw: The frame's draw handle.
            left: Left edge.
            top: Top edge.
            value: Filled amount.
            capacity: Full amount. A zero capacity draws an empty bar.
            colour: Fill colour.
        """
        width = 58
        draw.rectangle((left, top, left + width, top + 7), fill=COLOR_BAR_BACK)
        if capacity > 0 and value > 0:
            filled = width * min(1.0, float(value) / float(capacity))
            draw.rectangle((left, top, left + filled, top + 7), fill=colour)

    def _caption(self, step: StepData) -> str:
        """Return the caption naming what the robots are about to do.

        Args:
            step: The recorded step.

        Returns:
            A one-line caption. The terminal bookkeeping step took no action
            and says so.
        """
        if step.action is None:
            return "Episode ended - final state"
        per_robot = self.env.decode_action(step.action)
        moves = "  ".join(
            f"R{robot}:{ACTION_LABELS[int(per_robot[robot])]}"
            for robot in range(self.env.num_robots)
        )
        reward = "" if step.reward is None else f"   reward {float(step.reward):+.2f}"
        return f"Joint action {int(step.action)}   {moves}{reward}"

    # -- output ----------------------------------------------------------

    def _save(self, images: Sequence[Image.Image], cache_path: Path) -> None:
        """Quantize every frame against the fixed palette and write the GIF.

        Args:
            images: The rendered frames.
            cache_path: Destination path.
        """
        master = _build_master_palette()
        indexed = [image.quantize(palette=master, dither=Image.Dither.NONE) for image in images]
        durations = [FRAME_MS] * (len(indexed) - 1) + [FINAL_FRAME_MS]
        indexed[0].save(
            cache_path,
            save_all=True,
            append_images=list(indexed[1:]),
            duration=durations,
            loop=0,
            optimize=True,
        )


def action_labels() -> Dict[int, str]:
    """Return the per-robot action labels the caption uses.

    Exists so a test can assert the renderer's labels and the environment's
    action enum have not drifted apart.

    Returns:
        A mapping from :class:`FirefightingAction` code to label.
    """
    return {int(member): ACTION_LABELS[int(member)] for member in FirefightingAction}
