# SPDX-License-Identifier: MIT

"""Renders a Chicheck Invaders episode as an animated GIF that reads like the game.

Two panels side by side, on one dark arcade page.

* **The sky** -- the true world. Ship, chickens, the beam of a shot fired this
  step, and the two sensor footprints drawn over the grid: the camera cone as a
  pale wedge opening upward from the ship, the radar as a ring at its radius.
  The gun is hitscan, so a shot is a flash down the ship's whole column on the
  step that fired it and nothing on any other step -- there is no bolt to
  follow, because a shot never survives the step it was fired in. The chicken it
  killed is still drawn, because the frame shows the state the ship fired
  *from*; the next frame is where it is gone. A chicken's sprite
  says which mode it is in, because the mode is the hidden variable the whole
  observation model exists to reveal: a patrolling chicken is drawn upright with
  its walking direction marked, a diving one head-down. The two sensors are
  drawn as boundaries rather than as shaded cells -- with the default cone and
  radius almost every cell is covered, and a panel tinted almost everywhere says
  nothing. This panel is the one a reviewer checks the environment against.
* **What the ship believes** -- the same grid, with the belief's weighted
  marginal chance that a chicken occupies each cell. Particles are what this
  repository's low-dimensional visualizers draw, and the belief here *is* a
  cloud of particles, but a particle is a whole flock; overlaying a few hundred
  flocks produces a smear rather than a picture. The per-cell weighted marginal
  is the projection the task turns on -- where is there a chicken -- and it is
  still the belief rather than a fit to it. The sensor boundaries are repeated
  here so the two panels can be read against each other: cells well inside both
  sensors that the belief leaves dark are cells the sensors have ruled out.

Each frame shows the state a step was taken *from*, because that is what
:class:`~POMDPPlanners.core.simulation.StepData` records, and its caption names
the action about to be taken and the reward it earned. Rendering is Pillow and
numpy only, in the style the Light-Dark and occupancy-grid renderers established
here: everything that never changes is drawn once per episode and pasted, so the
per-frame cost is the handful of sprites that actually move.

The output is deterministic. Nothing is jittered for readability and nothing
iterates over an unordered collection, so re-rendering one history twice gives
two byte-identical files -- which is what the golden test hashes.

Classes:
    ChicheckInvadersVisualizer: Renders Chicheck Invaders episodes.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_schema import (
    CHICKEN_ALIVE,
    CHICKEN_COLUMN,
    CHICKEN_DIRECTION,
    CHICKEN_MODE,
    CHICKEN_ROW,
    MODE_DIVE,
    SHIP_COLUMN_INDEX,
    STEP_INDEX,
    chicken_slots,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.chicheck_invaders_pomdp.chicheck_invaders_pomdp import (
        ChicheckInvadersPOMDP,
    )


_ACTION_LABELS = ("hold", "move left", "move right", "fire")

COLOR_PAGE = (12, 14, 24)
COLOR_PANEL = (20, 24, 40)
COLOR_GRID = (38, 44, 68)
COLOR_TEXT = (226, 232, 245)
COLOR_TEXT_DIM = (132, 142, 170)
COLOR_SHIP = (96, 214, 255)
COLOR_SHIP_DARK = (30, 120, 160)
COLOR_CHICKEN = (246, 200, 84)
COLOR_CHICKEN_DIVE = (240, 96, 84)
COLOR_SHOT = (180, 255, 210)
COLOR_CAMERA = (74, 120, 190)
COLOR_RADAR = (96, 190, 150)
COLOR_BELIEF = (150, 240, 220)

CELL = 34
MARGIN = 28
PANEL_GAP = 40
HEADER = 78
CAPTION = 74


def _blend(
    base: Tuple[int, int, int], other: Tuple[int, int, int], weight: float
) -> Tuple[int, int, int]:
    """Mix two colours, ``weight`` of the way from ``base`` to ``other``."""
    amount = min(max(float(weight), 0.0), 1.0)
    return tuple(int(round(b + (o - b) * amount)) for b, o in zip(base, other))  # type: ignore[return-value]


def _font(size: int) -> ImageFont.FreeTypeFont:
    """Return a font of roughly ``size`` pixels, from Pillow's own bundled face.

    Pillow 10.1 and later scale their bundled Aileron face, which keeps text
    identical on every machine carrying the same Pillow -- and that is what the
    golden hash depends on. Searching the host for ``DejaVuSans.ttf`` instead
    would make the rendered pixels depend on which fonts happen to be installed,
    so a golden generated on a laptop would never match one rendered in CI.
    Older Pillow ignores the size and hands back one fixed bitmap face, which
    would silently collapse every text size into one, so this raises instead.

    Args:
        size: Requested height in pixels.

    Returns:
        A scalable font.

    Raises:
        RuntimeError: If Pillow is too old, or was built without FreeType.
    """
    try:
        font = ImageFont.load_default(size=size)
    except TypeError as exc:  # pragma: no cover - only on Pillow < 10.1
        raise RuntimeError(
            "The Chicheck Invaders renderer needs Pillow >= 10.1 for scalable default fonts."
        ) from exc
    if not isinstance(font, ImageFont.FreeTypeFont):  # pragma: no cover
        raise RuntimeError(
            "The Chicheck Invaders renderer needs a FreeType-enabled Pillow for scalable "
            f"fonts; this one returned {type(font).__name__}."
        )
    return font


class ChicheckInvadersVisualizer:
    """Renders an episode of :class:`ChicheckInvadersPOMDP` to an animated GIF.

    Attributes:
        environment: The environment whose geometry every frame is drawn from.
    """

    def __init__(self, environment: "ChicheckInvadersPOMDP"):
        """Initialize the renderer.

        Args:
            environment: The environment being rendered.
        """
        self.environment = environment
        self._columns = environment.num_columns
        self._rows = environment.num_rows
        self._panel = (self._columns * CELL, self._rows * CELL)
        self._width = MARGIN * 2 + self._panel[0] * 2 + PANEL_GAP
        self._height = HEADER + self._panel[1] + CAPTION
        self._title_font = _font(20)
        self._label_font = _font(13)
        self._caption_font = _font(14)

    # -- geometry -------------------------------------------------------

    def _panel_origin(self, index: int) -> Tuple[int, int]:
        """Top-left pixel of panel ``index`` (0 = the sky, 1 = the belief)."""
        return MARGIN + index * (self._panel[0] + PANEL_GAP), HEADER

    def _cell_box(
        self, origin: Tuple[int, int], column: int, row: int
    ) -> Tuple[int, int, int, int]:
        """Pixel box of grid cell ``(column, row)``, with row 0 at the bottom."""
        left = origin[0] + int(column) * CELL
        top = origin[1] + (self._rows - 1 - int(row)) * CELL
        return left, top, left + CELL - 1, top + CELL - 1

    # -- static chrome --------------------------------------------------

    def _background(self) -> Image.Image:
        """Draw the page, the two panel wells and their headings, once."""
        page = Image.new("RGB", (self._width, self._height), COLOR_PAGE)
        draw = ImageDraw.Draw(page)
        draw.text((MARGIN, 18), "Chicheck Invaders", font=self._title_font, fill=COLOR_TEXT)
        headings = (
            ("The sky", "true flock, ship and shots; hidden from the ship"),
            ("What the ship believes", "weighted chance of a chicken per cell"),
        )
        for index, (heading, blurb) in enumerate(headings):
            origin = self._panel_origin(index)
            draw.text((origin[0], HEADER - 38), heading, font=self._label_font, fill=COLOR_TEXT)
            draw.text((origin[0], HEADER - 21), blurb, font=self._label_font, fill=COLOR_TEXT_DIM)
            draw.rectangle(
                [
                    origin[0],
                    origin[1],
                    origin[0] + self._panel[0] - 1,
                    origin[1] + self._panel[1] - 1,
                ],
                fill=COLOR_PANEL,
                outline=COLOR_GRID,
            )
            for column in range(self._columns + 1):
                x = origin[0] + column * CELL
                draw.line([x, origin[1], x, origin[1] + self._panel[1] - 1], fill=COLOR_GRID)
            for row in range(self._rows + 1):
                y = origin[1] + row * CELL
                draw.line([origin[0], y, origin[0] + self._panel[0] - 1, y], fill=COLOR_GRID)
        return page

    # -- sprites --------------------------------------------------------

    def _draw_sensors(self, frame: Image.Image, origin: Tuple[int, int], ship: int) -> None:
        """Draw the camera cone's edges and the radar's ring for this ship column.

        Outlines rather than shaded cells. With the default 45-degree cone and a
        six-cell radar, filling every covered cell tints most of the panel, and a
        panel that is tinted almost everywhere says nothing; the two boundaries
        say exactly where each sensor stops. Both are drawn onto a panel-sized
        overlay and pasted, which is what clips the ring at the panel's edge
        instead of letting it spill into the neighbouring one.

        Args:
            frame: The frame being built.
            origin: Top-left pixel of the panel to draw into.
            ship: The ship's column.
        """
        overlay = Image.new("RGBA", self._panel, (0, 0, 0, 0))
        pen = ImageDraw.Draw(overlay)
        left, top, right, bottom = self._cell_box((0, 0), ship, 0)
        centre = ((left + right) / 2.0, (top + bottom) / 2.0)

        radius = self.environment.radar_radius * CELL
        pen.ellipse(
            [centre[0] - radius, centre[1] - radius, centre[0] + radius, centre[1] + radius],
            outline=COLOR_RADAR + (190,),
            width=2,
        )
        reach = self._rows * CELL
        for side in (-1.0, 1.0):
            pen.line(
                [
                    centre,
                    (
                        centre[0] + side * self.environment.camera_slope * reach,
                        centre[1] - reach,
                    ),
                ],
                fill=COLOR_CAMERA + (210,),
                width=2,
            )
        frame.paste(overlay, origin, overlay)

    def _draw_ship(self, draw: ImageDraw.ImageDraw, origin: Tuple[int, int], ship: int) -> None:
        """Draw the ship as a wedge on row 0 of its column."""
        left, top, right, bottom = self._cell_box(origin, ship, 0)
        mid = (left + right) // 2
        draw.polygon(
            [(mid, top + 4), (right - 4, bottom - 5), (left + 4, bottom - 5)],
            fill=COLOR_SHIP,
            outline=COLOR_SHIP_DARK,
        )
        draw.rectangle([mid - 2, top + 1, mid + 2, top + 6], fill=COLOR_SHIP)

    def _draw_chicken(
        self,
        draw: ImageDraw.ImageDraw,
        origin: Tuple[int, int],
        column: int,
        row: int,
        diving: bool,
        direction: int,
    ) -> None:
        """Draw one live chicken; head down when diving, walking when not."""
        left, top, right, bottom = self._cell_box(origin, column, row)
        colour = COLOR_CHICKEN_DIVE if diving else COLOR_CHICKEN
        body = [left + 7, top + 8, right - 7, bottom - 7]
        draw.ellipse(body, fill=colour)
        mid_x = (left + right) // 2
        if diving:
            # Head below the body and a pair of swept-back wings: a silhouette
            # that reads as "coming down" at this size, where an arrow would not.
            draw.polygon(
                [(mid_x, bottom - 2), (mid_x - 4, bottom - 9), (mid_x + 4, bottom - 9)],
                fill=colour,
            )
            draw.line([left + 4, top + 5, mid_x - 4, top + 12], fill=colour, width=2)
            draw.line([right - 4, top + 5, mid_x + 4, top + 12], fill=colour, width=2)
        else:
            draw.ellipse([mid_x - 4, top + 3, mid_x + 4, top + 11], fill=colour)
            tip = (right - 3) if direction > 0 else (left + 3)
            draw.line([mid_x, top + 7, tip, top + 7], fill=colour, width=2)

    def _draw_beam(
        self, draw: ImageDraw.ImageDraw, origin: Tuple[int, int], ship: int, struck: int
    ) -> None:
        """Flash the ship's column, and ring the chicken the shot took.

        Drawn only on a step whose action actually discharged the gun. The beam
        stops at the chicken it killed rather than running to the top of the
        grid, so a reader can see what the shot bought; a shot into an empty
        column runs the full height and rings nothing.

        Args:
            draw: The frame's drawing context.
            origin: Top-left pixel of the panel.
            ship: The ship's column.
            struck: Row of the chicken killed, or ``-1`` for a miss.
        """
        top_row = self._rows - 1 if struck < 0 else struck
        left, _, right, bottom = self._cell_box(origin, ship, 0)
        _, top, _, _ = self._cell_box(origin, ship, top_row)
        mid = (left + right) // 2
        draw.line([mid, bottom - 8, mid, top + 4], fill=COLOR_SHOT, width=3)
        if struck >= 0:
            box = self._cell_box(origin, ship, struck)
            draw.ellipse(
                [box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2], outline=COLOR_SHOT, width=2
            )

    # -- belief ---------------------------------------------------------

    def _occupancy_marginal(self, belief: Optional[object]) -> np.ndarray:
        """Weighted chance that each cell holds *at least one* live chicken.

        This is an occupancy probability, not an expected count: each particle
        contributes its whole weight to a cell once, no matter how many of its
        chickens are standing on it.

        Args:
            belief: The step's belief, or ``None``.

        Returns:
            A ``(num_rows, num_columns)`` array in ``[0, 1]``; all zeros when
            there is no belief to read.
        """
        grid = np.zeros((self._rows, self._columns), dtype=np.float64)
        particles = getattr(belief, "particles", None)
        if not particles:
            return grid
        weights = getattr(belief, "normalized_weights", None)
        if weights is None:
            weights = np.full(len(particles), 1.0 / len(particles))
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if len(weights) != len(particles) or not np.isfinite(weights).all():
            weights = np.full(len(particles), 1.0 / len(particles))
        for particle, weight in zip(particles, weights):
            state = np.asarray(particle, dtype=np.float64)
            if state.shape != (self.environment.state_size,):
                continue
            # One particle contributes its weight to a cell *once*, however many
            # of its chickens stand there. Adding a weight per chicken would
            # render the expected chicken **count**, which is a different
            # quantity: two particles agreeing that one cell holds two chickens
            # would paint it as though it were certain, and a ``clip`` to 1
            # would hide the overflow rather than fix it. Patrols walking into
            # each other do stack, so this is reachable rather than theoretical.
            occupied = np.zeros_like(grid, dtype=bool)
            flock = chicken_slots(state, self.environment.num_chickens)
            for slot in flock:
                if slot[CHICKEN_ALIVE] <= 0.0:
                    continue
                column = int(round(float(slot[CHICKEN_COLUMN])))
                row = int(round(float(slot[CHICKEN_ROW])))
                if 0 <= column < self._columns and 0 <= row < self._rows:
                    occupied[row, column] = True
            grid += float(weight) * occupied
        # A probability by construction now: a convex combination of indicator
        # grids. The bound is belt and braces against a weight vector that does
        # not quite sum to one.
        return np.clip(grid, 0.0, 1.0)

    def _draw_belief(
        self, draw: ImageDraw.ImageDraw, origin: Tuple[int, int], grid: np.ndarray
    ) -> None:
        """Paint the belief marginal, one shaded cell per grid cell."""
        for row in range(self._rows):
            for column in range(self._columns):
                mass = float(grid[row, column])
                if mass <= 0.0:
                    continue
                box = self._cell_box(origin, column, row)
                draw.rectangle(
                    [box[0] + 1, box[1] + 1, box[2], box[3]],
                    fill=_blend(COLOR_PANEL, COLOR_BELIEF, 0.15 + 0.85 * mass),
                )

    # -- frames ---------------------------------------------------------

    def _frame(self, step: StepData, background: Image.Image) -> Image.Image:
        """Render one frame from one recorded step."""
        frame = background.copy()
        draw = ImageDraw.Draw(frame)
        state = np.asarray(step.state, dtype=np.float64)
        ship = int(round(float(state[SHIP_COLUMN_INDEX])))

        sky_origin = self._panel_origin(0)
        belief_origin = self._panel_origin(1)
        self._draw_belief(draw, belief_origin, self._occupancy_marginal(step.belief))
        self._draw_sensors(frame, sky_origin, ship)
        self._draw_sensors(frame, belief_origin, ship)
        draw = ImageDraw.Draw(frame)

        for slot in chicken_slots(state, self.environment.num_chickens):
            # A dead chicken is drawn as nothing: its slot survives in the state
            # and in every observation, but it has left the sky.
            if slot[CHICKEN_ALIVE] <= 0.0:
                continue
            self._draw_chicken(
                draw,
                sky_origin,
                int(round(float(slot[CHICKEN_COLUMN]))),
                int(round(float(slot[CHICKEN_ROW]))),
                slot[CHICKEN_MODE] == MODE_DIVE,
                int(round(float(slot[CHICKEN_DIRECTION]))),
            )
        if step.action is not None and self.environment.fires(state, step.action):
            target = self.environment.shot_target(state)
            struck = (
                -1
                if target < 0
                else int(
                    round(
                        float(
                            chicken_slots(state, self.environment.num_chickens)[target][CHICKEN_ROW]
                        )
                    )
                )
            )
            self._draw_beam(draw, sky_origin, ship, struck)
        self._draw_ship(draw, sky_origin, ship)
        self._draw_ship(draw, belief_origin, ship)
        draw.text(
            (MARGIN, self._height - CAPTION + 10),
            self._caption(step, state),
            font=self._caption_font,
            fill=COLOR_TEXT,
        )
        for line, text in enumerate(self._legend()):
            draw.text(
                (MARGIN, self._height - CAPTION + 32 + line * 17),
                text,
                font=self._label_font,
                fill=COLOR_TEXT_DIM,
            )
        return frame

    def _caption(self, step: StepData, state: np.ndarray) -> str:
        """One line naming the step, the action about to be taken and its reward."""
        index = int(round(float(state[STEP_INDEX])))
        alive = self.environment.live_chicken_count(state)
        if step.action is None:
            return f"step {index} - final state - {alive} chicken(s) left"
        label = _ACTION_LABELS[int(step.action)]
        reward = "" if step.reward is None else f" - reward {float(step.reward):+.2f}"
        return f"step {index} - about to {label} - {alive} chicken(s) left{reward}"

    @staticmethod
    def _legend() -> Tuple[str, str]:
        """The key, spelled out rather than drawn, so it survives any font.

        Two lines rather than one because the single line ran past the right
        edge of the narrower default grid and lost its last entry.
        """
        return (
            "blue lines: edges of the camera cone    green arc: radar range",
            "yellow: patrolling    red, head down: diving    pale beam: this step's shot",
        )

    def create_visualization(self, history: List[StepData], output_path: Path) -> None:
        """Write ``history`` to ``output_path`` as an animated GIF.

        Args:
            history: The episode's recorded steps.
            output_path: File to write. Its parent is created if missing.

        Raises:
            ValueError: If ``history`` is empty -- an animation of no steps is a
                file that decodes but shows nothing, and the coverage matrix
                rejects a single-frame GIF for exactly that reason.
        """
        if not history:
            raise ValueError("cannot render a Chicheck Invaders episode with no recorded steps")
        background = self._background()
        frames = [self._frame(step, background) for step in history]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=420,
            loop=0,
            optimize=False,
        )
