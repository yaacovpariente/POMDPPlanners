# SPDX-License-Identifier: MIT

"""Visualization for the occupancy-grid mapping POMDP.

Renders an episode as an animated GIF built to read like a game screen: a dark
console page, three instrument panels in machined metal, and a robot that
sweeps fog off a map.  Pillow and numpy only, in the style the Light-Dark
renderer established here, for two reasons rather than Matplotlib.  Pillow
gives direct control of pixels, which is what buys the textures, the lighting
and the sprite work; and everything that never changes -- page, chrome,
headings, keys and the whole ground-truth panel -- is rasterised once per
episode and copied, so the treatment costs almost nothing per frame.

The three panels per frame are unchanged in meaning, and each is given its own
material so that a reader never has to check a title to tell them apart:

* **Observed map** -- the map the robot has built, as the occupancy
  probability of each cell, drawn as fog burning off a chart.  This is the
  thing the task is *about*: the reward is the rate at which this panel's fog
  resolves into swept floor or into a wall.  A cell's colour is driven by
  evidence away from ``p = 0.5`` in either direction and the static over it is
  scaled by how unsure the robot is, so the panel gets calmer exactly as the
  entropy the reward pays for falls.  The robot's trail and its measured beams
  are drawn here, since this panel is what those beams produce.
* **Map estimate** -- the per-cell posterior probability that the cell is
  occupied, averaged over the belief's particles with their weights, drawn as a
  cyan holographic readout.  This is the belief view the environment contract
  asks for.  Drawing the particles themselves is the usual choice, but one
  particle is a whole hundred-cell world and a hundred overlaid worlds are not
  a picture of anything; the weighted marginal is the projection the task turns
  on, and it is still the belief rather than a fit to it.  No robot and no
  beams are drawn here, which keeps the panel unmistakably a belief and not a
  second map.
* **Ground truth** -- the real map as the level it is, stone floor and lit
  blocks, with the robot's pose, heading and trail, drawn only here and
  labelled as hidden from the robot.  It exists so a reviewer can check the
  other two panels against what was really there.

Both probability panels are keyed, because they share a layout and mean
different things: a planner that is exploring well drives fog out of the left
panel, while a filter that is tracking well makes the middle panel approach the
right one.

Every frame is drawn *before* its own action resolves, because
:class:`~POMDPPlanners.core.simulation.StepData` records the state a step was
taken *from*: each frame shows what the robot chose from, and the caption says
what it is about to do.  The beams shown are the ones stored in that state, the
readings that produced the map beside them.  The initial state carries zero
range placeholders rather than a reading, so the first frame draws no beams.

Classes:
    OccupancyGridMappingVisualizer: Renders occupancy-grid mapping episodes.
"""

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_assets import (
    COLOR_BEAM,
    COLOR_CHART_FOG,
    COLOR_CHART_FREE,
    COLOR_CHART_OCCUPIED,
    COLOR_HOLO_BLANK,
    COLOR_HOLO_HIGH,
    COLOR_HOLO_LOW,
    COLOR_HOLO_MID,
    COLOR_METAL,
    COLOR_METAL_DARK,
    COLOR_METER_FILL,
    COLOR_METER_GOAL,
    COLOR_PAGE,
    COLOR_ROBOT,
    COLOR_TEXT,
    COLOR_TEXT_DIM,
    COLOR_TRAIL,
    COLOR_TRUTH_FLOOR,
    COLOR_TRUTH_WALL,
    draw_bezel,
    draw_gradient_bar,
    draw_screen_edge,
    get_font,
    mix_colors,
    robot_sprite_facing,
    static_field,
    stone_floor,
    stone_shade,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_pomdp import (
    STEP_INDEX,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_sensor import (
    HEADING_LABELS,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_pomdp import (  # noqa: E501
        OccupancyGridMappingPOMDP,
    )


_ACTION_LABELS = ("move forward", "turn left", "turn right")

_PANEL_HEADINGS = (
    ("Observed map", "approximate occupancy from the measured scans"),
    ("Map estimate", "weighted chance a cell is occupied, across the belief's worlds"),
    ("Ground truth", "hidden occupancy; the robot knows its pose"),
)

# Canvas and layout, in pixels. The canvas keeps the dimensions the earlier
# render used so saved episodes stay comparable frame for frame.
CANVAS_SIZE = (1500, 850)
MARGIN = 41
CARD_WIDTH = 450
CARD_GAP = 34
CARD_TOP = 124
CARD_BOTTOM = 704
#: Side of a panel's square screen. Divisible by the common grid sizes, so the
#: default 10x10 world lands on whole 40-pixel cells with no resampling seam.
SCREEN_SIZE = 400
SCREEN_INSET = 25
SCREEN_TOP = 188

CAPTION_TOP = 716
CAPTION_BOTTOM = 812

METER_LEFT = 1154
METER_RIGHT = 1454
METER_TOP = 74
METER_BOTTOM = 90

#: Milliseconds each frame is held, and the longer hold on the last one.
FRAME_MS = 900
FINAL_FRAME_MS = 2400

#: Blocks in a panel's colour scale.
_RAMP_BLOCKS = 14

#: Below this occupancy probability a cell is drawn as swept floor, above
#: ``1 - _RESOLVED`` as a wall. Only affects the tile detailing, never the fill.
_RESOLVED = 0.34

# Colours that must survive GIF quantization even though they cover few pixels.
# Median cut weights a colour by how many pixels carry it, so the large dark
# areas would otherwise take every slot and the beams, the trail and the robot
# would come out muddy.
_ACCENT_COLORS: Tuple[Tuple[int, int, int], ...] = (
    COLOR_BEAM,
    COLOR_ROBOT,
    COLOR_TRAIL,
    COLOR_CHART_OCCUPIED,
    COLOR_CHART_FOG,
    COLOR_CHART_FREE,
    COLOR_HOLO_HIGH,
    COLOR_HOLO_MID,
    COLOR_HOLO_LOW,
    COLOR_HOLO_BLANK,
    COLOR_TRUTH_WALL,
    COLOR_TRUTH_FLOOR,
    COLOR_METAL,
    COLOR_METAL_DARK,
    COLOR_METER_FILL,
    COLOR_METER_GOAL,
    COLOR_TEXT,
    COLOR_TEXT_DIM,
    COLOR_PAGE,
    (255, 255, 255),
    (0, 0, 0),
)


class OccupancyGridMappingVisualizer:
    """Renders an occupancy-grid mapping episode as an animated GIF.

    Attributes:
        env: The environment the episode was run in, used for grid geometry and
            to read state and belief in its own layout.
        num_rows: Grid rows, copied from the environment.
        num_cols: Grid columns, copied from the environment.
        num_cells: Cell count, copied from the environment.
    """

    def __init__(self, env: "OccupancyGridMappingPOMDP"):
        """Initialize the visualizer.

        Args:
            env: The environment instance to visualize.
        """
        self.env = env
        self.num_rows = env.num_rows
        self.num_cols = env.num_cols
        self.num_cells = env.num_cells
        self._background: Optional[Image.Image] = None
        self._background_key: Optional[Tuple[Any, ...]] = None
        self._truth_cache: Optional[Image.Image] = None

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
        frames = self._build_frames(history)
        images = self.render_frames(frames)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._save(images, cache_path)

    def render_frames(self, frames: List[Dict[str, Any]]) -> List[Image.Image]:
        """Render drawing records to full RGB frames.

        Exposed so tests and contact sheets can inspect a single frame without
        decoding a GIF.

        Args:
            frames: Records from :meth:`_build_frames`.

        Returns:
            One RGB image per record, all :data:`CANVAS_SIZE`.
        """
        background = self._background_for(frames[0])
        return [self._render_frame(background, frame) for frame in frames]

    # -- history to drawing records --------------------------------------

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

    def _belief_marginal(self, belief: Any) -> np.ndarray:
        """Per-cell occupancy probability of the true map under ``belief``.

        The weighted mean over particles, which is the exact marginal of a
        weighted particle belief rather than a summary of it. Deterministic in
        the belief, which the golden-file hash depends on.

        Args:
            belief: The belief recorded on the step.

        Returns:
            ``(num_rows, num_cols)`` array of probabilities, all ``NaN`` when
            the belief carries nothing usable, which the renderer draws blank.
        """
        blank = np.full((self.num_rows, self.num_cols), np.nan)
        if belief is None:
            return blank
        particles = getattr(belief, "particles", None)
        if particles is None or len(particles) == 0:
            return blank
        weights = np.asarray(
            getattr(
                belief,
                "normalized_weights",
                np.full(len(particles), 1.0 / len(particles)),
            ),
            dtype=np.float64,
        )
        offset = self.env.map_offset
        occupancy = (
            np.asarray(particles, dtype=np.float64)[:, offset : offset + self.num_cells] > 0.5
        )
        marginal = (weights[:, None] * occupancy).sum(axis=0) / weights.sum()
        return marginal.reshape(self.num_rows, self.num_cols)

    def _measured_ranges(self, state: np.ndarray) -> Optional[np.ndarray]:
        """The noisy ranges stored in ``state``, or ``None`` before the first scan.

        The environment's initial observation is a sentinel: the known start
        pose and zero range placeholders, never applied to the map. Drawing it
        as a scan would put twenty-four zero-length beams on the first frame and
        imply the robot had measured its own cell, so it is reported as absent.

        Args:
            state: A state vector in the environment's layout.

        Returns:
            ``(num_beams,)`` ranges in cell widths, or ``None``.
        """
        if float(state[STEP_INDEX]) <= 0.0:
            return None
        return np.asarray(state[self.env.scan_offset :], dtype=np.float64)

    def _beam_directions(self, heading: int) -> np.ndarray:
        """Unit ``(row, col)`` direction of every beam at ``heading``.

        Reproduces the bin-centre bearings
        :func:`~POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_sensor.build_ray_templates`
        uses, so a drawn beam points exactly where the sensor cast it.

        Args:
            heading: Heading index, ``0`` north through ``3`` west.

        Returns:
            ``(num_beams, 2)`` array of unit directions in ``(row, col)``.
        """
        span = float(self.env.field_of_view_degrees)
        count = int(self.env.num_beams)
        base = 90.0 * int(heading)
        directions = np.empty((count, 2), dtype=np.float64)
        for index in range(count):
            bearing = math.radians(base - span / 2.0 + span * (index + 0.5) / count)
            directions[index] = (-math.cos(bearing), math.sin(bearing))
        return directions

    def _build_frames(self, history: List[StepData]) -> List[Dict[str, Any]]:
        """Turn the history into one self-contained drawing record per step."""
        frames: List[Dict[str, Any]] = []
        trail: List[Tuple[int, int]] = []
        for index, step in enumerate(history):
            state = np.asarray(step.state, dtype=np.float64)
            row, col, heading = self.env.pose(state)
            trail.append((row, col))
            frames.append(
                {
                    "index": index,
                    "total": len(history),
                    "grid": self.env.occupancy_probabilities(state),
                    "belief": self._belief_marginal(step.belief),
                    "truth": np.asarray(self.env.true_map(state), dtype=np.float64),
                    "row": row,
                    "col": col,
                    "heading": heading,
                    "trail": list(trail),
                    "ranges": self._measured_ranges(state),
                    "entropy": self.env.entropy_bits(state),
                    "action": step.action,
                    "reward": step.reward,
                }
            )
        return frames

    # -- geometry --------------------------------------------------------

    @staticmethod
    def _card_left(panel: int) -> int:
        return MARGIN + panel * (CARD_WIDTH + CARD_GAP)

    def _screen_box(self, panel: int) -> Tuple[int, int, int, int]:
        left = self._card_left(panel) + SCREEN_INSET
        return left, SCREEN_TOP, left + SCREEN_SIZE, SCREEN_TOP + SCREEN_SIZE

    @property
    def _cell_size(self) -> Tuple[float, float]:
        """Screen pixels per cell, as ``(width, height)``."""
        return SCREEN_SIZE / self.num_cols, SCREEN_SIZE / self.num_rows

    def _cell_centre(self, row: float, col: float) -> Tuple[float, float]:
        """Screen-local pixel centre of a cell, as ``(x, y)``."""
        cell_w, cell_h = self._cell_size
        return (col + 0.5) * cell_w, (row + 0.5) * cell_h

    def _cell_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Integer screen-local pixel boundaries of the columns and the rows."""
        xs = np.round(np.arange(self.num_cols + 1) * SCREEN_SIZE / self.num_cols).astype(int)
        ys = np.round(np.arange(self.num_rows + 1) * SCREEN_SIZE / self.num_rows).astype(int)
        return xs, ys

    def _upscale(self, values: np.ndarray) -> np.ndarray:
        """Blow a per-cell array up to screen pixels with no interpolation."""
        xs, ys = self._cell_edges()
        column_index = np.zeros(SCREEN_SIZE, dtype=np.int64)
        row_index = np.zeros(SCREEN_SIZE, dtype=np.int64)
        for col in range(self.num_cols):
            column_index[xs[col] : xs[col + 1]] = col
        for row in range(self.num_rows):
            row_index[ys[row] : ys[row + 1]] = row
        return values[np.ix_(row_index, column_index)]

    # -- panel screens ---------------------------------------------------

    def _observed_screen(self, probabilities: np.ndarray) -> Image.Image:
        """Draw the observed map: fog resolving into floor or wall."""
        probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
        free = np.clip((0.5 - probabilities) / 0.5, 0.0, 1.0)
        occupied = np.clip((probabilities - 0.5) / 0.5, 0.0, 1.0)

        fog = np.array(COLOR_CHART_FOG, dtype=np.float64)
        toward_free = np.array(COLOR_CHART_FREE, dtype=np.float64) - fog
        toward_occupied = np.array(COLOR_CHART_OCCUPIED, dtype=np.float64) - fog
        cells = (
            fog[None, None, :]
            + free[:, :, None] * toward_free[None, None, :]
            + occupied[:, :, None] * toward_occupied[None, None, :]
        )

        rgb = np.stack([self._upscale(cells[:, :, i]) for i in range(3)], axis=2)
        # Static, scaled by how unsure the robot is of the cell under it. An
        # unknown cell fizzes; a resolved one is flat.
        uncertainty = self._upscale(1.0 - np.abs(2.0 * probabilities - 1.0))
        speckle = static_field(SCREEN_SIZE, SCREEN_SIZE)
        rgb *= 1.0 + (speckle - 1.0)[:, :, None] * 0.55 * uncertainty[:, :, None]

        screen = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(screen, "RGBA")
        self._draw_cell_detail(draw, probabilities)
        return screen

    def _draw_cell_detail(self, draw: ImageDraw.ImageDraw, probabilities: np.ndarray) -> None:
        """Seams between cells, tile insets on swept floor, bevels on walls."""
        xs, ys = self._cell_edges()
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                x0, x1 = int(xs[col]), int(xs[col + 1]) - 1
                y0, y1 = int(ys[row]), int(ys[row + 1]) - 1
                probability = float(probabilities[row, col])
                if probability >= 1.0 - _RESOLVED:
                    draw.line([(x0 + 1, y0 + 1), (x1 - 1, y0 + 1)], fill=(255, 226, 176, 150))
                    draw.line([(x0 + 1, y0 + 1), (x0 + 1, y1 - 1)], fill=(255, 226, 176, 150))
                    draw.line([(x0 + 1, y1 - 1), (x1 - 1, y1 - 1)], fill=(120, 66, 12, 190))
                    draw.line([(x1 - 1, y0 + 1), (x1 - 1, y1 - 1)], fill=(120, 66, 12, 190))
                elif probability <= _RESOLVED:
                    draw.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], outline=(94, 168, 178, 55))
        for x in xs[1:-1]:
            draw.line([(int(x), 0), (int(x), SCREEN_SIZE)], fill=(6, 12, 16, 130))
        for y in ys[1:-1]:
            draw.line([(0, int(y)), (SCREEN_SIZE, int(y))], fill=(6, 12, 16, 130))

    def _belief_screen(self, marginal: np.ndarray) -> Image.Image:
        """Draw the belief marginal as a cyan holographic readout."""
        values = np.asarray(marginal, dtype=np.float64)
        missing = ~np.isfinite(values)
        filled = np.where(missing, 0.0, np.clip(values, 0.0, 1.0))

        low = np.array(COLOR_HOLO_LOW, dtype=np.float64)
        mid = np.array(COLOR_HOLO_MID, dtype=np.float64)
        high = np.array(COLOR_HOLO_HIGH, dtype=np.float64)
        lower = np.clip(filled / 0.5, 0.0, 1.0)
        upper = np.clip((filled - 0.5) / 0.5, 0.0, 1.0)
        cells = (
            low[None, None, :]
            + lower[:, :, None] * (mid - low)[None, None, :]
            + upper[:, :, None] * (high - mid)[None, None, :]
        )
        blank = np.array(COLOR_HOLO_BLANK, dtype=np.float64)
        cells = np.where(missing[:, :, None], blank[None, None, :], cells)

        rgb = np.stack([self._upscale(cells[:, :, i]) for i in range(3)], axis=2)
        # Scanlines. A projected readout is the one thing in the frame that is
        # not a physical surface, and this is what says so.
        rgb[1::3, :, :] *= 0.84

        screen = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(screen, "RGBA")
        xs, ys = self._cell_edges()
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if missing[row, col]:
                    # Hatched, not just a different colour. A reader scanning
                    # the ramp reads any flat fill as a probability; the hatch
                    # says this cell is not on the ramp at all.
                    x0, x1 = int(xs[col]) + 1, int(xs[col + 1]) - 2
                    y0, y1 = int(ys[row]) + 1, int(ys[row + 1]) - 2
                    for start in range(x0 - (y1 - y0), x1, 7):
                        draw.line(
                            [
                                (max(start, x0), y0 + max(0, x0 - start)),
                                (
                                    min(start + (y1 - y0), x1),
                                    y0 + min(y1 - y0, x1 - start),
                                ),
                            ],
                            fill=(38, 38, 42, 220),
                        )
                    continue
                if filled[row, col] < 0.55:
                    continue
                x0, x1 = int(xs[col]) + 2, int(xs[col + 1]) - 3
                y0, y1 = int(ys[row]) + 2, int(ys[row + 1]) - 3
                alpha = int(90 + 140 * min(1.0, (float(filled[row, col]) - 0.55) / 0.45))
                draw.rectangle([x0, y0, x1, y1], outline=COLOR_HOLO_HIGH + (alpha,))
        # A lattice bright enough to show over the dark low-probability cells
        # as well as under the bright ones, so the panel reads as a projected
        # grid everywhere rather than only where the belief is confident.
        for x in xs[1:-1]:
            draw.line([(int(x), 0), (int(x), SCREEN_SIZE)], fill=(58, 150, 172, 130))
        for y in ys[1:-1]:
            draw.line([(0, int(y)), (SCREEN_SIZE, int(y))], fill=(58, 150, 172, 130))
        return screen

    def _truth_screen(self, truth: np.ndarray) -> Image.Image:
        """Draw the hidden map as the level it is: stone floor and lit blocks."""
        occupied = np.asarray(truth, dtype=np.float64) > 0.5
        shade = stone_shade(SCREEN_SIZE, SCREEN_SIZE)
        floor = stone_floor(SCREEN_SIZE, SCREEN_SIZE) * 255.0
        wall = (
            np.array(COLOR_TRUTH_WALL, dtype=np.float64)[None, None, :]
            * np.clip(0.74 + 0.34 * shade, 0.0, 1.35)[:, :, None]
        )
        mask = self._upscale(occupied.astype(np.float64))[:, :, None]
        rgb = floor * (1.0 - mask) + wall * mask

        screen = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(screen, "RGBA")
        xs, ys = self._cell_edges()
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                x0, x1 = int(xs[col]), int(xs[col + 1]) - 1
                y0, y1 = int(ys[row]), int(ys[row + 1]) - 1
                if occupied[row, col]:
                    draw.line([(x0 + 1, y0 + 1), (x1 - 1, y0 + 1)], fill=(226, 238, 246, 190))
                    draw.line([(x0 + 1, y0 + 1), (x0 + 1, y1 - 1)], fill=(226, 238, 246, 190))
                    draw.line([(x0 + 1, y1 - 1), (x1 - 1, y1 - 1)], fill=(24, 34, 42, 210))
                    draw.line([(x1 - 1, y0 + 1), (x1 - 1, y1 - 1)], fill=(24, 34, 42, 210))
                elif x1 - x0 >= 8 and y1 - y0 >= 8:
                    # The tile inset is a fixed 3px, so on a grid fine enough
                    # that a cell is under 8px the inset would invert and draw
                    # clutter instead of a tile. Skipped rather than scaled: at
                    # that size the tile is not legible either way.
                    draw.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], outline=(96, 122, 138, 45))
        for x in xs[1:-1]:
            draw.line([(int(x), 0), (int(x), SCREEN_SIZE)], fill=(8, 14, 18, 120))
        for y in ys[1:-1]:
            draw.line([(0, int(y)), (SCREEN_SIZE, int(y))], fill=(8, 14, 18, 120))
        return screen

    # -- moving parts -----------------------------------------------------

    def _draw_beams(self, draw: ImageDraw.ImageDraw, frame: Dict[str, Any]) -> None:
        """Draw the swept region and every measured beam of the stored scan.

        Ranges are the noisy readings the state carries, not the noise-free
        cast, so a beam that overshot a wall visibly overshoots it and one that
        fell short visibly falls short. That mismatch is the environment's
        whole difficulty, and hiding it would make the render a lie.

        Args:
            draw: Screen-local RGBA draw context. Pillow clips to the screen,
                so a beam leaving the grid is cut at the panel edge.
            frame: One record from :meth:`_build_frames`.
        """
        ranges = frame["ranges"]
        if ranges is None:
            return
        cell_w, cell_h = self._cell_size
        origin_x, origin_y = self._cell_centre(frame["row"], frame["col"])
        directions = self._beam_directions(frame["heading"])
        limit = float(self.env.max_range_cells)

        endpoints: List[Tuple[float, float]] = []
        hits: List[bool] = []
        for (delta_row, delta_col), measured in zip(directions, np.asarray(ranges, dtype=float)):
            # The noise is unclipped, so a reading can land past the sensor's
            # range or below zero. Both are drawn clamped, because that is what
            # the inverse model does with them: it treats anything at or beyond
            # max_range_cells as a miss that frees the whole ray, and the beam
            # is drawn to the range with no end spark to say exactly that.
            drawn = float(np.clip(measured, 0.0, limit))
            endpoints.append(
                (
                    origin_x + delta_col * drawn * cell_w,
                    origin_y + delta_row * drawn * cell_h,
                )
            )
            hits.append(bool(measured < limit))

        if len(endpoints) >= 3:
            # A full-surround fan already encloses the robot; a narrower one
            # needs the sensor's own position to close the wedge.
            polygon = (
                list(endpoints)
                if float(self.env.field_of_view_degrees) >= 359.9
                else [(origin_x, origin_y)] + list(endpoints)
            )
            draw.polygon(polygon, fill=COLOR_BEAM + (34,))

        for (end_x, end_y), hit in zip(endpoints, hits):
            draw.line(
                [(origin_x, origin_y), (end_x, end_y)],
                fill=COLOR_BEAM + (120 if hit else 70,),
                width=2,
            )
            if hit:
                draw.ellipse(
                    [end_x - 3, end_y - 3, end_x + 3, end_y + 3],
                    fill=COLOR_BEAM + (235,),
                )

    def _draw_trail(self, draw: ImageDraw.ImageDraw, trail: Sequence[Tuple[int, int]]) -> None:
        """Draw the cells the robot has already stood in, oldest faintest."""
        points = [self._cell_centre(row, col) for row, col in trail]
        # A turn leaves the robot where it was, so consecutive points repeat.
        # Pillow draws a zero-length segment as a blob, so they are dropped.
        line = [points[0]]
        for point in points[1:]:
            if point != line[-1]:
                line.append(point)
        if len(line) > 1:
            draw.line(line, fill=COLOR_TRAIL + (150,), width=4, joint="curve")
        count = len(points)
        for index, (x, y) in enumerate(points[:-1]):
            alpha = int(70 + 130 * (index + 1) / max(count, 1))
            draw.ellipse([x - 3.5, y - 3.5, x + 3.5, y + 3.5], fill=COLOR_TRAIL + (alpha,))

    def _paste_robot(self, screen: Image.Image, frame: Dict[str, Any]) -> None:
        """Paste the robot sprite onto an opaque screen at its recorded pose.

        Pasted onto the opaque screen rather than into the transparent overlay
        because compositing an RGBA sprite through a transparent layer
        multiplies by alpha twice and leaves a dark fringe on its rim.

        Args:
            screen: The panel's RGB screen image, modified in place.
            frame: One record from :meth:`_build_frames`.
        """
        cell_w, cell_h = self._cell_size
        size = max(14, int(round(1.12 * min(cell_w, cell_h))))
        sprite = robot_sprite_facing(size, int(frame["heading"]))
        centre_x, centre_y = self._cell_centre(frame["row"], frame["col"])
        screen.paste(sprite, (int(centre_x - size / 2), int(centre_y - size / 2)), sprite)

    # -- static background ------------------------------------------------

    def _background_for(self, frame: Dict[str, Any]) -> Image.Image:
        """Return the cached page, rebuilding it if the hidden map changed."""
        key = (
            self.num_rows,
            self.num_cols,
            np.asarray(frame["truth"], dtype=float).tobytes(),
        )
        if self._background is None or self._background_key != key:
            self._truth_cache = self._truth_screen(frame["truth"])
            self._background = self._build_background()
            self._background_key = key
        return self._background

    def _build_background(self) -> Image.Image:
        """Rasterise everything that does not move. Called once per episode.

        The ground-truth screen belongs here: the hidden map is fixed for the
        whole episode, so it is drawn once and only the trail and the robot on
        top of it are redrawn.

        Returns:
            The page with its chrome, headings, keys and truth panel in place.
        """
        assert self._truth_cache is not None
        canvas = Image.new("RGB", CANVAS_SIZE, COLOR_PAGE)
        draw = ImageDraw.Draw(canvas, "RGBA")

        draw.text(
            (46, 22),
            "OCCUPANCY GRID MAPPING",
            font=get_font(33),
            fill=COLOR_TEXT,
            anchor="la",
            stroke_width=1,
            stroke_fill=COLOR_TEXT,
        )
        draw.text(
            (46, 66),
            "Explore the unknown. Build the map.",
            font=get_font(14),
            fill=COLOR_TEXT_DIM,
            anchor="la",
        )

        tints = (COLOR_CHART_OCCUPIED, COLOR_HOLO_HIGH, COLOR_TRUTH_WALL)
        for panel, (name, description) in enumerate(_PANEL_HEADINGS):
            left = self._card_left(panel)
            draw_bezel(draw, (left, CARD_TOP, left + CARD_WIDTH, CARD_BOTTOM))
            centre = left + CARD_WIDTH // 2
            draw.text(
                (centre, CARD_TOP + 16),
                name,
                font=get_font(19),
                fill=COLOR_TEXT,
                anchor="ma",
                stroke_width=1,
                stroke_fill=COLOR_TEXT,
            )
            draw.text(
                (centre, CARD_TOP + 44),
                description,
                font=get_font(12),
                fill=COLOR_TEXT_DIM,
                anchor="ma",
            )
            draw_screen_edge(draw, self._screen_box(panel), tints[panel])

        canvas.paste(self._truth_cache, self._screen_box(2)[:2])

        self._draw_keys(canvas, draw)
        draw_bezel(
            draw,
            (MARGIN, CAPTION_TOP, CANVAS_SIZE[0] - MARGIN, CAPTION_BOTTOM),
            radius=10,
        )
        return canvas

    def _gradient_bar(
        self,
        draw: ImageDraw.ImageDraw,
        panel: int,
        top: int,
        stops: Sequence[Tuple[float, Tuple[int, int, int]]],
    ) -> None:
        """Draw ``panel``'s colour ramp as a stepped scale under its screen."""
        left, _, right, _ = self._screen_box(panel)
        draw_gradient_bar(draw, left, right, top, stops, _RAMP_BLOCKS)

    def _swatch(
        self,
        draw: ImageDraw.ImageDraw,
        panel: int,
        top: int,
        colour: Tuple[int, int, int],
        label: str,
    ) -> None:
        """Draw one keyed colour chip and its label inside a panel."""
        left, _, _, _ = self._screen_box(panel)
        draw.rectangle([left, top, left + 15, top + 15], fill=colour, outline=COLOR_METAL_DARK)
        draw.text(
            (left + 24, top + 8),
            label,
            font=get_font(12),
            fill=COLOR_TEXT_DIM,
            anchor="lm",
        )

    def _draw_keys(self, canvas: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        """Draw each panel's key: what its colours and its marks mean.

        Each panel is keyed in its own materials rather than in one shared
        legend, because the three panels no longer share a colour ramp and a
        single key would have to claim they did.

        Args:
            canvas: The page, for pasting the robot chip.
            draw: RGBA draw context on ``canvas``.
        """
        font = get_font(11)
        threshold = 0.5

        # Observed map: a diverging ramp, keyed at both ends and the
        # middle, since the middle is where an unmapped cell starts.
        self._gradient_bar(
            draw,
            0,
            602,
            (
                (0.0, COLOR_CHART_FREE),
                (threshold, COLOR_CHART_FOG),
                (1.0, COLOR_CHART_OCCUPIED),
            ),
        )
        left, _, right, _ = self._screen_box(0)
        for position, text, anchor in (
            (0.0, "believed free  p = 0", "la"),
            (0.5, "unknown  p = 0.5", "ma"),
            (1.0, "believed occupied  p = 1", "ra"),
        ):
            draw.text(
                (left + position * (right - left), 623),
                text,
                font=font,
                fill=COLOR_TEXT_DIM,
                anchor=anchor,
            )
        draw.text(
            (left, 648),
            "chance the cell is occupied, from the measured scans",
            font=font,
            fill=COLOR_TEXT_DIM,
            anchor="la",
        )
        draw.line([(left + 2, 676), (left + 30, 676)], fill=COLOR_BEAM + (200,), width=2)
        draw.ellipse([left + 27, 673, left + 33, 679], fill=COLOR_BEAM + (235,))
        draw.text(
            (left + 42, 676),
            "measured beam, and where it stopped",
            font=font,
            fill=COLOR_TEXT_DIM,
            anchor="lm",
        )

        # Map estimate: an ordinary probability ramp plus the blank.
        self._gradient_bar(
            draw,
            1,
            602,
            (
                (0.0, COLOR_HOLO_LOW),
                (threshold, COLOR_HOLO_MID),
                (1.0, COLOR_HOLO_HIGH),
            ),
        )
        left, _, right, _ = self._screen_box(1)
        for position, text, anchor in (
            (0.0, "0", "la"),
            (0.5, "0.5", "ma"),
            (1.0, "1", "ra"),
        ):
            draw.text(
                (left + position * (right - left), 623),
                text,
                font=font,
                fill=COLOR_TEXT_DIM,
                anchor=anchor,
            )
        draw.text(
            (left, 648),
            "weighted occupancy probability over the belief's worlds",
            font=font,
            fill=COLOR_TEXT_DIM,
            anchor="la",
        )
        self._swatch(draw, 1, 668, COLOR_HOLO_BLANK, "no belief recorded")

        # Ground truth: two categories and the robot.
        self._swatch(draw, 2, 602, COLOR_TRUTH_WALL, "wall or obstacle")
        self._swatch(draw, 2, 628, COLOR_TRUTH_FLOOR, "free space")
        left, _, _, _ = self._screen_box(2)
        draw.rectangle([left, 654, left + 15, 669], fill=COLOR_TRUTH_FLOOR)
        sprite = robot_sprite_facing(22, 1)
        canvas.paste(sprite, (left - 3, 651), sprite)
        draw.text(
            (left + 24, 662),
            "robot, pointing along its heading",
            font=get_font(12),
            fill=COLOR_TEXT_DIM,
            anchor="lm",
        )
        draw.line([(left + 2, 690), (left + 30, 690)], fill=COLOR_TRAIL + (200,), width=4)
        draw.text(
            (left + 42, 690),
            "where it has already been",
            font=font,
            fill=COLOR_TEXT_DIM,
            anchor="lm",
        )

    # -- one frame --------------------------------------------------------

    def _render_frame(self, background: Image.Image, frame: Dict[str, Any]) -> Image.Image:
        """Compose one full frame over the cached page.

        Args:
            background: The cached page from :meth:`_build_background`.
            frame: One record from :meth:`_build_frames`.

        Returns:
            One RGB image of :data:`CANVAS_SIZE`.
        """
        canvas = background.copy()
        observed = self._observed_screen(frame["grid"])
        overlay = Image.new("RGBA", (SCREEN_SIZE, SCREEN_SIZE), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay, "RGBA")
        self._draw_beams(overlay_draw, frame)
        self._draw_trail(overlay_draw, frame["trail"])
        observed.paste(overlay, (0, 0), overlay)
        self._paste_robot(observed, frame)
        canvas.paste(observed, self._screen_box(0)[:2])

        canvas.paste(self._belief_screen(frame["belief"]), self._screen_box(1)[:2])

        assert self._truth_cache is not None
        truth = self._truth_cache.copy()
        truth_overlay = Image.new("RGBA", (SCREEN_SIZE, SCREEN_SIZE), (0, 0, 0, 0))
        self._draw_trail(ImageDraw.Draw(truth_overlay, "RGBA"), frame["trail"])
        truth.paste(truth_overlay, (0, 0), truth_overlay)
        self._paste_robot(truth, frame)
        canvas.paste(truth, self._screen_box(2)[:2])

        draw = ImageDraw.Draw(canvas, "RGBA")
        self._draw_status(draw, frame)
        self._draw_caption(draw, frame)
        return canvas

    def _draw_status(self, draw: ImageDraw.ImageDraw, frame: Dict[str, Any]) -> None:
        """Step counter, entropy readout and the entropy meter."""
        initial = float(self.env.initial_entropy_bits)
        threshold = float(self.env.entropy_threshold_bits)
        entropy = float(frame["entropy"])
        draw.text(
            (METER_RIGHT, 22),
            f"STEP {frame['index'] + 1:02d} / {frame['total']:02d}",
            font=get_font(18),
            fill=COLOR_TEXT,
            anchor="ra",
            stroke_width=1,
            stroke_fill=COLOR_TEXT,
        )
        draw.text(
            (METER_RIGHT, 50),
            f"map entropy {entropy:.1f} of {initial:.0f} bits",
            font=get_font(12),
            fill=COLOR_TEXT_DIM,
            anchor="ra",
        )
        draw.rectangle(
            [METER_LEFT, METER_TOP, METER_RIGHT, METER_BOTTOM],
            fill=(20, 28, 34),
            outline=COLOR_METAL_DARK,
        )
        width = METER_RIGHT - METER_LEFT
        filled = int(round(width * float(np.clip(entropy / max(initial, 1e-9), 0.0, 1.0))))
        if filled > 0:
            colour = COLOR_METER_GOAL if entropy <= threshold else COLOR_METER_FILL
            draw.rectangle(
                [METER_LEFT + 1, METER_TOP + 2, METER_LEFT + filled, METER_BOTTOM - 2],
                fill=colour,
            )
        mark = METER_LEFT + int(round(width * float(np.clip(threshold / max(initial, 1e-9), 0, 1))))
        draw.line(
            [(mark, METER_TOP - 3), (mark, METER_BOTTOM + 3)],
            fill=COLOR_METER_GOAL,
            width=2,
        )
        # The label is centred on the goal mark, which sits wherever the
        # threshold falls, so it is clamped to keep it inside the canvas when
        # the threshold is a large fraction of the initial entropy.
        label_x = int(np.clip(mark, METER_LEFT + 44, CANVAS_SIZE[0] - 90))
        draw.text(
            (label_x, METER_BOTTOM + 6),
            f"Target: {threshold:.0f} bits",
            font=get_font(11),
            fill=COLOR_METER_GOAL,
            anchor="ma",
        )

    def _draw_caption(self, draw: ImageDraw.ImageDraw, frame: Dict[str, Any]) -> None:
        """The two-line story of the step, plus the note on what a panel shows."""
        threshold = float(self.env.entropy_threshold_bits)
        step_text = f"Step {frame['index'] + 1} of {frame['total']}"
        heading_text = HEADING_LABELS[frame["heading"]]
        if frame["action"] is None:
            headline = f"{step_text} - episode over, no further action"
            detail = (
                f"Final map entropy {frame['entropy']:.2f} bits; "
                f"the map counts as resolved at {threshold:.2f} bits or below"
            )
            note = "Final recorded state"
        else:
            action_text = _ACTION_LABELS[int(frame["action"])]
            reward_text = "-" if frame["reward"] is None else f"{float(frame['reward']):+.2f}"
            headline = (
                f"{step_text} - robot at row {frame['row']}, column {frame['col']}, "
                f"facing {heading_text}; about to {action_text}"
            )
            detail = (
                f"Observed map entropy reduction: {reward_text} bits.   "
                f"Map entropy before it: {frame['entropy']:.2f} bits "
                f"(resolved at {threshold:.2f} or below)"
            )
            note = "Panels show the state before this action"
        centre = CANVAS_SIZE[0] // 2
        draw.text((centre, 738), headline, font=get_font(14), fill=COLOR_TEXT, anchor="ma")
        draw.text((centre, 764), detail, font=get_font(12), fill=COLOR_TEXT_DIM, anchor="ma")
        draw.text((centre, 788), note, font=get_font(11), fill=COLOR_TEXT_DIM, anchor="ma")

    # -- saving -----------------------------------------------------------

    @staticmethod
    def _build_palette(reference: Image.Image) -> Image.Image:
        """One GIF palette for the whole animation.

        Median cut on its own weights a colour by how many pixels carry it, so
        the large dark panels would take nearly every slot and the beams, the
        trail and the robot would quantize to mud. The adaptive palette gets the
        bulk of the slots and the accent colours are appended by hand.

        Args:
            reference: A representative frame.

        Returns:
            A one-pixel ``P`` image carrying the palette.
        """
        sample = reference.resize(
            (max(1, reference.width // 2), max(1, reference.height // 2)),
            Image.Resampling.NEAREST,
        )
        adaptive = sample.quantize(
            colors=256 - len(_ACCENT_COLORS), method=Image.Quantize.MEDIANCUT
        )
        entries = list(adaptive.getpalette() or [])
        entries = entries[: 3 * (256 - len(_ACCENT_COLORS))]
        for colour in _ACCENT_COLORS:
            entries.extend(colour)
        entries.extend([0] * (768 - len(entries)))
        master = Image.new("P", (1, 1))
        master.putpalette(entries)
        return master

    def _save(self, images: List[Image.Image], cache_path: Path) -> None:
        """Quantize every frame to one shared palette and write the GIF."""
        master = self._build_palette(images[-1])
        indexed = [image.quantize(palette=master, dither=Image.Dither.NONE) for image in images]
        durations = [FRAME_MS] * (len(indexed) - 1) + [FINAL_FRAME_MS]
        indexed[0].save(
            cache_path,
            save_all=True,
            append_images=indexed[1:],
            duration=durations,
            loop=0,
            optimize=True,
        )
