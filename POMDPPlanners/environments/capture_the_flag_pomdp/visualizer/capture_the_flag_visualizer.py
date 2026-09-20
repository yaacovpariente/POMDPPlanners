# SPDX-License-Identifier: MIT

"""Isometric episode renderer for the CaptureTheFlag POMDP.

Produces an animated GIF with a real-time-strategy framing: an isometric
camera, soldier sprites that run between cells rather than teleporting, a
scoreboard, and a minimap. Every frame is a function of the recorded history
and of seeded art, so rendering the same episode twice gives the same bytes --
the golden-file test depends on that.

The belief is drawn as its own overlay layer, never baked into the world art,
because a picture of the true state alone is an MDP picture of a POMDP: it
cannot tell a planner that handled uncertainty from one that got lucky. Red
positions and the red flag cell are the two hidden variables, so those are what
the overlay shows.

Classes:
    CaptureTheFlagVisualizer: Renders an episode history to a GIF.
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.capture_the_flag_pomdp.visualizer.capture_the_flag_assets import (
    BLUE,
    RED,
    TILE_HEIGHT,
    TILE_WIDTH,
    WOOD,
    banner_sprite,
    contact_shadow,
    ground_plane,
    iso,
    keep_sprite,
    panel,
    river_axis,
    soldier_sprite,
    tree_sprite,
)
from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_pomdp_utils import (
    ACTION_SCAN,
    decode_joint_action,
)

_VIEW_WIDTH = 640
_TOP_BAR = 26
_FIELD_HEIGHT = 384
_PANEL_HEIGHT = 120
_VIEW_HEIGHT = _TOP_BAR + _FIELD_HEIGHT + _PANEL_HEIGHT
_UPSCALE = 1
_SUBSTEPS = 3
_GIF_COLOURS = 128
_FRAME_MS = 140


class CaptureTheFlagVisualizer:
    """Render CaptureTheFlag episodes as isometric RTS-style GIFs.

    Attributes:
        env: The environment whose field and state layout is being drawn.
    """

    def __init__(self, env: Any):
        """Initialize the visualizer.

        Args:
            env: The :class:`CaptureTheFlagPOMDP` the history came from.
        """
        self.env = env
        self.layout = env.layout
        width, height = env.grid_size
        # Centre the camera on the middle of the field so the terrain, which is
        # generated across the whole viewport, reaches every edge.
        centre = iso(width / 2.0, height / 2.0, (0, 0))
        self.origin = (
            _VIEW_WIDTH // 2 - centre[0],
            _TOP_BAR + _FIELD_HEIGHT // 2 - centre[1],
        )
        self.paths = self._worn_paths()
        self.terrain = ground_plane(
            (_VIEW_WIDTH, _VIEW_HEIGHT),
            _TOP_BAR,
            self.origin,
            env.midline,
            self.paths,
            env.trees,
        )
        self.font = ImageFont.load_default()

    def _worn_paths(self) -> List[Tuple[float, float, float, float]]:
        """Return the trails worn between the bases and the crossing."""
        env = self.env
        crossing = float(river_axis(np.array(float(env.blue_base[1])), env.midline, env.trees))
        return [
            (
                float(env.blue_base[0]),
                float(env.blue_base[1]),
                crossing - 0.6,
                float(env.blue_base[1]),
            ),
            (
                crossing + 0.6,
                float(env.blue_base[1]),
                float(env.red_base[0]),
                float(env.red_base[1]),
            ),
        ]

    # --------------------------------------------------------------- belief

    def _belief_marginals(
        self, belief: Any
    ) -> Tuple[List[Dict[Tuple[int, int], float]], Dict[Tuple[int, int], float]]:
        """Reduce a particle belief to the marginals over the hidden variables.

        Args:
            belief: The belief recorded on a step, or ``None``.

        Returns:
            One cell-to-weight marginal per red player, and the marginal over
            the red flag's candidate cells. Both empty when no belief was
            recorded.
        """
        empty: List[Dict[Tuple[int, int], float]] = [{} for _ in range(self.env.n_red)]
        particles = getattr(belief, "particles", None)
        if particles is None or len(particles) == 0:
            return empty, {}
        weights = getattr(belief, "weights", None)
        if weights is None:
            weights = np.ones(len(particles)) / len(particles)
        weights = np.asarray(weights, dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            return empty, {}
        weights = weights / total

        red_marginals: List[Dict[Tuple[int, int], float]] = [{} for _ in range(self.env.n_red)]
        flag_marginal: Dict[Tuple[int, int], float] = {}
        for particle, weight in zip(particles, weights):
            state = np.asarray(particle, dtype=np.float64)
            if state.shape != (self.layout.size,):
                continue
            for index, cell in enumerate(self.layout.red_cells(state)):
                red_marginals[index][cell] = red_marginals[index].get(cell, 0.0) + float(weight)
            flag_cell = self.env.red_flag_cell(state)
            flag_marginal[flag_cell] = flag_marginal.get(flag_cell, 0.0) + float(weight)
        return red_marginals, flag_marginal

    def _draw_belief(
        self,
        canvas: Image.Image,
        red_marginals: Sequence[Dict[Tuple[int, int], float]],
        flag_marginal: Dict[Tuple[int, int], float],
    ) -> None:
        """Overlay the belief on top of the world, as its own layer.

        Particles are drawn as weighted markers rather than summarised by an
        ellipse: the belief over a hidden pursuer goes multi-modal routinely,
        and an ellipse would assert a Gaussian shape that is exactly what the
        picture exists to let a reader check.

        Args:
            canvas: The frame being composed.
            red_marginals: Per-red-player cell marginals.
            flag_marginal: Marginal over the red flag's candidate cells.
        """
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        hues = [RED[2], (176, 92, 200), (232, 148, 72), (120, 200, 220)]
        for index, marginal in enumerate(red_marginals):
            if not marginal:
                continue
            peak = max(marginal.values())
            colour = hues[index % len(hues)]
            for cell, weight in sorted(marginal.items()):
                share = weight / peak
                radius = 5 + 13 * share
                centre = iso(cell[0], cell[1], self.origin)
                offset = -7 + 14 * (index / max(1, len(red_marginals) - 1)) if len(
                    red_marginals
                ) > 1 else 0
                draw.ellipse(
                    [
                        centre[0] + offset - radius,
                        centre[1] - radius * 0.55,
                        centre[0] + offset + radius,
                        centre[1] + radius * 0.55,
                    ],
                    outline=(250, 250, 250, int(90 + 120 * share)),
                    fill=colour + (int(110 + 145 * share),),
                )
        if flag_marginal:
            peak = max(flag_marginal.values())
            for cell, weight in sorted(flag_marginal.items()):
                share = weight / peak
                centre = iso(cell[0], cell[1], self.origin)
                draw.polygon(
                    [
                        (centre[0], centre[1] - TILE_HEIGHT // 2),
                        (centre[0] + TILE_WIDTH // 2, centre[1]),
                        (centre[0], centre[1] + TILE_HEIGHT // 2),
                        (centre[0] - TILE_WIDTH // 2, centre[1]),
                    ],
                    fill=(236, 188, 92, int(26 + 70 * share)),
                    outline=(244, 212, 120, 200),
                )
                draw.text(
                    (centre[0] - 8, centre[1] - 6),
                    f"{weight:.2f}",
                    font=self.font,
                    fill=(250, 232, 170, 255),
                )
        canvas.alpha_composite(overlay)

    # ---------------------------------------------------------------- world

    # pylint: disable-next=too-many-locals
    def _draw_world(
        self,
        canvas: Image.Image,
        state: np.ndarray,
        blue_cells: Sequence[Tuple[float, float]],
        red_cells: Sequence[Tuple[float, float]],
        stride_index: int,
        scanning: Sequence[bool],
    ) -> None:
        """Paint terrain props and both teams, sorted back to front.

        Args:
            canvas: The frame being composed.
            state: The state being drawn.
            blue_cells: Blue positions, fractional mid-stride.
            red_cells: Red positions, fractional mid-stride.
            stride_index: Frame index within the run cycle.
            scanning: Whether each blue player is scanning this step.
        """
        env, layout = self.env, self.layout
        carrier_red_flag = int(state[layout.carrier_red_flag])
        carrier_blue_flag = int(state[layout.carrier_blue_flag])

        self._draw_bridge(canvas)

        drawables: List[Tuple[float, str, Any]] = []
        for cell in env.trees:
            drawables.append((cell[0] + cell[1], "tree", cell))
        drawables.append((env.blue_base[0] + env.blue_base[1] - 0.2, "keep_blue", env.blue_base))
        drawables.append((env.red_base[0] + env.red_base[1] - 0.2, "keep_red", env.red_base))
        if carrier_red_flag == 0:
            flag_cell = env.red_flag_cell(state)
            drawables.append((flag_cell[0] + flag_cell[1] - 0.1, "banner_red", flag_cell))
        if carrier_blue_flag == 0:
            cell = env.blue_flag_cell
            drawables.append((cell[0] + cell[1] - 0.1, "banner_blue", cell))
        for index, cell in enumerate(blue_cells):
            drawables.append((cell[0] + cell[1], "blue", index))
        for index, cell in enumerate(red_cells):
            drawables.append((cell[0] + cell[1], "red", index))

        for _, kind, payload in sorted(drawables, key=lambda item: item[0]):
            self._draw_one(
                canvas,
                kind,
                payload,
                state,
                blue_cells,
                red_cells,
                stride_index,
                scanning,
            )

    # pylint: disable-next=too-many-arguments,too-many-locals
    def _draw_one(
        self,
        canvas: Image.Image,
        kind: str,
        payload: Any,
        state: np.ndarray,
        blue_cells: Sequence[Tuple[float, float]],
        red_cells: Sequence[Tuple[float, float]],
        stride_index: int,
        scanning: Sequence[bool],
    ) -> None:
        """Paint one depth-sorted drawable."""
        layout = self.layout
        if kind == "tree":
            centre = iso(payload[0], payload[1], self.origin)
            contact_shadow(canvas, (centre[0] + 12, centre[1] + 4), (20, 7), 95, skew=8)
            sprite = tree_sprite(payload[0] * 31 + payload[1] * 17, payload[0] > self.env.midline)
            self._blit(canvas, sprite, centre, lift=6)
        elif kind.startswith("keep"):
            centre = iso(payload[0], payload[1], self.origin)
            contact_shadow(canvas, (centre[0] + 26, centre[1] + 6), (40, 14), 100, skew=14)
            self._blit(canvas, keep_sprite(kind.endswith("blue")), centre, lift=16)
        elif kind.startswith("banner"):
            centre = iso(payload[0], payload[1], self.origin)
            contact_shadow(canvas, (centre[0] + 7, centre[1] + 3), (9, 4), 80)
            self._blit(canvas, banner_sprite(kind.endswith("blue")), centre, lift=4)
        else:
            team_blue = kind == "blue"
            index = int(payload)
            cell = blue_cells[index] if team_blue else red_cells[index]
            centre = iso(cell[0], cell[1], self.origin)
            frozen = (
                int(state[(layout.freeze_blue if team_blue else layout.freeze_red) + index]) > 0
            )
            previous = (
                blue_cells[index] if team_blue else red_cells[index]
            )
            facing = 1 if previous[0] >= cell[0] else -1
            carrying_red = team_blue and int(state[layout.carrier_red_flag]) == index + 1
            carrying_blue = (not team_blue) and int(state[layout.carrier_blue_flag]) == index + 1
            contact_shadow(canvas, (centre[0] + 7, centre[1] + 4), (13, 5), 110)
            if team_blue and scanning[index]:
                self._draw_scan(canvas, centre)
            sprite = soldier_sprite(
                team_blue,
                facing,
                stride_index if not frozen else 0,
                carrying_blue,
                carrying_red,
                frozen,
                team_blue,
            )
            self._blit(canvas, sprite, centre, lift=10)

    def _blit(self, canvas: Image.Image, sprite: Image.Image, centre: Tuple[int, int], lift: int) -> None:
        """Composite a sprite so its feet sit on a cell centre."""
        canvas.alpha_composite(
            sprite, (centre[0] - sprite.width // 2, centre[1] - sprite.height + lift)
        )

    def _draw_bridge(self, canvas: Image.Image) -> None:
        """Draw the log crossing, centred on the channel it spans."""
        env = self.env
        row = float(env.blue_base[1])
        axis = float(river_axis(np.array(row), env.midline, env.trees))
        start = iso(axis - 1.05, row, self.origin)
        end = iso(axis + 1.05, row, self.origin)
        draw = ImageDraw.Draw(canvas)
        contact_shadow(
            canvas, ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 + 9), (44, 9), 90
        )
        for step in range(-2, 3):
            t = (step + 2) / 4
            px = int(start[0] + (end[0] - start[0]) * t)
            py = int(start[1] + (end[1] - start[1]) * t)
            draw.rectangle([px - 3, py + 2, px + 3, py + 14], fill=WOOD[0])
        for step in range(20):
            t = step / 19
            px = int(start[0] + (end[0] - start[0]) * t)
            py = int(start[1] + (end[1] - start[1]) * t)
            draw.line([(px, py - 11), (px, py + 4)], fill=WOOD[2] if step % 2 else WOOD[1], width=5)
        draw.line([start, end], fill=WOOD[3], width=2)
        for landing in (start, end):
            draw.ellipse(
                [landing[0] - 16, landing[1] - 6, landing[0] + 16, landing[1] + 10],
                fill=(118, 96, 62),
            )

    def _draw_scan(self, canvas: Image.Image, centre: Tuple[int, int]) -> None:
        """Draw the expanding ping that marks a scan action."""
        for radius, alpha in ((22, 200), (38, 140), (56, 86), (76, 44)):
            ring = Image.new("RGBA", (radius * 2 + 2, radius + 2), (0, 0, 0, 0))
            ImageDraw.Draw(ring).ellipse(
                [0, 0, radius * 2, radius], outline=(186, 234, 250, alpha)
            )
            canvas.alpha_composite(ring, (centre[0] - radius, centre[1] - radius // 2))

    # ------------------------------------------------------------------ hud

    # pylint: disable-next=too-many-locals
    def _draw_hud(
        self,
        canvas: Image.Image,
        state: np.ndarray,
        step_index: int,
        total_steps: int,
        red_marginals: Sequence[Dict[Tuple[int, int], float]],
    ) -> None:
        """Draw the scoreboard, the minimap and the status readout."""
        env, layout = self.env, self.layout
        draw = ImageDraw.Draw(canvas)
        panel(draw, (0, 0, _VIEW_WIDTH - 1, _TOP_BAR - 1), (26, 26, 33))
        draw.rectangle([8, 7, 18, 17], fill=BLUE[1])
        draw.rectangle([9, 8, 17, 12], fill=BLUE[2])
        draw.text((24, 8), str(int(state[layout.score_blue])), font=self.font, fill=(240, 238, 230))
        draw.text((36, 8), "-", font=self.font, fill=(120, 120, 132))
        draw.rectangle([48, 7, 58, 17], fill=RED[1])
        draw.rectangle([49, 8, 57, 12], fill=RED[2])
        draw.text((64, 8), str(int(state[layout.score_red])), font=self.font, fill=(240, 238, 230))
        draw.text(
            (100, 8), f"TURN {step_index:03d} / {total_steps:03d}", font=self.font, fill=(176, 176, 190)
        )
        blue_taken = int(state[layout.carrier_blue_flag]) != 0
        red_taken = int(state[layout.carrier_red_flag]) != 0
        draw.text(
            (220, 8),
            f"OWN BANNER: {'STOLEN' if blue_taken else 'HOME'}",
            font=self.font,
            fill=(234, 176, 88) if blue_taken else (176, 176, 190),
        )
        draw.text(
            (400, 8),
            f"ENEMY BANNER: {'CARRIED' if red_taken else 'PLANTED'}",
            font=self.font,
            fill=(150, 232, 140) if red_taken else (176, 176, 190),
        )

        base_y = _TOP_BAR + _FIELD_HEIGHT
        panel(draw, (0, base_y, _VIEW_WIDTH - 1, _VIEW_HEIGHT - 1), (24, 24, 31))
        map_box = (10, base_y + 9, 126, base_y + 109)
        panel(draw, map_box, (18, 20, 18))
        mini = self.terrain.resize((map_box[2] - map_box[0] - 7, map_box[3] - map_box[1] - 7))
        canvas.paste(mini, (map_box[0] + 4, map_box[1] + 4))
        for cell, colour in [
            (cell, BLUE[3]) for cell in layout.blue_cells(state)
        ] + [(cell, RED[3]) for cell in layout.red_cells(state)]:
            point = iso(cell[0], cell[1], self.origin)
            mx = map_box[0] + 4 + int(point[0] / _VIEW_WIDTH * (map_box[2] - map_box[0] - 7))
            my = map_box[1] + 4 + int(
                (point[1] - _TOP_BAR) / _FIELD_HEIGHT * (map_box[3] - map_box[1] - 7)
            )
            draw.rectangle([mx - 2, my - 2, mx + 2, my + 2], fill=colour)

        text_x = map_box[2] + 14
        draw.text((text_x, base_y + 12), "BLUE SQUAD", font=self.font, fill=(240, 238, 230))
        for index in range(env.n_blue):
            freeze = int(state[layout.freeze_blue + index])
            cell = layout.blue_cells(state)[index]
            status = f"FROZEN {freeze}" if freeze else "READY"
            carrying = " + BANNER" if int(state[layout.carrier_red_flag]) == index + 1 else ""
            draw.text(
                (text_x, base_y + 30 + index * 14),
                f"P{index + 1}  {cell}  {status}{carrying}",
                font=self.font,
                fill=(146, 208, 130) if not freeze else (216, 170, 60),
            )
        spread = sum(len(marginal) for marginal in red_marginals)
        draw.text(
            (text_x, base_y + 34 + env.n_blue * 14),
            f"BELIEF: {spread} CELLS HOLD RED MASS",
            font=self.font,
            fill=(226, 138, 122),
        )
        draw.text(
            (text_x, base_y + 48 + env.n_blue * 14),
            "OVERLAY: DOTS = RED MARGINALS, DIAMONDS = BANNER",
            font=self.font,
            fill=(150, 148, 162),
        )

    # -------------------------------------------------------------- episode

    def _interpolate(
        self, previous: Sequence[Tuple[int, int]], current: Sequence[Tuple[int, int]], t: float
    ) -> List[Tuple[float, float]]:
        """Blend two cell lists so units run rather than teleport.

        A respawn is not interpolated: a tagged player is carried home across
        the map, and tweening that would read as a unit sprinting through the
        enemy line.

        Args:
            previous: Cells at the start of the step.
            current: Cells at the end of the step.
            t: Fraction through the step.

        Returns:
            Fractional cells for this sub-frame.
        """
        blended: List[Tuple[float, float]] = []
        for start, end in zip(previous, current):
            if abs(start[0] - end[0]) + abs(start[1] - end[1]) > 1:
                blended.append((float(end[0]), float(end[1])))
            else:
                blended.append(
                    (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
                )
        return blended

    def render_episode(self, history: Sequence[StepData], output_path: Path) -> None:
        """Render an episode history to an animated GIF.

        Args:
            history: The recorded steps.
            output_path: File the ``.gif`` is written to.

        Raises:
            ValueError: If the history holds no usable state.
        """
        states = [
            np.asarray(step.state, dtype=np.float64)
            for step in history
            if step.state is not None
        ]
        if not states:
            raise ValueError("cannot render an episode with no recorded state")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frames: List[Image.Image] = []
        for step_index, step in enumerate(history):
            if step.state is None:
                continue
            state = np.asarray(step.state, dtype=np.float64)
            next_state = (
                np.asarray(step.next_state, dtype=np.float64)
                if step.next_state is not None
                else state
            )
            scanning = self._scanning_flags(step.action)
            red_marginals, flag_marginal = self._belief_marginals(step.belief)
            for substep in range(_SUBSTEPS):
                t = substep / _SUBSTEPS
                frames.append(
                    self._compose(
                        state,
                        next_state,
                        t,
                        substep,
                        scanning,
                        red_marginals,
                        flag_marginal,
                        step_index,
                        len(history),
                    )
                )

        # One shared adaptive palette, derived from a single frame: a
        # per-frame palette would be both larger and a second thing that has to
        # stay byte-stable for the golden test.
        palette = frames[len(frames) // 2].quantize(colors=_GIF_COLOURS, method=Image.Quantize.MEDIANCUT)
        quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        quantized[0].save(
            output_path,
            save_all=True,
            append_images=quantized[1:],
            duration=_FRAME_MS,
            loop=0,
            optimize=True,
        )

    def _scanning_flags(self, action: Any) -> List[bool]:
        """Return, per blue player, whether it scanned on this step."""
        if action is None:
            return [False] * self.env.n_blue
        player_actions = decode_joint_action(int(action), self.env.n_blue)
        return [player_action == ACTION_SCAN for player_action in player_actions]

    # pylint: disable-next=too-many-arguments
    def _compose(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        t: float,
        substep: int,
        scanning: Sequence[bool],
        red_marginals: Sequence[Dict[Tuple[int, int], float]],
        flag_marginal: Dict[Tuple[int, int], float],
        step_index: int,
        total_steps: int,
    ) -> Image.Image:
        """Compose one frame."""
        canvas = Image.new("RGBA", (_VIEW_WIDTH, _VIEW_HEIGHT), (14, 13, 17, 255))
        canvas.paste(self.terrain, (0, _TOP_BAR))
        blue_cells = self._interpolate(
            self.layout.blue_cells(state), self.layout.blue_cells(next_state), t
        )
        red_cells = self._interpolate(
            self.layout.red_cells(state), self.layout.red_cells(next_state), t
        )
        self._draw_belief(canvas, red_marginals, flag_marginal)
        self._draw_world(canvas, state, blue_cells, red_cells, substep * 2, scanning)
        self._draw_hud(canvas, state, step_index, total_steps, red_marginals)
        return canvas.convert("RGB").resize(
            (_VIEW_WIDTH * _UPSCALE, _VIEW_HEIGHT * _UPSCALE), Image.Resampling.NEAREST
        )
