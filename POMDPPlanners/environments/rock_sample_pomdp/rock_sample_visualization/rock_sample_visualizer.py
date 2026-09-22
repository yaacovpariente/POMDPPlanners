# SPDX-License-Identifier: MIT

"""Fast, deterministic RockSample episode rendering with Pillow."""

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_assets import sprite, terrain

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
    get_robot_pos,
    get_rocks,
)

if TYPE_CHECKING:
    from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import (
        RockSamplePOMDP,
        RockSampleState,
    )


CANVAS_SIZE = (1000, 800)
PLOT_LEFT = 76
PLOT_TOP = 111
PLOT_MAX_RIGHT = 771
PLOT_MAX_BOTTOM = 668
LEGEND_LEFT = 811
GIF_DURATION_MS = 1000

COLOR_PAGE = (23, 25, 27)
COLOR_GRID = (210, 154, 102)
COLOR_BORDER = (96, 84, 65)
COLOR_TEXT = (237, 229, 214)
COLOR_GOOD_ROCK = (68, 160, 68)
COLOR_BAD_ROCK = (248, 72, 72)
COLOR_DANGER = (232, 55, 40)
COLOR_EXIT = (244, 196, 40)
COLOR_ROBOT = (0, 35, 235)
COLOR_PATH = (108, 108, 255)
COLOR_ARROW = (238, 30, 30)
COLOR_SENSOR = (232, 134, 20)
COLOR_INFO = (35, 39, 42)
COLOR_SUCCESS = (144, 238, 144)
COLOR_SUCCESS_EDGE = (0, 112, 32)
COLOR_FAILURE = (240, 128, 128)
COLOR_FAILURE_EDGE = (160, 0, 0)

ACCENT_COLORS: Tuple[Tuple[int, int, int], ...] = (
    COLOR_PAGE,
    COLOR_GRID,
    COLOR_BORDER,
    COLOR_TEXT,
    COLOR_GOOD_ROCK,
    COLOR_BAD_ROCK,
    COLOR_DANGER,
    COLOR_EXIT,
    COLOR_ROBOT,
    COLOR_PATH,
    COLOR_ARROW,
    COLOR_SENSOR,
    COLOR_INFO,
    COLOR_SUCCESS,
    COLOR_SUCCESS_EDGE,
    COLOR_FAILURE,
    COLOR_FAILURE_EDGE,
)


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


class RockSampleVisualizer:
    """Render RockSample paths without rebuilding the fixed scene per frame."""

    def __init__(self, env: "RockSamplePOMDP"):
        self.env = env
        self.map_size = env.map_size
        self.rock_positions = env.rock_positions
        self.action_names = env.action_names
        self.action_to_vector = env.action_to_vector
        self.dangerous_areas = env.dangerous_areas
        self.dangerous_area_radius = env.dangerous_area_radius
        self._background: Optional[Image.Image] = None
        self._background_key: Optional[Tuple[object, ...]] = None

    def create_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Save one GIF frame for every state in an episode history."""
        self._validate_visualization_inputs(history, cache_path)
        path, actions = self._extract_path_and_actions(history)
        self.visualize_path(path, actions, cache_path)

    def visualize_path(
        self, path: List["RockSampleState"], actions: List[int], cache_path: Path
    ) -> None:
        """Save a path as a 1-frame-per-second animated GIF."""
        self._validate_path_cache_inputs(cache_path)
        if not path:
            raise ValueError("Cannot visualize empty path")
        frames = self.render_frames(path, actions)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        master = self._build_palette(frames[0])
        indexed = [frame.quantize(palette=master, dither=Image.Dither.NONE) for frame in frames]
        indexed[0].save(
            cache_path,
            save_all=True,
            append_images=indexed[1:],
            duration=GIF_DURATION_MS,
            loop=0,
            optimize=False,
            disposal=2,
        )

    def render_frames(
        self, path: Sequence["RockSampleState"], actions: Sequence[int]
    ) -> List[Image.Image]:
        """Return RGB frames, copying one cached static background per state."""
        key = self._scene_key()
        if self._background is None or self._background_key != key:
            self._background = self._build_static_background()
            self._background_key = key

        frames: List[Image.Image] = []
        valid_path: List[Tuple[int, int]] = []
        for frame_index, state in enumerate(path):
            canvas = self._background.copy()
            draw = ImageDraw.Draw(canvas)
            robot_pos = get_robot_pos(state)
            if robot_pos != (-1, -1):
                valid_path.append(robot_pos)
            self._draw_path(draw, valid_path)
            self._draw_rocks(canvas, state)
            action = actions[frame_index] if frame_index < len(actions) else None
            if robot_pos != (-1, -1):
                self._draw_robot(canvas, robot_pos)
                self._draw_action(draw, robot_pos, action)
            self._draw_rock_badges(draw, state)
            self._draw_status(draw, frame_index, len(path), state, action)
            frames.append(canvas)
        return frames

    def _validate_visualization_inputs(self, history: List[StepData], cache_path: Path) -> None:
        if not isinstance(history, List):
            raise TypeError("history must be a List object")
        if not history:
            raise ValueError("Cannot visualize empty history")
        for step in history:
            if not isinstance(step, StepData):
                raise TypeError("history must be a List of StepData objects")
        self._validate_path_cache_inputs(cache_path)

    @staticmethod
    def _extract_path_and_actions(
        history: List[StepData],
    ) -> Tuple[List["RockSampleState"], List[int]]:
        path = [step.state for step in history]
        actions = [step.action for step in history[:-1]]
        return path, actions

    @staticmethod
    def _validate_path_cache_inputs(cache_path: Path) -> None:
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not str(cache_path).endswith(".gif"):
            raise ValueError("cache_path must end with .gif")

    def _scene_key(self) -> Tuple[object, ...]:
        return (
            tuple(self.map_size),
            tuple(self.rock_positions),
            tuple(self.dangerous_areas),
            float(self.dangerous_area_radius),
        )

    @property
    def _plot_bounds(self) -> Tuple[int, int, int, int]:
        rows, cols = self.map_size
        unit = min((PLOT_MAX_RIGHT - PLOT_LEFT) / cols, (PLOT_MAX_BOTTOM - PLOT_TOP) / rows)
        return PLOT_LEFT, PLOT_TOP, round(PLOT_LEFT + cols * unit), round(PLOT_TOP + rows * unit)

    def _cell_center(self, row: float, col: float) -> Tuple[float, float]:
        left, top, right, bottom = self._plot_bounds
        rows, cols = self.map_size
        x = left + (col + 0.5) / cols * (right - left)
        y = top + (row + 0.5) / rows * (bottom - top)
        return x, y

    @property
    def _pixels_per_cell(self) -> float:
        left, top, right, bottom = self._plot_bounds
        rows, cols = self.map_size
        return min((right - left) / cols, (bottom - top) / rows)

    def _build_static_background(self) -> Image.Image:
        canvas = Image.new("RGB", CANVAS_SIZE, COLOR_PAGE)
        draw = ImageDraw.Draw(canvas)
        left, top, right, bottom = self._plot_bounds

        draw.text(
            (left, 40),
            "RockSample",
            fill=COLOR_TEXT,
            font=_font(30),
            anchor="lm",
        )
        draw.text(
            (left, 68), "MINERAL SURVEY  /  WORLD STATE", fill=(161, 168, 173), font=_font(12)
        )
        draw.rounded_rectangle(
            (left - 7, top - 7, right + 7, bottom + 7),
            radius=4,
            fill=(12, 14, 15),
            outline=(100, 90, 77),
            width=2,
        )
        canvas.paste(terrain(right - left, bottom - top), (left, top))
        self._draw_danger_areas(canvas)
        for row in range(self.map_size[0]):
            _, y = self._cell_center(row, 0)
            edge = top + row * self._pixels_per_cell
            draw.line([(left, edge), (right, edge)], fill=COLOR_GRID, width=1)
            draw.text((left - 13, y), str(row), fill=COLOR_TEXT, font=_font(14), anchor="rm")
        for col in range(self.map_size[1]):
            x, _ = self._cell_center(0, col)
            edge = left + col * self._pixels_per_cell
            draw.line([(edge, top), (edge, bottom)], fill=COLOR_GRID, width=1)
            draw.text((x, bottom + 10), str(col), fill=COLOR_TEXT, font=_font(14), anchor="ma")
        draw.rectangle((left, top, right, bottom), outline=COLOR_BORDER, width=1)
        draw.line([(right, top), (right, bottom)], fill=COLOR_EXIT, width=3)
        draw.text(
            (right + 14, bottom + 28), "EAST EXIT", fill=COLOR_EXIT, font=_font(12), anchor="rm"
        )
        draw.text(
            ((left + right) / 2, bottom + 36),
            "Column",
            fill=COLOR_TEXT,
            font=_font(15),
            anchor="ma",
        )
        draw.text((18, (top + bottom) / 2), "Row", fill=COLOR_TEXT, font=_font(15), anchor="lm")
        self._draw_legend(draw)
        rover = sprite("rover", 27)
        canvas.paste(rover, (LEGEND_LEFT + 8, PLOT_TOP + 170), rover)
        return canvas

    def _draw_danger_areas(self, canvas: Image.Image) -> None:
        # Draw on the plot crop so edge hazards cannot cover axis labels.
        left, top, right, bottom = self._plot_bounds
        plot = canvas.crop((left, top, right, bottom))
        draw = ImageDraw.Draw(plot, "RGBA")
        radius = float(self.dangerous_area_radius) * self._pixels_per_cell
        for row, col in self.dangerous_areas:
            x, y = self._cell_center(row, col)
            x, y = x - left, y - top
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(*COLOR_DANGER, 95),
                outline=(*COLOR_DANGER, 240),
                width=2,
            )
        canvas.paste(plot, (left, top))

    def _draw_legend(self, draw: ImageDraw.ImageDraw) -> None:
        x = LEGEND_LEFT
        y = PLOT_TOP + 54
        draw.rounded_rectangle(
            (x - 15, PLOT_TOP - 5, 984, PLOT_TOP + 276),
            radius=9,
            fill=(30, 34, 37),
            outline=(81, 84, 83),
            width=1,
        )
        draw.text((x, PLOT_TOP + 18), "WORLD STATE", fill=(167, 177, 183), font=_font(14))
        entries = [
            ("Good Rock", COLOR_GOOD_ROCK),
            ("Bad Rock", COLOR_BAD_ROCK),
            ("Dangerous Areas", COLOR_DANGER),
            ("Exit", COLOR_EXIT),
            ("Robot", COLOR_ROBOT),
            ("Path", COLOR_PATH),
            ("Sensor Check", COLOR_SENSOR),
        ]
        for label, color in entries:
            if label == "Path":
                draw.line([(x + 4, y + 9), (x + 34, y + 9)], fill=color, width=3)
            elif label == "Exit":
                draw.line([(x + 4, y), (x + 34, y)], fill=color, width=5)
            elif label == "Robot":
                pass  # The sprite is pasted after the legend panel is drawn.
            else:
                draw.rectangle((x + 4, y, x + 34, y + 18), fill=color)
            draw.text((x + 44, y + 9), label, fill=COLOR_TEXT, font=_font(14), anchor="lm")
            y += 30
        draw.text((x, PLOT_TOP + 300), "R number = rock ID", fill=(172, 175, 172), font=_font(12))
        draw.text(
            (x, PLOT_TOP + 320), "Border = true quality", fill=(172, 175, 172), font=_font(12)
        )

    def _draw_path(self, draw: ImageDraw.ImageDraw, path: Sequence[Tuple[int, int]]) -> None:
        points = [self._cell_center(row, col) for row, col in path]
        if len(points) > 1:
            draw.line(points, fill=COLOR_PATH, width=3, joint="curve")

    def _draw_rocks(self, canvas: Image.Image, state: "RockSampleState") -> None:
        draw = ImageDraw.Draw(canvas)
        rocks = get_rocks(state)
        size = max(8, round(self._pixels_per_cell * 0.53))
        for index, (row, col) in enumerate(self.rock_positions):
            if index >= len(rocks):
                continue
            x, y = self._cell_center(row, col)
            art = sprite(("ore-blue", "ore-green", "ore-red")[index % 3], size)
            canvas.paste(art, (round(x - art.width / 2), round(y - art.height / 2)), art)

    def _draw_rock_badges(self, draw: ImageDraw.ImageDraw, state: "RockSampleState") -> None:
        cell = self._pixels_per_cell
        height = min(17, cell * 0.18)
        half_width = min(17, cell * 0.23)
        for index, ((row, col), good) in enumerate(zip(self.rock_positions, get_rocks(state))):
            x, y = self._cell_center(row, col)
            color = COLOR_GOOD_ROCK if good else COLOR_BAD_ROCK
            badge_y = y + cell * 0.28
            draw.rounded_rectangle(
                (x - half_width, badge_y, x + half_width, badge_y + height),
                radius=min(3, height / 4),
                fill=(22, 25, 24),
                outline=color,
                width=2,
            )
            draw.text(
                (x, badge_y + height / 2),
                f"R{index}",
                fill=COLOR_TEXT,
                font=_font(max(5, min(12, round(height * 0.7)))),
                anchor="mm",
            )

    def _draw_robot(self, canvas: Image.Image, robot_pos: Tuple[int, int]) -> None:
        x, y = self._cell_center(*robot_pos)
        art = sprite("rover", max(10, round(self._pixels_per_cell * 0.65)))
        canvas.paste(art, (round(x - art.width / 2), round(y - art.height / 2)), art)

    def _draw_action(
        self, draw: ImageDraw.ImageDraw, robot_pos: Tuple[int, int], action: Optional[int]
    ) -> None:
        if action is None:
            return
        x, y = self._cell_center(*robot_pos)
        dx, dy = self.action_to_vector.get(action, (0, 0))
        if dx or dy:
            length = self._pixels_per_cell * 0.38
            end = (x + dx * length, y + dy * length)
            draw.line([(x, y), end], fill=COLOR_ARROW, width=3)
            self._draw_arrow_head(draw, x, y, end[0], end[1])
        elif action >= 5:
            rock_index = action - 5
            if rock_index < len(self.rock_positions):
                target = self._cell_center(*self.rock_positions[rock_index])
                self._draw_dashed_line(draw, (x, y), target)
                radius = min(17.0, self._pixels_per_cell * 0.16)
                draw.ellipse(
                    (
                        target[0] - radius,
                        target[1] - radius,
                        target[0] + radius,
                        target[1] + radius,
                    ),
                    outline=COLOR_SENSOR,
                    width=3,
                )

    @staticmethod
    def _draw_arrow_head(
        draw: ImageDraw.ImageDraw, start_x: float, start_y: float, end_x: float, end_y: float
    ) -> None:
        vx, vy = end_x - start_x, end_y - start_y
        length = max((vx * vx + vy * vy) ** 0.5, 1.0)
        ux, uy = vx / length, vy / length
        size = 9.0
        base_x, base_y = end_x - ux * size, end_y - uy * size
        draw.polygon(
            [
                (end_x, end_y),
                (base_x - uy * size * 0.55, base_y + ux * size * 0.55),
                (base_x + uy * size * 0.55, base_y - ux * size * 0.55),
            ],
            fill=COLOR_ARROW,
        )

    @staticmethod
    def _draw_dashed_line(
        draw: ImageDraw.ImageDraw,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> None:
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1.0)
        ux, uy = dx / length, dy / length
        cursor = 12.0
        while cursor < length - 12.0:
            finish = min(cursor + 8.0, length - 12.0)
            draw.line(
                [
                    (start[0] + ux * cursor, start[1] + uy * cursor),
                    (start[0] + ux * finish, start[1] + uy * finish),
                ],
                fill=COLOR_SENSOR,
                width=3,
            )
            cursor += 14.0

    def _draw_status(
        self,
        draw: ImageDraw.ImageDraw,
        frame_index: int,
        frame_count: int,
        state: "RockSampleState",
        action: Optional[int],
    ) -> None:
        left, top, _, bottom = self._plot_bounds
        action_name = "Terminal" if action is None else self.action_names[action]
        text = f"STEP {frame_index + 1:02d} / {frame_count:02d}     Action: {action_name}"
        bbox = draw.multiline_textbbox((left + 12, 735), text, font=_font(17), spacing=2)
        draw.rounded_rectangle(
            (bbox[0] - 10, bbox[1] - 8, bbox[2] + 10, bbox[3] + 8),
            radius=5,
            fill=COLOR_INFO,
            outline=(80, 70, 50),
        )
        draw.multiline_text((left + 12, 735), text, fill=COLOR_TEXT, font=_font(17), spacing=2)

        if action == 0:
            success = self._check_sample_success(state)
            label = "VALUABLE!" if success else "WORTHLESS!"
            fill = COLOR_SUCCESS if success else COLOR_FAILURE
            edge = COLOR_SUCCESS_EDGE if success else COLOR_FAILURE_EDGE
            label_bbox = draw.textbbox((left + 12, bottom - 16), label, font=_font(20), anchor="ls")
            draw.rounded_rectangle(
                (
                    label_bbox[0] - 8,
                    label_bbox[1] - 7,
                    label_bbox[2] + 8,
                    label_bbox[3] + 7,
                ),
                radius=7,
                fill=fill,
                outline=edge,
                width=3,
            )
            draw.text((left + 12, bottom - 16), label, fill=edge, font=_font(20), anchor="ls")

    def _check_sample_success(self, state: "RockSampleState") -> bool:
        robot_pos = get_robot_pos(state)
        rocks = get_rocks(state)
        for index, rock_pos in enumerate(self.rock_positions):
            if robot_pos == rock_pos:
                return rocks[index]
        return False

    @staticmethod
    def _build_palette(reference: Image.Image) -> Image.Image:
        # Reserve colors for small sprites: terrain otherwise consumes nearly
        # every palette entry and turns the blue rover into a flat neon patch.
        art_colors = 64
        art = Image.new("RGB", (256, 64), COLOR_PAGE)
        for index, name in enumerate(("rover", "ore-blue", "ore-green", "ore-red")):
            icon = sprite(name, 64)
            art.paste(icon, (index * 64, 0), icon)
        art_palette = art.quantize(colors=art_colors, method=Image.Quantize.MEDIANCUT)
        adaptive = reference.quantize(
            colors=256 - len(ACCENT_COLORS) - art_colors, method=Image.Quantize.MEDIANCUT
        )
        entries = (adaptive.getpalette() or [])[: 3 * (256 - len(ACCENT_COLORS) - art_colors)]
        entries.extend((art_palette.getpalette() or [])[: 3 * art_colors])
        for color in ACCENT_COLORS:
            entries.extend(color)
        entries.extend([0] * (768 - len(entries)))
        master = Image.new("P", (1, 1))
        master.putpalette(entries)
        return master
