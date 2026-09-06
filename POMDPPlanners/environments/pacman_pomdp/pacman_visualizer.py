# SPDX-License-Identifier: MIT

"""Cached, state-driven PacMan episode rendering."""

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.pacman_pomdp.pacman_art import character, ghost, pellet, tile

if TYPE_CHECKING:
    from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP


@lru_cache(maxsize=48)
def _font(size: int, bold: bool = False) -> Any:
    path = Path(__file__).with_name("img") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default(size=size)


class PacManVisualizer:
    """One native cell per tile; cached art is read-only between frames."""

    def __init__(self, environment: "PacManPOMDP", tile_size: int = 32):
        if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size < 1:
            raise ValueError("tile_size must be a positive integer")
        self.env = environment
        self.tile_size = tile_size
        self.sprites = {"pacman": character("player", tile_size)}
        self.sprites.update({f"ghost_{i}": ghost(tile_size, i) for i in range(8)})
        self.font_regular = _font(13)
        self.font_bold = _font(14, True)
        self._background: Optional[Image.Image] = None
        self._background_key: Optional[tuple] = None
        self._palette: Optional[Image.Image] = None

    def _scene_key(self, tile_size: int) -> tuple:
        return (
            tuple(self.env.maze_size),
            tuple(sorted(self.env.walls)),
            tile_size,
            tuple(sorted(getattr(self.env, "dangerous_areas", ()))),
            float(getattr(self.env, "dangerous_area_radius", 1.0)),
        )

    def _build_background(self, tile_size: int) -> Image.Image:
        rows, cols = self.env.maze_size
        canvas = Image.new("RGBA", (cols * tile_size, rows * tile_size + 80), (8, 10, 16, 255))
        for row in range(rows):
            for col in range(cols):
                canvas.paste(
                    tile(tile_size, (row, col) in self.env.walls),
                    (col * tile_size, row * tile_size),
                )
        self._draw_dangerous_areas(canvas, tile_size)
        return canvas

    def _draw_dangerous_areas(self, canvas: Image.Image, tile_size: int) -> None:
        """Clip red danger circles to the map, below belief and entities."""
        areas = getattr(self.env, "dangerous_areas", None)
        if not areas:
            return
        rows, cols = self.env.maze_size
        overlay = Image.new("RGBA", (cols * tile_size, rows * tile_size))
        draw = ImageDraw.Draw(overlay)
        radius = max(1, round(float(getattr(self.env, "dangerous_area_radius", 1.0)) * tile_size))
        for row, col in areas:
            x, y = (col + 0.5) * tile_size, (row + 0.5) * tile_size
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(255, 0, 0, 90),
                outline=(255, 64, 40, 160),
            )
        canvas.alpha_composite(overlay)

    def _draw_ghost_belief(
        self, belief: Optional[Belief], canvas: Image.Image, tile_size: int
    ) -> None:
        """Show marginal ghost mass relative to the most occupied belief cell."""
        if belief is None:
            return
        particles = getattr(belief, "particles", None)
        weights = getattr(belief, "normalized_weights", None)
        if particles is None or weights is None or len(particles) == 0:
            return
        rows, cols = self.env.maze_size
        heatmap = np.zeros((rows, cols), dtype=np.float64)
        for particle, weight in zip(particles, weights):
            for row, col in self.env.get_ghost_positions(particle):
                if 0 <= row < rows and 0 <= col < cols:
                    heatmap[row, col] += float(weight)
        maximum = float(heatmap.max())
        if maximum <= 0:
            return
        overlay = Image.new("RGBA", (cols * tile_size, rows * tile_size))
        draw = ImageDraw.Draw(overlay)
        for row, col in zip(*np.nonzero(heatmap)):
            alpha = int(180 * heatmap[row, col] / maximum)
            x, y = int(col) * tile_size, int(row) * tile_size
            draw.rectangle((x, y, x + tile_size - 1, y + tile_size - 1), fill=(255, 0, 0, alpha))
        canvas.alpha_composite(overlay)

    def _draw_pellets(self, state: np.ndarray, canvas: Image.Image, tile_size: int) -> None:
        rows, cols = self.env.maze_size
        art = pellet(tile_size)
        for row, col in self.env.get_pellets(state):
            if 0 <= row < rows and 0 <= col < cols:
                canvas.alpha_composite(art, (col * tile_size, row * tile_size))

    def _draw_ghosts(
        self, state: np.ndarray, canvas: Image.Image, sprites: dict, tile_size: int
    ) -> None:
        rows, cols = self.env.maze_size
        positions = self.env.get_ghost_positions(state)
        for index, (row, col) in enumerate(positions):
            if 0 <= row < rows and 0 <= col < cols:
                # Each sprite contains exactly one ghost, not the old four-ghost atlas.
                canvas.alpha_composite(
                    sprites[f"ghost_{index % 8}"], (col * tile_size, row * tile_size)
                )
        for row, col in set(positions):
            count = positions.count((row, col))
            if count > 1 and 0 <= row < rows and 0 <= col < cols:
                ImageDraw.Draw(canvas).text(
                    (col * tile_size + 2, row * tile_size + 2),
                    str(count),
                    font=_font(max(6, min(12, tile_size // 3)), True),
                    fill=(255, 255, 255),
                )

    def _draw_pacman(
        self, state: np.ndarray, canvas: Image.Image, tile_size: int, direction: str
    ) -> None:
        rows, cols = self.env.maze_size
        row, col = self.env.get_pacman_pos(state)
        if 0 <= row < rows and 0 <= col < cols:
            canvas.alpha_composite(
                character("player", tile_size, direction), (col * tile_size, row * tile_size)
            )
            if tile_size >= 3 and (row, col) in self.env.get_ghost_positions(state):
                edge = max(1, tile_size // 16)
                ImageDraw.Draw(canvas).rounded_rectangle(
                    (
                        col * tile_size + edge,
                        row * tile_size + edge,
                        (col + 1) * tile_size - edge,
                        (row + 1) * tile_size - edge,
                    ),
                    radius=max(1, tile_size // 6),
                    outline=(255, 66, 40),
                    width=edge,
                )

    @staticmethod
    def _fit_text(
        draw: ImageDraw.ImageDraw, text: str, width: int, size: int = 14, bold: bool = False
    ) -> Any:
        while size > 4 and draw.textlength(text, font=_font(size, bold)) > width:
            size -= 1
        return _font(size, bold)

    def _draw_text_overlay(
        self,
        state: np.ndarray,
        draw: ImageDraw.ImageDraw,
        step_num: int,
        action_name: str,
        tile_size: int,
    ) -> None:
        rows, cols = self.env.maze_size
        width, top = cols * tile_size, rows * tile_size
        draw.rectangle((0, top, width, top + 80), fill=(8, 10, 16, 255))
        draw.line((0, top, width, top), fill=(61, 80, 107), width=1)
        pellets = len(self.env.get_pellets(state))
        score = self.env.get_score(state)
        score_text = str(int(score)) if float(score).is_integer() else str(score)
        terminal = self.env.get_terminal(state)
        narrow = width < 150
        tiny = width < 70
        action_short = {
            "north": "N",
            "east": "E",
            "south": "S",
            "west": "W",
            "stay": "stay",
            "Terminal": "end",
            "No action": "—",
        }.get(action_name, action_name)
        first = f"Step {step_num}: {action_name}" if not narrow else f"S{step_num} {action_short}"
        if width >= 400:
            first = f"PACMAN  /  Step {step_num}: {action_name}"
        second = (
            f"Score: {score_text}   Pellets: {pellets}"
            if not narrow
            else f"+{score_text}  P:{pellets}"
        )
        third = (
            "Ghost belief: relative mass"
            if not narrow
            else ("G rel." if tiny else "Ghost: rel. mass")
        )
        if terminal:
            fourth = (
                ("WIN" if pellets == 0 else "OVER")
                if tiny
                else ("YOU WIN!" if pellets == 0 else "GAME OVER")
            )
        else:
            fourth = (
                "Danger zone"
                if getattr(self.env, "dangerous_areas", None)
                else "World state / selected action"
            )
            if narrow:
                fourth = "Danger" if getattr(self.env, "dangerous_areas", None) else "State/action"
        pad = min(6, max(1, width // 16))
        for index, text in enumerate((first, second, third, fourth)):
            size = (18 if index == 0 else 14) if width >= 400 else (14 if index == 0 else 12)
            font = self._fit_text(draw, text, max(1, width - 2 * pad), size, index == 0)
            color = (241, 229, 177) if index == 0 else (195, 207, 224)
            if index == 2:
                color = (255, 124, 119)
            if index == 3 and terminal:
                color = (135, 247, 151) if pellets == 0 else (255, 110, 93)
            draw.text((pad, top + 4 + index * 18), text, font=font, fill=color)

    def _render_frame(
        self,
        state: np.ndarray,
        step_num: int,
        action_name: str,
        sprites: dict,
        tile_size: int,
        belief: Optional[Belief] = None,
    ) -> Image.Image:
        key = self._scene_key(tile_size)
        if self._background is None or self._background_key != key:
            self._background = self._build_background(tile_size)
            self._background_key = key
            self._palette = None
        canvas = self._background.copy()
        self._draw_ghost_belief(belief, canvas, tile_size)
        self._draw_pellets(state, canvas, tile_size)
        self._draw_ghosts(state, canvas, sprites, tile_size)
        self._draw_pacman(state, canvas, tile_size, action_name)
        self._draw_text_overlay(state, ImageDraw.Draw(canvas), step_num, action_name, tile_size)
        return canvas

    def _generate_frames(
        self,
        path: List[np.ndarray],
        actions: List[int],
        sprites: dict,
        tile_size: int,
        beliefs: Optional[List[Optional[Belief]]] = None,
    ) -> List[Image.Image]:
        frames = []
        for index, state in enumerate(path):
            action = actions[index] if index < len(actions) else None
            name = (
                self.env.action_names[action]
                if action is not None
                else ("Terminal" if self.env.get_terminal(state) else "No action")
            )
            belief = beliefs[index] if beliefs is not None and index < len(beliefs) else None
            frames.append(self._render_frame(state, index + 1, name, sprites, tile_size, belief))
        return frames

    def _build_palette(self, reference: Image.Image) -> Image.Image:
        # Put entity art and tinted tiles next to the scene when choosing colors,
        # so small ghosts, yellow highlights and red mass retain their detail.
        atlas = Image.new("RGB", (512, 256), (8, 10, 16))
        for index, (name, direction) in enumerate((("player", "east"), ("ghost", "east"))):
            art = character(name, 96, direction)
            atlas.paste(art, (index * 128, 0), art)
        for index, alpha in enumerate((45, 90, 135, 180)):
            tint = tile(64, False).copy()
            tint.alpha_composite(Image.new("RGBA", (64, 64), (255, 0, 0, alpha)))
            atlas.paste(tint, (index * 64, 128))
        for index in range(8):
            art = ghost(64, index)
            atlas.paste(art, (index * 64, 192), art)
        colors = atlas.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
        scene = reference.convert("RGB").quantize(colors=144, method=Image.Quantize.MEDIANCUT)
        entries = (scene.getpalette() or [])[:432] + (colors.getpalette() or [])[:288]
        for color in (
            (255, 255, 245),
            (241, 229, 177),
            (195, 207, 224),
            (255, 124, 119),
            (135, 247, 151),
            (255, 110, 93),
            (8, 10, 16),
            (61, 80, 107),
        ):
            entries.extend(color)
        entries.extend([0] * (768 - len(entries)))
        master = Image.new("P", (1, 1))
        master.putpalette(entries)
        return master

    def _save_animated_gif(self, frames: List[Image.Image], cache_path: Path) -> None:
        if not frames:
            return
        if self._palette is None:
            self._palette = self._build_palette(
                self._background if self._background is not None else frames[0]
            )
        indexed = [
            frame.convert("RGB").quantize(palette=self._palette, dither=Image.Dither.NONE)
            for frame in frames
        ]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        indexed[0].save(
            cache_path,
            save_all=True,
            append_images=indexed[1:],
            duration=1000,
            loop=0,
            optimize=False,
            disposal=1,
        )

    def visualize_path(
        self,
        path: List[np.ndarray],
        actions: List[int],
        cache_path: Path,
        beliefs: Optional[List[Optional[Belief]]] = None,
    ) -> None:
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if not path:
            raise ValueError("Cannot visualize empty path")
        frames = self._generate_frames(path, actions, self.sprites, self.tile_size, beliefs)
        self._save_animated_gif(frames, cache_path)

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        if not isinstance(history, List):
            raise TypeError("history must be a List object")
        if not history:
            raise ValueError("Cannot visualize empty history")
        if any(not isinstance(step, StepData) for step in history):
            raise TypeError("history must be a List of StepData objects")
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")
        if cache_path.suffix != ".gif":
            raise ValueError("cache_path must end with .gif")
        self.visualize_path(
            [step.state for step in history],
            [step.action for step in history],
            cache_path,
            beliefs=[step.belief for step in history],
        )
