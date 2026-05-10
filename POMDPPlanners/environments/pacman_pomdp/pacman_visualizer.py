"""Visualization module for PacMan POMDP environment.

This module provides sprite-based visualization capabilities for PacMan POMDP
episodes, rendering animated GIFs of agent behavior and game state.

Classes:
    PacManVisualizer: Handles sprite-based rendering and GIF generation
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from POMDPPlanners.core.belief import Belief, WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData

if TYPE_CHECKING:
    from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP


class PacManVisualizer:
    """Handles visualization for PacMan POMDP environments.

    This class manages sprite loading, frame rendering, and GIF generation for
    visualizing PacMan POMDP episodes. It renders the maze, PacMan, ghosts, pellets,
    and game state information.

    Attributes:
        env: Reference to the PacMan POMDP environment
        tile_size: Size of each tile in pixels
        sprites: Dictionary of loaded sprite images
    """

    def __init__(self, environment: "PacManPOMDP", tile_size: int = 32):
        """Initialize visualizer with reference to environment.

        Args:
            environment: PacMan POMDP environment instance
            tile_size: Size of each tile in pixels. Defaults to 32.
        """
        self.env = environment
        self.tile_size = tile_size

        # Load sprites from the img directory
        module_dir = Path(__file__).parent
        sprite_dir = module_dir / "img"
        self.sprites = self._load_sprites(sprite_dir, tile_size)
        # Fonts are bundled in the same img/ directory so rendering is
        # byte-identical across systems regardless of OS-installed fonts
        # (golden-file tests run in a slim Docker image without DejaVu).
        self.font_regular: Any = self._load_font(sprite_dir, 13)
        self.font_bold: Any = self._load_font(sprite_dir, 14, bold=True)

    @staticmethod
    def _load_font(sprite_dir: Path, size: int, bold: bool = False) -> Any:
        """Load the bundled DejaVu Sans font; fall back to PIL's default."""
        font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        font_path = sprite_dir / font_name
        try:
            return ImageFont.truetype(str(font_path), size)
        except (OSError, IOError):
            return ImageFont.load_default()

    def _colorize_sprite(self, image: Image.Image, color: Tuple[int, int, int, int]) -> Image.Image:
        """Apply color overlay to sprite image."""
        overlay = Image.new("RGBA", image.size, color)  # type: ignore[arg-type]
        result = Image.blend(image.convert("RGBA"), overlay, 0.3)
        result.putalpha(image.split()[-1])
        return result

    def _load_pacman_sprite(self, sprite_dir: Path, tile_size: int) -> Image.Image:
        """Load or generate PacMan sprite."""
        pacman_head_path = sprite_dir / "pacman_head.jpg"
        pacman_png_path = sprite_dir / "pocman.png"

        if pacman_head_path.exists():
            return Image.open(pacman_head_path).convert("RGBA").resize((tile_size, tile_size))
        if pacman_png_path.exists():
            return Image.open(pacman_png_path).convert("RGBA").resize((tile_size, tile_size))
        img = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))  # type: ignore[arg-type]
        draw = ImageDraw.Draw(img)
        draw.ellipse([4, 4, tile_size - 4, tile_size - 4], fill=(255, 255, 0, 255))
        return img

    def _load_ghost_sprites(self, sprite_dir: Path, tile_size: int) -> dict:
        """Load or generate ghost sprites with different colors."""
        ghost_colors = [
            (255, 0, 0, 255),  # Red ghost
            (0, 255, 0, 255),  # Green ghost
            (0, 0, 255, 255),  # Blue ghost
            (255, 0, 255, 255),  # Magenta ghost
            (255, 165, 0, 255),  # Orange ghost
            (0, 255, 255, 255),  # Cyan ghost
            (255, 255, 0, 255),  # Yellow ghost
            (128, 0, 128, 255),  # Purple ghost
        ]

        sprites = {}
        ghost_path = sprite_dir / "ghosts.png"
        if ghost_path.exists():
            base_ghost = Image.open(ghost_path).convert("RGBA")
            for i, color in enumerate(ghost_colors):
                colored_ghost = self._colorize_sprite(base_ghost, color)
                sprites[f"ghost_{i}"] = colored_ghost.resize((tile_size, tile_size))
        else:
            for i, color in enumerate(ghost_colors):
                img = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))  # type: ignore[arg-type]
                draw = ImageDraw.Draw(img)
                draw.rectangle([4, 4, tile_size - 4, tile_size - 4], fill=color)
                sprites[f"ghost_{i}"] = img

        return sprites

    def _load_sprites(self, sprite_dir: Path, tile_size: int) -> dict:
        """Load all sprites for visualization."""
        sprites = {}
        sprites["pacman"] = self._load_pacman_sprite(sprite_dir, tile_size)
        ghost_sprites = self._load_ghost_sprites(sprite_dir, tile_size)
        sprites.update(ghost_sprites)
        return sprites

    def _draw_maze_background(self, draw: ImageDraw.ImageDraw, tile_size: int) -> None:
        """Draw maze background with walls and corridors."""
        rows, cols = self.env.maze_size
        for r in range(rows):
            for c in range(cols):
                x, y = c * tile_size, r * tile_size
                if (r, c) in self.env.walls:
                    draw.rectangle([x, y, x + tile_size, y + tile_size], fill=(20, 20, 80, 255))
                else:
                    draw.rectangle([x, y, x + tile_size, y + tile_size], fill=(0, 0, 0, 255))

    def _draw_pellets(self, state: np.ndarray, draw: ImageDraw.ImageDraw, tile_size: int) -> None:
        """Draw pellets on the maze."""
        rows, cols = self.env.maze_size
        pellets = self.env.get_pellets(state)
        for r in range(rows):
            for c in range(cols):
                if (r, c) in pellets:
                    x, y = c * tile_size, r * tile_size
                    cx, cy = x + tile_size // 2, y + tile_size // 2
                    rdot = 4
                    draw.ellipse(
                        [cx - rdot, cy - rdot, cx + rdot, cy + rdot],
                        fill=(255, 255, 255, 255),
                    )

    def _draw_ghost_belief(
        self, belief: Optional[Belief], canvas: Image.Image, tile_size: int
    ) -> None:
        """Overlay the belief over ghost positions as a translucent red heatmap.

        Each cell's red intensity is proportional to the marginal probability
        (sum of normalized particle weights) that any ghost occupies that cell.
        """
        if belief is None or not isinstance(belief, WeightedParticleBelief):
            return
        particles = belief.particles
        weights = belief.normalized_weights
        if particles is None or weights is None or len(particles) == 0:
            return

        rows, cols = self.env.maze_size
        heatmap = np.zeros((rows, cols), dtype=np.float64)
        for particle, weight in zip(particles, weights):
            for ghost_pos in self.env.get_ghost_positions(particle):
                gr, gc = int(ghost_pos[0]), int(ghost_pos[1])
                if 0 <= gr < rows and 0 <= gc < cols:
                    heatmap[gr, gc] += float(weight)

        max_w = float(heatmap.max())
        if max_w <= 0.0:
            return

        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))  # type: ignore[arg-type]
        draw_overlay = ImageDraw.Draw(overlay)
        for r in range(rows):
            for c in range(cols):
                w = heatmap[r, c]
                if w <= 0.0:
                    continue
                alpha = int(180.0 * (w / max_w))
                x, y = c * tile_size, r * tile_size
                draw_overlay.rectangle(
                    [x, y, x + tile_size, y + tile_size],
                    fill=(255, 0, 0, alpha),
                )
        canvas.alpha_composite(overlay)

    def _draw_ghosts(
        self, state: np.ndarray, canvas: Image.Image, sprites: dict, tile_size: int
    ) -> None:
        """Draw ghosts on the canvas."""
        rows, cols = self.env.maze_size
        for i, ghost_pos in enumerate(self.env.get_ghost_positions(state)):
            gr, gc = ghost_pos
            if 0 <= gr < rows and 0 <= gc < cols:
                ghost_x, ghost_y = gc * tile_size, gr * tile_size
                ghost_sprite_key = f"ghost_{i % 8}"
                if ghost_sprite_key in sprites:
                    canvas.paste(
                        sprites[ghost_sprite_key],
                        (ghost_x, ghost_y),
                        sprites[ghost_sprite_key],
                    )

    def _draw_pacman(
        self,
        state: np.ndarray,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        sprites: dict,
        tile_size: int,
    ) -> None:
        """Draw PacMan on the canvas."""
        rows, cols = self.env.maze_size
        pacman_pos = self.env.get_pacman_pos(state)
        ghost_positions = self.env.get_ghost_positions(state)
        pr, pc = pacman_pos
        if 0 <= pr < rows and 0 <= pc < cols:
            pacman_x, pacman_y = pc * tile_size, pr * tile_size
            if pacman_pos in ghost_positions:
                draw.ellipse(
                    [
                        pacman_x,
                        pacman_y,
                        pacman_x + tile_size,
                        pacman_y + tile_size,
                    ],
                    fill=(255, 0, 0, 200),
                )
                num_colliding_ghosts = sum(
                    1 for ghost_pos in ghost_positions if ghost_pos == pacman_pos
                )
                explosion_text = "💥" * min(num_colliding_ghosts, 3)
                draw.text(
                    (pacman_x + 2, pacman_y + 2),
                    explosion_text,
                    fill=(255, 255, 255),
                )
            else:
                canvas.paste(sprites["pacman"], (pacman_x, pacman_y), sprites["pacman"])

    def _draw_text_overlay(
        self,
        state: np.ndarray,
        draw: ImageDraw.ImageDraw,
        step_num: int,
        action_name: str,
        tile_size: int,
    ) -> None:
        """Draw text overlay with game state information and legend."""
        rows, cols = self.env.maze_size
        canvas_w = cols * tile_size
        panel_top = rows * tile_size
        # Paint an opaque dark panel under the text so the white/grey
        # foreground text isn't washed out by the GIF's transparent
        # default background.
        draw.rectangle(
            [0, panel_top, canvas_w, panel_top + 80],
            fill=(15, 15, 30, 255),
        )

        text_y = panel_top + 6
        pellets = self.env.get_pellets(state)
        score = self.env.get_score(state)
        score_display = int(score) if float(score).is_integer() else score

        # Line 1: step + action
        draw.text(
            (6, text_y),
            f"Step {step_num}: {action_name}",
            fill=(255, 255, 255),
            font=self.font_bold,
        )
        # Line 2: score / pellets
        draw.text(
            (6, text_y + 18),
            f"Score: {score_display}    Pellets: {len(pellets)}",
            fill=(255, 255, 255),
            font=self.font_regular,
        )
        # Line 3: belief legend (red swatch + caption). Caption text is
        # kept short so it fits the maze-width canvas at small tile sizes.
        legend_x = 6
        legend_y = text_y + 36
        draw.rectangle(
            [legend_x, legend_y + 2, legend_x + 14, legend_y + 14],
            fill=(255, 0, 0, 220),
        )
        draw.text(
            (legend_x + 20, legend_y),
            "= ghost belief",
            fill=(255, 255, 255),
            font=self.font_regular,
        )

        if self.env.get_terminal(state):
            banner = "YOU WIN!" if len(pellets) == 0 else "GAME OVER"
            color = (0, 255, 0) if len(pellets) == 0 else (255, 80, 80)
            draw.text(
                (legend_x + 130, legend_y),
                banner,
                fill=color,
                font=self.font_bold,
            )

    def _render_frame(
        self,
        state: np.ndarray,
        step_num: int,
        action_name: str,
        sprites: dict,
        tile_size: int,
        belief: Optional[Belief] = None,
    ) -> Image.Image:
        """Render a single frame of the visualization."""
        rows, cols = self.env.maze_size
        # Reserve 80 px below the maze for: step+action, score+pellets,
        # belief legend, and an optional terminal-state banner.
        canvas = Image.new("RGBA", (cols * tile_size, rows * tile_size + 80))
        draw = ImageDraw.Draw(canvas)

        self._draw_maze_background(draw, tile_size)
        # Belief overlay sits beneath pellets/ghosts/pacman so it doesn't
        # obscure them; pellets remain visible on top of the heatmap.
        self._draw_ghost_belief(belief, canvas, tile_size)
        self._draw_pellets(state, draw, tile_size)
        self._draw_ghosts(state, canvas, sprites, tile_size)
        self._draw_pacman(state, canvas, draw, sprites, tile_size)
        self._draw_text_overlay(state, draw, step_num, action_name, tile_size)

        return canvas

    def _generate_frames(
        self,
        path: List[np.ndarray],
        actions: List[int],
        sprites: dict,
        tile_size: int,
        beliefs: Optional[List[Optional[Belief]]] = None,
    ) -> List[Image.Image]:
        """Generate all frames for the visualization."""
        frames = []
        for i, state in enumerate(path):
            if i < len(actions):
                action_name = self.env.action_names[actions[i]]
            else:
                action_name = "Terminal"
            belief = beliefs[i] if beliefs is not None and i < len(beliefs) else None
            frame = self._render_frame(state, i + 1, action_name, sprites, tile_size, belief=belief)
            frames.append(frame)

        return frames

    def _save_animated_gif(self, frames: List[Image.Image], cache_path: Path) -> None:
        """Save frames as an animated GIF."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if frames:
            frames[0].save(
                cache_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000,
                loop=0,
            )
            print(f"Sprite-based visualization saved as GIF: {cache_path}")
        else:
            print("No frames generated for visualization")

    def visualize_path(
        self,
        path: List[np.ndarray],
        actions: List[int],
        cache_path: Path,
        beliefs: Optional[List[Optional[Belief]]] = None,
    ) -> None:
        """Visualize PacMan path through the maze using sprite-based rendering.

        Args:
            path: List of state arrays representing the path through the maze.
            actions: List of actions taken at each step.
            cache_path: Path where the GIF should be saved.
            beliefs: Optional per-frame beliefs. When supplied, each frame
                overlays a translucent red heatmap over the cells the
                belief assigns non-zero ghost-occupation probability.

        Raises:
            TypeError: If cache_path is not a Path object.
        """
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")

        frames = self._generate_frames(path, actions, self.sprites, self.tile_size, beliefs=beliefs)
        self._save_animated_gif(frames, cache_path)

    def cache_visualization(self, history: List[StepData], cache_path: Path) -> None:
        """Cache visualization of episode history.

        Args:
            history: List of StepData objects representing the episode
            cache_path: Path where the GIF should be saved

        Raises:
            TypeError: If history or cache_path have wrong types
            ValueError: If history is empty or cache_path doesn't end with .gif
        """
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

        # Extract path, actions, and per-step beliefs from history.
        path = [step.state for step in history]
        actions = [step.action for step in history[:-1]]  # Last step has no action
        beliefs: List[Optional[Belief]] = [step.belief for step in history]

        self.visualize_path(path, actions, cache_path, beliefs=beliefs)
