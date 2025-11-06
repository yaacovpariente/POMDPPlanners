"""Visualization module for PacMan POMDP environment.

This module provides sprite-based visualization capabilities for PacMan POMDP
episodes, rendering animated GIFs of agent behavior and game state.

Classes:
    PacManVisualizer: Handles sprite-based rendering and GIF generation
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

from PIL import Image, ImageDraw

from POMDPPlanners.core.simulation import StepData

if TYPE_CHECKING:
    from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP, PacManState


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
        elif pacman_png_path.exists():
            return Image.open(pacman_png_path).convert("RGBA").resize((tile_size, tile_size))
        else:
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

    def _draw_pellets(
        self, state: "PacManState", draw: ImageDraw.ImageDraw, tile_size: int
    ) -> None:
        """Draw pellets on the maze."""
        rows, cols = self.env.maze_size
        for r in range(rows):
            for c in range(cols):
                if (r, c) in state.pellets:
                    x, y = c * tile_size, r * tile_size
                    cx, cy = x + tile_size // 2, y + tile_size // 2
                    rdot = 4
                    draw.ellipse(
                        [cx - rdot, cy - rdot, cx + rdot, cy + rdot],
                        fill=(255, 255, 255, 255),
                    )

    def _draw_ghosts(
        self, state: "PacManState", canvas: Image.Image, sprites: dict, tile_size: int
    ) -> None:
        """Draw ghosts on the canvas."""
        rows, cols = self.env.maze_size
        for i, ghost_pos in enumerate(state.ghost_positions):
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
        state: "PacManState",
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        sprites: dict,
        tile_size: int,
    ) -> None:
        """Draw PacMan on the canvas."""
        rows, cols = self.env.maze_size
        pr, pc = state.pacman_pos
        if 0 <= pr < rows and 0 <= pc < cols:
            pacman_x, pacman_y = pc * tile_size, pr * tile_size
            if state.pacman_pos in state.ghost_positions:
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
                    1 for ghost_pos in state.ghost_positions if ghost_pos == state.pacman_pos
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
        state: "PacManState",
        draw: ImageDraw.ImageDraw,
        step_num: int,
        action_name: str,
        tile_size: int,
    ) -> None:
        """Draw text overlay with game state information."""
        rows, _ = self.env.maze_size
        text_y = rows * tile_size + 5
        draw.text((5, text_y), f"Step {step_num}: {action_name}", fill=(255, 255, 255))
        draw.text(
            (5, text_y + 15),
            f"Score: {state.score}, Pellets: {len(state.pellets)}",
            fill=(255, 255, 255),
        )

        if state.terminal:
            if len(state.pellets) == 0:
                draw.text((5, text_y + 30), "🎉 YOU WIN! 🎉", fill=(0, 255, 0))
            else:
                draw.text((5, text_y + 30), "👻 GAME OVER! 👻", fill=(255, 0, 0))

    def _render_frame(
        self, state: "PacManState", step_num: int, action_name: str, sprites: dict, tile_size: int
    ) -> Image.Image:
        """Render a single frame of the visualization."""
        rows, cols = self.env.maze_size
        canvas = Image.new("RGBA", (cols * tile_size, rows * tile_size + 60))
        draw = ImageDraw.Draw(canvas)

        self._draw_maze_background(draw, tile_size)
        self._draw_pellets(state, draw, tile_size)
        self._draw_ghosts(state, canvas, sprites, tile_size)
        self._draw_pacman(state, canvas, draw, sprites, tile_size)
        self._draw_text_overlay(state, draw, step_num, action_name, tile_size)

        return canvas

    def _generate_frames(
        self, path: List["PacManState"], actions: List[int], sprites: dict, tile_size: int
    ) -> List[Image.Image]:
        """Generate all frames for the visualization."""
        frames = []
        for i, state in enumerate(path):
            if i < len(actions):
                action_name = self.env.action_names[actions[i]]
            else:
                action_name = "Terminal"

            frame = self._render_frame(state, i + 1, action_name, sprites, tile_size)
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
        self, path: List["PacManState"], actions: List[int], cache_path: Path
    ) -> None:
        """Visualize PacMan path through the maze using sprite-based rendering.

        Args:
            path: List of states representing the path through the maze
            actions: List of actions taken at each step
            cache_path: Path where the GIF should be saved

        Raises:
            TypeError: If cache_path is not a Path object
        """
        if not isinstance(cache_path, Path):
            raise TypeError("cache_path must be a Path object")

        frames = self._generate_frames(path, actions, self.sprites, self.tile_size)
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

        # Extract path and actions
        path = [step.state for step in history]
        actions = [step.action for step in history[:-1]]  # Last step has no action

        self.visualize_path(path, actions, cache_path)
