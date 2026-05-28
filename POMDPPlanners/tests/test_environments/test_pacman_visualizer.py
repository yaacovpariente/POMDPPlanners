# SPDX-License-Identifier: MIT

"""Tests for ``PacManVisualizer`` belief-overlay rendering."""

# Tests intentionally exercise the protected ``_draw_ghost_belief`` method
# because the bug under test lives in its isinstance gate.
# pylint: disable=protected-access

import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from PIL import Image

from POMDPPlanners.core.belief import (
    VectorizedWeightedParticleBelief,
    WeightedParticleBelief,
)
from POMDPPlanners.core.simulation.history import StepData
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_vectorized_updater import (
    PacManVectorizedUpdater,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_visualizer import PacManVisualizer
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import pacman_pinned_kwargs


def _make_env() -> PacManPOMDP:
    return PacManPOMDP(
        discount_factor=0.95,
        **pacman_pinned_kwargs(
            maze_size=(5, 5),
            walls=set(),
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(4, 4)],
            initial_pellets=[(2, 2)],
        ),
    )


def _single_particle(env: PacManPOMDP, ghost_pos: Tuple[int, int]) -> np.ndarray:
    return env.make_state(
        pacman_pos=(0, 0),
        ghost_positions=(ghost_pos,),
        pellets=((2, 2),),
        score=0.0,
        terminal=False,
    )


def _make_vectorized_belief(
    env: PacManPOMDP, ghost_pos: Tuple[int, int], n_particles: int = 10
) -> VectorizedWeightedParticleBelief:
    particle = _single_particle(env, ghost_pos)
    particles = np.tile(particle, (n_particles, 1))
    log_weights = np.log(np.ones(n_particles) / n_particles)
    updater = PacManVectorizedUpdater.from_environment(env)
    return VectorizedWeightedParticleBelief(particles, log_weights, updater)


def _make_state_keyed_belief(
    env: PacManPOMDP, ghost_pos: Tuple[int, int], n_particles: int = 10
) -> WeightedParticleBelief:
    particle = _single_particle(env, ghost_pos)
    particles: List[np.ndarray] = [particle.copy() for _ in range(n_particles)]
    log_weights = np.log(np.ones(n_particles) / n_particles)
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


def _blank_canvas(env: PacManPOMDP, tile_size: int) -> Image.Image:
    rows, cols = env.maze_size
    return Image.new("RGBA", (cols * tile_size, rows * tile_size + 80))


def _tile_center_alpha(canvas: Image.Image, row: int, col: int, tile_size: int) -> int:
    cx = col * tile_size + tile_size // 2
    cy = row * tile_size + tile_size // 2
    pixel = canvas.getpixel((cx, cy))
    assert isinstance(pixel, tuple) and len(pixel) == 4
    return int(pixel[3])


def test_draw_ghost_belief_renders_vectorized_belief() -> None:
    """Vectorized belief produces a non-empty heatmap on the target ghost cell.

    Purpose: Regression for the bug where ``_draw_ghost_belief`` gated on
        ``isinstance(belief, WeightedParticleBelief)`` and silently skipped
        ``VectorizedWeightedParticleBelief``.

    Given: A ``PacManPOMDP`` with maze_size=(5,5) and a
        ``VectorizedWeightedParticleBelief`` whose particles all place the
        single ghost at cell (3, 1).
    When: ``_draw_ghost_belief`` is invoked on a fresh transparent canvas.
    Then: The center pixel of tile (3, 1) has alpha > 0 (red heatmap), and
        every other tile center has alpha == 0.

    Test type: unit
    """
    env = _make_env()
    visualizer = PacManVisualizer(env, tile_size=16)
    ghost_pos = (3, 1)
    belief = _make_vectorized_belief(env, ghost_pos)
    canvas = _blank_canvas(env, visualizer.tile_size)

    visualizer._draw_ghost_belief(belief, canvas, visualizer.tile_size)

    assert _tile_center_alpha(canvas, *ghost_pos, visualizer.tile_size) > 0
    rows, cols = env.maze_size
    for r in range(rows):
        for c in range(cols):
            if (r, c) == ghost_pos:
                continue
            assert _tile_center_alpha(canvas, r, c, visualizer.tile_size) == 0


def test_draw_ghost_belief_renders_state_keyed_belief() -> None:
    """State-keyed belief still renders the heatmap (regression guard).

    Purpose: Confirms the fix does not regress the existing
        ``WeightedParticleBelief`` rendering path.

    Given: A ``PacManPOMDP`` with maze_size=(5,5) and a
        ``WeightedParticleBelief`` whose particles all place the single ghost
        at cell (2, 4).
    When: ``_draw_ghost_belief`` is invoked on a fresh transparent canvas.
    Then: The center pixel of tile (2, 4) has alpha > 0.

    Test type: unit
    """
    env = _make_env()
    visualizer = PacManVisualizer(env, tile_size=16)
    ghost_pos = (2, 4)
    belief = _make_state_keyed_belief(env, ghost_pos)
    canvas = _blank_canvas(env, visualizer.tile_size)

    visualizer._draw_ghost_belief(belief, canvas, visualizer.tile_size)

    assert _tile_center_alpha(canvas, *ghost_pos, visualizer.tile_size) > 0


def test_draw_ghost_belief_noop_when_belief_none() -> None:
    """``None`` belief leaves the canvas unmodified.

    Purpose: Verifies the explicit ``None``-guard remains intact.

    Given: A ``PacManPOMDP`` and a fresh transparent canvas.
    When: ``_draw_ghost_belief`` is called with ``belief=None``.
    Then: Every tile center pixel still has alpha == 0.

    Test type: unit
    """
    env = _make_env()
    visualizer = PacManVisualizer(env, tile_size=16)
    canvas = _blank_canvas(env, visualizer.tile_size)

    visualizer._draw_ghost_belief(None, canvas, visualizer.tile_size)

    rows, cols = env.maze_size
    for r in range(rows):
        for c in range(cols):
            assert _tile_center_alpha(canvas, r, c, visualizer.tile_size) == 0


def test_draw_ghost_belief_noop_when_particles_empty() -> None:
    """Empty particle set leaves the canvas unmodified.

    Purpose: Verifies the empty-particle guard remains intact after the
        isinstance gate is removed.

    Given: A ``WeightedParticleBelief``-like object with zero particles.
    When: ``_draw_ghost_belief`` is invoked.
    Then: Every tile center pixel still has alpha == 0.

    Test type: unit
    """
    env = _make_env()
    visualizer = PacManVisualizer(env, tile_size=16)
    canvas = _blank_canvas(env, visualizer.tile_size)

    class _EmptyBelief:
        particles: List[np.ndarray] = []
        normalized_weights = np.array([])

    visualizer._draw_ghost_belief(_EmptyBelief(), canvas, visualizer.tile_size)  # type: ignore[arg-type]

    rows, cols = env.maze_size
    for r in range(rows):
        for c in range(cols):
            assert _tile_center_alpha(canvas, r, c, visualizer.tile_size) == 0


def test_cache_visualization_vectorized_belief_renders_red_overlay() -> None:
    """End-to-end: a history carrying a vectorized belief renders a red overlay.

    Purpose: Mirrors the bug-report acceptance criterion that a PFT-DPW-style
        episode (i.e. one whose ``StepData.belief`` is a
        ``VectorizedWeightedParticleBelief``) produces a GIF whose frames
        contain at least one red-dominant pixel inside the maze region.

    Given: A two-step ``StepData`` history whose belief is a
        ``VectorizedWeightedParticleBelief`` concentrated on a single ghost
        cell.
    When: ``env.cache_visualization(history, gif_path)`` is invoked.
    Then: At least one pixel in the maze region of the first frame is
        red-dominant (R > G, R > B, alpha > 0).

    Test type: integration
    """
    env = _make_env()
    actual_ghost_pos = (0, 4)
    belief_ghost_pos = (3, 1)
    belief = _make_vectorized_belief(env, belief_ghost_pos)
    state = _single_particle(env, actual_ghost_pos)
    obs = np.array([actual_ghost_pos[0], actual_ghost_pos[1]])
    history = [
        StepData(state, 0, state, obs, -1.0, belief),
        StepData(state, None, state, obs, None, belief),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        gif_path = Path(tmpdir) / "viz.gif"
        env.cache_visualization(history, gif_path)
        assert gif_path.exists()

        frame = Image.open(gif_path).convert("RGBA")
        tile_size = PacManVisualizer(env).tile_size
        br, bc = belief_ghost_pos
        tile = np.array(
            frame.crop((bc * tile_size, br * tile_size, (bc + 1) * tile_size, (br + 1) * tile_size))
        )

        red = tile[..., 0].astype(int)
        green = tile[..., 1].astype(int)
        blue = tile[..., 2].astype(int)
        red_dominant = (red > green) & (red > blue) & (red > 0)
        assert (
            red_dominant.any()
        ), "Expected at least one red-dominant pixel inside the belief-concentrated tile"


def test_draw_dangerous_areas_renders_red_overlay_on_configured_cell() -> None:
    """Dangerous-area overlay tints the configured center cell red.

    Purpose: Smoke-tests :meth:`PacManVisualizer._draw_dangerous_areas` so
        that future refactors of the visualizer cannot silently drop the
        hazard rendering — without it, planners and reviewers reading a
        cached GIF would have no visual cue that the danger penalty was
        active.

    Given: A PacManPOMDP with one configured zone at (2, 2) radius 1.0 and
        a fresh blank RGBA canvas matching the visualizer dimensions.
    When: ``_draw_dangerous_areas`` is invoked on the canvas.
    Then: The center pixel of tile (2, 2) has alpha > 0 (red overlay
        present), and a clearly distant tile (0, 0) still has alpha == 0.

    Test type: unit
    """
    env = PacManPOMDP(
        discount_factor=0.95,
        **pacman_pinned_kwargs(
            maze_size=(5, 5),
            walls=set(),
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(4, 4)],
            initial_pellets=[(2, 2)],
            dangerous_areas={(2, 2)},
            dangerous_area_radius=1.0,
            dangerous_area_penalty=5.0,
        ),
    )
    visualizer = PacManVisualizer(env, tile_size=16)
    canvas = _blank_canvas(env, visualizer.tile_size)

    visualizer._draw_dangerous_areas(canvas, visualizer.tile_size)

    assert _tile_center_alpha(canvas, 2, 2, visualizer.tile_size) > 0
    assert _tile_center_alpha(canvas, 0, 0, visualizer.tile_size) == 0


def test_draw_dangerous_areas_noop_when_feature_disabled() -> None:
    """No danger zones configured ⇒ no overlay touches the canvas.

    Purpose: Guards the opt-in contract — vanilla PacMan envs (default
        ``dangerous_areas``) should produce pixel-identical canvases before
        and after the new hook fires.

    Given: A PacManPOMDP constructed without ``dangerous_areas``.
    When: ``_draw_dangerous_areas`` is invoked on a fresh canvas.
    Then: Every tile center still has alpha == 0.

    Test type: unit
    """
    env = _make_env()
    visualizer = PacManVisualizer(env, tile_size=16)
    canvas = _blank_canvas(env, visualizer.tile_size)

    visualizer._draw_dangerous_areas(canvas, visualizer.tile_size)

    rows, cols = env.maze_size
    for r in range(rows):
        for c in range(cols):
            assert _tile_center_alpha(canvas, r, c, visualizer.tile_size) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
