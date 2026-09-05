# SPDX-License-Identifier: MIT

"""Focused tests for the Pillow-based Light-Dark renderer.

These cover the properties the renderer is supposed to guarantee rather than the
exact picture: determinism, the cached-background compositing model, hazards
drawn as translucent probability fields rather than walls, the preserved world
geometry and public API, and that all three Light-Dark variants use the same
renderer.  The exact pixels are covered by the golden-file test.
"""

import ast
import hashlib
from pathlib import Path
from typing import List, cast

import numpy as np
import pytest
from PIL import Image

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ContinuousLightDarkPOMDPDiscreteActions,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils import (
    light_dark_assets,
    light_dark_visualizer as visualizer_module,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_visualizer import (
    CANVAS_SIZE,
    PLOT_BOTTOM,
    PLOT_LEFT,
    PLOT_RIGHT,
    PLOT_TOP,
    LightDarkPOMDPVisualizer,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    continuous_light_dark_pinned_kwargs,
)


# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def continuous_env():
    return ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        name="TestContinuousLightDark",
        **continuous_light_dark_pinned_kwargs(),
    )


@pytest.fixture
def continuous_discrete_env():
    return ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)


@pytest.fixture
def discrete_env():
    return DiscreteLightDarkPOMDP(discount_factor=0.95, name="TestDiscreteLightDark")


def _episode(n_steps: int = 5, n_particles: int = 8) -> List[StepData]:
    """Deterministic episode: the rover walks right, belief spreads around it."""
    rng = np.random.default_rng(42)
    steps: List[StepData] = []
    for i in range(n_steps):
        state = np.array([float(i), 5.0])
        particles = [state + rng.normal(0.0, 0.3, size=2) for _ in range(n_particles)]
        belief = WeightedParticleBelief(
            particles=particles,
            log_weights=np.full(n_particles, -np.log(n_particles)),
        )
        steps.append(
            StepData(
                state=state,
                action="right",
                next_state=np.array([float(i) + 1.0, 5.0]),
                observation=None,
                reward=-1.0,
                belief=belief,
            )
        )
    return steps


def _paths(steps: List[StepData]):
    path = [s.state for s in steps]
    beliefs = [
        cast(WeightedParticleBelief, s.belief).to_unique_support_distribution() for s in steps
    ]
    actions = [s.action for s in steps]
    return path, beliefs, actions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- output shape --------------------------------------------------------


@pytest.mark.parametrize(
    "env_fixture", ["continuous_env", "continuous_discrete_env", "discrete_env"]
)
def test_gif_frame_count_and_size(env_fixture, request, tmp_path):
    """All variants produce one GIF frame per recorded step at the fixed size."""
    env = request.getfixturevalue(env_fixture)
    steps = _episode(n_steps=6)
    out = tmp_path / "episode.gif"

    LightDarkPOMDPVisualizer(env).cache_visualization(steps, out)

    assert out.exists()
    with Image.open(out) as gif:
        assert gif.size == CANVAS_SIZE
        assert getattr(gif, "n_frames") == len(steps)


@pytest.mark.parametrize(
    "env_fixture", ["continuous_env", "continuous_discrete_env", "discrete_env"]
)
def test_environment_cache_visualization_uses_this_renderer(env_fixture, request, tmp_path):
    """The inherited ``Environment.cache_visualization`` path still works.

    The variants do not own a renderer; they delegate to the shared one through
    the base class, so a regression there would be invisible to the tests above.
    """
    env = request.getfixturevalue(env_fixture)

    env.cache_visualization(_episode(n_steps=3), tmp_path, 0)

    out = tmp_path / "agent_path_0.gif"
    assert out.exists()
    with Image.open(out) as gif:
        assert getattr(gif, "n_frames") == 3
        assert gif.size == CANVAS_SIZE


def test_frames_change_as_the_rover_moves(continuous_env):
    """Consecutive frames must differ; a cached background must not freeze them."""
    steps = _episode(n_steps=4)
    frames = LightDarkPOMDPVisualizer(continuous_env).render_frames(*_paths(steps))

    assert len(frames) == 4
    for a, b in zip(frames, frames[1:]):
        assert np.asarray(a).tobytes() != np.asarray(b).tobytes()


# --- determinism ---------------------------------------------------------


def test_same_inputs_give_identical_gif_bytes(continuous_env, tmp_path):
    """Two independent renders of one episode must be byte-identical.

    The golden-file test compares hashes, so any hidden randomness in the
    procedural art would make it flap.
    """
    steps = _episode(n_steps=4)
    first, second = tmp_path / "a.gif", tmp_path / "b.gif"

    LightDarkPOMDPVisualizer(continuous_env).cache_visualization(steps, first)
    LightDarkPOMDPVisualizer(continuous_env).cache_visualization(steps, second)

    assert _sha256(first) == _sha256(second)


def test_ground_texture_is_seeded_and_size_dependent():
    a = light_dark_assets.ground_texture(64, 48)
    b = light_dark_assets.ground_texture(64, 48)
    assert np.array_equal(a, b)
    assert a.shape == (48, 64, 3)
    assert light_dark_assets.ground_texture(32, 48).shape == (48, 32, 3)
    # Texture must actually vary, otherwise "textured ground" is a flat fill.
    assert float(a.std()) > 0.01


def test_ground_texture_cache_is_not_mutated_by_a_render(continuous_env):
    """The renderer copies the cached texture before lighting it.

    Lighting is done in place for speed; writing into the lru_cache entry would
    make the second render of a process darker than the first.
    """
    width, height = PLOT_RIGHT - PLOT_LEFT, PLOT_BOTTOM - PLOT_TOP
    before = light_dark_assets.ground_texture(width, height).copy()

    LightDarkPOMDPVisualizer(continuous_env)._build_background()

    assert np.array_equal(light_dark_assets.ground_texture(width, height), before)


# --- compositing model ---------------------------------------------------


def test_background_is_built_once_per_visualizer(continuous_env, monkeypatch):
    """N frames must cost one background build, not N."""
    vis = LightDarkPOMDPVisualizer(continuous_env)
    calls = {"n": 0}
    real = vis._build_background

    def counted():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(vis, "_build_background", counted)
    steps = _episode(n_steps=7)
    vis.render_frames(*_paths(steps))
    vis.render_frames(*_paths(steps))

    assert calls["n"] == 1


def test_renderer_modules_do_not_import_matplotlib():
    """The Light-Dark render path must stay off Matplotlib.

    Checked on the source, not ``sys.modules``: importing the environments
    package pulls Matplotlib in through other environments' visualizers, which
    would mask a regression here.
    """
    for module in (visualizer_module, light_dark_assets):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith("matplotlib"), f"{module.__name__} imports {name}"


# --- scene semantics -----------------------------------------------------


def test_world_geometry_and_view_limits_are_preserved(discrete_env):
    """The view still spans -1 .. grid_size + 1 and maps onto the plot box."""
    vis = LightDarkPOMDPVisualizer(discrete_env)
    lo, hi = vis._world_limits
    assert (lo, hi) == (-1.0, float(discrete_env.grid_size) + 1.0)

    assert vis._to_px(lo, hi) == pytest.approx((PLOT_LEFT, PLOT_TOP))
    assert vis._to_px(hi, lo) == pytest.approx((PLOT_RIGHT, PLOT_BOTTOM))
    # y grows upward in the world and downward in pixels.
    assert vis._to_px(0.0, 1.0)[1] < vis._to_px(0.0, 0.0)[1]


def test_hazards_are_translucent_probability_fields_not_walls(continuous_env):
    """Inside a hazard the ground underneath must still show through.

    A hazard is a probability of being hit, not a wall, so it has to read as a
    wash over the world exactly as in the mockup.  The check compares the same
    pixels rendered with and without the obstacles: a translucent wash leaves
    the two strongly correlated, a solid fill destroys the correlation.  It also
    excludes the opaque centre dot, whose own contrast would otherwise supply
    all the variance a weaker test looks for.
    """
    vis = LightDarkPOMDPVisualizer(continuous_env)
    with_hazard = np.asarray(vis._build_background()).astype(float)

    bare_env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        name="TestNoObstacles",
        **{**continuous_light_dark_pinned_kwargs(), "obstacles": []},
    )
    without = np.asarray(LightDarkPOMDPVisualizer(bare_env)._build_background()).astype(float)

    ox, oy = float(continuous_env.obstacles[0, 0]), float(continuous_env.obstacles[1, 0])
    radius = float(continuous_env.obstacle_radius)
    cx, cy = vis._to_px(ox, oy)
    half = 0.6 * radius * vis._px_per_unit
    box = (slice(int(cy - half), int(cy + half)), slice(int(cx - half), int(cx + half)))

    # Annulus: inside the hazard but clear of the centre dot and of the rim.
    yy, xx = np.mgrid[box[0], box[1]]
    dist = np.hypot(xx - cx, yy - cy)
    annulus = (dist > 0.25 * half) & (dist < 0.9 * half)
    assert annulus.sum() > 500

    inside = with_hazard[box][annulus]
    bare = without[box][annulus]

    # The wash reddens the region...
    assert inside[:, 0].mean() > bare[:, 0].mean() + 10.0
    # ...but the ground underneath still drives the pixels. A solid fill would
    # make the correlation collapse.
    correlation = np.corrcoef(inside[:, 0], bare[:, 0])[0, 1]
    assert correlation > 0.8, f"hazard is not translucent: correlation {correlation:.3f}"


def test_belief_particles_are_drawn(continuous_env):
    """A frame rendered with a belief must differ from one rendered without."""
    steps = _episode(n_steps=3, n_particles=12)
    path, beliefs, actions = _paths(steps)
    vis = LightDarkPOMDPVisualizer(continuous_env)

    with_belief = np.asarray(vis.render_frames(path, beliefs, actions)[-1]).astype(int)
    # An empty belief list is the "no particles" case the renderer must tolerate.
    without = np.asarray(vis.render_frames(path, [], actions)[-1]).astype(int)

    assert not np.array_equal(with_belief, without)
    changed = np.any(with_belief != without, axis=2)
    # The added pixels are the yellow end of the belief ramp.
    added = with_belief[changed]
    assert added.size > 0
    assert added[:, 0].mean() > added[:, 2].mean()


@pytest.mark.parametrize("sprite_name", ["rover", "beacon", "goal"])
def test_packaged_art_is_transparent_detailed_and_cached(sprite_name):
    sprite_factory = getattr(light_dark_assets, f"{sprite_name}_sprite")
    sprite = sprite_factory(60)
    assert sprite is sprite_factory(60)
    assert sprite.mode == "RGBA"
    assert sprite.size == (60, 60)
    rgba = np.asarray(sprite)
    assert np.all(rgba[[0, -1]][:, [0, -1], 3] == 0)
    assert np.count_nonzero(rgba[:, :, 3] > 200) > 300
    assert len(np.unique(rgba[rgba[:, :, 3] > 200, :3], axis=0)) > 100


@pytest.mark.parametrize("size", [CANVAS_SIZE, (1, 1), (3, 7), (20, 4)])
def test_palette_analysis_samples_without_resizing_frames(monkeypatch, size):
    original = Image.Image.quantize
    sizes = []

    def tracked(image, *args, **kwargs):
        sizes.append(image.size)
        return original(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "quantize", tracked)
    reference = Image.new("RGB", size, (130, 105, 80))
    master = LightDarkPOMDPVisualizer._build_palette(reference)
    assert sizes == [(max(1, size[0] // 2), max(1, size[1] // 2))]
    assert reference.size == size
    palette = master.getpalette()
    assert palette is not None and len(palette) == 768
    entries = [tuple(palette[i : i + 3]) for i in range(0, 768, 3)]
    assert all(color in entries for color in visualizer_module.ACCENT_COLORS)


def _headlight_centroid(sprite):
    """Where the rover's two bright headlights sit, in sprite pixels."""
    rgba = np.asarray(sprite.convert("RGBA")).astype(int)
    # Ignore the arbitrary RGB values of transparent antialiasing pixels.
    mask = (
        (rgba[:, :, 0] > 240)
        & (rgba[:, :, 1] > 230)
        & (rgba[:, :, 2] > 180)
        & (rgba[:, :, 3] > 200)
    )
    ys, xs = np.nonzero(mask)
    assert xs.size > 0, "rover sprite has no headlights to locate"
    return float(xs.mean()), float(ys.mean())


def test_rover_sprite_rotates_to_its_heading():
    """The sprite really turns, and turns the right way.

    Comparing whole rendered frames would not prove this: the action arrow and
    the HUD text differ between two actions anyway, so such a test passes even
    with the rotation removed.  This looks at the sprite itself and at where its
    headlights end up.
    """
    east = light_dark_assets.rover_sprite_facing(60, 0)
    north = light_dark_assets.rover_sprite_facing(60, 90)
    west = light_dark_assets.rover_sprite_facing(60, 180)

    assert np.asarray(east).tobytes() != np.asarray(north).tobytes()

    ex, ey = _headlight_centroid(east)
    nx, ny = _headlight_centroid(north)
    wx, wy = _headlight_centroid(west)
    centre = east.width / 2.0

    tolerance = 3.0
    assert ex > centre and abs(ey - centre) < tolerance, "heading 0 must point +x"
    assert ny < centre and abs(nx - centre) < tolerance, "heading 90 must point up on screen"
    assert wx < centre and abs(wy - centre) < tolerance, "heading 180 must point -x"


def test_headings_follow_the_actions_and_hold_when_there_is_none(continuous_env):
    """Heading per frame: from the action, held over frames that have none."""
    vis = LightDarkPOMDPVisualizer(continuous_env)
    headings = vis._headings([(1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (-1.0, 0.0)])
    assert headings == [0, 90, 90, 180]


def test_rendered_frame_uses_the_heading(continuous_env, monkeypatch):
    """The renderer passes the frame's heading to the sprite, not a constant."""
    vis = LightDarkPOMDPVisualizer(continuous_env)
    seen = []
    real = visualizer_module.rover_sprite_facing

    def spy(size, heading):
        seen.append(heading)
        return real(size, heading)

    monkeypatch.setattr(visualizer_module, "rover_sprite_facing", spy)
    vis.render_frames([np.array([5.0, 5.0]), np.array([5.0, 6.0])], [], ["right", "up"])

    assert seen == [0, 90]


# --- public API contract -------------------------------------------------


def test_cache_path_must_be_a_gif_path(continuous_env, tmp_path):
    vis = LightDarkPOMDPVisualizer(continuous_env)
    path, beliefs, actions = _paths(_episode(n_steps=2))

    with pytest.raises(TypeError):
        not_a_path = str(tmp_path / "x.gif")
        vis.visualize_path(path, beliefs, actions, not_a_path)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        vis.visualize_path(path, beliefs, actions, tmp_path / "x.png")


def test_history_validation(continuous_env, tmp_path):
    vis = LightDarkPOMDPVisualizer(continuous_env)
    out = tmp_path / "x.gif"

    with pytest.raises(TypeError):
        vis.cache_visualization("not a list", out)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        vis.cache_visualization([], out)
    with pytest.raises(TypeError):
        vis.cache_visualization([object()], out)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        vis.cache_visualization(_episode(n_steps=1), str(out))  # type: ignore[arg-type]


def test_missing_parent_directory_is_created(continuous_env, tmp_path):
    out = tmp_path / "nested" / "deeper" / "episode.gif"
    LightDarkPOMDPVisualizer(continuous_env).cache_visualization(_episode(n_steps=2), out)
    assert out.exists()


def test_rendering_carries_no_state_between_calls(continuous_env):
    """Reusing one visualizer must not let an earlier episode change a later one.

    The rover heading is held across frames within an episode; holding it on the
    instance instead would make the same episode render differently depending on
    what was rendered before it, and the golden-file test would flap.
    """
    vis = LightDarkPOMDPVisualizer(continuous_env)
    path = [np.array([5.0, 5.0]), np.array([5.0, 6.0])]

    first = np.asarray(vis.render_frames(path, [], [None, None])[0])
    vis.render_frames(path, [], ["up", "left"])
    again = np.asarray(vis.render_frames(path, [], [None, None])[0])

    assert np.array_equal(first, again)


def test_belief_trail_length_follows_the_path_not_the_belief_list(continuous_env):
    """Trail depth is bounded by the path, as in the old renderer.

    `cache_visualization` always passes lists of equal length, but
    `visualize_path` is public and does not require that, so the two lengths can
    differ and the old behaviour is the compatible one.
    """
    vis = LightDarkPOMDPVisualizer(continuous_env)
    one_particle = (np.array([[100.0, 100.0]]), np.array([4.0]))
    belief_px = [one_particle] * 12

    drawn = []

    class _CountingDraw:
        def ellipse(self, *args, **kwargs):
            drawn.append(args)

    vis._draw_belief(_CountingDraw(), belief_px, 3, 2)
    assert len(drawn) == 3, "trail should be min(len(path), MAX_BELIEF_TRAIL)"

    drawn.clear()
    vis._draw_belief(_CountingDraw(), belief_px, 12, 11)
    assert len(drawn) == 10, "trail is capped at MAX_BELIEF_TRAIL"


def test_visualize_path_accepts_mismatched_belief_list_length(continuous_env, tmp_path):
    """A shorter belief list than path must render, not raise."""
    path = [np.array([float(i), 5.0]) for i in range(4)]
    out = tmp_path / "mismatched.gif"

    LightDarkPOMDPVisualizer(continuous_env).visualize_path(path, [], ["right"], out)

    with Image.open(out) as gif:
        assert getattr(gif, "n_frames") == 4
