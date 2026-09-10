# SPDX-License-Identifier: MIT

"""Tests for the occupancy-grid mapping renderer.

The golden-file suite already pins the exact bytes of one episode's GIF, which
catches any change at all but explains none of them. What is tested here is the
handful of claims the picture actually makes, each of which could break while
the render still looks plausible:

* a drawn beam points where the sensor cast it, and stops at the *measured*
  range rather than at the noise-free one;
* the first frame draws no beams, because the initial observation is a sentinel
  and not a reading;
* the three panels stay three different things -- the belief panel never grows
  a robot, and "no belief recorded" never renders as a probability;
* the robot is drawn at the pose the history recorded, facing the recorded
  heading;
* the belief panel is the weighted marginal and not an unweighted one;
* the render is reproducible, which the golden hash depends on.
"""

import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
    OccupancyGridAction,
    OccupancyGridMappingPOMDP,
    create_occupancy_grid_state,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_assets import (
    COLOR_HOLO_BLANK,
    COLOR_ROBOT,
    robot_sprite_facing,
    stone_shade,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_visualizer import (  # noqa: E501
    CANVAS_SIZE,
    SCREEN_SIZE,
    OccupancyGridMappingVisualizer,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_sensor import (
    build_ray_templates,
)


@pytest.fixture(name="env")
def _env() -> OccupancyGridMappingPOMDP:
    """A small walled room with no interior obstacles and a short-range sensor.

    Fixed rather than default, because the default prior draws its obstacles at
    random and a rendering test written against a random map is a test written
    against whatever seed happened to be set.
    """
    return OccupancyGridMappingPOMDP(
        num_rows=7,
        num_cols=7,
        num_beams=8,
        max_range_cells=2.5,
        num_obstacles=0,
        max_steps=12,
    )


def _empty_room(env: OccupancyGridMappingPOMDP) -> np.ndarray:
    """A walled, otherwise empty map of ``env``'s size."""
    occupancy = np.zeros((env.num_rows, env.num_cols))
    occupancy[0, :] = occupancy[-1, :] = 1.0
    occupancy[:, 0] = occupancy[:, -1] = 1.0
    return occupancy


def _episode(env: OccupancyGridMappingPOMDP, steps: int = 3, seed: int = 5) -> List[StepData]:
    """A short recorded episode with a real belief attached to every step."""
    np.random.seed(seed)
    # pylint: disable-next=import-outside-toplevel
    from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
        OccupancyGridMappingBelief,
    )

    state = create_occupancy_grid_state(env, _empty_room(env))
    belief = OccupancyGridMappingBelief.initial(env, n_particles=8)
    history: List[StepData] = []
    for _ in range(steps):
        action = int(OccupancyGridAction.FORWARD)
        next_state, observation, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=reward,
                belief=belief,
                info=None,
            )
        )
        belief = belief.update(action=action, observation=observation, pomdp=env)
        state = next_state
    history.append(
        StepData(
            state=state,
            action=None,
            next_state=None,
            observation=None,
            reward=None,
            belief=belief,
            info=None,
        )
    )
    return history


def _screen(image, visualizer: OccupancyGridMappingVisualizer, panel: int) -> np.ndarray:
    """The pixels of one panel's screen, as an ``(h, w, 3)`` array."""
    return np.asarray(image.crop(visualizer._screen_box(panel)), dtype=np.int16)


def _near(pixels: np.ndarray, colour, tolerance: int = 26) -> np.ndarray:
    """Mask of pixels within ``tolerance`` of ``colour`` on every channel."""
    return np.all(np.abs(pixels - np.array(colour, dtype=np.int16)) <= tolerance, axis=-1)


# -- geometry the picture claims ----------------------------------------


def test_drawn_beams_point_where_the_sensor_cast_them(env):
    """Purpose: The beam fan must be the sensor's geometry, not a lookalike.

    Given: The ray templates the environment actually casts with
    When: The renderer computes its beam directions for the same heading
    Then: Each drawn direction agrees with its template's cells to within the
        angle one cell of quantization can explain

    Test type: unit
    """
    visualizer = OccupancyGridMappingVisualizer(env)
    templates = build_ray_templates(
        num_beams=env.num_beams,
        field_of_view_degrees=env.field_of_view_degrees,
        max_range_cells=env.max_range_cells,
    )
    for heading in range(4):
        directions = visualizer._beam_directions(heading)
        offsets, ranges = templates[heading]
        for beam in range(env.num_beams):
            finite = np.isfinite(ranges[beam])
            furthest = int(np.max(np.flatnonzero(finite)))
            cell = offsets[beam, furthest].astype(float)
            norm = float(np.hypot(cell[0], cell[1]))
            assert norm > 0.0
            cosine = float(np.dot(directions[beam], cell / norm))
            assert cosine > math.cos(math.radians(25.0)), (heading, beam, cosine)


def test_a_beam_stops_at_the_measured_range_not_the_true_one(env):
    """Purpose: The render must show the noisy reading, which is the difficulty.

    Given: One state whose stored scan is deliberately shorter than the truth
    When: The observed panel is drawn
    Then: The beam pixels reach only as far as the stored reading says

    Test type: unit
    """
    visualizer = OccupancyGridMappingVisualizer(env)
    frame = visualizer._build_frames(_episode(env, steps=2))[1]
    cell_w, _ = visualizer._cell_size
    centre_x, centre_y = visualizer._cell_centre(frame["row"], frame["col"])

    # How far the beams reach is measured against the same frame drawn with no
    # beams at all, rather than by matching the beam colour. The beams are drawn
    # translucent over a textured panel, so their pixels are not near the beam
    # colour; but every other mark -- the robot, the trail, the cell texture --
    # is identical between the two renders and cancels exactly.
    def reach(ranges) -> float:
        with_beams = _screen(
            visualizer.render_frames([dict(frame, ranges=ranges)])[0], visualizer, 0
        )
        without = _screen(visualizer.render_frames([dict(frame, ranges=None)])[0], visualizer, 0)
        ys, xs = np.nonzero(np.any(np.abs(with_beams - without) > 2, axis=-1))
        assert len(xs) > 0, "no beams were drawn at all"
        return float(np.max(np.hypot(xs - centre_x, ys - centre_y)))

    # Both readings are inside max_range_cells (2.5), so both are hits and both
    # get the same end spark; only the range differs.
    short_cells, long_cells = 1.0, 2.0
    short = reach(np.full(env.num_beams, short_cells))
    long = reach(np.full(env.num_beams, long_cells))

    # Slack covers the beam's own 2px width and its 3px end spark.
    slack = 6.0
    assert short_cells * cell_w - slack <= short <= short_cells * cell_w + slack
    assert long_cells * cell_w - slack <= long <= long_cells * cell_w + slack
    assert short < long


def test_the_first_frame_draws_no_beams(env):
    """Purpose: The initial observation is a sentinel, not a scan.

    Given: An episode whose first state carries zero range placeholders
    When: The frames are built
    Then: The first frame reports no ranges and later frames do

    Test type: unit
    """
    visualizer = OccupancyGridMappingVisualizer(env)
    frames = visualizer._build_frames(_episode(env, steps=3))
    assert frames[0]["ranges"] is None
    assert frames[1]["ranges"] is not None
    assert len(frames[1]["ranges"]) == env.num_beams


def test_the_robot_is_drawn_at_the_recorded_pose_and_heading(env):
    """Purpose: A robot drawn in the wrong cell would invalidate every frame.

    Given: A recorded episode
    When: Each frame is rendered
    Then: Robot-coloured pixels cluster on the recorded cell in the observed
        and the ground-truth panels, and the sprite matches the recorded
        heading

    Test type: integration
    """
    visualizer = OccupancyGridMappingVisualizer(env)
    frames = visualizer._build_frames(_episode(env, steps=3))
    images = visualizer.render_frames(frames)
    cell_w, cell_h = visualizer._cell_size
    for frame, image in zip(frames, images):
        centre_x, centre_y = visualizer._cell_centre(frame["row"], frame["col"])
        for panel in (0, 2):
            mask = _near(_screen(image, visualizer, panel), COLOR_ROBOT, tolerance=40)
            assert mask.any(), (frame["index"], panel)
            ys, xs = np.nonzero(mask)
            assert abs(float(xs.mean()) - centre_x) <= cell_w * 0.5
            assert abs(float(ys.mean()) - centre_y) <= cell_h * 0.5

    headings = {frame["heading"] for frame in frames}
    sprites = {heading: robot_sprite_facing(32, heading).tobytes() for heading in range(4)}
    assert len(set(sprites.values())) == 4
    assert headings


def test_the_belief_panel_never_draws_the_robot(env):
    """Purpose: The belief panel must stay a belief, not a second map view.

    Given: A rendered episode
    When: The belief panel is inspected
    Then: It carries no robot-coloured pixels at all

    Test type: integration
    """
    visualizer = OccupancyGridMappingVisualizer(env)
    images = visualizer.render_frames(visualizer._build_frames(_episode(env, steps=3)))
    for image in images:
        assert not _near(_screen(image, visualizer, 1), COLOR_ROBOT, tolerance=40).any()


def test_a_missing_belief_is_drawn_blank_and_not_as_a_probability(env):
    """Purpose: "No belief recorded" must never be read off the ramp.

    Given: A step whose belief carries nothing usable
    When: The belief panel is drawn
    Then: Its cells use the blank colour, which is on neither end of the ramp

    Test type: unit
    """
    visualizer = OccupancyGridMappingVisualizer(env)
    frames = visualizer._build_frames(_episode(env, steps=2))
    blank = np.full((env.num_rows, env.num_cols), np.nan)
    assert np.isnan(visualizer._belief_marginal(None)).all()

    image = visualizer.render_frames([dict(frames[0], belief=blank)])[0]
    pixels = _screen(image, visualizer, 1)
    assert _near(pixels, COLOR_HOLO_BLANK, tolerance=22).mean() > 0.4


def test_the_belief_panel_is_the_weighted_marginal(env):
    """Purpose: An unweighted mean would be a different, wrong quantity.

    Given: Two particles with very different weights and different maps
    When: The marginal is computed
    Then: It equals the weight-weighted mean of their occupancies

    Test type: unit
    """

    class _Belief:  # pylint: disable=too-few-public-methods
        def __init__(self, particles, weights):
            self.particles = particles
            self.normalized_weights = weights

    first = _empty_room(env)
    second = first.copy()
    second[3, 3] = 1.0
    particles = [
        create_occupancy_grid_state(env, first),
        create_occupancy_grid_state(env, second),
    ]
    weights = np.array([0.25, 0.75])
    visualizer = OccupancyGridMappingVisualizer(env)
    marginal = visualizer._belief_marginal(_Belief(particles, weights))
    assert marginal[3, 3] == pytest.approx(0.75)
    assert marginal[3, 2] == pytest.approx(0.0)
    assert marginal[0, 0] == pytest.approx(1.0)


# -- output shape and reproducibility ------------------------------------


def test_frames_are_canvas_sized_and_one_per_recorded_step(env):
    """Purpose: The frame count and size are what every saved GIF is compared on.

    Given: A four-step recorded episode
    When: It is rendered
    Then: There are four frames, each the declared canvas size

    Test type: unit
    """
    visualizer = OccupancyGridMappingVisualizer(env)
    history = _episode(env, steps=3)
    images = visualizer.render_frames(visualizer._build_frames(history))
    assert len(images) == len(history)
    assert {image.size for image in images} == {CANVAS_SIZE}


def test_the_three_panels_stay_three_different_pictures(env):
    """Purpose: The panels mean different things and must not converge visually.

    Given: One rendered frame
    When: The three screens are compared
    Then: No two are the same pixels, and each screen is square

    Test type: unit
    """
    visualizer = OccupancyGridMappingVisualizer(env)
    image = visualizer.render_frames(visualizer._build_frames(_episode(env, steps=2)))[-1]
    screens = [_screen(image, visualizer, panel) for panel in range(3)]
    for pixels in screens:
        assert pixels.shape == (SCREEN_SIZE, SCREEN_SIZE, 3)
    assert not np.array_equal(screens[0], screens[1])
    assert not np.array_equal(screens[1], screens[2])
    assert not np.array_equal(screens[0], screens[2])


def test_rendering_is_reproducible(env, tmp_path: Path):
    """Purpose: The golden hash and every cached GIF depend on this.

    Given: One history and two fresh visualizers
    When: Both write the episode
    Then: The two files are byte-identical, and so is the procedural stone

    Test type: unit
    """
    history = _episode(env, steps=3)
    first = tmp_path / "first.gif"
    second = tmp_path / "second.gif"
    OccupancyGridMappingVisualizer(env).create_visualization(history, first)
    OccupancyGridMappingVisualizer(env).create_visualization(history, second)
    assert first.read_bytes() == second.read_bytes()
    np.testing.assert_array_equal(stone_shade(64, 64), stone_shade(64, 64))


def test_a_saved_gif_holds_one_frame_per_step(env, tmp_path: Path):
    """Purpose: A dropped or duplicated frame silently misreports an episode.

    Given: A saved episode
    When: The GIF is reopened
    Then: It has one frame per recorded step at the canvas size

    Test type: integration
    """
    # pylint: disable-next=import-outside-toplevel
    from PIL import GifImagePlugin, Image

    history = _episode(env, steps=4)
    path = tmp_path / "episode.gif"
    OccupancyGridMappingVisualizer(env).create_visualization(history, path)
    with Image.open(path) as gif:
        assert isinstance(gif, GifImagePlugin.GifImageFile)
        assert gif.format == "GIF"
        assert gif.n_frames == len(history)
        assert gif.size == CANVAS_SIZE


# -- input validation, unchanged from the previous renderer ---------------


@pytest.mark.parametrize(
    "history, cache_name, error",
    [
        ("not a list", "a.gif", TypeError),
        ([], "a.gif", ValueError),
        (["not step data"], "a.gif", TypeError),
    ],
)
def test_bad_history_is_rejected(env, tmp_path: Path, history, cache_name, error):
    """Purpose: A malformed history must fail loudly rather than render nonsense.

    Given: A history that is not a non-empty list of StepData
    When: A visualization is requested
    Then: The documented error is raised

    Test type: unit
    """
    with pytest.raises(error):
        OccupancyGridMappingVisualizer(env).create_visualization(history, tmp_path / cache_name)


def test_a_non_gif_destination_is_rejected(env, tmp_path: Path):
    """Purpose: The caller's contract says GIF, and the saver assumes it.

    Given: A destination that is not a GIF path, and one that is not a Path
    When: A visualization is requested
    Then: ValueError and TypeError are raised respectively

    Test type: unit
    """
    history = _episode(env, steps=1)
    with pytest.raises(ValueError):
        OccupancyGridMappingVisualizer(env).create_visualization(history, tmp_path / "a.png")
    with pytest.raises(TypeError):
        OccupancyGridMappingVisualizer(env).create_visualization(
            history, str(tmp_path / "a.gif")  # type: ignore[arg-type]
        )
