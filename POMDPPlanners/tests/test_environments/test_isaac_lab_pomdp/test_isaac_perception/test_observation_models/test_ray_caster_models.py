# SPDX-License-Identifier: MIT

"""Unit tests for the ray-cast observation models.

The geometry is checked against hand-computed ranges rather than a golden array: a ray caster that
is subtly wrong still returns plausible-looking numbers, and a planner reading them would simply
localise badly rather than fail.
"""

from typing import Any, Dict

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
    HeightScanObservationModel,
    RayCasterObservationModel,
    grid_scan_pattern,
)

ORIGIN = {"base_pose": np.array([0.0, 0.0, 0.0])}


def _four_ray_model(**overrides: Any) -> RayCasterObservationModel:
    """A 4-ray full-circle caster with one disc 1 m ahead, radius 0.5 m."""
    settings: Dict[str, Any] = {
        "channel": "lidar",
        "pose_channel": "base_pose",
        "num_rays": 4,
        "max_range": 5.0,
        "range_std": 1e-6,
        "obstacle_centers": [(1.0, 0.0)],
        "obstacle_radii": [0.5],
    }
    settings.update(overrides)
    return RayCasterObservationModel(**settings)


def test_ranges_match_the_hand_computed_disc_geometry() -> None:
    """The forward ray hits the disc's near face; the others miss and saturate.

    Purpose: Validates the ray-disc intersection against a case solvable by hand

    Given: A 4-ray full-circle caster at the origin facing +x, and a disc centred at (1, 0)
        with radius 0.5
    When: The clean ranges are computed
    Then: The +x ray reads 0.5 and the other three read the max range

    Test type: unit
    """
    ranges = _four_ray_model().clean_ranges(ORIGIN)
    assert ranges == pytest.approx([0.5, 5.0, 5.0, 5.0])


def test_ring_rotates_with_the_heading() -> None:
    """A LiDAR is body-fixed, so turning the robot must move which ray sees the obstacle.

    Purpose: Validates that ray bearings are taken relative to the pose's yaw

    Given: The same disc ahead in the world frame
    When: The robot's yaw is rotated by +pi/2
    Then: The obstacle moves from ray 0 to the ray one quarter-turn back

    Test type: unit
    """
    model = _four_ray_model()
    turned = model.clean_ranges({"base_pose": np.array([0.0, 0.0, np.pi / 2.0])})
    assert turned[3] == pytest.approx(0.5)
    assert turned[0] == pytest.approx(5.0)


def test_a_ray_starting_inside_a_disc_reports_the_far_surface() -> None:
    """Standing inside an obstacle must not report a negative or zero range.

    Purpose: Validates the near-root-behind-the-origin branch of the intersection

    Given: A caster placed at the centre of a radius-0.5 disc
    When: The clean ranges are computed
    Then: Every ray reads the disc radius, the distance to the surface ahead of it

    Test type: unit
    """
    model = _four_ray_model(obstacle_centers=[(0.0, 0.0)], obstacle_radii=[0.5])
    assert model.clean_ranges(ORIGIN) == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_an_obstacle_behind_the_ray_is_not_reported() -> None:
    """A ray only sees forward; a disc behind the origin must not shorten it.

    Purpose: Validates that intersections at negative t are discarded

    Given: A single ray pointing +x and a disc entirely behind the origin
    When: The clean ranges are computed
    Then: The ray saturates at the max range

    Test type: unit
    """
    model = _four_ray_model(num_rays=1, field_of_view=0.0, obstacle_centers=[(-2.0, 0.0)])
    assert model.clean_ranges(ORIGIN) == pytest.approx([5.0])


def test_the_nearest_of_several_discs_wins() -> None:
    """A ray reports the first surface it meets, not the last one tested.

    Purpose: Validates the minimum over obstacles

    Given: Two discs on the +x axis at 1 m and 3 m
    When: The forward ray is cast
    Then: It reports the near disc's face

    Test type: unit
    """
    model = _four_ray_model(obstacle_centers=[(3.0, 0.0), (1.0, 0.0)], obstacle_radii=[0.5, 0.25])
    assert model.clean_ranges(ORIGIN)[0] == pytest.approx(0.75)


def test_no_obstacles_saturates_every_ray() -> None:
    """An empty scene is a legitimate configuration, not a degenerate one.

    Purpose: Validates the zero-obstacle fast path

    Given: A caster with no obstacles configured
    When: The clean ranges are computed
    Then: Every ray reads the max range

    Test type: unit
    """
    model = _four_ray_model(obstacle_centers=None, obstacle_radii=None)
    assert model.clean_ranges(ORIGIN) == pytest.approx([5.0] * 4)


def test_perceived_ranges_stay_within_the_sensor_bounds() -> None:
    """A negative or over-range reading is not something a real sensor can emit.

    Purpose: Validates the clipping applied after the range noise

    Given: A caster with noise large against the distance to the obstacle
    When: Many readings are drawn
    Then: Every reading lies in [0, max_range]

    Test type: unit
    """
    np.random.seed(0)
    model = _four_ray_model(range_std=3.0)
    draws = np.stack([model.perceive(ORIGIN) for _ in range(500)])
    assert draws.min() >= 0.0
    assert draws.max() <= 5.0


def test_range_density_peaks_at_the_true_pose() -> None:
    """Localisation only works if the likelihood prefers the pose that explains the scan.

    Purpose: Validates that the range density discriminates between candidate poses

    Given: A clean scan taken from the origin
    When: It is scored at the origin and at a pose half a metre away
    Then: The origin scores higher

    Test type: unit
    """
    model = _four_ray_model(range_std=0.05)
    reading = model.clean_ranges(ORIGIN)
    at_truth = model.log_probability(ORIGIN, reading)
    displaced = model.log_probability({"base_pose": np.array([0.5, 0.0, 0.0])}, reading)
    assert at_truth > displaced


def test_range_density_rejects_a_reading_of_the_wrong_width() -> None:
    """A scan of the wrong length cannot have come from this sensor.

    Purpose: Validates the shape guard on the range density

    Given: A 4-ray caster
    When: A 3-entry reading is scored
    Then: The score is -inf

    Test type: unit
    """
    assert _four_ray_model().log_probability(ORIGIN, np.zeros(3)) == float("-inf")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"num_rays": 0}, "num_rays"),
        ({"max_range": 0.0}, "max_range"),
        ({"range_std": 0.0}, "range_std"),
        ({"obstacle_radii": [0.5, 0.5]}, "obstacle_radii"),
    ],
)
def test_invalid_sensor_configuration_is_rejected_at_construction(
    overrides: Dict[str, Any], message: str
) -> None:
    """A misconfigured sensor must fail at build time, not thousands of simulations in.

    Purpose: Validates the construction-time guards on the ray caster

    Given: A configuration with one invalid setting
    When: The model is constructed
    Then: ValueError is raised naming the offending setting

    Test type: unit
    """
    with pytest.raises(ValueError, match=message):
        _four_ray_model(**overrides)


def test_height_scan_reads_the_obstacle_under_each_grid_point() -> None:
    """The scanner's job is to say what is underfoot, disc by disc.

    Purpose: Validates the height lookup over the body-frame grid

    Given: A two-point grid with one point over a 0.4 m obstacle and one over bare floor
    When: The clean heights are computed
    Then: They read 0.4 and 0.0 respectively

    Test type: unit
    """
    model = HeightScanObservationModel(
        channel="height_scan",
        pattern=[(0.0, 0.0), (2.0, 0.0)],
        obstacle_centers=[(0.0, 0.0)],
        obstacle_radii=[0.5],
        obstacle_heights=[0.4],
        height_std=1e-6,
    )
    assert model.clean_heights(ORIGIN) == pytest.approx([0.4, 0.0])


def test_height_scan_grid_rotates_with_the_heading() -> None:
    """The grid is body-fixed, so a turn changes which world point each sample covers.

    Purpose: Validates that the scan pattern is rotated into the world frame

    Given: A single grid point 2 m ahead and an obstacle 2 m to the robot's left
    When: The robot turns a quarter-turn left
    Then: The sample moves onto the obstacle

    Test type: unit
    """
    model = HeightScanObservationModel(
        channel="height_scan",
        pattern=[(2.0, 0.0)],
        obstacle_centers=[(0.0, 2.0)],
        obstacle_radii=[0.5],
        obstacle_heights=[0.3],
        height_std=1e-6,
    )
    assert model.clean_heights(ORIGIN) == pytest.approx([0.0])
    turned = model.clean_heights({"base_pose": np.array([0.0, 0.0, np.pi / 2.0])})
    assert turned == pytest.approx([0.3])


def test_height_scan_defaults_to_a_square_grid_of_the_requested_size() -> None:
    """The default pattern must have the width the configuration asks for.

    Purpose: Validates the default grid construction

    Given: A scanner built with grid_size 3 and no explicit pattern
    When: A scan is taken
    Then: It has nine samples

    Test type: unit
    """
    np.random.seed(0)
    model = HeightScanObservationModel(channel="height_scan", grid_size=3, grid_extent=0.6)
    assert model.perceive(ORIGIN).shape == (9,)


def test_height_scan_density_prefers_the_pose_that_explains_the_scan() -> None:
    """A height scan is a localisation cue only if its likelihood is pose-sensitive.

    Purpose: Validates that the height density discriminates between candidate poses

    Given: A clean scan taken while standing on an obstacle
    When: It is scored on the obstacle and well off it
    Then: The on-obstacle pose scores higher

    Test type: unit
    """
    model = HeightScanObservationModel(
        channel="height_scan",
        pattern=[(0.0, 0.0)],
        obstacle_centers=[(0.0, 0.0)],
        obstacle_radii=[0.5],
        obstacle_heights=[0.4],
        height_std=0.02,
    )
    reading = model.clean_heights(ORIGIN)
    off = {"base_pose": np.array([3.0, 0.0, 0.0])}
    assert model.log_probability(ORIGIN, reading) > model.log_probability(off, reading)


def test_height_scan_rejects_obstacle_arrays_of_unequal_length() -> None:
    """Centres, radii and heights describe the same discs, so a mismatch is a config bug.

    Purpose: Validates the construction-time guard on the obstacle description

    Given: Two centres but one radius
    When: The scanner is constructed
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="equal length"):
        HeightScanObservationModel(
            channel="height_scan",
            obstacle_centers=[(0.0, 0.0), (1.0, 1.0)],
            obstacle_radii=[0.5],
            obstacle_heights=[0.4, 0.4],
        )


class TestGridScanPattern:
    """The grid layout a height-scan model must share with the sensor it predicts.

    Ordering is the trap. IsaacLab flattens ``meshgrid(x, y, indexing="xy")``, so the reading runs
    x fastest. A model laying the same points out y-fastest scores every particle against permuted
    cells, and the belief degrades in a way that reads as sensor noise.
    """

    def test_cell_count_matches_the_sensor_grid(self) -> None:
        """A width mismatch makes every likelihood -inf, which looks like a dead belief.

        Purpose: Validates the number of cells for a given extent and resolution

        Given: A 0.8 x 0.6 m grid at 0.1 m resolution
        When: The pattern is built
        Then: It has 9 x 7 = 63 cells

        Test type: unit
        """
        assert grid_scan_pattern((0.8, 0.6), 0.1).shape == (63, 2)

    def test_x_varies_fastest_within_a_row(self) -> None:
        """This is IsaacLab's own flattening order, and it is not the intuitive one.

        Purpose: Validates the cell ordering against IsaacLab's "xy" meshgrid

        Given: A 0.2 x 0.1 m grid at 0.1 m resolution
        When: The pattern is built
        Then: The first three cells share a y and sweep x

        Test type: unit
        """
        pattern = grid_scan_pattern((0.2, 0.1), 0.1)
        assert pattern[:3, 1] == pytest.approx(np.full(3, -0.05))
        assert pattern[:3, 0] == pytest.approx(np.array([-0.1, 0.0, 0.1]))

    def test_a_non_positive_resolution_is_rejected(self) -> None:
        """A zero resolution would build an empty or unbounded grid.

        Purpose: Validates rejection of a non-positive resolution

        Given: A resolution of zero
        When: The pattern is built
        Then: ValueError is raised

        Test type: unit
        """
        with pytest.raises(ValueError, match="resolution must be positive"):
            grid_scan_pattern((0.8, 0.6), 0.0)

    def test_the_pattern_drives_the_height_scan_model(self) -> None:
        """The pattern exists to be handed to the model, so the pairing is the contract.

        Purpose: Validates that a model built on the pattern reports one height per cell

        Given: A height-scan model built with a 0.2 x 0.1 m pattern over one obstacle disc
        When: Clean heights are computed at the origin
        Then: There is one height per pattern cell and the covered cells read the disc height

        Test type: unit
        """
        pattern = grid_scan_pattern((0.2, 0.1), 0.1)
        model = HeightScanObservationModel(
            channel="height_scan",
            pattern=pattern,
            obstacle_centers=[(0.0, 0.0)],
            obstacle_radii=[0.06],
            obstacle_heights=[0.3],
        )
        heights = model.clean_heights({"base_pose": np.zeros(3)})
        assert heights.shape == (pattern.shape[0],)
        assert heights.max() == pytest.approx(0.3)
