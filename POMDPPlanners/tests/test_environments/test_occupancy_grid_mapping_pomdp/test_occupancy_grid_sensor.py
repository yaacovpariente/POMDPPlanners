# SPDX-License-Identifier: MIT

"""Tests for the ray caster and the inverse sensor model.

The cross-environment conformance suite covers the ``Environment`` contracts.
What it cannot see is whether the sensor is the sensor this environment claims
to have, and that is the whole observation model: a ray caster that stopped one
cell early, or an inverse sensor model that marked cells behind an obstacle,
would still pass every shared contract while quietly measuring the wrong thing.
"""

import math

import numpy as np
import pytest

from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_sensor import (
    NUM_HEADINGS,
    build_ray_templates,
    cast_scan,
    grid_entropy_bits,
    log_odds_from_probability,
    scan_log_odds_delta,
)


@pytest.fixture(name="template")
def _template():
    """One four-beam, full-surround template set at range 4."""
    return build_ray_templates(
        num_beams=4, field_of_view_degrees=360.0, max_range_cells=4.0
    )


def test_a_single_beam_points_along_the_heading(template):
    """Bearing zero is north, rows grow downwards, columns grow right.

    Purpose: Pins the geometry convention the rest of the environment is written
        against, because every later check reads as correct under a mirrored
        convention too. A one-beam fan is the clean way to ask: its only beam
        sits at the centre of the fan, which is the heading itself.

    Given: A one-beam narrow fan, one template per heading
    When: The nearest cell of the single beam is read for each heading
    Then: They are north, east, south and west in heading-index order

    Test type: unit
    """
    del template
    single = build_ray_templates(
        num_beams=1, field_of_view_degrees=1.0, max_range_cells=4.0
    )
    nearest = [tuple(int(v) for v in single[h][0][0, 0]) for h in range(NUM_HEADINGS)]
    assert nearest == [(-1, 0), (0, 1), (1, 0), (0, -1)]


def test_full_surround_beams_sit_at_bin_centres(template):
    """A four-beam 360-degree fan is the four diagonals, not the four axes.

    Purpose: Beams sit at bin *centres*, so a full-circle fan does not put two
        beams on the same bearing at plus and minus 180 degrees. That choice is
        invisible until someone counts beams and expects an axis-aligned one, so
        it is pinned here rather than left to be rediscovered.

    Given: Four beams over 360 degrees from a north-facing robot
    When: The nearest cell of each beam is read
    Then: The four are the diagonal neighbours

    Test type: unit
    """
    offsets, _ = template[0]
    nearest = {tuple(int(v) for v in offsets[beam, 0]) for beam in range(4)}
    assert nearest == {(-1, -1), (-1, 1), (1, -1), (1, 1)}


def test_the_default_fan_still_sweeps_every_axis_cell():
    """Bin centres do not leave the cardinal directions unswept.

    Purpose: The default 24-beam fan has no beam pointing exactly north, so the
        obvious worry is that the cells straight ahead go unobserved and leave an
        entropy floor the robot cannot get under. At 7.5 degrees off-axis the
        beam is still inside the axis cell out to the full sensor range, and
        this is what says so.

    Given: The environment's default fan -- 24 beams, 360 degrees, range 3.5
    When: The cells every beam crosses are collected
    Then: All three axis cells in each cardinal direction are among them

    Test type: unit
    """
    offsets, _ = build_ray_templates(
        num_beams=24, field_of_view_degrees=360.0, max_range_cells=3.5
    )[0]
    swept = {tuple(int(v) for v in cell) for cell in offsets.reshape(-1, 2)}
    for distance in (1, 2, 3):
        for cell in (
            (-distance, 0),
            (distance, 0),
            (0, -distance),
            (0, distance),
        ):
            assert cell in swept, f"{cell} is never swept by the default fan"


def test_turning_permutes_the_same_beam_set(template):
    """A 360-degree scan sees the same cells whichever way the robot faces.

    Purpose: Under the default field of view a turn re-measures exactly the
        cells the robot could already see, and never a new one -- which is why
        turning only ever buys the sharpening of a repeat reading, decaying to
        nothing at the log-odds clamp. If a heading swept even one cell the
        others do not, turning would start opening new ground and the trade-off
        the environment is built around would quietly change.

    Given: The four heading templates of a full-surround fan
    When: Each template's set of visited cells is collected
    Then: All four sets are identical

    Test type: unit
    """
    sets = [
        {tuple(int(v) for v in cell) for cell in template[h][0].reshape(-1, 2)}
        for h in range(NUM_HEADINGS)
    ]
    for other in sets[1:]:
        assert other == sets[0]


def test_ray_cells_are_ordered_outwards_and_within_range(template):
    """Every beam lists its cells nearest-first and never beyond the range.

    Purpose: The inverse sensor model marks "everything before the stop slot"
        free, which is only the traversed part if the ordering is monotone in
        range. An out-of-order template would mark cells behind an obstacle free.

    Given: A four-beam template at range 4
    When: The per-cell ranges of every beam are read
    Then: Each beam's real (non-padding) ranges increase and stay within range

    Test type: unit
    """
    for heading in range(NUM_HEADINGS):
        _, ranges = template[heading]
        for beam in range(ranges.shape[0]):
            real = ranges[beam][np.isfinite(ranges[beam])]
            assert np.all(np.diff(real) > 0.0)
            assert np.all(real <= 4.0 + 1e-9)


def test_beam_stops_at_the_first_occupied_cell(template):
    """A beam reports the distance to the nearest obstacle on its line.

    Purpose: The forward sensor model is the whole observation likelihood; a
        beam that reported the *furthest* obstacle would still produce plausible
        numbers on an open map and be wrong on every occluded one.

    Given: A 9x9 empty grid with obstacles two and three cells north of centre,
        and a single north-facing beam
    When: A scan is cast from the centre
    Then: The beam reports 2.0, the nearer obstacle, and is flagged as a hit

    Test type: unit
    """
    del template
    north_beam = build_ray_templates(
        num_beams=1, field_of_view_degrees=1.0, max_range_cells=4.0
    )[0]
    occupancy = np.zeros((9, 9))
    occupancy[2, 4] = 1.0
    occupancy[1, 4] = 1.0
    ranges, hit, _ = cast_scan(occupancy, 4, 4, north_beam, max_range_cells=4.0)
    assert bool(hit[0])
    assert ranges[0] == pytest.approx(2.0)


def test_beam_with_no_obstacle_reports_max_range_and_no_hit(template):
    """An open beam is a miss at maximum range, not a return.

    Purpose: The max-range miss is what makes exploration pay -- it is the case
        that marks a whole open corridor free. Reporting it as a hit would put
        occupied evidence at the end of every open sweep.

    Given: An empty 21x21 grid, so no beam meets anything inside its range
    When: A scan is cast from the centre
    Then: Every beam is a miss reporting exactly the maximum range

    Test type: unit
    """
    ranges, hit, _ = cast_scan(np.zeros((21, 21)), 10, 10, template[0], max_range_cells=4.0)
    assert not hit.any()
    assert np.allclose(ranges, 4.0)


def test_leaving_the_grid_is_a_miss_not_a_return(template):
    """A beam that runs off the grid detected nothing.

    Purpose: Off-grid is absence of world, not presence of obstacle. Treating it
        as a hit would paint a phantom wall around any unwalled map.

    Given: An empty 3x3 grid with the robot in its corner
    When: A scan is cast
    Then: No beam reports a hit

    Test type: unit
    """
    _, hit, _ = cast_scan(np.zeros((3, 3)), 0, 0, template[0], max_range_cells=4.0)
    assert not hit.any()


def test_range_is_the_centre_to_centre_distance():
    """A diagonal beam's range is the Euclidean distance, not a step count.

    Purpose: The observation likelihood is a Gaussian about this number, so a
        range quantised by the ray-stepping resolution would bias every
        diagonal beam's weight.

    Given: A four-beam full-surround fan, whose beams are the diagonals, and
        an obstacle on the north-east diagonal
    When: The scan is cast
    Then: The diagonal beam reports ``hypot(2, 2)``

    Test type: unit
    """
    template = build_ray_templates(
        num_beams=4, field_of_view_degrees=360.0, max_range_cells=5.0
    )
    occupancy = np.zeros((11, 11))
    occupancy[3, 7] = 1.0  # two north and two east of (5, 5)
    ranges, hit, _ = cast_scan(occupancy, 5, 5, template[0], max_range_cells=5.0)
    assert ranges[hit].min() == pytest.approx(math.hypot(2, 2))


def test_inverse_sensor_model_marks_free_then_occupied_then_nothing():
    """Free before the hit, occupied at it, untouched behind it.

    Purpose: This three-way split *is* the inverse sensor model. Marking behind
        the hit would claim knowledge of cells the beam never reached, which is
        the difference between mapping and guessing.

    Given: A single north-facing beam and an obstacle three cells north
    When: The scan's log-odds increments are computed
    Then: The two cells in between are negative, the obstacle positive, and
        everything further north is exactly zero

    Test type: unit
    """
    template = build_ray_templates(
        num_beams=1, field_of_view_degrees=1.0, max_range_cells=6.0
    )[0]
    occupancy = np.zeros((11, 11))
    occupancy[2, 5] = 1.0
    _, hit, stop_slot = cast_scan(occupancy, 5, 5, template, max_range_cells=6.0)
    delta = scan_log_odds_delta(
        hit=hit,
        stop_slot=stop_slot,
        template=template,
        row=5,
        col=5,
        num_rows=11,
        num_cols=11,
        free_log_odds=-1.0,
        occupied_log_odds=2.0,
    ).reshape(11, 11)

    assert delta[4, 5] == pytest.approx(-1.0)
    assert delta[3, 5] == pytest.approx(-1.0)
    assert delta[2, 5] == pytest.approx(2.0)
    assert delta[1, 5] == 0.0
    assert delta[0, 5] == 0.0


def test_max_range_miss_marks_the_whole_ray_free_and_nothing_occupied():
    """A beam that returns nothing still buys information.

    Purpose: Discarding max-range misses is the classic occupancy-grid bug, and
        it removes precisely the reward signal that makes a robot drive into
        open space.

    Given: A single beam into an empty grid
    When: The scan's log-odds increments are computed
    Then: Every cell on the ray is negative and none is positive

    Test type: unit
    """
    template = build_ray_templates(
        num_beams=1, field_of_view_degrees=1.0, max_range_cells=4.0
    )[0]
    _, hit, stop_slot = cast_scan(np.zeros((21, 21)), 10, 10, template, max_range_cells=4.0)
    delta = scan_log_odds_delta(
        hit=hit,
        stop_slot=stop_slot,
        template=template,
        row=10,
        col=10,
        num_rows=21,
        num_cols=21,
        free_log_odds=-1.0,
        occupied_log_odds=2.0,
    )
    assert delta.max() <= 0.0
    assert np.count_nonzero(delta) == template[0].shape[1]


def test_entropy_is_one_bit_per_unknown_cell_and_zero_when_certain():
    """The reward's unit is a bit, and an unknown grid holds one per cell.

    Purpose: Both the reward scale and the termination threshold are quoted in
        bits and read as fractions of the cell count, which is only meaningful
        if this holds exactly.

    Given: An all-zero log-odds grid and a strongly clamped one
    When: Their entropies are computed
    Then: The first is the cell count and the second is near zero

    Test type: unit
    """
    assert grid_entropy_bits(np.zeros((7, 5))) == pytest.approx(35.0)
    # Not exactly zero: the probability is clipped off 0 and 1 before the log,
    # so a certain cell keeps a floor of about 4e-11 bits. That floor is why
    # entropy_threshold_fraction is a fraction and not a test for zero.
    assert grid_entropy_bits(np.full((7, 5), 40.0)) == pytest.approx(0.0, abs=1e-6)
    assert grid_entropy_bits(np.full((7, 5), -40.0)) == pytest.approx(0.0, abs=1e-6)


def test_entropy_survives_log_odds_that_would_underflow_in_probability_space():
    """Log-odds is the representation for a reason.

    Purpose: The module's stated reason for holding log-odds is that repeated
        Bayesian updates underflow in probability space. A stable sigmoid is
        what makes that true rather than merely claimed.

    Given: Log-odds far outside the range where ``exp(l)`` is finite
    When: The entropy is computed
    Then: It is finite and zero, with no overflow and nothing invalid.
        Underflow is allowed and is the point: ``exp`` is only ever evaluated on
        a non-positive argument, so an extreme value flushes quietly to zero
        instead of becoming ``inf`` and then ``nan``

    Test type: unit
    """
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        value = grid_entropy_bits(np.array([[900.0, -900.0]]))
    assert np.isfinite(value)
    assert value == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.1])
def test_certain_evidence_is_rejected(probability):
    """A sensor model probability of 0 or 1 is refused, not silently clipped.

    Purpose: Infinite evidence cannot be revised by any later reading, so a cell
        given it is stuck forever. Clipping it quietly would hide that.

    Given: A degenerate or out-of-range probability
    When: It is converted to log-odds
    Then: ``ValueError``

    Test type: unit
    """
    with pytest.raises(ValueError):
        log_odds_from_probability(probability)
