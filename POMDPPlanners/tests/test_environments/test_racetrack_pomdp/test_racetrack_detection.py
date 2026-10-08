# SPDX-License-Identifier: MIT

"""Tests for the arithmetic behind "was there something there, and how sure are we?".

Every piece here is shared by both arms of the matched pair and mirrored by the torch model,
so a change to one of these functions moves the POMDP arm's radar likelihood and the MDP
arm's slot flags at once. That is the reason they are tested apart from either model: these
tests are pure NumPy and never construct an environment.

What used to live in this file — the occupancy grid's soft rasteriser and its cell-mismatch
scorer — is gone with the grid itself. What survives is the detection composition, the two
closed-form densities the likelihood is built from, the clutter model, and the packing rule
that decides which detection lands in which rank.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.racetrack_pomdp.racetrack_detection import (
    bernoulli_log_prob,
    cauchy_draw,
    cauchy_log_prob,
    detected_probabilities,
    gaussian_log_prob,
    pack_detections,
    validate_detection_rates,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    DETECTION_PRESENT,
    DETECTION_REL_VX,
    DETECTION_REL_VY,
    DETECTION_REL_X,
    DETECTION_REL_Y,
    DETECTION_SLOT_WIDTH,
)

# One report as the world hands it over: [rel_x, rel_y, rel_vx, rel_vy], with no presence
# flag — pack_detections is what decides which slot a report lands in and sets the flag.
_REPORT_WIDTH = 4


def test_equal_detection_rates_reproduce_the_symmetric_squash_exactly():
    """Miss and false alarm set equal give back the two-sided squash they generalise.

    Purpose: The detection model replaced a symmetric squash into [m, 1 - m], and the claim
        that it generalises rather than replaces it is what makes every earlier measurement
        still comparable — a claim worth pinning rather than asserting in a docstring

    Given: Occupancy probabilities across the whole unit interval, and one rate m = 0.05
    When: They are pushed through the detection channel with both rates at m
    Then: Every entry equals m + (1 - 2m) q, the old formula, to 1e-15

    Test type: unit
    """
    occupancy = np.linspace(0.0, 1.0, 101)
    rate = 0.05

    detected = detected_probabilities(occupancy, miss_prob=rate, false_alarm_prob=rate)

    np.testing.assert_allclose(detected, rate + (1.0 - 2.0 * rate) * occupancy, atol=1e-15)


def test_the_two_detection_rates_bound_the_probabilities_from_either_side():
    """A configured lossy radar bounds the probabilities away from 0 and 1 on both sides.

    Purpose: Above zero the rates are what a lossy radar is modelled with, and the bound is
        the property that makes them one: no occupancy, however certain, can drive a slot's
        reported probability past the rate that contradicts it

    Given: Occupancy probabilities spanning [0, 1] and two different rates
    When: They are pushed through the detection channel
    Then: Nothing falls below the false-alarm rate or above 1 - the miss rate, and the two
        ends are attained exactly at q = 0 and q = 1

    Test type: unit
    """
    occupancy = np.linspace(0.0, 1.0, 101)
    miss, false_alarm = 0.05, 0.02

    detected = detected_probabilities(occupancy, miss_prob=miss, false_alarm_prob=false_alarm)

    assert np.all(detected >= false_alarm - 1e-15)
    assert np.all(detected <= 1.0 - miss + 1e-15)
    assert detected[0] == pytest.approx(false_alarm)
    assert detected[-1] == pytest.approx(1.0 - miss)


def test_bernoulli_log_prob_matches_the_counted_form_when_probabilities_are_uniform():
    """The scorer is the plain Bernoulli sum, checked against a closed form.

    Purpose: Every presence term in either arm routes through this one function, so an error
        of a constant factor here would rescale every particle weight while still ranking
        them plausibly

    Given: A uniform 0.25 probability array of 20 entries and an observation setting 3 of them
    When: The log-likelihood is taken
    Then: It equals 3 * log(0.25) + 17 * log(0.75)

    Test type: unit
    """
    probabilities = np.full(20, 0.25)
    observed = np.zeros(20, dtype=bool)
    observed[:3] = True

    scored = bernoulli_log_prob(probabilities, observed)

    assert scored == pytest.approx(3 * np.log(0.25) + 17 * np.log(0.75))


def test_a_certain_slot_does_not_produce_an_infinite_penalty():
    """A probability of exactly 0 or 1 is clipped, so one slot cannot annihilate a particle.

    Purpose: Without the clip a zero-rate model scoring a single disagreeing slot returns
        -inf, the particle's weight underflows to zero, and the filter loses it permanently
        for a one-slot error

    Given: An array of exact zeros and ones, and an observation that disagrees on every entry
    When: The log-likelihood is taken
    Then: It is finite and strictly negative

    Test type: unit
    """
    probabilities = np.array([0.0, 1.0, 0.0, 1.0])
    observed = probabilities < 0.5

    scored = bernoulli_log_prob(probabilities, observed)

    assert np.isfinite(scored)
    assert scored < 0.0


def test_gaussian_log_prob_sums_the_closed_form_over_every_entry():
    """The Gaussian term is the isotropic density summed, normaliser included.

    Purpose: Four of the POMDP arm's five likelihood terms are this function, so dropping the
        normaliser would leave every ranking intact while making the density improper — and
        the MDP arm's clutter comparison, which weighs a density against a bare rate, would
        then be measuring the wrong quantity

    Given: A (2, 2) residual block of 0.5 m entries at a 0.5 m width
    When: The log-density is taken
    Then: It equals -4 * 0.5 * (0.5 / 0.5)^2 / ... the hand-written sum, and a zero residual
        scores strictly higher

    Test type: unit
    """
    deviation = np.full((2, 2), 0.5)
    std = 0.5

    scored = gaussian_log_prob(deviation, std)
    expected = -0.5 * 4 * (0.5 / 0.5) ** 2 - 0.5 * 4 * np.log(2.0 * np.pi * std**2)

    assert scored == pytest.approx(float(expected), rel=1e-12)
    assert gaussian_log_prob(np.zeros((2, 2)), std) > scored


def test_the_cauchy_clutter_density_keeps_a_distant_report_at_a_log_cost():
    """A far-out phantom is charged logarithmically, not quadratically.

    Purpose: This is why the clutter model is Cauchy rather than Gaussian. The detection
        ranks have no window to be uniform over, so a genuinely distant report has to stay
        cheap enough that the particle survives it; a Gaussian tail would make one stray
        detection cost hundreds of nats

    Given: One report at the scale, and one at forty times the scale, at an 18 m scale
    When: Both are scored, and the far one is compared with the same residual under a Gaussian
    Then: Both are finite, the near one scores higher, and the far one costs under 10 nats
        where the Gaussian of the same width would cost over 700

    Test type: unit
    """
    scale = 18.0
    near = cauchy_log_prob(np.array([scale]), scale)
    far = cauchy_log_prob(np.array([40.0 * scale]), scale)

    assert near == pytest.approx(-np.log(np.pi * scale) - np.log(2.0), rel=1e-12)
    assert np.isfinite(far)
    assert near > far
    assert near - far < 10.0
    assert near - gaussian_log_prob(np.array([40.0 * scale]), scale) > 700.0


def test_cauchy_draws_are_centred_on_zero_with_a_tail_no_gaussian_has():
    """The sampler is the inverse transform of the density it is paired with.

    Purpose: A sampler and a scorer that disagree make every self-consistency measurement in
        this package meaningless, and a Cauchy is the case where the disagreement is easiest
        to miss — its mean does not exist, so the usual "check the sample mean" does not fire

    Given: 20000 draws at a scale of 2.0 from a fixed seed
    When: Their median and their tail are measured
    Then: The median sits within 0.1 of zero, half the draws fall inside +/- the scale as the
        quartiles require, and the extreme draw is more than ten scales out

    Test type: unit
    """
    np.random.seed(11)

    draws = cauchy_draw((20000, 1), 2.0)

    assert abs(float(np.median(draws))) < 0.1
    assert abs(float(np.mean(np.abs(draws) <= 2.0)) - 0.5) < 0.02
    assert float(np.max(np.abs(draws))) > 20.0


def test_pack_detections_orders_reports_by_range_and_pads_the_unused_slots():
    """The packed reading is nearest-first, which is what rank association pairs against.

    Purpose: The likelihood matches the particle's i-th visible slot to the reading's i-th
        detection, so the order this function imposes *is* the association rule. Packing in
        arrival order instead would silently pair every detection with the wrong slot

    Given: Three four-wide reports handed over out of order — at 20 m, 5 m and roughly
        12.4 m — and four slots to pack them into
    When: They are packed
    Then: The rows come back nearest-first with both relative velocity components carried
        along, the first three are flagged detected, and the fourth is left at zero

    Test type: unit
    """
    reports = np.array([[20.0, 0.0, 1.0, -0.5], [5.0, 0.0, -2.0, 0.25], [12.0, 3.0, 0.5, 1.5]])

    packed = pack_detections(reports, 4)

    assert packed.shape == (4, DETECTION_SLOT_WIDTH)
    np.testing.assert_array_equal(packed[:, DETECTION_PRESENT], [1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(
        packed[0, DETECTION_REL_X : DETECTION_REL_VY + 1], [5.0, 0.0, -2.0, 0.25]
    )
    np.testing.assert_allclose(
        packed[1, DETECTION_REL_X : DETECTION_REL_VY + 1], [12.0, 3.0, 0.5, 1.5]
    )
    np.testing.assert_allclose(
        packed[2, DETECTION_REL_X : DETECTION_REL_VY + 1], [20.0, 0.0, 1.0, -0.5]
    )
    np.testing.assert_array_equal(packed[3], np.zeros(DETECTION_SLOT_WIDTH))


def test_pack_detections_carries_both_velocity_components_through_the_reordering():
    """Reordering moves whole rows: neither velocity column is dropped or swapped.

    Purpose: A detection now carries the vehicle's full relative velocity rather than a
        single closing rate, and the packer writes that pair into the slot by a column
        slice. A slice that was one entry short would silently zero every crossing rate,
        and a transposed one would report the crossing rate as the closing rate — both are
        plausible-looking readings the filter would absorb without complaint. Checking the
        two components separately, with values that differ in sign and magnitude, is what
        separates those two failures from the correct packing

    Given: Two reports whose ranges put them in the opposite order from the one handed over,
        each carrying a distinct rel_vx and rel_vy that no swap could reproduce
    When: They are packed into two slots
    Then: The nearer report is in slot zero with both of its own components in their own
        columns, the further one is in slot one with its own, and neither pair is swapped

    Test type: unit
    """
    reports = np.array([[30.0, 0.0, 1.0, -7.0], [4.0, 0.0, -3.0, 2.0]])

    packed = pack_detections(reports, 2)

    assert packed[0, DETECTION_REL_X] == pytest.approx(4.0)
    assert packed[0, DETECTION_REL_VX] == pytest.approx(-3.0)
    assert packed[0, DETECTION_REL_VY] == pytest.approx(2.0)
    assert packed[1, DETECTION_REL_X] == pytest.approx(30.0)
    assert packed[1, DETECTION_REL_VX] == pytest.approx(1.0)
    assert packed[1, DETECTION_REL_VY] == pytest.approx(-7.0)


def test_pack_detections_drops_the_furthest_reports_when_there_are_more_than_slots():
    """A reading with more vehicles than slots keeps the near ones and loses the far ones.

    Purpose: The state carries a fixed number of slots, so a reading the model has no slot for
        is exactly what its false-alarm rate stands for. Dropping the *nearest* instead would
        hide the vehicle the planner most needs to avoid

    Given: Three reports at 5 m, 12.4 m and 20 m, and only two slots
    When: They are packed
    Then: Both slots are flagged detected and hold the 5 m and 12.4 m reports; the 20 m one
        is absent

    Test type: unit
    """
    reports = np.array([[20.0, 0.0, 1.0, -0.5], [5.0, 0.0, -2.0, 0.25], [12.0, 3.0, 0.5, 1.5]])

    packed = pack_detections(reports, 2)

    assert packed.shape == (2, DETECTION_SLOT_WIDTH)
    np.testing.assert_array_equal(packed[:, DETECTION_PRESENT], [1.0, 1.0])
    np.testing.assert_allclose(packed[:, DETECTION_REL_X], [5.0, 12.0])
    np.testing.assert_allclose(packed[:, DETECTION_REL_Y], [0.0, 3.0])
    np.testing.assert_allclose(packed[:, DETECTION_REL_VX], [-2.0, 0.5])
    np.testing.assert_allclose(packed[:, DETECTION_REL_VY], [0.25, 1.5])


def test_pack_detections_with_nothing_to_report_returns_an_all_zero_block():
    """A step where the radar sees nothing still produces a fixed-width reading.

    Purpose: The observation width is fixed and the belief reshapes it by position, so an
        empty step must return the same block shape as a full one rather than an empty array

    Given: No reports at all, and four slots
    When: They are packed
    Then: The result is a (4, 5) block of zeros, so every slot reads undetected

    Test type: unit
    """
    packed = pack_detections(np.zeros((0, _REPORT_WIDTH)), 4)

    assert packed.shape == (4, DETECTION_SLOT_WIDTH)
    assert not np.any(packed)


@pytest.mark.parametrize("rates", [(0.0, 0.0), (0.05, 0.02), (0.5, 0.49)])
def test_validate_detection_rates_accepts_the_rates_the_composition_can_use(rates):
    """Rates in [0, 1) summing below 1 pass, zero included.

    Purpose: Zero is the shipped setting — this world's detection decision is deterministic —
        so a validator that rejected it would reject every model built from the defaults.
        Nonzero rates stay valid because a lossy radar remains a legitimate configuration

    Given: Both rates at zero, a lossy-radar pair, and a pair summing to just under 1
    When: Each is validated
    Then: Nothing is raised

    Test type: unit
    """
    miss, false_alarm = rates

    assert validate_detection_rates(miss, false_alarm) is None


@pytest.mark.parametrize(
    "miss, false_alarm, message",
    [
        (-0.01, 0.02, "presence_miss_prob must be in"),
        (0.05, 1.0, "presence_false_alarm_prob must be in"),
        (1.5, 0.02, "presence_miss_prob must be in"),
        (0.6, 0.4, "must stay below 1"),
        (0.9, 0.2, "must stay below 1"),
    ],
)
def test_validate_detection_rates_rejects_the_pairs_that_invert_the_likelihood(
    miss, false_alarm, message
):
    """A rate outside [0, 1), or a pair summing to 1, is refused at the call site.

    Purpose: At miss + false_alarm >= 1 a tracked vehicle makes a slot no *more* likely to be
        reported than an empty one, so the likelihood runs backwards while every value stays
        finite and plausible — the kind of error a filter absorbs without complaint

    Given: A negative miss rate, a false-alarm rate at exactly 1, a miss rate above 1, and two
        pairs summing to 1 or more
    When: Each is validated
    Then: ValueError is raised, naming the offending parameter or the sum

    Test type: unit
    """
    with pytest.raises(ValueError, match=message):
        validate_detection_rates(miss, false_alarm)
