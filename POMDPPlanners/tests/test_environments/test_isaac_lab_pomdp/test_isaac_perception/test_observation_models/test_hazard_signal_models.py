# SPDX-License-Identifier: MIT

"""Unit tests for the latent-type signal observation model.

The load-bearing property is the reveal rule: the likelihood must be *flat* outside a zone and
separating inside it. Get that wrong in the flat direction and the agent learns the type before it
commits, which removes the decision the construction exists to create; get it wrong in the
separating direction and the belief never splits at all.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
    UNINFORMATIVE_ACCURACY,
    LatentTypeSignalObservationModel,
)

LIGHT_CENTER = (-1.5, 0.0)
HEAVY_CENTER = (1.5, 0.0)
RADIUS = 0.9
LIGHT_INDEX = 0
HEAVY_INDEX = 1

#: One radius short of the heavy zone: the last position at which nothing has been learned.
GATE = np.array([HEAVY_CENTER[0] - 1.6, 0.0])
INSIDE_HEAVY = np.array(HEAVY_CENTER)


@pytest.fixture(name="model")
def _model() -> LatentTypeSignalObservationModel:
    return LatentTypeSignalObservationModel(
        channel="hazard_signal",
        type_channel="hazard_type",
        position_channel="position",
        zone_centers=[LIGHT_CENTER, HEAVY_CENTER],
        zone_radii=[RADIUS, RADIUS],
        accuracy_inside=0.9,
        rng=np.random.default_rng(0),
    )


def _clean(position: np.ndarray, types: np.ndarray) -> dict:
    return {"position": position, "hazard_type": types}


def test_signal_is_informative_only_inside_its_own_zone(
    model: LatentTypeSignalObservationModel,
) -> None:
    """Standing in one zone must teach nothing about the other.

    Purpose: Validates the per-zone accuracy the reveal rule depends on

    Given: A two-zone model and an agent standing in the heavy zone
    When: The per-zone accuracies are read there and at the gate
    Then: Only the occupied zone is informative; every other entry is a coin flip

    Test type: unit
    """
    types = np.array([0.0, 1.0])
    assert model.accuracy_at(_clean(GATE, types)) == pytest.approx([0.5, 0.5])
    inside = model.accuracy_at(_clean(INSIDE_HEAVY, types))
    assert inside[HEAVY_INDEX] == pytest.approx(0.9)
    assert inside[LIGHT_INDEX] == pytest.approx(UNINFORMATIVE_ACCURACY)


def test_likelihood_is_flat_outside_the_zone(model: LatentTypeSignalObservationModel) -> None:
    """Outside, every candidate type explains the signal equally, so nothing is learned.

    Purpose: Validates that the reveal is deferred until after the agent commits

    Given: A signal observed at the gate
    When: It is scored against a low-type and a high-type candidate state
    Then: The two log-densities are equal

    Test type: unit
    """
    signals = np.array([1.0, 1.0])
    low = model.log_probability(_clean(GATE, np.array([0.0, 0.0])), signals)
    high = model.log_probability(_clean(GATE, np.array([0.0, 1.0])), signals)
    assert low == pytest.approx(high)


def test_likelihood_separates_inside_the_zone(model: LatentTypeSignalObservationModel) -> None:
    """Once inside, the signal must actually discriminate between the types.

    Purpose: Validates that entering a zone makes the type learnable

    Given: A high signal observed inside the heavy zone
    When: It is scored against a high-type and a low-type candidate state
    Then: The high-type state scores strictly higher

    Test type: unit
    """
    signals = np.array([1.0, 1.0])
    low = model.log_probability(_clean(INSIDE_HEAVY, np.array([0.0, 0.0])), signals)
    high = model.log_probability(_clean(INSIDE_HEAVY, np.array([0.0, 1.0])), signals)
    assert high > low


def test_sampled_signal_is_truthful_at_the_configured_rate(
    model: LatentTypeSignalObservationModel,
) -> None:
    """The empirical accuracy must match the configured one, inside and outside.

    Purpose: Validates the sampler against the accuracy the density assumes

    Given: A high-type heavy zone
    When: Many signals are drawn inside it and at the gate
    Then: The inside rate is 0.9 and the gate rate is 0.5

    Test type: unit
    """
    types = np.array([0.0, 1.0])
    inside = np.stack([model.perceive(_clean(INSIDE_HEAVY, types)) for _ in range(4000)])
    at_gate = np.stack([model.perceive(_clean(GATE, types)) for _ in range(4000)])
    assert float((inside[:, HEAVY_INDEX] == 1.0).mean()) == pytest.approx(0.9, abs=0.02)
    assert float((at_gate[:, HEAVY_INDEX] == 1.0).mean()) == pytest.approx(0.5, abs=0.03)


def test_seed_makes_the_signal_stream_reproducible(
    model: LatentTypeSignalObservationModel,
) -> None:
    """A planning run has to be repeatable, and this channel carries its own randomness.

    Purpose: Validates the seed hook on the signal sampler

    Given: A model reseeded to the same value twice
    When: The same number of signals is drawn after each reseed
    Then: The two streams are identical

    Test type: unit
    """
    clean = _clean(INSIDE_HEAVY, np.array([0.0, 1.0]))
    model.seed(7)
    first = np.stack([model.perceive(clean) for _ in range(20)])
    model.seed(7)
    second = np.stack([model.perceive(clean) for _ in range(20)])
    assert np.array_equal(first, second)


def test_posterior_splits_far_enough_to_grade_after_entering(
    model: LatentTypeSignalObservationModel,
) -> None:
    """The gate check: entering must move the posterior, or there is no dispersion to price.

    Purpose: Validates that one in-zone signal separates the belief materially

    Given: A 0.3 prior on the heavy zone's high type and an agent one step inside it
    When: The posterior is taken under each of the two possible signals
    Then: The two posteriors are more than 0.2 apart

    Test type: unit
    """
    prior = np.array([0.0, 0.3])
    clean = _clean(INSIDE_HEAVY, np.array([0.0, 1.0]))
    low_child = model.posterior_after_signal(prior, clean, np.array([0.0, 0.0]))
    high_child = model.posterior_after_signal(prior, clean, np.array([0.0, 1.0]))
    gap = abs(float(high_child[HEAVY_INDEX]) - float(low_child[HEAVY_INDEX]))
    assert gap > 0.2, f"posteriors only {gap:.3f} apart"


def test_posterior_does_not_move_outside_the_zone(
    model: LatentTypeSignalObservationModel,
) -> None:
    """The control direction of the gate check: at the gate the belief must be untouched.

    Purpose: Validates that an uninformative signal leaves the prior in place

    Given: A 0.3 prior and an agent at the gate
    When: The posterior is taken under either signal
    Then: Both equal the prior

    Test type: unit
    """
    prior = np.array([0.0, 0.3])
    clean = _clean(GATE, np.array([0.0, 1.0]))
    for signal in ([0.0, 0.0], [0.0, 1.0]):
        posterior = model.posterior_after_signal(prior, clean, np.array(signal))
        assert posterior[HEAVY_INDEX] == pytest.approx(0.3)


def test_density_rejects_a_signal_of_the_wrong_width(
    model: LatentTypeSignalObservationModel,
) -> None:
    """A signal vector must have one entry per zone, or the zone mapping is meaningless.

    Purpose: Validates the shape guard on the signal density

    Given: A two-zone model
    When: A three-entry signal is scored
    Then: The score is -inf rather than a broadcast

    Test type: unit
    """
    clean = _clean(INSIDE_HEAVY, np.array([0.0, 1.0]))
    assert model.log_probability(clean, np.zeros(3)) == float("-inf")


def test_an_uninformative_accuracy_is_rejected_at_construction() -> None:
    """An in-zone accuracy of 0.5 collapses the construction to the case it exists to avoid.

    Purpose: Validates the construction-time guard on accuracy_inside

    Given: An accuracy of exactly 0.5
    When: The model is constructed
    Then: ValueError is raised explaining the requirement

    Test type: unit
    """
    with pytest.raises(ValueError, match="informative"):
        LatentTypeSignalObservationModel(
            channel="hazard_signal",
            type_channel="hazard_type",
            position_channel="position",
            zone_centers=[HEAVY_CENTER],
            zone_radii=[RADIUS],
            accuracy_inside=0.5,
        )


def test_mismatched_zone_arrays_are_rejected_at_construction() -> None:
    """Centres and radii describe the same zones, so a length mismatch is a config bug.

    Purpose: Validates the construction-time guard on the zone geometry

    Given: Two centres but one radius
    When: The model is constructed
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="zone_radii"):
        LatentTypeSignalObservationModel(
            channel="hazard_signal",
            type_channel="hazard_type",
            position_channel="position",
            zone_centers=[LIGHT_CENTER, HEAVY_CENTER],
            zone_radii=[RADIUS],
        )


def test_position_is_read_from_the_declared_indices() -> None:
    """The position channel is often a wider block, so the x-y indices must be honoured.

    Purpose: Validates the position_indices indirection

    Given: A model reading x and y from indices 2 and 3 of the position block
    When: A block whose trailing pair sits inside the heavy zone is supplied
    Then: The occupancy reports the heavy zone

    Test type: unit
    """
    model = LatentTypeSignalObservationModel(
        channel="hazard_signal",
        type_channel="hazard_type",
        position_channel="robot",
        zone_centers=[LIGHT_CENTER, HEAVY_CENTER],
        zone_radii=[RADIUS, RADIUS],
        position_indices=(2, 3),
    )
    clean = {
        "robot": np.array([99.0, 99.0, HEAVY_CENTER[0], HEAVY_CENTER[1]]),
        "hazard_type": np.array([0.0, 1.0]),
    }
    assert model.occupancy(clean).tolist() == [0.0, 1.0]
