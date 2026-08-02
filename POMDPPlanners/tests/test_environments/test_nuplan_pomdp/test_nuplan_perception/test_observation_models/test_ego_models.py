# SPDX-License-Identifier: MIT

"""Unit tests for the ego-channel observation model."""

import numpy as np
import pytest

from POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_models.ego_models import (
    EgoObservationModel,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp import EGO_STATE_WIDTH


def test_perceive_returns_ego_width_vector() -> None:
    """perceive returns a noised ego proprioception vector of the ego width.

    Purpose: Validates the ego sampler shape and additive-noise behaviour.

    Given: An EgoObservationModel with a small noise std and a clean ego vector
    When: perceive is called on the clean ego block
    Then: the result has the ego width and differs from the clean value

    Test type: unit
    """
    np.random.seed(0)
    model = EgoObservationModel(ego_std=0.1)
    clean = np.arange(EGO_STATE_WIDTH, dtype=float)
    perceived = model.perceive(clean)
    assert perceived.shape == (EGO_STATE_WIDTH,)
    assert not np.array_equal(perceived, clean)


def test_log_probability_peaks_at_the_clean_value() -> None:
    """The ego density is maximised when the observation equals the clean value.

    Purpose: Validates the Gaussian density peaks at zero measurement error.

    Given: An EgoObservationModel and a clean ego vector
    When: log_probability is scored at the clean value and at a displaced value
    Then: the clean value scores strictly higher

    Test type: unit
    """
    model = EgoObservationModel(ego_std=0.1)
    clean = np.zeros(EGO_STATE_WIDTH)
    displaced = clean.copy()
    displaced[0] = 1.0
    assert model.log_probability(clean, clean) > model.log_probability(clean, displaced)


def test_ego_density_is_a_normalised_gaussian() -> None:
    """The ego density carries its Gaussian normalising constant.

    Purpose: Validates log_probability is a proper log-density, not an unnormalised score.

    Given: An EgoObservationModel with ego_std 1.0 and an observation at the clean value
    When: log_probability scores it
    Then: the score equals -0.5 * EGO_STATE_WIDTH * log(2 pi)

    Test type: unit
    """
    model = EgoObservationModel(ego_std=1.0)
    clean = np.zeros(EGO_STATE_WIDTH)
    expected = -0.5 * EGO_STATE_WIDTH * np.log(2.0 * np.pi)

    assert model.log_probability(clean, clean) == pytest.approx(expected)


def test_supports_density_flag_is_set() -> None:
    """The ego model advertises a density so it can back a scoring model.

    Purpose: Validates supports_density is True for the ego channel.

    Given: An EgoObservationModel instance
    When: its supports_density attribute is read
    Then: it is True

    Test type: unit
    """
    assert EgoObservationModel().supports_density is True
