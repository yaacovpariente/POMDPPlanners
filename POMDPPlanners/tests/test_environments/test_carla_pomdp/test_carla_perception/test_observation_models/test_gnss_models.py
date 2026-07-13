# SPDX-License-Identifier: MIT

"""Tests for the GNSS-channel observation model (gnss_models.py)."""

import numpy as np

from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models import (
    GnssObservationModel,
)


def test_perceive_returns_two_vector():
    """perceive returns a 2-D GNSS reading.

    Purpose: Validates the GNSS sampler shape and channel handling

    Given: A GnssObservationModel and a clean 2-D reading
    When: perceive is called
    Then: A length-2 array is returned

    Test type: unit
    """
    np.random.seed(0)
    model = GnssObservationModel(gnss_std=1e-5)
    perceived = model.perceive(np.array([1.0, 2.0]))
    assert perceived.shape == (2,)


def test_log_probability_prefers_the_truth():
    """The density scores the true reading above a shifted one.

    Purpose: Validates the GNSS Gaussian density ranks the truthful reading higher

    Given: A GnssObservationModel and a clean 2-D reading
    When: log_probability scores the truth vs a 3 m-shifted reading
    Then: The truthful reading has the higher log-probability

    Test type: unit
    """
    model = GnssObservationModel(gnss_std=0.5)
    truth = np.array([1.0, 2.0])
    assert model.log_probability(truth, truth) > model.log_probability(truth, truth + 3.0)
