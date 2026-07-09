# SPDX-License-Identifier: MIT

"""Tests for the per-channel observation-model registry (registry.py)."""

from typing import Any

import numpy as np
import pytest

from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_model import (
    CarlaObservationModel,
)
from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models import (
    GnssObservationModel,
    available_observation_models,
    build_observation_model,
    register_observation_model,
)


def test_default_models_are_registered():
    """The shipped catalog registers the gnss/agents reference models.

    Purpose: Validates importing the catalog populates the registry per channel

    Given: The observation_models subpackage has been imported
    When: available_observation_models is queried per channel
    Then: The gnss 'gaussian' and agents 'factored' entries are present

    Test type: unit
    """
    assert "gaussian" in available_observation_models("gnss")
    assert "factored" in available_observation_models("agents")


def test_build_resolves_registered_model_with_kwargs():
    """build_observation_model instantiates the registered factory with kwargs.

    Purpose: Validates name resolution and kwargs forwarding

    Given: The registered gnss 'gaussian' model
    When: build_observation_model is called with a gnss_std kwarg
    Then: A GnssObservationModel carrying that gnss_std is returned

    Test type: unit
    """
    model = build_observation_model("gnss", "gaussian", gnss_std=0.1)
    assert isinstance(model, GnssObservationModel)
    assert model.gnss_std == 0.1


def test_build_unknown_name_raises_keyerror():
    """Resolving an unregistered name raises a descriptive KeyError.

    Purpose: Validates the error path for an unknown selection

    Given: No model registered as gnss 'no-such-model'
    When: build_observation_model is called for it
    Then: KeyError is raised naming the missing model

    Test type: unit
    """
    with pytest.raises(KeyError, match="no-such-model"):
        build_observation_model("gnss", "no-such-model")


def test_register_decorator_adds_a_selectable_model():
    """The decorator registers a new model that build/available then expose.

    Purpose: Validates user-extensibility via the registration decorator

    Given: A custom sample-only gnss model registered under a fresh name
    When: available_observation_models and build_observation_model are used
    Then: The name is listed and the built instance is the custom class

    Test type: unit
    """

    @register_observation_model("gnss", "test_identity_gnss")
    class _IdentityGnss(CarlaObservationModel):
        channel = "gnss"
        supports_density = False

        def perceive(self, clean_channel: Any) -> np.ndarray:
            return np.asarray(clean_channel, dtype=float)

    assert "test_identity_gnss" in available_observation_models("gnss")
    assert isinstance(build_observation_model("gnss", "test_identity_gnss"), _IdentityGnss)
