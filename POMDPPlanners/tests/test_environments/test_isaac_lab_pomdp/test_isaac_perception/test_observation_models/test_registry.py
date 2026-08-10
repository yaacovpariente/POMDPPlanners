# SPDX-License-Identifier: MIT

"""Unit tests for the name-keyed Isaac observation-model registry."""

from typing import Mapping

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import IsaacObservationModel
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models import (
    GaussianChannelObservationModel,
    available_observation_models,
    build_observation_model,
    register_observation_model,
)


def test_shipped_models_are_registered_under_their_documented_names() -> None:
    """Selection by name is the public contract, so the names must actually be there.

    Purpose: Validates that importing the catalog registers every shipped model

    Given: The observation_models package has been imported
    When: The registry is listed
    Then: The five shipped model names are present

    Test type: unit
    """
    names = available_observation_models()
    assert {"gaussian", "exact", "ray_caster", "height_scan", "latent_type_signal"} <= set(names)
    assert names == sorted(names)


def test_build_resolves_a_name_and_forwards_construction_arguments() -> None:
    """A selection is only useful if the per-channel arguments reach the model.

    Purpose: Validates that build_observation_model instantiates and passes kwargs through

    Given: The "gaussian" model registered in the catalog
    When: It is built with an explicit channel and noise_std
    Then: The instance carries both values

    Test type: unit
    """
    model = build_observation_model("gaussian", channel="base_pose", noise_std=0.25)
    assert isinstance(model, GaussianChannelObservationModel)
    assert model.channel == "base_pose"
    assert model.noise_std == pytest.approx(0.25)


def test_unknown_name_raises_and_lists_what_is_available() -> None:
    """A typo in a config must fail loudly with the correction in the message.

    Purpose: Validates the error path of an unregistered selection

    Given: A name that was never registered
    When: build_observation_model is called with it
    Then: KeyError is raised and the message lists the registered names

    Test type: unit
    """
    with pytest.raises(KeyError, match="available"):
        build_observation_model("no_such_model", channel="x")


def test_registration_decorator_returns_the_factory_unchanged() -> None:
    """A decorator that wrapped the class would break isinstance checks on it.

    Purpose: Validates that register_observation_model is transparent

    Given: A locally defined observation model
    When: It is registered under a fresh name
    Then: The decorated symbol is the class itself and the registry builds instances of it

    Test type: unit
    """

    @register_observation_model("_test_passthrough")
    class _Passthrough(IsaacObservationModel):
        channel = "probe"
        state_channels = ("probe",)

        def perceive(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
            return np.asarray(clean_state["probe"], dtype=float)

    assert _Passthrough.__name__ == "_Passthrough"
    assert isinstance(build_observation_model("_test_passthrough"), _Passthrough)
