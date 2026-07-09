# SPDX-License-Identifier: MIT

"""Unit tests for the per-channel nuPlan observation-model registry."""

import pytest

from POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_models import (
    EgoObservationModel,
    FactoredAgentObservationModel,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_models.registry import (
    available_observation_models,
    build_observation_model,
)


def test_registered_channels_resolve_to_their_models() -> None:
    """The default channel selections build their registered concrete models.

    Purpose: Validates build_observation_model resolves (channel, name) to instances.

    Given: The ego and agents channels populated by the catalog imports
    When: build_observation_model is called for each default selection
    Then: the returned instances are the registered concrete classes

    Test type: unit
    """
    ego = build_observation_model("ego", "gaussian")
    agents = build_observation_model("agents", "factored", max_tracked_agents=3)
    assert isinstance(ego, EgoObservationModel)
    assert isinstance(agents, FactoredAgentObservationModel)


def test_available_models_lists_registered_names() -> None:
    """available_observation_models reports the registered names per channel.

    Purpose: Validates the catalog listing for each channel.

    Given: The imported ego and agents channel catalogs
    When: available_observation_models is queried for each channel
    Then: the registered names appear in the returned lists

    Test type: unit
    """
    assert "gaussian" in available_observation_models("ego")
    assert "factored" in available_observation_models("agents")


def test_unknown_model_name_raises_keyerror() -> None:
    """Resolving an unregistered (channel, name) raises a descriptive KeyError.

    Purpose: Validates the error path for an unknown model selection.

    Given: The observation-model registry
    When: build_observation_model is called with an unregistered name
    Then: a KeyError naming the channel is raised

    Test type: unit
    """
    with pytest.raises(KeyError, match="agents"):
        build_observation_model("agents", "does_not_exist")
