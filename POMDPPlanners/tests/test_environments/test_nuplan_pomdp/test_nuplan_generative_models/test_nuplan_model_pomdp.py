# SPDX-License-Identifier: MIT

"""Unit tests for the abstract nuPlan generative-model base (schema + observation composition).

The base is exercised through the concrete :class:`FactoredNuPlanModelPOMDP`, which supplies the
dynamics the abstract base leaves open while inheriting the shared schema and observation model.
"""

import numpy as np

from POMDPPlanners.environments.nuplan_pomdp.nuplan_generative_models.nuplan_factored_model_pomdp import (
    FactoredNuPlanModelPOMDP,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp import (
    AGENT_SLOT_WIDTH,
    EGO_STATE_WIDTH,
    LIGHT_SLOT_WIDTH,
)

_MAX_AGENTS = 2


def _model() -> FactoredNuPlanModelPOMDP:
    return FactoredNuPlanModelPOMDP(discount_factor=0.95, max_tracked_agents=_MAX_AGENTS)


def _zero_state() -> np.ndarray:
    width = EGO_STATE_WIDTH + _MAX_AGENTS * AGENT_SLOT_WIDTH + LIGHT_SLOT_WIDTH
    return np.zeros(width)


def test_get_actions_indexes_the_presets() -> None:
    """The discrete action set enumerates the preset indices.

    Purpose: Validates get_actions returns indices into action_presets.

    Given: A factored nuPlan model with the default presets
    When: get_actions is called
    Then: it returns 0..len(presets)-1

    Test type: unit
    """
    model = _model()
    assert model.get_actions() == list(range(len(model.action_presets)))


def test_encode_observation_degrades_each_channel() -> None:
    """encode_observation perceives every channel of the raw world reading.

    Purpose: Validates the single raw-observation seam composes the channel models.

    Given: A raw {ego, agents} observation with one distant agent
    When: encode_observation is applied
    Then: the perceived observation keeps both channels and gates the distant agent out

    Test type: unit
    """
    np.random.seed(0)
    model = FactoredNuPlanModelPOMDP(
        discount_factor=0.95, max_tracked_agents=1, perception_range=50.0
    )
    raw = {"ego": np.zeros(EGO_STATE_WIDTH), "agents": np.array([1.0, 500.0, 0.0, 0.0, 0.0])}
    perceived = model.encode_observation(raw)
    assert sorted(perceived) == ["agents", "ego"]
    assert perceived["agents"][0] == 0.0


def test_observation_log_probability_sums_channel_densities() -> None:
    """The composed density is the sum of the per-channel log-densities.

    Purpose: Validates observation_log_probability composes channels additively.

    Given: A model and a clean-derived observation of a zero state
    When: observation_log_probability scores that observation
    Then: it equals the sum of the ego and agents channel densities

    Test type: unit
    """
    model = _model()
    state = _zero_state()
    clean = model._clean_observation(state)  # pylint: disable=protected-access
    observation = dict(clean)
    total = model.observation_log_probability(state, 0, [observation])[0]
    ego_lp = model.observation_models["ego"].log_probability(clean["ego"], observation["ego"])
    agents_lp = model.observation_models["agents"].log_probability(
        clean["agents"], observation["agents"]
    )
    assert total == float(ego_lp + agents_lp)


def test_hash_and_equality_agree_on_equal_observations() -> None:
    """Equal observations hash equally (planner belief-child indexing contract).

    Purpose: Validates is_equal_observation implies equal hash_observation.

    Given: Two structurally-equal {ego, agents} observations
    When: is_equal_observation and hash_observation are compared
    Then: they report equal and hash to the same key

    Test type: unit
    """
    model = _model()
    obs_a = {"ego": np.zeros(EGO_STATE_WIDTH), "agents": np.zeros(_MAX_AGENTS * AGENT_SLOT_WIDTH)}
    obs_b = {"ego": np.zeros(EGO_STATE_WIDTH), "agents": np.zeros(_MAX_AGENTS * AGENT_SLOT_WIDTH)}
    assert model.is_equal_observation(obs_a, obs_b)
    assert model.hash_observation(obs_a) == model.hash_observation(obs_b)
