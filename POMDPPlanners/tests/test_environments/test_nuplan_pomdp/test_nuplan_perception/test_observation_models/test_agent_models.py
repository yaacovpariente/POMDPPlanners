# SPDX-License-Identifier: MIT

"""Unit tests for the factored agent-channel observation model."""

import numpy as np

from POMDPPlanners.environments.nuplan_pomdp.nuplan_perception.observation_models.agent_models import (
    FactoredAgentObservationModel,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp import AGENT_SLOT_WIDTH


def test_out_of_range_agent_is_dropped() -> None:
    """An agent beyond the perception range is gated out of the perceived block.

    Purpose: Validates the range gate zeroes an out-of-range slot.

    Given: A model with a 50 m range and one agent 200 m ahead
    When: the clean agent block is perceived
    Then: the slot is emptied (present == 0)

    Test type: unit
    """
    np.random.seed(0)
    model = FactoredAgentObservationModel(max_tracked_agents=1, perception_range=50.0)
    agents = np.array([1.0, 200.0, 0.0, 0.0, 0.0])
    perceived = model.perceive(agents)
    assert perceived[0] == 0.0


def test_in_range_agent_is_detected() -> None:
    """A near agent within range is kept as a present slot.

    Purpose: Validates a visible slot survives gating.

    Given: A model with a 50 m range and one agent 10 m ahead
    When: the clean agent block is perceived
    Then: the slot stays present (present == 1)

    Test type: unit
    """
    np.random.seed(0)
    model = FactoredAgentObservationModel(max_tracked_agents=1, perception_range=50.0, pose_std=0.0)
    agents = np.array([1.0, 10.0, 0.0, 0.0, 0.0])
    perceived = model.perceive(agents)
    assert perceived[0] == 1.0


def test_log_probability_penalises_a_phantom_detection() -> None:
    """Reporting a present agent where the state has none is heavily penalised.

    Purpose: Validates the density floors an impossible phantom detection.

    Given: A clean state with an empty slot and an observation with that slot present
    When: log_probability scores the observation
    Then: the score is a large negative floor

    Test type: unit
    """
    model = FactoredAgentObservationModel(max_tracked_agents=1)
    clean = np.zeros(AGENT_SLOT_WIDTH)
    phantom = np.array([1.0, 5.0, 0.0, 0.0, 0.0])
    assert model.log_probability(clean, phantom) < -10.0


def test_render_without_noise_is_deterministic() -> None:
    """render(noisy=False) returns the gated block without pose noise.

    Purpose: Validates the noise-free render path is deterministic.

    Given: A visible agent and a model with non-zero pose_std
    When: render is called twice with noisy=False
    Then: the two renders are identical

    Test type: unit
    """
    model = FactoredAgentObservationModel(max_tracked_agents=1, pose_std=1.0)
    agents = np.array([1.0, 10.0, 1.0, 0.0, 2.0])
    first = model.render(agents, noisy=False)
    second = model.render(agents, noisy=False)
    assert np.array_equal(first, second)
