# SPDX-License-Identifier: MIT

"""Tests for the agent-channel observation model (agent_models.py)."""

import numpy as np

from POMDPPlanners.environments.carla_pomdp.carla_perception.observation_models import (
    FactoredAgentObservationModel,
)


def test_perceive_gates_out_of_range_agent():
    """perceive keeps a near agent and zeroes an out-of-range one.

    Purpose: Validates the range gate on the agents channel

    Given: A model with a 50 m range and a near + far agent (both present)
    When: perceive is called on the flat agents block
    Then: The near slot stays present and the far slot is zeroed

    Test type: unit
    """
    np.random.seed(0)
    model = FactoredAgentObservationModel(max_tracked_agents=2, perception_range=50.0)
    rows = np.zeros((2, 5))
    rows[0] = np.array([1.0, 10.0, 0.0, 0.0, 0.0])
    rows[1] = np.array([1.0, 100.0, 0.0, 0.0, 0.0])
    out = model.perceive(rows.reshape(-1)).reshape(2, 5)
    assert out[0, 0] == 1.0
    assert out[1, 0] == 0.0


def test_render_noiseless_preserves_pose():
    """render(noisy=False) gates without perturbing a detected pose.

    Purpose: Validates the noise-free rendering path used to build clean observations

    Given: A model with no range gate and a single detected agent with a nonzero pose
    When: render is called with noisy=False
    Then: The agent's pose is returned unchanged

    Test type: unit
    """
    model = FactoredAgentObservationModel(max_tracked_agents=1, perception_range=None, pose_std=0.5)
    rows = np.array([[1.0, 10.0, 1.0, 0.2, 3.0]])
    out = model.render(rows.reshape(-1), noisy=False).reshape(1, 5)
    assert np.allclose(out[0], rows[0])


def test_log_probability_penalizes_seeing_an_occluded_agent():
    """Reporting an occluded agent is scored as near-impossible.

    Purpose: Validates occlusion gating in the agent-channel density

    Given: A target occluded by a blocker on the ego->target sight line
    When: log_probability scores an observation that (wrongly) reports the occluded target
        vs one that correctly omits it
    Then: The correctly-omitted observation scores far higher

    Test type: unit
    """
    model = FactoredAgentObservationModel(max_tracked_agents=2, occlusion_radius=1.5)
    target = np.array([1.0, 10.0, 0.0, 0.0, 0.0])
    blocker = np.array([1.0, 5.0, 0.0, 0.0, 0.0])
    clean = np.stack([target, blocker]).reshape(-1)

    omitted = np.zeros((2, 5))
    omitted[1] = blocker
    seen = omitted.copy()
    seen[0] = target

    lp_omitted = model.log_probability(clean, omitted.reshape(-1))
    lp_seen = model.log_probability(clean, seen.reshape(-1))
    assert lp_omitted > lp_seen + 40.0
