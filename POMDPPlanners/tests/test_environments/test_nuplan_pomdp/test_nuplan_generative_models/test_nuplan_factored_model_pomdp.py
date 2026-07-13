# SPDX-License-Identifier: MIT

"""Unit tests for the reference factored nuPlan generative model."""

import numpy as np
import pytest

from POMDPPlanners.environments.nuplan_pomdp.nuplan_generative_models.nuplan_factored_model_pomdp import (
    FactoredNuPlanModelPOMDP,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp import (
    AGENT_SLOT_WIDTH,
    EGO_STATE_WIDTH,
    LIGHT_SLOT_WIDTH,
    REWARD_STEP_COST,
)

_MAX_AGENTS = 2


def _zero_state() -> np.ndarray:
    width = EGO_STATE_WIDTH + _MAX_AGENTS * AGENT_SLOT_WIDTH + LIGHT_SLOT_WIDTH
    return np.zeros(width)


def test_identity_transition_returns_state_unchanged() -> None:
    """The placeholder transition returns the input state unchanged.

    Purpose: Validates the documented identity-placeholder dynamics.

    Given: A factored nuPlan model and a state
    When: sample_next_state is called
    Then: the returned state equals the input

    Test type: unit
    """
    model = FactoredNuPlanModelPOMDP(discount_factor=0.95, max_tracked_agents=_MAX_AGENTS)
    state = _zero_state()
    state[0] = 3.0
    assert np.array_equal(model.sample_next_state(state, 0), state)


def test_sample_next_step_emits_both_channels() -> None:
    """A full step yields the composed {ego, agents} observation.

    Purpose: Validates sample_next_step returns a factored observation dict.

    Given: A factored nuPlan model and a zero state
    When: sample_next_step is called
    Then: the observation carries the ego and agents channels

    Test type: unit
    """
    np.random.seed(42)
    model = FactoredNuPlanModelPOMDP(discount_factor=0.95, max_tracked_agents=_MAX_AGENTS)
    observation = model.sample_next_step(_zero_state(), 0)[1]
    assert sorted(observation) == ["agents", "ego"]


def test_reward_on_stationary_state_is_step_cost() -> None:
    """A stationary, on-route state earns just the per-step time cost.

    Purpose: Validates the driving-quality reward at zero motion.

    Given: A factored nuPlan model and a zero (stationary) state
    When: reward is scored for the cruise action
    Then: the reward equals the negative per-step time cost

    Test type: unit
    """
    model = FactoredNuPlanModelPOMDP(discount_factor=0.95, max_tracked_agents=_MAX_AGENTS)
    assert model.reward(_zero_state(), 0) == pytest.approx(-REWARD_STEP_COST)


def test_transition_density_not_implemented() -> None:
    """The placeholder transition has no density yet.

    Purpose: Validates transition_log_probability raises for the placeholder dynamics.

    Given: A factored nuPlan model
    When: transition_log_probability is called
    Then: a NotImplementedError is raised

    Test type: unit
    """
    model = FactoredNuPlanModelPOMDP(discount_factor=0.95, max_tracked_agents=_MAX_AGENTS)
    with pytest.raises(NotImplementedError):
        model.transition_log_probability(_zero_state(), 0, [_zero_state()])


def test_initial_distributions_defer_to_the_world() -> None:
    """The model's initial distributions defer to the world's observation.

    Purpose: Validates the model does not seed its own initial belief.

    Given: A factored nuPlan model
    When: initial_state_dist / initial_observation_dist are requested
    Then: both raise NotImplementedError directing the caller to the world

    Test type: unit
    """
    model = FactoredNuPlanModelPOMDP(discount_factor=0.95, max_tracked_agents=_MAX_AGENTS)
    with pytest.raises(NotImplementedError):
        model.initial_state_dist()
    with pytest.raises(NotImplementedError):
        model.initial_observation_dist()
