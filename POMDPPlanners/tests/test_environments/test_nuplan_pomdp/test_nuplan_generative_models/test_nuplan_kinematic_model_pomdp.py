# SPDX-License-Identifier: MIT

"""Unit tests for the kinematic-bicycle nuPlan generative model."""

from typing import Any

import numpy as np
import pytest

from POMDPPlanners.environments.nuplan_pomdp.nuplan_generative_models.nuplan_kinematic_model_pomdp import (
    KinematicNuPlanModelPOMDP,
)
from POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp import (
    AGENT_SLOT_WIDTH,
    EGO_STATE_WIDTH,
    LIGHT_SLOT_WIDTH,
)

_MAX_AGENTS = 2


def _model(**kwargs: Any) -> KinematicNuPlanModelPOMDP:
    return KinematicNuPlanModelPOMDP(
        discount_factor=0.95, dt=0.1, max_tracked_agents=_MAX_AGENTS, **kwargs
    )


def _zero_state() -> np.ndarray:
    width = EGO_STATE_WIDTH + _MAX_AGENTS * AGENT_SLOT_WIDTH + LIGHT_SLOT_WIDTH
    return np.zeros(width)


def _state_with_agent(rel_x: float, rel_y: float = 0.0, rel_speed: float = 0.0) -> np.ndarray:
    state = _zero_state()
    state[EGO_STATE_WIDTH : EGO_STATE_WIDTH + AGENT_SLOT_WIDTH] = [
        1.0,
        rel_x,
        rel_y,
        0.0,
        rel_speed,
    ]
    return state


def test_propagated_angles_stay_wrapped_to_pi() -> None:
    """Yaw and heading error stay inside [-pi, pi] after propagation.

    Purpose: Validates the transition preserves the documented wrapped-angle state layout,
        so densities and comparisons never straddle the branch cut.

    Given: A state whose yaw and heading error sit just below +pi, moving forward
    When: a right-steering action propagates them past pi
    Then: both angles come back wrapped into [-pi, pi]

    Test type: unit
    """
    model = _model()
    state = _zero_state()
    state[2] = np.pi - 0.01  # yaw
    state[3] = 5.0  # vx, so the yaw rate is non-zero
    state[6] = np.pi - 0.01  # heading error vs the route baseline

    next_state = model.sample_next_state(state, 2)  # (0.0, 0.3) coast, steer right

    assert -np.pi <= next_state[2] <= np.pi
    assert -np.pi <= next_state[6] <= np.pi


def test_acceleration_produces_forward_velocity() -> None:
    """The accelerate preset moves the ego forward.

    Purpose: Validates the kinematic transition responds to acceleration.

    Given: A kinematic nuPlan model and a stationary state
    When: the accelerate action (index 0) is applied
    Then: the next state has positive longitudinal velocity and advanced position

    Test type: unit
    """
    model = _model()
    next_state = model.sample_next_state(_zero_state(), 0)
    assert next_state[3] > 0.0
    assert next_state[0] > 0.0


def test_agent_slot_range_closes_with_ego_motion() -> None:
    """A lead agent's forward range shrinks as the accelerating ego closes on it.

    Purpose: Validates the agent-slot motion model closes range under ego motion.

    Given: A stationary lead agent 20 m ahead and an accelerating ego
    When: sample_next_state is applied
    Then: the agent's ego-frame forward range decreases

    Test type: unit
    """
    model = _model()
    state = _state_with_agent(rel_x=20.0)
    next_state = model.sample_next_state(state, 0)
    assert next_state[EGO_STATE_WIDTH + 1] < 20.0


def test_agent_in_footprint_is_terminal() -> None:
    """An agent inside the ego footprint just ahead is a predicted collision.

    Purpose: Validates the model's predictive terminal check.

    Given: A present agent 2 m ahead within the collision corridor
    When: is_terminal is evaluated
    Then: it reports True

    Test type: unit
    """
    model = _model()
    assert model.is_terminal(_state_with_agent(rel_x=2.0)) is True


def test_obstacle_aware_speed_lowers_target_near_lead() -> None:
    """A close lead agent lowers the obstacle-aware desired speed used by the reward.

    Purpose: Validates the obstacle-aware target-speed ramp.

    Given: A model with stop_gap/safe_distance shaping and a near lead agent
    When: the internal obstacle-aware desired speed is computed close vs far
    Then: the close-lead target is below the far-lead (full) target

    Test type: unit
    """
    model = _model(stop_gap=2.0, safe_distance=12.0, desired_speed=8.0)
    near = model._obstacle_aware_desired_speed(  # pylint: disable=protected-access
        _state_with_agent(rel_x=6.0)
    )
    far = model._obstacle_aware_desired_speed(  # pylint: disable=protected-access
        _state_with_agent(rel_x=50.0)
    )
    assert near < far == pytest.approx(8.0)


def test_batch_transition_matches_single() -> None:
    """The batched transition equals stacking single-state transitions.

    Purpose: Validates sample_next_state_batch consistency with sample_next_state.

    Given: Two identical states and the accelerate action
    When: sample_next_state_batch is applied
    Then: each row equals the single-state transition

    Test type: unit
    """
    model = _model()
    state = _zero_state()
    single = model.sample_next_state(state, 0)
    batch = model.sample_next_state_batch([state, state], 0)
    assert np.allclose(batch[0], single) and np.allclose(batch[1], single)
