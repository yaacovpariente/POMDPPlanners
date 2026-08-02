# SPDX-License-Identifier: MIT

"""Tests for the kinematic CARLA generative model's ego transition.

Covers :class:`KinematicCarlaModelPOMDP` replacing the factored model's identity
transition with a real bicycle propagation: throttle produces forward motion, brake does
not, and the batched transition matches the single-sample one. All tests run on hand-built
state arrays with no CARLA server.
"""

import numpy as np

from POMDPPlanners.environments.carla_pomdp.carla_generative_models import (
    KinematicCarlaModelPOMDP,
)
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    DEFAULT_MAX_TRACKED_AGENTS,
    EGO_STATE_WIDTH,
)

_WIDTH = EGO_STATE_WIDTH + DEFAULT_MAX_TRACKED_AGENTS * AGENT_SLOT_WIDTH
_THROTTLE_ACTION = 0  # preset (0.5, 0.0, 0.0)
_BRAKE_ACTION = 3  # preset (0.0, 0.0, 1.0)


def _rest_state() -> np.ndarray:
    return np.zeros(_WIDTH)


def _state_with_agent(rel_x: float, rel_y: float, slot: int = 0) -> np.ndarray:
    """Rest state carrying one present agent at ``(rel_x, rel_y)`` in ego frame."""
    state = _rest_state()
    base = EGO_STATE_WIDTH + slot * AGENT_SLOT_WIDTH
    state[base] = 1.0  # present flag
    state[base + 1] = rel_x
    state[base + 2] = rel_y
    return state


def test_throttle_from_rest_produces_forward_velocity():
    """Throttle accelerates the resting ego forward.

    Purpose: Validates the kinematic transition converts throttle into speed.

    Given: A kinematic model and an ego at rest
    When: sample_next_state is called with the throttle preset
    Then: The resulting longitudinal velocity and position advance beyond zero

    Test type: unit
    """
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)

    next_state = env.sample_next_state(_rest_state(), _THROTTLE_ACTION)

    assert next_state[3] > 0.0  # vx grew
    assert next_state[0] > 0.0  # x advanced


def test_brake_from_rest_keeps_ego_stationary():
    """Braking from rest leaves the ego at rest.

    Purpose: Validates the transition does not manufacture motion under brake.

    Given: A kinematic model and an ego at rest
    When: sample_next_state is called with the brake preset
    Then: Velocity and position remain zero

    Test type: unit
    """
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)

    next_state = env.sample_next_state(_rest_state(), _BRAKE_ACTION)

    assert next_state[3] == 0.0
    assert next_state[0] == 0.0


def test_moving_ego_decelerates_under_brake():
    """Brake reduces the speed of a moving ego.

    Purpose: Validates the longitudinal brake deceleration term.

    Given: A kinematic model and an ego moving forward at 10 m/s
    When: sample_next_state is called with the brake preset
    Then: The resulting longitudinal velocity is lower than the initial speed

    Test type: unit
    """
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)
    state = _rest_state()
    state[3] = 10.0  # vx = 10 m/s along +x, yaw = 0

    next_state = env.sample_next_state(state, _BRAKE_ACTION)

    assert next_state[3] < 10.0


def test_batch_transition_matches_single_sample():
    """The batched transition equals the per-particle single transition.

    Purpose: Validates sample_next_state_batch is consistent with sample_next_state.

    Given: A kinematic model and three distinct particles
    When: sample_next_state_batch and per-particle sample_next_state are compared
    Then: The batched result matches the stacked single-sample results

    Test type: unit
    """
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)
    particles = np.stack([_rest_state(), _rest_state() + 0.5, _rest_state() + 1.0])

    batched = env.sample_next_state_batch(particles, _THROTTLE_ACTION)
    singles = np.stack([env.sample_next_state(p, _THROTTLE_ACTION) for p in particles])

    np.testing.assert_allclose(batched, singles)


def test_is_terminal_true_for_agent_directly_ahead():
    """A present agent within the forward collision box makes the state terminal.

    Purpose: Validates the kinematic model foresees a collision with a lead vehicle.

    Given: A kinematic model and a state with one present agent 2 m directly ahead
    When: is_terminal is queried
    Then: It returns True so the reward applies the terminal collision penalty

    Test type: unit
    """
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)

    assert env.is_terminal(_state_with_agent(rel_x=2.0, rel_y=0.0)) is True


def test_is_terminal_false_for_agent_behind_or_far_or_beside():
    """Agents behind, far ahead, or in an adjacent lane are not collisions.

    Purpose: Validates the collision box excludes non-threatening agent positions.

    Given: A kinematic model with default collision_gap/collision_halfwidth
    When: is_terminal is queried for agents behind, beyond the gap, and laterally offset
    Then: It returns False for every non-colliding placement

    Test type: unit
    """
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)

    assert env.is_terminal(_state_with_agent(rel_x=-2.0, rel_y=0.0)) is False  # behind
    assert env.is_terminal(_state_with_agent(rel_x=20.0, rel_y=0.0)) is False  # far ahead
    assert env.is_terminal(_state_with_agent(rel_x=2.0, rel_y=3.0)) is False  # adjacent lane


def test_is_terminal_false_when_slot_absent():
    """An empty agent slot at the collision position is ignored.

    Purpose: Validates only present agents trigger the terminal collision.

    Given: A kinematic model and a state whose agent slot carries a pose but present == 0
    When: is_terminal is queried
    Then: It returns False because the slot is not marked present

    Test type: unit
    """
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)
    state = _state_with_agent(rel_x=2.0, rel_y=0.0)
    state[EGO_STATE_WIDTH] = 0.0  # clear the present flag, keep the pose

    assert env.is_terminal(state) is False


def test_reward_applies_collision_penalty_on_predicted_collision():
    """The inherited reward charges the collision penalty on a terminal state.

    Purpose: Validates is_terminal feeds the reward's terminal collision term.

    Given: A kinematic model and a state with an agent inside the forward collision box
    When: reward is evaluated with that state as the resulting next_state
    Then: The reward is lower than for the same state with no agent present

    Test type: unit
    """
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)
    colliding = _state_with_agent(rel_x=2.0, rel_y=0.0)
    clear = _rest_state()

    reward_colliding = env.reward(clear, _THROTTLE_ACTION, next_state=colliding)
    reward_clear = env.reward(clear, _THROTTLE_ACTION, next_state=clear)

    assert reward_colliding < reward_clear - env.collision_penalty / 2


def test_multiple_samples_return_stacked_copies():
    """Requesting several next states returns a stacked array of the propagation.

    Purpose: Validates the n_samples>1 convention of the kinematic transition.

    Given: A kinematic model and a resting ego
    When: sample_next_state is called with n_samples=4
    Then: A (4, width) array is returned whose rows all equal the single propagation

    Test type: unit
    """
    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)

    single = env.sample_next_state(_rest_state(), _THROTTLE_ACTION)
    batch = env.sample_next_state(_rest_state(), _THROTTLE_ACTION, n_samples=4)

    assert batch.shape == (4, _WIDTH)
    np.testing.assert_allclose(batch, np.stack([single] * 4))
