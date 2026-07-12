# SPDX-License-Identifier: MIT

"""Draw-coupled hazard-termination tests for the Light-Dark POMDPs.

Covers the ``hazard-terminates-episode v2`` redesign for both
:class:`ContinuousLightDarkPOMDP` (default flag-on) and
:class:`DiscreteLightDarkPOMDP` (opt-in flag-on):

* enabling the flag appends an absorbing terminal slot to the state;
* landing on an obstacle (not at goal) draws one hazard uniform and couples
  termination to a deterministic obstacle penalty;
* the terminal slot is absorbing;
* the shock reward model is rejected with the continuous flag;
* the flag-off path keeps the historical state width and behaviour.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    RewardModelType,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_beliefs.discrete_light_dark_vectorized_updater import (
    DiscreteLightDarkVectorizedUpdater,
)


# ---------------------------------------------------------------------------
# Continuous Light-Dark
# ---------------------------------------------------------------------------


def _continuous_env(**overrides):
    kwargs = {
        "obstacle_hit_probability": 1.0,
        "state_transition_cov_matrix": np.eye(2) * 1e-12,
        "obstacles": [(3.0, 7.0)],
        "obstacle_radius": 1.5,
    }
    kwargs.update(overrides)
    return ContinuousLightDarkPOMDP(discount_factor=0.95, **kwargs)


def test_continuous_flag_on_is_three_dim_and_draw_couples():
    """Flag-on continuous LD appends a slot and couples termination + penalty.

    Purpose: Validates the draw-coupled transition and deterministic reward at
        ``obstacle_hit_probability=1.0``.

    Given: A flag-on continuous env with a single obstacle at (3, 7) and a 3-D
        state whose robot sits inside that obstacle.
    When: A near-zero-noise transition is sampled and its reward taken.
    Then: The next state is 3-D and terminal, and the reward includes the
        deterministic obstacle penalty.

    Test type: unit
    """
    env = _continuous_env()
    state = np.array([3.0, 7.0, 0.0])
    next_state = env.sample_next_state(state, np.array([0.0, 0.0]))
    assert next_state.shape == (3,)
    assert next_state[2] == 1.0
    assert env.is_terminal(next_state)
    reward = env.reward(state, np.array([0.0, 0.0]), next_state=next_state)
    base = -env.fuel_cost - float(np.linalg.norm(next_state[:2] - env.goal_state))
    np.testing.assert_allclose(reward, base + env.obstacle_reward, atol=1e-9)


def test_continuous_terminal_is_absorbing():
    """An already-terminal continuous state freezes with the slot latched.

    Purpose: Validates the absorbing contract.

    Given: A flag-on continuous env and a 3-D state whose slot is set.
    When: A transition is sampled.
    Then: The next state equals the input.

    Test type: unit
    """
    env = _continuous_env()
    state = np.array([3.0, 7.0, 1.0])
    next_state = env.sample_next_state(state, np.array([1.0, 0.0]))
    np.testing.assert_array_equal(next_state, state)


def test_continuous_no_penalty_when_slot_unset():
    """Flag-on continuous reward applies the penalty only when the slot is set.

    Purpose: Validates the deterministic coupling in the reward.

    Given: Two next states differing only in the terminal slot, robot in an
        obstacle zone away from the goal.
    When: reward_batch is evaluated on both.
    Then: The reward difference equals ``obstacle_reward``.

    Test type: unit
    """
    env = _continuous_env()
    state = np.array([[3.0, 7.0, 0.0]])
    live = np.array([[3.0, 7.0, 0.0]])
    dead = np.array([[3.0, 7.0, 1.0]])
    action = np.array([0.0, 0.0])
    r_live = env.reward_batch(state, action, next_states=live)[0]
    r_dead = env.reward_batch(state, action, next_states=dead)[0]
    np.testing.assert_allclose(r_dead - r_live, env.obstacle_reward, atol=1e-9)


def test_continuous_shock_with_flag_raises():
    """The shock reward model is incompatible with the continuous flag.

    Purpose: Validates the construction-time guard.

    Given: is_obstacle_hit_terminal=True and the shock reward model.
    When: The env is constructed.
    Then: ValueError is raised.

    Test type: unit
    """
    with pytest.raises(ValueError):
        ContinuousLightDarkPOMDP(
            discount_factor=0.95,
            is_obstacle_hit_terminal=True,
            reward_model_type=RewardModelType.ZERO_MEAN_HAZARD_SHOCK,
        )


def test_continuous_flag_off_is_two_dim():
    """Flag-off continuous LD keeps the historical 2-D layout.

    Purpose: Validates the conditional slot (flag-off is untouched).

    Given: A flag-off continuous env.
    When: An initial state and a transition are sampled.
    Then: Both are 2-D and reward_requires_next_state is False.

    Test type: unit
    """
    env = ContinuousLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=False)
    assert env.reward_requires_next_state is False
    assert env.initial_state_dist().sample()[0].shape == (2,)
    assert env.sample_next_state(np.array([1.0, 5.0]), np.array([1.0, 0.0])).shape == (2,)


# ---------------------------------------------------------------------------
# Discrete Light-Dark
# ---------------------------------------------------------------------------


def _discrete_env(**overrides):
    kwargs = {
        "is_obstacle_hit_terminal": True,
        "obstacle_hit_probability": 1.0,
        "transition_error_prob": 0.0,
        "obstacles": [(5, 5)],
    }
    kwargs.update(overrides)
    return DiscreteLightDarkPOMDP(discount_factor=0.95, **kwargs)


def test_discrete_flag_on_draw_couples_and_penalises():
    """Flag-on discrete LD terminates on the obstacle Bernoulli and penalises.

    Purpose: Validates draw-coupled termination at ``hit_probability=1.0``.

    Given: A flag-on discrete env with an obstacle at (5, 5) and a 3-D state
        one cell away.
    When: The action stepping onto the obstacle is taken.
    Then: The next state is 3-D terminal and the reward includes the penalty.

    Test type: unit
    """
    env = _discrete_env()
    state = np.array([4, 5, 0])
    np.random.seed(0)
    next_state = env.sample_next_state(state, "right")
    assert next_state.shape == (3,)
    assert next_state[2] == 1.0
    assert env.is_terminal(next_state)
    reward = env.reward(state, "right", next_state=next_state)
    base = -env.fuel_cost - float(np.linalg.norm(next_state[:2] - env.goal_state))
    np.testing.assert_allclose(reward, base + env.obstacle_reward, atol=1e-9)


def test_discrete_obstacle_is_bernoulli_terminal_not_geometric():
    """Flag-on discrete LD: an obstacle cell is terminal only via a set slot.

    Purpose: Validates the geometric always-terminal gate is replaced by the
        draw-coupled slot.

    Given: A flag-on discrete env.
    When: is_terminal is queried on an obstacle cell with a live vs set slot.
    Then: Live is not terminal; set is terminal.

    Test type: unit
    """
    env = _discrete_env()
    assert not env.is_terminal(np.array([5, 5, 0]))
    assert env.is_terminal(np.array([5, 5, 1]))


def test_discrete_terminal_is_absorbing():
    """An already-terminal discrete state freezes with the slot latched.

    Purpose: Validates the absorbing contract.

    Given: A flag-on discrete env and a 3-D state whose slot is set.
    When: A transition is sampled.
    Then: The next state equals the input.

    Test type: unit
    """
    env = _discrete_env()
    state = np.array([5, 5, 1])
    next_state = env.sample_next_state(state, "up")
    np.testing.assert_array_equal(next_state, state)


def test_discrete_flag_off_is_two_dim_and_geometric_terminal():
    """Flag-off discrete LD keeps 2-D states and always-terminal obstacles.

    Purpose: Validates the default (opt-out) path is unchanged.

    Given: A default discrete env.
    When: An initial state is sampled and an obstacle cell is queried.
    Then: The state is 2-D and the obstacle cell is terminal (geometric).

    Test type: unit
    """
    env = DiscreteLightDarkPOMDP(discount_factor=0.95, obstacles=[(5, 5)])
    assert env.reward_requires_next_state is False
    assert env.initial_state_dist().sample()[0].shape == (2,)
    assert env.is_terminal(np.array([5, 5]))


def test_discrete_flag_on_vectorized_updater_batch_transition():
    """The flag-on discrete updater produces 3-D next particles.

    Purpose: Validates the vectorized updater threads the terminal slot.

    Given: A flag-on discrete env and its updater.
    When: batch_transition is called on 3-D particles.
    Then: The output is 3-D and a particle stepping onto the obstacle is
        terminal (hit_probability=1.0).

    Test type: integration
    """
    env = _discrete_env()
    updater = DiscreteLightDarkVectorizedUpdater.from_environment(env)
    particles = np.array([[4.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
    np.random.seed(0)
    nxt = updater.batch_transition(particles, "right")
    assert nxt.shape == (2, 3)
    assert nxt[0, 2] == 1.0  # stepped onto (5, 5)
