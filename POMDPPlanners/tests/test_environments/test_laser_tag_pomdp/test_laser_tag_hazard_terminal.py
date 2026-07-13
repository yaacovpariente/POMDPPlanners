# SPDX-License-Identifier: MIT

"""Draw-coupled hazard-termination tests for both LaserTag environments.

Covers the ``is_dangerous_area_hit_terminal`` opt-in for
:class:`~POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp.LaserTagPOMDP`
(discrete) and
:class:`~POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp.ContinuousLaserTagPOMDP`
(continuous): flag-off is behaviour-preserving, flag-on couples termination to
the hazard hit and makes the reward deterministic, the shock reward model is
rejected, and the reward_range lower bound includes the dangerous-area penalty.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.laser_tag_pomdp import _native
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDP,
    ContinuousLaserTagPOMDPDiscreteActions,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import (
    LaserTagPOMDP,
    RewardModelType,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_beliefs.laser_tag_vectorized_updater import (  # noqa: E501
    LaserTagVectorizedUpdater,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


class _CyclingActionSampler(ActionSampler):
    """Deterministic action sampler cycling through a fixed list."""

    def __init__(self, actions):
        self._actions = actions
        self._idx = 0

    def sample(self, belief_node=None):  # noqa: D401
        action = self._actions[self._idx % len(self._actions)]
        self._idx += 1
        return action


# ─────────────────────────── Discrete LaserTag ───────────────────────────


def test_discrete_flag_defaults_false_and_no_next_state_dependency():
    """Discrete LaserTag defaults to the legacy (non-terminal-hazard) behaviour.

    Purpose: Validates the opt-in flag defaults off and does not alter the
        reward-timing contract when disabled.

    Given: A default ``LaserTagPOMDP``.
    When: The flag and ``reward_requires_next_state`` are inspected.
    Then: The flag is ``False`` and reward does not require the next state.

    Test type: unit
    """
    env = LaserTagPOMDP(discount_factor=0.95)
    assert env.is_dangerous_area_hit_terminal is False
    assert env.reward_requires_next_state is False


def test_discrete_flag_with_shock_model_raises():
    """Combining the flag with the shock reward model is rejected.

    Purpose: Validates the guard against an ill-defined hazard coupling.

    Given: ``is_dangerous_area_hit_terminal=True`` and ``ZERO_MEAN_HAZARD_SHOCK``.
    When: The environment is constructed.
    Then: A ``ValueError`` is raised.

    Test type: unit
    """
    with pytest.raises(ValueError, match="ZERO_MEAN_HAZARD_SHOCK"):
        LaserTagPOMDP(
            discount_factor=0.95,
            is_dangerous_area_hit_terminal=True,
            reward_model_type=RewardModelType.ZERO_MEAN_HAZARD_SHOCK,
        )


def test_discrete_constant_flag_terminates_and_penalises_on_danger_entry():
    """CONSTANT hazard entry terminates (p=1) with a deterministic penalty.

    Purpose: Validates draw-coupled termination and the deterministic penalty
        for the CONSTANT reward model.

    Given: The robot moves North into the dangerous cell ``(4, 3)`` (within
        radius 1 of centre ``(5, 3)``).
    When: The transition and reward are evaluated.
    Then: The next state is terminal and the reward is ``-step_cost - penalty``.

    Test type: integration
    """
    env = LaserTagPOMDP(discount_factor=0.95, is_dangerous_area_hit_terminal=True)
    np.random.seed(0)
    state = np.array([5.0, 3.0, 0.0, 0.0, 0.0])
    next_state = env.sample_next_state(state, 0)  # North -> (4, 3)
    assert bool(next_state[4]) is True
    reward = env.reward(state, 0, next_state=next_state)
    assert reward == pytest.approx(-env.step_cost - env.dangerous_area_penalty)


def test_discrete_flag_off_never_terminates_via_hazard():
    """Flag-off transitions only terminate on a successful tag, never on hazard.

    Purpose: Validates flag-off behaviour is unchanged by the feature.

    Given: A default (flag-off) env, robot standing on a danger centre.
    When: Many non-tag transitions are sampled.
    Then: No sampled next state is terminal.

    Test type: unit
    """
    env = LaserTagPOMDP(discount_factor=0.95)
    np.random.seed(0)
    terminals = [
        env.sample_next_state(np.array([5.0, 3.0, 0.0, 0.0, 0.0]), 0)[4] for _ in range(50)
    ]
    assert not any(terminals)


def test_discrete_decayed_flag_termination_rate_matches_hit_probability():
    """DECAYED hazard termination fires with probability exp(-min_dist/decay).

    Purpose: Validates the decayed hazard couples termination to the decayed
        hit probability.

    Given: The robot holds (tag) at distance 1 from the nearest centre, decay 1.
    When: Many transitions are sampled.
    Then: The empirical termination rate approximates ``exp(-1)``.

    Test type: performance
    """
    env = LaserTagPOMDP(
        discount_factor=0.95,
        is_dangerous_area_hit_terminal=True,
        reward_model_type=RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY,
        penalty_decay=1.0,
    )
    np.random.seed(3)
    # Robot at (4, 3): min distance to centre (5, 3) is 1.0; tag keeps it put.
    rate = np.mean(
        [env.sample_next_state(np.array([4.0, 3.0, 0.0, 0.0, 0.0]), 4)[4] for _ in range(4000)]
    )
    assert rate == pytest.approx(np.exp(-1.0), abs=0.03)


def test_discrete_native_and_python_transition_agree_under_flag():
    """Native and Python single-step paths agree on the hazard outcome.

    Purpose: Validates python⇔C++ parity of the draw-coupled transition.

    Given: The same seed and state fed to the native fast path and the pure
        Python path.
    When: Both sample the next state under the flag.
    Then: The resulting next states (including the terminal slot) are identical.

    Test type: integration
    """
    env = LaserTagPOMDP(discount_factor=0.95, is_dangerous_area_hit_terminal=True)
    state = np.array([5.0, 3.0, 8.0, 2.0, 0.0])
    params = env._get_native_step_params()  # pylint: disable=protected-access
    assert params is not None
    np.random.seed(11)
    native = env.sample_next_state(state, 0)
    np.random.seed(11)
    # Force the pure-Python path (params=None disables the native fast path).
    result = env._python_sample_next_state(state, 0, 1)  # pylint: disable=protected-access
    python = env._maybe_terminate_one(result)  # pylint: disable=protected-access
    np.testing.assert_array_equal(np.asarray(native), np.asarray(python))


def test_discrete_rollout_under_flag_is_finite():
    """The flag-on discrete rollout runs via the Python draw-coupled path.

    Purpose: Validates rollout dispatch does not use the hazard-unaware native
        kernel when the flag is on.

    Given: A flag-on env and a cycling action sampler.
    When: A random rollout is simulated.
    Then: A finite discounted return is produced.

    Test type: integration
    """
    env = LaserTagPOMDP(discount_factor=0.95, is_dangerous_area_hit_terminal=True)
    np.random.seed(7)
    sampler = _CyclingActionSampler([0, 1, 2, 3, 4])
    value = env.simulate_random_rollout(
        state=np.array([5.0, 3.0, 8.0, 2.0, 0.0]),
        action_sampler=sampler,
        max_depth=20,
        discount_factor=0.95,
        depth=0,
    )
    assert np.isfinite(value)


def test_discrete_belief_updater_marks_hazard_terminals_under_flag():
    """The discrete belief updater terminates particles on a hazard hit.

    Purpose: Validates hazard params are threaded into the vectorized updater.

    Given: A flag-on CONSTANT env with all particles on a danger centre.
    When: A batch transition (non-tag, blocked-in-danger) is applied.
    Then: All next particles are terminal (p=1 hazard hit in the zone).

    Test type: integration
    """
    env = LaserTagPOMDP(discount_factor=0.95, is_dangerous_area_hit_terminal=True)
    updater = LaserTagVectorizedUpdater.from_environment(env)
    _native.set_seed(0)
    np.random.seed(0)
    # Robot on centre (5, 3); North -> (4, 3) still within the danger radius.
    particles = np.tile(np.array([5.0, 3.0, 0.0, 0.0, 0.0]), (64, 1))
    next_particles = updater.batch_transition(particles, np.asarray(0))
    assert np.all(next_particles[:, 4] == 1.0)


# ────────────────────────── Continuous LaserTag ──────────────────────────


def test_continuous_flag_defaults_false():
    """Continuous LaserTag defaults to the legacy stochastic-reward behaviour.

    Purpose: Validates the opt-in flag defaults off with no next-state
        dependency.

    Given: A default ``ContinuousLaserTagPOMDP``.
    When: The flag and ``reward_requires_next_state`` are inspected.
    Then: Both are ``False``.

    Test type: unit
    """
    env = ContinuousLaserTagPOMDP(discount_factor=0.95)
    assert env.is_dangerous_area_hit_terminal is False
    assert env.reward_requires_next_state is False


def test_continuous_reward_range_includes_dangerous_area_penalty():
    """The continuous reward_range lower bound accounts for the danger penalty.

    Purpose: Validates the reward_range fix (danger penalty was omitted).

    Given: A continuous env with explicit penalties.
    When: The reward_range is read.
    Then: The lower bound equals ``-tag_penalty - step_cost - danger_penalty``.

    Test type: unit
    """
    env = ContinuousLaserTagPOMDP(
        discount_factor=0.95, tag_penalty=10.0, step_cost=1.0, dangerous_area_penalty=5.0
    )
    assert env.reward_range[0] == pytest.approx(-16.0)
    assert env.reward_range[1] == pytest.approx(10.0)


def test_continuous_flag_on_terminates_and_penalises_in_danger_area():
    """Continuous hazard entry terminates (p=1) with a deterministic penalty.

    Purpose: Validates draw-coupled termination + deterministic reward for the
        continuous env.

    Given: Near-zero robot noise and the robot inside danger centre ``(5, 3)``.
    When: The transition and reward are evaluated with the realised next state.
    Then: The next state is terminal and the reward is ``-step_cost - penalty``.

    Test type: integration
    """
    env = ContinuousLaserTagPOMDP(
        discount_factor=0.95,
        is_dangerous_area_hit_terminal=True,
        robot_transition_cov_matrix=np.eye(2) * 1e-9,
    )
    np.random.seed(0)
    _native.set_seed(0)
    state = np.array([5.0, 3.0, 8.0, 2.0, 0.0])
    action = np.array([0.0, 0.0, 0.0])
    next_state = env.sample_next_state(state, action)
    assert bool(next_state[4]) is True
    reward = env.reward(state, action, next_state=next_state)
    assert reward == pytest.approx(-env.step_cost - env.dangerous_area_penalty)


def test_continuous_flag_on_partial_probability_terminates_statistically():
    """Continuous hazard termination fires at the configured probability.

    Purpose: Validates the constant hazard probability couples termination.

    Given: ``dangerous_area_hit_probability=0.5`` and the robot in a zone.
    When: Many transitions are sampled.
    Then: The empirical termination rate approximates 0.5.

    Test type: performance
    """
    env = ContinuousLaserTagPOMDP(
        discount_factor=0.95,
        is_dangerous_area_hit_terminal=True,
        dangerous_area_hit_probability=0.5,
        robot_transition_cov_matrix=np.eye(2) * 1e-9,
    )
    _native.set_seed(123)
    rate = np.mean(
        [
            env.sample_next_state(np.array([5.0, 3.0, 8.0, 2.0, 0.0]), np.array([0.0, 0.0, 0.0]))[4]
            for _ in range(4000)
        ]
    )
    assert rate == pytest.approx(0.5, abs=0.04)


def test_continuous_flag_off_transition_never_terminates_via_hazard():
    """Flag-off continuous transitions never terminate on a hazard.

    Purpose: Validates flag-off byte-preserving behaviour of the transition.

    Given: A default (flag-off) continuous env, robot inside a danger zone.
    When: Many non-tag transitions are sampled.
    Then: No sampled next state is terminal.

    Test type: unit
    """
    env = ContinuousLaserTagPOMDP(
        discount_factor=0.95, robot_transition_cov_matrix=np.eye(2) * 1e-9
    )
    _native.set_seed(0)
    terminals = [
        env.sample_next_state(np.array([5.0, 3.0, 8.0, 2.0, 0.0]), np.array([0.0, 0.0, 0.0]))[4]
        for _ in range(50)
    ]
    assert not any(terminals)


def test_continuous_discrete_actions_variant_threads_flag():
    """The continuous discrete-action variant forwards the hazard flag.

    Purpose: Validates the subclass ctor threads the flag to the parent.

    Given: A ``ContinuousLaserTagPOMDPDiscreteActions`` with the flag on.
    When: The flag and reward contract are inspected.
    Then: The flag is set and reward requires the next state.

    Test type: unit
    """
    env = ContinuousLaserTagPOMDPDiscreteActions(
        discount_factor=0.95,
        is_dangerous_area_hit_terminal=True,
        robot_transition_cov_matrix=np.eye(2) * 1e-9,
    )
    assert env.is_dangerous_area_hit_terminal is True
    assert env.reward_requires_next_state is True
    # reward with next_state=None must resample via the vector-accepting parent
    # transition (the str-label override would otherwise KeyError).
    np.random.seed(0)
    _native.set_seed(0)
    reward = env.reward(np.array([5.0, 3.0, 8.0, 2.0, 0.0]), "up")
    assert reward == pytest.approx(-env.step_cost - env.dangerous_area_penalty)


def test_continuous_belief_updater_marks_hazard_terminals_under_flag():
    """The continuous belief updater terminates particles on a hazard hit.

    Purpose: Validates hazard params are threaded into the C++ batch kernel via
        the continuous vectorized updater.

    Given: A flag-on continuous env (near-zero noise) with all particles inside
        a danger zone.
    When: A batch transition is applied.
    Then: All next particles are terminal.

    Test type: integration
    """
    from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_beliefs.continuous_laser_tag_vectorized_updater import (  # noqa: E501  pylint: disable=import-outside-toplevel
        ContinuousLaserTagVectorizedUpdater,
    )

    env = ContinuousLaserTagPOMDP(
        discount_factor=0.95,
        is_dangerous_area_hit_terminal=True,
        robot_transition_cov_matrix=np.eye(2) * 1e-9,
    )
    updater = ContinuousLaserTagVectorizedUpdater.from_environment(env)
    _native.set_seed(0)
    particles = np.tile(np.array([5.0, 3.0, 8.0, 2.0, 0.0]), (64, 1))
    next_particles = updater.batch_transition(particles, np.array([0.0, 0.0, 0.0]))
    assert np.all(next_particles[:, 4] == 1.0)
