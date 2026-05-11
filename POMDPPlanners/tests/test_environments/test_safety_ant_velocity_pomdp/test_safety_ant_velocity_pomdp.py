"""Tests for Safety Ant Velocity POMDP environment.

This module tests the Safety Ant Velocity POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

# pylint: disable=too-many-lines

from unittest.mock import Mock

import numpy as np
import pytest

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.safety_ant_velocity_pomdp import SafeAntVelocityPOMDP
from POMDPPlanners.tests.test_utils.confidence_interval_utils import (
    verify_metrics_within_confidence_intervals,
)
from POMDPPlanners.tests.test_utils.metric_invariants_utils import (
    verify_history_returns_bounded,
    verify_metric_sanity,
    verify_return_shift_linearity,
)

# Set seeds for reproducible tests


@pytest.fixture
def pomdp():
    return SafeAntVelocityPOMDP(discount_factor=0.95)


def test_safe_velocity_pomdp_initialization():
    """Test safe velocity pomdp initialization.

    Purpose: Validates proper initialization of safe velocity pomdp

    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes

    Test type: unit
    """
    # Test basic initialization
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        safe_velocity_threshold=2.0,
        max_force=1.0,
        dt=0.1,
        mass=1.0,
        damping=0.1,
        position_noise=0.1,
        velocity_noise=0.2,
        safety_violation_penalty=-100.0,
        movement_reward_scale=1.0,
    )

    assert env.discount_factor == 0.95
    assert env.safe_velocity_threshold == 2.0
    assert env.max_force == 1.0
    assert env.dt == 0.1
    assert env.mass == 1.0
    assert env.damping == 0.1
    assert env.position_noise == 0.1
    assert env.velocity_noise == 0.2
    assert env.safety_violation_penalty == -100.0
    assert env.movement_reward_scale == 1.0
    assert env.actions == [0, 1, 2, 3]


def test_state_transition_with_force_via_env(pomdp):
    """Test state transition behavior with force application via env API.

    Purpose: Validates that env.sample_next_state correctly updates position and
    velocity when force is applied.

    Given: SafeAntVelocityPOMDP environment, state [0.0, 0.0, 1.0, 1.0], action 3
    When: env.sample_next_state is called
    Then: Next state has correct 4D shape, position changes due to velocity, and
        velocity changes due to applied force.

    Test type: unit
    """
    state = np.array([0.0, 0.0, 1.0, 1.0])  # [pos_x, pos_y, vel_x, vel_y]
    action = 3  # Maximum force

    next_state = pomdp.sample_next_state(state=state, action=action)

    # Verify state dimensions
    assert next_state.shape == (4,)

    # Verify position changed due to velocity
    assert not np.array_equal(next_state[:2], state[:2])

    # Verify velocity changed due to force
    assert not np.array_equal(next_state[2:], state[2:])


def test_state_transition_no_force_damping_via_env(pomdp):
    """Test state transition behavior without force application via env API.

    Purpose: Validates that env.sample_next_state correctly handles damping when no
    force is applied.

    Given: SafeAntVelocityPOMDP environment, state [0.0, 0.0, 1.0, 1.0], action 0
    When: env.sample_next_state is called
    Then: Velocity decreases due to damping effects while maintaining state shape.

    Test type: unit
    """
    state = np.array([0.0, 0.0, 1.0, 1.0])
    action = 0  # No force

    next_state = pomdp.sample_next_state(state=state, action=action)

    # Verify shape
    assert next_state.shape == (4,)

    # Verify velocity decreased due to damping
    assert np.linalg.norm(next_state[2:]) < np.linalg.norm(state[2:])


def test_observation_adds_noise_via_env(pomdp):
    """Test observation behavior with noise addition via env API.

    Purpose: Validates that env.sample_observation correctly adds noise to position
    and velocity measurements.

    Given: SafeAntVelocityPOMDP environment and next_state [0.0, 0.0, 1.0, 1.0]
    When: env.sample_observation is called
    Then: Observation has correct 4D shape and contains noise in both position and
        velocity components.

    Test type: unit
    """
    next_state = np.array([0.0, 0.0, 1.0, 1.0])
    action = 1

    observation = pomdp.sample_observation(next_state=next_state, action=action)

    # Verify observation dimensions
    assert observation.shape == (4,)

    # Verify position has noise
    assert not np.array_equal(observation[:2], next_state[:2])

    # Verify velocity has noise
    assert not np.array_equal(observation[2:], next_state[2:])


def test_reward_function():
    """Test reward function for safe and unsafe velocity states.

    Purpose: Validates that reward function correctly penalizes safety violations and rewards safe movement

    Given: SafeAntVelocityPOMDP environment with safe_velocity_threshold=2.0 and safety_violation_penalty=-100.0
    When: Reward is calculated for safe velocity state [0.0, 0.0, 1.0, 1.0] and unsafe velocity state [0.0, 0.0, 2.0, 2.0]
    Then: Safe state receives positive reward, unsafe state receives negative penalty

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        safe_velocity_threshold=2.0,
        safety_violation_penalty=-100.0,
        movement_reward_scale=1.0,
    )

    # Test reward for safe velocity
    state_safe = np.array([0.0, 0.0, 1.0, 1.0])  # Speed = sqrt(2) ≈ 1.4
    reward_safe = env.reward(state_safe, 1)
    assert reward_safe > 0  # Positive reward for movement within safe limits

    # Test reward for unsafe velocity
    state_unsafe = np.array([0.0, 0.0, 2.0, 2.0])  # Speed = 2*sqrt(2) ≈ 2.8
    reward_unsafe = env.reward(state_unsafe, 1)
    assert reward_unsafe < 0  # Negative reward due to safety violation


def test_terminal_state():
    """Test terminal state detection based on velocity threshold.

    Purpose: Validates that environment correctly identifies terminal states when velocity exceeds safety threshold

    Given: SafeAntVelocityPOMDP environment with safe_velocity_threshold=2.0
    When: is_terminal is called for safe velocity state [0.0, 0.0, 1.0, 1.0] and unsafe velocity state [0.0, 0.0, 3.0, 3.0]
    Then: Safe state returns False, unsafe state returns True indicating terminal condition

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        safe_velocity_threshold=2.0,
    )

    # Test non-terminal state (safe velocity)
    state_safe = np.array([0.0, 0.0, 1.0, 1.0])  # Speed = sqrt(2) ≈ 1.4
    assert not env.is_terminal(state_safe)

    # Test terminal state (very unsafe velocity)
    state_unsafe = np.array([0.0, 0.0, 3.0, 3.0])  # Speed = 3*sqrt(2) ≈ 4.2
    assert env.is_terminal(state_unsafe)


def test_reward_range():
    """Test that reward range is correctly calculated.

    Purpose: Validates that SafeAntVelocityPOMDP reward range is properly calculated based on environment parameters

    Given: A SafeAntVelocityPOMDP environment with specific safety parameters
    When: Environment reward_range attribute is checked
    Then: Returns range based on min_reward (no movement + safety penalty) and max_reward (max safe speed * movement scale)

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        safe_velocity_threshold=2.0,
        safety_violation_penalty=-50.0,
        movement_reward_scale=1.5,
    )

    # Expected calculations from SafeAntVelocityPOMDP constructor:
    # min_reward = 0.0 + safety_violation_penalty = 0.0 + (-50.0) = -50.0
    # max_reward = safe_velocity_threshold * 1.5 * movement_reward_scale = 2.0 * 1.5 * 1.5 = 4.5
    expected_min = 0.0 + (-50.0)  # -50.0
    expected_max = 2.0 * 1.5 * 1.5  # 4.5

    assert env.reward_range == (expected_min, expected_max)

    # Test with different parameters
    env2 = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        safe_velocity_threshold=3.0,
        safety_violation_penalty=-100.0,
        movement_reward_scale=2.0,
    )

    expected_min2 = 0.0 + (-100.0)  # -100.0
    expected_max2 = 3.0 * 1.5 * 2.0  # 9.0

    assert env2.reward_range == (expected_min2, expected_max2)


def test_initial_state_distribution():
    """Test initial state distribution sampling and bounds.

    Purpose: Validates that initial state distribution generates valid states within expected bounds

    Given: SafeAntVelocityPOMDP environment with default parameters
    When: Initial state distribution is sampled multiple times
    Then: All samples have 4D shape, position components are within [-1, 1] bounds, and velocity components are zero

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(discount_factor=0.95)
    initial_dist = env.initial_state_dist()

    # Test multiple samples
    for _ in range(10):
        state = initial_dist.sample()[0]

        # Verify state dimensions
        assert state.shape == (4,)

        # Verify position bounds
        assert np.all(state[:2] >= -1)
        assert np.all(state[:2] <= 1)

        # Verify zero initial velocity
        assert np.allclose(state[2:], 0)


def test_get_actions():
    """Test action space retrieval.

    Purpose: Validates that environment provides correct discrete action space

    Given: SafeAntVelocityPOMDP environment with default configuration
    When: get_actions method is called
    Then: Returns list of 4 actions [0, 1, 2, 3] representing different force levels from no force to maximum force

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(discount_factor=0.95)
    actions = env.get_actions()

    assert len(actions) == 4
    assert 0 in actions  # No force
    assert 3 in actions  # Maximum force


def test_is_equal_observation():
    """Test observation equality comparison.

    Purpose: Validates that observation equality comparison works correctly for identical and different observations

    Given: SafeAntVelocityPOMDP environment and observation pairs
    When: is_equal_observation is called for identical observations [0.0, 0.0, 1.0, 1.0] and different observations [0.1, 0.0, 1.0, 1.0]
    Then: Returns True for identical observations, False for different observations

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(discount_factor=0.95)

    # Test equal observations
    obs1 = np.array([0.0, 0.0, 1.0, 1.0])
    obs2 = np.array([0.0, 0.0, 1.0, 1.0])
    assert env.is_equal_observation(obs1, obs2)

    # Test different observations
    obs3 = np.array([0.1, 0.0, 1.0, 1.0])
    assert not env.is_equal_observation(obs1, obs3)


def test_sample_next_step():
    """Test complete environment step simulation.

    Purpose: Validates that sample_next_step correctly simulates state transitions, observations, and rewards

    Given: SafeAntVelocityPOMDP environment, initial state [0.0, 0.0, 1.0, 1.0], and action 2 (medium force)
    When: sample_next_step is called
    Then: Returns valid next_state, observation, and reward with correct types, shapes, and state changes due to force application

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(discount_factor=0.95)
    state = np.array([0.0, 0.0, 1.0, 1.0])
    action = 2  # Medium force

    next_state, observation, reward = env.sample_next_step(state, action)

    # Verify return types and shapes
    assert isinstance(next_state, np.ndarray)
    assert isinstance(observation, np.ndarray)
    assert isinstance(reward, float)
    assert next_state.shape == (4,)
    assert observation.shape == (4,)

    # Verify state transition
    assert not np.array_equal(next_state[:2], state[:2])  # Position changed
    assert not np.array_equal(next_state[2:], state[2:])  # Velocity changed

    # Verify observation has noise
    assert not np.array_equal(observation, next_state)

    # Verify reward is calculated (using np.isclose for float comparison)
    expected_reward = env.reward(next_state, action)
    assert np.isclose(reward, expected_reward, rtol=1e-10)


def test_compute_metrics():
    """Test safety metrics computation from simulation histories.

    Purpose: Validates that environment correctly computes safety violation rates and critical violation rates from episode histories

    Given: SafeAntVelocityPOMDP environment and test histories with different velocity scenarios (safe, mixed safety violations, critical violations)
    When: compute_metrics is called with the histories
    Then: Returns correct safety_violation_rate and critical_violation_rate metrics with proper confidence intervals

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        safe_velocity_threshold=2.0,
    )

    # Create test histories with different velocity scenarios
    histories = []

    # History 1: All safe velocities
    safe_steps = [
        StepData(
            state=np.array([0.0, 0.0, 1.0, 1.0]),
            action=1,
            observation=np.array([0.0, 0.0, 1.0, 1.0]),
            reward=1.0,
            next_state=np.array([0.1, 0.1, 1.1, 1.1]),
            belief=Mock(spec=Belief),
        ),
        StepData(
            state=np.array([0.1, 0.1, 1.1, 1.1]),
            action=1,
            observation=np.array([0.1, 0.1, 1.1, 1.1]),
            reward=1.0,
            next_state=np.array([0.2, 0.2, 1.2, 1.2]),
            belief=Mock(spec=Belief),
        ),
    ]
    histories.append(
        History(
            history=safe_steps,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=len(safe_steps),
            reach_terminal_state=False,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )
    )

    # History 2: Some safety violations
    mixed_steps = [
        StepData(
            state=np.array([0.0, 0.0, 1.0, 1.0]),
            action=1,
            observation=np.array([0.0, 0.0, 1.0, 1.0]),
            reward=1.0,
            next_state=np.array([0.1, 0.1, 1.1, 1.1]),
            belief=Mock(spec=Belief),
        ),
        StepData(
            state=np.array([0.1, 0.1, 2.1, 2.1]),  # Only this state has unsafe velocity
            action=3,
            observation=np.array([0.1, 0.1, 2.1, 2.1]),
            reward=-100.0,
            next_state=np.array([0.2, 0.2, 1.2, 1.2]),  # Back to safe velocity
            belief=Mock(spec=Belief),
        ),
    ]
    histories.append(
        History(
            history=mixed_steps,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=len(mixed_steps),
            reach_terminal_state=False,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )
    )

    # History 3: Critical violation
    critical_steps = [
        StepData(
            state=np.array([0.0, 0.0, 3.1, 3.1]),
            action=3,
            observation=np.array([0.0, 0.0, 3.1, 3.1]),
            reward=-100.0,
            next_state=np.array([0.1, 0.1, 3.2, 3.2]),
            belief=Mock(spec=Belief),
        ),
    ]
    histories.append(
        History(
            history=critical_steps,
            discount_factor=0.95,
            average_state_sampling_time=0.0,
            average_action_time=0.0,
            average_observation_time=0.0,
            average_belief_update_time=0.0,
            average_reward_time=0.0,
            actual_num_steps=len(critical_steps),
            reach_terminal_state=True,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )
    )

    # Compute metrics
    metrics = env.compute_metrics(histories)

    # Convert metrics to dictionary for easier access
    metrics_dict = {m.name: m for m in metrics}

    # Calculate expected rates
    total_steps = len(safe_steps) + len(mixed_steps) + len(critical_steps)
    safety_violations = 0 + 1 + 1  # One violation in mixed_steps, one in critical_steps
    critical_violations = 0 + 0 + 1  # One violation in critical_steps
    expected_safety_rate = safety_violations / total_steps
    expected_critical_rate = critical_violations / total_steps

    # Verify metrics
    assert np.isclose(metrics_dict["safety_violation_rate"].value, expected_safety_rate)
    assert np.isclose(metrics_dict["critical_violation_rate"].value, expected_critical_rate)

    # Verify confidence intervals
    assert (
        metrics_dict["safety_violation_rate"].lower_confidence_bound
        <= metrics_dict["safety_violation_rate"].value
    )
    assert (
        metrics_dict["safety_violation_rate"].upper_confidence_bound
        >= metrics_dict["safety_violation_rate"].value
    )
    assert (
        metrics_dict["critical_violation_rate"].lower_confidence_bound
        <= metrics_dict["critical_violation_rate"].value
    )
    assert (
        metrics_dict["critical_violation_rate"].upper_confidence_bound
        >= metrics_dict["critical_violation_rate"].value
    )


def test_compute_metrics_values_within_confidence_intervals():
    """Test SafeAntVelocityPOMDP metric values are inside CIs and pass invariants.

    Purpose: Validates that metrics produced by compute_metrics lie inside
        their CI bounds and that all structural invariants hold (rate-in-[0,1],
        counts >= 0, finite CI for n>=2, returns inside reward bounds, and
        return-shift linearity).

    Given: A SafeAntVelocityPOMDP and 3 hand-built histories with varied
        velocity profiles (all-safe, mixed, critical). Rewards lie in
        [-100, 3.0] as declared by reward_range.
    When: compute_metrics is called and the four invariant helpers are run.
    Then: All checks pass without raising.

    Test type: integration
    """
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        safe_velocity_threshold=2.0,
    )

    # History 0: all-safe.
    safe_steps = [
        StepData(
            state=np.array([0.0, 0.0, 1.0, 1.0]),
            action=1,
            observation=np.array([0.0, 0.0, 1.0, 1.0]),
            reward=1.0,
            next_state=np.array([0.1, 0.1, 1.1, 1.1]),
            belief=Mock(spec=Belief),
        ),
        StepData(
            state=np.array([0.1, 0.1, 1.1, 1.1]),
            action=1,
            observation=np.array([0.1, 0.1, 1.1, 1.1]),
            reward=1.0,
            next_state=np.array([0.2, 0.2, 1.2, 1.2]),
            belief=Mock(spec=Belief),
        ),
    ]

    # History 1: mixed (one safety violation, no critical).
    mixed_steps = [
        StepData(
            state=np.array([0.0, 0.0, 1.0, 1.0]),
            action=1,
            observation=np.array([0.0, 0.0, 1.0, 1.0]),
            reward=1.0,
            next_state=np.array([0.1, 0.1, 1.1, 1.1]),
            belief=Mock(spec=Belief),
        ),
        StepData(
            state=np.array([0.1, 0.1, 2.1, 2.1]),
            action=3,
            observation=np.array([0.1, 0.1, 2.1, 2.1]),
            reward=-100.0,
            next_state=np.array([0.2, 0.2, 1.2, 1.2]),
            belief=Mock(spec=Belief),
        ),
    ]

    # History 2: critical violation.
    critical_steps = [
        StepData(
            state=np.array([0.0, 0.0, 3.1, 3.1]),
            action=3,
            observation=np.array([0.0, 0.0, 3.1, 3.1]),
            reward=-100.0,
            next_state=np.array([0.1, 0.1, 3.2, 3.2]),
            belief=Mock(spec=Belief),
        ),
    ]

    histories = []
    for steps, reach_terminal in (
        (safe_steps, False),
        (mixed_steps, False),
        (critical_steps, True),
    ):
        histories.append(
            History(
                history=steps,
                discount_factor=0.95,
                average_state_sampling_time=0.0,
                average_action_time=0.0,
                average_observation_time=0.0,
                average_belief_update_time=0.0,
                average_reward_time=0.0,
                actual_num_steps=len(steps),
                reach_terminal_state=reach_terminal,
                policy_run_data=[PolicyRunData(info_variables=[])],
            )
        )

    metrics = env.compute_metrics(histories)
    verify_metrics_within_confidence_intervals(metrics)
    verify_metric_sanity(metrics, histories, env)
    verify_history_returns_bounded(histories, env)
    verify_return_shift_linearity(histories, env, shift=1.5)


def test_environment_equality():
    """Test environment equality comparison.

    Purpose: Validates that environment equality comparison works correctly for identical and different configurations

    Given: Two SafeAntVelocityPOMDP environments with identical parameters and one with different discount factor
    When: Equality comparison is performed
    Then: Identical environments are equal, environments with different parameters are not equal

    Test type: unit
    """
    # Create two identical environments
    env1 = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        safe_velocity_threshold=2.0,
        max_force=1.0,
        dt=0.1,
        mass=1.0,
        damping=0.1,
        position_noise=0.1,
        velocity_noise=0.2,
        safety_violation_penalty=-100.0,
        movement_reward_scale=1.0,
    )
    env2 = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        safe_velocity_threshold=2.0,
        max_force=1.0,
        dt=0.1,
        mass=1.0,
        damping=0.1,
        position_noise=0.1,
        velocity_noise=0.2,
        safety_violation_penalty=-100.0,
        movement_reward_scale=1.0,
    )

    # Test equality
    assert env1 == env2

    # Test inequality with different parameters
    env3 = SafeAntVelocityPOMDP(
        discount_factor=0.9,  # Different discount factor
        safe_velocity_threshold=2.0,
        max_force=1.0,
        dt=0.1,
        mass=1.0,
        damping=0.1,
        position_noise=0.1,
        velocity_noise=0.2,
        safety_violation_penalty=-100.0,
        movement_reward_scale=1.0,
    )
    assert env1 != env3


def test_config_id():
    """Test configuration ID generation and consistency.

    Purpose: Validates that config_id generates consistent identifiers for identical environments and different identifiers for different configurations

    Given: SafeAntVelocityPOMDP environments with same and different parameters
    When: Config IDs are generated and compared
    Then: Identical environments have same config_id, different environments have different config_ids

    Test type: configuration
    """
    # Create two environments with same parameters
    env1 = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        safe_velocity_threshold=2.0,
        max_force=1.0,
        dt=0.1,
        mass=1.0,
        damping=0.1,
        position_noise=0.1,
        velocity_noise=0.2,
        safety_violation_penalty=-100.0,
        movement_reward_scale=1.0,
    )
    env2 = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        safe_velocity_threshold=2.0,
        max_force=1.0,
        dt=0.1,
        mass=1.0,
        damping=0.1,
        position_noise=0.1,
        velocity_noise=0.2,
        safety_violation_penalty=-100.0,
        movement_reward_scale=1.0,
    )

    # Test same config_id for identical environments
    assert env1.config_id == env2.config_id

    # Test different config_id for different environments
    env3 = SafeAntVelocityPOMDP(
        discount_factor=0.9,  # Different discount factor
        safe_velocity_threshold=2.0,
        max_force=1.0,
        dt=0.1,
        mass=1.0,
        damping=0.1,
        position_noise=0.1,
        velocity_noise=0.2,
        safety_violation_penalty=-100.0,
        movement_reward_scale=1.0,
    )
    assert env1.config_id != env3.config_id


def test_env_sample_next_state_shape(pomdp):
    """Test env.sample_next_state via env API.

    Purpose: Validates that env.sample_next_state produces correctly shaped states
    for different actions.

    Given: SafeAntVelocityPOMDP environment, state [0.0, 0.0, 0.0, 0.0], actions 0 and 1
    When: env.sample_next_state is called for each action
    Then: Returns ndarray of shape (4,) for both actions.

    Test type: unit
    """
    state = np.array([0.0, 0.0, 0.0, 0.0])

    next_state_a0 = pomdp.sample_next_state(state=state, action=0)
    assert isinstance(next_state_a0, np.ndarray)
    assert next_state_a0.shape == (4,)

    next_state_a1 = pomdp.sample_next_state(state=state, action=1)
    assert isinstance(next_state_a1, np.ndarray)
    assert next_state_a1.shape == (4,)


def test_env_sample_observation_shape(pomdp):
    """Test env.sample_observation produces correctly shaped observations.

    Purpose: Validates that env.sample_observation generates noisy observations of
    the right shape.

    Given: SafeAntVelocityPOMDP environment and next_state [0.0, 0.0, 0.0, 0.0]
    When: env.sample_observation is called
    Then: Returns ndarray observation of shape (4,).

    Test type: unit
    """
    next_state = np.array([0.0, 0.0, 0.0, 0.0])
    obs = pomdp.sample_observation(next_state=next_state, action=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)


def test_initial_state_distribution_with_pomdp(pomdp):
    """Test initial state distribution creation and sampling.

    Purpose: Validates that environment correctly creates initial state distribution

    Given: SafeAntVelocityPOMDP environment
    When: initial_state_dist is called and state is sampled
    Then: Returns valid initial state distribution that produces correctly shaped states

    Test type: unit
    """
    # Test initial state distribution
    dist = pomdp.initial_state_dist()
    state = dist.sample()[0]
    assert isinstance(state, np.ndarray)
    assert state.shape == (4,)


def test_observation_log_probability_invalid_observation_raises():
    """Test that env.observation_log_probability rejects malformed observations.

    Purpose: Validates that the env-level observation log-probability path
    surfaces the native marshalling layer's length validation.

    Given: A valid SafeAntVelocityPOMDP environment, a valid 4-D next_state, and
        an empty observation array.
    When: env.observation_log_probability is called with the empty observation.
    Then: A ValueError mentioning "length 4" is raised.

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        position_noise=0.1,
        velocity_noise=0.2,
    )
    next_state = np.array([0.5, -0.2, 1.0, 0.5])
    empty_observation = np.array([])

    with pytest.raises(ValueError) as exc_info:
        env.observation_log_probability(next_state, 1, [empty_observation])

    # The error comes from the native marshalling layer and mentions the expected length.
    assert "length 4" in str(exc_info.value)


def test_observation_log_probability_various_invalid_shapes():
    """Test env.observation_log_probability rejects all malformed observation shapes.

    Purpose: Validates that env.observation_log_probability surfaces shape
    validation errors for observations of the wrong length.

    Given: A valid SafeAntVelocityPOMDP environment, a 4-D next_state, and
        observations with incorrect shapes (lengths 0, 1, 2, 3, 5, 6).
    When: env.observation_log_probability is called with each malformed shape.
    Then: A ValueError is raised in every case.

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        position_noise=0.1,
        velocity_noise=0.2,
    )
    next_state = np.array([0.5, -0.2, 1.0, 0.5])

    invalid_observations = [
        np.array([]),  # shape (0,)
        np.array([1.0]),  # shape (1,)
        np.array([1.0, 2.0]),  # shape (2,)
        np.array([1.0, 2.0, 3.0]),  # shape (3,)
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # shape (5,)
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),  # shape (6,)
    ]

    for invalid_obs in invalid_observations:
        with pytest.raises(ValueError):
            env.observation_log_probability(next_state, 1, [invalid_obs])


def test_env_sample_observation_never_empty():
    """Test that env.sample_observation never produces empty observations.

    Purpose: Validates that env.sample_observation always produces correctly shaped
    finite observations across many draws.

    Given: A SafeAntVelocityPOMDP environment and a fixed next_state.
    When: env.sample_observation is called many times.
    Then: Every observation has shape (4,), is finite, and is usable for the
        env-level observation log-probability path.

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        position_noise=0.1,
        velocity_noise=0.2,
    )
    next_state = np.array([0.5, -0.2, 1.0, 0.5])
    action = 1

    for _ in range(20):
        for _ in range(5):
            obs = env.sample_observation(next_state=next_state, action=action)
            assert isinstance(obs, np.ndarray)
            assert obs.shape == (4,)
            assert np.all(np.isfinite(obs))
            log_probs = env.observation_log_probability(next_state, action, [obs])
            assert len(log_probs) == 1
            assert np.isfinite(log_probs[0])


def test_sample_next_step_observation_never_empty():
    """Test that sample_next_step never produces empty observations.

    Purpose: Validates that the high-level sample_next_step method produces valid observations

    Given: A valid SafeAntVelocityPOMDP environment and various states
    When: sample_next_step is called multiple times
    Then: All returned observations should be properly shaped and non-empty

    Test type: integration
    """
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        safe_velocity_threshold=2.0,
        position_noise=0.1,
        velocity_noise=0.2,
    )

    # Test with various initial states
    test_states = [
        np.array([0.0, 0.0, 0.0, 0.0]),  # Zero state
        np.array([-0.5, 0.5, 1.0, -1.0]),  # Mixed positive/negative
        np.array([0.8, -0.3, 1.5, 0.8]),  # Near safe threshold
        np.array([0.1, 0.1, 0.1, 0.1]),  # Small values
    ]

    actions = env.get_actions()

    for state in test_states:
        for action in actions:
            # Call sample_next_step multiple times to check consistency
            for _ in range(5):
                next_state, observation, _ = env.sample_next_step(state, action)

                # Check observation properties
                assert isinstance(observation, np.ndarray), "Observation should be numpy array"
                assert observation.shape == (
                    4,
                ), f"Expected observation shape (4,), got {observation.shape}"
                assert len(observation) > 0, "Observation should not be empty"
                assert np.all(np.isfinite(observation)), "All observation values should be finite"

                # Verify observation can be used in probability calculation
                log_probs = env.observation_log_probability(next_state, action, [observation])
                assert len(log_probs) == 1, "Should return one log-probability"
                log_prob = log_probs[0]
                assert np.isfinite(log_prob), "Log-probability should be finite"
                prob = float(np.exp(log_prob))
                assert prob >= 0.0, "Probability should be non-negative"


def test_env_observation_log_probability_list_interface():
    """Test that env.observation_log_probability handles a list of observations.

    Purpose: Validates that env.observation_log_probability works with lists of
    observations and returns arrays.

    Given: A SafeAntVelocityPOMDP environment, a fixed next_state, and a mix of
        nearby and far observations.
    When: env.observation_log_probability is called with the list.
    Then: Returns a finite ndarray, and observations near the true state have
        higher log-probability than those far away.

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        position_noise=0.1,
        velocity_noise=0.2,
    )
    next_state = np.array([0.5, -0.2, 1.0, 0.5])
    action = 1

    obs1 = np.array([0.51, -0.19, 1.02, 0.48])  # Close to true state
    log_probs = env.observation_log_probability(next_state, action, [obs1])
    assert isinstance(log_probs, np.ndarray)
    assert len(log_probs) == 1
    assert np.isfinite(log_probs[0])

    obs2 = np.array([0.49, -0.21, 0.98, 0.52])  # Also close to true state
    obs3 = np.array([1.5, 1.8, 2.0, 2.5])  # Far from true state

    log_probs_multi = env.observation_log_probability(next_state, action, [obs1, obs2, obs3])
    assert isinstance(log_probs_multi, np.ndarray)
    assert len(log_probs_multi) == 3
    assert np.all(np.isfinite(log_probs_multi))

    # Closer observations should have higher log-probability
    assert log_probs_multi[0] > log_probs_multi[2]
    assert log_probs_multi[1] > log_probs_multi[2]


def test_reward_batch_matches_scalar_reward():
    """Test that reward_batch returns results consistent with scalar reward.

    Purpose: Validates that the vectorized reward_batch gives identical outputs
    to calling reward() individually for each state.

    Given: A SafeAntVelocityPOMDP environment and an array of states with mixed safe/unsafe velocities
    When: reward_batch is called with the state array
    Then: Output shape is (N,) and values match element-wise reward() calls exactly

    Test type: unit
    """
    env = SafeAntVelocityPOMDP(
        discount_factor=0.95,
        safe_velocity_threshold=2.0,
        safety_violation_penalty=-100.0,
        movement_reward_scale=1.0,
    )
    np.random.seed(42)
    # 4-dim states: [pos_x, pos_y, vel_x, vel_y]
    states = np.random.randn(100, 4) * 2.0
    action = 1

    batch_rewards = env.reward_batch(states, action)

    assert batch_rewards.shape == (100,)
    expected = np.array([env.reward(states[i], action) for i in range(100)])
    np.testing.assert_allclose(batch_rewards, expected)

    # Also test with N=1
    single = env.reward_batch(states[:1], action)
    assert single.shape == (1,)
    np.testing.assert_allclose(single[0], env.reward(states[0], action))


# ---------------------------------------------------------------------------
# Native simulate_rollout equivalence and smoke tests
# ---------------------------------------------------------------------------


class _FixedActionSamplerSafeAnt:
    def __init__(self, action: int) -> None:
        self._action = action

    def sample(self) -> int:
        return self._action


def _safe_ant_python_rollout_native_semantics(
    env_local: SafeAntVelocityPOMDP,
    initial_state: np.ndarray,
    action_indices: np.ndarray,
    max_depth: int,
    start_depth: int,
    discount_factor: float,
) -> float:
    terminal_threshold = env_local.safe_velocity_threshold * 1.5
    state = np.array(initial_state, dtype=np.float64)
    total = 0.0
    gamma_power = 1.0
    depth = start_depth
    n_actions = len(env_local.actions)

    for action_int in action_indices:
        if depth >= max_depth:
            break
        vx, vy = state[2], state[3]
        if np.sqrt(vx * vx + vy * vy) > terminal_threshold:
            break
        ai = int(action_int) % n_actions
        next_state = env_local.sample_next_state(state=state, action=ai)
        nx, ny = next_state[2], next_state[3]
        next_speed = float(np.sqrt(nx * nx + ny * ny))
        step_reward = next_speed * env_local.movement_reward_scale
        if next_speed > env_local.safe_velocity_threshold:
            step_reward += env_local.safety_violation_penalty
        total += gamma_power * step_reward
        gamma_power *= discount_factor
        state = np.asarray(next_state, dtype=np.float64)
        depth += 1

    return total


def test_native_simulate_rollout_safe_ant_matches_python_reference():
    """Native simulate_rollout matches a Python reference with same reward semantics.

    Purpose: Validates that the C++ rollout accumulates the same discounted
    return as a Python loop using identical physics and reward conventions
    (reward computed from next state) under a fixed C++ RNG seed and
    identical pre-drawn action indices.

    Given: A SafeAntVelocityPOMDP, a fixed initial state, and identical
        action_indices and C++ RNG seed for both implementations.
    When: Both the native ``_native.simulate_rollout`` and the Python
        reference walk the same trajectory.
    Then: Returned discounted returns agree within atol=1e-9.

    Test type: integration
    """
    from POMDPPlanners.environments.safety_ant_velocity_pomdp import (
        _native as sa_native,
    )  # pylint: disable=import-outside-toplevel
    from POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp import (
        DEFAULT_FORCE_SCALES,
    )  # pylint: disable=import-outside-toplevel

    np.random.seed(0)
    env_local = SafeAntVelocityPOMDP(discount_factor=0.99)
    initial_state = np.array([0.2, -0.1, 0.5, 0.3], dtype=np.float64)
    max_depth = 8
    start_depth = 0
    discount_factor = 0.99
    steps_left = max_depth - start_depth

    action_indices = np.random.randint(0, 4, size=steps_left, dtype=np.int32)

    sa_native.set_seed(42)
    native_result = sa_native.simulate_rollout(
        initial_state=np.ascontiguousarray(initial_state),
        action_indices=action_indices,
        force_scales=np.ascontiguousarray(DEFAULT_FORCE_SCALES, dtype=np.float64),
        max_depth=max_depth,
        start_depth=start_depth,
        discount_factor=discount_factor,
        dt=float(env_local.dt),
        mass=float(env_local.mass),
        damping=float(env_local.damping),
        max_force=float(env_local.max_force),
        safe_velocity_threshold=float(env_local.safe_velocity_threshold),
        safety_violation_penalty=float(env_local.safety_violation_penalty),
        movement_reward_scale=float(env_local.movement_reward_scale),
    )

    sa_native.set_seed(42)
    python_result = _safe_ant_python_rollout_native_semantics(
        env_local=env_local,
        initial_state=initial_state,
        action_indices=action_indices,
        max_depth=max_depth,
        start_depth=start_depth,
        discount_factor=discount_factor,
    )

    np.testing.assert_allclose(native_result, python_result, atol=1e-9, rtol=0.0)


def test_simulate_random_rollout_safe_ant_returns_finite_float():
    """simulate_random_rollout returns a finite float from an initial state.

    Purpose: Smoke-test that the native override does not raise or produce
    non-finite values from a fresh initial state with shallow horizon.

    Given: A SafeAntVelocityPOMDP, its initial state, and a fixed action.
    When: simulate_random_rollout is called with max_depth=6.
    Then: The result is a finite float.

    Test type: integration
    """
    from POMDPPlanners.environments.safety_ant_velocity_pomdp import (
        _native as sa_native,
    )  # pylint: disable=import-outside-toplevel

    np.random.seed(0)
    sa_native.set_seed(0)
    env_local = SafeAntVelocityPOMDP(discount_factor=0.99)
    state = env_local.initial_state_dist().sample()[0]
    sampler = _FixedActionSamplerSafeAnt(action=1)

    result = env_local.simulate_random_rollout(
        state=state,
        action_sampler=sampler,
        max_depth=6,
        discount_factor=0.99,
    )

    assert isinstance(result, float)
    assert np.isfinite(result)


def test_simulate_random_rollout_safe_ant_returns_zero_at_max_depth():
    """simulate_random_rollout returns 0.0 when depth equals max_depth.

    Purpose: Validates the depth-bounded base case for the override.

    Given: A SafeAntVelocityPOMDP, any state, and depth == max_depth.
    When: simulate_random_rollout is called.
    Then: The return is exactly 0.0.

    Test type: unit
    """
    env_local = SafeAntVelocityPOMDP(discount_factor=0.99)
    state = env_local.initial_state_dist().sample()[0]
    sampler = _FixedActionSamplerSafeAnt(action=0)

    result = env_local.simulate_random_rollout(
        state=state,
        action_sampler=sampler,
        max_depth=5,
        discount_factor=0.99,
        depth=5,
    )

    assert result == 0.0


def test_simulate_random_rollout_safe_ant_terminal_returns_zero():
    """simulate_random_rollout returns 0.0 from a terminal state.

    Purpose: Validates that a state with critically-high speed terminates
    the rollout immediately and returns 0.0.

    Given: A state where speed exceeds 1.5 * safe_velocity_threshold.
    When: simulate_random_rollout is called with depth < max_depth.
    Then: The return is exactly 0.0.

    Test type: unit
    """
    env_local = SafeAntVelocityPOMDP(discount_factor=0.99)
    critical_speed = env_local.safe_velocity_threshold * 2.0
    terminal_state = np.array([0.0, 0.0, critical_speed, 0.0], dtype=np.float64)
    sampler = _FixedActionSamplerSafeAnt(action=0)

    result = env_local.simulate_random_rollout(
        state=terminal_state,
        action_sampler=sampler,
        max_depth=10,
        discount_factor=0.99,
        depth=0,
    )

    assert result == 0.0
