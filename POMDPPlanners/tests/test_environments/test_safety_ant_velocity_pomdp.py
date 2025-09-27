"""Tests for Safety Ant Velocity POMDP environment.

This module tests the Safety Ant Velocity POMDP environment, focusing on:
- Basic environment functionality
- State transitions and observations
- Reward calculations
- Terminal conditions
"""

import random
from unittest.mock import Mock

import numpy as np
import pytest

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.policy import PolicyInfoVariable, PolicyRunData
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.safety_ant_velocity_pomdp import (
    SafeAntVelocityObservation,
    SafeAntVelocityPOMDP,
    SafeAntVelocityStateTransition,
)


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


def test_state_transition():
    """Test state transition model with force application.

    Purpose: Validates that state transitions correctly update position and velocity when force is applied

    Given: A SafeAntVelocityStateTransition model with state [0.0, 0.0, 1.0, 1.0] and maximum force action 3
    When: State transition is sampled
    Then: Next state has correct 4D shape, position changes due to velocity, and velocity changes due to applied force

    Test type: unit
    """
    # Test state transition with known parameters
    state = np.array([0.0, 0.0, 1.0, 1.0])  # [pos_x, pos_y, vel_x, vel_y]
    action = 3  # Maximum force
    dt = 0.1
    mass = 1.0
    damping = 0.1
    max_force = 1.0

    transition = SafeAntVelocityStateTransition(
        state=state,
        action=action,
        dt=dt,
        mass=mass,
        damping=damping,
        max_force=max_force,
    )

    next_state = transition.sample()[0]

    # Verify state dimensions
    assert next_state.shape == (4,)

    # Verify position changed due to velocity
    assert not np.array_equal(next_state[:2], state[:2])

    # Verify velocity changed due to force
    assert not np.array_equal(next_state[2:], state[2:])


def test_state_transition_no_force():
    """Test state transition model without force application.

    Purpose: Validates that state transitions correctly handle damping when no force is applied

    Given: A SafeAntVelocityStateTransition model with state [0.0, 0.0, 1.0, 1.0] and no-force action 0
    When: State transition is sampled
    Then: Velocity decreases due to damping effects while maintaining correct state dimensions

    Test type: unit
    """
    # Test state transition with no force
    state = np.array([0.0, 0.0, 1.0, 1.0])
    action = 0  # No force
    dt = 0.1
    mass = 1.0
    damping = 0.1
    max_force = 1.0

    transition = SafeAntVelocityStateTransition(
        state=state,
        action=action,
        dt=dt,
        mass=mass,
        damping=damping,
        max_force=max_force,
    )

    next_state = transition.sample()

    # Verify velocity decreased due to damping
    assert np.linalg.norm(next_state[2:]) < np.linalg.norm(state[2:])


def test_observation_model():
    """Test observation model with noise addition.

    Purpose: Validates that observation model correctly adds noise to position and velocity measurements

    Given: A SafeAntVelocityObservation model with state [0.0, 0.0, 1.0, 1.0] and noise parameters
    When: Observation is sampled
    Then: Observation has correct 4D shape and contains noise in both position and velocity components

    Test type: unit
    """
    # Test observation model
    state = np.array([0.0, 0.0, 1.0, 1.0])
    action = 1
    position_noise = 0.1
    velocity_noise = 0.2

    observation_model = SafeAntVelocityObservation(
        next_state=state,
        action=action,
        position_noise=position_noise,
        velocity_noise=velocity_noise,
    )

    observation = observation_model.sample()[0]

    # Verify observation dimensions
    assert observation.shape == (4,)

    # Verify position has noise
    assert not np.array_equal(observation[:2], state[:2])

    # Verify velocity has noise
    assert not np.array_equal(observation[2:], state[2:])


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
            policy_run_data=PolicyRunData(info_variables=[]),
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
            policy_run_data=PolicyRunData(info_variables=[]),
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
            policy_run_data=PolicyRunData(info_variables=[]),
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


def test_state_transition_model(pomdp):
    """Test state transition model creation and sampling.

    Purpose: Validates that environment correctly creates state transition models for different actions

    Given: SafeAntVelocityPOMDP environment and test states with actions 0 and 1
    When: state_transition_model is called and next states are sampled
    Then: Returns valid state transition models that produce correctly shaped next states

    Test type: unit
    """
    # Test state transition
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = 0
    transition = pomdp.state_transition_model(state, action)
    next_state = transition.sample()[0]
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (4,)

    # Test with different action
    action = 1
    transition = pomdp.state_transition_model(state, action)
    next_state = transition.sample()[0]
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (4,)


def test_observation_model_with_pomdp(pomdp):
    """Test observation model creation and sampling.

    Purpose: Validates that environment correctly creates observation models that generate noisy observations

    Given: SafeAntVelocityPOMDP environment and test state [0.0, 0.0, 0.0, 0.0] with action 0
    When: observation_model is called and observation is sampled
    Then: Returns valid observation model that produces correctly shaped observations

    Test type: unit
    """
    # Test observation model
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = 0
    observation = pomdp.observation_model(state, action)
    obs = observation.sample()[0]
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


def test_observation_model_empty_observation_error():
    """Test that observation model properly handles invalid empty observations.

    Purpose: Validates that the observation probability method handles invalid inputs gracefully

    Given: A valid SafeAntVelocityObservation model and an empty observation array
    When: The probability method is called with an empty observation
    Then: A descriptive error should be raised explaining the invalid observation format

    Test type: unit
    """
    # Create observation model
    state = np.array([0.5, -0.2, 1.0, 0.5])
    obs_model = SafeAntVelocityObservation(
        next_state=state,
        action=1,
        position_noise=0.1,
        velocity_noise=0.2,
    )

    # Test with empty observation (this is the error case we fixed)
    empty_observation = np.array([])

    with pytest.raises(ValueError) as exc_info:
        obs_model.probability([empty_observation])  # Now expects list

    # The error should mention expected array format
    assert "Expected non-empty numpy array observation" in str(exc_info.value)


def test_observation_model_invalid_observation_shapes():
    """Test observation model with various invalid observation shapes.

    Purpose: Validates that observation model handles all invalid observation shapes properly

    Given: A valid SafeAntVelocityObservation model and observations with incorrect shapes
    When: The probability method is called with malformed observations
    Then: Appropriate errors should be raised for each invalid shape

    Test type: unit
    """
    # Create observation model
    state = np.array([0.5, -0.2, 1.0, 0.5])
    obs_model = SafeAntVelocityObservation(
        next_state=state,
        action=1,
        position_noise=0.1,
        velocity_noise=0.2,
    )

    # Test with various invalid observation shapes
    invalid_observations = [
        np.array([]),  # Empty array (shape (0,))
        np.array([1.0]),  # Too short (shape (1,))
        np.array([1.0, 2.0]),  # Too short (shape (2,))
        np.array([1.0, 2.0, 3.0]),  # Too short (shape (3,))
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # Too long (shape (5,))
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),  # Too long (shape (6,))
    ]

    for i, invalid_obs in enumerate(invalid_observations):
        with pytest.raises(ValueError) as exc_info:
            obs_model.probability([invalid_obs])  # Now expects list

        print(
            f"Invalid observation {i} shape {invalid_obs.shape} correctly raised: {type(exc_info.value).__name__}"
        )


def test_observation_never_empty_from_sample():
    """Test that observation model never produces empty observations from sample method.

    Purpose: Validates that the sample method always produces correctly shaped observations

    Given: A valid SafeAntVelocityObservation model
    When: Multiple observations are sampled
    Then: All observations should have shape (4,) and valid content

    Test type: unit
    """
    # Create observation model
    state = np.array([0.5, -0.2, 1.0, 0.5])
    obs_model = SafeAntVelocityObservation(
        next_state=state,
        action=1,
        position_noise=0.1,
        velocity_noise=0.2,
    )

    # Sample many observations to check consistency
    for _ in range(20):
        observations = obs_model.sample(n_samples=5)

        assert len(observations) == 5, "Should return requested number of observations"

        for obs in observations:
            assert isinstance(obs, np.ndarray), "Each observation should be numpy array"
            assert obs.shape == (4,), f"Expected shape (4,), got {obs.shape}"
            assert len(obs) > 0, "Observation should not be empty"
            assert np.all(np.isfinite(obs)), "All observation values should be finite"

            # Test that this observation can be used in probability calculation
            try:
                probs = obs_model.probability([obs])  # Now expects list
                assert len(probs) == 1, "Should return one probability"
                prob = probs[0]
                assert np.isfinite(prob), "Probability should be finite"
                assert prob >= 0.0, "Probability should be non-negative"
            except Exception as e:
                pytest.fail(
                    f"Valid observation {obs} with shape {obs.shape} failed probability calculation: {e}"
                )


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
                next_state, observation, reward = env.sample_next_step(state, action)

                # Check observation properties
                assert isinstance(observation, np.ndarray), "Observation should be numpy array"
                assert observation.shape == (
                    4,
                ), f"Expected observation shape (4,), got {observation.shape}"
                assert len(observation) > 0, "Observation should not be empty"
                assert np.all(np.isfinite(observation)), "All observation values should be finite"

                # Verify observation can be used in probability calculation
                obs_model = env.observation_model(next_state, action)
                try:
                    probs = obs_model.probability([observation])  # Now expects list
                    assert len(probs) == 1, "Should return one probability"
                    prob = probs[0]
                    assert np.isfinite(prob), "Probability should be finite"
                    assert prob >= 0.0, "Probability should be non-negative"
                except Exception as e:
                    pytest.fail(
                        f"sample_next_step produced invalid observation {observation} with shape {observation.shape}: {e}"
                    )


def test_observation_probability_list_interface():
    """Test that observation model probability method correctly handles list interface.

    Purpose: Validates that probability method works with lists of observations and returns arrays

    Given: A valid SafeAntVelocityObservation model and multiple valid observations
    When: The probability method is called with list of observations
    Then: Returns array of probabilities with correct length and finite values

    Test type: unit
    """
    # Create observation model
    state = np.array([0.5, -0.2, 1.0, 0.5])
    obs_model = SafeAntVelocityObservation(
        next_state=state,
        action=1,
        position_noise=0.1,
        velocity_noise=0.2,
    )

    # Test single observation
    obs1 = np.array([0.51, -0.19, 1.02, 0.48])  # Close to true state
    probs = obs_model.probability([obs1])

    assert isinstance(probs, np.ndarray), "Should return numpy array"
    assert len(probs) == 1, "Should return one probability for one observation"
    assert np.isfinite(probs[0]), "Probability should be finite"
    assert probs[0] >= 0.0, "Probability should be non-negative"

    # Test multiple observations
    obs2 = np.array([0.49, -0.21, 0.98, 0.52])  # Also close to true state
    obs3 = np.array([1.5, 1.8, 2.0, 2.5])  # Far from true state

    probs_multi = obs_model.probability([obs1, obs2, obs3])

    assert isinstance(probs_multi, np.ndarray), "Should return numpy array"
    assert len(probs_multi) == 3, "Should return three probabilities for three observations"
    assert np.all(np.isfinite(probs_multi)), "All probabilities should be finite"
    assert np.all(probs_multi >= 0.0), "All probabilities should be non-negative"

    # Closer observations should have higher probability
    assert probs_multi[0] > probs_multi[2], "Closer observation should have higher probability"
    assert probs_multi[1] > probs_multi[2], "Closer observation should have higher probability"
