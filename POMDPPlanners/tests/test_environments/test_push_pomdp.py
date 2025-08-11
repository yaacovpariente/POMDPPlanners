import numpy as np
import pytest
from pathlib import Path

from POMDPPlanners.environments.push_pomdp import (
    PushPOMDP,
    PushStateTransition,
    PushObservation,
)

def test_push_pomdp_initialization():
    """Test push pomdp initialization.
    
    Purpose: Validates proper initialization of push pomdp 
    
    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes
    
    Test type: unit
    """
    # Test basic initialization
    env = PushPOMDP(
        discount_factor=0.95,
        grid_size=10,
        push_threshold=1.0,
        friction_coefficient=0.3,
        observation_noise=0.1,
    )
    
    assert env.discount_factor == 0.95
    assert env.grid_size == 10
    assert env.push_threshold == 1.0
    assert env.friction_coefficient == 0.3
    assert env.observation_noise == 0.1
    assert env.actions == ["up", "down", "right", "left"]
    assert np.array_equal(env.target_pos, np.array([9, 9]))  # grid_size - 1

def test_state_transition():
    """Test state transition.
    
    Purpose: Validates that PushStateTransition correctly moves robot and pushes object when within threshold distance
    
    Given: Robot at position [5,5] next to object at [6,5], action="right", push_threshold=2.0, grid_size=10
    When: PushStateTransition samples next state after robot moves right
    Then: Robot x-position increases, object x-position increases due to push, target unchanged, all positions within bounds
    
    Test type: unit
    """
    # Test state transition with known parameters
    state = np.array([5.0, 5.0, 6.0, 5.0, 9.0, 9.0])  # Robot to the left of object
    action = "right"
    grid_size = 10
    push_threshold = 2.0  # Increased threshold to ensure pushing
    friction_coefficient = 0.3

    transition = PushStateTransition(
        state=state,
        action=action,
        grid_size=grid_size,
        push_threshold=push_threshold,
        friction_coefficient=friction_coefficient,
    )

    next_state = transition.sample()[0]  # Get first element from list

    # Verify state dimensions
    assert next_state.shape == (6,)
    
    # Verify robot moved right
    assert next_state[0] > state[0]  # robot_x increased
    
    # Verify object was pushed (since robot was close enough)
    assert next_state[2] > state[2]  # object_x increased
    
    # Verify target position unchanged
    assert np.array_equal(next_state[4:], state[4:])
    
    # Verify positions within bounds
    assert np.all(next_state >= 0)
    assert np.all(next_state < grid_size)

def test_state_transition_no_push():
    """Test state transition no push.
    
    Purpose: Validates that PushStateTransition moves robot without affecting object when distance exceeds push threshold
    
    Given: Robot at position [1,1] far from object at [8,8], action="right", push_threshold=1.0
    When: PushStateTransition samples next state for distant robot-object pair
    Then: Robot x-position increases but object position remains unchanged due to distance > push_threshold
    
    Test type: unit
    """
    # Test state transition when robot is too far to push
    state = np.array([1.0, 1.0, 8.0, 8.0, 9.0, 9.0])  # Robot far from object
    action = "right"
    grid_size = 10
    push_threshold = 1.0
    friction_coefficient = 0.3

    transition = PushStateTransition(
        state=state,
        action=action,
        grid_size=grid_size,
        push_threshold=push_threshold,
        friction_coefficient=friction_coefficient,
    )

    next_state = transition.sample()[0]  # Get first element from list

    # Verify robot moved
    assert next_state[0] > state[0]
    
    # Verify object didn't move (too far from robot)
    assert np.array_equal(next_state[2:4], state[2:4])

def test_observation_model():
    """Test observation model.
    
    Purpose: Validates that PushObservation adds noise to object position while keeping robot and target positions exact
    
    Given: State with robot [5,5], object [4,5], target [9,9], observation_noise=0.1
    When: PushObservation samples observation with noise applied
    Then: Robot position exact, object position has noise (not equal to true position), target position exact, correct dimensions
    
    Test type: unit
    """
    # Test observation model
    state = np.array([5.0, 5.0, 4.0, 5.0, 9.0, 9.0])
    action = "right"
    observation_noise = 0.1
    grid_size = 10

    observation_model = PushObservation(
        next_state=state,
        action=action,
        observation_noise=observation_noise,
        grid_size=grid_size,
    )

    observation = observation_model.sample()[0]  # Get first element from list

    # Verify observation dimensions
    assert observation.shape == (6,)
    
    # Verify robot position is exact
    assert np.array_equal(observation[:2], state[:2])
    
    # Verify object position has noise
    assert not np.array_equal(observation[2:4], state[2:4])
    
    # Verify target position is exact
    assert np.array_equal(observation[4:], state[4:])

def test_reward_function():
    """Test reward function.
    
    Purpose: Validates that PushPOMDP reward function returns higher rewards for objects closer to target position
    
    Given: Three states with object at different distances from target [9,9]: far [5,5], near [8.5,8.5], at target [9,9]
    When: Reward is calculated for each state with action="right"
    Then: Negative reward for far object, higher reward for near object, positive reward for object at target
    
    Test type: unit
    """
    env = PushPOMDP(discount_factor=0.95)
    
    # Test reward for object far from target
    state_far = np.array([5.0, 5.0, 5.0, 5.0, 9.0, 9.0])
    reward_far = env.reward(state_far, "right")
    assert reward_far < 0  # Negative reward for distance
    
    # Test reward for object near target
    state_near = np.array([8.0, 8.0, 8.5, 8.5, 9.0, 9.0])
    reward_near = env.reward(state_near, "right")
    assert reward_near > reward_far  # Higher reward for being closer
    
    # Test reward for object at target
    state_at_target = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0])
    reward_at_target = env.reward(state_at_target, "right")
    assert reward_at_target > 0  # Positive reward for reaching target

def test_terminal_state():
    """Test terminal state.
    
    Purpose: Validates that PushPOMDP correctly identifies terminal states when object reaches target vicinity
    
    Given: Two states: object far from target [5,5] vs [9,9] and object near target [8.8,8.8] vs [9,9]
    When: is_terminal method evaluates proximity to target position
    Then: Far state returns False (non-terminal), near state returns True (terminal) based on distance threshold
    
    Test type: unit
    """
    env = PushPOMDP(discount_factor=0.95)
    
    # Test non-terminal state
    state_far = np.array([5.0, 5.0, 5.0, 5.0, 9.0, 9.0])
    assert not env.is_terminal(state_far)
    
    # Test terminal state (object near target)
    state_near = np.array([9.0, 9.0, 8.8, 8.8, 9.0, 9.0])  # Closer to target
    assert env.is_terminal(state_near)

def test_initial_state_distribution():
    """Test initial state distribution.
    
    Purpose: Validates that PushPOMDP initial state distribution generates valid states with minimum distance constraint
    
    Given: PushPOMDP environment with grid_size=10, minimum distance=2.0 from target
    When: initial_state_dist samples multiple initial states
    Then: All states have 6D shape, positions within [0,grid_size), object-target distance >= 2.0
    
    Test type: unit
    """
    env = PushPOMDP(discount_factor=0.95, grid_size=10)
    initial_dist = env.initial_state_dist()
    
    # Test multiple samples
    for _ in range(10):
        state = initial_dist.sample()[0]  # Get first element from list
        
        # Verify state dimensions
        assert state.shape == (6,)
        
        # Verify positions within bounds
        assert np.all(state >= 0)
        assert np.all(state < env.grid_size)
        
        # Verify minimum distance from target
        object_pos = state[2:4]
        target_pos = state[4:6]
        distance = np.linalg.norm(object_pos - target_pos)
        assert distance >= 2.0  # Minimum distance constraint

def test_get_actions():
    """Test get actions.
    
    Purpose: Validates that PushPOMDP provides correct set of directional movement actions
    
    Given: PushPOMDP environment with default configuration
    When: get_actions method is called to retrieve available actions
    Then: Returns 4 directional actions: ["up", "down", "right", "left"]
    
    Test type: unit
    """
    env = PushPOMDP(discount_factor=0.95)
    actions = env.get_actions()
    
    assert len(actions) == 4
    assert "up" in actions
    assert "down" in actions
    assert "right" in actions
    assert "left" in actions

def test_is_equal_observation():
    """Test is equal observation.
    
    Purpose: Validates that PushPOMDP correctly compares observation arrays for equality
    
    Given: Two identical observation arrays [1,1,2,2,9,9] and one different array [1,1,2.1,2,9,9]
    When: is_equal_observation compares observation arrays element-wise
    Then: Identical arrays return True, different arrays return False
    
    Test type: unit
    """
    env = PushPOMDP(discount_factor=0.95)
    
    # Test equal observations
    obs1 = np.array([1.0, 1.0, 2.0, 2.0, 9.0, 9.0])
    obs2 = np.array([1.0, 1.0, 2.0, 2.0, 9.0, 9.0])
    assert env.is_equal_observation(obs1, obs2)
    
    # Test different observations
    obs3 = np.array([1.0, 1.0, 2.1, 2.0, 9.0, 9.0])
    assert not env.is_equal_observation(obs1, obs3)

def test_sample_next_step():
    """Test sample next step.
    
    Purpose: Validates sampling behavior for  next step
    
    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution
    
    Test type: unit
    """
    env = PushPOMDP(discount_factor=0.95)
    state = np.array([5.0, 5.0, 4.0, 5.0, 9.0, 9.0])
    action = "right"
    
    next_state, observation, reward = env.sample_next_step(state, action)
    
    # Verify return types and shapes
    assert isinstance(next_state, np.ndarray)
    assert isinstance(observation, np.ndarray)
    assert isinstance(reward, float)
    assert next_state.shape == (6,)
    assert observation.shape == (6,)
    
    # Verify state transition
    assert next_state[0] > state[0]  # Robot moved right
    
    # Verify observation has noise on object position
    assert not np.array_equal(observation[2:4], next_state[2:4])
    
    # Verify reward is calculated
    assert reward == env.reward(next_state, action)

def test_environment_equality():
    """Test environment equality.
    
    Purpose: Validates equality comparison for environment 
    
    Given: Objects with same or different configurations
    When: Equality comparison is performed
    Then: Objects are correctly identified as equal or unequal
    
    Test type: unit
    """
    # Create two identical environments
    env1 = PushPOMDP(
        discount_factor=0.95,
        grid_size=10,
        push_threshold=1.0,
        friction_coefficient=0.3,
        observation_noise=0.1,
    )
    env2 = PushPOMDP(
        discount_factor=0.95,
        grid_size=10,
        push_threshold=1.0,
        friction_coefficient=0.3,
        observation_noise=0.1,
    )
    
    # Test equality
    assert env1 == env2
    
    # Test inequality with different parameters
    env3 = PushPOMDP(
        discount_factor=0.9,  # Different discount factor
        grid_size=10,
        push_threshold=1.0,
        friction_coefficient=0.3,
        observation_noise=0.1,
    )
    assert env1 != env3

def test_config_id():
    """Test config id.
    
    Purpose: Validates config_id behavior for 
    
    Given: Belief objects with specific configurations
    When: Config IDs are generated or compared
    Then: Config IDs behave as expected (deterministic, unique, etc.)
    
    Test type: configuration
    """
    # Create two environments with same parameters
    env1 = PushPOMDP(
        discount_factor=0.95,
        grid_size=10,
        push_threshold=1.0,
        friction_coefficient=0.3,
        observation_noise=0.1,
    )
    env2 = PushPOMDP(
        discount_factor=0.95,
        grid_size=10,
        push_threshold=1.0,
        friction_coefficient=0.3,
        observation_noise=0.1,
    )
    
    # Test same config_id for identical environments
    assert env1.config_id == env2.config_id
    
    # Test different config_id for different environments
    env3 = PushPOMDP(
        discount_factor=0.9,  # Different discount factor
        grid_size=10,
        push_threshold=1.0,
        friction_coefficient=0.3,
        observation_noise=0.1,
    )
    assert env1.config_id != env3.config_id 