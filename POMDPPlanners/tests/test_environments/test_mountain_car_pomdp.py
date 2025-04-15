import pytest
import numpy as np
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.core.simulation import StepData

def test_mountain_car_initialization():
    pomdp = MountainCarPOMDP()
    assert pomdp.discount_factor == 0.95
    assert pomdp.min_position == -1.2
    assert pomdp.max_position == 0.6
    assert pomdp.max_speed == 0.07
    assert pomdp.goal_position == 0.5
    assert pomdp.power == 0.001
    assert pomdp.gravity == 0.0025
    assert len(pomdp.actions) == 3
    assert -1 in pomdp.actions
    assert 0 in pomdp.actions
    assert 1 in pomdp.actions

def test_mountain_car_state_transition():
    pomdp = MountainCarPOMDP()
    
    # Test left acceleration
    state = (0.0, 0.0)
    transition = pomdp.state_transition_model(state, -1)
    new_state = transition.sample()
    assert isinstance(new_state, tuple)
    assert len(new_state) == 2
    assert isinstance(new_state[0], float)
    assert isinstance(new_state[1], float)
    assert new_state[0] <= pomdp.max_position
    assert new_state[0] >= pomdp.min_position
    assert abs(new_state[1]) <= pomdp.max_speed
    
    # Test right acceleration
    state = (0.0, 0.0)
    transition = pomdp.state_transition_model(state, 1)
    new_state = transition.sample()
    assert new_state[0] <= pomdp.max_position
    assert new_state[0] >= pomdp.min_position
    assert abs(new_state[1]) <= pomdp.max_speed
    
    # Test no acceleration
    state = (0.0, 0.0)
    transition = pomdp.state_transition_model(state, 0)
    new_state = transition.sample()
    assert new_state[0] <= pomdp.max_position
    assert new_state[0] >= pomdp.min_position
    assert abs(new_state[1]) <= pomdp.max_speed

def test_mountain_car_observation():
    pomdp = MountainCarPOMDP()
    
    # Test observation with known state
    state = (0.0, 0.0)
    observation = pomdp.observation_model(state, 0).sample()
    assert isinstance(observation, tuple)
    assert len(observation) == 2
    assert isinstance(observation[0], float)
    assert isinstance(observation[1], float)
    
    # Test observation noise
    state = (0.0, 0.0)
    observations = [pomdp.observation_model(state, 0).sample() for _ in range(100)]
    positions = [obs[0] for obs in observations]
    velocities = [obs[1] for obs in observations]
    
    # Check that observations are noisy
    assert np.std(positions) > 0
    assert np.std(velocities) > 0

def test_mountain_car_reward():
    pomdp = MountainCarPOMDP()
    
    # Test reward when not at goal
    state = (0.0, 0.0)
    reward = pomdp.reward(state, 0)
    assert reward == -1.0
    
    # Test reward when at goal
    state = (pomdp.goal_position, 0.0)
    reward = pomdp.reward(state, 0)
    assert reward == 0.0
    
    # Test reward when past goal
    state = (pomdp.goal_position + 0.1, 0.0)
    reward = pomdp.reward(state, 0)
    assert reward == 0.0

def test_mountain_car_terminal():
    pomdp = MountainCarPOMDP()
    
    # Test non-terminal state
    state = (0.0, 0.0)
    assert not pomdp.is_terminal(state)
    
    # Test terminal state
    state = (pomdp.goal_position, 0.0)
    assert pomdp.is_terminal(state)
    
    # Test state past goal
    state = (pomdp.goal_position + 0.1, 0.0)
    assert pomdp.is_terminal(state)

def test_mountain_car_initial_state():
    pomdp = MountainCarPOMDP()
    initial_state = pomdp.initial_state_dist().sample()
    
    assert isinstance(initial_state, tuple)
    assert len(initial_state) == 2
    assert isinstance(initial_state[0], float)
    assert isinstance(initial_state[1], float)
    assert initial_state[0] >= -0.6
    assert initial_state[0] <= -0.4
    assert initial_state[1] == 0.0

def test_mountain_car_initial_observation():
    pomdp = MountainCarPOMDP()
    initial_observation = pomdp.initial_observation_dist().sample()
    
    assert isinstance(initial_observation, tuple)
    assert len(initial_observation) == 2
    assert initial_observation[0] == 0.0
    assert initial_observation[1] == 0.0

def test_mountain_car_actions():
    pomdp = MountainCarPOMDP()
    actions = pomdp.get_actions()
    
    assert len(actions) == 3
    assert -1 in actions
    assert 0 in actions
    assert 1 in actions

def test_mountain_car_state_bounds():
    pomdp = MountainCarPOMDP()
    
    # Test position bounds
    state = (pomdp.min_position - 0.1, 0.0)
    transition = pomdp.state_transition_model(state, 0).sample()
    assert transition[0] >= pomdp.min_position
    
    state = (pomdp.max_position + 0.1, 0.0)
    transition = pomdp.state_transition_model(state, 0).sample()
    assert transition[0] <= pomdp.max_position
    
    # Test velocity bounds
    state = (0.0, pomdp.max_speed + 0.1)
    transition = pomdp.state_transition_model(state, 0).sample()
    assert abs(transition[1]) <= pomdp.max_speed
    
    state = (0.0, -pomdp.max_speed - 0.1)
    transition = pomdp.state_transition_model(state, 0).sample()
    assert abs(transition[1]) <= pomdp.max_speed 