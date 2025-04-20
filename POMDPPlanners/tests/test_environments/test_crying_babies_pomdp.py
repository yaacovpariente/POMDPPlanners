import pytest
import numpy as np
from POMDPPlanners.environments.crying_babies_pomdp import CryingBabiesPOMDP

def test_crying_babies_initialization():
    pomdp = CryingBabiesPOMDP(discount_factor=0.95)
    assert pomdp.num_babies == 2
    assert len(pomdp.states) == 4  # All combinations of 2 babies being hungry/not hungry
    assert len(pomdp.actions) == 3
    assert 'feed_baby1' in pomdp.actions
    assert 'feed_baby2' in pomdp.actions
    assert 'do_nothing' in pomdp.actions
    assert len(pomdp.observations) == 4
    assert 'cry_baby1' in pomdp.observations
    assert 'cry_baby2' in pomdp.observations
    assert 'cry_both' in pomdp.observations
    assert 'cry_none' in pomdp.observations

def test_crying_babies_state_transition():
    pomdp = CryingBabiesPOMDP(discount_factor=0.95)
    
    # Test feeding baby1
    state = (True, True)
    transition = pomdp.state_transition_model(state, 'feed_baby1')
    new_state = transition.sample()
    assert new_state[0] == False  # Baby1 should be fed
    assert new_state[1] == True   # Baby2 should remain hungry
    
    # Test feeding baby2
    state = (True, True)
    transition = pomdp.state_transition_model(state, 'feed_baby2')
    new_state = transition.sample()
    assert new_state[0] == True   # Baby1 should remain hungry
    assert new_state[1] == False  # Baby2 should be fed
    
    # Test do_nothing
    state = (True, True)
    transition = pomdp.state_transition_model(state, 'do_nothing')
    new_state = transition.sample()
    # Both babies could remain hungry or get fed randomly
    assert isinstance(new_state[0], bool)
    assert isinstance(new_state[1], bool)

def test_crying_babies_observation():
    pomdp = CryingBabiesPOMDP(discount_factor=0.95)
    
    # Test observation when both babies are hungry
    state = (True, True)
    observation = pomdp.observation_model(state, 'do_nothing').sample()
    assert observation in ['cry_baby1', 'cry_baby2', 'cry_both', 'cry_none']
    
    # Test observation when only baby1 is hungry
    state = (True, False)
    observation = pomdp.observation_model(state, 'do_nothing').sample()
    assert observation in ['cry_baby1', 'cry_none']
    
    # Test observation when only baby2 is hungry
    state = (False, True)
    observation = pomdp.observation_model(state, 'do_nothing').sample()
    assert observation in ['cry_baby2', 'cry_none']
    
    # Test observation when no babies are hungry
    state = (False, False)
    observation = pomdp.observation_model(state, 'do_nothing').sample()
    assert observation == 'cry_none'

def test_crying_babies_reward():
    pomdp = CryingBabiesPOMDP(discount_factor=0.95)
    
    # Test reward when both babies are hungry and doing nothing
    state = (True, True)
    reward = pomdp.reward(state, 'do_nothing')
    assert reward == -10.0  # -5.0 for each hungry baby
    
    # Test reward when feeding baby1
    state = (True, True)
    reward = pomdp.reward(state, 'feed_baby1')
    assert reward == -6.0  # -5.0 for baby2, -1.0 for feeding
    
    # Test reward when no babies are hungry
    state = (False, False)
    reward = pomdp.reward(state, 'do_nothing')
    assert reward == 0.0
    
    # Test reward when feeding with no hungry babies
    state = (False, False)
    reward = pomdp.reward(state, 'feed_baby1')
    assert reward == -1.0  # Only the cost of feeding

def test_crying_babies_terminal():
    pomdp = CryingBabiesPOMDP(discount_factor=0.95)
    
    # Game should never be terminal
    assert not pomdp.is_terminal((True, True))
    assert not pomdp.is_terminal((False, False))
    assert not pomdp.is_terminal((True, False))
    assert not pomdp.is_terminal((False, True))

def test_crying_babies_initial_state():
    pomdp = CryingBabiesPOMDP(discount_factor=0.95)
    initial_state = pomdp.initial_state_dist().sample()
    
    assert isinstance(initial_state, tuple)
    assert len(initial_state) == 2
    assert initial_state == (False, False)  # Both babies start not hungry

def test_crying_babies_initial_observation():
    pomdp = CryingBabiesPOMDP(discount_factor=0.95)
    initial_observation = pomdp.initial_observation_dist().sample()
    
    assert initial_observation == 'cry_none'  # No babies crying initially

def test_crying_babies_actions():
    pomdp = CryingBabiesPOMDP(discount_factor=0.95)
    actions = pomdp.get_actions()
    
    assert len(actions) == 3
    assert 'feed_baby1' in actions
    assert 'feed_baby2' in actions
    assert 'do_nothing' in actions

def test_crying_babies_state_transition_probabilities():
    pomdp = CryingBabiesPOMDP(discount_factor=0.95)
    
    # Test deterministic feeding transitions
    state = (True, True)
    transition = pomdp.state_transition_model(state, 'feed_baby1')
    assert transition.probability((False, True)) == 1.0
    
    state = (True, True)
    transition = pomdp.state_transition_model(state, 'feed_baby2')
    assert transition.probability((True, False)) == 1.0

def test_crying_babies_observation_probabilities():
    pomdp = CryingBabiesPOMDP(discount_factor=0.95)
    
    # Test observation probabilities for different states
    state = (True, True)
    observation = pomdp.observation_model(state, 'do_nothing')
    # Since observations are probabilistic, we can't assert exact probabilities
    # but we can verify the observation is valid
    obs = observation.sample()
    assert obs in pomdp.observations
    
    state = (False, False)
    observation = pomdp.observation_model(state, 'do_nothing')
    assert observation.sample() == 'cry_none'
