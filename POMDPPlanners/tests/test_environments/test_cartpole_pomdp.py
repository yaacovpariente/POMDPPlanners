import numpy as np
import pytest
from POMDPPlanners.environments.cartpole_pomdp import (
    CartPolePOMDP,
    CartPoleStateTransition,
    CartPoleObservation,
    CartPoleInitialStateDistribution
)

def test_cartpole_state_transition():
    # Test state transition with known parameters
    state = np.array([0.0, 0.0, 0.0, 0.0])  # x, x_dot, theta, theta_dot
    action = np.array([1])  # push right
    force_mag = 10.0
    total_mass = 1.1
    polemass_length = 0.05
    gravity = 9.8
    length = 0.5
    tau = 0.02
    masspole = 0.1
    
    transition = CartPoleStateTransition(
        state=state,
        action=action,
        force_mag=force_mag,
        total_mass=total_mass,
        polemass_length=polemass_length,
        gravity=gravity,
        length=length,
        kinematics_integrator="euler",
        tau=tau,
        masspole=masspole
    )
    
    next_state = transition.sample()
    
    # Verify state dimensions
    assert next_state.shape == (4,)
    # Verify state bounds are reasonable
    assert np.all(np.isfinite(next_state))
    # Verify reasonable initial movement
    assert np.abs(next_state[0]) < 0.1  # x position should change little in one step
    assert np.abs(next_state[1]) < 0.5  # x_dot should be reasonable for the force applied
    assert np.abs(next_state[2]) < 0.1  # theta should change little in one step
    assert np.abs(next_state[3]) < 0.5  # theta_dot should be reasonable

def test_cartpole_observation():
    # Test observation model with known parameters
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = np.array([0])
    noise_cov = np.eye(4) * 0.1  # Small noise
    
    observation = CartPoleObservation(
        next_state=state,
        action=action,
        noise_cov=noise_cov
    )
    
    obs = observation.sample()
    
    # Verify observation dimensions
    assert obs.shape == (4,)
    # Verify observation is close to state with noise
    assert np.allclose(obs, state, atol=1.0)
    # Verify noise is applied
    assert not np.array_equal(obs, state)

def test_cartpole_initial_state_distribution():
    # Test initial state distribution
    dist = CartPoleInitialStateDistribution()
    state = dist.sample()
    
    # Verify state dimensions
    assert state.shape == (4,)
    # Verify state is within expected bounds
    assert np.all(state >= -0.05)
    assert np.all(state <= 0.05)

def test_cartpole_pomdp_initialization():
    # Test POMDP initialization
    noise_cov = np.eye(4) * 0.1
    
    env = CartPolePOMDP(
        discount_factor=0.95,
        noise_cov=noise_cov
    )
    
    # Verify parameters
    assert np.array_equal(env.noise_cov, noise_cov)
    assert env.gravity == 9.8
    assert env.masscart == 1.0
    assert env.masspole == 0.1
    assert env.length == 0.5

def test_cartpole_pomdp_reward():
    # Test reward function
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.1)
    
    # Test non-terminal state
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = np.array([0])
    reward = env.reward(state, action)
    assert reward == 1.0
    
    # Test terminal state (pole angle too large)
    state = np.array([0.0, 0.0, 0.3, 0.0])  # theta > theta_threshold
    reward = env.reward(state, action)
    assert reward == 0.0

def test_cartpole_pomdp_terminal():
    # Test terminal state detection
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.1)
    
    # Test non-terminal state
    state = np.array([0.0, 0.0, 0.0, 0.0])
    assert not env.is_terminal(state)
    
    # Test terminal state (cart position too far)
    state = np.array([2.5, 0.0, 0.0, 0.0])  # x > x_threshold
    assert env.is_terminal(state)
    
    # Test terminal state (pole angle too large)
    state = np.array([0.0, 0.0, 0.3, 0.0])  # theta > theta_threshold
    assert env.is_terminal(state)

def test_cartpole_pomdp_models():
    # Test model creation
    env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.1)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = np.array([0])
    
    # Test state transition model
    transition_model = env.state_transition_model(state, action)
    assert isinstance(transition_model, CartPoleStateTransition)
    
    # Test observation model
    observation_model = env.observation_model(state, action)
    assert isinstance(observation_model, CartPoleObservation)
    
    # Test initial state distribution
    initial_dist = env.initial_state_dist()
    assert isinstance(initial_dist, CartPoleInitialStateDistribution)
    
    # Test initial observation distribution
    initial_obs_dist = env.initial_observation_dist()
    assert isinstance(initial_obs_dist, CartPoleInitialStateDistribution)
