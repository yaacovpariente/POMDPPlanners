import numpy as np
import pytest
from POMDPPlanners.environments.cartpole_pomdp import (
    CartPolePOMDP,
    CartPoleStateTransition,
    CartPoleObservation,
    CartPoleInitialStateDistribution,
)

np.random.seed(42)

@pytest.fixture
def base_cartpole_environment() -> CartPolePOMDP:
    """Fixture providing a base CartPolePOMDP environment for comparison."""
    noise_cov = np.eye(4) * 0.1
    return CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)


class TestCartPolePOMDPEquality:
    """Test suite for CartPolePOMDP equality comparisons."""
    
    def test_same_discount_factor(self, base_cartpole_environment: CartPolePOMDP):
        """Test that CartPolePOMDPs with same discount factor are equal."""
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        assert base_cartpole_environment == other_env
        assert other_env == base_cartpole_environment  # Test symmetry
    
    def test_different_discount_factor(self, base_cartpole_environment: CartPolePOMDP):
        """Test that CartPolePOMDPs with different discount factors are not equal."""
        other_env = CartPolePOMDP(
            discount_factor=0.8,
            noise_cov=base_cartpole_environment.noise_cov
        )
        assert base_cartpole_environment != other_env
        assert other_env != base_cartpole_environment  # Test symmetry
    
    def test_different_noise_covariance(self, base_cartpole_environment: CartPolePOMDP):
        """Test that CartPolePOMDPs with different noise covariance are not equal."""
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=np.eye(4) * 0.2  # Different noise covariance
        )
        assert base_cartpole_environment != other_env
        assert other_env != base_cartpole_environment  # Test symmetry
    
    def test_different_physical_parameters(self, base_cartpole_environment: CartPolePOMDP):
        """Test that CartPolePOMDPs with different physical parameters are not equal."""
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        other_env.gravity = 10.0  # Different gravity
        assert base_cartpole_environment != other_env
        
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        other_env.masscart = 2.0  # Different cart mass
        assert base_cartpole_environment != other_env
    
    def test_comparison_with_non_environment(self, base_cartpole_environment: CartPolePOMDP):
        """Test comparison with non-Environment objects."""
        assert base_cartpole_environment != "not an environment"
        assert base_cartpole_environment != 42
        assert base_cartpole_environment != None
    
    def test_missing_attributes(self, base_cartpole_environment: CartPolePOMDP):
        """Test equality when attributes are missing."""
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        delattr(other_env, 'gravity')
        assert base_cartpole_environment != other_env
        
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        delattr(other_env, 'masscart')
        assert base_cartpole_environment != other_env
    
    def test_deep_copy_equality(self, base_cartpole_environment: CartPolePOMDP):
        """Test that a deep copy of CartPolePOMDP is equal to original."""
        import copy
        copied_env = copy.deepcopy(base_cartpole_environment)
        assert copied_env == base_cartpole_environment
        assert base_cartpole_environment == copied_env  # Test symmetry


class TestCartPolePOMDPConfigId:
    """Test suite for CartPolePOMDP config_id functionality."""
    
    def test_config_id_consistency(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id is consistent for identical environments."""
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        assert base_cartpole_environment.config_id == other_env.config_id
    
    def test_config_id_different_discount_factor(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id changes with different discount factor."""
        other_env = CartPolePOMDP(
            discount_factor=0.8,
            noise_cov=base_cartpole_environment.noise_cov
        )
        assert base_cartpole_environment.config_id != other_env.config_id
    
    def test_config_id_different_noise_covariance(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id changes with different noise covariance."""
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=np.eye(4) * 0.2  # Different noise covariance
        )
        assert base_cartpole_environment.config_id != other_env.config_id
    
    def test_config_id_different_physical_parameters(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id changes with different physical parameters."""
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        other_env.gravity = 10.0  # Different gravity
        assert base_cartpole_environment.config_id != other_env.config_id
        
        other_env = CartPolePOMDP(
            discount_factor=0.95,
            noise_cov=base_cartpole_environment.noise_cov
        )
        other_env.masscart = 2.0  # Different cart mass
        assert base_cartpole_environment.config_id != other_env.config_id
    
    def test_config_id_format(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id is a valid SHA-256 hash."""
        config_id = base_cartpole_environment.config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 64  # SHA-256 hash length
        assert all(c in '0123456789abcdef' for c in config_id)  # Valid hex characters
    
    def test_config_id_deterministic(self, base_cartpole_environment: CartPolePOMDP):
        """Test that config_id is deterministic (same input always produces same output)."""
        config_id1 = base_cartpole_environment.config_id
        config_id2 = base_cartpole_environment.config_id
        assert config_id1 == config_id2


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
        masspole=masspole,
    )

    next_state = transition.sample()

    # Verify state dimensions
    assert next_state.shape == (4,)
    # Verify state bounds are reasonable
    assert np.all(np.isfinite(next_state))
    # Verify reasonable initial movement
    assert np.abs(next_state[0]) < 0.1  # x position should change little in one step
    assert (
        np.abs(next_state[1]) < 0.5
    )  # x_dot should be reasonable for the force applied
    assert np.abs(next_state[2]) < 0.1  # theta should change little in one step
    assert np.abs(next_state[3]) < 0.5  # theta_dot should be reasonable


def test_cartpole_observation():
    # Test observation model with known parameters
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = np.array([0])
    noise_cov = np.eye(4) * 0.1  # Small noise

    observation = CartPoleObservation(
        next_state=state, action=action, noise_cov=noise_cov
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

    env = CartPolePOMDP(discount_factor=0.95, noise_cov=noise_cov)

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
