import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask
from POMDPPlanners.core.simulation import History
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT


def create_test_belief():
    """Helper function to create a valid belief state for testing."""
    particles = ["tiger_left", "tiger_right"]
    log_weights = np.array([np.log(0.5), np.log(0.5)])
    return WeightedParticleBelief(
        particles=particles,
        log_weights=log_weights,
        resampling=False
    )

@pytest.fixture
def environment():
    """Fixture to create a Tiger POMDP environment."""
    return TigerPOMDP(discount_factor=0.95)

@pytest.fixture
def policy(environment):
    """Fixture to create a SparsePFT policy."""
    return SparsePFT(
        environment=environment,
        discount_factor=0.95,
        gamma=0.95,
        depth=3,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=4,
        n_simulations=2
    )

def test_episode_simulation_task_creation(environment, policy):
    """Test creation and basic properties of EpisodeSimulationTask.
    
    Purpose: Validates that EpisodeSimulationTask can be created with correct attributes and properties
    
    Given: A TigerPOMDP environment, SparsePFT policy, and test belief state
    When: EpisodeSimulationTask is created with specific parameters (num_steps=2, episode_id=1, seed=42)
    Then: Task has correct attributes matching input parameters and generates valid cache key
    
    Test type: unit
    """
    belief = create_test_belief()
    
    task = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1,
        console_output=False
    )
    
    assert task.environment == environment
    assert task.policy == policy
    assert task.initial_belief == belief
    assert task.num_steps == 2
    assert task.episode_id == 1
    assert task.seed == 42
    assert task.discount_factor == 0.95
    assert task.episode_number == 1
    assert isinstance(task._cache_key, str)
    assert len(task._cache_key) > 0

def test_episode_simulation_task_invalid_steps(environment, policy):
    """Test that EpisodeSimulationTask raises error for invalid num_steps.
    
    Purpose: Validates that EpisodeSimulationTask raises ValueError for invalid num_steps parameter
    
    Given: A TigerPOMDP environment, SparsePFT policy, and test belief state
    When: EpisodeSimulationTask is created with num_steps=-1 (invalid)
    Then: ValueError is raised with appropriate error message about num_steps being positive
    
    Test type: unit
    """
    belief = create_test_belief()
    
    with pytest.raises(ValueError, match="num_steps must be a positive integer"):
        EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=-1,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1
        )

def test_episode_simulation_task_equality(environment, policy):
    """Test task equality and hashing for EpisodeSimulationTask.
    
    Purpose: Validates that EpisodeSimulationTask equality comparison and hashing work correctly
    
    Given: Three EpisodeSimulationTask instances: two identical and one with different parameters
    When: Equality comparison and hashing are performed between tasks
    Then: Identical tasks are equal and have same hash, different tasks are unequal with different hashes
    
    Test type: unit
    """
    belief = create_test_belief()
    
    # Create identical tasks
    task1 = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1
    )
    
    task2 = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1
    )
    
    # Create different task (different num_steps)
    task3 = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=belief,
        num_steps=5,  # Different
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1
    )
    
    # Test equality
    assert task1 == task2
    assert task1 != task3
    assert task2 != task3
    
    # Test hashing
    assert hash(task1) == hash(task2)
    assert hash(task1) != hash(task3)
    assert hash(task2) != hash(task3)

def test_episode_simulation_task_serialization(environment, policy):
    """Test task serialization for EpisodeSimulationTask.
    
    Purpose: Validates that EpisodeSimulationTask can be properly serialized to dictionary and reconstructed
    
    Given: A EpisodeSimulationTask with specific configuration parameters
    When: Task is serialized to dictionary with to_dict() method and reconstructed with from_dict()
    Then: Reconstructed task has identical attributes to original task
    
    Test type: unit
    """
    belief = create_test_belief()
    
    original_task = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1
    )
    
    # Test to_dict
    task_dict = original_task.to_dict()
    assert isinstance(task_dict, dict)
    assert task_dict['environment'] == environment
    assert task_dict['policy'] == policy
    assert task_dict['initial_belief'] == belief
    assert task_dict['num_steps'] == 2
    assert task_dict['episode_id'] == 1
    assert task_dict['seed'] == 42
    assert task_dict['discount_factor'] == 0.95
    assert task_dict['episode_number'] == 1
    
    # Test from_dict
    reconstructed_task = EpisodeSimulationTask.from_dict(task_dict)
    assert reconstructed_task.environment == original_task.environment
    assert reconstructed_task.policy == original_task.policy
    assert reconstructed_task.initial_belief == original_task.initial_belief
    assert reconstructed_task.num_steps == original_task.num_steps
    assert reconstructed_task.episode_id == original_task.episode_id
    assert reconstructed_task.seed == original_task.seed
    assert reconstructed_task.discount_factor == original_task.discount_factor
    assert reconstructed_task.episode_number == original_task.episode_number

def test_episode_simulation_task_execution(environment, policy):
    """Test task execution for EpisodeSimulationTask.
    
    Purpose: Validates that EpisodeSimulationTask can execute and return results
    
    Given: A EpisodeSimulationTask with valid configuration
    When: Task is executed with run() method
    Then: Task executes successfully and returns appropriate result or None
    
    Test type: integration
    """
    belief = create_test_belief()
    
    task = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1,
        console_output=False
    )
    
    # Mock the run_episode function to avoid actual execution
    with patch('POMDPPlanners.simulations.episodes.run_episode') as mock_run_episode:
        mock_run_episode.return_value = Mock()
        mock_run_episode.return_value.history = [Mock(), Mock()]
        mock_run_episode.return_value.reach_terminal_state = True
        mock_run_episode.return_value.actual_num_steps = 2
        
        result = task.run()
        
        # Verify run_episode was called with correct parameters
        mock_run_episode.assert_called_once_with(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            logger=task.logger
        )
        
        # Verify result
        assert result is not None
        assert hasattr(result, 'history')
        assert hasattr(result, 'reach_terminal_state')
        assert hasattr(result, 'actual_num_steps')

def test_episode_simulation_task_value_error_logging(caplog):
    """Test that EpisodeSimulationTask logs ValueError exceptions properly.
    
    Purpose: Validates that EpisodeSimulationTask logs ValueError exceptions with appropriate detail
    
    Given: A EpisodeSimulationTask that encounters a ValueError during execution
    When: Task execution raises a ValueError
    Then: Error is logged with appropriate level and message
    
    Test type: unit
    """
    from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask
    
    # Create a mock environment and policy
    mock_env = Mock()
    mock_env.name = "test_env"
    mock_env.config_id = "test_env_config"
    
    mock_policy = Mock()
    mock_policy.name = "test_policy"
    mock_policy.config_id = "test_policy_config"
    
    mock_belief = Mock()
    mock_belief.config_id = "test_belief_config"
    
    # Create task
    task = EpisodeSimulationTask(
        environment=mock_env,
        policy=mock_policy,
        initial_belief=mock_belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1,
        console_output=False
    )
    
    # Mock run_episode to raise ValueError
    with patch('POMDPPlanners.simulations.episodes.run_episode') as mock_run_episode:
        mock_run_episode.side_effect = ValueError("Test value error")
        
        # Run task and expect it to handle the error gracefully
        result = task.run()
        
        # Verify error was logged
        assert "Error running episode 1: Test value error" in caplog.text
        assert result is None

def test_episode_simulation_task_runtime_error_logging(caplog):
    """Test that EpisodeSimulationTask logs RuntimeError exceptions properly.
    
    Purpose: Validates that EpisodeSimulationTask logs RuntimeError exceptions with appropriate detail
    
    Given: A EpisodeSimulationTask that encounters a RuntimeError during execution
    When: Task execution raises a RuntimeError
    Then: Error is logged with appropriate level and message
    
    Test type: unit
    """
    from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask
    
    # Create a mock environment and policy
    mock_env = Mock()
    mock_env.name = "test_env"
    mock_env.config_id = "test_env_config"
    
    mock_policy = Mock()
    mock_policy.name = "test_policy"
    mock_policy.config_id = "test_policy_config"
    
    mock_belief = Mock()
    mock_belief.config_id = "test_belief_config"
    
    # Create task
    task = EpisodeSimulationTask(
        environment=mock_env,
        policy=mock_policy,
        initial_belief=mock_belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1,
        console_output=False
    )
    
    # Mock run_episode to raise RuntimeError
    with patch('POMDPPlanners.simulations.episodes.run_episode') as mock_run_episode:
        mock_run_episode.side_effect = RuntimeError("Test runtime error")
        
        # Run task and expect it to handle the error gracefully
        result = task.run()
        
        # Verify error was logged
        assert "Error running episode 1: Test runtime error" in caplog.text
        assert result is None

def test_episode_simulation_task_type_error_logging(caplog):
    """Test that EpisodeSimulationTask logs TypeError exceptions properly.
    
    Purpose: Validates that EpisodeSimulationTask logs TypeError exceptions with appropriate detail
    
    Given: A EpisodeSimulationTask that encounters a TypeError during execution
    When: Task execution raises a TypeError
    Then: Error is logged with appropriate level and message
    
    Test type: unit
    """
    from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask
    
    # Create a mock environment and policy
    mock_env = Mock()
    mock_env.name = "test_env"
    mock_env.config_id = "test_env_config"
    
    mock_policy = Mock()
    mock_policy.name = "test_policy"
    mock_policy.config_id = "test_policy_config"
    
    mock_belief = Mock()
    mock_belief.config_id = "test_belief_config"
    
    # Create task
    task = EpisodeSimulationTask(
        environment=mock_env,
        policy=mock_policy,
        initial_belief=mock_belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1,
        console_output=False
    )
    
    # Mock run_episode to raise TypeError
    with patch('POMDPPlanners.simulations.episodes.run_episode') as mock_run_episode:
        mock_run_episode.side_effect = TypeError("Test type error")
        
        # Run task and expect it to handle the error gracefully
        result = task.run()
        
        # Verify error was logged
        assert "Error running episode 1: Test type error" in caplog.text
        assert result is None

def test_episode_simulation_task_custom_exception_logging(caplog):
    """Test that EpisodeSimulationTask logs custom exceptions properly.
    
    Purpose: Validates that EpisodeSimulationTask logs custom exceptions with appropriate detail
    
    Given: A EpisodeSimulationTask that encounters a custom exception during execution
    When: Task execution raises a custom exception
    Then: Error is logged with appropriate level and message
    
    Test type: unit
    """
    from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask
    
    # Create a custom exception class
    class CustomTestException(Exception):
        pass
    
    # Create a mock environment and policy
    mock_env = Mock()
    mock_env.name = "test_env"
    mock_env.config_id = "test_env_config"
    
    mock_policy = Mock()
    mock_policy.name = "test_policy"
    mock_policy.config_id = "test_policy_config"
    
    mock_belief = Mock()
    mock_belief.config_id = "test_belief_config"
    
    # Create task
    task = EpisodeSimulationTask(
        environment=mock_env,
        policy=mock_policy,
        initial_belief=mock_belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1,
        console_output=False
    )
    
    # Mock run_episode to raise custom exception
    with patch('POMDPPlanners.simulations.episodes.run_episode') as mock_run_episode:
        mock_run_episode.side_effect = CustomTestException("Test custom exception")
        
        # Run task and expect it to handle the error gracefully
        result = task.run()
        
        # Verify error was logged
        assert "Error running episode 1: Test custom exception" in caplog.text
        assert result is None

def test_episode_simulation_task_logging_includes_traceback(caplog):
    """Test that EpisodeSimulationTask logs include full traceback information.
    
    Purpose: Validates that EpisodeSimulationTask logs include full exception traceback for debugging
    
    Given: A EpisodeSimulationTask that encounters an exception during execution
    When: Task execution raises an exception
    Then: Full exception details including traceback are logged
    
    Test type: unit
    """
    from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask
    
    # Create a mock environment and policy
    mock_env = Mock()
    mock_env.name = "test_env"
    mock_env.config_id = "test_env_config"
    
    mock_policy = Mock()
    mock_policy.name = "test_policy"
    mock_policy.config_id = "test_policy_config"
    
    mock_belief = Mock()
    mock_belief.config_id = "test_belief_config"
    
    # Create task
    task = EpisodeSimulationTask(
        environment=mock_env,
        policy=mock_policy,
        initial_belief=mock_belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1,
        console_output=False
    )
    
    # Mock run_episode to raise an exception
    with patch('POMDPPlanners.simulations.episodes.run_episode') as mock_run_episode:
        mock_run_episode.side_effect = Exception("Test exception with traceback")
        
        # Run task and expect it to handle the error gracefully
        result = task.run()
        
        # Verify error was logged with traceback
        assert "Error running episode 1: Test exception with traceback" in caplog.text
        assert "Full exception details:" in caplog.text
        assert result is None
