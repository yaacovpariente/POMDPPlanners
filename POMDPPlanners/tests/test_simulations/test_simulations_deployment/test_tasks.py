import pytest
import numpy as np

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
            console_output=False
        )

def test_episode_simulation_task_equality(environment, policy):
    """Test task equality and hashing.
    
    Purpose: Validates that EpisodeSimulationTask equality comparison and hashing work correctly
    
    Given: Three EpisodeSimulationTask instances: two identical and one with different episode_id
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
        console_output=False
    )
    
    task2 = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        console_output=False
    )
    
    # Create different task
    task3 = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=2,  # Different episode_id
        seed=42,
        console_output=False
    )
    
    assert task1 == task2
    assert task1 != task3
    assert hash(task1) == hash(task2)
    assert hash(task1) != hash(task3)

def test_episode_simulation_task_serialization(environment, policy):
    """Test task serialization and deserialization.
    
    Purpose: Validates that EpisodeSimulationTask can be properly serialized and deserialized
    
    Given: An EpisodeSimulationTask with specific configuration parameters
    When: Task is serialized to dictionary and then deserialized back to task object
    Then: Deserialized task has identical attributes to original task, including environment, policy, and all parameters
    
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
        episode_number=1,
        console_output=False
    )
    
    # Test serialization
    task_dict = original_task.to_dict()
    assert isinstance(task_dict, dict)
    assert task_dict['num_steps'] == 2
    assert task_dict['episode_id'] == 1
    assert task_dict['seed'] == 42
    assert task_dict['discount_factor'] == 0.95
    assert task_dict['episode_number'] == 1
    
    # Test deserialization
    reconstructed_task = EpisodeSimulationTask.from_dict(task_dict)
    assert reconstructed_task == original_task

def test_episode_simulation_task_execution(environment, policy):
    """Test actual task execution with real environment and policy.
    
    Purpose: Validates that EpisodeSimulationTask can execute successfully and produce valid results
    
    Given: An EpisodeSimulationTask with TigerPOMDP environment and SparsePFT policy
    When: Task.run() method is executed
    Then: Returns History object with correct structure, step count, and episode completion status
    
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
        console_output=False
    )
    
    history = task.run()
    assert isinstance(history, History)
    assert len(history.history) <= 2  # May be less if terminal state reached
    assert history.discount_factor == 0.95  # Should match environment's discount factor
    assert history.actual_num_steps <= 2
    
    # Verify history contents
    for step in history.history:
        assert step.state in ["tiger_left", "tiger_right"]
        assert step.action in ["listen", "open_left", "open_right"]
        assert step.observation in ["hear_left", "hear_right", "hear_nothing"]
        assert isinstance(step.reward, float)
        assert isinstance(step.belief, WeightedParticleBelief)


class FailingEnvironment(TigerPOMDP):
    """Mock environment that raises different types of exceptions."""
    
    def __init__(self, exception_type=ValueError, error_message="Test error", fail_method="reward"):
        # Initialize as TigerPOMDP but override methods to raise exceptions
        super().__init__(discount_factor=0.95)
        self.name = "FailingEnvironment"
        self.exception_type = exception_type
        self.error_message = error_message
        self.fail_method = fail_method
    
    def reward(self, state, action):
        if self.fail_method == "reward":
            raise self.exception_type(self.error_message)
        return super().reward(state, action)
    
    def state_transition_model(self, state, action):
        if self.fail_method == "state_transition":
            raise self.exception_type(self.error_message)
        return super().state_transition_model(state, action)
    
    def observation_model(self, next_state, action):
        if self.fail_method == "observation":
            raise self.exception_type(self.error_message)
        return super().observation_model(next_state, action)


class FailingPolicy(SparsePFT):
    """Mock policy that raises exceptions."""
    
    def __init__(self, environment, exception_type=RuntimeError, error_message="Policy failure"):
        # Initialize as SparsePFT but override methods to raise exceptions
        super().__init__(
            environment=environment,
            discount_factor=0.95,
            gamma=0.95,
            depth=3,
            c_ucb=1.0,
            beta_ucb=0.5,
            belief_child_num=4,
            n_simulations=2
        )
        self.name = "FailingPolicy"
        self.exception_type = exception_type
        self.error_message = error_message
    
    def action(self, belief):
        raise self.exception_type(self.error_message)


def test_episode_simulation_task_value_error_logging(caplog):
    """Test that ValueError in task execution is properly logged with details.
    
    Purpose: Validates that ValueError exceptions during task execution are logged with informative error messages
    
    Given: EpisodeSimulationTask with environment that raises ValueError in reward method
    When: Task execution is performed and fails with ValueError
    Then: Error details are logged including exception type, message, and full traceback
    
    Test type: unit
    """
    import logging
    
    # Create failing environment that raises ValueError in reward method
    failing_env = FailingEnvironment(
        exception_type=ValueError, 
        error_message="Invalid state configuration", 
        fail_method="reward"
    )
    
    # Create normal working policy
    normal_policy = SparsePFT(
        environment=failing_env,
        discount_factor=0.95,
        gamma=0.95,
        depth=3,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=4,
        n_simulations=2
    )
    belief = create_test_belief()
    
    # Create task with failing environment
    task = EpisodeSimulationTask(
        environment=failing_env,
        policy=normal_policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=999,
        seed=42,
        console_output=True  # Enable logging to caplog
    )
    
    # Capture logs at DEBUG level to get full exception details
    with caplog.at_level(logging.DEBUG):
        # Run task - should fail and log error details
        result = task.run()
    
    # Verify task failed
    assert result is None, "Task should return None on failure"
    
    # Verify error logging occurred
    error_logs = [record for record in caplog.records if record.levelname == 'ERROR']
    exception_logs = [record for record in caplog.records if 'exception' in record.message.lower()]
    
    # Should have at least one error log entry
    assert len(error_logs) >= 1, "Should have at least one ERROR log entry"
    
    # Check that error message contains relevant information
    error_messages = [record.message for record in error_logs]
    error_text = ' '.join(error_messages)
    
    assert "Error running episode 999" in error_text, "Should log specific episode ID"
    assert "Invalid state configuration" in error_text, "Should include original error message"
    
    # Verify exception details were logged (logger.exception() call)
    assert len(exception_logs) >= 1, "Should have exception details logged"
    
    # Check that ValueError is mentioned in logs (either in message or traceback)
    all_log_text = ' '.join([record.message for record in caplog.records])
    all_log_with_exc = ' '.join([getattr(record, 'exc_text', '') or '' for record in caplog.records if hasattr(record, 'exc_text')])
    full_log_content = all_log_text + ' ' + all_log_with_exc
    
    # Verify ValueError is mentioned somewhere in the logging
    has_value_error = "ValueError" in full_log_content
    # Also check if the exception type shows up in the error logs
    has_exception_type = any("ValueError" in str(record.exc_info) if record.exc_info else False for record in caplog.records)
    
    assert has_value_error or has_exception_type, "Should mention ValueError in logs or exception info"


def test_episode_simulation_task_runtime_error_logging(caplog):
    """Test that RuntimeError in task execution is properly logged with details.
    
    Purpose: Validates that RuntimeError exceptions during task execution are logged with informative error messages
    
    Given: EpisodeSimulationTask with policy that raises RuntimeError  
    When: Task execution is performed and fails with RuntimeError
    Then: Error details are logged including exception type, message, and full traceback
    
    Test type: unit
    """
    import logging
    
    # Create normal environment but failing policy
    environment = TigerPOMDP(discount_factor=0.95)
    failing_policy = FailingPolicy(environment=environment, exception_type=RuntimeError, error_message="Policy computation failed")
    belief = create_test_belief()
    
    # Create task with failing policy
    task = EpisodeSimulationTask(
        environment=environment,
        policy=failing_policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=888,
        seed=42,
        console_output=True  # Enable logging to caplog
    )
    
    # Capture logs at DEBUG level
    with caplog.at_level(logging.DEBUG):
        # Run task - should fail and log error details
        result = task.run()
    
    # Verify task failed
    assert result is None, "Task should return None on failure"
    
    # Verify error logging occurred
    error_logs = [record for record in caplog.records if record.levelname == 'ERROR']
    
    # Should have at least one error log entry
    assert len(error_logs) >= 1, "Should have at least one ERROR log entry"
    
    # Check that error message contains relevant information
    error_messages = [record.message for record in error_logs]
    error_text = ' '.join(error_messages)
    
    assert "Error running episode 888" in error_text, "Should log specific episode ID"
    assert "Policy computation failed" in error_text, "Should include original error message"
    
    # Check that RuntimeError is mentioned in logs (either in message or traceback)
    all_log_text = ' '.join([record.message for record in caplog.records])
    all_log_with_exc = ' '.join([getattr(record, 'exc_text', '') or '' for record in caplog.records if hasattr(record, 'exc_text')])
    full_log_content = all_log_text + ' ' + all_log_with_exc
    
    # Verify RuntimeError is mentioned somewhere in the logging
    has_runtime_error = "RuntimeError" in full_log_content
    # Also check if the exception type shows up in the error logs
    has_exception_type = any("RuntimeError" in str(record.exc_info) if record.exc_info else False for record in caplog.records)
    
    assert has_runtime_error or has_exception_type, "Should mention RuntimeError in logs or exception info"


def test_episode_simulation_task_type_error_logging(caplog):
    """Test that TypeError in task execution is properly logged with details.
    
    Purpose: Validates that TypeError exceptions during task execution are logged with informative error messages
    
    Given: EpisodeSimulationTask with environment that raises TypeError
    When: Task execution is performed and fails with TypeError  
    Then: Error details are logged including exception type, message, and full traceback
    
    Test type: unit
    """
    import logging
    
    # Create failing environment that raises TypeError in state transition
    failing_env = FailingEnvironment(
        exception_type=TypeError, 
        error_message="Expected float, got string", 
        fail_method="state_transition"
    )
    
    # Create normal working policy
    policy = SparsePFT(
        environment=failing_env,
        discount_factor=0.95,
        gamma=0.95,
        depth=3,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=4,
        n_simulations=2
    )
    belief = create_test_belief()
    
    # Create task with failing environment
    task = EpisodeSimulationTask(
        environment=failing_env,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=777,
        seed=42,
        console_output=True  # Enable logging to caplog
    )
    
    # Capture logs at DEBUG level
    with caplog.at_level(logging.DEBUG):
        # Run task - should fail and log error details
        result = task.run()
    
    # Verify task failed
    assert result is None, "Task should return None on failure"
    
    # Verify error logging occurred
    error_logs = [record for record in caplog.records if record.levelname == 'ERROR']
    
    # Should have at least one error log entry
    assert len(error_logs) >= 1, "Should have at least one ERROR log entry"
    
    # Check that error message contains relevant information
    error_messages = [record.message for record in error_logs]
    error_text = ' '.join(error_messages)
    
    assert "Error running episode 777" in error_text, "Should log specific episode ID"
    assert "Expected float, got string" in error_text, "Should include original error message"
    
    # Check that TypeError is mentioned in logs (either in message or traceback)
    all_log_text = ' '.join([record.message for record in caplog.records])
    all_log_with_exc = ' '.join([getattr(record, 'exc_text', '') or '' for record in caplog.records if hasattr(record, 'exc_text')])
    full_log_content = all_log_text + ' ' + all_log_with_exc
    
    # Verify TypeError is mentioned somewhere in the logging
    has_type_error = "TypeError" in full_log_content
    # Also check if the exception type shows up in the error logs
    has_exception_type = any("TypeError" in str(record.exc_info) if record.exc_info else False for record in caplog.records)
    
    assert has_type_error or has_exception_type, "Should mention TypeError in logs or exception info"


def test_episode_simulation_task_custom_exception_logging(caplog):
    """Test that custom exceptions in task execution are properly logged with details.
    
    Purpose: Validates that custom exception classes during task execution are logged with informative error messages
    
    Given: EpisodeSimulationTask with environment that raises custom exception
    When: Task execution is performed and fails with custom exception
    Then: Error details are logged including exception type, message, and full traceback
    
    Test type: unit
    """
    import logging
    
    # Define custom exception
    class CustomPlanningError(Exception):
        """Custom exception for testing error logging."""
        pass
    
    # Create failing environment that raises custom exception in observation method
    failing_env = FailingEnvironment(
        exception_type=CustomPlanningError, 
        error_message="Custom planning algorithm failed with invalid parameters",
        fail_method="observation"
    )
    
    # Create normal working policy
    policy = SparsePFT(
        environment=failing_env,
        discount_factor=0.95,
        gamma=0.95,
        depth=3,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=4,
        n_simulations=2
    )
    belief = create_test_belief()
    
    # Create task with failing environment
    task = EpisodeSimulationTask(
        environment=failing_env,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=666,
        seed=42,
        console_output=True  # Enable logging to caplog
    )
    
    # Capture logs at DEBUG level
    with caplog.at_level(logging.DEBUG):
        # Run task - should fail and log error details
        result = task.run()
    
    # Verify task failed
    assert result is None, "Task should return None on failure"
    
    # Verify error logging occurred
    error_logs = [record for record in caplog.records if record.levelname == 'ERROR']
    
    # Should have at least one error log entry
    assert len(error_logs) >= 1, "Should have at least one ERROR log entry"
    
    # Check that error message contains relevant information
    error_messages = [record.message for record in error_logs]
    error_text = ' '.join(error_messages)
    
    assert "Error running episode 666" in error_text, "Should log specific episode ID"
    assert "Custom planning algorithm failed with invalid parameters" in error_text, "Should include original error message"
    
    # Check that custom exception name is mentioned in logs (either in message or traceback)
    all_log_text = ' '.join([record.message for record in caplog.records])
    all_log_with_exc = ' '.join([getattr(record, 'exc_text', '') or '' for record in caplog.records if hasattr(record, 'exc_text')])
    full_log_content = all_log_text + ' ' + all_log_with_exc
    
    # Verify CustomPlanningError is mentioned somewhere in the logging
    has_custom_error = "CustomPlanningError" in full_log_content
    # Also check if the exception type shows up in the error logs
    has_exception_type = any("CustomPlanningError" in str(record.exc_info) if record.exc_info else False for record in caplog.records)
    
    assert has_custom_error or has_exception_type, "Should mention CustomPlanningError in logs or exception info"


def test_episode_simulation_task_logging_includes_traceback(caplog):
    """Test that exception logging includes full traceback information.
    
    Purpose: Validates that exception logging includes complete stack trace for debugging
    
    Given: EpisodeSimulationTask that raises nested exceptions with stack trace  
    When: Task execution fails with complex exception stack
    Then: Full traceback information is logged including file names, line numbers, and call stack
    
    Test type: unit
    """
    import logging
    
    class DeepFailingEnvironment(TigerPOMDP):
        """Environment that creates a deeper call stack before failing."""
        
        def __init__(self):
            super().__init__(discount_factor=0.95)
            self.name = "DeepFailingEnvironment"
        
        def reward(self, state, action):
            return self._helper_method_1()
        
        def _helper_method_1(self):
            return self._helper_method_2()
        
        def _helper_method_2(self):
            raise ValueError("Deep nested error in helper method 2")
    
    # Create environment with nested failure
    deep_failing_env = DeepFailingEnvironment()
    policy = SparsePFT(
        environment=deep_failing_env,
        discount_factor=0.95,
        gamma=0.95,
        depth=3,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=4,
        n_simulations=2
    )
    belief = create_test_belief()
    
    # Create task
    task = EpisodeSimulationTask(
        environment=deep_failing_env,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=555,
        seed=42,
        console_output=True
    )
    
    # Capture logs including traceback
    with caplog.at_level(logging.DEBUG):
        result = task.run()
    
    # Verify task failed
    assert result is None, "Task should return None on failure"
    
    # Get all log text including exception details to check for traceback information
    all_log_text = ' '.join([record.message for record in caplog.records])
    all_log_with_exc = ' '.join([getattr(record, 'exc_text', '') or '' for record in caplog.records if hasattr(record, 'exc_text')])
    
    # Also check caplog's output which includes the full logging output
    full_output = caplog.text
    
    # Combine all sources of logging information
    comprehensive_log = all_log_text + ' ' + all_log_with_exc + ' ' + full_output
    
    # Verify traceback information is present (either in messages or exception info)
    has_traceback = "Traceback" in comprehensive_log or "File " in comprehensive_log
    assert has_traceback, "Should include traceback information"
    
    # Verify method names appear in the stack trace
    assert "_helper_method_1" in comprehensive_log, "Should show method names in stack trace"
    assert "_helper_method_2" in comprehensive_log, "Should show nested method calls"
    assert "Deep nested error in helper method 2" in comprehensive_log, "Should include original error message"
    
    # Verify line numbers or file paths are mentioned (common in tracebacks)
    assert any(char.isdigit() for char in comprehensive_log), "Should include line numbers in traceback" 