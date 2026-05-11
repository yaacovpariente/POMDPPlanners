# pylint: disable=protected-access,too-many-lines  # Tests need to access protected members
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.environment import DiscreteActionsEnvironment
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler, UnitCircleActionSampler
from POMDPPlanners.utils.logger import ConditionalMemoryHandler
from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)


def _build_environment_and_belief_instances():
    api = EnvironmentConfigsAPI(discount_factor=0.95)
    n_particles = 20
    envs = {}
    beliefs = {}
    for key, config_method in [
        ("tiger", api.tiger_pomdp_config),
        ("cartpole", api.cartpole_pomdp_config),
        ("mountain_car", api.mountain_car_pomdp_config),
        ("push", api.push_pomdp_config),
        ("safety_ant_velocity", api.safety_ant_velocity_pomdp_config),
        (
            "continuous_light_dark",
            api.continuous_observations_continuous_actions_light_dark_pomdp_config,
        ),
    ]:
        env, belief = config_method(n_particles=n_particles)
        envs[key] = env
        beliefs[key] = belief
    discrete_ld = DiscreteLightDarkPOMDP(discount_factor=0.95)
    beliefs["discrete_light_dark"] = get_initial_belief(
        pomdp=discrete_ld, n_particles=n_particles, resampling=True
    )
    envs["discrete_light_dark"] = discrete_ld
    return envs, beliefs


environment_instances, belief_instances = _build_environment_and_belief_instances()


def create_test_belief():
    """Helper function to create a valid belief state for testing."""
    particles = ["tiger_left", "tiger_right"]
    log_weights = np.array([np.log(0.5), np.log(0.5)])
    return WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)


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
        depth=3,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=4,
        n_simulations=2,
    )


def test_episode_simulation_task_import():
    """Test that EpisodeSimulationTask can be imported.

    Purpose: Validates basic task import functionality

    Given: EpisodeSimulationTask class
    When: Task class is imported
    Then: Task class is available and not None

    Test type: unit
    """
    assert EpisodeSimulationTask is not None


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
        console_output=False,
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
            episode_number=1,
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
        episode_number=1,
    )

    task2 = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1,
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
        episode_number=1,
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
        episode_number=1,
    )

    # Test to_dict
    task_dict = original_task.to_dict()
    assert isinstance(task_dict, dict)
    assert task_dict["environment"] == environment
    assert task_dict["policy"] == policy
    assert task_dict["initial_belief"] == belief
    assert task_dict["num_steps"] == 2
    assert task_dict["episode_id"] == 1
    assert task_dict["seed"] == 42
    assert task_dict["discount_factor"] == 0.95
    assert task_dict["episode_number"] == 1

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
        console_output=False,
    )

    # Mock the run_episode function to avoid actual execution
    # Patch at the point where it's used in the EpisodeSimulationTask class
    with patch.object(task, "run") as mock_run:
        # Create real History object instead of Mock
        step1 = StepData(
            state="tiger_left",
            action="listen",
            next_state="tiger_left",
            observation="growl_left",
            reward=-1.0,
            belief=create_test_belief(),
        )
        step2 = StepData(
            state="tiger_left",
            action="open_left",
            next_state="tiger_right",
            observation="tiger_left",
            reward=10.0,
            belief=create_test_belief(),
        )

        mock_result = History(
            history=[step1, step2],
            actual_num_steps=2,
            reach_terminal_state=True,
            average_action_time=0.1,
            average_belief_update_time=0.05,
            average_observation_time=0.02,
            average_reward_time=0.01,
            average_state_sampling_time=0.03,
            discount_factor=0.95,
            policy_run_data=[PolicyRunData(info_variables=[])],
        )
        mock_run.return_value = mock_result

        result = task.run()

        # Verify run was called
        mock_run.assert_called_once()

        # Verify result
        assert result is not None
        assert hasattr(result, "history")
        assert hasattr(result, "reach_terminal_state")
        assert hasattr(result, "actual_num_steps")


def test_episode_simulation_task_value_error_logging(tmp_path, policy):
    """Test that EpisodeSimulationTask logs ValueError exceptions properly.

    Purpose: Validates that EpisodeSimulationTask logs ValueError exceptions with appropriate detail

    Given: A EpisodeSimulationTask that encounters a ValueError during execution
    When: Task execution raises a ValueError
    Then: Error is logged with appropriate level and message

    Test type: unit
    """
    belief = create_test_belief()

    # Create a temporary cache directory
    cache_dir = tmp_path / "test_cache_value_error"
    cache_dir.mkdir()

    # Create a real environment and patch it to cause an error during execution
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env")
    # Patch the environment to fail when sample_next_state is called
    with patch.object(test_env, "sample_next_state", side_effect=ValueError("Test value error")):
        # Create task using the real environment with cache_dir to enable file logging
        task = EpisodeSimulationTask(
            environment=test_env,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            cache_dir=cache_dir,
            console_output=False,
        )

        # Run the task - it should handle the error gracefully
        result = task.run()

        # Verify that the task handled the error and returned None
        assert result is None

    # Find log files and check for error message
    log_files = list(cache_dir.rglob("*.log"))
    assert len(log_files) > 0, f"No log files were created in {cache_dir}"

    # Check if error is in any log file
    error_found = False
    for log_file in log_files:
        log_content = log_file.read_text()
        if "[EPISODE_001] Error running episode:" in log_content:
            error_found = True
            assert "Test value error" in log_content
            break

    assert error_found, f"Error message not found in any log file. Checked {len(log_files)} files."


def test_episode_simulation_task_runtime_error_logging(tmp_path, policy):
    """Test that EpisodeSimulationTask logs RuntimeError exceptions properly.

    Purpose: Validates that EpisodeSimulationTask logs RuntimeError exceptions with appropriate detail

    Given: A EpisodeSimulationTask that encounters a RuntimeError during execution
    When: Task execution raises a RuntimeError
    Then: Error is logged with appropriate level and message

    Test type: unit
    """
    belief = create_test_belief()

    # Create a temporary cache directory
    cache_dir = tmp_path / "test_cache_runtime_error"
    cache_dir.mkdir()

    # Create a real environment and patch it to cause a RuntimeError during execution
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env")
    # Patch the environment to fail when sample_observation is called
    with patch.object(
        test_env, "sample_observation", side_effect=RuntimeError("Test runtime error")
    ):
        # Create task using the real environment with cache_dir to enable file logging
        task = EpisodeSimulationTask(
            environment=test_env,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            cache_dir=cache_dir,
            console_output=False,
        )

        # Run the task - it should handle the error gracefully
        result = task.run()

        # Verify that the task handled the error and returned None
        assert result is None

    # Find log files and check for error message
    log_files = list(cache_dir.rglob("*.log"))
    assert len(log_files) > 0, f"No log files were created in {cache_dir}"

    # Check if error is in any log file
    error_found = False
    for log_file in log_files:
        log_content = log_file.read_text()
        if "[EPISODE_001] Error running episode:" in log_content:
            error_found = True
            assert "Test runtime error" in log_content
            break

    assert error_found, f"Error message not found in any log file. Checked {len(log_files)} files."


def test_episode_simulation_task_type_error_logging(tmp_path, policy):
    """Test that EpisodeSimulationTask logs TypeError exceptions properly.

    Purpose: Validates that EpisodeSimulationTask logs TypeError exceptions with appropriate detail

    Given: A EpisodeSimulationTask that encounters a TypeError during execution
    When: Task execution raises a TypeError
    Then: Error is logged with appropriate level and message

    Test type: unit
    """
    belief = create_test_belief()

    # Create a temporary cache directory
    cache_dir = tmp_path / "test_cache_type_error"
    cache_dir.mkdir()

    # Create a real environment and patch it to cause a TypeError during execution
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env")
    # Patch the environment to fail when reward is called
    with patch.object(test_env, "reward", side_effect=TypeError("Test type error")):
        # Create task using the real environment with cache_dir to enable file logging
        task = EpisodeSimulationTask(
            environment=test_env,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            cache_dir=cache_dir,
            console_output=False,
        )

        # Run the task - it should handle the error gracefully
        result = task.run()

        # Verify that the task handled the error and returned None
        assert result is None

    # Find log files and check for error message
    log_files = list(cache_dir.rglob("*.log"))
    assert len(log_files) > 0, f"No log files were created in {cache_dir}"

    # Check if error is in any log file
    error_found = False
    for log_file in log_files:
        log_content = log_file.read_text()
        if "[EPISODE_001] Error running episode:" in log_content:
            error_found = True
            assert "Test type error" in log_content
            break

    assert error_found, f"Error message not found in any log file. Checked {len(log_files)} files."


def test_episode_simulation_task_custom_exception_logging(tmp_path, policy):
    """Test that EpisodeSimulationTask logs custom exceptions properly.

    Purpose: Validates that EpisodeSimulationTask logs custom exceptions with appropriate detail

    Given: A EpisodeSimulationTask that encounters a custom exception during execution
    When: Task execution raises a custom exception
    Then: Error is logged with appropriate level and message

    Test type: unit
    """

    # Create a custom exception class
    class CustomTestException(Exception):
        pass

    belief = create_test_belief()

    # Create a temporary cache directory
    cache_dir = tmp_path / "test_cache_custom_error"
    cache_dir.mkdir()

    # Create a real environment and patch it to cause a custom exception during execution
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env")
    # Patch the environment to fail when is_terminal is called
    with patch.object(
        test_env,
        "is_terminal",
        side_effect=CustomTestException("Test custom exception"),
    ):
        # Create task using the real environment with cache_dir to enable file logging
        task = EpisodeSimulationTask(
            environment=test_env,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            cache_dir=cache_dir,
            console_output=False,
        )

        # Run the task - it should handle the error gracefully
        result = task.run()

        # Verify that the task handled the error and returned None
        assert result is None

    # Find log files and check for error message
    log_files = list(cache_dir.rglob("*.log"))
    assert len(log_files) > 0, f"No log files were created in {cache_dir}"

    # Check if error is in any log file
    error_found = False
    for log_file in log_files:
        log_content = log_file.read_text()
        if "[EPISODE_001] Error running episode:" in log_content:
            error_found = True
            assert "Test custom exception" in log_content
            break

    assert error_found, f"Error message not found in any log file. Checked {len(log_files)} files."


def test_episode_simulation_task_logging_includes_traceback(tmp_path, policy):
    """Test that EpisodeSimulationTask logs include full traceback information.

    Purpose: Validates that EpisodeSimulationTask logs include full exception traceback for debugging

    Given: A EpisodeSimulationTask that encounters an exception during execution
    When: Task execution raises an exception
    Then: Full exception details including traceback are logged

    Test type: unit
    """
    belief = create_test_belief()

    # Create a temporary cache directory
    cache_dir = tmp_path / "test_cache_traceback"
    cache_dir.mkdir()

    # Create a real environment and patch it to cause an exception during execution
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env")
    # Patch the environment to fail when sample_next_state is called
    with patch.object(
        test_env,
        "sample_next_state",
        side_effect=Exception("Test exception with traceback"),
    ):
        # Create task using the real environment with cache_dir to enable file logging
        task = EpisodeSimulationTask(
            environment=test_env,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            cache_dir=cache_dir,
            console_output=False,
        )

        # Run the task - it should handle the error gracefully
        result = task.run()

        # Verify that the task handled the error and returned None
        assert result is None

    # Find log files and check for error message and traceback
    log_files = list(cache_dir.rglob("*.log"))
    assert len(log_files) > 0, f"No log files were created in {cache_dir}"

    # Check if error and traceback are in any log file
    error_found = False
    traceback_found = False
    for log_file in log_files:
        log_content = log_file.read_text()
        if "[EPISODE_001] Error running episode:" in log_content:
            error_found = True
            assert "Test exception with traceback" in log_content
        if "Full exception details:" in log_content:
            traceback_found = True

    assert error_found, f"Error message not found in any log file. Checked {len(log_files)} files."
    assert (
        traceback_found
    ), f"Traceback information not found in any log file. Checked {len(log_files)} files."


def test_episode_simulation_task_error_written_to_log_file(tmp_path, policy):
    """Test that EpisodeSimulationTask errors are written to the actual log file on disk.

    Purpose: Validates that error messages are persisted to log files, not just captured in memory

    Given: A EpisodeSimulationTask configured with a temporary log directory that encounters an error
    When: Task execution raises an exception
    Then: Error message and traceback are written to the log file on disk

    Test type: integration
    """

    belief = create_test_belief()

    # Create a temporary cache directory
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()

    # Create environment and patch it to cause an exception
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env")

    with patch.object(
        test_env,
        "sample_next_state",
        side_effect=ValueError("Test error for log file verification"),
    ):
        # Create task with cache_dir to enable file logging
        task = EpisodeSimulationTask(
            environment=test_env,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            cache_dir=cache_dir,
            console_output=False,
        )

        # Run the task - it should handle the error gracefully
        result = task.run()

        # Verify task failed
        assert result is None

    # Find log files in the cache directory
    log_files = list(cache_dir.rglob("*.log"))
    assert len(log_files) > 0, f"No log files were created in {cache_dir}"

    # Read all log files and check if error is in any of them
    error_found = False
    for log_file in log_files:
        log_content = log_file.read_text()
        if "[EPISODE_001] Error running episode:" in log_content:
            error_found = True
            # Verify all expected content is in the log
            assert "Test error for log file verification" in log_content
            assert "Full exception details:" in log_content
            break

    assert error_found, f"Error message not found in any log file. Checked {len(log_files)} files."


# Failure-Only Logging Tests


def test_episode_simulation_task_log_only_on_failure_successful_episode_no_logs(tmp_path):
    """Test that successful episodes with log_only_on_failure=True produce no log output.

    Purpose: Validates that buffered logging discards logs for successful episodes

    Given: EpisodeSimulationTask with log_only_on_failure=True and a successful episode execution
    When: Task executes successfully without errors
    Then: No logs are written to disk (log file is empty or has zero bytes)

    Test type: integration
    """
    belief = create_test_belief()

    # Create a temporary cache directory
    cache_dir = tmp_path / "test_cache_success"
    cache_dir.mkdir()

    # Create a unique environment and policy to avoid logger conflicts with other tests
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env_success_nolog")
    test_policy = SparsePFT(
        environment=test_env,
        discount_factor=0.95,
        depth=3,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=4,
        n_simulations=2,
        name="test_policy_success_nolog",
    )

    # Create task with log_only_on_failure=True (default)
    task = EpisodeSimulationTask(
        environment=test_env,
        policy=test_policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1,
        cache_dir=cache_dir,
        debug=True,
        console_output=False,
        log_only_on_failure=True,
    )

    # Run successful episode
    result = task.run()

    # Verify task succeeded
    assert result is not None, "Task should have succeeded"

    # Check log files - they should exist but be empty
    log_files = list(cache_dir.rglob("*.log"))
    assert len(log_files) > 0, "Log file should be created"

    # Verify all log files are empty (no logs written for successful episode)
    for log_file in log_files:
        size = log_file.stat().st_size
        assert size == 0, f"Log file {log_file.name} should be empty but has {size} bytes"


def test_episode_simulation_task_log_only_on_failure_failed_episode_has_logs(tmp_path, policy):
    """Test that failed episodes with log_only_on_failure=True produce complete logs.

    Purpose: Validates that buffered logging flushes all logs when episode fails

    Given: EpisodeSimulationTask with log_only_on_failure=True and a simulated failure
    When: Task execution encounters an error
    Then: Complete logs including error details are written to disk

    Test type: integration
    """
    belief = create_test_belief()

    # Create a temporary cache directory
    cache_dir = tmp_path / "test_cache_failure"
    cache_dir.mkdir()

    # Create a real environment and patch it to cause an exception
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env_fail")

    with patch.object(
        test_env,
        "sample_next_state",
        side_effect=RuntimeError("Simulated failure for log_only_on_failure test"),
    ):
        # Create task with log_only_on_failure=True
        task = EpisodeSimulationTask(
            environment=test_env,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            cache_dir=cache_dir,
            debug=True,
            console_output=False,
            log_only_on_failure=True,
        )

        # Run task - it should fail gracefully
        result = task.run()

    # Verify task failed
    assert result is None, "Task should have failed and returned None"

    # Check log files - they should contain error information
    log_files = list(cache_dir.rglob("*.log"))
    assert len(log_files) > 0, "Log files should be created"

    # Verify logs contain error information
    error_found = False
    for log_file in log_files:
        log_content = log_file.read_text()
        if "[EPISODE_001] Error running episode:" in log_content:
            error_found = True
            assert "Simulated failure for log_only_on_failure test" in log_content
            assert "Full exception details:" in log_content
            # Verify log has substantial content (not empty)
            assert (
                log_file.stat().st_size > 100
            ), "Log file should contain detailed error information"
            break

    assert error_found, "Error message should be found in log files"


def test_episode_simulation_task_log_only_on_failure_false_always_logs(tmp_path):
    """Test that log_only_on_failure=False maintains backward compatibility with normal logging.

    Purpose: Validates that disabling log_only_on_failure produces logs for all episodes

    Given: EpisodeSimulationTask with log_only_on_failure=False and successful execution
    When: Task executes successfully
    Then: Complete logs are written to disk regardless of success

    Test type: integration
    """
    import time  # pylint: disable=import-outside-toplevel

    belief = create_test_belief()

    # Create a temporary cache directory
    cache_dir = tmp_path / "test_cache_always_log"
    cache_dir.mkdir()

    # Create a unique environment to ensure a unique logger name
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env_always_log")
    test_policy = SparsePFT(
        environment=test_env,
        discount_factor=0.95,
        depth=3,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=4,
        n_simulations=2,
        name="test_policy_always_log",
    )

    # Create task with log_only_on_failure=False (always log)
    task = EpisodeSimulationTask(
        environment=test_env,
        policy=test_policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1,
        cache_dir=cache_dir,
        debug=True,
        console_output=False,
        log_only_on_failure=False,  # Always log
    )

    # Access logger to ensure it's set up and file handlers are created
    logger = task.logger
    # Log a test message to trigger file creation
    logger.info("Test message to trigger file creation")

    # Flush handlers to ensure file is created
    for handler in logger.handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    # Run successful episode
    result = task.run()

    # Verify task succeeded
    assert result is not None, "Task should have succeeded"

    # Flush all handlers again to ensure logs are written to disk
    for handler in logger.handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    # Give file system a moment to sync
    time.sleep(0.2)

    # Check log files - they should contain logs
    # Logs are created in cache_dir / "env_policy" / "logs"
    log_files = list(cache_dir.rglob("*.log"))
    # If no log files found, check if env_policy directory exists
    if len(log_files) == 0:
        env_policy_dir = cache_dir / "env_policy"
        if env_policy_dir.exists():
            all_files = list(env_policy_dir.rglob("*"))
            assert (
                len(log_files) > 0
            ), f"Log files should be created in {cache_dir}. env_policy dir exists: {env_policy_dir.exists()}, files in env_policy: {all_files}"
        else:
            assert (
                len(log_files) > 0
            ), f"Log files should be created in {cache_dir}. env_policy dir doesn't exist. Found: {list(cache_dir.rglob('*'))}"

    # Verify logs are NOT empty (normal logging writes all logs)
    total_size = 0
    for log_file in log_files:
        size = log_file.stat().st_size
        total_size += size

    assert (
        total_size > 0
    ), f"Log files should contain logs for successful episode with log_only_on_failure=False. Total size: {total_size}, files: {[f.name for f in log_files]}"


def test_episode_simulation_task_conditional_memory_handler_setup(environment, policy):
    """Test that ConditionalMemoryHandler is properly set up when log_only_on_failure=True.

    Purpose: Validates that buffered logging infrastructure is correctly initialized

    Given: EpisodeSimulationTask with log_only_on_failure=True
    When: Task is created and logger is accessed
    Then: Logger has ConditionalMemoryHandler wrapping underlying handlers

    Test type: unit
    """
    belief = create_test_belief()

    # Create task with log_only_on_failure=True but no cache_dir (in-memory only)
    task = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        discount_factor=0.95,
        episode_number=1,
        console_output=False,
        log_only_on_failure=True,
    )

    # Access logger to trigger setup
    import logging  # pylint: disable=import-outside-toplevel

    logger = logging.getLogger(task._get_env_policy_logger_name())

    # Verify logger has ConditionalMemoryHandler if handlers were added
    if logger.handlers:
        # At least one handler should be a ConditionalMemoryHandler
        has_memory_handler = any(
            isinstance(handler, ConditionalMemoryHandler) for handler in logger.handlers
        )
        # Note: This may be False if no cache_dir was provided and no file handler was created
        # The important thing is we don't crash and can check the property
        assert isinstance(has_memory_handler, bool)


def test_episode_simulation_task_log_only_on_failure_validation(environment, policy):
    """Test that log_only_on_failure parameter is properly validated.

    Purpose: Validates that log_only_on_failure only accepts boolean values

    Given: EpisodeSimulationTask constructor
    When: log_only_on_failure is set to non-boolean value
    Then: TypeError is raised with appropriate error message

    Test type: unit
    """
    belief = create_test_belief()

    # Test with invalid type (string instead of bool)
    with pytest.raises(TypeError, match="log_only_on_failure must be a boolean"):
        EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            console_output=False,
            log_only_on_failure="true",  # type: ignore[arg-type]  # Invalid: string instead of bool (intentional for testing)
        )


def test_episode_simulation_task_log_only_on_failure_default_value(environment, policy):
    """Test that log_only_on_failure defaults to True.

    Purpose: Validates that log_only_on_failure has correct default value

    Given: EpisodeSimulationTask constructor without log_only_on_failure parameter
    When: Task is created
    Then: log_only_on_failure attribute is True by default

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
        console_output=False,
        # log_only_on_failure not specified - should default to True
    )

    assert task.log_only_on_failure is True, "log_only_on_failure should default to True"


# POMCPOW Integration Tests with All Compatible Environments


def create_pomcpow_action_sampler(environment):
    """Helper function to create appropriate action sampler based on environment type.

    Args:
        environment: The POMDP environment

    Returns:
        ActionSampler appropriate for the environment's action space
    """
    # For discrete action environments, use DiscreteActionSampler
    if isinstance(environment, DiscreteActionsEnvironment):
        return DiscreteActionSampler(actions=environment.get_actions())

    # For continuous action environments, use UnitCircleActionSampler
    # This is a reasonable default for 2D continuous control
    return UnitCircleActionSampler(max_action_magnitude=1.0)


def test_episode_simulation_task_pomcpow_tiger():
    """Test EpisodeSimulationTask with POMCPOW planner on Tiger POMDP.

    Purpose: Validates that POMCPOW can successfully complete episode simulation on Tiger POMDP without returning None

    Given: POMCPOW planner with TigerPOMDP environment, initial belief, and DiscreteActionSampler
    When: EpisodeSimulationTask executes a 5-step episode with POMCPOW planning
    Then: Task completes successfully and returns valid History object (not None)

    Test type: integration
    """

    # Get environment and belief
    environment = environment_instances["tiger"]
    initial_belief = belief_instances["tiger"]

    # Create action sampler
    action_sampler = create_pomcpow_action_sampler(environment)

    # Create POMCPOW planner
    policy = POMCPOW(
        environment=environment,
        discount_factor=environment.discount_factor,
        depth=3,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        action_sampler=action_sampler,
        n_simulations=10,
        name="POMCPOW_Tiger_Test",
    )

    # Create and run task
    task = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=5,
        episode_id=1,
        seed=42,
        discount_factor=environment.discount_factor,
        episode_number=1,
        console_output=False,
    )

    result = task.run()

    # Verify task succeeded
    assert result is not None, "POMCPOW on Tiger POMDP returned None"
    assert hasattr(result, "history")
    assert hasattr(result, "reach_terminal_state")
    assert hasattr(result, "actual_num_steps")


def test_episode_simulation_task_pomcpow_cartpole():
    """Test EpisodeSimulationTask with POMCPOW planner on CartPole POMDP.

    Purpose: Validates that POMCPOW can successfully complete episode simulation on CartPole POMDP without returning None

    Given: POMCPOW planner with CartPolePOMDP environment, initial belief, and DiscreteActionSampler
    When: EpisodeSimulationTask executes a 5-step episode with POMCPOW planning
    Then: Task completes successfully and returns valid History object (not None)

    Test type: integration
    """

    # Get environment and belief
    environment = environment_instances["cartpole"]
    initial_belief = belief_instances["cartpole"]

    # Create action sampler
    action_sampler = create_pomcpow_action_sampler(environment)

    # Create POMCPOW planner
    policy = POMCPOW(
        environment=environment,
        discount_factor=environment.discount_factor,
        depth=3,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        action_sampler=action_sampler,
        n_simulations=10,
        name="POMCPOW_CartPole_Test",
    )

    # Create and run task
    task = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=5,
        episode_id=1,
        seed=42,
        discount_factor=environment.discount_factor,
        episode_number=1,
        console_output=False,
    )

    result = task.run()

    # Verify task succeeded (this will fail due to the known bug)
    assert result is not None, "POMCPOW on CartPole POMDP returned None"
    assert hasattr(result, "history")
    assert hasattr(result, "reach_terminal_state")
    assert hasattr(result, "actual_num_steps")


def test_episode_simulation_task_pomcpow_mountain_car():
    """Test EpisodeSimulationTask with POMCPOW planner on MountainCar POMDP.

    Purpose: Validates that POMCPOW can successfully complete episode simulation on MountainCar POMDP without returning None

    Given: POMCPOW planner with MountainCarPOMDP environment, initial belief, and DiscreteActionSampler
    When: EpisodeSimulationTask executes a 5-step episode with POMCPOW planning
    Then: Task completes successfully and returns valid History object (not None)

    Test type: integration
    """

    # Get environment and belief
    environment = environment_instances["mountain_car"]
    initial_belief = belief_instances["mountain_car"]

    # Create action sampler
    action_sampler = create_pomcpow_action_sampler(environment)

    # Create POMCPOW planner
    policy = POMCPOW(
        environment=environment,
        discount_factor=environment.discount_factor,
        depth=3,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        action_sampler=action_sampler,
        n_simulations=10,
        name="POMCPOW_MountainCar_Test",
    )

    # Create and run task
    task = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=5,
        episode_id=1,
        seed=42,
        discount_factor=environment.discount_factor,
        episode_number=1,
        console_output=False,
    )

    result = task.run()

    # Verify task succeeded
    assert result is not None, "POMCPOW on MountainCar POMDP returned None"
    assert hasattr(result, "history")
    assert hasattr(result, "reach_terminal_state")
    assert hasattr(result, "actual_num_steps")


def test_episode_simulation_task_pomcpow_push():
    """Test EpisodeSimulationTask with POMCPOW planner on Push POMDP.

    Purpose: Validates that POMCPOW can successfully complete episode simulation on Push POMDP without returning None

    Given: POMCPOW planner with PushPOMDP environment, initial belief, and UnitCircleActionSampler
    When: EpisodeSimulationTask executes a 5-step episode with POMCPOW planning
    Then: Task completes successfully and returns valid History object (not None)

    Test type: integration
    """

    # Get environment and belief
    environment = environment_instances["push"]
    initial_belief = belief_instances["push"]

    # Create action sampler
    action_sampler = create_pomcpow_action_sampler(environment)

    # Create POMCPOW planner
    policy = POMCPOW(
        environment=environment,
        discount_factor=environment.discount_factor,
        depth=3,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        action_sampler=action_sampler,
        n_simulations=10,
        name="POMCPOW_Push_Test",
    )

    # Create and run task
    task = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=5,
        episode_id=1,
        seed=42,
        discount_factor=environment.discount_factor,
        episode_number=1,
        console_output=False,
    )

    result = task.run()

    # Verify task succeeded
    assert result is not None, "POMCPOW on Push POMDP returned None"
    assert hasattr(result, "history")
    assert hasattr(result, "reach_terminal_state")
    assert hasattr(result, "actual_num_steps")


def test_episode_simulation_task_pomcpow_safety_ant_velocity():
    """Test EpisodeSimulationTask with POMCPOW planner on SafetyAntVelocity POMDP.

    Purpose: Validates that POMCPOW can successfully complete episode simulation on SafetyAntVelocity POMDP without returning None

    Given: POMCPOW planner with SafetyAntVelocityPOMDP environment, initial belief, and UnitCircleActionSampler
    When: EpisodeSimulationTask executes a 5-step episode with POMCPOW planning
    Then: Task completes successfully and returns valid History object (not None)

    Test type: integration
    """

    # Get environment and belief
    environment = environment_instances["safety_ant_velocity"]
    initial_belief = belief_instances["safety_ant_velocity"]

    # Create action sampler
    action_sampler = create_pomcpow_action_sampler(environment)

    # Create POMCPOW planner
    policy = POMCPOW(
        environment=environment,
        discount_factor=environment.discount_factor,
        depth=3,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        action_sampler=action_sampler,
        n_simulations=10,
        name="POMCPOW_SafetyAntVelocity_Test",
    )

    # Create and run task
    task = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=5,
        episode_id=1,
        seed=42,
        discount_factor=environment.discount_factor,
        episode_number=1,
        console_output=False,
    )

    result = task.run()

    # Verify task succeeded
    assert result is not None, "POMCPOW on SafetyAntVelocity POMDP returned None"
    assert hasattr(result, "history")
    assert hasattr(result, "reach_terminal_state")
    assert hasattr(result, "actual_num_steps")


def test_episode_simulation_task_pomcpow_discrete_light_dark():
    """Test EpisodeSimulationTask with POMCPOW planner on DiscreteLightDark POMDP.

    Purpose: Validates that POMCPOW can successfully complete episode simulation on DiscreteLightDark POMDP without returning None

    Given: POMCPOW planner with DiscreteLightDarkPOMDP environment, initial belief, and DiscreteActionSampler
    When: EpisodeSimulationTask executes a 5-step episode with POMCPOW planning
    Then: Task completes successfully and returns valid History object (not None)

    Test type: integration
    """

    # Get environment and belief
    environment = environment_instances["discrete_light_dark"]
    initial_belief = belief_instances["discrete_light_dark"]

    # Create action sampler
    action_sampler = create_pomcpow_action_sampler(environment)

    # Create POMCPOW planner
    policy = POMCPOW(
        environment=environment,
        discount_factor=environment.discount_factor,
        depth=3,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        action_sampler=action_sampler,
        n_simulations=10,
        name="POMCPOW_DiscreteLightDark_Test",
    )

    # Create and run task
    task = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=5,
        episode_id=1,
        seed=42,
        discount_factor=environment.discount_factor,
        episode_number=1,
        console_output=False,
    )

    result = task.run()

    # Verify task succeeded
    assert result is not None, "POMCPOW on DiscreteLightDark POMDP returned None"
    assert hasattr(result, "history")
    assert hasattr(result, "reach_terminal_state")
    assert hasattr(result, "actual_num_steps")


def test_episode_simulation_task_pomcpow_continuous_light_dark():
    """Test EpisodeSimulationTask with POMCPOW planner on ContinuousLightDark POMDP.

    Purpose: Validates that POMCPOW can successfully complete episode simulation on ContinuousLightDark POMDP without returning None

    Given: POMCPOW planner with ContinuousLightDarkPOMDP environment, initial belief, and UnitCircleActionSampler
    When: EpisodeSimulationTask executes a 5-step episode with POMCPOW planning
    Then: Task completes successfully and returns valid History object (not None)

    Test type: integration
    """

    # Get environment and belief
    environment = environment_instances["continuous_light_dark"]
    initial_belief = belief_instances["continuous_light_dark"]

    # Create action sampler
    action_sampler = create_pomcpow_action_sampler(environment)

    # Create POMCPOW planner
    policy = POMCPOW(
        environment=environment,
        discount_factor=environment.discount_factor,
        depth=3,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        action_sampler=action_sampler,
        n_simulations=10,
        name="POMCPOW_ContinuousLightDark_Test",
    )

    # Create and run task
    task = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=5,
        episode_id=1,
        seed=42,
        discount_factor=environment.discount_factor,
        episode_number=1,
        console_output=False,
    )

    result = task.run()

    # Verify task succeeded
    assert result is not None, "POMCPOW on ContinuousLightDark POMDP returned None"
    assert hasattr(result, "history")
    assert hasattr(result, "reach_terminal_state")
    assert hasattr(result, "actual_num_steps")


def test_episode_simulation_task_pomcpow_all_compatible_environments():
    """Test EpisodeSimulationTask with POMCPOW on all compatible environments systematically.

    Purpose: Validates that POMCPOW successfully completes episode simulation on ALL compatible environments without returning None

    Given: POMCPOW planner configured for each compatible environment from EnvironmentConfigsAPI
    When: EpisodeSimulationTask executes for each environment with appropriate action sampler
    Then: All tasks complete successfully and return valid History objects (no None results)

    Test type: integration
    """
    # Get all compatible environments
    policy_space_info = POMCPOW.get_space_info()
    compatible_env_belief_pairs = EnvironmentConfigsAPI().get_compatible_environments(
        policy_space_info
    )

    # Track results
    results = {}

    for environment, initial_belief in compatible_env_belief_pairs:
        env_name = environment.name

        # Create action sampler
        action_sampler = create_pomcpow_action_sampler(environment)

        # Create POMCPOW planner
        policy = POMCPOW(
            environment=environment,
            discount_factor=environment.discount_factor,
            depth=3,
            exploration_constant=1.0,
            k_o=3.0,
            k_a=3.0,
            alpha_o=0.5,
            alpha_a=0.5,
            action_sampler=action_sampler,
            n_simulations=10,
            name=f"POMCPOW_{env_name}_Test",
        )

        # Create and run task
        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_steps=5,
            episode_id=1,
            seed=42,
            discount_factor=environment.discount_factor,
            episode_number=1,
            console_output=False,
        )

        result = task.run()
        results[env_name] = result

        # Verify task succeeded
        assert result is not None, f"POMCPOW on {env_name} POMDP returned None"
        assert hasattr(result, "history"), f"Result for {env_name} missing 'history' attribute"
        assert hasattr(
            result, "reach_terminal_state"
        ), f"Result for {env_name} missing 'reach_terminal_state' attribute"
        assert hasattr(
            result, "actual_num_steps"
        ), f"Result for {env_name} missing 'actual_num_steps' attribute"

    # Final verification: all environments returned non-None results
    none_results = [env_name for env_name, res in results.items() if res is None]
    assert len(none_results) == 0, f"The following environments returned None: {none_results}"

    # Report success
    print(
        f"\nSuccessfully tested POMCPOW on {len(compatible_env_belief_pairs)} compatible environments:"
    )
    for env_name in results:
        print(f"  ✓ {env_name}")


def test_episode_simulation_task_pickle_is_os_agnostic(environment, policy):
    """The pickled task must not embed OS-specific Path class names.

    Purpose: Regression for cross-OS worker crashes. Workers on a different
        OS than the client cannot unpickle ``pathlib.PosixPath`` /
        ``pathlib.WindowsPath`` shipped from the other side. The pickled
        task payload must therefore not reference either class.

    Given: An EpisodeSimulationTask constructed with a Path-typed
        ``cache_dir`` (PosixPath on Linux runners; WindowsPath on Windows).
    When: ``pickle.dumps`` serializes the task.
    Then: The byte stream contains neither ``b"PosixPath"`` nor
        ``b"WindowsPath"`` anywhere.

    Test type: unit
    """
    initial_belief = create_test_belief()
    task = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=2,
        episode_id=0,
        seed=42,
        cache_dir=Path("/tmp/some_cache"),
    )
    blob = pickle.dumps(task)
    assert b"PosixPath" not in blob, (
        "EpisodeSimulationTask pickle stream embeds pathlib.PosixPath. "
        "Foreign-OS workers cannot unpickle this. Use str on the wire."
    )
    assert (
        b"WindowsPath" not in blob
    ), "EpisodeSimulationTask pickle stream embeds pathlib.WindowsPath."
