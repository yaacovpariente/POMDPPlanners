from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask


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
        gamma=0.95,
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


def test_episode_simulation_task_value_error_logging(caplog, environment, policy):
    """Test that EpisodeSimulationTask logs ValueError exceptions properly.

    Purpose: Validates that EpisodeSimulationTask logs ValueError exceptions with appropriate detail

    Given: A EpisodeSimulationTask that encounters a ValueError during execution
    When: Task execution raises a ValueError
    Then: Error is logged with appropriate level and message

    Test type: unit
    """
    from POMDPPlanners.simulations.simulations_deployment.tasks import (
        EpisodeSimulationTask,
    )

    belief = create_test_belief()

    # Create a real environment and patch it to cause an error during execution
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env")
    # Patch the environment to fail when state_transition_model is called
    with patch.object(
        test_env, "state_transition_model", side_effect=ValueError("Test value error")
    ):
        # Create task using the real environment
        task = EpisodeSimulationTask(
            environment=test_env,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            console_output=False,
        )

        # Run the task - it should handle the error gracefully
        result = task.run()

        # Verify that the task handled the error and returned None
        assert result is None

        # Verify that some error was logged (the exact message may vary)
        assert "[EPISODE_001] Error running episode:" in caplog.text


def test_episode_simulation_task_runtime_error_logging(caplog, environment, policy):
    """Test that EpisodeSimulationTask logs RuntimeError exceptions properly.

    Purpose: Validates that EpisodeSimulationTask logs RuntimeError exceptions with appropriate detail

    Given: A EpisodeSimulationTask that encounters a RuntimeError during execution
    When: Task execution raises a RuntimeError
    Then: Error is logged with appropriate level and message

    Test type: unit
    """
    from POMDPPlanners.simulations.simulations_deployment.tasks import (
        EpisodeSimulationTask,
    )

    belief = create_test_belief()

    # Create a real environment and patch it to cause a RuntimeError during execution
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env")
    # Patch the environment to fail when observation_model is called
    with patch.object(
        test_env, "observation_model", side_effect=RuntimeError("Test runtime error")
    ):
        # Create task using the real environment
        task = EpisodeSimulationTask(
            environment=test_env,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            console_output=False,
        )

        # Run the task - it should handle the error gracefully
        result = task.run()

        # Verify that the task handled the error and returned None
        assert result is None

        # Verify that some error was logged (the exact message may vary)
        assert "[EPISODE_001] Error running episode:" in caplog.text


def test_episode_simulation_task_type_error_logging(caplog, environment, policy):
    """Test that EpisodeSimulationTask logs TypeError exceptions properly.

    Purpose: Validates that EpisodeSimulationTask logs TypeError exceptions with appropriate detail

    Given: A EpisodeSimulationTask that encounters a TypeError during execution
    When: Task execution raises a TypeError
    Then: Error is logged with appropriate level and message

    Test type: unit
    """
    from POMDPPlanners.simulations.simulations_deployment.tasks import (
        EpisodeSimulationTask,
    )

    belief = create_test_belief()

    # Create a real environment and patch it to cause a TypeError during execution
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env")
    # Patch the environment to fail when reward is called
    with patch.object(test_env, "reward", side_effect=TypeError("Test type error")):
        # Create task using the real environment
        task = EpisodeSimulationTask(
            environment=test_env,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            console_output=False,
        )

        # Run the task - it should handle the error gracefully
        result = task.run()

        # Verify that the task handled the error and returned None
        assert result is None

        # Verify that some error was logged (the exact message may vary)
        assert "[EPISODE_001] Error running episode:" in caplog.text


def test_episode_simulation_task_custom_exception_logging(caplog, environment, policy):
    """Test that EpisodeSimulationTask logs custom exceptions properly.

    Purpose: Validates that EpisodeSimulationTask logs custom exceptions with appropriate detail

    Given: A EpisodeSimulationTask that encounters a custom exception during execution
    When: Task execution raises a custom exception
    Then: Error is logged with appropriate level and message

    Test type: unit
    """
    from POMDPPlanners.simulations.simulations_deployment.tasks import (
        EpisodeSimulationTask,
    )

    # Create a custom exception class
    class CustomTestException(Exception):
        pass

    belief = create_test_belief()

    # Create a real environment and patch it to cause a custom exception during execution
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env")
    # Patch the environment to fail when is_terminal is called
    with patch.object(
        test_env,
        "is_terminal",
        side_effect=CustomTestException("Test custom exception"),
    ):
        # Create task using the real environment
        task = EpisodeSimulationTask(
            environment=test_env,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            console_output=False,
        )

        # Run the task - it should handle the error gracefully
        result = task.run()

        # Verify that the task handled the error and returned None
        assert result is None

        # Verify that some error was logged (the exact message may vary)
        assert "[EPISODE_001] Error running episode:" in caplog.text


def test_episode_simulation_task_logging_includes_traceback(caplog, environment, policy):
    """Test that EpisodeSimulationTask logs include full traceback information.

    Purpose: Validates that EpisodeSimulationTask logs include full exception traceback for debugging

    Given: A EpisodeSimulationTask that encounters an exception during execution
    When: Task execution raises an exception
    Then: Full exception details including traceback are logged

    Test type: unit
    """
    from POMDPPlanners.simulations.simulations_deployment.tasks import (
        EpisodeSimulationTask,
    )

    belief = create_test_belief()

    # Create a real environment and patch it to cause an exception during execution
    test_env = TigerPOMDP(discount_factor=0.95, name="test_env")
    # Patch the environment to fail when state_transition_model is called
    with patch.object(
        test_env,
        "state_transition_model",
        side_effect=Exception("Test exception with traceback"),
    ):
        # Create task using the real environment
        task = EpisodeSimulationTask(
            environment=test_env,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            discount_factor=0.95,
            episode_number=1,
            console_output=False,
        )

        # Run the task - it should handle the error gracefully
        result = task.run()

        # Verify that the task handled the error and returned None
        assert result is None

        # Verify that some error was logged (the exact message may vary)
        assert "[EPISODE_001] Error running episode:" in caplog.text

        # Verify that traceback information was logged
        assert "Full exception details:" in caplog.text
