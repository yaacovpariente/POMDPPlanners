"""Tests for simulation functionality.

This module tests the simulation functionality, focusing on:
- Basic simulation operations
- Episode simulation
- History tracking
- Metrics computation
"""

import random
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import History, StepData, TaskManagerExternalDB
from POMDPPlanners.core.simulation.tasks import SimulationTask, DataBaseInterface

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


def create_test_belief():
    """Helper function to create a valid belief state for testing."""
    particles = ["tiger_left", "tiger_right"]
    # Use equal weights (log(0.5) for each particle)
    log_weights = np.array([np.log(0.5), np.log(0.5)])
    return WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)


def test_history_equality():
    """Test History class equality comparison.

    Purpose: Validates equality comparison for history

    Given: Objects with same or different configurations
    When: Equality comparison is performed
    Then: Objects are correctly identified as equal or unequal

    Test type: unit
    """
    # Create test data
    step_data1 = StepData(
        state="state1",
        action="action1",
        next_state="next_state1",
        observation="obs1",
        reward=1.0,
        belief=create_test_belief(),
    )

    step_data2 = StepData(
        state="state2",
        action="action2",
        next_state="next_state2",
        observation="obs2",
        reward=2.0,
        belief=create_test_belief(),
    )

    # Create identical histories
    history1 = History(
        history=[step_data1, step_data2],
        discount_factor=0.95,
        average_state_sampling_time=0.1,
        average_action_time=0.2,
        average_observation_time=0.3,
        average_belief_update_time=0.4,
        average_reward_time=0.5,
        actual_num_steps=2,
        reach_terminal_state=True,
        policy_run_data=[],
    )

    history2 = History(
        history=[step_data1, step_data2],
        discount_factor=0.95,
        average_state_sampling_time=0.1,
        average_action_time=0.2,
        average_observation_time=0.3,
        average_belief_update_time=0.4,
        average_reward_time=0.5,
        actual_num_steps=2,
        reach_terminal_state=True,
        policy_run_data=[],
    )

    # Create different history
    history3 = History(
        history=[step_data2, step_data1],  # Different order
        discount_factor=0.95,
        average_state_sampling_time=0.1,
        average_action_time=0.2,
        average_observation_time=0.3,
        average_belief_update_time=0.4,
        average_reward_time=0.5,
        actual_num_steps=2,
        reach_terminal_state=True,
        policy_run_data=[],
    )

    # Test equality
    assert history1 == history2, "Identical histories should be equal"
    assert history1 != history3, "Different histories should not be equal"

    # Test comparison with non-History object
    assert history1 != "not a history", "History should not equal non-History object"

    # Test with different field values
    history4 = History(
        history=[step_data1, step_data2],
        discount_factor=0.90,  # Different discount factor
        average_state_sampling_time=0.1,
        average_action_time=0.2,
        average_observation_time=0.3,
        average_belief_update_time=0.4,
        average_reward_time=0.5,
        actual_num_steps=2,
        reach_terminal_state=True,
        policy_run_data=[],
    )
    assert history1 != history4, "Histories with different discount factors should not be equal"

    # Test with different history length
    history5 = History(
        history=[step_data1],  # Different length
        discount_factor=0.95,
        average_state_sampling_time=0.1,
        average_action_time=0.2,
        average_observation_time=0.3,
        average_belief_update_time=0.4,
        average_reward_time=0.5,
        actual_num_steps=1,  # Different num_steps
        reach_terminal_state=True,
        policy_run_data=[],
    )
    assert history1 != history5, "Histories with different lengths should not be equal"


def test_history_serialization():
    """Test History serialization and deserialization.

    Purpose: Validates that History objects can be serialized to dictionaries and deserialized back to equivalent objects

    Given: History object with StepData, timing attributes, and configuration parameters
    When: to_dict() and from_dict() methods are used for serialization and deserialization
    Then: Serialized dictionary contains all key fields and deserialized History equals original object

    Test type: unit
    """
    # Create test data
    step_data = StepData(
        state="state1",
        action="action1",
        next_state="next_state1",
        observation="obs1",
        reward=1.0,
        belief=create_test_belief(),
    )

    history = History(
        history=[step_data],
        discount_factor=0.95,
        average_state_sampling_time=0.1,
        average_action_time=0.2,
        average_observation_time=0.3,
        average_belief_update_time=0.4,
        average_reward_time=0.5,
        actual_num_steps=1,
        reach_terminal_state=True,
        policy_run_data=[],
    )

    # Test serialization
    history_dict = history.to_dict()
    assert isinstance(history_dict, dict)
    assert history_dict["discount_factor"] == 0.95
    assert history_dict["actual_num_steps"] == 1
    assert history_dict["reach_terminal_state"] is True
    assert len(history_dict["history"]) == 1

    # Test deserialization
    reconstructed_history = History.from_dict(history_dict)
    assert reconstructed_history == history


class MockSimulationTask(SimulationTask):
    def __init__(self, config_id: str, should_succeed: bool = True):
        self._config_id = config_id
        self._should_succeed = should_succeed

    def run(self) -> Any:
        if self._should_succeed:
            return {"result": "success"}
        return None

    def get_config_id(self) -> str:
        return self._config_id


class MockDatabase(DataBaseInterface):
    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._cache[key]

    def is_key_in_cache(self, key: str) -> bool:
        return key in self._cache

    def set(self, key: str, value: Any):
        self._cache[key] = value

    def clear(self):
        self._cache.clear()


class TestTaskManagerExternalDB(TaskManagerExternalDB):
    def _run_tasks(self, tasks: List[SimulationTask]) -> List[Any]:
        return [task.run() for task in tasks]


def test_task_manager_external_db():
    """Test TaskManagerExternalDB with successful and failed tasks.

    Purpose: Validates that TaskManagerExternalDB correctly handles mixed success/failure scenarios and caching behavior

    Given: MockDatabase, TestTaskManagerExternalDB, and 4 MockSimulationTasks (3 successful, 1 failed) with identifiers
    When: run_tasks is called with mixed success/failure tasks
    Then: Only successful tasks (3) are returned and cached, failed tasks are excluded, subsequent runs use cache

    Test type: unit
    """
    # Create mock database and task manager
    mock_db = MockDatabase()
    task_manager = TestTaskManagerExternalDB(mock_db)

    # Create test tasks
    tasks = [
        MockSimulationTask("task1", should_succeed=True),
        MockSimulationTask("task2", should_succeed=False),
        MockSimulationTask("task3", should_succeed=True),
        MockSimulationTask("task4", should_succeed=True),
    ]
    task_identifiers = ["id1", "id2", "id3", "id4"]

    # Test running tasks
    results, successful_ids = task_manager.run_tasks(
        cast(List[SimulationTask], tasks), task_identifiers
    )

    # Verify results
    assert len(results) == 3  # Only successful tasks
    assert len(successful_ids) == 3  # Only successful task identifiers
    assert all(result is not None for result in results)
    assert "id1" in successful_ids
    assert "id3" in successful_ids
    assert "id4" in successful_ids
    assert "id2" not in successful_ids  # Failed task should not be in successful_ids

    # Verify caching - only successful tasks should be cached
    assert mock_db.is_key_in_cache("task1")
    assert not mock_db.is_key_in_cache("task2")  # Failed task should not be cached
    assert mock_db.is_key_in_cache("task3")
    assert mock_db.is_key_in_cache("task4")

    # Test running same tasks again (should use cache)
    results2, successful_ids2 = task_manager.run_tasks(
        cast(List[SimulationTask], tasks), task_identifiers
    )
    assert results2 == results
    assert successful_ids2 == successful_ids


def test_task_manager_external_db_all_failed():
    """Test TaskManagerExternalDB when all tasks fail.

    Purpose: Validates that TaskManagerExternalDB correctly handles edge case where all tasks fail

    Given: MockDatabase, TestTaskManagerExternalDB, and 2 MockSimulationTasks that both fail (should_succeed=False)
    When: run_tasks is called with all failing tasks
    Then: Empty results and successful_ids lists are returned (no tasks succeed or get cached)

    Test type: unit
    """
    mock_db = MockDatabase()
    task_manager = TestTaskManagerExternalDB(mock_db)

    # Create all failing tasks
    tasks = [
        MockSimulationTask("task1", should_succeed=False),
        MockSimulationTask("task2", should_succeed=False),
    ]
    task_identifiers = ["id1", "id2"]

    # Test running tasks
    results, successful_ids = task_manager.run_tasks(
        cast(List[SimulationTask], tasks), task_identifiers
    )

    # Verify results
    assert len(results) == 0
    assert len(successful_ids) == 0


def test_task_manager_external_db_all_cached():
    """Test TaskManagerExternalDB when all tasks are cached.

    Purpose: Validates that TaskManagerExternalDB correctly retrieves all results from cache when tasks are pre-cached

    Given: MockDatabase with pre-cached results, TestTaskManagerExternalDB, and 2 MockSimulationTasks with cached entries
    When: run_tasks is called with tasks that have pre-existing cache entries
    Then: All cached results (2) are returned with correct identifiers without re-executing tasks

    Test type: unit
    """
    mock_db = MockDatabase()
    task_manager = TestTaskManagerExternalDB(mock_db)

    # Create tasks and cache their results
    tasks = [
        MockSimulationTask("task1", should_succeed=True),
        MockSimulationTask("task2", should_succeed=True),
    ]
    task_identifiers = ["id1", "id2"]

    # Pre-cache results
    mock_db.set("task1", {"result": "success"})
    mock_db.set("task2", {"result": "success"})

    # Test running tasks
    results, successful_ids = task_manager.run_tasks(
        cast(List[SimulationTask], tasks), task_identifiers
    )

    # Verify results
    assert len(results) == 2
    assert len(successful_ids) == 2
    assert "id1" in successful_ids
    assert "id2" in successful_ids
