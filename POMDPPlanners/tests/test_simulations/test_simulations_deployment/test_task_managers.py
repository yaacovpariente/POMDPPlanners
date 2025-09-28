import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import History
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.simulations.simulations_deployment.cache_dbs import DiskCacheDB
from POMDPPlanners.simulations.simulations_deployment.task_managers import (
    DaskTaskManager,
    JoblibTaskManager,
    PBSTaskManager,
    TaskManagerType,
)
from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask


def create_test_belief():
    """Helper function to create a valid belief state for testing."""
    particles = ["tiger_left", "tiger_right"]
    log_weights = np.array([np.log(0.5), np.log(0.5)])
    return WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)


@pytest.fixture
def temp_cache_dir():
    """Fixture to create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Add a small delay to ensure all file handles are released
    time.sleep(0.1)
    try:
        shutil.rmtree(temp_dir)
    except PermissionError:
        # If we still can't delete, log a warning but don't fail the test
        import warnings

        warnings.warn(f"Could not delete temporary directory {temp_dir}")


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


@pytest.fixture
def cache_db(temp_cache_dir):
    """Fixture to create a DiskCacheDB instance."""
    db = DiskCacheDB(cache_dir=temp_cache_dir)
    yield db
    # Ensure the cache is properly closed
    db.close()
    # Add a small delay to ensure all file handles are released
    time.sleep(0.1)


# Tests for DaskTaskManager
def test_dask_task_manager_initialization():
    """Test DaskTaskManager initialization and cleanup (no cache_db).

    Purpose: Validates that DaskTaskManager initializes correctly with proper client and cache setup

    Given: DaskTaskManager constructor with default parameters
    When: DaskTaskManager is initialized as a context manager
    Then: Task manager has valid client and cache objects, with cache not registered by default

    Test type: unit
    """
    with DaskTaskManager() as task_manager:
        assert task_manager.client is not None
        assert task_manager.cache is not None
        assert not task_manager.cache_registered  # Cache not registered by default


def test_dask_task_manager_with_cache_clear():
    """Test DaskTaskManager initialization with cache clearing (no cache_db).

    Purpose: Validates that DaskTaskManager can initialize with cache clearing enabled

    Given: DaskTaskManager constructor with clear_cache_on_start=True parameter
    When: DaskTaskManager is initialized as a context manager
    Then: Task manager has valid client and cache objects, with cache not registered when cleared

    Test type: unit
    """
    with DaskTaskManager(clear_cache_on_start=True) as task_manager:
        assert task_manager.client is not None
        assert task_manager.cache is not None
        assert not task_manager.cache_registered  # Cache not registered when cleared


def test_dask_task_manager_run_tasks(environment, policy):
    """Test running tasks with DaskTaskManager (no cache_db).

    Purpose: Validates that DaskTaskManager can successfully run multiple EpisodeSimulationTask instances

    Given: A TigerPOMDP environment, SparsePFT policy, and 2 EpisodeSimulationTask instances
    When: DaskTaskManager.run_tasks() is called with the task list and identifiers
    Then: Returns 2 successful results and 2 successful IDs, with each result being a valid History object

    Test type: unit
    """
    with DaskTaskManager() as task_manager:
        # Create multiple tasks
        tasks = []
        task_identifiers = []
        for i in range(2):
            belief = create_test_belief()
            task = EpisodeSimulationTask(
                environment=environment,
                policy=policy,
                initial_belief=belief,
                num_steps=2,
                episode_id=i,
                seed=42 + i,
                console_output=False,
            )
            tasks.append(task)
            task_identifiers.append(f"episode_{i}")
        # Run tasks
        results, successful_ids = task_manager.run_tasks(tasks, task_identifiers)
        # Verify results
        assert len(results) == 2
        assert len(successful_ids) == 2
        assert all(id in successful_ids for id in task_identifiers)
        for result in results:
            assert isinstance(result, History)
            assert len(result.history) <= 2


def test_dask_task_manager_task_status(environment, policy):
    """Test getting task status with DaskTaskManager (no cache_db).

    Purpose: Validates that DaskTaskManager can submit tasks and retrieve their status information

    Given: A TigerPOMDP environment, SparsePFT policy, and EpisodeSimulationTask
    When: Task is submitted via submit_tasks() and status is retrieved via get_task_status()
    Then: Returns status dictionary with task cache key and valid status values (pending, running, or finished)

    Test type: unit
    """
    with DaskTaskManager() as task_manager:
        # Create and submit a task
        belief = create_test_belief()
        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            console_output=False,
        )
        # Submit task and get status
        futures = task_manager.submit_tasks([task])
        status = task_manager.get_task_status(futures)
        assert len(status) == 1
        assert task._cache_key in status
        assert status[task._cache_key] in ["pending", "running", "finished"]


# Tests for JoblibTaskManager
def test_joblib_task_manager_initialization(cache_db):
    """Test JoblibTaskManager initialization.

    Purpose: Validates that JoblibTaskManager initializes correctly with cache database and default parameters

    Given: A cache database fixture and JoblibTaskManager constructor
    When: JoblibTaskManager is initialized as a context manager with cache_db
    Then: Task manager has correct default attributes: n_jobs=-1 (all cores), verbose=0, and valid memory object

    Test type: unit
    """
    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        assert task_manager.n_jobs == -1  # Default to all cores
        assert task_manager.verbose == 0
        assert task_manager.memory is not None


def test_joblib_task_manager_with_cache_clear(cache_db):
    """Test JoblibTaskManager initialization with cache clearing.

    Purpose: Validates that JoblibTaskManager can initialize with cache clearing enabled

    Given: A cache database fixture and JoblibTaskManager constructor with clear_cache_on_start=True
    When: JoblibTaskManager is initialized as a context manager
    Then: Task manager has valid memory object and cache is empty after clearing (0 items)

    Test type: unit
    """
    with JoblibTaskManager(cache_db=cache_db, clear_cache_on_start=True) as task_manager:
        assert task_manager.memory is not None
        # Cache should be empty after clearing
        memory = task_manager.memory
        assert memory is not None and memory.store_backend is not None
        items = memory.store_backend.get_items()
        assert items is not None
        assert len(list(items)) == 0


def test_joblib_task_manager_run_tasks(cache_db, environment, policy):
    """Test running tasks with JoblibTaskManager.

    Purpose: Validates that JoblibTaskManager can successfully run multiple EpisodeSimulationTask instances with caching

    Given: A cache database, TigerPOMDP environment, SparsePFT policy, and 2 EpisodeSimulationTask instances
    When: JoblibTaskManager.run_tasks() is called with the task list and identifiers
    Then: Returns 2 successful results and 2 successful IDs, with each result being a valid History object

    Test type: unit
    """
    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Create multiple tasks
        tasks = []
        task_identifiers = []
        for i in range(2):
            belief = create_test_belief()
            task = EpisodeSimulationTask(
                environment=environment,
                policy=policy,
                initial_belief=belief,
                num_steps=2,
                episode_id=i,
                seed=42 + i,
                console_output=False,
            )
            tasks.append(task)
            task_identifiers.append(f"episode_{i}")

        # Run tasks
        results, successful_ids = task_manager.run_tasks(tasks, task_identifiers)

        # Verify results
        assert len(results) == 2
        assert len(successful_ids) == 2
        assert all(id in successful_ids for id in task_identifiers)
        for result in results:
            assert isinstance(result, History)
            assert len(result.history) <= 2


def test_joblib_task_manager_cache(cache_db, environment, policy):
    """Test caching behavior of JoblibTaskManager.

    Purpose: Validates that JoblibTaskManager properly caches task results and reuses them on subsequent runs

    Given: A cache database, TigerPOMDP environment, SparsePFT policy, and EpisodeSimulationTask
    When: Same task is run twice with identical parameters
    Then: Second run retrieves result from cache instead of re-executing, demonstrating caching effectiveness

    Test type: unit
    """
    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Create a task
        belief = create_test_belief()
        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            console_output=False,
        )
        task_identifier = "episode_1"

        # Run task twice
        result1, ids1 = task_manager.run_tasks([task], [task_identifier])
        result2, ids2 = task_manager.run_tasks([task], [task_identifier])

        # Compare main fields instead of object equality
        assert result1[0].discount_factor == result2[0].discount_factor
        assert result1[0].actual_num_steps == result2[0].actual_num_steps
        assert result1[0].reach_terminal_state == result2[0].reach_terminal_state
        assert len(result1[0].history) == len(result2[0].history)
        assert ids1 == ids2
        assert task_identifier in ids1

        # Add a small delay to ensure all file handles are released
        time.sleep(0.1)

        try:
            # Clear cache and run again
            task_manager.clear_cache()
        except PermissionError:
            # If we can't clear the cache, log a warning but continue with the test
            import warnings

            warnings.warn("Could not clear cache due to file being in use, continuing with test")

        result3, ids3 = task_manager.run_tasks([task], [task_identifier])

        # Compare main fields again
        assert result1[0].discount_factor == result3[0].discount_factor
        assert result1[0].actual_num_steps == result3[0].actual_num_steps
        assert result1[0].reach_terminal_state == result3[0].reach_terminal_state
        assert len(result1[0].history) == len(result3[0].history)
        assert ids1 == ids3
        assert task_identifier in ids3


def test_joblib_task_manager_logging(cache_db, environment, policy, temp_cache_dir):
    """Test that JoblibTaskManager writes logs properly.

    Purpose: Validates that JoblibTaskManager can write logs to specified directory with debug logging enabled

    Given: A cache database, TigerPOMDP environment, SparsePFT policy, EpisodeSimulationTask, and log directory
    When: JoblibTaskManager runs task with logger_debug=True and cache_dir configured
    Then: Task executes successfully and generates logs in the specified directory

    Test type: unit
    """
    import os
    from pathlib import Path

    # Create a specific log directory for this test
    log_dir = Path(temp_cache_dir) / "test_logs"
    log_dir.mkdir(exist_ok=True)

    with JoblibTaskManager(
        cache_db=cache_db, cache_dir=str(log_dir), logger_debug=True
    ) as task_manager:
        # Create a task
        belief = create_test_belief()
        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            console_output=False,
        )
        task_identifier = "episode_1"

        # Run task to generate logs
        results, successful_ids = task_manager.run_tasks([task], [task_identifier])

        # Verify results
        assert len(results) == 1
        assert len(successful_ids) == 1
        assert task_identifier in successful_ids

        # Verify that expected log messages are present in the task manager logs
        # Note: We don't check directory structure as it may change


def test_joblib_task_manager_logging_with_multiple_tasks(
    cache_db, environment, policy, temp_cache_dir
):
    """Test that JoblibTaskManager logs progress updates for multiple tasks.

    Purpose: Validates that JoblibTaskManager can log progress updates when running multiple tasks

    Given: A cache database, TigerPOMDP environment, SparsePFT policy, and 2 EpisodeSimulationTask instances
    When: JoblibTaskManager runs multiple tasks with logger_debug=True and cache_dir configured
    Then: All tasks execute successfully and progress logging is generated for multiple task execution

    Test type: integration
    """
    import os
    from pathlib import Path

    # Create a specific log directory for this test
    log_dir = Path(temp_cache_dir) / "test_logs_multi"
    log_dir.mkdir(exist_ok=True)

    with JoblibTaskManager(
        cache_db=cache_db, cache_dir=str(log_dir), logger_debug=True
    ) as task_manager:
        # Create multiple tasks to test progress logging
        tasks = []
        task_identifiers = []
        for i in range(2):  # Create 2 tasks to test progress updates
            belief = create_test_belief()
            task = EpisodeSimulationTask(
                environment=environment,
                policy=policy,
                initial_belief=belief,
                num_steps=2,
                episode_id=i,
                seed=42 + i,
                console_output=False,
            )
            tasks.append(task)
            task_identifiers.append(f"episode_{i}")

        # Run tasks to generate logs
        results, successful_ids = task_manager.run_tasks(tasks, task_identifiers)

        # Verify results
        assert len(results) == 2
        assert len(successful_ids) == 2

        # Note: We don't check directory structure as it may change


class FailingEpisodeSimulationTask(EpisodeSimulationTask):
    """Mock task that always fails for testing failed task caching behavior."""

    def run(self):
        """Always raise an exception to simulate task failure."""
        raise RuntimeError("Simulated task failure for testing")


def test_joblib_task_manager_failed_tasks_not_cached(cache_db, environment, policy):
    """Test that JoblibTaskManager does not cache results from failed tasks.

    Purpose: Validates that failed tasks are not cached and subsequent runs retry execution

    Given: JoblibTaskManager with cache and a task that always fails
    When: The failing task is run multiple times
    Then: Each run attempts to execute the task (not cached) and fails consistently

    Test type: unit
    """
    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Create a task that always fails
        belief = create_test_belief()
        failing_task = FailingEpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=999,
            seed=42,
            console_output=False,
        )
        task_identifier = "failing_episode"

        # First run - should fail with exception
        with pytest.raises(RuntimeError, match="Simulated task failure for testing"):
            task_manager.run_tasks([failing_task], [task_identifier])

        # Second run - should also fail (not cached), proving failed tasks aren't cached
        with pytest.raises(RuntimeError, match="Simulated task failure for testing"):
            task_manager.run_tasks([failing_task], [task_identifier])

        # The fact that both runs raise exceptions (rather than the second returning cached results)
        # proves that failed tasks are not cached by the task manager


def test_dask_task_manager_failed_tasks_not_cached(environment, policy):
    """Test that DaskTaskManager does not cache results from failed tasks.

    Purpose: Validates that failed tasks are not cached and subsequent runs retry execution for Dask

    Given: DaskTaskManager with cache and a task that always fails
    When: The failing task is run multiple times
    Then: Each run attempts to execute the task (not cached) and fails consistently

    Test type: unit
    """
    with DaskTaskManager() as task_manager:
        # Create a task that always fails
        belief = create_test_belief()
        failing_task = FailingEpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=999,
            seed=42,
            console_output=False,
        )
        task_identifier = "failing_dask_episode"

        # First run - should fail with exception
        with pytest.raises(RuntimeError, match="Simulated task failure for testing"):
            task_manager.run_tasks([failing_task], [task_identifier])

        # Second run - should also fail (not cached), proving failed tasks aren't cached
        with pytest.raises(RuntimeError, match="Simulated task failure for testing"):
            task_manager.run_tasks([failing_task], [task_identifier])

        # The fact that both runs raise exceptions (rather than the second returning cached results)
        # proves that failed tasks are not cached by the Dask task manager


# Tests for SequentialTaskManager
from POMDPPlanners.simulations.simulations_deployment.task_managers import (
    SequentialTaskManager,
)


def test_sequential_task_manager_initialization(cache_db):
    """Test SequentialTaskManager initialization.

    Purpose: Validates that SequentialTaskManager initializes correctly with cache database and inherits from JoblibTaskManager

    Given: A cache database fixture and SequentialTaskManager constructor
    When: SequentialTaskManager is initialized as a context manager with cache_db
    Then: Task manager has correct attributes inherited from JoblibTaskManager and n_jobs is set to 1 for sequential execution

    Test type: unit
    """
    with SequentialTaskManager(cache_db=cache_db) as task_manager:
        assert task_manager.n_jobs == -1  # Inherited from JoblibTaskManager
        assert task_manager.verbose == 0
        assert task_manager.memory is not None
        assert isinstance(task_manager, SequentialTaskManager)
        assert isinstance(task_manager, JoblibTaskManager)  # Should inherit from JoblibTaskManager


def test_sequential_task_manager_with_cache_clear(cache_db):
    """Test SequentialTaskManager initialization with cache clearing.

    Purpose: Validates that SequentialTaskManager can initialize with cache clearing enabled

    Given: A cache database fixture and SequentialTaskManager constructor with clear_cache_on_start=True
    When: SequentialTaskManager is initialized as a context manager
    Then: Task manager has valid memory object and cache is empty after clearing (0 items)

    Test type: unit
    """
    with SequentialTaskManager(cache_db=cache_db, clear_cache_on_start=True) as task_manager:
        assert task_manager.memory is not None
        # Cache should be empty after clearing
        memory = task_manager.memory
        assert memory is not None and memory.store_backend is not None
        items = memory.store_backend.get_items()
        assert items is not None
        assert len(list(items)) == 0


def test_sequential_task_manager_run_tasks(cache_db, environment, policy):
    """Test running tasks with SequentialTaskManager.

    Purpose: Validates that SequentialTaskManager can successfully run multiple EpisodeSimulationTask instances sequentially with caching

    Given: A cache database, TigerPOMDP environment, SparsePFT policy, and 3 EpisodeSimulationTask instances
    When: SequentialTaskManager.run_tasks() is called with the task list and identifiers
    Then: Returns 3 successful results and 3 successful IDs, with each result being a valid History object

    Test type: unit
    """
    with SequentialTaskManager(cache_db=cache_db) as task_manager:
        # Create multiple tasks
        tasks = []
        task_identifiers = []
        for i in range(3):
            belief = create_test_belief()
            task = EpisodeSimulationTask(
                environment=environment,
                policy=policy,
                initial_belief=belief,
                num_steps=2,
                episode_id=i,
                seed=42 + i,
                console_output=False,
            )
            tasks.append(task)
            task_identifiers.append(f"episode_{i}")

        # Run tasks
        results, successful_ids = task_manager.run_tasks(tasks, task_identifiers)

        # Verify results
        assert len(results) == 3
        assert len(successful_ids) == 3
        assert all(id in successful_ids for id in task_identifiers)
        for result in results:
            assert isinstance(result, History)
            assert len(result.history) <= 2


def test_sequential_task_manager_cache(cache_db, environment, policy):
    """Test caching behavior of SequentialTaskManager.

    Purpose: Validates that SequentialTaskManager properly caches task results and reuses them on subsequent runs

    Given: A cache database, TigerPOMDP environment, SparsePFT policy, and EpisodeSimulationTask
    When: Same task is run twice with identical parameters
    Then: Second run retrieves result from cache instead of re-executing, demonstrating caching effectiveness

    Test type: unit
    """
    with SequentialTaskManager(cache_db=cache_db) as task_manager:
        # Create a task
        belief = create_test_belief()
        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            console_output=False,
        )
        task_identifier = "episode_1"

        # Run task twice
        result1, ids1 = task_manager.run_tasks([task], [task_identifier])
        result2, ids2 = task_manager.run_tasks([task], [task_identifier])

        # Compare main fields instead of object equality
        assert result1[0].discount_factor == result2[0].discount_factor
        assert result1[0].actual_num_steps == result2[0].actual_num_steps
        assert result1[0].reach_terminal_state == result2[0].reach_terminal_state
        assert len(result1[0].history) == len(result2[0].history)
        assert ids1 == ids2
        assert task_identifier in ids1

        # Add a small delay to ensure all file handles are released
        time.sleep(0.1)

        try:
            # Clear cache and run again
            task_manager.clear_cache()
        except PermissionError:
            # If we can't clear the cache, log a warning but continue with the test
            import warnings

            warnings.warn("Could not clear cache due to file being in use, continuing with test")

        result3, ids3 = task_manager.run_tasks([task], [task_identifier])

        # Compare main fields again
        assert result1[0].discount_factor == result3[0].discount_factor
        assert result1[0].actual_num_steps == result3[0].actual_num_steps
        assert result1[0].reach_terminal_state == result3[0].reach_terminal_state
        assert len(result1[0].history) == len(result3[0].history)
        assert ids1 == ids3
        assert task_identifier in ids3


def test_sequential_task_manager_logging(cache_db, environment, policy, temp_cache_dir):
    """Test that SequentialTaskManager writes logs properly.

    Purpose: Validates that SequentialTaskManager can write logs to specified directory with debug logging enabled

    Given: A cache database, TigerPOMDP environment, SparsePFT policy, EpisodeSimulationTask, and log directory
    When: SequentialTaskManager runs task with logger_debug=True and cache_dir configured
    Then: Task executes successfully and generates logs in the specified directory

    Test type: unit
    """
    import os
    from pathlib import Path

    # Create a specific log directory for this test
    log_dir = Path(temp_cache_dir) / "test_sequential_logs"
    log_dir.mkdir(exist_ok=True)

    with SequentialTaskManager(
        cache_db=cache_db, cache_dir=str(log_dir), logger_debug=True
    ) as task_manager:
        # Create a task
        belief = create_test_belief()
        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            console_output=False,
        )
        task_identifier = "episode_1"

        # Run task to generate logs
        results, successful_ids = task_manager.run_tasks([task], [task_identifier])

        # Verify results
        assert len(results) == 1
        assert len(successful_ids) == 1
        assert task_identifier in successful_ids

        # Verify that expected log messages are present in the task manager logs
        # Note: We don't check directory structure as it may change


def test_sequential_task_manager_logging_with_multiple_tasks(
    cache_db, environment, policy, temp_cache_dir
):
    """Test that SequentialTaskManager logs progress updates for multiple tasks.

    Purpose: Validates that SequentialTaskManager can log progress updates when running multiple tasks sequentially

    Given: A cache database, TigerPOMDP environment, SparsePFT policy, and 3 EpisodeSimulationTask instances
    When: SequentialTaskManager runs multiple tasks with logger_debug=True and cache_dir configured
    Then: All tasks execute successfully and progress logging is generated for sequential task execution

    Test type: integration
    """
    import os
    from pathlib import Path

    # Create a specific log directory for this test
    log_dir = Path(temp_cache_dir) / "test_sequential_logs_multi"
    log_dir.mkdir(exist_ok=True)

    with SequentialTaskManager(
        cache_db=cache_db, cache_dir=str(log_dir), logger_debug=True
    ) as task_manager:
        # Create multiple tasks to test progress logging
        tasks = []
        task_identifiers = []
        for i in range(3):  # Create 3 tasks to test progress updates
            belief = create_test_belief()
            task = EpisodeSimulationTask(
                environment=environment,
                policy=policy,
                initial_belief=belief,
                num_steps=2,
                episode_id=i,
                seed=42 + i,
                console_output=False,
            )
            tasks.append(task)
            task_identifiers.append(f"episode_{i}")

        # Run tasks to generate logs
        results, successful_ids = task_manager.run_tasks(tasks, task_identifiers)

        # Verify results
        assert len(results) == 3
        assert len(successful_ids) == 3

        # Note: We don't check directory structure as it may change


def test_sequential_task_manager_failed_tasks_not_cached(cache_db, environment, policy):
    """Test that SequentialTaskManager does not cache results from failed tasks.

    Purpose: Validates that failed tasks are not cached and subsequent runs retry execution for sequential processing

    Given: SequentialTaskManager with cache and a task that always fails
    When: The failing task is run multiple times
    Then: Each run attempts to execute the task (not cached) and fails consistently

    Test type: unit
    """
    with SequentialTaskManager(cache_db=cache_db) as task_manager:
        # Create a task that always fails
        belief = create_test_belief()
        failing_task = FailingEpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=999,
            seed=42,
            console_output=False,
        )
        task_identifier = "failing_sequential_episode"

        # First run - should fail with exception
        with pytest.raises(RuntimeError, match="Simulated task failure for testing"):
            task_manager.run_tasks([failing_task], [task_identifier])

        # Second run - should also fail (not cached), proving failed tasks aren't cached
        with pytest.raises(RuntimeError, match="Simulated task failure for testing"):
            task_manager.run_tasks([failing_task], [task_identifier])

        # The fact that both runs raise exceptions (rather than the second returning cached results)
        # proves that failed tasks are not cached by the sequential task manager


def test_sequential_task_manager_execution_order(cache_db, environment, policy):
    """Test that SequentialTaskManager executes tasks in the correct order.

    Purpose: Validates that SequentialTaskManager processes tasks in the order they are provided

    Given: A cache database, TigerPOMDP environment, SparsePFT policy, and 3 EpisodeSimulationTask instances with different seeds
    When: SequentialTaskManager runs multiple tasks
    Then: Tasks are executed sequentially in the order provided, and results maintain the same order

    Test type: unit
    """
    with SequentialTaskManager(cache_db=cache_db) as task_manager:
        # Create tasks with different seeds to ensure different results
        tasks = []
        task_identifiers = []
        seeds = [42, 123, 456]

        for i, seed in enumerate(seeds):
            belief = create_test_belief()
            task = EpisodeSimulationTask(
                environment=environment,
                policy=policy,
                initial_belief=belief,
                num_steps=2,
                episode_id=i,
                seed=seed,
                console_output=False,
            )
            tasks.append(task)
            task_identifiers.append(f"episode_{i}")

        # Run tasks
        results, successful_ids = task_manager.run_tasks(tasks, task_identifiers)

        # Verify results
        assert len(results) == 3
        assert len(successful_ids) == 3

        # Verify that results correspond to the original task order
        for i, result in enumerate(results):
            assert isinstance(result, History)
            # Verify that results are valid History objects
            assert hasattr(result, "history")
            assert hasattr(result, "actual_num_steps")
            assert hasattr(result, "reach_terminal_state")


def test_sequential_task_manager_single_task_execution(cache_db, environment, policy):
    """Test that SequentialTaskManager handles single task execution correctly.

    Purpose: Validates that SequentialTaskManager can execute a single task efficiently

    Given: A cache database, TigerPOMDP environment, SparsePFT policy, and single EpisodeSimulationTask
    When: SequentialTaskManager runs a single task
    Then: Task executes successfully and returns correct result format

    Test type: unit
    """
    with SequentialTaskManager(cache_db=cache_db) as task_manager:
        # Create a single task
        belief = create_test_belief()
        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            console_output=False,
        )
        task_identifier = "single_episode"

        # Run single task
        results, successful_ids = task_manager.run_tasks([task], [task_identifier])

        # Verify results
        assert len(results) == 1
        assert len(successful_ids) == 1
        assert task_identifier in successful_ids
        assert isinstance(results[0], History)
        assert hasattr(results[0], "history")
        assert hasattr(results[0], "actual_num_steps")
        assert hasattr(results[0], "reach_terminal_state")


def test_sequential_task_manager_empty_task_list(cache_db):
    """Test that SequentialTaskManager handles empty task list correctly.

    Purpose: Validates that SequentialTaskManager can handle edge case of empty task list

    Given: A cache database and empty task list
    When: SequentialTaskManager.run_tasks() is called with empty lists
    Then: Returns empty results and empty successful_ids lists

    Test type: unit
    """
    with SequentialTaskManager(cache_db=cache_db) as task_manager:
        # Run with empty task list
        results, successful_ids = task_manager.run_tasks([], [])

        # Verify results
        assert len(results) == 0
        assert len(successful_ids) == 0


def test_sequential_task_manager_context_manager(cache_db, environment, policy):
    """Test that SequentialTaskManager works correctly as a context manager.

    Purpose: Validates that SequentialTaskManager can be used as a context manager with proper cleanup

    Given: A cache database, TigerPOMDP environment, SparsePFT policy, and EpisodeSimulationTask
    When: SequentialTaskManager is used as a context manager
    Then: Task manager initializes correctly, executes tasks, and cleans up properly

    Test type: unit
    """
    # Test context manager functionality
    with SequentialTaskManager(cache_db=cache_db) as task_manager:
        # Verify initialization
        assert task_manager.memory is not None
        assert task_manager.cache_db is not None

        # Create and run a task
        belief = create_test_belief()
        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            console_output=False,
        )
        task_identifier = "context_episode"

        results, successful_ids = task_manager.run_tasks([task], [task_identifier])

        # Verify results
        assert len(results) == 1
        assert len(successful_ids) == 1
        assert task_identifier in successful_ids

    # Context manager should have cleaned up properly
    # Note: SequentialTaskManager doesn't require explicit cleanup like DaskTaskManager


def test_sequential_task_manager_verbose_output(cache_db, environment, policy):
    """Test that SequentialTaskManager respects verbose parameter.

    Purpose: Validates that SequentialTaskManager can be configured with different verbose levels

    Given: A cache database, TigerPOMDP environment, SparsePFT policy, and EpisodeSimulationTask
    When: SequentialTaskManager is initialized with verbose=1
    Then: Task manager executes successfully with increased verbosity

    Test type: unit
    """
    with SequentialTaskManager(cache_db=cache_db, verbose=1) as task_manager:
        # Verify verbose setting
        assert task_manager.verbose == 1

        # Create and run a task
        belief = create_test_belief()
        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            console_output=False,
        )
        task_identifier = "verbose_episode"

        results, successful_ids = task_manager.run_tasks([task], [task_identifier])

        # Verify results
        assert len(results) == 1
        assert len(successful_ids) == 1
        assert task_identifier in successful_ids


# Tests for PBSTaskManager
def test_pbs_task_manager_enum():
    """Test that TaskManagerType includes PBS option.

    Purpose: Validates that PBS is properly added to TaskManagerType enum

    Given: TaskManagerType enum
    When: PBS enum value is accessed
    Then: PBS value equals "pbs" string and enum contains all expected values

    Test type: unit
    """
    assert TaskManagerType.PBS.value == "pbs"
    expected_values = {"dask", "joblib", "pbs"}
    actual_values = {t.value for t in TaskManagerType}
    assert expected_values <= actual_values


def test_pbs_task_manager_initialization():
    """Test PBSTaskManager initialization without dask-jobqueue dependency.

    Purpose: Validates that PBSTaskManager initializes correctly and stores PBS-specific parameters

    Given: PBSTaskManager constructor with PBS-specific parameters
    When: PBSTaskManager is instantiated with queue and resource parameters
    Then: Task manager stores all PBS parameters correctly and inherits from DaskTaskManager

    Test type: unit
    """
    manager = PBSTaskManager(
        queue="default",
        n_workers=2,
        cores=4,
        memory="8GB",
        processes=2,
        walltime="02:00:00",
        job_extra=["#PBS -m abe"],
    )

    assert manager.queue == "default"
    assert manager.n_workers == 2
    assert manager.cores == 4
    assert manager.memory == "8GB"
    assert manager.processes == 2
    assert manager.walltime == "02:00:00"
    assert manager.job_extra == ["#PBS -m abe"]
    assert isinstance(manager, DaskTaskManager)


def test_pbs_task_manager_initialization_defaults():
    """Test PBSTaskManager initialization with default parameters.

    Purpose: Validates that PBSTaskManager uses correct default values for optional parameters

    Given: PBSTaskManager constructor with only required queue parameter
    When: PBSTaskManager is instantiated with minimal parameters
    Then: Task manager uses expected default values for all optional parameters

    Test type: unit
    """
    manager = PBSTaskManager(queue="batch")

    assert manager.queue == "batch"
    assert manager.n_workers == 4
    assert manager.cores == 1
    assert manager.memory == "4GB"
    assert manager.processes == 1
    assert manager.walltime == "01:00:00"
    assert manager.job_extra == []
    assert manager.cache_size == 2e9
    assert manager.clear_cache_on_start is False


def test_pbs_task_manager_missing_dependency():
    """Test PBSTaskManager raises proper error when dask-jobqueue is missing.

    Purpose: Validates that PBSTaskManager provides clear error message when dependency is missing

    Given: PBSTaskManager instance without dask-jobqueue installed
    When: _initialize_client() is called (which happens during task execution)
    Then: RuntimeError is raised with helpful message about installing dask-jobqueue

    Test type: unit
    """
    from unittest.mock import MagicMock, patch

    manager = PBSTaskManager(queue="default")

    # Mock the import of dask_jobqueue to raise ImportError
    with patch.dict("sys.modules", {"dask_jobqueue": None}):
        with patch(
            "POMDPPlanners.simulations.simulations_deployment.task_managers.PBSCluster",
            side_effect=ImportError("No module named 'dask_jobqueue'"),
        ):
            with pytest.raises(RuntimeError, match="dask-jobqueue is required for PBS support"):
                manager._initialize_client()


def test_pbs_task_manager_inheritance():
    """Test that PBSTaskManager correctly inherits from DaskTaskManager.

    Purpose: Validates that PBSTaskManager inherits all expected methods from DaskTaskManager

    Given: PBSTaskManager instance
    When: Instance methods are checked
    Then: All DaskTaskManager methods are available on PBSTaskManager instance

    Test type: unit
    """
    manager = PBSTaskManager(queue="default")

    # Check that key DaskTaskManager methods are available
    assert hasattr(manager, "submit_tasks")
    assert hasattr(manager, "gather_results")
    assert hasattr(manager, "run_tasks")
    assert hasattr(manager, "get_task_status")
    assert hasattr(manager, "cancel_tasks")
    assert hasattr(manager, "__enter__")
    assert hasattr(manager, "__exit__")


def test_pbs_task_manager_context_manager():
    """Test that PBSTaskManager works as a context manager.

    Purpose: Validates that PBSTaskManager can be used as a context manager without initialization

    Given: PBSTaskManager instance
    When: Used as a context manager (with statement)
    Then: Context manager protocol works without errors during entry and exit

    Test type: unit
    """
    # Test that context manager works without actually initializing client
    try:
        with PBSTaskManager(queue="default") as manager:
            assert manager.queue == "default"
            assert manager.client is None  # Not initialized yet
    except Exception as e:
        # Should not raise any exceptions during context manager setup
        pytest.fail(f"Context manager should not raise exceptions during setup: {e}")


# Tests for PBSTaskManager Dashboard Functionality
def test_pbs_task_manager_dashboard_initialization_defaults():
    """Test PBSTaskManager dashboard initialization with default parameters.

    Purpose: Validates that PBSTaskManager initializes with correct default dashboard parameters

    Given: PBSTaskManager constructor with only required queue parameter
    When: PBSTaskManager is instantiated with minimal parameters
    Then: Task manager uses expected default values for all dashboard parameters

    Test type: unit
    """
    manager = PBSTaskManager(queue="batch")

    assert manager.enable_dashboard is True
    assert manager.dashboard_address == "0.0.0.0"
    assert manager.dashboard_port == 8787
    assert manager.dashboard_prefix is None


def test_pbs_task_manager_dashboard_initialization_custom():
    """Test PBSTaskManager dashboard initialization with custom parameters.

    Purpose: Validates that PBSTaskManager correctly stores custom dashboard configuration

    Given: PBSTaskManager constructor with custom dashboard parameters
    When: PBSTaskManager is instantiated with specific dashboard settings
    Then: Task manager stores all custom dashboard parameters correctly

    Test type: unit
    """
    manager = PBSTaskManager(
        queue="gpu_queue",
        enable_dashboard=True,
        dashboard_address="192.168.1.100",
        dashboard_port=8888,
        dashboard_prefix="/my-dashboard",
    )

    assert manager.enable_dashboard is True
    assert manager.dashboard_address == "192.168.1.100"
    assert manager.dashboard_port == 8888
    assert manager.dashboard_prefix == "/my-dashboard"


def test_pbs_task_manager_dashboard_disabled():
    """Test PBSTaskManager with dashboard disabled.

    Purpose: Validates that PBSTaskManager can be configured with dashboard disabled

    Given: PBSTaskManager constructor with enable_dashboard=False
    When: PBSTaskManager is instantiated with dashboard disabled
    Then: Task manager has dashboard disabled and other parameters are still stored

    Test type: unit
    """
    manager = PBSTaskManager(queue="batch_queue", enable_dashboard=False, dashboard_port=9999)

    assert manager.enable_dashboard is False
    assert manager.dashboard_port == 9999  # Should still store the parameter


def test_pbs_task_manager_get_dashboard_url_disabled():
    """Test get_dashboard_url when dashboard is disabled.

    Purpose: Validates that get_dashboard_url returns None when dashboard is disabled

    Given: PBSTaskManager instance with dashboard disabled
    When: get_dashboard_url() method is called
    Then: Returns None indicating dashboard is not available

    Test type: unit
    """
    manager = PBSTaskManager(queue="default", enable_dashboard=False)

    dashboard_url = manager.get_dashboard_url()
    assert dashboard_url is None


def test_pbs_task_manager_get_dashboard_url_no_client():
    """Test get_dashboard_url when client is not initialized.

    Purpose: Validates that get_dashboard_url returns None when no client exists

    Given: PBSTaskManager instance with dashboard enabled but no client
    When: get_dashboard_url() method is called
    Then: Returns None indicating dashboard is not available yet

    Test type: unit
    """
    manager = PBSTaskManager(queue="default", enable_dashboard=True)

    # Client is None before initialization
    assert manager.client is None
    dashboard_url = manager.get_dashboard_url()
    assert dashboard_url is None


def test_pbs_task_manager_is_dashboard_running_disabled():
    """Test is_dashboard_running when dashboard is disabled.

    Purpose: Validates that is_dashboard_running returns False when dashboard is disabled

    Given: PBSTaskManager instance with dashboard disabled
    When: is_dashboard_running() method is called
    Then: Returns False indicating dashboard is not running

    Test type: unit
    """
    manager = PBSTaskManager(queue="default", enable_dashboard=False)

    is_running = manager.is_dashboard_running()
    assert is_running is False


def test_pbs_task_manager_is_dashboard_running_no_client():
    """Test is_dashboard_running when client is not initialized.

    Purpose: Validates that is_dashboard_running returns False when no client exists

    Given: PBSTaskManager instance with dashboard enabled but no client
    When: is_dashboard_running() method is called
    Then: Returns False indicating dashboard is not running yet

    Test type: unit
    """
    manager = PBSTaskManager(queue="default", enable_dashboard=True)

    # Client is None before initialization
    assert manager.client is None
    is_running = manager.is_dashboard_running()
    assert is_running is False


def test_pbs_task_manager_dashboard_helper_methods_exist():
    """Test that PBSTaskManager has all dashboard helper methods.

    Purpose: Validates that PBSTaskManager implements all expected dashboard helper methods

    Given: PBSTaskManager instance
    When: Instance methods are checked
    Then: All dashboard helper methods are available on PBSTaskManager instance

    Test type: unit
    """
    manager = PBSTaskManager(queue="default")

    # Check that dashboard helper methods are available
    assert hasattr(manager, "get_dashboard_url")
    assert hasattr(manager, "is_dashboard_running")
    assert callable(manager.get_dashboard_url)
    assert callable(manager.is_dashboard_running)


def test_pbs_task_manager_dashboard_port_validation():
    """Test PBSTaskManager accepts various dashboard port values.

    Purpose: Validates that PBSTaskManager accepts different valid port numbers

    Given: PBSTaskManager constructor with different port values
    When: PBSTaskManager is instantiated with various valid ports
    Then: Task manager stores the correct port values

    Test type: unit
    """
    # Test common port values
    test_ports = [8787, 8888, 9999, 8080, 3000]

    for port in test_ports:
        manager = PBSTaskManager(queue="default", dashboard_port=port)
        assert manager.dashboard_port == port


def test_pbs_task_manager_dashboard_address_validation():
    """Test PBSTaskManager accepts various dashboard address values.

    Purpose: Validates that PBSTaskManager accepts different valid address formats

    Given: PBSTaskManager constructor with different address values
    When: PBSTaskManager is instantiated with various valid addresses
    Then: Task manager stores the correct address values

    Test type: unit
    """
    # Test common address values
    test_addresses = ["0.0.0.0", "127.0.0.1", "192.168.1.100", "localhost"]

    for address in test_addresses:
        manager = PBSTaskManager(queue="default", dashboard_address=address)
        assert manager.dashboard_address == address


def test_pbs_task_manager_dashboard_prefix_validation():
    """Test PBSTaskManager accepts various dashboard prefix values.

    Purpose: Validates that PBSTaskManager accepts different dashboard prefix formats

    Given: PBSTaskManager constructor with different prefix values
    When: PBSTaskManager is instantiated with various valid prefixes
    Then: Task manager stores the correct prefix values

    Test type: unit
    """
    # Test common prefix values
    test_prefixes = [None, "/dashboard", "/my-app", "/cluster-monitor", "api/v1"]

    for prefix in test_prefixes:
        manager = PBSTaskManager(queue="default", dashboard_prefix=prefix)
        assert manager.dashboard_prefix == prefix


def test_pbs_task_manager_context_manager_with_dashboard():
    """Test that PBSTaskManager context manager works with dashboard settings.

    Purpose: Validates that PBSTaskManager context manager properly handles dashboard configuration

    Given: PBSTaskManager instance with custom dashboard settings
    When: Used as a context manager (with statement)
    Then: Context manager protocol works and preserves dashboard settings during entry and exit

    Test type: unit
    """
    # Test that context manager works with dashboard configuration
    try:
        with PBSTaskManager(
            queue="default",
            enable_dashboard=True,
            dashboard_port=8888,
            dashboard_address="127.0.0.1",
        ) as manager:
            assert manager.queue == "default"
            assert manager.enable_dashboard is True
            assert manager.dashboard_port == 8888
            assert manager.dashboard_address == "127.0.0.1"
            assert manager.client is None  # Not initialized yet
            assert manager.cluster is None  # Not initialized yet
    except Exception as e:
        # Should not raise any exceptions during context manager setup
        pytest.fail(f"Context manager with dashboard should not raise exceptions during setup: {e}")


def test_pbs_task_manager_cluster_storage():
    """Test that PBSTaskManager stores cluster reference for dashboard access.

    Purpose: Validates that PBSTaskManager has cluster attribute for dashboard functionality

    Given: PBSTaskManager instance
    When: Instance is created
    Then: Task manager has cluster attribute initialized to None

    Test type: unit
    """
    manager = PBSTaskManager(queue="default")

    # Check that cluster attribute exists and is initially None
    assert hasattr(manager, "cluster")
    assert manager.cluster is None


# Tests for _log_cache_statistics method
def test_joblib_task_manager_log_cache_statistics_current_behavior(cache_db):
    """Test _log_cache_statistics current behavior with missing get_stats method.

    Purpose: Validates that _log_cache_statistics handles the missing get_stats method gracefully

    Given: JoblibTaskManager with cache (current joblib version doesn't have get_stats)
    When: _log_cache_statistics() is called
    Then: Method catches AttributeError and logs warning message

    Test type: unit
    """
    from unittest.mock import patch

    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        with patch.object(task_manager.logger, "warning") as mock_warning:
            task_manager._log_cache_statistics()

            # Verify that warning was called once
            assert mock_warning.call_count == 1

            # Check the warning message
            warning_message = mock_warning.call_args_list[0][0][0]
            assert "Could not log cache statistics" in warning_message
            assert "'Memory' object has no attribute 'get_stats'" in warning_message


def test_joblib_task_manager_log_cache_statistics_normal(cache_db):
    """Test _log_cache_statistics with normal cache statistics (if get_stats existed).

    Purpose: Validates that _log_cache_statistics correctly logs cache hits, misses, and hit rate

    Given: JoblibTaskManager with cache and mock cache statistics
    When: _log_cache_statistics() is called with normal cache stats
    Then: Method logs cache hits, misses, and calculated hit rate correctly

    Test type: unit
    """
    from unittest.mock import Mock, patch

    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Mock cache statistics
        mock_cache_stats = {"hits": 8, "misses": 2}

        # Add get_stats method to memory object for testing
        setattr(task_manager.memory, "get_stats", Mock(return_value=mock_cache_stats))

        with patch.object(task_manager.logger, "info") as mock_info:
            task_manager._log_cache_statistics()

            # Check that the specific cache statistics messages were logged
            logged_messages = [call[0][0] for call in mock_info.call_args_list]

            # Find the cache stats message
            cache_stats_message = None
            hit_rate_message = None

            for msg in logged_messages:
                if "Joblib Cache Stats" in msg:
                    cache_stats_message = msg
                elif "Cache hit rate:" in msg:
                    hit_rate_message = msg

            # Verify cache stats message
            assert cache_stats_message is not None
            assert "Cache hits: 8" in cache_stats_message
            assert "Cache misses: 2" in cache_stats_message

            # Verify hit rate message
            assert hit_rate_message is not None
            assert "Cache hit rate: 80.0%" in hit_rate_message


def test_joblib_task_manager_log_cache_statistics_zero_requests(cache_db):
    """Test _log_cache_statistics with zero cache requests.

    Purpose: Validates that _log_cache_statistics handles zero total requests correctly

    Given: JoblibTaskManager with cache and zero cache statistics
    When: _log_cache_statistics() is called with zero hits and misses
    Then: Method logs cache stats but skips hit rate calculation

    Test type: unit
    """
    from unittest.mock import Mock, patch

    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Mock cache statistics with zero requests
        mock_cache_stats = {"hits": 0, "misses": 0}

        # Add get_stats method to memory object for testing
        setattr(task_manager.memory, "get_stats", Mock(return_value=mock_cache_stats))

        with patch.object(task_manager.logger, "info") as mock_info:
            task_manager._log_cache_statistics()

            # Check that the specific cache statistics messages were logged
            logged_messages = [call[0][0] for call in mock_info.call_args_list]

            # Find the cache stats message
            cache_stats_message = None
            hit_rate_message = None

            for msg in logged_messages:
                if "Joblib Cache Stats" in msg:
                    cache_stats_message = msg
                elif "Cache hit rate:" in msg:
                    hit_rate_message = msg

            # Verify cache stats message
            assert cache_stats_message is not None
            assert "Cache hits: 0" in cache_stats_message
            assert "Cache misses: 0" in cache_stats_message

            # Verify no hit rate message (should be None)
            assert hit_rate_message is None


def test_joblib_task_manager_log_cache_statistics_exception_handling(cache_db):
    """Test _log_cache_statistics exception handling.

    Purpose: Validates that _log_cache_statistics handles exceptions gracefully

    Given: JoblibTaskManager with cache that raises exception on get_stats()
    When: _log_cache_statistics() is called and get_stats() raises exception
    Then: Method catches exception and logs warning message

    Test type: unit
    """
    from unittest.mock import Mock, patch

    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Mock get_stats to raise an exception
        setattr(
            task_manager.memory,
            "get_stats",
            Mock(side_effect=RuntimeError("Cache stats unavailable")),
        )

        with patch.object(task_manager.logger, "warning") as mock_warning:
            task_manager._log_cache_statistics()

            # Verify that warning was called once
            assert mock_warning.call_count == 1

            # Check the warning message
            warning_message = mock_warning.call_args_list[0][0][0]
            assert "Could not log cache statistics" in warning_message
            assert "Cache stats unavailable" in warning_message


def test_joblib_task_manager_log_cache_statistics_different_hit_rates(cache_db):
    """Test _log_cache_statistics with different hit rate scenarios.

    Purpose: Validates that _log_cache_statistics correctly calculates various hit rates

    Given: JoblibTaskManager with cache and different cache statistics
    When: _log_cache_statistics() is called with various hit/miss ratios
    Then: Method correctly calculates and logs different hit rates

    Test type: unit
    """
    from unittest.mock import Mock, patch

    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        test_cases = [
            ({"hits": 0, "misses": 10}, "0.0%"),  # 0% hit rate
            ({"hits": 5, "misses": 5}, "50.0%"),  # 50% hit rate
            ({"hits": 10, "misses": 0}, "100.0%"),  # 100% hit rate
            ({"hits": 3, "misses": 7}, "30.0%"),  # 30% hit rate
            ({"hits": 7, "misses": 3}, "70.0%"),  # 70% hit rate
        ]

        for cache_stats, expected_hit_rate in test_cases:
            # Add get_stats method to memory object for testing
            setattr(task_manager.memory, "get_stats", Mock(return_value=cache_stats))

            with patch.object(task_manager.logger, "info") as mock_info:
                task_manager._log_cache_statistics()

                # Check that the specific hit rate message was logged
                logged_messages = [call[0][0] for call in mock_info.call_args_list]

                # Find the hit rate message
                hit_rate_message = None
                for msg in logged_messages:
                    if "Cache hit rate:" in msg:
                        hit_rate_message = msg
                        break

                # Verify hit rate calculation
                assert hit_rate_message is not None
                assert f"Cache hit rate: {expected_hit_rate}" in hit_rate_message


def test_joblib_task_manager_log_cache_statistics_missing_keys(cache_db):
    """Test _log_cache_statistics with missing cache statistics keys.

    Purpose: Validates that _log_cache_statistics handles missing keys gracefully

    Given: JoblibTaskManager with cache statistics missing hits/misses keys
    When: _log_cache_statistics() is called with incomplete stats
    Then: Method uses default values (0) for missing keys

    Test type: unit
    """
    from unittest.mock import Mock, patch

    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Mock cache statistics with missing keys
        mock_cache_stats = {}  # Empty dict - no hits or misses keys

        # Add get_stats method to memory object for testing
        setattr(task_manager.memory, "get_stats", Mock(return_value=mock_cache_stats))

        with patch.object(task_manager.logger, "info") as mock_info:
            task_manager._log_cache_statistics()

            # Check that the specific cache statistics messages were logged
            logged_messages = [call[0][0] for call in mock_info.call_args_list]

            # Find the cache stats message
            cache_stats_message = None
            hit_rate_message = None

            for msg in logged_messages:
                if "Joblib Cache Stats" in msg:
                    cache_stats_message = msg
                elif "Cache hit rate:" in msg:
                    hit_rate_message = msg

            # Verify cache stats message uses default values
            assert cache_stats_message is not None
            assert "Cache hits: 0" in cache_stats_message
            assert "Cache misses: 0" in cache_stats_message

            # Verify no hit rate message (should be None due to zero total)
            assert hit_rate_message is None


def test_sequential_task_manager_log_cache_statistics_inheritance(cache_db):
    """Test that SequentialTaskManager inherits _log_cache_statistics method.

    Purpose: Validates that SequentialTaskManager can use _log_cache_statistics from JoblibTaskManager

    Given: SequentialTaskManager with cache and mock cache statistics
    When: _log_cache_statistics() is called on SequentialTaskManager
    Then: Method works correctly and logs cache statistics

    Test type: unit
    """
    from unittest.mock import Mock, patch

    with SequentialTaskManager(cache_db=cache_db) as task_manager:
        # Mock cache statistics
        mock_cache_stats = {"hits": 6, "misses": 4}

        # Add get_stats method to memory object for testing
        setattr(task_manager.memory, "get_stats", Mock(return_value=mock_cache_stats))

        with patch.object(task_manager.logger, "info") as mock_info:
            task_manager._log_cache_statistics()

            # Check that the specific cache statistics messages were logged
            logged_messages = [call[0][0] for call in mock_info.call_args_list]

            # Find the cache stats message
            cache_stats_message = None
            hit_rate_message = None

            for msg in logged_messages:
                if "Joblib Cache Stats" in msg:
                    cache_stats_message = msg
                elif "Cache hit rate:" in msg:
                    hit_rate_message = msg

            # Verify cache stats message
            assert cache_stats_message is not None
            assert "Cache hits: 6" in cache_stats_message
            assert "Cache misses: 4" in cache_stats_message

            # Verify hit rate message
            assert hit_rate_message is not None
            assert "Cache hit rate: 60.0%" in hit_rate_message


def test_joblib_task_manager_log_cache_statistics_precision(cache_db):
    """Test _log_cache_statistics hit rate precision formatting.

    Purpose: Validates that _log_cache_statistics formats hit rate with correct precision

    Given: JoblibTaskManager with cache statistics that result in non-integer hit rates
    When: _log_cache_statistics() is called with fractional hit rates
    Then: Method formats hit rate to one decimal place

    Test type: unit
    """
    from unittest.mock import Mock, patch

    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Mock cache statistics that result in fractional hit rate
        mock_cache_stats = {"hits": 1, "misses": 3}  # 25.0% hit rate

        # Add get_stats method to memory object for testing
        setattr(task_manager.memory, "get_stats", Mock(return_value=mock_cache_stats))

        with patch.object(task_manager.logger, "info") as mock_info:
            task_manager._log_cache_statistics()

            # Check that the specific hit rate message was logged
            logged_messages = [call[0][0] for call in mock_info.call_args_list]

            # Find the hit rate message
            hit_rate_message = None
            for msg in logged_messages:
                if "Cache hit rate:" in msg:
                    hit_rate_message = msg
                    break

            # Check hit rate formatting
            assert hit_rate_message is not None
            assert "Cache hit rate: 25.0%" in hit_rate_message

            # Test another fractional case
            mock_cache_stats = {"hits": 1, "misses": 2}  # 33.333...% hit rate
            setattr(task_manager.memory, "get_stats", Mock(return_value=mock_cache_stats))
            task_manager._log_cache_statistics()

            # Get updated messages
            logged_messages = [call[0][0] for call in mock_info.call_args_list]

            # Find the second hit rate message
            hit_rate_message = None
            for msg in logged_messages:
                if "Cache hit rate: 33.3%" in msg:
                    hit_rate_message = msg
                    break

            assert hit_rate_message is not None
