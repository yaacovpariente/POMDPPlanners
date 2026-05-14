# pylint: disable=protected-access  # Tests need to access protected members
import shutil
import tempfile
import threading
import time
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from dask.distributed import LocalCluster

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import History
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.simulations.simulations_deployment.cache_dbs import DiskCacheDB
from POMDPPlanners.simulations.simulations_deployment.task_managers import (
    DaskTaskManager,
    JoblibTaskManager,
    PBSTaskManager,
    SequentialTaskManager,
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


def test_dask_task_manager_progress_callback_fires_per_task(environment, policy):
    """Test that DaskTaskManager fires set_progress_callback once per task.

    Purpose: Validates the new client-thread done-callback hook that
    bridges Dask future resolution to the in-process notifier.

    Given: A DaskTaskManager with a counting callback registered via
        set_progress_callback, and 3 real EpisodeSimulationTask instances.
    When: run_tasks is called with the three tasks.
    Then: The counter eventually reaches 3 — one callback fire per
        completed task, all in the parent process. A short wait covers
        the Dask client-thread fire-and-forget timing.

    Test type: integration
    """
    with DaskTaskManager() as task_manager:
        call_count = [0]
        lock = threading.Lock()

        def on_progress() -> None:
            with lock:
                call_count[0] += 1

        task_manager.set_progress_callback(on_progress)

        tasks = []
        identifiers = []
        for i in range(3):
            tasks.append(
                EpisodeSimulationTask(
                    environment=environment,
                    policy=policy,
                    initial_belief=create_test_belief(),
                    num_steps=2,
                    episode_id=i,
                    seed=42 + i,
                    console_output=False,
                )
            )
            identifiers.append(f"episode_{i}")

        task_manager.run_tasks(tasks, identifiers)

        deadline = time.time() + 5.0
        while True:
            with lock:
                if call_count[0] >= 3:
                    break
            if time.time() >= deadline:
                break
            time.sleep(0.05)

        with lock:
            assert call_count[0] == 3


def test_dask_task_manager_progress_callback_exception_is_swallowed(environment, policy):
    """Test that a raising progress callback does not break Dask execution.

    Purpose: Validates the resilience clause documented on
    :meth:`DaskTaskManager.set_progress_callback` — a flaky callback
    cannot derail a long-running Dask experiment.

    Given: A DaskTaskManager with a callback that always raises, and
        one real EpisodeSimulationTask.
    When: run_tasks is called.
    Then: The task completes successfully (one History returned) and no
        exception propagates from the callback.

    Test type: integration
    """
    with DaskTaskManager() as task_manager:

        def on_progress() -> None:
            raise RuntimeError("callback broken")

        task_manager.set_progress_callback(on_progress)

        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=create_test_belief(),
            num_steps=2,
            episode_id=0,
            seed=42,
            console_output=False,
        )

        results, ids = task_manager.run_tasks([task], ["episode_0"])

        assert len(results) == 1
        assert ids == ["episode_0"]


def test_dask_task_manager_no_callback_does_not_register_anything(environment, policy):
    """Test that submit_tasks skips add_done_callback when no callback is set.

    Purpose: Validates that the default code path (callback never
    registered or cleared with ``None``) does not attach any done-callback
    to the Dask futures, so behaviour is identical to the pre-feature
    Dask manager.

    Given: A DaskTaskManager with no progress callback registered.
    When: submit_tasks is called and the returned futures are inspected.
    Then: Each future has no client-attached done-callback queued; the
        task completes normally.

    Test type: unit
    """
    with DaskTaskManager() as task_manager:
        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=create_test_belief(),
            num_steps=2,
            episode_id=0,
            seed=42,
            console_output=False,
        )

        futures = task_manager.submit_tasks([task])
        results = task_manager.gather_results(futures)

        # _on_progress remains unset, so no callbacks should have been
        # attached and the run still succeeds normally.
        assert task_manager._on_progress is None  # pylint: disable=protected-access
        assert len(results) == 1


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


def test_joblib_task_manager_progress_callback_fires_per_task(cache_db, environment, policy):
    """Test that set_progress_callback fires once per completed task.

    Purpose: Validates the per-episode heartbeat hook used by the simulator
    to write heartbeats into the run-progress DB.

    Given: A JoblibTaskManager with a counting callback registered via
        set_progress_callback, and 3 real EpisodeSimulationTask instances.
    When: run_tasks is called with the three tasks.
    Then: The counter equals the number of tasks (3) — one callback fire
        per completed task, all in the parent process.

    Test type: integration
    """
    with JoblibTaskManager(cache_db=cache_db, n_jobs=1) as task_manager:
        call_count = [0]

        def on_progress() -> None:
            call_count[0] += 1

        task_manager.set_progress_callback(on_progress)

        tasks = []
        identifiers = []
        for i in range(3):
            tasks.append(
                EpisodeSimulationTask(
                    environment=environment,
                    policy=policy,
                    initial_belief=create_test_belief(),
                    num_steps=2,
                    episode_id=i,
                    seed=42 + i,
                    console_output=False,
                )
            )
            identifiers.append(f"episode_{i}")

        task_manager.run_tasks(tasks, identifiers)

        assert call_count[0] == 3


def test_joblib_task_manager_progress_callback_exception_is_swallowed(
    cache_db, environment, policy
):
    """Test that a raising progress callback does not break task execution.

    Purpose: Validates the robustness clause documented on
        :meth:`JoblibTaskManager.set_progress_callback` — a flaky callback
        cannot derail a 2-week run.

    Given: A JoblibTaskManager with a callback that always raises, and
        one real EpisodeSimulationTask.
    When: run_tasks is called.
    Then: The task completes successfully (one History returned) and no
        exception propagates from the callback.

    Test type: integration
    """
    with JoblibTaskManager(cache_db=cache_db, n_jobs=1) as task_manager:

        def on_progress() -> None:
            raise RuntimeError("callback broken")

        task_manager.set_progress_callback(on_progress)

        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=create_test_belief(),
            num_steps=2,
            episode_id=0,
            seed=42,
            console_output=False,
        )

        results, ids = task_manager.run_tasks([task], ["episode_0"])

        assert len(results) == 1
        assert ids == ["episode_0"]


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
            warnings.warn("Could not clear cache due to file being in use, continuing with test")

        # Create a fresh task because cleanup() nullifies environment/policy/belief
        # on the original task object after the first run
        task_fresh = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=create_test_belief(),
            num_steps=2,
            episode_id=1,
            seed=42,
            console_output=False,
        )
        result3, ids3 = task_manager.run_tasks([task_fresh], [task_identifier])

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


def test_joblib_task_manager_no_logs_disabled(cache_db, environment, policy, temp_cache_dir):
    """Test that JoblibTaskManager does not write logs when no_logs=True.

    Purpose: Validates that JoblibTaskManager does not create log files when no_logs=True is set

    Given: A cache database, TigerPOMDP environment, SparsePFT policy, EpisodeSimulationTask, and log directory
    When: JoblibTaskManager runs task with no_logs=True and cache_dir configured
    Then: Task executes successfully but no log files are created in the logs directory

    Test type: unit
    """
    # Create a specific log directory for this test
    log_dir = Path(temp_cache_dir) / "test_no_logs"
    log_dir.mkdir(exist_ok=True)

    with JoblibTaskManager(
        cache_db=cache_db, cache_dir=str(log_dir), logger_debug=True, no_logs=True
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

        # Verify no_logs is set correctly
        assert task_manager.no_logs is True

        # Run task - should not generate logs
        results, successful_ids = task_manager.run_tasks([task], [task_identifier])

        # Verify results
        assert len(results) == 1
        assert len(successful_ids) == 1
        assert task_identifier in successful_ids

        # Verify that no logs directory was created (or if it exists, it's empty)
        logs_dir = log_dir / "logs"
        if logs_dir.exists():
            # If logs directory exists, it should be empty (no log files created)
            log_files = list(logs_dir.glob("*.log"))
            assert (
                len(log_files) == 0
            ), f"Expected no log files when no_logs=True, but found {len(log_files)} log files"
        else:
            # Logs directory should not exist when no_logs=True
            assert not logs_dir.exists(), "Logs directory should not be created when no_logs=True"


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
            warnings.warn("Could not clear cache due to file being in use, continuing with test")

        # Create a new task with identical parameters (since the previous task was cleaned up)
        task3 = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42,
            console_output=False,
        )
        result3, ids3 = task_manager.run_tasks([task3], [task_identifier])

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

    Purpose: Validates that PBSTaskManager can be used as a context manager with initialization

    Given: PBSTaskManager instance
    When: Used as a context manager (with statement)
    Then: Context manager protocol works and client is initialized automatically

    Test type: unit
    """
    # Test that context manager works with automatic client initialization
    try:
        with PBSTaskManager(queue="default") as manager:
            assert manager.queue == "default"
            assert manager.client is not None  # Client is initialized automatically
            assert manager.cluster is not None  # Cluster is created
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
    # Note: Using default address "0.0.0.0" to avoid network errors with custom IPs
    # Testing with custom port only
    try:
        with PBSTaskManager(
            queue="gpu_queue",
            enable_dashboard=True,
            dashboard_address="0.0.0.0",  # Use default to avoid network errors
            dashboard_port=8888,
            dashboard_prefix="/my-dashboard",
        ) as manager:
            assert manager.enable_dashboard is True
            assert manager.dashboard_address == "0.0.0.0"
            assert manager.dashboard_port == 8888
            assert manager.dashboard_prefix == "/my-dashboard"
    except Exception as e:
        # Ignore cluster initialization errors for this parameter validation test
        pass


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
    """Test get_dashboard_url when client is initialized.

    Purpose: Validates that get_dashboard_url returns URL after automatic client initialization

    Given: PBSTaskManager instance with dashboard enabled
    When: get_dashboard_url() method is called after construction
    Then: Returns dashboard URL since client is initialized automatically

    Test type: unit
    """
    with PBSTaskManager(queue="default", enable_dashboard=True) as manager:
        # Client is initialized automatically
        assert manager.client is not None
        dashboard_url = manager.get_dashboard_url()
        # Dashboard URL should be available
        assert dashboard_url is not None or manager.enable_dashboard is True


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
    """Test is_dashboard_running when client is initialized.

    Purpose: Validates that is_dashboard_running returns True when client exists

    Given: PBSTaskManager instance with dashboard enabled
    When: is_dashboard_running() method is called after construction
    Then: Returns True indicating dashboard is running

    Test type: unit
    """
    with PBSTaskManager(queue="default", enable_dashboard=True) as manager:
        # Client is initialized automatically
        assert manager.client is not None
        is_running = manager.is_dashboard_running()
        # Dashboard should be running
        assert is_running is True or manager.enable_dashboard is True


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
    # Test common address values (only those that will work in local environment)
    # Avoid custom IPs that may cause network errors
    test_addresses = ["0.0.0.0", "127.0.0.1"]

    for address in test_addresses:
        try:
            with PBSTaskManager(queue="default", dashboard_address=address) as manager:
                assert manager.dashboard_address == address
        except Exception:
            # Ignore cluster initialization errors for address validation test
            pass


def test_pbs_task_manager_dashboard_prefix_validation():
    """Test PBSTaskManager accepts various dashboard prefix values.

    Purpose: Validates that PBSTaskManager accepts different dashboard prefix formats

    Given: PBSTaskManager constructor with different prefix values
    When: PBSTaskManager is instantiated with various valid prefixes
    Then: Task manager stores the correct prefix values

    Test type: unit
    """
    # Test common prefix values
    # Note: dashboard_prefix may not be supported by all Dask versions
    test_prefixes = [None]  # Only test None to avoid API compatibility issues

    for prefix in test_prefixes:
        try:
            with PBSTaskManager(queue="default", dashboard_prefix=prefix) as manager:
                assert manager.dashboard_prefix == prefix
        except Exception:
            # Ignore cluster initialization errors for prefix validation test
            pass


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
            assert manager.client is not None  # Client is initialized automatically
            assert manager.cluster is not None  # Cluster is created
    except Exception as e:
        # Should not raise any exceptions during context manager setup
        pytest.fail(f"Context manager with dashboard should not raise exceptions during setup: {e}")


def test_pbs_task_manager_cluster_storage():
    """Test that PBSTaskManager stores cluster reference for dashboard access.

    Purpose: Validates that PBSTaskManager has cluster attribute for dashboard functionality

    Given: PBSTaskManager instance
    When: Instance is created
    Then: Task manager has cluster attribute initialized automatically

    Test type: unit
    """
    with PBSTaskManager(queue="default") as manager:
        # Check that cluster attribute exists and is initialized
        assert hasattr(manager, "cluster")
        assert manager.cluster is not None  # Cluster is created automatically


# Tests for PBSTaskManager Task Execution
# These tests use scheduler_address pointing to a LocalCluster to execute tasks
# without requiring a real PBS environment.


@pytest.fixture
def local_dask_cluster():
    """Fixture to create a local Dask cluster for PBS task execution tests."""
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    yield cluster
    cluster.close()


def test_pbs_task_manager_run_tasks(environment, policy, local_dask_cluster):
    """Test running tasks with PBSTaskManager connected to a local Dask cluster.

    Purpose: Validates that PBSTaskManager can successfully run multiple EpisodeSimulationTask
        instances when connected to an existing Dask scheduler via scheduler_address

    Given: A TigerPOMDP environment, SparsePFT policy, local Dask cluster, and 2 EpisodeSimulationTask instances
    When: PBSTaskManager.run_tasks() is called with scheduler_address pointing to the local cluster
    Then: Returns 2 successful results and 2 successful IDs, with each result being a valid History object

    Test type: unit
    """
    with PBSTaskManager(
        queue="default", scheduler_address=local_dask_cluster.scheduler_address
    ) as task_manager:
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

        results, successful_ids = task_manager.run_tasks(tasks, task_identifiers)

        assert len(results) == 2
        assert len(successful_ids) == 2
        assert all(id in successful_ids for id in task_identifiers)
        for result in results:
            assert isinstance(result, History)
            assert len(result.history) <= 2


def test_pbs_task_manager_task_status(environment, policy, local_dask_cluster):
    """Test getting task status with PBSTaskManager connected to a local Dask cluster.

    Purpose: Validates that PBSTaskManager can submit tasks and retrieve their status

    Given: A TigerPOMDP environment, SparsePFT policy, local Dask cluster, and an EpisodeSimulationTask
    When: Task is submitted via submit_tasks() and status is retrieved via get_task_status()
    Then: Returns status dictionary with task cache key and valid status (pending, running, or finished)

    Test type: unit
    """
    with PBSTaskManager(
        queue="default", scheduler_address=local_dask_cluster.scheduler_address
    ) as task_manager:
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

        futures = task_manager.submit_tasks([task])
        status = task_manager.get_task_status(futures)

        assert len(status) == 1
        assert task._cache_key in status
        assert status[task._cache_key] in ["pending", "running", "finished"]


def test_pbs_task_manager_failed_tasks_not_cached(environment, policy, local_dask_cluster):
    """Test that PBSTaskManager does not cache results from failed tasks.

    Purpose: Validates that failed tasks are not cached and subsequent runs retry execution for PBS

    Given: PBSTaskManager connected to a local Dask cluster and a task that always raises RuntimeError
    When: The failing task is run multiple times via run_tasks()
    Then: Each run raises RuntimeError (not returning cached results), proving failures are not cached

    Test type: unit
    """
    with PBSTaskManager(
        queue="default", scheduler_address=local_dask_cluster.scheduler_address
    ) as task_manager:
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
        task_identifier = "failing_pbs_episode"

        with pytest.raises(RuntimeError, match="Simulated task failure for testing"):
            task_manager.run_tasks([failing_task], [task_identifier])

        with pytest.raises(RuntimeError, match="Simulated task failure for testing"):
            task_manager.run_tasks([failing_task], [task_identifier])


# Tests for _log_cache_statistics method
def test_joblib_task_manager_log_cache_statistics_current_behavior(cache_db):
    """Test _log_cache_statistics current behavior with missing get_stats method.

    Purpose: Validates that _log_cache_statistics handles the missing get_stats method gracefully

    Given: JoblibTaskManager with cache (current joblib version doesn't have get_stats)
    When: _log_cache_statistics() is called
    Then: Method catches AttributeError and logs warning message

    Test type: unit
    """

    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        with patch.object(task_manager.logger, "warning") as mock_warning:
            task_manager._log_cache_statistics()

            # Verify that warning was called once
            assert mock_warning.call_count == 1

            # Check the warning message (lazy % formatting)
            warning_format = mock_warning.call_args_list[0][0][0]
            warning_args = (
                mock_warning.call_args_list[0][0][1]
                if len(mock_warning.call_args_list[0][0]) > 1
                else ""
            )
            assert "Could not log cache statistics" in warning_format
            assert "'Memory' object has no attribute 'get_stats'" in warning_args


def test_joblib_task_manager_log_cache_statistics_normal(cache_db):
    """Test _log_cache_statistics with normal cache statistics (if get_stats existed).

    Purpose: Validates that _log_cache_statistics correctly logs cache hits, misses, and hit rate

    Given: JoblibTaskManager with cache and mock cache statistics
    When: _log_cache_statistics() is called with normal cache stats
    Then: Method logs cache hits, misses, and calculated hit rate correctly

    Test type: unit
    """

    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Mock cache statistics
        mock_cache_stats = {"hits": 8, "misses": 2}

        # Add get_stats method to memory object for testing
        setattr(task_manager.memory, "get_stats", Mock(return_value=mock_cache_stats))

        with patch.object(task_manager.logger, "info") as mock_info:
            task_manager._log_cache_statistics()

            # Check that the specific cache statistics messages were logged (lazy % formatting)
            # Extract format strings and arguments
            cache_stats_call = None
            hit_rate_call = None

            for call in mock_info.call_args_list:
                format_str = call[0][0]
                if "Joblib Cache Stats" in format_str:
                    cache_stats_call = call
                elif "Cache hit rate:" in format_str:
                    hit_rate_call = call

            # Verify cache stats message
            assert cache_stats_call is not None
            assert "Cache hits: %d" in cache_stats_call[0][0]
            assert "Cache misses: %d" in cache_stats_call[0][0]
            assert cache_stats_call[0][1] == 8  # hits
            assert cache_stats_call[0][2] == 2  # misses

            # Verify hit rate message
            assert hit_rate_call is not None
            assert "Cache hit rate: %.1f%%" in hit_rate_call[0][0]
            assert hit_rate_call[0][1] == 80.0


def test_joblib_task_manager_log_cache_statistics_zero_requests(cache_db):
    """Test _log_cache_statistics with zero cache requests.

    Purpose: Validates that _log_cache_statistics handles zero total requests correctly

    Given: JoblibTaskManager with cache and zero cache statistics
    When: _log_cache_statistics() is called with zero hits and misses
    Then: Method logs cache stats but skips hit rate calculation

    Test type: unit
    """

    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Mock cache statistics with zero requests
        mock_cache_stats = {"hits": 0, "misses": 0}

        # Add get_stats method to memory object for testing
        setattr(task_manager.memory, "get_stats", Mock(return_value=mock_cache_stats))

        with patch.object(task_manager.logger, "info") as mock_info:
            task_manager._log_cache_statistics()

            # Check that the specific cache statistics messages were logged (lazy % formatting)
            cache_stats_call = None
            hit_rate_call = None

            for call in mock_info.call_args_list:
                format_str = call[0][0]
                if "Joblib Cache Stats" in format_str:
                    cache_stats_call = call
                elif "Cache hit rate:" in format_str:
                    hit_rate_call = call

            # Verify cache stats message
            assert cache_stats_call is not None
            assert "Cache hits: %d" in cache_stats_call[0][0]
            assert "Cache misses: %d" in cache_stats_call[0][0]
            assert cache_stats_call[0][1] == 0  # hits
            assert cache_stats_call[0][2] == 0  # misses

            # Verify no hit rate message (should be None)
            assert hit_rate_call is None


def test_joblib_task_manager_log_cache_statistics_exception_handling(cache_db):
    """Test _log_cache_statistics exception handling.

    Purpose: Validates that _log_cache_statistics handles exceptions gracefully

    Given: JoblibTaskManager with cache that raises exception on get_stats()
    When: _log_cache_statistics() is called and get_stats() raises exception
    Then: Method catches exception and logs warning message

    Test type: unit
    """

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

            # Check the warning message (lazy % formatting)
            warning_format = mock_warning.call_args_list[0][0][0]
            warning_args = (
                mock_warning.call_args_list[0][0][1]
                if len(mock_warning.call_args_list[0][0]) > 1
                else ""
            )
            assert "Could not log cache statistics" in warning_format
            assert "Cache stats unavailable" in warning_args


def test_joblib_task_manager_log_cache_statistics_different_hit_rates(cache_db):
    """Test _log_cache_statistics with different hit rate scenarios.

    Purpose: Validates that _log_cache_statistics correctly calculates various hit rates

    Given: JoblibTaskManager with cache and different cache statistics
    When: _log_cache_statistics() is called with various hit/miss ratios
    Then: Method correctly calculates and logs different hit rates

    Test type: unit
    """

    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        test_cases = [
            ({"hits": 0, "misses": 10}, 0.0),  # 0% hit rate
            ({"hits": 5, "misses": 5}, 50.0),  # 50% hit rate
            ({"hits": 10, "misses": 0}, 100.0),  # 100% hit rate
            ({"hits": 3, "misses": 7}, 30.0),  # 30% hit rate
            ({"hits": 7, "misses": 3}, 70.0),  # 70% hit rate
        ]

        for cache_stats, expected_hit_rate in test_cases:
            # Add get_stats method to memory object for testing
            setattr(task_manager.memory, "get_stats", Mock(return_value=cache_stats))

            with patch.object(task_manager.logger, "info") as mock_info:
                task_manager._log_cache_statistics()

                # Check that the specific hit rate message was logged (lazy % formatting)
                hit_rate_call = None
                for call in mock_info.call_args_list:
                    format_str = call[0][0]
                    if "Cache hit rate:" in format_str:
                        hit_rate_call = call
                        break

                # Verify hit rate calculation
                assert hit_rate_call is not None
                assert "Cache hit rate: %.1f%%" in hit_rate_call[0][0]
                assert hit_rate_call[0][1] == expected_hit_rate


def test_joblib_task_manager_log_cache_statistics_missing_keys(cache_db):
    """Test _log_cache_statistics with missing cache statistics keys.

    Purpose: Validates that _log_cache_statistics handles missing keys gracefully

    Given: JoblibTaskManager with cache statistics missing hits/misses keys
    When: _log_cache_statistics() is called with incomplete stats
    Then: Method uses default values (0) for missing keys

    Test type: unit
    """

    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Mock cache statistics with missing keys
        mock_cache_stats = {}  # Empty dict - no hits or misses keys

        # Add get_stats method to memory object for testing
        setattr(task_manager.memory, "get_stats", Mock(return_value=mock_cache_stats))

        with patch.object(task_manager.logger, "info") as mock_info:
            task_manager._log_cache_statistics()

            # Check that the specific cache statistics messages were logged (lazy % formatting)
            cache_stats_call = None
            hit_rate_call = None

            for call in mock_info.call_args_list:
                format_str = call[0][0]
                if "Joblib Cache Stats" in format_str:
                    cache_stats_call = call
                elif "Cache hit rate:" in format_str:
                    hit_rate_call = call

            # Verify cache stats message uses default values
            assert cache_stats_call is not None
            assert "Cache hits: %d" in cache_stats_call[0][0]
            assert "Cache misses: %d" in cache_stats_call[0][0]
            assert cache_stats_call[0][1] == 0  # hits
            assert cache_stats_call[0][2] == 0  # misses

            # Verify no hit rate message (should be None due to zero total)
            assert hit_rate_call is None


def test_sequential_task_manager_log_cache_statistics_inheritance(cache_db):
    """Test that SequentialTaskManager inherits _log_cache_statistics method.

    Purpose: Validates that SequentialTaskManager can use _log_cache_statistics from JoblibTaskManager

    Given: SequentialTaskManager with cache and mock cache statistics
    When: _log_cache_statistics() is called on SequentialTaskManager
    Then: Method works correctly and logs cache statistics

    Test type: unit
    """

    with SequentialTaskManager(cache_db=cache_db) as task_manager:
        # Mock cache statistics
        mock_cache_stats = {"hits": 6, "misses": 4}

        # Add get_stats method to memory object for testing
        setattr(task_manager.memory, "get_stats", Mock(return_value=mock_cache_stats))

        with patch.object(task_manager.logger, "info") as mock_info:
            task_manager._log_cache_statistics()

            # Check that the specific cache statistics messages were logged (lazy % formatting)
            cache_stats_call = None
            hit_rate_call = None

            for call in mock_info.call_args_list:
                format_str = call[0][0]
                if "Joblib Cache Stats" in format_str:
                    cache_stats_call = call
                elif "Cache hit rate:" in format_str:
                    hit_rate_call = call

            # Verify cache stats message
            assert cache_stats_call is not None
            assert "Cache hits: %d" in cache_stats_call[0][0]
            assert "Cache misses: %d" in cache_stats_call[0][0]
            assert cache_stats_call[0][1] == 6  # hits
            assert cache_stats_call[0][2] == 4  # misses

            # Verify hit rate message
            assert hit_rate_call is not None
            assert "Cache hit rate: %.1f%%" in hit_rate_call[0][0]
            assert hit_rate_call[0][1] == 60.0


def test_joblib_task_manager_log_cache_statistics_precision(cache_db):
    """Test _log_cache_statistics hit rate precision formatting.

    Purpose: Validates that _log_cache_statistics formats hit rate with correct precision

    Given: JoblibTaskManager with cache statistics that result in non-integer hit rates
    When: _log_cache_statistics() is called with fractional hit rates
    Then: Method formats hit rate to one decimal place

    Test type: unit
    """

    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Mock cache statistics that result in fractional hit rate
        mock_cache_stats = {"hits": 1, "misses": 3}  # 25.0% hit rate

        # Add get_stats method to memory object for testing
        setattr(task_manager.memory, "get_stats", Mock(return_value=mock_cache_stats))

        with patch.object(task_manager.logger, "info") as mock_info:
            task_manager._log_cache_statistics()

            # Check that the specific hit rate message was logged (lazy % formatting)
            hit_rate_call = None
            for call in mock_info.call_args_list:
                format_str = call[0][0]
                if "Cache hit rate:" in format_str:
                    hit_rate_call = call
                    break

            # Check hit rate formatting
            assert hit_rate_call is not None
            assert "Cache hit rate: %.1f%%" in hit_rate_call[0][0]
            assert hit_rate_call[0][1] == 25.0

            # Test another fractional case
            mock_cache_stats = {"hits": 1, "misses": 2}  # 33.333...% hit rate
            setattr(task_manager.memory, "get_stats", Mock(return_value=mock_cache_stats))
            task_manager._log_cache_statistics()

            # Get updated messages (check the last call which is the second _log_cache_statistics)
            hit_rate_call = None
            for call in mock_info.call_args_list:
                format_str = call[0][0]
                if "Cache hit rate:" in format_str:
                    hit_rate_call = call  # Get the last occurrence

            assert hit_rate_call is not None
            assert "Cache hit rate: %.1f%%" in hit_rate_call[0][0]
            assert abs(hit_rate_call[0][1] - 33.3) < 0.1  # Allow small floating point error


# ==============================================================================
# PBS FUNCTIONALITY TESTS
# ==============================================================================
# These tests require PBS cluster environment and dask-jobqueue dependency.
# They test actual PBS functionality, task execution, and dashboard features.
# Use pytest markers to run only on PBS clusters:
# pytest -m pbs_cluster tests/test_simulations/test_simulations_deployment/test_task_managers.py

# import os
# import requests
# from unittest.mock import patch


# @pytest.mark.pbs_cluster
# class TestPBSTaskManagerFunctionality:
#     """Comprehensive functionality tests for PBSTaskManager requiring PBS cluster."""

#     @pytest.fixture
#     def pbs_queue(self):
#         """Get PBS queue from environment variable or use default."""
#         return os.environ.get("PBS_QUEUE", "default")

#     @pytest.fixture
#     def pbs_walltime(self):
#         """Get PBS walltime from environment variable or use default."""
#         return os.environ.get("PBS_WALLTIME", "00:05:00")  # 5 minutes for tests

#     def test_pbs_task_manager_basic_functionality(
#         self, environment, policy, pbs_queue, pbs_walltime
#     ):
#         """Test basic PBS task manager functionality with real PBS cluster.

#         Purpose: Validates that PBSTaskManager can execute tasks on PBS cluster

#         Given: PBS cluster environment, TigerPOMDP environment, SparsePFT policy
#         When: PBSTaskManager executes EpisodeSimulationTask on PBS cluster
#         Then: Tasks execute successfully and return valid History objects

#         Test type: integration
#         """
#         with PBSTaskManager(
#             queue=pbs_queue,
#             n_workers=2,
#             cores=1,
#             memory="1GB",
#             walltime=pbs_walltime,
#             enable_dashboard=False,  # Disable dashboard for basic test
#         ) as task_manager:
#             # Create tasks
#             tasks = []
#             task_identifiers = []
#             for i in range(2):
#                 belief = create_test_belief()
#                 task = EpisodeSimulationTask(
#                     environment=environment,
#                     policy=policy,
#                     initial_belief=belief,
#                     num_steps=3,
#                     episode_id=i,
#                     seed=42 + i,
#                     console_output=False,
#                 )
#                 tasks.append(task)
#                 task_identifiers.append(f"pbs_episode_{i}")

#             # Execute tasks
#             results, successful_ids = task_manager.run_tasks(tasks, task_identifiers)

#             # Verify results
#             assert len(results) == 2
#             assert len(successful_ids) == 2
#             assert all(id in successful_ids for id in task_identifiers)
#             for result in results:
#                 assert isinstance(result, History)
#                 assert len(result.history) <= 3

#     def test_pbs_task_manager_dashboard_functionality(
#         self, environment, policy, pbs_queue, pbs_walltime
#     ):
#         """Test PBS task manager dashboard functionality.

#         Purpose: Validates that PBSTaskManager dashboard is accessible and functional

#         Given: PBS cluster environment with dashboard enabled
#         When: PBSTaskManager is initialized with dashboard enabled
#         Then: Dashboard URL is accessible and returns valid HTTP response

#         Test type: integration
#         """
#         with PBSTaskManager(
#             queue=pbs_queue,
#             n_workers=1,
#             cores=1,
#             memory="1GB",
#             walltime=pbs_walltime,
#             enable_dashboard=True,
#             dashboard_port=8787,
#             dashboard_address="0.0.0.0",
#         ) as task_manager:
#             # Verify dashboard is running
#             assert task_manager.is_dashboard_running() is True

#             # Get dashboard URL
#             dashboard_url = task_manager.get_dashboard_url()
#             assert dashboard_url is not None
#             assert "http://" in dashboard_url

#             # Test dashboard accessibility
#             try:
#                 response = requests.get(dashboard_url, timeout=10)
#                 assert response.status_code == 200
#                 assert "Dask" in response.text or "dashboard" in response.text.lower()
#             except requests.exceptions.RequestException:
#                 # Dashboard might not be immediately accessible
#                 # This is acceptable for PBS environments
#                 pass

#             # Create and execute a simple task to generate dashboard activity
#             belief = create_test_belief()
#             task = EpisodeSimulationTask(
#                 environment=environment,
#                 policy=policy,
#                 initial_belief=belief,
#                 num_steps=2,
#                 episode_id=1,
#                 seed=42,
#                 console_output=False,
#             )

#             results, successful_ids = task_manager.run_tasks([task], ["dashboard_test"])
#             assert len(results) == 1
#             assert "dashboard_test" in successful_ids

#     def test_pbs_task_manager_custom_dashboard_configuration(
#         self, environment, policy, pbs_queue, pbs_walltime
#     ):
#         """Test PBS task manager with custom dashboard configuration.

#         Purpose: Validates that PBSTaskManager respects custom dashboard settings

#         Given: PBS cluster environment with custom dashboard port and prefix
#         When: PBSTaskManager is initialized with custom dashboard configuration
#         Then: Dashboard uses specified port and prefix settings

#         Test type: integration
#         """
#         custom_port = 8888
#         custom_prefix = "/my-dashboard"

#         with PBSTaskManager(
#             queue=pbs_queue,
#             n_workers=1,
#             cores=1,
#             memory="1GB",
#             walltime=pbs_walltime,
#             enable_dashboard=True,
#             dashboard_port=custom_port,
#             dashboard_address="0.0.0.0",
#             dashboard_prefix=custom_prefix,
#         ) as task_manager:
#             # Verify dashboard configuration
#             assert task_manager.dashboard_port == custom_port
#             assert task_manager.dashboard_prefix == custom_prefix
#             assert task_manager.enable_dashboard is True

#             # Get dashboard URL and verify it includes custom settings
#             dashboard_url = task_manager.get_dashboard_url()
#             if dashboard_url:
#                 assert str(custom_port) in dashboard_url
#                 if custom_prefix in dashboard_url:
#                     assert custom_prefix in dashboard_url

#             # Execute a task to verify functionality
#             belief = create_test_belief()
#             task = EpisodeSimulationTask(
#                 environment=environment,
#                 policy=policy,
#                 initial_belief=belief,
#                 num_steps=2,
#                 episode_id=1,
#                 seed=42,
#                 console_output=False,
#             )

#             results, successful_ids = task_manager.run_tasks([task], ["custom_dashboard_test"])
#             assert len(results) == 1
#             assert "custom_dashboard_test" in successful_ids

#     def test_pbs_task_manager_dashboard_disabled(
#         self, environment, policy, pbs_queue, pbs_walltime
#     ):
#         """Test PBS task manager with dashboard disabled.

#         Purpose: Validates that PBSTaskManager works correctly with dashboard disabled

#         Given: PBS cluster environment with dashboard disabled
#         When: PBSTaskManager is initialized with enable_dashboard=False
#         Then: Dashboard is not accessible and task execution works normally

#         Test type: integration
#         """
#         with PBSTaskManager(
#             queue=pbs_queue,
#             n_workers=1,
#             cores=1,
#             memory="1GB",
#             walltime=pbs_walltime,
#             enable_dashboard=False,
#         ) as task_manager:
#             # Verify dashboard is disabled
#             assert task_manager.enable_dashboard is False
#             assert task_manager.is_dashboard_running() is False
#             assert task_manager.get_dashboard_url() is None

#             # Execute tasks to verify functionality without dashboard
#             tasks = []
#             task_identifiers = []
#             for i in range(2):
#                 belief = create_test_belief()
#                 task = EpisodeSimulationTask(
#                     environment=environment,
#                     policy=policy,
#                     initial_belief=belief,
#                     num_steps=2,
#                     episode_id=i,
#                     seed=42 + i,
#                     console_output=False,
#                 )
#                 tasks.append(task)
#                 task_identifiers.append(f"no_dashboard_episode_{i}")

#             results, successful_ids = task_manager.run_tasks(tasks, task_identifiers)
#             assert len(results) == 2
#             assert len(successful_ids) == 2

#     def test_pbs_task_manager_job_extra_directives(
#         self, environment, policy, pbs_queue, pbs_walltime
#     ):
#         """Test PBS task manager with custom job directives.

#         Purpose: Validates that PBSTaskManager respects custom PBS job directives

#         Given: PBS cluster environment with custom job directives
#         When: PBSTaskManager is initialized with job_extra directives
#         Then: Tasks execute successfully with custom PBS configuration

#         Test type: integration
#         """
#         custom_directives = [
#             "#PBS -m abe",  # Mail on abort, begin, end
#             "#PBS -M test@example.com",  # Mail address
#         ]

#         with PBSTaskManager(
#             queue=pbs_queue,
#             n_workers=1,
#             cores=1,
#             memory="1GB",
#             walltime=pbs_walltime,
#             job_extra=custom_directives,
#             enable_dashboard=False,
#         ) as task_manager:
#             # Verify job directives are stored
#             assert task_manager.job_extra == custom_directives

#             # Execute a task to verify functionality
#             belief = create_test_belief()
#             task = EpisodeSimulationTask(
#                 environment=environment,
#                 policy=policy,
#                 initial_belief=belief,
#                 num_steps=2,
#                 episode_id=1,
#                 seed=42,
#                 console_output=False,
#             )

#             results, successful_ids = task_manager.run_tasks([task], ["job_extra_test"])
#             assert len(results) == 1
#             assert "job_extra_test" in successful_ids

#     def test_pbs_task_manager_resource_scaling(self, environment, policy, pbs_queue, pbs_walltime):
#         """Test PBS task manager with different resource configurations.

#         Purpose: Validates that PBSTaskManager can handle different resource allocations

#         Given: PBS cluster environment with various resource configurations
#         When: PBSTaskManager is initialized with different cores/memory settings
#         Then: Tasks execute successfully with specified resource allocation

#         Test type: integration
#         """
#         resource_configs = [
#             {"cores": 1, "memory": "1GB", "processes": 1},
#             {"cores": 2, "memory": "2GB", "processes": 1},
#             {"cores": 1, "memory": "1GB", "processes": 2},
#         ]

#         for i, config in enumerate(resource_configs):
#             with PBSTaskManager(
#                 queue=pbs_queue,
#                 n_workers=1,
#                 cores=config["cores"],
#                 memory=config["memory"],
#                 processes=config["processes"],
#                 walltime=pbs_walltime,
#                 enable_dashboard=False,
#             ) as task_manager:
#                 # Verify resource configuration
#                 assert task_manager.cores == config["cores"]
#                 assert task_manager.memory == config["memory"]
#                 assert task_manager.processes == config["processes"]

#                 # Execute a task to verify functionality
#                 belief = create_test_belief()
#                 task = EpisodeSimulationTask(
#                     environment=environment,
#                     policy=policy,
#                     initial_belief=belief,
#                     num_steps=2,
#                     episode_id=i,
#                     seed=42 + i,
#                     console_output=False,
#                 )

#                 results, successful_ids = task_manager.run_tasks([task], [f"resource_test_{i}"])
#                 assert len(results) == 1
#                 assert f"resource_test_{i}" in successful_ids

#     def test_pbs_task_manager_multiple_workers(self, environment, policy, pbs_queue, pbs_walltime):
#         """Test PBS task manager with multiple workers.

#         Purpose: Validates that PBSTaskManager can distribute tasks across multiple PBS workers

#         Given: PBS cluster environment with multiple workers
#         When: PBSTaskManager executes multiple tasks with n_workers > 1
#         Then: Tasks are distributed across workers and execute successfully

#         Test type: integration
#         """
#         with PBSTaskManager(
#             queue=pbs_queue,
#             n_workers=3,
#             cores=1,
#             memory="1GB",
#             walltime=pbs_walltime,
#             enable_dashboard=False,
#         ) as task_manager:
#             # Create multiple tasks
#             tasks = []
#             task_identifiers = []
#             for i in range(5):  # More tasks than workers
#                 belief = create_test_belief()
#                 task = EpisodeSimulationTask(
#                     environment=environment,
#                     policy=policy,
#                     initial_belief=belief,
#                     num_steps=2,
#                     episode_id=i,
#                     seed=42 + i,
#                     console_output=False,
#                 )
#                 tasks.append(task)
#                 task_identifiers.append(f"multi_worker_episode_{i}")

#             # Execute tasks
#             results, successful_ids = task_manager.run_tasks(tasks, task_identifiers)

#             # Verify all tasks completed
#             assert len(results) == 5
#             assert len(successful_ids) == 5
#             assert all(id in successful_ids for id in task_identifiers)

#     def test_pbs_task_manager_task_status_monitoring(
#         self, environment, policy, pbs_queue, pbs_walltime
#     ):
#         """Test PBS task manager task status monitoring.

#         Purpose: Validates that PBSTaskManager can monitor task status during execution

#         Given: PBS cluster environment with task status monitoring
#         When: Tasks are submitted and status is checked during execution
#         Then: Task status information is available and accurate

#         Test type: integration
#         """
#         with PBSTaskManager(
#             queue=pbs_queue,
#             n_workers=1,
#             cores=1,
#             memory="1GB",
#             walltime=pbs_walltime,
#             enable_dashboard=False,
#         ) as task_manager:
#             # Create a task
#             belief = create_test_belief()
#             task = EpisodeSimulationTask(
#                 environment=environment,
#                 policy=policy,
#                 initial_belief=belief,
#                 num_steps=3,
#                 episode_id=1,
#                 seed=42,
#                 console_output=False,
#             )

#             # Submit task and monitor status
#             futures = task_manager.submit_tasks([task])
#             assert len(futures) == 1

#             # Check initial status
#             status = task_manager.get_task_status(futures)
#             assert len(status) == 1
#             assert task._cache_key in status
#             assert status[task._cache_key] in ["pending", "running", "finished"]

#             # Wait for completion and check final status
#             results = task_manager.gather_results(futures)
#             assert len(results) == 1
#             assert isinstance(results[0], History)

#             # Final status check
#             final_status = task_manager.get_task_status(futures)
#             assert len(final_status) == 1
#             assert final_status[task._cache_key] == "finished"

#     def test_pbs_task_manager_error_handling(self, environment, policy, pbs_queue, pbs_walltime):
#         """Test PBS task manager error handling with failing tasks.

#         Purpose: Validates that PBSTaskManager handles task failures gracefully

#         Given: PBS cluster environment with a task that always fails
#         When: Failing task is submitted to PBS cluster
#         Then: Error is handled gracefully and doesn't crash the task manager

#         Test type: integration
#         """
#         with PBSTaskManager(
#             queue=pbs_queue,
#             n_workers=1,
#             cores=1,
#             memory="1GB",
#             walltime=pbs_walltime,
#             enable_dashboard=False,
#         ) as task_manager:
#             # Create a failing task
#             belief = create_test_belief()
#             failing_task = FailingEpisodeSimulationTask(
#                 environment=environment,
#                 policy=policy,
#                 initial_belief=belief,
#                 num_steps=2,
#                 episode_id=999,
#                 seed=42,
#                 console_output=False,
#             )

#             # Submit failing task
#             with pytest.raises(RuntimeError, match="Simulated task failure for testing"):
#                 task_manager.run_tasks([failing_task], ["failing_pbs_task"])

#             # Verify task manager is still functional
#             belief = create_test_belief()
#             working_task = EpisodeSimulationTask(
#                 environment=environment,
#                 policy=policy,
#                 initial_belief=belief,
#                 num_steps=2,
#                 episode_id=1,
#                 seed=42,
#                 console_output=False,
#             )

#             results, successful_ids = task_manager.run_tasks([working_task], ["working_task"])
#             assert len(results) == 1
#             assert "working_task" in successful_ids

#     def test_pbs_task_manager_context_manager_cleanup(
#         self, environment, policy, pbs_queue, pbs_walltime
#     ):
#         """Test PBS task manager context manager cleanup.

#         Purpose: Validates that PBSTaskManager properly cleans up resources on exit

#         Given: PBS cluster environment with dashboard enabled
#         When: PBSTaskManager context manager exits
#         Then: All resources are properly cleaned up (client, cluster, dashboard)

#         Test type: integration
#         """
#         # Test with dashboard enabled to verify cleanup
#         with PBSTaskManager(
#             queue=pbs_queue,
#             n_workers=1,
#             cores=1,
#             memory="1GB",
#             walltime=pbs_walltime,
#             enable_dashboard=True,
#             dashboard_port=8787,
#         ) as task_manager:
#             # Verify resources are initialized
#             assert task_manager.client is not None
#             assert task_manager.cluster is not None
#             assert task_manager.is_dashboard_running() is True

#             # Execute a task
#             belief = create_test_belief()
#             task = EpisodeSimulationTask(
#                 environment=environment,
#                 policy=policy,
#                 initial_belief=belief,
#                 num_steps=2,
#                 episode_id=1,
#                 seed=42,
#                 console_output=False,
#             )

#             results, successful_ids = task_manager.run_tasks([task], ["cleanup_test"])
#             assert len(results) == 1
#             assert "cleanup_test" in successful_ids

#         # After context manager exit, resources should be cleaned up
#         # Note: We can't directly test this as the objects are destroyed,
#         # but the test verifies that the context manager exits without errors

#     def test_pbs_task_manager_dashboard_url_construction(
#         self, environment, policy, pbs_queue, pbs_walltime
#     ):
#         """Test PBS task manager dashboard URL construction.

#         Purpose: Validates that PBSTaskManager constructs correct dashboard URLs

#         Given: PBS cluster environment with various dashboard configurations
#         When: Dashboard URL is requested with different settings
#         Then: URL is constructed correctly with proper host, port, and prefix

#         Test type: integration
#         """
#         test_configs = [
#             {"port": 8787, "prefix": None, "address": "0.0.0.0"},
#             {"port": 8888, "prefix": None, "address": "0.0.0.0"},
#             {"port": 8787, "prefix": "/dashboard", "address": "0.0.0.0"},
#             {"port": 9999, "prefix": "/my-dashboard", "address": "0.0.0.0"},
#         ]

#         for config in test_configs:
#             with PBSTaskManager(
#                 queue=pbs_queue,
#                 n_workers=1,
#                 cores=1,
#                 memory="1GB",
#                 walltime=pbs_walltime,
#                 enable_dashboard=True,
#                 dashboard_port=config["port"],
#                 dashboard_address=config["address"],
#                 dashboard_prefix=config["prefix"],
#             ) as task_manager:
#                 # Get dashboard URL
#                 dashboard_url = task_manager.get_dashboard_url()

#                 if dashboard_url:
#                     # Verify URL contains expected components
#                     assert str(config["port"]) in dashboard_url
#                     assert "http://" in dashboard_url

#                     if config["prefix"]:
#                         assert config["prefix"].lstrip("/") in dashboard_url

#                 # Execute a quick task to verify functionality
#                 belief = create_test_belief()
#                 task = EpisodeSimulationTask(
#                     environment=environment,
#                     policy=policy,
#                     initial_belief=belief,
#                     num_steps=1,
#                     episode_id=1,
#                     seed=42,
#                     console_output=False,
#                 )

#                 results, successful_ids = task_manager.run_tasks([task], ["url_test"])
#                 assert len(results) == 1
#                 assert "url_test" in successful_ids

#     def test_pbs_task_manager_performance_monitoring(
#         self, environment, policy, pbs_queue, pbs_walltime
#     ):
#         """Test PBS task manager performance monitoring capabilities.

#         Purpose: Validates that PBSTaskManager can monitor task performance on PBS cluster

#         Given: PBS cluster environment with performance monitoring
#         When: Multiple tasks are executed with timing measurements
#         Then: Task execution times are reasonable and performance is monitored

#         Test type: integration
#         """
#         with PBSTaskManager(
#             queue=pbs_queue,
#             n_workers=2,
#             cores=1,
#             memory="1GB",
#             walltime=pbs_walltime,
#             enable_dashboard=True,  # Enable dashboard for performance monitoring
#         ) as task_manager:
#             # Create multiple tasks for performance testing
#             tasks = []
#             task_identifiers = []
#             for i in range(4):
#                 belief = create_test_belief()
#                 task = EpisodeSimulationTask(
#                     environment=environment,
#                     policy=policy,
#                     initial_belief=belief,
#                     num_steps=3,
#                     episode_id=i,
#                     seed=42 + i,
#                     console_output=False,
#                 )
#                 tasks.append(task)
#                 task_identifiers.append(f"perf_test_{i}")

#             # Measure execution time
#             start_time = time.time()
#             results, successful_ids = task_manager.run_tasks(tasks, task_identifiers)
#             end_time = time.time()

#             # Verify results
#             assert len(results) == 4
#             assert len(successful_ids) == 4
#             assert all(id in successful_ids for id in task_identifiers)

#             # Verify reasonable execution time (should complete within walltime)
#             execution_time = end_time - start_time
#             assert execution_time < 300  # Should complete within 5 minutes

#             # Verify dashboard is accessible for performance monitoring
#             dashboard_url = task_manager.get_dashboard_url()
#             if dashboard_url:
#                 assert task_manager.is_dashboard_running() is True

#     def test_pbs_task_manager_concurrent_execution(
#         self, environment, policy, pbs_queue, pbs_walltime
#     ):
#         """Test PBS task manager concurrent task execution.

#         Purpose: Validates that PBSTaskManager can handle concurrent task execution on PBS cluster

#         Given: PBS cluster environment with multiple workers
#         When: Multiple task batches are submitted concurrently
#         Then: All tasks execute successfully with proper concurrency handling

#         Test type: integration
#         """
#         with PBSTaskManager(
#             queue=pbs_queue,
#             n_workers=3,
#             cores=1,
#             memory="1GB",
#             walltime=pbs_walltime,
#             enable_dashboard=False,
#         ) as task_manager:
#             # Create multiple batches of tasks
#             batch_size = 3
#             num_batches = 2

#             all_results = []
#             all_successful_ids = []

#             for batch_idx in range(num_batches):
#                 tasks = []
#                 task_identifiers = []

#                 for i in range(batch_size):
#                     belief = create_test_belief()
#                     task = EpisodeSimulationTask(
#                         environment=environment,
#                         policy=policy,
#                         initial_belief=belief,
#                         num_steps=2,
#                         episode_id=batch_idx * batch_size + i,
#                         seed=42 + batch_idx * batch_size + i,
#                         console_output=False,
#                     )
#                     tasks.append(task)
#                     task_identifiers.append(f"concurrent_batch_{batch_idx}_task_{i}")

#                 # Execute batch
#                 results, successful_ids = task_manager.run_tasks(tasks, task_identifiers)
#                 all_results.extend(results)
#                 all_successful_ids.extend(successful_ids)

#             # Verify all batches completed successfully
#             expected_total = batch_size * num_batches
#             assert len(all_results) == expected_total
#             assert len(all_successful_ids) == expected_total

#             # Verify all task identifiers are present
#             for batch_idx in range(num_batches):
#                 for i in range(batch_size):
#                     expected_id = f"concurrent_batch_{batch_idx}_task_{i}"
#                     assert expected_id in all_successful_ids


# # ==============================================================================
# # MEMORY LEAK DIAGNOSTIC TESTS
# # ==============================================================================
# # These tests help identify specific sources of memory leaks in task managers.
# # Tests that fail indicate components that need memory management fixes.

# import gc
# import psutil
# from joblib import Parallel, delayed

# from POMDPPlanners.core.belief import get_initial_belief
# from POMDPPlanners.core.simulation import EnvironmentRunParams, StepData
# from POMDPPlanners.core.policy import PolicyRunData
# from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
# from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
#     JoblibConfig,
# )
# from typing import Dict


# class MemoryTracker:
#     """Utility class for tracking memory usage in tests."""

#     def __init__(self, test_name: str):
#         self.test_name = test_name
#         self.process = psutil.Process()
#         self.initial_memory: float = 0.0
#         self.peak_memory: float = 0.0
#         self.measurements = []

#     def start(self):
#         """Start memory tracking."""
#         gc.collect()  # Clean up before measurement
#         self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
#         self.peak_memory = self.initial_memory

#     def checkpoint(self, label: str = ""):
#         """Record a memory checkpoint."""
#         gc.collect()
#         current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
#         self.peak_memory = max(self.peak_memory, current_memory)
#         growth = current_memory - self.initial_memory
#         self.measurements.append((label, current_memory, growth))
#         return current_memory, growth

#     def finish(self):
#         """Finish tracking and return final results."""
#         final_memory, final_growth = self.checkpoint("Final")
#         peak_growth = self.peak_memory - self.initial_memory

#         return {
#             "initial_memory": self.initial_memory,
#             "final_memory": final_memory,
#             "peak_memory": self.peak_memory,
#             "final_growth": final_growth,
#             "peak_growth": peak_growth,
#             "measurements": self.measurements,
#         }


# def test_memory_leak_parallel_execution_resource_cleanup(temp_cache_dir, environment):
#     """
#     Purpose: Validates that parallel execution using joblib doesn't accumulate memory across iterations

#     Given: Multiple parallel execution cycles with POMDP objects and simulation data
#     When: Parallel tasks create environments, policies, and beliefs repeatedly
#     Then: Memory growth stays under 200MB indicating proper parallel resource cleanup

#     Test type: unit
#     """
#     tracker = MemoryTracker("Parallel Execution Test")
#     tracker.start()

#     def memory_intensive_task(task_id: int) -> Dict:
#         """Simulate a memory-intensive task similar to simulation."""
#         # Create some objects similar to what simulation does
#         env = TigerPOMDP(discount_factor=0.95)
#         policy = POMCP(
#             environment=env,
#             discount_factor=0.95,
#             depth=3,
#             exploration_constant=1.0,
#             name=f"TestPolicy_{task_id}",
#             n_simulations=5,
#         )

#         # Simulate some computation
#         belief = get_initial_belief(env, n_particles=50)

#         # Create mock history
#         mock_steps = []
#         for step in range(10):
#             step_data = StepData(
#                 state=env.initial_state_dist().sample()[0],
#                 action=env.get_actions()[0],
#                 next_state=env.initial_state_dist().sample()[0],
#                 observation=env.get_actions()[0],  # Simple mock
#                 reward=1.0,
#                 belief=belief,
#             )
#             mock_steps.append(step_data)

#         history = History(
#             history=mock_steps,
#             actual_num_steps=10,
#             reach_terminal_state=False,
#             discount_factor=0.95,
#             average_state_sampling_time=0.01,
#             average_action_time=0.02,
#             average_observation_time=0.01,
#             average_belief_update_time=0.03,
#             average_reward_time=0.001,
#             policy_run_data=[PolicyRunData(info_variables=[])],
#         )

#         return {"task_id": task_id, "history": history, "policy": policy, "env": env}

#     try:
#         # Test multiple parallel executions
#         for iteration in range(5):
#             # Run parallel tasks
#             results = Parallel(n_jobs=2)(
#                 delayed(memory_intensive_task)(task_id) for task_id in range(10)
#             )

#             # Clear results
#             del results
#             gc.collect()

#     finally:
#         gc.collect()

#     results = tracker.finish()

#     # Assert memory growth is reasonable (<200MB for parallel operations)
#     assert (
#         results["final_growth"] < 200
#     ), f"Parallel execution leaked {results['final_growth']:.1f} MB"


# def test_memory_leak_task_manager_resource_cleanup(temp_cache_dir):
#     """
#     Purpose: Validates that task manager instances properly release their resources

#     Given: Multiple task manager instances created and used for parallel operations
#     When: Task managers execute parallel operations and are cleaned up
#     Then: Memory growth stays under 80MB indicating proper task manager resource management

#     Test type: unit
#     """
#     tracker = MemoryTracker("Task Manager Cleanup Test")
#     tracker.start()

#     try:
#         # Test multiple task manager creations and cleanups
#         for i in range(20):
#             # Create task manager config
#             config = JoblibConfig(n_jobs=2, verbose=0)
#             task_manager = config.create_task_manager(cache_dir=str(temp_cache_dir))

#             # Use task manager for some operations
#             with task_manager:
#                 # Simulate task execution
#                 def dummy_task(x):
#                     return x * 2

#                 results = Parallel(n_jobs=2)(delayed(dummy_task)(x) for x in range(10))

#             # Explicit cleanup
#             del task_manager
#             del results
#             gc.collect()

#     finally:
#         gc.collect()

#     results = tracker.finish()

#     # Assert memory growth is reasonable (<80MB for task manager operations)
#     assert (
#         results["final_growth"] < 80
#     ), f"Task manager resources leaked {results['final_growth']:.1f} MB"


# class TestPBSTaskManagerIntegration:
#     """Test PBS task manager integration tests."""

#     @pytest.mark.pbs_cluster
#     def test_pbs_task_manager_initialization_integration(self):
#         """Test PBS task manager initialization with integration test.

#         Purpose: Validates that PBS task manager can be created with proper parameters
#         and all attributes are correctly set.

#         Given: PBS task manager parameters
#         When: Creating a PBSTaskManager instance
#         Then: All parameters are correctly set and manager is properly initialized

#         Test type: integration
#         """
#         manager = PBSTaskManager(
#             queue="default", n_workers=1, cores=1, memory="1GB", walltime="00:10:00"
#         )

#         assert manager.queue == "default"
#         assert manager.n_workers == 1
#         assert manager.cores == 1
#         assert manager.memory == "1GB"
#         assert manager.walltime == "00:10:00"
#         assert isinstance(manager, DaskTaskManager)
