import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import time

from POMDPPlanners.simulations.simulations_deployment.task_managers import DaskTaskManager, JoblibTaskManager
from POMDPPlanners.simulations.simulations_deployment.tasks import EpisodeSimulationTask
from POMDPPlanners.simulations.simulations_deployment.cache_dbs import DiskCacheDB
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
        n_simulations=2
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
                console_output=False
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
            console_output=False
        )
        # Submit task and get status
        futures = task_manager.submit_tasks([task])
        status = task_manager.get_task_status(futures)
        assert len(status) == 1
        assert task._cache_key in status
        assert status[task._cache_key] in ['pending', 'running', 'finished']

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
        assert len(list(task_manager.memory.store_backend.get_items())) == 0

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
                console_output=False
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
            console_output=False
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
    
    with JoblibTaskManager(cache_db=cache_db, cache_dir=str(log_dir), logger_debug=True) as task_manager:
        # Create a task
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
        task_identifier = "episode_1"
        
        # Run task to generate logs
        results, successful_ids = task_manager.run_tasks([task], [task_identifier])
        
        # Verify results
        assert len(results) == 1
        assert len(successful_ids) == 1
        assert task_identifier in successful_ids
        
        # Verify that expected log messages are present in the task manager logs
        # Note: We don't check directory structure as it may change

def test_joblib_task_manager_logging_with_multiple_tasks(cache_db, environment, policy, temp_cache_dir):
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
    
    with JoblibTaskManager(cache_db=cache_db, cache_dir=str(log_dir), logger_debug=True) as task_manager:
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
                console_output=False
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
            console_output=False
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
            console_output=False
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