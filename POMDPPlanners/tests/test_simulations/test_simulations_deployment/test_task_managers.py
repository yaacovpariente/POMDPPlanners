import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

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
    shutil.rmtree(temp_dir)

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
    db.close()

# Tests for DaskTaskManager
def test_dask_task_manager_initialization():
    """Test DaskTaskManager initialization and cleanup (no cache_db)."""
    with DaskTaskManager() as task_manager:
        assert task_manager.client is not None
        assert task_manager.cache is not None
        assert not task_manager.cache_registered  # Cache not registered by default

def test_dask_task_manager_with_cache_clear():
    """Test DaskTaskManager initialization with cache clearing (no cache_db)."""
    with DaskTaskManager(clear_cache_on_start=True) as task_manager:
        assert task_manager.client is not None
        assert task_manager.cache is not None
        assert not task_manager.cache_registered  # Cache not registered when cleared

def test_dask_task_manager_run_tasks(environment, policy):
    """Test running tasks with DaskTaskManager (no cache_db)."""
    with DaskTaskManager() as task_manager:
        # Create multiple tasks
        tasks = []
        for i in range(2):
            belief = create_test_belief()
            task = EpisodeSimulationTask(
                environment=environment,
                policy=policy,
                initial_belief=belief,
                num_steps=2,
                episode_id=i,
                seed=42 + i
            )
            tasks.append(task)
        # Run tasks
        results = task_manager.run_tasks(tasks)
        # Verify results
        assert len(results) == 2
        for result in results:
            assert isinstance(result, History)
            assert len(result.history) <= 2

def test_dask_task_manager_task_status(environment, policy):
    """Test getting task status with DaskTaskManager (no cache_db)."""
    with DaskTaskManager() as task_manager:
        # Create and submit a task
        belief = create_test_belief()
        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42
        )
        # Submit task and get status
        futures = task_manager.submit_tasks([task])
        status = task_manager.get_task_status(futures)
        assert len(status) == 1
        assert task._cache_key in status
        assert status[task._cache_key] in ['pending', 'running', 'finished']

# Tests for JoblibTaskManager
def test_joblib_task_manager_initialization(cache_db):
    """Test JoblibTaskManager initialization."""
    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        assert task_manager.n_jobs == -1  # Default to all cores
        assert task_manager.verbose == 0
        assert task_manager.memory is not None

def test_joblib_task_manager_with_cache_clear(cache_db):
    """Test JoblibTaskManager initialization with cache clearing."""
    with JoblibTaskManager(cache_db=cache_db, clear_cache_on_start=True) as task_manager:
        assert task_manager.memory is not None
        # Cache should be empty after clearing
        assert len(list(task_manager.memory.store_backend.get_items())) == 0

def test_joblib_task_manager_run_tasks(cache_db, environment, policy):
    """Test running tasks with JoblibTaskManager."""
    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Create multiple tasks
        tasks = []
        for i in range(2):
            belief = create_test_belief()
            task = EpisodeSimulationTask(
                environment=environment,
                policy=policy,
                initial_belief=belief,
                num_steps=2,
                episode_id=i,
                seed=42 + i
            )
            tasks.append(task)
        
        # Run tasks
        results = task_manager.run_tasks(tasks)
        
        # Verify results
        assert len(results) == 2
        for result in results:
            assert isinstance(result, History)
            assert len(result.history) <= 2

def test_joblib_task_manager_cache(cache_db, environment, policy):
    """Test caching behavior of JoblibTaskManager."""
    with JoblibTaskManager(cache_db=cache_db) as task_manager:
        # Create a task
        belief = create_test_belief()
        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42
        )
        
        # Run task twice
        result1 = task_manager.run_tasks([task])[0]
        result2 = task_manager.run_tasks([task])[0]
        
        # Compare main fields instead of object equality
        assert result1.discount_factor == result2.discount_factor
        assert result1.actual_num_steps == result2.actual_num_steps
        assert result1.reach_terminal_state == result2.reach_terminal_state
        assert len(result1.history) == len(result2.history)
        # Optionally compare more fields as needed
        
        # Clear cache and run again
        task_manager.clear_cache()
        result3 = task_manager.run_tasks([task])[0]
        
        # Compare main fields again
        assert result1.discount_factor == result3.discount_factor
        assert result1.actual_num_steps == result3.actual_num_steps
        assert result1.reach_terminal_state == result3.reach_terminal_state
        assert len(result1.history) == len(result3.history)
        # Optionally compare more fields as needed 