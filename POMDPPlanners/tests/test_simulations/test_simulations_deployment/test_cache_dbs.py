import shutil
import tempfile
import time
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import History
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.simulations.simulations_deployment.cache_dbs import DiskCacheDB
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
    temp_path = Path(temp_dir)
    yield temp_path
    # Add a small delay to ensure file handles are released
    time.sleep(0.1)
    if temp_path.exists():
        try:
            shutil.rmtree(temp_path)
        except PermissionError:
            # If we can't remove due to file handles, that's okay for tests
            pass


@pytest.fixture
def cache_db(temp_cache_dir):
    """Fixture to create a DiskCacheDB instance."""
    cache_db = DiskCacheDB(cache_dir=temp_cache_dir)
    yield cache_db
    # Ensure proper cleanup
    try:
        cache_db.close()
    except:
        pass
    # Add a small delay to ensure file handles are released
    time.sleep(0.1)


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


def test_disk_cache_db_initialization(temp_cache_dir):
    """Test DiskCacheDB initialization and cleanup.

    Purpose: Validates proper initialization of disk cache db

    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes

    Test type: unit
    """
    cache_db = DiskCacheDB(cache_dir=temp_cache_dir)
    try:
        assert isinstance(cache_db.cache_dir, Path)
        assert cache_db.cache_dir.exists()
        assert cache_db.cache_dir.is_dir()
    finally:
        cache_db.close()


def test_disk_cache_db_operations(temp_cache_dir, environment, policy):
    """Test basic cache operations (set, get, is_key_in_cache).

    Purpose: Validates that DiskCacheDB correctly implements core cache operations including storage, retrieval, and key existence checking

    Given: DiskCacheDB with temporary cache directory, EpisodeSimulationTask with TigerPOMDP environment and SparsePFT policy
    When: Task is executed and result cached using set(), then retrieved using get() and is_key_in_cache()
    Then: Cache correctly stores History object, retrieval returns equivalent History with matching discount_factor and step counts, key existence checks work correctly

    Test type: unit
    """
    cache_db = DiskCacheDB(cache_dir=temp_cache_dir)
    try:
        # Create a real simulation task and run it
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
        history = task.run()
        assert history is not None  # Ensure task.run() didn't return None
        history = cast(History, history)

        # Test setting and getting
        cache_db.set(task._cache_key, history)
        assert cache_db.is_key_in_cache(task._cache_key)

        retrieved = cache_db.get(task._cache_key)
        assert isinstance(retrieved, History)
        # Type cast for pyright - we know it's a History from the assert
        retrieved = cast(History, retrieved)
        # Compare main fields instead of object equality
        assert retrieved.discount_factor == history.discount_factor
        assert retrieved.actual_num_steps == history.actual_num_steps
        assert retrieved.reach_terminal_state == history.reach_terminal_state
        assert len(retrieved.history) == len(history.history)
        # Optionally compare more fields as needed

        # Test non-existent key
        assert not cache_db.is_key_in_cache("non_existent")
        assert cache_db.get("non_existent") is None
    finally:
        cache_db.close()


def test_disk_cache_db_store_and_retrieve(cache_db, environment, policy):
    """Test storing and retrieving tasks from disk cache.

    Purpose: Validates that DiskCacheDB can store task results with config IDs and retrieve identical History objects

    Given: DiskCacheDB instance, EpisodeSimulationTask, and mock History with specific timing and result attributes
    When: History is stored using task config ID and then retrieved from cache
    Then: Retrieved History object matches original exactly including discount_factor, timing attributes, and terminal state flag

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
        console_output=False,
    )

    # Create a mock history result
    history = History(
        history=[],
        discount_factor=0.95,
        average_state_sampling_time=0.001,
        average_action_time=0.002,
        average_observation_time=0.003,
        average_belief_update_time=0.004,
        average_reward_time=0.005,
        actual_num_steps=2,
        reach_terminal_state=True,
        policy_run_data=None,
    )

    # Store in cache
    task_id = task.get_config_id()
    cache_db.set(task_id, history)

    # Check if key exists
    assert cache_db.is_key_in_cache(task_id)

    # Retrieve from cache
    retrieved_history = cache_db.get(task_id)
    assert retrieved_history == history


def test_disk_cache_db_clear(cache_db, environment, policy):
    """Test clearing the disk cache.

    Purpose: Validates that DiskCacheDB clear operation removes all cached entries and makes them inaccessible

    Given: DiskCacheDB with cached History object stored using task config ID
    When: clear() method is called on the cache database
    Then: Previously cached key is no longer found in cache (is_key_in_cache returns False)

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
        console_output=False,
    )

    # Create a mock history result
    history = History(
        history=[],
        discount_factor=0.95,
        average_state_sampling_time=0.001,
        average_action_time=0.002,
        average_observation_time=0.003,
        average_belief_update_time=0.004,
        average_reward_time=0.005,
        actual_num_steps=2,
        reach_terminal_state=True,
        policy_run_data=None,
    )

    # Store in cache
    task_id = task.get_config_id()
    cache_db.set(task_id, history)

    # Verify it's in cache
    assert cache_db.is_key_in_cache(task_id)

    # Clear cache
    cache_db.clear()

    # Verify it's no longer in cache
    assert not cache_db.is_key_in_cache(task_id)
