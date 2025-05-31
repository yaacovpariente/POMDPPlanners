import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path

from POMDPPlanners.simulations.simulations_deployment.cache_dbs import DiskCacheDB
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
def temp_cache_dir():
    """Fixture to create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)

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

def test_disk_cache_db_initialization(temp_cache_dir):
    """Test DiskCacheDB initialization and cleanup."""
    cache_db = DiskCacheDB(cache_dir=temp_cache_dir)
    try:
        assert isinstance(cache_db.cache_dir, Path)
        assert cache_db.cache_dir.exists()
        assert cache_db.cache_dir.is_dir()
    finally:
        cache_db.close()

def test_disk_cache_db_operations(temp_cache_dir, environment, policy):
    """Test basic cache operations (set, get, is_key_in_cache)."""
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
            seed=42
        )
        history = task.run()
        
        # Test setting and getting
        cache_db.set(task._cache_key, history)
        assert cache_db.is_key_in_cache(task._cache_key)
        
        retrieved = cache_db.get(task._cache_key)
        assert isinstance(retrieved, History)
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

def test_disk_cache_db_invalid_value(temp_cache_dir):
    """Test that DiskCacheDB rejects invalid values."""
    cache_db = DiskCacheDB(cache_dir=temp_cache_dir)
    try:
        with pytest.raises(TypeError, match="Cache can only store History objects"):
            cache_db.set("test_key", "not a history")
    finally:
        cache_db.close()

def test_disk_cache_db_clear(temp_cache_dir, environment, policy):
    """Test cache clearing functionality."""
    cache_db = DiskCacheDB(cache_dir=temp_cache_dir)
    try:
        # Create and run a real simulation task
        belief = create_test_belief()
        task = EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=1,
            seed=42
        )
        history = task.run()
        
        # Store the result
        cache_db.set(task._cache_key, history)
        assert cache_db.is_key_in_cache(task._cache_key)
        
        # Clear cache
        cache_db.clear()
        assert not cache_db.is_key_in_cache(task._cache_key)
        assert cache_db.get(task._cache_key) is None
    finally:
        cache_db.close() 