import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

from POMDPPlanners.simulations.simulations_deployment import RemoteRaySimulationDeployment, LocalSimulationDeployment
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.policy import PolicyRunData, PolicyInfoVariable


@pytest.fixture
def temp_redis_path(tmp_path):
    """Create a temporary directory for Redis data."""
    redis_path = tmp_path / "redis_data"
    redis_path.mkdir()
    return redis_path


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    with patch('redis.Redis') as mock_redis:
        yield mock_redis


@pytest.fixture
def deployment(temp_redis_path, mock_redis_client):
    """Create a RemoteRaySimulationDeployment instance with mocked Redis."""
    with patch('subprocess.Popen') as mock_popen:
        deployment = RemoteRaySimulationDeployment(
            redis_host='localhost',
            redis_port=6379,
            redis_db_path=str(temp_redis_path)
        )
        yield deployment


@pytest.fixture
def sample_history():
    """Create a sample History object for testing."""
    env = TigerPOMDP(discount_factor=0.95, name="test_env")
    particles = [env.initial_state_dist().sample() for _ in range(3)]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)
    step_data = StepData(
        state="state1",
        action="action1",
        next_state="state2",
        observation="obs1",
        reward=1.0,
        belief=belief
    )
    return History(
        history=[step_data],
        discount_factor=0.9,
        average_state_sampling_time=0.1,
        average_action_time=0.2,
        average_observation_time=0.3,
        average_belief_update_time=0.4,
        average_reward_time=0.5,
        actual_num_steps=1,
        reach_terminal_state=True,
        policy_run_data=PolicyRunData(info_variables=[
            PolicyInfoVariable(name="test_var", value=1.0),
            PolicyInfoVariable(name="test_var2", value=2.0)
        ])
    )


def test_redis_initialization(deployment, temp_redis_path):
    """Test that Redis is properly initialized with the correct path."""
    assert os.path.exists(temp_redis_path)
    assert os.path.exists(os.path.join(temp_redis_path, 'redis.conf'))


def test_generate_cache_key(deployment):
    """Test that cache keys are generated correctly."""
    env = TigerPOMDP(discount_factor=0.95, name="test_env")
    policy = POMCP(
        environment=env,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="test_policy",
        n_simulations=2
    )
    particles = [env.initial_state_dist().sample() for _ in range(3)]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)
    config = {"test": "config"}

    key = deployment._generate_cache_key(env, policy, belief, config)
    key_data = json.loads(key.split(':', 1)[1])

    assert key.startswith('simulation:')
    assert key_data['env'] == env.__class__.__name__
    assert key_data['policy'] == 'POMCP'
    assert key_data['belief'] == 'WeightedParticleBelief'
    assert key_data['config'] == config


def test_save_and_load_episode_results(deployment, sample_history, mock_redis_client):
    """Test saving and loading episode results."""
    env = TigerPOMDP(discount_factor=0.95, name="test_env")
    policy = POMCP(
        environment=env,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="test_policy",
        n_simulations=2
    )
    particles = [env.initial_state_dist().sample() for _ in range(3)]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)
    config = {"test": "config"}
    cache_dir = Path("/tmp/test_cache")

    # Save results
    deployment.save_episode_simulation_results(
        environment=env,
        policy=policy,
        initial_belief=belief,
        results=[sample_history],
        cache_dir_path=cache_dir,
        general_config=config
    )

    # Verify Redis set was called with correct data
    mock_redis_client.return_value.set.assert_called_once()
    saved_data = json.loads(mock_redis_client.return_value.set.call_args[0][1])
    assert len(saved_data) == 1
    assert saved_data[0]['discount_factor'] == sample_history.discount_factor
    assert saved_data[0]['history'][0]['belief']['type'] == 'WeightedParticleBelief'

    # Mock Redis get to return the serialized data
    mock_redis_client.return_value.get.return_value = json.dumps([{
        'history': [{
            'state': "state1",
            'action': "action1",
            'next_state': "state2",
            'observation': "obs1",
            'reward': 1.0,
            'belief': {
                'type': 'WeightedParticleBelief',
                'particles': particles,
                'log_weights': log_weights.tolist(),
                'resampling': False
            }
        }],
        'discount_factor': 0.9,
        'average_state_sampling_time': 0.1,
        'average_action_time': 0.2,
        'average_observation_time': 0.3,
        'average_belief_update_time': 0.4,
        'average_reward_time': 0.5,
        'actual_num_steps': 1,
        'reach_terminal_state': True,
        'policy_run_data': {
            'info_variables': [
                {'name': 'test_var', 'value': 1.0},
                {'name': 'test_var2', 'value': 2.0}
            ]
        }
    }])

    # Load results
    loaded_results = deployment.load_episode_simulation_results(
        environment=env,
        policy=policy,
        initial_belief=belief,
        cache_dir_path=cache_dir,
        general_config=config
    )

    # Verify results
    assert len(loaded_results) == 1
    assert isinstance(loaded_results[0], History)
    assert loaded_results[0].discount_factor == sample_history.discount_factor
    assert loaded_results[0].actual_num_steps == sample_history.actual_num_steps
    assert isinstance(loaded_results[0].history[0].belief, WeightedParticleBelief)


def test_load_empty_results(deployment, mock_redis_client):
    """Test loading when no results are cached."""
    env = TigerPOMDP(discount_factor=0.95, name="test_env")
    policy = POMCP(
        environment=env,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="test_policy",
        n_simulations=2
    )
    particles = [env.initial_state_dist().sample() for _ in range(3)]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)
    config = {"test": "config"}
    cache_dir = Path("/tmp/test_cache")

    # Mock Redis get to return None (no cached data)
    mock_redis_client.return_value.get.return_value = None

    # Load results
    loaded_results = deployment.load_episode_simulation_results(
        environment=env,
        policy=policy,
        initial_belief=belief,
        cache_dir_path=cache_dir,
        general_config=config
    )

    # Verify empty results
    assert len(loaded_results) == 0


def test_load_invalid_data(deployment, mock_redis_client):
    """Test loading with invalid cached data."""
    env = TigerPOMDP(discount_factor=0.95, name="test_env")
    policy = POMCP(
        environment=env,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="test_policy",
        n_simulations=2
    )
    particles = [env.initial_state_dist().sample() for _ in range(3)]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)
    config = {"test": "config"}
    cache_dir = Path("/tmp/test_cache")

    # Mock Redis get to return invalid JSON
    mock_redis_client.return_value.get.return_value = "invalid json"

    # Load results
    loaded_results = deployment.load_episode_simulation_results(
        environment=env,
        policy=policy,
        initial_belief=belief,
        cache_dir_path=cache_dir,
        general_config=config
    )

    # Verify empty results on error
    assert len(loaded_results) == 0


def test_cleanup_on_del(deployment, mock_redis_client):
    """Test that Redis process is properly cleaned up."""
    # Create a mock process
    mock_process = Mock()
    mock_process.wait.return_value = None  # Ensure wait doesn't raise TimeoutExpired
    deployment.redis_process = mock_process

    # Trigger __del__ explicitly since garbage collection timing is unpredictable
    deployment.__del__()

    # Verify process was terminated
    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once_with(timeout=5)


# Tests for LocalSimulationDeployment
@pytest.fixture
def local_deployment(temp_cache_dir):
    """Create a LocalSimulationDeployment instance."""
    return LocalSimulationDeployment(n_jobs=1, cache_dir=temp_cache_dir)

def test_local_deployment_save_and_load_episode_results(local_deployment, temp_cache_dir, sample_history):
    """Test saving and loading episode results with LocalSimulationDeployment."""
    env = TigerPOMDP(discount_factor=0.95, name="test_env")
    policy = POMCP(
        environment=env,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="test_policy",
        n_simulations=2
    )
    particles = [env.initial_state_dist().sample() for _ in range(3)]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)
    config = {"test": "config"}

    # Save results
    local_deployment.save_episode_simulation_results(
        environment=env,
        policy=policy,
        initial_belief=belief,
        results=[sample_history],
        cache_dir_path=temp_cache_dir,
        general_config=config
    )

    # Load results
    loaded_results = local_deployment.load_episode_simulation_results(
        environment=env,
        policy=policy,
        initial_belief=belief,
        cache_dir_path=temp_cache_dir,
        general_config=config
    )

    # Verify results
    assert len(loaded_results) == 1
    assert isinstance(loaded_results[0], History)
    assert loaded_results[0].discount_factor == sample_history.discount_factor
    assert loaded_results[0].actual_num_steps == sample_history.actual_num_steps
    assert isinstance(loaded_results[0].history[0].belief, WeightedParticleBelief)
    assert loaded_results[0].history[0].state == sample_history.history[0].state
    assert loaded_results[0].history[0].action == sample_history.history[0].action
    assert loaded_results[0].history[0].next_state == sample_history.history[0].next_state
    assert loaded_results[0].history[0].observation == sample_history.history[0].observation
    assert loaded_results[0].history[0].reward == sample_history.history[0].reward

def test_local_deployment_load_empty_results(local_deployment, temp_cache_dir):
    """Test loading when no results are cached with LocalSimulationDeployment."""
    env = TigerPOMDP(discount_factor=0.95, name="test_env")
    policy = POMCP(
        environment=env,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="test_policy",
        n_simulations=2
    )
    particles = [env.initial_state_dist().sample() for _ in range(3)]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)
    config = {"test": "config"}

    # Load results from empty cache
    loaded_results = local_deployment.load_episode_simulation_results(
        environment=env,
        policy=policy,
        initial_belief=belief,
        cache_dir_path=temp_cache_dir,
        general_config=config
    )

    # Verify empty results
    assert len(loaded_results) == 0

def test_local_deployment_parallel_execution(local_deployment, temp_cache_dir, sample_history):
    """Test parallel execution with LocalSimulationDeployment."""
    env = TigerPOMDP(discount_factor=0.95, name="test_env")
    policy = POMCP(
        environment=env,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="test_policy",
        n_simulations=2
    )
    particles = [env.initial_state_dist().sample() for _ in range(3)]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)
    config = {"test": "config"}

    # Create a function that returns the sample history
    def mock_episode_func(**kwargs):
        return sample_history

    # Create episode configurations
    episode_configs = [
        {
            'environment': env,
            'policy': policy,
            'initial_belief': belief,
            'num_steps': 5,
            'episode_id': i,
            'seed': i,
            'cache_dir_path': temp_cache_dir,
            'simulation_deployment': local_deployment
        }
        for i in range(3)
    ]

    # Run episodes in parallel
    results = local_deployment.run_multiple_episodes(
        func=mock_episode_func,
        episode_configs=episode_configs
    )

    # Verify results
    assert len(results) == 3
    for result in results:
        assert isinstance(result, History)
        assert result.discount_factor == sample_history.discount_factor
        assert result.actual_num_steps == sample_history.actual_num_steps

def test_local_deployment_cache_persistence(local_deployment, temp_cache_dir, sample_history):
    """Test that cached results persist between deployment instances."""
    env = TigerPOMDP(discount_factor=0.95, name="test_env")
    policy = POMCP(
        environment=env,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="test_policy",
        n_simulations=2
    )
    particles = [env.initial_state_dist().sample() for _ in range(3)]
    log_weights = np.log(np.ones(3) / 3)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)
    config = {"test": "config"}

    # Save results with first deployment instance
    local_deployment.save_episode_simulation_results(
        environment=env,
        policy=policy,
        initial_belief=belief,
        results=[sample_history],
        cache_dir_path=temp_cache_dir,
        general_config=config
    )

    # Create a new deployment instance
    new_deployment = LocalSimulationDeployment(n_jobs=1)

    # Load results with new deployment instance
    loaded_results = new_deployment.load_episode_simulation_results(
        environment=env,
        policy=policy,
        initial_belief=belief,
        cache_dir_path=temp_cache_dir,
        general_config=config
    )

    # Verify results
    assert len(loaded_results) == 1
    assert isinstance(loaded_results[0], History)
    assert loaded_results[0].discount_factor == sample_history.discount_factor
    assert loaded_results[0].actual_num_steps == sample_history.actual_num_steps

def test_local_deployment_caching_performance(local_deployment, temp_cache_dir):
    """Test that cached parallel tasks run significantly faster on subsequent executions."""
    # Create a computationally expensive function that we can cache
    def expensive_computation(x, sleep_time=0.1):
        # Simulate expensive computation with sleep
        time.sleep(sleep_time)
        # Use deterministic computation instead of random
        result = np.sum(np.arange(1000) * x)
        return result

    # Create a wrapper function that will be called by the deployment
    def task_wrapper(x, sleep_time=0.1, **kwargs):
        return expensive_computation(x, sleep_time)

    # Create multiple task configurations
    task_configs = [
        {
            'x': i,
            'sleep_time': 0.1,  # 100ms per task
            'cache_dir_path': temp_cache_dir,
            'simulation_deployment': local_deployment
        }
        for i in range(5)  # Run 5 tasks
    ]

    # First run - should be slow
    start_time = time.time()
    first_results = local_deployment.run_multiple_episodes(
        func=task_wrapper,
        episode_configs=task_configs
    )
    first_run_time = time.time() - start_time

    # Second run - should be much faster due to caching
    start_time = time.time()
    second_results = local_deployment.run_multiple_episodes(
        func=task_wrapper,
        episode_configs=task_configs
    )
    second_run_time = time.time() - start_time

    # Verify results are identical
    assert len(first_results) == len(second_results)
    for first, second in zip(first_results, second_results):
        assert np.isclose(first, second), "Cached results should be identical"

    # Verify second run is significantly faster
    # Since each task takes 100ms, first run should take at least 500ms
    # Second run should be much faster (ideally < 50ms)
    assert first_run_time >= 0.5, "First run should take at least 500ms"
    assert second_run_time < 0.1, "Second run should be much faster (< 100ms)"
    assert second_run_time < first_run_time / 5, "Second run should be at least 5x faster"

def test_local_deployment_caching_with_different_configs(local_deployment, temp_cache_dir):
    """Test that different task configurations result in different cache entries."""
    def task_wrapper(x, y, **kwargs):
        # Add a small sleep to ensure measurable time
        time.sleep(0.05)  # Increased sleep time
        return x + y

    # First set of configurations
    configs1 = [
        {
            'x': i,
            'y': 1,
            'cache_dir_path': temp_cache_dir,
            'simulation_deployment': local_deployment
        }
        for i in range(3)
    ]

    # Second set of configurations (different y value)
    configs2 = [
        {
            'x': i,
            'y': 2,  # Different y value
            'cache_dir_path': temp_cache_dir,
            'simulation_deployment': local_deployment
        }
        for i in range(3)
    ]

    # Run first set
    start_time = time.time()
    results1 = local_deployment.run_multiple_episodes(
        func=task_wrapper,
        episode_configs=configs1
    )
    first_run_time = time.time() - start_time

    # Run second set (should be slow as it's different configs)
    start_time = time.time()
    results2 = local_deployment.run_multiple_episodes(
        func=task_wrapper,
        episode_configs=configs2
    )
    second_run_time = time.time() - start_time

    # Run first set again (should be fast due to caching)
    start_time = time.time()
    results1_cached = local_deployment.run_multiple_episodes(
        func=task_wrapper,
        episode_configs=configs1
    )
    cached_run_time = time.time() - start_time

    # Verify results
    assert len(results1) == len(results2) == len(results1_cached)
    for r1, r2, r1c in zip(results1, results2, results1_cached):
        assert r1 == r1c, "Cached results should be identical"
        assert r1 != r2, "Different configs should give different results"

    # Verify timing - account for parallel execution
    # With 3 tasks running in parallel, each taking 50ms, we expect at least 50ms total
    assert first_run_time >= 0.05, "First run should take at least 50ms"
    assert second_run_time >= 0.05, "Second run should take at least 50ms"
    # For cached run, we just verify it's not significantly slower
    assert cached_run_time <= first_run_time * 0.5, "Cached run should not be significantly slower"

def test_local_deployment_caching_with_large_data(local_deployment, temp_cache_dir):
    """Test caching behavior with large data structures."""
    def task_wrapper(data, **kwargs):
        # Simulate processing large data
        time.sleep(0.2)  # Increased sleep time
        # Use deterministic computation
        return np.sum(data)

    # Create large arrays with deterministic values
    large_data = [np.arange(1000).reshape(100, 10) for _ in range(3)]
    
    configs = [
        {
            'data': data,
            'cache_dir_path': temp_cache_dir,
            'simulation_deployment': local_deployment
        }
        for data in large_data
    ]

    # First run
    start_time = time.time()
    first_results = local_deployment.run_multiple_episodes(
        func=task_wrapper,
        episode_configs=configs
    )
    first_run_time = time.time() - start_time

    # Second run
    start_time = time.time()
    second_results = local_deployment.run_multiple_episodes(
        func=task_wrapper,
        episode_configs=configs
    )
    second_run_time = time.time() - start_time

    # Verify results
    assert len(first_results) == len(second_results)
    for first, second in zip(first_results, second_results):
        assert np.isclose(first, second), "Cached results should be identical"

    # Verify timing - account for parallel execution
    # With 3 tasks running in parallel, each taking 200ms, we expect at least 200ms total
    assert first_run_time >= 0.2, "First run should take at least 200ms"
    # For second run, we just verify it's not significantly slower
    assert second_run_time <= first_run_time * 0.5, "Second run should not be significantly slower"


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for caching."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    # Ensure the directory exists and is empty
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)
    yield temp_path
    # Cleanup
    try:
        if temp_path.exists():
            # Force close any open file handles
            import gc
            gc.collect()
            # Try to remove the directory
            shutil.rmtree(temp_path, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Failed to clean up temporary directory {temp_path}: {e}")
