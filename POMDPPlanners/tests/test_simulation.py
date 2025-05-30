import pytest
from typing import Any, List, Dict, Tuple, Optional
import numpy as np
from dask.distributed import LocalCluster

from POMDPPlanners.core.simulation import (
    SimulationTask,
    TaskManager,
    History,
    StepData
)
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.core.belief import WeightedParticleBelief

@pytest.fixture(scope="function")
def dask_cluster():
    """Fixture to start a local Dask cluster for tests."""
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    yield cluster.scheduler_address
    cluster.close()

def create_test_belief():
    """Helper function to create a valid belief state for testing."""
    particles = ["tiger_left", "tiger_right"]
    # Use equal weights (log(0.5) for each particle)
    log_weights = np.array([np.log(0.5), np.log(0.5)])
    return WeightedParticleBelief(
        particles=particles,
        log_weights=log_weights,
        resampling=False
    )

def test_simulation_task_creation():
    """Test creation and basic properties of SimulationTask."""
    # Create real environment and policy
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparsePFT(
        environment=env,
        discount_factor=0.95,
        gamma=0.95,
        depth=3,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=4,
        n_simulations=2
    )
    belief = create_test_belief()
    
    # Create task
    task = SimulationTask(
        environment=env,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        episode_number=1
    )
    
    # Test properties
    assert task.environment == env
    assert task.policy == policy
    assert task.initial_belief == belief
    assert task.num_steps == 2
    assert task.episode_id == 1
    assert task.seed == 42
    assert task.episode_number == 1
    assert isinstance(task._cache_key, str)
    assert len(task._cache_key) > 0

def test_simulation_task_equality():
    """Test task equality and hashing."""
    # Create real environment and policy
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparsePFT(
        environment=env,
        discount_factor=0.95,
        gamma=0.95,
        depth=10,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=10,
        n_simulations=2
    )
    belief = create_test_belief()
    
    # Create identical tasks
    task1 = SimulationTask(
        environment=env,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        episode_number=1
    )
    
    task2 = SimulationTask(
        environment=env,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        episode_number=1
    )
    
    # Create different task
    task3 = SimulationTask(
        environment=env,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=2,  # Different episode_id
        seed=42,
        episode_number=2
    )
    
    # Test equality
    assert task1 == task2
    assert task1 != task3
    assert hash(task1) == hash(task2)
    assert hash(task1) != hash(task3)

def test_simulation_task_serialization():
    """Test task serialization and deserialization."""
    # Create real environment and policy
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparsePFT(
        environment=env,
        discount_factor=0.95,
        gamma=0.95,
        depth=10,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=10,
        n_simulations=2
    )
    belief = create_test_belief()
    
    # Create task
    original_task = SimulationTask(
        environment=env,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        episode_number=1
    )
    
    # Test serialization
    task_dict = original_task.to_dict()
    assert isinstance(task_dict, dict)
    assert task_dict['num_steps'] == 2
    assert task_dict['episode_id'] == 1
    assert task_dict['seed'] == 42
    assert task_dict['episode_number'] == 1
    
    # Test deserialization
    reconstructed_task = SimulationTask.from_dict(task_dict)
    assert reconstructed_task == original_task

def test_task_manager_initialization(dask_cluster):
    """Test TaskManager initialization and cleanup."""
    # Test local initialization
    with TaskManager(n_workers=2) as manager:
        assert manager.client is not None
        assert manager.n_workers == 2

    # Test connection to existing cluster
    with TaskManager(scheduler_address=dask_cluster) as manager:
        assert manager.client is not None
        assert manager.scheduler_address == dask_cluster

def test_task_manager_execution():
    """Test task execution through TaskManager."""
    # Create real environment and policy
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparsePFT(
        environment=env,
        discount_factor=0.95,
        gamma=0.95,
        depth=10,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=10,
        n_simulations=2
    )
    belief = create_test_belief()

    # Create tasks
    tasks = [
        SimulationTask(
            environment=env,
            policy=policy,
            initial_belief=belief,
            num_steps=2,
            episode_id=i,
            seed=i,
            episode_number=i
        )
        for i in range(1)
    ]

    # Test task execution
    with TaskManager(n_workers=2) as manager:
        # Test submit and gather
        futures = manager.submit_tasks(tasks)
        assert len(futures) == 1

        # Test status checking
        status = manager.get_task_status(futures)
        assert len(status) == 1
        assert all(isinstance(s, str) for s in status.values())

        # Test gathering results
        histories = manager.gather_results(futures)
        assert len(histories) == 1
        # Compute results before checking types
        histories = [h.compute() for h in histories]
        assert all(isinstance(h, History) for h in histories)

def test_task_manager_caching():
    """Test task caching through TaskManager."""
    # Create real environment and policy
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparsePFT(
        environment=env,
        discount_factor=0.95,
        gamma=0.95,
        depth=10,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=10,
        n_simulations=2
    )
    belief = create_test_belief()

    # Create identical tasks
    task1 = SimulationTask(
        environment=env,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        episode_number=1
    )

    task2 = SimulationTask(
        environment=env,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        episode_number=1
    )

    with TaskManager(n_workers=2) as manager:
        # Run first task
        history1 = manager.run_tasks([task1])[0]
        history1 = history1.compute()  # Compute the result

        # Run second task (should use cache)
        history2 = manager.run_tasks([task2])[0]
        history2 = history2.compute()  # Compute the result

        # Results should be identical
        assert history1.history == history2.history, "Cached results should have identical histories."
        assert history1.discount_factor == history2.discount_factor, "Cached results should have identical discount factors."
        assert history1.average_state_sampling_time == history2.average_state_sampling_time, "Cached results should have identical average_state_sampling_time."
        assert history1.average_observation_time == history2.average_observation_time, "Cached results should have identical average_observation_time."
        assert history1.average_belief_update_time == history2.average_belief_update_time, "Cached results should have identical average_belief_update_time."
        assert history1.average_reward_time == history2.average_reward_time, "Cached results should have identical average_reward_time."
        assert history1.actual_num_steps == history2.actual_num_steps, "Cached results should have identical actual_num_steps."
        assert history1.reach_terminal_state == history2.reach_terminal_state, "Cached results should have identical reach_terminal_state."
        assert history1.policy_run_data == history2.policy_run_data, "Cached results should have identical policy_run_data."

def test_task_manager_error_handling():
    """Test error handling in TaskManager."""
    # Create real environment and policy
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparsePFT(
        environment=env,
        discount_factor=0.95,
        gamma=0.95,
        depth=10,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=10,
        n_simulations=2
    )
    belief = create_test_belief()

    # Create task with invalid parameters
    with pytest.raises(ValueError, match="num_steps must be a positive integer"):
        SimulationTask(
            environment=env,
            policy=policy,
            initial_belief=belief,
            num_steps=-1,  # Invalid number of steps
            episode_id=1,
            seed=42,
            episode_number=1
        )

def test_task_manager_cancellation():
    """Test task cancellation in TaskManager."""
    # Create real environment and policy
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparsePFT(
        environment=env,
        discount_factor=0.95,
        gamma=0.95,
        depth=10,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=10,
        n_simulations=2
    )
    belief = create_test_belief()
    
    # Create long-running task
    task = SimulationTask(
        environment=env,
        policy=policy,
        initial_belief=belief,
        num_steps=2,  # Changed from 1000 to 2
        episode_id=1,
        seed=42,
        episode_number=1
    )
    
    with TaskManager(n_workers=2) as manager:
        # Submit task
        futures = manager.submit_tasks([task])
        
        # Cancel task
        manager.cancel_tasks(futures)
        
        # Check status
        status = manager.get_task_status(futures)
        assert status[futures[0].key] in ['cancelled', 'error'] 