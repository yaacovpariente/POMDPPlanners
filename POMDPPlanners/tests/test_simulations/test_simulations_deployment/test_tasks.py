import pytest
import numpy as np

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

def test_episode_simulation_task_creation(environment, policy):
    """Test creation and basic properties of EpisodeSimulationTask."""
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
        console_output=False
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
    """Test that EpisodeSimulationTask raises error for invalid num_steps."""
    belief = create_test_belief()
    
    with pytest.raises(ValueError, match="num_steps must be a positive integer"):
        EpisodeSimulationTask(
            environment=environment,
            policy=policy,
            initial_belief=belief,
            num_steps=-1,
            episode_id=1,
            seed=42,
            console_output=False
        )

def test_episode_simulation_task_equality(environment, policy):
    """Test task equality and hashing."""
    belief = create_test_belief()
    
    # Create identical tasks
    task1 = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        console_output=False
    )
    
    task2 = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=1,
        seed=42,
        console_output=False
    )
    
    # Create different task
    task3 = EpisodeSimulationTask(
        environment=environment,
        policy=policy,
        initial_belief=belief,
        num_steps=2,
        episode_id=2,  # Different episode_id
        seed=42,
        console_output=False
    )
    
    assert task1 == task2
    assert task1 != task3
    assert hash(task1) == hash(task2)
    assert hash(task1) != hash(task3)

def test_episode_simulation_task_serialization(environment, policy):
    """Test task serialization and deserialization."""
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
        console_output=False
    )
    
    # Test serialization
    task_dict = original_task.to_dict()
    assert isinstance(task_dict, dict)
    assert task_dict['num_steps'] == 2
    assert task_dict['episode_id'] == 1
    assert task_dict['seed'] == 42
    assert task_dict['discount_factor'] == 0.95
    assert task_dict['episode_number'] == 1
    
    # Test deserialization
    reconstructed_task = EpisodeSimulationTask.from_dict(task_dict)
    assert reconstructed_task == original_task

def test_episode_simulation_task_execution(environment, policy):
    """Test actual task execution with real environment and policy."""
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
    
    history = task.run()
    assert isinstance(history, History)
    assert len(history.history) <= 2  # May be less if terminal state reached
    assert history.discount_factor == 0.95  # Should match environment's discount factor
    assert history.actual_num_steps <= 2
    
    # Verify history contents
    for step in history.history:
        assert step.state in ["tiger_left", "tiger_right"]
        assert step.action in ["listen", "open_left", "open_right"]
        assert step.observation in ["hear_left", "hear_right", "hear_nothing"]
        assert isinstance(step.reward, float)
        assert isinstance(step.belief, WeightedParticleBelief) 