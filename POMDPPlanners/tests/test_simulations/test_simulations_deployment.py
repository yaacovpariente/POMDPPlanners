import pytest
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch

from POMDPPlanners.simulations.simulations_deployment import RemoteRaySimulationDeployment
from POMDPPlanners.core.simulation import History, StepData


class MockEnvironment:
    def __init__(self, name="test_env"):
        self.name = name

    @property
    def __class__(self):
        return type("MockEnvironment", (), {"__name__": "MockEnvironment"})


class MockPolicy:
    def __init__(self, name="test_policy"):
        self.name = name

    @property
    def __class__(self):
        return type("MockPolicy", (), {"__name__": "MockPolicy"})


class MockBelief:
    def __init__(self, name="test_belief"):
        self.name = name

    @property
    def __class__(self):
        return type("MockBelief", (), {"__name__": "MockBelief"})
    
    def to_dict(self):
        """Convert belief to dictionary for serialization."""
        return {
            "name": self.name,
            "type": "MockBelief"
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create belief from dictionary."""
        return cls(name=data["name"])

    def __eq__(self, other):
        if not isinstance(other, MockBelief):
            return False
        return self.name == other.name


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
    step_data = StepData(
        state="state1",
        action="action1",
        next_state="state2",
        observation="obs1",
        reward=1.0,
        belief=MockBelief()
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
        reach_terminal_state=True
    )


def test_redis_initialization(deployment, temp_redis_path):
    """Test that Redis is properly initialized with the correct path."""
    assert os.path.exists(temp_redis_path)
    assert os.path.exists(os.path.join(temp_redis_path, 'redis.conf'))


def test_generate_cache_key(deployment):
    """Test that cache keys are generated correctly."""
    env = MockEnvironment()
    policy = MockPolicy()
    belief = MockBelief()
    config = {"test": "config"}

    key = deployment._generate_cache_key(env, policy, belief, config)
    key_data = json.loads(key.split(':', 1)[1])

    assert key.startswith('simulation:')
    assert key_data['env'] == 'MockEnvironment'
    assert key_data['policy'] == 'MockPolicy'
    assert key_data['belief'] == 'MockBelief'
    assert key_data['config'] == config


def test_save_and_load_episode_results(deployment, sample_history, mock_redis_client):
    """Test saving and loading episode results."""
    env = MockEnvironment()
    policy = MockPolicy()
    belief = MockBelief()
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
    assert saved_data[0]['history'][0]['belief']['type'] == 'MockBelief'

    # Mock Redis get to return the serialized data
    mock_redis_client.return_value.get.return_value = json.dumps([{
        'history': [{
            'state': "state1",
            'action': "action1",
            'next_state': "state2",
            'observation': "obs1",
            'reward': 1.0,
            'belief': {'name': "test_belief", 'type': "MockBelief"}
        }],
        'discount_factor': 0.9,
        'average_state_sampling_time': 0.1,
        'average_action_time': 0.2,
        'average_observation_time': 0.3,
        'average_belief_update_time': 0.4,
        'average_reward_time': 0.5,
        'actual_num_steps': 1,
        'reach_terminal_state': True
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
    assert isinstance(loaded_results[0].history[0].belief, MockBelief)


def test_load_empty_results(deployment, mock_redis_client):
    """Test loading when no results are cached."""
    env = MockEnvironment()
    policy = MockPolicy()
    belief = MockBelief()
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
    env = MockEnvironment()
    policy = MockPolicy()
    belief = MockBelief()
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
