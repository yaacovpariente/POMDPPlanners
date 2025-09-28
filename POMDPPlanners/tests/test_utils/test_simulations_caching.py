"""Tests for simulations caching module.

This module tests the caching functionality for simulation results, focusing on:
- Cache key generation for different object combinations
- Cache directory path management
- Storing and retrieving simulation results
- Cache hit/miss scenarios
- Edge cases and error conditions
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, cast
from unittest.mock import Mock, patch

import pytest

from POMDPPlanners.core.simulation.history import History, StepData
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.utils.simulations_caching import (
    cache_episode_simulation_results,
    get_cache_dir_path,
    get_cache_key,
    load_episode_simulation_results,
)


class MockEnvironment:
    """Mock environment for testing purposes."""

    def __init__(self, config_id="mock_env_id"):
        self.config_id = config_id


class MockPolicy:
    """Mock policy for testing purposes."""

    def __init__(self, config_id="mock_policy_id"):
        self.config_id = config_id


class MockBelief:
    """Mock belief for testing purposes."""

    def __init__(self, config_id="mock_belief_id"):
        self.config_id = config_id


@dataclass
class SimpleHistory:
    """Simple serializable history object for testing caching functionality."""

    episode_id: int
    total_reward: float
    num_steps: int

    def __eq__(self, other):
        if not isinstance(other, SimpleHistory):
            return False
        return (
            self.episode_id == other.episode_id
            and self.total_reward == other.total_reward
            and self.num_steps == other.num_steps
        )


@pytest.fixture
def mock_environment():
    """Create a mock environment for testing.

    Returns:
        MockEnvironment: Mock environment with predictable config_id
    """
    return MockEnvironment("test_env_123")


@pytest.fixture
def mock_policy():
    """Create a mock policy for testing.

    Returns:
        MockPolicy: Mock policy with predictable config_id
    """
    return MockPolicy("test_policy_456")


@pytest.fixture
def mock_belief():
    """Create a mock belief for testing.

    Returns:
        MockBelief: Mock belief with predictable config_id
    """
    return MockBelief("test_belief_789")


@pytest.fixture
def mock_history():
    """Create a mock history for testing.

    Returns:
        History: Mock history object for simulation results (non-serializable)
    """
    return Mock(spec=History)


@pytest.fixture
def simple_history():
    """Create a simple serializable history for testing.

    Returns:
        SimpleHistory: Simple serializable history object for cache testing
    """
    return SimpleHistory(episode_id=1, total_reward=10.5, num_steps=5)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing.

    Returns:
        Path: Temporary directory path that gets cleaned up after test
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def test_get_cache_key_basic_functionality(mock_environment, mock_policy, mock_belief):
    """Test basic cache key generation with mock objects.

    Purpose: Validates that get_cache_key generates deterministic keys from component IDs

    Given: Mock environment, policy, and belief objects with known config_id values
    When: Cache key is generated using get_cache_key function
    Then: Key contains all component IDs separated by pipe characters

    Test type: unit
    """
    key = get_cache_key(mock_environment, mock_policy, mock_belief)

    expected_parts = [
        "test_env_123",
        "test_policy_456",
        "test_belief_789",
        "[]",  # Empty general_config as sorted list
    ]
    expected_key = "|".join(expected_parts)

    assert key == expected_key


def test_get_cache_key_with_general_config(mock_environment, mock_policy, mock_belief):
    """Test cache key generation with general configuration parameters.

    Purpose: Validates that general_config parameters are properly included in cache keys

    Given: Mock objects and a general_config dictionary with various parameter types
    When: Cache key is generated with the general_config
    Then: Key includes sorted config parameters as string representation

    Test type: unit
    """
    general_config = {"num_episodes": 100, "max_steps": 50, "param_a": "value_a"}

    key = get_cache_key(mock_environment, mock_policy, mock_belief, general_config)

    # Should contain sorted config items
    assert "test_env_123" in key
    assert "test_policy_456" in key
    assert "test_belief_789" in key
    assert str(sorted(general_config.items())) in key


def test_get_cache_key_different_configs_produce_different_keys(
    mock_environment, mock_policy, mock_belief
):
    """Test that different configurations produce different cache keys.

    Purpose: Validates that cache keys are unique for different configurations

    Given: Same mock objects but different general_config dictionaries
    When: Cache keys are generated for each configuration
    Then: All cache keys are unique to prevent cache collisions

    Test type: unit
    """
    config1 = {"episodes": 100}
    config2 = {"episodes": 200}
    config3 = {"episodes": 100, "steps": 50}

    key1 = get_cache_key(mock_environment, mock_policy, mock_belief, config1)
    key2 = get_cache_key(mock_environment, mock_policy, mock_belief, config2)
    key3 = get_cache_key(mock_environment, mock_policy, mock_belief, config3)

    assert key1 != key2
    assert key1 != key3
    assert key2 != key3


def test_get_cache_key_empty_config_consistency(mock_environment, mock_policy, mock_belief):
    """Test cache key consistency with empty vs no config.

    Purpose: Validates that empty config and no config produce identical cache keys

    Given: Mock objects tested with empty dict vs no general_config parameter
    When: Cache keys are generated for both scenarios
    Then: Both keys are identical for consistent caching behavior

    Test type: unit
    """
    key_no_config = get_cache_key(mock_environment, mock_policy, mock_belief)
    key_empty_config = get_cache_key(mock_environment, mock_policy, mock_belief, {})

    assert key_no_config == key_empty_config


def test_get_cache_dir_path_creates_correct_subdirectory(temp_cache_dir):
    """Test cache directory path generation.

    Purpose: Validates that get_cache_dir_path creates correct subdirectory path

    Given: A temporary base cache directory path
    When: get_cache_dir_path is called with the base path
    Then: Returns path with 'simulations_cache' subdirectory appended

    Test type: unit
    """
    result_path = get_cache_dir_path(temp_cache_dir)
    expected_path = temp_cache_dir / "simulations_cache"

    assert result_path == expected_path


def test_cache_episode_simulation_results_new_entry(
    mock_environment, mock_policy, mock_belief, simple_history, temp_cache_dir
):
    """Test caching new simulation results.

    Purpose: Validates that new simulation results are successfully stored in cache

    Given: Mock objects, simulation results, and empty cache directory
    When: cache_episode_simulation_results is called with the data
    Then: Results are stored in cache and can be retrieved with same key

    Test type: unit
    """
    results = [simple_history, SimpleHistory(2, 20.0, 10)]

    # Cache the results
    cache_episode_simulation_results(
        mock_environment, mock_policy, mock_belief, results, temp_cache_dir
    )

    # Verify results were cached by loading them
    loaded_results = load_episode_simulation_results(
        mock_environment, mock_policy, mock_belief, temp_cache_dir
    )

    assert loaded_results == results


def test_cache_episode_simulation_results_with_config(
    mock_environment, mock_policy, mock_belief, simple_history, temp_cache_dir
):
    """Test caching results with general configuration parameters.

    Purpose: Validates that simulation results with config parameters are properly cached

    Given: Mock objects, results, cache directory, and general_config parameters
    When: Results are cached with the configuration
    Then: Results can be retrieved using the same configuration parameters

    Test type: unit
    """
    results = [simple_history]
    general_config = {"episodes": 50, "max_steps": 100}

    cache_episode_simulation_results(
        mock_environment, mock_policy, mock_belief, results, temp_cache_dir, general_config
    )

    loaded_results = load_episode_simulation_results(
        mock_environment, mock_policy, mock_belief, temp_cache_dir, general_config
    )

    assert loaded_results == results


@patch("POMDPPlanners.utils.simulations_caching.logger")
def test_cache_episode_simulation_results_already_cached(
    mock_logger, mock_environment, mock_policy, mock_belief, simple_history, temp_cache_dir
):
    """Test behavior when results are already cached.

    Purpose: Validates that attempting to cache existing results logs appropriate message

    Given: Simulation results that are already stored in cache
    When: cache_episode_simulation_results is called again with same parameters
    Then: Info message is logged indicating results are already cached

    Test type: unit
    """
    results = [simple_history]

    # Cache results twice
    cache_episode_simulation_results(
        mock_environment, mock_policy, mock_belief, results, temp_cache_dir
    )
    cache_episode_simulation_results(
        mock_environment, mock_policy, mock_belief, results, temp_cache_dir
    )

    # Verify logger was called with appropriate message
    mock_logger.info.assert_called()
    call_args = mock_logger.info.call_args[0][0]
    assert "already cached" in call_args


def test_load_episode_simulation_results_cache_miss(
    mock_environment, mock_policy, mock_belief, temp_cache_dir
):
    """Test loading results when cache is empty.

    Purpose: Validates that loading from empty cache returns empty list

    Given: Empty cache directory with no stored simulation results
    When: load_episode_simulation_results is called
    Then: Empty list is returned indicating cache miss

    Test type: unit
    """
    loaded_results = load_episode_simulation_results(
        mock_environment, mock_policy, mock_belief, temp_cache_dir
    )

    assert loaded_results == []


@patch("POMDPPlanners.utils.simulations_caching.logger")
def test_load_episode_simulation_results_cache_miss_logs_debug(
    mock_logger, mock_environment, mock_policy, mock_belief, temp_cache_dir
):
    """Test that cache miss logs debug message.

    Purpose: Validates that cache miss scenario logs appropriate debug message

    Given: Empty cache directory
    When: load_episode_simulation_results is called for non-existent key
    Then: Debug message is logged indicating cache miss

    Test type: unit
    """
    load_episode_simulation_results(mock_environment, mock_policy, mock_belief, temp_cache_dir)

    mock_logger.debug.assert_called()
    call_args = mock_logger.debug.call_args[0][0]
    assert "not found in cache" in call_args


def test_load_episode_simulation_results_different_config_returns_empty(
    mock_environment, mock_policy, mock_belief, simple_history, temp_cache_dir
):
    """Test loading with different config returns empty results.

    Purpose: Validates that different configurations result in cache miss

    Given: Results cached with specific configuration parameters
    When: load_episode_simulation_results is called with different config
    Then: Empty list is returned due to different cache key

    Test type: unit
    """
    results = [simple_history]
    config1 = {"episodes": 100}
    config2 = {"episodes": 200}

    # Cache with config1
    cache_episode_simulation_results(
        mock_environment, mock_policy, mock_belief, results, temp_cache_dir, config1
    )

    # Try to load with config2 - should be cache miss
    loaded_results = load_episode_simulation_results(
        mock_environment, mock_policy, mock_belief, temp_cache_dir, config2
    )

    assert loaded_results == []


def test_cache_key_deterministic_across_calls(mock_environment, mock_policy, mock_belief):
    """Test that cache keys are deterministic across multiple calls.

    Purpose: Validates that identical inputs always produce identical cache keys

    Given: Same mock objects and configuration parameters
    When: get_cache_key is called multiple times with identical inputs
    Then: All generated keys are identical for reliable caching

    Test type: unit
    """
    config = {"param": "value", "number": 42}

    key1 = get_cache_key(mock_environment, mock_policy, mock_belief, config)
    key2 = get_cache_key(mock_environment, mock_policy, mock_belief, config)
    key3 = get_cache_key(mock_environment, mock_policy, mock_belief, config)

    assert key1 == key2 == key3


def test_cache_survives_multiple_objects_same_ids():
    """Test caching with different object instances having same config_ids.

    Purpose: Validates that cache works correctly with different object instances

    Given: Different object instances with identical config_id values
    When: Cache operations are performed with different object instances
    Then: Cache correctly identifies matching keys regardless of object identity

    Test type: unit
    """
    env1 = MockEnvironment("same_id")
    env2 = MockEnvironment("same_id")
    policy1 = MockPolicy("same_policy")
    policy2 = MockPolicy("same_policy")
    belief1 = MockBelief("same_belief")
    belief2 = MockBelief("same_belief")

    key1 = get_cache_key(cast(Environment, env1), cast(Policy, policy1), cast(Belief, belief1))
    key2 = get_cache_key(cast(Environment, env2), cast(Policy, policy2), cast(Belief, belief2))

    assert key1 == key2


def test_cache_with_complex_general_config():
    """Test cache key generation with complex nested general config.

    Purpose: Validates that complex configuration structures are properly serialized

    Given: Mock objects and general_config with nested dictionaries and lists
    When: Cache key is generated with the complex configuration
    Then: Key is generated successfully and remains deterministic

    Test type: unit
    """
    env = MockEnvironment()
    policy = MockPolicy()
    belief = MockBelief()

    complex_config = {
        "nested": {"param1": 1, "param2": [1, 2, 3]},
        "list_param": [{"a": 1}, {"b": 2}],
        "simple": 42,
    }

    key1 = get_cache_key(
        cast(Environment, env), cast(Policy, policy), cast(Belief, belief), complex_config
    )
    key2 = get_cache_key(
        cast(Environment, env), cast(Policy, policy), cast(Belief, belief), complex_config
    )

    assert key1 == key2
    assert isinstance(key1, str)
    assert len(key1) > 0


def test_cache_dir_path_with_relative_path():
    """Test cache directory path generation with relative paths.

    Purpose: Validates that get_cache_dir_path works correctly with relative paths

    Given: A relative path as base cache directory
    When: get_cache_dir_path is called with the relative path
    Then: Correct subdirectory path is appended regardless of path type

    Test type: unit
    """
    relative_path = Path("./cache")
    result_path = get_cache_dir_path(relative_path)
    expected_path = relative_path / "simulations_cache"

    assert result_path == expected_path


def test_end_to_end_cache_workflow(temp_cache_dir):
    """Test complete cache workflow from storing to retrieving results.

    Purpose: Validates the complete caching workflow with realistic scenario

    Given: Mock objects and multiple simulation results with different configurations
    When: Multiple cache and load operations are performed
    Then: All operations work correctly with proper cache hits and misses

    Test type: integration
    """
    env = MockEnvironment("workflow_env")
    policy = MockPolicy("workflow_policy")
    belief = MockBelief("workflow_belief")

    # Create different result sets
    results1 = [SimpleHistory(1, 100.0, 10), SimpleHistory(2, 200.0, 20)]
    results2 = [SimpleHistory(3, 300.0, 30)]

    config1 = {"scenario": "test1"}
    config2 = {"scenario": "test2"}

    # Cache different results with different configs
    cache_episode_simulation_results(
        cast(Environment, env),
        cast(Policy, policy),
        cast(Belief, belief),
        cast(List[History], results1),
        temp_cache_dir,
        config1,
    )
    cache_episode_simulation_results(
        cast(Environment, env),
        cast(Policy, policy),
        cast(Belief, belief),
        cast(List[History], results2),
        temp_cache_dir,
        config2,
    )

    # Load and verify results
    loaded1 = load_episode_simulation_results(
        cast(Environment, env), cast(Policy, policy), cast(Belief, belief), temp_cache_dir, config1
    )
    loaded2 = load_episode_simulation_results(
        cast(Environment, env), cast(Policy, policy), cast(Belief, belief), temp_cache_dir, config2
    )

    assert loaded1 == results1
    assert loaded2 == results2

    # Test cache miss
    config3 = {"scenario": "test3"}
    loaded3 = load_episode_simulation_results(
        cast(Environment, env), cast(Policy, policy), cast(Belief, belief), temp_cache_dir, config3
    )
    assert loaded3 == []


def test_get_cache_key_returns_string_type(mock_environment, mock_policy, mock_belief):
    """Test that get_cache_key returns correct type.

    Purpose: Validates that get_cache_key always returns a string type

    Given: Mock environment, policy, and belief objects
    When: get_cache_key is called with various configurations
    Then: Return value is always of type str

    Test type: unit
    """
    # Test with no config
    key1 = get_cache_key(mock_environment, mock_policy, mock_belief)
    assert isinstance(key1, str)

    # Test with empty config
    key2 = get_cache_key(mock_environment, mock_policy, mock_belief, {})
    assert isinstance(key2, str)

    # Test with complex config
    complex_config = {"num": 42, "nested": {"a": 1, "b": [1, 2, 3]}}
    key3 = get_cache_key(mock_environment, mock_policy, mock_belief, complex_config)
    assert isinstance(key3, str)


def test_get_cache_dir_path_returns_path_type(temp_cache_dir):
    """Test that get_cache_dir_path returns correct type.

    Purpose: Validates that get_cache_dir_path returns a Path object

    Given: A base cache directory path
    When: get_cache_dir_path is called
    Then: Return value is of type Path

    Test type: unit
    """
    result = get_cache_dir_path(temp_cache_dir)
    assert isinstance(result, Path)


def test_cache_episode_simulation_results_returns_none(
    mock_environment, mock_policy, mock_belief, simple_history, temp_cache_dir
):
    """Test that cache_episode_simulation_results returns None.

    Purpose: Validates that cache_episode_simulation_results has no return value (returns None)

    Given: Mock objects, simulation results, and cache directory
    When: cache_episode_simulation_results is called
    Then: Function returns None (void function)

    Test type: unit
    """
    results = [simple_history]

    return_value = cache_episode_simulation_results(
        mock_environment, mock_policy, mock_belief, results, temp_cache_dir
    )

    assert return_value is None


def test_load_episode_simulation_results_returns_list_type(
    mock_environment, mock_policy, mock_belief, simple_history, temp_cache_dir
):
    """Test that load_episode_simulation_results returns correct type.

    Purpose: Validates that load_episode_simulation_results always returns a list

    Given: Mock objects and cache directory in various states
    When: load_episode_simulation_results is called
    Then: Return value is always of type list

    Test type: unit
    """
    # Test cache miss - should return empty list
    result_miss = load_episode_simulation_results(
        mock_environment, mock_policy, mock_belief, temp_cache_dir
    )
    assert isinstance(result_miss, list)
    assert result_miss == []

    # Test cache hit - should return list with contents
    results = [simple_history, SimpleHistory(2, 50.0, 8)]
    cache_episode_simulation_results(
        mock_environment, mock_policy, mock_belief, results, temp_cache_dir
    )

    result_hit = load_episode_simulation_results(
        mock_environment, mock_policy, mock_belief, temp_cache_dir
    )
    assert isinstance(result_hit, list)
    assert len(result_hit) == 2
    assert all(isinstance(item, SimpleHistory) for item in result_hit)
