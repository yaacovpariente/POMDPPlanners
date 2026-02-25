import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling_planner import (
    SparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.utils.visualization import (
    plot_discounted_returns_histogram,
    plot_discounted_returns_histogram_multiple_policies,
    plot_environment_policy_pair_comparison,
)

np.random.seed(42)
random.seed(42)


def create_mock_history(rewards, discount_factor=0.95):
    """Helper function to create mock History objects for testing."""
    steps = []
    for i, reward in enumerate(rewards):
        step = StepData(
            state=f"state_{i}",
            action=f"action_{i}",
            next_state=f"state_{i+1}",
            observation=f"obs_{i}",
            reward=reward,
            belief=WeightedParticleBelief(particles=[f"state_{i}"], log_weights=np.array([0.1])),
        )
        steps.append(step)

    return History(
        history=steps,
        discount_factor=discount_factor,
        average_state_sampling_time=0.001,
        average_action_time=0.002,
        average_observation_time=0.001,
        average_belief_update_time=0.003,
        average_reward_time=0.001,
        actual_num_steps=len(rewards),
        reach_terminal_state=True,
        policy_run_data=[],
    )


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for visualization testing with proper cleanup.

    Creates a unique temporary directory for each test to store generated plots
    and MLFlow artifacts, ensuring test isolation and automatic cleanup.

    Yields:
        Path: Temporary directory path for storing test artifacts

    Note:
        Uses force cleanup with garbage collection to handle any remaining
        file handles that may prevent directory removal on some systems.
    """
    import gc
    import shutil
    import uuid

    temp_dir = Path(tempfile.gettempdir())
    unique_dir = temp_dir / f"test_{uuid.uuid4().hex}"
    unique_dir.mkdir(parents=True, exist_ok=True)
    temp_cache_dir = unique_dir
    try:
        # Ensure the directory exists and is empty
        if temp_cache_dir.exists():
            shutil.rmtree(temp_cache_dir)
        temp_cache_dir.mkdir(parents=True, exist_ok=True)
        yield temp_cache_dir
    finally:
        # Cleanup
        try:
            if temp_cache_dir.exists():
                # Force close any open file handles
                gc.collect()
                # Try to remove the directory
                shutil.rmtree(temp_cache_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory {temp_cache_dir}: {e}")


def test_plot_discounted_returns_histogram(temp_cache_dir):
    """Test that plot_discounted_returns_histogram generates histogram visualization for single policy.

    Purpose: Validates that discounted returns histogram plotting creates visualization for a single policy

    Given: TigerPOMDP environment, StandardSparseSampling policy, and mock History objects with discounted returns
    When: plot_discounted_returns_histogram is called with histories and policy
    Then: Histogram plot is generated and saved to expected file location

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=2, depth=1)

    # Create mock histories with discounted returns
    mock_histories = []
    for i in range(10):
        history = create_mock_history([-1.0, -2.0, 5.0])
        mock_histories.append(history)

    # Execute
    output_path = temp_cache_dir / "discounted_returns_histogram.png"
    plot_discounted_returns_histogram(
        histories=mock_histories, policy=policy, environment=env, cache_path=output_path
    )

    # Verify plot was created
    assert output_path.exists()


def test_plot_discounted_returns_histogram_multiple_policies(temp_cache_dir):
    """Test that plot_discounted_returns_histogram_multiple_policies generates overlapping histograms for multiple policies.

    Purpose: Validates that multiple policy histogram plotting creates overlapping visualizations

    Given: TigerPOMDP environment, multiple policies, and histories dictionary mapping policy names to History lists
    When: plot_discounted_returns_histogram_multiple_policies is called with multiple policy histories
    Then: Overlapping histogram plot is generated showing distribution differences between policies

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    policy1 = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=2, depth=1)
    policy2 = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=3, depth=2)

    # Create mock histories for each policy
    histories_dict = {}

    # Policy 1 histories
    policy1_histories = []
    for i in range(8):
        history = create_mock_history([-1.0, -2.0, 5.0])
        policy1_histories.append(history)
    histories_dict[policy1.name] = policy1_histories

    # Policy 2 histories
    policy2_histories = []
    for i in range(8):
        history = create_mock_history([-0.5, -1.5, 4.5])  # Slightly different rewards
        policy2_histories.append(history)
    histories_dict[policy2.name] = policy2_histories

    # Execute
    output_path = temp_cache_dir / "multiple_policies_histogram.png"
    plot_discounted_returns_histogram_multiple_policies(
        histories=histories_dict,
        policies=[policy1, policy2],
        environment=env,
        cache_path=output_path,
    )

    # Verify plot was created
    assert output_path.exists()


def test_plot_environment_policy_pair_comparison(temp_cache_dir):
    """Test that plot_environment_policy_pair_comparison generates histogram for environment-policy pair.

    Purpose: Validates that environment-policy pair comparison plotting creates histogram visualization

    Given: TigerPOMDP environment, StandardSparseSampling policy, and mock History objects
    When: plot_environment_policy_pair_comparison is called with histories, policy, and environment
    Then: Histogram plot is generated showing discounted returns distribution for the specific pair

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=2, depth=1)

    # Create mock histories
    mock_histories = []
    for i in range(10):
        history = create_mock_history([-1.0, -2.0, 5.0])
        mock_histories.append(history)

    # Execute
    output_path = temp_cache_dir / "environment_policy_comparison.png"
    plot_environment_policy_pair_comparison(
        histories=mock_histories, policy=policy, environment=env, cache_path=output_path
    )

    # Verify plot was created
    assert output_path.exists()


def test_plot_discounted_returns_histogram_empty_histories(temp_cache_dir):
    """Test that plot_discounted_returns_histogram properly handles empty histories input.

    Purpose: Validates proper error handling when empty histories list is provided

    Given: TigerPOMDP environment, StandardSparseSampling policy, and empty histories list
    When: plot_discounted_returns_histogram is called with empty histories
    Then: Function handles empty input gracefully (creates empty histogram)

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=2, depth=1)

    # Execute with empty histories
    output_path = temp_cache_dir / "empty_histogram.png"
    plot_discounted_returns_histogram(
        histories=[], policy=policy, environment=env, cache_path=output_path
    )

    # Verify plot was created (even if empty)
    assert output_path.exists()


def test_plot_discounted_returns_histogram_multiple_policies_empty_histories(
    temp_cache_dir,
):
    """Test that plot_discounted_returns_histogram_multiple_policies properly handles empty histories.

    Purpose: Validates proper error handling when policies have no histories

    Given: TigerPOMDP environment, multiple policies, and histories dictionary with empty lists
    When: plot_discounted_returns_histogram_multiple_policies is called with empty histories
    Then: Function handles empty input gracefully (creates empty histogram)

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    policy1 = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=2, depth=1)
    policy2 = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=3, depth=2)

    # Create empty histories dictionary
    histories_dict = {policy1.name: [], policy2.name: []}

    # Execute with empty histories
    output_path = temp_cache_dir / "empty_multiple_histogram.png"
    plot_discounted_returns_histogram_multiple_policies(
        histories=histories_dict,
        policies=[policy1, policy2],
        environment=env,
        cache_path=output_path,
    )

    # Verify plot was created (even if empty)
    assert output_path.exists()


def test_plot_discounted_returns_histogram_single_history(temp_cache_dir):
    """Test that plot_discounted_returns_histogram properly handles single history input.

    Purpose: Validates proper handling when only one history is provided (edge case)

    Given: TigerPOMDP environment, StandardSparseSampling policy, and single History object
    When: plot_discounted_returns_histogram is called with exactly one history
    Then: Histogram plot is generated successfully for single data point

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=2, depth=1)

    # Create single history
    single_history = create_mock_history([-1.0, -2.0, 5.0])

    # Execute with single history
    output_path = temp_cache_dir / "single_history_histogram.png"
    plot_discounted_returns_histogram(
        histories=[single_history],
        policy=policy,
        environment=env,
        cache_path=output_path,
    )

    # Verify plot was created
    assert output_path.exists()


def test_plot_discounted_returns_histogram_two_histories(temp_cache_dir):
    """Test that plot_discounted_returns_histogram properly handles two histories input.

    Purpose: Validates proper handling when exactly two histories are provided (minimal comparison case)

    Given: TigerPOMDP environment, StandardSparseSampling policy, and two History objects
    When: plot_discounted_returns_histogram is called with exactly two histories
    Then: Histogram plot is generated successfully for minimal dataset

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=2, depth=1)

    # Create two histories with different rewards
    history1 = create_mock_history([-1.0, -2.0, 5.0])
    history2 = create_mock_history([-0.5, -1.5, 4.5])

    # Execute with two histories
    output_path = temp_cache_dir / "two_histories_histogram.png"
    plot_discounted_returns_histogram(
        histories=[history1, history2],
        policy=policy,
        environment=env,
        cache_path=output_path,
    )

    # Verify plot was created
    assert output_path.exists()


def test_plot_discounted_returns_histogram_multiple_policies_single_history_per_policy(
    temp_cache_dir,
):
    """Test that plot_discounted_returns_histogram_multiple_policies handles single history per policy.

    Purpose: Validates proper handling when each policy has exactly one history (edge case)

    Given: TigerPOMDP environment, multiple policies, and histories dictionary with single history per policy
    When: plot_discounted_returns_histogram_multiple_policies is called with single histories
    Then: Overlapping histogram plot is generated successfully for minimal datasets

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    policy1 = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=2, depth=1)
    policy2 = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=3, depth=2)

    # Create histories dictionary with single history per policy
    histories_dict = {
        policy1.name: [create_mock_history([-1.0, -2.0, 5.0])],
        policy2.name: [create_mock_history([-0.5, -1.5, 4.5])],
    }

    # Execute with single histories per policy
    output_path = temp_cache_dir / "single_history_per_policy_histogram.png"
    plot_discounted_returns_histogram_multiple_policies(
        histories=histories_dict,
        policies=[policy1, policy2],
        environment=env,
        cache_path=output_path,
    )

    # Verify plot was created
    assert output_path.exists()


def test_plot_discounted_returns_histogram_zero_rewards(temp_cache_dir):
    """Test that plot_discounted_returns_histogram properly handles histories with zero rewards.

    Purpose: Validates proper handling when all rewards are zero (edge case)

    Given: TigerPOMDP environment, StandardSparseSampling policy, and History objects with zero rewards
    When: plot_discounted_returns_histogram is called with zero-reward histories
    Then: Histogram plot is generated successfully for zero-value data

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=2, depth=1)

    # Create histories with zero rewards
    mock_histories = []
    for i in range(5):
        history = create_mock_history([0.0, 0.0, 0.0])
        mock_histories.append(history)

    # Execute with zero-reward histories
    output_path = temp_cache_dir / "zero_rewards_histogram.png"
    plot_discounted_returns_histogram(
        histories=mock_histories, policy=policy, environment=env, cache_path=output_path
    )

    # Verify plot was created
    assert output_path.exists()


def test_plot_discounted_returns_histogram_negative_rewards(temp_cache_dir):
    """Test that plot_discounted_returns_histogram properly handles histories with all negative rewards.

    Purpose: Validates proper handling when all rewards are negative (edge case)

    Given: TigerPOMDP environment, StandardSparseSampling policy, and History objects with negative rewards
    When: plot_discounted_returns_histogram is called with negative-reward histories
    Then: Histogram plot is generated successfully for negative-value data

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=2, depth=1)

    # Create histories with negative rewards
    mock_histories = []
    for i in range(5):
        history = create_mock_history([-10.0, -5.0, -2.0])
        mock_histories.append(history)

    # Execute with negative-reward histories
    output_path = temp_cache_dir / "negative_rewards_histogram.png"
    plot_discounted_returns_histogram(
        histories=mock_histories, policy=policy, environment=env, cache_path=output_path
    )

    # Verify plot was created
    assert output_path.exists()


def test_plot_discounted_returns_histogram_large_rewards(temp_cache_dir):
    """Test that plot_discounted_returns_histogram properly handles histories with large rewards.

    Purpose: Validates proper handling when rewards are very large (edge case)

    Given: TigerPOMDP environment, StandardSparseSampling policy, and History objects with large rewards
    When: plot_discounted_returns_histogram is called with large-reward histories
    Then: Histogram plot is generated successfully for large-value data

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    policy = SparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=2, depth=1)

    # Create histories with large rewards
    mock_histories = []
    for i in range(5):
        history = create_mock_history([1000.0, 2000.0, 5000.0])
        mock_histories.append(history)

    # Execute with large-reward histories
    output_path = temp_cache_dir / "large_rewards_histogram.png"
    plot_discounted_returns_histogram(
        histories=mock_histories, policy=policy, environment=env, cache_path=output_path
    )

    # Verify plot was created
    assert output_path.exists()
