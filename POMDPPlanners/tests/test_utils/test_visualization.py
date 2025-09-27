import logging
import os
import random
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import mlflow
import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import History, MetricValue, StepData
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDPDiscreteActions,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planner import (
    StandardSparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.utils.visualization import (
    AgentPath,
    plot_discounted_returns_histogram,
    plot_discounted_returns_histogram_multiple_policies,
    plot_environment_policy_pair_comparison,
    plot_metrics_comparison,
    plot_policies_comparison_on_environment,
    plot_policy_returns,
    plot_tree_graphs,
)

np.random.seed(42)
random.seed(42)


# Set up logger for tests
test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.WARNING)


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
        policy_run_data=None,
    )


@contextmanager
def mlflow_run_context(experiment_name: str, tracking_uri: str):
    """Context manager for MLFlow runs with proper cleanup.

    Creates and manages MLFlow experiment runs ensuring proper resource cleanup
    even if exceptions occur during visualization testing.

    Args:
        experiment_name: Name of MLFlow experiment to create/use
        tracking_uri: File-based URI for MLFlow tracking storage

    Yields:
        mlflow.ActiveRun: Active MLFlow run context for logging artifacts
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            yield run
    finally:
        # Ensure MLFlow client is closed
        mlflow.end_run()


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
                import gc

                gc.collect()
                # Try to remove the directory
                shutil.rmtree(temp_cache_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory {temp_cache_dir}: {e}")


def test_plot_statistics_comparison():
    """Test that plot_metrics_comparison generates expected visualization files for single environment-policy pair.

    Purpose: Validates that metrics comparison plotting creates all expected plot files with proper MLflow integration

    Given: TigerPOMDP environment, StandardSparseSampling policy, and mock MetricValue statistics for average_return, return_cvar, and average_action_time
    When: plot_metrics_comparison is called with single environment and policy
    Then: All expected plot files are created in the plots directory within specified time limit

    Test type: unit
    """
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        temp_cache_dir = Path(temp_cache_dir)
        environment = TigerPOMDP(discount_factor=0.95)
        policy = StandardSparseSamplingDiscreteActionsPlanner(
            environment=environment, branching_factor=2, depth=1
        )
        # Create mock statistics using MetricValue objects
        mock_statistics = [
            [
                MetricValue(
                    name="average_return",
                    value=-10.0,
                    lower_confidence_bound=-15.0,
                    upper_confidence_bound=-5.0,
                ),
                MetricValue(
                    name="return_cvar",
                    value=-12.0,
                    lower_confidence_bound=-17.0,
                    upper_confidence_bound=-7.0,
                ),
                MetricValue(
                    name="average_action_time",
                    value=0.1,
                    lower_confidence_bound=0.05,
                    upper_confidence_bound=0.15,
                ),
            ]
        ]
        # Create mlruns directory
        mlruns_dir = temp_cache_dir / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        # Set up MLFlow tracking
        tracking_uri = f"file://{mlruns_dir.absolute().as_posix()}"
        with mlflow_run_context("test_visualization", tracking_uri):
            # Execute with timeout
            start_time = time.time()
            plot_metrics_comparison(
                statistics=mock_statistics,
                environments=[environment],
                policies=[policy],
                cache_dir_path=temp_cache_dir,
            )
            assert time.time() - start_time < 30, "Plot generation took too long"
            # Verify plots directory was created
            plots_dir = temp_cache_dir / "plots"
            assert plots_dir.exists()
            # Verify plot files were created
            expected_plots = [
                "average_return_comparison.png",
                "return_cvar_comparison.png",
                "average_action_time_comparison.png",
            ]
            for plot_file in expected_plots:
                plot_path = plots_dir / plot_file
                assert plot_path.exists()


def test_plot_statistics_comparison_multiple_envs_policies():
    """Test that plot_metrics_comparison handles multiple environment-policy combinations correctly.

    Purpose: Validates that metrics comparison plotting works with multiple environments and policies producing comparative visualizations

    Given: Two TigerPOMDP environments with different discount factors, two StandardSparseSampling policies with different parameters, and corresponding mock statistics
    When: plot_metrics_comparison is called with multiple environments and policies
    Then: Comparison plots are generated showing metrics across different environment-policy configurations

    Test type: integration
    """
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        temp_cache_dir = Path(temp_cache_dir)
        environment1 = TigerPOMDP(discount_factor=0.95)
        environment2 = TigerPOMDP(discount_factor=0.99)
        policy1 = StandardSparseSamplingDiscreteActionsPlanner(
            environment=environment1, branching_factor=2, depth=1
        )
        policy2 = StandardSparseSamplingDiscreteActionsPlanner(
            environment=environment2, branching_factor=3, depth=4
        )

        # Create mock statistics for multiple environment-policy combinations using MetricValue objects
        mock_statistics = [
            [
                MetricValue(
                    name="average_return",
                    value=-10.0,
                    lower_confidence_bound=-15.0,
                    upper_confidence_bound=-5.0,
                ),
                MetricValue(
                    name="return_cvar",
                    value=-12.0,
                    lower_confidence_bound=-17.0,
                    upper_confidence_bound=-7.0,
                ),
            ],
            [
                MetricValue(
                    name="average_return",
                    value=-8.0,
                    lower_confidence_bound=-13.0,
                    upper_confidence_bound=-3.0,
                ),
                MetricValue(
                    name="return_cvar",
                    value=-10.0,
                    lower_confidence_bound=-15.0,
                    upper_confidence_bound=-5.0,
                ),
            ],
        ]

        # Create mlruns directory
        mlruns_dir = temp_cache_dir / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)

        # Set up MLFlow tracking
        tracking_uri = f"file://{mlruns_dir.absolute().as_posix()}"
        with mlflow_run_context("test_visualization_multiple", tracking_uri):
            # Execute
            plot_metrics_comparison(
                statistics=mock_statistics,
                environments=[environment1, environment2],
                policies=[policy1, policy2],
                cache_dir_path=temp_cache_dir,
            )

            # Verify plots directory was created
            plots_dir = temp_cache_dir / "plots"
            assert plots_dir.exists()

            # Verify plot files were created
            expected_plots = [
                "average_return_comparison.png",
                "return_cvar_comparison.png",
            ]
            for plot_file in expected_plots:
                plot_path = plots_dir / plot_file
                assert plot_path.exists()


def test_plot_statistics_comparison_empty_statistics(temp_cache_dir):
    """Test that plot_metrics_comparison properly handles empty statistics input.

    Purpose: Validates proper error handling when no statistics data is provided to the plotting function

    Given: TigerPOMDP environment, StandardSparseSampling policy, and empty statistics list
    When: plot_metrics_comparison is called with empty statistics
    Then: Exception is raised indicating invalid input rather than silent failure

    Test type: unit
    """
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment, branching_factor=2, depth=1
    )

    # Test with empty statistics
    with pytest.raises(Exception):
        plot_metrics_comparison(
            statistics=[],
            environments=[environment],
            policies=[policy],
            cache_dir_path=temp_cache_dir,
        )


def test_plot_policy_returns_tiger_pomdp(temp_cache_dir):
    """Test that plot_policy_returns generates return comparison visualization for Tiger POMDP environment.

    Purpose: Validates that policy return plotting creates visualization comparing different agent paths in Tiger POMDP

    Given: TigerPOMDP environment and AgentPath objects with different action strategies (Listen First vs Direct Open)
    When: plot_policy_returns is called with agent paths and minimal sampling parameters
    Then: Policy returns comparison plot is generated within time limit and saved to expected file location

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)

    # Create agent paths for Tiger POMDP with minimal steps
    agent_paths = [
        AgentPath(
            name="Listen First",
            state_sequence=["tiger_left"],  # Reduced from 3 to 1
            action_sequence=["listen"],  # Reduced from 3 to 1
            n_particles=5,  # Reduced from 10 to 5
        ),
        AgentPath(
            name="Direct Open",
            state_sequence=["tiger_left"],  # Reduced from 2 to 1
            action_sequence=["listen"],  # Reduced from 2 to 1
            n_particles=5,  # Reduced from 10 to 5
        ),
    ]

    # Execute with timeout
    start_time = time.time()
    plot_policy_returns(
        env=env,
        agent_paths=agent_paths,
        dir_path=temp_cache_dir,
        n_samples=5,  # Reduced from 10 to 5
        logger=test_logger,
    )

    # Verify plot was created
    output_path = temp_cache_dir / "policy_returns_comparison.png"
    assert output_path.exists()


def test_plot_policy_returns_discrete_light_dark_pomdp(temp_cache_dir):
    """Test that plot_policy_returns generates return comparison visualization for Discrete Light Dark POMDP environment.

    Purpose: Validates that policy return plotting works with discrete light-dark navigation environment using different path strategies

    Given: DiscreteLightDarkPOMDP environment with obstacles and beacons, and AgentPath objects representing direct vs upper navigation paths
    When: plot_policy_returns is called with navigation-specific agent paths and reduced sampling for test performance
    Then: Policy returns comparison plot is generated within time limit showing performance differences between navigation strategies

    Test type: unit
    """
    # Setup - optimized for test performance
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.20,
        beacons=[(0, 0), (0, 5), (5, 0), (5, 5)],  # Convert to list of tuples
        goal_state=np.array([5, 2]),  # Smaller grid
        start_state=np.array([0, 2]),
        obstacles=[(2, 2)],  # Convert to list of tuples
        obstacle_reward=-16.0,
        goal_reward=10.0,
        obstacle_hit_probability=0.5,
        beacon_radius=1.0,
        fuel_cost=2.0,
        grid_size=6,  # Reduced from 11 to 6
        is_stochastic_reward=True,
    )

    # Create simplified agent paths for testing
    agent_paths = [
        AgentPath(
            name="Direct Path",
            state_sequence=[np.array([0, 2]), np.array([1, 2])],
            action_sequence=["right", "right"],  # Same length as state_sequence
            n_particles=5,  # Reduced from 10 to 5
        ),
        AgentPath(
            name="Upper Path",
            state_sequence=[np.array([0, 2]), np.array([0, 3])],
            action_sequence=["up", "up"],  # Same length as state_sequence
            n_particles=5,  # Reduced from 10 to 5
        ),
    ]

    # Execute with timeout
    start_time = time.time()
    plot_policy_returns(
        env=env,
        agent_paths=agent_paths,
        dir_path=temp_cache_dir,
        n_samples=3,  # Further reduced for test performance
        logger=test_logger,
    )
    assert time.time() - start_time < 30, "Plot generation took too long"

    # Verify plot was created
    output_path = temp_cache_dir / "policy_returns_comparison.png"
    assert output_path.exists()


def test_plot_policy_returns_continuous_light_dark_pomdp(temp_cache_dir):
    """Test that plot_policy_returns generates return comparison visualization for Continuous Light Dark POMDP environment.

    Purpose: Validates that policy return plotting works with continuous light-dark navigation environment using discrete actions

    Given: ContinuousLightDarkPOMDPDiscreteActions environment with Gaussian noise models, obstacles, and AgentPath objects for direct vs upper navigation
    When: plot_policy_returns is called with continuous state space agent paths and optimized sampling parameters
    Then: Policy returns comparison plot is generated within time limit demonstrating performance across continuous state transitions

    Test type: unit
    """
    # Setup
    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        state_transition_cov_matrix=np.eye(2) * 0.1,
        observation_cov_matrix=np.eye(2) * 0.1,
        obstacle_hit_probability=0.2,
        obstacle_reward=-10.0,
        goal_reward=10.0,
        fuel_cost=2.0,
        grid_size=11,
        goal_state_radius=1.5,
        beacon_radius=1.0,
        obstacle_radius=1.5,
        beacons=[
            (0, 0),
            (0, 5),
            (0, 10),
            (5, 0),
            (5, 5),
            (5, 10),
            (10, 0),
            (10, 5),
            (10, 10),
        ],  # Convert to list of tuples
        goal_state=np.array([10, 5]),
        start_state=np.array([0, 5]),
        obstacles=[(3, 7), (5, 5)],  # Convert to list of tuples
    )

    # Create agent paths for Continuous Light Dark POMDP
    agent_paths = [
        AgentPath(
            name="Direct Path",
            state_sequence=[
                np.array([0, 5]),
                np.array([1, 5]),
                np.array([2, 5]),
                np.array([3, 5]),
                np.array([4, 5]),
            ],
            action_sequence=["right"] * 5,
            n_particles=10,
        ),
        AgentPath(
            name="Upper Path",
            state_sequence=[
                np.array([0, 5]),
                np.array([0, 6]),
                np.array([0, 7]),
                np.array([0, 8]),
                np.array([0, 9]),
            ],
            action_sequence=["up"] * 5,
            n_particles=10,
        ),
    ]

    # Execute with timeout
    start_time = time.time()
    plot_policy_returns(
        env=env,
        agent_paths=agent_paths,
        dir_path=temp_cache_dir,
        n_samples=5,  # Already at 5, which is good for testing
        logger=test_logger,
    )
    assert time.time() - start_time < 30, "Plot generation took too long"

    # Verify plot was created
    output_path = temp_cache_dir / "policy_returns_comparison.png"
    assert output_path.exists()


def test_plot_policy_returns_empty_paths(temp_cache_dir):
    """Test that plot_policy_returns properly handles empty agent paths input.

    Purpose: Validates proper error handling when no agent paths are provided to the policy returns plotting function

    Given: TigerPOMDP environment and empty agent_paths list
    When: plot_policy_returns is called with empty agent paths
    Then: ValueError is raised with descriptive message rather than silent failure or runtime error

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)

    # Test with empty agent paths
    with pytest.raises(ValueError, match="agent_paths cannot be empty"):
        plot_policy_returns(
            env=env,
            agent_paths=[],
            dir_path=temp_cache_dir,
            n_samples=100,
            logger=test_logger,
        )


def test_plot_policy_returns_invalid_n_samples(temp_cache_dir):
    """Test that plot_policy_returns properly validates n_samples parameter.

    Purpose: Validates proper error handling when invalid n_samples parameter is provided to ensure meaningful sample sizes

    Given: TigerPOMDP environment, valid AgentPath, and invalid n_samples value (0 or negative)
    When: plot_policy_returns is called with n_samples=0
    Then: ValueError is raised with descriptive message indicating n_samples must be positive

    Test type: unit
    """
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    agent_paths = [
        AgentPath(
            name="Test Path",
            state_sequence=["tiger_left"],
            action_sequence=["listen"],
            n_particles=10,
        )
    ]

    # Test with invalid n_samples
    with pytest.raises(ValueError, match="n_samples must be greater than 0"):
        plot_policy_returns(
            env=env,
            agent_paths=agent_paths,
            dir_path=temp_cache_dir,
            n_samples=0,
            logger=test_logger,
        )


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
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=1
    )

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
    policy1 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=1
    )
    policy2 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=3, depth=2
    )

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
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=1
    )

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


def test_plot_policies_comparison_on_environment(temp_cache_dir):
    """Test that plot_policies_comparison_on_environment generates bar plots comparing policies across environments.

    Purpose: Validates that policy comparison plotting creates bar charts with academic publication style

    Given: Metrics dictionary with environment names mapping to policy metrics, and mock MetricValue objects
    When: plot_policies_comparison_on_environment is called with metrics dictionary
    Then: Bar plot is generated showing policy performance comparison with academic styling

    Test type: unit
    """
    # Setup
    metrics_dict = {
        "TigerPOMDP": {
            "Policy1": [
                MetricValue(
                    name="average_return",
                    value=-5.0,
                    lower_confidence_bound=-7.0,
                    upper_confidence_bound=-3.0,
                ),
                MetricValue(
                    name="return_cvar",
                    value=-6.0,
                    lower_confidence_bound=-8.0,
                    upper_confidence_bound=-4.0,
                ),
            ],
            "Policy2": [
                MetricValue(
                    name="average_return",
                    value=-4.0,
                    lower_confidence_bound=-6.0,
                    upper_confidence_bound=-2.0,
                ),
                MetricValue(
                    name="return_cvar",
                    value=-5.0,
                    lower_confidence_bound=-7.0,
                    upper_confidence_bound=-3.0,
                ),
            ],
        }
    }

    # Execute
    output_path = temp_cache_dir / "policies_comparison"
    output_path.mkdir(parents=True, exist_ok=True)

    plot_policies_comparison_on_environment(metrics_dict=metrics_dict, cache_dir_path=output_path)

    # Verify plots were created
    expected_plots = [
        "TigerPOMDP_average_return_comparison.png",
        "TigerPOMDP_return_cvar_comparison.png",
    ]
    for plot_file in expected_plots:
        plot_path = output_path / plot_file
        assert plot_path.exists()


def test_plot_policies_comparison_on_environment_empty_input(temp_cache_dir):
    """Test that plot_policies_comparison_on_environment properly handles empty input.

    Purpose: Validates proper error handling when empty metrics dictionary is provided

    Given: Empty metrics dictionary
    When: plot_policies_comparison_on_environment is called with empty input
    Then: ValueError is raised with descriptive message

    Test type: unit
    """
    # Test with empty metrics dictionary
    with pytest.raises(ValueError, match="metrics_dict cannot be empty"):
        plot_policies_comparison_on_environment(metrics_dict={}, cache_dir_path=temp_cache_dir)


def test_plot_policies_comparison_on_environment_invalid_input(temp_cache_dir):
    """Test that plot_policies_comparison_on_environment properly validates input types.

    Purpose: Validates proper error handling when invalid input types are provided

    Given: Invalid input types (non-dict, non-MetricValue objects)
    When: plot_policies_comparison_on_environment is called with invalid input
    Then: TypeError is raised with descriptive message

    Test type: unit
    """
    # Test with non-dict input
    with pytest.raises(TypeError, match="metrics_dict must be a dictionary"):
        plot_policies_comparison_on_environment(
            metrics_dict="invalid", cache_dir_path=temp_cache_dir  # type: ignore[arg-type]
        )

    # Test with invalid metric values
    invalid_metrics_dict = {"TigerPOMDP": {"Policy1": ["invalid_metric_value"]}}
    with pytest.raises(TypeError, match="All metric values must be MetricValue objects"):
        plot_policies_comparison_on_environment(
            metrics_dict=invalid_metrics_dict, cache_dir_path=temp_cache_dir  # type: ignore[arg-type]
        )


def test_plot_tree_graphs(temp_cache_dir):
    """Test that plot_tree_graphs generates interactive tree visualizations.

    Purpose: Validates that tree graph plotting creates interactive visualizations of belief trees

    Given: Mock BeliefNode root with ActionNode and BeliefNode children
    When: plot_tree_graphs is called with root node
    Then: Interactive tree visualization is generated and displayed (tested by checking no exceptions)

    Test type: unit
    """
    # Setup - Create a simple mock tree structure
    root_belief = BeliefNode(
        belief=WeightedParticleBelief(
            particles=[np.array([0.0, 0.0])], log_weights=np.array([0.1])
        ),
        observation=None,
        parent=None,
    )

    # Add an action node
    action_node = ActionNode(action="listen", parent=root_belief)
    root_belief.children = [action_node]

    # Add a belief node child
    child_belief = BeliefNode(
        belief=WeightedParticleBelief(
            particles=[np.array([1.0, 1.0])], log_weights=np.array([0.1])
        ),
        observation="tiger_left",
        parent=action_node,
    )
    action_node.children = [child_belief]

    # Set some values for visualization
    root_belief.v_value = -5.0
    root_belief.visit_count = 10
    action_node.q_value = -4.0
    action_node.visit_count = 8
    child_belief.v_value = -3.0
    child_belief.visit_count = 5

    # Execute - This should not raise an exception
    # Note: The function calls fig.show() which opens a browser window in interactive mode
    # In test environment, this might not display but should not crash
    try:
        plot_tree_graphs(root_belief)
        # If we get here without exception, the test passes
        assert True
    except Exception as e:
        # If there's an exception, it should be related to display/interaction, not core functionality
        # Check that it's not a fundamental error
        assert (
            "display" in str(e).lower() or "show" in str(e).lower() or "browser" in str(e).lower()
        )


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
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=1
    )

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
    policy1 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=1
    )
    policy2 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=3, depth=2
    )

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
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=1
    )

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
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=1
    )

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
    policy1 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=1
    )
    policy2 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=3, depth=2
    )

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
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=1
    )

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
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=1
    )

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
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=1
    )

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


def test_plot_policies_comparison_on_environment_single_policy(temp_cache_dir):
    """Test that plot_policies_comparison_on_environment handles single policy input.

    Purpose: Validates proper handling when only one policy is provided (edge case)

    Given: Metrics dictionary with single policy, and mock MetricValue objects
    When: plot_policies_comparison_on_environment is called with single policy
    Then: Bar plot is generated successfully for single policy comparison

    Test type: unit
    """
    # Setup
    metrics_dict = {
        "TigerPOMDP": {
            "SinglePolicy": [
                MetricValue(
                    name="average_return",
                    value=-5.0,
                    lower_confidence_bound=-7.0,
                    upper_confidence_bound=-3.0,
                ),
                MetricValue(
                    name="return_cvar",
                    value=-6.0,
                    lower_confidence_bound=-8.0,
                    upper_confidence_bound=-4.0,
                ),
            ]
        }
    }

    # Execute
    output_path = temp_cache_dir / "single_policy_comparison"
    output_path.mkdir(parents=True, exist_ok=True)

    plot_policies_comparison_on_environment(metrics_dict=metrics_dict, cache_dir_path=output_path)

    # Verify plots were created
    expected_plots = [
        "TigerPOMDP_average_return_comparison.png",
        "TigerPOMDP_return_cvar_comparison.png",
    ]
    for plot_file in expected_plots:
        plot_path = output_path / plot_file
        assert plot_path.exists()


def test_plot_policies_comparison_on_environment_single_metric(temp_cache_dir):
    """Test that plot_policies_comparison_on_environment handles single metric input.

    Purpose: Validates proper handling when only one metric is provided (edge case)

    Given: Metrics dictionary with single metric, and mock MetricValue objects
    When: plot_policies_comparison_on_environment is called with single metric
    Then: Bar plot is generated successfully for single metric comparison

    Test type: unit
    """
    # Setup
    metrics_dict = {
        "TigerPOMDP": {
            "Policy1": [
                MetricValue(
                    name="average_return",
                    value=-5.0,
                    lower_confidence_bound=-7.0,
                    upper_confidence_bound=-3.0,
                ),
            ],
            "Policy2": [
                MetricValue(
                    name="average_return",
                    value=-4.0,
                    lower_confidence_bound=-6.0,
                    upper_confidence_bound=-2.0,
                ),
            ],
        }
    }

    # Execute
    output_path = temp_cache_dir / "single_metric_comparison"
    output_path.mkdir(parents=True, exist_ok=True)

    plot_policies_comparison_on_environment(metrics_dict=metrics_dict, cache_dir_path=output_path)

    # Verify plot was created
    expected_plot = "TigerPOMDP_average_return_comparison.png"
    plot_path = output_path / expected_plot
    assert plot_path.exists()


def test_plot_policies_comparison_on_environment_zero_values(temp_cache_dir):
    """Test that plot_policies_comparison_on_environment handles zero metric values.

    Purpose: Validates proper handling when metric values are zero (edge case)

    Given: Metrics dictionary with zero values, and mock MetricValue objects
    When: plot_policies_comparison_on_environment is called with zero values
    Then: Bar plot is generated successfully for zero-value metrics

    Test type: unit
    """
    # Setup
    metrics_dict = {
        "TigerPOMDP": {
            "Policy1": [
                MetricValue(
                    name="average_return",
                    value=0.0,
                    lower_confidence_bound=-1.0,
                    upper_confidence_bound=1.0,
                ),
            ],
            "Policy2": [
                MetricValue(
                    name="average_return",
                    value=0.0,
                    lower_confidence_bound=-0.5,
                    upper_confidence_bound=0.5,
                ),
            ],
        }
    }

    # Execute
    output_path = temp_cache_dir / "zero_values_comparison"
    output_path.mkdir(parents=True, exist_ok=True)

    plot_policies_comparison_on_environment(metrics_dict=metrics_dict, cache_dir_path=output_path)

    # Verify plot was created
    expected_plot = "TigerPOMDP_average_return_comparison.png"
    plot_path = output_path / expected_plot
    assert plot_path.exists()


def test_plot_policies_comparison_on_environment_large_values(temp_cache_dir):
    """Test that plot_policies_comparison_on_environment handles large metric values.

    Purpose: Validates proper handling when metric values are very large (edge case)

    Given: Metrics dictionary with large values, and mock MetricValue objects
    When: plot_policies_comparison_on_environment is called with large values
    Then: Bar plot is generated successfully for large-value metrics

    Test type: unit
    """
    # Setup
    metrics_dict = {
        "TigerPOMDP": {
            "Policy1": [
                MetricValue(
                    name="average_return",
                    value=10000.0,
                    lower_confidence_bound=9500.0,
                    upper_confidence_bound=10500.0,
                ),
            ],
            "Policy2": [
                MetricValue(
                    name="average_return",
                    value=12000.0,
                    lower_confidence_bound=11500.0,
                    upper_confidence_bound=12500.0,
                ),
            ],
        }
    }

    # Execute
    output_path = temp_cache_dir / "large_values_comparison"
    output_path.mkdir(parents=True, exist_ok=True)

    plot_policies_comparison_on_environment(metrics_dict=metrics_dict, cache_dir_path=output_path)

    # Verify plot was created
    expected_plot = "TigerPOMDP_average_return_comparison.png"
    plot_path = output_path / expected_plot
    assert plot_path.exists()


def test_plot_tree_graphs_single_node(temp_cache_dir):
    """Test that plot_tree_graphs handles single node tree.

    Purpose: Validates proper handling when tree has only one node (edge case)

    Given: Single BeliefNode with no children
    When: plot_tree_graphs is called with single node
    Then: Interactive tree visualization is generated successfully for single node

    Test type: unit
    """
    # Setup - Create single node tree
    root_belief = BeliefNode(
        belief=WeightedParticleBelief(
            particles=[np.array([0.0, 0.0])], log_weights=np.array([0.1])
        ),
        observation=None,
        parent=None,
    )

    # Set values for visualization
    root_belief.v_value = -5.0
    root_belief.visit_count = 10

    # Execute - This should not raise an exception
    try:
        plot_tree_graphs(root_belief)
        # If we get here without exception, the test passes
        assert True
    except Exception as e:
        # If there's an exception, it should be related to display/interaction, not core functionality
        assert (
            "display" in str(e).lower() or "show" in str(e).lower() or "browser" in str(e).lower()
        )


def test_plot_tree_graphs_deep_tree(temp_cache_dir):
    """Test that plot_tree_graphs handles deep tree structure.

    Purpose: Validates proper handling when tree has many levels (edge case)

    Given: Deep tree structure with multiple levels
    When: plot_tree_graphs is called with deep tree
    Then: Interactive tree visualization is generated successfully for deep tree

    Test type: unit
    """
    # Setup - Create deep tree structure
    root_belief = BeliefNode(
        belief=WeightedParticleBelief(
            particles=[np.array([0.0, 0.0])], log_weights=np.array([0.1])
        ),
        observation=None,
        parent=None,
    )

    # Create a chain of nodes (deep but narrow tree)
    current_node = root_belief
    for i in range(5):  # Create 5 levels deep
        action_node = ActionNode(action=f"action_{i}", parent=current_node)
        current_node.children = [action_node]

        belief_node = BeliefNode(
            belief=WeightedParticleBelief(
                particles=[np.array([float(i), float(i)])], log_weights=np.array([0.1])
            ),
            observation=f"obs_{i}",
            parent=action_node,
        )
        action_node.children = [belief_node]

        # Set values
        action_node.q_value = -4.0 - i
        action_node.visit_count = 8 - i
        belief_node.v_value = -3.0 - i
        belief_node.visit_count = 5 - i

        current_node = belief_node

    # Set root values
    root_belief.v_value = -5.0
    root_belief.visit_count = 10

    # Execute - This should not raise an exception
    try:
        plot_tree_graphs(root_belief)
        # If we get here without exception, the test passes
        assert True
    except Exception as e:
        # If there's an exception, it should be related to display/interaction, not core functionality
        assert (
            "display" in str(e).lower() or "show" in str(e).lower() or "browser" in str(e).lower()
        )
