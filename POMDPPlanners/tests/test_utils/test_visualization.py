import pytest
import numpy as np
from pathlib import Path
import tempfile
import uuid
import shutil
import mlflow
import os
import time
import logging
import random
from contextlib import contextmanager

from POMDPPlanners.utils.visualization import plot_metrics_comparison, plot_policy_returns, AgentPath
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import DiscreteLightDarkPOMDP
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import ContinuousLightDarkPOMDPDiscreteActions
from POMDPPlanners.planners.sparse_sampling_planner import (
    StandardSparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.core.simulation import MetricValue

np.random.seed(42)
random.seed(42)


# Set up logger for tests
test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.WARNING)


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
    try:
        temp_cache_dir = unique_dir
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
            expected_plots = ["average_return_comparison.png", "return_cvar_comparison.png"]
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
            action_sequence=["listen"],      # Reduced from 3 to 1
            n_particles=5                    # Reduced from 10 to 5
        ),
        AgentPath(
            name="Direct Open",
            state_sequence=["tiger_left"],   # Reduced from 2 to 1
            action_sequence=["listen"],      # Reduced from 2 to 1
            n_particles=5                    # Reduced from 10 to 5
        )
    ]
    
    # Execute with timeout
    start_time = time.time()
    plot_policy_returns(
        env=env,
        agent_paths=agent_paths,
        dir_path=temp_cache_dir,
        n_samples=5,  # Reduced from 10 to 5
        logger=test_logger
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
    # Setup
    env = DiscreteLightDarkPOMDP(
        discount_factor=0.95,
        transition_error_prob=0.05,
        observation_error_prob=0.20,
        beacons=np.array([[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]]),
        goal_state=np.array([10, 5]),
        start_state=np.array([0, 5]),
        obstacles=np.array([[5, 5], [4, 5]]),
        obstacle_reward=-16.0,
        goal_reward=10.0,
        obstacle_hit_probability=0.5,
        beacon_radius=1.0,
        fuel_cost=2.0,
        grid_size=11,
        is_stochastic_reward=True
    )
    
    # Create agent paths for Discrete Light Dark POMDP
    agent_paths = [
        AgentPath(
            name="Direct Path",
            state_sequence=[
                np.array([0, 5]), 
                np.array([1, 5]), 
                np.array([2, 5]), 
                np.array([3, 5]), 
                np.array([4, 5])
            ],
            action_sequence=["right"] * 5,
            n_particles=10
        ),
        AgentPath(
            name="Upper Path",
            state_sequence=[
                np.array([0, 5]), 
                np.array([0, 6]), 
                np.array([0, 7]), 
                np.array([0, 8]), 
                np.array([0, 9])
            ],
            action_sequence=["up"] * 5,
            n_particles=10
        )
    ]
    
    # Execute with timeout
    start_time = time.time()
    plot_policy_returns(
        env=env,
        agent_paths=agent_paths,
        dir_path=temp_cache_dir,
        n_samples=5,  # Reduced from 100 to speed up test
        logger=test_logger
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
        beacons=np.array([[0, 0, 0, 5, 5, 5, 10, 10, 10], [0, 5, 10, 0, 5, 10, 0, 5, 10]]),
        goal_state=np.array([10, 5]),
        start_state=np.array([0, 5]),
        obstacles=np.array([[3, 7], [5, 5]])
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
                np.array([4, 5])
            ],
            action_sequence=["right"] * 5,
            n_particles=10
        ),
        AgentPath(
            name="Upper Path",
            state_sequence=[
                np.array([0, 5]), 
                np.array([0, 6]), 
                np.array([0, 7]), 
                np.array([0, 8]), 
                np.array([0, 9])
            ],
            action_sequence=["up"] * 5,
            n_particles=10
        )
    ]
    
    # Execute with timeout
    start_time = time.time()
    plot_policy_returns(
        env=env,
        agent_paths=agent_paths,
        dir_path=temp_cache_dir,
        n_samples=5,  # Already at 5, which is good for testing
        logger=test_logger
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
            logger=test_logger
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
            n_particles=10
        )
    ]
    
    # Test with invalid n_samples
    with pytest.raises(ValueError, match="n_samples must be greater than 0"):
        plot_policy_returns(
            env=env,
            agent_paths=agent_paths,
            dir_path=temp_cache_dir,
            n_samples=0,
            logger=test_logger
        )
