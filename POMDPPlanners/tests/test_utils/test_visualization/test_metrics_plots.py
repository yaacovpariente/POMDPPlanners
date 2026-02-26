import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import mlflow
import pytest

from POMDPPlanners.core.simulation import MetricValue
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import (
    SparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.utils.visualization import (
    plot_metrics_comparison,
    plot_policies_comparison_on_environment,
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
        policy = SparseSamplingDiscreteActionsPlanner(
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
        policy1 = SparseSamplingDiscreteActionsPlanner(
            environment=environment1, branching_factor=2, depth=1
        )
        policy2 = SparseSamplingDiscreteActionsPlanner(
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
    policy = SparseSamplingDiscreteActionsPlanner(
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
