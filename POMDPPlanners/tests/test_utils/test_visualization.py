import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import mlflow
import os

from POMDPPlanners.utils.visualization import plot_statistics_comparison
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planner import (
    StandardSparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.core.simulation import MetricValue


@pytest.fixture
def temp_cache_dir():
    # Create a temporary directory for MLFlow cache
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    # Ensure the directory exists and is empty
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


def test_plot_statistics_comparison(temp_cache_dir):
    # Setup
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
    tracking_uri = f"file:///{mlruns_dir.absolute().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test_visualization")

    with mlflow.start_run():
        # Execute
        plot_statistics_comparison(
            statistics=mock_statistics,
            environments=[environment],
            policies=[policy],
            cache_dir_path=temp_cache_dir,
        )

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


def test_plot_statistics_comparison_multiple_envs_policies(temp_cache_dir):
    # Setup
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
    tracking_uri = f"file:///{mlruns_dir.absolute().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test_visualization_multiple")

    with mlflow.start_run():
        # Execute
        plot_statistics_comparison(
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
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment, branching_factor=2, depth=1
    )

    # Test with empty statistics
    with pytest.raises(Exception):
        plot_statistics_comparison(
            statistics=[],
            environments=[environment],
            policies=[policy],
            cache_dir_path=temp_cache_dir,
        )
