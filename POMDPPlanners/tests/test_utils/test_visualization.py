import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import mlflow
import os

from POMDPPlanners.utils.visualization import plot_metrics_comparison, plot_policy_returns, AgentPath
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import DiscreteLightDarkPOMDP
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import ContinuousLightDarkPOMDPDiscreteActions
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
        plot_metrics_comparison(
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
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    
    # Create agent paths for Tiger POMDP
    agent_paths = [
        AgentPath(
            name="Listen First",
            state_sequence=["tiger_left"] * 3,
            action_sequence=["listen", "listen", "open_right"],
            n_particles=10
        ),
        AgentPath(
            name="Direct Open",
            state_sequence=["tiger_left"] * 2,
            action_sequence=["listen", "open_right"],
            n_particles=10
        )
    ]
    
    # Execute
    output_path = temp_cache_dir / "tiger_policy_returns.png"
    plot_policy_returns(
        env=env,
        agent_paths=agent_paths,
        output_path=output_path,
        n_samples=100
    )
    
    # Verify plot was created
    assert output_path.exists()


def test_plot_policy_returns_discrete_light_dark_pomdp(temp_cache_dir):
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
            action_sequence=["right"] * 4,
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
            action_sequence=["up"] * 4,
            n_particles=10
        )
    ]
    
    # Execute
    output_path = temp_cache_dir / "discrete_ld_policy_returns.png"
    plot_policy_returns(
        env=env,
        agent_paths=agent_paths,
        output_path=output_path,
        n_samples=100
    )
    
    # Verify plot was created
    assert output_path.exists()


def test_plot_policy_returns_continuous_light_dark_pomdp(temp_cache_dir):
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
            action_sequence=["right"] * 4,
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
            action_sequence=["up"] * 4,
            n_particles=10
        )
    ]
    
    # Execute
    output_path = temp_cache_dir / "continuous_ld_policy_returns.png"
    plot_policy_returns(
        env=env,
        agent_paths=agent_paths,
        output_path=output_path,
        n_samples=100
    )
    
    # Verify plot was created
    assert output_path.exists()


def test_plot_policy_returns_empty_paths(temp_cache_dir):
    # Setup
    env = TigerPOMDP(discount_factor=0.95)
    
    # Test with empty agent paths
    with pytest.raises(ValueError, match="agent_paths cannot be empty"):
        plot_policy_returns(
            env=env,
            agent_paths=[],
            output_path=temp_cache_dir / "empty_paths.png",
            n_samples=100
        )


def test_plot_policy_returns_invalid_n_samples(temp_cache_dir):
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
            output_path=temp_cache_dir / "invalid_samples.png",
            n_samples=0
        )
