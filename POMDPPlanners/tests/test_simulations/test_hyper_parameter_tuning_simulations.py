import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import pandas as pd
import time

from POMDPPlanners.core.simulation import NumericalHyperParameter
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import HyperParameterOptimizer


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
        # Wait a bit to ensure all Redis connections are closed
        time.sleep(0.1)
        # Try multiple times to delete the directory
        for _ in range(5):
            try:
                shutil.rmtree(temp_path)
                break
            except PermissionError:
                time.sleep(0.1)
            except Exception as e:
                print(f"Error cleaning up temp directory: {e}")
                break


@pytest.fixture
def optimizer(temp_cache_dir):
    """Create a HyperParameterOptimizer instance for testing."""
    return HyperParameterOptimizer(
        cache_dir_path=temp_cache_dir,
        experiment_name="test_optimization",
        n_jobs=1,
        confidence_interval_level=0.95,
    )


def test_optimizer_initialization(temp_cache_dir):
    """Test that the optimizer initializes correctly."""
    optimizer = HyperParameterOptimizer(
        cache_dir_path=temp_cache_dir,
        experiment_name="test_init",
        n_jobs=2,
        confidence_interval_level=0.99,
    )
    
    assert optimizer.cache_dir_path == temp_cache_dir
    assert optimizer.experiment_name == "test_init"
    assert optimizer.n_jobs == 2
    assert optimizer.confidence_interval_level == 0.99
    assert optimizer.mlruns_path == temp_cache_dir / "mlruns"
    assert optimizer.mlruns_path.exists()


def test_optimize_policy_parameters(optimizer):
    """Test optimizing parameters for a single environment-policy pair."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    param_ranges = [
        NumericalHyperParameter(name="branching_factor", low=2, high=3),
        NumericalHyperParameter(name="depth", low=2, high=3),
    ]

    # Execute
    best_params, best_value, histories = optimizer.optimize_policy_parameters(
        environment=environment,
        policy_class=StandardSparseSamplingDiscreteActionsPlanner,
        param_ranges=param_ranges,
        num_episodes=2,
        num_steps=2,
        n_particles=10,
        n_trials=2,
    )

    # Assert
    assert isinstance(best_params, dict)
    assert "branching_factor" in best_params
    assert "depth" in best_params
    assert isinstance(best_value, float)
    assert isinstance(histories, list)
    assert len(histories) > 0


def test_optimize_policy_parameters_invalid_params(optimizer):
    """Test that invalid parameters raise appropriate errors."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    param_ranges = [
        NumericalHyperParameter(name="branching_factor", low=2, high=3),
        NumericalHyperParameter(name="depth", low=2, high=3),
    ]

    # Test invalid parameters
    with pytest.raises(AssertionError):
        optimizer.optimize_policy_parameters(
            environment=environment,
            policy_class=StandardSparseSamplingDiscreteActionsPlanner,
            param_ranges=param_ranges,
            num_episodes=0,  # Invalid num_episodes
            num_steps=2,
            n_particles=10,
            n_trials=2,
        )


def test_optimize_multiple_environments(optimizer):
    """Test optimizing parameters for multiple environment-policy pairs."""
    # Setup
    environment1 = TigerPOMDP(discount_factor=0.95)
    environment2 = TigerPOMDP(discount_factor=0.99)
    param_ranges = [
        NumericalHyperParameter(name="branching_factor", low=2, high=3),
        NumericalHyperParameter(name="depth", low=2, high=3),
    ]

    environment_policy_pairs = [
        (environment1, (StandardSparseSamplingDiscreteActionsPlanner, param_ranges)),
        (environment2, (StandardSparseSamplingDiscreteActionsPlanner, param_ranges)),
    ]

    # Execute
    results, df = optimizer.optimize_multiple_environments(
        environment_policy_pairs=environment_policy_pairs,
        num_episodes=2,
        num_steps=2,
        n_particles=10,
        n_trials=2,
    )

    # Assert
    assert isinstance(results, list)
    assert len(results) == 2  # One result per environment
    for result in results:
        assert isinstance(result, tuple)
        assert len(result) == 3  # (best_params, best_value, histories)
        best_params, best_value, histories = result
        assert isinstance(best_params, dict)
        assert isinstance(best_value, float)
        assert isinstance(histories, list)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "param_range_branching_factor" in df.columns
    assert "param_range_depth" in df.columns


def test_simulation_method(optimizer):
    """Test the simulation method directly."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=2,
    )
    initial_belief = environment.get_initial_belief(n_particles=10)

    # Execute
    histories, statistics = optimizer.simulation(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_episodes=2,
        num_steps=2,
        alpha=0.05,
    )

    # Assert
    assert isinstance(histories, list)
    assert len(histories) == 2
    assert isinstance(statistics, list)
    assert len(statistics) > 0


def test_run_multiple_episodes(optimizer):
    """Test running multiple episodes in parallel."""
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=2,
        depth=2,
    )
    initial_belief = environment.get_initial_belief(n_particles=10)

    # Execute
    histories = optimizer.run_multiple_episodes(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_episodes=2,
        num_steps=2,
    )

    # Assert
    assert isinstance(histories, list)
    assert len(histories) == 2
    for history in histories:
        assert isinstance(history, list)
        assert len(history) == 2  # num_steps
