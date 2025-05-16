import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import mlflow
import json
import os
import optuna
import pandas as pd
import time

from POMDPPlanners.core.simulation import MetricValue, NumericalHyperParameter
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    create_policy_optimization_objective,
    optimize_policy_parameters_with_optuna,
    optimize_policy_parameters_for_multiple_environments,
)


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


def test_create_policy_optimization_objective():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)

    # Define parameter ranges for testing
    param_ranges = {
        "branching_factor": {"type": "int", "low": 2, "high": 5},
        "depth": {"type": "int", "low": 2, "high": 4},
    }

    # Create a simple evaluation function
    def evaluation_function(policy):
        assert isinstance(policy, StandardSparseSamplingDiscreteActionsPlanner)
        assert hasattr(policy, "environment")
        return (1.0, (0.5, 1.5))  # Return a tuple to simulate statistics

    # Create a mock trial that implements all suggest methods
    class MockTrial:
        def suggest_int(self, name, low, high):
            return (low + high) // 2

        def suggest_float(self, name, low, high, log=False):
            return (low + high) / 2

        def suggest_categorical(self, name, choices):
            return choices[0]

    # Execute
    objective = create_policy_optimization_objective(
        policy_class=StandardSparseSamplingDiscreteActionsPlanner,
        param_ranges=param_ranges,
        evaluation_function=evaluation_function,
        environment=environment,
    )

    # Test the objective function
    result = objective(MockTrial())

    # Assert
    assert result == 1.0  # Should return the mean value from the tuple


def test_optimize_policy_parameters_with_optuna(temp_cache_dir):
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    param_ranges = [
        NumericalHyperParameter(name="branching_factor", low=2, high=3),
        NumericalHyperParameter(name="depth", low=2, high=3),
    ]

    # Execute
    best_params, best_value, histories = optimize_policy_parameters_with_optuna(
        environment=environment,
        policy_class=StandardSparseSamplingDiscreteActionsPlanner,
        param_ranges=param_ranges,
        num_episodes=2,
        num_steps=2,
        n_particles=10,
        cache_dir_path=temp_cache_dir,
        n_trials=2,
    )

    # Assert
    assert isinstance(best_params, dict)
    assert "branching_factor" in best_params
    assert "depth" in best_params
    assert isinstance(best_value, float)
    assert isinstance(histories, list)
    assert len(histories) > 0


def test_optimize_policy_parameters_with_optuna_invalid_params(temp_cache_dir):
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    param_ranges = [
        NumericalHyperParameter(name="branching_factor", low=2, high=3),
        NumericalHyperParameter(name="depth", low=2, high=3),
    ]

    # Test invalid parameters
    with pytest.raises(AssertionError):
        optimize_policy_parameters_with_optuna(
            environment=environment,
            policy_class=StandardSparseSamplingDiscreteActionsPlanner,
            param_ranges=param_ranges,
            num_episodes=0,  # Invalid num_episodes
            num_steps=2,
            n_particles=10,
            cache_dir_path=temp_cache_dir,
            n_trials=2,
        )


def test_optimize_policy_parameters_for_multiple_environments(temp_cache_dir):
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
    results, df = optimize_policy_parameters_for_multiple_environments(
        environment_policy_pairs=environment_policy_pairs,
        num_episodes=2,
        num_steps=2,
        n_particles=10,
        cache_dir_path=temp_cache_dir,
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
