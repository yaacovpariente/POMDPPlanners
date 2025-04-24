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

from POMDPPlanners.core.simulation import MetricValue
from POMDPPlanners.simulations.simulations import (
    run_episode,
    simulation,
    compare_planners,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.planners.sparse_sampling_planner import (
    StandardSparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import History, NumericalHyperParameter
from POMDPPlanners.simulations.simulations import (
    create_policy_optimization_objective,
    optimize_policy_parameters_with_optuna,
    optimize_policy_parameters_for_multiple_environments,
)


def test_run_episode_returns_history():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment, branching_factor=2, depth=3
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_steps = 5

    # Execute
    history = run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
    )

    # Assert
    assert isinstance(history, History)
    assert len(history.history) == num_steps


def test_run_episode_timing_statistics():
    # Setup
    environment = MountainCarPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment, branching_factor=2, depth=3
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_steps = 5

    # Execute
    history = run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
    )

    # Assert timing statistics are positive and reasonable
    assert (
        history.average_state_sampling_time >= 0
    )  # Can be 0 if state transitions are deterministic
    assert history.average_action_time > 0
    assert (
        history.average_observation_time >= 0
    )  # Can be 0 if observations are deterministic
    assert history.average_belief_update_time > 0
    assert history.average_reward_time >= 0  # Can be 0 if rewards are deterministic


def test_run_episode_valid_transitions():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment, branching_factor=2, depth=3
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_steps = 5

    # Execute
    history = run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
    )

    # Assert valid state transitions and observations
    for step in history.history:
        assert step.state in ["tiger_left", "tiger_right"]
        assert step.action in ["listen", "open_left", "open_right"]
        assert step.next_state in ["tiger_left", "tiger_right"]
        assert step.observation in ["hear_left", "hear_right", "hear_nothing"]
        assert isinstance(step.reward, float)


def test_run_episode_reward_calculation():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment, branching_factor=2, depth=3
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_steps = 5

    # Execute
    history = run_episode(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_steps=num_steps,
    )

    # Assert reward values are correct based on TigerPOMDP rules
    for step in history.history:
        if step.action == "listen":
            assert step.reward == -1.0
        elif step.action == "open_left":
            assert step.reward in [-100.0, 10.0]
        elif step.action == "open_right":
            assert step.reward in [-100.0, 10.0]


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


def test_simulation_parameter_validation():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment, branching_factor=2, depth=3
    )
    initial_belief = get_initial_belief(environment, n_particles=100)

    # Test invalid parameters
    with pytest.raises(AssertionError):
        simulation(
            environment="not_an_environment",
            policy=policy,
            initial_belief=initial_belief,
            num_episodes=10,
            num_steps=5,
            alpha=0.1,
        )

    with pytest.raises(AssertionError):
        simulation(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_episodes=0,  # Invalid num_episodes
            num_steps=5,
            alpha=0.1,
        )

    with pytest.raises(AssertionError):
        simulation(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_episodes=10,
            num_steps=5,
            alpha=0.1,
            confidence_interval_level=1.5,  # Invalid confidence interval
        )


def test_simulation_returns_histories_and_statistics():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment, branching_factor=2, depth=3
    )
    initial_belief = get_initial_belief(environment, n_particles=100)
    num_episodes = 2
    num_steps = 3

    # Execute
    histories, statistics = simulation(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_episodes=num_episodes,
        num_steps=num_steps,
        alpha=0.1,
    )

    # Assert
    assert len(histories) == num_episodes
    for history in histories:
        assert isinstance(history, History)
        assert len(history.history) == num_steps

    # Check statistics type and content
    assert isinstance(statistics, list)
    assert all(isinstance(metric, MetricValue) for metric in statistics)

    # Check that all expected metrics are present
    metric_names = {metric.name for metric in statistics}
    expected_metrics = {
        "average_return",
        "return_cvar",
        "return_value_at_risk",
        "average_state_sampling_time",
        "average_action_time",
        "average_observation_time",
        "average_belief_update_time",
        "average_reward_time",
    }
    assert metric_names == expected_metrics

    # Check that values are reasonable
    for metric in statistics:
        assert isinstance(metric.value, (int, float, np.floating))
        if not np.isnan(
            metric.lower_confidence_bound
        ):  # Some metrics might have NaN confidence bounds
            assert metric.lower_confidence_bound <= metric.value
        if not np.isnan(metric.upper_confidence_bound):
            assert metric.value <= metric.upper_confidence_bound


def test_simulation_different_alphas():
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment, branching_factor=5, depth=3
    )
    initial_belief = get_initial_belief(environment, n_particles=5)

    # Execute with different alphas
    _, statistics1 = simulation(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_episodes=5,
        num_steps=3,
        alpha=1,
    )

    _, statistics2 = simulation(
        environment=environment,
        policy=policy,
        initial_belief=initial_belief,
        num_episodes=5,
        num_steps=3,
        alpha=0.05,
    )

    # Get CVaR values from the statistics
    cvar1 = next(metric.value for metric in statistics1 if metric.name == "return_cvar")
    cvar2 = next(metric.value for metric in statistics2 if metric.name == "return_cvar")

    # Assert CVaR values are different for different alphas
    assert cvar1 != cvar2


def test_compare_planners_parameter_validation(temp_cache_dir):
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment, branching_factor=2, depth=3
    )

    # Test invalid parameters
    with pytest.raises(AssertionError):
        compare_planners(
            environment_policy_pairs=[("not_an_environment", policy)],
            num_episodes=10,
            num_steps=5,
            alpha=0.1,
            n_particles=100,
            cache_dir_path=temp_cache_dir,
        )

    with pytest.raises(AssertionError):
        compare_planners(
            environment_policy_pairs=[],  # Empty policy list
            num_episodes=10,
            num_steps=5,
            alpha=0.1,
            n_particles=100,
            cache_dir_path=temp_cache_dir,
        )


def test_compare_planners_mlflow_integration(temp_cache_dir):
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    policy = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment, branching_factor=2, depth=3
    )
    initial_belief = get_initial_belief(environment, n_particles=100)

    # Create mlruns directory
    mlruns_dir = temp_cache_dir / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    # Set up MLFlow tracking
    experiment_name = "test_compare_planners"
    tracking_uri = f"file:///{mlruns_dir.absolute().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Execute
    compare_planners(
        environment_policy_pairs=[(environment, policy)],
        num_episodes=2,
        num_steps=2,
        alpha=0.1,
        n_particles=5,
        cache_dir_path=temp_cache_dir,
        experiment_name=experiment_name,
    )

    # Assert MLFlow artifacts were created
    assert mlruns_dir.exists()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    assert experiment is not None

    # Check for run artifacts
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0
    for _, run in runs.iterrows():
        assert run.status == "FINISHED"


def test_compare_planners_different_parameters(temp_cache_dir):
    # Setup
    environment1 = TigerPOMDP(discount_factor=0.95)
    environment2 = TigerPOMDP(discount_factor=0.99)
    policy1 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment1, branching_factor=2, depth=3
    )
    policy2 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment2, branching_factor=3, depth=4
    )
    initial_belief = get_initial_belief(environment1, n_particles=100)

    # Execute
    planner_statistics, results_df = compare_planners(
        environment_policy_pairs=[(environment1, policy1), (environment2, policy2)],
        num_episodes=2,
        num_steps=2,
        alpha=0.1,
        n_particles=5,
        cache_dir_path=temp_cache_dir,
    )

    # Assert
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 2  # One row per policy
    assert "policy_branching_factor" in results_df.columns
    assert "policy_depth" in results_df.columns
    assert all(results_df["policy_branching_factor"] == [2, 3])
    assert all(results_df["policy_depth"] == [3, 4])


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


def test_optimize_policy_parameters_with_optuna_different_metrics(temp_cache_dir):
    # Setup
    environment = TigerPOMDP(discount_factor=0.95)
    param_ranges = [
        NumericalHyperParameter(name="branching_factor", low=2, high=3),
        NumericalHyperParameter(name="depth", low=2, high=3),
    ]

    # Execute with different optimization metrics
    best_params1, _, _ = optimize_policy_parameters_with_optuna(
        environment=environment,
        policy_class=StandardSparseSamplingDiscreteActionsPlanner,
        param_ranges=param_ranges,
        num_episodes=5,
        num_steps=5,
        n_particles=10,
        cache_dir_path=temp_cache_dir,
        parameter_to_optimize="average_return",
        n_trials=2,
    )

    best_params2, _, _ = optimize_policy_parameters_with_optuna(
        environment=environment,
        policy_class=StandardSparseSamplingDiscreteActionsPlanner,
        param_ranges=param_ranges,
        num_episodes=5,
        num_steps=5,
        n_particles=10,
        cache_dir_path=temp_cache_dir,
        parameter_to_optimize="return_cvar",
        n_trials=2,
    )

    # Assert different metrics can lead to different optimal parameters
    assert best_params1 != best_params2


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
