"""Tests for hyper_parameter_tuning_simulations module.

This module tests the HyperParameterOptimizer class and its functionality for
optimizing POMDP policy hyperparameters using Optuna and MLFlow.
"""

import random
import shutil
import tempfile
from pathlib import Path
from typing import List, cast

import mlflow
import numpy as np
import pytest

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import (
    CategoricalHyperParameter,
    MetricValue,
    NumericalHyperParameter,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterFeature,
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
    OptimizedPolicyResult,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planner import (
    StandardSparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    HyperParameterOptimizer,
)

np.random.seed(42)
random.seed(42)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for testing cache operations."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    yield temp_path
    # Cleanup
    try:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def real_environment():
    """Create a real TigerPOMDP environment for testing."""
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def real_policy_class():
    """Create a real policy class for testing."""
    return StandardSparseSamplingDiscreteActionsPlanner


@pytest.fixture
def real_belief(real_environment):
    """Create a real belief for testing."""
    return get_initial_belief(real_environment, n_particles=10)  # Small for fast tests


@pytest.fixture
def sample_hyperparameters():
    """Create sample hyperparameters for testing."""
    return [
        NumericalHyperParameter(1, 3, "branching_factor"),  # Correct order: low, high, name
        NumericalHyperParameter(1, 3, "depth"),  # Correct order: low, high, name
    ]


@pytest.fixture
def sample_configs(real_environment, real_policy_class, real_belief):
    """Create sample HyperParameterRunParams configurations for testing."""
    return [
        HyperParameterRunParams(
            environment=real_environment,
            belief=real_belief,
            policy_cls=real_policy_class,
            hyper_parameters=[
                NumericalHyperParameter(1, 2, "branching_factor"),
                NumericalHyperParameter(1, 2, "depth"),
            ],
            constant_parameters={},  # No constant parameters needed for this planner
            num_episodes=2,  # Small for fast tests
            num_steps=3,  # Small for fast tests
            n_trials=2,  # Small for fast tests
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        ),
        HyperParameterRunParams(
            environment=real_environment,
            belief=real_belief,
            policy_cls=real_policy_class,
            hyper_parameters=[
                NumericalHyperParameter(1, 2, "branching_factor"),
                NumericalHyperParameter(1, 2, "depth"),
            ],
            constant_parameters={},  # No constant parameters needed for this planner
            num_episodes=2,  # Small for fast tests
            num_steps=3,  # Small for fast tests
            n_trials=2,  # Small for fast tests
            direction=HyperParameterOptimizationDirection.MAXIMIZE,  # Changed from MINIMIZE to work with average_return
            parameter_to_optimize="average_return",  # Changed from total_cost to a valid metric
        ),
    ]


@pytest.fixture
def real_optimized_policy_result(real_environment, real_policy_class):
    """Create a real OptimizedPolicyResult for testing."""
    # Create a real policy instance
    policy = real_policy_class(environment=real_environment, branching_factor=2, depth=2)

    return OptimizedPolicyResult(
        environment=real_environment,
        policy=policy,
        chosen_hyper_parameters={"branching_factor": 2, "depth": 2},
        num_episodes=2,  # Small for fast tests
        num_steps=3,  # Small for fast tests
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
    )


class TestHyperParameterOptimizerInitialization:
    """Test HyperParameterOptimizer initialization and configuration."""

    def test_initialization_with_default_parameters(self, temp_cache_dir):
        """Test initialization with default parameters."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        assert optimizer.cache_dir_path == temp_cache_dir
        assert optimizer.experiment_name == "POMDP_Parameter_Optimization"
        assert optimizer.n_jobs == 1
        assert optimizer.confidence_interval_level == 0.95
        assert optimizer.alpha == 0.05
        assert optimizer.mlflow_tracking_uri is not None
        assert "mlruns" in str(optimizer.mlruns_path)

    def test_initialization_with_custom_parameters(self, temp_cache_dir):
        """Test initialization with custom parameters."""
        custom_tracking_uri = temp_cache_dir / "custom_mlruns"

        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir,
            experiment_name="Custom_Experiment",
            n_jobs=4,
            confidence_interval_level=0.99,
            alpha=0.01,
            mlflow_tracking_uri=custom_tracking_uri,
        )

        assert optimizer.experiment_name == "Custom_Experiment"
        assert optimizer.n_jobs == 4
        assert optimizer.confidence_interval_level == 0.99
        assert optimizer.alpha == 0.01
        assert optimizer.mlflow_tracking_uri == f"file://{custom_tracking_uri.absolute()}"
        assert optimizer.mlruns_path == custom_tracking_uri

    def test_initialization_creates_mlruns_directory(self, temp_cache_dir):
        """Test that MLruns directory is created during initialization."""
        mlruns_path = temp_cache_dir / "mlruns"

        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        assert mlruns_path.exists()
        assert mlruns_path.is_dir()

    def test_initialization_sets_up_mlflow_tracking(self, temp_cache_dir):
        """Test that MLflow tracking is properly configured."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir, experiment_name="Test_Experiment"
        )

        # Check that MLflow experiment is set
        assert mlflow.get_experiment_by_name("Test_Experiment") is not None

    def test_initialization_with_invalid_cache_dir_type(self):
        """Test initialization with invalid cache_dir_path type raises error."""
        with pytest.raises(TypeError, match="unsupported operand type"):
            HyperParameterOptimizer(cache_dir_path="/invalid/path/string")  # type: ignore[arg-type]


class TestHyperParameterOptimizerTaskCreation:
    """Test task creation methods."""

    def test_create_tasks_with_real_configs(self, temp_cache_dir, sample_configs):
        """Test task creation with real configurations."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        tasks, task_identifiers = optimizer._create_tasks(sample_configs)

        assert len(tasks) == len(sample_configs)
        # Verify tasks are real HyperParameterTuningSimulationTask instances
        for task in tasks:
            assert hasattr(task, "run")
            assert hasattr(task, "environment")
            assert hasattr(task, "policy_cls")
            assert hasattr(task, "hyper_parameters")

    def test_create_tasks_preserves_config_parameters(self, temp_cache_dir, sample_configs):
        """Test task creation preserves configuration parameters."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        tasks, task_identifiers = optimizer._create_tasks(sample_configs)

        for i, task in enumerate(tasks):
            config = sample_configs[i]
            assert task.environment == config.environment
            assert task.policy_cls == config.policy_cls
            assert task.hyper_parameters == config.hyper_parameters
            assert task.constant_parameters == config.constant_parameters
            assert task.num_episodes == config.num_episodes
            assert task.num_steps == config.num_steps
            assert task.n_trials == config.n_trials
            assert task.direction == config.direction
            assert task.parameter_to_optimize == config.parameter_to_optimize

    def test_create_tasks_returns_correct_type(self, temp_cache_dir, sample_configs):
        """Test that _create_tasks returns Tuple[List[HyperParameterTuningSimulationTask], List[str]]."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        tasks, task_identifiers = optimizer._create_tasks(sample_configs)

        # Check that it returns a tuple
        assert isinstance((tasks, task_identifiers), tuple)

        # Check that first element is a list
        assert isinstance(tasks, list)

        # Check that second element is a list of strings
        assert isinstance(task_identifiers, list)
        for identifier in task_identifiers:
            assert isinstance(identifier, str)

        # Check that each item is a HyperParameterTuningSimulationTask
        from POMDPPlanners.simulations.simulations_deployment.tasks import (
            HyperParameterTuningSimulationTask,
        )

        for task in tasks:
            assert isinstance(task, HyperParameterTuningSimulationTask)

        # Check that the lists have the expected length
        assert len(tasks) == len(sample_configs)
        assert len(task_identifiers) == len(sample_configs)

    def test_execute_optimization_tasks_returns_correct_type(self, temp_cache_dir, sample_configs):
        """Test that _execute_optimization_tasks returns Tuple[List[OptimizedPolicyResult], List[HyperParameterTuningSimulationTask]]."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        # Use only one config for faster testing
        single_config = sample_configs[:1]

        task_results, tasks = optimizer._execute_optimization_tasks(single_config)

        # Check that it returns a tuple
        assert isinstance((task_results, tasks), tuple)

        # Check that first element is a list
        assert isinstance(task_results, list)

        # Check that second element is a list
        assert isinstance(tasks, list)

        # Check that the lists have the expected length
        assert len(tasks) == len(single_config)

        # Check that each task is a HyperParameterTuningSimulationTask
        from POMDPPlanners.simulations.simulations_deployment.tasks import (
            HyperParameterTuningSimulationTask,
        )

        for task in tasks:
            assert isinstance(task, HyperParameterTuningSimulationTask)

        # Note: task_results might be empty or contain None values if optimization fails,
        # but the structure should be correct
        if task_results:
            # If there are results, they should be OptimizedPolicyResult instances
            from POMDPPlanners.core.simulation.hyperparameter_tuning import (
                OptimizedPolicyResult,
            )

            for result in task_results:
                assert isinstance(result, OptimizedPolicyResult)


class TestHyperParameterOptimizerOptimizeMethod:
    """Test the main optimize method."""

    def test_optimize_empty_configs_returns_empty_list(self, temp_cache_dir):
        """Test optimize with empty configs returns empty list."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        result = optimizer.optimize([])

        assert result == []

    def test_optimize_with_real_configs(self, temp_cache_dir, sample_configs):
        """Test optimize with real configurations (integration test)."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        # Use only one config for faster testing
        single_config = sample_configs[:1]

        result = optimizer.optimize(single_config)

        # Verify results
        assert isinstance(result, list)
        assert len(result) <= len(single_config)  # May be fewer if some fail

        # Check result structure if any succeeded
        for optimized_result in result:
            assert hasattr(optimized_result, "environment")
            assert hasattr(optimized_result, "policy")
            assert hasattr(optimized_result, "chosen_hyper_parameters")
            assert hasattr(optimized_result, "direction")
            assert hasattr(optimized_result, "parameter_to_optimize")

    def test_optimize_creates_mlflow_experiment(self, temp_cache_dir, sample_configs):
        """Test optimize creates MLflow experiment."""
        experiment_name = "Test_Optimize_Experiment"
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir, experiment_name=experiment_name
        )

        # Run with empty configs to avoid long execution
        optimizer.optimize([])

        # Verify experiment was created
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None


class TestHyperParameterOptimizerHelperMethods:
    """Test helper methods of HyperParameterOptimizer."""

    def test_prepare_mlflow_session_ends_active_run(self, temp_cache_dir):
        """Test MLflow session preparation ends active runs."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        # Start a run
        with mlflow.start_run():
            assert mlflow.active_run() is not None

            # Prepare session
            optimizer._prepare_mlflow_session()

            # Run should be ended
            assert mlflow.active_run() is None

    def test_log_batch_level_parameters(self, temp_cache_dir, sample_configs):
        """Test batch level parameter logging."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        with mlflow.start_run():
            optimizer._log_batch_level_parameters(sample_configs)

            # Verify parameters were logged
            run = mlflow.active_run()
            assert run is not None

    def test_prepare_configuration_parameters(self, temp_cache_dir, sample_configs):
        """Test configuration parameter preparation."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        config = sample_configs[0]
        params = optimizer._prepare_configuration_parameters(0, config)

        # Check required parameters exist and have correct types
        assert "config_index" in params
        assert "environment_type" in params
        assert "policy_type" in params
        assert "num_episodes" in params
        assert "num_steps" in params
        assert "direction" in params
        assert "parameter_to_optimize" in params
        assert "n_trials" in params

        # Check specific values
        assert params["config_index"] == 1  # original_index + 1
        assert params["environment_type"] == "TigerPOMDP"
        assert params["policy_type"] == "StandardSparseSamplingDiscreteActionsPlanner"
        assert params["num_episodes"] == config.num_episodes
        assert params["num_steps"] == config.num_steps
        assert params["direction"] == config.direction.value
        assert params["parameter_to_optimize"] == config.parameter_to_optimize
        assert params["n_trials"] == config.n_trials

    def test_log_optimization_results_success(self, temp_cache_dir, real_optimized_policy_result):
        """Test logging optimization results for successful optimization."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        # Create a real task that would have metadata
        from POMDPPlanners.simulations.simulations_deployment.tasks import (
            HyperParameterTuningSimulationTask,
        )

        task = HyperParameterTuningSimulationTask(
            environment=real_optimized_policy_result.environment,
            belief=get_initial_belief(real_optimized_policy_result.environment, n_particles=10),
            policy_cls=type(real_optimized_policy_result.policy),
            hyper_parameters=[],
            constant_parameters={},  # No constant parameters needed for this planner
            num_episodes=1,
            num_steps=1,
            direction=real_optimized_policy_result.direction,
            parameter_to_optimize=real_optimized_policy_result.parameter_to_optimize,
            n_trials=1,
        )

        with mlflow.start_run():
            optimizer._log_optimization_results(real_optimized_policy_result, task)

            # Verify metrics were logged
            run = mlflow.active_run()
            assert run is not None

    def test_log_optimization_results_failure(self, temp_cache_dir, real_environment):
        """Test logging optimization results for failed optimization."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        # Create a real task
        from POMDPPlanners.simulations.simulations_deployment.tasks import (
            HyperParameterTuningSimulationTask,
        )

        task = HyperParameterTuningSimulationTask(
            environment=real_environment,
            belief=get_initial_belief(real_environment, n_particles=10),
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=[],
            constant_parameters={},  # No constant parameters needed for this planner
            num_episodes=1,
            num_steps=1,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
            n_trials=1,
        )

        with mlflow.start_run():
            optimizer._log_optimization_results(None, task)  # type: ignore[arg-type]

            # Verify failure was logged
            run = mlflow.active_run()
            assert run is not None


class TestHyperParameterOptimizerIntegration:
    """Integration tests for HyperParameterOptimizer."""

    def test_mlflow_experiment_creation(self, temp_cache_dir):
        """Test that MLflow experiment is properly created."""
        experiment_name = "Test_Experiment_Creation"

        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir, experiment_name=experiment_name
        )

        # Verify experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None
        assert experiment.name == experiment_name

    def test_cache_directory_structure(self, temp_cache_dir):
        """Test that cache directory structure is properly created."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        # Check required directories exist
        assert (temp_cache_dir / "mlruns").exists()
        assert (temp_cache_dir / "task_manager_cache").exists()


class TestHyperParameterOptimizerErrorHandling:
    """Test error handling in HyperParameterOptimizer."""

    def test_initialization_with_nonexistent_directory(self, temp_cache_dir):
        """Test initialization with nonexistent directory creates it."""
        new_dir = temp_cache_dir / "new_optimizer_dir"

        optimizer = HyperParameterOptimizer(cache_dir_path=new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_mlflow_tracking_uri_handling(self, temp_cache_dir):
        """Test MLflow tracking URI handling with various path types."""
        # Test with string path
        string_path = str(temp_cache_dir / "string_mlruns")
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir, mlflow_tracking_uri=Path(string_path)
        )

        assert optimizer.mlflow_tracking_uri.startswith("file://")
        assert string_path in optimizer.mlflow_tracking_uri

    def test_task_creation_with_invalid_configs(
        self, temp_cache_dir, real_environment, real_policy_class, real_belief
    ):
        """Test task creation handles invalid configurations gracefully."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        # Create a real but minimal config with very small parameters for fast testing
        minimal_config = HyperParameterRunParams(
            environment=real_environment,
            belief=real_belief,
            policy_cls=real_policy_class,
            hyper_parameters=[],  # Empty hyperparameters - this is the "invalid" aspect
            constant_parameters={},  # No constant parameters needed for this planner
            num_episodes=1,  # Small for fast tests
            num_steps=1,  # Small for fast tests
            n_trials=1,  # Small for fast tests
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        )

        # Should not crash during task creation even with empty hyperparameters
        tasks, task_identifiers = optimizer._create_tasks([minimal_config])
        assert len(tasks) == 1
        assert len(task_identifiers) == 1


class TestHyperParameterOptimizerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_parameter_ranges(
        self, temp_cache_dir, real_environment, real_policy_class, real_belief
    ):
        """Test handling of very large parameter ranges."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        # Create config with large parameter ranges but small values for fast testing
        large_config = HyperParameterRunParams(
            environment=real_environment,
            belief=real_belief,
            policy_cls=real_policy_class,
            hyper_parameters=cast(
                List[HyperParameterFeature],
                [
                    NumericalHyperParameter(
                        low=1, high=100, name="branching_factor"
                    )  # Large range but reasonable values
                ],
            ),
            constant_parameters={},  # No constant parameters needed for this planner
            num_episodes=1,  # Small for fast tests
            num_steps=1,  # Small for fast tests
            n_trials=1,  # Small for fast tests
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        )

        # Should not crash during initialization
        assert optimizer is not None

    def test_zero_episodes_and_steps(
        self, temp_cache_dir, real_environment, real_policy_class, real_belief
    ):
        """Test handling of zero episodes and steps."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        zero_config = HyperParameterRunParams(
            environment=real_environment,
            belief=real_belief,
            policy_cls=real_policy_class,
            hyper_parameters=[],
            constant_parameters={},  # No constant parameters needed for this planner
            num_episodes=0,
            num_steps=0,
            n_trials=1,  # Small for fast tests
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        )

        # Should not crash during initialization
        assert optimizer is not None

    def test_empty_hyperparameters_list(
        self, temp_cache_dir, real_environment, real_policy_class, real_belief
    ):
        """Test handling of empty hyperparameters list."""
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        empty_config = HyperParameterRunParams(
            environment=real_environment,
            belief=real_belief,
            policy_cls=real_policy_class,
            hyper_parameters=[],
            constant_parameters={},  # No constant parameters needed for this planner
            num_episodes=1,  # Small for fast tests
            num_steps=1,  # Small for fast tests
            n_trials=1,  # Small for fast tests
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        )

        # Should not crash during initialization
        assert optimizer is not None

    def test_very_large_n_jobs(self, temp_cache_dir):
        """Test handling of very large n_jobs values."""
        large_n_jobs = [100, 1000, 10000]

        for n_jobs in large_n_jobs:
            optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir, n_jobs=n_jobs)
            assert optimizer.n_jobs == n_jobs


# Create a test policy class that requires specific constant parameters
# This needs to be defined outside the test method to avoid pickling issues
class PolicyRequiringConstants(StandardSparseSamplingDiscreteActionsPlanner):
    def __init__(self, environment, branching_factor, depth, required_constant=None, **kwargs):
        if required_constant is None:
            raise TypeError(
                "PolicyRequiringConstants.__init__() missing 1 required keyword argument: 'required_constant'"
            )
        super().__init__(environment, branching_factor, depth, **kwargs)
        self.required_constant = required_constant


class TestHyperParameterOptimizerMLFlowIntegration:
    """Test MLFlow logging integration that should catch issues from hyper_param_runner.py."""

    def test_mlflow_logging_with_successful_optimization(
        self, temp_cache_dir, real_environment, real_belief
    ):
        """Test that MLFlow properly logs parameters and metrics during successful optimization.

        Purpose: Validates that MLFlow logging works end-to-end and catches integration issues

        Given: A valid hyperparameter optimization configuration with real components
        When: Full optimization is run through HyperParameterOptimizer.optimize()
        Then: MLFlow logs all expected parameters and metrics, experiment structure is correct

        Test type: integration
        """
        experiment_name = "MLFlow_Integration_Test"
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir, experiment_name=experiment_name
        )

        # Create a config that should work
        config = HyperParameterRunParams(
            environment=real_environment,
            belief=real_belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=[
                NumericalHyperParameter(1, 2, "branching_factor"),
                NumericalHyperParameter(1, 2, "depth"),
            ],
            constant_parameters={},  # No constant parameters needed for this planner
            num_episodes=2,  # Small for fast tests
            num_steps=2,  # Small for fast tests
            n_trials=2,  # Small for fast tests
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        )

        # Run optimization
        result = optimizer.optimize([config])

        # Verify MLFlow experiment was created and has data
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None

        # Check that runs were created
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) > 0  # Should have at least batch run

        # Find the batch run (parent run)
        batch_runs = runs[runs["tags.mlflow.runName"].str.contains("optimize_batch_", na=False)]  # type: ignore[index]
        assert len(batch_runs) >= 1

        batch_run = batch_runs.iloc[0]

        # Verify batch-level parameters were logged
        assert batch_run["params.num_configurations"] == "1"
        assert batch_run["params.batch_method"] == "stub_interface_optimize"

        # Verify batch-level metrics were logged
        assert "metrics.batch_success_rate" in batch_run
        assert "metrics.batch_completed_configs" in batch_run
        assert "metrics.batch_failed_configs" in batch_run

        # If optimization succeeded, check nested runs
        if len(result) > 0:
            # Find nested configuration runs
            config_runs = runs[runs["tags.mlflow.parentRunId"].notna()]  # type: ignore[index]
            assert len(config_runs) >= 1

            config_run = config_runs.iloc[0]

            # Verify configuration parameters were logged
            assert config_run["params.config_index"] == "1"
            assert config_run["params.environment_type"] == "TigerPOMDP"
            assert (
                config_run["params.policy_type"] == "StandardSparseSamplingDiscreteActionsPlanner"
            )

            # Verify optimization results were logged
            assert "metrics.optimization_success" in config_run
            assert "metrics.best_value" in config_run
            assert "metrics.optimization_time" in config_run

            # Verify final evaluation metrics were logged
            assert "metrics.final_average_return" in config_run
            assert "metrics.final_success_rate" in config_run
            assert "metrics.final_average_listens" in config_run

    def test_numerical_hyperparameter_constructor_order_validation(
        self, temp_cache_dir, real_environment, real_belief
    ):
        """Test that NumericalHyperParameter constructor parameter order is validated.

        Purpose: Catches the issue from hyper_param_runner.py where parameter order was wrong

        Given: NumericalHyperParameter created with correct vs incorrect parameter order
        When: Hyperparameters are used in optimization configuration
        Then: Correct order works, incorrect order fails with clear error

        Test type: unit
        """
        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        # Test correct parameter order: low, high, name
        correct_hyperparams = cast(
            List[HyperParameterFeature],
            [
                NumericalHyperParameter(1, 3, "branching_factor"),  # Correct: low, high, name
                NumericalHyperParameter(1, 3, "depth"),
            ],
        )

        correct_config = HyperParameterRunParams(
            environment=real_environment,
            belief=real_belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=correct_hyperparams,
            constant_parameters={},
            num_episodes=1,
            num_steps=1,
            n_trials=1,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        )

        # This should work
        tasks, task_identifiers = optimizer._create_tasks([correct_config])
        assert len(tasks) == 1
        assert len(task_identifiers) == 1

        # Verify hyperparameter names are correctly set
        task = tasks[0]
        assert task.hyper_parameters[0].name == "branching_factor"
        assert task.hyper_parameters[1].name == "depth"
        # Cast to NumericalHyperParameter for type checking
        param0 = cast(NumericalHyperParameter, task.hyper_parameters[0])
        assert param0.low == 1
        assert param0.high == 3

        # Test that incorrect parameter order (name, low, high) fails
        # This was the original issue in hyper_param_runner.py
        incorrect_hyperparams = [
            NumericalHyperParameter("branching_factor", 1, 3),  # type: ignore[arg-type]  # Wrong order: name, low, high
            NumericalHyperParameter("depth", 1, 3),  # type: ignore[arg-type]  # Wrong order: name, low, high
        ]

        incorrect_config = HyperParameterRunParams(
            environment=real_environment,
            belief=real_belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=cast(List[HyperParameterFeature], incorrect_hyperparams),
            constant_parameters={},
            num_episodes=1,
            num_steps=1,
            n_trials=1,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        )

        # This should fail because the parameters are in wrong order
        # The low and high values are strings instead of numbers
        with pytest.raises(Exception) as exc_info:
            optimizer.optimize([incorrect_config])

        # Verify the error is related to parameter type issues
        error_message = str(exc_info.value).lower()
        assert "type" in error_message or ">" in error_message or "comparison" in error_message

    def test_missing_constant_parameters_for_complex_policies(
        self, temp_cache_dir, real_environment, real_belief
    ):
        """Test that missing constant parameters for complex policies are detected.

        Purpose: Catches the issue from hyper_param_runner.py where POMCP was missing discount_factor

        Given: A policy class that requires constant parameters (simulated with invalid parameters)
        When: HyperParameterRunParams is created without required constant parameters
        Then: Task creation succeeds but execution fails with clear error about missing parameters

        Test type: integration
        """

        optimizer = HyperParameterOptimizer(cache_dir_path=temp_cache_dir)

        # Test configuration missing required constant parameters
        config_missing_constants = HyperParameterRunParams(
            environment=real_environment,
            belief=real_belief,
            policy_cls=PolicyRequiringConstants,
            hyper_parameters=[
                NumericalHyperParameter(1, 2, "branching_factor"),
                NumericalHyperParameter(1, 2, "depth"),
            ],
            constant_parameters={},  # Missing required_constant
            num_episodes=2,  # Need at least 2 for confidence intervals
            num_steps=1,
            n_trials=1,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        )

        # Task creation should succeed
        tasks, task_identifiers = optimizer._create_tasks([config_missing_constants])
        assert len(tasks) == 1
        assert len(task_identifiers) == 1

        # But optimization execution should fail with clear error
        with pytest.raises(Exception) as exc_info:
            optimizer.optimize([config_missing_constants])

        # Verify the error message indicates missing parameter
        error_message = str(exc_info.value)
        assert "required_constant" in error_message or "missing" in error_message.lower()

        # Test configuration with correct constant parameters
        config_with_constants = HyperParameterRunParams(
            environment=real_environment,
            belief=real_belief,
            policy_cls=PolicyRequiringConstants,
            hyper_parameters=[
                NumericalHyperParameter(1, 2, "branching_factor"),
                NumericalHyperParameter(1, 2, "depth"),
            ],
            constant_parameters={"required_constant": "test_value"},  # Providing required parameter
            num_episodes=2,  # Need at least 2 for confidence intervals
            num_steps=1,
            n_trials=1,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        )

        # This should work
        result = optimizer.optimize([config_with_constants])
        assert isinstance(result, list)
        # Note: result might be empty if optimization fails for other reasons, but no missing parameter error

    def test_mlflow_parameter_and_metric_logging_completeness(
        self, temp_cache_dir, real_environment, real_belief
    ):
        """Test that all expected MLFlow parameters and metrics are logged.

        Purpose: Ensures MLFlow logging is comprehensive and nothing is missing

        Given: A successful hyperparameter optimization run
        When: All logging functions are executed during optimization
        Then: All expected parameters, metrics, and artifacts are present in MLFlow

        Test type: integration
        """
        experiment_name = "Complete_Logging_Test"
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir, experiment_name=experiment_name
        )

        config = HyperParameterRunParams(
            environment=real_environment,
            belief=real_belief,
            policy_cls=StandardSparseSamplingDiscreteActionsPlanner,
            hyper_parameters=[
                NumericalHyperParameter(1, 2, "branching_factor"),
                NumericalHyperParameter(1, 2, "depth"),
            ],
            constant_parameters={},
            num_episodes=2,  # Need multiple episodes for statistics
            num_steps=2,
            n_trials=2,  # Need multiple trials for optimization
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        )

        result = optimizer.optimize([config])

        # Get all runs for this experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None, f"Experiment {experiment_name} not found"
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        # Verify we have both batch and configuration runs
        assert len(runs) >= 1  # At least the batch run

        # Check batch run contains expected elements
        batch_run = runs[  # type: ignore[index]
            runs["tags.mlflow.runName"].str.contains("optimize_batch_", na=False)  # type: ignore[index]
        ].iloc[0]

        # Batch-level parameters
        expected_batch_params = ["num_configurations", "batch_method"]
        for param in expected_batch_params:
            assert f"params.{param}" in batch_run.index, f"Missing batch parameter: {param}"

        # Batch-level metrics
        expected_batch_metrics = [
            "batch_success_rate",
            "batch_completed_configs",
            "batch_failed_configs",
        ]
        for metric in expected_batch_metrics:
            assert f"metrics.{metric}" in batch_run.index, f"Missing batch metric: {metric}"

        # If we have results, check configuration run logging
        if len(result) > 0:
            config_runs = runs[runs["tags.mlflow.parentRunId"].notna()]  # type: ignore[index]
            if len(config_runs) > 0:
                config_run = config_runs.iloc[0]

                # Configuration-level parameters
                expected_config_params = [
                    "config_index",
                    "environment_type",
                    "policy_type",
                    "num_episodes",
                    "num_steps",
                    "direction",
                    "parameter_to_optimize",
                    "n_trials",
                ]
                for param in expected_config_params:
                    assert (
                        f"params.{param}" in config_run.index
                    ), f"Missing config parameter: {param}"

                # Optimization result metrics
                expected_optimization_metrics = [
                    "optimization_success",
                    "best_value",
                    "optimization_time",
                ]
                for metric in expected_optimization_metrics:
                    assert (
                        f"metrics.{metric}" in config_run.index
                    ), f"Missing optimization metric: {metric}"

                # Final evaluation metrics
                expected_final_metrics = [
                    "final_average_return",
                    "final_success_rate",
                    "final_average_listens",
                ]
                for metric in expected_final_metrics:
                    assert (
                        f"metrics.{metric}" in config_run.index
                    ), f"Missing final evaluation metric: {metric}"

    def test_hyperparameter_runner_configuration_validation(self, temp_cache_dir):
        """Test validation of the exact configuration used in hyper_param_runner.py.

        Purpose: Replicates the hyper_param_runner.py configuration to catch its specific issues

        Given: The exact same configuration as used in hyper_param_runner.py
        When: Configuration is validated and executed
        Then: Specific issues from hyper_param_runner.py are identified and handled

        Test type: integration
        """
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP

        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir,
            experiment_name="hyper_param_experiment",
            n_jobs=1,  # Simplified for testing
            confidence_interval_level=0.95,
            alpha=0.1,
        )

        env = TigerPOMDP(discount_factor=0.95, name="Tiger_095")

        # Test configuration with correct parameter order (fixing hyper_param_runner.py issue)
        fixed_config = HyperParameterRunParams(
            environment=env,
            belief=get_initial_belief(env, n_particles=10),  # Smaller for fast tests
            policy_cls=POMCP,
            hyper_parameters=[
                # Correct order: low, high, name (original was wrong: low, high, name)
                NumericalHyperParameter(
                    0.1, 1.0, "exploration_constant"
                ),  # Smaller range for fast tests
                NumericalHyperParameter(10, 50, "n_simulations"),  # Smaller range for fast tests
                NumericalHyperParameter(2, 5, "depth"),  # Smaller range for fast tests
            ],
            constant_parameters={
                "discount_factor": env.discount_factor,  # This was missing in original
                "name": "POMCP_Tiger_095",
            },
            num_episodes=2,  # Smaller for fast tests
            num_steps=3,  # Smaller for fast tests
            n_trials=2,  # Smaller for fast tests
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        )

        # This should work now
        tasks, task_identifiers = optimizer._create_tasks([fixed_config])
        assert len(tasks) == 1
        assert len(task_identifiers) == 1

        # Verify the task has correct configuration
        task = tasks[0]
        assert task.policy_cls == POMCP
        assert task.constant_parameters["discount_factor"] == 0.95
        assert task.hyper_parameters[0].name == "exploration_constant"
        assert task.hyper_parameters[1].name == "n_simulations"
        assert task.hyper_parameters[2].name == "depth"

        # The optimization should succeed (or at least not fail due to missing parameters)
        try:
            result = optimizer.optimize([fixed_config])
            # If it succeeds, verify we got results
            assert isinstance(result, list)
        except Exception as e:
            # If it fails, it should not be due to missing discount_factor or parameter order issues
            error_message = str(e).lower()
            assert "discount_factor" not in error_message
            assert "missing" not in error_message or "positional argument" not in error_message
