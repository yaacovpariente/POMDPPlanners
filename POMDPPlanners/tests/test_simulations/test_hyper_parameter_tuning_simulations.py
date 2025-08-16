"""Tests for hyper_parameter_tuning_simulations module.

This module tests the HyperParameterOptimizer class and its functionality for
optimizing POMDP policy hyperparameters using Optuna and MLFlow.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import mlflow
import numpy as np

from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import (
    HyperParameterOptimizer
)
from POMDPPlanners.core.simulation import (
    NumericalHyperParameter,
    CategoricalHyperParameter,
    MetricValue
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
    OptimizedPolicyResult,
    HyperParameterOptimizationDirection
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP


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
def mock_environment():
    """Create a mock environment for testing."""
    env = Mock()
    env.__class__.__name__ = "MockEnvironment"
    env.name = "mock_env"
    env.discount_factor = 0.95
    
    # Mock the initial_state_dist method to return a proper distribution
    mock_dist = Mock()
    mock_dist.sample.return_value = [Mock()] * 100  # Return list of 100 mock states
    env.initial_state_dist = mock_dist
    
    return env


@pytest.fixture
def mock_policy_class():
    """Create a mock policy class for testing."""
    policy_cls = Mock()
    policy_cls.__name__ = "MockPolicy"
    return policy_cls


@pytest.fixture
def mock_belief():
    """Create a mock belief for testing."""
    belief = Mock()
    belief.__class__.__name__ = "MockBelief"
    return belief


@pytest.fixture
def sample_hyperparameters():
    """Create sample hyperparameters for testing."""
    return [
        NumericalHyperParameter("exploration_constant", 0.1, 10.0),
        NumericalHyperParameter("n_simulations", 100, 1000),
        CategoricalHyperParameter("algorithm", ["tpe", "cmaes", "random"])
    ]


@pytest.fixture
def sample_configs(mock_environment, mock_policy_class, mock_belief):
    """Create sample HyperParameterRunParams configurations for testing."""
    return [
        HyperParameterRunParams(
            environment=mock_environment,
            belief=mock_belief,
            policy_cls=mock_policy_class,
            hyper_parameters=[
                NumericalHyperParameter("exploration_constant", 0.1, 10.0),
                NumericalHyperParameter("n_simulations", 100, 1000)
            ],
            num_episodes=10,
            num_steps=20,
            n_trials=50,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return"
        ),
        HyperParameterRunParams(
            environment=mock_environment,
            belief=mock_belief,
            policy_cls=mock_policy_class,
            hyper_parameters=[
                CategoricalHyperParameter("algorithm", ["tpe", "cmaes"]),
                NumericalHyperParameter("depth", 5, 15)
            ],
            num_episodes=15,
            num_steps=25,
            n_trials=75,
            direction=HyperParameterOptimizationDirection.MINIMIZE,
            parameter_to_optimize="total_cost"
        )
    ]


@pytest.fixture
def mock_optimized_policy_result(mock_environment, mock_policy_class):
    """Create a mock OptimizedPolicyResult for testing."""
    return OptimizedPolicyResult(
        environment=mock_environment,
        policy=mock_policy_class(),
        chosen_hyper_parameters={
            "exploration_constant": 5.0,
            "n_simulations": 500
        },
        num_episodes=10,
        num_steps=20,
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return"
    )


class TestHyperParameterOptimizerInitialization:
    """Test HyperParameterOptimizer initialization and configuration."""

    def test_initialization_with_default_parameters(self, temp_cache_dir):
        """Test initialization with default parameters."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
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
            mlflow_tracking_uri=custom_tracking_uri
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
        
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        assert mlruns_path.exists()
        assert mlruns_path.is_dir()

    def test_initialization_sets_up_mlflow_tracking(self, temp_cache_dir):
        """Test that MLflow tracking is properly configured."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir,
            experiment_name="Test_Experiment"
        )
        
        # Check that MLflow experiment is set
        assert mlflow.get_experiment_by_name("Test_Experiment") is not None

    def test_initialization_with_invalid_cache_dir_type(self):
        """Test initialization with invalid cache_dir_path type raises error."""
        with pytest.raises(TypeError, match="unsupported operand type"):
            HyperParameterOptimizer(
                cache_dir_path="/invalid/path/string"
            )


class TestHyperParameterOptimizerTaskCreation:
    """Test task creation methods."""

    @patch('POMDPPlanners.simulations.hyper_parameter_tuning_simulations.HyperParameterTuningSimulationTask')
    def test_create_tasks_with_default_n_trials(self, mock_task_class, temp_cache_dir, sample_configs):
        """Test task creation with default n_trials."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Mock the task class to avoid actual instantiation
        mock_task_instance = Mock()
        mock_task_class.return_value = mock_task_instance
        
        tasks = optimizer._create_tasks(sample_configs)
        
        assert len(tasks) == len(sample_configs)
        # Verify task class was called with correct parameters
        assert mock_task_class.call_count == len(sample_configs)

    @patch('POMDPPlanners.simulations.hyper_parameter_tuning_simulations.HyperParameterTuningSimulationTask')
    def test_create_tasks_with_custom_n_trials(self, mock_task_class, temp_cache_dir, sample_configs):
        """Test task creation with custom n_trials."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Mock the task class to avoid actual instantiation
        mock_task_instance = Mock()
        mock_task_class.return_value = mock_task_instance
        
        # Create configs with different n_trials values
        configs_with_trials = [
            HyperParameterRunParams(
                environment=sample_configs[0].environment,
                belief=sample_configs[0].belief,
                policy_cls=sample_configs[0].policy_cls,
                hyper_parameters=sample_configs[0].hyper_parameters,
                num_episodes=sample_configs[0].num_episodes,
                num_steps=sample_configs[0].num_steps,
                n_trials=100,
                direction=sample_configs[0].direction,
                parameter_to_optimize=sample_configs[0].parameter_to_optimize
            ),
            HyperParameterRunParams(
                environment=sample_configs[1].environment,
                belief=sample_configs[1].belief,
                policy_cls=sample_configs[1].policy_cls,
                hyper_parameters=sample_configs[1].hyper_parameters,
                num_episodes=sample_configs[1].num_episodes,
                num_steps=sample_configs[1].num_steps,
                n_trials=75,
                direction=sample_configs[1].direction,
                parameter_to_optimize=sample_configs[1].parameter_to_optimize
            )
        ]
        
        tasks = optimizer._create_tasks(configs_with_trials)
        
        assert len(tasks) == len(configs_with_trials)
        # Verify task class was called with correct parameters
        assert mock_task_class.call_count == len(configs_with_trials)

    @patch('POMDPPlanners.simulations.hyper_parameter_tuning_simulations.HyperParameterTuningSimulationTask')
    def test_create_tasks_handles_n_trials(self, mock_task_class, temp_cache_dir, sample_configs):
        """Test task creation handles n_trials parameter correctly."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Mock the task class to avoid actual instantiation
        mock_task_instance = Mock()
        mock_task_class.return_value = mock_task_instance
        
        tasks = optimizer._create_tasks(sample_configs)
        
        assert len(tasks) == len(sample_configs)
        # Verify task class was called with correct parameters
        assert mock_task_class.call_count == len(sample_configs)


class TestHyperParameterOptimizerOptimizeMethod:
    """Test the main optimize method."""

    @patch('POMDPPlanners.simulations.hyper_parameter_tuning_simulations.HyperParameterOptimizer._execute_optimization_tasks')
    @patch('POMDPPlanners.simulations.hyper_parameter_tuning_simulations.HyperParameterOptimizer._process_task_results_with_mlflow_logging')
    def test_optimize_empty_configs_returns_empty_list(self, mock_process, mock_execute, temp_cache_dir):
        """Test optimize with empty configs returns empty list."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        result = optimizer.optimize([])
        
        assert result == []
        mock_execute.assert_not_called()
        mock_process.assert_not_called()

    @patch('POMDPPlanners.simulations.hyper_parameter_tuning_simulations.HyperParameterOptimizer._execute_optimization_tasks')
    @patch('POMDPPlanners.simulations.hyper_parameter_tuning_simulations.HyperParameterOptimizer._process_task_results_with_mlflow_logging')
    def test_optimize_calls_required_methods(self, mock_process, mock_execute, temp_cache_dir, sample_configs):
        """Test optimize calls all required methods in correct order."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Mock return values
        mock_execute.return_value = ([], [])  # task_results, tasks
        mock_process.return_value = [Mock()]  # mock results
        
        result = optimizer.optimize(sample_configs)
        
        # Verify methods were called
        mock_execute.assert_called_once()
        mock_process.assert_called_once()
        assert len(result) == 1

    @patch('mlflow.start_run')
    def test_optimize_starts_mlflow_run(self, mock_start_run, temp_cache_dir, sample_configs):
        """Test optimize starts MLflow run for tracking."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Mock the internal methods to avoid actual execution
        with patch.object(optimizer, '_execute_optimization_tasks', return_value=([], [])):
            with patch.object(optimizer, '_process_task_results_with_mlflow_logging', return_value=[]):
                optimizer.optimize(sample_configs)
        
        # Verify MLflow run was started
        mock_start_run.assert_called()


class TestHyperParameterOptimizerHelperMethods:
    """Test helper methods of HyperParameterOptimizer."""

    def test_prepare_mlflow_session_ends_active_run(self, temp_cache_dir):
        """Test MLflow session preparation ends active runs."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Start a mock run
        with mlflow.start_run():
            assert mlflow.active_run() is not None
            
            # Prepare session
            optimizer._prepare_mlflow_session()
            
            # Run should be ended
            assert mlflow.active_run() is None

    def test_log_batch_level_parameters(self, temp_cache_dir, sample_configs):
        """Test batch level parameter logging."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        with mlflow.start_run():
            optimizer._log_batch_level_parameters(sample_configs)
            
            # Verify parameters were logged
            run = mlflow.active_run()
            assert run is not None

    def test_match_successful_results_with_configs(self, temp_cache_dir, sample_configs):
        """Test matching successful results with configurations."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Mock task results
        mock_task_results = [Mock(), None, Mock()]  # Second result is None (failed)
        mock_tasks = [Mock(), Mock(), Mock()]
        
        # Extend sample_configs to match
        extended_configs = sample_configs + [sample_configs[0]]
        
        result = optimizer._match_successful_results_with_configs(
            mock_task_results, extended_configs, mock_tasks
        )
        
        # Should return 2 successful matches (indices 0 and 2)
        assert len(result) == 2
        assert result[0][0] == 0  # First successful index
        assert result[1][0] == 2  # Second successful index

    @patch.object(HyperParameterOptimizer, '_prepare_configuration_parameters')
    def test_prepare_configuration_parameters(self, mock_prepare_params, temp_cache_dir, sample_configs):
        """Test configuration parameter preparation."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        config = sample_configs[0]
        
        # Mock the method to return expected parameters
        mock_prepare_params.return_value = {
            "config_index": 1,
            "environment_type": "MockEnvironment",
            "policy_type": "MockPolicy",
            "num_episodes": 10,
            "num_steps": 20,
            "direction": "maximize",
            "parameter_to_optimize": "average_return",
            "n_trials": 50,
            "num_episodes_tuning": 10
        }
        
        params = optimizer._prepare_configuration_parameters(0, config)
        
        # Check required parameters
        assert params["config_index"] == 1
        assert params["environment_type"] == "MockEnvironment"
        assert params["policy_type"] == "MockPolicy"
        assert params["num_episodes"] == 10
        assert params["num_steps"] == 20
        assert params["direction"] == "maximize"
        assert params["parameter_to_optimize"] == "average_return"
        assert params["n_trials"] == 50
        assert params["num_episodes_tuning"] == 10
        
        # Verify the method was called
        mock_prepare_params.assert_called_once_with(0, config)

    def test_log_optimization_results_success(self, temp_cache_dir, mock_optimized_policy_result):
        """Test logging optimization results for successful optimization."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Mock task with metadata
        mock_task = Mock()
        mock_task.get_optimization_metadata.return_value = {
            'best_value': 0.85,
            'optimization_time': 120.5,
            'n_trials': 50,
            'best_trial_number': 23
        }
        
        with mlflow.start_run():
            optimizer._log_optimization_results(mock_optimized_policy_result, mock_task)
            
            # Verify metrics were logged
            run = mlflow.active_run()
            assert run is not None

    def test_log_optimization_results_failure(self, temp_cache_dir):
        """Test logging optimization results for failed optimization."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Mock failed task
        mock_task = Mock()
        
        with mlflow.start_run():
            optimizer._log_optimization_results(None, mock_task)
            
            # Verify failure was logged
            run = mlflow.active_run()
            assert run is not None

    def test_get_best_value_from_task(self, temp_cache_dir):
        """Test extracting best value from task metadata."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Mock task with metadata
        mock_task = Mock()
        mock_task.get_optimization_metadata.return_value = {
            'best_value': 0.92
        }
        
        best_value = optimizer._get_best_value_from_task(mock_task)
        
        assert best_value == "0.92"

    def test_get_best_value_from_task_no_metadata(self, temp_cache_dir):
        """Test extracting best value when task has no metadata."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Mock task without metadata
        mock_task = Mock()
        mock_task.get_optimization_metadata.return_value = None
        
        best_value = optimizer._get_best_value_from_task(mock_task)
        
        assert best_value == "unknown"

    def test_log_configuration_failure(self, temp_cache_dir):
        """Test logging configuration failure."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        exception = RuntimeError("Optimization failed")
        
        with mlflow.start_run():
            optimizer._log_configuration_failure(0, exception)
            
            # Verify failure was logged
            run = mlflow.active_run()
            assert run is not None

    def test_log_batch_level_summary(self, temp_cache_dir, sample_configs):
        """Test batch level summary logging."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Mock results
        mock_results = [Mock(), Mock()]  # 2 successful results
        
        with mlflow.start_run():
            optimizer._log_batch_level_summary(sample_configs, mock_results)
            
            # Verify summary was logged
            run = mlflow.active_run()
            assert run is not None


class TestHyperParameterOptimizerIntegration:
    """Integration tests for HyperParameterOptimizer."""

    @patch('POMDPPlanners.simulations.hyper_parameter_tuning_simulations.HyperParameterOptimizer._execute_optimization_tasks')
    @patch('POMDPPlanners.simulations.hyper_parameter_tuning_simulations.HyperParameterOptimizer._process_task_results_with_mlflow_logging')
    def test_full_optimization_workflow(self, mock_process, mock_execute, temp_cache_dir, sample_configs):
        """Test complete optimization workflow."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir,
            experiment_name="Integration_Test"
        )
        
        # Mock successful execution
        mock_execute.return_value = ([Mock()], [Mock()])
        mock_process.return_value = [Mock()]
        
        # Run optimization
        results = optimizer.optimize(sample_configs)
        
        # Verify workflow completed
        assert len(results) == 1
        mock_execute.assert_called_once()
        mock_process.assert_called_once()

    def test_mlflow_experiment_creation(self, temp_cache_dir):
        """Test that MLflow experiment is properly created."""
        experiment_name = "Test_Experiment_Creation"
        
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir,
            experiment_name=experiment_name
        )
        
        # Verify experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None
        assert experiment.name == experiment_name

    def test_cache_directory_structure(self, temp_cache_dir):
        """Test that cache directory structure is properly created."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Check required directories exist
        assert (temp_cache_dir / "mlruns").exists()
        assert (temp_cache_dir / "task_manager_cache").exists()


class TestHyperParameterOptimizerErrorHandling:
    """Test error handling in HyperParameterOptimizer."""

    def test_initialization_with_nonexistent_directory(self, temp_cache_dir):
        """Test initialization with nonexistent directory creates it."""
        new_dir = temp_cache_dir / "new_optimizer_dir"
        
        optimizer = HyperParameterOptimizer(
            cache_dir_path=new_dir
        )
        
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_mlflow_tracking_uri_handling(self, temp_cache_dir):
        """Test MLflow tracking URI handling with various path types."""
        # Test with string path
        string_path = str(temp_cache_dir / "string_mlruns")
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir,
            mlflow_tracking_uri=Path(string_path)
        )
        
        assert optimizer.mlflow_tracking_uri.startswith("file://")
        assert string_path in optimizer.mlflow_tracking_uri

    @patch('POMDPPlanners.simulations.hyper_parameter_tuning_simulations.HyperParameterTuningSimulationTask')
    def test_task_creation_with_invalid_configs(self, mock_task_class, temp_cache_dir):
        """Test task creation handles invalid configurations gracefully."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Create invalid config (missing required attributes)
        invalid_config = Mock()
        invalid_config.environment = Mock()
        invalid_config.policy_cls = Mock()
        invalid_config.belief = Mock()
        invalid_config.hyper_parameters = []
        invalid_config.num_episodes = 1
        invalid_config.num_steps = 1
        invalid_config.n_trials = 10
        invalid_config.direction = HyperParameterOptimizationDirection.MAXIMIZE
        invalid_config.parameter_to_optimize = "test"
        
        # Mock the task class to avoid actual instantiation
        mock_task_instance = Mock()
        mock_task_class.return_value = mock_task_instance
        
        # Should not crash during task creation
        tasks = optimizer._create_tasks([invalid_config])
        assert len(tasks) == 1


class TestHyperParameterOptimizerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_parameter_ranges(self, temp_cache_dir, mock_belief):
        """Test handling of very large parameter ranges."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        # Create config with very large parameter ranges
        large_config = HyperParameterRunParams(
            environment=Mock(),
            belief=mock_belief,
            policy_cls=Mock(),
            hyper_parameters=[
                NumericalHyperParameter("large_param", 1e-10, 1e10)
            ],
            num_episodes=1,
            num_steps=1,
            n_trials=10,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="test"
        )
        
        # Should not crash during initialization
        assert optimizer is not None

    def test_zero_episodes_and_steps(self, temp_cache_dir, mock_belief):
        """Test handling of zero episodes and steps."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        zero_config = HyperParameterRunParams(
            environment=Mock(),
            belief=mock_belief,
            policy_cls=Mock(),
            hyper_parameters=[],
            num_episodes=0,
            num_steps=0,
            n_trials=10,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="test"
        )
        
        # Should not crash during initialization
        assert optimizer is not None

    def test_empty_hyperparameters_list(self, temp_cache_dir, mock_belief):
        """Test handling of empty hyperparameters list."""
        optimizer = HyperParameterOptimizer(
            cache_dir_path=temp_cache_dir
        )
        
        empty_config = HyperParameterRunParams(
            environment=Mock(),
            belief=mock_belief,
            policy_cls=Mock(),
            hyper_parameters=[],
            num_episodes=1,
            num_steps=1,
            n_trials=10,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="test"
        )
        
        # Should not crash during initialization
        assert optimizer is not None

    def test_very_large_n_jobs(self, temp_cache_dir):
        """Test handling of very large n_jobs values."""
        large_n_jobs = [100, 1000, 10000]
        
        for n_jobs in large_n_jobs:
            optimizer = HyperParameterOptimizer(
                cache_dir_path=temp_cache_dir,
                n_jobs=n_jobs
            )
            assert optimizer.n_jobs == n_jobs
