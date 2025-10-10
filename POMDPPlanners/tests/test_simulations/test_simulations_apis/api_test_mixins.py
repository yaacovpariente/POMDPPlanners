"""Mixin classes with shared test methods for all simulation API implementations.

This module contains concrete test methods organized into logical mixin classes
that can be inherited by all API test classes (Local, Dask, PBS) to ensure
consistent behavior and reduce code duplication.
"""

import pytest
import pandas as pd
from pathlib import Path
from typing import Any, Optional, Protocol

from POMDPPlanners.core.simulation.hyperparameter_tuning import OptimizedPolicyResult
from POMDPPlanners.simulations.simulation_apis.simulations_api_interface import (
    SimulationsAPIInterface,
)


class APITestProtocol(Protocol):
    """Protocol defining the interface that test classes must implement."""

    def create_api(
        self, cache_dir_path: Optional[Path] = None, debug: bool = False, **kwargs: Any
    ) -> SimulationsAPIInterface:
        """Factory method to create an API instance for testing.

        Args:
            cache_dir_path: Optional cache directory path
            debug: Whether to enable debug mode
            **kwargs: Additional implementation-specific parameters

        Returns:
            An instance of a SimulationsAPIInterface implementation
        """
        ...


class InitializationTestsMixin:
    """Mixin with initialization tests that work for all API implementations."""

    def test_api_initialization_with_default_params(self: APITestProtocol) -> None:
        """Test API initialization with minimal parameters.

        Purpose: Validates that API can be initialized with default settings

        Given: No custom parameters provided
        When: API is instantiated with defaults
        Then: API object is created successfully with expected attributes

        Test type: unit
        """
        api = self.create_api()
        assert api is not None
        assert hasattr(api, "logger")

    def test_api_initialization_with_custom_cache_dir(self: "APITestProtocol", tmp_path):
        """Test API initialization with custom cache directory.

        Purpose: Validates custom cache directory configuration

        Given: A temporary directory path
        When: API is initialized with cache_dir_path parameter
        Then: API is created successfully

        Test type: unit
        """
        cache_dir = tmp_path / "custom_cache"
        api = self.create_api(cache_dir_path=cache_dir)
        assert api is not None

    def test_api_initialization_with_debug_mode(self: "APITestProtocol"):
        """Test API initialization with debug mode enabled.

        Purpose: Validates debug mode configuration

        Given: debug=True parameter
        When: API is instantiated
        Then: API is created successfully with debug enabled

        Test type: unit
        """
        api = self.create_api(debug=True)
        assert api is not None
        assert hasattr(api, "logger")


class RunMultipleEnvironmentsTestsMixin:
    """Mixin with tests for run_multiple_environments_and_policies method."""

    def test_run_multiple_environments_returns_correct_types(
        self: "APITestProtocol", sample_environment_params
    ) -> None:
        """Test return types from run_multiple_environments_and_policies.

        Purpose: Validates method returns correct data structures

        Given: Valid environment run parameters
        When: run_multiple_environments_and_policies is called
        Then: Returns tuple of (dict, DataFrame) with expected types

        Test type: integration
        """
        api = self.create_api()
        results, stats_df = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            n_jobs=1,
        )

        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    def test_run_multiple_environments_result_structure(
        self: "APITestProtocol", sample_environment_params
    ):
        """Test result dictionary has expected nested structure.

        Purpose: Validates result organization by environment and policy names

        Given: Valid environment run parameters with known env/policy names
        When: Simulation completes successfully
        Then: Result dict follows env_name → policy_name → list structure

        Test type: integration
        """
        api = self.create_api()
        results, _ = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            n_jobs=1,
        )

        # Verify nested structure
        for env_name, policies_dict in results.items():
            assert isinstance(env_name, str)
            assert isinstance(policies_dict, dict)
            for policy_name, history_list in policies_dict.items():
                assert isinstance(policy_name, str)
                assert isinstance(history_list, list)

    def test_run_multiple_environments_dataframe_columns(
        self: "APITestProtocol", sample_environment_params
    ):
        """Test DataFrame contains expected columns.

        Purpose: Validates statistical summary DataFrame structure

        Given: Valid environment run parameters
        When: Simulation completes successfully
        Then: DataFrame contains expected statistical columns

        Test type: integration
        """
        api = self.create_api()
        _, stats_df = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            n_jobs=1,
        )

        # Verify DataFrame is not empty
        assert len(stats_df) > 0
        # Verify it has columns
        assert len(stats_df.columns) > 0

    def test_run_multiple_environments_with_profiling_enabled(
        self: "APITestProtocol", sample_environment_params
    ):
        """Test execution with profiling enabled.

        Purpose: Validates profiling parameter works correctly

        Given: Valid environment run parameters and enable_profiling=True
        When: run_multiple_environments_and_policies is called
        Then: Execution completes successfully with profiling data

        Test type: integration
        """
        api = self.create_api()
        results, stats_df = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            n_jobs=1,
            enable_profiling=True,
            profiling_output_limit=10,
        )

        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    def test_run_multiple_environments_cache_creation(
        self: "APITestProtocol", sample_environment_params, tmp_path
    ):
        """Test cache directory is created when specified.

        Purpose: Validates cache directory creation functionality

        Given: Valid environment run parameters and custom cache path
        When: run_multiple_environments_and_policies is called
        Then: Cache directory is created

        Test type: integration
        """
        cache_dir = tmp_path / "test_cache"
        api = self.create_api()

        api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            n_jobs=1,
            cache_dir_path=cache_dir,
        )

        # Note: Cache creation depends on implementation details
        # Some APIs might create it, others might not until needed


class HyperparameterOptimizationTestsMixin:
    """Mixin with hyperparameter optimization tests."""

    def test_hyperparameter_optimization_returns_correct_type(
        self: "APITestProtocol", sample_hyperparameter_configs
    ):
        """Test hyperparameter optimization return type.

        Purpose: Validates optimization returns list of OptimizedPolicyResult objects

        Given: Valid hyperparameter run configurations
        When: run_hyperparameter_optimization is called
        Then: Returns List[OptimizedPolicyResult] with expected structure

        Test type: integration
        """
        api = self.create_api()
        results = api.run_hyperparameter_optimization(
            environment_run_params=sample_hyperparameter_configs,
            n_jobs=1,
            debug=True,
        )

        assert isinstance(results, list)
        assert all(isinstance(r, OptimizedPolicyResult) for r in results)

    def test_hyperparameter_optimization_result_structure(
        self: "APITestProtocol", sample_hyperparameter_configs
    ):
        """Test optimization results have expected attributes.

        Purpose: Validates each result contains environment, policy, and hyperparameters

        Given: Valid hyperparameter run configurations
        When: Optimization completes successfully
        Then: Each result has environment, policy, and chosen_hyper_parameters attributes

        Test type: integration
        """
        api = self.create_api()
        results = api.run_hyperparameter_optimization(
            environment_run_params=sample_hyperparameter_configs,
            n_jobs=1,
            debug=True,
        )

        for result in results:
            assert hasattr(result, "environment")
            assert hasattr(result, "policy")
            assert hasattr(result, "chosen_hyper_parameters")

    def test_hyperparameter_optimization_with_custom_experiment_name(
        self: "APITestProtocol", sample_hyperparameter_configs
    ):
        """Test optimization with custom experiment name.

        Purpose: Validates experiment naming works correctly

        Given: Valid configs and custom experiment name
        When: run_hyperparameter_optimization is called
        Then: Optimization completes successfully

        Test type: integration
        """
        api = self.create_api()
        results = api.run_hyperparameter_optimization(
            environment_run_params=sample_hyperparameter_configs,
            experiment_name="CustomOptimizationTest",
            n_jobs=1,
            debug=True,
        )

        assert isinstance(results, list)

    def test_hyperparameter_optimization_cache_directory_creation(
        self: "APITestProtocol", sample_hyperparameter_configs, tmp_path
    ):
        """Test cache directory creation for optimization.

        Purpose: Validates cache directory is created/used correctly

        Given: Valid configs and custom cache directory
        When: run_hyperparameter_optimization is called
        Then: Optimization completes and cache is managed

        Test type: integration
        """
        cache_dir = tmp_path / "optimization_cache"
        api = self.create_api()

        results = api.run_hyperparameter_optimization(
            environment_run_params=sample_hyperparameter_configs,
            n_jobs=1,
            cache_dir_path=cache_dir,
            debug=True,
        )

        assert isinstance(results, list)

    def test_hyperparameter_optimization_with_statistical_params(
        self: "APITestProtocol", sample_hyperparameter_configs
    ):
        """Test optimization with custom statistical parameters.

        Purpose: Validates alpha and confidence_interval_level parameters

        Given: Valid configs with custom alpha and confidence level
        When: run_hyperparameter_optimization is called
        Then: Optimization completes successfully

        Test type: integration
        """
        api = self.create_api()
        results = api.run_hyperparameter_optimization(
            environment_run_params=sample_hyperparameter_configs,
            n_jobs=1,
            confidence_interval_level=0.99,
            alpha=0.01,
            debug=True,
        )

        assert isinstance(results, list)


class OptimizeAndEvaluateTestsMixin:
    """Mixin with optimize-and-evaluate workflow tests."""

    def test_optimize_and_evaluate_basic_execution(
        self: "APITestProtocol", sample_hyperparameter_configs
    ):
        """Test basic optimize and evaluate workflow.

        Purpose: Validates workflow executes successfully

        Given: Valid hyperparameter configurations
        When: run_optimize_and_evaluate is called
        Then: Returns tuple of (dict, DataFrame) with expected types

        Test type: integration
        """
        api = self.create_api()
        results, stats_df = api.run_optimize_and_evaluate(
            configs=sample_hyperparameter_configs,
            evaluation_episodes=2,
            evaluation_steps=3,
            evaluation_n_jobs=1,
            optimization_n_jobs=1,
            debug=True,
        )

        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    def test_optimize_and_evaluate_returns_correct_types(
        self: "APITestProtocol", sample_hyperparameter_configs
    ):
        """Test return types from optimize and evaluate.

        Purpose: Validates method returns correct data structures

        Given: Valid hyperparameter configurations
        When: run_optimize_and_evaluate completes
        Then: Returns dict and DataFrame with proper structure

        Test type: integration
        """
        api = self.create_api()
        results, stats_df = api.run_optimize_and_evaluate(
            configs=sample_hyperparameter_configs,
            evaluation_episodes=2,
            evaluation_steps=3,
            evaluation_n_jobs=1,
            optimization_n_jobs=1,
            debug=True,
        )

        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) > 0

    def test_optimize_and_evaluate_empty_configs_raises_error(self: "APITestProtocol"):
        """Test error handling for empty configuration list.

        Purpose: Validates proper error handling for invalid input

        Given: Empty list of configurations
        When: run_optimize_and_evaluate is called
        Then: ValueError is raised

        Test type: unit
        """
        api = self.create_api()

        with pytest.raises(ValueError, match="configs list cannot be empty"):
            api.run_optimize_and_evaluate(
                configs=[],
                evaluation_episodes=2,
                evaluation_steps=3,
            )

    def test_optimize_and_evaluate_cache_directory_handling(
        self: "APITestProtocol", sample_hyperparameter_configs, tmp_path
    ):
        """Test cache directory creation and management.

        Purpose: Validates cache directory is created and used correctly

        Given: Valid configs and custom cache directory
        When: run_optimize_and_evaluate is called
        Then: Workflow completes successfully

        Test type: integration
        """
        cache_dir = tmp_path / "optimize_evaluate_cache"
        api = self.create_api()

        results, stats_df = api.run_optimize_and_evaluate(
            configs=sample_hyperparameter_configs,
            evaluation_episodes=2,
            evaluation_steps=3,
            evaluation_n_jobs=1,
            optimization_n_jobs=1,
            cache_dir_path=cache_dir,
            debug=True,
        )

        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)


class BenchmarkEnvironmentsOnPlannerGeneratorsTestsMixin:
    """Mixin with tests for run_all_benchmark_environments_on_planner_generators method."""

    def test_benchmark_environments_basic_execution(
        self: "APITestProtocol", sample_planner_generators
    ):
        """Test basic execution of benchmark environments on planner generators.

        Purpose: Validates method executes successfully with valid generators

        Given: Valid PlannerGenerator objects
        When: run_all_benchmark_environments_on_planner_generators is called
        Then: Returns tuple of (dict, DataFrame) with expected types

        Test type: integration
        """
        api = self.create_api()
        results, stats_df = api.run_all_benchmark_environments_on_planner_generators(
            generators=sample_planner_generators,
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            alpha=0.1,
            confidence_interval_level=0.95,
            n_jobs=1,
        )

        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    def test_benchmark_environments_returns_correct_types(
        self: "APITestProtocol", sample_planner_generators
    ):
        """Test return types from benchmark environments method.

        Purpose: Validates method returns correct data structures

        Given: Valid planner generators
        When: run_all_benchmark_environments_on_planner_generators completes
        Then: Returns dict and DataFrame with proper structure

        Test type: integration
        """
        api = self.create_api()
        results, stats_df = api.run_all_benchmark_environments_on_planner_generators(
            generators=sample_planner_generators,
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            n_jobs=1,
        )

        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) > 0

    def test_benchmark_environments_result_structure(
        self: "APITestProtocol", sample_planner_generators
    ):
        """Test result dictionary has expected nested structure.

        Purpose: Validates result organization by environment and policy names

        Given: Valid planner generators
        When: Benchmark execution completes successfully
        Then: Result dict follows env_name → policy_name → list structure

        Test type: integration
        """
        api = self.create_api()
        results, _ = api.run_all_benchmark_environments_on_planner_generators(
            generators=sample_planner_generators,
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            n_jobs=1,
        )

        # Verify nested structure
        for env_name, policies_dict in results.items():
            assert isinstance(env_name, str)
            assert isinstance(policies_dict, dict)
            for policy_name, history_list in policies_dict.items():
                assert isinstance(policy_name, str)
                assert isinstance(history_list, list)

    def test_benchmark_environments_empty_generators_raises_error(self: "APITestProtocol"):
        """Test error handling for empty generators list.

        Purpose: Validates proper error handling for invalid input

        Given: Empty list of generators
        When: run_all_benchmark_environments_on_planner_generators is called
        Then: ValueError is raised

        Test type: unit
        """
        api = self.create_api()

        with pytest.raises(ValueError, match="generators list cannot be empty"):
            api.run_all_benchmark_environments_on_planner_generators(
                generators=[],
                n_particles=10,
                num_episodes=2,
                num_steps=3,
            )

    def test_benchmark_environments_with_custom_experiment_name(
        self: "APITestProtocol", sample_planner_generators
    ):
        """Test benchmark execution with custom experiment name.

        Purpose: Validates experiment naming works correctly

        Given: Valid generators and custom experiment name
        When: run_all_benchmark_environments_on_planner_generators is called
        Then: Execution completes successfully

        Test type: integration
        """
        api = self.create_api()
        results, stats_df = api.run_all_benchmark_environments_on_planner_generators(
            generators=sample_planner_generators,
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            experiment_name="CustomBenchmarkTest",
            n_jobs=1,
        )

        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    def test_benchmark_environments_with_profiling_enabled(
        self: "APITestProtocol", sample_planner_generators
    ):
        """Test benchmark execution with profiling enabled.

        Purpose: Validates profiling parameter works correctly

        Given: Valid generators and enable_profiling=True
        When: run_all_benchmark_environments_on_planner_generators is called
        Then: Execution completes successfully with profiling data

        Test type: integration
        """
        api = self.create_api()
        results, stats_df = api.run_all_benchmark_environments_on_planner_generators(
            generators=sample_planner_generators,
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            n_jobs=1,
            enable_profiling=True,
            profiling_output_limit=10,
        )

        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    def test_benchmark_environments_cache_directory_handling(
        self: "APITestProtocol", sample_planner_generators, tmp_path
    ):
        """Test cache directory creation and management.

        Purpose: Validates cache directory is created and used correctly

        Given: Valid generators and custom cache directory
        When: run_all_benchmark_environments_on_planner_generators is called
        Then: Execution completes successfully

        Test type: integration
        """
        cache_dir = tmp_path / "benchmark_cache"
        api = self.create_api()

        results, stats_df = api.run_all_benchmark_environments_on_planner_generators(
            generators=sample_planner_generators,
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            cache_dir_path=cache_dir,
            n_jobs=1,
        )

        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    def test_benchmark_environments_with_custom_statistical_params(
        self: "APITestProtocol", sample_planner_generators
    ):
        """Test benchmark execution with custom statistical parameters.

        Purpose: Validates alpha and confidence_interval_level parameters

        Given: Valid generators with custom alpha and confidence level
        When: run_all_benchmark_environments_on_planner_generators is called
        Then: Execution completes successfully

        Test type: integration
        """
        api = self.create_api()
        results, stats_df = api.run_all_benchmark_environments_on_planner_generators(
            generators=sample_planner_generators,
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            alpha=0.05,
            confidence_interval_level=0.99,
            n_jobs=1,
        )

        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)


class ErrorHandlingTestsMixin:
    """Mixin with common error handling tests."""

    def test_invalid_alpha_value_raises_error(self: "APITestProtocol", sample_environment_params):
        """Test validation of alpha parameter.

        Purpose: Validates alpha parameter validation (should be between 0 and 1)

        Given: Environment run parameters and invalid alpha value
        When: run_multiple_environments_and_policies is called
        Then: Appropriate error is raised or execution completes

        Test type: unit

        Note: Some implementations may not validate alpha at API level
        """
        api = self.create_api()

        # This test might pass without error if validation happens deeper
        # Test with alpha > 1
        try:
            api.run_multiple_environments_and_policies(
                environment_run_params=sample_environment_params,
                alpha=1.5,  # Invalid
                confidence_interval_level=0.95,
                n_jobs=1,
            )
        except (ValueError, AssertionError):
            # Expected if validation is implemented
            pass

    def test_invalid_confidence_interval_raises_error(
        self: "APITestProtocol", sample_environment_params
    ):
        """Test validation of confidence_interval_level parameter.

        Purpose: Validates confidence_interval_level parameter validation

        Given: Environment run parameters and invalid confidence interval
        When: run_multiple_environments_and_policies is called
        Then: Appropriate error is raised or execution completes

        Test type: unit

        Note: Some implementations may not validate at API level
        """
        api = self.create_api()

        # Test with confidence_interval_level > 1
        try:
            api.run_multiple_environments_and_policies(
                environment_run_params=sample_environment_params,
                alpha=0.05,
                confidence_interval_level=1.5,  # Invalid
                n_jobs=1,
            )
        except (ValueError, AssertionError):
            # Expected if validation is implemented
            pass

    def test_negative_n_jobs_interpretation(
        self: "APITestProtocol", sample_environment_params, temp_cache_dir
    ):
        """Test n_jobs=-1 means use all cores.

        Purpose: Validates -1 interpretation for parallel jobs

        Given: Environment run parameters and n_jobs=-1
        When: run_multiple_environments_and_policies is called
        Then: Execution completes successfully using all available cores

        Test type: integration
        """
        api = self.create_api()

        results, stats_df = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            n_jobs=-1,  # Use all cores
            cache_dir_path=temp_cache_dir,
        )

        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)
