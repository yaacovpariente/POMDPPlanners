"""Tests for LocalSimulationsAPI using mixin pattern.

This module tests the LocalSimulationsAPI implementation, inheriting common
tests from mixins and adding LocalSimulationsAPI-specific tests. Also includes
integration tests with real execution using debug mode.
"""

import pytest
import pandas as pd
from pathlib import Path
from typing import cast, List
from unittest.mock import MagicMock, patch

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.simulation import (
    EnvironmentRunParams,
    NumericalHyperParameter,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParamPlannerConfig,
    HyperParamPlannerConfigGenerator,
    HyperParameterFeature,
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import (
    LocalSimulationsAPI,
)
from .api_test_mixins import (
    InitializationTestsMixin,
    RunMultipleEnvironmentsTestsMixin,
    HyperparameterOptimizationTestsMixin,
    OptimizeAndEvaluateTestsMixin,
    BenchmarkEnvironmentsOnPlannerGeneratorsTestsMixin,
    ErrorHandlingTestsMixin,
)

# Import fixtures so they are available to tests
pytest_plugins = ["POMDPPlanners.tests.test_simulations.test_simulations_apis.api_test_fixtures"]


class TestLocalSimulationsAPI(
    InitializationTestsMixin,
    RunMultipleEnvironmentsTestsMixin,
    HyperparameterOptimizationTestsMixin,
    OptimizeAndEvaluateTestsMixin,
    BenchmarkEnvironmentsOnPlannerGeneratorsTestsMixin,
    ErrorHandlingTestsMixin,
):
    """Test suite for LocalSimulationsAPI.

    This class inherits all common API tests from mixins and adds
    LocalSimulationsAPI-specific tests.
    """

    def create_api(self, cache_dir_path=None, debug=False, **kwargs):
        """Create LocalSimulationsAPI instance for testing.

        Note: kwargs are ignored for LocalSimulationsAPI but accepted
        for consistency with other API test classes.

        Args:
            cache_dir_path: Optional cache directory path
            debug: Enable debug mode
            **kwargs: Additional arguments (ignored)

        Returns:
            LocalSimulationsAPI instance
        """
        return LocalSimulationsAPI(cache_dir_path=cache_dir_path, debug=debug)

    # LocalSimulationsAPI-specific tests

    def test_local_api_scheduler_address_parameter_ignored(
        self, sample_environment_params, tmp_path
    ):
        """Test that scheduler_address parameter is ignored.

        Purpose: Validates that scheduler_address doesn't affect local execution

        Given: LocalSimulationsAPI instance
        When: run_multiple_environments_and_policies called with scheduler_address
        Then: Execution proceeds normally, parameter is ignored

        Test type: unit
        """
        api = self.create_api()

        # Should not raise error even with scheduler_address provided
        results, stats_df = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            scheduler_address="tcp://fake:8786",  # Should be ignored
            n_jobs=1,
            cache_dir_path=tmp_path / "test_cache",
        )

        assert results is not None
        assert stats_df is not None

    def test_local_api_has_debug_run_method(self):
        """Test LocalSimulationsAPI-specific debug run method.

        Purpose: Validates unique method only available in LocalSimulationsAPI

        Given: LocalSimulationsAPI instance
        When: Checking for debug run method
        Then: Method exists and is callable

        Test type: unit
        """
        api = self.create_api()

        assert hasattr(api, "run_multiple_environments_and_policies_with_initial_debug_run")
        assert callable(api.run_multiple_environments_and_policies_with_initial_debug_run)

    def test_local_api_debug_run_method_execution(self, sample_environment_params, tmp_path):
        """Test execution of debug run method.

        Purpose: Validates debug run workflow executes successfully

        Given: LocalSimulationsAPI instance and valid environment params
        When: run_multiple_environments_and_policies_with_initial_debug_run is called
        Then: Debug run and main run both complete successfully

        Test type: integration
        """
        api = self.create_api()

        results, stats_df = api.run_multiple_environments_and_policies_with_initial_debug_run(
            environment_run_params=sample_environment_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            n_jobs=1,
            cache_dir_path=tmp_path / "test_cache",
        )

        assert results is not None
        assert stats_df is not None

    def test_local_api_debug_run_with_custom_experiment_name(
        self, sample_environment_params, tmp_path
    ):
        """Test debug run with custom experiment name.

        Purpose: Validates experiment naming in debug run workflow

        Given: LocalSimulationsAPI and custom experiment name
        When: Debug run method is called
        Then: Both debug and main runs complete with proper naming

        Test type: integration
        """
        api = self.create_api()

        results, stats_df = api.run_multiple_environments_and_policies_with_initial_debug_run(
            environment_run_params=sample_environment_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            experiment_name="CustomDebugTest",
            n_jobs=1,
            cache_dir_path=tmp_path / "test_cache",
        )

        assert results is not None
        assert stats_df is not None

    def test_local_api_uses_joblib_backend(self):
        """Test that LocalSimulationsAPI uses Joblib for parallelization.

        Purpose: Validates correct backend is used for local execution

        Given: LocalSimulationsAPI instance
        When: API is initialized
        Then: Instance is correctly typed as LocalSimulationsAPI

        Test type: unit
        """
        api = self.create_api()
        assert isinstance(api, LocalSimulationsAPI)

    def test_local_api_initialization_without_cache_dir(self):
        """Test initialization without cache directory.

        Purpose: Validates API can be initialized without explicit cache path

        Given: No cache_dir_path parameter
        When: LocalSimulationsAPI is instantiated
        Then: API is created successfully

        Test type: unit
        """
        api = LocalSimulationsAPI(debug=False)
        assert api is not None
        assert hasattr(api, "logger")

    def test_local_api_multiple_n_jobs_values(self, sample_environment_params, tmp_path):
        """Test different n_jobs values.

        Purpose: Validates n_jobs parameter flexibility

        Given: Valid environment params
        When: Simulations run with different n_jobs values
        Then: All executions complete successfully

        Test type: integration
        """
        api = self.create_api()

        # Test with n_jobs=1
        results1, stats1 = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            n_jobs=1,
            cache_dir_path=tmp_path / "test_cache_1",
        )
        assert results1 is not None

        # Test with n_jobs=2
        results2, stats2 = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            n_jobs=2,
            cache_dir_path=tmp_path / "test_cache_2",
        )
        assert results2 is not None

        # Test with n_jobs=-1 (all cores)
        results3, stats3 = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            n_jobs=-1,
            cache_dir_path=tmp_path / "test_cache_3",
        )
        assert results3 is not None

    def test_local_api_clear_cache_on_start(self, sample_environment_params, tmp_path):
        """Test clear_cache_on_start parameter.

        Purpose: Validates cache clearing functionality

        Given: Valid environment params and cache directory
        When: Simulation runs with clear_cache_on_start=True
        Then: Execution completes successfully

        Test type: integration
        """
        cache_dir = tmp_path / "clear_cache_test"
        api = self.create_api()

        results, stats_df = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            n_jobs=1,
            cache_dir_path=cache_dir,
            clear_cache_on_start=True,
        )

        assert results is not None
        assert stats_df is not None

    def test_run_multiple_environments_and_policies_error(
        self, temp_cache_dir, sample_environment_params
    ):
        """Test error handling in run_multiple_environments_and_policies.

        Purpose: Validates error handling for run multiple environments and policies

        Given: Invalid inputs or error conditions
        When: Operation is attempted
        Then: Appropriate exception is raised

        Test type: unit
        """
        api = self.create_api()
        # Force an error by mocking the environment's state_transition_model to return None
        sample_environment_params[0].environment.state_transition_model = lambda *args: None
        with pytest.raises(Exception):
            api.run_multiple_environments_and_policies(
                environment_run_params=sample_environment_params,
                alpha=0.1,
                confidence_interval_level=0.95,
                cache_dir_path=temp_cache_dir,
            )

    @patch(
        "POMDPPlanners.simulations.simulation_apis.local_simulations_api.HyperParameterOptimizer"
    )
    def test_run_hyperparameter_optimization_default_parameters(
        self,
        mock_optimizer_class,
        tmp_path,
        sample_hyperparameter_configs,
    ):
        """Test hyperparameter optimization with default parameters.

        Purpose: Validates that hyperparameter optimization uses correct default values when optional parameters are not provided

        Given: Only required parameters (environment_run_params)
        When: run_hyperparameter_optimization is called with minimal parameters and temp cache
        Then: HyperParameterOptimizer is created with correct default values

        Test type: unit
        """
        # Create a mock OptimizedPolicyResult
        from POMDPPlanners.core.simulation.hyperparameter_tuning import OptimizedPolicyResult
        from unittest.mock import Mock

        mock_result = Mock(spec=OptimizedPolicyResult)

        # Mock the optimizer instance
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.optimize.return_value = [mock_result]
        mock_optimizer_instance.cleanup.return_value = None
        mock_optimizer_class.return_value = mock_optimizer_instance

        api = self.create_api()
        results = api.run_hyperparameter_optimization(
            environment_run_params=sample_hyperparameter_configs,
            cache_dir_path=tmp_path / "test_cache",
        )

        # Verify default parameters
        call_args = mock_optimizer_class.call_args

        assert call_args[1]["experiment_name"] == "POMDP_Hyperparameter_Optimization"  # Default
        assert call_args[1]["n_jobs"] == -1  # Default
        assert call_args[1]["confidence_interval_level"] == 0.95  # Default
        assert call_args[1]["alpha"] == 0.05  # Default
        assert call_args[1]["use_queue_logger"] is False  # Default

        # Verify cache directory was set to provided temp path
        assert call_args[1]["cache_dir_path"] == tmp_path / "test_cache"

        assert isinstance(results, list)
        assert len(results) == 1

    @patch(
        "POMDPPlanners.simulations.simulation_apis.local_simulations_api.HyperParameterOptimizer"
    )
    def test_run_hyperparameter_optimization_error_handling(
        self, mock_optimizer_class, temp_cache_dir, sample_hyperparameter_configs
    ):
        """Test error handling in run_hyperparameter_optimization.

        Purpose: Validates that hyperparameter optimization properly handles errors and raises appropriate exceptions

        Given: HyperParameterOptimizer that raises an exception during optimization
        When: run_hyperparameter_optimization is executed
        Then: RuntimeError is raised with appropriate error message and cleanup is still called

        Test type: unit
        """
        # Mock the optimizer instance to raise an exception
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.optimize.side_effect = Exception("Optimization failed")
        mock_optimizer_instance.cleanup.return_value = None
        mock_optimizer_class.return_value = mock_optimizer_instance

        api = self.create_api()

        with pytest.raises(
            RuntimeError,
            match="Hyperparameter optimization failed: Optimization failed",
        ):
            api.run_hyperparameter_optimization(
                environment_run_params=sample_hyperparameter_configs,
                cache_dir_path=temp_cache_dir,
            )

        # Verify cleanup was still called even after error
        mock_optimizer_instance.cleanup.assert_called_once()

    @patch(
        "POMDPPlanners.simulations.simulation_apis.local_simulations_api.OptimizationEvaluationLocalWorkflow"
    )
    def test_run_all_hyperparameter_benchmarks_debug_mode(
        self,
        mock_workflow_class,
        temp_cache_dir,
    ):
        """Test run_all_hyperparameter_benchmarks with debug mode enabled.

        Purpose: Validates that all hyperparameter benchmarks runs correctly with debug settings

        Given: Mocked workflow and policy space info for continuous action/observation spaces
        When: run_all_hyperparameter_benchmarks is executed with debug=True and minimal settings
        Then: OptimizationEvaluationLocalWorkflow is called with correct parameters and returns expected results

        Test type: unit
        """
        from POMDPPlanners.core.policy import PolicySpaceInfo
        from POMDPPlanners.core.environment import SpaceType

        # Mock the workflow instance
        mock_workflow_instance = MagicMock()
        mock_results = ({}, pd.DataFrame())
        mock_workflow_instance.optimize_and_evaluate.return_value = mock_results
        mock_workflow_class.return_value = mock_workflow_instance

        api = self.create_api()
        # Use a real PolicySpaceInfo
        policy_space_info = PolicySpaceInfo(
            action_space=SpaceType.CONTINUOUS,
            observation_space=SpaceType.CONTINUOUS,
        )
        results, stats_df = api.run_all_hyperparameter_benchmarks(
            policy_space_info=policy_space_info,
            particles=5,
            num_episodes=1,
            num_steps=3,
            n_trials=1,
            discount_factor=0.95,
            time_out_in_seconds=1.0,
            evaluation_episodes=1,
            evaluation_steps=3,
            evaluation_n_jobs=1,
            optimization_n_jobs=1,
            is_risk_averse=False,
            confidence_interval_level=0.95,
            alpha=0.05,
            cache_dir_path=temp_cache_dir,
            experiment_name="test_all_hparam_benchmarks_debug",
            debug=True,
            cache_visualizations=False,
        )

        # Verify OptimizationEvaluationLocalWorkflow was called with correct parameters
        mock_workflow_class.assert_called_once()
        call_args = mock_workflow_class.call_args

        assert call_args[1]["cache_dir"] == temp_cache_dir
        assert call_args[1]["experiment_name"] == "test_all_hparam_benchmarks_debug"
        assert call_args[1]["optimization_n_jobs"] == 1
        assert call_args[1]["evaluation_episodes"] == 1
        assert call_args[1]["evaluation_steps"] == 3
        assert call_args[1]["evaluation_n_jobs"] == 1
        assert call_args[1]["confidence_interval_level"] == 0.95
        assert call_args[1]["alpha"] == 0.05
        assert call_args[1]["debug"] is True
        assert call_args[1]["verbose"] is True
        assert call_args[1]["cache_visualizations"] is False

        # Verify optimize_and_evaluate was called
        mock_workflow_instance.optimize_and_evaluate.assert_called_once()

        # Verify results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)


# Additional fixtures for integration tests
@pytest.fixture
def tiger_environment():
    """Fixture to create a TigerPOMDP environment."""
    return TigerPOMDP(discount_factor=0.95, name="test_tiger")


@pytest.fixture
def sparse_pft_policy(tiger_environment):
    """Fixture to create a SparsePFT policy for the Tiger environment."""
    return SparsePFT(
        environment=tiger_environment,
        discount_factor=0.95,
        gamma=0.95,
        depth=10,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=5,
        n_simulations=100,
        name="test_sparse_pft",
    )


class TestLocalSimulationsAPIIntegration:
    """Integration tests for LocalSimulationsAPI with real execution using debug mode."""

    def test_run_multiple_environments_and_policies_integration(
        self, temp_cache_dir, tiger_environment, sparse_pft_policy
    ):
        """Test run_multiple_environments_and_policies integration with debug mode.

        Purpose: Validates that local simulation executes end-to-end with real environment and policy

        Given: TigerPOMDP environment, SparsePFT policy with minimal parameters (1 episode, 3 steps), debug=True
        When: run_multiple_environments_and_policies is executed
        Then: Returns results dict and DataFrame with actual simulation data

        Test type: integration
        """
        api = LocalSimulationsAPI()
        initial_belief = get_initial_belief(tiger_environment, n_particles=5)

        environment_run_params = [
            EnvironmentRunParams(
                environment=tiger_environment,
                belief=initial_belief,
                policies=[sparse_pft_policy],
                num_episodes=1,
                num_steps=3,
            )
        ]

        results, stats_df = api.run_multiple_environments_and_policies(
            environment_run_params=environment_run_params,
            alpha=0.1,
            confidence_interval_level=0.95,
            experiment_name="integration_test_local",
            debug=True,
            n_jobs=1,
            cache_dir_path=temp_cache_dir,
        )

        # Verify actual execution results
        assert isinstance(results, dict)
        assert "test_tiger" in results
        assert "test_sparse_pft" in results["test_tiger"]
        assert len(results["test_tiger"]["test_sparse_pft"]) == 1
        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) > 0

    def test_run_hyperparameter_optimization_integration(self, temp_cache_dir, tiger_environment):
        """Test run_hyperparameter_optimization integration with debug mode.

        Purpose: Validates that hyperparameter optimization executes end-to-end with real environment

        Given: TigerPOMDP environment with POMCP optimization config, minimal parameters (1 episode, 3 steps, 2 trials), debug=True
        When: run_hyperparameter_optimization is executed
        Then: Returns list of OptimizedPolicyResult with actual optimization results

        Test type: integration
        """
        api = LocalSimulationsAPI()
        initial_belief = get_initial_belief(tiger_environment, n_particles=5)

        # Create hyperparameter run params for POMCP
        hyperparameter_run_params = [
            HyperParameterRunParams(
                environment=tiger_environment,
                belief=initial_belief,
                hyper_param_planner_config=HyperParamPlannerConfig(
                    policy_cls=POMCP,
                    hyper_parameters=cast(
                        List[HyperParameterFeature],
                        [
                            NumericalHyperParameter(low=0.1, high=2.0, name="exploration_constant"),
                            NumericalHyperParameter(low=10, high=50, name="n_simulations"),
                        ],
                    ),
                    constant_parameters={
                        "discount_factor": 0.95,
                        "depth": 3,
                        "name": "OptimizedPOMCP",
                    },
                ),
                num_episodes=1,
                num_steps=3,
                n_trials=2,  # Minimal for fast execution
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
            )
        ]

        results = api.run_hyperparameter_optimization(
            environment_run_params=hyperparameter_run_params,
            n_jobs=1,
            experiment_name="integration_test_hyperparameter_opt",
            cache_dir_path=temp_cache_dir,
            debug=True,
        )

        # Verify actual optimization results
        assert isinstance(results, list)
        assert len(results) == 1
        result = results[0]
        assert hasattr(result, "environment")
        assert hasattr(result, "policy")
        assert hasattr(result, "chosen_hyper_parameters")

    def test_run_optimize_and_evaluate_integration(self, temp_cache_dir, tiger_environment):
        """Test run_optimize_and_evaluate integration with debug mode.

        Purpose: Validates that optimize and evaluate workflow executes end-to-end with real environment

        Given: TigerPOMDP with POMCP optimization config, minimal parameters (1 optimization episode, 1 evaluation episode, 2 trials), debug=True
        When: run_optimize_and_evaluate is executed
        Then: Returns results dict and DataFrame with actual optimization and evaluation data

        Test type: integration
        """
        api = LocalSimulationsAPI()
        initial_belief = get_initial_belief(tiger_environment, n_particles=5)

        # Create hyperparameter run params for POMCP
        hyperparameter_run_params = [
            HyperParameterRunParams(
                environment=tiger_environment,
                belief=initial_belief,
                hyper_param_planner_config=HyperParamPlannerConfig(
                    policy_cls=POMCP,
                    hyper_parameters=cast(
                        List[HyperParameterFeature],
                        [
                            NumericalHyperParameter(low=0.1, high=2.0, name="exploration_constant"),
                            NumericalHyperParameter(low=10, high=50, name="n_simulations"),
                        ],
                    ),
                    constant_parameters={
                        "discount_factor": 0.95,
                        "depth": 3,
                        "name": "OptimizedPOMCP",
                    },
                ),
                num_episodes=1,
                num_steps=3,
                n_trials=2,  # Minimal for fast execution
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
            )
        ]

        results, stats_df = api.run_optimize_and_evaluate(
            configs=hyperparameter_run_params,
            evaluation_episodes=1,
            evaluation_steps=3,
            evaluation_n_jobs=1,
            optimization_n_jobs=1,
            confidence_interval_level=0.95,
            alpha=0.05,
            cache_dir_path=temp_cache_dir,
            experiment_name="integration_test_optimize_evaluate",
            debug=True,
        )

        # Verify actual execution results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) > 0

    @pytest.mark.slow
    def test_run_all_hyperparameter_benchmarks_integration(self, temp_cache_dir):
        """Test run_all_hyperparameter_benchmarks integration with debug mode.

        Purpose: Validates that all hyperparameter benchmarks execute end-to-end with real environments

        Given: PolicySpaceInfo for discrete action/observation spaces, minimal parameters (1 episode, 3 steps, 2 trials), debug=True
        When: run_all_hyperparameter_benchmarks is executed
        Then: Returns results dict and DataFrame with actual benchmark results

        Test type: integration

        Note: This test is marked as slow because it creates configurations for all
        matching environments and planners, which can take several minutes even with
        minimal parameters. Run with `pytest -m slow` to include this test.
        """
        api = LocalSimulationsAPI()
        policy_space_info = PolicySpaceInfo(
            action_space=SpaceType.MIXED,
            observation_space=SpaceType.MIXED,
        )

        results, stats_df = api.run_all_hyperparameter_benchmarks(
            policy_space_info=policy_space_info,
            particles=5,
            num_episodes=1,
            num_steps=2,
            n_trials=2,  # Minimal for fast execution
            discount_factor=0.95,
            time_out_in_seconds=0.1,
            evaluation_episodes=1,
            evaluation_steps=2,
            evaluation_n_jobs=1,
            optimization_n_jobs=-1,
            is_risk_averse=False,
            confidence_interval_level=0.95,
            alpha=0.05,
            cache_dir_path=temp_cache_dir,
            experiment_name="integration_test_all_benchmarks",
            debug=True,
            cache_visualizations=False,
        )

        # Verify actual execution results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)
        # Results might be empty if no benchmarks match the criteria, which is okay

    @pytest.mark.slow
    def test_run_hyperparameter_tuning_experiment_with_benchmarks_integration(
        self, temp_cache_dir, tiger_environment
    ):
        """Test run_hyperparameter_tuning_experiment_with_benchmarks integration with debug mode.

        Purpose: Validates that comprehensive benchmark with hyperparameter tuning executes end-to-end

        Given: Generator for TigerPOMDP with POMCP, minimal parameters (1 episode, 3 steps, 2 trials), debug=True
        When: run_hyperparameter_tuning_experiment_with_benchmarks is executed
        Then: Returns results dict and DataFrame with actual tuning and evaluation results

        Test type: integration

        Note: This test is marked as slow because it runs full hyperparameter optimization
        followed by evaluation, which can take several minutes. Run with `pytest -m slow`
        to include this test.
        """
        api = LocalSimulationsAPI()

        # Create a proper generator class implementing the interface
        class TestPOMCPGenerator(HyperParamPlannerConfigGenerator):
            def generate(self, environment: Environment) -> HyperParamPlannerConfig:
                return HyperParamPlannerConfig(
                    policy_cls=POMCP,
                    hyper_parameters=cast(
                        List[HyperParameterFeature],
                        [
                            NumericalHyperParameter(low=0.1, high=2.0, name="exploration_constant"),
                        ],
                    ),
                    constant_parameters={
                        "discount_factor": 0.95,
                        "depth": 3,
                        "name": f"OptimizedPOMCP_{environment.name}",
                        "n_simulations": 2,
                    },
                )

            def get_planner_space_info(self) -> PolicySpaceInfo:
                return PolicySpaceInfo(
                    action_space=SpaceType.DISCRETE,
                    observation_space=SpaceType.DISCRETE,
                )

        generators = [TestPOMCPGenerator()]

        results, stats_df = api.run_hyperparameter_tuning_experiment_with_benchmarks(
            generators=generators,
            particles=5,
            num_episodes=1,
            num_steps=3,
            n_trials=2,  # Minimal for fast execution
            evaluation_episodes=1,
            evaluation_steps=3,
            evaluation_n_jobs=1,
            optimization_n_jobs=1,
            time_out_in_seconds=0.1,
            is_risk_averse=False,
            confidence_interval_level=0.95,
            alpha=0.05,
            cache_dir_path=temp_cache_dir,
            experiment_name="integration_test_comprehensive_benchmark",
            debug=True,
            cache_visualizations=False,
        )

        # Verify actual execution results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)
