from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.simulations.simulation_apis.dask_simulations_api import DaskSimulationsAPI

# Fixtures temp_cache_dir, tiger_environment, sparse_pft_policy, and
# sample_environment_run_params are now provided by the top-level conftest.py


class TestDaskSimulationsAPI:
    @patch("POMDPPlanners.simulations.simulation_apis.dask_simulations_api.POMDPSimulator")
    def test_run_multiple_environments_and_policies_with_scheduler_address(
        self, mock_simulator_class, temp_cache_dir, sample_environment_run_params
    ):
        """Test run_multiple_environments_and_policies with custom Dask scheduler address.

        Purpose: Validates that Dask simulation uses provided scheduler address correctly

        Given: TigerPOMDP environment, SparsePFT policy, custom scheduler address "tcp://localhost:8786"
        When: run_multiple_environments_and_policies is executed with scheduler_address
        Then: DaskConfig is created with correct scheduler address and simulation executes

        Test type: unit
        """
        # Mock the simulator context manager
        mock_simulator_instance = MagicMock()
        mock_results = (
            {"test_tiger": {"test_sparse_pft": ["mock_history"]}},
            pd.DataFrame({"environment": ["test_tiger"], "policy": ["test_sparse_pft"]}),
        )
        mock_simulator_instance.compare_multiple_environments_policies.return_value = mock_results
        mock_simulator_class.return_value.__enter__.return_value = mock_simulator_instance
        mock_simulator_class.return_value.__exit__.return_value = None

        api = DaskSimulationsAPI()
        results, stats_df = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_run_params,
            alpha=0.1,
            confidence_interval_level=0.95,
            experiment_name="test_dask_experiment",
            scheduler_address="tcp://localhost:8786",
            n_jobs=4,
            cache_dir_path=temp_cache_dir,
        )

        # Verify POMDPSimulator was called
        mock_simulator_class.assert_called_once()
        call_args = mock_simulator_class.call_args

        # Verify DaskConfig parameters
        task_manager_config = call_args[1]["task_manager_config"]
        assert task_manager_config.n_workers == 4
        assert task_manager_config.scheduler_address == "tcp://localhost:8786"

        # Verify results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    @patch("POMDPPlanners.simulations.simulation_apis.dask_simulations_api.POMDPSimulator")
    def test_run_multiple_environments_and_policies_local_cluster(
        self, mock_simulator_class, temp_cache_dir, sample_environment_run_params
    ):
        """Test run_multiple_environments_and_policies with local Dask cluster.

        Purpose: Validates that Dask simulation creates local cluster when no scheduler address provided

        Given: TigerPOMDP environment, SparsePFT policy, no scheduler_address (None)
        When: run_multiple_environments_and_policies is executed
        Then: DaskConfig is created with scheduler_address=None for local cluster

        Test type: unit
        """
        # Mock the simulator context manager
        mock_simulator_instance = MagicMock()
        mock_results = (
            {"test_tiger": {"test_sparse_pft": ["mock_history"]}},
            pd.DataFrame({"environment": ["test_tiger"], "policy": ["test_sparse_pft"]}),
        )
        mock_simulator_instance.compare_multiple_environments_policies.return_value = mock_results
        mock_simulator_class.return_value.__enter__.return_value = mock_simulator_instance
        mock_simulator_class.return_value.__exit__.return_value = None

        api = DaskSimulationsAPI()
        results, stats_df = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_run_params,
            alpha=0.1,
            confidence_interval_level=0.95,
            experiment_name="test_dask_local_cluster",
            scheduler_address=None,  # Local cluster
            n_jobs=2,
            cache_dir_path=temp_cache_dir,
        )

        # Verify POMDPSimulator was called
        mock_simulator_class.assert_called_once()
        call_args = mock_simulator_class.call_args

        # Verify DaskConfig parameters for local cluster
        task_manager_config = call_args[1]["task_manager_config"]
        assert task_manager_config.n_workers == 2
        assert task_manager_config.scheduler_address is None  # Local cluster

        # Verify results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    @patch("POMDPPlanners.simulations.simulation_apis.dask_simulations_api.POMDPSimulator")
    def test_run_multiple_environments_and_policies_with_cache_clearing(
        self, mock_simulator_class, temp_cache_dir, sample_environment_run_params
    ):
        """Test run_multiple_environments_and_policies with cache clearing enabled.

        Purpose: Validates that Dask simulation correctly clears cache when requested

        Given: TigerPOMDP environment, clear_cache_on_start=True
        When: run_multiple_environments_and_policies is executed
        Then: DaskConfig is created with clear_cache_on_start=True

        Test type: unit
        """
        # Mock the simulator context manager
        mock_simulator_instance = MagicMock()
        mock_results = (
            {"test_tiger": {"test_sparse_pft": ["mock_history"]}},
            pd.DataFrame({"environment": ["test_tiger"], "policy": ["test_sparse_pft"]}),
        )
        mock_simulator_instance.compare_multiple_environments_policies.return_value = mock_results
        mock_simulator_class.return_value.__enter__.return_value = mock_simulator_instance
        mock_simulator_class.return_value.__exit__.return_value = None

        api = DaskSimulationsAPI()
        results, stats_df = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_run_params,
            alpha=0.1,
            confidence_interval_level=0.95,
            experiment_name="test_dask_cache_clear",
            clear_cache_on_start=True,
            cache_dir_path=temp_cache_dir,
        )

        # Verify DaskConfig has clear_cache_on_start=True
        call_args = mock_simulator_class.call_args
        task_manager_config = call_args[1]["task_manager_config"]
        assert task_manager_config.clear_cache_on_start is True

        # Verify results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    @patch("POMDPPlanners.simulations.simulation_apis.dask_simulations_api.POMDPSimulator")
    def test_run_multiple_environments_and_policies_default_parameters(
        self, mock_simulator_class, sample_environment_run_params
    ):
        """Test run_multiple_environments_and_policies with default parameters.

        Purpose: Validates that Dask simulation uses correct default values

        Given: Only required parameters (environment_run_params, alpha, confidence_interval_level)
        When: run_multiple_environments_and_policies is executed
        Then: POMDPSimulator and DaskConfig are created with correct default values

        Test type: unit
        """
        # Mock the simulator context manager
        mock_simulator_instance = MagicMock()
        mock_results = (
            {"test_tiger": {"test_sparse_pft": ["mock_history"]}},
            pd.DataFrame({"environment": ["test_tiger"], "policy": ["test_sparse_pft"]}),
        )
        mock_simulator_instance.compare_multiple_environments_policies.return_value = mock_results
        mock_simulator_class.return_value.__enter__.return_value = mock_simulator_instance
        mock_simulator_class.return_value.__exit__.return_value = None

        api = DaskSimulationsAPI()
        results, stats_df = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_run_params,
            alpha=0.1,
            confidence_interval_level=0.95,
        )

        # Verify default parameters
        call_args = mock_simulator_class.call_args

        assert call_args[1]["experiment_name"] == "POMDP_Planning_Comparison"  # Default
        assert call_args[1]["debug"] is False  # Default
        assert call_args[1]["enable_profiling"] is False  # Default
        assert call_args[1]["profiling_output_limit"] == 50  # Default

        # Verify DaskConfig defaults
        task_manager_config = call_args[1]["task_manager_config"]
        assert task_manager_config.n_workers == -1  # Default
        assert task_manager_config.scheduler_address is None  # Default
        assert task_manager_config.clear_cache_on_start is False  # Default

        # Verify results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)


class TestDaskSimulationsAPIHyperparameterOptimization:
    """Test hyperparameter optimization methods in DaskSimulationsAPI."""

    @patch(
        "POMDPPlanners.simulations.simulation_apis.dask_simulations_api.OptimizationEvaluationDaskWorkflow"
    )
    @patch(
        "POMDPPlanners.simulations.simulation_apis.dask_simulations_api.PolicyHyperparameterOptimizationExperimentConfigCreator"
    )
    def test_run_hyperparameter_tuning_experiment_with_benchmarks(
        self, mock_creator_class, mock_workflow_class, temp_cache_dir
    ):
        """Test run_hyperparameter_tuning_experiment_with_benchmarks with Dask.

        Purpose: Validates that comprehensive benchmark uses DaskWorkflow correctly

        Given: Generator list, experiment parameters, DaskConfig settings
        When: run_hyperparameter_tuning_experiment_with_benchmarks is executed
        Then: OptimizationEvaluationDaskWorkflow is created with correct Dask parameters

        Test type: unit
        """
        # Mock config creator
        mock_creator_instance = MagicMock()
        mock_configs = [Mock()]
        mock_creator_instance.get_experiment_configs.return_value = mock_configs
        mock_creator_class.return_value = mock_creator_instance

        # Mock workflow
        mock_workflow_instance = MagicMock()
        mock_results = (
            {"env": {"policy": ["history"]}},
            pd.DataFrame({"environment": ["env"], "policy": ["policy"]}),
        )
        mock_workflow_instance.optimize_and_evaluate.return_value = mock_results
        mock_workflow_class.return_value = mock_workflow_instance

        api = DaskSimulationsAPI()
        mock_generators = [Mock()]

        results, stats_df = api.run_hyperparameter_tuning_experiment_with_benchmarks(
            generators=mock_generators,
            particles=30,
            num_episodes=10,
            evaluation_episodes=5,
            optimization_n_jobs=8,
            scheduler_address="tcp://localhost:8786",
            cache_size=int(4e9),
            cache_dir_path=temp_cache_dir,
        )

        # Verify workflow was created with DaskConfig parameters
        mock_workflow_class.assert_called_once()
        call_args = mock_workflow_class.call_args[1]

        assert call_args["n_workers"] == 8
        assert call_args["scheduler_address"] == "tcp://localhost:8786"
        assert call_args["cache_size"] == int(4e9)
        assert call_args["evaluation_episodes"] == 5

        # Verify workflow execution
        mock_workflow_instance.optimize_and_evaluate.assert_called_once_with(mock_configs)

        # Verify results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    @patch(
        "POMDPPlanners.simulations.simulation_apis.dask_simulations_api.OptimizationEvaluationDaskWorkflow"
    )
    @patch(
        "POMDPPlanners.simulations.simulation_apis.dask_simulations_api.AllHyperparameterBenchmarksExperimentConfigCreator"
    )
    def test_run_all_hyperparameter_benchmarks(
        self, mock_creator_class, mock_workflow_class, temp_cache_dir
    ):
        """Test run_all_hyperparameter_benchmarks with Dask.

        Purpose: Validates that all hyperparameter benchmarks use DaskWorkflow correctly

        Given: PolicySpaceInfo, experiment parameters, DaskConfig settings
        When: run_all_hyperparameter_benchmarks is executed
        Then: OptimizationEvaluationDaskWorkflow is created with correct Dask parameters

        Test type: unit
        """
        # Mock config creator
        mock_creator_instance = MagicMock()
        mock_configs = [Mock()]
        mock_creator_instance.get_experiment_configs.return_value = mock_configs
        mock_creator_class.return_value = mock_creator_instance

        # Mock workflow
        mock_workflow_instance = MagicMock()
        mock_results = (
            {"env": {"policy": ["history"]}},
            pd.DataFrame({"environment": ["env"], "policy": ["policy"]}),
        )
        mock_workflow_instance.optimize_and_evaluate.return_value = mock_results
        mock_workflow_class.return_value = mock_workflow_instance

        api = DaskSimulationsAPI()
        policy_space_info = PolicySpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.DISCRETE,
        )

        results, stats_df = api.run_all_hyperparameter_benchmarks(
            policy_space_info=policy_space_info,
            particles=30,
            num_episodes=10,
            evaluation_episodes=5,
            optimization_n_jobs=10,
            scheduler_address="tcp://localhost:8786",
            cache_size=int(5e9),
            cache_dir_path=temp_cache_dir,
        )

        # Verify workflow was created with DaskConfig parameters
        mock_workflow_class.assert_called_once()
        call_args = mock_workflow_class.call_args[1]

        assert call_args["n_workers"] == 10
        assert call_args["scheduler_address"] == "tcp://localhost:8786"
        assert call_args["cache_size"] == int(5e9)
        assert call_args["evaluation_episodes"] == 5

        # Verify workflow execution
        mock_workflow_instance.optimize_and_evaluate.assert_called_once_with(mock_configs)

        # Verify results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)


# Note: Integration tests for DaskSimulationsAPI would require an actual Dask cluster
# or local Dask setup, which may not be suitable for unit test environments.
# The tests above use mocking to verify the correct configuration and API usage.
