import random
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import (
    EnvironmentRunParams,
    History,
    NumericalHyperParameter,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
    HyperParamPlannerConfig,
    HyperParameterOptimizationDirection,
    OptimizedPolicyResult,
)
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.simulations.simulation_apis.dask_simulations_api import DaskSimulationsAPI

np.random.seed(42)
random.seed(42)


@pytest.fixture
def temp_cache_dir():
    """Fixture to create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        # Add a small delay to ensure all file handles are released
        time.sleep(0.1)
        # Ensure cleanup happens even if test fails
        try:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


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


@pytest.fixture
def sample_environment_run_params(tiger_environment, sparse_pft_policy):
    """Fixture to create sample environment run parameters using TigerPOMDP and SparsePFT."""
    initial_belief = get_initial_belief(tiger_environment, n_particles=100)
    return [
        EnvironmentRunParams(
            environment=tiger_environment,
            belief=initial_belief,
            policies=[sparse_pft_policy],
            num_episodes=2,
            num_steps=10,
        )
    ]


class TestDaskSimulationsAPI:
    def test_init(self):
        """Test DaskSimulationsAPI initialization.

        Purpose: Validates that DaskSimulationsAPI can be instantiated without errors

        Given: No parameters required for DaskSimulationsAPI constructor
        When: DaskSimulationsAPI instance is created
        Then: Object is properly initialized as DaskSimulationsAPI instance

        Test type: unit
        """
        api = DaskSimulationsAPI()
        assert isinstance(api, DaskSimulationsAPI)

    def test_init_with_parameters(self, temp_cache_dir):
        """Test DaskSimulationsAPI initialization with parameters.

        Purpose: Validates that DaskSimulationsAPI can be instantiated with custom parameters

        Given: Custom cache_dir_path and debug=True
        When: DaskSimulationsAPI instance is created
        Then: Object is properly initialized with custom parameters

        Test type: unit
        """
        api = DaskSimulationsAPI(cache_dir_path=temp_cache_dir, debug=True)
        assert isinstance(api, DaskSimulationsAPI)

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
    def test_run_multiple_environments_and_policies_with_profiling(
        self, mock_simulator_class, temp_cache_dir, sample_environment_run_params
    ):
        """Test run_multiple_environments_and_policies with profiling enabled.

        Purpose: Validates that Dask simulation correctly enables profiling when requested

        Given: TigerPOMDP environment, enable_profiling=True, profiling_output_limit=100
        When: run_multiple_environments_and_policies is executed
        Then: POMDPSimulator is created with profiling parameters

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
            experiment_name="test_dask_profiling",
            enable_profiling=True,
            profiling_output_limit=100,
            cache_dir_path=temp_cache_dir,
        )

        # Verify POMDPSimulator was called with profiling parameters
        call_args = mock_simulator_class.call_args
        assert call_args[1]["enable_profiling"] is True
        assert call_args[1]["profiling_output_limit"] == 100

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

    @patch("POMDPPlanners.simulations.simulation_apis.dask_simulations_api.HyperParameterOptimizer")
    def test_run_hyperparameter_optimization(
        self, mock_optimizer_class, temp_cache_dir, tiger_environment
    ):
        """Test run_hyperparameter_optimization with Dask.

        Purpose: Validates that hyperparameter optimization uses Dask correctly

        Given: Tiger environment, POMCP policy config, DaskConfig parameters
        When: run_hyperparameter_optimization is executed
        Then: HyperParameterOptimizer is created with DaskConfig and optimization runs

        Test type: unit
        """
        # Create hyperparameter configuration
        initial_belief = get_initial_belief(tiger_environment, n_particles=10)
        planner_config = HyperParamPlannerConfig(
            policy_cls=POMCP,
            hyper_parameters=[
                NumericalHyperParameter(0.1, 2.0, "exploration_constant"),
            ],
            constant_parameters={
                "discount_factor": 0.95,
                "name": "TestPOMCP",
                "depth": 5,
            },
        )

        environment_run_params = [
            HyperParameterRunParams(
                environment=tiger_environment,
                belief=initial_belief,
                hyper_param_planner_config=planner_config,
                num_episodes=2,
                num_steps=3,
                n_trials=3,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
            )
        ]

        # Mock optimizer
        mock_optimizer_instance = MagicMock()
        mock_policy = Mock()
        mock_policy.__class__.__name__ = "POMCP"
        mock_results = [
            OptimizedPolicyResult(
                environment=tiger_environment,
                policy=mock_policy,
                chosen_hyper_parameters={"exploration_constant": 1.0},
                num_episodes=2,
                num_steps=3,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
                optimized_metric_values={"average_return": 10.5},
            )
        ]
        mock_optimizer_instance.optimize.return_value = mock_results
        mock_optimizer_class.return_value = mock_optimizer_instance

        api = DaskSimulationsAPI()
        results = api.run_hyperparameter_optimization(
            environment_run_params=environment_run_params,
            experiment_name="Test_Hyperparameter_Optimization",
            n_jobs=4,
            scheduler_address="tcp://localhost:8786",
            cache_size=int(3e9),
            cache_dir_path=temp_cache_dir,
        )

        # Verify HyperParameterOptimizer was created with DaskConfig
        mock_optimizer_class.assert_called_once()
        call_args = mock_optimizer_class.call_args[1]

        task_manager_config = call_args["task_manager_config"]
        assert task_manager_config.n_workers == 4
        assert task_manager_config.scheduler_address == "tcp://localhost:8786"
        assert task_manager_config.cache_size == int(3e9)

        # Verify optimization was called
        mock_optimizer_instance.optimize.assert_called_once_with(environment_run_params)

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], OptimizedPolicyResult)

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
    def test_run_optimize_and_evaluate(
        self, mock_workflow_class, temp_cache_dir, tiger_environment
    ):
        """Test run_optimize_and_evaluate with Dask.

        Purpose: Validates that optimize and evaluate uses DaskWorkflow correctly

        Given: Hyperparameter run configs, DaskConfig settings
        When: run_optimize_and_evaluate is executed
        Then: OptimizationEvaluationDaskWorkflow is created and executes optimization

        Test type: unit
        """
        # Create hyperparameter configuration
        initial_belief = get_initial_belief(tiger_environment, n_particles=10)
        planner_config = HyperParamPlannerConfig(
            policy_cls=POMCP,
            hyper_parameters=[
                NumericalHyperParameter(0.1, 2.0, "exploration_constant"),
            ],
            constant_parameters={
                "discount_factor": 0.95,
                "name": "TestPOMCP",
                "depth": 5,
            },
        )

        configs = [
            HyperParameterRunParams(
                environment=tiger_environment,
                belief=initial_belief,
                hyper_param_planner_config=planner_config,
                num_episodes=2,
                num_steps=3,
                n_trials=3,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
            )
        ]

        # Mock workflow
        mock_workflow_instance = MagicMock()
        mock_results = (
            {"env": {"policy": ["history"]}},
            pd.DataFrame({"environment": ["env"], "policy": ["policy"]}),
        )
        mock_workflow_instance.optimize_and_evaluate.return_value = mock_results
        mock_workflow_class.return_value = mock_workflow_instance

        api = DaskSimulationsAPI()
        results, stats_df = api.run_optimize_and_evaluate(
            configs=configs,
            evaluation_episodes=10,
            optimization_n_jobs=6,
            scheduler_address="tcp://localhost:8786",
            cache_dir_path=temp_cache_dir,
        )

        # Verify workflow was created with correct parameters
        mock_workflow_class.assert_called_once()
        call_args = mock_workflow_class.call_args[1]

        assert call_args["n_workers"] == 6
        assert call_args["scheduler_address"] == "tcp://localhost:8786"
        assert call_args["evaluation_episodes"] == 10

        # Verify workflow execution
        mock_workflow_instance.optimize_and_evaluate.assert_called_once_with(configs)

        # Verify results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    @patch(
        "POMDPPlanners.simulations.simulation_apis.dask_simulations_api.OptimizationEvaluationDaskWorkflow"
    )
    def test_run_optimize_and_evaluate_empty_configs_raises_error(
        self, mock_workflow_class, temp_cache_dir
    ):
        """Test run_optimize_and_evaluate raises error with empty configs.

        Purpose: Validates that empty configs list raises ValueError

        Given: Empty configs list
        When: run_optimize_and_evaluate is executed
        Then: ValueError is raised with appropriate message

        Test type: unit
        """
        api = DaskSimulationsAPI()

        with pytest.raises(ValueError, match="configs list cannot be empty"):
            api.run_optimize_and_evaluate(
                configs=[],
                cache_dir_path=temp_cache_dir,
            )

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
