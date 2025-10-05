import os
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, List, cast
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from POMDPPlanners.core.belief import Belief, get_initial_belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import (
    CategoricalHyperParameter,
    EnvironmentRunParams,
    History,
    NumericalHyperParameter,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParamPlannerConfig,
    HyperParameterFeature,
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
    OptimizedPolicyResult,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.simulations.simulations_api import SimulationsAPI
from POMDPPlanners.simulations.simulator import POMDPSimulator

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


@pytest.fixture
def pomcp_policy(tiger_environment):
    """Fixture to create a POMCP policy for the Tiger environment."""
    return POMCP(
        environment=tiger_environment,
        discount_factor=0.95,
        depth=10,
        exploration_constant=1.0,
        name="test_pomcp",
        n_simulations=100,
    )


@pytest.fixture
def sample_hyperparameter_run_params(tiger_environment, pomcp_policy):
    """Fixture to create sample hyperparameter run parameters for optimization testing."""
    initial_belief = get_initial_belief(tiger_environment, n_particles=100)
    return [
        HyperParameterRunParams(
            environment=tiger_environment,
            belief=initial_belief,
            hyper_param_planner_config=HyperParamPlannerConfig(
                policy_cls=POMCP,
                hyper_parameters=cast(
                    List[HyperParameterFeature],
                    [
                        NumericalHyperParameter(low=0.1, high=2.0, name="exploration_constant"),
                        NumericalHyperParameter(low=50, high=200, name="n_simulations"),
                        NumericalHyperParameter(low=5, high=15, name="depth"),
                    ],
                ),
                constant_parameters={
                    "discount_factor": 0.95,
                    "name": "OptimizedPOMCP",
                },
            ),
            num_episodes=2,  # Small number for testing
            num_steps=5,  # Small number for testing
            n_trials=3,  # Small number for testing
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        )
    ]


@pytest.fixture
def mock_optimized_policy_result(tiger_environment):
    """Fixture to create a mock OptimizedPolicyResult for testing."""
    optimized_policy = POMCP(
        environment=tiger_environment,
        discount_factor=0.95,
        depth=10,
        exploration_constant=1.5,
        name="OptimizedPOMCP",
        n_simulations=150,
    )

    return OptimizedPolicyResult(
        policy=optimized_policy,
        environment=tiger_environment,
        chosen_hyper_parameters={
            "exploration_constant": 1.5,
            "n_simulations": 150,
            "depth": 10,
        },
        num_episodes=2,
        num_steps=5,
        direction=HyperParameterOptimizationDirection.MAXIMIZE,
        parameter_to_optimize="average_return",
    )


class TestSimulationsAPI:
    def test_init(self):
        """Test SimulationsAPI initialization.

        Purpose: Validates that SimulationsAPI can be instantiated without errors

        Given: No parameters required for SimulationsAPI constructor
        When: SimulationsAPI instance is created
        Then: Object is properly initialized as SimulationsAPI instance

        Test type: unit
        """
        api = SimulationsAPI()
        assert isinstance(api, SimulationsAPI)

    def test_run_multiple_environments_and_policies_local_run_success(
        self, temp_cache_dir, sample_environment_run_params
    ):
        """Test successful execution of run_multiple_environments_and_policies_local_run.

        Purpose: Validates that local simulation execution completes successfully and returns expected results structure

        Given: TigerPOMDP environment, SparsePFT policy, 2 episodes, 10 steps, alpha=0.1, confidence_interval_level=0.95, single job execution
        When: run_multiple_environments_and_policies_local_run is executed with valid parameters
        Then: Returns dict with environment-policy results (2 episodes) and stats DataFrame with environment, policy, success_rate, average_listens columns

        Test type: integration
        """
        api = SimulationsAPI()
        results, stats_df = api.run_multiple_environments_and_policies_local_run(
            environment_run_params=sample_environment_run_params,
            alpha=0.1,
            confidence_interval_level=0.95,
            experiment_name="test_experiment",
            debug=True,
            n_jobs=1,  # Use single job for testing
            cache_dir_path=temp_cache_dir,
        )
        assert isinstance(results, dict)
        assert "test_tiger" in results
        assert "test_sparse_pft" in results["test_tiger"]
        assert len(results["test_tiger"]["test_sparse_pft"]) == 2  # num_episodes
        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) == 1
        assert stats_df["environment"].iloc[0] == "test_tiger"
        assert stats_df["policy"].iloc[0] == "test_sparse_pft"
        assert "success_rate" in stats_df.columns
        assert "average_listens" in stats_df.columns
        history = results["test_tiger"]["test_sparse_pft"][0]
        assert isinstance(history, History)
        assert len(history.history) > 0
        assert all(hasattr(step, "state") for step in history.history)
        assert all(hasattr(step, "action") for step in history.history)
        assert all(hasattr(step, "observation") for step in history.history)
        assert all(hasattr(step, "reward") for step in history.history)

    def test_run_multiple_environments_and_policies_local_run_error(
        self, temp_cache_dir, sample_environment_run_params
    ):
        """Test error handling in run_multiple_environments_and_policies_local_run.

        Purpose: Validates error handling for run multiple environments and policies local run

        Given: Invalid inputs or error conditions
        When: Operation is attempted
        Then: Appropriate exception is raised

        Test type: integration
        """
        api = SimulationsAPI()
        # Force an error by mocking the environment's state_transition_model to return None
        sample_environment_run_params[0].environment.state_transition_model = lambda *args: None
        with pytest.raises(Exception):
            api.run_multiple_environments_and_policies_local_run(
                environment_run_params=sample_environment_run_params,
                alpha=0.1,
                confidence_interval_level=0.95,
                cache_dir_path=temp_cache_dir,
            )

    def test_run_multiple_environments_and_policies_local_run_invalid_params(
        self, temp_cache_dir, sample_environment_run_params
    ):
        """Test run_multiple_environments_and_policies_local_run with invalid parameters.

        Purpose: Validates that invalid parameter values raise appropriate ValueError exceptions with descriptive messages

        Given: SimulationsAPI instance and sample environment parameters with invalid alpha (-0.1), confidence_interval_level (1.5), and n_jobs (0) values
        When: run_multiple_environments_and_policies_local_run is called with each invalid parameter
        Then: ValueError is raised with appropriate error messages for alpha, confidence_interval_level, and n_jobs validation

        Test type: integration
        """
        api = SimulationsAPI()
        # Test invalid alpha
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            api.run_multiple_environments_and_policies_local_run(
                environment_run_params=sample_environment_run_params,
                alpha=-0.1,  # Invalid alpha
                confidence_interval_level=0.95,
                cache_dir_path=temp_cache_dir,
            )
        # Test invalid confidence interval
        with pytest.raises(ValueError, match="confidence_interval_level must be between 0 and 1"):
            api.run_multiple_environments_and_policies_local_run(
                environment_run_params=sample_environment_run_params,
                alpha=0.1,
                confidence_interval_level=1.5,  # Invalid confidence interval
                cache_dir_path=temp_cache_dir,
            )
        # Test invalid n_jobs
        with pytest.raises(ValueError, match="n_jobs must be a positive integer or -1"):
            api.run_multiple_environments_and_policies_local_run(
                environment_run_params=sample_environment_run_params,
                alpha=0.1,
                confidence_interval_level=0.95,
                n_jobs=0,  # Invalid n_jobs
                cache_dir_path=temp_cache_dir,
            )

    @patch("POMDPPlanners.simulations.simulations_api.POMDPSimulator")
    def test_run_multiple_environments_and_policies_pbs_run_success(
        self, mock_simulator, temp_cache_dir, sample_environment_run_params
    ):
        """Test successful execution of run_multiple_environments_and_policies_pbs_run.

        Purpose: Validates that PBS cluster simulation execution configures correctly and returns expected results

        Given: TigerPOMDP environment, SparsePFT policy, PBS configuration with queue='normal', and mocked POMDPSimulator
        When: run_multiple_environments_and_policies_pbs_run is executed with valid PBS parameters
        Then: POMDPSimulator is called with correct PBSConfig and returns mocked results structure

        Test type: unit
        """
        # Mock the simulator and its context manager
        mock_simulator_instance = MagicMock()
        mock_simulator_instance.__enter__.return_value = mock_simulator_instance
        mock_simulator_instance.__exit__.return_value = None

        # Mock the comparison results
        mock_results = (
            {"test_tiger": {"test_sparse_pft": ["mock_history_1", "mock_history_2"]}},
            pd.DataFrame(
                {
                    "environment": ["test_tiger"],
                    "policy": ["test_sparse_pft"],
                    "success_rate": [0.95],
                    "average_listens": [2.5],
                }
            ),
        )
        mock_simulator_instance.compare_multiple_environments_policies.return_value = mock_results
        mock_simulator.return_value = mock_simulator_instance

        api = SimulationsAPI()
        results, stats_df = api.run_multiple_environments_and_policies_pbs_run(
            environment_run_params=sample_environment_run_params,
            alpha=0.1,
            confidence_interval_level=0.95,
            queue="normal",
            experiment_name="test_pbs_experiment",
            debug=True,
            n_workers=4,
            cores=2,
            memory="8GB",
            processes=1,
            walltime="02:00:00",
            job_extra=["#PBS -l feature=gpu"],
            cache_dir_path=temp_cache_dir,
        )

        # Verify POMDPSimulator was called with correct configuration
        mock_simulator.assert_called_once()
        call_args = mock_simulator.call_args

        # Check that task_manager_config is a PBSConfig with correct parameters
        task_manager_config = call_args[1]["task_manager_config"]
        from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
            PBSConfig,
        )

        assert isinstance(task_manager_config, PBSConfig)
        assert task_manager_config.queue == "normal"
        assert task_manager_config.n_workers == 4
        assert task_manager_config.cores == 2
        assert task_manager_config.memory == "8GB"
        assert task_manager_config.processes == 1
        assert task_manager_config.walltime == "02:00:00"
        assert task_manager_config.job_extra == ["#PBS -l feature=gpu"]
        # Default dashboard values
        assert task_manager_config.enable_dashboard is True
        assert task_manager_config.dashboard_address == "0.0.0.0"
        assert task_manager_config.dashboard_port == 8787
        assert task_manager_config.dashboard_prefix is None

        # Check other simulator parameters
        assert call_args[1]["cache_dir_path"] == temp_cache_dir
        assert call_args[1]["experiment_name"] == "test_pbs_experiment"
        assert call_args[1]["debug"] is True

        # Verify results - results and stats_df are unpacked from the tuple returned by compare_multiple_environments_policies
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    @patch("POMDPPlanners.simulations.simulations_api.POMDPSimulator")
    def test_run_multiple_environments_and_policies_pbs_run_default_parameters(
        self, mock_simulator, temp_cache_dir, sample_environment_run_params
    ):
        """Test PBS run with default parameters.

        Purpose: Validates that PBS function uses correct default values when optional parameters are not provided

        Given: Only required parameters (environment_run_params, alpha, confidence_interval_level, queue)
        When: run_multiple_environments_and_policies_pbs_run is called with minimal parameters
        Then: PBSConfig is created with correct default values (n_workers=4, cores=1, memory="4GB", etc.)

        Test type: unit
        """
        # Mock the simulator and its context manager
        mock_simulator_instance = MagicMock()
        mock_simulator_instance.__enter__.return_value = mock_simulator_instance
        mock_simulator_instance.__exit__.return_value = None

        mock_results = ({"test_tiger": {"test_sparse_pft": []}}, pd.DataFrame())
        mock_simulator_instance.compare_multiple_environments_policies.return_value = mock_results
        mock_simulator.return_value = mock_simulator_instance

        api = SimulationsAPI()
        results, stats_df = api.run_multiple_environments_and_policies_pbs_run(
            environment_run_params=sample_environment_run_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            queue="default_queue",
        )

        # Verify default parameters
        call_args = mock_simulator.call_args
        task_manager_config = call_args[1]["task_manager_config"]

        assert task_manager_config.queue == "default_queue"
        assert task_manager_config.n_workers == 4  # Default
        assert task_manager_config.cores == 1  # Default
        assert task_manager_config.memory == "4GB"  # Default
        assert task_manager_config.processes == 1  # Default
        assert task_manager_config.walltime == "01:00:00"  # Default
        assert task_manager_config.job_extra is None  # Default
        assert task_manager_config.clear_cache_on_start is False  # Default
        # Default dashboard values
        assert task_manager_config.enable_dashboard is True  # Default
        assert task_manager_config.dashboard_address == "0.0.0.0"  # Default
        assert task_manager_config.dashboard_port == 8787  # Default
        assert task_manager_config.dashboard_prefix is None  # Default

        assert call_args[1]["experiment_name"] == "POMDP_Planning_Comparison"  # Default
        assert call_args[1]["debug"] is False  # Default
        assert call_args[1]["enable_profiling"] is False  # Default

    def test_run_multiple_environments_and_policies_pbs_run_invalid_queue(
        self, temp_cache_dir, sample_environment_run_params
    ):
        """Test PBS run with invalid queue parameter.

        Purpose: Validates that PBS function properly validates required queue parameter

        Given: Missing or invalid queue parameter
        When: run_multiple_environments_and_policies_pbs_run is called
        Then: Appropriate error is raised for missing required parameter

        Test type: unit
        """
        api = SimulationsAPI()

        # Test with empty queue
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'queue'"):
            api.run_multiple_environments_and_policies_pbs_run(  # type: ignore[call-arg]
                environment_run_params=sample_environment_run_params,
                alpha=0.05,
                confidence_interval_level=0.95,
                # Missing queue parameter
            )

    @patch("POMDPPlanners.simulations.simulations_api.POMDPSimulator")
    def test_run_multiple_environments_and_policies_pbs_run_custom_job_extra(
        self, mock_simulator, temp_cache_dir, sample_environment_run_params
    ):
        """Test PBS run with custom job_extra directives.

        Purpose: Validates that PBS function correctly handles custom job directives

        Given: Custom PBS job directives including GPU features and email notifications
        When: run_multiple_environments_and_policies_pbs_run is called with job_extra parameter
        Then: PBSConfig includes all specified job_extra directives

        Test type: unit
        """
        # Mock the simulator
        mock_simulator_instance = MagicMock()
        mock_simulator_instance.__enter__.return_value = mock_simulator_instance
        mock_simulator_instance.__exit__.return_value = None

        mock_results = ({"test_tiger": {"test_sparse_pft": []}}, pd.DataFrame())
        mock_simulator_instance.compare_multiple_environments_policies.return_value = mock_results
        mock_simulator.return_value = mock_simulator_instance

        custom_job_extra = [
            "#PBS -l feature=gpu",
            "#PBS -m ae",
            "#PBS -M user@example.com",
        ]

        api = SimulationsAPI()
        results, stats_df = api.run_multiple_environments_and_policies_pbs_run(
            environment_run_params=sample_environment_run_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            queue="gpu_queue",
            job_extra=custom_job_extra,
            cache_dir_path=temp_cache_dir,
        )

        # Verify job_extra directives are properly passed
        call_args = mock_simulator.call_args
        task_manager_config = call_args[1]["task_manager_config"]

        assert task_manager_config.job_extra == custom_job_extra
        assert "#PBS -l feature=gpu" in task_manager_config.job_extra
        assert "#PBS -m ae" in task_manager_config.job_extra
        assert "#PBS -M user@example.com" in task_manager_config.job_extra

    @patch("POMDPPlanners.simulations.simulations_api.POMDPSimulator")
    def test_run_multiple_environments_and_policies_pbs_run_custom_dashboard(
        self, mock_simulator, temp_cache_dir, sample_environment_run_params
    ):
        """Test PBS run with custom dashboard parameters.

        Purpose: Validates that PBS function correctly handles custom dashboard configuration

        Given: Custom dashboard parameters including enabled dashboard, custom port, address, and prefix
        When: run_multiple_environments_and_policies_pbs_run is called with dashboard parameters
        Then: PBSConfig includes all specified dashboard settings

        Test type: unit
        """
        # Mock the simulator
        mock_simulator_instance = MagicMock()
        mock_simulator_instance.__enter__.return_value = mock_simulator_instance
        mock_simulator_instance.__exit__.return_value = None

        mock_results = ({"test_tiger": {"test_sparse_pft": []}}, pd.DataFrame())
        mock_simulator_instance.compare_multiple_environments_policies.return_value = mock_results
        mock_simulator.return_value = mock_simulator_instance

        api = SimulationsAPI()
        results, stats_df = api.run_multiple_environments_and_policies_pbs_run(
            environment_run_params=sample_environment_run_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            queue="gpu_queue",
            enable_dashboard=True,
            dashboard_address="192.168.1.100",
            dashboard_port=9999,
            dashboard_prefix="/cluster-monitor",
            cache_dir_path=temp_cache_dir,
        )

        # Verify dashboard parameters are properly passed
        call_args = mock_simulator.call_args
        task_manager_config = call_args[1]["task_manager_config"]

        assert task_manager_config.enable_dashboard is True
        assert task_manager_config.dashboard_address == "192.168.1.100"
        assert task_manager_config.dashboard_port == 9999
        assert task_manager_config.dashboard_prefix == "/cluster-monitor"

    @patch("POMDPPlanners.simulations.simulations_api.POMDPSimulator")
    def test_run_multiple_environments_and_policies_pbs_run_dashboard_disabled(
        self, mock_simulator, temp_cache_dir, sample_environment_run_params
    ):
        """Test PBS run with dashboard disabled.

        Purpose: Validates that PBS function correctly handles disabled dashboard configuration

        Given: Dashboard disabled (enable_dashboard=False) with custom port and address
        When: run_multiple_environments_and_policies_pbs_run is called with dashboard disabled
        Then: PBSConfig shows dashboard disabled but other dashboard parameters are stored

        Test type: unit
        """
        # Mock the simulator
        mock_simulator_instance = MagicMock()
        mock_simulator_instance.__enter__.return_value = mock_simulator_instance
        mock_simulator_instance.__exit__.return_value = None

        mock_results = ({"test_tiger": {"test_sparse_pft": []}}, pd.DataFrame())
        mock_simulator_instance.compare_multiple_environments_policies.return_value = mock_results
        mock_simulator.return_value = mock_simulator_instance

        api = SimulationsAPI()
        results, stats_df = api.run_multiple_environments_and_policies_pbs_run(
            environment_run_params=sample_environment_run_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            queue="batch_queue",
            enable_dashboard=False,
            dashboard_port=8888,
            dashboard_address="127.0.0.1",
            cache_dir_path=temp_cache_dir,
        )

        # Verify dashboard is disabled but parameters are stored
        call_args = mock_simulator.call_args
        task_manager_config = call_args[1]["task_manager_config"]

        assert task_manager_config.enable_dashboard is False
        assert task_manager_config.dashboard_port == 8888  # Should still store the parameter
        assert (
            task_manager_config.dashboard_address == "127.0.0.1"
        )  # Should still store the parameter

    @patch("POMDPPlanners.simulations.simulations_api.HyperParameterOptimizer")
    def test_run_hyperparameter_optimization_success(
        self,
        mock_optimizer_class,
        temp_cache_dir,
        sample_hyperparameter_run_params,
        mock_optimized_policy_result,
    ):
        """Test successful execution of run_hyperparameter_optimization.

        Purpose: Validates that hyperparameter optimization completes successfully and returns expected results structure

        Given: TigerPOMDP environment, POMCP policy class, hyperparameter ranges, and mocked HyperParameterOptimizer
        When: run_hyperparameter_optimization is executed with valid parameters
        Then: Returns list of OptimizedPolicyResult objects with optimized policies and their parameters

        Test type: unit
        """
        # Mock the optimizer instance and its methods
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.optimize.return_value = [mock_optimized_policy_result]
        mock_optimizer_instance.cleanup.return_value = None
        mock_optimizer_class.return_value = mock_optimizer_instance

        api = SimulationsAPI()
        results = api.run_hyperparameter_optimization(
            environment_run_params=sample_hyperparameter_run_params,
            experiment_name="test_hyperparameter_optimization",
            n_jobs=1,  # Use single job for testing
            cache_dir_path=temp_cache_dir,
            debug=True,
        )

        # Verify HyperParameterOptimizer was called with correct parameters
        mock_optimizer_class.assert_called_once()
        call_args = mock_optimizer_class.call_args

        assert call_args[1]["cache_dir_path"] == temp_cache_dir
        assert call_args[1]["experiment_name"] == "test_hyperparameter_optimization"
        assert call_args[1]["n_jobs"] == 1
        assert call_args[1]["confidence_interval_level"] == 0.95
        assert call_args[1]["alpha"] == 0.05
        assert call_args[1]["use_queue_logger"] is False

        # Verify optimizer.optimize was called with correct parameters
        mock_optimizer_instance.optimize.assert_called_once_with(sample_hyperparameter_run_params)

        # Verify results
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], OptimizedPolicyResult)
        assert results[0].policy.name == "OptimizedPOMCP"
        assert results[0].chosen_hyper_parameters["exploration_constant"] == 1.5
        assert results[0].chosen_hyper_parameters["n_simulations"] == 150
        assert results[0].chosen_hyper_parameters["depth"] == 10

        # Verify cleanup was called
        mock_optimizer_instance.cleanup.assert_called_once()

    @patch("POMDPPlanners.simulations.simulations_api.HyperParameterOptimizer")
    def test_run_hyperparameter_optimization_default_parameters(
        self,
        mock_optimizer_class,
        temp_cache_dir,
        sample_hyperparameter_run_params,
        mock_optimized_policy_result,
    ):
        """Test hyperparameter optimization with default parameters.

        Purpose: Validates that hyperparameter optimization uses correct default values when optional parameters are not provided

        Given: Only required parameters (environment_run_params)
        When: run_hyperparameter_optimization is called with minimal parameters
        Then: HyperParameterOptimizer is created with correct default values

        Test type: unit
        """
        # Mock the optimizer instance
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.optimize.return_value = [mock_optimized_policy_result]
        mock_optimizer_instance.cleanup.return_value = None
        mock_optimizer_class.return_value = mock_optimizer_instance

        api = SimulationsAPI()
        results = api.run_hyperparameter_optimization(
            environment_run_params=sample_hyperparameter_run_params,
        )

        # Verify default parameters
        call_args = mock_optimizer_class.call_args

        assert call_args[1]["experiment_name"] == "POMDP_Hyperparameter_Optimization"  # Default
        assert call_args[1]["n_jobs"] == -1  # Default
        assert call_args[1]["confidence_interval_level"] == 0.95  # Default
        assert call_args[1]["alpha"] == 0.05  # Default
        assert call_args[1]["use_queue_logger"] is False  # Default

        # Verify cache directory was created with default name
        expected_cache_dir = Path("./hyperparameter_optimization_results")
        assert call_args[1]["cache_dir_path"] == expected_cache_dir

        assert isinstance(results, list)
        assert len(results) == 1

    @patch("POMDPPlanners.simulations.simulations_api.HyperParameterOptimizer")
    def test_run_hyperparameter_optimization_custom_cache_dir(
        self,
        mock_optimizer_class,
        temp_cache_dir,
        sample_hyperparameter_run_params,
        mock_optimized_policy_result,
    ):
        """Test hyperparameter optimization with custom cache directory.

        Purpose: Validates that hyperparameter optimization correctly handles custom cache directory path

        Given: Custom cache directory path
        When: run_hyperparameter_optimization is called with custom cache_dir_path
        Then: HyperParameterOptimizer uses the specified cache directory

        Test type: unit
        """
        # Mock the optimizer instance
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.optimize.return_value = [mock_optimized_policy_result]
        mock_optimizer_instance.cleanup.return_value = None
        mock_optimizer_class.return_value = mock_optimizer_instance

        custom_cache_dir = temp_cache_dir / "custom_optimization_cache"
        api = SimulationsAPI()
        results = api.run_hyperparameter_optimization(
            environment_run_params=sample_hyperparameter_run_params,
            cache_dir_path=custom_cache_dir,
        )

        # Verify custom cache directory was used
        call_args = mock_optimizer_class.call_args
        assert call_args[1]["cache_dir_path"] == custom_cache_dir

        assert isinstance(results, list)
        assert len(results) == 1

    @patch("POMDPPlanners.simulations.simulations_api.HyperParameterOptimizer")
    def test_run_hyperparameter_optimization_multiple_configurations(
        self,
        mock_optimizer_class,
        temp_cache_dir,
        sample_hyperparameter_run_params,
        mock_optimized_policy_result,
    ):
        """Test hyperparameter optimization with multiple configurations.

        Purpose: Validates that hyperparameter optimization correctly handles multiple environment-policy configurations

        Given: Multiple HyperParameterRunParams configurations
        When: run_hyperparameter_optimization is called with multiple configurations
        Then: Returns list of OptimizedPolicyResult objects for each configuration

        Test type: unit
        """
        # Create multiple configurations
        tiger_env = TigerPOMDP(discount_factor=0.95, name="test_tiger")
        initial_belief = get_initial_belief(tiger_env, n_particles=100)

        multiple_configs = [
            HyperParameterRunParams(
                environment=tiger_env,
                belief=initial_belief,
                hyper_param_planner_config=HyperParamPlannerConfig(
                    policy_cls=POMCP,
                    hyper_parameters=cast(
                        List[HyperParameterFeature],
                        [
                            NumericalHyperParameter(low=0.1, high=2.0, name="exploration_constant"),
                            NumericalHyperParameter(low=50, high=200, name="n_simulations"),
                        ],
                    ),
                    constant_parameters={"discount_factor": 0.95, "name": "POMCP_Config1"},
                ),
                num_episodes=2,
                num_steps=5,
                n_trials=3,
                direction=HyperParameterOptimizationDirection.MAXIMIZE,
                parameter_to_optimize="average_return",
            ),
            HyperParameterRunParams(
                environment=tiger_env,
                belief=initial_belief,
                hyper_param_planner_config=HyperParamPlannerConfig(
                    policy_cls=POMCP,
                    hyper_parameters=cast(
                        List[HyperParameterFeature],
                        [
                            NumericalHyperParameter(low=5, high=15, name="depth"),
                            CategoricalHyperParameter(choices=["tpe", "random"], name="algorithm"),
                        ],
                    ),
                    constant_parameters={"discount_factor": 0.95, "name": "POMCP_Config2"},
                ),
                num_episodes=2,
                num_steps=5,
                n_trials=3,
                direction=HyperParameterOptimizationDirection.MINIMIZE,
                parameter_to_optimize="total_cost",
            ),
        ]

        # Create mock results for multiple configurations
        mock_result_1 = OptimizedPolicyResult(
            policy=POMCP(
                environment=tiger_env,
                discount_factor=0.95,
                depth=10,
                exploration_constant=1.0,
                name="POMCP_Config1",
                n_simulations=100,
            ),
            environment=tiger_env,
            chosen_hyper_parameters={"exploration_constant": 1.0, "n_simulations": 100},
            num_episodes=2,
            num_steps=5,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
        )

        mock_result_2 = OptimizedPolicyResult(
            policy=POMCP(
                environment=tiger_env,
                discount_factor=0.95,
                depth=12,
                exploration_constant=1.0,
                name="POMCP_Config2",
                n_simulations=100,
            ),
            environment=tiger_env,
            chosen_hyper_parameters={"depth": 12, "algorithm": "tpe"},
            num_episodes=2,
            num_steps=5,
            direction=HyperParameterOptimizationDirection.MINIMIZE,
            parameter_to_optimize="total_cost",
        )

        # Mock the optimizer instance
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.optimize.return_value = [mock_result_1, mock_result_2]
        mock_optimizer_instance.cleanup.return_value = None
        mock_optimizer_class.return_value = mock_optimizer_instance

        api = SimulationsAPI()
        results = api.run_hyperparameter_optimization(
            environment_run_params=multiple_configs,
            cache_dir_path=temp_cache_dir,
        )

        # Verify optimizer was called with multiple configurations
        mock_optimizer_instance.optimize.assert_called_once_with(multiple_configs)

        # Verify results
        assert isinstance(results, list)
        assert len(results) == 2
        assert isinstance(results[0], OptimizedPolicyResult)
        assert isinstance(results[1], OptimizedPolicyResult)
        assert results[0].policy.name == "POMCP_Config1"
        assert results[1].policy.name == "POMCP_Config2"

    @patch("POMDPPlanners.simulations.simulations_api.HyperParameterOptimizer")
    def test_run_hyperparameter_optimization_error_handling(
        self, mock_optimizer_class, temp_cache_dir, sample_hyperparameter_run_params
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

        api = SimulationsAPI()

        with pytest.raises(
            RuntimeError,
            match="Hyperparameter optimization failed: Optimization failed",
        ):
            api.run_hyperparameter_optimization(
                environment_run_params=sample_hyperparameter_run_params,
                cache_dir_path=temp_cache_dir,
            )

        # Verify cleanup was still called even after error
        mock_optimizer_instance.cleanup.assert_called_once()

    @patch("POMDPPlanners.simulations.simulations_api.HyperParameterOptimizer")
    def test_run_hyperparameter_optimization_cleanup_error(
        self,
        mock_optimizer_class,
        temp_cache_dir,
        sample_hyperparameter_run_params,
        mock_optimized_policy_result,
    ):
        """Test hyperparameter optimization with cleanup error handling.

        Purpose: Validates that hyperparameter optimization handles cleanup errors gracefully

        Given: HyperParameterOptimizer that succeeds but cleanup raises an exception
        When: run_hyperparameter_optimization is executed
        Then: Results are returned successfully despite cleanup error (error is logged as warning)

        Test type: unit
        """
        # Mock the optimizer instance
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.optimize.return_value = [mock_optimized_policy_result]
        mock_optimizer_instance.cleanup.side_effect = Exception("Cleanup failed")
        mock_optimizer_class.return_value = mock_optimizer_instance

        api = SimulationsAPI()

        # Should not raise an exception despite cleanup error
        results = api.run_hyperparameter_optimization(
            environment_run_params=sample_hyperparameter_run_params,
            cache_dir_path=temp_cache_dir,
        )

        # Verify results are still returned
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], OptimizedPolicyResult)

        # Verify cleanup was attempted
        mock_optimizer_instance.cleanup.assert_called_once()

    def test_run_hyperparameter_optimization_invalid_parameters(
        self, temp_cache_dir, sample_hyperparameter_run_params
    ):
        """Test run_hyperparameter_optimization with invalid parameters.

        Purpose: Validates that hyperparameter optimization properly validates parameter values

        Given: Invalid parameter values (negative alpha, invalid confidence interval, etc.)
        When: run_hyperparameter_optimization is called with invalid parameters
        Then: Appropriate ValueError exceptions are raised

        Test type: unit
        """
        api = SimulationsAPI()

        # Test invalid alpha - this should be caught by the HyperParameterOptimizer
        with pytest.raises(RuntimeError, match="Hyperparameter optimization failed"):
            api.run_hyperparameter_optimization(
                environment_run_params=sample_hyperparameter_run_params,
                alpha=-0.1,  # Invalid alpha
                cache_dir_path=temp_cache_dir,
            )

        # Test invalid confidence interval - this should be caught by the HyperParameterOptimizer
        with pytest.raises(RuntimeError, match="Hyperparameter optimization failed"):
            api.run_hyperparameter_optimization(
                environment_run_params=sample_hyperparameter_run_params,
                confidence_interval_level=1.5,  # Invalid confidence interval
                cache_dir_path=temp_cache_dir,
            )

        # Test invalid n_jobs - this should be caught by the HyperParameterOptimizer
        with pytest.raises(RuntimeError, match="Hyperparameter optimization failed"):
            api.run_hyperparameter_optimization(
                environment_run_params=sample_hyperparameter_run_params,
                n_jobs=0,  # Invalid n_jobs
                cache_dir_path=temp_cache_dir,
            )

    def test_run_hyperparameter_optimization_empty_configurations(self, temp_cache_dir):
        """Test run_hyperparameter_optimization with empty configurations list.

        Purpose: Validates that hyperparameter optimization handles empty configurations gracefully

        Given: Empty list of HyperParameterRunParams
        When: run_hyperparameter_optimization is called with empty configurations
        Then: Returns empty list without error

        Test type: unit
        """
        api = SimulationsAPI()

        results = api.run_hyperparameter_optimization(
            environment_run_params=[],  # Empty list
            cache_dir_path=temp_cache_dir,
        )

        assert isinstance(results, list)
        assert len(results) == 0

    @patch("POMDPPlanners.simulations.simulations_api.OptimizationEvaluationLocalWorkflow")
    def test_run_optimize_and_evaluate_local_success(
        self,
        mock_workflow_class,
        temp_cache_dir,
        sample_hyperparameter_run_params,
    ):
        """Test successful execution of run_optimize_and_evaluate_local.

        Purpose: Validates that optimize and evaluate workflow completes successfully with local execution

        Given: TigerPOMDP environment, POMCP policy class, hyperparameter configs, and mocked workflow
        When: run_optimize_and_evaluate_local is executed with valid parameters
        Then: OptimizationEvaluationLocalWorkflow is called with correct config and returns expected results

        Test type: unit
        """
        # Mock the workflow instance and its methods
        mock_workflow_instance = MagicMock()
        mock_results = (
            {"test_tiger": {"OptimizedPOMCP": ["mock_history_1", "mock_history_2"]}},
            pd.DataFrame(
                {
                    "environment": ["test_tiger"],
                    "policy": ["OptimizedPOMCP"],
                    "average_return": [10.5],
                }
            ),
        )
        mock_workflow_instance.optimize_and_evaluate.return_value = mock_results
        mock_workflow_class.return_value = mock_workflow_instance

        api = SimulationsAPI()
        results, stats_df = api.run_optimize_and_evaluate_local(
            configs=sample_hyperparameter_run_params,
            evaluation_episodes=100,
            evaluation_steps=100,
            evaluation_n_jobs=1,
            optimization_n_jobs=2,
            confidence_interval_level=0.95,
            alpha=0.05,
            cache_dir_path=temp_cache_dir,
            experiment_name="test_optimize_evaluate",
            debug=True,
            cache_visualizations=True,
        )

        # Verify OptimizationEvaluationLocalWorkflow was called with correct parameters
        mock_workflow_class.assert_called_once()
        call_args = mock_workflow_class.call_args

        assert call_args[1]["cache_dir"] == temp_cache_dir
        assert call_args[1]["experiment_name"] == "test_optimize_evaluate"
        assert call_args[1]["optimization_n_jobs"] == 2
        assert call_args[1]["evaluation_episodes"] == 100
        assert call_args[1]["evaluation_steps"] == 100
        assert call_args[1]["evaluation_n_jobs"] == 1
        assert call_args[1]["confidence_interval_level"] == 0.95
        assert call_args[1]["alpha"] == 0.05
        assert call_args[1]["debug"] is True
        assert call_args[1]["verbose"] is True
        assert call_args[1]["cache_visualizations"] is True

        # Verify optimize_and_evaluate was called with correct configs
        mock_workflow_instance.optimize_and_evaluate.assert_called_once_with(
            sample_hyperparameter_run_params
        )

        # Verify results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    @patch("POMDPPlanners.simulations.simulations_api.OptimizationEvaluationLocalWorkflow")
    def test_run_optimize_and_evaluate_local_default_parameters(
        self,
        mock_workflow_class,
        sample_hyperparameter_run_params,
    ):
        """Test run_optimize_and_evaluate_local with default parameters.

        Purpose: Validates that optimize and evaluate uses correct default values when optional parameters are not provided

        Given: Only required parameters (configs)
        When: run_optimize_and_evaluate_local is called with minimal parameters
        Then: OptimizationEvaluationLocalWorkflow is created with correct default values

        Test type: unit
        """
        # Mock the workflow instance
        mock_workflow_instance = MagicMock()
        mock_results = ({}, pd.DataFrame())
        mock_workflow_instance.optimize_and_evaluate.return_value = mock_results
        mock_workflow_class.return_value = mock_workflow_instance

        api = SimulationsAPI()
        results, stats_df = api.run_optimize_and_evaluate_local(
            configs=sample_hyperparameter_run_params,
        )

        # Verify default parameters
        call_args = mock_workflow_class.call_args

        assert call_args[1]["experiment_name"] == "Optimize_And_Evaluate"  # Default
        assert call_args[1]["optimization_n_jobs"] == -1  # Default
        assert call_args[1]["evaluation_episodes"] == 100  # Default
        assert call_args[1]["evaluation_steps"] == 100  # Default
        assert call_args[1]["evaluation_n_jobs"] == 1  # Default
        assert call_args[1]["confidence_interval_level"] == 0.95  # Default
        assert call_args[1]["alpha"] == 0.05  # Default
        assert call_args[1]["debug"] is False  # Default
        assert call_args[1]["cache_visualizations"] is True  # Default

        # Verify cache directory was created with default name
        expected_cache_dir = Path("./optimize_and_evaluate_results")
        assert call_args[1]["cache_dir"] == expected_cache_dir

    @patch("POMDPPlanners.simulations.simulations_api.OptimizationEvaluationLocalWorkflow")
    def test_run_optimize_and_evaluate_local_empty_configs(
        self,
        mock_workflow_class,
        temp_cache_dir,
    ):
        """Test run_optimize_and_evaluate_local with empty configurations list.

        Purpose: Validates that optimize and evaluate handles empty configurations with appropriate error

        Given: Empty list of HyperParameterRunParams
        When: run_optimize_and_evaluate_local is called with empty configurations
        Then: ValueError is raised with descriptive error message

        Test type: unit
        """
        api = SimulationsAPI()

        with pytest.raises(ValueError, match="configs list cannot be empty"):
            api.run_optimize_and_evaluate_local(
                configs=[],  # Empty list
                cache_dir_path=temp_cache_dir,
            )

    @patch("POMDPPlanners.simulations.simulations_api.OptimizationEvaluationPBSWorkflow")
    def test_run_optimize_and_evaluate_pbs_success(
        self,
        mock_workflow_class,
        temp_cache_dir,
        sample_hyperparameter_run_params,
    ):
        """Test successful execution of run_optimize_and_evaluate_pbs.

        Purpose: Validates that optimize and evaluate workflow completes successfully with PBS cluster execution

        Given: TigerPOMDP environment, POMCP policy class, hyperparameter configs, PBS settings, and mocked workflow
        When: run_optimize_and_evaluate_pbs is executed with valid parameters
        Then: OptimizationEvaluationPBSWorkflow is called with correct config and returns expected results

        Test type: unit
        """
        # Mock the workflow instance and its methods
        mock_workflow_instance = MagicMock()
        mock_results = (
            {"test_tiger": {"OptimizedPOMCP": ["mock_history_1", "mock_history_2"]}},
            pd.DataFrame(
                {
                    "environment": ["test_tiger"],
                    "policy": ["OptimizedPOMCP"],
                    "average_return": [10.5],
                }
            ),
        )
        mock_workflow_instance.optimize_and_evaluate.return_value = mock_results
        mock_workflow_class.return_value = mock_workflow_instance

        api = SimulationsAPI()
        results, stats_df = api.run_optimize_and_evaluate_pbs(
            configs=sample_hyperparameter_run_params,
            queue="test_queue",
            evaluation_episodes=100,
            evaluation_steps=100,
            evaluation_n_jobs=1,
            n_workers=4,
            cores=2,
            memory="8GB",
            processes=1,
            walltime="02:00:00",
            job_extra=["#PBS -l feature=gpu"],
            confidence_interval_level=0.95,
            alpha=0.05,
            cache_dir_path=temp_cache_dir,
            experiment_name="test_optimize_evaluate_pbs",
            debug=True,
            cache_visualizations=True,
        )

        # Verify OptimizationEvaluationPBSWorkflow was called with correct parameters
        mock_workflow_class.assert_called_once()
        call_args = mock_workflow_class.call_args

        assert call_args[1]["cache_dir"] == temp_cache_dir
        assert call_args[1]["experiment_name"] == "test_optimize_evaluate_pbs"
        assert call_args[1]["queue"] == "test_queue"
        assert call_args[1]["n_workers"] == 4
        assert call_args[1]["cores"] == 2
        assert call_args[1]["memory"] == "8GB"
        assert call_args[1]["processes"] == 1
        assert call_args[1]["walltime"] == "02:00:00"
        assert call_args[1]["job_extra"] == ["#PBS -l feature=gpu"]
        assert call_args[1]["evaluation_episodes"] == 100
        assert call_args[1]["evaluation_steps"] == 100
        assert call_args[1]["evaluation_n_jobs"] == 1
        assert call_args[1]["confidence_interval_level"] == 0.95
        assert call_args[1]["alpha"] == 0.05
        assert call_args[1]["debug"] is True
        assert call_args[1]["verbose"] is True
        assert call_args[1]["cache_visualizations"] is True

        # Verify optimize_and_evaluate was called with correct configs
        mock_workflow_instance.optimize_and_evaluate.assert_called_once_with(
            sample_hyperparameter_run_params
        )

        # Verify results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    @patch("POMDPPlanners.simulations.simulations_api.OptimizationEvaluationPBSWorkflow")
    def test_run_optimize_and_evaluate_pbs_default_parameters(
        self,
        mock_workflow_class,
        sample_hyperparameter_run_params,
    ):
        """Test run_optimize_and_evaluate_pbs with default parameters.

        Purpose: Validates that optimize and evaluate PBS uses correct default values when optional parameters are not provided

        Given: Only required parameters (configs, queue)
        When: run_optimize_and_evaluate_pbs is called with minimal parameters
        Then: OptimizationEvaluationPBSWorkflow is created with correct default values

        Test type: unit
        """
        # Mock the workflow instance
        mock_workflow_instance = MagicMock()
        mock_results = ({}, pd.DataFrame())
        mock_workflow_instance.optimize_and_evaluate.return_value = mock_results
        mock_workflow_class.return_value = mock_workflow_instance

        api = SimulationsAPI()
        results, stats_df = api.run_optimize_and_evaluate_pbs(
            configs=sample_hyperparameter_run_params,
            queue="default_queue",
        )

        # Verify default parameters
        call_args = mock_workflow_class.call_args

        assert call_args[1]["experiment_name"] == "Optimize_And_Evaluate_PBS"  # Default
        assert call_args[1]["queue"] == "default_queue"
        assert call_args[1]["n_workers"] == 4  # Default
        assert call_args[1]["cores"] == 1  # Default
        assert call_args[1]["memory"] == "4GB"  # Default
        assert call_args[1]["processes"] == 1  # Default
        assert call_args[1]["walltime"] == "03:00:00"  # Default
        assert call_args[1]["job_extra"] is None  # Default
        assert call_args[1]["evaluation_episodes"] == 100  # Default
        assert call_args[1]["evaluation_steps"] == 100  # Default
        assert call_args[1]["evaluation_n_jobs"] == 1  # Default
        assert call_args[1]["confidence_interval_level"] == 0.95  # Default
        assert call_args[1]["alpha"] == 0.05  # Default
        assert call_args[1]["debug"] is False  # Default
        assert call_args[1]["cache_visualizations"] is True  # Default

        # Verify cache directory was created with default name
        expected_cache_dir = Path("./optimize_and_evaluate_pbs_results")
        assert call_args[1]["cache_dir"] == expected_cache_dir

    @patch("POMDPPlanners.simulations.simulations_api.OptimizationEvaluationPBSWorkflow")
    def test_run_optimize_and_evaluate_pbs_empty_configs(
        self,
        mock_workflow_class,
        temp_cache_dir,
    ):
        """Test run_optimize_and_evaluate_pbs with empty configurations list.

        Purpose: Validates that optimize and evaluate PBS handles empty configurations with appropriate error

        Given: Empty list of HyperParameterRunParams
        When: run_optimize_and_evaluate_pbs is called with empty configurations
        Then: ValueError is raised with descriptive error message

        Test type: unit
        """
        api = SimulationsAPI()

        with pytest.raises(ValueError, match="configs list cannot be empty"):
            api.run_optimize_and_evaluate_pbs(
                configs=[],  # Empty list
                queue="test_queue",
                cache_dir_path=temp_cache_dir,
            )

    @patch("POMDPPlanners.simulations.simulations_api.OptimizationEvaluationLocalWorkflow")
    def test_run_hyperparameter_tuning_comprehensive_benchmark_local_success(
        self,
        mock_workflow_class,
        temp_cache_dir,
    ):
        """Test successful execution of run_hyperparameter_tuning_comprehensive_benchmark_local.

        Purpose: Validates that comprehensive benchmark with local execution completes successfully

        Given: Mocked workflow and generators
        When: run_hyperparameter_tuning_comprehensive_benchmark_local is executed
        Then: OptimizationEvaluationLocalWorkflow is called with correct config and returns expected results

        Test type: unit
        """
        # Mock the workflow instance and its methods
        mock_workflow_instance = MagicMock()
        mock_results = (
            {"test_tiger": {"OptimizedPOMCP": ["mock_history_1", "mock_history_2"]}},
            pd.DataFrame(
                {
                    "environment": ["test_tiger"],
                    "policy": ["OptimizedPOMCP"],
                    "average_return": [10.5],
                }
            ),
        )
        mock_workflow_instance.optimize_and_evaluate.return_value = mock_results
        mock_workflow_class.return_value = mock_workflow_instance

        # Mock generators
        mock_generators = [Mock(), Mock()]

        api = SimulationsAPI()
        results, stats_df = api.run_hyperparameter_tuning_experiment_with_benchmarks_local(
            generators=mock_generators,
            particles=50,
            num_episodes=15,
            num_steps=25,
            n_trials=200,
            evaluation_episodes=5,
            evaluation_steps=10,
            evaluation_n_jobs=2,
            optimization_n_jobs=4,
            is_risk_averse=False,
            confidence_interval_level=0.95,
            alpha=0.05,
            cache_dir_path=temp_cache_dir,
            experiment_name="test_comprehensive_benchmark",
            debug=True,
            cache_visualizations=True,
        )

        # Verify OptimizationEvaluationLocalWorkflow was called with correct parameters
        mock_workflow_class.assert_called_once()
        call_args = mock_workflow_class.call_args

        assert call_args[1]["cache_dir"] == temp_cache_dir
        assert call_args[1]["experiment_name"] == "test_comprehensive_benchmark"
        assert call_args[1]["optimization_n_jobs"] == 4
        assert call_args[1]["evaluation_episodes"] == 5
        assert call_args[1]["evaluation_steps"] == 10
        assert call_args[1]["evaluation_n_jobs"] == 2
        assert call_args[1]["confidence_interval_level"] == 0.95
        assert call_args[1]["alpha"] == 0.05
        assert call_args[1]["debug"] is True
        assert call_args[1]["verbose"] is True
        assert call_args[1]["cache_visualizations"] is True

        # Verify optimize_and_evaluate was called with correct configs
        mock_workflow_instance.optimize_and_evaluate.assert_called_once()

        # Verify results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    @patch("POMDPPlanners.simulations.simulations_api.OptimizationEvaluationLocalWorkflow")
    def test_run_hyperparameter_tuning_comprehensive_benchmark_local_default_parameters(
        self,
        mock_workflow_class,
    ):
        """Test comprehensive benchmark local with default parameters.

        Purpose: Validates that comprehensive benchmark uses correct default values when optional parameters are not provided

        Given: Only required parameters (generators)
        When: run_hyperparameter_tuning_comprehensive_benchmark_local is called with minimal parameters
        Then: OptimizationEvaluationLocalWorkflow is created with correct default values

        Test type: unit
        """
        # Mock the workflow instance
        mock_workflow_instance = MagicMock()
        mock_results = ({}, pd.DataFrame())
        mock_workflow_instance.optimize_and_evaluate.return_value = mock_results
        mock_workflow_class.return_value = mock_workflow_instance

        # Mock generators
        mock_generators = [Mock()]

        api = SimulationsAPI()
        results, stats_df = api.run_hyperparameter_tuning_experiment_with_benchmarks_local(
            generators=mock_generators,
        )

        # Verify default parameters
        call_args = mock_workflow_class.call_args

        assert call_args[1]["experiment_name"] == "Comprehensive_Benchmark"  # Default
        assert call_args[1]["optimization_n_jobs"] == -1  # Default
        assert call_args[1]["evaluation_episodes"] == 3  # Default
        assert call_args[1]["evaluation_steps"] == 6  # Default
        assert call_args[1]["evaluation_n_jobs"] == 1  # Default
        assert call_args[1]["confidence_interval_level"] == 0.95  # Default
        assert call_args[1]["alpha"] == 0.05  # Default
        assert call_args[1]["debug"] is False  # Default
        assert call_args[1]["cache_visualizations"] is True  # Default

    @patch("POMDPPlanners.simulations.simulations_api.OptimizationEvaluationPBSWorkflow")
    def test_run_hyperparameter_tuning_comprehensive_benchmark_pbs_success(
        self,
        mock_workflow_class,
        temp_cache_dir,
    ):
        """Test successful execution of run_hyperparameter_tuning_comprehensive_benchmark_pbs.

        Purpose: Validates that comprehensive benchmark with PBS execution completes successfully

        Given: Mocked workflow, generators, and PBS settings
        When: run_hyperparameter_tuning_comprehensive_benchmark_pbs is executed
        Then: OptimizationEvaluationPBSWorkflow is called with correct config and returns expected results

        Test type: unit
        """
        # Mock the workflow instance and its methods
        mock_workflow_instance = MagicMock()
        mock_results = (
            {"test_tiger": {"OptimizedPOMCP": ["mock_history_1", "mock_history_2"]}},
            pd.DataFrame(
                {
                    "environment": ["test_tiger"],
                    "policy": ["OptimizedPOMCP"],
                    "average_return": [10.5],
                }
            ),
        )
        mock_workflow_instance.optimize_and_evaluate.return_value = mock_results
        mock_workflow_class.return_value = mock_workflow_instance

        # Mock generators
        mock_generators = [Mock(), Mock()]

        api = SimulationsAPI()
        results, stats_df = api.run_hyperparameter_tuning_experiment_with_benchmarks_pbs(
            generators=mock_generators,
            queue="test_queue",
            particles=50,
            num_episodes=15,
            num_steps=25,
            n_trials=200,
            evaluation_episodes=5,
            evaluation_steps=10,
            evaluation_n_jobs=2,
            is_risk_averse=False,
            n_workers=8,
            cores=2,
            memory="16GB",
            processes=2,
            walltime="06:00:00",
            job_extra=["#PBS -l feature=gpu"],
            confidence_interval_level=0.95,
            alpha=0.05,
            cache_dir_path=temp_cache_dir,
            experiment_name="test_comprehensive_benchmark_pbs",
            debug=True,
            cache_visualizations=True,
        )

        # Verify OptimizationEvaluationPBSWorkflow was called with correct parameters
        mock_workflow_class.assert_called_once()
        call_args = mock_workflow_class.call_args

        assert call_args[1]["cache_dir"] == temp_cache_dir
        assert call_args[1]["experiment_name"] == "test_comprehensive_benchmark_pbs"
        assert call_args[1]["queue"] == "test_queue"
        assert call_args[1]["n_workers"] == 8
        assert call_args[1]["cores"] == 2
        assert call_args[1]["memory"] == "16GB"
        assert call_args[1]["processes"] == 2
        assert call_args[1]["walltime"] == "06:00:00"
        assert call_args[1]["job_extra"] == ["#PBS -l feature=gpu"]
        assert call_args[1]["evaluation_episodes"] == 5
        assert call_args[1]["evaluation_steps"] == 10
        assert call_args[1]["evaluation_n_jobs"] == 2
        assert call_args[1]["confidence_interval_level"] == 0.95
        assert call_args[1]["alpha"] == 0.05
        assert call_args[1]["debug"] is True
        assert call_args[1]["verbose"] is True
        assert call_args[1]["cache_visualizations"] is True

        # Verify optimize_and_evaluate was called with correct configs
        mock_workflow_instance.optimize_and_evaluate.assert_called_once()

        # Verify results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)

    @patch("POMDPPlanners.simulations.simulations_api.OptimizationEvaluationPBSWorkflow")
    def test_run_hyperparameter_tuning_comprehensive_benchmark_pbs_default_parameters(
        self,
        mock_workflow_class,
    ):
        """Test comprehensive benchmark PBS with default parameters.

        Purpose: Validates that comprehensive benchmark PBS uses correct default values when optional parameters are not provided

        Given: Only required parameters (generators, queue)
        When: run_hyperparameter_tuning_comprehensive_benchmark_pbs is called with minimal parameters
        Then: OptimizationEvaluationPBSWorkflow is created with correct default values

        Test type: unit
        """
        # Mock the workflow instance
        mock_workflow_instance = MagicMock()
        mock_results = ({}, pd.DataFrame())
        mock_workflow_instance.optimize_and_evaluate.return_value = mock_results
        mock_workflow_class.return_value = mock_workflow_instance

        # Mock generators
        mock_generators = [Mock()]

        api = SimulationsAPI()
        results, stats_df = api.run_hyperparameter_tuning_experiment_with_benchmarks_pbs(
            generators=mock_generators,
            queue="default_queue",
        )

        # Verify default parameters
        call_args = mock_workflow_class.call_args

        assert call_args[1]["experiment_name"] == "Comprehensive_Benchmark_PBS"  # Default
        assert call_args[1]["queue"] == "default_queue"
        assert call_args[1]["n_workers"] == 4  # Default
        assert call_args[1]["cores"] == 1  # Default
        assert call_args[1]["memory"] == "4GB"  # Default
        assert call_args[1]["processes"] == 1  # Default
        assert call_args[1]["walltime"] == "03:00:00"  # Default
        assert call_args[1]["job_extra"] is None  # Default
        assert call_args[1]["evaluation_episodes"] == 3  # Default
        assert call_args[1]["evaluation_steps"] == 6  # Default
        assert call_args[1]["evaluation_n_jobs"] == 1  # Default
        assert call_args[1]["confidence_interval_level"] == 0.95  # Default
        assert call_args[1]["alpha"] == 0.05  # Default
        assert call_args[1]["debug"] is False  # Default
        assert call_args[1]["cache_visualizations"] is True  # Default

        # Verify cache directory was created with default name
        expected_cache_dir = Path("./comprehensive_benchmark_pbs_results")
        assert call_args[1]["cache_dir"] == expected_cache_dir
