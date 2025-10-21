import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, cast
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import EnvironmentRunParams, NumericalHyperParameter
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParamPlannerConfig,
    HyperParameterFeature,
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.simulations.simulation_apis.pbs_simulations_api import PBSSimulationsAPI
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import PBSConfig

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
def sample_hyperparameter_run_params(tiger_environment):
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
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
        )
    ]


class TestPBSSimulationsAPI:
    @patch("POMDPPlanners.simulations.simulation_apis.pbs_simulations_api.POMDPSimulator")
    def test_run_multiple_environments_and_policies_custom_dashboard(
        self, mock_simulator, temp_cache_dir, sample_environment_run_params
    ):
        """Test PBS run with custom dashboard parameters.

        Purpose: Validates that PBS function correctly handles custom dashboard configuration

        Given: Custom dashboard parameters including enabled dashboard, custom port, address, and prefix
        When: run_multiple_environments_and_policies is called with dashboard parameters
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

        api = PBSSimulationsAPI(
            queue="gpu_queue",
            enable_dashboard=True,
            dashboard_address="192.168.1.100",
            dashboard_port=9999,
            dashboard_prefix="/cluster-monitor",
        )

        results, stats_df = api.run_multiple_environments_and_policies(
            environment_run_params=sample_environment_run_params,
            alpha=0.05,
            confidence_interval_level=0.95,
            cache_dir_path=temp_cache_dir,
        )

        # Verify dashboard parameters are properly passed
        call_args = mock_simulator.call_args
        task_manager_config = call_args[1]["task_manager_config"]

        assert task_manager_config.enable_dashboard is True
        assert task_manager_config.dashboard_address == "192.168.1.100"
        assert task_manager_config.dashboard_port == 9999
        assert task_manager_config.dashboard_prefix == "/cluster-monitor"

    @patch(
        "POMDPPlanners.simulations.simulation_apis.pbs_simulations_api.OptimizationEvaluationPBSWorkflow"
    )
    @patch(
        "POMDPPlanners.simulations.simulation_apis.pbs_simulations_api.PolicyHyperparameterOptimizationExperimentConfigCreator"
    )
    def test_run_hyperparameter_tuning_experiment_with_benchmarks_success(
        self,
        mock_creator_class,
        mock_workflow_class,
        temp_cache_dir,
    ):
        """Test successful execution of run_hyperparameter_tuning_experiment_with_benchmarks.

        Purpose: Validates that comprehensive benchmark with PBS execution completes successfully

        Given: Mocked workflow, generators, and PBS settings
        When: run_hyperparameter_tuning_experiment_with_benchmarks is executed
        Then: OptimizationEvaluationPBSWorkflow is called with correct config and returns expected results

        Test type: unit
        """
        # Mock config creator
        mock_creator_instance = MagicMock()
        mock_configs = [Mock()]
        mock_creator_instance.get_experiment_configs.return_value = mock_configs
        mock_creator_class.return_value = mock_creator_instance

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

        api = PBSSimulationsAPI(
            queue="test_queue",
            n_workers=8,
            cores=2,
            memory="16GB",
            processes=2,
            walltime="06:00:00",
            job_extra=["#PBS -l feature=gpu"],
        )

        results, stats_df = api.run_hyperparameter_tuning_experiment_with_benchmarks(
            generators=mock_generators,
            particles=50,
            num_episodes=15,
            num_steps=25,
            n_trials=200,
            evaluation_episodes=5,
            evaluation_steps=10,
            evaluation_n_jobs=2,
            is_risk_averse=False,
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


class TestPBSSimulationsAPIIntegration:
    """Test PBS simulation API integration tests."""

    @pytest.mark.pbs_cluster
    def test_pbs_simulation_api_integration(self, temp_cache_dir):
        """Test PBS simulation API with integration test.

        Purpose: Validates that PBS simulation API can be used with proper mocking
        to test the PBS configuration and parameter passing.

        Given: PBS simulation parameters and mocked POMDPSimulator
        When: Calling run_multiple_environments_and_policies
        Then: PBS configuration is correctly created and passed to simulator

        Test type: integration
        """
        # Create test environment and policy
        tiger = TigerPOMDP(discount_factor=0.95)
        policy = POMCP(
            environment=tiger,
            discount_factor=0.95,
            depth=3,
            exploration_constant=1.0,
            name="POMCP_Test",
            n_simulations=10,
        )

        env_params = [
            EnvironmentRunParams(
                environment=tiger,
                belief=get_initial_belief(tiger, n_particles=5),
                policies=[policy],
                num_episodes=2,
                num_steps=3,
            )
        ]

        # Mock the POMDPSimulator to avoid actual PBS execution
        with patch(
            "POMDPPlanners.simulations.simulation_apis.pbs_simulations_api.POMDPSimulator"
        ) as mock_simulator:
            mock_instance = Mock()
            mock_instance.__enter__ = Mock(return_value=mock_instance)
            mock_instance.__exit__ = Mock(return_value=None)
            mock_instance.compare_multiple_environments_policies.return_value = ({}, Mock())
            mock_simulator.return_value = mock_instance

            api = PBSSimulationsAPI(
                queue="test_queue",
                n_workers=1,
                cores=1,
                memory="1GB",
                walltime="00:05:00",
            )

            # Test PBS run
            results, stats = api.run_multiple_environments_and_policies(
                environment_run_params=env_params,
                alpha=0.05,
                confidence_interval_level=0.95,
                experiment_name="PBS_Test",
                cache_dir_path=temp_cache_dir,
            )

            # Verify the simulator was called with correct PBS config
            mock_simulator.assert_called_once()
            call_args = mock_simulator.call_args[1]

            task_manager_config = call_args["task_manager_config"]

            assert isinstance(task_manager_config, PBSConfig)
            assert task_manager_config.queue == "test_queue"
            assert task_manager_config.n_workers == 1
            assert task_manager_config.cores == 1
            assert task_manager_config.memory == "1GB"
            assert task_manager_config.walltime == "00:05:00"
