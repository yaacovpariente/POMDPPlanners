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
from POMDPPlanners.core.environment import Environment, SpaceType
from POMDPPlanners.core.policy import Policy, PolicySpaceInfo
from POMDPPlanners.core.simulation import (
    CategoricalHyperParameter,
    EnvironmentRunParams,
    History,
    NumericalHyperParameter,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParamPlannerConfig,
    HyperParamPlannerConfigGenerator,
    HyperParameterFeature,
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
    OptimizedPolicyResult,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
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
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
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
        parameters_to_optimize=[("average_return", HyperParameterOptimizationDirection.MAXIMIZE)],
        optimized_metric_values={"average_return": 10.5},
    )


class TestLocalSimulationsAPI:
    def test_init(self):
        """Test LocalSimulationsAPI initialization.

        Purpose: Validates that LocalSimulationsAPI can be instantiated without errors

        Given: No parameters required for LocalSimulationsAPI constructor
        When: LocalSimulationsAPI instance is created
        Then: Object is properly initialized as LocalSimulationsAPI instance

        Test type: unit
        """
        api = LocalSimulationsAPI()
        assert isinstance(api, LocalSimulationsAPI)

    def test_run_multiple_environments_and_policies_success(
        self, temp_cache_dir, sample_environment_run_params
    ):
        """Test successful execution of run_multiple_environments_and_policies.

        Purpose: Validates that local simulation execution completes successfully and returns expected results structure

        Given: TigerPOMDP environment, SparsePFT policy, 2 episodes, 10 steps, alpha=0.1, confidence_interval_level=0.95, single job execution
        When: run_multiple_environments_and_policies is executed with valid parameters
        Then: Returns dict with environment-policy results (2 episodes) and stats DataFrame with environment, policy, success_rate, average_listens columns

        Test type: integration
        """
        api = LocalSimulationsAPI()
        results, stats_df = api.run_multiple_environments_and_policies(
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

    def test_run_multiple_environments_and_policies_error(
        self, temp_cache_dir, sample_environment_run_params
    ):
        """Test error handling in run_multiple_environments_and_policies.

        Purpose: Validates error handling for run multiple environments and policies

        Given: Invalid inputs or error conditions
        When: Operation is attempted
        Then: Appropriate exception is raised

        Test type: integration
        """
        api = LocalSimulationsAPI()
        # Force an error by mocking the environment's state_transition_model to return None
        sample_environment_run_params[0].environment.state_transition_model = lambda *args: None
        with pytest.raises(Exception):
            api.run_multiple_environments_and_policies(
                environment_run_params=sample_environment_run_params,
                alpha=0.1,
                confidence_interval_level=0.95,
                cache_dir_path=temp_cache_dir,
            )

    def test_run_multiple_environments_and_policies_invalid_params(
        self, temp_cache_dir, sample_environment_run_params
    ):
        """Test run_multiple_environments_and_policies with invalid parameters.

        Purpose: Validates that invalid parameter values raise appropriate ValueError exceptions with descriptive messages

        Given: LocalSimulationsAPI instance and sample environment parameters with invalid alpha (-0.1), confidence_interval_level (1.5), and n_jobs (0) values
        When: run_multiple_environments_and_policies is called with each invalid parameter
        Then: ValueError is raised with appropriate error messages for alpha, confidence_interval_level, and n_jobs validation

        Test type: integration
        """
        api = LocalSimulationsAPI()
        # Test invalid alpha
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            api.run_multiple_environments_and_policies(
                environment_run_params=sample_environment_run_params,
                alpha=-0.1,  # Invalid alpha
                confidence_interval_level=0.95,
                cache_dir_path=temp_cache_dir,
            )
        # Test invalid confidence interval
        with pytest.raises(ValueError, match="confidence_interval_level must be between 0 and 1"):
            api.run_multiple_environments_and_policies(
                environment_run_params=sample_environment_run_params,
                alpha=0.1,
                confidence_interval_level=1.5,  # Invalid confidence interval
                cache_dir_path=temp_cache_dir,
            )
        # Test invalid n_jobs
        with pytest.raises(ValueError, match="n_jobs must be a positive integer or -1"):
            api.run_multiple_environments_and_policies(
                environment_run_params=sample_environment_run_params,
                alpha=0.1,
                confidence_interval_level=0.95,
                n_jobs=0,  # Invalid n_jobs
                cache_dir_path=temp_cache_dir,
            )

    @patch(
        "POMDPPlanners.simulations.simulation_apis.local_simulations_api.HyperParameterOptimizer"
    )
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

        api = LocalSimulationsAPI()
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

    @patch(
        "POMDPPlanners.simulations.simulation_apis.local_simulations_api.HyperParameterOptimizer"
    )
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

        api = LocalSimulationsAPI()
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

    @patch(
        "POMDPPlanners.simulations.simulation_apis.local_simulations_api.HyperParameterOptimizer"
    )
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

        api = LocalSimulationsAPI()

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

    @patch(
        "POMDPPlanners.simulations.simulation_apis.local_simulations_api.OptimizationEvaluationLocalWorkflow"
    )
    def test_run_optimize_and_evaluate_success(
        self,
        mock_workflow_class,
        temp_cache_dir,
        sample_hyperparameter_run_params,
    ):
        """Test successful execution of run_optimize_and_evaluate.

        Purpose: Validates that optimize and evaluate workflow completes successfully with local execution

        Given: TigerPOMDP environment, POMCP policy class, hyperparameter configs, and mocked workflow
        When: run_optimize_and_evaluate is executed with valid parameters
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

        api = LocalSimulationsAPI()
        results, stats_df = api.run_optimize_and_evaluate(
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
        # Mock the workflow instance
        mock_workflow_instance = MagicMock()
        mock_results = ({}, pd.DataFrame())
        mock_workflow_instance.optimize_and_evaluate.return_value = mock_results
        mock_workflow_class.return_value = mock_workflow_instance

        api = LocalSimulationsAPI()
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


# Integration Tests - using debug mode for fast execution
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

        planner_config = HyperParamPlannerConfig(
            policy_cls=POMCP,
            hyper_parameters=cast(
                List[HyperParameterFeature],
                [
                    NumericalHyperParameter(low=0.5, high=1.5, name="exploration_constant"),
                ],
            ),
            constant_parameters={
                "discount_factor": 0.95,
                "name": "IntegrationTestPOMCP",
                "depth": 5,
                "n_simulations": 20,
            },
        )

        optimization_configs = [
            HyperParameterRunParams(
                environment=tiger_environment,
                belief=initial_belief,
                hyper_param_planner_config=planner_config,
                num_episodes=1,
                num_steps=3,
                n_trials=2,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
            )
        ]

        results = api.run_hyperparameter_optimization(
            environment_run_params=optimization_configs,
            experiment_name="integration_test_hyperparam",
            n_jobs=1,
            cache_dir_path=temp_cache_dir,
            debug=True,
        )

        # Verify actual optimization results
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], OptimizedPolicyResult)
        assert "exploration_constant" in results[0].chosen_hyper_parameters
        assert results[0].policy.name == "IntegrationTestPOMCP"

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

        planner_config = HyperParamPlannerConfig(
            policy_cls=POMCP,
            hyper_parameters=cast(
                List[HyperParameterFeature],
                [
                    NumericalHyperParameter(low=0.5, high=1.5, name="exploration_constant"),
                ],
            ),
            constant_parameters={
                "discount_factor": 0.95,
                "name": "IntegrationOptEvalPOMCP",
                "depth": 5,
                "n_simulations": 20,
            },
        )

        configs = [
            HyperParameterRunParams(
                environment=tiger_environment,
                belief=initial_belief,
                hyper_param_planner_config=planner_config,
                num_episodes=1,
                num_steps=3,
                n_trials=2,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
            )
        ]

        results, stats_df = api.run_optimize_and_evaluate(
            configs=configs,
            evaluation_episodes=1,
            evaluation_steps=3,
            evaluation_n_jobs=1,
            optimization_n_jobs=1,
            cache_dir_path=temp_cache_dir,
            experiment_name="integration_test_opt_eval",
            debug=True,
        )

        # Verify actual workflow results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) > 0

    def test_run_all_hyperparameter_benchmarks_integration(self, temp_cache_dir):
        """Test run_all_hyperparameter_benchmarks integration with debug mode.

        Purpose: Validates that all hyperparameter benchmarks execute end-to-end with real environments

        Given: PolicySpaceInfo for discrete action/observation spaces, minimal parameters (1 episode, 3 steps, 2 trials), debug=True
        When: run_all_hyperparameter_benchmarks is executed
        Then: Returns results dict and DataFrame with actual benchmark results

        Test type: integration
        """
        api = LocalSimulationsAPI()
        policy_space_info = PolicySpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.DISCRETE,
        )

        results, stats_df = api.run_all_hyperparameter_benchmarks(
            policy_space_info=policy_space_info,
            particles=5,
            num_episodes=1,
            num_steps=3,
            n_trials=2,
            discount_factor=0.95,
            time_out_in_seconds=0.1,
            evaluation_episodes=1,
            evaluation_steps=3,
            evaluation_n_jobs=1,
            optimization_n_jobs=1,
            is_risk_averse=False,
            confidence_interval_level=0.95,
            alpha=0.05,
            cache_dir_path=temp_cache_dir,
            experiment_name="integration_test_all_benchmarks",
            debug=True,
            cache_visualizations=False,
        )

        # Verify actual benchmark results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) > 0

    def test_run_hyperparameter_tuning_experiment_with_benchmarks_integration(
        self, temp_cache_dir, tiger_environment
    ):
        """Test run_hyperparameter_tuning_experiment_with_benchmarks integration with debug mode.

        Purpose: Validates that comprehensive benchmark with hyperparameter tuning executes end-to-end

        Given: Generator for TigerPOMDP with POMCP, minimal parameters (1 episode, 3 steps, 2 trials), debug=True
        When: run_hyperparameter_tuning_experiment_with_benchmarks is executed
        Then: Returns results dict and DataFrame with actual tuning and evaluation results

        Test type: integration
        """
        api = LocalSimulationsAPI()

        # Create a simple generator for POMCP on Tiger
        class SimplePOMCPGenerator(HyperParamPlannerConfigGenerator):
            def __init__(self, environment):
                self.environment = environment

            def generate(self, environment: "Environment") -> HyperParamPlannerConfig:
                return HyperParamPlannerConfig(
                    policy_cls=POMCP,
                    hyper_parameters=cast(
                        List[HyperParameterFeature],
                        [
                            NumericalHyperParameter(low=0.5, high=1.5, name="exploration_constant"),
                        ],
                    ),
                    constant_parameters={
                        "discount_factor": environment.discount_factor,
                        "name": f"BenchmarkPOMCP_{environment.name}",
                        "depth": 5,
                        "n_simulations": 2,
                    },
                )

            def get_planner_space_info(self) -> "PolicySpaceInfo":
                return PolicySpaceInfo(
                    action_space=self.environment.space_info.action_space,
                    observation_space=self.environment.space_info.observation_space,
                )

        generator = SimplePOMCPGenerator(tiger_environment)

        results, stats_df = api.run_hyperparameter_tuning_experiment_with_benchmarks(
            generators=[generator],
            particles=5,
            num_episodes=2,
            num_steps=2,
            n_trials=2,
            discount_factor=0.95,
            time_out_in_seconds=0.1,
            evaluation_episodes=2,
            evaluation_steps=2,
            evaluation_n_jobs=1,
            optimization_n_jobs=1,
            is_risk_averse=False,
            confidence_interval_level=0.95,
            alpha=0.05,
            cache_dir_path=temp_cache_dir,
            experiment_name="integration_test_comprehensive_benchmark",
            debug=True,
            cache_visualizations=False,
        )

        # Verify actual benchmark results
        assert isinstance(results, dict)
        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) > 0
