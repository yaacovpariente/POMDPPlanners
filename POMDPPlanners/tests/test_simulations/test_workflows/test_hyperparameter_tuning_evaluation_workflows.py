# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for hyperparameter_tuning_evaluation_workflows module.

This module tests the workflow classes for running hyperparameter optimization
followed by policy evaluation in different execution environments.
"""

# pylint: disable=protected-access  # Tests need to access protected members

import random
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from POMDPPlanners.simulations.workflows.hyperparameter_tuning_evaluation_workflows import (
    OptimizationEvaluationLocalWorkflow,
    OptimizationEvaluationDaskWorkflow,
    OptimizationEvaluationPBSWorkflow,
)
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    DaskConfig,
    JoblibConfig,
    PBSConfig,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParamPlannerConfigGenerator,
    HyperParamPlannerConfig,
    NumericalHyperParameter,
)
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler

np.random.seed(42)
random.seed(42)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for testing cache operations.

    Purpose: Provides isolated cache directory for workflow testing

    Given: System temporary directory
    When: Tests need cache storage
    Then: Temporary path is created and cleaned up after test

    Test type: unit
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    yield temp_path
    try:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def sample_workflow_params(temp_cache_dir):
    """Create sample workflow parameters for testing.

    Purpose: Provides common workflow initialization parameters

    Given: Temporary cache directory
    When: Creating workflow instances
    Then: Returns dictionary of standard parameters

    Test type: unit
    """
    return {
        "cache_dir": temp_cache_dir,
        "experiment_name": "Test_Experiment",
        "evaluation_episodes": 2,  # Match the default value
        "evaluation_steps": 6,
        "evaluation_n_jobs": 1,
        "confidence_interval_level": 0.95,
        "alpha": 0.05,
        "debug": False,
        "verbose": True,
        "cache_visualizations": True,
    }


class TestOptimizationEvaluationLocalWorkflow:
    """Test OptimizationEvaluationLocalWorkflow for local execution."""

    def test_initialization_with_default_parameters(self, temp_cache_dir):
        """Test initialization with default parameters.

        Purpose: Validates that LocalWorkflow initializes with default values

        Given: Minimum required parameters (cache_dir and experiment_name)
        When: Creating LocalWorkflow instance
        Then: All attributes are set to expected default values

        Test type: unit
        """
        workflow = OptimizationEvaluationLocalWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Test_Experiment",
        )

        assert workflow.cache_dir == temp_cache_dir
        assert workflow.experiment_name == "Test_Experiment"
        assert workflow.evaluation_episodes == 2  # Default changed to 2
        assert workflow.evaluation_steps == 6
        assert workflow.evaluation_n_jobs == 1
        assert workflow.optimization_n_jobs == -1
        assert workflow.confidence_interval_level == 0.95
        assert workflow.alpha == 0.05
        assert workflow.debug is False
        assert workflow.verbose is True
        assert workflow.cache_visualizations is True

    def test_get_task_manager_config_hyperparameter_returns_joblib_config(
        self, sample_workflow_params
    ):
        """Test that hyperparameter task manager config is JoblibConfig.

        Purpose: Validates correct task manager type for hyperparameter optimization

        Given: LocalWorkflow instance with optimization_n_jobs=-1
        When: Calling _get_task_manager_config_hyperparameter
        Then: Returns JoblibConfig with n_jobs=1 (single task, parallelized by Optuna)

        Test type: unit
        """
        workflow = OptimizationEvaluationLocalWorkflow(
            optimization_n_jobs=-1, **sample_workflow_params
        )

        config = workflow._get_task_manager_config_hyperparameter()

        assert isinstance(config, JoblibConfig)
        assert config.n_jobs == 1  # Always 1 for local, parallelization handled by Optuna

    def test_get_task_manager_config_evaluation_returns_joblib_config(self, temp_cache_dir):
        """Test that evaluation task manager config is JoblibConfig.

        Purpose: Validates correct task manager type for evaluation phase

        Given: LocalWorkflow instance with evaluation_n_jobs=4
        When: Calling _get_task_manager_config_evaluation
        Then: Returns JoblibConfig with correct n_jobs value

        Test type: unit
        """
        workflow = OptimizationEvaluationLocalWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Test",
            evaluation_n_jobs=4,
        )

        config = workflow._get_task_manager_config_evaluation()

        assert isinstance(config, JoblibConfig)
        assert config.n_jobs == 4


class TestOptimizationEvaluationPBSWorkflow:
    """Test OptimizationEvaluationPBSWorkflow for PBS cluster execution."""

    def test_initialization_with_default_parameters(self, temp_cache_dir):
        """Test initialization with default PBS parameters.

        Purpose: Validates that PBSWorkflow initializes with default PBS values

        Given: Minimum required parameters (cache_dir and experiment_name)
        When: Creating PBSWorkflow instance
        Then: All PBS-related attributes are set to expected defaults

        Test type: unit
        """
        workflow = OptimizationEvaluationPBSWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="PBS_Test",
        )

        assert workflow.cache_dir == temp_cache_dir
        assert workflow.experiment_name == "PBS_Test"
        assert workflow.queue == "short"
        assert workflow.n_workers == 4
        assert workflow.cores == 1
        assert workflow.memory == "4GB"
        assert workflow.processes == 1
        assert workflow.walltime == "03:00:00"
        assert workflow.job_extra is None
        assert workflow.optimization_n_jobs == 1  # Should equal cores

    def test_get_task_manager_config_hyperparameter_returns_pbs_config(self, temp_cache_dir):
        """Test that hyperparameter task manager config is PBSConfig.

        Purpose: Validates correct task manager type for PBS hyperparameter optimization

        Given: PBSWorkflow instance with custom PBS settings
        When: Calling _get_task_manager_config_hyperparameter
        Then: Returns PBSConfig with correct PBS parameters and processes=1

        Test type: unit
        """
        workflow = OptimizationEvaluationPBSWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Test",
            queue="long",
            n_workers=8,
            cores=2,
            processes=4,
        )

        config = workflow._get_task_manager_config_hyperparameter()

        assert isinstance(config, PBSConfig)
        assert config.queue == "long"
        assert config.n_workers == 8
        assert config.cores == 2
        assert config.processes == 1  # Always 1 for hyperparameter optimization
        assert workflow.optimization_n_jobs == 2  # Should equal cores

    def test_get_task_manager_config_evaluation_returns_pbs_config(self, temp_cache_dir):
        """Test that evaluation task manager config is PBSConfig.

        Purpose: Validates correct task manager type for PBS evaluation phase

        Given: PBSWorkflow instance with custom PBS settings
        When: Calling _get_task_manager_config_evaluation
        Then: Returns PBSConfig with correct PBS parameters including processes

        Test type: unit
        """
        workflow = OptimizationEvaluationPBSWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Test",
            memory="32GB",
            walltime="24:00:00",
            processes=4,
        )

        config = workflow._get_task_manager_config_evaluation()

        assert isinstance(config, PBSConfig)
        assert config.memory == "32GB"
        assert config.walltime == "24:00:00"
        assert config.processes == 4  # Uses the configured processes value


class TestOptimizationEvaluationDaskWorkflow:
    """Test OptimizationEvaluationDaskWorkflow for Dask distributed execution."""

    def test_initialization_with_default_parameters(self, temp_cache_dir):
        """Test initialization with default Dask parameters.

        Purpose: Validates that DaskWorkflow initializes with default Dask values

        Given: Minimum required parameters (cache_dir and experiment_name)
        When: Creating DaskWorkflow instance
        Then: All Dask-related attributes are set to expected defaults

        Test type: unit
        """
        workflow = OptimizationEvaluationDaskWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Dask_Test",
        )

        assert workflow.cache_dir == temp_cache_dir
        assert workflow.experiment_name == "Dask_Test"
        assert workflow.n_workers == 4
        assert workflow.scheduler_address is None
        assert workflow.cache_size == int(2e9)
        assert workflow.clear_cache_on_start is False
        assert workflow.optimization_n_jobs == 4  # Should equal n_workers

    def test_initialization_with_custom_parameters(self, temp_cache_dir):
        """Test initialization with custom Dask parameters.

        Purpose: Validates that DaskWorkflow correctly stores custom parameters

        Given: Custom Dask configuration parameters
        When: Creating DaskWorkflow instance with custom values
        Then: All attributes reflect the custom configuration

        Test type: unit
        """
        workflow = OptimizationEvaluationDaskWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Custom_Dask_Test",
            n_workers=8,
            scheduler_address="tcp://localhost:8786",
            cache_size=int(4e9),
            clear_cache_on_start=True,
            evaluation_episodes=10,
            evaluation_steps=20,
        )

        assert workflow.n_workers == 8
        assert workflow.scheduler_address == "tcp://localhost:8786"
        assert workflow.cache_size == int(4e9)
        assert workflow.clear_cache_on_start is True
        assert workflow.evaluation_episodes == 10
        assert workflow.evaluation_steps == 20
        assert workflow.optimization_n_jobs == 8

    def test_get_task_manager_config_hyperparameter_returns_dask_config(self, temp_cache_dir):
        """Test that hyperparameter task manager config is DaskConfig.

        Purpose: Validates correct task manager type for Dask hyperparameter optimization

        Given: DaskWorkflow instance with custom Dask settings
        When: Calling _get_task_manager_config_hyperparameter
        Then: Returns DaskConfig with correct Dask parameters

        Test type: unit
        """
        workflow = OptimizationEvaluationDaskWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Test",
            n_workers=6,
            scheduler_address="tcp://localhost:8786",
            cache_size=int(3e9),
            clear_cache_on_start=True,
        )

        config = workflow._get_task_manager_config_hyperparameter()

        assert isinstance(config, DaskConfig)
        assert config.n_workers == 6
        assert config.scheduler_address == "tcp://localhost:8786"
        assert config.cache_size == int(3e9)
        assert config.clear_cache_on_start is True

    def test_get_task_manager_config_evaluation_returns_dask_config(self, temp_cache_dir):
        """Test that evaluation task manager config is DaskConfig.

        Purpose: Validates correct task manager type for Dask evaluation phase

        Given: DaskWorkflow instance with custom Dask settings
        When: Calling _get_task_manager_config_evaluation
        Then: Returns DaskConfig with correct parameters and clear_cache_on_start=False

        Test type: unit
        """
        workflow = OptimizationEvaluationDaskWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Test",
            n_workers=8,
            cache_size=int(5e9),
            clear_cache_on_start=True,  # Should be overridden for evaluation
        )

        config = workflow._get_task_manager_config_evaluation()

        assert isinstance(config, DaskConfig)
        assert config.n_workers == 8
        assert config.cache_size == int(5e9)
        assert config.clear_cache_on_start is False  # Always False for evaluation

    def test_get_task_manager_config_evaluation_preserves_scheduler_address(self, temp_cache_dir):
        """Test that evaluation config preserves scheduler address.

        Purpose: Validates that scheduler address is correctly passed to evaluation config

        Given: DaskWorkflow instance with scheduler address
        When: Calling _get_task_manager_config_evaluation
        Then: Returns DaskConfig with same scheduler address

        Test type: unit
        """
        scheduler_addr = "tcp://10.0.0.1:8786"
        workflow = OptimizationEvaluationDaskWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Test",
            scheduler_address=scheduler_addr,
        )

        config = workflow._get_task_manager_config_evaluation()

        assert isinstance(config, DaskConfig)
        assert config.scheduler_address == scheduler_addr


class IntegerActionSampler(ActionSampler):
    """Action sampler that returns integer actions for compatibility with environments."""

    def __init__(self, actions):
        self.actions = actions

    def sample(self, belief_node=None):
        # Return integer index instead of string action
        return random.randint(0, len(self.actions) - 1)


class PFT_DPW_TestGenerator(HyperParamPlannerConfigGenerator):
    """Test generator for PFT_DPW planner configuration."""

    def __init__(
        self,
        discount_factor: float,
        depth: int,
        name: str,
        action_sampler: ActionSampler,
        max_exploration_constant: float = 1.0,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        min_samples_per_node: int = 10,
        min_visit_count_per_action: int = 1,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        self.discount_factor = discount_factor
        self.depth = depth
        self.name = name
        self.action_sampler = action_sampler
        self.max_exploration_constant = max_exploration_constant
        self.time_out_in_seconds = time_out_in_seconds
        self.n_simulations = n_simulations
        self.min_samples_per_node = min_samples_per_node
        self.min_visit_count_per_action = min_visit_count_per_action

    def generate(self, environment) -> HyperParamPlannerConfig:
        hyper_parameters = [
            NumericalHyperParameter(0, self.max_exploration_constant, "exploration_constant"),
            NumericalHyperParameter(1, 10, "k_o"),
            NumericalHyperParameter(0.01, 0.5, "alpha_o"),
        ]

        constant_parameters = {
            "discount_factor": self.discount_factor,
            "environment": environment,
            "name": self.name,
            "depth": self.depth,
            "action_sampler": self.action_sampler,
            "n_simulations": self.n_simulations,
        }

        return HyperParamPlannerConfig(
            policy_cls=PFT_DPW,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,
        )

    def get_planner_space_info(self):
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.DISCRETE,
        )


class TestWorkflowValidation:
    """Test input validation for workflow optimize_and_evaluate method."""

    def test_validate_configs_empty_list_raises_error(self, temp_cache_dir):
        """Test that empty configs list raises ValueError.

        Purpose: Validates that empty configs list is rejected at workflow level

        Given: OptimizationEvaluationLocalWorkflow instance and empty configs list
        When: optimize_and_evaluate is called with empty list
        Then: ValueError is raised with appropriate message

        Test type: unit
        """
        workflow = OptimizationEvaluationLocalWorkflow(
            cache_dir=temp_cache_dir, experiment_name="test", optimization_n_jobs=1
        )

        with pytest.raises(ValueError, match="configs list cannot be empty"):
            workflow.optimize_and_evaluate([])

    def test_validate_configs_invalid_type_raises_error(self, temp_cache_dir):
        """Test that non-HyperParameterRunParams element raises TypeError.

        Purpose: Validates that configs must be proper type

        Given: Workflow and list with dict instead of HyperParameterRunParams
        When: optimize_and_evaluate is called
        Then: TypeError is raised indicating wrong type

        Test type: unit
        """
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief

        workflow = OptimizationEvaluationLocalWorkflow(
            cache_dir=temp_cache_dir, experiment_name="test", optimization_n_jobs=1
        )

        env = TigerPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=10)
        invalid_configs = [{"environment": env, "belief": belief}]

        with pytest.raises(TypeError, match="not a HyperParameterRunParams instance"):
            workflow.optimize_and_evaluate(invalid_configs)  # type: ignore

    def test_validate_configs_negative_num_episodes_raises_error(self, temp_cache_dir):
        """Test that negative num_episodes raises ValueError.

        Purpose: Validates that num_episodes must be positive

        Given: Config with num_episodes = -1
        When: HyperParameterRunParams is constructed
        Then: ValueError is raised about num_episodes

        Test type: unit
        """
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.core.simulation.hyperparameter_tuning import (
            HyperParameterRunParams,
            HyperParamPlannerConfig,
            HyperParameterOptimizationDirection,
        )

        workflow = OptimizationEvaluationLocalWorkflow(
            cache_dir=temp_cache_dir, experiment_name="test", optimization_n_jobs=1
        )

        env = TigerPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=10)

        with pytest.raises(ValueError, match="num_episodes must be positive"):
            config = HyperParameterRunParams(
                environment=env,
                belief=belief,
                hyper_param_planner_config=HyperParamPlannerConfig(
                    policy_cls=POMCP,
                    hyper_parameters=[NumericalHyperParameter(0.1, 2.0, "exploration_constant")],
                    constant_parameters={"depth": 5, "n_simulations": 100},
                ),
                num_episodes=-1,  # Invalid
                num_steps=10,
                n_trials=5,
                parameters_to_optimize=[
                    ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
                ],
            )

    def test_validate_configs_invalid_metric_name_raises_error(self, temp_cache_dir):
        """Test that invalid metric name in parameters_to_optimize raises ValueError.

        Purpose: Validates that metric names must match available metrics

        Given: Config with invalid metric name "nonexistent_metric"
        When: HyperParameterRunParams is constructed
        Then: ValueError is raised with available metrics listed

        Test type: unit
        """
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.core.simulation.hyperparameter_tuning import (
            HyperParameterRunParams,
            HyperParamPlannerConfig,
            HyperParameterOptimizationDirection,
        )

        workflow = OptimizationEvaluationLocalWorkflow(
            cache_dir=temp_cache_dir, experiment_name="test", optimization_n_jobs=1
        )

        env = TigerPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=10)

        with pytest.raises(ValueError, match="Invalid metric name 'nonexistent_metric'"):
            config = HyperParameterRunParams(
                environment=env,
                belief=belief,
                hyper_param_planner_config=HyperParamPlannerConfig(
                    policy_cls=POMCP,
                    hyper_parameters=[NumericalHyperParameter(0.1, 2.0, "exploration_constant")],
                    constant_parameters={"depth": 5, "n_simulations": 100},
                ),
                num_episodes=10,
                num_steps=5,
                n_trials=5,
                parameters_to_optimize=[
                    (
                        "nonexistent_metric",
                        HyperParameterOptimizationDirection.MAXIMIZE,
                    )  # Invalid metric
                ],
            )

    def test_validate_configs_valid_environment_specific_metric(self, temp_cache_dir):
        """Test that valid environment-specific metric passes validation at workflow level.

        Purpose: Validates that environment-specific metrics (like TigerPOMDP's success_rate) are accepted

        Given: TigerPOMDP environment and config with "success_rate" metric
        When: _validate_configs is called
        Then: No exception is raised

        Test type: unit
        """
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.core.simulation.hyperparameter_tuning import (
            HyperParameterRunParams,
            HyperParamPlannerConfig,
            HyperParameterOptimizationDirection,
        )

        workflow = OptimizationEvaluationLocalWorkflow(
            cache_dir=temp_cache_dir, experiment_name="test", optimization_n_jobs=1
        )

        env = TigerPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=10)

        config = HyperParameterRunParams(
            environment=env,
            belief=belief,
            hyper_param_planner_config=HyperParamPlannerConfig(
                policy_cls=POMCP,
                hyper_parameters=[NumericalHyperParameter(0.1, 2.0, "exploration_constant")],
                constant_parameters={"depth": 5, "n_simulations": 100},
            ),
            num_episodes=10,
            num_steps=5,
            n_trials=5,
            parameters_to_optimize=[
                (
                    "success_rate",
                    HyperParameterOptimizationDirection.MAXIMIZE,
                )  # TigerPOMDP-specific
            ],
        )

        # Should not raise any exception
        workflow._validate_configs([config])


class TestWorkflowOptimizedConfigUsage:
    """Test that optimized hyperparameters are used in evaluation."""

    @patch(
        "POMDPPlanners.simulations.workflows.hyperparameter_tuning_evaluation_workflows.POMDPSimulator"
    )
    @patch(
        "POMDPPlanners.simulations.workflows.hyperparameter_tuning_evaluation_workflows.HyperParameterOptimizer"
    )
    def test_workflow_uses_optimized_config_for_evaluation(
        self, mock_optimizer_class, mock_simulator_class, temp_cache_dir
    ):
        """Test that optimized hyperparameters are actually used in evaluation.

        Purpose: Validates that the policy configuration selected by hyperparameter tuning
                 is the actual configuration used for evaluation in the workflow

        Given: Optimization returns a policy with specific hyperparameters
        When: optimize_and_evaluate executes both optimization and evaluation phases
        Then: The policy passed to evaluation is the same policy from optimization,
              and its hyperparameters match the chosen_hyper_parameters from optimization

        Test type: integration
        """
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.core.simulation.hyperparameter_tuning import (
            HyperParameterRunParams,
            HyperParamPlannerConfig,
            HyperParameterOptimizationDirection,
            OptimizedPolicyResult,
        )

        # Create environment and belief
        env = TigerPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=10)

        # Create workflow
        workflow = OptimizationEvaluationLocalWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="test_optimized_config",
            optimization_n_jobs=1,
            evaluation_episodes=2,
            evaluation_steps=6,
        )

        # Create config
        config = HyperParameterRunParams(
            environment=env,
            belief=belief,
            hyper_param_planner_config=HyperParamPlannerConfig(
                policy_cls=POMCP,
                hyper_parameters=[NumericalHyperParameter(0.1, 2.0, "exploration_constant")],
                constant_parameters={"discount_factor": 0.95, "name": "TestPOMCP"},
            ),
            num_episodes=3,
            num_steps=6,
            n_trials=2,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
        )

        # Create policy with specific hyperparameters stored as attributes
        chosen_hyperparams = {"exploration_constant": 1.75, "n_simulations": 18}
        optimized_policy = POMCP(
            environment=env,
            discount_factor=0.95,
            depth=5,
            name="OptimizedPOMCP",
            **chosen_hyperparams,  # Store hyperparameters as policy attributes
        )

        # Create optimization result with the same hyperparameters
        optimization_result = OptimizedPolicyResult(
            environment=env,
            policy=optimized_policy,
            chosen_hyper_parameters=chosen_hyperparams,
            num_episodes=3,
            num_steps=6,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
            optimized_metric_values={"average_return": 10.5},
        )

        # Mock optimizer to return our optimization result
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.optimize.return_value = [optimization_result]

        # Mock simulator and its context manager
        mock_simulator = Mock()
        mock_simulator_class.return_value.__enter__ = Mock(return_value=mock_simulator)
        mock_simulator_class.return_value.__exit__ = Mock(return_value=None)

        # Mock evaluation results
        mock_evaluation_results = {"TigerPOMDP": {"OptimizedPOMCP": [Mock(), Mock()]}}
        mock_evaluation_statistics = pd.DataFrame(
            {"policy": ["OptimizedPOMCP"], "metric": ["average_return"], "value": [8.5]}
        )
        mock_simulator.compare_multiple_environments_policies.return_value = (
            mock_evaluation_results,
            mock_evaluation_statistics,
        )

        # Execute workflow
        results = workflow.optimize_and_evaluate([config])

        # Verify optimizer was created and called
        mock_optimizer_class.assert_called_once()
        mock_optimizer.optimize.assert_called_once_with([config])

        # Verify simulator was created
        mock_simulator_class.assert_called_once()

        # Verify evaluation was called
        mock_simulator.compare_multiple_environments_policies.assert_called_once()

        # Capture the arguments passed to compare_multiple_environments_policies
        eval_call_args = mock_simulator.compare_multiple_environments_policies.call_args
        eval_environment_run_params = eval_call_args[1]["environment_run_params"]

        # Verify that we have the correct number of environment run params
        assert len(eval_environment_run_params) == 1
        eval_run_params = eval_environment_run_params[0]

        # Verify that the policies passed to evaluation are the same policies from optimization
        assert len(eval_run_params.policies) == 1
        evaluated_policy = eval_run_params.policies[0]
        assert (
            evaluated_policy is optimized_policy
        ), "Evaluation should use the exact same policy object from optimization"

        # Verify that the policy's hyperparameters match the chosen_hyper_parameters
        for param_name, param_value in optimization_result.chosen_hyper_parameters.items():
            assert hasattr(
                evaluated_policy, param_name
            ), f"Policy should have hyperparameter attribute '{param_name}'"
            assert (
                getattr(evaluated_policy, param_name) == param_value
            ), f"Policy hyperparameter '{param_name}' should be {param_value}, got {getattr(evaluated_policy, param_name)}"

        # Verify that all chosen hyperparameters are present in the policy
        assert len(optimization_result.chosen_hyper_parameters) > 0
        for param_name in optimization_result.chosen_hyper_parameters:
            assert hasattr(
                evaluated_policy, param_name
            ), f"Policy missing hyperparameter '{param_name}' from optimization result"

        # Verify evaluation parameters
        assert eval_run_params.num_episodes == workflow.evaluation_episodes
        assert eval_run_params.num_steps == workflow.evaluation_steps
        assert eval_run_params.environment is env
        assert eval_run_params.belief is belief

    @patch(
        "POMDPPlanners.simulations.workflows.hyperparameter_tuning_evaluation_workflows.POMDPSimulator"
    )
    @patch(
        "POMDPPlanners.simulations.workflows.hyperparameter_tuning_evaluation_workflows.HyperParameterOptimizer"
    )
    def test_workflow_uses_optimized_config_for_evaluation_multiple_policies(
        self, mock_optimizer_class, mock_simulator_class, temp_cache_dir
    ):
        """Test that multiple optimized policies are correctly used in evaluation.

        Purpose: Validates that when multiple policies are optimized, all are correctly
                 passed to evaluation with their optimized hyperparameters

        Given: Optimization returns multiple policies with different hyperparameters
        When: optimize_and_evaluate executes with multiple configs
        Then: All optimized policies are passed to evaluation with correct hyperparameters

        Test type: integration
        """
        from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        from POMDPPlanners.core.belief import get_initial_belief
        from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        from POMDPPlanners.core.simulation.hyperparameter_tuning import (
            HyperParameterRunParams,
            HyperParamPlannerConfig,
            HyperParameterOptimizationDirection,
            OptimizedPolicyResult,
        )

        # Create environment and belief
        env = TigerPOMDP(discount_factor=0.95)
        belief = get_initial_belief(env, n_particles=10)

        # Create workflow
        workflow = OptimizationEvaluationLocalWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="test_multiple_policies",
            optimization_n_jobs=1,
        )

        # Create two configs
        config1 = HyperParameterRunParams(
            environment=env,
            belief=belief,
            hyper_param_planner_config=HyperParamPlannerConfig(
                policy_cls=POMCP,
                hyper_parameters=[NumericalHyperParameter(0.1, 2.0, "exploration_constant")],
                constant_parameters={"discount_factor": 0.95, "name": "POMCP1"},
            ),
            num_episodes=3,
            num_steps=6,
            n_trials=2,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
        )

        config2 = HyperParameterRunParams(
            environment=env,
            belief=belief,
            hyper_param_planner_config=HyperParamPlannerConfig(
                policy_cls=POMCP,
                hyper_parameters=[NumericalHyperParameter(0.1, 2.0, "exploration_constant")],
                constant_parameters={"discount_factor": 0.95, "name": "POMCP2"},
            ),
            num_episodes=3,
            num_steps=6,
            n_trials=2,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
        )

        # Create two policies with different hyperparameters
        chosen_hyperparams_1 = {"exploration_constant": 1.2, "n_simulations": 15}
        optimized_policy_1 = POMCP(
            environment=env, discount_factor=0.95, depth=5, name="POMCP1", **chosen_hyperparams_1
        )

        chosen_hyperparams_2 = {"exploration_constant": 2.1, "n_simulations": 25}
        optimized_policy_2 = POMCP(
            environment=env, discount_factor=0.95, depth=5, name="POMCP2", **chosen_hyperparams_2
        )

        # Create optimization results
        optimization_result_1 = OptimizedPolicyResult(
            environment=env,
            policy=optimized_policy_1,
            chosen_hyper_parameters=chosen_hyperparams_1,
            num_episodes=3,
            num_steps=6,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
            optimized_metric_values={"average_return": 12.0},
        )

        optimization_result_2 = OptimizedPolicyResult(
            environment=env,
            policy=optimized_policy_2,
            chosen_hyper_parameters=chosen_hyperparams_2,
            num_episodes=3,
            num_steps=6,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
            optimized_metric_values={"average_return": 15.0},
        )

        # Mock optimizer to return both results
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.optimize.return_value = [optimization_result_1, optimization_result_2]

        # Mock simulator
        mock_simulator = Mock()
        mock_simulator_class.return_value.__enter__ = Mock(return_value=mock_simulator)
        mock_simulator_class.return_value.__exit__ = Mock(return_value=None)

        mock_evaluation_results = {"TigerPOMDP": {"POMCP1": [Mock()], "POMCP2": [Mock()]}}
        mock_evaluation_statistics = pd.DataFrame()
        mock_simulator.compare_multiple_environments_policies.return_value = (
            mock_evaluation_results,
            mock_evaluation_statistics,
        )

        # Execute workflow
        workflow.optimize_and_evaluate([config1, config2])

        # Capture evaluation arguments
        eval_call_args = mock_simulator.compare_multiple_environments_policies.call_args
        eval_environment_run_params = eval_call_args[1]["environment_run_params"]

        # Verify we have one environment run params (both policies grouped by environment)
        assert len(eval_environment_run_params) == 1
        eval_run_params = eval_environment_run_params[0]

        # Verify both policies are present
        assert len(eval_run_params.policies) == 2

        # Verify first policy matches optimization result
        evaluated_policy_1 = eval_run_params.policies[0]
        assert evaluated_policy_1 is optimized_policy_1
        for param_name, param_value in optimization_result_1.chosen_hyper_parameters.items():
            assert getattr(evaluated_policy_1, param_name) == param_value

        # Verify second policy matches optimization result
        evaluated_policy_2 = eval_run_params.policies[1]
        assert evaluated_policy_2 is optimized_policy_2
        for param_name, param_value in optimization_result_2.chosen_hyper_parameters.items():
            assert getattr(evaluated_policy_2, param_name) == param_value
