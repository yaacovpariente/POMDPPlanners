"""Tests for hyperparameter_tuning_evaluation_workflows module.

This module tests the workflow classes for running hyperparameter optimization
followed by policy evaluation in different execution environments.
"""

import random
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from POMDPPlanners.simulations.hyperparameter_tuning_evaluation_workflows import (
    OptimizationEvaluationWorkflow,
    OptimizationEvaluationLocalWorkflow,
    OptimizationEvaluationPBSWorkflow,
)
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    JoblibConfig,
    PBSConfig,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParamPlannerConfigGenerator,
    HyperParamPlannerConfig,
    NumericalHyperParameter,
)
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

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
        "particles": 30,
        "num_episodes": 10,
        "num_steps": 20,
        "n_trials": 100,
        "evaluation_episodes": 3,
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
        assert workflow.particles == 30
        assert workflow.num_episodes == 10
        assert workflow.num_steps == 20
        assert workflow.n_trials == 100
        assert workflow.evaluation_episodes == 3
        assert workflow.evaluation_steps == 6
        assert workflow.evaluation_n_jobs == 1
        assert workflow.optimization_n_jobs == -1
        assert workflow.confidence_interval_level == 0.95
        assert workflow.alpha == 0.05
        assert workflow.debug is False
        assert workflow.verbose is True
        assert workflow.cache_visualizations is True

    def test_initialization_with_custom_parameters(self, sample_workflow_params):
        """Test initialization with custom parameters.

        Purpose: Validates that LocalWorkflow accepts custom parameter values

        Given: Custom workflow parameters
        When: Creating LocalWorkflow instance with custom values
        Then: All attributes match provided custom values

        Test type: unit
        """
        workflow = OptimizationEvaluationLocalWorkflow(
            optimization_n_jobs=4, **sample_workflow_params
        )

        assert workflow.cache_dir == sample_workflow_params["cache_dir"]
        assert workflow.experiment_name == sample_workflow_params["experiment_name"]
        assert workflow.particles == sample_workflow_params["particles"]
        assert workflow.num_episodes == sample_workflow_params["num_episodes"]
        assert workflow.optimization_n_jobs == 4

    def test_get_task_manager_config_hyperparameter_returns_joblib_config(
        self, sample_workflow_params
    ):
        """Test that hyperparameter task manager config is JoblibConfig.

        Purpose: Validates correct task manager type for hyperparameter optimization

        Given: LocalWorkflow instance with optimization_n_jobs=-1
        When: Calling _get_task_manager_config_hyperparameter
        Then: Returns JoblibConfig with correct n_jobs value

        Test type: unit
        """
        workflow = OptimizationEvaluationLocalWorkflow(
            optimization_n_jobs=-1, **sample_workflow_params
        )

        config = workflow._get_task_manager_config_hyperparameter()

        assert isinstance(config, JoblibConfig)
        assert config.n_jobs == -1

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

    @patch(
        "POMDPPlanners.simulations.hyperparameter_tuning_evaluation_workflows.complete_environments_and_benchmarks_hyperparameter_optimization_configs"
    )
    @patch.object(OptimizationEvaluationLocalWorkflow, "optimize_and_evaluate")
    def test_run_comprehensive_benchmark_calls_optimize_and_evaluate(
        self, mock_optimize_and_evaluate, mock_config_generator, sample_workflow_params
    ):
        """Test run_comprehensive_benchmark calls optimize_and_evaluate.

        Purpose: Validates that run_comprehensive_benchmark orchestrates optimization

        Given: LocalWorkflow instance and mocked generator
        When: Calling run_comprehensive_benchmark
        Then: optimize_and_evaluate is called with generated configs

        Test type: unit
        """
        workflow = OptimizationEvaluationLocalWorkflow(**sample_workflow_params)
        mock_gen = Mock()
        mock_configs = [Mock(), Mock()]
        mock_config_generator.return_value = mock_configs
        mock_optimize_and_evaluate.return_value = ({}, Mock())

        workflow.run_comprehensive_benchmark(mock_gen)

        mock_config_generator.assert_called_once()
        mock_optimize_and_evaluate.assert_called_once_with(configs=mock_configs)

    @patch(
        "POMDPPlanners.simulations.hyperparameter_tuning_evaluation_workflows.complete_environments_and_benchmarks_hyperparameter_optimization_configs"
    )
    @patch(
        "POMDPPlanners.simulations.hyperparameter_tuning_evaluation_workflows.AverageReturnParameterToOptimizeMapper"
    )
    @patch.object(OptimizationEvaluationLocalWorkflow, "optimize_and_evaluate")
    def test_run_comprehensive_benchmark_uses_average_return_mapper(
        self, mock_optimize, mock_mapper_class, mock_config_gen, sample_workflow_params
    ):
        """Test run_comprehensive_benchmark uses AverageReturnParameterToOptimizeMapper.

        Purpose: Validates correct parameter mapper for comprehensive benchmark

        Given: LocalWorkflow instance
        When: Calling run_comprehensive_benchmark
        Then: AverageReturnParameterToOptimizeMapper is instantiated and used

        Test type: unit
        """
        workflow = OptimizationEvaluationLocalWorkflow(**sample_workflow_params)
        mock_gen = Mock()
        mock_config_gen.return_value = []
        mock_optimize.return_value = ({}, Mock())

        workflow.run_comprehensive_benchmark(mock_gen)

        mock_mapper_class.assert_called_once()
        call_args = mock_config_gen.call_args
        assert call_args.kwargs["parameter_to_optimize_mapper"] == mock_mapper_class.return_value

    @patch(
        "POMDPPlanners.simulations.hyperparameter_tuning_evaluation_workflows.complete_environments_and_benchmarks_hyperparameter_optimization_configs"
    )
    @patch(
        "POMDPPlanners.simulations.hyperparameter_tuning_evaluation_workflows.RiskAverseParameterToOptimizeMapper"
    )
    @patch.object(OptimizationEvaluationLocalWorkflow, "optimize_and_evaluate")
    def test_run_risk_averse_benchmark_uses_risk_averse_mapper(
        self, mock_optimize, mock_mapper_class, mock_config_gen, sample_workflow_params
    ):
        """Test run_risk_averse_benchmark uses RiskAverseParameterToOptimizeMapper.

        Purpose: Validates correct parameter mapper for risk-averse benchmark

        Given: LocalWorkflow instance
        When: Calling run_risk_averse_benchmark
        Then: RiskAverseParameterToOptimizeMapper is instantiated and used

        Test type: unit
        """
        workflow = OptimizationEvaluationLocalWorkflow(**sample_workflow_params)
        mock_gen = Mock()
        mock_config_gen.return_value = []
        mock_optimize.return_value = ({}, Mock())

        workflow.run_comprehensive_risk_averse_benchmark(mock_gen)

        mock_mapper_class.assert_called_once()
        call_args = mock_config_gen.call_args
        assert call_args.kwargs["parameter_to_optimize_mapper"] == mock_mapper_class.return_value

    @patch(
        "POMDPPlanners.simulations.hyperparameter_tuning_evaluation_workflows.complete_environments_and_benchmarks_hyperparameter_optimization_configs"
    )
    @patch.object(OptimizationEvaluationLocalWorkflow, "optimize_and_evaluate")
    def test_run_risk_averse_benchmark_calls_optimize_and_evaluate(
        self, mock_optimize_and_evaluate, mock_config_generator, sample_workflow_params
    ):
        """Test run_risk_averse_benchmark calls optimize_and_evaluate.

        Purpose: Validates that run_risk_averse_benchmark orchestrates optimization

        Given: LocalWorkflow instance and mocked generator
        When: Calling run_risk_averse_benchmark
        Then: optimize_and_evaluate is called with generated configs

        Test type: unit
        """
        workflow = OptimizationEvaluationLocalWorkflow(**sample_workflow_params)
        mock_gen = Mock()
        mock_configs = [Mock(), Mock()]
        mock_config_generator.return_value = mock_configs
        mock_optimize_and_evaluate.return_value = ({}, Mock())

        workflow.run_comprehensive_risk_averse_benchmark(mock_gen)

        mock_config_generator.assert_called_once()
        mock_optimize_and_evaluate.assert_called_once_with(configs=mock_configs)


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

    def test_initialization_with_custom_pbs_parameters(self, temp_cache_dir):
        """Test initialization with custom PBS parameters.

        Purpose: Validates that PBSWorkflow accepts custom PBS configuration

        Given: Custom PBS parameters
        When: Creating PBSWorkflow instance with custom PBS values
        Then: All PBS attributes match provided custom values

        Test type: unit
        """
        workflow = OptimizationEvaluationPBSWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Custom_PBS",
            queue="long",
            n_workers=10,
            cores=4,
            memory="16GB",
            processes=2,
            walltime="12:00:00",
            job_extra=["#PBS -m abe"],
        )

        assert workflow.queue == "long"
        assert workflow.n_workers == 10
        assert workflow.cores == 4
        assert workflow.memory == "16GB"
        assert workflow.processes == 2
        assert workflow.walltime == "12:00:00"
        assert workflow.job_extra == ["#PBS -m abe"]

    def test_get_task_manager_config_hyperparameter_returns_pbs_config(self, temp_cache_dir):
        """Test that hyperparameter task manager config is PBSConfig.

        Purpose: Validates correct task manager type for PBS hyperparameter optimization

        Given: PBSWorkflow instance with custom PBS settings
        When: Calling _get_task_manager_config_hyperparameter
        Then: Returns PBSConfig with correct PBS parameters

        Test type: unit
        """
        workflow = OptimizationEvaluationPBSWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Test",
            queue="long",
            n_workers=8,
            cores=2,
        )

        config = workflow._get_task_manager_config_hyperparameter()

        assert isinstance(config, PBSConfig)
        assert config.queue == "long"
        assert config.n_workers == 8
        assert config.cores == 2

    def test_get_task_manager_config_evaluation_returns_pbs_config(self, temp_cache_dir):
        """Test that evaluation task manager config is PBSConfig.

        Purpose: Validates correct task manager type for PBS evaluation phase

        Given: PBSWorkflow instance with custom PBS settings
        When: Calling _get_task_manager_config_evaluation
        Then: Returns PBSConfig with correct PBS parameters

        Test type: unit
        """
        workflow = OptimizationEvaluationPBSWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Test",
            memory="32GB",
            walltime="24:00:00",
        )

        config = workflow._get_task_manager_config_evaluation()

        assert isinstance(config, PBSConfig)
        assert config.memory == "32GB"
        assert config.walltime == "24:00:00"

    @patch(
        "POMDPPlanners.simulations.hyperparameter_tuning_evaluation_workflows.complete_environments_and_benchmarks_hyperparameter_optimization_configs"
    )
    @patch.object(OptimizationEvaluationPBSWorkflow, "optimize_and_evaluate")
    def test_run_comprehensive_benchmark_calls_optimize_and_evaluate(
        self, mock_optimize_and_evaluate, mock_config_generator, temp_cache_dir
    ):
        """Test PBS run_comprehensive_benchmark calls optimize_and_evaluate.

        Purpose: Validates that PBS workflow orchestrates optimization correctly

        Given: PBSWorkflow instance and mocked generator
        When: Calling run_comprehensive_benchmark
        Then: optimize_and_evaluate is called with generated configs

        Test type: unit
        """
        workflow = OptimizationEvaluationPBSWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Test",
        )
        mock_gen = Mock()
        mock_configs = [Mock(), Mock()]
        mock_config_generator.return_value = mock_configs
        mock_optimize_and_evaluate.return_value = ({}, Mock())

        workflow.run_comprehensive_benchmark(mock_gen)

        mock_config_generator.assert_called_once()
        mock_optimize_and_evaluate.assert_called_once_with(configs=mock_configs)

    @patch(
        "POMDPPlanners.simulations.hyperparameter_tuning_evaluation_workflows.complete_environments_and_benchmarks_hyperparameter_optimization_configs"
    )
    @patch.object(OptimizationEvaluationPBSWorkflow, "optimize_and_evaluate")
    def test_run_risk_averse_benchmark_calls_optimize_and_evaluate(
        self, mock_optimize_and_evaluate, mock_config_generator, temp_cache_dir
    ):
        """Test PBS run_risk_averse_benchmark calls optimize_and_evaluate.

        Purpose: Validates that PBS risk-averse workflow orchestrates optimization

        Given: PBSWorkflow instance and mocked generator
        When: Calling run_risk_averse_benchmark
        Then: optimize_and_evaluate is called with generated configs

        Test type: unit
        """
        workflow = OptimizationEvaluationPBSWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Test",
        )
        mock_gen = Mock()
        mock_configs = [Mock()]
        mock_config_generator.return_value = mock_configs
        mock_optimize_and_evaluate.return_value = ({}, Mock())

        workflow.run_risk_averse_benchmark(mock_gen)

        mock_config_generator.assert_called_once()
        mock_optimize_and_evaluate.assert_called_once_with(configs=mock_configs)


class TestWorkflowParameterPropagation:
    """Test that workflow parameters are correctly propagated to internal methods."""

    @patch(
        "POMDPPlanners.simulations.hyperparameter_tuning_evaluation_workflows.complete_environments_and_benchmarks_hyperparameter_optimization_configs"
    )
    @patch.object(OptimizationEvaluationLocalWorkflow, "optimize_and_evaluate")
    def test_workflow_parameters_passed_to_config_generator(
        self, mock_optimize, mock_config_gen, temp_cache_dir
    ):
        """Test that workflow parameters are passed to config generator.

        Purpose: Validates parameter propagation from workflow to config generator

        Given: LocalWorkflow with custom particles, episodes, steps, trials
        When: Calling run_comprehensive_benchmark
        Then: Config generator receives correct parameter values

        Test type: unit
        """
        workflow = OptimizationEvaluationLocalWorkflow(
            cache_dir=temp_cache_dir,
            experiment_name="Test",
            particles=50,
            num_episodes=15,
            num_steps=25,
            n_trials=200,
        )
        mock_gen = Mock()
        mock_config_gen.return_value = []
        mock_optimize.return_value = ({}, Mock())

        workflow.run_comprehensive_benchmark(mock_gen)

        call_kwargs = mock_config_gen.call_args.kwargs
        assert call_kwargs["particles"] == 50
        assert call_kwargs["num_episodes"] == 15
        assert call_kwargs["num_steps"] == 25
        assert call_kwargs["n_trials"] == 200


class IntegerActionSampler(ActionSampler):
    """Action sampler that returns integer actions for compatibility with environments."""

    def __init__(self, actions):
        self.actions = actions

    def sample(self, belief_node=None):
        import random

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
