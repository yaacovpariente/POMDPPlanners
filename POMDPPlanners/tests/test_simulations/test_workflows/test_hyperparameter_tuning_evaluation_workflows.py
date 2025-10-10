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

from POMDPPlanners.simulations.workflows.hyperparameter_tuning_evaluation_workflows import (
    OptimizationEvaluationWorkflow,
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
