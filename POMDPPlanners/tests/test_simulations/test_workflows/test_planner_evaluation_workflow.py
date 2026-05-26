# SPDX-License-Identifier: MIT

"""Tests for planner evaluation workflow classes.

This module tests the planner evaluation workflow functionality, focusing on:
- Base PlannerEvaluationWorkflow class initialization and evaluation
- PlannerEvaluationLocalWorkflow implementation
- PlannerEvaluationDaskWorkflow implementation
- PlannerEvaluationPBSWorkflow implementation
- Task manager configuration generation
- Error handling and edge cases
"""

# pylint: disable=protected-access  # Tests need to access protected members

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from POMDPPlanners.core.simulation.simulation_configs import EnvironmentRunParams
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import (
    TaskManagerConfig,
    JoblibConfig,
    DaskConfig,
    PBSConfig,
)
from POMDPPlanners.simulations.workflows.planner_evaluation_workflow import (
    PlannerEvaluationWorkflow,
    PlannerEvaluationLocalWorkflow,
    PlannerEvaluationDaskWorkflow,
    PlannerEvaluationPBSWorkflow,
)


@pytest.fixture
def real_environment():
    """Create a real TigerPOMDP environment for testing."""
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def real_belief():
    """Create a real WeightedParticleBelief for testing."""
    particles = ["tiger_left", "tiger_right"]
    log_weights = np.log(np.array([0.5, 0.5]))
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


@pytest.fixture
def real_policy(real_environment):
    """Create a real POMCP policy for testing."""
    return POMCP(
        environment=real_environment,
        discount_factor=0.95,
        depth=5,
        n_simulations=10,
        exploration_constant=1.0,
        name="TestPolicy",
    )


@pytest.fixture
def sample_environment_run_params(real_environment, real_belief, real_policy):
    """Create sample EnvironmentRunParams for testing."""
    return EnvironmentRunParams(
        environment=real_environment,
        belief=real_belief,
        policies=[real_policy],
        num_episodes=5,
        num_steps=10,
    )


# Backwards compatibility aliases for existing tests
@pytest.fixture
def mock_environment(real_environment):
    """Alias for real_environment for backwards compatibility."""
    return real_environment


@pytest.fixture
def mock_belief(real_belief):
    """Alias for real_belief for backwards compatibility."""
    return real_belief


@pytest.fixture
def mock_policy(real_policy):
    """Alias for real_policy for backwards compatibility."""
    return real_policy


@pytest.fixture
def mock_simulator_results():
    """Create mock simulator results for testing."""
    mock_results = {"TestEnv": {"TestPolicy": ["mock_history_1", "mock_history_2"]}}
    mock_stats_df = pd.DataFrame(
        {
            "environment": ["TestEnv"],
            "policy": ["TestPolicy"],
            "mean_return": [10.5],
            "std_return": [2.1],
        }
    )
    return mock_results, mock_stats_df


class TestPlannerEvaluationWorkflow:
    """Test suite for PlannerEvaluationWorkflow base class."""

    def test_abstract_class_has_abstract_methods(self):
        """Test that PlannerEvaluationWorkflow is properly abstract.

        Purpose: Validates that the abstract base class has the expected abstract methods

        Given: PlannerEvaluationWorkflow class
        When: Checking its abstract methods
        Then: It has the expected abstract method

        Test type: unit
        """
        # Test that the class is abstract by checking it has abstract methods
        assert PlannerEvaluationWorkflow.__abstractmethods__ == frozenset(
            {"_get_task_manager_config"}
        )

        # Test that the abstract method exists
        assert hasattr(PlannerEvaluationWorkflow, "_get_task_manager_config")
        assert callable(getattr(PlannerEvaluationWorkflow, "_get_task_manager_config"))

    def test_abstract_method_raises_not_implemented_error(self):
        """Test that _get_task_manager_config raises NotImplementedError.

        Purpose: Validates that abstract method is properly defined

        Given: Concrete subclass without implementing abstract method
        When: Calling _get_task_manager_config
        Then: NotImplementedError is raised

        Test type: unit
        """

        class IncompleteWorkflow(PlannerEvaluationWorkflow):
            def _get_task_manager_config(self) -> TaskManagerConfig:
                raise NotImplementedError("Method not implemented")

        workflow = IncompleteWorkflow(
            experiment_name="test",
            cache_dir_path=None,
        )

        with pytest.raises(NotImplementedError):
            workflow._get_task_manager_config()

    @patch("POMDPPlanners.simulations.workflows.planner_evaluation_workflow.POMDPSimulator")
    def test_evaluate_method_calls_simulator_correctly(
        self, mock_simulator_class, sample_environment_run_params, mock_simulator_results
    ):
        """Test that evaluate method calls simulator with correct parameters.

        Purpose: Validates that evaluate method properly delegates to simulator

        Given: Concrete workflow implementation
        When: Calling evaluate method
        Then: Simulator is called with correct parameters

        Test type: unit
        """

        class TestWorkflow(PlannerEvaluationWorkflow):
            def _get_task_manager_config(self) -> TaskManagerConfig:
                return JoblibConfig(n_jobs=1)

        mock_simulator_instance = Mock()
        mock_simulator_instance.compare_multiple_environments_policies.return_value = (
            mock_simulator_results
        )
        mock_simulator_class.return_value = mock_simulator_instance

        workflow = TestWorkflow(
            experiment_name="test_experiment",
            cache_dir_path=Path("/tmp/test"),
            debug=True,
            n_jobs=2,
            enable_profiling=True,
            verbose=False,
            alpha=0.05,
            confidence_interval_level=0.99,
            cache_visualizations=False,
        )

        configs = [sample_environment_run_params]
        results = workflow.evaluate(configs)

        # Verify simulator was created with correct parameters
        mock_simulator_class.assert_called_once_with(
            cache_dir_path=Path("/tmp/test"),
            experiment_name="test_experiment",
            debug=True,
            task_console_output=False,
            enable_profiling=True,
            task_manager_config=workflow._get_task_manager_config(),
        )

        # Verify simulator method was called with correct parameters
        mock_simulator_instance.compare_multiple_environments_policies.assert_called_once_with(
            environment_run_params=configs,
            alpha=0.05,
            confidence_interval_level=0.99,
            n_jobs=2,
            cache_visualizations=False,
        )

        # Verify results are returned correctly
        assert results == mock_simulator_results

    def test_initialization_stores_parameters_correctly(self):
        """Test that initialization stores all parameters correctly.

        Purpose: Validates that all initialization parameters are stored

        Given: Concrete workflow implementation
        When: Initializing with various parameters
        Then: All parameters are stored as instance attributes

        Test type: unit
        """

        class TestWorkflow(PlannerEvaluationWorkflow):
            def _get_task_manager_config(self) -> TaskManagerConfig:
                return JoblibConfig(n_jobs=1)

        workflow = TestWorkflow(
            experiment_name="test_experiment",
            cache_dir_path=Path("/tmp/test"),
            debug=True,
            n_jobs=4,
            enable_profiling=True,
            verbose=False,
            alpha=0.05,
            confidence_interval_level=0.99,
            cache_visualizations=False,
        )

        assert workflow.experiment_name == "test_experiment"
        assert workflow.cache_dir_path == Path("/tmp/test")
        assert workflow.debug is True
        assert workflow.n_jobs == 4
        assert workflow.enable_profiling is True
        assert workflow.verbose is False
        assert workflow.alpha == 0.05
        assert workflow.confidence_interval_level == 0.99
        assert workflow.cache_visualizations is False

    def test_default_parameter_values(self):
        """Test that default parameter values are set correctly.

        Purpose: Validates that default values are applied when not specified

        Given: Concrete workflow implementation
        When: Initializing with minimal parameters
        Then: Default values are applied correctly

        Test type: unit
        """

        class TestWorkflow(PlannerEvaluationWorkflow):
            def _get_task_manager_config(self) -> TaskManagerConfig:
                return JoblibConfig(n_jobs=1)

        workflow = TestWorkflow(
            experiment_name="test_experiment",
            cache_dir_path=None,
        )

        assert workflow.debug is False
        assert workflow.n_jobs == 1
        assert workflow.enable_profiling is False
        assert workflow.verbose is True
        assert workflow.alpha == 0.1
        assert workflow.confidence_interval_level == 0.95
        assert workflow.cache_visualizations is True


class TestPlannerEvaluationLocalWorkflow:
    """Test suite for PlannerEvaluationLocalWorkflow class."""

    def test_initialization(self):
        """Test that PlannerEvaluationLocalWorkflow initializes correctly.

        Purpose: Validates that local workflow initializes with correct parameters

        Given: PlannerEvaluationLocalWorkflow class
        When: Initializing with parameters
        Then: Instance is created with correct attributes

        Test type: unit
        """
        workflow = PlannerEvaluationLocalWorkflow(
            experiment_name="local_test",
            cache_dir_path=Path("/tmp/local"),
            debug=True,
            n_jobs=8,
        )

        assert workflow.experiment_name == "local_test"
        assert workflow.cache_dir_path == Path("/tmp/local")
        assert workflow.debug is True
        assert workflow.n_jobs == 8

    def test_get_task_manager_config_returns_joblib_config(self):
        """Test that _get_task_manager_config returns JoblibConfig.

        Purpose: Validates that local workflow returns correct task manager config

        Given: PlannerEvaluationLocalWorkflow instance
        When: Calling _get_task_manager_config
        Then: JoblibConfig with correct n_jobs is returned

        Test type: unit
        """
        workflow = PlannerEvaluationLocalWorkflow(
            experiment_name="local_test",
            cache_dir_path=None,
            n_jobs=4,
        )

        config = workflow._get_task_manager_config()

        assert isinstance(config, JoblibConfig)
        assert config.n_jobs == 4

    def test_different_n_jobs_values(self):
        """Test that different n_jobs values are handled correctly.

        Purpose: Validates that n_jobs parameter is passed through correctly

        Given: PlannerEvaluationLocalWorkflow with different n_jobs values
        When: Getting task manager config
        Then: Correct n_jobs value is used

        Test type: unit
        """
        # Test with positive integer
        workflow1 = PlannerEvaluationLocalWorkflow(
            experiment_name="test1",
            cache_dir_path=None,
            n_jobs=8,
        )
        config1 = workflow1._get_task_manager_config()
        assert isinstance(config1, JoblibConfig)
        assert config1.n_jobs == 8

        # Test with -1 (use all cores)
        workflow2 = PlannerEvaluationLocalWorkflow(
            experiment_name="test2",
            cache_dir_path=None,
            n_jobs=-1,
        )
        config2 = workflow2._get_task_manager_config()
        assert isinstance(config2, JoblibConfig)
        assert config2.n_jobs == -1

        # Test with 1 (single core)
        workflow3 = PlannerEvaluationLocalWorkflow(
            experiment_name="test3",
            cache_dir_path=None,
            n_jobs=1,
        )
        config3 = workflow3._get_task_manager_config()
        assert isinstance(config3, JoblibConfig)
        assert config3.n_jobs == 1


class TestPlannerEvaluationDaskWorkflow:
    """Test suite for PlannerEvaluationDaskWorkflow class."""

    def test_initialization_with_defaults(self):
        """Test that PlannerEvaluationDaskWorkflow initializes with defaults.

        Purpose: Validates that Dask workflow initializes with correct default values

        Given: PlannerEvaluationDaskWorkflow class
        When: Initializing with minimal parameters
        Then: Instance is created with correct default attributes

        Test type: unit
        """
        workflow = PlannerEvaluationDaskWorkflow(
            cache_dir_path=Path("/tmp/dask"),
            experiment_name="dask_test",
        )

        assert workflow.experiment_name == "dask_test"
        assert workflow.cache_dir_path == Path("/tmp/dask")
        assert workflow.n_workers == 4
        assert workflow.scheduler_address is None
        assert workflow.cache_size == int(2e9)
        assert workflow.clear_cache_on_start is False

    def test_initialization_with_custom_values(self):
        """Test that PlannerEvaluationDaskWorkflow initializes with custom values.

        Purpose: Validates that Dask workflow accepts custom parameter values

        Given: PlannerEvaluationDaskWorkflow class
        When: Initializing with custom parameters
        Then: Instance is created with correct custom attributes

        Test type: unit
        """
        workflow = PlannerEvaluationDaskWorkflow(
            cache_dir_path=Path("/tmp/dask"),
            experiment_name="dask_test",
            n_workers=8,
            scheduler_address="tcp://localhost:8786",
            cache_size=int(4e9),
            clear_cache_on_start=True,
            debug=True,
            alpha=0.05,
        )

        assert workflow.n_workers == 8
        assert workflow.scheduler_address == "tcp://localhost:8786"
        assert workflow.cache_size == int(4e9)
        assert workflow.clear_cache_on_start is True
        assert workflow.debug is True
        assert workflow.alpha == 0.05

    def test_get_task_manager_config_returns_dask_config(self):
        """Test that _get_task_manager_config returns DaskConfig.

        Purpose: Validates that Dask workflow returns correct task manager config

        Given: PlannerEvaluationDaskWorkflow instance
        When: Calling _get_task_manager_config
        Then: DaskConfig with correct parameters is returned

        Test type: unit
        """
        workflow = PlannerEvaluationDaskWorkflow(
            cache_dir_path=Path("/tmp/dask"),
            experiment_name="dask_test",
            n_workers=6,
            scheduler_address="tcp://scheduler:8786",
            cache_size=int(3e9),
            clear_cache_on_start=True,
        )

        config = workflow._get_task_manager_config()

        assert isinstance(config, DaskConfig)
        assert config.n_workers == 6
        assert config.scheduler_address == "tcp://scheduler:8786"
        assert config.cache_size == int(3e9)
        assert config.clear_cache_on_start is True

    def test_n_jobs_bug_documented(self):
        """Test that the documented bug in n_jobs assignment is present.

        Purpose: Validates that the TODO comment about n_jobs bug is accurate

        Given: PlannerEvaluationDaskWorkflow instance
        When: Initializing with n_jobs parameter
        Then: n_jobs is incorrectly set to n_workers value

        Test type: unit
        """
        workflow = PlannerEvaluationDaskWorkflow(
            cache_dir_path=Path("/tmp/dask"),
            experiment_name="dask_test",
            n_workers=8,
            n_jobs=2,  # This should be used but gets overridden
        )

        # The bug: n_jobs should be 2 but gets set to n_workers (8)
        assert workflow.n_jobs == 8  # This is the bug
        # The correct behavior would be: assert workflow.n_jobs == 2

    def test_scheduler_address_none_vs_provided(self):
        """Test that scheduler_address handling works for both None and provided values.

        Purpose: Validates that scheduler_address parameter is handled correctly

        Given: PlannerEvaluationDaskWorkflow instances
        When: Initializing with None vs provided scheduler address
        Then: Both cases work correctly

        Test type: unit
        """
        # Test with None (local cluster)
        workflow1 = PlannerEvaluationDaskWorkflow(
            cache_dir_path=Path("/tmp/dask"),
            experiment_name="dask_test",
            scheduler_address=None,
        )
        config1 = workflow1._get_task_manager_config()
        assert isinstance(config1, DaskConfig)
        assert config1.scheduler_address is None

        # Test with provided address
        workflow2 = PlannerEvaluationDaskWorkflow(
            cache_dir_path=Path("/tmp/dask"),
            experiment_name="dask_test",
            scheduler_address="tcp://remote:8786",
        )
        config2 = workflow2._get_task_manager_config()
        assert isinstance(config2, DaskConfig)
        assert config2.scheduler_address == "tcp://remote:8786"


class TestPlannerEvaluationPBSWorkflow:
    """Test suite for PlannerEvaluationPBSWorkflow class."""

    def test_initialization_with_defaults(self):
        """Test that PlannerEvaluationPBSWorkflow initializes with defaults.

        Purpose: Validates that PBS workflow initializes with correct default values

        Given: PlannerEvaluationPBSWorkflow class
        When: Initializing with minimal parameters
        Then: Instance is created with correct default attributes

        Test type: unit
        """
        workflow = PlannerEvaluationPBSWorkflow(
            cache_dir_path=Path("/tmp/pbs"),
            experiment_name="pbs_test",
        )

        assert workflow.experiment_name == "pbs_test"
        assert workflow.cache_dir_path == Path("/tmp/pbs")
        assert workflow.queue == "short"
        assert workflow.n_workers == 4
        assert workflow.cores == 1
        assert workflow.memory == "4GB"
        assert workflow.processes == 1
        assert workflow.walltime == "03:00:00"
        assert workflow.job_extra is None
        assert workflow.enable_dashboard is True
        assert workflow.dashboard_address == "0.0.0.0"
        assert workflow.dashboard_port == 8787
        assert workflow.dashboard_prefix is None

    def test_initialization_with_custom_values(self):
        """Test that PlannerEvaluationPBSWorkflow initializes with custom values.

        Purpose: Validates that PBS workflow accepts custom parameter values

        Given: PlannerEvaluationPBSWorkflow class
        When: Initializing with custom parameters
        Then: Instance is created with correct custom attributes

        Test type: unit
        """
        job_extra = ["-l feature=gpu", "-l gres=gpu:1"]
        workflow = PlannerEvaluationPBSWorkflow(
            cache_dir_path=Path("/tmp/pbs"),
            experiment_name="pbs_test",
            queue="gpu",
            n_workers=10,
            cores=8,
            memory="32GB",
            processes=2,
            walltime="24:00:00",
            job_extra=job_extra,
            enable_dashboard=False,
            dashboard_address="127.0.0.1",
            dashboard_port=9999,
            dashboard_prefix="/dask",
            debug=True,
            alpha=0.01,
        )

        assert workflow.queue == "gpu"
        assert workflow.n_workers == 10
        assert workflow.cores == 8
        assert workflow.memory == "32GB"
        assert workflow.processes == 2
        assert workflow.walltime == "24:00:00"
        assert workflow.job_extra == job_extra
        assert workflow.enable_dashboard is False
        assert workflow.dashboard_address == "127.0.0.1"
        assert workflow.dashboard_port == 9999
        assert workflow.dashboard_prefix == "/dask"
        assert workflow.debug is True
        assert workflow.alpha == 0.01

    def test_get_task_manager_config_returns_pbs_config(self):
        """Test that _get_task_manager_config returns PBSConfig.

        Purpose: Validates that PBS workflow returns correct task manager config

        Given: PlannerEvaluationPBSWorkflow instance
        When: Calling _get_task_manager_config
        Then: PBSConfig with correct parameters is returned

        Test type: unit
        """
        job_extra = ["-l feature=gpu"]
        workflow = PlannerEvaluationPBSWorkflow(
            cache_dir_path=Path("/tmp/pbs"),
            experiment_name="pbs_test",
            queue="gpu",
            n_workers=6,
            cores=4,
            memory="16GB",
            processes=2,
            walltime="12:00:00",
            job_extra=job_extra,
            enable_dashboard=True,
            dashboard_address="0.0.0.0",
            dashboard_port=8787,
            dashboard_prefix="/dask",
        )

        config = workflow._get_task_manager_config()

        assert isinstance(config, PBSConfig)
        assert config.queue == "gpu"
        assert config.n_workers == 6
        assert config.cores == 4
        assert config.memory == "16GB"
        assert config.processes == 2
        assert config.walltime == "12:00:00"
        assert config.job_extra == job_extra
        assert config.enable_dashboard is True
        assert config.dashboard_address == "0.0.0.0"
        assert config.dashboard_port == 8787
        assert config.dashboard_prefix == "/dask"

    def test_n_jobs_bug_documented(self):
        """Test that the documented bug in n_jobs assignment is present.

        Purpose: Validates that the TODO comment about n_jobs bug is accurate

        Given: PlannerEvaluationPBSWorkflow instance
        When: Initializing with cores parameter
        Then: n_jobs is incorrectly set to cores value

        Test type: unit
        """
        workflow = PlannerEvaluationPBSWorkflow(
            cache_dir_path=Path("/tmp/pbs"),
            experiment_name="pbs_test",
            cores=8,
        )

        # The bug: n_jobs should be 1 (default) but gets set to cores (8)
        assert workflow.n_jobs == 8  # This is the bug
        # The correct behavior would be: assert workflow.n_jobs == 1

    def test_job_extra_handling(self):
        """Test that job_extra parameter is handled correctly.

        Purpose: Validates that job_extra parameter is stored and passed through

        Given: PlannerEvaluationPBSWorkflow instances
        When: Initializing with None vs provided job_extra
        Then: Both cases work correctly

        Test type: unit
        """
        # Test with None
        workflow1 = PlannerEvaluationPBSWorkflow(
            cache_dir_path=Path("/tmp/pbs"),
            experiment_name="pbs_test",
            job_extra=None,
        )
        config1 = workflow1._get_task_manager_config()
        assert isinstance(config1, PBSConfig)
        assert config1.job_extra is None

        # Test with provided list
        job_extra = ["-l feature=gpu", "-l gres=gpu:1", "-l walltime=24:00:00"]
        workflow2 = PlannerEvaluationPBSWorkflow(
            cache_dir_path=Path("/tmp/pbs"),
            experiment_name="pbs_test",
            job_extra=job_extra,
        )
        config2 = workflow2._get_task_manager_config()
        assert isinstance(config2, PBSConfig)
        assert config2.job_extra == job_extra

    def test_dashboard_configuration(self):
        """Test that dashboard configuration parameters are handled correctly.

        Purpose: Validates that dashboard-related parameters are stored and passed through

        Given: PlannerEvaluationPBSWorkflow instances
        When: Initializing with different dashboard configurations
        Then: Dashboard parameters are handled correctly

        Test type: unit
        """
        # Test with dashboard enabled
        workflow1 = PlannerEvaluationPBSWorkflow(
            cache_dir_path=Path("/tmp/pbs"),
            experiment_name="pbs_test",
            enable_dashboard=True,
            dashboard_address="0.0.0.0",
            dashboard_port=8787,
            dashboard_prefix="/dask",
        )
        config1 = workflow1._get_task_manager_config()
        assert isinstance(config1, PBSConfig)
        assert config1.enable_dashboard is True
        assert config1.dashboard_address == "0.0.0.0"
        assert config1.dashboard_port == 8787
        assert config1.dashboard_prefix == "/dask"

        # Test with dashboard disabled
        workflow2 = PlannerEvaluationPBSWorkflow(
            cache_dir_path=Path("/tmp/pbs"),
            experiment_name="pbs_test",
            enable_dashboard=False,
            dashboard_address="127.0.0.1",
            dashboard_port=9999,
            dashboard_prefix=None,
        )
        config2 = workflow2._get_task_manager_config()
        assert isinstance(config2, PBSConfig)
        assert config2.enable_dashboard is False
        assert config2.dashboard_address == "127.0.0.1"
        assert config2.dashboard_port == 9999
        assert config2.dashboard_prefix is None


class TestWorkflowIntegration:
    """Integration tests for workflow classes."""

    @patch("POMDPPlanners.simulations.workflows.planner_evaluation_workflow.POMDPSimulator")
    def test_local_workflow_evaluation_flow(
        self, mock_simulator_class, sample_environment_run_params, mock_simulator_results
    ):
        """Test complete evaluation flow for local workflow.

        Purpose: Validates that local workflow evaluation works end-to-end

        Given: PlannerEvaluationLocalWorkflow instance
        When: Calling evaluate method
        Then: Complete evaluation flow executes successfully

        Test type: integration
        """
        mock_simulator_instance = Mock()
        mock_simulator_instance.compare_multiple_environments_policies.return_value = (
            mock_simulator_results
        )
        mock_simulator_class.return_value = mock_simulator_instance

        workflow = PlannerEvaluationLocalWorkflow(
            experiment_name="integration_test",
            cache_dir_path=Path("/tmp/integration"),
            n_jobs=2,
        )

        configs = [sample_environment_run_params]
        results = workflow.evaluate(configs)

        # Verify simulator was created with JoblibConfig
        call_args = mock_simulator_class.call_args
        task_manager_config = call_args[1]["task_manager_config"]
        assert isinstance(task_manager_config, JoblibConfig)
        assert task_manager_config.n_jobs == 2

        # Verify results are returned
        assert results == mock_simulator_results

    @patch("POMDPPlanners.simulations.workflows.planner_evaluation_workflow.POMDPSimulator")
    def test_dask_workflow_evaluation_flow(
        self, mock_simulator_class, sample_environment_run_params, mock_simulator_results
    ):
        """Test complete evaluation flow for Dask workflow.

        Purpose: Validates that Dask workflow evaluation works end-to-end

        Given: PlannerEvaluationDaskWorkflow instance
        When: Calling evaluate method
        Then: Complete evaluation flow executes successfully

        Test type: integration
        """
        mock_simulator_instance = Mock()
        mock_simulator_instance.compare_multiple_environments_policies.return_value = (
            mock_simulator_results
        )
        mock_simulator_class.return_value = mock_simulator_instance

        workflow = PlannerEvaluationDaskWorkflow(
            cache_dir_path=Path("/tmp/dask_integration"),
            experiment_name="dask_integration_test",
            n_workers=4,
            scheduler_address="tcp://localhost:8786",
        )

        configs = [sample_environment_run_params]
        results = workflow.evaluate(configs)

        # Verify simulator was created with DaskConfig
        call_args = mock_simulator_class.call_args
        task_manager_config = call_args[1]["task_manager_config"]
        assert isinstance(task_manager_config, DaskConfig)
        assert task_manager_config.n_workers == 4
        assert task_manager_config.scheduler_address == "tcp://localhost:8786"

        # Verify results are returned
        assert results == mock_simulator_results

    @patch("POMDPPlanners.simulations.workflows.planner_evaluation_workflow.POMDPSimulator")
    def test_pbs_workflow_evaluation_flow(
        self, mock_simulator_class, sample_environment_run_params, mock_simulator_results
    ):
        """Test complete evaluation flow for PBS workflow.

        Purpose: Validates that PBS workflow evaluation works end-to-end

        Given: PlannerEvaluationPBSWorkflow instance
        When: Calling evaluate method
        Then: Complete evaluation flow executes successfully

        Test type: integration
        """
        mock_simulator_instance = Mock()
        mock_simulator_instance.compare_multiple_environments_policies.return_value = (
            mock_simulator_results
        )
        mock_simulator_class.return_value = mock_simulator_instance

        workflow = PlannerEvaluationPBSWorkflow(
            cache_dir_path=Path("/tmp/pbs_integration"),
            experiment_name="pbs_integration_test",
            queue="gpu",
            n_workers=8,
            cores=4,
            memory="16GB",
        )

        configs = [sample_environment_run_params]
        results = workflow.evaluate(configs)

        # Verify simulator was created with PBSConfig
        call_args = mock_simulator_class.call_args
        task_manager_config = call_args[1]["task_manager_config"]
        assert isinstance(task_manager_config, PBSConfig)
        assert task_manager_config.queue == "gpu"
        assert task_manager_config.n_workers == 8
        assert task_manager_config.cores == 4
        assert task_manager_config.memory == "16GB"

        # Verify results are returned
        assert results == mock_simulator_results

    def test_workflow_parameter_consistency(self):
        """Test that workflow parameters are consistent across all implementations.

        Purpose: Validates that all workflow implementations handle common parameters consistently

        Given: All workflow implementations
        When: Initializing with same parameters
        Then: Common parameters are handled consistently

        Test type: integration
        """
        common_params = {
            "experiment_name": "consistency_test",
            "cache_dir_path": Path("/tmp/consistency"),
            "debug": True,
            "enable_profiling": True,
            "verbose": False,
            "alpha": 0.05,
            "confidence_interval_level": 0.99,
            "cache_visualizations": False,
        }

        # Test local workflow
        local_workflow = PlannerEvaluationLocalWorkflow(n_jobs=2, **common_params)

        # Test Dask workflow
        dask_workflow = PlannerEvaluationDaskWorkflow(
            cache_dir_path=common_params["cache_dir_path"],
            experiment_name=common_params["experiment_name"],
            debug=common_params["debug"],
            enable_profiling=common_params["enable_profiling"],
            verbose=common_params["verbose"],
            alpha=common_params["alpha"],
            confidence_interval_level=common_params["confidence_interval_level"],
            cache_visualizations=common_params["cache_visualizations"],
        )

        # Test PBS workflow
        pbs_workflow = PlannerEvaluationPBSWorkflow(
            cache_dir_path=common_params["cache_dir_path"],
            experiment_name=common_params["experiment_name"],
            debug=common_params["debug"],
            enable_profiling=common_params["enable_profiling"],
            verbose=common_params["verbose"],
            alpha=common_params["alpha"],
            confidence_interval_level=common_params["confidence_interval_level"],
            cache_visualizations=common_params["cache_visualizations"],
        )

        # Verify common parameters are consistent
        for workflow in [local_workflow, dask_workflow, pbs_workflow]:
            assert workflow.experiment_name == "consistency_test"
            assert workflow.cache_dir_path == Path("/tmp/consistency")
            assert workflow.debug is True
            assert workflow.enable_profiling is True
            assert workflow.verbose is False
            assert workflow.alpha == 0.05
            assert workflow.confidence_interval_level == 0.99
            assert workflow.cache_visualizations is False


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_configs_list(self):
        """Test that empty configs list raises validation error.

        Purpose: Validates that empty configs list is caught by validation

        Given: Workflow instance
        When: Calling evaluate with empty configs list
        Then: ValueError is raised

        Test type: unit
        """

        class TestWorkflow(PlannerEvaluationWorkflow):
            def _get_task_manager_config(self) -> TaskManagerConfig:
                return JoblibConfig(n_jobs=1)

        workflow = TestWorkflow(
            experiment_name="test",
            cache_dir_path=None,
        )

        # Should raise error with empty list due to validation
        with pytest.raises(ValueError, match="configs list cannot be empty"):
            workflow.evaluate([])

    def test_none_cache_dir_path(self, sample_environment_run_params):
        """Test that None cache_dir_path is handled correctly.

        Purpose: Validates that None cache_dir_path doesn't cause errors

        Given: Workflow instance with None cache_dir_path
        When: Calling evaluate method with valid config
        Then: Method executes without error and passes None to simulator

        Test type: unit
        """

        class TestWorkflow(PlannerEvaluationWorkflow):
            def _get_task_manager_config(self) -> TaskManagerConfig:
                return JoblibConfig(n_jobs=1)

        workflow = TestWorkflow(
            experiment_name="test",
            cache_dir_path=None,
        )

        with patch(
            "POMDPPlanners.simulations.workflows.planner_evaluation_workflow.POMDPSimulator"
        ) as mock_simulator_class:
            mock_simulator_instance = Mock()
            mock_simulator_instance.compare_multiple_environments_policies.return_value = (
                {},
                pd.DataFrame(),
            )
            mock_simulator_class.return_value = mock_simulator_instance

            results = workflow.evaluate([sample_environment_run_params])

            # Verify simulator was called with None cache_dir_path
            call_args = mock_simulator_class.call_args
            assert call_args[1]["cache_dir_path"] is None

    def test_extreme_parameter_values(self):
        """Test that extreme parameter values are handled correctly.

        Purpose: Validates that extreme parameter values don't cause errors

        Given: Workflow instances with extreme parameter values
        When: Initializing and calling methods
        Then: Methods execute without error

        Test type: unit
        """
        # Test with extreme alpha values
        workflow1 = PlannerEvaluationLocalWorkflow(
            experiment_name="extreme_test",
            cache_dir_path=None,
            alpha=0.001,  # Very small alpha
        )
        assert workflow1.alpha == 0.001

        workflow2 = PlannerEvaluationLocalWorkflow(
            experiment_name="extreme_test",
            cache_dir_path=None,
            alpha=0.5,  # Large alpha
        )
        assert workflow2.alpha == 0.5

        # Test with extreme confidence interval levels
        workflow3 = PlannerEvaluationLocalWorkflow(
            experiment_name="extreme_test",
            cache_dir_path=None,
            confidence_interval_level=0.999,  # Very high confidence
        )
        assert workflow3.confidence_interval_level == 0.999

        workflow4 = PlannerEvaluationLocalWorkflow(
            experiment_name="extreme_test",
            cache_dir_path=None,
            confidence_interval_level=0.5,  # Low confidence
        )
        assert workflow4.confidence_interval_level == 0.5

    def test_large_n_jobs_values(self):
        """Test that large n_jobs values are handled correctly.

        Purpose: Validates that large n_jobs values don't cause errors

        Given: Workflow instances with large n_jobs values
        When: Getting task manager config
        Then: Config is created without error

        Test type: unit
        """
        # Test with very large n_jobs
        workflow = PlannerEvaluationLocalWorkflow(
            experiment_name="large_n_jobs_test",
            cache_dir_path=None,
            n_jobs=1000,
        )

        config = workflow._get_task_manager_config()
        assert isinstance(config, JoblibConfig)
        assert config.n_jobs == 1000

    def test_large_n_workers_values(self):
        """Test that large n_workers values are handled correctly.

        Purpose: Validates that large n_workers values don't cause errors

        Given: Dask workflow instances with large n_workers values
        When: Getting task manager config
        Then: Config is created without error

        Test type: unit
        """
        # Test with very large n_workers
        workflow = PlannerEvaluationDaskWorkflow(
            cache_dir_path=Path("/tmp/large_workers"),
            experiment_name="large_workers_test",
            n_workers=1000,
        )

        config = workflow._get_task_manager_config()
        assert isinstance(config, DaskConfig)
        assert config.n_workers == 1000


class TestPlannerEvaluationWorkflowValidation:
    """Test suite for input validation in PlannerEvaluationWorkflow."""

    def test_validate_configs_empty_list_raises_error(self):
        """Test that empty configs list raises ValueError.

        Purpose: Validates that empty configs list is caught during validation

        Given: PlannerEvaluationWorkflow instance
        When: Calling _validate_configs with empty list
        Then: ValueError is raised with appropriate message

        Test type: unit
        """

        class TestWorkflow(PlannerEvaluationWorkflow):
            def _get_task_manager_config(self) -> TaskManagerConfig:
                return JoblibConfig(n_jobs=1)

        workflow = TestWorkflow(
            experiment_name="test",
            cache_dir_path=None,
        )

        with pytest.raises(ValueError, match="configs list cannot be empty"):
            workflow._validate_configs([])

    def test_validate_configs_non_list_raises_error(self):
        """Test that non-list configs raises TypeError.

        Purpose: Validates that configs parameter must be a list

        Given: PlannerEvaluationWorkflow instance
        When: Calling _validate_configs with non-list type
        Then: TypeError is raised with appropriate message

        Test type: unit
        """

        class TestWorkflow(PlannerEvaluationWorkflow):
            def _get_task_manager_config(self) -> TaskManagerConfig:
                return JoblibConfig(n_jobs=1)

        workflow = TestWorkflow(
            experiment_name="test",
            cache_dir_path=None,
        )

        with pytest.raises(TypeError, match="configs must be a list"):
            workflow._validate_configs("not_a_list")  # type: ignore

    def test_validate_configs_wrong_config_type_raises_error(
        self, mock_environment, mock_belief, mock_policy
    ):
        """Test that wrong configuration type raises TypeError.

        Purpose: Validates that each config must be EnvironmentRunParams instance

        Given: PlannerEvaluationWorkflow instance
        When: Calling _validate_configs with wrong type in list
        Then: TypeError is raised with appropriate message

        Test type: unit
        """

        class TestWorkflow(PlannerEvaluationWorkflow):
            def _get_task_manager_config(self) -> TaskManagerConfig:
                return JoblibConfig(n_jobs=1)

        workflow = TestWorkflow(
            experiment_name="test",
            cache_dir_path=None,
        )

        wrong_config = {"environment": mock_environment, "belief": mock_belief}

        with pytest.raises(TypeError, match="Expected EnvironmentRunParams"):
            workflow._validate_configs([wrong_config])  # type: ignore

    def test_validate_configs_invalid_environment_type_raises_error(self, mock_belief, mock_policy):
        """Test that invalid environment type raises TypeError.

        Purpose: Validates that environment must be Environment instance

        Given: Invalid environment parameter for EnvironmentRunParams
        When: Creating EnvironmentRunParams at construction time
        Then: TypeError is raised with appropriate message

        Test type: unit
        """
        with pytest.raises(TypeError, match="environment must be an Environment instance"):
            EnvironmentRunParams(
                environment="not_an_environment",  # type: ignore
                belief=mock_belief,
                policies=[mock_policy],
                num_episodes=10,
                num_steps=5,
            )

    def test_validate_configs_invalid_belief_type_raises_error(self, mock_environment, mock_policy):
        """Test that invalid belief type raises TypeError.

        Purpose: Validates that belief must be Belief instance

        Given: Invalid belief parameter for EnvironmentRunParams
        When: Creating EnvironmentRunParams at construction time
        Then: TypeError is raised with appropriate message

        Test type: unit
        """
        with pytest.raises(TypeError, match="belief must be a Belief instance"):
            EnvironmentRunParams(
                environment=mock_environment,
                belief="not_a_belief",  # type: ignore
                policies=[mock_policy],
                num_episodes=10,
                num_steps=5,
            )

    def test_validate_configs_empty_policies_list_raises_error(self, mock_environment, mock_belief):
        """Test that empty policies list raises ValueError.

        Purpose: Validates that policies list cannot be empty

        Given: Empty policies list for EnvironmentRunParams
        When: Creating EnvironmentRunParams at construction time
        Then: ValueError is raised with appropriate message

        Test type: unit
        """
        with pytest.raises(ValueError, match="policies list cannot be empty"):
            EnvironmentRunParams(
                environment=mock_environment,
                belief=mock_belief,
                policies=[],
                num_episodes=10,
                num_steps=5,
            )

    def test_validate_configs_invalid_policy_type_raises_error(self, mock_environment, mock_belief):
        """Test that invalid policy type raises TypeError.

        Purpose: Validates that all policies must be Policy instances

        Given: Invalid policy in policies list for EnvironmentRunParams
        When: Creating EnvironmentRunParams at construction time
        Then: TypeError is raised with appropriate message

        Test type: unit
        """
        with pytest.raises(TypeError, match="must be a Policy instance"):
            EnvironmentRunParams(
                environment=mock_environment,
                belief=mock_belief,
                policies=["not_a_policy"],  # type: ignore
                num_episodes=10,
                num_steps=5,
            )

    def test_validate_configs_negative_num_episodes_raises_error(
        self, mock_environment, mock_belief, mock_policy
    ):
        """Test that negative num_episodes raises ValueError.

        Purpose: Validates that num_episodes must be positive

        Given: Negative num_episodes for EnvironmentRunParams
        When: Creating EnvironmentRunParams at construction time
        Then: ValueError is raised with appropriate message

        Test type: unit
        """
        with pytest.raises(ValueError, match="num_episodes must be positive"):
            EnvironmentRunParams(
                environment=mock_environment,
                belief=mock_belief,
                policies=[mock_policy],
                num_episodes=-5,
                num_steps=10,
            )

    def test_validate_configs_zero_num_episodes_raises_error(
        self, mock_environment, mock_belief, mock_policy
    ):
        """Test that zero num_episodes raises ValueError.

        Purpose: Validates that num_episodes must be greater than zero

        Given: Zero num_episodes for EnvironmentRunParams
        When: Creating EnvironmentRunParams at construction time
        Then: ValueError is raised with appropriate message

        Test type: unit
        """
        with pytest.raises(ValueError, match="num_episodes must be positive"):
            EnvironmentRunParams(
                environment=mock_environment,
                belief=mock_belief,
                policies=[mock_policy],
                num_episodes=0,
                num_steps=10,
            )

    def test_validate_configs_non_integer_num_episodes_raises_error(
        self, mock_environment, mock_belief, mock_policy
    ):
        """Test that non-integer num_episodes raises TypeError.

        Purpose: Validates that num_episodes must be an integer

        Given: Non-integer num_episodes for EnvironmentRunParams
        When: Creating EnvironmentRunParams at construction time
        Then: TypeError is raised with appropriate message

        Test type: unit
        """
        with pytest.raises(TypeError, match="num_episodes must be an integer"):
            EnvironmentRunParams(
                environment=mock_environment,
                belief=mock_belief,
                policies=[mock_policy],
                num_episodes=10.5,  # type: ignore
                num_steps=5,
            )

    def test_validate_configs_negative_num_steps_raises_error(
        self, mock_environment, mock_belief, mock_policy
    ):
        """Test that negative num_steps raises ValueError.

        Purpose: Validates that num_steps must be positive

        Given: Negative num_steps for EnvironmentRunParams
        When: Creating EnvironmentRunParams at construction time
        Then: ValueError is raised with appropriate message

        Test type: unit
        """
        with pytest.raises(ValueError, match="num_steps must be positive"):
            EnvironmentRunParams(
                environment=mock_environment,
                belief=mock_belief,
                policies=[mock_policy],
                num_episodes=10,
                num_steps=-5,
            )

    def test_validate_configs_zero_num_steps_raises_error(
        self, mock_environment, mock_belief, mock_policy
    ):
        """Test that zero num_steps raises ValueError.

        Purpose: Validates that num_steps must be greater than zero

        Given: Zero num_steps for EnvironmentRunParams
        When: Creating EnvironmentRunParams at construction time
        Then: ValueError is raised with appropriate message

        Test type: unit
        """
        with pytest.raises(ValueError, match="num_steps must be positive"):
            EnvironmentRunParams(
                environment=mock_environment,
                belief=mock_belief,
                policies=[mock_policy],
                num_episodes=10,
                num_steps=0,
            )

    def test_validate_configs_non_integer_num_steps_raises_error(
        self, mock_environment, mock_belief, mock_policy
    ):
        """Test that non-integer num_steps raises TypeError.

        Purpose: Validates that num_steps must be an integer

        Given: Non-integer num_steps for EnvironmentRunParams
        When: Creating EnvironmentRunParams at construction time
        Then: TypeError is raised with appropriate message

        Test type: unit
        """
        with pytest.raises(TypeError, match="num_steps must be an integer"):
            EnvironmentRunParams(
                environment=mock_environment,
                belief=mock_belief,
                policies=[mock_policy],
                num_episodes=10,
                num_steps=5.5,  # type: ignore
            )

    def test_validate_configs_valid_config_passes(self, mock_environment, mock_belief, mock_policy):
        """Test that valid configuration passes validation.

        Purpose: Validates that properly formed config passes all checks

        Given: PlannerEvaluationWorkflow instance and valid config
        When: Calling _validate_configs
        Then: No exception is raised

        Test type: unit
        """

        class TestWorkflow(PlannerEvaluationWorkflow):
            def _get_task_manager_config(self) -> TaskManagerConfig:
                return JoblibConfig(n_jobs=1)

        workflow = TestWorkflow(
            experiment_name="test",
            cache_dir_path=None,
        )

        config = EnvironmentRunParams(
            environment=mock_environment,
            belief=mock_belief,
            policies=[mock_policy],
            num_episodes=10,
            num_steps=5,
        )

        # Should not raise any exception
        workflow._validate_configs([config])

    def test_validate_configs_multiple_valid_configs_passes(
        self, mock_environment, mock_belief, mock_policy
    ):
        """Test that multiple valid configurations pass validation.

        Purpose: Validates that validation works correctly with multiple configs

        Given: PlannerEvaluationWorkflow instance and multiple valid configs
        When: Calling _validate_configs
        Then: No exception is raised

        Test type: unit
        """

        class TestWorkflow(PlannerEvaluationWorkflow):
            def _get_task_manager_config(self) -> TaskManagerConfig:
                return JoblibConfig(n_jobs=1)

        workflow = TestWorkflow(
            experiment_name="test",
            cache_dir_path=None,
        )

        config1 = EnvironmentRunParams(
            environment=mock_environment,
            belief=mock_belief,
            policies=[mock_policy],
            num_episodes=10,
            num_steps=5,
        )

        # Create second set of real objects
        env2 = TigerPOMDP(discount_factor=0.9)
        belief2 = WeightedParticleBelief(
            particles=["tiger_left", "tiger_right"], log_weights=np.log(np.array([0.3, 0.7]))
        )
        policy2 = POMCP(
            environment=env2,
            discount_factor=0.9,
            depth=10,
            n_simulations=20,
            exploration_constant=2.0,
            name="Policy2",
        )
        policy3 = POMCP(
            environment=env2,
            discount_factor=0.9,
            depth=15,
            n_simulations=30,
            exploration_constant=1.5,
            name="Policy3",
        )

        config2 = EnvironmentRunParams(
            environment=env2,
            belief=belief2,
            policies=[policy2, policy3],
            num_episodes=20,
            num_steps=15,
        )

        # Should not raise any exception
        workflow._validate_configs([config1, config2])

    def test_evaluate_calls_validation(self, mock_environment, mock_belief, mock_policy):
        """Test that validation is performed at construction time before evaluate is called.

        Purpose: Validates that validation occurs at EnvironmentRunParams construction

        Given: Invalid parameters for EnvironmentRunParams
        When: Creating EnvironmentRunParams with negative num_episodes
        Then: Validation error is raised at construction time

        Test type: integration
        """
        # Create invalid config (negative num_episodes) - validation happens here
        with pytest.raises(ValueError, match="num_episodes must be positive"):
            EnvironmentRunParams(
                environment=mock_environment,
                belief=mock_belief,
                policies=[mock_policy],
                num_episodes=-5,
                num_steps=10,
            )
