import random
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import EnvironmentRunParams, History
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
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


# Note: Integration tests for DaskSimulationsAPI would require an actual Dask cluster
# or local Dask setup, which may not be suitable for unit test environments.
# The tests above use mocking to verify the correct configuration and API usage.
