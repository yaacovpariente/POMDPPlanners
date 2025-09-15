import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import List, Dict
import tempfile
import time
import random

import pandas as pd
import numpy as np

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief, get_initial_belief
from POMDPPlanners.core.simulation import EnvironmentRunParams, History
from POMDPPlanners.simulations.simulations_api import SimulationsAPI
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT

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
        sample_environment_run_params[0].environment.state_transition_model = (
            lambda *args: None
        )
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
        with pytest.raises(
            ValueError, match="confidence_interval_level must be between 0 and 1"
        ):
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
        mock_simulator_instance.compare_multiple_environments_policies.return_value = (
            mock_results
        )
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
        mock_simulator_instance.compare_multiple_environments_policies.return_value = (
            mock_results
        )
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
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'queue'"
        ):
            api.run_multiple_environments_and_policies_pbs_run(
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
        mock_simulator_instance.compare_multiple_environments_policies.return_value = (
            mock_results
        )
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
