import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import List, Dict
import tempfile
import time

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
        name="test_sparse_pft"
    )

@pytest.fixture
def sample_environment_run_params(tiger_environment, sparse_pft_policy):
    """Fixture to create sample environment run parameters using TigerPOMDP and SparsePFT."""
    initial_belief = get_initial_belief(tiger_environment, n_particles=100)
    return [EnvironmentRunParams(
        environment=tiger_environment,
        belief=initial_belief,
        policies=[sparse_pft_policy],
        num_episodes=2,
        num_steps=10
    )]

class TestSimulationsAPI:
    def test_init(self):
        """Test SimulationsAPI initialization.
    
    Purpose: Validates init
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
        api = SimulationsAPI()
        assert isinstance(api, SimulationsAPI)

    def test_run_multiple_environments_and_policies_local_run_success(
        self,
        temp_cache_dir,
        sample_environment_run_params
    ):
        """Test successful execution of run_multiple_environments_and_policies_local_run.
    
    Purpose: Validates run multiple environments and policies local run success
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
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
            cache_dir_path=temp_cache_dir
        )
        assert isinstance(results, dict)
        assert "test_tiger" in results
        assert "test_sparse_pft" in results["test_tiger"]
        assert len(results["test_tiger"]["test_sparse_pft"]) == 2  # num_episodes
        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) == 1
        assert stats_df['environment'].iloc[0] == 'test_tiger'
        assert stats_df['policy'].iloc[0] == 'test_sparse_pft'
        assert 'success_rate' in stats_df.columns
        assert 'average_listens' in stats_df.columns
        history = results["test_tiger"]["test_sparse_pft"][0]
        assert isinstance(history, History)
        assert len(history.history) > 0
        assert all(hasattr(step, 'state') for step in history.history)
        assert all(hasattr(step, 'action') for step in history.history)
        assert all(hasattr(step, 'observation') for step in history.history)
        assert all(hasattr(step, 'reward') for step in history.history)
        
    def test_run_multiple_environments_and_policies_local_run_error(
        self,
        temp_cache_dir,
        sample_environment_run_params
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
                cache_dir_path=temp_cache_dir
            )

    def test_run_multiple_environments_and_policies_local_run_invalid_params(
        self,
        temp_cache_dir,
        sample_environment_run_params
    ):
        """Test run_multiple_environments_and_policies_local_run with invalid parameters.
    
    Purpose: Validates run multiple environments and policies local run invalid params
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: integration
    """
        api = SimulationsAPI()
        # Test invalid alpha
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            api.run_multiple_environments_and_policies_local_run(
                environment_run_params=sample_environment_run_params,
                alpha=-0.1,  # Invalid alpha
                confidence_interval_level=0.95,
                cache_dir_path=temp_cache_dir
            )
        # Test invalid confidence interval
        with pytest.raises(ValueError, match="confidence_interval_level must be between 0 and 1"):
            api.run_multiple_environments_and_policies_local_run(
                environment_run_params=sample_environment_run_params,
                alpha=0.1,
                confidence_interval_level=1.5,  # Invalid confidence interval
                cache_dir_path=temp_cache_dir
            )
        # Test invalid n_jobs
        with pytest.raises(ValueError, match="n_jobs must be a positive integer or -1"):
            api.run_multiple_environments_and_policies_local_run(
                environment_run_params=sample_environment_run_params,
                alpha=0.1,
                confidence_interval_level=0.95,
                n_jobs=0,  # Invalid n_jobs
                cache_dir_path=temp_cache_dir
            )
