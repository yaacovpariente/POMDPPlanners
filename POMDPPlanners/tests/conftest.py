"""Shared pytest fixtures for all test modules.

This conftest.py provides common fixtures that are automatically available
to all test files in the POMDPPlanners/tests directory and its subdirectories.
"""

import random
import shutil
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pytest

from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import EnvironmentRunParams

np.random.seed(42)
random.seed(42)


@pytest.fixture(autouse=True)
def cleanup_mlflow_runs():
    """Automatically cleanup any active MLflow runs before and after each test.

    This fixture ensures that MLflow runs are properly ended between tests,
    preventing the "Run already active" error that occurs when tests don't
    properly clean up their MLflow runs.
    """
    # Before test: end any existing runs
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        # Ignore any errors during cleanup
        pass

    yield

    # After test: end any remaining runs
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        # Ignore any errors during cleanup
        pass


@pytest.fixture
def temp_cache_dir():
    """Fixture to create a temporary cache directory.

    Automatically cleans up the directory after test completion.
    Includes delay to ensure all file handles are released before cleanup.
    """
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
    """Fixture to create a TigerPOMDP environment for testing."""
    return TigerPOMDP(discount_factor=0.95, name="test_tiger")


@pytest.fixture
def sparse_pft_policy(tiger_environment):
    """Fixture to create a SparsePFT policy for the Tiger environment."""
    return SparsePFT(
        environment=tiger_environment,
        discount_factor=0.95,
        depth=10,
        c_ucb=1.0,
        beta_ucb=0.5,
        belief_child_num=5,
        n_simulations=100,
        name="test_sparse_pft",
    )


@pytest.fixture
def sample_environment_run_params(tiger_environment, sparse_pft_policy):
    """Fixture to create sample environment run parameters.

    Uses TigerPOMDP and SparsePFT with minimal configuration for fast testing.
    """
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
