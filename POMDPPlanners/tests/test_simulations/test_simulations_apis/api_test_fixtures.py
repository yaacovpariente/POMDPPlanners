"""Shared fixtures and test data generators for simulation API tests.

This module provides reusable pytest fixtures for testing all simulation API
implementations (LocalSimulationsAPI, DaskSimulationsAPI, PBSSimulationsAPI).
"""

import pytest
import tempfile
import time
from pathlib import Path

from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.environment import (
    Environment,
    DiscreteActionsEnvironment,
    SpaceType,
)
from POMDPPlanners.core.policy import Policy, PolicySpaceInfo
from POMDPPlanners.core.simulation import (
    EnvironmentRunParams,
    NumericalHyperParameter,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
    HyperParameterOptimizationDirection,
    HyperParamPlannerConfig,
)
from POMDPPlanners.core.simulation.simulation_configs import PlannerGenerator


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
def sample_environment():
    """Create a simple test environment (Tiger POMDP)."""
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def sample_policy(sample_environment):
    """Create a simple test policy (POMCP)."""
    return POMCP(
        environment=sample_environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="TestPOMCP",
        n_simulations=10,
    )


@pytest.fixture
def sample_environment_params(sample_environment, sample_policy):
    """Create sample EnvironmentRunParams for testing.

    Returns a list with one EnvironmentRunParams configuration using
    minimal episodes and steps for fast testing.
    """
    return [
        EnvironmentRunParams(
            environment=sample_environment,
            belief=get_initial_belief(sample_environment, n_particles=10),
            policies=[sample_policy],
            num_episodes=2,
            num_steps=3,
        )
    ]


@pytest.fixture
def sample_hyperparameter_configs(sample_environment):
    """Create sample hyperparameter optimization configs.

    Returns a list with one HyperParameterRunParams configuration using
    minimal trials, episodes, and steps for fast testing.
    """
    planner_config = HyperParamPlannerConfig(
        policy_cls=POMCP,
        hyper_parameters=[
            NumericalHyperParameter(0.1, 2.0, "exploration_constant"),
            NumericalHyperParameter(5, 20, "n_simulations"),
        ],
        constant_parameters={
            "discount_factor": 0.95,
            "name": "OptimizedPOMCP",
            "depth": 3,
        },
    )

    return [
        HyperParameterRunParams(
            environment=sample_environment,
            belief=get_initial_belief(sample_environment, n_particles=10),
            hyper_param_planner_config=planner_config,
            num_episodes=2,
            num_steps=3,
            n_trials=2,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
        )
    ]


def create_temp_cache_dir(tmp_path: Path, name: str) -> Path:
    """Helper function to create temporary cache directories.

    Args:
        tmp_path: pytest tmp_path fixture
        name: Name for the cache directory

    Returns:
        Path to the created cache directory
    """
    cache_dir = tmp_path / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class MockPlannerGenerator(PlannerGenerator):
    """Mock implementation of PlannerGenerator for testing."""

    def __init__(self, space_info: PolicySpaceInfo, name: str = "MockPlanner"):
        self.space_info = space_info
        self.planner_name = name

    def generate(self, environment: Environment) -> Policy:
        """Generate a mock policy for the given environment.

        Args:
            environment: The environment to generate a policy for.
                Should be a DiscreteActionsEnvironment for POMCP compatibility.

        Returns:
            A POMCP policy configured for the environment.
        """
        # Cast to DiscreteActionsEnvironment for POMCP
        # This is safe as we only use this with discrete environments in tests
        discrete_env = environment
        assert isinstance(
            discrete_env, DiscreteActionsEnvironment
        ), "MockPlannerGenerator requires DiscreteActionsEnvironment"

        return POMCP(
            environment=discrete_env,
            discount_factor=0.95,
            depth=3,
            exploration_constant=1.0,
            name=f"{self.planner_name}_{environment.name}",
            n_simulations=10,
        )

    def get_planner_space_info(self) -> PolicySpaceInfo:
        """Return the stored space info."""
        return self.space_info


@pytest.fixture
def sample_planner_generators():
    """Create sample PlannerGenerator objects for testing.

    Returns a list with PlannerGenerator instances compatible with
    discrete action/observation spaces (e.g., TigerPOMDP).
    """
    discrete_space_info = PolicySpaceInfo(
        action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
    )

    return [
        MockPlannerGenerator(discrete_space_info, name="TestPOMCP_1"),
        MockPlannerGenerator(discrete_space_info, name="TestPOMCP_2"),
    ]
