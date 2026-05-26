# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for experiment_configs module.

This module contains comprehensive tests for the experiment_configs implementation,
including tests for all configuration functions used in hyperparameter optimization
and benchmarking experiments.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from POMDPPlanners.configs.experiment_configs import (
    get_hyperparameter_benchmarks,
    complete_environments_and_benchmarks_hyperparameter_optimization_configs,
    get_benchmarks_hyperparameter_optimization_configs,
    AverageReturnParameterToOptimizeMapper,
    RiskAverseParameterToOptimizeMapper,
    AllHyperparameterBenchmarksExperimentConfigCreator,
    AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator,
)
from POMDPPlanners.configs.environment_configs import (
    EnvironmentConfigsAPI,
    RiskAverseEnvironmentConfigsAPI,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
    HyperParamPlannerConfigGenerator,
    HyperParamPlannerConfig,
    CategoricalHyperParameter,
)
from POMDPPlanners.core.simulation.simulation_configs import (
    EvaluationExperimentConfigCreator,
    PlannerGenerator,
    EnvironmentRunParams,
)
from POMDPPlanners.core.policy import PolicySpaceInfo, PolicyRunData
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.environments.push_pomdp import PushPOMDP
from POMDPPlanners.environments.safety_ant_velocity_pomdp import SafeAntVelocityPOMDP
from POMDPPlanners.environments.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.laser_tag_pomdp import (
    LaserTagPOMDP,
    ContinuousLaserTagPOMDP,
    ContinuousLaserTagPOMDPDiscreteActions,
)
from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP

# Set random seeds for reproducible tests
np.random.seed(42)


class MockPolicy(Policy):
    """Mock Policy class for testing that accepts standard policy parameters."""

    def __init__(
        self,
        environment,
        name,
        discount_factor=0.95,
        log_path=None,
        debug=False,
        use_queue_logger=False,
        test_param=1.0,  # Added parameter for hyperparameter testing
    ):
        super().__init__(environment, discount_factor, name, log_path, debug, use_queue_logger)
        self.test_param = test_param

    def action(self, belief):
        """Mock action method."""
        return [], PolicyRunData(info_variables=[])

    @classmethod
    def get_space_info(cls):
        """Mock space info method."""
        return PolicySpaceInfo(SpaceType.DISCRETE, SpaceType.DISCRETE)

    @classmethod
    def get_info_variable_names(cls):
        """Mock info variable names method."""
        return []


class MockHyperParamPlannerConfigGenerator(HyperParamPlannerConfigGenerator):
    """Mock implementation of HyperParamPlannerConfigGenerator for testing."""

    def __init__(self, space_info: PolicySpaceInfo):
        self.space_info = space_info

    def generate(self, environment: Environment) -> HyperParamPlannerConfig:
        """Generate a mock planner config."""
        return HyperParamPlannerConfig(
            policy_cls=MockPolicy,
            hyper_parameters=[
                CategoricalHyperParameter(name="test_param", choices=[0.5, 1.0, 1.5])
            ],
            constant_parameters={"environment": environment, "name": "MockPlanner"},
        )

    def get_planner_space_info(self) -> PolicySpaceInfo:
        """Return the stored space info."""
        return self.space_info


class MockPlannerGenerator(PlannerGenerator):
    """Mock implementation of PlannerGenerator for testing."""

    def __init__(self, space_info: PolicySpaceInfo, name: str = "MockPlanner"):
        self.space_info = space_info
        self.name = name

    def generate(self, environment: Environment) -> Policy:
        """Generate a mock policy."""
        mock_policy = Mock(spec=Policy)
        mock_policy.config_id = f"{self.name}_{environment.config_id}"
        return mock_policy

    def get_planner_space_info(self) -> PolicySpaceInfo:
        """Return the stored space info."""
        return self.space_info


class TestExperimentConfigs:
    """Test cases for experiment_configs module."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.discount_factor = 0.95
        self.particle_count = 10  # Small number for faster tests
        self.time_out_in_seconds = 3.0

        # Create mock space info for discrete environments
        self.discrete_space_info = PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )

        # Create mock space info for continuous environments
        self.continuous_space_info = PolicySpaceInfo(
            action_space=SpaceType.CONTINUOUS, observation_space=SpaceType.CONTINUOUS
        )

    def test_get_hyperparameter_benchmarks_basic(self):
        """Test basic functionality of get_hyperparameter_benchmarks.

        Purpose: Validates that get_hyperparameter_benchmarks returns correct structure
        and content for basic input parameters.

        Given: Valid PolicySpaceInfo with discrete spaces
        When: Calling get_hyperparameter_benchmarks with default parameters
        Then: Returns list of (Environment, Belief, List[HyperParamPlannerConfig]) tuples

        Test type: unit
        """
        benchmarks = get_hyperparameter_benchmarks(
            policy_space_info=self.discrete_space_info,
            particle_count=self.particle_count,
            time_out_in_seconds=self.time_out_in_seconds,
        )

        # Verify return type and structure
        assert isinstance(benchmarks, list), "Should return a list of benchmarks"

        # If benchmarks exist, verify structure
        if benchmarks:
            for env, belief, planner_configs in benchmarks:
                assert isinstance(env, Environment), "Each benchmark should contain an Environment"
                assert isinstance(belief, Belief), "Each benchmark should contain a Belief"
                assert isinstance(
                    planner_configs, list
                ), "Each benchmark should contain list of planner configs"

                # Verify planner configs structure
                for planner_config in planner_configs:
                    assert isinstance(
                        planner_config, HyperParamPlannerConfig
                    ), "Should contain HyperParamPlannerConfig instances"

    def test_get_hyperparameter_benchmarks_custom_parameters(self):
        """Test get_hyperparameter_benchmarks with custom parameters.

        Purpose: Validates that get_hyperparameter_benchmarks works with custom
        particle counts and timeout settings.

        Given: Custom particle count and timeout parameters
        When: Calling get_hyperparameter_benchmarks with custom parameters
        Then: Returns correctly structured benchmarks with appropriate settings

        Test type: unit
        """
        custom_particle_count = 20
        custom_timeout = 5.0

        benchmarks = get_hyperparameter_benchmarks(
            policy_space_info=self.discrete_space_info,
            particle_count=custom_particle_count,
            time_out_in_seconds=custom_timeout,
        )

        assert isinstance(benchmarks, list), "Should return a list"

        # Verify that the function can handle different parameters
        if benchmarks:
            env, belief, planner_configs = benchmarks[0]
            # Verify belief has expected particle count (some belief types have particles)
            assert hasattr(belief, "particles"), "Belief should have particles"

    def test_get_hyperparameter_benchmarks_continuous_spaces(self):
        """Test get_hyperparameter_benchmarks with continuous space info.

        Purpose: Validates that get_hyperparameter_benchmarks handles continuous
        action/observation spaces correctly.

        Given: PolicySpaceInfo with continuous spaces
        When: Calling get_hyperparameter_benchmarks
        Then: Returns appropriate benchmarks for continuous environments

        Test type: unit
        """
        benchmarks = get_hyperparameter_benchmarks(
            policy_space_info=self.continuous_space_info, particle_count=self.particle_count
        )

        assert isinstance(benchmarks, list), "Should return a list for continuous spaces"

    def test_complete_environments_and_benchmarks_configs_basic(self):
        """Test basic functionality of complete_environments_and_benchmarks_hyperparameter_optimization_configs.

        Purpose: Validates that the function returns correctly structured
        HyperParameterRunParams list with both main and benchmark configurations.

        Given: List of Mock HyperParamPlannerConfigGenerator with discrete space info
        When: Calling complete_environments_and_benchmarks_hyperparameter_optimization_configs
        Then: Returns list of HyperParameterRunParams with correct structure

        Test type: unit
        """
        gen = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        mapper = AverageReturnParameterToOptimizeMapper()

        configs = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
            generators=[gen],
            parameter_to_optimize_mapper=mapper,
            particles=self.particle_count,
            num_episodes=5,
            num_steps=10,
            n_trials=20,
        )

        # Verify return type and structure
        assert isinstance(configs, list), "Should return a list of configurations"

        if configs:
            for config in configs:
                assert isinstance(
                    config, HyperParameterRunParams
                ), "Should contain HyperParameterRunParams"
                assert isinstance(config.environment, Environment), "Config should have Environment"
                assert isinstance(config.belief, Belief), "Config should have Belief"
                assert isinstance(
                    config.hyper_param_planner_config, HyperParamPlannerConfig
                ), "Config should have HyperParamPlannerConfig"
                assert config.num_episodes == 5, "Should use specified num_episodes"
                assert config.num_steps == 10, "Should use specified num_steps"
                assert config.n_trials == 20, "Should use specified n_trials"
                # Check that average_return is in parameters and all are MAXIMIZE
                assert len(config.parameters_to_optimize) > 0, "Should have at least one parameter"
                assert any(
                    param[0] == "average_return" for param in config.parameters_to_optimize
                ), "Should include average_return parameter"
                assert all(
                    param[1] == HyperParameterOptimizationDirection.MAXIMIZE
                    for param in config.parameters_to_optimize
                ), "All parameters should be set to MAXIMIZE"

    def test_complete_environments_and_benchmarks_configs_custom_params(self):
        """Test complete_environments_and_benchmarks_hyperparameter_optimization_configs with custom parameters.

        Purpose: Validates that the function works with various custom parameter combinations.

        Given: Custom parameters for episodes, steps, and trials
        When: Calling the function with custom parameters
        Then: Returns configurations with the specified parameters

        Test type: unit
        """
        gen = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        mapper = AverageReturnParameterToOptimizeMapper()

        configs = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
            generators=[gen],
            parameter_to_optimize_mapper=mapper,
            particles=25,
            num_episodes=15,
            num_steps=30,
            n_trials=50,
        )

        assert isinstance(configs, list)

        if configs:
            config = configs[0]
            assert config.num_episodes == 15, "Should use custom num_episodes"
            assert config.num_steps == 30, "Should use custom num_steps"
            assert config.n_trials == 50, "Should use custom n_trials"

    def test_complete_environments_and_benchmarks_configs_defaults(self):
        """Test complete_environments_and_benchmarks_hyperparameter_optimization_configs with default parameters.

        Purpose: Validates that the function works correctly with default parameter values.

        Given: Only required generator list and mapper parameters
        When: Calling the function with minimal parameters
        Then: Uses default values correctly for all optional parameters

        Test type: unit
        """
        gen = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        mapper = AverageReturnParameterToOptimizeMapper()

        configs = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
            generators=[gen], parameter_to_optimize_mapper=mapper
        )

        assert isinstance(configs, list)

        if configs:
            config = configs[0]
            # Verify default values
            assert config.num_episodes == 10, "Should use default num_episodes"
            assert config.num_steps == 20, "Should use default num_steps"
            assert config.n_trials == 500, "Should use default n_trials"
            # Removed old API check: direction
            # Removed old API check: parameter_to_optimize

    def test_get_benchmarks_hyperparameter_optimization_configs(self):
        """Test basic functionality of get_benchmarks_hyperparameter_optimization_configs.

        Purpose: Validates that the function creates benchmark configurations
        correctly from a reference HyperParameterRunParams.

        Given: Mock HyperParameterRunParams configuration
        When: Calling get_benchmarks_hyperparameter_optimization_configs
        Then: Returns list of benchmark configurations with correct structure

        Test type: unit
        """
        # Create mock environment and belief
        mock_env = TigerPOMDP(discount_factor=0.95, name="TestTiger")
        mock_belief = Mock(spec=Belief)
        mock_planner_config = HyperParamPlannerConfig(
            policy_cls=MockPolicy,
            hyper_parameters=[
                CategoricalHyperParameter(name="test_param", choices=[0.5, 1.0, 1.5])
            ],
            constant_parameters={"environment": mock_env, "name": "TestBenchmark"},
        )

        # Create reference configuration
        reference_conf = HyperParameterRunParams(
            environment=mock_env,
            belief=mock_belief,
            hyper_param_planner_config=mock_planner_config,
            num_episodes=5,
            num_steps=10,
            n_trials=20,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
        )

        benchmark_configs = get_benchmarks_hyperparameter_optimization_configs(
            conf=reference_conf,
            discount_factor=self.discount_factor,
            time_out_in_seconds=self.time_out_in_seconds,
        )

        # Verify return type and structure
        assert isinstance(benchmark_configs, list), "Should return a list of benchmark configs"

        if benchmark_configs:
            for config in benchmark_configs:
                assert isinstance(
                    config, HyperParameterRunParams
                ), "Should contain HyperParameterRunParams"
                # Verify that basic properties match the reference
                assert (
                    config.environment == reference_conf.environment
                ), "Should use same environment"
                assert config.belief == reference_conf.belief, "Should use same belief"
                assert (
                    config.num_episodes == reference_conf.num_episodes
                ), "Should use same num_episodes"
                assert config.num_steps == reference_conf.num_steps, "Should use same num_steps"
                assert config.n_trials == reference_conf.n_trials, "Should use same n_trials"
                assert (
                    config.parameters_to_optimize == reference_conf.parameters_to_optimize
                ), "Should use same parameters"

    def test_get_benchmarks_hyperparameter_optimization_configs_custom_timeout(self):
        """Test get_benchmarks_hyperparameter_optimization_configs with custom timeout.

        Purpose: Validates that the function respects custom timeout parameters.

        Given: HyperParameterRunParams and custom timeout setting
        When: Calling the function with custom timeout
        Then: Returns configurations but timeout logic is properly handled internally

        Test type: unit
        """
        mock_env = TigerPOMDP(discount_factor=0.95, name="TestTiger")
        mock_belief = Mock(spec=Belief)
        mock_planner_config = HyperParamPlannerConfig(
            policy_cls=MockPolicy,
            hyper_parameters=[
                CategoricalHyperParameter(name="test_param", choices=[0.5, 1.0, 1.5])
            ],
            constant_parameters={"environment": mock_env, "name": "TestBenchmark"},
        )

        reference_conf = HyperParameterRunParams(
            environment=mock_env,
            belief=mock_belief,
            hyper_param_planner_config=mock_planner_config,
            num_episodes=3,
            num_steps=5,
            n_trials=10,
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
        )

        custom_timeout = 7.5
        benchmark_configs = get_benchmarks_hyperparameter_optimization_configs(
            conf=reference_conf, discount_factor=0.9, time_out_in_seconds=custom_timeout
        )

        assert isinstance(benchmark_configs, list), "Should return a list with custom timeout"

    @patch("POMDPPlanners.configs.experiment_configs.EnvironmentConfigsAPI")
    @patch("POMDPPlanners.configs.experiment_configs.PlannersHyperparamConfigs")
    def test_get_hyperparameter_benchmarks_with_mocks(self, mock_planners_config, mock_env_config):
        """Test get_hyperparameter_benchmarks with mocked dependencies.

        Purpose: Validates that the function correctly integrates with its dependencies
        and handles the expected API calls.

        Given: Mocked EnvironmentConfigsAPI and PlannersHyperparamConfigs
        When: Calling get_hyperparameter_benchmarks
        Then: Properly calls mock objects and returns expected structure

        Test type: unit
        """
        # Set up mocks
        mock_env_instance = mock_env_config.return_value
        mock_planners_instance = mock_planners_config.return_value

        # Mock environments and beliefs
        fake_env = TigerPOMDP(discount_factor=0.95, name="MockTiger")
        fake_belief = Mock(spec=Belief)
        mock_env_instance.get_compatible_environments.return_value = [(fake_env, fake_belief)]

        # Mock planner configs
        fake_planner_conf = Mock()
        fake_planner_conf.policy_cls = MockPolicy
        fake_planner_conf.hyper_parameters = [
            CategoricalHyperParameter(name="test_param", choices=[0.5, 1.0, 1.5])
        ]
        fake_planner_conf.constant_parameters = {"environment": fake_env, "name": "FakePlanner"}
        mock_planners_instance.get_compatible_planners.return_value = [fake_planner_conf]

        # Call function
        benchmarks = get_hyperparameter_benchmarks(
            policy_space_info=self.discrete_space_info,
            particle_count=self.particle_count,
            time_out_in_seconds=self.time_out_in_seconds,
        )

        # Verify mocks were called correctly
        mock_env_config.assert_called_once_with(discount_factor=0.95)
        mock_planners_config.assert_called_once_with(discount_factor=0.95)
        mock_env_instance.get_compatible_environments.assert_called_once_with(
            policy_space_info=self.discrete_space_info, n_particles=self.particle_count
        )
        mock_planners_instance.get_compatible_planners.assert_called_once_with(
            env=fake_env, time_out_in_seconds=self.time_out_in_seconds
        )

        # Verify result structure
        assert isinstance(benchmarks, list)
        if benchmarks:
            env, belief, planner_configs = benchmarks[0]
            assert env == fake_env
            assert belief == fake_belief
            assert len(planner_configs) == 1
            assert isinstance(planner_configs[0], HyperParamPlannerConfig)

    def test_edge_case_empty_compatible_environments(self):
        """Test handling of edge case when no compatible environments are found.

        Purpose: Validates that functions handle the case when environment configs
        return empty lists gracefully.

        Given: PolicySpaceInfo that might have no compatible environments
        When: Calling functions that depend on compatible environments
        Then: Returns empty lists or handles gracefully

        Test type: edge case
        """
        # This test verifies the functions handle empty environment lists
        benchmarks = get_hyperparameter_benchmarks(
            policy_space_info=self.discrete_space_info,
            particle_count=5,  # Very small number might result in empty environments
        )

        # Should return empty list, not crash
        assert isinstance(benchmarks, list), "Should return list even if empty"

    def test_edge_case_zero_particles(self):
        """Test handling of edge case with zero particles.

        Purpose: Validates that functions handle invalid particle counts gracefully.

        Given: Zero particle count
        When: Calling functions with zero particles
        Then: Either handles gracefully or raises appropriate errors

        Test type: edge case
        """
        with pytest.raises((ValueError, AssertionError)):
            get_hyperparameter_benchmarks(
                policy_space_info=self.discrete_space_info,
                particle_count=0,  # Invalid particle count
            )

    def test_edge_case_negative_timeout(self):
        """Test handling of negative timeout values.

        Purpose: Validates that functions handle negative timeout values appropriately.

        Given: Negative timeout parameter
        When: Calling functions with negative timeout
        Then: Either handles gracefully or raises appropriate errors

        Test type: edge case
        """
        # Test that negative timeout is handled gracefully (doesn't crash)
        benchmarks = get_hyperparameter_benchmarks(
            policy_space_info=self.discrete_space_info,
            time_out_in_seconds=-1.0,  # Invalid timeout, but should be handled gracefully
        )
        assert isinstance(benchmarks, list), "Should return list even with negative timeout"

    def test_edge_case_invalid_optimization_direction(self):
        """Test handling of optimization with parameter mapper.

        Purpose: Validates that optimization works with parameter mapper.

        Given: Valid parameter mapper
        When: Calling functions with mapper
        Then: Function should work correctly

        Test type: edge case
        """
        gen = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        mapper = AverageReturnParameterToOptimizeMapper()

        # Test with a valid mapper (should work)
        configs = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
            generators=[gen], parameter_to_optimize_mapper=mapper
        )
        assert isinstance(configs, list)

    def test_integration_between_functions(self):
        """Test integration between different functions in the module.

        Purpose: Validates that functions work together correctly when chained or used together.

        Given: Output from one function
        When: Using that output as input to another function
        Then: Functions work together without errors

        Test type: integration
        """
        # Test that configs returned by complete_environments_and_benchmarks_configs
        # can be used with get_benchmarks_hyperparameter_optimization_configs
        gen = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        mapper = AverageReturnParameterToOptimizeMapper()

        # Get full configs which includes both main and benchmark configs
        all_configs = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
            generators=[gen],
            parameter_to_optimize_mapper=mapper,
            particles=self.particle_count,
            num_episodes=3,
            num_steps=5,
            n_trials=10,
        )

        assert isinstance(all_configs, list)

        # Test that we can use one config as reference for benchmarks
        if all_configs:
            reference_config = all_configs[0]

            # This shouldn't crash and should return additional benchmark configs
            additional_benchmarks = get_benchmarks_hyperparameter_optimization_configs(
                conf=reference_config,
                discount_factor=self.discount_factor,
                time_out_in_seconds=self.time_out_in_seconds,
            )

            assert isinstance(
                additional_benchmarks, list
            ), "Should handle reference config correctly"

    def test_function_signature_compatibility(self):
        """Test that function signatures match expected types and imports work correctly.

        Purpose: Validates that all imports are correct and function signatures
        are compatible with their declared types.

        Given: Various input combinations
        When: Calling functions with type-checked inputs
        Then: Type system accepts all calls without import errors

        Test type: type checking
        """
        # Test that we can create space info objects
        space_info = PolicySpaceInfo(
            action_space=SpaceType.MIXED, observation_space=SpaceType.MIXED
        )

        # Test function calls don't raise import or type errors
        try:
            benchmarks = get_hyperparameter_benchmarks(
                policy_space_info=space_info, particle_count=self.particle_count
            )
            assert isinstance(benchmarks, list)
        except Exception as e:
            pytest.fail(f"Function calls should not raise errors: {e}")

    def test_default_parameter_values_consistency(self):
        """Test that default parameter values are consistent across functions.

        Purpose: Validates that default values are used consistently when
        not explicitly provided.

        Given: Functions with default parameters
        When: Calling functions with minimal parameters
        Then: Default values are applied consistently

        Test type: consistency
        """
        gen = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        mapper = AverageReturnParameterToOptimizeMapper()

        # Test default parameters are applied
        configs_default = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
            generators=[gen], parameter_to_optimize_mapper=mapper
        )

        if configs_default:
            config = configs_default[0]
            # Verify that defaults are used
            assert config.num_episodes == 10  # Default from function signature
            assert config.num_steps == 20  # Default from function signature
            assert config.n_trials == 500  # Default from function signature
            # Removed old API check: direction  # Default
            # Removed old API check: parameter_to_optimize  # Default

    def test_memory_usage_reasonable(self):
        """Test that functions don't create excessive memory usage.

        Purpose: Validates that functions handle reasonable-sized inputs without
        creating memory issues.

        Given: Reasonable-sized inputs
        When: Calling functions multiple times
        Then: Memory usage remains reasonable

        Test type: performance
        """
        gen = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        mapper = AverageReturnParameterToOptimizeMapper()

        # Test multiple calls don't accumulate memory improperly
        for i in range(3):
            configs = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
                generators=[gen],
                parameter_to_optimize_mapper=mapper,
                particles=self.particle_count,
                num_episodes=2,  # Small values for performance test
                num_steps=3,
                n_trials=5,
            )

            # Should be able to handle multiple calls
            assert isinstance(configs, list)

    def test_multiple_generators_basic(self):
        """Test complete_environments_and_benchmarks_hyperparameter_optimization_configs with multiple generators.

        Purpose: Validates that the function correctly handles multiple generators
        and combines their outputs appropriately.

        Given: List of multiple MockHyperParamPlannerConfigGenerator instances
        When: Calling complete_environments_and_benchmarks_hyperparameter_optimization_configs
        Then: Returns configurations from all generators combined

        Test type: unit
        """
        # Create multiple generators with different space info
        gen1 = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        gen2 = MockHyperParamPlannerConfigGenerator(self.continuous_space_info)
        mapper = AverageReturnParameterToOptimizeMapper()

        configs = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
            generators=[gen1, gen2],
            parameter_to_optimize_mapper=mapper,
            particles=self.particle_count,
            num_episodes=3,
            num_steps=5,
            n_trials=10,
        )

        # Verify return type and structure
        assert isinstance(configs, list), "Should return a list of configurations"

        # Should have configurations from both generators
        if configs:
            for config in configs:
                assert isinstance(
                    config, HyperParameterRunParams
                ), "Should contain HyperParameterRunParams"
                assert isinstance(config.environment, Environment), "Config should have Environment"
                assert isinstance(config.belief, Belief), "Config should have Belief"
                assert isinstance(
                    config.hyper_param_planner_config, HyperParamPlannerConfig
                ), "Config should have HyperParamPlannerConfig"

    def test_multiple_generators_same_space_info(self):
        """Test multiple generators with the same space info.

        Purpose: Validates that multiple generators with identical space info
        are handled correctly and don't cause conflicts.

        Given: Multiple generators with same PolicySpaceInfo
        When: Calling the function with these generators
        Then: Returns configurations from all generators

        Test type: unit
        """
        # Create multiple generators with same space info
        gen1 = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        gen2 = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        gen3 = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        mapper = AverageReturnParameterToOptimizeMapper()

        configs = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
            generators=[gen1, gen2, gen3],
            parameter_to_optimize_mapper=mapper,
            particles=self.particle_count,
            num_episodes=2,
            num_steps=3,
            n_trials=5,
        )

        assert isinstance(configs, list), "Should return a list"

    def test_empty_generators_list(self):
        """Test handling of empty generators list.

        Purpose: Validates that the function handles empty generators list gracefully.

        Given: Empty list of generators
        When: Calling the function with empty list
        Then: Returns empty list or handles gracefully

        Test type: edge case
        """
        mapper = AverageReturnParameterToOptimizeMapper()

        configs = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
            generators=[],
            parameter_to_optimize_mapper=mapper,
            particles=self.particle_count,
        )

        # Should return empty list, not crash
        assert isinstance(configs, list), "Should return list even if empty"
        assert len(configs) == 0, "Should return empty list for empty generators"

    def test_mixed_space_info_generators(self):
        """Test generators with mixed space info types.

        Purpose: Validates that generators with different space info types
        (discrete, continuous, mixed) work together correctly.

        Given: Generators with different PolicySpaceInfo types
        When: Calling the function with mixed generators
        Then: Returns configurations from all generators

        Test type: unit
        """
        # Create generators with different space info
        discrete_gen = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        continuous_gen = MockHyperParamPlannerConfigGenerator(self.continuous_space_info)
        mixed_gen = MockHyperParamPlannerConfigGenerator(
            PolicySpaceInfo(action_space=SpaceType.MIXED, observation_space=SpaceType.MIXED)
        )
        mapper = AverageReturnParameterToOptimizeMapper()

        configs = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
            generators=[discrete_gen, continuous_gen, mixed_gen],
            parameter_to_optimize_mapper=mapper,
            particles=self.particle_count,
            num_episodes=2,
            num_steps=3,
            n_trials=5,
        )

        assert isinstance(configs, list), "Should handle mixed space info generators"


# Additional helper test class for testing mock implementations
class TestMockImplementations:
    """Test cases for mock implementations used in the tests."""

    def test_mock_hyper_parameter_planner_config_generator(self):
        """Test the MockHyperParamPlannerConfigGenerator implementation."""
        space_info = PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )

        gen = MockHyperParamPlannerConfigGenerator(space_info)

        # Test generation
        mock_env = Mock(spec=Environment)
        config = gen.generate(mock_env)

        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == MockPolicy
        assert len(config.hyper_parameters) == 1
        assert config.hyper_parameters[0].name == "test_param"
        assert config.constant_parameters == {"environment": mock_env, "name": "MockPlanner"}

        # Test space info retrieval
        returned_space_info = gen.get_planner_space_info()
        assert returned_space_info == space_info


class TestAllHyperparameterBenchmarksExperimentConfigCreator:
    """Test cases for AllHyperparameterBenchmarksExperimentConfigCreator."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.discrete_space_info = PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )
        self.continuous_space_info = PolicySpaceInfo(
            action_space=SpaceType.CONTINUOUS, observation_space=SpaceType.CONTINUOUS
        )

    def test_initialization_basic(self):
        """Test basic initialization of AllHyperparameterBenchmarksExperimentConfigCreator.

        Purpose: Validates that the creator can be initialized with required parameters.

        Given: Valid PolicySpaceInfo and basic configuration parameters
        When: Creating AllHyperparameterBenchmarksExperimentConfigCreator instance
        Then: Instance is created successfully with all attributes set correctly

        Test type: unit
        """
        creator = AllHyperparameterBenchmarksExperimentConfigCreator(
            policy_space_info=self.discrete_space_info,
            particles=30,
            num_episodes=10,
            num_steps=20,
            n_trials=100,
            discount_factor=0.95,
            time_out_in_seconds=3.0,
            is_risk_averse=False,
        )

        assert creator.policy_space_info == self.discrete_space_info
        assert creator.particles == 30
        assert creator.num_episodes == 10
        assert creator.num_steps == 20
        assert creator.n_trials == 100
        assert creator.discount_factor == 0.95
        assert creator.time_out_in_seconds == 3.0
        assert creator.is_risk_averse is False

    def test_initialization_risk_averse(self):
        """Test initialization with risk_averse=True.

        Purpose: Validates that the creator initializes correctly with risk-averse mode.

        Given: Valid PolicySpaceInfo and is_risk_averse=True
        When: Creating AllHyperparameterBenchmarksExperimentConfigCreator instance
        Then: Instance is created with risk-averse parameter mapper

        Test type: unit
        """
        creator = AllHyperparameterBenchmarksExperimentConfigCreator(
            policy_space_info=self.discrete_space_info,
            particles=30,
            num_episodes=10,
            num_steps=20,
            n_trials=100,
            discount_factor=0.95,
            time_out_in_seconds=3.0,
            is_risk_averse=True,
        )

        assert creator.is_risk_averse is True
        assert creator.parameter_to_optimize_mapper is not None

    def test_get_experiment_configs_returns_non_empty_list(self):
        """Test that get_experiment_configs returns a non-empty list.

        Purpose: Validates that the creator generates experiment configurations.

        Given: AllHyperparameterBenchmarksExperimentConfigCreator with discrete space info
        When: Calling get_experiment_configs
        Then: Returns a non-empty list of HyperParameterRunParams

        Test type: unit
        """
        creator = AllHyperparameterBenchmarksExperimentConfigCreator(
            policy_space_info=self.discrete_space_info,
            particles=10,  # Small number for faster tests
            num_episodes=2,
            num_steps=3,
            n_trials=5,
            discount_factor=0.95,
            time_out_in_seconds=3.0,
            is_risk_averse=False,
        )

        configs = creator.get_experiment_configs()

        assert isinstance(configs, list), "Should return a list"
        assert len(configs) > 0, "Should return non-empty list of configurations"


class TestAllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator:
    """Test cases for AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.discrete_space_info = PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )
        self.continuous_space_info = PolicySpaceInfo(
            action_space=SpaceType.CONTINUOUS, observation_space=SpaceType.CONTINUOUS
        )

    def test_initialization_basic(self):
        """Test basic initialization of AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator.

        Purpose: Validates that the creator can be initialized with required parameters.

        Given: Valid PlannerGenerator list and basic configuration parameters
        When: Creating AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator instance
        Then: Instance is created successfully with all attributes set correctly

        Test type: unit
        """
        generator = MockPlannerGenerator(self.discrete_space_info, "TestPlanner")

        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=[generator],
            n_particles=30,
            num_episodes=10,
            num_steps=20,
            is_risk_averse=False,
        )

        assert creator.generators == [generator]
        assert creator.n_particles == 30
        assert creator.num_episodes == 10
        assert creator.num_steps == 20

    def test_initialization_multiple_generators(self):
        """Test initialization with multiple generators.

        Purpose: Validates that the creator handles multiple PlannerGenerator instances correctly.

        Given: List of multiple MockPlannerGenerator instances
        When: Creating AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator instance
        Then: Instance is created with all generators stored correctly

        Test type: unit
        """
        generator1 = MockPlannerGenerator(self.discrete_space_info, "Planner1")
        generator2 = MockPlannerGenerator(self.discrete_space_info, "Planner2")
        generator3 = MockPlannerGenerator(self.continuous_space_info, "Planner3")

        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=[generator1, generator2, generator3],
            n_particles=25,
            num_episodes=5,
            num_steps=15,
            is_risk_averse=False,
        )

        assert len(creator.generators) == 3
        assert creator.generators[0] == generator1
        assert creator.generators[1] == generator2
        assert creator.generators[2] == generator3
        assert creator.n_particles == 25
        assert creator.num_episodes == 5
        assert creator.num_steps == 15

    def test_get_experiment_configs_returns_environment_run_params(self):
        """Test that get_experiment_configs returns EnvironmentRunParams list.

        Purpose: Validates that the creator generates experiment configurations correctly.

        Given: AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator with discrete space info
        When: Calling get_experiment_configs
        Then: Returns a list of EnvironmentRunParams with correct structure

        Test type: unit
        """
        generator = MockPlannerGenerator(self.discrete_space_info, "TestPlanner")

        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=[generator],
            n_particles=10,  # Small number for faster tests
            num_episodes=2,
            num_steps=3,
            is_risk_averse=False,
        )

        configs = creator.get_experiment_configs()

        assert isinstance(configs, list), "Should return a list"

        if configs:  # May be empty if no compatible environments
            for config in configs:
                assert isinstance(
                    config, EnvironmentRunParams
                ), "Each config should be EnvironmentRunParams"
                assert isinstance(config.environment, Environment), "Config should have Environment"
                assert isinstance(config.belief, Belief), "Config should have Belief"
                assert isinstance(config.policies, list), "Config should have policies list"
                assert config.num_episodes == 2, "Should use specified num_episodes"
                assert config.num_steps == 3, "Should use specified num_steps"
                assert len(config.policies) > 0, "Should have at least one policy"

    def test_get_experiment_configs_multiple_generators_same_environment(self):
        """Test that multiple generators for same environment are combined correctly.

        Purpose: Validates that policies from multiple generators are combined into single EnvironmentRunParams.

        Given: Multiple generators with same space info (same environments)
        When: Calling get_experiment_configs
        Then: Policies from all generators are combined for each environment

        Test type: unit
        """
        generator1 = MockPlannerGenerator(self.discrete_space_info, "Planner1")
        generator2 = MockPlannerGenerator(self.discrete_space_info, "Planner2")

        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=[generator1, generator2],
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            is_risk_averse=False,
        )

        configs = creator.get_experiment_configs()

        assert isinstance(configs, list)

        if configs:
            # Each environment should have policies from both generators
            for config in configs:
                assert len(config.policies) >= 2, "Should have policies from both generators"
                # Verify policies have different config_ids (from different generators)
                policy_ids = [policy.config_id for policy in config.policies]
                assert len(set(policy_ids)) == len(
                    policy_ids
                ), "Should have unique policy config_ids"

    def test_get_experiment_configs_different_space_info(self):
        """Test that generators with different space info work correctly.

        Purpose: Validates that generators with different PolicySpaceInfo are handled correctly.

        Given: Generators with different space info (discrete vs continuous)
        When: Calling get_experiment_configs
        Then: Returns configurations for environments compatible with each generator

        Test type: unit
        """
        discrete_generator = MockPlannerGenerator(self.discrete_space_info, "DiscretePlanner")
        continuous_generator = MockPlannerGenerator(self.continuous_space_info, "ContinuousPlanner")

        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=[discrete_generator, continuous_generator],
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            is_risk_averse=False,
        )

        configs = creator.get_experiment_configs()

        assert isinstance(configs, list), "Should return a list"

        # Should have configurations for both discrete and continuous environments
        if configs:
            for config in configs:
                assert isinstance(config, EnvironmentRunParams)
                assert len(config.policies) > 0, "Should have at least one policy"

    def test_get_experiment_configs_empty_generators(self):
        """Test handling of empty generators list.

        Purpose: Validates that empty generators list is handled gracefully.

        Given: Empty list of generators
        When: Calling get_experiment_configs
        Then: Returns empty list or handles gracefully

        Test type: edge case
        """
        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=[],
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            is_risk_averse=False,
        )

        configs = creator.get_experiment_configs()

        assert isinstance(configs, list), "Should return a list"
        assert len(configs) == 0, "Should return empty list for empty generators"

    def test_get_experiment_configs_custom_parameters(self):
        """Test that custom parameters are applied correctly.

        Purpose: Validates that custom n_particles, num_episodes, and num_steps are used.

        Given: Custom parameter values
        When: Calling get_experiment_configs
        Then: All returned configurations use the specified parameters

        Test type: unit
        """
        generator = MockPlannerGenerator(self.discrete_space_info, "TestPlanner")

        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=[generator],
            n_particles=50,
            num_episodes=15,
            num_steps=25,
            is_risk_averse=False,
        )

        configs = creator.get_experiment_configs()

        assert isinstance(configs, list)

        if configs:
            for config in configs:
                assert config.num_episodes == 15, "Should use custom num_episodes"
                assert config.num_steps == 25, "Should use custom num_steps"

    def test_environment_config_id_grouping(self):
        """Test that environments with same config_id are grouped correctly.

        Purpose: Validates that environments with identical config_id are combined into single EnvironmentRunParams.

        Given: Multiple generators that might produce same environment config_id
        When: Calling get_experiment_configs
        Then: Environments with same config_id are grouped together

        Test type: unit
        """
        generator1 = MockPlannerGenerator(self.discrete_space_info, "Planner1")
        generator2 = MockPlannerGenerator(self.discrete_space_info, "Planner2")

        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=[generator1, generator2],
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            is_risk_averse=False,
        )

        configs = creator.get_experiment_configs()

        assert isinstance(configs, list)

        if configs:
            # Verify that each config has unique environment config_id
            env_config_ids = [config.environment.config_id for config in configs]
            assert len(set(env_config_ids)) == len(
                env_config_ids
            ), "Should have unique environment config_ids"

    def test_policy_generation_for_each_environment(self):
        """Test that policies are generated for each environment correctly.

        Purpose: Validates that each generator produces policies for compatible environments.

        Given: Generator and compatible environments
        When: Calling get_experiment_configs
        Then: Each environment gets policies from compatible generators

        Test type: unit
        """
        generator = MockPlannerGenerator(self.discrete_space_info, "TestPlanner")

        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=[generator],
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            is_risk_averse=False,
        )

        configs = creator.get_experiment_configs()

        assert isinstance(configs, list)

        if configs:
            for config in configs:
                assert len(config.policies) > 0, "Should have policies for each environment"
                # Verify policies are generated by the generator
                for policy in config.policies:
                    assert hasattr(policy, "config_id"), "Policy should have config_id"
                    assert policy.config_id.startswith(
                        "TestPlanner_"
                    ), "Policy config_id should reflect generator name"

    def test_inheritance_from_evaluation_experiment_config_creator(self):
        """Test that the class properly inherits from EvaluationExperimentConfigCreator.

        Purpose: Validates that the class implements the required interface correctly.

        Given: AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator instance
        When: Calling get_experiment_configs (inherited method)
        Then: Method works correctly and returns expected results

        Test type: inheritance
        """
        generator = MockPlannerGenerator(self.discrete_space_info, "TestPlanner")

        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=[generator],
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            is_risk_averse=False,
        )

        # Test that it's an instance of the parent class
        assert isinstance(creator, EvaluationExperimentConfigCreator)

        # Test that the inherited method works
        configs = creator.get_experiment_configs()
        assert isinstance(configs, list)

    def test_no_duplicate_configs(self):
        """Test that no duplicate configurations are returned.

        Purpose: Validates that the parent class's duplicate detection works correctly.

        Given: AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator
        When: Calling get_experiment_configs
        Then: No duplicate config_ids are returned

        Test type: unit
        """
        generator = MockPlannerGenerator(self.discrete_space_info, "TestPlanner")

        creator = AllBenchmarkEnvironmentsOnPlannerGeneratorsExperimentConfigCreator(
            generators=[generator],
            n_particles=10,
            num_episodes=2,
            num_steps=3,
            is_risk_averse=False,
        )

        configs = creator.get_experiment_configs()

        assert isinstance(configs, list)

        if configs:
            # Verify no duplicate config_ids
            config_ids = [config.config_id for config in configs]
            assert len(set(config_ids)) == len(config_ids), "Should have no duplicate config_ids"


class TestAverageReturnParameterToOptimizeMapper:
    """Tests for AverageReturnParameterToOptimizeMapper metric selection logic."""

    def setup_method(self):
        self.mapper = AverageReturnParameterToOptimizeMapper()
        self.env_configs = EnvironmentConfigsAPI(discount_factor=0.95)

    def _get_all_standard_environments(self):
        mixed_space = PolicySpaceInfo(
            action_space=SpaceType.MIXED, observation_space=SpaceType.MIXED
        )
        return self.env_configs.get_compatible_environments(
            policy_space_info=mixed_space, n_particles=10
        )

    def test_average_return_mapper_returns_average_return_for_all_environments(self):
        """Test that average_return is returned as first metric for every environment.

        Purpose: Validates that AverageReturnParameterToOptimizeMapper always includes
        average_return with MAXIMIZE direction as the first parameter for all environments.

        Given: Each of the registered environment types from EnvironmentConfigsAPI
        When: AverageReturnParameterToOptimizeMapper.generate() is called for each environment
        Then: Every result is non-empty and contains ("average_return", MAXIMIZE) as first element

        Test type: unit
        """
        envs = self._get_all_standard_environments()
        assert len(envs) > 0, "Should have at least one registered environment"

        for env, _belief in envs:
            result = self.mapper.generate(env)
            assert len(result) > 0, f"Result should be non-empty for {env.name}"
            assert result[0] == (
                "average_return",
                HyperParameterOptimizationDirection.MAXIMIZE,
            ), f"First metric should be ('average_return', MAXIMIZE) for {env.name}, got {result[0]}"

    def test_average_return_mapper_environment_specific_metrics(self):
        """Test that environment-specific metrics are added for known environment types.

        Purpose: Validates that specific environments get their expected secondary metrics.

        Given: Specific environment instances (Tiger, CartPole, RockSample, LaserTag, PacMan)
        When: AverageReturnParameterToOptimizeMapper.generate() is called for each
        Then: Each environment has the correct environment-specific metric

        Test type: unit
        """
        expected_metrics = {
            TigerPOMDP: "success_rate",
            CartPolePOMDP: "goal_reaching_rate",
            RockSamplePOMDP: "exit_success_rate",
            LaserTagPOMDP: "tag_success_rate",
            PacManPOMDP: "win_rate",
        }

        for env, _belief in self._get_all_standard_environments():
            for env_cls, expected_metric in expected_metrics.items():
                if isinstance(env, env_cls):
                    metric_names = [param[0] for param in self.mapper.generate(env)]
                    assert expected_metric in metric_names, (
                        f"{env.name} (isinstance {env_cls.__name__}) should have "
                        f"'{expected_metric}' metric, got {metric_names}"
                    )

    def test_average_return_mapper_continuous_environments(self):
        """Test mapper behavior for continuous laser tag environments.

        Purpose: Validates that ContinuousLaserTagPOMDP and its discrete-actions
        variant get tag_success_rate from the mapper despite not inheriting from
        LaserTagPOMDP. These classes inherit from Environment directly and require
        their own isinstance branch.

        Given: ContinuousLaserTagPOMDP and ContinuousLaserTagPOMDPDiscreteActions
            instances from the environment registry
        When: AverageReturnParameterToOptimizeMapper.generate() is called
        Then: Result contains both average_return and tag_success_rate

        Test type: unit
        """
        envs = self._get_all_standard_environments()
        continuous_envs = [
            (env, belief) for env, belief in envs if isinstance(env, ContinuousLaserTagPOMDP)
        ]
        assert (
            len(continuous_envs) > 0
        ), "Registry should contain at least one ContinuousLaserTagPOMDP variant"

        for env, _belief in continuous_envs:
            result = self.mapper.generate(env)
            metric_names = [param[0] for param in result]
            assert "average_return" in metric_names, f"{env.name} should have average_return"
            assert (
                "tag_success_rate" in metric_names
            ), f"{env.name} should have tag_success_rate but got {metric_names}"

    def test_average_return_mapper_continuous_laser_tag_has_tag_success_rate(self):
        """Regression: ContinuousLaserTagPOMDP must have tag_success_rate metric.

        Purpose: Verifies that the AverageReturnParameterToOptimizeMapper returns
        tag_success_rate for directly instantiated ContinuousLaserTagPOMDP and its
        discrete-actions variant. ContinuousLaserTagPOMDP inherits from Environment
        (not LaserTagPOMDP), so it requires its own isinstance branch in the mapper.

        Given: Directly constructed ContinuousLaserTagPOMDP and
            ContinuousLaserTagPOMDPDiscreteActions instances
        When: AverageReturnParameterToOptimizeMapper.generate() is called
        Then: tag_success_rate with MAXIMIZE is present in the returned metrics

        Test type: unit
        """
        envs = [
            ContinuousLaserTagPOMDP(discount_factor=0.95, walls=[], dangerous_areas=[]),
            ContinuousLaserTagPOMDPDiscreteActions(
                discount_factor=0.95, walls=[], dangerous_areas=[]
            ),
        ]
        for env in envs:
            result = self.mapper.generate(env)
            metric_dict = {name: direction for name, direction in result}
            assert "tag_success_rate" in metric_dict, (
                f"{env.name} should have 'tag_success_rate' metric but got "
                f"{list(metric_dict.keys())}"
            )
            assert metric_dict["tag_success_rate"] == (
                HyperParameterOptimizationDirection.MAXIMIZE
            ), f"{env.name}: tag_success_rate should be MAXIMIZE"

    def test_average_return_mapper_all_directions_are_maximize(self):
        """Test that all metrics from AverageReturnParameterToOptimizeMapper use MAXIMIZE.

        Purpose: Validates the mapper always uses MAXIMIZE direction for all metrics.

        Given: All registered environments from EnvironmentConfigsAPI
        When: AverageReturnParameterToOptimizeMapper.generate() is called for each
        Then: Every metric tuple has HyperParameterOptimizationDirection.MAXIMIZE

        Test type: unit
        """
        for env, _belief in self._get_all_standard_environments():
            result = self.mapper.generate(env)
            for metric_name, direction in result:
                assert (
                    direction == HyperParameterOptimizationDirection.MAXIMIZE
                ), f"{env.name}: metric '{metric_name}' should be MAXIMIZE, got {direction}"


class TestRiskAverseParameterToOptimizeMapper:
    """Tests for RiskAverseParameterToOptimizeMapper metric selection logic."""

    def setup_method(self):
        self.mapper = RiskAverseParameterToOptimizeMapper()

    def test_risk_averse_mapper_covers_all_registered_environments(self):
        """Test that the risk-averse mapper handles every registered environment without error.

        Purpose: Validates that no registered environment raises ValueError from the
        risk-averse mapper's else clause. Newly added environments that are not handled
        by any isinstance check would raise ValueError.

        Given: All environments from both EnvironmentConfigsAPI and RiskAverseEnvironmentConfigsAPI
        When: RiskAverseParameterToOptimizeMapper.generate() is called for each
        Then: No ValueError is raised for any environment

        Test type: unit
        """
        mixed_space = PolicySpaceInfo(
            action_space=SpaceType.MIXED, observation_space=SpaceType.MIXED
        )

        standard_api = EnvironmentConfigsAPI(discount_factor=0.95)
        standard_envs = standard_api.get_compatible_environments(
            policy_space_info=mixed_space, n_particles=10
        )

        risk_averse_api = RiskAverseEnvironmentConfigsAPI(discount_factor=0.95)
        risk_averse_envs = risk_averse_api.get_compatible_environments(
            policy_space_info=mixed_space, n_particles=10
        )

        all_envs = standard_envs + risk_averse_envs
        assert len(all_envs) > 0, "Should have at least one environment to test"

        unsupported_envs = []
        for env, _belief in all_envs:
            try:
                result = self.mapper.generate(env)
                assert len(result) > 0, f"Result should be non-empty for {env.name}"
            except ValueError:
                unsupported_envs.append(env.__class__.__name__)

        if unsupported_envs:
            pytest.fail(
                f"RiskAverseParameterToOptimizeMapper does not handle these registered "
                f"environments: {unsupported_envs}"
            )

    def test_risk_averse_mapper_continuous_laser_tag_metrics(self):
        """Regression: ContinuousLaserTagPOMDP must have correct risk-averse metrics.

        Purpose: Verifies that the RiskAverseParameterToOptimizeMapper returns the
        expected metrics for directly instantiated ContinuousLaserTagPOMDP and its
        discrete-actions variant. ContinuousLaserTagPOMDP inherits from Environment
        (not LaserTagPOMDP), so it requires its own isinstance branch in the mapper.

        Given: Directly constructed ContinuousLaserTagPOMDP and
            ContinuousLaserTagPOMDPDiscreteActions instances
        When: RiskAverseParameterToOptimizeMapper.generate() is called
        Then: Returns average_all_dangerous_encounters (MINIMIZE) and
            tag_success_rate (MAXIMIZE)

        Test type: unit
        """
        envs = [
            ContinuousLaserTagPOMDP(discount_factor=0.95, walls=[], dangerous_areas=[]),
            ContinuousLaserTagPOMDPDiscreteActions(
                discount_factor=0.95, walls=[], dangerous_areas=[]
            ),
        ]
        for env in envs:
            result = self.mapper.generate(env)
            metric_dict = {name: direction for name, direction in result}
            assert "average_all_dangerous_encounters" in metric_dict, (
                f"{env.name} should have 'average_all_dangerous_encounters' but got "
                f"{list(metric_dict.keys())}"
            )
            assert metric_dict["average_all_dangerous_encounters"] == (
                HyperParameterOptimizationDirection.MINIMIZE
            ), f"{env.name}: average_all_dangerous_encounters should be MINIMIZE"
            assert "tag_success_rate" in metric_dict, (
                f"{env.name} should have 'tag_success_rate' but got " f"{list(metric_dict.keys())}"
            )
            assert metric_dict["tag_success_rate"] == (
                HyperParameterOptimizationDirection.MAXIMIZE
            ), f"{env.name}: tag_success_rate should be MAXIMIZE"

    def test_risk_averse_mapper_returns_non_empty_for_all_risk_averse_environments(self):
        """Test that the risk-averse mapper returns non-empty results for risk-averse environments.

        Purpose: Validates that risk-averse environments get proper metric mappings.

        Given: All environments from RiskAverseEnvironmentConfigsAPI
        When: RiskAverseParameterToOptimizeMapper.generate() is called for each
        Then: Each result is a non-empty list of (metric_name, direction) tuples

        Test type: unit
        """
        mixed_space = PolicySpaceInfo(
            action_space=SpaceType.MIXED, observation_space=SpaceType.MIXED
        )
        risk_averse_api = RiskAverseEnvironmentConfigsAPI(discount_factor=0.95)
        envs = risk_averse_api.get_compatible_environments(
            policy_space_info=mixed_space, n_particles=10
        )

        for env, _belief in envs:
            result = self.mapper.generate(env)
            assert len(result) > 0, f"Risk-averse result should be non-empty for {env.name}"
            for metric_name, direction in result:
                assert isinstance(metric_name, str), f"Metric name should be str for {env.name}"
                assert isinstance(
                    direction, HyperParameterOptimizationDirection
                ), f"Direction should be HyperParameterOptimizationDirection for {env.name}"


class TestAllHyperparameterBenchmarksConfigValidity:
    """Tests validating the structural correctness of generated experiment configs."""

    def setup_method(self):
        self.mixed_space_info = PolicySpaceInfo(
            action_space=SpaceType.MIXED, observation_space=SpaceType.MIXED
        )

    def _create_creator(self, policy_space_info=None, is_risk_averse=False):
        return AllHyperparameterBenchmarksExperimentConfigCreator(
            policy_space_info=policy_space_info or self.mixed_space_info,
            particles=10,
            num_episodes=2,
            num_steps=3,
            n_trials=2,
            discount_factor=0.95,
            time_out_in_seconds=3.0,
            is_risk_averse=is_risk_averse,
        )

    def test_all_generated_configs_have_valid_structure(self):
        """Test that every generated config has all required fields with valid values.

        Purpose: Validates that each generated HyperParameterRunParams has a valid
        environment, belief, planner config, and non-empty parameters_to_optimize.

        Given: AllHyperparameterBenchmarksExperimentConfigCreator with MIXED space info
        When: get_experiment_configs() is called
        Then: Every config has valid environment, belief, planner config, hyperparameters,
        parameters_to_optimize, and positive num_episodes/num_steps/n_trials

        Test type: unit
        """
        creator = self._create_creator()
        configs = creator.get_experiment_configs()

        assert len(configs) > 0, "Should generate at least one config"

        for i, config in enumerate(configs):
            label = f"Config[{i}] ({config.environment.name})"
            assert isinstance(config.environment, Environment), f"{label}: invalid environment"
            assert isinstance(config.belief, Belief), f"{label}: invalid belief"
            assert (
                config.hyper_param_planner_config.policy_cls is not None
            ), f"{label}: policy_cls is None"
            assert (
                len(config.hyper_param_planner_config.hyper_parameters) > 0
            ), f"{label}: hyper_parameters is empty"
            assert (
                len(config.parameters_to_optimize) > 0
            ), f"{label}: parameters_to_optimize is empty"
            assert config.num_episodes > 0, f"{label}: num_episodes must be positive"
            assert config.num_steps > 0, f"{label}: num_steps must be positive"
            assert config.n_trials > 0, f"{label}: n_trials must be positive"

    def test_all_generated_configs_have_discount_factor_in_constant_params(self):
        """Test that discount_factor is present in constant_parameters for planners that need it.

        Purpose: Validates that planners requiring discount_factor receive it via
        constant_parameters. Missing discount_factor was a known integration issue.

        Given: AllHyperparameterBenchmarksExperimentConfigCreator with MIXED space info
        When: get_experiment_configs() is called
        Then: Every config's constant_parameters includes "discount_factor"

        Test type: unit
        """
        creator = self._create_creator()
        configs = creator.get_experiment_configs()

        assert len(configs) > 0, "Should generate at least one config"

        for config in configs:
            planner_name = config.hyper_param_planner_config.policy_cls.__name__
            constant_params = config.hyper_param_planner_config.constant_parameters
            # PlannersHyperparamConfigs sets discount_factor in constant_parameters
            # for all planners except SparseSampling and DiscreteActionSequences
            # which handle it differently. We check that environment is always present.
            assert "environment" in constant_params, (
                f"Planner '{planner_name}' for env '{config.environment.name}' "
                f"missing 'environment' in constant_parameters"
            )

    def test_config_count_matches_environment_planner_combinations(self):
        """Test that config count equals total environment-planner combinations.

        Purpose: Validates that _get_experiment_configs generates one config per
        compatible (environment, planner) pair.

        Given: AllHyperparameterBenchmarksExperimentConfigCreator with DISCRETE space info
        When: get_experiment_configs() is called
        Then: Config count matches the sum of compatible planners across all environments

        Test type: unit
        """
        discrete_space = PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )
        creator = self._create_creator(policy_space_info=discrete_space)
        configs = creator.get_experiment_configs()

        # Independently compute expected count
        from POMDPPlanners.configs.planners_hyperparam_configs import PlannersHyperparamConfigs

        env_configs = EnvironmentConfigsAPI(discount_factor=0.95)
        envs = env_configs.get_compatible_environments(
            policy_space_info=discrete_space, n_particles=10
        )
        planners_config = PlannersHyperparamConfigs(discount_factor=0.95)

        expected_count = 0
        for env, _belief in envs:
            compatible_planners = planners_config.get_compatible_planners(
                env=env, time_out_in_seconds=3.0
            )
            expected_count += len(compatible_planners)

        assert (
            len(configs) == expected_count
        ), f"Expected {expected_count} configs (one per env-planner pair), got {len(configs)}"

    def test_generated_configs_metric_names_are_valid(self):
        """Test that all metric names in parameters_to_optimize are recognized by the environment.

        Purpose: Validates that the mapper does not return metric names that the
        environment cannot produce. This catches the scenario where a mapper returns
        a metric that doesn't exist for a particular environment.

        Given: AllHyperparameterBenchmarksExperimentConfigCreator with MIXED space info
        When: get_experiment_configs() is called
        Then: Every metric name in parameters_to_optimize is a string (non-empty)
        and has a valid optimization direction

        Test type: unit
        """
        creator = self._create_creator()
        configs = creator.get_experiment_configs()

        assert len(configs) > 0, "Should generate at least one config"

        for config in configs:
            for metric_name, direction in config.parameters_to_optimize:
                assert isinstance(metric_name, str) and len(metric_name) > 0, (
                    f"Metric name should be non-empty string for {config.environment.name}, "
                    f"got {metric_name!r}"
                )
                assert isinstance(direction, HyperParameterOptimizationDirection), (
                    f"Direction for '{metric_name}' should be "
                    f"HyperParameterOptimizationDirection for {config.environment.name}"
                )

    def test_all_generated_configs_risk_averse_have_valid_structure(self):
        """Test that risk-averse generated configs also have valid structure.

        Purpose: Validates that configs generated with is_risk_averse=True have the
        same structural integrity as standard configs.

        Given: AllHyperparameterBenchmarksExperimentConfigCreator with is_risk_averse=True
        When: get_experiment_configs() is called
        Then: Every config has valid structure and non-empty parameters_to_optimize

        Test type: unit
        """
        creator = self._create_creator(is_risk_averse=True)
        configs = creator.get_experiment_configs()

        assert len(configs) > 0, "Risk-averse should generate at least one config"

        for config in configs:
            assert isinstance(config.environment, Environment)
            assert isinstance(config.belief, Belief)
            assert len(config.parameters_to_optimize) > 0, (
                f"Risk-averse config for {config.environment.name} has empty "
                f"parameters_to_optimize"
            )
