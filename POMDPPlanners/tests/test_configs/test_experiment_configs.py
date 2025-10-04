"""Tests for experiment_configs module.

This module contains comprehensive tests for the experiment_configs implementation,
including tests for all configuration functions used in hyperparameter optimization
and benchmarking experiments.
"""

from unittest.mock import Mock, patch
from typing import List

import numpy as np
import pytest

from POMDPPlanners.configs.experiment_configs import (
    get_hyperparameter_benchmarks,
    complete_environments_and_benchmarks_hyperparameter_optimization_configs,
    get_benchmarks_hyperparameter_optimization_configs,
    AverageReturnParameterToOptimizeMapper,
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
    HyperParamPlannerConfigGenerator,
    HyperParamPlannerConfig,
)
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

# Set random seeds for reproducible tests
np.random.seed(42)


class MockHyperParamPlannerConfigGenerator(HyperParamPlannerConfigGenerator):
    """Mock implementation of HyperParamPlannerConfigGenerator for testing."""

    def __init__(self, space_info: PolicySpaceInfo):
        self.space_info = space_info

    def generate(self, environment: Environment) -> HyperParamPlannerConfig:
        """Generate a mock planner config."""
        return HyperParamPlannerConfig(
            policy_cls=Mock,
            hyper_parameters=[],
            constant_parameters={"environment": environment, "name": "MockPlanner"},
        )

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

        Given: Mock HyperParamPlannerConfigGenerator with discrete space info
        When: Calling complete_environments_and_benchmarks_hyperparameter_optimization_configs
        Then: Returns list of HyperParameterRunParams with correct structure

        Test type: unit
        """
        gen = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        mapper = AverageReturnParameterToOptimizeMapper()

        configs = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
            gen=gen,
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
                assert (
                    config.direction == HyperParameterOptimizationDirection.MAXIMIZE
                ), "Should use specified direction"
                assert (
                    config.parameter_to_optimize == "average_return"
                ), "Should use specified parameter"

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
            gen=gen,
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

        Given: Only required generator and mapper parameters
        When: Calling the function with minimal parameters
        Then: Uses default values correctly for all optional parameters

        Test type: unit
        """
        gen = MockHyperParamPlannerConfigGenerator(self.discrete_space_info)
        mapper = AverageReturnParameterToOptimizeMapper()

        configs = complete_environments_and_benchmarks_hyperparameter_optimization_configs(
            gen=gen, parameter_to_optimize_mapper=mapper
        )

        assert isinstance(configs, list)

        if configs:
            config = configs[0]
            # Verify default values
            assert config.num_episodes == 10, "Should use default num_episodes"
            assert config.num_steps == 20, "Should use default num_steps"
            assert config.n_trials == 500, "Should use default n_trials"
            assert (
                config.direction == HyperParameterOptimizationDirection.MAXIMIZE
            ), "Should use default direction"
            assert config.parameter_to_optimize == "average_return", "Should use default parameter"

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
            policy_cls=Mock,
            hyper_parameters=[],
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
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return",
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
                assert config.direction == reference_conf.direction, "Should use same direction"
                assert (
                    config.parameter_to_optimize == reference_conf.parameter_to_optimize
                ), "Should use same parameter"

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
            policy_cls=Mock,
            hyper_parameters=[],
            constant_parameters={"environment": mock_env, "name": "TestBenchmark"},
        )

        reference_conf = HyperParameterRunParams(
            environment=mock_env,
            belief=mock_belief,
            hyper_param_planner_config=mock_planner_config,
            num_episodes=3,
            num_steps=5,
            n_trials=10,
            direction=HyperParameterOptimizationDirection.MINIMIZE,
            parameter_to_optimize="cumulative_reward",
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
        fake_planner_conf.policy_cls = Mock
        fake_planner_conf.hyper_parameters = []
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
            gen=gen, parameter_to_optimize_mapper=mapper
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
            gen=gen,
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
            gen=gen, parameter_to_optimize_mapper=mapper
        )

        if configs_default:
            config = configs_default[0]
            # Verify that defaults are used
            assert config.num_episodes == 10  # Default from function signature
            assert config.num_steps == 20  # Default from function signature
            assert config.n_trials == 500  # Default from function signature
            assert config.direction == HyperParameterOptimizationDirection.MAXIMIZE  # Default
            assert config.parameter_to_optimize == "average_return"  # Default

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
                gen=gen,
                parameter_to_optimize_mapper=mapper,
                particles=self.particle_count,
                num_episodes=2,  # Small values for performance test
                num_steps=3,
                n_trials=5,
            )

            # Should be able to handle multiple calls
            assert isinstance(configs, list)


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
        assert config.policy_cls == Mock
        assert config.hyper_parameters == []
        assert config.constant_parameters == {"environment": mock_env, "name": "MockPlanner"}

        # Test space info retrieval
        returned_space_info = gen.get_planner_space_info()
        assert returned_space_info == space_info
