"""Tests for PlannersHyperparamConfigs class.

This module contains comprehensive tests for the PlannersHyperparamConfigs implementation,
including tests for all planner configuration methods and their hyperparameter ranges.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from POMDPPlanners.configs.planners_hyperparam_configs import PlannersHyperparamConfigs
from POMDPPlanners.utils.hyperparameter_tuning_and_eval import HyperParamPlannerConfig
from POMDPPlanners.core.simulation import NumericalHyperParameter, CategoricalHyperParameter
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.mcts_planners.pomcp_dpw import POMCP_DPW
from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import DiscreteActionSequencesPlanner
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


class TestPlannersHyperparamConfigs:
    """Test cases for PlannersHyperparamConfigs class."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.discount_factor = 0.95
        self.config_api = PlannersHyperparamConfigs(discount_factor=self.discount_factor)
        
        # Create mock environment with reward range
        self.mock_env = Mock()
        self.mock_env.reward_range = (-10.0, 100.0)  # Example reward range
        
        # Create mock action sampler
        self.mock_action_sampler = Mock(spec=ActionSampler)
        
        self.planner_name = "TestPlanner"

    def test_initialization(self):
        """Test PlannersHyperparamConfigs initialization.

        Purpose: Validates that PlannersHyperparamConfigs initializes correctly with discount factor

        Given: A discount factor value
        When: PlannersHyperparamConfigs is instantiated
        Then: The discount factor is stored correctly

        Test type: unit
        """
        config_api = PlannersHyperparamConfigs(discount_factor=0.9)
        assert config_api.discount_factor == 0.9

    def test_pft_dpw_config(self):
        """Test PFT_DPW configuration creation.

        Purpose: Validates that PFT_DPW configuration is created correctly with proper hyperparameters

        Given: A mock environment, action sampler, and planner name
        When: pft_dpw_config is called
        Then: Returns valid HyperParamPlannerConfig with correct policy class and parameters

        Test type: unit
        """
        config = self.config_api.pft_dpw_config(
            env=self.mock_env,
            action_sampler=self.mock_action_sampler,
            name=self.planner_name
        )

        # Verify return type
        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == PFT_DPW

        # Verify hyperparameters
        assert len(config.hyper_parameters) == 7
        param_names = [param.name for param in config.hyper_parameters]
        expected_params = ["exploration_constant", "time_out_in_seconds", "depth", "k_a", "alpha_a", "k_o", "alpha_o"]
        assert set(param_names) == set(expected_params)

        # Verify constant parameters
        assert config.constant_parameters["discount_factor"] == self.discount_factor
        assert config.constant_parameters["name"] == self.planner_name
        assert config.constant_parameters["environment"] == self.mock_env
        assert config.constant_parameters["action_sampler"] == self.mock_action_sampler

    def test_pft_dpw_config_hyperparameter_ranges(self):
        """Test PFT_DPW hyperparameter ranges are correct.

        Purpose: Validates that PFT_DPW hyperparameters have correct ranges and types

        Given: A mock environment with specific reward range
        When: pft_dpw_config is called
        Then: All hyperparameters have expected ranges and are NumericalHyperParameter instances

        Test type: unit
        """
        config = self.config_api.pft_dpw_config(
            env=self.mock_env,
            action_sampler=self.mock_action_sampler,
            name=self.planner_name
        )

        # Check specific parameter ranges
        param_dict = {param.name: param for param in config.hyper_parameters}
        
        # Exploration constant should be based on reward range
        exploration_param = param_dict["exploration_constant"]
        expected_max = (self.mock_env.reward_range[1] - self.mock_env.reward_range[0]) * 15
        assert isinstance(exploration_param, NumericalHyperParameter)
        assert exploration_param.low == 0.0
        assert exploration_param.high == expected_max

        # Check other parameters
        assert param_dict["time_out_in_seconds"].low == 0.1
        assert param_dict["time_out_in_seconds"].high == 15.0
        assert param_dict["depth"].low == 2
        assert param_dict["depth"].high == 15
        assert param_dict["k_a"].low == 1.0
        assert param_dict["k_a"].high == 8.0

    def test_pomcpow_config(self):
        """Test POMCPOW configuration creation.

        Purpose: Validates that POMCPOW configuration is created correctly with proper hyperparameters

        Given: A mock environment, action sampler, and planner name
        When: pomcpow_config is called
        Then: Returns valid HyperParamPlannerConfig with correct policy class and parameters

        Test type: unit
        """
        config = self.config_api.pomcpow_config(
            env=self.mock_env,
            action_sampler=self.mock_action_sampler,
            name=self.planner_name
        )

        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == POMCPOW
        assert len(config.hyper_parameters) == 7

        # Verify constant parameters
        assert config.constant_parameters["discount_factor"] == self.discount_factor
        assert config.constant_parameters["action_sampler"] == self.mock_action_sampler

    def test_sparse_pft_config(self):
        """Test SparsePFT configuration creation.

        Purpose: Validates that SparsePFT configuration is created correctly with proper hyperparameters

        Given: A mock environment and planner name (no action sampler needed)
        When: sparse_pft_config is called
        Then: Returns valid HyperParamPlannerConfig with correct policy class and parameters

        Test type: unit
        """
        config = self.config_api.sparse_pft_config(
            env=self.mock_env,
            name=self.planner_name
        )

        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == SparsePFT
        assert len(config.hyper_parameters) == 6

        param_names = [param.name for param in config.hyper_parameters]
        expected_params = ["exploration_constant", "time_out_in_seconds", "depth", "c_ucb", "beta_ucb", "belief_child_num"]
        assert set(param_names) == set(expected_params)

        # Check unique parameter "gamma" instead of "discount_factor"
        assert config.constant_parameters["gamma"] == self.discount_factor
        assert "action_sampler" not in config.constant_parameters

    def test_sparse_sampling_config(self):
        """Test StandardSparseSampling configuration creation.

        Purpose: Validates that StandardSparseSampling configuration is created correctly with mixed parameter types

        Given: A mock environment and planner name
        When: sparse_sampling_config is called
        Then: Returns valid HyperParamPlannerConfig with numerical and categorical parameters

        Test type: unit
        """
        config = self.config_api.sparse_sampling_config(
            env=self.mock_env,
            name=self.planner_name
        )

        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == StandardSparseSamplingDiscreteActionsPlanner
        assert len(config.hyper_parameters) == 3

        # Check parameter types
        param_dict = {param.name: param for param in config.hyper_parameters}
        assert isinstance(param_dict["branching_factor"], NumericalHyperParameter)
        assert isinstance(param_dict["depth"], NumericalHyperParameter)
        assert isinstance(param_dict["resampling"], CategoricalHyperParameter)

        # Check categorical parameter values
        resampling_param = param_dict["resampling"]
        assert resampling_param.choices == [True, False]

    def test_pomcp_config(self):
        """Test POMCP configuration creation.

        Purpose: Validates that POMCP configuration is created correctly with basic MCTS parameters

        Given: A mock environment and planner name
        When: pomcp_config is called
        Then: Returns valid HyperParamPlannerConfig with correct policy class and parameters

        Test type: unit
        """
        config = self.config_api.pomcp_config(
            env=self.mock_env,
            name=self.planner_name
        )

        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == POMCP
        assert len(config.hyper_parameters) == 3

        param_names = [param.name for param in config.hyper_parameters]
        expected_params = ["exploration_constant", "time_out_in_seconds", "depth"]
        assert set(param_names) == set(expected_params)

        # Check unique parameter
        assert config.constant_parameters["min_samples_per_node"] == 1

    def test_pomcp_dpw_config(self):
        """Test POMCP_DPW configuration creation.

        Purpose: Validates that POMCP_DPW configuration is created correctly with DPW-specific parameters

        Given: A mock environment, action sampler, and planner name
        When: pomcp_dpw_config is called
        Then: Returns valid HyperParamPlannerConfig with DPW hyperparameters

        Test type: unit
        """
        config = self.config_api.pomcp_dpw_config(
            env=self.mock_env,
            action_sampler=self.mock_action_sampler,
            name=self.planner_name
        )

        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == POMCP_DPW
        assert len(config.hyper_parameters) == 8  # Most parameters including min_samples_per_node

        param_names = [param.name for param in config.hyper_parameters]
        expected_params = ["exploration_constant", "time_out_in_seconds", "depth", "k_a", "alpha_a", "k_o", "alpha_o", "min_samples_per_node"]
        assert set(param_names) == set(expected_params)

    def test_discrete_action_sequences_config(self):
        """Test DiscreteActionSequences configuration creation.

        Purpose: Validates that DiscreteActionSequences configuration is created correctly with open-loop parameters

        Given: A mock environment and planner name
        When: discrete_action_sequences_config is called
        Then: Returns valid HyperParamPlannerConfig with planning horizon and sampling parameters

        Test type: unit
        """
        config = self.config_api.discrete_action_sequences_config(
            env=self.mock_env,
            name=self.planner_name
        )

        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == DiscreteActionSequencesPlanner
        assert len(config.hyper_parameters) == 2

        param_names = [param.name for param in config.hyper_parameters]
        expected_params = ["depth", "n_return_samples"]
        assert set(param_names) == set(expected_params)

        # Check parameter ranges
        param_dict = {param.name: param for param in config.hyper_parameters}
        assert param_dict["depth"].low == 2
        assert param_dict["depth"].high == 3
        assert param_dict["n_return_samples"].low == 10
        assert param_dict["n_return_samples"].high == 500

    def test_reward_range_scaling(self):
        """Test that exploration constant scales with reward range.

        Purpose: Validates that exploration constant hyperparameters scale correctly with environment reward ranges

        Given: Environments with different reward ranges
        When: Configuration methods are called
        Then: Exploration constant ranges scale proportionally to reward range

        Test type: unit
        """
        # Test with different reward ranges
        small_range_env = Mock()
        small_range_env.reward_range = (-1.0, 1.0)  # Range of 2

        large_range_env = Mock()
        large_range_env.reward_range = (-100.0, 500.0)  # Range of 600

        small_config = self.config_api.pomcp_config(small_range_env, "SmallRange")
        large_config = self.config_api.pomcp_config(large_range_env, "LargeRange")

        small_exploration = next(p for p in small_config.hyper_parameters if p.name == "exploration_constant")
        large_exploration = next(p for p in large_config.hyper_parameters if p.name == "exploration_constant")

        # Large range should have much larger exploration constant maximum
        assert large_exploration.high > small_exploration.high
        
        # Check proportionality (approximately)
        small_range = small_range_env.reward_range[1] - small_range_env.reward_range[0]
        large_range = large_range_env.reward_range[1] - large_range_env.reward_range[0]
        
        expected_ratio = large_range / small_range
        actual_ratio = large_exploration.high / small_exploration.high
        assert abs(actual_ratio - expected_ratio) < 0.01  # Allow small floating point error

    def test_all_configs_return_hyperparamplannerconfig(self):
        """Test that all configuration methods return HyperParamPlannerConfig instances.

        Purpose: Validates that all configuration methods return the correct type

        Given: Mock environment and required parameters
        When: Each configuration method is called
        Then: All methods return HyperParamPlannerConfig instances

        Test type: unit
        """
        configs = [
            self.config_api.pft_dpw_config(self.mock_env, self.mock_action_sampler, "PFT_DPW"),
            self.config_api.pomcpow_config(self.mock_env, self.mock_action_sampler, "POMCPOW"),
            self.config_api.sparse_pft_config(self.mock_env, "SparsePFT"),
            self.config_api.sparse_sampling_config(self.mock_env, "SparseSampling"),
            self.config_api.pomcp_config(self.mock_env, "POMCP"),
            self.config_api.pomcp_dpw_config(self.mock_env, self.mock_action_sampler, "POMCP_DPW"),
            self.config_api.discrete_action_sequences_config(self.mock_env, "DiscreteSequences")
        ]

        for config in configs:
            assert isinstance(config, HyperParamPlannerConfig)
            assert hasattr(config, 'policy_cls')
            assert hasattr(config, 'hyper_parameters')
            assert hasattr(config, 'constant_parameters')

    def test_hyperparameter_names_uniqueness(self):
        """Test that hyperparameter names are unique within each configuration.

        Purpose: Validates that no configuration has duplicate hyperparameter names

        Given: Any planner configuration
        When: Hyperparameters are examined
        Then: All parameter names are unique within each configuration

        Test type: unit
        """
        configs = [
            self.config_api.pft_dpw_config(self.mock_env, self.mock_action_sampler, "PFT_DPW"),
            self.config_api.pomcpow_config(self.mock_env, self.mock_action_sampler, "POMCPOW"),
            self.config_api.sparse_pft_config(self.mock_env, "SparsePFT"),
            self.config_api.sparse_sampling_config(self.mock_env, "SparseSampling"),
            self.config_api.pomcp_config(self.mock_env, "POMCP"),
            self.config_api.pomcp_dpw_config(self.mock_env, self.mock_action_sampler, "POMCP_DPW"),
            self.config_api.discrete_action_sequences_config(self.mock_env, "DiscreteSequences")
        ]

        for config in configs:
            param_names = [param.name for param in config.hyper_parameters]
            assert len(param_names) == len(set(param_names)), f"Duplicate parameter names in {config.policy_cls.__name__}"

    def test_constant_parameters_include_required_fields(self):
        """Test that constant parameters include all required fields.

        Purpose: Validates that constant parameters contain necessary fields for policy instantiation

        Given: Any planner configuration
        When: Constant parameters are examined
        Then: Required fields like environment and name are present

        Test type: unit
        """
        config = self.config_api.pomcp_config(self.mock_env, self.planner_name)

        required_fields = ["environment", "name"]
        for field in required_fields:
            assert field in config.constant_parameters
            assert config.constant_parameters[field] is not None

    def test_edge_case_zero_reward_range(self):
        """Test configuration with zero reward range.

        Purpose: Validates that configurations handle edge case of zero reward range

        Given: Environment with zero reward range
        When: Configuration method is called
        Then: Configuration is created successfully with appropriate exploration constant

        Test type: unit
        """
        zero_range_env = Mock()
        zero_range_env.reward_range = (0.0, 0.0)  # Zero range

        config = self.config_api.pomcp_config(zero_range_env, "ZeroRange")
        exploration_param = next(p for p in config.hyper_parameters if p.name == "exploration_constant")
        
        # Should handle zero range gracefully (exploration constant max should be 0)
        assert exploration_param.high == 0.0
        assert exploration_param.low == 0.0

    def test_negative_reward_range(self):
        """Test configuration with negative reward range.

        Purpose: Validates that configurations handle negative reward ranges correctly

        Given: Environment with negative reward range
        When: Configuration method is called
        Then: Configuration uses absolute difference for exploration constant scaling

        Test type: unit
        """
        negative_env = Mock()
        negative_env.reward_range = (-50.0, -10.0)  # Range of 40

        config = self.config_api.pomcp_config(negative_env, "Negative")
        exploration_param = next(p for p in config.hyper_parameters if p.name == "exploration_constant")
        
        # Should use absolute difference: (-10.0 - (-50.0)) * 15 = 40 * 15 = 600
        expected_max = 40.0 * 15
        assert exploration_param.high == expected_max


def test_planners_hyperparam_configs_usage_example():
    """Test the usage example for PlannersHyperparamConfigs.

    Purpose: Validates that the class can be used as intended in practice

    Given: A typical usage scenario with mock environment
    When: Configuration is created and used
    Then: All operations complete successfully without errors

    Test type: example
    """
    # Example usage
    discount_factor = 0.95
    config_api = PlannersHyperparamConfigs(discount_factor=discount_factor)

    # Create mock environment
    mock_env = Mock()
    mock_env.reward_range = (-10.0, 100.0)
    
    # Create mock action sampler
    mock_action_sampler = Mock(spec=ActionSampler)

    # Get POMCP configuration
    pomcp_config = config_api.pomcp_config(mock_env, "POMCP_Example")

    # Verify the configuration
    assert isinstance(pomcp_config, HyperParamPlannerConfig)
    assert pomcp_config.policy_cls == POMCP
    assert len(pomcp_config.hyper_parameters) > 0
    assert pomcp_config.constant_parameters["discount_factor"] == discount_factor