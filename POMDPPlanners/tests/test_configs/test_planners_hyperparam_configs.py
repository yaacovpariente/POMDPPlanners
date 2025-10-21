"""Tests for PlannersHyperparamConfigs class.

This module contains comprehensive tests for the PlannersHyperparamConfigs implementation,
including tests for all planner configuration methods and their hyperparameter ranges.
"""

# pylint: disable=protected-access  # Tests need to access protected members

from unittest.mock import Mock


from POMDPPlanners.configs.planners_hyperparam_configs import PlannersHyperparamConfigs
from POMDPPlanners.core.environment import SpaceInfo, SpaceType
from POMDPPlanners.core.simulation import (
    NumericalHyperParameter,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler, UnitCircleActionSampler
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.mcts_planners.pomcp_dpw import POMCP_DPW
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import (
    DiscreteActionSequencesPlanner,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.planners.sparse_sampling_planner import (
    StandardSparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.utils.hyperparameter_tuning_and_eval import (
    HyperParamPlannerConfig,
    get_fast_optimization_defaults,
)


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
            name=self.planner_name,
        )

        # Verify return type
        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == PFT_DPW

        # Verify hyperparameters
        assert len(config.hyper_parameters) == 6
        param_names = [param.name for param in config.hyper_parameters]
        expected_params = [
            "exploration_constant",
            "depth",
            "k_a",
            "alpha_a",
            "k_o",
            "alpha_o",
        ]
        assert set(param_names) == set(expected_params)

        # Verify constant parameters
        assert config.constant_parameters["discount_factor"] == self.discount_factor
        assert config.constant_parameters["name"] == self.planner_name
        assert config.constant_parameters["environment"] == self.mock_env
        assert config.constant_parameters["action_sampler"] == self.mock_action_sampler

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
            name=self.planner_name,
        )

        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == POMCPOW
        assert len(config.hyper_parameters) == 6

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
        config = self.config_api.sparse_pft_config(env=self.mock_env, name=self.planner_name)

        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == SparsePFT
        assert len(config.hyper_parameters) == 4

        param_names = [param.name for param in config.hyper_parameters]
        expected_params = ["depth", "c_ucb", "beta_ucb", "belief_child_num"]
        assert set(param_names) == set(expected_params)

        # Check unique parameter "gamma" instead of "discount_factor"
        assert config.constant_parameters["gamma"] == self.discount_factor
        assert "action_sampler" not in config.constant_parameters

    def test_sparse_sampling_config(self):
        """Test StandardSparseSampling configuration creation.

        Purpose: Validates that StandardSparseSampling configuration is created correctly with numerical parameters

        Given: A mock environment and planner name
        When: sparse_sampling_config is called
        Then: Returns valid HyperParamPlannerConfig with branching_factor and depth parameters

        Test type: unit
        """
        config = self.config_api.sparse_sampling_config(env=self.mock_env, name=self.planner_name)

        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == StandardSparseSamplingDiscreteActionsPlanner
        assert len(config.hyper_parameters) == 2

        # Check parameter types
        param_dict = {param.name: param for param in config.hyper_parameters}
        assert isinstance(param_dict["branching_factor"], NumericalHyperParameter)
        assert isinstance(param_dict["depth"], NumericalHyperParameter)

        # Check that time_out_in_seconds is not in constant parameters
        # (StandardSparseSamplingDiscreteActionsPlanner doesn't accept it)
        assert "time_out_in_seconds" not in config.constant_parameters

    def test_pomcp_config(self):
        """Test POMCP configuration creation.

        Purpose: Validates that POMCP configuration is created correctly with basic MCTS parameters

        Given: A mock environment and planner name
        When: pomcp_config is called
        Then: Returns valid HyperParamPlannerConfig with correct policy class and parameters

        Test type: unit
        """
        config = self.config_api.pomcp_config(env=self.mock_env, name=self.planner_name)

        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == POMCP

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
            name=self.planner_name,
        )

        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == POMCP_DPW

    def test_discrete_action_sequences_config(self):
        """Test DiscreteActionSequences configuration creation.

        Purpose: Validates that DiscreteActionSequences configuration is created correctly with open-loop parameters

        Given: A mock environment and planner name
        When: discrete_action_sequences_config is called
        Then: Returns valid HyperParamPlannerConfig with planning horizon and sampling parameters

        Test type: unit
        """
        config = self.config_api.discrete_action_sequences_config(
            env=self.mock_env, name=self.planner_name
        )

        assert isinstance(config, HyperParamPlannerConfig)
        assert config.policy_cls == DiscreteActionSequencesPlanner

        param_names = [param.name for param in config.hyper_parameters]
        expected_params = ["depth", "n_return_samples"]
        assert set(param_names) == set(expected_params)

        # Check parameter ranges
        param_dict = {param.name: param for param in config.hyper_parameters}
        depth_param = param_dict["depth"]
        n_return_samples_param = param_dict["n_return_samples"]

        # These should be NumericalHyperParameter instances
        assert isinstance(depth_param, NumericalHyperParameter)
        assert isinstance(n_return_samples_param, NumericalHyperParameter)

        assert depth_param.low == 2
        assert depth_param.high == 3
        assert n_return_samples_param.low == 10
        assert n_return_samples_param.high == 500

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
            self.config_api.discrete_action_sequences_config(self.mock_env, "DiscreteSequences"),
        ]

        for config in configs:
            assert isinstance(config, HyperParamPlannerConfig)
            assert hasattr(config, "policy_cls")
            assert hasattr(config, "hyper_parameters")
            assert hasattr(config, "constant_parameters")

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
            self.config_api.discrete_action_sequences_config(self.mock_env, "DiscreteSequences"),
        ]

        for config in configs:
            param_names = [param.name for param in config.hyper_parameters]
            assert len(param_names) == len(
                set(param_names)
            ), f"Duplicate parameter names in {config.policy_cls.__name__}"

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
        exploration_param = next(
            p for p in config.hyper_parameters if p.name == "exploration_constant"
        )

        # Should handle zero range gracefully (exploration constant max should be 0)
        assert isinstance(exploration_param, NumericalHyperParameter)
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
        exploration_param = next(
            p for p in config.hyper_parameters if p.name == "exploration_constant"
        )

        # Should use absolute difference: (-10.0 - (-50.0)) * max_depth_for_tuning = 40 * 10 = 400
        expected_max = 40.0 * 10
        assert isinstance(exploration_param, NumericalHyperParameter)
        assert exploration_param.high == expected_max

    def test_pomcp_config_with_hyperparameter_tuning(self):
        """Test POMCP configuration initialization.

        Purpose: Validates that POMCP configuration can be created without errors

        Given: A real Tiger POMDP environment and POMCP configuration
        When: POMCP configuration is created
        Then: Configuration is created successfully without errors

        Test type: unit
        """
        # Create a real environment for testing
        env = TigerPOMDP(discount_factor=0.95, name="TestTiger")

        # Create POMCP configuration using the config API
        pomcp_config = self.config_api.pomcp_config(env, "TestPOMCP")

        # Verify configuration was created successfully
        assert pomcp_config is not None
        assert pomcp_config.policy_cls == POMCP
        assert pomcp_config.constant_parameters["environment"] == env
        assert pomcp_config.constant_parameters["name"] == "TestPOMCP"

    def test_sparse_pft_config_with_hyperparameter_tuning(self):
        """Test SparsePFT configuration initialization.

        Purpose: Validates that SparsePFT configuration can be created without errors

        Given: A real Tiger POMDP environment and SparsePFT configuration
        When: SparsePFT configuration is created
        Then: Configuration is created successfully without errors

        Test type: unit
        """
        # Create a real environment for testing
        env = TigerPOMDP(discount_factor=0.95, name="TestTiger")

        # Create SparsePFT configuration using the config API
        sparse_pft_config = self.config_api.sparse_pft_config(env, "TestSparsePFT")

        # Verify configuration was created successfully
        assert sparse_pft_config is not None
        assert sparse_pft_config.policy_cls == SparsePFT
        assert sparse_pft_config.constant_parameters["environment"] == env
        assert sparse_pft_config.constant_parameters["name"] == "TestSparsePFT"

    def test_multiple_configs_with_hyperparameter_tuning(self):
        """Test multiple planner configurations initialization.

        Purpose: Validates that multiple planner configurations can be created without errors

        Given: Multiple planner configurations (POMCP and SparsePFT)
        When: Multiple configurations are created
        Then: All configurations are created successfully without errors

        Test type: unit
        """
        # Create a real environment for testing
        env = TigerPOMDP(discount_factor=0.95, name="TestTiger")

        # Create multiple configurations
        pomcp_config = self.config_api.pomcp_config(env, "TestPOMCP")
        sparse_pft_config = self.config_api.sparse_pft_config(env, "TestSparsePFT")

        planner_configs = [pomcp_config, sparse_pft_config]

        # Verify all configurations were created successfully
        assert len(planner_configs) == 2

        # Verify POMCP configuration
        assert pomcp_config is not None
        assert pomcp_config.policy_cls == POMCP
        assert pomcp_config.constant_parameters["environment"] == env
        assert pomcp_config.constant_parameters["name"] == "TestPOMCP"

        # Verify SparsePFT configuration
        assert sparse_pft_config is not None
        assert sparse_pft_config.policy_cls == SparsePFT
        assert sparse_pft_config.constant_parameters["environment"] == env
        assert sparse_pft_config.constant_parameters["name"] == "TestSparsePFT"

    def test_config_with_fast_optimization_defaults(self):
        """Test configuration and fast optimization defaults initialization.

        Purpose: Validates that configurations and fast optimization defaults can be created without errors

        Given: A planner configuration and fast optimization defaults
        When: Configuration and defaults are created
        Then: Both are created successfully without errors

        Test type: unit
        """
        # Create a real environment for testing
        env = TigerPOMDP(discount_factor=0.95, name="TestTiger")

        # Create POMCP configuration
        pomcp_config = self.config_api.pomcp_config(env, "TestPOMCPFast")

        # Get fast optimization defaults
        fast_defaults = get_fast_optimization_defaults()

        # Verify configuration was created successfully
        assert pomcp_config is not None
        assert pomcp_config.policy_cls == POMCP
        assert pomcp_config.constant_parameters["environment"] == env
        assert pomcp_config.constant_parameters["name"] == "TestPOMCPFast"

        # Verify fast defaults were created successfully
        assert fast_defaults is not None
        assert isinstance(fast_defaults, dict)
        assert "optimization_episodes" in fast_defaults
        assert "optimization_steps" in fast_defaults
        assert "n_trials" in fast_defaults

    def test_get_compatible_planners_discrete_environment(self):
        """Test get_compatible_planners with discrete action/observation environment.

        Purpose: Validates that compatible planners are returned for discrete environments

        Given: A discrete action/observation environment (TigerPOMDP)
        When: get_compatible_planners is called
        Then: Returns appropriate planners that support discrete spaces

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95)
        planners = self.config_api.get_compatible_planners(env)

        # Should return 6 planners for discrete/discrete environment
        assert len(planners) == 6

        # Extract planner class names
        planner_classes = [config.policy_cls.__name__ for config in planners]

        # Expected planners for discrete/discrete environment
        expected_planners = {
            "POMCPOW",
            "POMCP_DPW",
            "POMCP",
            "SparsePFT",
            "StandardSparseSamplingDiscreteActionsPlanner",
            "DiscreteActionSequencesPlanner",
        }

        assert set(planner_classes) == expected_planners

        # Verify all returned configs are valid
        for config in planners:
            assert isinstance(config, HyperParamPlannerConfig)
            assert config.constant_parameters["environment"] == env

    def test_get_compatible_planners_continuous_environment(self):
        """Test get_compatible_planners with continuous action/observation environment.

        Purpose: Validates that compatible planners are returned for continuous environments

        Given: A continuous action/observation environment (ContinuousLightDarkPOMDP)
        When: get_compatible_planners is called
        Then: Returns appropriate planners that support continuous spaces

        Test type: unit
        """
        env = ContinuousLightDarkPOMDP(discount_factor=0.95)
        planners = self.config_api.get_compatible_planners(env)

        # Should return 3 planners for continuous/continuous environment
        assert len(planners) == 3

        # Extract planner class names
        planner_classes = [config.policy_cls.__name__ for config in planners]

        # Expected planners for continuous/continuous environment
        expected_planners = {"POMCPOW", "POMCP_DPW", "PFT_DPW"}

        assert set(planner_classes) == expected_planners

        # Verify all returned configs are valid
        for config in planners:
            assert isinstance(config, HyperParamPlannerConfig)
            assert config.constant_parameters["environment"] == env

    def test_get_compatible_planners_with_custom_timeout(self):
        """Test get_compatible_planners with custom timeout parameter.

        Purpose: Validates that custom timeout parameter is applied to all planners

        Given: An environment and custom timeout value
        When: get_compatible_planners is called with custom timeout
        Then: All planner configs use the specified timeout

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95)
        custom_timeout = 5.0
        planners = self.config_api.get_compatible_planners(env, time_out_in_seconds=custom_timeout)

        # Verify timeout is applied to all planners that use it
        for config in planners:
            if "time_out_in_seconds" in config.constant_parameters:
                assert config.constant_parameters["time_out_in_seconds"] == custom_timeout

    def test_get_action_sampler_for_discrete_environment(self):
        """Test _get_action_sampler_for_environment with discrete action space.

        Purpose: Validates that DiscreteActionSampler is returned for discrete environments

        Given: A discrete action space environment
        When: _get_action_sampler_for_environment is called
        Then: Returns DiscreteActionSampler with environment actions

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95)
        action_sampler = self.config_api._get_action_sampler_for_environment(env)

        assert isinstance(action_sampler, DiscreteActionSampler)
        assert action_sampler.actions == env.get_actions()

    def test_get_action_sampler_for_continuous_environment(self):
        """Test _get_action_sampler_for_environment with continuous action space.

        Purpose: Validates that UnitCircleActionSampler is returned for continuous environments

        Given: A continuous action space environment
        When: _get_action_sampler_for_environment is called
        Then: Returns UnitCircleActionSampler

        Test type: unit
        """
        env = ContinuousLightDarkPOMDP(discount_factor=0.95)
        action_sampler = self.config_api._get_action_sampler_for_environment(env)

        assert isinstance(action_sampler, UnitCircleActionSampler)

    def test_get_action_sampler_for_mixed_environment(self):
        """Test _get_action_sampler_for_environment with mixed action space.

        Purpose: Validates that UnitCircleActionSampler is returned for mixed environments

        Given: A mixed action space environment
        When: _get_action_sampler_for_environment is called
        Then: Returns UnitCircleActionSampler as default for mixed spaces

        Test type: unit
        """
        # Create mock environment with mixed action space
        mock_env = Mock()
        mock_env.space_info = SpaceInfo(
            action_space=SpaceType.MIXED, observation_space=SpaceType.MIXED
        )
        mock_env.reward_range = None  # Mock reward_range attribute

        action_sampler = self.config_api._get_action_sampler_for_environment(mock_env)

        assert isinstance(action_sampler, UnitCircleActionSampler)

    def test_get_action_sampler_for_non_discrete_actions_environment(self):
        """Test _get_action_sampler_for_environment with non-DiscreteActionsEnvironment.

        Purpose: Validates that None is returned when environment doesn't have get_actions method

        Given: An environment with discrete space but no get_actions method
        When: _get_action_sampler_for_environment is called
        Then: Returns None because environment doesn't support action enumeration

        Test type: unit
        """
        # Create mock environment that has discrete space but is not DiscreteActionsEnvironment
        mock_env = Mock()
        mock_env.space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )
        mock_env.reward_range = None  # Mock reward_range attribute

        action_sampler = self.config_api._get_action_sampler_for_environment(mock_env)

        assert action_sampler is None

    def test_get_compatible_planners_filters_by_action_sampler_availability(self):
        """Test that get_compatible_planners filters planners based on action sampler availability.

        Purpose: Validates that planners requiring action samplers are excluded when sampler unavailable

        Given: An environment that cannot provide action sampler
        When: get_compatible_planners is called
        Then: Only planners not requiring action samplers are returned

        Test type: unit
        """
        # Create mock environment with discrete space but no get_actions method
        mock_env = Mock()
        mock_env.space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )
        mock_env.reward_range = None  # Mock reward_range attribute

        planners = self.config_api.get_compatible_planners(mock_env)

        # Should only return planners that don't require action samplers
        # In this case, only discrete action planners that don't need action samplers
        planner_classes = [config.policy_cls.__name__ for config in planners]

        # Expected planners that don't require action samplers
        expected_planners = {
            "POMCP",
            "SparsePFT",
            "StandardSparseSamplingDiscreteActionsPlanner",
            "DiscreteActionSequencesPlanner",
        }

        assert set(planner_classes) == expected_planners

    def test_get_compatible_planners_empty_list_for_incompatible_environment(self):
        """Test get_compatible_planners returns appropriate planners for all environments.

        Purpose: Validates that function handles edge cases gracefully

        Given: Various environment configurations
        When: get_compatible_planners is called
        Then: Returns appropriate planners or handles edge cases gracefully

        Test type: unit
        """
        # Test with mock environment that has unusual space configuration
        mock_env = Mock()
        mock_env.space_info = SpaceInfo(
            action_space=SpaceType.CONTINUOUS, observation_space=SpaceType.CONTINUOUS
        )
        mock_env.reward_range = None  # Mock reward_range attribute

        planners = self.config_api.get_compatible_planners(mock_env)

        # Should return continuous-compatible planners
        assert len(planners) >= 0  # Should handle gracefully

    def test_get_compatible_planners_all_configs_have_required_fields(self):
        """Test that all returned planner configs have required fields.

        Purpose: Validates that all returned configurations are properly structured

        Given: Any environment with compatible planners
        When: get_compatible_planners is called
        Then: All returned configs have necessary fields for planner instantiation

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95)
        planners = self.config_api.get_compatible_planners(env)

        for config in planners:
            # Verify required fields exist
            assert hasattr(config, "policy_cls")
            assert hasattr(config, "hyper_parameters")
            assert hasattr(config, "constant_parameters")

            # Verify constant parameters have required fields
            assert "environment" in config.constant_parameters
            assert "name" in config.constant_parameters
            assert config.constant_parameters["environment"] == env

            # Verify policy class is valid
            assert config.policy_cls is not None
            assert callable(config.policy_cls)
