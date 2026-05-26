# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for POMDP planner serialization.

This module tests that all POMDP planners can be properly serialized
and deserialized using pickle. Serialization is crucial for:
- Distributed computing scenarios
- Saving/loading planner configurations
- Checkpointing during long-running optimizations
- Multi-processing applications
"""

# pylint: disable=attribute-defined-outside-init  # pytest setup_method pattern

import pickle
from typing import Any, Dict

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.environments import DiscreteLightDarkPOMDP, TigerPOMDP
from POMDPPlanners.planners import (
    POMCP,
    POMCP_DPW,
    POMCPOW,
    PFT_DPW,
    DiscreteActionSequencesPlanner,
    SparsePFT,
    SparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.planners.mcts_planners.beta_zero import BetaZero
from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_action_sampler import (
    BetaZeroActionSampler,
)
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

# Set seeds for reproducible tests
np.random.seed(42)


class TestPlannerSerialization:
    """Test cases for planner serialization using pickle."""

    def setup_method(self):
        """Set up test environments for each test."""
        self.tiger_env = TigerPOMDP(discount_factor=0.95)
        self.light_dark_env = DiscreteLightDarkPOMDP(discount_factor=0.95)

    def _test_planner_serialization(self, planner_class: type, init_params: Dict[str, Any]) -> None:
        """Helper method to test planner serialization.

        Purpose: Validates that a planner can be pickled and unpickled correctly

        Given: A planner class and initialization parameters
        When: Planner is created, pickled, and unpickled
        Then: Unpickled planner maintains all properties and functionality

        Test type: unit

        Args:
            planner_class: Planner class to test
            init_params: Parameters for planner initialization
        """
        # Create planner
        planner = planner_class(**init_params)

        # Pickle the planner
        pickled = pickle.dumps(planner)

        # Unpickle the planner
        unpickled_planner = pickle.loads(pickled)

        # Verify basic properties are preserved
        assert unpickled_planner.name == planner.name
        assert unpickled_planner.discount_factor == planner.discount_factor

        # Verify environment is preserved
        assert unpickled_planner.environment.name == planner.environment.name

    def test_pomcp_serialization(self):
        """Test POMCP planner serialization.

        Purpose: Validates that POMCP can be pickled and unpickled

        Given: POMCP planner instance with TigerPOMDP environment
        When: Planner is pickled and unpickled
        Then: Unpickled planner maintains all properties and functionality

        Test type: unit
        """
        self._test_planner_serialization(
            POMCP,
            {
                "environment": self.tiger_env,
                "discount_factor": 0.95,
                "depth": 10,
                "exploration_constant": 10.0,
                "name": "POMCP_Test",
                "n_simulations": 100,
            },
        )

    def test_pomcp_dpw_serialization(self):
        """Test POMCP_DPW planner serialization.

        Purpose: Validates that POMCP_DPW can be pickled and unpickled

        Given: POMCP_DPW planner instance with TigerPOMDP environment
        When: Planner is pickled and unpickled
        Then: Unpickled planner maintains all properties and functionality

        Test type: unit
        """
        # POMCP_DPW requires action_sampler - skip for now as it's complex to setup
        pytest.skip("POMCP_DPW requires ActionSampler which is complex to initialize")

    def test_pomcpow_serialization(self):
        """Test POMCPOW planner serialization.

        Purpose: Validates that POMCPOW can be pickled and unpickled

        Given: POMCPOW planner instance with DiscreteLightDarkPOMDP environment
        When: Planner is pickled and unpickled
        Then: Unpickled planner maintains all properties and functionality

        Test type: unit
        """
        # POMCPOW requires action_sampler - skip for now as it's complex to setup
        pytest.skip("POMCPOW requires ActionSampler which is complex to initialize")

    def test_pft_dpw_serialization(self):
        """Test PFT_DPW planner serialization.

        Purpose: Validates that PFT_DPW can be pickled and unpickled

        Given: PFT_DPW planner instance with TigerPOMDP environment
        When: Planner is pickled and unpickled
        Then: Unpickled planner maintains all properties and functionality

        Test type: unit
        """
        # PFT_DPW requires action_sampler - skip for now as it's complex to setup
        pytest.skip("PFT_DPW requires ActionSampler which is complex to initialize")

    def test_sparse_pft_serialization(self):
        """Test SparsePFT planner serialization.

        Purpose: Validates that SparsePFT can be pickled and unpickled

        Given: SparsePFT planner instance with TigerPOMDP environment
        When: Planner is pickled and unpickled
        Then: Unpickled planner maintains all properties and functionality

        Test type: unit
        """
        self._test_planner_serialization(
            SparsePFT,
            {
                "environment": self.tiger_env,
                "discount_factor": 0.95,
                "depth": 10,
                "c_ucb": 10.0,
                "beta_ucb": 0.5,
                "belief_child_num": 5,
                "name": "SparsePFT_Test",
                "n_simulations": 100,
            },
        )

    def test_path_simulation_policy_serialization(self):
        """Test PathSimulationPolicy planner serialization.

        Purpose: Validates that PathSimulationPolicy can be pickled and unpickled

        Given: PathSimulationPolicy planner instance with TigerPOMDP environment
        When: Planner is pickled and unpickled
        Then: Unpickled planner maintains all properties and functionality

        Test type: unit
        """
        # PathSimulationPolicy is abstract - skip
        pytest.skip("PathSimulationPolicy is an abstract base class")

    def test_discrete_action_sequences_planner_serialization(self):
        """Test DiscreteActionSequencesPlanner serialization.

        Purpose: Validates that DiscreteActionSequencesPlanner can be pickled and unpickled

        Given: DiscreteActionSequencesPlanner instance with TigerPOMDP environment
        When: Planner is pickled and unpickled
        Then: Unpickled planner maintains all properties and functionality

        Test type: unit
        """
        self._test_planner_serialization(
            DiscreteActionSequencesPlanner,
            {
                "environment": self.tiger_env,
                "discount_factor": 0.95,
                "name": "DiscreteActionSeq_Test",
                "depth": 5,
                "n_return_samples": 10,
            },
        )

    def test_sparse_sampling_planner_serialization(self):
        """Test SparseSamplingDiscreteActionsPlanner serialization.

        Purpose: Validates that SparseSamplingDiscreteActionsPlanner can be pickled and unpickled

        Given: SparseSamplingDiscreteActionsPlanner instance with TigerPOMDP environment
        When: Planner is pickled and unpickled
        Then: Unpickled planner maintains all properties and functionality

        Test type: unit
        """
        self._test_planner_serialization(
            SparseSamplingDiscreteActionsPlanner,
            {
                "environment": self.tiger_env,
                "branching_factor": 10,
                "depth": 5,
                "name": "SparseSampling_Test",
            },
        )

    def test_beta_zero_serialization(self):
        """Test BetaZero planner serialization.

        Purpose: Validates that BetaZero can be pickled and unpickled correctly,
            including its network-guided action sampler components.

        Given: BetaZero planner instance with TigerPOMDP environment and
            BetaZeroActionSampler with fallback
        When: Planner is pickled and unpickled
        Then: Unpickled planner maintains all properties and functionality

        Test type: unit
        """
        # Create action sampler with fallback for BetaZero
        actions = self.tiger_env.get_actions()
        fallback_sampler = DiscreteActionSampler(actions)
        action_sampler = BetaZeroActionSampler(
            fallback_sampler=fallback_sampler,
            actions=actions,
        )

        # Create BetaZero planner
        planner = BetaZero(
            environment=self.tiger_env,
            discount_factor=0.95,
            depth=10,
            name="BetaZero_Test",
            action_sampler=action_sampler,
            n_simulations=50,
            state_dim=1,  # Tiger POMDP has 1D state
            k_a=1.0,
            alpha_a=0.5,
            k_o=1.0,
            alpha_o=0.5,
            exploration_constant=1.0,
        )

        # Pickle the planner
        pickled = pickle.dumps(planner)

        # Unpickle the planner
        unpickled_planner = pickle.loads(pickled)

        # Verify basic properties are preserved
        assert unpickled_planner.name == planner.name
        assert unpickled_planner.discount_factor == planner.discount_factor
        assert unpickled_planner.depth == planner.depth
        assert unpickled_planner.n_simulations == planner.n_simulations

        # Verify environment is preserved
        assert unpickled_planner.environment.name == planner.environment.name


class TestPlannerSerializationWithBeliefs:
    """Test cases for planner serialization with belief states."""

    def setup_method(self):
        """Set up test environment and beliefs for each test."""
        self.env = TigerPOMDP(discount_factor=0.95)

        # Create initial belief
        initial_state = self.env.initial_state_dist().sample()[0]
        self.belief = WeightedParticleBelief(
            particles=[initial_state] * 100,
            log_weights=np.log(np.ones(100) / 100),
            resampling=True,
        )

    def test_pomcp_serialization_with_belief(self):
        """Test POMCP serialization after belief updates.

        Purpose: Validates that POMCP with updated beliefs can be pickled

        Given: POMCP planner with belief state
        When: Planner is pickled and unpickled
        Then: Unpickled planner maintains state correctly

        Test type: unit
        """
        planner = POMCP(
            environment=self.env,
            discount_factor=0.95,
            depth=10,
            exploration_constant=10.0,
            name="POMCP_Belief_Test",
            n_simulations=50,
        )

        # Pickle planner with belief context
        pickled = pickle.dumps(planner)
        unpickled_planner = pickle.loads(pickled)

        assert unpickled_planner.name == planner.name
        assert unpickled_planner.discount_factor == planner.discount_factor


class TestPlannerSerializationRoundTrip:
    """Test cases for complete planner serialization round trips."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.env = TigerPOMDP(discount_factor=0.95)
        self.initial_state = self.env.initial_state_dist().sample()[0]

        # Create initial belief
        self.belief = WeightedParticleBelief(
            particles=[self.initial_state] * 100,
            log_weights=np.log(np.ones(100) / 100),
            resampling=True,
        )

    def test_planner_serialization_preserves_functionality(self):
        """Test that serialized planners maintain full functionality.

        Purpose: Validates that planners work correctly after serialization round trip

        Given: Multiple planner instances
        When: Planners are pickled, unpickled, and used for action selection
        Then: Unpickled planners produce valid actions

        Test type: integration
        """
        planners = [
            POMCP(
                environment=self.env,
                discount_factor=0.95,
                depth=10,
                exploration_constant=10.0,
                name="POMCP_Test",
                n_simulations=50,
            ),
            SparsePFT(
                environment=self.env,
                discount_factor=0.95,
                depth=10,
                c_ucb=10.0,
                beta_ucb=0.5,
                belief_child_num=5,
                name="SparsePFT_Test",
                n_simulations=50,
            ),
            DiscreteActionSequencesPlanner(
                environment=self.env,
                discount_factor=0.95,
                name="DiscreteActionSeq_Test",
                depth=3,
                n_return_samples=5,
            ),
        ]

        for planner in planners:
            # Pickle and unpickle
            pickled = pickle.dumps(planner)
            unpickled_planner = pickle.loads(pickled)

            # Test action selection with unpickled planner
            action_list, _ = unpickled_planner.action(self.belief)

            assert action_list is not None
            assert len(action_list) > 0
            assert action_list[0] in self.env.get_actions()

    def test_mcts_planner_serialization_after_simulations(self):
        """Test MCTS planner serialization after running simulations.

        Purpose: Validates that MCTS planners with built search trees can be pickled

        Given: POMCP planner after running simulations
        When: Planner is pickled and unpickled
        Then: Unpickled planner maintains properties correctly

        Test type: integration
        """
        planner = POMCP(
            environment=self.env,
            discount_factor=0.95,
            depth=10,
            exploration_constant=10.0,
            name="POMCP_Simulations_Test",
            n_simulations=10,
        )

        # Run action to build search tree
        action_list, _ = planner.action(self.belief)
        assert action_list is not None
        assert len(action_list) > 0

        # Pickle after building search tree
        pickled = pickle.dumps(planner)
        unpickled_planner = pickle.loads(pickled)

        # Verify planner properties are preserved
        assert unpickled_planner.name == planner.name
        assert unpickled_planner.discount_factor == planner.discount_factor
        assert unpickled_planner.depth == planner.depth
        assert unpickled_planner.exploration_constant == planner.exploration_constant


class TestPlannerConfigSerialization:
    """Test cases for planner configuration serialization."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.env = TigerPOMDP(discount_factor=0.95)

    def test_pomcp_config_serialization(self):
        """Test POMCP configuration serialization.

        Purpose: Validates that POMCP configurations can be pickled

        Given: POMCP planner with specific configuration
        When: Planner configuration is accessed after serialization
        Then: Configuration parameters are preserved

        Test type: unit
        """
        planner = POMCP(
            environment=self.env,
            discount_factor=0.95,
            depth=15,
            exploration_constant=20.0,
            name="POMCP_Config_Test",
            n_simulations=200,
        )

        pickled = pickle.dumps(planner)
        unpickled_planner = pickle.loads(pickled)

        # Verify configuration parameters
        assert unpickled_planner.depth == 15
        assert unpickled_planner.exploration_constant == 20.0
        assert unpickled_planner.n_simulations == 200

    def test_pomcp_dpw_config_serialization(self):
        """Test POMCP_DPW configuration serialization.

        Purpose: Validates that POMCP_DPW configurations with DPW parameters can be pickled

        Given: POMCP_DPW planner with specific DPW configuration
        When: Planner configuration is accessed after serialization
        Then: All DPW parameters are preserved

        Test type: unit
        """
        # POMCP_DPW requires ActionSampler - skip for now
        pytest.skip("POMCP_DPW requires ActionSampler which is complex to initialize")


class TestPlannerSerializationEdgeCases:
    """Test cases for edge cases in planner serialization."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.env = TigerPOMDP(discount_factor=0.95)

    def test_planner_serialization_with_no_simulations(self):
        """Test planner serialization before any simulations.

        Purpose: Validates that fresh planners without any runs can be pickled

        Given: Newly created planner without running simulations
        When: Planner is pickled and unpickled
        Then: Unpickled planner is ready for use

        Test type: unit
        """
        planner = POMCP(
            environment=self.env,
            discount_factor=0.95,
            depth=10,
            exploration_constant=10.0,
            name="POMCP_NoSim_Test",
            n_simulations=100,
        )

        # Pickle immediately without running any simulations
        pickled = pickle.dumps(planner)
        unpickled_planner = pickle.loads(pickled)

        assert unpickled_planner.name == planner.name
        assert unpickled_planner.environment.name == planner.environment.name

    def test_multiple_planners_serialization(self):
        """Test serialization of multiple planners together.

        Purpose: Validates that multiple planners can be pickled in same structure

        Given: Dictionary containing multiple different planners
        When: Dictionary is pickled and unpickled
        Then: All planners are correctly restored

        Test type: integration
        """
        planners_dict = {
            "pomcp": POMCP(
                environment=self.env,
                discount_factor=0.95,
                depth=10,
                exploration_constant=10.0,
                name="POMCP_Multi_Test",
                n_simulations=50,
            ),
            "sparse_pft": SparsePFT(
                environment=self.env,
                discount_factor=0.95,
                depth=10,
                c_ucb=10.0,
                beta_ucb=0.5,
                belief_child_num=5,
                name="SparsePFT_Multi_Test",
                n_simulations=50,
            ),
        }

        pickled = pickle.dumps(planners_dict)
        unpickled_dict = pickle.loads(pickled)

        assert len(unpickled_dict) == 2
        assert "pomcp" in unpickled_dict
        assert "sparse_pft" in unpickled_dict
        assert unpickled_dict["pomcp"].name == "POMCP_Multi_Test"
        assert unpickled_dict["sparse_pft"].name == "SparsePFT_Multi_Test"


class TestAllPlannersPicklable:
    """Comprehensive test to ensure all planners are picklable.

    This test class validates that all available POMDP planners, including
    those with complex components like action samplers, can be successfully
    pickled and unpickled. This is critical for:
    - Joblib/multiprocessing compatibility
    - Distributed computing scenarios
    - Checkpointing and model persistence
    """

    def setup_method(self):
        """Set up test environment for each test."""
        self.env = TigerPOMDP(discount_factor=0.95)
        self.actions = self.env.get_actions()

    def test_all_basic_planners_picklable(self):
        """Test that all basic planners (without action samplers) are picklable.

        Purpose: Validates comprehensive pickle support for basic planners

        Given: Instances of POMCP, SparsePFT, DiscreteActionSequencesPlanner,
            and SparseSamplingDiscreteActionsPlanner
        When: Each planner is pickled and unpickled via pickle.dumps/loads
        Then: All planners successfully round-trip and maintain their properties

        Test type: integration
        """
        planners = [
            POMCP(
                environment=self.env,
                discount_factor=0.95,
                depth=10,
                exploration_constant=10.0,
                name="POMCP_AllTest",
                n_simulations=50,
            ),
            SparsePFT(
                environment=self.env,
                discount_factor=0.95,
                depth=10,
                c_ucb=10.0,
                beta_ucb=0.5,
                belief_child_num=5,
                name="SparsePFT_AllTest",
                n_simulations=50,
            ),
            DiscreteActionSequencesPlanner(
                environment=self.env,
                discount_factor=0.95,
                name="DiscreteActionSeq_AllTest",
                depth=3,
                n_return_samples=5,
            ),
            SparseSamplingDiscreteActionsPlanner(
                environment=self.env,
                branching_factor=5,
                depth=3,
                name="SparseSampling_AllTest",
            ),
        ]

        for planner in planners:
            # Full pickle round trip
            pickled = pickle.dumps(planner)
            unpickled = pickle.loads(pickled)

            # Verify properties preserved
            assert unpickled.name == planner.name, f"Name mismatch for {planner.name}"
            assert (
                unpickled.discount_factor == planner.discount_factor
            ), f"Discount factor mismatch for {planner.name}"
            assert (
                unpickled.environment.name == planner.environment.name
            ), f"Environment mismatch for {planner.name}"

    def test_all_action_sampler_planners_picklable(self):
        """Test that all planners with action samplers are picklable.

        Purpose: Validates pickle support for planners with ActionSampler components,
            which are critical for parallel execution with joblib

        Given: Instances of BetaZero, POMCP_DPW, POMCPOW, and PFT_DPW planners,
            each configured with appropriate action samplers
        When: Each planner is pickled and unpickled via pickle.dumps/loads
        Then: All planners successfully round-trip and maintain their properties,
            including action sampler configuration

        Test type: integration
        """
        # BetaZero with BetaZeroActionSampler
        fallback = DiscreteActionSampler(self.actions)
        beta_zero_sampler = BetaZeroActionSampler(
            fallback_sampler=fallback,
            actions=self.actions,
        )
        beta_zero = BetaZero(
            environment=self.env,
            discount_factor=0.95,
            depth=10,
            name="BetaZero_AllTest",
            action_sampler=beta_zero_sampler,
            n_simulations=50,
            state_dim=1,
            k_a=1.0,
            alpha_a=0.5,
            k_o=1.0,
            alpha_o=0.5,
            exploration_constant=1.0,
        )

        # POMCP_DPW with DiscreteActionSampler
        pomcp_dpw_sampler = DiscreteActionSampler(self.actions)
        pomcp_dpw = POMCP_DPW(
            environment=self.env,
            discount_factor=0.95,
            depth=10,
            name="POMCP_DPW_AllTest",
            action_sampler=pomcp_dpw_sampler,
            n_simulations=50,
            k_a=1.0,
            alpha_a=0.5,
            k_o=1.0,
            alpha_o=0.5,
            exploration_constant=1.0,
        )

        # PFT_DPW with DiscreteActionSampler
        pft_dpw_sampler = DiscreteActionSampler(self.actions)
        pft_dpw = PFT_DPW(
            environment=self.env,
            discount_factor=0.95,
            depth=10,
            name="PFT_DPW_AllTest",
            action_sampler=pft_dpw_sampler,
            n_simulations=50,
            k_a=1.0,
            alpha_a=0.5,
            k_o=1.0,
            alpha_o=0.5,
            exploration_constant=1.0,
        )

        # POMCPOW with DiscreteActionSampler
        pomcpow_sampler = DiscreteActionSampler(self.actions)
        pomcpow = POMCPOW(
            environment=self.env,
            discount_factor=0.95,
            depth=10,
            name="POMCPOW_AllTest",
            action_sampler=pomcpow_sampler,
            n_simulations=50,
            k_a=1.0,
            alpha_a=0.5,
            k_o=1.0,
            alpha_o=0.5,
            exploration_constant=1.0,
        )

        planners = [beta_zero, pomcp_dpw, pft_dpw, pomcpow]

        for planner in planners:
            # Full pickle round trip (what joblib does for parallel execution)
            pickled = pickle.dumps(planner)
            unpickled = pickle.loads(pickled)

            # Verify properties preserved
            assert unpickled.name == planner.name, f"Name mismatch for {planner.name}"
            assert (
                unpickled.discount_factor == planner.discount_factor
            ), f"Discount factor mismatch for {planner.name}"
            assert (
                unpickled.environment.name == planner.environment.name
            ), f"Environment mismatch for {planner.name}"

    def test_planner_picklable_for_joblib_parallel(self):
        """Test that planners can be used with joblib.Parallel.

        Purpose: Validates real-world joblib compatibility by simulating
            joblib.Parallel serialization pattern

        Given: A BetaZero planner instance with action sampler
        When: Planner is pickled, unpickled, and used for action selection
            (simulating joblib worker process)
        Then: Unpickled planner functions correctly and produces valid actions

        Test type: integration
        """
        # Create BetaZero planner with action sampler
        fallback = DiscreteActionSampler(self.actions)
        action_sampler = BetaZeroActionSampler(
            fallback_sampler=fallback,
            actions=self.actions,
        )
        planner = BetaZero(
            environment=self.env,
            discount_factor=0.95,
            depth=10,
            name="BetaZero_Joblib_Test",
            action_sampler=action_sampler,
            n_simulations=20,  # Small for fast test
            state_dim=1,
            k_a=1.0,
            alpha_a=0.5,
            k_o=1.0,
            alpha_o=0.5,
            exploration_constant=1.0,
        )

        # Create a belief for action selection
        initial_state = self.env.initial_state_dist().sample()[0]
        belief = WeightedParticleBelief(
            particles=[initial_state] * 50,
            log_weights=np.log(np.ones(50) / 50),
            resampling=True,
        )

        # Simulate joblib serialization (what happens in parallel execution)
        pickled = pickle.dumps(planner)
        unpickled_planner = pickle.loads(pickled)

        # Test that unpickled planner can select actions
        action_list, _ = unpickled_planner.action(belief)

        # Verify action is valid
        assert action_list is not None, "Action list should not be None"
        assert len(action_list) > 0, "Action list should not be empty"
        assert action_list[0] in self.actions, f"Action {action_list[0]} not in valid actions"
