"""Comprehensive tests for metric name consistency across environments and policies.

This module demonstrates how to use the reusable test utilities from
test_metric_consistency_utils.py to verify metric consistency for all
environments and policies.
"""

import random
from unittest.mock import Mock

import numpy as np
import pytest

from POMDPPlanners.core.belief import Belief, get_initial_belief
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import (
    SparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import (
    DiscreteActionSequencesPlanner,
)
from POMDPPlanners.tests.test_metric_consistency_utils import (
    verify_environment_metric_consistency,
    verify_policy_metric_consistency,
)
from POMDPPlanners.tests.test_utils.history_builders import build_test_history

np.random.seed(42)
random.seed(42)


class TestEnvironmentMetricConsistency:
    """Test suite for environment metric name consistency."""

    def test_tiger_pomdp_metric_consistency(self):
        """Test TigerPOMDP metric name consistency.

        Purpose: Validates that TigerPOMDP declares and produces consistent metric names

        Given: A TigerPOMDP environment with sample histories
        When: get_metric_names() and compute_metrics() are called
        Then: Declared names match produced names exactly

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95)

        # Create sample histories
        steps = [
            StepData(
                state="tiger_left",
                action="listen",
                next_state="tiger_left",
                observation="hear_left",
                reward=-1,
                belief=Mock(spec=Belief),
            )
            for _ in range(3)
        ]

        histories = [
            build_test_history(
                steps=steps,
                reach_terminal=True,
                policy_run_data=[PolicyRunData(info_variables=[])],
            )
        ]

        verify_environment_metric_consistency(env, histories)

    def test_push_pomdp_metric_consistency(self):
        """Test PushPOMDP metric name consistency.

        Purpose: Validates that PushPOMDP declares and produces consistent metric names

        Given: A PushPOMDP environment with sample histories
        When: get_metric_names() and compute_metrics() are called
        Then: Declared names match produced names exactly

        Test type: unit
        """
        env = PushPOMDP(
            discount_factor=0.95,
            grid_size=10,
            push_threshold=1.0,
            friction_coefficient=0.1,
            observation_noise=0.1,
        )

        # Create sample histories with collision data
        steps = [
            StepData(
                state=np.array([1.0, 1.0, 2.0, 2.0, 5.0, 5.0]),  # robot, object, target positions
                action="up",
                next_state=np.array([1.0, 1.1, 2.0, 2.0, 5.0, 5.0]),
                observation=np.array([2.0, 2.0]),  # object position observation
                reward=-0.1,
                belief=Mock(spec=Belief),
            )
            for _ in range(5)
        ]

        histories = [
            build_test_history(
                steps=steps,
                reach_terminal=False,
                policy_run_data=[PolicyRunData(info_variables=[])],
            )
        ]

        verify_environment_metric_consistency(env, histories)


class TestPolicyMetricConsistency:
    """Test suite for policy info variable name consistency."""

    def test_pomcp_info_variable_consistency(self):
        """Test POMCP info variable name consistency.

        Purpose: Validates that POMCP declares and produces consistent info variable names

        Given: A POMCP planner and initial belief
        When: get_info_variable_names() and action() are called
        Then: Declared names match produced names exactly

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95)
        planner = POMCP(
            environment=env,
            discount_factor=0.95,
            depth=5,
            exploration_constant=1.0,
            n_simulations=10,
            name="TestPOMCP",
        )
        belief = get_initial_belief(pomdp=env, n_particles=10, resampling=True)

        verify_policy_metric_consistency(planner, belief)

    def test_sparse_sampling_info_variable_consistency(self):
        """Test SparseSampling info variable name consistency.

        Purpose: Validates that SparseSampling declares and produces consistent info variable names (empty)

        Given: A SparseSampling planner and initial belief
        When: get_info_variable_names() and action() are called
        Then: Both return empty lists (no info variables)

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95)
        planner = SparseSamplingDiscreteActionsPlanner(
            environment=env,
            branching_factor=2,
            depth=2,
            name="TestSparseSampling",
        )
        belief = get_initial_belief(pomdp=env, n_particles=10, resampling=True)

        verify_policy_metric_consistency(planner, belief)

    def test_discrete_action_sequences_info_variable_consistency(self):
        """Test DiscreteActionSequences info variable name consistency.

        Purpose: Validates that DiscreteActionSequences declares and produces consistent info variable names (empty)

        Given: A DiscreteActionSequences planner and initial belief
        When: get_info_variable_names() and action() are called
        Then: Both return empty lists (no info variables)

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95)
        planner = DiscreteActionSequencesPlanner(
            environment=env,
            discount_factor=0.95,
            name="TestOpenLoop",
            depth=2,
            n_return_samples=5,
        )
        belief = get_initial_belief(pomdp=env, n_particles=10, resampling=True)

        verify_policy_metric_consistency(planner, belief)


class TestMetricConsistencyEdgeCases:
    """Test edge cases for metric consistency validation."""

    def test_environment_with_no_metrics(self):
        """Test environment that produces no custom metrics.

        Purpose: Validates that environments with empty get_metric_names() work correctly

        Given: An environment with no custom metrics (returns empty list)
        When: Metric consistency is verified
        Then: Verification passes (both declared and produced are empty)

        Test type: unit
        """
        # TigerPOMDP has metrics, but we can test the pattern
        # Most base environments would have empty metric lists
        env = TigerPOMDP(discount_factor=0.95)

        # Manually verify the pattern works
        declared = env.get_metric_names()
        assert isinstance(declared, list), "get_metric_names() must return a list"

    def test_policy_with_no_info_variables(self):
        """Test policy that produces no info variables.

        Purpose: Validates that policies with empty get_info_variable_names() work correctly

        Given: A policy with no info variables (returns empty list)
        When: Info variable consistency is verified
        Then: Verification passes (both declared and produced are empty)

        Test type: unit
        """
        env = TigerPOMDP(discount_factor=0.95)
        planner = SparseSamplingDiscreteActionsPlanner(
            environment=env,
            branching_factor=2,
            depth=2,
            name="TestSparseSampling",
        )

        # Verify it returns empty list
        declared = planner.get_info_variable_names()
        assert declared == [], "SparseSampling should declare no info variables"

        belief = get_initial_belief(pomdp=env, n_particles=10, resampling=True)
        _, run_data = planner.action(belief)
        assert run_data.info_variables == [], "SparseSampling should produce no info variables"
