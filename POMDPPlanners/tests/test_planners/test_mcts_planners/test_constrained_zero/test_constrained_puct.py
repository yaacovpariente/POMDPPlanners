"""Tests for the SPUCT action selection module.

This module tests the safety-constrained PUCT selection and progressive
widening functions used by ConstrainedZero.
"""

import random

import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_puct import (
    _compute_safety_mask,
    spuct_action_progressive_widening,
    spuct_selection,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler

np.random.seed(42)
random.seed(42)


class SimpleActionSampler(ActionSampler):
    def __init__(self):
        self._counter = 0

    def sample(self, belief_node=None):  # pylint: disable=unused-argument
        self._counter += 1
        return f"new_action_{self._counter}"


def _make_belief_node_with_children(q_values, visit_counts, parent_visits):
    particles = [[0.0], [1.0]]
    log_weights = np.log([0.5, 0.5])
    belief = WeightedParticleBelief(particles, log_weights)
    node = BeliefNode(belief=belief)
    node.visit_count = parent_visits
    children = []
    for i, (q, n) in enumerate(zip(q_values, visit_counts)):
        child = ActionNode(action=f"a_{i}", parent=node)
        child.q_value = q
        child.visit_count = n
        children.append(child)
    return node, children


class TestComputeSafetyMask:
    """Tests for the _compute_safety_mask helper."""

    def test_all_safe(self):
        """Test mask when all actions are safe.

        Purpose: Validates safety mask for all-safe case.

        Given: Failure probabilities [0.05, 0.08] and delta_prime=0.1.
        When: _compute_safety_mask is called.
        Then: Mask is [1, 1].

        Test type: unit
        """
        mask = _compute_safety_mask(np.array([0.05, 0.08]), 0.1)
        np.testing.assert_array_equal(mask, [1.0, 1.0])

    def test_some_unsafe(self):
        """Test mask when some actions are unsafe.

        Purpose: Validates safety mask masks only unsafe actions.

        Given: Failure probabilities [0.05, 0.2] and delta_prime=0.1.
        When: _compute_safety_mask is called.
        Then: Mask is [1, 0].

        Test type: unit
        """
        mask = _compute_safety_mask(np.array([0.05, 0.2]), 0.1)
        np.testing.assert_array_equal(mask, [1.0, 0.0])

    def test_all_unsafe_fallback(self):
        """Test mask when all actions are unsafe.

        Purpose: Validates fallback to unconstrained when no safe action exists.

        Given: Failure probabilities [0.5, 0.8] and delta_prime=0.1.
        When: _compute_safety_mask is called.
        Then: Mask is [1, 1] (fallback to unconstrained).

        Test type: unit
        """
        mask = _compute_safety_mask(np.array([0.5, 0.8]), 0.1)
        np.testing.assert_array_equal(mask, [1.0, 1.0])

    def test_exact_threshold(self):
        """Test action at exactly the threshold is considered safe.

        Purpose: Validates boundary condition (f <= delta).

        Given: Failure probability [0.1] and delta_prime=0.1.
        When: _compute_safety_mask is called.
        Then: Mask is [1] (safe).

        Test type: unit
        """
        mask = _compute_safety_mask(np.array([0.1]), 0.1)
        np.testing.assert_array_equal(mask, [1.0])


class TestSpuctSelection:
    """Tests for the spuct_selection function."""

    def test_safe_action_preferred_over_unsafe(self):
        """Test SPUCT prefers safe actions over unsafe ones.

        Purpose: Validates that unsafe actions are masked out.

        Given: Two actions with equal Q/visits but action 0 is safe, action 1 is unsafe.
        When: spuct_selection is called.
        Then: The safe action (a_0) is selected.

        Test type: unit
        """
        node, children = _make_belief_node_with_children(
            q_values=[1.0, 2.0], visit_counts=[5, 5], parent_visits=10
        )
        # action 1 has higher Q but is unsafe
        failure_dict = {id(children[0]): 0.05, id(children[1]): 0.5}

        selected = spuct_selection(
            belief_node=node,
            exploration_constant=1.0,
            failure_dict=failure_dict,
            delta_prime=0.1,
        )
        assert selected.action == "a_0"

    def test_all_unsafe_selects_best_q(self):
        """Test all-unsafe fallback selects highest PUCT score.

        Purpose: Validates unconstrained fallback when all actions unsafe.

        Given: Two unsafe actions with Q=[1.0, 3.0] and many visits.
        When: spuct_selection is called.
        Then: The higher-Q action is selected (fallback to unconstrained).

        Test type: unit
        """
        node, children = _make_belief_node_with_children(
            q_values=[1.0, 3.0], visit_counts=[100, 100], parent_visits=200
        )
        failure_dict = {id(children[0]): 0.5, id(children[1]): 0.8}

        selected = spuct_selection(
            belief_node=node,
            exploration_constant=1.0,
            failure_dict=failure_dict,
            delta_prime=0.1,
        )
        assert selected.action == "a_1"

    def test_matches_puct_when_all_safe(self):
        """Test SPUCT matches PUCT when all actions are safe.

        Purpose: Validates SPUCT reduces to standard PUCT when safety mask is all ones.

        Given: Two actions with Q=[1.0, 3.0], both safe.
        When: spuct_selection is called.
        Then: Selects the same action as standard PUCT (higher Q with many visits).

        Test type: unit
        """
        node, children = _make_belief_node_with_children(
            q_values=[1.0, 3.0], visit_counts=[100, 100], parent_visits=200
        )
        failure_dict = {id(children[0]): 0.01, id(children[1]): 0.02}

        selected = spuct_selection(
            belief_node=node,
            exploration_constant=1.0,
            failure_dict=failure_dict,
            delta_prime=0.1,
        )
        # With many visits, exploration is small, so higher Q wins
        assert selected.action == "a_1"

    def test_uniform_priors_when_none(self):
        """Test SPUCT uses uniform priors when action_priors is None.

        Purpose: Validates default prior behavior.

        Given: Three actions with different Q-values, no explicit priors.
        When: spuct_selection is called with action_priors=None.
        Then: Returns a valid action (no error).

        Test type: unit
        """
        node, _ = _make_belief_node_with_children(
            q_values=[1.0, 2.0, 3.0],
            visit_counts=[10, 10, 10],
            parent_visits=30,
        )
        failure_dict = {}  # all zero failures

        selected = spuct_selection(
            belief_node=node,
            exploration_constant=1.0,
            failure_dict=failure_dict,
            delta_prime=0.5,
            action_priors=None,
        )
        assert selected.action in ["a_0", "a_1", "a_2"]


class TestSpuctActionProgressiveWidening:
    """Tests for the spuct_action_progressive_widening function."""

    def test_creates_new_action_on_empty_node(self):
        """Test new action is created for a leaf belief node.

        Purpose: Validates widening creates new action on first visit.

        Given: A leaf belief node with no children.
        When: spuct_action_progressive_widening is called.
        Then: A new action node is created and returned.

        Test type: unit
        """
        particles = [[0.0], [1.0]]
        log_weights = np.log([0.5, 0.5])
        belief = WeightedParticleBelief(particles, log_weights)
        node = BeliefNode(belief=belief)

        sampler = SimpleActionSampler()
        action_node = spuct_action_progressive_widening(
            belief_node=node,
            alpha_a=0.5,
            action_sampler=sampler,
            exploration_constant=1.0,
            k_a=1.0,
            failure_dict={},
            delta_prime=0.1,
        )
        assert action_node is not None
        assert action_node.action.startswith("new_action_")

    def test_selects_via_spuct_when_not_widening(self):
        """Test SPUCT selection when widening threshold not met.

        Purpose: Validates SPUCT is used instead of creating new actions.

        Given: A node with many children relative to k_a * N^alpha_a.
        When: spuct_action_progressive_widening is called.
        Then: An existing child is selected via SPUCT.

        Test type: unit
        """
        # With k_a=0.1, alpha_a=0.5, N=150: threshold = 0.1 * 150^0.5 ~ 1.22
        # 3 children > 1.22 -> no widening -> SPUCT selection
        node, children = _make_belief_node_with_children(
            q_values=[1.0, 2.0, 3.0],
            visit_counts=[50, 50, 50],
            parent_visits=150,
        )
        failure_dict = {id(c): 0.01 for c in children}

        sampler = SimpleActionSampler()
        selected = spuct_action_progressive_widening(
            belief_node=node,
            alpha_a=0.5,
            action_sampler=sampler,
            exploration_constant=1.0,
            k_a=0.1,
            failure_dict=failure_dict,
            delta_prime=0.5,
        )
        assert selected.action in ["a_0", "a_1", "a_2"]

    def test_min_visit_count_at_root(self):
        """Test min_visit_count_per_action enforced at root.

        Purpose: Validates that at depth 0, unvisited children are returned first.

        Given: A root node with two children, one unvisited.
        When: spuct_action_progressive_widening is called with min_visit_count=1.
        Then: The unvisited child is returned.

        Test type: unit
        """
        particles = [[0.0], [1.0]]
        log_weights = np.log([0.5, 0.5])
        belief = WeightedParticleBelief(particles, log_weights)
        node = BeliefNode(belief=belief)
        node.visit_count = 10

        child1 = ActionNode(action="a_0", parent=node)
        child1.visit_count = 5
        child2 = ActionNode(action="a_1", parent=node)
        child2.visit_count = 0  # unvisited

        failure_dict = {}
        sampler = SimpleActionSampler()

        selected = spuct_action_progressive_widening(
            belief_node=node,
            alpha_a=0.5,
            action_sampler=sampler,
            exploration_constant=1.0,
            k_a=1.0,
            failure_dict=failure_dict,
            delta_prime=0.1,
            min_visit_count_per_action=1,
        )
        assert selected.action == "a_1"
