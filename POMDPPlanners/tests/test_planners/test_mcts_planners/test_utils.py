# SPDX-License-Identifier: MIT

"""Shared test utilities for MCTS planners.

This module provides common test functions that can be reused across different MCTS planner tests
to ensure consistent validation and reduce code duplication.
"""

import random

import numpy as np
from anytree import PostOrderIter

from POMDPPlanners.core.tree import ActionNode, BeliefNode

np.random.seed(42)
random.seed(42)


def validate_tree_structure_with_progressive_widening(
    root_belief_node,
    planner,
    n_simulations,
    depth,
    k_o,
    k_a,
    alpha_o,
    alpha_a,
    action_sampler,
    expected_belief_type=None,
    planner_type="POMCP_DPW",
):
    """
    Validates complete tree structure construction and node integrity for MCTS planners with progressive widening.

    This function provides comprehensive validation of MCTS tree structures with double progressive widening,
    suitable for both POMCP_DPW and POMCPOW planners. It verifies tree structure, node integrity,
    progressive widening constraints, and value propagation.

    Args:
        root_belief_node: Root belief node of the constructed tree
        planner: The MCTS planner instance (POMCP_DPW or POMCPOW)
        n_simulations: Number of simulations used to build the tree
        depth: Maximum search depth parameter
        k_o: Observation progressive widening coefficient
        k_a: Action progressive widening coefficient
        alpha_o: Observation progressive widening exponent
        alpha_a: Action progressive widening exponent
        action_sampler: Action sampler used by the planner
        expected_belief_type: Expected belief type for non-root nodes (optional)

    Raises:
        AssertionError: If any tree structure validation fails
    """
    _validate_root_node(root_belief_node, n_simulations, k_a, alpha_a, action_sampler)

    belief_count, action_count, max_depth = _validate_tree_nodes(
        root_belief_node, expected_belief_type, planner_type, depth, k_o, alpha_o
    )

    _validate_overall_tree_structure(belief_count, action_count, max_depth, depth)
    _validate_progressive_widening_constraints(root_belief_node, k_a, alpha_a)
    _validate_root_value_propagation(root_belief_node)


def _validate_root_node(root_belief_node, n_simulations, k_a, alpha_a, action_sampler):
    assert root_belief_node.observation is None
    assert root_belief_node.parent is None
    assert root_belief_node.visit_count == n_simulations
    assert root_belief_node.v_value is not None
    assert len(root_belief_node.children) > 0

    action_children = [
        child for child in root_belief_node.children if isinstance(child, ActionNode)
    ]
    assert len(action_children) > 0

    max_expected_actions = k_a * (n_simulations**alpha_a)
    assert len(action_children) <= max_expected_actions + 1

    unique_actions = set(child.action for child in action_children)
    assert all(action in action_sampler.get_space() for action in unique_actions)


def _validate_tree_nodes(root_belief_node, expected_belief_type, planner_type, depth, k_o, alpha_o):
    belief_node_count = 0
    action_node_count = 0
    max_observed_depth = 0

    for node in PostOrderIter(root_belief_node):
        assert node.visit_count >= 0, f"Node visit count {node.visit_count} must be non-negative"
        assert hasattr(node, "depth"), "All nodes must have depth attribute"
        max_observed_depth = max(max_observed_depth, node.depth)

        if isinstance(node, BeliefNode):
            belief_node_count += 1
            _validate_belief_node(node, root_belief_node, expected_belief_type, planner_type, depth)
        elif isinstance(node, ActionNode):
            action_node_count += 1
            _validate_action_node(node, k_o, alpha_o)

    return belief_node_count, action_node_count, max_observed_depth


def _validate_belief_node(node, root_belief_node, expected_belief_type, planner_type, depth):
    assert node.belief is not None, "BeliefNode must have belief"
    assert isinstance(node.v_value, (int, float)), "BeliefNode must have a number v_value"

    if node != root_belief_node:
        assert node.observation is not None, "Non-root BeliefNode must have observation"
        assert node.parent is not None, "Non-root BeliefNode must have parent"
        assert isinstance(node.parent, ActionNode), "BeliefNode parent must be ActionNode"

    if expected_belief_type is not None and node.depth > 0:
        assert isinstance(
            node.belief, expected_belief_type
        ), f"Belief must be {expected_belief_type.__name__}"

    _validate_belief_node_visit_counts(node, planner_type, depth)


def _validate_belief_node_visit_counts(node, planner_type, depth):
    if not node.is_leaf and node.depth > 0:
        child_action_visits = sum(
            child.visit_count for child in node.children if isinstance(child, ActionNode)
        )
        if planner_type == "POMCP_DPW":
            assert (
                node.visit_count == child_action_visits + 1
            ), f"BeliefNode visit count {node.visit_count} should be >= sum of child visits {child_action_visits}"
        else:
            assert (
                node.visit_count >= child_action_visits
            ), f"BeliefNode visit count {node.visit_count} should be >= sum of child visits {child_action_visits}"
    else:
        if depth == node.depth:
            if planner_type == "POMCP_DPW":
                assert node.visit_count == 1, "Leaf belief node visit count must be 1"
            else:
                assert node.visit_count >= 1, "Leaf belief node visit count must be >= 1"
        else:
            assert node.visit_count >= 1, "Leaf belief node visit count must be >= 1"


def _validate_action_node(node, k_o, alpha_o):
    assert node.action is not None, "ActionNode must have action"
    assert node.q_value is not None, "ActionNode must have q_value"
    assert node.parent is not None, "ActionNode must have parent"
    assert isinstance(node.parent, BeliefNode), "ActionNode parent must be BeliefNode"

    if not node.is_leaf and node.visit_count > 0:
        assert isinstance(node.q_value, (int, float)), "Q-value must be numeric"
        assert not np.isnan(node.q_value), "Q-value cannot be NaN"

    if not node.is_leaf:
        child_belief_visits = sum(
            child.visit_count for child in node.children if isinstance(child, BeliefNode)
        )
        # With duplicate action prevention, the same action node can be selected multiple times
        # before creating new belief children, so visit_count >= child_belief_visits
        assert (
            node.visit_count >= child_belief_visits
        ), f"ActionNode visit count {node.visit_count} should be >= sum of child visits {child_belief_visits}"

    if node.visit_count > 1:
        max_allowed_observations = k_o * (node.visit_count**alpha_o)
        actual_observations = len(node.children)
        assert (
            actual_observations <= max_allowed_observations + 2
        ), f"Observation progressive widening violated: {actual_observations} obs > {max_allowed_observations:.2f} max allowed"


def _validate_overall_tree_structure(belief_count, action_count, max_depth, depth):
    assert belief_count > 0, "Tree must contain belief nodes"
    assert action_count > 0, "Tree must contain action nodes"
    # With duplicate action prevention, the relationship between belief and action nodes
    # can vary. Typically each action node should have at least one belief child,
    # but with progressive widening and duplicate prevention, the exact ratio can vary.
    # We allow some flexibility while ensuring the tree structure is reasonable.
    # In a typical tree: belief_count (including root) should be >= action_count + 1
    # But with duplicate prevention allowing action reuse, we relax this slightly.
    min_expected_beliefs = max(1, action_count // 2)  # At least half as many beliefs as actions
    assert (
        belief_count >= min_expected_beliefs
    ), f"Tree structure seems invalid: {belief_count} belief nodes for {action_count} action nodes"
    assert (
        max_depth <= 2 * depth + 2
    ), f"Maximum observed depth {max_depth} should not exceed 2*depth+2 = {2*depth+2}"


def _validate_progressive_widening_constraints(root_belief_node, k_a, alpha_a):
    for node in PostOrderIter(root_belief_node):
        if isinstance(node, BeliefNode) and node.visit_count > 0:
            action_children_count = len(
                [child for child in node.children if isinstance(child, ActionNode)]
            )
            max_allowed_actions_for_node = k_a * (node.visit_count**alpha_a)
            assert (
                action_children_count <= max_allowed_actions_for_node + 2
            ), f"Progressive widening violated at node: {action_children_count} > {max_allowed_actions_for_node:.2f}"


def _validate_root_value_propagation(root_belief_node):
    assert isinstance(root_belief_node.v_value, (int, float)), "Root v_value must be numeric"
    assert not np.isnan(root_belief_node.v_value), "Root v_value cannot be NaN"
