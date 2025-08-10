"""Shared test utilities for MCTS planners.

This module provides common test functions that can be reused across different MCTS planner tests
to ensure consistent validation and reduce code duplication.
"""

import numpy as np
from anytree import PostOrderIter
from POMDPPlanners.core.tree import BeliefNode, ActionNode


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
    planner_type="POMCP_DPW"
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
    # ASSERT: Verify tree structure and node integrity
    # Root node verification
    assert root_belief_node.observation is None  # Root has no observation
    assert root_belief_node.parent is None      # Root has no parent
    assert root_belief_node.visit_count == n_simulations  # Root visited by all simulations
    assert root_belief_node.v_value is not None  # Root must have value
    assert len(root_belief_node.children) > 0    # Root must have action children
    
    # Progressive widening verification for root actions
    # Due to action progressive widening, creates multiple action nodes (can exceed unique actions)
    action_children = [child for child in root_belief_node.children if isinstance(child, ActionNode)]
    assert len(action_children) > 0
    
    # Progressive widening creates action nodes based on k_a * n^alpha_a formula
    max_expected_actions = k_a * (n_simulations ** alpha_a)
    assert len(action_children) <= max_expected_actions + 1  # Allow small variance
    
    # Actions in the nodes should be from the valid action space
    unique_actions = set(child.action for child in action_children)
    assert all(action in action_sampler.get_space() for action in unique_actions)
    
    # Detailed tree structure verification using PostOrderIter
    belief_node_count = 0
    action_node_count = 0
    max_observed_depth = 0
    
    for node in PostOrderIter(root_belief_node):
        # Common node properties
        assert node.visit_count >= 0, f"Node visit count {node.visit_count} must be non-negative"
        assert hasattr(node, 'depth'), "All nodes must have depth attribute"
        max_observed_depth = max(max_observed_depth, node.depth)
        
        if isinstance(node, BeliefNode):
            belief_node_count += 1
            
            # BeliefNode-specific validations
            assert node.belief is not None, "BeliefNode must have belief"
            assert isinstance(node.v_value, (int, float)), "BeliefNode must have a number v_value"
            
            # Non-root belief nodes must have observations and proper structure
            if node != root_belief_node:
                assert node.observation is not None, "Non-root BeliefNode must have observation"
                assert node.parent is not None, "Non-root BeliefNode must have parent"
                assert isinstance(node.parent, ActionNode), "BeliefNode parent must be ActionNode"

            # Verify belief type for specific planners if provided
            if expected_belief_type is not None and node.depth > 0:  # Non-root nodes created during tree construction
                # Check if it's the expected type or a compatible belief type
                assert isinstance(node.belief, expected_belief_type), f"Belief must be {expected_belief_type.__name__}"
            
            # Visit count consistency for belief nodes
            if not node.is_leaf and node.depth > 0:
                child_action_visits = sum(child.visit_count for child in node.children if isinstance(child, ActionNode))
                # In MCTS with progressive widening, belief node visits should be consistent with action visits
                # Allow some variance due to progressive widening and rollouts
                if planner_type == "POMCP_DPW":
                    # POMCP_DPW has strict visit count relationship due to rollouts
                    assert node.visit_count == child_action_visits + 1, \
                        f"BeliefNode visit count {node.visit_count} should be >= sum of child visits {child_action_visits}"
                else:
                    # POMCPOW and other planners may have different visit count patterns
                    assert node.visit_count >= child_action_visits, \
                        f"BeliefNode visit count {node.visit_count} should be >= sum of child visits {child_action_visits}"
            else:
                if depth == node.depth:
                    if planner_type == "POMCP_DPW":
                        assert node.visit_count == 1, "Leaf belief node visit count must be 1"
                    else:
                        assert node.visit_count >= 1, "Leaf belief node visit count must be >= 1"
                else:
                    assert node.visit_count >= 1, "Leaf belief node visit count must be >= 1"

        elif isinstance(node, ActionNode):
            action_node_count += 1
            
            # ActionNode-specific validations  
            assert node.action is not None, "ActionNode must have action"
            assert node.q_value is not None, "ActionNode must have q_value"
            assert node.parent is not None, "ActionNode must have parent"
            assert isinstance(node.parent, BeliefNode), "ActionNode parent must be BeliefNode"
            
            # Action must be from valid action space
            assert node.action in action_sampler.get_space(), \
                f"Action {node.action} must be in action sampler space {action_sampler.get_space()}"
            
            # Non-leaf action nodes with visits should have meaningful q_values
            if not node.is_leaf and node.visit_count > 0:
                # Q-value should be a valid number (allow 0 but not None/NaN)
                assert isinstance(node.q_value, (int, float)), "Q-value must be numeric"
                assert not np.isnan(node.q_value), "Q-value cannot be NaN"
            
            # Visit count consistency for action nodes
            if not node.is_leaf:
                child_belief_visits = sum(child.visit_count for child in node.children if isinstance(child, BeliefNode))
                # In MCTS, action visit count should equal sum of child belief visits
                assert node.visit_count == child_belief_visits, \
                    f"ActionNode visit count {node.visit_count} should equal sum of child visits {child_belief_visits}"
            
            # Progressive widening validation for observation nodes
            if node.visit_count > 1:  # Only check nodes with sufficient visits
                max_allowed_observations = k_o * (node.visit_count ** alpha_o)
                actual_observations = len(node.children)
                # Allow small variance due to sampling and progressive widening mechanics
                assert actual_observations <= max_allowed_observations + 2, \
                    f"Observation progressive widening violated: {actual_observations} obs > {max_allowed_observations:.2f} max allowed"
    
    # Overall tree structure validations
    assert belief_node_count > 0, "Tree must contain belief nodes"
    assert action_node_count > 0, "Tree must contain action nodes"
    assert belief_node_count >= action_node_count, "Should have at least as many belief nodes as action nodes"
    
    # Tree depth should be reasonable given the depth parameter
    # In MCTS trees, depth can vary due to progressive widening and termination conditions
    assert max_observed_depth <= 2 * depth + 2, \
        f"Maximum observed depth {max_observed_depth} should not exceed 2*depth+2 = {2*depth+2}"
    
    # Progressive widening effectiveness check
    # In MCTS planners with progressive widening, action nodes are created based on visit counts and alpha parameters
    # This can actually create more action nodes than naive expansion due to multiple nodes per action
    # Instead verify that the progressive widening constraint is respected at individual nodes
    for node in PostOrderIter(root_belief_node):
        if isinstance(node, BeliefNode) and node.visit_count > 0:
            action_children_count = len([child for child in node.children if isinstance(child, ActionNode)])
            max_allowed_actions_for_node = k_a * (node.visit_count ** alpha_a)
            # Allow some variance due to the discrete nature of progressive widening checks
            assert action_children_count <= max_allowed_actions_for_node + 2, \
                f"Progressive widening violated at node: {action_children_count} > {max_allowed_actions_for_node:.2f}"
    
    # Value propagation verification - root should have reasonable value after many simulations
    assert isinstance(root_belief_node.v_value, (int, float)), "Root v_value must be numeric"
    assert not np.isnan(root_belief_node.v_value), "Root v_value cannot be NaN"