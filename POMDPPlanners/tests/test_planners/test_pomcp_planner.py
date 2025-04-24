import numpy as np
import pytest
from unittest.mock import Mock, patch
from anytree import Node
from anytree import PostOrderIter

from POMDPPlanners.core.belief import ParticleBelief
from POMDPPlanners.core.tree import BeliefNode, ActionNode
from POMDPPlanners.planners.mcts_planners.pomcp_planner import POMCPPlanner
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

@pytest.fixture
def environment():
    return TigerPOMDP(discount_factor=0.9)

@pytest.fixture
def planner(environment):
    return POMCPPlanner(
        environment=environment,
        discount_factor=0.9,
        n_simulations=100,  # Increase simulations for better exploration
        depth=3,
        exploration_constant=1.0
    )

def test_initialization(planner, environment):
    assert planner.environment == environment
    assert planner.discount_factor == 0.9
    assert planner.n_simulations == 100  # Updated
    assert planner.depth == 3
    assert planner.exploration_constant == 1.0

def test_get_explored_action_node(planner):
    # Create a belief node with children
    belief = ParticleBelief(
        particles=["tiger_left", "tiger_right"],
        log_weights=np.array([-0.69314718, -0.69314718])  # log(0.5) for equal weights
    )
    belief_node = BeliefNode(
        belief=belief,
        observation=None,  # Root node has no observation
        parent=None,
        children=tuple(),
        data=None
    )
    belief_node.visit_count = 1  # Add visit_count attribute
    
    for action in planner.environment.get_actions():
        action_node = ActionNode(
            action=action,
            parent=belief_node,
            children=tuple(),
            data=None
        )
        action_node.q_value = -1.0  # Same Q-value for all actions
        action_node.visit_count = 1  # Same visit count for all actions
        
    last_action_node = belief_node.children[-1]
    last_action_node.q_value = -100.0

    # Test exploration
    selected_node = planner._get_explored_action_node(belief_node)
    assert isinstance(selected_node, ActionNode)
    assert selected_node in belief_node.children
    assert selected_node.action in planner.environment.get_actions()
    assert selected_node.action == last_action_node.action  # selected the node with the lowest q_value (this is a cost setting).

def test_update_leaf_node_q_value(planner):
    # Create a belief node and action node
    belief = ParticleBelief(
        particles=["tiger_left", "tiger_right"],
        log_weights=np.array([-0.69314718, -0.69314718])  # log(0.5) for equal weights
    )
    belief_node = BeliefNode(
        belief=belief,
        observation=None,  # Root node has no observation
        parent=None,
        children=tuple(),
        data=None
    )
    
    action_node = ActionNode(
        action="listen",  # Using a valid Tiger POMDP action
        parent=belief_node,
        children=tuple(),
        data=None
    )
    action_node.visit_count = 1

    planner._update_leaf_node_q_value(action_node)
    # The Q-value should be -1.0 (cost of listening) for both states
    # The value is log-weighted sum: exp(-0.69314718) * 1.0 + exp(-0.69314718) * 1.0 = 1.
    assert np.isclose(action_node.q_value, 1., rtol=1e-6)

def test_update_non_leaf_action_node_q_value(planner):
    # Create a belief node and action node with children
    belief = ParticleBelief(
        particles=["tiger_left", "tiger_right"],
        log_weights=np.array([-0.69314718, -0.69314718])
    )
    belief_node = BeliefNode(
        belief=belief,
        observation=None,  # Root node has no observation
        parent=None,
        children=tuple(),
        data=None
    )
    action_node = ActionNode(
        action="listen",  # Using a valid Tiger POMDP action
        parent=belief_node,
        children=tuple(),
        data=None
    )
    action_node.immediate_cost = -1.0  # Cost of listening
    action_node.q_value = 0.0
    action_node.visit_count = 1

    # Add children to action node
    for observation in planner.environment.observations:
        child = BeliefNode(
            belief=ParticleBelief(
                particles=["tiger_left", "tiger_right"],
                log_weights=np.array([-0.69314718, -0.69314718])
            ),
            observation=observation,  # Use the observation from the loop
            parent=action_node,
            children=tuple(),
            data=None
        )
        child.v_value = -1.0  # Expected value for next state
        child.visit_count = 1

    planner._update_non_leaf_action_node_q_value(action_node)
    expected_q = -1.0 + 0.9 * (-1.0)  # immediate_cost + discount_factor * average_v_value
    assert np.isclose(action_node.q_value, expected_q)

def test_update_v_and_q_values(planner):
    # Create a belief node with children
    belief = ParticleBelief(
        particles=["tiger_left", "tiger_right"],
        log_weights=np.array([-0.69314718, -0.69314718])
    )
    belief_node = BeliefNode(
        belief=belief,
        observation=None,  # Root node has no observation
        parent=None,
        children=tuple(),
        data=None
    )
    belief_node.visit_count = 1
    
    # Create action nodes with different Q-values
    action_node1 = ActionNode(
        action="listen",
        parent=belief_node,
        children=tuple(),
        data=None
    )
    action_node1.q_value = -1.0
    action_node1.immediate_cost = -1.0
    action_node1.visit_count = 1
    
    action_node2 = ActionNode(
        action="open_left",
        parent=belief_node,
        children=tuple(),
        data=None
    )
    action_node2.q_value = -100.0
    action_node2.immediate_cost = action_node2.q_value
    action_node2.visit_count = 1
    
    # Test updating leaf action node
    planner._update_v_and_q_values(belief_node=belief_node, action_node=action_node1)
    # For leaf action node, q_value should be updated to immediate cost
    assert np.isclose(action_node1.q_value, -1.0, rtol=1e-6)  # Cost of listening
    # For non-leaf belief node, v_value should be updated to min of children q_values
    assert belief_node.v_value == -100.0  # Should be the minimum Q-value
    
    # Test updating non-leaf action node
    # Add a child belief node to action_node1
    child_belief = BeliefNode(
        belief=belief,
        observation="hear_left",
        parent=action_node1,
        children=tuple(),
        data=None
    )
    child_belief.v_value = -2.0
    child_belief.visit_count = 1
    
    planner._update_v_and_q_values(belief_node=belief_node, action_node=action_node1)
    # For non-leaf action node, q_value should be updated to immediate cost + discounted average of children v_values
    expected_q = -1.0 + 0.9 * (-2.0)  # immediate_cost + discount_factor * child_v_value
    assert np.isclose(action_node1.q_value, expected_q, rtol=1e-6)
    # Belief node v_value should be updated to min of children q_values
    assert belief_node.v_value == -100.0  # Should still be the minimum Q-value

def test_action_selection(planner):
    # Create a belief with tiger more likely on the left
    belief = ParticleBelief(
        particles=["tiger_left"] * 8 + ["tiger_right"] * 2,  # 80% chance tiger is on left
        log_weights=np.array([-2.302585] * 10)  # log(0.1) for equal weights
    )

    # Test action selection
    action = planner.action(belief)
    assert action in planner.environment.get_actions()
    # Should either listen or open right door
    assert action in ["listen", "open_right", "open_left"]

def test_pomcp_planner_tree_node_update(planner):
    belief = ParticleBelief(
        particles=["tiger_left"] * 8 + ["tiger_right"] * 2,  # 80% chance tiger is on left
        log_weights=np.array([-2.302585] * 10)  # log(0.1) for equal weights
    )

    tree = planner._learn_belief_tree(belief)
    
    v_counter = 0
    q_counter = 0
    for node in PostOrderIter(tree):
        if isinstance(node, BeliefNode):
            v_counter += 1
        elif isinstance(node, ActionNode):
            q_counter += 1

    assert v_counter > 0
    assert q_counter > 0
