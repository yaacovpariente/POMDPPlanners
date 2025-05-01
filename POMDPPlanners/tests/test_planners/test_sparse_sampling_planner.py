import pytest
import numpy as np
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.planners.sparse_sampling_planner import (
    StandardSparseSamplingDiscreteActionsPlanner,
)
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from anytree import PostOrderIter


@pytest.fixture
def tiger_pomdp():
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def initial_belief(tiger_pomdp):
    # Create a uniform belief over states using particles
    states = list(tiger_pomdp.states)
    particles = states * 10  # Create 10 particles for each state
    log_weights = np.log(
        np.ones(len(particles)) / len(particles)
    )  # Uniform log weights
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


@pytest.fixture
def planner(tiger_pomdp):
    return StandardSparseSamplingDiscreteActionsPlanner(
        environment=tiger_pomdp,
        branching_factor=2,
        depth=2
    )


def test_initialization(planner, tiger_pomdp):
    """Test that the planner initializes correctly"""
    assert planner.environment == tiger_pomdp
    assert planner.branching_factor == 2
    assert planner.depth == 2


def test_action_selection(planner, initial_belief):
    """Test that action selection returns a valid action"""
    action = planner.action(initial_belief)
    assert action in planner.environment.get_actions()


def test_belief_tree_construction(planner, initial_belief):
    """Test that the belief tree is constructed correctly"""
    tree = BeliefNode(belief=initial_belief)
    planner._build_tree(belief_node=tree, current_depth=0)

    # Check root node
    assert tree.belief == initial_belief
    assert tree.parent is None
    assert len(tree.children) > 0

    # The +1 is for the root node, and the * 2 is because both a
    # belief and an action node can be at the same depth.
    assert tree.height == planner.depth * 2 + 1


def test_node_statistics(planner, initial_belief):
    """Test that node statistics are updated correctly"""
    tree = BeliefNode(
        belief=initial_belief,
        parent=None,
        children=tuple(),  # Initialize with empty tuple
    )
    planner._build_tree(belief_node=tree, current_depth=0)

    # Check that all nodes have statistics
    for node in PostOrderIter(tree):
        if node.is_leaf:
            assert hasattr(node, "immediate_cost")
            assert hasattr(node, "q_value")
            assert hasattr(node, "visit_count")
        elif isinstance(node, ActionNode):
            assert hasattr(node, "visit_count")
            assert hasattr(node, "q_value")
        elif isinstance(node, BeliefNode):
            assert hasattr(node, "v_value")
            assert hasattr(node, "visit_count")


def test_leaf_node_statistics(planner, initial_belief):
    """Test that leaf node statistics are calculated correctly"""
    tree = planner._learn_belief_tree(initial_belief)

    for node in PostOrderIter(tree):
        if node.is_leaf:
            assert isinstance(node, ActionNode)
            assert node.visit_count == 1
            assert node.q_value == node.immediate_cost


def test_non_leaf_action_node_statistics(planner, initial_belief):
    """Test that non-leaf action node statistics are calculated correctly using fixed expected values"""
    tree = BeliefNode(belief=initial_belief)

    # Create an action node
    action_node = ActionNode(action="listen", parent=tree, children=tuple(), data=None)

    # Create two belief nodes as children with known v_values
    belief_node1 = BeliefNode(
        belief=initial_belief, parent=action_node, children=tuple(), data=None
    )
    belief_node1.v_value = 2.0

    belief_node2 = BeliefNode(
        belief=initial_belief, parent=action_node, children=tuple(), data=None
    )
    belief_node2.v_value = 4.0

    # Set immediate cost for action node
    action_node.immediate_cost = 1.0

    # Update action node statistics
    planner._update_non_leaf_action_node_statistics(action_node)

    # Expected q_value = immediate_cost + discount_factor * mean(children_v_values)
    # = 1.0 + 0.95 * ((2.0 + 4.0) / 2) = 1.0 + 0.95 * 3.0 = 3.85
    expected_q_value = 1.0 + planner.discount_factor * 3.0

    assert action_node.visit_count > 0
    assert np.isclose(action_node.q_value, expected_q_value)


def test_belief_node_statistics(planner, initial_belief):
    """Test that belief node statistics are calculated correctly using fixed expected values"""
    # Create a belief node
    belief_node = BeliefNode(belief=initial_belief)

    # Create two action nodes as children with known q_values
    action_node1 = ActionNode(
        action="listen", parent=belief_node, children=tuple(), data=None
    )
    action_node1.q_value = 3.0
    action_node1.visit_count = 2

    action_node2 = ActionNode(
        action="open-left", parent=belief_node, children=tuple(), data=None
    )
    action_node2.q_value = 5.0
    action_node2.visit_count = 3

    # Update belief node statistics
    planner._update_belief_node_statistics(belief_node)

    # Expected v_value should be min of children q_values (3.0)
    # Expected visit_count should be sum of children visit_counts (2 + 3 = 5)
    assert np.isclose(belief_node.v_value, 3.0)
    assert belief_node.visit_count == 5


def test_invalid_branching_factor():
    """Test that invalid branching factor raises an error"""
    env = TigerPOMDP(discount_factor=0.95)
    with pytest.raises(ValueError):
        StandardSparseSamplingDiscreteActionsPlanner(
            environment=env,
            branching_factor=0,
            depth=1
        )


def test_invalid_depth():
    """Test that invalid depth raises an error"""
    env = TigerPOMDP(discount_factor=0.95)
    with pytest.raises(ValueError):
        StandardSparseSamplingDiscreteActionsPlanner(
            environment=env,
            branching_factor=2,
            depth=0
        )
