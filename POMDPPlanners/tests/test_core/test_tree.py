import pytest
import numpy as np
from anytree import NodeMixin, RenderTree, PostOrderIter

from POMDPPlanners.core.tree import ActionNode, BeliefNode, get_optimal_action_cost_setting
from POMDPPlanners.core.belief import ParticleBelief
from POMDPPlanners.core.environment import Environment


# Create a simple test environment for belief updates
class TestEnvironment(Environment):
    def state_transition(self, state, action):
        return state

    def observation_model(self, state, action):
        class DummyModel:
            def probability(self, obs):
                return 1.0

        return DummyModel()


@pytest.fixture
def test_belief():
    # Create a simple particle belief with two particles
    particles = [1, 2]  # Simple integer particles
    log_weights = np.log(np.array([0.6, 0.4]))  # Convert to log weights
    return ParticleBelief(particles=particles, log_weights=log_weights)


@pytest.fixture
def test_env():
    return TestEnvironment()


def test_action_node_initialization():
    # Test basic initialization
    action = "move_forward"
    node = ActionNode(
        action, children=()
    )  # Initialize with empty tuple instead of None
    assert node.action == action
    assert node.q_value == 0.0
    assert node.visit_count == 0
    assert node.immediate_cost == None
    assert node.sample == []
    assert node.lower_confidence_bound == 0.0
    assert node.upper_confidence_bound == 0.0
    assert node.parent is None
    assert node.children == ()

    # Test initialization with parent and data
    parent = ActionNode("parent_action", children=())
    data = {"test": "data"}
    node = ActionNode(action, parent=parent, data=data, children=())
    assert node.parent == parent
    assert node.data == data


def test_belief_node_initialization(test_belief):
    # Test basic initialization
    node = BeliefNode(test_belief, children=())
    assert node.belief == test_belief
    assert node.v_value == 0.0
    assert node.visit_count == 0
    assert node.lower_confidence_bound == 0.0
    assert node.upper_confidence_bound == 0.0
    assert node.parent is None
    assert node.children == ()

    # Test initialization with parent and data
    parent = BeliefNode(test_belief, children=())
    data = {"test": "data"}
    node = BeliefNode(test_belief, parent=parent, data=data, children=())
    assert node.parent == parent
    assert node.data == data


def test_tree_structure(test_belief):
    # Create a simple tree structure
    root = BeliefNode(test_belief, children=())
    action1 = ActionNode("action1", parent=root, children=())
    belief1 = BeliefNode(test_belief, parent=action1, children=())
    action2 = ActionNode("action2", parent=root, children=())
    belief2 = BeliefNode(test_belief, parent=action2, children=())

    # Test parent-child relationships
    assert root.children == (action1, action2)
    assert action1.parent == root
    assert action2.parent == root
    assert belief1.parent == action1
    assert belief2.parent == action2


def test_get_optimal_action(test_belief):
    # Create a belief node with multiple action children
    belief_node = BeliefNode(test_belief, children=())

    # Create action nodes with different q-values
    action1 = ActionNode("action1", parent=belief_node, children=())
    action1.q_value = 0.5

    action2 = ActionNode("action2", parent=belief_node, children=())
    action2.q_value = 0.8

    action3 = ActionNode("action3", parent=belief_node, children=())
    action3.q_value = 0.3

    # Test that the action with highest q-value is selected
    optimal_action = get_optimal_action_cost_setting(belief_node)
    assert optimal_action == "action3"

def test_node_properties(test_belief):
    # Test updating node properties
    action_node = ActionNode("test_action", children=())
    action_node.q_value = 1.5
    action_node.visit_count = 10
    action_node.immediate_cost = 2
    action_node.sample = [1, 2, 3]
    action_node.lower_confidence_bound = 0.1
    action_node.upper_confidence_bound = 0.9

    assert action_node.q_value == 1.5
    assert action_node.visit_count == 10
    assert action_node.immediate_cost == 2
    assert action_node.sample == [1, 2, 3]
    assert action_node.lower_confidence_bound == 0.1
    assert action_node.upper_confidence_bound == 0.9

    belief_node = BeliefNode(test_belief, children=())
    belief_node.v_value = 2.0
    belief_node.visit_count = 5
    belief_node.lower_confidence_bound = 0.2
    belief_node.upper_confidence_bound = 0.8

    assert belief_node.v_value == 2.0
    assert belief_node.visit_count == 5
    assert belief_node.lower_confidence_bound == 0.2
    assert belief_node.upper_confidence_bound == 0.8
