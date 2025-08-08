import pytest
import numpy as np
from anytree import NodeMixin, RenderTree, PostOrderIter

from POMDPPlanners.core.tree import ActionNode, BeliefNode, get_optimal_action_cost_setting
from POMDPPlanners.core.belief import WeightedParticleBelief
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
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


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


def test_sample_child_node(test_belief):
    """Test the sample_child_node method of ActionNode."""
    # Create an action node with multiple belief node children
    action_node = ActionNode("test_action", children=())
    
    # Create belief nodes with different visit counts
    belief1 = BeliefNode(test_belief, observation="obs1", parent=action_node, children=())
    belief1.visit_count = 10
    
    belief2 = BeliefNode(test_belief, observation="obs2", parent=action_node, children=())
    belief2.visit_count = 5
    
    belief3 = BeliefNode(test_belief, observation="obs3", parent=action_node, children=())
    belief3.visit_count = 15
    
    # Test sampling with different visit counts
    # Since this is probabilistic, we'll test multiple times to ensure it works
    sampled_nodes = []
    for _ in range(100):
        sampled_node = action_node.sample_child_node()
        sampled_nodes.append(sampled_node)
        assert sampled_node in [belief1, belief2, belief3]
    
    # Check that all nodes are sampled at least once (very likely with 100 samples)
    unique_sampled = set(sampled_nodes)
    assert len(unique_sampled) >= 2  # At least 2 different nodes should be sampled
    
    # Test with equal visit counts
    action_node_equal = ActionNode("test_action_equal", children=())
    belief_equal1 = BeliefNode(test_belief, observation="obs1", parent=action_node_equal, children=())
    belief_equal1.visit_count = 5
    
    belief_equal2 = BeliefNode(test_belief, observation="obs2", parent=action_node_equal, children=())
    belief_equal2.visit_count = 5
    
    # Test sampling with equal visit counts
    sampled_nodes_equal = []
    for _ in range(50):
        sampled_node = action_node_equal.sample_child_node()
        sampled_nodes_equal.append(sampled_node)
        assert sampled_node in [belief_equal1, belief_equal2]
    
    # Both nodes should be sampled roughly equally
    count_belief1 = sampled_nodes_equal.count(belief_equal1)
    count_belief2 = sampled_nodes_equal.count(belief_equal2)
    assert count_belief1 > 0 and count_belief2 > 0


def test_sample_child_node_single_child(test_belief):
    """Test sample_child_node with only one child."""
    action_node = ActionNode("test_action", children=())
    belief = BeliefNode(test_belief, observation="obs1", parent=action_node, children=())
    belief.visit_count = 5
    
    # Should always return the single child
    sampled_node = action_node.sample_child_node()
    assert sampled_node == belief


def test_sample_child_node_no_children():
    """Test sample_child_node with no children (should raise error)."""
    action_node = ActionNode("test_action", children=())
    
    # This should raise a ValueError because sum(child_visit_counts) would be 0
    with pytest.raises(ValueError):
        action_node.sample_child_node()


def test_get_belief_node_child(test_belief):
    """Test the get_belief_node_child method of ActionNode."""
    # Create an action node with multiple belief node children
    action_node = ActionNode("test_action", children=())
    
    # Create belief nodes with different observations
    belief1 = BeliefNode(test_belief, observation="obs1", parent=action_node, children=())
    belief2 = BeliefNode(test_belief, observation="obs2", parent=action_node, children=())
    belief3 = BeliefNode(test_belief, observation="obs3", parent=action_node, children=())
    
    # Test getting existing observations
    result1 = action_node.get_belief_node_child("obs1")
    assert result1 == belief1
    
    result2 = action_node.get_belief_node_child("obs2")
    assert result2 == belief2
    
    result3 = action_node.get_belief_node_child("obs3")
    assert result3 == belief3
    
    # Test getting non-existing observation
    result_none = action_node.get_belief_node_child("non_existent_obs")
    assert result_none is None


def test_get_belief_node_child_no_children(test_belief):
    """Test get_belief_node_child with no children."""
    action_node = ActionNode("test_action", children=())
    
    # Should return None for any observation
    result = action_node.get_belief_node_child("any_observation")
    assert result is None


def test_get_belief_node_child_duplicate_observations(test_belief):
    """Test get_belief_node_child with duplicate observations (should return first match)."""
    action_node = ActionNode("test_action", children=())
    
    # Create belief nodes with the same observation
    belief1 = BeliefNode(test_belief, observation="same_obs", parent=action_node, children=())
    belief2 = BeliefNode(test_belief, observation="same_obs", parent=action_node, children=())
    
    # Should return the first child with matching observation
    result = action_node.get_belief_node_child("same_obs")
    assert result == belief1


def test_get_belief_node_child_none_observation(test_belief):
    """Test get_belief_node_child with None observation."""
    action_node = ActionNode("test_action", children=())
    
    # Create belief nodes with None observation
    belief_none = BeliefNode(test_belief, observation=None, parent=action_node, children=())
    belief_obs = BeliefNode(test_belief, observation="obs1", parent=action_node, children=())
    
    # Test getting None observation
    result = action_node.get_belief_node_child(None)
    assert result == belief_none
    
    # Test getting regular observation
    result_obs = action_node.get_belief_node_child("obs1")
    assert result_obs == belief_obs
