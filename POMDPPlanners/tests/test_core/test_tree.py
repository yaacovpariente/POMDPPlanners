# SPDX-License-Identifier: MIT

"""Tests for tree data structures.

This module tests the tree data structures, focusing on:
- Basic tree functionality
- Node operations
- Tree traversal
- Tree statistics
"""

import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.distributions import DiscreteDistribution
from POMDPPlanners.core.environment import (
    Environment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.tree import (
    ActionNode,
    BeliefNode,
    get_optimal_action_cost_setting,
)

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


# Create a simple test environment for belief updates
class MockEnvironment(Environment):
    def __init__(self):
        space_info = SpaceInfo(SpaceType.DISCRETE, SpaceType.DISCRETE)
        super().__init__(discount_factor=0.95, name="MockEnvironment", space_info=space_info)

    def sample_next_state(self, state, action, n_samples: int = 1):
        del action
        if n_samples == 1:
            return state
        return [state] * n_samples

    def sample_observation(self, next_state, action, n_samples: int = 1):
        del next_state, action
        if n_samples == 1:
            return "obs"
        return ["obs"] * n_samples

    def transition_log_probability(self, state, action, next_states) -> np.ndarray:
        del state, action
        return np.zeros(len(next_states))

    def observation_log_probability(self, next_state, action, observations) -> np.ndarray:
        del next_state, action
        return np.zeros(len(observations))

    def is_equal_observation(self, observation1, observation2):
        """Check if two observations are equal."""
        return observation1 == observation2

    def hash_action(self, action):
        return action

    def is_terminal(self, state):
        """Check if state is terminal."""
        return False

    def reward(self, state, action, next_state=None):
        """Return reward for state transition."""
        return 0.0

    def initial_state_dist(self):
        """Return initial state distribution."""
        return DiscreteDistribution(values=[0], probs=np.array([1.0]))

    def initial_observation_dist(self):
        """Return initial observation distribution."""
        return DiscreteDistribution(values=["obs"], probs=np.array([1.0]))


@pytest.fixture
def test_belief():
    """Create weighted particle belief for tree node testing."""
    # Create a simple particle belief with two particles for tree tests
    particles = [1, 2]  # Simple integer particles
    log_weights = np.log(np.array([0.6, 0.4]))  # Convert to log weights
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


@pytest.fixture
def test_env():
    """Test env.

    Purpose: Provides MockEnvironment fixture for tree node testing with POMDP interface

    Given: MockEnvironment implementation with discrete spaces, dummy models, and deterministic transitions
    When: Fixture is used in tree structure tests
    Then: Returns MockEnvironment instance with proper POMDP interface methods for belief updates

    Test type: unit
    """
    return MockEnvironment()


def test_action_node_initialization_creates_mcts_tree_node():
    """
    Purpose: Validates ActionNode initializes correctly for MCTS tree construction

    Given: Action string "move_forward" and optional parent node with test data
    When: ActionNode instances are created with basic and parent-child configurations
    Then: Nodes are initialized with correct action, default values, and proper tree relationships

    Test type: unit
    """
    # ARRANGE: Setup action and test data for node initialization
    test_action = "move_forward"
    parent_action = "parent_action"
    test_data = {"test": "data"}

    # ACT: Create basic ActionNode without parent
    basic_node = ActionNode(test_action, children=())

    # Create ActionNode with parent and data
    parent_node = ActionNode(parent_action, children=())
    child_node = ActionNode(test_action, parent=parent_node, data=test_data, children=())

    # ASSERT: Verify basic node initialization with correct defaults
    assert basic_node.action == test_action
    assert basic_node.q_value == 0.0  # Default Q-value for new nodes
    assert basic_node.visit_count == 0  # No visits initially
    assert basic_node.immediate_cost is None
    assert basic_node.sample == []  # Empty sample list
    assert basic_node.lower_confidence_bound == 0.0
    assert basic_node.upper_confidence_bound == 0.0
    assert basic_node.parent is None  # Root node has no parent
    assert basic_node.children == ()  # No children initially

    # Verify parent-child relationship and data storage
    assert child_node.parent == parent_node
    assert child_node.data == test_data
    assert child_node.action == test_action


def test_belief_node_initialization(test_belief):
    """Test belief node initialization.

    Purpose: Validates that BeliefNode initializes correctly with belief state and default MCTS values

    Given: WeightedParticleBelief with particles [1,2] and weights [0.6,0.4], optional parent and data parameters
    When: BeliefNode instances are created with basic and parent-child configurations
    Then: Nodes have correct belief reference, default v_value=0.0, visit_count=0, confidence bounds=0.0, and proper tree relationships

    Test type: unit
    """
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
    """Test tree structure.

    Purpose: Validates that MCTS tree structure correctly maintains parent-child relationships between BeliefNode and ActionNode

    Given: Root BeliefNode and multiple ActionNodes (action1, action2) with child BeliefNodes
    When: Tree hierarchy is constructed with proper parent-child assignments
    Then: Root has correct children tuple, ActionNodes have proper parents, and BeliefNodes maintain correct action parent references

    Test type: unit
    """
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
    """Test get optimal action.

    Purpose: Validates that get_optimal_action_cost_setting correctly selects action with lowest cost (highest negative q_value)

    Given: BeliefNode with 3 ActionNode children having different q_values (0.5, 0.8, 0.3)
    When: get_optimal_action_cost_setting evaluates the action choices
    Then: Returns action3 with lowest cost (0.3) for cost-minimization MCTS setting

    Test type: unit
    """
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
    """Test node properties.

    Purpose: Validates that ActionNode and BeliefNode properties can be updated correctly during MCTS simulation

    Given: ActionNode and BeliefNode instances with initial default values
    When: MCTS properties are updated (q_value, visit_count, v_value, confidence bounds)
    Then: All property updates persist correctly with expected values maintained

    Test type: unit
    """
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
    """Test the sample_child_node method of ActionNode.

    Purpose: Validates sampling behavior for  child node

    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution

    Test type: unit
    """
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
    belief_equal1 = BeliefNode(
        test_belief, observation="obs1", parent=action_node_equal, children=()
    )
    belief_equal1.visit_count = 5

    belief_equal2 = BeliefNode(
        test_belief, observation="obs2", parent=action_node_equal, children=()
    )
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
    """Test sample_child_node with only one child.

    Purpose: Validates sampling behavior for  child node single child

    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution

    Test type: unit
    """
    action_node = ActionNode("test_action", children=())
    belief = BeliefNode(test_belief, observation="obs1", parent=action_node, children=())
    belief.visit_count = 5

    # Should always return the single child
    sampled_node = action_node.sample_child_node()
    assert sampled_node == belief


def test_sample_child_node_no_children():
    """Test sample_child_node with no children (should raise error).

    Purpose: Validates sampling behavior for  child node no children

    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution

    Test type: unit
    """
    action_node = ActionNode("test_action", children=())

    # This should raise a ValueError because sum(child_visit_counts) would be 0
    with pytest.raises(ValueError):
        action_node.sample_child_node()


def test_get_belief_node_child(test_belief, test_env):
    """Test the get_belief_node_child method of ActionNode.

    Purpose: Validates that get_belief_node_child correctly retrieves BeliefNode children based on observation matching

    Given: ActionNode with 3 BeliefNode children having different observations (obs1, obs2, obs3)
    When: get_belief_node_child is called with existing and non-existing observation identifiers
    Then: Returns correct BeliefNode for existing observations and None for non-existent observations

    Test type: unit
    """
    # Create an action node with multiple belief node children
    action_node = ActionNode("test_action", children=())

    # Create belief nodes with different observations
    belief1 = BeliefNode(test_belief, observation="obs1", parent=action_node, children=())
    belief2 = BeliefNode(test_belief, observation="obs2", parent=action_node, children=())
    belief3 = BeliefNode(test_belief, observation="obs3", parent=action_node, children=())

    # Test getting existing observations
    result1 = action_node.get_belief_node_child("obs1", test_env)
    assert result1 == belief1

    result2 = action_node.get_belief_node_child("obs2", test_env)
    assert result2 == belief2

    result3 = action_node.get_belief_node_child("obs3", test_env)
    assert result3 == belief3

    # Test getting non-existing observation
    result_none = action_node.get_belief_node_child("non_existent_obs", test_env)
    assert result_none is None


def test_get_belief_node_child_no_children(test_belief, test_env):
    """Test get_belief_node_child with no children.

    Purpose: Validates that get_belief_node_child handles edge case of ActionNode with no children gracefully

    Given: ActionNode with empty children tuple
    When: get_belief_node_child is called with any observation identifier
    Then: Returns None indicating no matching child found

    Test type: unit
    """
    action_node = ActionNode("test_action", children=())

    # Should return None for any observation
    result = action_node.get_belief_node_child("any_observation", test_env)
    assert result is None


def test_get_belief_node_child_duplicate_observations(test_belief, test_env):
    """Test get_belief_node_child with duplicate observations (should return first match).

    Purpose: Validates that get_belief_node_child returns first matching BeliefNode when multiple children have identical observations

    Given: ActionNode with 2 BeliefNode children both having observation "same_obs"
    When: get_belief_node_child searches for "same_obs"
    Then: Returns first child (belief1) with matching observation due to first-match search behavior

    Test type: unit
    """
    action_node = ActionNode("test_action", children=())

    # Create belief nodes with the same observation
    belief1 = BeliefNode(test_belief, observation="same_obs", parent=action_node, children=())
    belief2 = BeliefNode(test_belief, observation="same_obs", parent=action_node, children=())

    # Should return the first child with matching observation
    result = action_node.get_belief_node_child("same_obs", test_env)
    assert result == belief1


def test_get_belief_node_child_none_observation(test_belief, test_env):
    """Test get_belief_node_child with None observation.

    Purpose: Validates that get_belief_node_child correctly handles None observations using proper equality comparison

    Given: ActionNode with BeliefNode children having None and string observations
    When: get_belief_node_child searches for None observation and string observation
    Then: Returns correct BeliefNode for None observation match and string observation match respectively

    Test type: unit
    """
    action_node = ActionNode("test_action", children=())

    # Create belief nodes with None observation
    belief_none = BeliefNode(test_belief, observation=None, parent=action_node, children=())
    belief_obs = BeliefNode(test_belief, observation="obs1", parent=action_node, children=())

    # Test getting None observation
    result = action_node.get_belief_node_child(None, test_env)
    assert result == belief_none

    # Test getting regular observation
    result_obs = action_node.get_belief_node_child("obs1", test_env)
    assert result_obs == belief_obs


def test_belief_node_get_child(test_belief):
    """Test the get_child method of BeliefNode.

    Purpose: Validates that get_child correctly retrieves ActionNode children based on action matching

    Given: BeliefNode with 3 ActionNode children having different actions (action1, action2, action3)
    When: get_child is called with existing and non-existing action identifiers
    Then: Returns correct ActionNode for existing actions and None for non-existent actions

    Test type: unit
    """
    # Create a belief node with multiple action children
    belief_node = BeliefNode(test_belief, children=())

    # Create action nodes with different actions
    action1 = ActionNode("action1", parent=belief_node, children=())
    action2 = ActionNode("action2", parent=belief_node, children=())
    action3 = ActionNode("action3", parent=belief_node, children=())

    # Test getting existing actions
    result1 = belief_node.get_child("action1")
    assert result1 == action1

    result2 = belief_node.get_child("action2")
    assert result2 == action2

    result3 = belief_node.get_child("action3")
    assert result3 == action3

    # Test getting non-existing action
    result_none = belief_node.get_child("non_existent_action")
    assert result_none is None


def test_belief_node_get_child_no_children(test_belief):
    """Test get_child with no children.

    Purpose: Validates that get_child handles edge case of BeliefNode with no children gracefully

    Given: BeliefNode with empty children tuple
    When: get_child is called with any action identifier
    Then: Returns None indicating no matching child found

    Test type: unit
    """
    belief_node = BeliefNode(test_belief, children=())

    # Should return None for any action
    result = belief_node.get_child("any_action")
    assert result is None


def test_belief_node_get_child_duplicate_actions(test_belief):
    """Test get_child with duplicate actions (should return first match).

    Purpose: Validates that get_child returns first matching ActionNode when multiple children have identical actions

    Given: BeliefNode with 2 ActionNode children both having action "same_action"
    When: get_child searches for "same_action"
    Then: Returns first child (action1) with matching action due to first-match search behavior

    Test type: unit
    """
    belief_node = BeliefNode(test_belief, children=())

    # Create action nodes with the same action
    action1 = ActionNode("same_action", parent=belief_node, children=())
    action2 = ActionNode("same_action", parent=belief_node, children=())

    # Should return the first child with matching action
    result = belief_node.get_child("same_action")
    assert result == action1


def test_belief_node_get_child_none_action(test_belief):
    """Test get_child with None action.

    Purpose: Validates that get_child correctly handles None actions using proper equality comparison

    Given: BeliefNode with ActionNode children having None and string actions
    When: get_child searches for None action and string action
    Then: Returns correct ActionNode for None action match and string action match respectively

    Test type: unit
    """
    belief_node = BeliefNode(test_belief, children=())

    # Create action nodes with None action
    action_none = ActionNode(None, parent=belief_node, children=())
    action_str = ActionNode("action1", parent=belief_node, children=())

    # Test getting None action
    result = belief_node.get_child(None)
    assert result == action_none

    # Test getting regular action
    result_str = belief_node.get_child("action1")
    assert result_str == action_str


def test_belief_node_get_child_numeric_actions(test_belief):
    """Test get_child with numeric actions.

    Purpose: Validates that get_child works correctly with numeric action identifiers

    Given: BeliefNode with ActionNode children having integer and float actions
    When: get_child searches for matching numeric actions
    Then: Returns correct ActionNode for exact numeric matches

    Test type: unit
    """
    belief_node = BeliefNode(test_belief, children=())

    # Create action nodes with numeric actions
    action_int = ActionNode(1, parent=belief_node, children=())
    action_float = ActionNode(2.5, parent=belief_node, children=())
    action_zero = ActionNode(0, parent=belief_node, children=())

    # Test getting integer action
    result_int = belief_node.get_child(1)
    assert result_int == action_int

    # Test getting float action
    result_float = belief_node.get_child(2.5)
    assert result_float == action_float

    # Test getting zero action
    result_zero = belief_node.get_child(0)
    assert result_zero == action_zero

    # Test non-existing numeric action
    result_none = belief_node.get_child(999)
    assert result_none is None


def test_immediate_cost_setter_updates_immediate_reward(test_belief):
    """Test that setting immediate_cost automatically updates immediate_reward to its negative value.

    Purpose: Validates that the immediate_cost setter property correctly updates immediate_reward to -immediate_cost

    Given: ActionNode and BeliefNode instances with initial immediate_cost=None and immediate_reward=None
    When: immediate_cost is set to various values (positive, negative, zero, None)
    Then: immediate_reward is automatically set to -immediate_cost for non-None values, and immediate_reward remains unchanged when immediate_cost is set to None

    Test type: unit
    """
    # Test with ActionNode - positive cost
    action_node = ActionNode("test_action", children=())
    action_node.immediate_cost = 5.0
    assert action_node.immediate_cost == 5.0
    assert action_node.immediate_reward == -5.0

    # Test with ActionNode - negative cost
    action_node.immediate_cost = -3.5
    assert action_node.immediate_cost == -3.5
    assert action_node.immediate_reward == 3.5

    # Test with ActionNode - zero cost
    action_node.immediate_cost = 0.0
    assert action_node.immediate_cost == 0.0
    assert action_node.immediate_reward == 0.0

    # Test with ActionNode - None (should not update immediate_reward)
    action_node._immediate_reward = 10.0  # Set private attribute directly to avoid setter
    action_node.immediate_cost = None
    assert action_node.immediate_cost is None
    assert action_node.immediate_reward == 10.0  # Should remain unchanged

    # Test with BeliefNode - positive cost
    belief_node = BeliefNode(test_belief, children=())
    belief_node.immediate_cost = 7.5
    assert belief_node.immediate_cost == 7.5
    assert belief_node.immediate_reward == -7.5

    # Test with BeliefNode - negative cost
    belief_node.immediate_cost = -2.0
    assert belief_node.immediate_cost == -2.0
    assert belief_node.immediate_reward == 2.0

    # Test with BeliefNode - zero cost
    belief_node.immediate_cost = 0.0
    assert belief_node.immediate_cost == 0.0
    assert belief_node.immediate_reward == 0.0

    # Test with BeliefNode - None (should not update immediate_reward)
    belief_node._immediate_reward = 15.0  # Set private attribute directly to avoid setter
    belief_node.immediate_cost = None
    assert belief_node.immediate_cost is None
    assert belief_node.immediate_reward == 15.0  # Should remain unchanged


def test_immediate_reward_setter_updates_immediate_cost(test_belief):
    """Test that setting immediate_reward automatically updates immediate_cost to its negative value.

    Purpose: Validates that the immediate_reward setter property correctly updates immediate_cost to -immediate_reward

    Given: ActionNode and BeliefNode instances with initial immediate_cost=None and immediate_reward=None
    When: immediate_reward is set to various values (positive, negative, zero, None)
    Then: immediate_cost is automatically set to -immediate_reward for non-None values, and immediate_cost remains unchanged when immediate_reward is set to None

    Test type: unit
    """
    # Test with ActionNode - positive reward
    action_node = ActionNode("test_action", children=())
    action_node.immediate_reward = 5.0
    assert action_node.immediate_reward == 5.0
    assert action_node.immediate_cost == -5.0

    # Test with ActionNode - negative reward
    action_node.immediate_reward = -3.5
    assert action_node.immediate_reward == -3.5
    assert action_node.immediate_cost == 3.5

    # Test with ActionNode - zero reward
    action_node.immediate_reward = 0.0
    assert action_node.immediate_reward == 0.0
    assert action_node.immediate_cost == 0.0

    # Test with ActionNode - None (should not update immediate_cost)
    action_node._immediate_cost = 10.0  # Set private attribute directly to avoid setter
    action_node.immediate_reward = None
    assert action_node.immediate_reward is None
    assert action_node.immediate_cost == 10.0  # Should remain unchanged

    # Test with BeliefNode - positive reward
    belief_node = BeliefNode(test_belief, children=())
    belief_node.immediate_reward = 7.5
    assert belief_node.immediate_reward == 7.5
    assert belief_node.immediate_cost == -7.5

    # Test with BeliefNode - negative reward
    belief_node.immediate_reward = -2.0
    assert belief_node.immediate_reward == -2.0
    assert belief_node.immediate_cost == 2.0

    # Test with BeliefNode - zero reward
    belief_node.immediate_reward = 0.0
    assert belief_node.immediate_reward == 0.0
    assert belief_node.immediate_cost == 0.0

    # Test with BeliefNode - None (should not update immediate_cost)
    belief_node._immediate_cost = 15.0  # Set private attribute directly to avoid setter
    belief_node.immediate_reward = None
    assert belief_node.immediate_reward is None
    assert belief_node.immediate_cost == 15.0  # Should remain unchanged
