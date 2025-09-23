import random

import numpy as np
import pytest
from anytree import PostOrderIter

from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.policy import PolicyRunData
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planner import (
    StandardSparseSamplingDiscreteActionsPlanner,
)

np.random.seed(42)
random.seed(42)


@pytest.fixture
def tiger_pomdp():
    return TigerPOMDP(discount_factor=0.95)


@pytest.fixture
def initial_belief(tiger_pomdp):
    # Create a uniform belief over states using particles
    states = list(tiger_pomdp.states)
    particles = states * 10  # Create 10 particles for each state
    log_weights = np.log(np.ones(len(particles)) / len(particles))  # Uniform log weights
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


@pytest.fixture
def planner(tiger_pomdp):
    return StandardSparseSamplingDiscreteActionsPlanner(
        environment=tiger_pomdp, branching_factor=2, depth=2
    )


def test_initialization(planner, tiger_pomdp):
    """Test that the planner initializes correctly

    Purpose: Validates proper initialization of

    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes

    Test type: unit
    """
    assert planner.environment == tiger_pomdp
    assert planner.branching_factor == 2
    assert planner.depth == 2


def test_action_selection(planner, initial_belief):
    """Test that action selection returns a valid action

    Purpose: Validates that StandardSparseSamplingDiscreteActionsPlanner returns valid TigerPOMDP actions through sparse sampling

    Given: StandardSparseSamplingDiscreteActionsPlanner with branching_factor=2, depth=2, TigerPOMDP environment, uniform initial belief
    When: action method performs sparse sampling tree construction and action selection
    Then: Returns list with single valid tiger action, PolicyRunData with no info_variables (as expected)

    Test type: unit
    """
    action, run_data = planner.action(initial_belief)
    assert isinstance(action, list)
    assert len(action) == 1
    assert action[0] in planner.environment.get_actions()
    assert isinstance(run_data, PolicyRunData)
    assert len(run_data.info_variables) == 0


def test_belief_tree_construction(planner, initial_belief):
    """Test that the belief tree is constructed correctly

    Purpose: Validates that sparse sampling belief tree construction creates proper structure with correct depth and branching

    Given: BeliefNode with uniform initial belief, StandardSparseSamplingPlanner with depth=2
    When: _build_tree constructs sparse sampling tree from root node at current_depth=0
    Then: Tree has correct root belief, no parent, >0 children, height=2*depth+1=5 for alternating belief-action levels

    Test type: unit
    """
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
    """Test that node statistics are updated correctly

    Purpose: Validates that sparse sampling node statistics are properly initialized and computed for all node types

    Given: BeliefNode with uniform initial belief, sparse sampling tree built with branching_factor=2, depth=2
    When: _build_tree creates tree structure and computes statistics for all nodes
    Then: Leaf nodes have immediate_cost/q_value/visit_count, ActionNodes have visit_count/q_value, BeliefNodes have v_value/visit_count

    Test type: unit
    """
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
    """Test that leaf node statistics are calculated correctly

    Purpose: Validates that sparse sampling leaf node statistics are computed correctly using rollout estimation

    Given: StandardSparseSamplingPlanner with uniform initial belief, learned belief tree with depth=2
    When: _learn_belief_tree constructs full tree and computes leaf node statistics
    Then: All leaf nodes are ActionNodes with visit_count=1 and q_value=immediate_cost (rollout result)

    Test type: unit
    """
    tree = planner._learn_belief_tree(initial_belief)

    for node in PostOrderIter(tree):
        if node.is_leaf:
            assert isinstance(node, ActionNode)
            assert node.visit_count == 1
            assert node.q_value == node.immediate_cost


def test_non_leaf_action_node_statistics(planner, initial_belief):
    """Test that non-leaf action node statistics are calculated correctly using fixed expected values

    Purpose: Validates that non-leaf ActionNode statistics are computed using immediate cost plus discounted future value

    Given: ActionNode with immediate_cost=1.0, two BeliefNode children with v_values [2.0, 4.0], discount_factor=0.95
    When: _update_non_leaf_action_node_statistics computes q_value from children
    Then: q_value = 1.0 + 0.95 * mean([2.0, 4.0]) = 3.85, visit_count > 0

    Test type: unit
    """
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
    """Test that belief node statistics are calculated correctly using fixed expected values

    Purpose: Validates that BeliefNode statistics are computed using minimum q_value and sum of visit counts from children

    Given: BeliefNode with two ActionNode children having q_values [3.0, 5.0] and visit_counts [2, 3]
    When: _update_belief_node_statistics computes v_value and visit_count from children
    Then: v_value = min([3.0, 5.0]) = 3.0, visit_count = sum([2, 3]) = 5

    Test type: unit
    """
    # Create a belief node
    belief_node = BeliefNode(belief=initial_belief)

    # Create two action nodes as children with known q_values
    action_node1 = ActionNode(action="listen", parent=belief_node, children=tuple(), data=None)
    action_node1.q_value = 3.0
    action_node1.visit_count = 2

    action_node2 = ActionNode(action="open-left", parent=belief_node, children=tuple(), data=None)
    action_node2.q_value = 5.0
    action_node2.visit_count = 3

    # Update belief node statistics
    planner._update_belief_node_statistics(belief_node)

    # Expected v_value should be min of children q_values (3.0)
    # Expected visit_count should be sum of children visit_counts (2 + 3 = 5)
    assert np.isclose(belief_node.v_value, 3.0)
    assert belief_node.visit_count == 5


def test_invalid_branching_factor():
    """Test that invalid branching factor raises an error

    Purpose: Validates that StandardSparseSamplingPlanner rejects invalid branching_factor parameters

    Given: TigerPOMDP environment, invalid branching_factor=0, valid depth=1
    When: StandardSparseSamplingDiscreteActionsPlanner constructor is called with branching_factor=0
    Then: ValueError is raised for non-positive branching factor parameter

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=0.95)
    with pytest.raises(ValueError):
        StandardSparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=0, depth=1)


def test_invalid_depth():
    """Test that invalid depth raises an error

    Purpose: Validates that StandardSparseSamplingPlanner rejects invalid depth parameters

    Given: TigerPOMDP environment, valid branching_factor=2, invalid depth=0
    When: StandardSparseSamplingDiscreteActionsPlanner constructor is called with depth=0
    Then: ValueError is raised for non-positive depth parameter

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=0.95)
    with pytest.raises(ValueError):
        StandardSparseSamplingDiscreteActionsPlanner(environment=env, branching_factor=2, depth=0)


def test_sanity_pomdp_action_selection():
    """Test that the sparse sampling planner correctly identifies the better action in SanityPOMDP

    Purpose: Validates that StandardSparseSamplingPlanner identifies optimal actions in SanityPOMDP with deterministic reward structure

    Given: SanityPOMDP environment with clear optimal action, sparse sampling planner with branching_factor=4, depth=3, 100 particles
    When: Multiple trials of action selection using sparse sampling tree search
    Then: Action 0 (better action) selected ≥70% of trials, demonstrating systematic planning advantage over random selection

    Test type: unit
    """
    # Create environment and planner with higher branching factor and depth for better accuracy
    environment = SanityPOMDP()
    planner = StandardSparseSamplingDiscreteActionsPlanner(
        environment=environment,
        branching_factor=4,  # Higher branching factor for better exploration
        depth=3,  # Deeper tree for better planning
    )

    # Get initial belief
    belief = get_initial_belief(pomdp=environment, n_particles=100, resampling=True)

    # Run multiple trials to ensure consistent behavior
    n_trials = 10
    action_0_count = 0

    for _ in range(n_trials):
        action, run_data = planner.action(belief)
        assert isinstance(action, list)
        assert len(action) == 1
        assert isinstance(run_data, PolicyRunData)
        assert len(run_data.info_variables) == 0
        if action[0] == 0:  # Count how many times action 0 is selected
            action_0_count += 1

    # Verify that action 0 (the better action) is selected most of the time
    # We expect at least 70% success rate since sparse sampling is more systematic than MCTS
    assert (
        action_0_count >= 0.7 * n_trials
    ), f"Sparse sampling planner selected action 0 only {action_0_count}/{n_trials} times, expected at least {0.7 * n_trials}"


# Config ID Tests


def test_sparse_sampling_config_id_consistency_identical_parameters(tiger_pomdp):
    """Test that config_id is consistent for identical StandardSparseSampling parameters.

    Purpose: Validates that StandardSparseSamplingPlanner with identical parameters produces identical config_id

    Given: Two StandardSparseSamplingPlanner instances with identical parameters
    When: config_id is accessed on both instances
    Then: Both instances return the same config_id

    Test type: unit
    """
    # Create two StandardSparseSamplingPlanner instances with identical parameters
    planner1 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=tiger_pomdp,
        branching_factor=5,
        depth=8,
        name="SparseSampling_Test1",
    )

    planner2 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=tiger_pomdp,
        branching_factor=5,
        depth=8,
        name="SparseSampling_Test1",  # Same name
    )

    # Config IDs should be identical
    config_id1 = planner1.config_id
    config_id2 = planner2.config_id

    assert config_id1 == config_id2
    assert isinstance(config_id1, str)
    assert len(config_id1) > 0


def test_sparse_sampling_config_id_different_branching_factor(tiger_pomdp):
    """Test that config_id changes when branching_factor parameter differs.

    Purpose: Validates that config_id changes when branching_factor parameter differs

    Given: Two StandardSparseSamplingPlanner instances with different branching_factor values
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
    planner1 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=tiger_pomdp,
        branching_factor=3,
        depth=8,
        name="SparseSampling_Test",
    )

    planner2 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=tiger_pomdp,
        branching_factor=6,  # Different branching factor
        depth=8,
        name="SparseSampling_Test",
    )

    config_id1 = planner1.config_id
    config_id2 = planner2.config_id

    assert config_id1 != config_id2


def test_sparse_sampling_config_id_different_depth(tiger_pomdp):
    """Test that config_id changes when depth parameter differs.

    Purpose: Validates that config_id changes when depth parameter differs

    Given: Two StandardSparseSamplingPlanner instances with different depth values
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
    planner1 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=tiger_pomdp,
        branching_factor=5,
        depth=5,
        name="SparseSampling_Test",
    )

    planner2 = StandardSparseSamplingDiscreteActionsPlanner(
        environment=tiger_pomdp,
        branching_factor=5,
        depth=12,  # Different depth
        name="SparseSampling_Test",
    )

    config_id1 = planner1.config_id
    config_id2 = planner2.config_id

    assert config_id1 != config_id2


def test_sparse_sampling_config_id_consistency_across_evaluations(tiger_pomdp):
    """Test that config_id remains consistent across different policy evaluations.

    Purpose: Validates that config_id is stable across multiple accesses and policy actions

    Given: Single StandardSparseSamplingPlanner instance and initial belief
    When: config_id is accessed before and after policy actions
    Then: config_id remains identical across all evaluations

    Test type: integration
    """
    planner = StandardSparseSamplingDiscreteActionsPlanner(
        environment=tiger_pomdp,
        branching_factor=3,  # Reduced for testing
        depth=3,  # Reduced for testing
        name="SparseSampling_Consistency_Test",
    )

    # Get initial config_id
    initial_config_id = planner.config_id

    # Create initial belief and perform policy actions
    initial_belief = get_initial_belief(tiger_pomdp, n_particles=50)

    # Perform multiple policy evaluations
    for i in range(3):
        action, run_data = planner.action(initial_belief)

        # Check config_id remains the same
        current_config_id = planner.config_id
        assert current_config_id == initial_config_id

        # Verify the action and run_data are valid
        assert isinstance(action, list)
        assert len(action) == 1
        assert action[0] in tiger_pomdp.get_actions()
        assert run_data is not None

    # Final check
    final_config_id = planner.config_id
    assert final_config_id == initial_config_id


def test_sparse_sampling_config_id_hash_properties(tiger_pomdp):
    """Test that config_id has proper hash properties.

    Purpose: Validates that config_id produces valid hash strings

    Given: StandardSparseSamplingPlanner instance
    When: config_id is accessed
    Then: config_id is a valid hash string with expected properties

    Test type: unit
    """
    planner = StandardSparseSamplingDiscreteActionsPlanner(
        environment=tiger_pomdp,
        branching_factor=5,
        depth=8,
        name="SparseSampling_Hash_Test",
    )

    config_id = planner.config_id

    # Should be a non-empty string
    assert isinstance(config_id, str)
    assert len(config_id) > 0

    # Should be a valid hexadecimal hash (SHA-256 produces 64 hex characters)
    assert len(config_id) == 64
    assert all(c in "0123456789abcdef" for c in config_id.lower())
