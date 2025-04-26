import pytest
import numpy as np
from anytree import PostOrderIter

from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.core.tree import BeliefNode, ActionNode
from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP


@pytest.fixture
def discount_factor():
    return 0.9


@pytest.fixture
def depth():
    return 3


@pytest.fixture
def c_ucb():
    return 1.0


@pytest.fixture
def beta_ucb():
    return 1.0


@pytest.fixture
def belief_child_num():
    return 2


@pytest.fixture
def n_simulations():
    return 100


@pytest.fixture
def n_particles():
    return 100


@pytest.fixture
def environment(discount_factor):
    return TigerPOMDP(discount_factor=discount_factor)


@pytest.fixture
def initial_belief(environment, n_particles):
    return get_initial_belief(
        pomdp=environment,
        n_particles=n_particles,
        resampling=True
    )


@pytest.fixture
def planner(environment, discount_factor, depth, c_ucb, beta_ucb, belief_child_num, n_simulations):
    return SparsePFT(
        environment=environment,
        discount_factor=discount_factor,
        gamma=discount_factor,
        depth=depth,
        c_ucb=c_ucb,
        beta_ucb=beta_ucb,
        belief_child_num=belief_child_num,
        n_simulations=n_simulations
    )


def test_initialization(planner, environment):
    """Test that the planner initializes correctly"""
    assert planner.environment == environment
    assert planner.discount_factor == 0.9
    assert planner.gamma == 0.9
    assert planner.depth == 3
    assert planner.c_ucb == 1.0
    assert planner.beta_ucb == 1.0
    assert planner.belief_child_num == 2
    assert planner.n_simulations == 100


def test_action_selection(planner, initial_belief):
    """Test that action selection returns a valid action"""
    action = planner.action(belief=initial_belief)
    assert action in planner.environment.get_actions()


def test_get_explored_action_node(planner):
    """Test that action node exploration works correctly"""
    # Create a belief node with children
    belief = WeightedParticleBelief(
        particles=["tiger_left", "tiger_right"],
        log_weights=np.array([-0.69314718, -0.69314718])  # log(0.5) for equal weights
    )
    belief_node = BeliefNode(
        belief=belief,
        observation=None,
        parent=None,
        children=tuple()
    )
    belief_node.visit_count = 1
    
    # Create action nodes with different Q-values
    for action in planner.environment.get_actions():
        action_node = ActionNode(
            action=action,
            parent=belief_node,
            children=tuple()
        )
        action_node.q_value = -1.0
        action_node.visit_count = 1
    
    # Set one action node to have a much lower Q-value
    last_action_node = belief_node.children[-1]
    last_action_node.q_value = 100.0
    
    # Test exploration
    selected_node = planner.get_explored_action_node(belief_node=belief_node)
    assert isinstance(selected_node, ActionNode)
    assert selected_node in belief_node.children
    assert selected_node.action in planner.environment.get_actions()
    # Should select the node with the lowest Q-value due to exploration term
    assert selected_node.q_value == 100.0  # Check Q-value instead of action


def test_sample_next_existing_belief(planner):
    """Test sampling from existing belief nodes"""
    # Create a belief node and action node
    belief = WeightedParticleBelief(
        particles=["tiger_left", "tiger_right"],
        log_weights=np.array([-0.69314718, -0.69314718])
    )
    belief_node = BeliefNode(
        belief=belief,
        observation=None,
        parent=None,
        children=tuple()
    )
    
    action_node = ActionNode(
        action="listen",
        parent=belief_node,
        children=tuple()
    )
    
    # Add some child belief nodes with proper visit counts
    for _ in range(2):
        child_belief = BeliefNode(
            belief=belief,
            observation="hear_left",
            parent=action_node,
            children=tuple()
        )
        child_belief.immediate_cost = -1.0
        child_belief.visit_count = 1  # Add visit count
    
    next_belief_node, immediate_reward = planner._sample_next_existing_belief(action_node=action_node)
    assert isinstance(next_belief_node, BeliefNode)
    assert next_belief_node in action_node.children
    assert immediate_reward == -next_belief_node.immediate_cost


def test_generate_belief(planner):
    """Test generating a new belief node"""
    # Create a belief node and action node
    belief = WeightedParticleBelief(
        particles=["tiger_left", "tiger_right"],
        log_weights=np.array([-0.69314718, -0.69314718])
    )
    belief_node = BeliefNode(
        belief=belief,
        observation=None,
        parent=None,
        children=tuple()
    )
    
    action_node = ActionNode(
        action="listen",
        parent=belief_node,
        children=tuple()
    )
    
    next_belief_node, immediate_reward = planner._generate_belief(action_node=action_node)
    assert isinstance(next_belief_node, BeliefNode)
    assert next_belief_node.parent == action_node
    assert next_belief_node.observation is not None
    assert next_belief_node.immediate_cost is not None
    assert immediate_reward == -next_belief_node.immediate_cost


def test_random_rollout(planner):
    """Test random rollout from a state"""
    state = "tiger_left"
    return_value = planner.random_rollout(state=state, depth=0)
    assert isinstance(return_value, float)
    # The Tiger POMDP has negative rewards, so the return value should be negative
    # We'll allow for some numerical error
    assert return_value < 0.1  # Allow for small positive values due to numerical error


def test_update_node_statistics(planner):
    """Test updating node statistics"""
    # Create a belief node and action node
    belief = WeightedParticleBelief(
        particles=["tiger_left", "tiger_right"],
        log_weights=np.array([-0.69314718, -0.69314718])
    )
    belief_node = BeliefNode(
        belief=belief,
        observation=None,
        parent=None,
        children=tuple()
    )
    belief_node.visit_count = 1
    
    action_node = ActionNode(
        action="listen",
        parent=belief_node,
        children=tuple()
    )
    action_node.visit_count = 1
    action_node.q_value = 0.0
    
    # Update statistics
    return_sample = -1.0
    planner.update_nodes(
        belief_node=belief_node,
        action_node=action_node,
        return_sample=return_sample
    )
    
    assert belief_node.visit_count == 2
    assert action_node.visit_count == 2
    assert action_node.q_value == -0.5  # (0.0 + -1.0) / 2
    assert belief_node.v_value is not None


def test_integration_with_tiger_pomdp(planner, initial_belief, environment, n_particles):
    """Test integration with Tiger POMDP environment"""
    current_belief = initial_belief
    for _ in range(5):
        # Create a belief node with children
        belief_node = BeliefNode(
            belief=current_belief,
            observation=None,
            parent=None,
            children=tuple()
        )
        
        # Add action nodes
        for action in planner.environment.get_actions():
            action_node = ActionNode(
                action=action,
                parent=belief_node,
                children=tuple()
            )
            action_node.q_value = -1.0
            action_node.visit_count = 1
        
        action = planner.action(current_belief)
        assert action in environment.get_actions()
        
        # Simulate environment step
        state = current_belief.sample()
        next_state = environment.state_transition_model(state=state, action=action).sample()
        next_observation = environment.observation_model(next_state=next_state, action=action).sample()
        
        # Update belief
        current_belief = current_belief.update(
            action=action,
            observation=next_observation,
            pomdp=environment
        )
        
        # Verify belief is valid
        assert isinstance(current_belief, WeightedParticleBelief)
        assert len(current_belief.particles) == n_particles


def test_tree_structure_construction(planner, initial_belief, environment):
    """Test that the tree structure is constructed correctly"""
    # Learn the tree
    tree = planner._learn_tree(belief=initial_belief)
    
    # Verify root node
    assert isinstance(tree, BeliefNode)
    assert tree.belief == initial_belief
    assert tree.observation is None
    assert tree.parent is None
    assert len(tree.children) == len(environment.get_actions())
    assert tree.visit_count > 0
    assert tree.v_value is not None
    
    n_actions = len(environment.get_actions())
    # Verify tree structure and node properties
    for node in PostOrderIter(tree):
        assert node.visit_count >= 0
        
        if isinstance(node, BeliefNode):
            assert node.belief is not None
            assert node.v_value is not None
            
            if node.height > 1 and node.depth > 0:
                assert node.observation is not None
                # In SparsePFT, each action node can have at most belief_child_num children
                assert len(node.children) <= n_actions
                n_children_visits = sum(child.visit_count for child in node.children)
                assert node.visit_count == n_children_visits + 1  # +1 for the rollout
        
        elif isinstance(node, ActionNode):
            assert node.action is not None
            assert node.q_value is not None
            if not node.is_leaf:
                assert node.visit_count == sum(child.visit_count for child in node.children)
                # In SparsePFT, each action node can have at most belief_child_num children
                assert len(node.children) <= planner.belief_child_num
    
    # Verify tree depth
    assert tree.height == 2 * planner.depth + 1  # Each level has both belief and action nodes
    
    # Verify that all actions are explored
    root_action_nodes = tree.children
    assert len(root_action_nodes) == len(environment.get_actions())
    assert all(isinstance(node, ActionNode) for node in root_action_nodes)
    assert all(node.visit_count > 0 for node in root_action_nodes)
