import pytest
import numpy as np
from anytree import PostOrderIter

from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.core.tree import BeliefNode, ActionNode
from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP


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
    """Test that the planner initializes correctly
    
    Purpose: Validates proper initialization of 
    
    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes
    
    Test type: unit
    """
    assert planner.environment == environment
    assert planner.discount_factor == 0.9
    assert planner.gamma == 0.9
    assert planner.depth == 3
    assert planner.c_ucb == 1.0
    assert planner.beta_ucb == 1.0
    assert planner.belief_child_num == 2
    assert planner.n_simulations == 100


def test_action_selection(planner, initial_belief):
    """Test that action selection returns a valid action
    
    Purpose: Validates action selection
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    action, policy_run_data = planner.action(belief=initial_belief)
    assert isinstance(action, list)
    assert len(action) == 1
    assert action[0] in planner.environment.get_actions()


def test_get_explored_action_node(planner):
    """Test that action node exploration works correctly
    
    Purpose: Validates get explored action node
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
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
    """Test sampling from existing belief nodes
    
    Purpose: Validates sampling behavior for  next existing belief
    
    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution
    
    Test type: unit
    """
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
    """Test generating a new belief node
    
    Purpose: Validates generate belief
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
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
    """Test random rollout from a state
    
    Purpose: Validates random rollout
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    state = "tiger_left"
    return_value = planner.random_rollout(state=state, depth=0)
    assert isinstance(return_value, float)
    # The Tiger POMDP has negative rewards, so the return value should be negative
    # We'll allow for some numerical error
    min_reward = -100
    max_reward = 10
    depth = 5
    
    assert return_value >= min_reward * depth
    assert return_value <= max_reward * depth


def test_update_node_statistics(planner):
    """Test updating node statistics
    
    Purpose: Validates update functionality for  node statistics
    
    Given: Initial object state and update parameters
    When: Update operation is performed
    Then: Object state is correctly modified
    
    Test type: unit
    """
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
    """Test integration with Tiger POMDP environment
    
    Purpose: Validates integration with tiger pomdp
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: integration
    """
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
        
        action, policy_run_data = planner.action(current_belief)
        assert isinstance(action, list)
        assert len(action) == 1
        assert action[0] in environment.get_actions()
        
        # Simulate environment step
        state = current_belief.sample()
        next_state = environment.state_transition_model(state=state, action=action[0]).sample()[0]
        next_observation = environment.observation_model(next_state=next_state, action=action[0]).sample()[0]
        
        # Update belief
        current_belief = current_belief.update(
            action=action[0],
            observation=next_observation,
            pomdp=environment
        )
        
        # Verify belief is valid
        assert isinstance(current_belief, WeightedParticleBelief)
        assert len(current_belief.particles) == n_particles


def test_tree_structure_construction(planner, initial_belief, environment):
    """Test that the tree structure is constructed correctly
    
    Purpose: Validates tree structure construction
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
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


def test_sanity_pomdp_action_selection():
    """Test that SparsePFT correctly identifies the better action in SanityPOMDP
    
    Purpose: Validates sanity pomdp action selection
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Create environment and planner with appropriate parameters
    environment = SanityPOMDP()
    planner = SparsePFT(
        environment=environment,
        discount_factor=0.95,
        gamma=0.95,
        depth=3,
        c_ucb=1.0,
        beta_ucb=1.0,
        belief_child_num=3,  # More belief children for better exploration
        n_simulations=1000  # More simulations for better accuracy
    )
    
    # Get initial belief
    belief = get_initial_belief(
        pomdp=environment,
        n_particles=10,
        resampling=True
    )
    
    # Run multiple trials to ensure consistent behavior
    n_trials = 10
    action_0_count = 0
    
    for _ in range(n_trials):
        action, policy_run_data = planner.action(belief)
        assert isinstance(action, list)
        assert len(action) == 1
        if action[0] == 0:  # Count how many times action 0 is selected
            action_0_count += 1
    
    # Verify that action 0 (the better action) is selected most of the time
    # We expect at least 70% success rate since SparsePFT combines MCTS with particle filtering
    assert action_0_count >= 0.7 * n_trials, \
        f"SparsePFT selected action 0 only {action_0_count}/{n_trials} times, expected at least {0.7 * n_trials}"


def test_sanity_pomdp_belief_children():
    """Test that SparsePFT generates appropriate belief children for SanityPOMDP
    
    Purpose: Validates sanity pomdp belief children
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    environment = SanityPOMDP()
    planner = SparsePFT(
        environment=environment,
        discount_factor=0.95,
        gamma=0.95,
        depth=3,
        c_ucb=1.0,
        beta_ucb=1.0,
        belief_child_num=3,
        n_simulations=10
    )
    
    # Get initial belief and create nodes
    belief = get_initial_belief(
        pomdp=environment,
        n_particles=100,
        resampling=True
    )
    
    belief_node = BeliefNode(belief=belief, observation=None)
    action_node = ActionNode(action=0, parent=belief_node)  # Test with the better action
    
    # Generate a few belief children
    for _ in range(3):
        next_belief_node, _ = planner._generate_belief(action_node=action_node)
        
        # Verify the generated belief node
        assert isinstance(next_belief_node, BeliefNode)
        assert next_belief_node.parent == action_node
        assert isinstance(next_belief_node.belief, WeightedParticleBelief)
        assert len(next_belief_node.belief.particles) == 100
        assert next_belief_node.observation in [0, 1]  # SanityPOMDP has binary observations
        assert next_belief_node.immediate_cost is not None
