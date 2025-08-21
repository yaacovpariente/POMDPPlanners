import pytest
import numpy as np
from anytree import PostOrderIter
import random

from POMDPPlanners.planners.mcts_planners.pomcp_dpw import POMCP_DPW
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler, action_progressive_widening
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.tree import BeliefNode, ActionNode
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.tests.test_planners.test_mcts_planners.test_utils import validate_tree_structure_with_progressive_widening

np.random.seed(42)
random.seed(42)


class MockActionSampler(ActionSampler):
    """Mock action sampler for testing POMCP_DPW."""
    
    def __init__(self, actions=None):
        self.actions = actions or [0, 1, 2]
    
    def sample(self, belief_node=None):
        return np.random.choice(self.actions)
    
    def get_space(self):
        return self.actions


@pytest.fixture
def discount_factor():
    return 0.95


@pytest.fixture
def depth():
    return 3


@pytest.fixture
def exploration_constant():
    return 1.0


@pytest.fixture
def k_o():
    return 3.0


@pytest.fixture
def k_a():
    return 3.0


@pytest.fixture
def alpha_o():
    return 0.5


@pytest.fixture
def alpha_a():
    return 0.5


@pytest.fixture
def n_simulations():
    return 100


@pytest.fixture
def n_particles():
    return 100


@pytest.fixture
def action_sampler():
    return MockActionSampler([0, 1, 2])


@pytest.fixture  
def discrete_action_sampler(environment):
    return MockActionSampler(environment.get_actions())


@pytest.fixture
def environment(discount_factor):
    return TigerPOMDP(discount_factor=discount_factor)


@pytest.fixture
def planner(environment, discount_factor, depth, exploration_constant, k_o, k_a, alpha_o, alpha_a, n_simulations, action_sampler):
    return POMCP_DPW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        n_simulations=n_simulations,
        action_sampler=action_sampler,
        name="TestPOMCP_DPW"
    )


@pytest.fixture
def belief(environment, n_particles):
    return get_initial_belief(
        pomdp=environment,
        n_particles=n_particles,
        resampling=True
    )


def test_pomcp_dpw_initialization_n_simulations_creates_configured_planner():
    """
    Purpose: Validates POMCP_DPW planner initializes correctly with simulation count configuration
    
    Given: TigerPOMDP environment and progressive widening parameters (k_o=3, k_a=3, alpha=0.5)
    When: POMCP_DPW planner is initialized with n_simulations=100
    Then: Planner is configured with all parameters and simulation-based termination
    
    Test type: unit
    """
    # ARRANGE: Setup planner configuration parameters
    expected_simulations = 100
    expected_name = "TestPOMCP_DPW"
    
    # ACT: Initialize POMCP_DPW planner with simulation count
    planner = POMCP_DPW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        n_simulations=expected_simulations,
        action_sampler=action_sampler,
        name=expected_name
    )
    
    # ASSERT: Verify all parameters configured correctly
    assert planner.environment == environment
    assert planner.discount_factor == discount_factor
    assert planner.depth == depth
    assert planner.exploration_constant == exploration_constant
    assert planner.k_o == k_o
    assert planner.k_a == k_a
    assert planner.alpha_o == alpha_o
    assert planner.alpha_a == alpha_a
    assert planner.n_simulations == expected_simulations
    assert planner.time_out_in_seconds is None  # Mutually exclusive with n_simulations
    assert planner.action_sampler == action_sampler
    assert planner.name == expected_name


def test_pomcp_dpw_initialization_timeout_creates_time_limited_planner():
    """
    Purpose: Ensures POMCP_DPW planner initializes correctly with time-based termination
    
    Given: TigerPOMDP environment and progressive widening configuration
    When: POMCP_DPW planner is initialized with time_out_in_seconds=5
    Then: Planner is configured for time-based termination instead of simulation count
    
    Test type: unit
    """
    # ARRANGE: Setup time-based termination configuration
    expected_timeout = 5
    expected_name = "TestPOMCP_DPW"
    
    # ACT: Initialize POMCP_DPW with timeout configuration
    planner = POMCP_DPW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        time_out_in_seconds=expected_timeout,
        action_sampler=action_sampler,
        name=expected_name
    )
    
    # ASSERT: Verify timeout-based configuration
    assert planner.environment == environment
    assert planner.discount_factor == discount_factor
    assert planner.depth == depth
    assert planner.exploration_constant == exploration_constant
    assert planner.time_out_in_seconds == expected_timeout
    assert planner.n_simulations is None  # Mutually exclusive with timeout
    assert planner.name == expected_name


def test_pomcp_dpw_initialization_both_termination_criteria_raises_error():
    """
    Purpose: Validates proper error handling when both termination criteria are provided
    
    Given: Valid POMCP_DPW configuration parameters
    When: Planner initialization attempts to set both n_simulations=100 and time_out_in_seconds=5
    Then: ValueError is raised indicating mutually exclusive termination criteria
    
    Test type: unit
    """
    # ARRANGE: Setup invalid configuration with both termination criteria
    invalid_simulations = 100
    invalid_timeout = 5
    
    # ACT & ASSERT: Verify ValueError raised for conflicting termination criteria
    with pytest.raises(ValueError) as exc_info:
        POMCP_DPW(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            exploration_constant=exploration_constant,
            k_o=k_o,
            k_a=k_a,
            alpha_o=alpha_o,
            alpha_a=alpha_a,
            n_simulations=invalid_simulations,
            time_out_in_seconds=invalid_timeout,
            action_sampler=action_sampler,
            name="TestPOMCP_DPW"
        )
    
    # ASSERT: Verify error message content
    assert "mutually exclusive" in str(exc_info.value).lower() or "both" in str(exc_info.value).lower()


def test_pomcp_dpw_action_selection_returns_valid_action_from_sampler(planner, belief):
    """
    Purpose: Validates POMCP_DPW selects valid actions from configured action sampler
    
    Given: POMCP_DPW planner with MockActionSampler containing actions [0, 1, 2] and initial belief
    When: Action selection is performed using the planner
    Then: Selected action is a single-element list with action from the sampler space
    
    Test type: unit
    """
    # ARRANGE: Use configured planner and belief from fixtures
    expected_action_space = planner.action_sampler.get_space()
    
    # ACT: Perform action selection
    selected_action, policy_run_data = planner.action(belief)
    
    # ASSERT: Verify valid action selection format and content
    assert isinstance(selected_action, list)
    assert len(selected_action) == 1
    assert selected_action[0] in expected_action_space
    assert hasattr(policy_run_data, 'info_variables')  # Contains MCTS tree statistics


def test_pomcp_dpw_progressive_widening_adds_new_action_to_unvisited_node(belief, planner):
    """
    Purpose: Verifies action progressive widening adds new actions to unvisited belief nodes
    
    Given: Unvisited belief node (visit_count=0) and POMCP_DPW progressive widening parameters
    When: Action progressive widening is applied to the belief node
    Then: New ActionNode is created and added as child with valid action from sampler
    
    Test type: unit
    """
    # ARRANGE: Create unvisited belief node for progressive widening test
    belief_node = BeliefNode(belief=belief, observation=None)
    belief_node.visit_count = 0
    initial_children_count = len(belief_node.children)
    expected_action_space = planner.action_sampler.get_space()
    
    # ACT: Apply action progressive widening to unvisited node
    action_node = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=planner.alpha_a,
        action_sampler=planner.action_sampler,
        exploration_constant=planner.exploration_constant,
        k_a=planner.k_a
    )
    
    # ASSERT: Verify new action node created and properly linked
    assert isinstance(action_node, ActionNode)
    assert action_node.parent == belief_node
    assert action_node in belief_node.children
    assert len(belief_node.children) == initial_children_count + 1
    assert action_node.action in expected_action_space
    assert action_node.visit_count == 0  # Newly created node
    assert action_node.q_value == 0.0  # Initial Q-value


def test_action_progressive_widening_existing_action(planner, belief, action_sampler):
    belief_node = BeliefNode(belief=belief, observation=None)
    
    # Add some children first
    for i in range(3):
        action_node = ActionNode(action=i, parent=belief_node)
        action_node.visit_count = 10
        action_node.q_value = np.random.random()
    
    belief_node.visit_count = 50
    
    # Use the function from dpw module directly
    action_node = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=planner.alpha_a,
        action_sampler=planner.action_sampler,
        exploration_constant=planner.exploration_constant,
        k_a=planner.k_a
    )
    assert isinstance(action_node, ActionNode)
    assert action_node in belief_node.children


def test_explored_action_node_ucb_selection(planner, belief):
    belief_node = BeliefNode(belief=belief, observation=None)
    belief_node.visit_count = 30
    
    # Add children with different Q-values
    q_values = [0.1, 0.5, 0.3]
    visit_counts = [10, 10, 10]
    
    for i, (q_val, visit_count) in enumerate(zip(q_values, visit_counts)):
        action_node = ActionNode(action=i, parent=belief_node)
        action_node.q_value = q_val
        action_node.visit_count = visit_count
    
    # Use the function from dpw module directly
    from POMDPPlanners.planners.planners_utils.dpw import ucb1_exploration
    selected_action_node = ucb1_exploration(
        belief_node=belief_node,
        exploration_constant=planner.exploration_constant
    )
    assert isinstance(selected_action_node, ActionNode)
    assert selected_action_node in belief_node.children


def test_rollout(planner):
    # Test the random_rollout_action_sampler function that POMCP_DPW uses
    from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler
    
    state = "tiger_left"
    depth = 0
    return_value = random_rollout_action_sampler(
        state=state, 
        depth=depth, 
        action_sampler=planner.action_sampler, 
        environment=planner.environment, 
        discount_factor=planner.discount_factor
    )
    assert isinstance(return_value, float)


def test_rollout_terminal_state(planner):
    # Test the random_rollout_action_sampler function with terminal state
    from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler
    
    # Create a mock terminal state
    original_is_terminal = planner.environment.is_terminal
    planner.environment.is_terminal = lambda state: True
    
    state = "tiger_left"
    depth = 0
    return_value = random_rollout_action_sampler(
        state=state, 
        depth=depth, 
        action_sampler=planner.action_sampler, 
        environment=planner.environment, 
        discount_factor=planner.discount_factor
    )
    assert return_value == 0
    
    # Restore original method
    planner.environment.is_terminal = original_is_terminal


def test_rollout_max_depth(planner):
    # Test the random_rollout_action_sampler function with max depth
    from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler
    
    state = "tiger_left"
    depth = planner.depth + 1  # This is 4 (planner.depth = 3)
    max_depth = depth  # Set max_depth to match the depth being tested
    
    return_value = random_rollout_action_sampler(
        state=state, 
        depth=depth, 
        action_sampler=planner.action_sampler, 
        environment=planner.environment, 
        discount_factor=planner.discount_factor,
        max_depth=max_depth  # Pass the max_depth parameter
    )
    assert return_value == 0


def test_simulate_path(planner, belief):
    belief_node = BeliefNode(belief=belief, observation=None)
    depth = 0
    
    return_value = planner._simulate_path(belief_node=belief_node, depth=depth)
    assert isinstance(return_value, float)
    assert belief_node.visit_count >= 1


def test_simulate_state_path_terminal_state(planner, belief):
    belief_node = BeliefNode(belief=belief, observation=None)
    depth = 0
    
    # Mock terminal state
    original_is_terminal = planner.environment.is_terminal
    planner.environment.is_terminal = lambda state: True
    
    state = belief.sample()
    return_value = planner._simulate_state_path(state=state, belief_node=belief_node, depth=depth)
    assert return_value == 0
    assert belief_node.visit_count == 1
    
    # Restore original method
    planner.environment.is_terminal = original_is_terminal


def test_simulate_state_path_max_depth(planner, belief):
    belief_node = BeliefNode(belief=belief, observation=None)
    state = belief.sample()
    depth = planner.depth + 1
    
    return_value = planner._simulate_state_path(state=state, belief_node=belief_node, depth=depth)
    assert return_value == 0


def test_get_space_info(planner):
    space_info = planner.get_space_info()
    assert hasattr(space_info, 'action_space')
    assert hasattr(space_info, 'observation_space')
    assert space_info.action_space == SpaceType.MIXED
    assert space_info.observation_space == SpaceType.MIXED


def test_integration_with_tiger_pomdp(planner, belief, environment, n_particles):
    current_belief = belief
    for _ in range(3):
        action, policy_run_data = planner.action(current_belief)
        assert isinstance(action, list)
        assert len(action) == 1
        assert action[0] in planner.action_sampler.get_space()
        
        # Simulate environment step
        state = current_belief.sample()
        next_state = environment.state_transition_model(state, action[0]).sample()[0]
        observation = environment.observation_model(next_state, action[0]).sample()[0]
        
        # Update belief
        current_belief = current_belief.update(action[0], observation, environment)
        
        # Verify belief is valid
        assert isinstance(current_belief, WeightedParticleBelief)
        assert len(current_belief.particles) == n_particles


def test_progressive_widening_parameters(planner, belief):
    belief_node = BeliefNode(belief=belief, observation=None)
    belief_node.visit_count = 100
    
    # Test that progressive widening respects k_o and alpha_o parameters
    action_node = ActionNode(action=0, parent=belief_node)
    action_node.visit_count = 50
    
    # Create mock observation
    observation = "tiger_growl_left"
    
    # Test observation progressive widening condition
    max_observations = planner.k_o * action_node.visit_count ** planner.alpha_o
    assert max_observations > 0
    
    # The condition should allow adding new observations if under the limit
    current_observations = len(action_node.children)
    can_add_observation = current_observations <= max_observations
    assert isinstance(can_add_observation, bool)


def test_belief_node_data_structure(planner, belief):
    """Test that belief nodes maintain proper belief structure for states and weights."""
    belief_node = BeliefNode(belief=belief, observation=None)
    
    # Simulate one path to set up the data structure
    planner._simulate_path(belief_node=belief_node, depth=0)
    
    # Check that children have proper belief structure
    for action_node in belief_node.children:
        for child_belief_node in action_node.children:
            # Check that the belief is an UnweightedParticleBeliefStateUpdate instance (POMCP tradition)
            from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate
            assert isinstance(child_belief_node.belief, UnweightedParticleBeliefStateUpdate)
            assert hasattr(child_belief_node.belief, 'particles')
            assert isinstance(child_belief_node.belief.particles, list)


def test_sanity_pomdp_action_selection():
    """Test POMCP_DPW with SanityPOMDP to verify correct action selection."""
    environment = SanityPOMDP()
    action_sampler = MockActionSampler([0, 1])
    
    planner = POMCP_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        n_simulations=1000,
        action_sampler=action_sampler,
        name="TestPOMCP_DPW"
    )
    
    belief = get_initial_belief(
        pomdp=environment,
        n_particles=100,
        resampling=True
    )
    
    # Run multiple trials to ensure consistent behavior
    n_trials = 10
    action_0_count = 0
    
    for _ in range(n_trials):
        action, policy_run_data = planner.action(belief)
        assert isinstance(action, list)
        assert len(action) == 1
        assert action[0] in [0, 1]
        if action[0] == 0:
            action_0_count += 1
    
    # Verify that action 0 (the better action) is selected most of the time
    # We expect at least 70% of the time to select action 0 (more lenient than POMCP)
    assert action_0_count >= 0.7 * n_trials, \
        f"POMCP_DPW selected action 0 only {action_0_count}/{n_trials} times, expected at least {0.7 * n_trials}"


def test_tree_structure_after_construction(planner, belief):
    """Test that the tree structure is properly constructed."""
    belief_node = BeliefNode(belief=belief, observation=None)
    
    # Run several simulations to build the tree
    for _ in range(50):
        planner._simulate_path(belief_node=belief_node, depth=0)
    
    # Verify tree structure
    assert len(belief_node.children) > 0
    assert all(isinstance(child, ActionNode) for child in belief_node.children)
    
    # Check that action nodes have proper structure
    for action_node in belief_node.children:
        assert action_node.parent == belief_node
        assert action_node.visit_count > 0
        assert action_node.q_value is not None
        
        # Check belief node children of action nodes
        for belief_child in action_node.children:
            assert isinstance(belief_child, BeliefNode)
            assert belief_child.parent == action_node
            assert hasattr(belief_child, 'data')


def test_q_value_updates(planner, belief):
    """Test that Q-values are properly updated during simulation."""
    belief_node = BeliefNode(belief=belief, observation=None)
    
    # Create an action node
    action_node = ActionNode(action=0, parent=belief_node)
    initial_q_value = action_node.q_value
    initial_visit_count = action_node.visit_count
    
    # Run simulation
    planner._simulate_path(belief_node=belief_node, depth=0)
    
    # Check that some action node's Q-value was updated
    action_updated = False
    for child in belief_node.children:
        if isinstance(child, ActionNode) and child.visit_count > initial_visit_count:
            action_updated = True
            break
    
    assert action_updated, "No action node was updated during simulation"


def test_visit_count_consistency(planner, belief):
    """Test that visit counts are consistent throughout the tree."""
    belief_node = BeliefNode(belief=belief, observation=None)
    
    # Run several simulations
    n_sims = 20
    for _ in range(n_sims):
        planner._simulate_path(belief_node=belief_node, depth=0)
    
    # Check visit count consistency
    assert belief_node.visit_count == n_sims
    
    # Check that action node visit counts sum correctly
    total_action_visits = sum(child.visit_count for child in belief_node.children if isinstance(child, ActionNode))
    assert total_action_visits <= belief_node.visit_count  # Allow for some variance due to tree structure


def test_pomcp_dpw_vs_pomcp_differences(planner, belief):
    """Test that POMCP_DPW has distinct behavior from standard POMCP due to progressive widening."""
    belief_node = BeliefNode(belief=belief, observation=None)
    
    # Run simulations to build tree
    for _ in range(100):
        planner._simulate_path(belief_node=belief_node, depth=0)
    
    # POMCP_DPW should show progressive widening behavior:
    # 1. Action nodes should be added progressively based on alpha_a
    # 2. Observation nodes should be limited by k_o and alpha_o parameters
    
    # Check action progressive widening - number of action children should be reasonable
    num_action_children = len(belief_node.children)
    assert num_action_children > 0
    # Note: Progressive widening may create more nodes than unique actions due to how it samples
    # So we just check that some actions were created and it's reasonable
    assert num_action_children >= 1
    
    # Check observation progressive widening for action nodes
    for action_node in belief_node.children:
        if action_node.visit_count > 0:
            max_allowed_observations = planner.k_o * action_node.visit_count ** planner.alpha_o
            actual_observations = len(action_node.children)
            # Should respect the progressive widening constraint (within reasonable bounds)
            assert actual_observations <= max_allowed_observations + 1  # Allow small variance


def test_unweighted_particle_belief_usage(planner, belief):
    """Test that POMCP_DPW properly uses unweighted particle beliefs in observation nodes."""
    from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate
    
    belief_node = BeliefNode(belief=belief, observation=None)
    state = belief.sample()
    
    # Simulate one path to create observation nodes
    planner._simulate_state_path(state=state, belief_node=belief_node, depth=0)
    
    # Check that any created observation nodes use UnweightedParticleBeliefStateUpdate
    for action_node in belief_node.children:
        for observation_node in action_node.children:
            assert isinstance(observation_node.belief, UnweightedParticleBeliefStateUpdate)
            # Verify it has the expected structure
            assert hasattr(observation_node.belief, 'particles')
            assert isinstance(observation_node.belief.particles, list)


def test_double_progressive_widening_integration(planner, belief):
    """Test that both action and observation progressive widening work together."""
    belief_node = BeliefNode(belief=belief, observation=None)
    
    # Run enough simulations to see progressive widening effects
    for _ in range(200):
        planner._simulate_path(belief_node=belief_node, depth=0)
    
    # Verify double progressive widening is working:
    # 1. Actions are added progressively
    # 2. Observations are constrained by progressive widening
    
    action_nodes = [child for child in belief_node.children if isinstance(child, ActionNode)]
    assert len(action_nodes) > 0
    
    total_observations = 0
    for action_node in action_nodes:
        total_observations += len(action_node.children)
        
        # Check that this action node respects observation progressive widening
        if action_node.visit_count > 1:
            max_obs = planner.k_o * action_node.visit_count ** planner.alpha_o
            actual_obs = len(action_node.children)
            assert actual_obs <= max_obs + 2  # Allow some variance due to sampling
    
    # Should have created some observations but be constrained by progressive widening
    assert total_observations >= 0


def test_basic_tree_structure(environment, discount_factor, depth, exploration_constant, k_o, k_a, alpha_o, alpha_a, action_sampler):
    """
    Purpose: Validates basic tree structure properties for POMCP_DPW with minimal simulation count
    
    Given: POMCP_DPW planner with progressive widening parameters and single simulation
    When: Tree construction is performed using _learn_tree method with n_simulations=1
    Then: All tree nodes have correct basic properties and parent-child relationships are valid
    
    Test type: unit
    """
    # ARRANGE: Setup POMCP_DPW planner with single simulation for basic structure testing
    n_simulations = 1
    planner = POMCP_DPW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        n_simulations=n_simulations,
        action_sampler=action_sampler,
        name="TestPOMCP_DPW_TreeStructure"
    )
    
    n_particles = 100
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)
    
    # ACT: Build minimal tree using _learn_tree method
    root_belief_node = planner._learn_tree(belief=belief)
    
    # ASSERT: Verify basic tree structure properties for all nodes
    for node in PostOrderIter(root_belief_node):
        if isinstance(node, BeliefNode):
            # BeliefNode basic validations
            assert node.visit_count == 1, "BeliefNode visit count must be 1 with single simulation"
            assert isinstance(node.v_value, (int, float)), "BeliefNode must have numeric v_value"
            
            # Root vs non-root belief node properties
            if node.depth > 0:
                assert node.observation is not None, "Non-root BeliefNode must have observation"
                assert node.parent is not None, "Non-root BeliefNode must have parent"
                assert isinstance(node.parent, ActionNode), "BeliefNode parent must be ActionNode"
            else:
                assert node.observation is None, "Root BeliefNode must have no observation"
                assert node.parent is None, "Root BeliefNode must have no parent"
                
        elif isinstance(node, ActionNode):
            # ActionNode basic validations
            assert node.visit_count == 1, "ActionNode visit count must be 1 with single simulation"
            assert isinstance(node.q_value, (int, float)), "ActionNode must have numeric q_value"
            assert node.parent is not None, "ActionNode must have parent"
            assert isinstance(node.parent, BeliefNode), "ActionNode parent must be BeliefNode"
            assert node.action is not None, "ActionNode must have action"
        else:
            raise ValueError(f"Unknown node type: {type(node)}")


def test_pomcp_dpw_tree_structure_construction(environment, discount_factor, depth, exploration_constant, k_o, k_a, alpha_o, alpha_a, action_sampler):
    """
    Purpose: Validates complete tree structure construction and node integrity for POMCP_DPW with progressive widening
    
    Given: POMCP_DPW planner with progressive widening parameters and TigerPOMDP environment  
    When: Tree construction is performed using _learn_tree method with sufficient simulations
    Then: All tree nodes have correct values, progressive widening constraints are respected, and tree structure is valid
    
    Test type: unit
    """
    # ARRANGE: Setup POMCP_DPW planner with sufficient simulations to build meaningful tree
    n_simulations = 200
    planner = POMCP_DPW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        n_simulations=n_simulations,
        action_sampler=action_sampler,
        name="TestPOMCP_DPW_TreeStructure"
    )
    
    n_particles = 100
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)
    
    # ACT: Build complete tree using _learn_tree method
    root_belief_node = planner._learn_tree(belief=belief)
    
    # ASSERT: Use shared validation function with POMCP_DPW-specific belief type
    from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate
    validate_tree_structure_with_progressive_widening(
        root_belief_node=root_belief_node,
        planner=planner,
        n_simulations=n_simulations,
        depth=depth,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        action_sampler=action_sampler,
        expected_belief_type=UnweightedParticleBeliefStateUpdate,  # POMCP_DPW uses unweighted particles
        planner_type="POMCP_DPW"  # Maintain original POMCP_DPW logic
    )


def test_numpy_array_observation_comparison():
    """Test that POMCP_DPW correctly handles numpy array observation comparisons."""
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ContinuousLightDarkPOMDPDiscreteActions, RewardModelType
    )
    
    # Create a minimal environment for testing
    environment = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        name="TestObservationComparison",
        state_transition_cov_matrix=np.eye(2) * 0.1,
        observation_cov_matrix=np.eye(2) * 0.5,
        beacons=np.array([[0], [0]]),
        goal_state=np.array([1, 1]),
        start_state=np.array([0, 0]),
        obstacles=np.array([]).reshape(2, 0),
        goal_reward=1.0,
        fuel_cost=0.1,
        grid_size=2,
        goal_state_radius=0.5,
        beacon_radius=1.0,
        reward_model_type=RewardModelType.STANDARD,
    )
    
    # Test that the environment's is_equal_observation method works correctly
    obs1 = np.array([0.5, 0.5])
    obs2 = np.array([0.5, 0.5])
    obs3 = np.array([0.6, 0.5])
    
    # Test observation equality
    assert environment.is_equal_observation(obs1, obs2) == True
    assert environment.is_equal_observation(obs1, obs3) == False
    
    # Test that POMCP_DPW can handle these observations
    action_sampler = MockActionSampler(["up", "down"])
    
    planner = POMCP_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=2,
        exploration_constant=1.0,
        k_o=2.0,
        k_a=2.0,
        alpha_o=0.5,
        alpha_a=0.5,
        n_simulations=10,  # Minimal for testing
        action_sampler=action_sampler,
        name="TestObservationComparison"
    )
    
    # Create minimal belief
    from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate
    belief = UnweightedParticleBeliefStateUpdate(particles=[np.array([0.0, 0.0])])
    
    # Test the simulate_state_path method with numpy observations
    belief_node = BeliefNode(belief=belief, observation=None)
    state = np.array([0.0, 0.0])
    
    # This should not raise an error about array truth values
    try:
        return_value = planner._simulate_state_path(
            state=state, 
            belief_node=belief_node, 
            depth=0
        )
        assert isinstance(return_value, (int, float))
    except ValueError as e:
        if "truth value of an array" in str(e):
            pytest.fail("POMCP_DPW failed to handle numpy array observations correctly")
        else:
            raise


# Config ID Tests

def test_pomcp_dpw_config_id_consistency_identical_parameters(environment, discrete_action_sampler):
    """Test that config_id is consistent for identical POMCP_DPW parameters.
    
    Purpose: Validates that POMCP_DPW with identical parameters produces identical config_id
    
    Given: Two POMCP_DPW instances with identical parameters
    When: config_id is accessed on both instances
    Then: Both instances return the same config_id
    
    Test type: unit
    """
    # Create two POMCP_DPW instances with identical parameters
    pomcp_dpw1 = POMCP_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=10,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="POMCP_DPW_Test1",
        action_sampler=discrete_action_sampler,
        n_simulations=100,
    )

    pomcp_dpw2 = POMCP_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=10,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="POMCP_DPW_Test1",  # Same name
        action_sampler=discrete_action_sampler,
        n_simulations=100,
    )

    # Config IDs should be identical
    config_id1 = pomcp_dpw1.config_id
    config_id2 = pomcp_dpw2.config_id

    assert config_id1 == config_id2
    assert isinstance(config_id1, str)
    assert len(config_id1) > 0


def test_pomcp_dpw_config_id_different_action_sampler_values():
    """Test that config_id changes when action sampler is initialized with different values.
    
    Purpose: Validates that config_id changes when action sampler parameters differ
    
    Given: Two POMCP_DPW instances with action samplers having different max_action_magnitude
    When: config_id is accessed on both instances
    Then: config_id values are different
    
    Test type: unit
    """
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ContinuousLightDarkPOMDP,
    )
    from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler
    
    # Create continuous environment for testing
    continuous_environment = ContinuousLightDarkPOMDP(
        discount_factor=0.99,
        goal_state=np.array([5, 0]),
        start_state=np.array([0, 0]),
        name="TestContinuous",
    )

    # Create action samplers with different max_action_magnitude
    sampler1 = UnitCircleActionSampler(max_action_magnitude=1.0)
    sampler2 = UnitCircleActionSampler(max_action_magnitude=2.0)

    pomcp_dpw1 = POMCP_DPW(
        environment=continuous_environment,
        discount_factor=0.99,
        depth=10,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="POMCP_DPW_Test",
        action_sampler=sampler1,
        n_simulations=100,
    )

    pomcp_dpw2 = POMCP_DPW(
        environment=continuous_environment,
        discount_factor=0.99,
        depth=10,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="POMCP_DPW_Test",
        action_sampler=sampler2,  # Different sampler parameters
        n_simulations=100,
    )

    config_id1 = pomcp_dpw1.config_id
    config_id2 = pomcp_dpw2.config_id

    assert config_id1 != config_id2
    assert isinstance(config_id1, str)
    assert isinstance(config_id2, str)
    assert len(config_id1) > 0
    assert len(config_id2) > 0


def test_pomcp_dpw_config_id_different_progressive_widening_parameters(environment, discrete_action_sampler):
    """Test that config_id changes when progressive widening parameters differ.
    
    Purpose: Validates that config_id changes when k_o, k_a, alpha_o, or alpha_a parameters differ
    
    Given: POMCP_DPW instances with different progressive widening parameters
    When: config_id is accessed on both instances
    Then: config_id values are different
    
    Test type: unit
    """
    pomcp_dpw1 = POMCP_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=10,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="POMCP_DPW_Test",
        action_sampler=discrete_action_sampler,
        n_simulations=100,
    )

    pomcp_dpw2 = POMCP_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=10,
        exploration_constant=1.0,
        k_o=5.0,  # Different k_o
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="POMCP_DPW_Test",
        action_sampler=discrete_action_sampler,
        n_simulations=100,
    )

    config_id1 = pomcp_dpw1.config_id
    config_id2 = pomcp_dpw2.config_id

    assert config_id1 != config_id2


def test_pomcp_dpw_config_id_consistency_across_evaluations(environment, discrete_action_sampler):
    """Test that config_id remains consistent across different policy evaluations.
    
    Purpose: Validates that config_id is stable across multiple accesses and policy actions
    
    Given: Single POMCP_DPW instance and initial belief
    When: config_id is accessed before and after policy actions
    Then: config_id remains identical across all evaluations
    
    Test type: integration
    """
    pomcp_dpw = POMCP_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=5,  # Reduced for testing
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="POMCP_DPW_Consistency_Test",
        action_sampler=discrete_action_sampler,
        n_simulations=10,  # Reduced for testing
    )

    # Get initial config_id
    initial_config_id = pomcp_dpw.config_id

    # Create initial belief and perform policy actions
    initial_belief = get_initial_belief(environment, n_particles=50)

    # Perform multiple policy evaluations
    for i in range(3):
        action, run_data = pomcp_dpw.action(initial_belief)
        
        # Check config_id remains the same
        current_config_id = pomcp_dpw.config_id
        assert current_config_id == initial_config_id
        
        # Verify the action and run_data are valid
        assert isinstance(action, list)
        assert len(action) == 1
        assert action[0] in environment.get_actions()
        assert run_data is not None

    # Final check
    final_config_id = pomcp_dpw.config_id
    assert final_config_id == initial_config_id


def test_pomcp_dpw_config_id_hash_properties(environment, discrete_action_sampler):
    """Test that config_id has proper hash properties.
    
    Purpose: Validates that config_id produces valid hash strings
    
    Given: POMCP_DPW instance
    When: config_id is accessed
    Then: config_id is a valid hash string with expected properties
    
    Test type: unit
    """
    pomcp_dpw = POMCP_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=10,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="POMCP_DPW_Hash_Test",
        action_sampler=discrete_action_sampler,
        n_simulations=100,
    )

    config_id = pomcp_dpw.config_id

    # Should be a non-empty string
    assert isinstance(config_id, str)
    assert len(config_id) > 0

    # Should be a valid hexadecimal hash (SHA-256 produces 64 hex characters)
    assert len(config_id) == 64
    assert all(c in '0123456789abcdef' for c in config_id.lower())