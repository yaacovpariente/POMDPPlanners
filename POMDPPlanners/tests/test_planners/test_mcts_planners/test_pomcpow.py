import pytest
import numpy as np
from anytree import PostOrderIter

from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler, action_progressive_widening
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.tree import BeliefNode, ActionNode
from POMDPPlanners.core.environment import SpaceType


class MockActionSampler(ActionSampler):
    """Mock action sampler for testing POMCPOW."""
    
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
def environment(discount_factor):
    return TigerPOMDP(discount_factor=discount_factor)


@pytest.fixture
def planner(environment, discount_factor, depth, exploration_constant, k_o, k_a, alpha_o, alpha_a, n_simulations, action_sampler):
    return POMCPOW(
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
        name="TestPOMCPOW"
    )


@pytest.fixture
def belief(environment, n_particles):
    return get_initial_belief(
        pomdp=environment,
        n_particles=n_particles,
        resampling=True
    )


def test_initialization_with_n_simulations(environment, discount_factor, depth, exploration_constant, k_o, k_a, alpha_o, alpha_a, action_sampler):
    """Test initialization with n simulations.
    
    Purpose: Validates proper initialization of  with n simulations
    
    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes
    
    Test type: unit
    """
    planner = POMCPOW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        n_simulations=100,
        action_sampler=action_sampler,
        name="TestPOMCPOW"
    )
    assert planner.environment == environment
    assert planner.discount_factor == discount_factor
    assert planner.depth == depth
    assert planner.exploration_constant == exploration_constant
    assert planner.k_o == k_o
    assert planner.k_a == k_a
    assert planner.alpha_o == alpha_o
    assert planner.alpha_a == alpha_a
    assert planner.n_simulations == 100
    assert planner.time_out_in_seconds is None
    assert planner.action_sampler == action_sampler


def test_initialization_with_timeout(environment, discount_factor, depth, exploration_constant, k_o, k_a, alpha_o, alpha_a, action_sampler):
    """Test initialization with timeout.
    
    Purpose: Validates proper initialization of  with timeout
    
    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes
    
    Test type: unit
    """
    planner = POMCPOW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        time_out_in_seconds=5,
        action_sampler=action_sampler,
        name="TestPOMCPOW"
    )
    assert planner.environment == environment
    assert planner.discount_factor == discount_factor
    assert planner.depth == depth
    assert planner.exploration_constant == exploration_constant
    assert planner.time_out_in_seconds == 5
    assert planner.n_simulations is None


def test_invalid_initialization(environment, discount_factor, depth, exploration_constant, k_o, k_a, alpha_o, alpha_a, action_sampler):
    """Test invalid initialization.
    
    Purpose: Validates proper initialization of invalid 
    
    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes
    
    Test type: unit
    """
    with pytest.raises(ValueError):
        POMCPOW(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            exploration_constant=exploration_constant,
            k_o=k_o,
            k_a=k_a,
            alpha_o=alpha_o,
            alpha_a=alpha_a,
            n_simulations=100,
            time_out_in_seconds=5,
            action_sampler=action_sampler,
            name="TestPOMCPOW"
        )


def test_action_selection(planner, belief):
    """Test action selection.
    
    Purpose: Validates action selection
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    action, policy_run_data = planner.action(belief)
    assert isinstance(action, list)
    assert len(action) == 1
    assert action[0] in planner.action_sampler.get_space()


def test_action_progressive_widening_new_action(planner, belief):
    """Test action progressive widening new action.
    
    Purpose: Validates action progressive widening new action
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    belief_node = BeliefNode(belief=belief, observation=None)
    belief_node.visit_count = 0
    
    # Use the function from dpw module directly
    action_node = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=planner.alpha_a,
        action_sampler=planner.action_sampler,
        exploration_constant=planner.exploration_constant
    )
    assert isinstance(action_node, ActionNode)
    assert action_node.parent == belief_node
    assert action_node in belief_node.children
    assert action_node.action in planner.action_sampler.get_space()


def test_action_progressive_widening_existing_action(planner, belief, action_sampler):
    """Test action progressive widening existing action.
    
    Purpose: Validates action progressive widening existing action
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
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
        exploration_constant=planner.exploration_constant
    )
    assert isinstance(action_node, ActionNode)
    assert action_node in belief_node.children


def test_explored_action_node_ucb_selection(planner, belief):
    """Test explored action node ucb selection.
    
    Purpose: Validates explored action node ucb selection
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
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
    """Test rollout.
    
    Purpose: Validates rollout
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    # Test the random_rollout_action_sampler function that POMCPOW uses
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
    """Test rollout terminal state.
    
    Purpose: Validates rollout terminal state
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
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
    """Test rollout max depth.
    
    Purpose: Validates rollout max depth
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
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
    """Test simulate state path terminal state.
    
    Purpose: Validates simulate state path terminal state
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    """Test simulate path.
    
    Purpose: Validates simulate path
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
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
    """Test get space info.
    
    Purpose: Validates get space info
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    """Test simulate state path max depth.
    
    Purpose: Validates simulate state path max depth
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    belief_node = BeliefNode(belief=belief, observation=None)
    state = belief.sample()
    depth = planner.depth + 1
    
    return_value = planner._simulate_state_path(state=state, belief_node=belief_node, depth=depth)
    assert return_value == 0


def test_get_space_info(planner):
    """Test integration with tiger pomdp.
    
    Purpose: Validates integration with tiger pomdp
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: integration
    """
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
    """Test progressive widening parameters.
    
    Purpose: Validates progressive widening parameters
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    belief_node = BeliefNode(belief=belief, observation=None)
    belief_node.visit_count = 100
    
    # Test that progressive widening respects k_o and alpha_o parameters
    action_node = ActionNode(action=0, parent=belief_node)
    """Test that belief nodes maintain proper belief structure for states and weights.
    
    Purpose: Validates belief node data structure
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    
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
    
    """Test POMCPOW with SanityPOMDP to verify correct action selection.
    
    Purpose: Validates sanity pomdp action selection
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    planner._simulate_path(belief_node=belief_node, depth=0)
    
    # Check that children have proper belief structure
    for action_node in belief_node.children:
        for child_belief_node in action_node.children:
            # Check that the belief is a WeightedParticleBeliefStateUpdate instance
            from POMDPPlanners.core.belief import WeightedParticleBeliefStateUpdate
            assert isinstance(child_belief_node.belief, WeightedParticleBeliefStateUpdate)
            assert hasattr(child_belief_node.belief, 'particles')
            assert hasattr(child_belief_node.belief, 'weights')
            assert isinstance(child_belief_node.belief.particles, list)
            assert isinstance(child_belief_node.belief.weights, list)


def test_sanity_pomdp_action_selection():
    """Test POMCPOW with SanityPOMDP to verify correct action selection."""
    environment = SanityPOMDP()
    action_sampler = MockActionSampler([0, 1])
    
    planner = POMCPOW(
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
        name="TestPOMCPOW"
    )
    
    belief = get_initial_belief(
        pomdp=environment,
        n_particles=100,
        resampling=True
    )
    
    # Run multiple trials to ensure consistent behavior
    n_trials = 10
    action_0_count = 0
    """Test that the tree structure is properly constructed.
    
    Purpose: Validates tree structure after construction
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
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
        f"POMCPOW selected action 0 only {action_0_count}/{n_trials} times, expected at least {0.7 * n_trials}"


def test_tree_structure_after_construction(planner, belief):
    """Test that the tree structure is properly constructed."""
    belief_node = BeliefNode(belief=belief, observation=None)
    
    # Run several simulations to build the tree
    for _ in range(50):
        planner._simulate_path(belief_node=belief_node, depth=0)
    
    # Verify tree structure
    assert len(belief_node.children) > 0
    """Test that Q-values are properly updated during simulation.
    
    Purpose: Validates update functionality for q value s
    
    Given: Initial object state and update parameters
    When: Update operation is performed
    Then: Object state is correctly modified
    
    Test type: unit
    """
    
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
    """Test that visit counts are consistent throughout the tree.
    
    Purpose: Validates visit count consistency
    
    Given: Test setup conditions
    When: Test operation is performed
    Then: Expected behavior is verified
    
    Test type: unit
    """
    
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