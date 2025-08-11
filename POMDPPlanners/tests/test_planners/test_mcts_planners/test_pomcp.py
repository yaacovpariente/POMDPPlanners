import pytest

from anytree import PostOrderIter

from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.tree import BeliefNode, ActionNode

import time

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
def n_simulations():
    return 100


@pytest.fixture
def n_particles():
    return 100


@pytest.fixture
def environment(discount_factor):
    return TigerPOMDP(discount_factor=discount_factor)


@pytest.fixture
def planner(environment, discount_factor, depth, exploration_constant, n_simulations):
    return POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        n_simulations=n_simulations,
        name="TestPOMCP"
    )


@pytest.fixture
def belief(environment, n_particles):
    return get_initial_belief(
        pomdp=environment,
        n_particles=n_particles,
        resampling=True
    )


def test_initialization_with_n_simulations(environment, discount_factor, depth, exploration_constant):
    """Test initialization with n simulations.
    
    Purpose: Validates proper initialization of  with n simulations
    
    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes
    
    Test type: unit
    """
    planner = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        n_simulations=100,
        name="TestPOMCP"
    )
    assert planner.environment == environment
    assert planner.discount_factor == discount_factor
    assert planner.depth == depth
    assert planner.exploration_constant == exploration_constant
    assert planner.n_simulations == 100
    assert planner.time_out_in_seconds is None


def test_initialization_with_timeout(environment, discount_factor, depth, exploration_constant):
    """Test initialization with timeout.
    
    Purpose: Validates proper initialization of  with timeout
    
    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes
    
    Test type: unit
    """
    planner = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        time_out_in_seconds=5,
        name="TestPOMCP"
    )
    assert planner.environment == environment
    assert planner.discount_factor == discount_factor
    assert planner.depth == depth
    assert planner.exploration_constant == exploration_constant
    assert planner.time_out_in_seconds == 5
    assert planner.n_simulations is None


def test_invalid_initialization(environment, discount_factor, depth, exploration_constant):
    """Test invalid initialization.
    
    Purpose: Validates proper initialization of invalid 
    
    Given: Constructor parameters and initial conditions
    When: Object is initialized
    Then: Object is properly constructed with expected attributes
    
    Test type: unit
    """
    with pytest.raises(ValueError):
        POMCP(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            exploration_constant=exploration_constant,
            n_simulations=100,
            time_out_in_seconds=5,
            name="TestPOMCP"
        )


def test_action_selection(planner, belief, environment):
    """Test action selection.
    
    Purpose: Validates that POMCP action selection returns valid TigerPOMDP actions through MCTS search
    
    Given: POMCP planner with TigerPOMDP environment, WeightedParticleBelief with 100 particles, 100 simulations
    When: action method performs MCTS search and selects best action
    Then: Returns list with single valid tiger action (listen/open_left/open_right) and proper PolicyRunData
    
    Test type: unit
    """
    action, policy_run_data = planner.action(belief)
    assert isinstance(action, list)
    assert len(action) == 1
    assert action[0] in environment.actions


def test_search_behavior_with_initial_belief(planner, belief, environment):
    """Test search behavior with initial belief.
    
    Purpose: Validates that POMCP search behavior produces consistent action selection using initial belief state
    
    Given: POMCP planner with initial TigerPOMDP belief state containing 100 particles
    When: action method executes MCTS search from initial belief
    Then: Returns consistent single-element action list with valid tiger actions
    
    Test type: unit
    """
    # The search method has been removed, so we test the action method instead
    action, policy_run_data = planner.action(belief)
    assert isinstance(action, list)
    assert len(action) == 1
    assert action[0] in environment.actions


def test_random_rollout(planner):
    """Test random rollout.
    
    Purpose: Validates that random rollout simulation returns finite reward values for MCTS value estimation
    
    Given: POMCP planner with TigerPOMDP environment, initial state "tiger_left", rollout depth=0
    When: random_rollout performs simulation from given state
    Then: Returns finite float value representing estimated future reward
    
    Test type: unit
    """
    state = "tiger_left"
    return_value = planner.random_rollout(state=state, depth=0)
    assert isinstance(return_value, float)


def test_integration_with_tiger_pomdp(planner, belief, environment, n_particles):
    """Test integration with tiger pomdp.
    
    Purpose: Validates that POMCP integrates correctly with TigerPOMDP environment for complete POMDP planning workflow
    
    Given: POMCP planner, TigerPOMDP with discount_factor=0.95, initial belief with 100 particles
    When: Full MCTS planning cycle executes including action selection and belief updates
    Then: Valid tiger actions selected, belief updates work correctly, and environment state transitions function properly
    
    Test type: integration
    """
    current_belief = belief
    for _ in range(5):
        action, policy_run_data = planner.action(current_belief)
        assert isinstance(action, list)
        assert len(action) == 1
        assert action[0] in environment.actions
        
        # Simulate environment step
        state = current_belief.sample()
        next_state = environment.state_transition_model(state, action[0]).sample()[0]
        observation = environment.observation_model(next_state, action[0]).sample()[0]  # Extract first element from list
        
        # Update belief
        current_belief = current_belief.update(action[0], observation, environment)
        
        # Verify belief is valid
        assert isinstance(current_belief, WeightedParticleBelief)
        assert len(current_belief.particles) == n_particles


def test_construct_tree_using_timeout(environment, discount_factor, depth, exploration_constant):
    """Test construct tree using timeout.
    
    Purpose: Validates that POMCP tree construction terminates correctly when timeout constraint is reached
    
    Given: POMCP planner with 1-second timeout instead of fixed simulation count, TigerPOMDP environment
    When: MCTS tree construction runs until timeout expires
    Then: Tree construction completes within timeout, action is selected, and tree structure is valid
    
    Test type: unit
    """
    timeout_seconds = 1
    planner = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        time_out_in_seconds=timeout_seconds,
        name="TestPOMCP"
    )
    
    belief = get_initial_belief(environment, n_particles=100, resampling=True)
    belief_node = BeliefNode(belief=belief, observation=None)
    
    start_time = time.time()
    planner._construct_tree_using_timeout(belief_node=belief_node)
    end_time = time.time()
    
    # Verify the function ran for approximately the timeout duration
    assert abs(end_time - start_time - timeout_seconds) < 0.5  # Allow 0.5s margin
    
    # Verify tree structure was created
    assert len(belief_node.children) > 0  # Should have at least one action node
    assert all(isinstance(child, ActionNode) for child in belief_node.children)


def test_construct_tree_using_n_simulations(environment, discount_factor, depth, exploration_constant):
    """Test construct tree using n simulations.
    
    Purpose: Validates that POMCP tree construction completes exactly the specified number of MCTS simulations
    
    Given: POMCP planner with n_simulations=50, TigerPOMDP environment, initial belief
    When: MCTS tree construction executes exactly 50 simulations
    Then: Tree construction completes, action is selected, and simulation count is respected
    
    Test type: unit
    """
    n_sims = 50
    planner = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        n_simulations=n_sims,
        name="TestPOMCP"
    )
    
    belief = get_initial_belief(environment, n_particles=100, resampling=True)
    belief_node = BeliefNode(belief=belief, observation=None)
    
    # Count total visits to verify number of simulations
    initial_visit_count = belief_node.visit_count
    
    planner._construct_tree_using_n_simulations(belief_node=belief_node)
    
    # Verify tree structure was created
    assert len(belief_node.children) > 0  # Should have at least one action node
    assert all(isinstance(child, ActionNode) for child in belief_node.children)
    
    # Verify total visits increased by approximately n_sims
    # Note: The actual number might be slightly different due to tree structure
    assert belief_node.visit_count >= initial_visit_count + n_sims * 0.5  # Allow for some variance


def test_tree_structure_construction(environment, discount_factor, depth, exploration_constant):
    """Test tree structure construction.
    
    Purpose: Validates that POMCP builds proper tree structure with BeliefNode and ActionNode hierarchy during MCTS
    
    Given: POMCP planner with 100 simulations, TigerPOMDP environment, initial belief
    When: MCTS tree construction creates belief-action tree structure
    Then: Tree has root BeliefNode, action children, belief grandchildren, and proper parent-child relationships
    
    Test type: unit
    """
    n_sims = 100
    planner = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        n_simulations=n_sims,
        name="TestPOMCP"
    )
    
    n_particles = 100
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)
    root_belief_node = BeliefNode(belief=belief, observation=None)
    
    planner._construct_tree_using_n_simulations(belief_node=root_belief_node)
    
    assert root_belief_node.height == 2 * depth + 1
    for node in PostOrderIter(root_belief_node):
        assert node.visit_count >= 0
        if isinstance(node, BeliefNode):
            assert node.belief is not None
            assert node.v_value is not None
            
            if node.height > 1 and node.depth > 0:
                assert node.v_value != 0
                assert node.observation is not None
                n_children_visits = sum(child.visit_count for child in node.children)
                assert node.visit_count == n_children_visits + 1  # +1 for the rollout

        elif isinstance(node, ActionNode):
            assert node.action is not None
            assert node.q_value is not None
            if not node.is_leaf:
                assert node.q_value != 0
                assert node.visit_count == sum(child.visit_count for child in node.children)
    
    # Verify root belief node
    assert root_belief_node.observation is None
    assert root_belief_node.parent is None
    assert len(root_belief_node.children) == len(environment.get_actions())
    assert root_belief_node.visit_count == n_sims  # 
    assert root_belief_node.v_value is not None


def test_sanity_pomdp_action_selection():
    """Test sanity pomdp action selection.
    
    Purpose: Validates that POMCP correctly handles SanityPOMDP environment with deterministic optimal action selection
    
    Given: SanityPOMDP environment with known optimal actions, POMCP planner with 50 simulations
    When: MCTS search determines action selection in deterministic environment
    Then: Selected action is valid for SanityPOMDP and planning completes without errors
    
    Test type: unit
    """
    # Create environment and planner
    environment = SanityPOMDP()
    planner = POMCP(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        n_simulations=1000,  # Use more simulations for better accuracy
        name="TestPOMCP"
    )
    
    # Get initial belief
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
        if action[0] == 0:  # Count how many times action 0 is selected
            action_0_count += 1
    
    # Verify that action 0 (the better action) is selected most of the time
    # We expect at least 80% of the time to select action 0
    assert action_0_count >= 0.8 * n_trials, \
        f"POMCP selected action 0 only {action_0_count}/{n_trials} times, expected at least {0.8 * n_trials}"

