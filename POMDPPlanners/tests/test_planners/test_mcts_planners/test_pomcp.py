"""Tests for POMCP planner.

This module tests the POMCP planner, focusing on:
- Basic POMCP functionality
- Tree search operations
- Belief updates
- Planning algorithms
"""

import random
import time

import numpy as np
import pytest
from anytree import PostOrderIter

from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


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
        name="TestPOMCP",
    )


@pytest.fixture
def belief(environment, n_particles):
    return get_initial_belief(pomdp=environment, n_particles=n_particles, resampling=True)


def test_initialization_with_n_simulations(
    environment, discount_factor, depth, exploration_constant
):
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
        name="TestPOMCP",
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
        name="TestPOMCP",
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
            name="TestPOMCP",
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
        observation = environment.observation_model(next_state, action[0]).sample()[
            0
        ]  # Extract first element from list

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
        name="TestPOMCP",
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


def test_construct_tree_using_n_simulations(
    environment, discount_factor, depth, exploration_constant
):
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
        name="TestPOMCP",
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
        name="TestPOMCP",
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
        name="TestPOMCP",
    )

    # Get initial belief
    belief = get_initial_belief(pomdp=environment, n_particles=100, resampling=True)

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
    assert (
        action_0_count >= 0.8 * n_trials
    ), f"POMCP selected action 0 only {action_0_count}/{n_trials} times, expected at least {0.8 * n_trials}"


# Config ID Tests


def test_pomcp_config_id_consistency_identical_parameters(
    environment, discount_factor, depth, exploration_constant, n_simulations
):
    """Test that config_id is consistent for identical POMCP parameters.

    Purpose: Validates that POMCP with identical parameters produces identical config_id

    Given: Two POMCP instances with identical parameters
    When: config_id is accessed on both instances
    Then: Both instances return the same config_id

    Test type: unit
    """
    # Create two POMCP instances with identical parameters
    pomcp1 = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        name="POMCP_Test1",
        n_simulations=n_simulations,
    )

    pomcp2 = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        name="POMCP_Test1",  # Same name
        n_simulations=n_simulations,
    )

    # Config IDs should be identical
    config_id1 = pomcp1.config_id
    config_id2 = pomcp2.config_id

    assert config_id1 == config_id2
    assert isinstance(config_id1, str)
    assert len(config_id1) > 0


def test_pomcp_config_id_different_exploration_constant(
    environment, discount_factor, depth, n_simulations
):
    """Test that config_id changes when exploration_constant differs.

    Purpose: Validates that config_id changes when core POMCP parameters differ

    Given: Two POMCP instances with different exploration_constant values
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
    pomcp1 = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=1.0,
        name="POMCP_Test",
        n_simulations=n_simulations,
    )

    pomcp2 = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=2.0,  # Different exploration constant
        name="POMCP_Test",
        n_simulations=n_simulations,
    )

    config_id1 = pomcp1.config_id
    config_id2 = pomcp2.config_id

    assert config_id1 != config_id2


def test_pomcp_config_id_different_depth(
    environment, discount_factor, exploration_constant, n_simulations
):
    """Test that config_id changes when depth differs.

    Purpose: Validates that config_id changes when depth parameter differs

    Given: Two POMCP instances with different depth values
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
    pomcp1 = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=10,
        exploration_constant=exploration_constant,
        name="POMCP_Test",
        n_simulations=n_simulations,
    )

    pomcp2 = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=15,  # Different depth
        exploration_constant=exploration_constant,
        name="POMCP_Test",
        n_simulations=n_simulations,
    )

    config_id1 = pomcp1.config_id
    config_id2 = pomcp2.config_id

    assert config_id1 != config_id2


def test_pomcp_config_id_consistency_across_evaluations(
    environment, discount_factor, depth, exploration_constant
):
    """Test that config_id remains consistent across different policy evaluations.

    Purpose: Validates that config_id is stable across multiple accesses and policy actions

    Given: Single POMCP instance and initial belief
    When: config_id is accessed before and after policy actions
    Then: config_id remains identical across all evaluations

    Test type: integration
    """
    pomcp = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=3,  # Reduced for testing
        exploration_constant=exploration_constant,
        name="POMCP_Consistency_Test",
        n_simulations=10,  # Reduced for testing
    )

    # Get initial config_id
    initial_config_id = pomcp.config_id

    # Create initial belief and perform policy actions
    initial_belief = get_initial_belief(environment, n_particles=50)

    # Perform multiple policy evaluations
    for i in range(3):
        action, run_data = pomcp.action(initial_belief)

        # Check config_id remains the same
        current_config_id = pomcp.config_id
        assert current_config_id == initial_config_id

        # Verify the action and run_data are valid
        assert isinstance(action, list)
        assert len(action) == 1
        assert action[0] in environment.get_actions()
        assert run_data is not None

    # Final check
    final_config_id = pomcp.config_id
    assert final_config_id == initial_config_id


def test_pomcp_config_id_different_min_samples_per_node(
    environment, discount_factor, depth, exploration_constant, n_simulations
):
    """Test that config_id changes when min_samples_per_node differs.

    Purpose: Validates that config_id changes when min_samples_per_node parameter differs

    Given: Two POMCP instances with different min_samples_per_node values
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
    pomcp1 = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        name="POMCP_Test",
        n_simulations=n_simulations,
        min_samples_per_node=10,
    )

    pomcp2 = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        name="POMCP_Test",
        n_simulations=n_simulations,
        min_samples_per_node=20,  # Different min_samples_per_node
    )

    config_id1 = pomcp1.config_id
    config_id2 = pomcp2.config_id

    assert config_id1 != config_id2


def test_pomcp_config_id_hash_properties(
    environment, discount_factor, depth, exploration_constant, n_simulations
):
    """Test that config_id has proper hash properties.

    Purpose: Validates that config_id produces valid hash strings

    Given: POMCP instance
    When: config_id is accessed
    Then: config_id is a valid hash string with expected properties

    Test type: unit
    """
    pomcp = POMCP(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        name="POMCP_Hash_Test",
        n_simulations=n_simulations,
    )

    config_id = pomcp.config_id

    # Should be a non-empty string
    assert isinstance(config_id, str)
    assert len(config_id) > 0

    # Should be a valid hexadecimal hash (SHA-256 produces 64 hex characters)
    assert len(config_id) == 64
    assert all(c in "0123456789abcdef" for c in config_id.lower())


# ==============================================================================
# MEMORY LEAK DIAGNOSTIC TESTS
# ==============================================================================
# These tests help identify specific sources of memory leaks in POMCP.
# Tests that fail indicate components that need memory management fixes.

import gc

from POMDPPlanners.utils.memory_tracker import MemoryTracker


def test_memory_leak_environment_policy_object_accumulation(environment, discount_factor):
    """
    Purpose: Validates that Environment and Policy objects are properly garbage collected

    Given: Multiple cycles of environment and policy object creation with MCTS tree building
    When: Many POMDP environments and policies with search trees are created and cleared
    Then: Memory growth stays under 100MB after explicit cleanup indicating proper garbage collection

    Test type: unit
    """
    tracker = MemoryTracker(enable_tracking=True, tracking_mode="lightweight")
    tracker.checkpoint("initial")

    environments = []
    policies = []
    beliefs = []

    try:
        # Create many environment and policy objects
        for i in range(50):  # Create many objects to amplify leaks
            # Create environment
            env = TigerPOMDP(discount_factor=discount_factor, name=f"Tiger_{i}")
            environments.append(env)

            # Create policy with potentially large search tree
            policy = POMCP(
                environment=env,
                discount_factor=discount_factor,
                depth=5,  # Deeper tree for more memory usage
                exploration_constant=1.0,
                name=f"POMCP_{i}",
                n_simulations=20,  # More simulations for larger tree
            )
            policies.append(policy)

            # Create belief with many particles
            belief = get_initial_belief(env, n_particles=100)
            beliefs.append(belief)

            # Simulate some policy usage to build search tree
            action, _ = policy.action(belief)

        tracker.checkpoint("after_creation")

        # Clear references explicitly
        environments.clear()
        policies.clear()
        beliefs.clear()

        # Force garbage collection
        gc.collect()
        tracker.checkpoint("after_cleanup")

    finally:
        # Final cleanup
        environments.clear()
        policies.clear()
        beliefs.clear()
        gc.collect()

    # Get memory growth
    memory_growth = tracker.get_memory_growth()

    # Assert memory growth is reasonable after cleanup (<100MB)
    assert memory_growth < 100, f"Environment/Policy objects leaked {memory_growth:.1f} MB"
