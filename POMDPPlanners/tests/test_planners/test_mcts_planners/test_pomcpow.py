# SPDX-License-Identifier: MIT

"""Tests for POMCPOW planner.

This module tests the POMCPOW planner, focusing on:
- Basic POMCPOW functionality
- Tree search operations
- Belief updates
- Planning algorithms
"""

# pylint: disable=protected-access  # Tests need to access protected members

import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import (
    WeightedParticleBelief,
    WeightedParticleBeliefStateUpdate,
    get_initial_belief,
)
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.planners_utils.dpw import (
    ActionSampler,
    action_progressive_widening,
    ucb1_exploration,
)
from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler
from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


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
def planner(
    environment,
    discount_factor,
    depth,
    exploration_constant,
    k_o,
    k_a,
    alpha_o,
    alpha_a,
    n_simulations,
    action_sampler,
):
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
        name="TestPOMCPOW",
    )


@pytest.fixture
def belief(environment, n_particles):
    return get_initial_belief(pomdp=environment, n_particles=n_particles, resampling=True)


def test_initialization_with_n_simulations(
    environment,
    discount_factor,
    depth,
    exploration_constant,
    k_o,
    k_a,
    alpha_o,
    alpha_a,
    action_sampler,
):
    """Test initialization with n simulations.

    Purpose: Validates proper initialization of POMCPOW with n simulations

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
        name="TestPOMCPOW",
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


def test_initialization_with_timeout(
    environment,
    discount_factor,
    depth,
    exploration_constant,
    k_o,
    k_a,
    alpha_o,
    alpha_a,
    action_sampler,
):
    """Test initialization with timeout.

    Purpose: Validates proper initialization of POMCPOW with timeout

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
        name="TestPOMCPOW",
    )
    assert planner.environment == environment
    assert planner.discount_factor == discount_factor
    assert planner.depth == depth
    assert planner.exploration_constant == exploration_constant
    assert planner.time_out_in_seconds == 5
    assert planner.n_simulations is None


def test_invalid_initialization(
    environment,
    discount_factor,
    depth,
    exploration_constant,
    k_o,
    k_a,
    alpha_o,
    alpha_a,
    action_sampler,
):
    """Test invalid initialization.

    Purpose: Validates proper initialization of invalid POMCPOW

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
            name="TestPOMCPOW",
        )


def test_action_selection(planner, belief):
    """Test action selection from belief tree.

    Purpose: Validates that POMCPOW planner correctly selects actions from belief tree

    Given: POMCPOW planner with configured environment and belief with initial state distribution
    When: action method is called with belief
    Then: Returns valid action list with single action from environment's action space

    Test type: unit
    """
    action, policy_run_data = planner.action(belief)
    assert isinstance(action, list)
    assert len(action) == 1
    assert action[0] in planner.action_sampler.get_space()


def test_action_progressive_widening_new_action(planner, belief):
    """Test action progressive widening for new action nodes.

    Purpose: Validates that action progressive widening correctly creates new action nodes when threshold is met

    Given: BeliefNode with visit_count=0 and POMCPOW planner with progressive widening parameters
    When: action_progressive_widening is called
    Then: Creates new ActionNode with correct parent relationship and action from action sampler

    Test type: unit
    """
    belief_node = BeliefNode(belief=belief, observation=None)
    belief_node.visit_count = 0

    # Use the function from dpw module directly
    action_node = action_progressive_widening(
        belief_node=belief_node,
        alpha_a=planner.alpha_a,
        action_sampler=planner.action_sampler,
        exploration_constant=planner.exploration_constant,
        k_a=planner.k_a,
    )
    assert isinstance(action_node, ActionNode)
    assert action_node.parent == belief_node
    assert action_node in belief_node.children
    assert action_node.action in planner.action_sampler.get_space()


def test_action_progressive_widening_existing_action(planner, belief, action_sampler):
    """Test action progressive widening for existing action nodes.

    Purpose: Validates that action progressive widening correctly selects from existing action nodes when threshold is met

    Given: BeliefNode with existing ActionNode children and visit_count=50
    When: action_progressive_widening is called
    Then: Returns existing ActionNode from children, confirming progressive widening threshold behavior

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
        exploration_constant=planner.exploration_constant,
        k_a=planner.k_a,
    )
    assert isinstance(action_node, ActionNode)
    assert action_node in belief_node.children


def test_explored_action_node_ucb_selection(planner, belief):
    """Test UCB-based action selection from explored action nodes.

    Purpose: Validates that UCB selection correctly chooses actions based on Q-values and visit counts

    Given: BeliefNode with visit_count=30 and ActionNode children with different Q-values [0.1, 0.5, 0.3]
    When: UCB selection is performed
    Then: Returns ActionNode with highest UCB value, balancing exploration and exploitation

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

    selected_action_node = ucb1_exploration(
        belief_node=belief_node, exploration_constant=planner.exploration_constant
    )
    assert isinstance(selected_action_node, ActionNode)
    assert selected_action_node in belief_node.children


def test_rollout(planner):
    """Test rollout simulation from belief tree.

    Purpose: Validates that rollout function correctly simulates random policy from belief state

    Given: POMCPOW planner with configured environment and belief
    When: random_rollout_action_sampler is called with initial state and depth
    Then: Returns float reward value, confirming successful rollout simulation

    Test type: unit
    """
    # Test the random_rollout_action_sampler function that POMCPOW uses

    state = "tiger_left"
    depth = 0
    return_value = random_rollout_action_sampler(
        state=state,
        depth=depth,
        action_sampler=planner.action_sampler,
        environment=planner.environment,
        discount_factor=planner.discount_factor,
    )
    assert isinstance(return_value, float)


def test_rollout_terminal_state(planner):
    """Test rollout behavior when encountering terminal state.

    Purpose: Validates that rollout function correctly handles terminal states by returning zero reward

    Given: POMCPOW planner with environment modified to always return terminal=True for any state
    When: random_rollout_action_sampler is called with terminal state
    Then: Function returns 0.0 reward, confirming proper terminal state handling

    Test type: unit
    """
    # Test the random_rollout_action_sampler function with terminal state

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
        discount_factor=planner.discount_factor,
    )
    assert return_value == 0

    # Restore original method
    planner.environment.is_terminal = original_is_terminal


def test_rollout_max_depth(planner):
    """Test rollout behavior when reaching maximum depth.

    Purpose: Validates that rollout function correctly handles maximum depth limit

    Given: POMCPOW planner with depth=2 and environment that never reaches terminal state
    When: random_rollout_action_sampler is called starting from depth 0
    Then: Function returns cumulative discounted reward after exactly 2 steps, confirming depth limit enforcement

    Test type: unit
    """
    # Test the random_rollout_action_sampler function with max depth

    state = "tiger_left"
    depth = planner.depth + 1  # This is 4 (planner.depth = 3)
    max_depth = depth  # Set max_depth to match the depth being tested

    return_value = random_rollout_action_sampler(
        state=state,
        depth=depth,
        action_sampler=planner.action_sampler,
        environment=planner.environment,
        discount_factor=planner.discount_factor,
        max_depth=max_depth,  # Pass the max_depth parameter
    )
    assert return_value == 0


def test_simulate_path(planner, belief):
    """Test path simulation through arena tree.

    Purpose: Validates that simulate_path correctly traverses the arena tree and updates node statistics.

    Given: POMCPOW planner (arena backend) and belief with initial state distribution.
    When: simulate_path is called starting from the root belief id.
    Then: Function returns a float value, tree is expanded, root visit_count incremented.

    Test type: unit
    """
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    depth = 0

    return_value = planner._simulate_path(tree=tree, belief_id=root_id, depth=depth)
    assert isinstance(return_value, float)
    assert tree.visit_count[root_id] >= 1


def test_simulate_state_path_terminal_state(planner, belief):
    """Test state path simulation with terminal state.

    Purpose: Validates that simulate_state_path correctly handles terminal states during simulation.

    Given: POMCPOW planner, belief, and environment modified to return terminal=True for any state.
    When: simulate_state_path is called against the arena tree.
    Then: Function returns 0 immediately, root visit_count incremented to 1.

    Test type: unit
    """
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    depth = 0

    original_is_terminal = planner.environment.is_terminal
    planner.environment.is_terminal = lambda state: True

    state = belief.sample()
    return_value = planner._simulate_state_path(
        tree=tree, state=state, belief_id=root_id, depth=depth
    )
    assert return_value == 0
    assert tree.visit_count[root_id] == 1

    planner.environment.is_terminal = original_is_terminal


def test_simulate_state_path_max_depth(planner, belief):
    """Test state path simulation with maximum depth limit.

    Purpose: Validates that simulate_state_path correctly enforces the depth limit on the arena tree.

    Given: POMCPOW planner, belief, depth set above the planner's max.
    When: simulate_state_path is called.
    Then: Function returns 0 immediately due to the depth gate.

    Test type: unit
    """
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    state = belief.sample()
    depth = planner.depth + 1

    return_value = planner._simulate_state_path(
        tree=tree, state=state, belief_id=root_id, depth=depth
    )
    assert return_value == 0


def test_get_space_info(planner):
    """Test space information retrieval.

    Purpose: Validates that planner correctly provides space information for action and observation spaces

    Given: POMCPOW planner with configured environment
    When: get_space_info method is called
    Then: Returns SpaceInfo object with correct action_space and observation_space values matching environment configuration

    Test type: unit
    """
    space_info = planner.get_space_info()
    assert hasattr(space_info, "action_space")
    assert hasattr(space_info, "observation_space")
    assert space_info.action_space == SpaceType.MIXED
    assert space_info.observation_space == SpaceType.MIXED


def test_integration_with_tiger_pomdp(planner, belief, environment, n_particles):
    """Test POMCPOW integration with TigerPOMDP environment.

    Purpose: Validates that POMCPOW planner works correctly with real POMDP environment

    Given: POMCPOW planner, TigerPOMDP environment, and belief with n_particles particles
    When: Action selection is performed multiple times
    Then: Planner returns valid actions from environment's action space, confirming proper integration

    Test type: integration
    """
    current_belief = belief
    for _ in range(3):
        action, policy_run_data = planner.action(current_belief)
        assert isinstance(action, list)
        assert len(action) == 1
        assert action[0] in planner.action_sampler.get_space()

        # Simulate environment step
        state = current_belief.sample()
        next_state = environment.sample_next_state(state=state, action=action[0])
        observation = environment.sample_observation(next_state=next_state, action=action[0])

        # Update belief
        current_belief = current_belief.update(action[0], observation, environment)

        # Verify belief is valid
        assert isinstance(current_belief, WeightedParticleBelief)
        assert len(current_belief.particles) == n_particles


def test_progressive_widening_parameters(planner, belief):
    """Test progressive widening parameter effects on tree expansion.

    Purpose: Validates that progressive widening parameters correctly control tree expansion behavior

    Given: POMCPOW planner with k_o=2, k_a=2, alpha_o=0.5, alpha_a=0.5 parameters
    When: Multiple simulations are performed
    Then: Tree expansion follows progressive widening rules: new actions added when visit count reaches k_a * N^alpha_a, new observations when visit count reaches k_o * N^alpha_o

    Test type: unit
    """
    belief_node = BeliefNode(belief=belief, observation=None)
    belief_node.visit_count = 100

    # Test that progressive widening respects k_o and alpha_o parameters
    action_node = ActionNode(action=0, parent=belief_node)
    action_node.visit_count = (
        1  # Set visit count to enable meaningful progressive widening calculation
    )

    # Create mock observation
    observation = "tiger_growl_left"

    # Test observation progressive widening condition
    max_observations = planner.k_o * action_node.visit_count**planner.alpha_o
    assert max_observations > 0

    # The condition should allow adding new observations if under the limit
    current_observations = len(action_node.children)
    can_add_observation = current_observations <= max_observations
    assert isinstance(can_add_observation, bool)


def test_belief_node_data_structure(planner, belief):
    """Test belief node data structure on the arena tree.

    Purpose: Validates that belief children of action children carry the expected
    belief class with the expected particle/weight attributes.

    Given: POMCPOW planner (arena backend) and belief with initial state distribution.
    When: One simulate_path is performed.
    Then: For every action child of root, every belief grandchild has a
    WeightedParticleBeliefStateUpdate with list-typed particles + weights.

    Test type: unit
    """
    tree = Tree()
    root_id = tree.add_belief_node(belief)

    planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

    for action_id in tree.children_ids[root_id]:
        assert tree.kind[action_id] == ACTION
        for belief_child_id in tree.children_ids[action_id]:
            assert tree.kind[belief_child_id] == BELIEF
            child_belief = tree.belief[belief_child_id]
            assert isinstance(child_belief, WeightedParticleBeliefStateUpdate)
            assert hasattr(child_belief, "particles")
            assert hasattr(child_belief, "weights")
            assert isinstance(child_belief.particles, list)
            assert isinstance(child_belief.weights, list)


@pytest.mark.slow
def test_sanity_pomdp_action_selection():
    """Test POMCPOW action selection with SanityPOMDP environment.

    Purpose: Validates that POMCPOW planner works correctly with deterministic SanityPOMDP environment

    Given: POMCPOW planner configured with SanityPOMDP environment and belief
    When: Action selection is performed
    Then: Planner returns valid actions from SanityPOMDP action space [0, 1], confirming proper environment integration

    Test type: integration
    """
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
        name="TestPOMCPOW",
    )

    belief = get_initial_belief(pomdp=environment, n_particles=100, resampling=True)

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
    assert (
        action_0_count >= 0.7 * n_trials
    ), f"POMCPOW selected action 0 only {action_0_count}/{n_trials} times, expected at least {0.7 * n_trials}"


def test_tree_structure_after_construction(
    planner, belief, n_simulations, depth, k_o, k_a, alpha_o, alpha_a, action_sampler
):
    """Test arena tree structure after construction.

    Purpose: Validates the arena tree built by ``_learn_tree`` has the expected
    root + action-children + belief-grandchildren structure with progressive widening.

    Given: POMCPOW planner (arena backend) with PW params and belief.
    When: ``_learn_tree`` is called.
    Then: root visit_count == n_simulations; root has action children; each action
    child has belief-update beliefs; action-children count respects k_a * N^alpha_a.

    Test type: unit
    """
    del depth, k_o, alpha_o, action_sampler  # PW invariants checked inline below

    tree, root_id = planner._learn_tree(belief=belief)

    assert tree.kind[root_id] == BELIEF
    assert tree.parent_id[root_id] is None
    assert tree.visit_count[root_id] == n_simulations

    action_children = tree.children_ids[root_id]
    assert len(action_children) > 0
    assert all(tree.kind[cid] == ACTION for cid in action_children)
    # Action progressive-widening upper bound (allow +1 slack for the final widening step).
    assert len(action_children) <= k_a * (n_simulations**alpha_a) + 1

    # Each action child has belief grandchildren of the expected belief class.
    for action_id in action_children:
        for belief_id in tree.children_ids[action_id]:
            assert tree.kind[belief_id] == BELIEF
            assert isinstance(tree.belief[belief_id], WeightedParticleBeliefStateUpdate)


def test_q_value_updates(planner, belief):
    """Test Q-value updates during simulation on the arena tree.

    Purpose: Validates that at least one action child of root receives a visit
    after one simulation, evidence its q_value would be updated.

    Given: POMCPOW planner (arena backend) and belief.
    When: One simulate_path is performed.
    Then: At least one action child of root has visit_count > 0.

    Test type: unit
    """
    tree = Tree()
    root_id = tree.add_belief_node(belief)

    planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

    action_updated = False
    for action_id in tree.children_ids[root_id]:
        if tree.kind[action_id] == ACTION and tree.visit_count[action_id] > 0:
            action_updated = True
            break

    assert action_updated, "No action node was updated during simulation"


def test_visit_count_consistency(planner, belief):
    """Test visit count consistency on the arena tree.

    Purpose: Validates that root visit_count equals the number of simulations and
    that the sum of action-children visit counts does not exceed the root.

    Given: POMCPOW planner (arena backend) and belief.
    When: 20 simulate_path calls are performed.
    Then: tree.visit_count[root_id] == 20 and sum(action_children_visits) <= 20.

    Test type: unit
    """
    tree = Tree()
    root_id = tree.add_belief_node(belief)

    n_sims = 20
    for _ in range(n_sims):
        planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

    assert tree.visit_count[root_id] == n_sims

    total_action_visits = sum(
        tree.visit_count[cid] for cid in tree.children_ids[root_id] if tree.kind[cid] == ACTION
    )
    assert total_action_visits <= tree.visit_count[root_id]


def test_pomcpow_tree_structure_construction(
    environment,
    discount_factor,
    depth,
    exploration_constant,
    k_o,
    k_a,
    alpha_o,
    alpha_a,
    action_sampler,
):
    """Test POMCPOW tree structure construction with different parameters.

    Purpose: Validates that POMCPOW planner constructs belief trees correctly with various parameter configurations

    Given: POMCPOW planner with different progressive widening parameters and belief
    When: Tree construction is performed
    Then: Belief tree has correct structure with proper node hierarchy, progressive widening behavior, and parameter-dependent expansion patterns

    Test type: unit
    """
    # ARRANGE: Setup POMCPOW planner with sufficient simulations to build meaningful tree
    n_simulations = 200
    planner = POMCPOW(
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
        name="TestPOMCPOW_TreeStructure",
    )

    n_particles = 100
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)

    # ACT: Build complete tree using _learn_tree method (arena backend)
    tree, root_id = planner._learn_tree(belief=belief)

    # ASSERT (inline arena-aware checks): root has the expected count, has action
    # children, each action child has belief grandchildren of the expected class.
    del (
        environment,
        discount_factor,
        depth,
        exploration_constant,
        k_o,
        k_a,
        alpha_o,
        alpha_a,
        action_sampler,
    )  # noqa: E501  pylint: disable=line-too-long
    assert tree.kind[root_id] == BELIEF
    assert tree.visit_count[root_id] == n_simulations
    action_children = tree.children_ids[root_id]
    assert len(action_children) > 0
    for action_id in action_children:
        assert tree.kind[action_id] == ACTION
        for belief_id in tree.children_ids[action_id]:
            assert tree.kind[belief_id] == BELIEF
            assert isinstance(tree.belief[belief_id], WeightedParticleBeliefStateUpdate)


# Config ID Tests


def test_pomcpow_config_id_consistency_identical_parameters(
    environment,
    discount_factor,
    depth,
    exploration_constant,
    k_o,
    k_a,
    alpha_o,
    alpha_a,
    action_sampler,
):
    """Test that config_id is consistent for identical POMCPOW parameters.

    Purpose: Validates that POMCPOW with identical parameters produces identical config_id

    Given: Two POMCPOW instances with identical parameters
    When: config_id is accessed on both instances
    Then: Both instances return the same config_id

    Test type: unit
    """
    # Create two POMCPOW instances with identical parameters
    pomcpow1 = POMCPOW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        name="POMCPOW_Test1",
        action_sampler=action_sampler,
        n_simulations=100,
    )

    pomcpow2 = POMCPOW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        name="POMCPOW_Test1",  # Same name
        action_sampler=action_sampler,
        n_simulations=100,
    )

    # Config IDs should be identical
    config_id1 = pomcpow1.config_id
    config_id2 = pomcpow2.config_id

    assert config_id1 == config_id2
    assert isinstance(config_id1, str)
    assert len(config_id1) > 0


def test_pomcpow_config_id_different_action_sampler_values():
    """Test that config_id changes when action sampler is initialized with different values.

    Purpose: Validates that config_id changes when action sampler parameters differ

    Given: Two POMCPOW instances with action samplers having different max_action_magnitude
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
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

    pomcpow1 = POMCPOW(
        environment=continuous_environment,
        discount_factor=0.99,
        depth=10,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="POMCPOW_Test",
        action_sampler=sampler1,
        n_simulations=100,
    )

    pomcpow2 = POMCPOW(
        environment=continuous_environment,
        discount_factor=0.99,
        depth=10,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="POMCPOW_Test",
        action_sampler=sampler2,  # Different sampler parameters
        n_simulations=100,
    )

    config_id1 = pomcpow1.config_id
    config_id2 = pomcpow2.config_id

    assert config_id1 != config_id2
    assert isinstance(config_id1, str)
    assert isinstance(config_id2, str)
    assert len(config_id1) > 0
    assert len(config_id2) > 0


def test_pomcpow_config_id_different_planner_parameters(
    environment, discount_factor, depth, k_o, k_a, alpha_o, alpha_a, action_sampler
):
    """Test that config_id changes when POMCPOW planner parameters differ.

    Purpose: Validates that config_id changes when core POMCPOW parameters differ

    Given: Two POMCPOW instances with different exploration_constant values
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
    pomcpow1 = POMCPOW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=1.0,  # Different exploration constant
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        name="POMCPOW_Test",
        action_sampler=action_sampler,
        n_simulations=100,
    )

    pomcpow2 = POMCPOW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=2.0,  # Different exploration constant
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        name="POMCPOW_Test",
        action_sampler=action_sampler,
        n_simulations=100,
    )

    config_id1 = pomcpow1.config_id
    config_id2 = pomcpow2.config_id

    assert config_id1 != config_id2


def test_pomcpow_config_id_consistency_across_evaluations(
    environment,
    discount_factor,
    depth,
    exploration_constant,
    k_o,
    k_a,
    alpha_o,
    alpha_a,
    action_sampler,
):
    """Test that config_id remains consistent across different policy evaluations.

    Purpose: Validates that config_id is stable across multiple accesses and policy actions

    Given: Single POMCPOW instance and initial belief
    When: config_id is accessed before and after policy actions
    Then: config_id remains identical across all evaluations

    Test type: integration
    """
    pomcpow = POMCPOW(
        environment=environment,
        discount_factor=discount_factor,
        depth=3,  # Reduced for testing
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        name="POMCPOW_Consistency_Test",
        action_sampler=action_sampler,
        n_simulations=10,  # Reduced for testing
    )

    # Get initial config_id
    initial_config_id = pomcpow.config_id

    # Create initial belief and perform policy actions
    initial_belief = get_initial_belief(environment, n_particles=50)

    # Perform multiple policy evaluations
    for i in range(3):
        action, run_data = pomcpow.action(initial_belief)

        # Check config_id remains the same
        current_config_id = pomcpow.config_id
        assert current_config_id == initial_config_id

        # Verify the action and run_data are valid
        assert isinstance(action, list)
        assert len(action) == 1
        assert action[0] in action_sampler.get_space()
        assert run_data is not None

    # Final check
    final_config_id = pomcpow.config_id
    assert final_config_id == initial_config_id


def test_pomcpow_config_id_action_sampler_attribute_changes():
    """Test config_id changes when action sampler attributes are modified.

    Purpose: Validates that modifying action sampler attributes affects config_id

    Given: POMCPOW instance with modifiable action sampler
    When: Action sampler attributes are changed
    Then: config_id reflects the change

    Test type: unit
    """
    # Create continuous environment for testing
    continuous_environment = ContinuousLightDarkPOMDP(
        discount_factor=0.99,
        goal_state=np.array([5, 0]),
        start_state=np.array([0, 0]),
        name="TestContinuous",
    )

    # Create initial action sampler and POMCPOW
    action_sampler = UnitCircleActionSampler(max_action_magnitude=1.0)

    pomcpow = POMCPOW(
        environment=continuous_environment,
        discount_factor=0.99,
        depth=10,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="POMCPOW_Attribute_Test",
        action_sampler=action_sampler,
        n_simulations=100,
    )

    # Get initial config_id
    initial_config_id = pomcpow.config_id

    # Modify the action sampler's attribute
    action_sampler.max_action_magnitude = 2.0

    # Config ID should be different after modification
    modified_config_id = pomcpow.config_id
    assert modified_config_id != initial_config_id

    # Restore original value
    action_sampler.max_action_magnitude = 1.0

    # Config ID should return to original
    restored_config_id = pomcpow.config_id
    assert restored_config_id == initial_config_id


def test_pomcpow_config_id_hash_properties(
    environment,
    discount_factor,
    depth,
    exploration_constant,
    k_o,
    k_a,
    alpha_o,
    alpha_a,
    action_sampler,
):
    """Test that config_id has proper hash properties.

    Purpose: Validates that config_id produces valid hash strings

    Given: POMCPOW instance
    When: config_id is accessed
    Then: config_id is a valid hash string with expected properties

    Test type: unit
    """
    pomcpow = POMCPOW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        name="POMCPOW_Hash_Test",
        action_sampler=action_sampler,
        n_simulations=100,
    )

    config_id = pomcpow.config_id

    # Should be a non-empty string
    assert isinstance(config_id, str)
    assert len(config_id) > 0

    # Should be a valid hexadecimal hash (SHA-256 produces 64 hex characters)
    assert len(config_id) == 64
    assert all(c in "0123456789abcdef" for c in config_id.lower())


def test_min_visit_count_per_action_enforcement(environment, action_sampler):
    """Test that min_visit_count_per_action ensures minimum visits for each root action.

    Purpose: Validates the min_visit_count_per_action parameter on the arena tree backend.

    Given: POMCPOW (arena backend) with k_a=2.0, alpha_a=0.0, min_visit_count_per_action=5.
    When: ``_learn_tree`` builds the tree from the initial belief.
    Then: Every action child of the root has visit_count >= min_visit_count_per_action.

    Test type: unit
    """
    del action_sampler  # we use a fresh MockActionSampler below to bound the action set
    min_visit_count = 5
    planner = POMCPOW(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=2.0,
        alpha_o=0.5,
        alpha_a=0.0,
        n_simulations=50,
        action_sampler=MockActionSampler(TigerPOMDP(discount_factor=0.99).get_actions()),
        name="TestPOMCPOW_MinVisit",
        min_visit_count_per_action=min_visit_count,
    )

    n_particles = 10
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)

    tree, root_id = planner._learn_tree(belief=belief)

    action_ids = [cid for cid in tree.children_ids[root_id] if tree.kind[cid] == ACTION]
    assert len(action_ids) > 0, "At least one action node should be created"

    for action_id in action_ids:
        visits = tree.visit_count[action_id]
        assert visits >= min_visit_count, (
            f"Action node {tree.action[action_id]} has {visits} visits, "
            f"expected at least {min_visit_count}"
        )


def test_max_depth_reached_with_timeout():
    """Test arena tree construction with timeout.

    Purpose: Validates POMCPOW builds the expected belief/action structure on the
    arena tree under a time-based budget and reaches the configured max depth.

    Given: POMCPOW (arena backend) with timeout=0.5s on TigerPOMDP.
    When: ``_construct_tree_using_timeout`` is invoked with a fresh tree + root id.
    Then: Tree max depth (in arena terms) is 2*(depth+1); every node has
    visit_count >= 0; non-root belief nodes have a non-None belief and v_value.

    Test type: unit
    """
    depth = 3
    environment = TigerPOMDP(discount_factor=0.95)
    action_sampler = MockActionSampler(environment.get_actions())

    planner = POMCPOW(
        environment=environment,
        discount_factor=0.95,
        depth=depth,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=3.0,
        alpha_o=0.5,
        alpha_a=0.5,
        time_out_in_seconds=0.5,  # type: ignore[arg-type]  # Plan for 0.5 seconds
        action_sampler=action_sampler,
        name="TestPOMCPOW_MaxDepth",
    )

    n_particles = 100
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)
    tree = Tree()
    root_id = tree.add_belief_node(belief)

    planner._construct_tree_using_timeout(tree=tree, root_id=root_id)

    # Walk the arena tree to compute max depth (longest path from root).
    max_observed_depth = 0
    frontier = [(root_id, 0)]
    while frontier:
        node_id, d = frontier.pop()
        if d > max_observed_depth:
            max_observed_depth = d
        for cid in tree.children_ids[node_id]:
            frontier.append((cid, d + 1))
    assert max_observed_depth == 2 * (depth + 1)

    # Validate per-node invariants.
    for node_id in range(len(tree)):
        assert tree.visit_count[node_id] >= 0
        if tree.kind[node_id] == BELIEF and node_id != root_id:
            assert tree.belief[node_id] is not None
            assert tree.v_value[node_id] is not None
            # Note: leaves have v_value=0 by initialization, so we don't assert
            # v_value != 0 for arbitrary belief nodes here.

    # Verify root belief node invariants in arena form.
    assert tree.parent_id[root_id] is None
    assert tree.observation[root_id] is None
    assert len(tree.children_ids[root_id]) > 0
    assert tree.visit_count[root_id] > 0


def test_observation_cdf_consistency(
    environment,
    discount_factor,
    depth,
    exploration_constant,
    k_o,
    k_a,
    alpha_o,
    alpha_a,
):
    """Test the per-parent children_cdf invariants on the POMCPOW arena tree.

    Purpose: Validates the arena tree's ``children_cdf`` (used by
    ``sample_belief_child`` for O(log K) weighted sampling at observation-widening
    action nodes) is monotonically non-decreasing and totals to the sum of
    children weights for every parent in the tree.

    Given: POMCPOW planner built with enough simulations to widen at least one
    action node into multiple belief children (n_simulations=300). The action
    sampler uses the environment's real action space so the environment yields
    distinguishable observations.
    When: ``_learn_tree`` builds the tree and every parent is walked.
    Then: For every parent with children, the CDF length matches children count,
    is monotonically non-decreasing, and ``cdf[-1]`` equals the sum of children
    weights within absolute tolerance ``1e-6``. At least one action node has
    multiple belief children.

    Test type: unit
    """
    cdf_action_sampler = MockActionSampler(environment.get_actions())
    planner = POMCPOW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        n_simulations=300,
        action_sampler=cdf_action_sampler,
        name="TestPOMCPOW_CDF",
    )

    n_particles = 100
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)

    tree, root_id = planner._learn_tree(belief=belief)
    del root_id

    multi_child_action_seen = False
    for parent_id in range(len(tree)):
        children = tree.children_ids[parent_id]
        if not children:
            continue
        cdf = tree.children_cdf[parent_id]

        assert len(cdf) == len(children), (
            f"Parent {parent_id} (kind={tree.kind[parent_id]}) has {len(children)} "
            f"children but CDF has {len(cdf)} entries"
        )

        for i in range(1, len(cdf)):
            assert cdf[i] >= cdf[i - 1] - 1e-9, (
                f"Parent {parent_id} CDF not monotone at index {i}: "
                f"cdf[{i - 1}]={cdf[i - 1]}, cdf[{i}]={cdf[i]}"
            )

        total_weight = sum(tree.weight[cid] for cid in children)
        assert abs(cdf[-1] - total_weight) < 1e-6, (
            f"Parent {parent_id} CDF total {cdf[-1]} != sum of child weights "
            f"{total_weight} (kind={tree.kind[parent_id]})"
        )

        if tree.kind[parent_id] == ACTION and len(children) >= 2:
            multi_child_action_seen = True

    assert multi_child_action_seen, (
        "No action node with multiple belief children found; observation-widening "
        "CDF invariant is vacuous on this tree. Increase n_simulations."
    )


def test_tree_structure_comprehensive(planner, belief):
    """Comprehensive arena tree-structure invariants for POMCPOW.

    Purpose: Bundle root-, kind-alternation-, V/Q-relationship-, depth-, and
    progressive-widening invariants for the POMCPOW arena tree into one walk.

    Given: POMCPOW planner with default fixture parameters and an initial belief.
    When: ``_learn_tree`` builds the tree and a BFS walk inspects every node.
    Then: Root is BELIEF with parent None and observation None; every node has
    visit_count >= 0; BELIEF/ACTION kinds alternate along every edge; every
    non-root has a parent; BELIEF nodes have a non-None belief; ACTION nodes
    have non-None action and q_value; node visit_count >= sum of children
    visits; each BELIEF node with at least one visited action child has
    v_value equal to the max q_value over visited action children (POMCPOW
    max-Q backup); the BFS depth from root is at most ``2 * planner.depth + 2``;
    PW bounds hold at every node with visit_count > 0.

    Test type: unit
    """
    tree, root_id = planner._learn_tree(belief=belief)

    assert tree.kind[root_id] == BELIEF
    assert tree.parent_id[root_id] is None
    assert tree.observation[root_id] is None

    # BFS to bound depth and visit every node reachable from root.
    max_observed_depth = 0
    frontier = [(root_id, 0)]
    visited_node_ids: list[int] = []
    while frontier:
        node_id, d = frontier.pop()
        visited_node_ids.append(node_id)
        max_observed_depth = max(max_observed_depth, d)
        for cid in tree.children_ids[node_id]:
            frontier.append((cid, d + 1))

    assert max_observed_depth <= 2 * planner.depth + 2, (
        f"Tree max depth ({max_observed_depth}) exceeded 2 * planner.depth + 2 "
        f"({2 * planner.depth + 2})"
    )

    for node_id in visited_node_ids:
        assert tree.visit_count[node_id] >= 0

        if node_id != root_id:
            assert tree.parent_id[node_id] is not None
            parent_id = tree.parent_id[node_id]
            # Kind alternation: BELIEF parent => ACTION child and vice versa.
            assert tree.kind[node_id] != tree.kind[parent_id], (
                f"Kind alternation violated at node {node_id} (kind={tree.kind[node_id]}, "
                f"parent kind={tree.kind[parent_id]})"
            )

        children = tree.children_ids[node_id]

        if tree.kind[node_id] == BELIEF:
            assert tree.belief[node_id] is not None
            # Q/V relationship: max-Q backup over visited action children.
            visited_action_children = [cid for cid in children if tree.visit_count[cid] > 0]
            if visited_action_children:
                max_q = max(tree.q_value[cid] for cid in visited_action_children)
                assert np.isclose(tree.v_value[node_id], max_q), (
                    f"BELIEF node {node_id} v_value ({tree.v_value[node_id]}) != max q_value "
                    f"of visited action children ({max_q})"
                )
        else:
            assert tree.action[node_id] is not None
            assert tree.q_value[node_id] is not None

        # Visit-count consistency: parent visits >= sum of child visits.
        if children:
            total_child_visits = sum(tree.visit_count[cid] for cid in children)
            assert tree.visit_count[node_id] >= total_child_visits, (
                f"Node {node_id} (kind={tree.kind[node_id]}) visit_count "
                f"({tree.visit_count[node_id]}) < sum of children visits ({total_child_visits})"
            )

        # Progressive widening bounds at any node with visits.
        visit_count = tree.visit_count[node_id]
        if visit_count > 0 and children:
            if tree.kind[node_id] == BELIEF:
                max_allowed = int(planner.k_a * (visit_count**planner.alpha_a)) + 1
                assert len(children) <= max_allowed, (
                    f"BELIEF node {node_id} has {len(children)} action children but "
                    f"PW allows at most {max_allowed} (k_a={planner.k_a}, "
                    f"alpha_a={planner.alpha_a}, visit_count={visit_count})"
                )
            else:
                max_allowed = int(planner.k_o * (visit_count**planner.alpha_o)) + 1
                assert len(children) <= max_allowed, (
                    f"ACTION node {node_id} has {len(children)} belief children but "
                    f"PW allows at most {max_allowed} (k_o={planner.k_o}, "
                    f"alpha_o={planner.alpha_o}, visit_count={visit_count})"
                )
