# SPDX-License-Identifier: MIT

# pylint: disable=protected-access  # Tests need to access protected members
import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import (
    UnweightedParticleBeliefStateUpdate,
    WeightedParticleBelief,
    get_initial_belief,
)
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
    ContinuousLightDarkPOMDPDiscreteActions,
    RewardModelType,
)
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp_dpw import POMCP_DPW
from POMDPPlanners.planners.planners_utils.dpw import (
    ActionSampler,
    action_progressive_widening,
    ucb1_exploration,
)
from POMDPPlanners.planners.planners_utils.rollout import random_rollout_action_sampler
from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler

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
        name="TestPOMCP_DPW",
    )


@pytest.fixture
def belief(environment, n_particles):
    return get_initial_belief(pomdp=environment, n_particles=n_particles, resampling=True)


def test_pomcp_dpw_initialization_n_simulations_creates_configured_planner(
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
        name=expected_name,
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


def test_pomcp_dpw_initialization_timeout_creates_time_limited_planner(
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
        name=expected_name,
    )

    # ASSERT: Verify timeout-based configuration
    assert planner.environment == environment
    assert planner.discount_factor == discount_factor
    assert planner.depth == depth
    assert planner.exploration_constant == exploration_constant
    assert planner.time_out_in_seconds == expected_timeout
    assert planner.n_simulations is None  # Mutually exclusive with timeout
    assert planner.name == expected_name


def test_pomcp_dpw_initialization_both_termination_criteria_raises_error(
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
            name="TestPOMCP_DPW",
        )

    # ASSERT: Verify error message content
    assert (
        "mutually exclusive" in str(exc_info.value).lower() or "both" in str(exc_info.value).lower()
    )


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
    assert hasattr(policy_run_data, "info_variables")  # Contains MCTS tree statistics


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
        k_a=planner.k_a,
    )

    # ASSERT: Verify new action node created and properly linked
    assert isinstance(action_node, ActionNode)
    assert action_node.parent == belief_node
    assert action_node in belief_node.children
    assert len(belief_node.children) == initial_children_count + 1
    assert action_node.action in expected_action_space
    assert action_node.visit_count == 0  # Newly created node
    assert action_node.q_value == 0.0  # Initial Q-value


def test_action_progressive_widening_existing_action(planner, belief):
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
    # Test the random_rollout_action_sampler function that POMCP_DPW uses
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
    tree = Tree()
    root_id = tree.add_belief_node(belief)

    return_value = planner._simulate_path(tree=tree, belief_id=root_id, depth=0)
    assert isinstance(return_value, float)
    assert tree.visit_count[root_id] >= 1


def test_simulate_state_path_terminal_state(planner, belief):
    tree = Tree()
    root_id = tree.add_belief_node(belief)

    original_is_terminal = planner.environment.is_terminal
    planner.environment.is_terminal = lambda state: True

    state = belief.sample()
    return_value = planner._simulate_state_path(tree=tree, state=state, belief_id=root_id, depth=0)
    assert return_value == 0
    assert tree.visit_count[root_id] == 1

    planner.environment.is_terminal = original_is_terminal


def test_simulate_state_path_max_depth(planner, belief):
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    state = belief.sample()
    depth = planner.depth + 1

    return_value = planner._simulate_state_path(
        tree=tree, state=state, belief_id=root_id, depth=depth
    )
    assert return_value == 0


def test_simulate_state_path_saturated_branch_uses_parent_state_for_reward(
    planner, environment, belief
):
    """Saturated DPW branch must compute reward from the parent state, not the sampled next_state.

    Purpose: pins the reward semantics in the DPW saturated branch. The
    widening branch (line 158) computes ``environment.reward(state=state,
    action=action)`` from the *parent* state. A previous implementation in
    the saturated branch (line 199) used the sampled next_state instead, so
    the two execution paths produced different reward signals whenever
    progressive widening saturated — biasing Q-estimates against actions
    whose successor states have systematically different per-state rewards.

    Given: A POMCP_DPW tree with a root belief, a single action child
        (visit_count=1), and a single belief grandchild whose particle
        list contains a state distinct from the parent state. Planner
        configured with ``k_o=0.1`` / ``alpha_o=0.5`` so the saturated
        branch fires (1 child > 0.1 threshold), ``k_a=0.1`` /
        ``alpha_a=0.5`` so action widening does not fire (UCB picks the
        existing action child), and ``depth=0`` so the recursive
        ``_simulate_state_path`` call short-circuits at depth=1.
    When: ``_simulate_state_path`` runs from the root with the parent
        state.
    Then: ``environment.reward`` is invoked with the parent state, never
        with the next-state particle stored in the belief grandchild.

    Test type: unit
    """
    parent_state = "tiger_left"
    particle_state = "tiger_right"  # distinct from parent_state
    action_label = "listen"  # non-terminal for both states under TigerPOMDP

    planner.k_o = 0.1
    planner.alpha_o = 0.5
    planner.k_a = 0.1
    planner.alpha_a = 0.5
    planner.depth = 0

    tree = Tree()
    root_id = tree.add_belief_node(belief)
    tree.visit_count[root_id] = 1

    action_id = tree.add_action_node(action=action_label, parent_id=root_id)
    tree.visit_count[action_id] = 1

    child_belief = UnweightedParticleBeliefStateUpdate()
    child_belief.particles = [particle_state]
    child_id = tree.add_belief_node(
        belief=child_belief,
        observation="dummy_obs",
        parent_id=action_id,
        weight=1.0,
    )
    tree.visit_count[child_id] = 1

    captured_calls = []
    real_reward = environment.reward

    def spy_reward(state, action, next_state=None):
        captured_calls.append((state, next_state))
        return real_reward(state=state, action=action, next_state=next_state)

    environment.reward = spy_reward
    try:
        planner._simulate_state_path(tree=tree, state=parent_state, belief_id=root_id, depth=0)
    finally:
        environment.reward = real_reward

    captured_states = [s for s, _ in captured_calls]
    captured_next_states = [ns for _, ns in captured_calls]
    assert parent_state in captured_states, (
        f"saturated branch must call environment.reward(state=parent_state); "
        f"captured states: {captured_states}"
    )
    assert particle_state not in captured_states, (
        f"saturated branch must not call environment.reward with the sampled "
        f"next_state as the parent state; captured states: {captured_states}"
    )
    assert particle_state in captured_next_states, (
        f"saturated branch must thread the sampled next_state into "
        f"environment.reward(..., next_state=...); captured next_states: "
        f"{captured_next_states}"
    )


def test_simulate_state_path_widening_branch_threads_next_state_to_reward(
    planner, environment, belief
):
    """Widening DPW branch must thread the freshly sampled next_state into reward().

    Purpose: Pins reward semantics in the DPW widening branch
        (``pomcp_dpw.py`` line 158). After ``sample_next_state`` draws the
        post-transition state, ``environment.reward`` must be called with
        ``next_state=`` set to that draw so transition-dependent reward
        terms (collision penalties, win bonuses) score against the same
        outcome as the trajectory rather than resampling.

    Given: A POMCP_DPW tree whose root action child has zero existing
        belief children and ``visit_count=1``, with ``k_o=10`` /
        ``alpha_o=1.0`` so ``children_count <= k_o * action_visits**alpha_o``
        holds and the widening branch fires; ``k_a=0.1`` / ``alpha_a=0.5``
        so action widening does not add a new action; ``depth=0`` so the
        recursive call short-circuits at depth=1.
    When: ``_simulate_state_path`` runs from the root with ``parent_state``.
    Then: ``environment.reward`` is invoked with ``state=parent_state`` and
        ``next_state`` set to a non-None value (the just-sampled draw).

    Test type: unit
    """
    parent_state = "tiger_left"
    action_label = "listen"

    planner.k_o = 10.0
    planner.alpha_o = 1.0
    planner.k_a = 0.1
    planner.alpha_a = 0.5
    planner.depth = 0

    tree = Tree()
    root_id = tree.add_belief_node(belief)
    tree.visit_count[root_id] = 1

    action_id = tree.add_action_node(action=action_label, parent_id=root_id)
    tree.visit_count[action_id] = 1

    captured_calls = []
    real_reward = environment.reward

    def spy_reward(state, action, next_state=None):
        captured_calls.append((state, next_state))
        return real_reward(state=state, action=action, next_state=next_state)

    environment.reward = spy_reward
    try:
        planner._simulate_state_path(tree=tree, state=parent_state, belief_id=root_id, depth=0)
    finally:
        environment.reward = real_reward

    assert captured_calls, "widening branch did not invoke environment.reward"
    states = [s for s, _ in captured_calls]
    next_states = [ns for _, ns in captured_calls]
    assert parent_state in states, (
        f"widening branch must call environment.reward(state=parent_state); "
        f"captured states: {states}"
    )
    assert any(ns is not None for ns in next_states), (
        f"widening branch must thread the sampled next_state into "
        f"environment.reward(..., next_state=...); captured next_states: "
        f"{next_states}"
    )


def test_get_space_info(planner):
    space_info = planner.get_space_info()
    assert hasattr(space_info, "action_space")
    assert hasattr(space_info, "observation_space")
    assert space_info.action_space == SpaceType.MIXED
    assert space_info.observation_space == SpaceType.MIXED


def test_integration_with_tiger_pomdp(planner, belief, environment, n_particles):
    current_belief = belief
    for _ in range(3):
        action, _ = planner.action(current_belief)
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
    belief_node = BeliefNode(belief=belief, observation=None)
    belief_node.visit_count = 100

    # Test that progressive widening respects k_o and alpha_o parameters
    action_node = ActionNode(action=0, parent=belief_node)
    action_node.visit_count = 50

    # Test observation progressive widening condition
    max_observations = planner.k_o * action_node.visit_count**planner.alpha_o
    assert max_observations > 0

    # The condition should allow adding new observations if under the limit
    current_observations = len(action_node.children)
    can_add_observation = current_observations <= max_observations
    assert isinstance(can_add_observation, bool)


def test_belief_node_data_structure(planner, belief):
    """Test that belief nodes maintain proper belief structure for states and weights."""
    tree = Tree()
    root_id = tree.add_belief_node(belief)

    planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

    for action_id in tree.children_ids[root_id]:
        assert tree.kind[action_id] == ACTION
        for child_belief_id in tree.children_ids[action_id]:
            assert tree.kind[child_belief_id] == BELIEF
            child_belief = tree.belief[child_belief_id]
            assert isinstance(child_belief, UnweightedParticleBeliefStateUpdate)
            assert hasattr(child_belief, "particles")
            assert isinstance(child_belief.particles, list)


@pytest.mark.slow
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
        name="TestPOMCP_DPW",
    )

    belief = get_initial_belief(pomdp=environment, n_particles=100, resampling=True)

    # Run multiple trials to ensure consistent behavior
    n_trials = 10
    action_0_count = 0

    for _ in range(n_trials):
        action, _ = planner.action(belief)
        assert isinstance(action, list)
        assert len(action) == 1
        assert action[0] in [0, 1]
        if action[0] == 0:
            action_0_count += 1

    # Verify that action 0 (the better action) is selected most of the time
    # We expect at least 70% of the time to select action 0 (more lenient than POMCP)
    assert (
        action_0_count >= 0.7 * n_trials
    ), f"POMCP_DPW selected action 0 only {action_0_count}/{n_trials} times, expected at least {0.7 * n_trials}"


def test_tree_structure_after_construction(planner, belief):
    """Test that the arena tree structure is properly constructed."""
    tree = Tree()
    root_id = tree.add_belief_node(belief)

    for _ in range(50):
        planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

    action_ids = tree.children_ids[root_id]
    assert len(action_ids) > 0
    assert all(tree.kind[cid] == ACTION for cid in action_ids)

    for action_id in action_ids:
        assert tree.parent_id[action_id] == root_id
        assert tree.visit_count[action_id] > 0
        assert tree.q_value[action_id] is not None

        for belief_child_id in tree.children_ids[action_id]:
            assert tree.kind[belief_child_id] == BELIEF
            assert tree.parent_id[belief_child_id] == action_id


def test_q_value_updates(planner, belief):
    """Test that Q-values are properly updated during simulation on the arena tree."""
    tree = Tree()
    root_id = tree.add_belief_node(belief)

    planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

    action_updated = any(
        tree.kind[cid] == ACTION and tree.visit_count[cid] > 0 for cid in tree.children_ids[root_id]
    )
    assert action_updated, "No action node was updated during simulation"


def test_visit_count_consistency(planner, belief):
    """Test that visit counts are consistent throughout the arena tree."""
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


@pytest.mark.slow
def test_pomcp_dpw_vs_pomcp_differences(planner, belief):
    """Test POMCP_DPW progressive widening behavior on the arena tree."""
    tree = Tree()
    root_id = tree.add_belief_node(belief)

    for _ in range(100):
        planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

    action_ids = tree.children_ids[root_id]
    assert len(action_ids) > 0

    for action_id in action_ids:
        action_visits = tree.visit_count[action_id]
        if action_visits > 0:
            max_allowed = planner.k_o * action_visits**planner.alpha_o
            actual = len(tree.children_ids[action_id])
            assert actual <= max_allowed + 1


def test_unweighted_particle_belief_usage(planner, belief):
    """Test that POMCP_DPW properly uses unweighted particle beliefs (arena tree)."""
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    state = belief.sample()

    planner._simulate_state_path(tree=tree, state=state, belief_id=root_id, depth=0)

    for action_id in tree.children_ids[root_id]:
        for obs_belief_id in tree.children_ids[action_id]:
            obs_belief = tree.belief[obs_belief_id]
            assert isinstance(obs_belief, UnweightedParticleBeliefStateUpdate)
            assert hasattr(obs_belief, "particles")
            assert isinstance(obs_belief.particles, list)


@pytest.mark.slow
def test_double_progressive_widening_integration(planner, belief):
    """Test that both action and observation progressive widening work together on arena."""
    tree = Tree()
    root_id = tree.add_belief_node(belief)

    for _ in range(200):
        planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

    action_ids = [cid for cid in tree.children_ids[root_id] if tree.kind[cid] == ACTION]
    assert len(action_ids) > 0

    total_observations = 0
    for action_id in action_ids:
        total_observations += len(tree.children_ids[action_id])
        action_visits = tree.visit_count[action_id]
        if action_visits > 1:
            max_obs = planner.k_o * action_visits**planner.alpha_o
            assert len(tree.children_ids[action_id]) <= max_obs + 2

    assert total_observations >= 0


def test_basic_tree_structure(
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
        name="TestPOMCP_DPW_TreeStructure",
    )

    n_particles = 100
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)

    tree, root_id = planner._learn_tree(belief=belief)

    for node_id in range(len(tree)):
        if tree.kind[node_id] == BELIEF:
            assert tree.visit_count[node_id] == 1
            assert isinstance(tree.v_value[node_id], (int, float))
            if node_id == root_id:
                assert tree.observation[node_id] is None
                assert tree.parent_id[node_id] is None
            else:
                assert tree.observation[node_id] is not None
                parent_id = tree.parent_id[node_id]
                assert parent_id is not None
                assert tree.kind[parent_id] == ACTION
        elif tree.kind[node_id] == ACTION:
            assert tree.visit_count[node_id] == 1
            assert isinstance(tree.q_value[node_id], (int, float))
            parent_id = tree.parent_id[node_id]
            assert parent_id is not None
            assert tree.kind[parent_id] == BELIEF
            assert tree.action[node_id] is not None
        else:
            raise ValueError(f"Unknown kind: {tree.kind[node_id]}")


def test_pomcp_dpw_tree_structure_construction(
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
        name="TestPOMCP_DPW_TreeStructure",
    )

    n_particles = 100
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)

    tree, root_id = planner._learn_tree(belief=belief)

    # Inline arena-aware validation (replaces the legacy
    # validate_tree_structure_with_progressive_widening helper, which still
    # serves planners not yet migrated to the arena tree).
    del environment, discount_factor, depth, exploration_constant, k_o, alpha_o, action_sampler
    assert tree.kind[root_id] == BELIEF
    assert tree.visit_count[root_id] == n_simulations
    action_children = [cid for cid in tree.children_ids[root_id] if tree.kind[cid] == ACTION]
    assert len(action_children) > 0
    assert len(action_children) <= k_a * (n_simulations**alpha_a) + 1
    for action_id in action_children:
        for belief_id in tree.children_ids[action_id]:
            assert tree.kind[belief_id] == BELIEF
            assert isinstance(tree.belief[belief_id], UnweightedParticleBeliefStateUpdate)


def test_numpy_array_observation_comparison():
    """Test that POMCP_DPW correctly handles numpy array observation comparisons."""
    # Create a minimal environment for testing
    environment = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        name="TestObservationComparison",
        state_transition_cov_matrix=np.eye(2) * 0.1,
        observation_cov_matrix=np.eye(2) * 0.5,
        beacons=[(0, 0)],  # List of tuples as expected
        goal_state=np.array([1, 1]),
        start_state=np.array([0, 0]),
        obstacles=[(0.5, 0.5)],  # Add one obstacle to avoid broadcasting issues
        goal_reward=1.0,
        fuel_cost=0.1,
        grid_size=2,
        goal_state_radius=0.5,
        beacon_radius=1.0,
        reward_model_type=RewardModelType.CONSTANT_HAZARD_PENALTY,
    )

    # Test that the environment's is_equal_observation method works correctly
    obs1 = np.array([0.5, 0.5])
    obs2 = np.array([0.5, 0.5])
    obs3 = np.array([0.6, 0.5])

    # Test observation equality
    assert environment.is_equal_observation(obs1, obs2) is True
    assert environment.is_equal_observation(obs1, obs3) is False

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
        name="TestObservationComparison",
    )

    belief = UnweightedParticleBeliefStateUpdate(particles=[np.array([0.0, 0.0])])

    tree = Tree()
    root_id = tree.add_belief_node(belief)
    state = np.array([0.0, 0.0])

    try:
        return_value = planner._simulate_state_path(
            tree=tree, state=state, belief_id=root_id, depth=0
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


def test_pomcp_dpw_config_id_different_progressive_widening_parameters(
    environment, discrete_action_sampler
):
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
    for _ in range(3):
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
    assert all(c in "0123456789abcdef" for c in config_id.lower())


def test_min_visit_count_per_action_enforcement(environment, discrete_action_sampler):
    """Test that min_visit_count_per_action ensures minimum visits for each action node.

    Purpose: Validates that min_visit_count_per_action parameter correctly enforces minimum visit counts
    for each action node at the root when using k_a=2.0 and alpha_a=0.0

    Given: POMCP_DPW planner with k_a=2.0, alpha_a=0.0, and min_visit_count_per_action=5
    When: Planning is performed with sufficient simulations
    Then: All action nodes at the root have at least min_visit_count_per_action visits

    Test type: unit
    """
    # ARRANGE: Setup POMCP_DPW with k_a=2.0, alpha_a=0.0 to limit to 2 actions
    # and min_visit_count_per_action=5 to ensure each action gets at least 5 visits
    min_visit_count = 5
    planner = POMCP_DPW(
        environment=environment,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=2.0,  # With alpha_a=0.0, this allows max 2 actions
        alpha_o=0.5,
        alpha_a=0.0,  # alpha_a=0.0 means k_a * n^0 = k_a = 2.0 (constant)
        n_simulations=60,  # Enough simulations to reach min_visit_count
        action_sampler=discrete_action_sampler,
        name="TestPOMCP_DPW_MinVisit",
        min_visit_count_per_action=min_visit_count,
    )

    n_particles = 10
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)

    tree, root_id = planner._learn_tree(belief=belief)

    action_ids = [cid for cid in tree.children_ids[root_id] if tree.kind[cid] == ACTION]
    assert len(action_ids) >= 0

    for action_id in action_ids:
        visits = tree.visit_count[action_id]
        assert (
            visits >= min_visit_count
        ), f"Action {tree.action[action_id]} has {visits} visits, expected at least {min_visit_count}"


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
    """Test that the per-parent children_cdf invariants hold throughout the tree.

    Purpose: Validates the arena tree's ``children_cdf`` (used by
    ``sample_belief_child`` for O(log K) weighted sampling at observation-widening
    action nodes) is monotonically non-decreasing and totals to the sum of
    children weights for every parent in the tree.

    Given: POMCP_DPW planner built with sufficient simulations to expand at least
    one action node with multiple belief children (n_simulations=300). The action
    sampler uses the environment's real action space so that observations are not
    collapsed to a single sentinel by the environment.
    When: The tree is built via ``_learn_tree`` and every parent (belief and
    action) is walked.
    Then: For every parent with children, the CDF length matches children count,
    is monotonically non-decreasing, and ``cdf[-1]`` equals the sum of children
    weights within absolute tolerance ``1e-6``. At least one action node has
    multiple belief children (otherwise the OW invariant is vacuous).

    Test type: unit
    """
    cdf_action_sampler = MockActionSampler(environment.get_actions())
    planner = POMCP_DPW(
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
        name="TestPOMCP_DPW_CDF",
    )

    n_particles = 100
    belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)

    tree, root_id = planner._learn_tree(belief=belief)
    del root_id  # walk all parents in the arena instead of subtree-only

    multi_child_action_seen = False
    for parent_id in range(len(tree)):
        children = tree.children_ids[parent_id]
        if not children:
            continue
        cdf = tree.children_cdf[parent_id]

        # Length matches.
        assert len(cdf) == len(children), (
            f"Parent {parent_id} (kind={tree.kind[parent_id]}) has {len(children)} "
            f"children but CDF has {len(cdf)} entries"
        )

        # Monotonic non-decreasing (allow tiny float slack).
        for i in range(1, len(cdf)):
            assert cdf[i] >= cdf[i - 1] - 1e-9, (
                f"Parent {parent_id} CDF not monotone at index {i}: "
                f"cdf[{i - 1}]={cdf[i - 1]}, cdf[{i}]={cdf[i]}"
            )

        # Total equals sum of children weights.
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
