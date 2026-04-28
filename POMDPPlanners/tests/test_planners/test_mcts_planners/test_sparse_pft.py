# pylint: disable=protected-access  # Tests need to access protected members
import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree
from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT

np.random.seed(42)
random.seed(42)


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
    return get_initial_belief(pomdp=environment, n_particles=n_particles, resampling=True)


@pytest.fixture
def planner(
    environment,
    discount_factor,
    depth,
    c_ucb,
    beta_ucb,
    belief_child_num,
    n_simulations,
):
    return SparsePFT(
        environment=environment,
        discount_factor=discount_factor,
        gamma=discount_factor,
        depth=depth,
        c_ucb=c_ucb,
        beta_ucb=beta_ucb,
        belief_child_num=belief_child_num,
        n_simulations=n_simulations,
        time_out_in_seconds=None,
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

    Purpose: Validates that SparsePFT action selection returns valid TigerPOMDP actions through MCTS with progressive widening

    Given: SparsePFT planner with TigerPOMDP environment, initial belief with 100 particles, belief_child_num=2
    When: action method executes Sparse Partially Observable Forward Tree search
    Then: Returns single-element action list containing valid tiger action (listen/open_left/open_right) and PolicyRunData

    Test type: unit
    """
    action, _ = planner.action(belief=initial_belief)
    assert isinstance(action, list)
    assert len(action) == 1
    assert action[0] in planner.environment.get_actions()


def test_get_explored_action_node(planner):
    """Test that action node exploration works correctly

    Purpose: Validates that get_explored_action_node correctly selects action nodes based on UCB1 exploration strategy

    Given: BeliefNode with ActionNode children having different q_values (-1.0 vs 100.0), visit counts, and UCB parameters
    When: get_explored_action_node applies UCB1 selection formula
    Then: Selects ActionNode with optimal UCB value balancing exploitation and exploration

    Test type: unit
    """
    belief = WeightedParticleBelief(
        particles=["tiger_left", "tiger_right"],
        log_weights=np.array([-0.69314718, -0.69314718]),
    )
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    tree.visit_count[root_id] = 1

    for action in planner.environment.get_actions():
        action_id = tree.add_action_node(action=action, parent_id=root_id)
        tree.q_value[action_id] = -1.0
        tree.visit_count[action_id] = 1

    last_action_id = tree.children_ids[root_id][-1]
    tree.q_value[last_action_id] = 100.0

    selected_id = planner.get_explored_action_node(tree=tree, belief_id=root_id)
    assert tree.kind[selected_id] == ACTION
    assert selected_id in tree.children_ids[root_id]
    assert tree.action[selected_id] in planner.environment.get_actions()
    assert tree.q_value[selected_id] == 100.0


def test_sample_next_existing_belief(planner):
    """Test sampling from existing belief nodes

    Purpose: Validates sampling behavior for  next existing belief

    Given: Configured object with sampling capabilities
    When: Sample method is called
    Then: Valid samples are returned according to distribution

    Test type: unit
    """
    belief = WeightedParticleBelief(
        particles=["tiger_left", "tiger_right"],
        log_weights=np.array([-0.69314718, -0.69314718]),
    )
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    action_id = tree.add_action_node(action="listen", parent_id=root_id)

    for _ in range(2):
        child_id = tree.add_belief_node(belief=belief, observation="hear_left", parent_id=action_id)
        tree.set_immediate_cost(child_id, -1.0)
        tree.visit_count[child_id] = 1

    next_belief_id, immediate_reward = planner._sample_next_existing_belief(
        tree=tree, action_id=action_id
    )
    assert tree.kind[next_belief_id] == BELIEF
    assert next_belief_id in tree.children_ids[action_id]
    assert tree.immediate_cost[next_belief_id] is not None
    assert immediate_reward == -tree.immediate_cost[next_belief_id]


def test_generate_belief(planner):
    """Test generating a new belief node

    Purpose: Validates that _generate_belief creates new BeliefNode children with proper observation sampling and cost computation

    Given: SparsePFT planner, BeliefNode and ActionNode with WeightedParticleBelief containing tiger states
    When: _generate_belief samples new observation and creates belief update
    Then: Returns new BeliefNode with correct parent, non-null observation, immediate_cost, and reward = -cost

    Test type: unit
    """
    belief = WeightedParticleBelief(
        particles=["tiger_left", "tiger_right"],
        log_weights=np.array([-0.69314718, -0.69314718]),
    )
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    action_id = tree.add_action_node(action="listen", parent_id=root_id)

    next_belief_id, immediate_reward = planner._generate_belief(tree=tree, action_id=action_id)
    assert tree.kind[next_belief_id] == BELIEF
    assert tree.parent_id[next_belief_id] == action_id
    assert tree.observation[next_belief_id] is not None
    assert tree.immediate_cost[next_belief_id] is not None
    assert immediate_reward == -tree.immediate_cost[next_belief_id]


def test_random_rollout(planner):
    """Test random rollout from a state

    Purpose: Validates that random rollout simulation returns rewards within expected TigerPOMDP bounds for value estimation

    Given: SparsePFT planner with TigerPOMDP environment, initial state "tiger_left", rollout depth=0
    When: random_rollout performs simulation from given state with random policy
    Then: Returns float reward within bounds [-500, 50] accounting for tiger environment reward structure

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
    belief = WeightedParticleBelief(
        particles=["tiger_left", "tiger_right"],
        log_weights=np.array([-0.69314718, -0.69314718]),
    )
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    tree.visit_count[root_id] = 1

    action_id = tree.add_action_node(action="listen", parent_id=root_id)
    tree.visit_count[action_id] = 1
    tree.q_value[action_id] = 0.0

    return_sample = -1.0
    planner.update_nodes(
        tree=tree, belief_id=root_id, action_id=action_id, return_sample=return_sample
    )

    assert tree.visit_count[root_id] == 2
    assert tree.visit_count[action_id] == 2
    assert tree.q_value[action_id] == -0.5  # (0.0 + -1.0) / 2
    assert tree.v_value[root_id] is not None


def test_integration_with_tiger_pomdp(planner, initial_belief, environment, n_particles):
    """Test integration with Tiger POMDP environment

    Purpose: Validates that SparsePFT integrates correctly with TigerPOMDP for complete POMDP planning workflow including belief updates

    Given: SparsePFT planner, TigerPOMDP environment, initial belief with 100 particles, 5 planning steps
    When: Full planning cycle executes including action selection, environment steps, and belief updates
    Then: Valid tiger actions selected, belief updates preserve particle count, and environment state transitions work correctly

    Test type: integration
    """
    current_belief = initial_belief
    for _ in range(5):
        # Create a belief node with children
        belief_node = BeliefNode(
            belief=current_belief, observation=None, parent=None, children=tuple()
        )

        # Add action nodes
        for action in planner.environment.get_actions():
            action_node = ActionNode(action=action, parent=belief_node, children=tuple())
            action_node.q_value = -1.0
            action_node.visit_count = 1

        action, _ = planner.action(current_belief)
        assert isinstance(action, list)
        assert len(action) == 1
        assert action[0] in environment.get_actions()

        # Simulate environment step
        state = current_belief.sample()
        next_state = environment.sample_next_state(state=state, action=action[0])
        next_observation = environment.sample_observation(next_state=next_state, action=action[0])

        # Update belief
        current_belief = current_belief.update(
            action=action[0], observation=next_observation, pomdp=environment
        )

        # Verify belief is valid
        assert isinstance(current_belief, WeightedParticleBelief)
        assert len(current_belief.particles) == n_particles


def test_tree_structure_construction(planner, initial_belief, environment):
    """Test that the tree structure is constructed correctly

    Purpose: Validates that SparsePFT builds proper MCTS tree structure with correct belief-action hierarchy and progressive widening

    Given: SparsePFT planner with belief_child_num=2, depth=3, TigerPOMDP environment, initial belief
    When: _learn_tree constructs MCTS tree with belief and action nodes
    Then: Tree has correct structure (root BeliefNode, action children, belief grandchildren), visit counts, progressive widening limits, and depth=2*depth+2

    Test type: unit
    """
    tree, root_id = planner._learn_tree(belief=initial_belief)

    assert tree.kind[root_id] == BELIEF
    assert tree.belief[root_id] == initial_belief
    assert tree.observation[root_id] is None
    assert tree.parent_id[root_id] is None
    assert len(tree.children_ids[root_id]) == len(environment.get_actions())
    assert tree.visit_count[root_id] > 0
    assert tree.v_value[root_id] is not None

    # Per-node invariants on the arena tree.
    for node_id in range(len(tree)):
        assert tree.visit_count[node_id] >= 0
        if tree.kind[node_id] == BELIEF:
            assert tree.belief[node_id] is not None
            assert tree.v_value[node_id] is not None
        elif tree.kind[node_id] == ACTION:
            assert tree.action[node_id] is not None
            assert tree.q_value[node_id] is not None
            if tree.children_ids[node_id]:
                assert len(tree.children_ids[node_id]) <= planner.belief_child_num

    # Walk the arena tree to compute max depth (longest path from root in node-id steps).
    # Arena tree alternates belief/action levels. With self.depth=N, the recursion creates
    # belief nodes at depths 0, 2, ..., 2N and action nodes at 1, 3, ..., 2N+1, plus one
    # final belief child at depth 2N+2 created before the next call hits the depth bound.
    # So the deepest node sits at 2 * planner.depth + 2 (matches POMCP's analogous test).
    max_observed_depth = 0
    frontier = [(root_id, 0)]
    while frontier:
        nid, d = frontier.pop()
        max_observed_depth = max(max_observed_depth, d)
        for cid in tree.children_ids[nid]:
            frontier.append((cid, d + 1))
    assert max_observed_depth == 2 * planner.depth + 2

    # All root action children explored.
    root_actions = tree.children_ids[root_id]
    assert len(root_actions) == len(environment.get_actions())
    assert all(tree.kind[cid] == ACTION for cid in root_actions)
    assert all(tree.visit_count[cid] > 0 for cid in root_actions)


@pytest.mark.slow
def test_sanity_pomdp_action_selection():
    """Test that SparsePFT correctly identifies the better action in SanityPOMDP

    Purpose: Validates that SparsePFT handles SanityPOMDP environment with deterministic reward structure and finds optimal actions

    Given: SanityPOMDP environment with clear optimal actions, SparsePFT with 1000 simulations and belief_child_num=3
    When: MCTS tree search explores action space with sufficient simulations
    Then: Selected action is valid for SanityPOMDP environment and planning completes successfully

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
        n_simulations=1000,  # More simulations for better accuracy
        time_out_in_seconds=None,
    )

    # Get initial belief
    belief = get_initial_belief(pomdp=environment, n_particles=10, resampling=True)

    # Run multiple trials to ensure consistent behavior
    n_trials = 10
    action_0_count = 0

    for _ in range(n_trials):
        action, _ = planner.action(belief)
        assert isinstance(action, list)
        assert len(action) == 1
        if action[0] == 0:  # Count how many times action 0 is selected
            action_0_count += 1

    # Verify that action 0 (the better action) is selected most of the time
    # We expect at least 70% success rate since SparsePFT combines MCTS with particle filtering
    assert (
        action_0_count >= 0.7 * n_trials
    ), f"SparsePFT selected action 0 only {action_0_count}/{n_trials} times, expected at least {0.7 * n_trials}"


def test_sanity_pomdp_belief_children():
    """Test that SparsePFT generates appropriate belief children for SanityPOMDP

    Purpose: Validates that SparsePFT generates proper belief children for SanityPOMDP environment with binary observations

    Given: SanityPOMDP environment with binary observations [0,1], SparsePFT with belief_child_num=3, initial belief with 100 particles
    When: _generate_belief creates belief children for action=0 (better action)
    Then: Generated BeliefNode has correct parent, WeightedParticleBelief with 100 particles, binary observation, and non-null immediate cost

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
        n_simulations=10,
        time_out_in_seconds=None,
    )

    belief = get_initial_belief(pomdp=environment, n_particles=100, resampling=True)

    tree = Tree()
    root_id = tree.add_belief_node(belief)
    action_id = tree.add_action_node(action=0, parent_id=root_id)  # better action

    for _ in range(3):
        next_belief_id, _ = planner._generate_belief(tree=tree, action_id=action_id)

        assert tree.kind[next_belief_id] == BELIEF
        assert tree.parent_id[next_belief_id] == action_id
        assert isinstance(tree.belief[next_belief_id], WeightedParticleBelief)
        assert len(tree.belief[next_belief_id].particles) == 100
        assert tree.observation[next_belief_id] in [0, 1]  # SanityPOMDP binary obs
        assert tree.immediate_cost[next_belief_id] is not None


# Config ID Tests


def test_sparse_pft_config_id_consistency_identical_parameters():
    """Test that config_id is consistent for identical SparsePFT parameters.

    Purpose: Validates that SparsePFT with identical parameters produces identical config_id

    Given: Two SparsePFT instances with identical parameters
    When: config_id is accessed on both instances
    Then: Both instances return the same config_id

    Test type: unit
    """
    environment = TigerPOMDP(discount_factor=0.95)

    # Create two SparsePFT instances with identical parameters
    sparse_pft1 = SparsePFT(
        environment=environment,
        discount_factor=0.95,
        gamma=0.95,
        depth=12,
        c_ucb=1.0,
        beta_ucb=2.0,
        belief_child_num=5,
        n_simulations=100,
        time_out_in_seconds=None,
        name="SparsePFT_Test1",
    )

    sparse_pft2 = SparsePFT(
        environment=environment,
        discount_factor=0.95,
        gamma=0.95,
        depth=12,
        c_ucb=1.0,
        beta_ucb=2.0,
        belief_child_num=5,
        n_simulations=100,
        time_out_in_seconds=None,
        name="SparsePFT_Test1",  # Same name
    )

    # Config IDs should be identical
    config_id1 = sparse_pft1.config_id
    config_id2 = sparse_pft2.config_id

    assert config_id1 == config_id2
    assert isinstance(config_id1, str)
    assert len(config_id1) > 0


def test_sparse_pft_config_id_different_c_ucb():
    """Test that config_id changes when c_ucb parameter differs.

    Purpose: Validates that config_id changes when c_ucb exploration parameter differs

    Given: Two SparsePFT instances with different c_ucb values
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
    environment = TigerPOMDP(discount_factor=0.95)

    sparse_pft1 = SparsePFT(
        environment=environment,
        discount_factor=0.95,
        gamma=0.95,
        depth=12,
        c_ucb=1.0,
        beta_ucb=2.0,
        belief_child_num=5,
        n_simulations=100,
        time_out_in_seconds=None,
        name="SparsePFT_Test",
    )

    sparse_pft2 = SparsePFT(
        environment=environment,
        discount_factor=0.95,
        gamma=0.95,
        depth=12,
        c_ucb=1.5,  # Different c_ucb
        beta_ucb=2.0,
        belief_child_num=5,
        n_simulations=100,
        time_out_in_seconds=None,
        name="SparsePFT_Test",
    )

    config_id1 = sparse_pft1.config_id
    config_id2 = sparse_pft2.config_id

    assert config_id1 != config_id2


def test_sparse_pft_config_id_different_belief_child_num():
    """Test that config_id changes when belief_child_num parameter differs.

    Purpose: Validates that config_id changes when belief_child_num branching parameter differs

    Given: Two SparsePFT instances with different belief_child_num values
    When: config_id is accessed on both instances
    Then: config_id values are different

    Test type: unit
    """
    environment = TigerPOMDP(discount_factor=0.95)

    sparse_pft1 = SparsePFT(
        environment=environment,
        discount_factor=0.95,
        gamma=0.95,
        depth=12,
        c_ucb=1.0,
        beta_ucb=2.0,
        belief_child_num=5,
        n_simulations=100,
        time_out_in_seconds=None,
        name="SparsePFT_Test",
    )

    sparse_pft2 = SparsePFT(
        environment=environment,
        discount_factor=0.95,
        gamma=0.95,
        depth=12,
        c_ucb=1.0,
        beta_ucb=2.0,
        belief_child_num=10,  # Different belief_child_num
        n_simulations=100,
        time_out_in_seconds=None,
        name="SparsePFT_Test",
    )

    config_id1 = sparse_pft1.config_id
    config_id2 = sparse_pft2.config_id

    assert config_id1 != config_id2


def test_sparse_pft_config_id_consistency_across_evaluations():
    """Test that config_id remains consistent across different policy evaluations.

    Purpose: Validates that config_id is stable across multiple accesses and policy actions

    Given: Single SparsePFT instance and initial belief
    When: config_id is accessed before and after policy actions
    Then: config_id remains identical across all evaluations

    Test type: integration
    """
    environment = TigerPOMDP(discount_factor=0.95)

    sparse_pft = SparsePFT(
        environment=environment,
        discount_factor=0.95,
        gamma=0.95,
        depth=5,  # Reduced for testing
        c_ucb=1.0,
        beta_ucb=2.0,
        belief_child_num=3,  # Reduced for testing
        n_simulations=10,  # Reduced for testing
        time_out_in_seconds=None,
        name="SparsePFT_Consistency_Test",
    )

    # Get initial config_id
    initial_config_id = sparse_pft.config_id

    # Create initial belief and perform policy actions
    initial_belief = get_initial_belief(environment, n_particles=50)

    # Perform multiple policy evaluations
    for _ in range(3):
        action, run_data = sparse_pft.action(initial_belief)

        # Check config_id remains the same
        current_config_id = sparse_pft.config_id
        assert current_config_id == initial_config_id

        # Verify the action and run_data are valid
        assert isinstance(action, list)
        assert len(action) == 1
        assert action[0] in environment.get_actions()
        assert run_data is not None

    # Final check
    final_config_id = sparse_pft.config_id
    assert final_config_id == initial_config_id


def test_sparse_pft_config_id_hash_properties():
    """Test that config_id has proper hash properties.

    Purpose: Validates that config_id produces valid hash strings

    Given: SparsePFT instance
    When: config_id is accessed
    Then: config_id is a valid hash string with expected properties

    Test type: unit
    """
    environment = TigerPOMDP(discount_factor=0.95)

    sparse_pft = SparsePFT(
        environment=environment,
        discount_factor=0.95,
        gamma=0.95,
        depth=12,
        c_ucb=1.0,
        beta_ucb=2.0,
        belief_child_num=5,
        n_simulations=100,
        time_out_in_seconds=None,
        name="SparsePFT_Hash_Test",
    )

    config_id = sparse_pft.config_id

    # Should be a non-empty string
    assert isinstance(config_id, str)
    assert len(config_id) > 0

    # Should be a valid hexadecimal hash (SHA-256 produces 64 hex characters)
    assert len(config_id) == 64
    assert all(c in "0123456789abcdef" for c in config_id.lower())


def test_q_value_v_value_consistency(planner, initial_belief):
    """Verify ``v_value`` of every BELIEF node equals max ``q_value`` over its children.

    Purpose: Pins down the contract enforced by ``update_nodes`` in SparsePFT:
        after each backup, ``tree.v_value[belief_id]`` equals the max
        ``tree.q_value[c]`` across all children of that belief node. This
        guards against silent drift if a future refactor changes the
        aggregator or filters children by visit count.

    Given: The ``planner`` fixture (SparsePFT on TigerPOMDP, depth=3,
        belief_child_num=2, n_simulations=100) and the ``initial_belief``
        fixture; ``_learn_tree`` builds the arena tree.
    When: Every BELIEF node with non-empty children is enumerated and the
        observed ``tree.v_value[belief_id]`` is compared against
        ``max(tree.q_value[c] for c in tree.children_ids[belief_id])`` taken
        over ALL action children (no visit-count filter), matching the
        ``update_nodes`` source behavior.
    Then: The values agree within ``atol=1e-9`` (float-tolerant equality via
        ``pytest.approx``).

    Test type: unit
    """
    tree, root_id = planner._learn_tree(belief=initial_belief)
    assert tree.kind[root_id] == BELIEF  # sanity: root is a belief node

    n_nodes = len(tree)
    n_belief_nodes_checked = 0
    for node_id in range(n_nodes):
        if tree.kind[node_id] != BELIEF:
            continue
        children = tree.children_ids[node_id]
        if not children:
            continue
        expected_v = max(tree.q_value[c] for c in children)
        observed_v = tree.v_value[node_id]
        assert observed_v == pytest.approx(expected_v, abs=1e-9)
        n_belief_nodes_checked += 1

    # Sanity: at least the root should have been checked.
    assert n_belief_nodes_checked >= 1
