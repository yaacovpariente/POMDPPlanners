# pylint: disable=protected-access
from logging import getLogger

import numpy as np
import pytest

from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.planners.mcts_planners.icvar_pft_dpw import ICVaR_PFT_DPW


@pytest.fixture
def env():
    return ContinuousLightDarkPOMDP(discount_factor=0.95)


@pytest.fixture
def belief():
    # Create a belief with particles in the continuous state space
    particles = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]  # Example particles in 2D space
    log_weights = np.log(np.ones(len(particles)) / len(particles))
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


@pytest.fixture
def action_sampler():
    return UnitCircleActionSampler()


@pytest.fixture
def planner(env, action_sampler):
    return ICVaR_PFT_DPW(
        environment=env,
        name="test_planner",
        depth=3,
        discount_factor=0.95,
        n_simulations=100,
        alpha=0.1,
        delta=0.1,
        k_o=5,
        min_immediate_cost=0.0,
        max_immediate_cost=1.0,
        min_visit_count_per_action=1,
        exploration_constant=1.0,
        action_sampler=action_sampler,
        k_a=2.0,
        alpha_a=0.0,
        alpha_o=0.0,
    )


def test_initialization(planner):
    assert planner.alpha == 0.1
    assert planner.delta == 0.1
    assert planner.depth == 3
    assert planner.discount_factor == 0.95
    assert planner.k_o == 5
    assert planner.min_immediate_cost == 0.0
    assert planner.max_immediate_cost == 1.0


def test_is_terminal_belief(planner, env):
    # Non-terminal belief (particles not at goal)
    particles = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    log_weights = np.log(np.ones(len(particles)) / len(particles))
    non_terminal_belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
    assert not planner.is_terminal_belief(non_terminal_belief)

    # Terminal belief (particles at goal)
    # Get the goal state from the environment
    goal_state = env.goal_state
    particles = [goal_state, goal_state]  # Both particles at goal
    log_weights = np.log(np.ones(len(particles)) / len(particles))
    terminal_belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
    assert planner.is_terminal_belief(terminal_belief)

    # Mixed belief
    particles = [np.array([0.0, 0.0]), goal_state]
    log_weights = np.log(np.ones(len(particles)) / len(particles))
    mixed_belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
    assert not planner.is_terminal_belief(mixed_belief)


def test_generate_belief(planner, belief):
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    action = np.array([1.0, 0.0])
    action_id = tree.add_action_node(action=action, parent_id=root_id)

    next_belief_id = planner._generate_belief(tree=tree, action_id=action_id)

    assert tree.kind[next_belief_id] == BELIEF
    assert tree.parent_id[next_belief_id] == action_id


def test_sample_next_existing_belief(planner, belief):
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    action = np.array([1.0, 0.0])
    action_id = tree.add_action_node(action=action, parent_id=root_id)

    for i in range(3):
        particles = [np.array([i * 1.0, i * 1.0]), np.array([(i + 1) * 1.0, (i + 1) * 1.0])]
        log_weights = np.log(np.ones(len(particles)) / len(particles))
        child_id = tree.add_belief_node(
            belief=WeightedParticleBelief(particles=particles, log_weights=log_weights),
            parent_id=action_id,
        )
        tree.visit_count[child_id] = i
        tree.set_immediate_cost(child_id, -float(i))

    next_belief_id = planner._sample_next_existing_belief(tree=tree, action_id=action_id)

    assert tree.kind[next_belief_id] == BELIEF
    assert next_belief_id in tree.children_ids[action_id]


def test_update_nodes(planner, belief):
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    action = np.array([1.0, 0.0])
    action_id = tree.add_action_node(action=action, parent_id=root_id)

    for i in range(3):
        particles = [np.array([i * 1.0, i * 1.0]), np.array([(i + 1) * 1.0, (i + 1) * 1.0])]
        log_weights = np.log(np.ones(len(particles)) / len(particles))
        child_id = tree.add_belief_node(
            belief=WeightedParticleBelief(particles=particles, log_weights=log_weights),
            parent_id=action_id,
        )
        tree.visit_count[child_id] = i
        tree.v_value[child_id] = -float(i)

    initial_visit_count = tree.visit_count[root_id]
    planner.update_nodes(tree=tree, belief_id=root_id, action_id=action_id)

    assert tree.visit_count[root_id] == initial_visit_count + 1
    assert tree.visit_count[action_id] == 1
    assert tree.q_value[action_id] is not None


def test_get_space_info(planner):
    space_info = planner.get_space_info()
    assert space_info.action_space == SpaceType.CONTINUOUS
    assert space_info.observation_space == SpaceType.CONTINUOUS


def test_immediate_cost_computed_once_per_action(planner, belief):
    """After _generate_belief, the action node carries the (parent_belief, action) cost
    and update_nodes does not recompute it."""
    tree = Tree()
    root_id = tree.add_belief_node(belief)
    action = np.array([1.0, 0.0])
    action_id = tree.add_action_node(action=action, parent_id=root_id)

    assert tree.immediate_cost[action_id] is None
    planner._generate_belief(tree=tree, action_id=action_id)
    after_generate = tree.immediate_cost[action_id]
    assert after_generate is not None

    planner.update_nodes(tree=tree, belief_id=root_id, action_id=action_id)
    assert tree.immediate_cost[action_id] == after_generate


def test_action(planner, belief):
    actions, policy_data = planner.action(belief)
    assert isinstance(actions, list)
    assert len(actions) == 1
    assert isinstance(actions[0], np.ndarray)  # For continuous actions
    assert actions[0].shape == (2,)  # 2D action vector
    assert hasattr(policy_data, "info_variables")  # Verify PolicyRunData structure


def test_progressive_widening_constraints(planner, belief):
    """Test that progressive widening constraints are respected in the tree."""
    tree, _ = planner._learn_tree(belief=belief)

    belief_count = 0
    action_count = 0
    for node_id in range(len(tree)):
        children = tree.children_ids[node_id]
        if tree.kind[node_id] == BELIEF:
            belief_count += 1
            max_allowed = int(planner.k_a + 1)
            assert len(children) <= max_allowed, (
                f"Belief node {node_id} has {len(children)} children but should have at most "
                f"{max_allowed} (k_a={planner.k_a}, visit_count={tree.visit_count[node_id]}, "
                f"alpha_a={planner.alpha_a})"
            )
        else:
            action_count += 1
            max_allowed = int(planner.k_o + 1)
            assert len(children) <= max_allowed, (
                f"Action node {node_id} has {len(children)} children but should have at most "
                f"{max_allowed} (k_o={planner.k_o}, visit_count={tree.visit_count[node_id]}, "
                f"alpha_o={planner.alpha_o})"
            )

    assert belief_count >= 1, "Tree should contain at least the root belief node"
    assert action_count >= 0, "Tree should contain zero or more action nodes"


def test_tree_structure_comprehensive(planner, belief):
    """Comprehensive structural validation of the iCVaR PFT-DPW search tree.

    Purpose: Validates that the search tree built by ICVaR_PFT_DPW respects all
    structural invariants: root metadata, BELIEF/ACTION kind alternation, parent
    pointers, value/visit-count consistency, the cost-channel V=min(Q) backup at
    BELIEF nodes (note: min, not max, since iCVaR is a cost channel), BFS depth
    bounds, and progressive widening bounds for both action and observation
    branches.

    Given: An ICVaR_PFT_DPW planner with progressive-widening parameters and a
    WeightedParticleBelief with two particles in the continuous light-dark POMDP.
    When: A full search tree is constructed via planner._learn_tree(belief) and
    every node is visited via BFS with depth tracking.
    Then: All structural invariants hold simultaneously across every node in
    the tree, and the cost-channel V=min(Q) backup matches within float
    tolerance at every BELIEF node with at least one visited action child.

    Test type: unit
    """
    tree, root_id = planner._learn_tree(belief=belief)

    # Root-level invariants
    assert tree.kind[root_id] == BELIEF
    assert tree.parent_id[root_id] is None
    assert tree.observation[root_id] is None
    assert len(tree.children_ids[root_id]) > 0

    # BFS walk tracking depth
    max_bfs_depth = 0
    frontier: list[tuple[int, int]] = [(root_id, 0)]
    while frontier:
        node_id, depth = frontier.pop(0)
        max_bfs_depth = max(max_bfs_depth, depth)
        _assert_node_invariants(tree, node_id, root_id)
        _assert_pw_bounds(tree, node_id, planner)
        for child_id in tree.children_ids[node_id]:
            frontier.append((child_id, depth + 1))

    max_depth_bound = 2 * planner.depth + 2
    assert max_bfs_depth <= max_depth_bound, (
        f"Max BFS depth {max_bfs_depth} exceeds bound {max_depth_bound} "
        f"(2 * planner.depth + 2 with planner.depth={planner.depth})"
    )


def _assert_node_invariants(tree, node_id, root_id):
    assert (
        tree.visit_count[node_id] >= 0
    ), f"Node {node_id} has negative visit_count {tree.visit_count[node_id]}"
    if node_id != root_id:
        assert tree.parent_id[node_id] is not None, f"Non-root node {node_id} has no parent"

    children = tree.children_ids[node_id]
    node_kind = tree.kind[node_id]

    if node_kind == BELIEF:
        assert tree.belief[node_id] is not None, f"BELIEF node {node_id} has no belief"
        assert tree.v_value[node_id] is not None, f"BELIEF node {node_id} has no v_value"
        for child_id in children:
            assert (
                tree.kind[child_id] == ACTION
            ), f"Child {child_id} of BELIEF node {node_id} is not ACTION"
        _assert_v_min_backup(tree, node_id, children)
    else:
        assert node_kind == ACTION
        assert tree.action[node_id] is not None, f"ACTION node {node_id} has no action"
        assert tree.q_value[node_id] is not None, f"ACTION node {node_id} has no q_value"
        for child_id in children:
            assert (
                tree.kind[child_id] == BELIEF
            ), f"Child {child_id} of ACTION node {node_id} is not BELIEF"

    if children:
        children_visit_sum = sum(tree.visit_count[c] for c in children)
        assert tree.visit_count[node_id] >= children_visit_sum, (
            f"Node {node_id} visit_count {tree.visit_count[node_id]} is less than "
            f"sum of children visit counts {children_visit_sum}"
        )


def _assert_v_min_backup(tree, belief_node_id, children):
    visited_action_children = [c for c in children if tree.visit_count[c] > 0]
    if not visited_action_children:
        return
    expected_v = min(tree.q_value[c] for c in visited_action_children)
    actual_v = tree.v_value[belief_node_id]
    assert actual_v == pytest.approx(expected_v), (
        f"BELIEF node {belief_node_id}: v_value={actual_v} does not equal "
        f"min over visited action children q_values={expected_v}"
    )


def _assert_pw_bounds(tree, node_id, planner):
    visit_count = tree.visit_count[node_id]
    if visit_count <= 0:
        return
    children = tree.children_ids[node_id]
    if tree.kind[node_id] == BELIEF:
        max_allowed = int(planner.k_a * (visit_count**planner.alpha_a)) + 1
        assert len(children) <= max_allowed, (
            f"BELIEF node {node_id} has {len(children)} children, exceeds "
            f"PW bound {max_allowed} (k_a={planner.k_a}, alpha_a={planner.alpha_a}, "
            f"visit_count={visit_count})"
        )
    else:
        max_allowed = int(planner.k_o * (visit_count**planner.alpha_o)) + 1
        assert len(children) <= max_allowed, (
            f"ACTION node {node_id} has {len(children)} children, exceeds "
            f"PW bound {max_allowed} (k_o={planner.k_o}, alpha_o={planner.alpha_o}, "
            f"visit_count={visit_count})"
        )


class TestICVaR_PFT_DPWEpisodeTests:
    """Test class for ICVaR_PFT_DPW episode execution tests."""

    def test_run_episode_continuous_light_dark_pomdp(self, planner, belief, env):
        """Test running a single episode with ContinuousLightDarkPOMDP environment."""
        logger = getLogger(__name__)
        num_steps = 10

        # Run the episode
        history = run_episode(
            environment=env,
            policy=planner,
            initial_belief=belief,
            num_steps=num_steps,
            logger=logger,
        )

        # Verify episode results
        assert history is not None
        assert len(history.history) > 0
        assert len(history.history) <= num_steps

        # Check that each step has required data
        for step in history.history:
            assert hasattr(step, "state")
            assert hasattr(step, "action")
            assert hasattr(step, "observation")
            assert hasattr(step, "reward")
            assert hasattr(step, "belief")

            # Verify action is valid for environment (continuous actions)
            if isinstance(step.action, list) and len(step.action) == 1:
                actual_action = step.action[0]
            else:
                actual_action = step.action

            # Skip None actions (they might occur in some edge cases)
            if actual_action is not None:
                # For continuous actions, check that it's a numpy array with correct shape
                assert isinstance(actual_action, np.ndarray)
                action_array: np.ndarray = actual_action
                assert action_array.shape == (2,)  # 2D action vector for ContinuousLightDarkPOMDP

        # Check episode summary
        assert hasattr(history, "actual_num_steps")
        assert hasattr(history, "reach_terminal_state")
        assert history.actual_num_steps > 0
        assert isinstance(history.reach_terminal_state, bool)

        # Verify timing metrics
        assert hasattr(history, "average_action_time")
        assert hasattr(history, "average_state_sampling_time")
        assert hasattr(history, "average_observation_time")
        assert hasattr(history, "average_belief_update_time")
        assert hasattr(history, "average_reward_time")

    def test_run_episode_different_alpha_values(self, env, action_sampler):
        """Test running episodes with different alpha values."""
        logger = getLogger(__name__)
        num_steps = 5
        belief = get_initial_belief(env, n_particles=20, resampling=True)

        alpha_values = [0.05, 0.1, 0.25, 0.5]

        for alpha in alpha_values:
            # Create planner with specific alpha
            planner = ICVaR_PFT_DPW(
                environment=env,
                name=f"test_planner_alpha_{alpha}",
                depth=3,
                discount_factor=0.95,
                n_simulations=50,  # Reduced for faster testing
                alpha=alpha,
                delta=0.1,
                k_o=5,
                min_immediate_cost=0.0,
                max_immediate_cost=1.0,
                min_visit_count_per_action=1,
                exploration_constant=1.0,
                action_sampler=action_sampler,
                k_a=2.0,
                alpha_a=0.0,
                alpha_o=0.0,
            )

            # Run the episode
            history = run_episode(
                environment=env,
                policy=planner,
                initial_belief=belief,
                num_steps=num_steps,
                logger=logger,
            )

            # Verify episode results
            assert history is not None
            assert len(history.history) > 0
            assert len(history.history) <= num_steps

            # Check that planner's alpha was used
            assert planner.alpha == alpha

    @pytest.mark.slow
    def test_run_episode_timeout_vs_n_simulations(self, env, action_sampler):
        """Test running episodes with timeout vs n_simulations configurations."""
        logger = getLogger(__name__)
        num_steps = 5
        belief = get_initial_belief(env, n_particles=20, resampling=True)

        # Test with n_simulations
        planner_n_sims = ICVaR_PFT_DPW(
            environment=env,
            name="test_planner_n_sims",
            depth=3,
            discount_factor=0.95,
            n_simulations=50,
            alpha=0.1,
            delta=0.1,
            k_o=5,
            min_immediate_cost=0.0,
            max_immediate_cost=1.0,
            min_visit_count_per_action=1,
            exploration_constant=1.0,
            action_sampler=action_sampler,
            k_a=2.0,
            alpha_a=0.0,
            alpha_o=0.0,
        )

        # Test with timeout
        planner_timeout = ICVaR_PFT_DPW(
            environment=env,
            name="test_planner_timeout",
            depth=3,
            discount_factor=0.95,
            time_out_in_seconds=2,  # 2 second timeout
            alpha=0.1,
            delta=0.1,
            k_o=5,
            min_immediate_cost=0.0,
            max_immediate_cost=1.0,
            min_visit_count_per_action=1,
            exploration_constant=1.0,
            action_sampler=action_sampler,
            k_a=2.0,
            alpha_a=0.0,
            alpha_o=0.0,
        )

        # Run episodes with both configurations
        history_n_sims = run_episode(
            environment=env,
            policy=planner_n_sims,
            initial_belief=belief,
            num_steps=num_steps,
            logger=logger,
        )

        history_timeout = run_episode(
            environment=env,
            policy=planner_timeout,
            initial_belief=belief,
            num_steps=num_steps,
            logger=logger,
        )

        # Verify both episodes completed successfully
        assert history_n_sims is not None
        assert history_timeout is not None
        assert len(history_n_sims.history) > 0
        assert len(history_timeout.history) > 0

        # Verify planner configurations
        assert planner_n_sims.n_simulations == 50
        assert planner_n_sims.time_out_in_seconds is None

        assert planner_timeout.n_simulations is None
        assert planner_timeout.time_out_in_seconds == 2

    @pytest.mark.slow
    def test_run_episode_early_termination(self, env, action_sampler):
        """Test that episode terminates early when reaching terminal state."""
        logger = getLogger(__name__)
        num_steps = 50  # Large number of steps
        belief = get_initial_belief(env, n_particles=20, resampling=True)

        # Create planner
        planner = ICVaR_PFT_DPW(
            environment=env,
            name="test_planner_termination",
            depth=3,
            discount_factor=0.95,
            n_simulations=100,
            alpha=0.1,
            delta=0.1,
            k_o=5,
            min_immediate_cost=0.0,
            max_immediate_cost=1.0,
            min_visit_count_per_action=1,
            exploration_constant=1.0,
            action_sampler=action_sampler,
            k_a=2.0,
            alpha_a=0.0,
            alpha_o=0.0,
        )

        # Run the episode
        history = run_episode(
            environment=env,
            policy=planner,
            initial_belief=belief,
            num_steps=num_steps,
            logger=logger,
        )

        # Verify episode results
        assert history is not None
        assert len(history.history) > 0

        # Check if episode terminated early due to terminal state
        if history.reach_terminal_state:
            assert len(history.history) < num_steps
            # Verify the last state is terminal
            last_step = history.history[-1]
            assert env.is_terminal(last_step.state)
        else:
            # If not terminal, should have used all steps
            assert len(history.history) == num_steps

    def test_run_episode_different_progressive_widening_params(self, env, action_sampler):
        """Test running episodes with different progressive widening parameters."""
        logger = getLogger(__name__)
        num_steps = 5
        belief = get_initial_belief(env, n_particles=20, resampling=True)

        # Test different k_a and k_o values
        test_configs = [
            {"k_a": 1.0, "alpha_a": 0.5, "k_o": 1.0, "alpha_o": 0.5},
            {"k_a": 2.0, "alpha_a": 0.3, "k_o": 3.0, "alpha_o": 0.3},
            {"k_a": 0.5, "alpha_a": 0.7, "k_o": 0.5, "alpha_o": 0.7},
        ]

        for config in test_configs:
            planner = ICVaR_PFT_DPW(
                environment=env,
                name=f"test_planner_k_a_{config['k_a']}_k_o_{config['k_o']}",
                depth=3,
                discount_factor=0.95,
                n_simulations=50,  # Reduced for faster testing
                alpha=0.1,
                delta=0.1,
                k_o=config["k_o"],
                min_immediate_cost=0.0,
                max_immediate_cost=1.0,
                min_visit_count_per_action=1,
                exploration_constant=1.0,
                action_sampler=action_sampler,
                k_a=config["k_a"],
                alpha_a=config["alpha_a"],
                alpha_o=config["alpha_o"],
            )

            # Run the episode
            history = run_episode(
                environment=env,
                policy=planner,
                initial_belief=belief,
                num_steps=num_steps,
                logger=logger,
            )

            # Verify episode results
            assert history is not None
            assert len(history.history) > 0
            assert len(history.history) <= num_steps

            # Check that planner's parameters were used
            assert planner.k_a == config["k_a"]
            assert planner.alpha_a == config["alpha_a"]
            assert planner.k_o == config["k_o"]
            assert planner.alpha_o == config["alpha_o"]
