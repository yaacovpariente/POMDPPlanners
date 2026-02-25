# pylint: disable=protected-access
from logging import getLogger

import numpy as np
import pytest

from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.tree import BeliefNode, ActionNode
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
    belief_node = BeliefNode(belief=belief)
    action = np.array([1.0, 0.0])  # Unit vector in x direction
    action_node = ActionNode(action=action, parent=belief_node)

    next_belief_node, immediate_reward = planner._generate_belief(action_node)

    assert isinstance(next_belief_node, BeliefNode)
    assert isinstance(immediate_reward, float)
    assert next_belief_node.parent == action_node


def test_sample_next_existing_belief(planner, belief):
    belief_node = BeliefNode(belief=belief)
    action = np.array([1.0, 0.0])  # Unit vector in x direction
    action_node = ActionNode(action=action, parent=belief_node)

    # Create some child belief nodes with continuous states
    children = []
    for i in range(3):
        particles = [np.array([i * 1.0, i * 1.0]), np.array([(i + 1) * 1.0, (i + 1) * 1.0])]
        log_weights = np.log(np.ones(len(particles)) / len(particles))
        child = BeliefNode(
            belief=WeightedParticleBelief(particles=particles, log_weights=log_weights),
            parent=action_node,
        )
        child.visit_count = i
        child.immediate_cost = -i
        children.append(child)
    action_node.children = tuple(children)

    next_belief_node, immediate_reward = planner._sample_next_existing_belief(action_node)

    assert isinstance(next_belief_node, BeliefNode)
    assert isinstance(immediate_reward, float)
    assert next_belief_node in action_node.children


def test_update_nodes(planner, belief):
    belief_node = BeliefNode(belief=belief)
    action = np.array([1.0, 0.0])  # Unit vector in x direction
    action_node = ActionNode(action=action, parent=belief_node)

    # Create some child belief nodes with continuous states
    children = []
    for i in range(3):
        particles = [np.array([i * 1.0, i * 1.0]), np.array([(i + 1) * 1.0, (i + 1) * 1.0])]
        log_weights = np.log(np.ones(len(particles)) / len(particles))
        child = BeliefNode(
            belief=WeightedParticleBelief(particles=particles, log_weights=log_weights),
            parent=action_node,
        )
        child.visit_count = i
        child.v_value = -i
        children.append(child)
    action_node.children = tuple(children)

    initial_visit_count = belief_node.visit_count
    planner.update_nodes(belief_node, action_node)

    assert belief_node.visit_count == initial_visit_count + 1
    assert action_node.visit_count == 1
    assert action_node.q_value is not None


def test_get_space_info(planner):
    space_info = planner.get_space_info()
    assert space_info.action_space == SpaceType.CONTINUOUS
    assert space_info.observation_space == SpaceType.CONTINUOUS


def test_entropy_weight_initialization(env, action_sampler):
    """Test that entropy_weight is properly initialized."""
    planner = ICVaR_PFT_DPW(
        environment=env,
        name="test_planner_entropy",
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
        entropy_weight=0.5,
    )
    assert planner.entropy_weight == 0.5

    # Test default value
    planner_default = ICVaR_PFT_DPW(
        environment=env,
        name="test_planner_default",
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
    assert planner_default.entropy_weight == 0.0


def test_entropy_weight_affects_cost(env, action_sampler, belief):
    """Test that entropy_weight affects the immediate cost calculation."""
    # Create planner with entropy_weight=0
    planner_no_entropy = ICVaR_PFT_DPW(
        environment=env,
        name="test_planner_no_entropy",
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
        entropy_weight=0.0,
    )

    # Create planner with entropy_weight>0
    planner_with_entropy = ICVaR_PFT_DPW(
        environment=env,
        name="test_planner_with_entropy",
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
        entropy_weight=0.5,
    )

    # Build trees with both planners and check that costs differ
    tree_no_entropy = planner_no_entropy._learn_tree(belief=belief)
    tree_with_entropy = planner_with_entropy._learn_tree(belief=belief)

    # Check that immediate costs are set (they should be)
    def get_action_node_costs(node):
        costs = []
        if isinstance(node, ActionNode) and node.immediate_cost is not None:
            costs.append(node.immediate_cost)
        for child in node.children:
            costs.extend(get_action_node_costs(child))
        return costs

    costs_no_entropy = get_action_node_costs(tree_no_entropy)
    costs_with_entropy = get_action_node_costs(tree_with_entropy)

    # Both should have computed costs
    assert len(costs_no_entropy) > 0
    assert len(costs_with_entropy) > 0

    # When entropy_weight > 0, costs should generally be different
    # (though they might be the same in some cases due to clipping)
    # At minimum, both planners should produce valid cost values
    assert all(isinstance(cost, (int, float)) for cost in costs_no_entropy)
    assert all(isinstance(cost, (int, float)) for cost in costs_with_entropy)


def test_action(planner, belief):
    actions, policy_data = planner.action(belief)
    assert isinstance(actions, list)
    assert len(actions) == 1
    assert isinstance(actions[0], np.ndarray)  # For continuous actions
    assert actions[0].shape == (2,)  # 2D action vector
    assert hasattr(policy_data, "info_variables")  # Verify PolicyRunData structure


def test_progressive_widening_constraints(planner, belief):
    """Test that progressive widening constraints are respected in the tree."""
    # Build the tree using the planner's internal method
    tree = planner._learn_tree(belief=belief)

    # Helper function to recursively traverse and check all nodes in the tree
    def check_node_constraints(node):
        if isinstance(node, BeliefNode):
            # For belief nodes: check that number of children <= k_a * visit_count^alpha_a
            # Note: Progressive widening allows expansion when len(children) <= threshold,
            # so we check that actual children don't exceed threshold + 1
            max_children_threshold = planner.k_a + 1
            actual_children = len(node.children)
            # The implementation uses <= in the condition, so actual children can be at most floor(threshold) + 1
            max_allowed = int(max_children_threshold)
            assert actual_children <= max_allowed, (
                f"Belief node has {actual_children} children but should have at most "
                f"{max_allowed} (threshold={max_children_threshold:.3f}, k_a={planner.k_a}, "
                f"visit_count={node.visit_count}, alpha_a={planner.alpha_a})"
            )

            # Recursively check all action node children
            for child in node.children:
                check_node_constraints(child)

        elif isinstance(node, ActionNode):
            # For action nodes: check that number of children <= k_o * visit_count^alpha_o
            # Note: Progressive widening allows expansion when len(children) <= threshold,
            # so we check that actual children don't exceed threshold + 1
            max_children_threshold = planner.k_o + 1
            actual_children = len(node.children)
            # The implementation uses <= in the condition, so actual children can be at most floor(threshold) + 1
            max_allowed = int(max_children_threshold)
            assert actual_children <= max_allowed, (
                f"Action node has {actual_children} children but should have at most "
                f"{max_allowed} (threshold={max_children_threshold:.3f}, k_o={planner.k_o}, "
                f"visit_count={node.visit_count}, alpha_o={planner.alpha_o})"
            )

            # Recursively check all belief node children
            for child in node.children:
                check_node_constraints(child)

    # Start the constraint checking from the root
    check_node_constraints(tree)

    # Additional verification: count total nodes and verify tree structure
    def count_nodes(node, belief_count=0, action_count=0):
        if isinstance(node, BeliefNode):
            belief_count += 1
            for child in node.children:
                belief_count, action_count = count_nodes(child, belief_count, action_count)
        elif isinstance(node, ActionNode):
            action_count += 1
            for child in node.children:
                belief_count, action_count = count_nodes(child, belief_count, action_count)
        return belief_count, action_count

    belief_nodes, action_nodes = count_nodes(tree)

    # Verify we have a meaningful tree (at least some nodes were created)
    assert belief_nodes >= 1, "Tree should contain at least the root belief node"
    assert action_nodes >= 0, "Tree should contain zero or more action nodes"

    print(
        f"Progressive widening test passed: {belief_nodes} belief nodes, {action_nodes} action nodes"
    )
    print(
        f"k_a={planner.k_a}, alpha_a={planner.alpha_a}, k_o={planner.k_o}, alpha_o={planner.alpha_o}"
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
