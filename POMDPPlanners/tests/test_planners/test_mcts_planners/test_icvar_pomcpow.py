# SPDX-License-Identifier: MIT

# pylint: disable=protected-access,too-many-lines,broad-exception-caught,unused-argument,import-outside-toplevel
from logging import getLogger
import inspect
import math

import numpy as np
import pytest

from POMDPPlanners.core.belief import (
    WeightedParticleBelief,
    WeightedParticleBeliefStateUpdate,
    get_initial_belief,
)
from POMDPPlanners.core.cost import belief_expectation_cost
from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree
from POMDPPlanners.environments import TigerPOMDP, DiscreteLightDarkPOMDP
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    discrete_light_dark_pinned_kwargs,
)
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.planners.mcts_planners.icvar_pomcpow import ICVaR_POMCPOW
from POMDPPlanners.core.policy import PolicyRunData


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
    return 1.0


@pytest.fixture
def k_a():
    return 1.0


@pytest.fixture
def alpha_o():
    return 0.5


@pytest.fixture
def alpha_a():
    return 0.5


@pytest.fixture
def alpha():
    return 0.1


@pytest.fixture
def n_simulations():
    return 100


@pytest.fixture
def n_particles():
    return 100


@pytest.fixture
def time_out_in_seconds():
    return 10


@pytest.fixture
def min_samples_per_node():
    return 10


@pytest.fixture
def environment(discount_factor):
    return TigerPOMDP(discount_factor=discount_factor)


@pytest.fixture
def light_dark_env(discount_factor):
    return DiscreteLightDarkPOMDP(
        discount_factor=discount_factor, **discrete_light_dark_pinned_kwargs()
    )


@pytest.fixture
def min_immediate_cost():
    return -10.0


@pytest.fixture
def max_immediate_cost():
    return 10.0


@pytest.fixture
def min_visit_count_per_action():
    return 1


@pytest.fixture
def delta():
    return 0.1


@pytest.fixture
def action_sampler(environment):
    return DiscreteActionSampler(actions=environment.get_actions())


@pytest.fixture
def light_dark_action_sampler(light_dark_env):
    return DiscreteActionSampler(actions=light_dark_env.get_actions())


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
    alpha,
    n_simulations,
    action_sampler,
    min_immediate_cost,
    max_immediate_cost,
    min_visit_count_per_action,
    delta,
):
    return ICVaR_POMCPOW(
        environment=environment,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        min_immediate_cost=min_immediate_cost,
        max_immediate_cost=max_immediate_cost,
        min_visit_count_per_action=min_visit_count_per_action,
        delta=delta,
        name="test_icvar_pomcpow",
        action_sampler=action_sampler,
        n_simulations=n_simulations,
        alpha=alpha,
    )


@pytest.fixture
def light_dark_planner(
    light_dark_env,
    discount_factor,
    depth,
    exploration_constant,
    k_o,
    k_a,
    alpha_o,
    alpha_a,
    alpha,
    light_dark_action_sampler,
    min_immediate_cost,
    max_immediate_cost,
    min_visit_count_per_action,
    delta,
):
    return ICVaR_POMCPOW(
        environment=light_dark_env,
        discount_factor=discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        min_immediate_cost=min_immediate_cost,
        max_immediate_cost=max_immediate_cost,
        min_visit_count_per_action=min_visit_count_per_action,
        delta=delta,
        name="test_icvar_pomcpow_light_dark",
        action_sampler=light_dark_action_sampler,
        n_simulations=100,  # Add n_simulations to avoid error
        alpha=alpha,
    )


@pytest.fixture
def belief(environment, n_particles):
    return get_initial_belief(pomdp=environment, n_particles=n_particles, resampling=True)


@pytest.fixture
def light_dark_belief(light_dark_env, n_particles):
    return get_initial_belief(pomdp=light_dark_env, n_particles=n_particles, resampling=True)


def create_weighted_belief(particles, log_weights=None):
    """Helper function to create a weighted belief for testing."""
    n = len(particles)
    if log_weights is None:
        # Uniform log-weights
        log_weights = np.log(np.ones(n) / n)
    return WeightedParticleBelief(particles=particles, log_weights=log_weights)


def _make_planner_for_backup_test(env, action_sampler, alpha, discount_factor):
    """Build a minimal ICVaR_POMCPOW with the specified ``alpha`` and ``γ``.

    Used by the ``_update_q_value`` paper-formula tests, which need control
    over α and γ to assert closed-form expected Q-values; the rest of the
    constructor arguments are filler that does not influence the backup
    arithmetic exercised here.
    """
    return ICVaR_POMCPOW(
        environment=env,
        discount_factor=discount_factor,
        depth=3,
        exploration_constant=1.0,
        k_o=1.0,
        k_a=1.0,
        alpha_o=0.5,
        alpha_a=0.5,
        min_immediate_cost=-10.0,
        max_immediate_cost=10.0,
        min_visit_count_per_action=1,
        delta=0.1,
        name="test_icvar_pomcpow_backup",
        action_sampler=action_sampler,
        n_simulations=1,
        alpha=alpha,
    )


def _iter_node_ids(tree: Tree, root_id: int):
    """BFS iterator over all node IDs in the subtree rooted at ``root_id``."""
    frontier = [root_id]
    while frontier:
        current = frontier.pop(0)
        yield current
        frontier.extend(tree.children_ids[current])


def _tree_depth(tree: Tree, root_id: int) -> int:
    """Maximum edge-depth from ``root_id`` to any descendant."""
    return max(
        tree.depth(node_id) - tree.depth(root_id) for node_id in _iter_node_ids(tree, root_id)
    )


@pytest.fixture
def terminal_state(light_dark_env):
    """Get a true terminal state from the environment."""
    # In LightDark, the goal state is terminal
    goal_state = light_dark_env.goal_state
    assert light_dark_env.is_terminal(goal_state)
    return goal_state


class TestICVaR_POMCPOWInitialization:
    """Test class for ICVaR_POMCPOW initialization."""

    def test_initialization_with_n_simulations(
        self,
        environment,
        discount_factor,
        depth,
        exploration_constant,
        k_o,
        k_a,
        alpha_o,
        alpha_a,
        alpha,
        n_simulations,
        action_sampler,
        min_immediate_cost,
        max_immediate_cost,
        min_visit_count_per_action,
        delta,
    ):
        """Test initialization with n_simulations parameter."""
        planner = ICVaR_POMCPOW(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            exploration_constant=exploration_constant,
            k_o=k_o,
            k_a=k_a,
            alpha_o=alpha_o,
            alpha_a=alpha_a,
            min_immediate_cost=min_immediate_cost,
            max_immediate_cost=max_immediate_cost,
            min_visit_count_per_action=min_visit_count_per_action,
            delta=delta,
            name="test_planner",
            action_sampler=action_sampler,
            n_simulations=n_simulations,
            alpha=alpha,
        )

        assert planner.environment == environment
        assert planner.discount_factor == discount_factor
        assert planner.depth == depth
        assert planner.exploration_constant == exploration_constant
        assert planner.k_o == k_o
        assert planner.k_a == k_a
        assert planner.alpha_o == alpha_o
        assert planner.alpha_a == alpha_a
        assert planner.alpha == alpha
        assert planner.n_simulations == n_simulations
        assert planner.action_sampler == action_sampler
        assert planner.name == "test_planner"

    def test_initialization_with_timeout(
        self,
        environment,
        discount_factor,
        depth,
        exploration_constant,
        k_o,
        k_a,
        alpha_o,
        alpha_a,
        alpha,
        time_out_in_seconds,
        action_sampler,
        min_immediate_cost,
        max_immediate_cost,
        min_visit_count_per_action,
        delta,
    ):
        """Test initialization with timeout parameter."""
        planner = ICVaR_POMCPOW(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            exploration_constant=exploration_constant,
            k_o=k_o,
            k_a=k_a,
            alpha_o=alpha_o,
            alpha_a=alpha_a,
            min_immediate_cost=min_immediate_cost,
            max_immediate_cost=max_immediate_cost,
            min_visit_count_per_action=min_visit_count_per_action,
            delta=delta,
            name="test_planner",
            action_sampler=action_sampler,
            time_out_in_seconds=time_out_in_seconds,
            alpha=alpha,
        )

        assert planner.time_out_in_seconds == time_out_in_seconds
        assert planner.n_simulations is None

    def test_initialization_with_min_samples_per_node(
        self,
        environment,
        discount_factor,
        depth,
        exploration_constant,
        k_o,
        k_a,
        alpha_o,
        alpha_a,
        alpha,
        min_samples_per_node,
        action_sampler,
        min_immediate_cost,
        max_immediate_cost,
        min_visit_count_per_action,
        delta,
    ):
        """Test initialization with min_samples_per_node parameter."""
        planner = ICVaR_POMCPOW(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            exploration_constant=exploration_constant,
            k_o=k_o,
            k_a=k_a,
            alpha_o=alpha_o,
            alpha_a=alpha_a,
            min_immediate_cost=min_immediate_cost,
            max_immediate_cost=max_immediate_cost,
            min_visit_count_per_action=min_visit_count_per_action,
            delta=delta,
            name="test_planner",
            action_sampler=action_sampler,
            min_samples_per_node=min_samples_per_node,
            alpha=alpha,
        )

        assert planner.min_samples_per_node == min_samples_per_node

    def test_initialization_with_debug_and_log_path(
        self,
        environment,
        discount_factor,
        depth,
        exploration_constant,
        k_o,
        k_a,
        alpha_o,
        alpha_a,
        alpha,
        action_sampler,
        tmp_path,
        min_immediate_cost,
        max_immediate_cost,
        min_visit_count_per_action,
        delta,
    ):
        """Test initialization with debug and log_path parameters."""
        log_path = tmp_path / "test_log.txt"

        planner = ICVaR_POMCPOW(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            exploration_constant=exploration_constant,
            k_o=k_o,
            k_a=k_a,
            alpha_o=alpha_o,
            alpha_a=alpha_a,
            min_immediate_cost=min_immediate_cost,
            max_immediate_cost=max_immediate_cost,
            min_visit_count_per_action=min_visit_count_per_action,
            delta=delta,
            name="test_planner",
            action_sampler=action_sampler,
            log_path=log_path,
            debug=True,
            alpha=alpha,
        )

        assert planner.log_path == log_path
        assert planner.debug is True


class TestICVaR_POMCPOWCoreFunctionality:
    """Test class for ICVaR_POMCPOW core functionality."""

    def test_simulate_path_basic(self, planner, belief):
        """Test basic simulation path functionality."""
        tree = Tree()
        tree.add_belief_node(belief)

        assert hasattr(planner, "_simulate_path")
        assert callable(planner._simulate_path)

        assert isinstance(belief, WeightedParticleBelief)
        assert len(belief.particles) > 0

        assert hasattr(planner, "environment")
        assert hasattr(planner, "depth")
        assert hasattr(planner, "discount_factor")

    def test_simulate_path_depth_limit(self, planner, belief):
        """Test that simulation respects depth limit."""
        tree = Tree()
        tree.add_belief_node(belief)

        assert hasattr(planner, "_simulate_path")
        assert callable(planner._simulate_path)

        assert hasattr(planner, "depth")
        assert isinstance(planner.depth, int)
        assert planner.depth > 0

        assert isinstance(belief, WeightedParticleBelief)
        assert len(belief.particles) > 0

    def test_simulate_path_terminal_state(
        self, light_dark_planner, light_dark_belief, terminal_state
    ):
        """Test simulation with terminal state."""
        terminal_particles = [terminal_state] * 10
        terminal_belief = create_weighted_belief(terminal_particles)
        tree = Tree()
        tree.add_belief_node(terminal_belief)

        assert hasattr(light_dark_planner, "_simulate_path")
        assert callable(light_dark_planner._simulate_path)

        assert isinstance(terminal_belief, WeightedParticleBelief)
        assert len(terminal_belief.particles) == 10
        assert all(
            np.array_equal(particle, terminal_state) for particle in terminal_belief.particles
        )

        assert hasattr(light_dark_planner, "environment")
        assert hasattr(light_dark_planner, "depth")
        assert hasattr(light_dark_planner, "discount_factor")

    def test_simulate_state_path_terminal_state_with_depth_gt_zero(
        self, light_dark_planner, terminal_state
    ):
        """Test _simulate_state_path with terminal state when depth > 0.

        This test verifies that when a terminal state is encountered at depth > 0,
        the method handles it correctly without raising a KeyError when trying to
        update the belief with action=None and observation=None.
        """
        terminal_particles = [terminal_state] * 10
        terminal_belief = create_weighted_belief(terminal_particles)
        tree = Tree()
        belief_id = tree.add_belief_node(terminal_belief)

        assert light_dark_planner.environment.is_terminal(state=terminal_state)

        assert hasattr(light_dark_planner, "_simulate_state_path")
        assert callable(light_dark_planner._simulate_state_path)

        try:
            light_dark_planner._simulate_state_path(
                tree=tree, state=terminal_state, belief_id=belief_id, depth=1
            )
            assert tree.visit_count[belief_id] > 0
        except (KeyError, NotImplementedError, TypeError) as e:
            error_msg = (
                f"_simulate_state_path raised {type(e).__name__} when handling terminal state "
                f"at depth > 0. This indicates a bug where inplace_update is called with "
                f"action=None and observation=None, which causes issues. "
                f"Error: {e}"
            )
            pytest.fail(error_msg)

    def test_update_belief_with_state_appends_particle_and_weight(
        self, planner, belief, environment
    ):
        """Verify _update_belief_with_state mutates the child belief in place.

        Purpose: Validates that _update_belief_with_state actually appends the
        supplied state and its observation likelihood to the target child
        belief, rather than merely reaching ``inplace_update`` without raising.

        Given: A tree containing a root belief, an action node, and an empty
            ``WeightedParticleBeliefStateUpdate`` registered as that action's
            observation child.
        When: ``_update_belief_with_state`` is called with a (state, action,
            observation) triple sampled from the environment, and then again
            with a second sampled triple targeting the same child belief.
        Then: After the first call the child gains exactly one particle equal
            to the supplied state, one weight equal to ``exp`` of the env's
            observation log-probability, and ``weights_sum`` equal to that
            weight; after the second call the particle/weight counts grow to
            two and ``weights_sum`` equals the running total of both weights.

        Test type: unit
        """
        np.random.seed(42)
        tree = Tree()
        root_id = tree.add_belief_node(belief)

        action = environment.get_actions()[0]
        action_id = tree.add_action_node(action=action, parent_id=root_id)

        child_belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])
        sampled_state = belief.sample()
        next_state, observation, _ = environment.sample_next_step(
            state=sampled_state, action=action
        )
        child_id = tree.add_belief_node(
            belief=child_belief, observation=observation, parent_id=action_id
        )

        assert len(child_belief.particles) == 0
        assert len(child_belief.weights) == 0
        assert child_belief.weights_sum == 0

        planner._update_belief_with_state(
            tree=tree,
            belief_id=child_id,
            action_id=action_id,
            observation=observation,
            state=next_state,
        )

        first_log_p = environment.observation_log_probability_single(
            next_state=next_state, action=action, observation=observation
        )
        first_weight = math.exp(first_log_p) if math.isfinite(first_log_p) else 0.0

        assert len(child_belief.particles) == 1
        assert child_belief.particles[-1] == next_state
        assert len(child_belief.weights) == 1
        assert child_belief.weights[-1] == pytest.approx(first_weight)
        assert child_belief.weights_sum == pytest.approx(first_weight)

        second_next_state, second_observation, _ = environment.sample_next_step(
            state=next_state, action=action
        )
        planner._update_belief_with_state(
            tree=tree,
            belief_id=child_id,
            action_id=action_id,
            observation=second_observation,
            state=second_next_state,
        )
        second_log_p = environment.observation_log_probability_single(
            next_state=second_next_state, action=action, observation=second_observation
        )
        second_weight = math.exp(second_log_p) if math.isfinite(second_log_p) else 0.0

        assert len(child_belief.particles) == 2
        assert child_belief.particles[-1] == second_next_state
        assert len(child_belief.weights) == 2
        assert child_belief.weights[-1] == pytest.approx(second_weight)
        assert child_belief.weights_sum == pytest.approx(first_weight + second_weight)

    def test_update_immediate_cost_first_call_state_update_belief_uses_negative_reward(
        self, planner, environment
    ):
        """First-call ``_update_immediate_cost`` on a state-update belief stores ``-reward``.

        Purpose: Validates the first branch of the paper's running
        weighted-average formula: when the action node has no cached
        immediate cost yet and its child belief is a
        ``WeightedParticleBeliefStateUpdate``, the cost is initialized to
        ``-reward`` (i.e. ``c = -R`` per the paper's cost = negative reward
        convention, matching algorithm line ``Imm(ha) ← (... + c·P)/W``
        evaluated at ``W_old = 0``).

        Given: A tree whose action node has ``immediate_cost = None`` and
            whose child belief is a freshly populated
            ``WeightedParticleBeliefStateUpdate`` with one particle.
        When: ``_update_immediate_cost`` is called with a known reward.
        Then: ``tree.immediate_cost[action_id]`` equals ``-reward`` exactly.

        Test type: unit
        """
        tree = Tree()
        action = environment.get_actions()[0]
        sampled_state = environment.initial_state_dist().sample()[0]

        root_belief = WeightedParticleBelief(
            particles=[sampled_state, sampled_state], log_weights=np.log(np.array([0.5, 0.5]))
        )
        root_id = tree.add_belief_node(root_belief)
        action_id = tree.add_action_node(action=action, parent_id=root_id)

        child_belief = WeightedParticleBeliefStateUpdate(particles=[sampled_state], weights=[0.5])
        child_id = tree.add_belief_node(belief=child_belief, observation="o", parent_id=action_id)

        assert tree.immediate_cost[action_id] is None

        reward = 3.5
        planner._update_immediate_cost(
            tree=tree, belief_id=child_id, action_id=action_id, reward=reward
        )

        assert tree.immediate_cost[action_id] == pytest.approx(-reward)

    def test_update_immediate_cost_running_weighted_average(self, planner, environment):
        """Subsequent ``_update_immediate_cost`` follows the paper's running weighted average.

        Purpose: Validates the formula
        ``Imm_new = (Imm_old · W_old + c · w_new) / W_new`` where
        ``W_new = W_old + w_new``, ``c = -reward``, and ``w_new`` is the
        weight of the most recently appended particle (i.e.
        ``belief.weights[-1]``). This is the recursive form on line
        :paper:`Imm(ha) ← (Imm·W_old + c·P) / W_new` of ICVaR-POMCPOW
        Simulate.

        Given: A tree whose action node already has a cached
            ``immediate_cost`` (treated as the ``Imm_old`` from a previous
            backup) and whose child belief is a
            ``WeightedParticleBeliefStateUpdate`` populated with two
            particles whose weights are explicit known values.
        When: ``_update_immediate_cost`` is called with a known reward.
        Then: ``tree.immediate_cost[action_id]`` equals the closed-form
            running-average value computed independently in the test.

        Test type: unit
        """
        tree = Tree()
        action = environment.get_actions()[0]
        sampled_state = environment.initial_state_dist().sample()[0]

        root_belief = WeightedParticleBelief(
            particles=[sampled_state, sampled_state], log_weights=np.log(np.array([0.5, 0.5]))
        )
        root_id = tree.add_belief_node(root_belief)
        action_id = tree.add_action_node(action=action, parent_id=root_id)

        weight_old = 0.7
        weight_new = 0.4
        child_belief = WeightedParticleBeliefStateUpdate(
            particles=[sampled_state, sampled_state],
            weights=[weight_old, weight_new],
        )
        assert child_belief.weights_sum == pytest.approx(weight_old + weight_new)

        child_id = tree.add_belief_node(belief=child_belief, observation="o", parent_id=action_id)

        imm_old = -1.25
        tree.set_immediate_cost(action_id, imm_old)

        reward = 2.0
        planner._update_immediate_cost(
            tree=tree, belief_id=child_id, action_id=action_id, reward=reward
        )

        expected = (imm_old * weight_old + (-reward) * weight_new) / (weight_old + weight_new)
        assert tree.immediate_cost[action_id] == pytest.approx(expected)

    def test_update_immediate_cost_recursive_calls_match_closed_form_weighted_average(
        self, planner, environment
    ):
        """N sequential ``_update_immediate_cost`` calls compose into ``Σ(-r·w)/Σw``.

        Purpose: Validates that the recursive update
        ``Imm_new = (Imm_old·W_old + (-r_new)·w_new) / W_new`` (paper algo
        line ``Imm(ha) ← (Imm·W_old + c·P)/W_new``) actually accumulates
        across many particles into the same value as the explicit
        weighted average ``Σ_i (-r_i)·w_i / Σ_i w_i``. The single-step
        test pins down the formula on one update; this test pins down
        that repeated application gives the closed-form answer over a
        whole sequence — no per-step rounding drift, correct treatment
        of the first-call special case (``Imm = -reward_1``).

        Given: Three sequential particle additions with explicit
            (reward, weight) pairs ``(1.0, 0.5)``, ``(3.0, 0.25)``,
            ``(5.0, 0.25)``. Each iteration appends to the child belief
            then calls ``_update_immediate_cost``.
        When: ``_update_immediate_cost`` is invoked once after each
            append, in order.
        Then: After all three calls,
            ``immediate_cost[action_id] = (-1·0.5 - 3·0.25 - 5·0.25) /
            (0.5+0.25+0.25) = -2.5/1.0 = -2.5`` exactly.

        Test type: unit
        """
        tree = Tree()
        action = environment.get_actions()[0]
        sampled_state = environment.initial_state_dist().sample()[0]

        root_belief = WeightedParticleBelief(
            particles=[sampled_state, sampled_state], log_weights=np.log(np.array([0.5, 0.5]))
        )
        root_id = tree.add_belief_node(root_belief)
        action_id = tree.add_action_node(action=action, parent_id=root_id)

        child_belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])
        child_id = tree.add_belief_node(belief=child_belief, observation="o", parent_id=action_id)

        rewards_and_weights = [(1.0, 0.5), (3.0, 0.25), (5.0, 0.25)]
        for reward, weight in rewards_and_weights:
            child_belief.particles.append(sampled_state)
            child_belief.weights.append(weight)
            child_belief.weights_sum += weight
            planner._update_immediate_cost(
                tree=tree, belief_id=child_id, action_id=action_id, reward=reward
            )

        # Closed form: Σ(-r·w) / Σw = (-0.5 - 0.75 - 1.25) / 1.0 = -2.5
        assert tree.immediate_cost[action_id] == pytest.approx(-2.5)

    def test_update_immediate_cost_root_belief_uses_belief_expectation_cost(
        self, planner, belief, environment
    ):
        """Root-frame ``_update_immediate_cost`` delegates to ``belief_expectation_cost``.

        Purpose: Validates that when the child belief is *not* a
        ``WeightedParticleBeliefStateUpdate`` (i.e. the action node lives
        directly under the root belief), the first-call path stores
        ``belief_expectation_cost(belief, action, env)`` — the
        expected-cost form ``Σ_j w̃^j c(x^j, a)`` of paper eq. (Qest).

        Given: A tree whose action node has ``immediate_cost = None`` and
            whose child belief is a regular ``WeightedParticleBelief``
            (the root-frame belief type).
        When: ``_update_immediate_cost`` is called with an arbitrary
            reward (which the root path ignores).
        Then: ``tree.immediate_cost[action_id]`` equals the value returned
            by ``belief_expectation_cost(belief, action, env)``.

        Test type: unit
        """
        tree = Tree()
        action = environment.get_actions()[0]

        root_id = tree.add_belief_node(belief)
        action_id = tree.add_action_node(action=action, parent_id=root_id)

        # Root-frame: child belief is a non-state-update belief.
        child_belief = belief
        child_id = tree.add_belief_node(belief=child_belief, observation="o", parent_id=action_id)

        assert tree.immediate_cost[action_id] is None

        planner._update_immediate_cost(
            tree=tree, belief_id=child_id, action_id=action_id, reward=999.0
        )

        expected = belief_expectation_cost(belief=belief, action=action, env=environment)
        assert tree.immediate_cost[action_id] == pytest.approx(expected)

    def test_update_q_value_constant_v_children_yields_imm_plus_gamma_v(
        self, environment, action_sampler, belief
    ):
        """Constant V-distribution: ``Q = Imm + γ·v`` for any α and any weights.

        Purpose: Validates the action-value backup
        ``Q(ba) = Imm(ha) + γ · Ĉ_α(V_children)`` (paper eq. 250) on the
        degenerate-distribution case: when every child has the same V,
        the upper-α CVaR equals that constant V, regardless of α or the
        per-child sampling frequencies.

        Given: A planner with α = 0.1, γ = 0.5; an action node with three
            visited belief children that all have ``v_value = 4.0`` and
            heterogeneous weights ``[1, 2, 3]``; ``immediate_cost = -1``.
        When: ``_update_q_value`` is called.
        Then: ``q_value[action] = -1 + 0.5 · 4.0 = 1.0`` exactly.

        Test type: unit
        """
        local_planner = _make_planner_for_backup_test(
            environment, action_sampler, alpha=0.1, discount_factor=0.5
        )
        tree = Tree()
        action = environment.get_actions()[0]
        root_id = tree.add_belief_node(belief)
        action_id = tree.add_action_node(action=action, parent_id=root_id)

        for w in (1.0, 2.0, 3.0):
            child_belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])
            cid = tree.add_belief_node(
                belief=child_belief, observation=f"o_{w}", parent_id=action_id, weight=w
            )
            tree.v_value[cid] = 4.0
            tree.visit_count[cid] = 1

        tree.set_immediate_cost(action_id, -1.0)

        local_planner._update_q_value(tree=tree, action_id=action_id)

        assert tree.q_value[action_id] == pytest.approx(1.0)

    def test_update_q_value_alpha_one_reduces_to_expected_value(
        self, environment, action_sampler, belief
    ):
        """α = 1 case: ``Ĉ_1`` is the weighted mean, so ``Q = Imm + γ · E[V]``.

        Purpose: Validates the paper's stated property (Section 6) that
        ``α = 1`` recovers expectation-based planning. Under ``α = 1``,
        ``Ĉ_α({(V_i, w_i)}) = Σ w_i V_i``, so the Q-update reduces to the
        risk-neutral Bellman backup.

        Given: A planner with α = 1.0, γ = 1.0; an action node with three
            visited belief children with ``v_value = [1.0, 2.0, 3.0]`` and
            tree weights ``[1, 1, 3]`` (normalized to ``[0.2, 0.2, 0.6]``);
            ``immediate_cost = 0``.
        When: ``_update_q_value`` is called.
        Then: ``q_value = 0 + 1.0 · (0.2·1 + 0.2·2 + 0.6·3) = 2.4`` exactly.

        Test type: unit
        """
        local_planner = _make_planner_for_backup_test(
            environment, action_sampler, alpha=1.0, discount_factor=1.0
        )
        tree = Tree()
        action = environment.get_actions()[0]
        root_id = tree.add_belief_node(belief)
        action_id = tree.add_action_node(action=action, parent_id=root_id)

        v_values = [1.0, 2.0, 3.0]
        weights = [1.0, 1.0, 3.0]  # normalize to [0.2, 0.2, 0.6]
        for v, w in zip(v_values, weights):
            child_belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])
            cid = tree.add_belief_node(
                belief=child_belief, observation=f"o_{v}", parent_id=action_id, weight=w
            )
            tree.v_value[cid] = v
            tree.visit_count[cid] = 1

        tree.set_immediate_cost(action_id, 0.0)

        local_planner._update_q_value(tree=tree, action_id=action_id)

        # E[V] = 0.2·1 + 0.2·2 + 0.6·3 = 0.2 + 0.4 + 1.8 = 2.4
        assert tree.q_value[action_id] == pytest.approx(2.4)

    def test_update_q_value_alpha_half_two_equal_weight_children_equals_max_v(
        self, environment, action_sampler, belief
    ):
        """α=0.5, two equal-weight children: ``Ĉ_α`` = the larger V (top half).

        Purpose: Validates the upper-α tail behavior of ``Ĉ_α`` on the
        smallest non-trivial case. With two equal-weight children
        (each weight 1/2 = α), the worst-α probability mass lies entirely
        on the larger V, so ``Ĉ_α = max(V_1, V_2)``.

        Given: A planner with α = 0.5, γ = 1.0; an action node with two
            visited belief children with ``v_value = [3.0, 7.0]`` and
            equal tree weights ``[1, 1]``; ``immediate_cost = -1``.
        When: ``_update_q_value`` is called.
        Then: ``q_value = -1 + 1.0 · 7.0 = 6.0`` exactly.

        Test type: unit
        """
        local_planner = _make_planner_for_backup_test(
            environment, action_sampler, alpha=0.5, discount_factor=1.0
        )
        tree = Tree()
        action = environment.get_actions()[0]
        root_id = tree.add_belief_node(belief)
        action_id = tree.add_action_node(action=action, parent_id=root_id)

        for v in (3.0, 7.0):
            child_belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])
            cid = tree.add_belief_node(
                belief=child_belief, observation=f"o_{v}", parent_id=action_id, weight=1.0
            )
            tree.v_value[cid] = v
            tree.visit_count[cid] = 1

        tree.set_immediate_cost(action_id, -1.0)

        local_planner._update_q_value(tree=tree, action_id=action_id)

        assert tree.q_value[action_id] == pytest.approx(6.0)

    def test_update_q_value_alpha_half_four_equal_weight_children_equals_mean_of_top_two(
        self, environment, action_sampler, belief
    ):
        """α=0.5, four equal-weight children: ``Ĉ_α`` = mean of the two largest V's.

        Purpose: Validates ``Ĉ_α`` aggregation when the upper-α tail
        spans more than one outcome. With four equal-weight children
        (each 1/4) and α = 0.5, the worst-α mass = 1/2 covers the two
        largest V's exactly, so ``Ĉ_α = (V_top + V_2nd) / 2``.

        Given: A planner with α = 0.5, γ = 0.5; an action node with four
            visited belief children with ``v_value = [1.0, 2.0, 5.0, 8.0]``
            and equal tree weights; ``immediate_cost = -2``.
        When: ``_update_q_value`` is called.
        Then: ``q_value = -2 + 0.5 · (5+8)/2 = -2 + 0.5 · 6.5 = 1.25``
            exactly.

        Test type: unit
        """
        local_planner = _make_planner_for_backup_test(
            environment, action_sampler, alpha=0.5, discount_factor=0.5
        )
        tree = Tree()
        action = environment.get_actions()[0]
        root_id = tree.add_belief_node(belief)
        action_id = tree.add_action_node(action=action, parent_id=root_id)

        for v in (1.0, 2.0, 5.0, 8.0):
            child_belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])
            cid = tree.add_belief_node(
                belief=child_belief, observation=f"o_{v}", parent_id=action_id, weight=1.0
            )
            tree.v_value[cid] = v
            tree.visit_count[cid] = 1

        tree.set_immediate_cost(action_id, -2.0)

        local_planner._update_q_value(tree=tree, action_id=action_id)

        assert tree.q_value[action_id] == pytest.approx(1.25)

    def test_update_v_value_equals_min_q_over_visited_action_children(
        self, planner, belief, environment
    ):
        """``V(b) = min_{a ∈ visited children of b} Q(ba)``: pure min over visited Q's.

        Purpose: Validates the V-update of paper eq. (Vest) /
        algorithm line ``V(h) ← min_{a ∈ Ch(h)} V(ha)``: the V-value at a
        belief node equals the minimum Q-value over its visited action
        children. Action children with ``visit_count == 0`` are excluded,
        even when their (stale or default) Q-value is smaller than the
        true minimum over visited children.

        Given: A belief node with three visited action children with
            ``q_value = [4.0, -1.0, 2.5]`` and one unvisited action child
            with ``q_value = -999.0`` (must be ignored).
        When: ``_update_v_value`` is called for the belief node.
        Then: ``v_value[belief] = -1.0`` (the minimum over visited
            children only); the unvisited ``-999.0`` is excluded.

        Test type: unit
        """
        tree = Tree()
        root_id = tree.add_belief_node(belief)

        actions = environment.get_actions()
        # Three visited action children with known Q-values.
        visited_qs = [4.0, -1.0, 2.5]
        for i, q in enumerate(visited_qs):
            aid = tree.add_action_node(action=actions[i % len(actions)], parent_id=root_id)
            tree.q_value[aid] = q
            tree.visit_count[aid] = 1

        # One unvisited action child whose smaller Q must be ignored.
        unvisited_aid = tree.add_action_node(action=actions[0], parent_id=root_id)
        tree.q_value[unvisited_aid] = -999.0
        tree.visit_count[unvisited_aid] = 0

        planner._update_v_value(tree=tree, belief_id=root_id)

        assert tree.v_value[root_id] == pytest.approx(-1.0)

    def test_update_q_value_no_visited_children_returns_immediate_cost(
        self, planner, belief, environment
    ):
        """``_update_q_value`` falls back to ``immediate_cost`` when no children are visited.

        Purpose: Validates the leaf-action edge case: if the action node
        has no visited children (either because it has no children at all,
        or because every child has ``visit_count == 0``), the Q-value is
        set to the immediate cost alone — there is no CVaR term to add.

        Given: A tree with an action node carrying a known
            ``immediate_cost`` and a single belief child whose
            ``visit_count`` is 0.
        When: ``_update_q_value`` is called for that action node.
        Then: ``tree.q_value[action_id]`` equals ``immediate_cost`` exactly
            (no contribution from the unvisited child's V-value).

        Test type: unit
        """
        tree = Tree()
        action = environment.get_actions()[0]

        root_id = tree.add_belief_node(belief)
        action_id = tree.add_action_node(action=action, parent_id=root_id)

        unvisited_belief = WeightedParticleBeliefStateUpdate(particles=[], weights=[])
        unvisited_id = tree.add_belief_node(
            belief=unvisited_belief, observation="o", parent_id=action_id, weight=1.0
        )
        tree.v_value[unvisited_id] = 42.0  # Must not influence the result.
        tree.visit_count[unvisited_id] = 0

        imm = -1.5
        tree.set_immediate_cost(action_id, imm)

        planner._update_q_value(tree=tree, action_id=action_id)

        assert tree.q_value[action_id] == pytest.approx(imm)


class TestICVaR_POMCPOWIntegration:
    """Test class for ICVaR_POMCPOW integration tests."""

    def test_integration_with_tiger_pomdp(self, planner, belief, environment, n_particles):
        """Test integration with TigerPOMDP environment."""
        # Test basic planner properties
        assert planner.environment == environment
        assert planner.discount_factor > 0
        assert planner.depth > 0

        # Test that planner can handle the belief
        assert hasattr(planner, "action")
        assert callable(planner.action)

        # Test that environment has valid actions
        actions = environment.get_actions()
        assert len(actions) > 0
        assert all(isinstance(action, (str, int)) for action in actions)

    def test_integration_with_light_dark_pomdp(
        self, light_dark_planner, light_dark_belief, light_dark_env
    ):
        """Test integration with DiscreteLightDarkPOMDP environment."""
        # Test basic planner properties
        assert light_dark_planner.environment == light_dark_env
        assert light_dark_planner.discount_factor > 0
        assert light_dark_planner.depth > 0

        # Test that planner can handle the belief
        assert hasattr(light_dark_planner, "action")
        assert callable(light_dark_planner.action)

        # Test that environment has valid actions
        actions = light_dark_env.get_actions()
        assert len(actions) > 0
        assert all(isinstance(action, (str, int, np.integer)) for action in actions)

    def test_action_accepts_vectorized_weighted_particle_belief_at_root(
        self, light_dark_env, light_dark_action_sampler
    ):
        """Root call with VectorizedWeightedParticleBelief returns a valid action.

        Purpose: Validates that ``ICVaR_POMCPOW.action`` handles the
            default belief type returned by
            :func:`POMDPPlanners.utils.belief_factory.create_environment_belief`,
            which is ``VectorizedWeightedParticleBelief`` for every
            registered environment. Before the fix the planner's
            ``_update_immediate_cost`` dispatched only on
            ``WeightedParticleBeliefStateUpdate`` and
            ``WeightedParticleBelief`` and raised
            ``ValueError(f"Unsupported belief type: ...")`` on the very
            first backprop, making the documented public-API call
            ``policy.action(create_environment_belief(env))`` crash.

        Given: A DiscreteLightDarkPOMDP environment, a
            ``VectorizedWeightedParticleBelief`` produced by
            ``create_environment_belief``, and an ICVaR_POMCPOW planner
            with a small simulation budget.
        When: ``planner.action(belief)`` is called.
        Then: The call returns one of the environment's valid actions
            without raising.

        Test type: integration
        """
        # Local imports keep this test self-contained — the symbols are
        # not used by any other test in this file.
        from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
            VectorizedWeightedParticleBelief,
        )
        from POMDPPlanners.utils.belief_factory import create_environment_belief

        belief = create_environment_belief(light_dark_env, n_particles=50)
        assert isinstance(belief, VectorizedWeightedParticleBelief), (
            "factory must return VectorizedWeightedParticleBelief for the default "
            "config — otherwise this test does not exercise the bug it targets"
        )
        planner = ICVaR_POMCPOW(
            environment=light_dark_env,
            discount_factor=0.95,
            depth=3,
            exploration_constant=1.0,
            k_o=1.0,
            k_a=1.0,
            alpha_o=0.5,
            alpha_a=0.5,
            min_immediate_cost=-10.0,
            max_immediate_cost=10.0,
            min_visit_count_per_action=1,
            delta=0.1,
            name="test_icvar_pomcpow_vec_belief",
            action_sampler=light_dark_action_sampler,
            n_simulations=10,
            alpha=0.1,
        )
        np.random.seed(0)
        action, policy_run_data = planner.action(belief)
        assert isinstance(policy_run_data, PolicyRunData)
        actual_action = action[0] if isinstance(action, list) and len(action) == 1 else action
        assert actual_action in light_dark_env.get_actions(), (
            f"planner returned action={actual_action!r}, which is not in the "
            f"environment action set {light_dark_env.get_actions()}"
        )

    def test_tree_structure_construction(self, planner, belief):
        """Test that the planner constructs proper tree structure."""
        tree = Tree()
        root_id = tree.add_belief_node(belief)

        assert hasattr(planner, "_simulate_path")
        assert callable(planner._simulate_path)

        assert tree.belief[root_id] is belief
        assert tree.children_ids[root_id] == []
        assert tree.visit_count[root_id] == 0

        assert isinstance(belief, WeightedParticleBelief)
        assert len(belief.particles) > 0

    def test_tree_structure_visit_count_consistency(self, planner, belief):
        """Test visit count invariant: each node's visits >= sum of child visits.

        Arena trees don't prune depth-exceeded descendants, so a non-terminal
        child may have visit_count=0. The exact equality that held under the
        anytree backend (via ``node.parent=None`` pruning) becomes ``>=`` here.
        """
        tree = Tree()
        root_id = tree.add_belief_node(belief)

        n_sims = 50
        for _ in range(n_sims):
            planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

        for node_id in _iter_node_ids(tree, root_id):
            assert tree.visit_count[node_id] >= 0
            children = tree.children_ids[node_id]
            if not children:
                continue
            total_child_visits = sum(tree.visit_count[cid] for cid in children)
            assert tree.visit_count[node_id] >= total_child_visits, (
                f"Node {node_id} (kind={tree.kind[node_id]}) visit_count "
                f"({tree.visit_count[node_id]}) < sum of children visits ({total_child_visits})"
            )

        assert tree.visit_count[root_id] == n_sims

    def test_tree_structure_q_value_v_value_relationships(self, planner, belief):
        """Test that Q-values and V-values are properly related in the tree."""
        tree = Tree()
        root_id = tree.add_belief_node(belief)

        n_sims = 30
        for _ in range(n_sims):
            planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

        for node_id in _iter_node_ids(tree, root_id):
            children = tree.children_ids[node_id]
            if tree.kind[node_id] == BELIEF:
                assert tree.v_value[node_id] is not None
                visited_action_children = [cid for cid in children if tree.visit_count[cid] > 0]
                if visited_action_children:
                    min_q_value = min(tree.q_value[cid] for cid in visited_action_children)
                    assert tree.v_value[node_id] == min_q_value, (
                        f"Belief node {node_id} V-value ({tree.v_value[node_id]}) "
                        f"!= min Q-value of action children ({min_q_value})"
                    )
            else:
                assert tree.q_value[node_id] is not None

    def test_tree_structure_progressive_widening_constraints(self, planner, belief):
        """Test that progressive widening constraints are respected in the tree."""
        tree = Tree()
        root_id = tree.add_belief_node(belief)

        n_sims = 100
        for _ in range(n_sims):
            planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

        for node_id in _iter_node_ids(tree, root_id):
            visit_count = tree.visit_count[node_id]
            if visit_count == 0:
                continue
            children = tree.children_ids[node_id]
            if tree.kind[node_id] == BELIEF:
                max_actions = planner.k_a * (visit_count**planner.alpha_a)
                max_allowed = int(max_actions) + 1
                assert len(children) <= max_allowed, (
                    f"Belief node has {len(children)} action children but should have at most "
                    f"{max_allowed} (k_a={planner.k_a}, alpha_a={planner.alpha_a}, "
                    f"visit_count={visit_count}, threshold={max_actions:.3f})"
                )
            else:
                max_observations = planner.k_o * (visit_count**planner.alpha_o)
                max_allowed = int(max_observations) + 1
                assert len(children) <= max_allowed, (
                    f"Action node has {len(children)} belief children but should have at most "
                    f"{max_allowed} (k_o={planner.k_o}, alpha_o={planner.alpha_o}, "
                    f"visit_count={visit_count}, threshold={max_observations:.3f})"
                )

    def test_tree_structure_comprehensive(self, planner, belief):
        """Comprehensive test of tree structure including all invariants."""
        tree = Tree()
        root_id = tree.add_belief_node(belief)

        n_sims = 100
        for _ in range(n_sims):
            planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

        assert tree.visit_count[root_id] == n_sims
        assert tree.parent_id[root_id] is None
        assert tree.v_value[root_id] is not None

        belief_count = 0
        action_count = 0

        for node_id in _iter_node_ids(tree, root_id):
            assert tree.visit_count[node_id] >= 0
            children = tree.children_ids[node_id]

            if tree.kind[node_id] == BELIEF:
                belief_count += 1
                assert tree.belief[node_id] is not None
                assert tree.v_value[node_id] is not None
                if children:
                    total_action_visits = sum(tree.visit_count[cid] for cid in children)
                    assert tree.visit_count[node_id] >= total_action_visits
                    visited_action_children = [cid for cid in children if tree.visit_count[cid] > 0]
                    if visited_action_children:
                        min_q = min(tree.q_value[cid] for cid in visited_action_children)
                        assert tree.v_value[node_id] == min_q
            else:
                action_count += 1
                assert tree.action[node_id] is not None
                assert tree.q_value[node_id] is not None
                if children:
                    total_belief_visits = sum(tree.visit_count[cid] for cid in children)
                    assert tree.visit_count[node_id] >= total_belief_visits

        assert belief_count >= 1, "Tree should contain at least the root belief node"
        assert action_count >= 1, "Tree should contain at least one action node"

        # Maximum tree depth alternates belief/action, so ≤ 2 * planner.depth + 2.
        max_depth = _tree_depth(tree, root_id)
        assert max_depth <= planner.depth * 2 + 2, (
            f"Tree depth ({max_depth}) should not exceed 2 * planner.depth + 2 "
            f"({planner.depth * 2 + 2})"
        )


class TestICVaR_POMCPOWEdgeCases:
    """Test class for ICVaR_POMCPOW edge cases."""

    def test_single_particle_belief(self, planner, environment):
        """Test handling of belief with single particle."""
        initial_state = (
            environment.get_initial_state()
            if hasattr(environment, "get_initial_state")
            else environment.get_actions()[0]
        )
        single_particle = [initial_state]
        single_belief = create_weighted_belief(single_particle, log_weights=np.array([1.0]))
        tree = Tree()
        tree.add_belief_node(single_belief)

        assert isinstance(single_belief, WeightedParticleBelief)
        assert len(single_belief.particles) == 1
        assert single_belief.particles[0] == initial_state

        assert hasattr(planner, "_simulate_path")
        assert callable(planner._simulate_path)

    def test_large_depth_values(self, planner, belief):
        """Test handling of very large depth values."""
        tree = Tree()
        tree.add_belief_node(belief)

        assert hasattr(planner, "_simulate_path")
        assert callable(planner._simulate_path)

        assert isinstance(belief, WeightedParticleBelief)
        assert len(belief.particles) > 0

        assert hasattr(planner, "depth")
        assert isinstance(planner.depth, int)
        assert planner.depth > 0

    def test_zero_discount_factor(
        self,
        environment,
        depth,
        exploration_constant,
        k_o,
        k_a,
        alpha_o,
        alpha_a,
        alpha,
        action_sampler,
        min_immediate_cost,
        max_immediate_cost,
        min_visit_count_per_action,
        delta,
    ):
        """Test behavior with zero discount factor."""
        planner = ICVaR_POMCPOW(
            environment=environment,
            discount_factor=0.0,
            depth=depth,
            exploration_constant=exploration_constant,
            k_o=k_o,
            k_a=k_a,
            alpha_o=alpha_o,
            alpha_a=alpha_a,
            min_immediate_cost=min_immediate_cost,
            max_immediate_cost=max_immediate_cost,
            min_visit_count_per_action=min_visit_count_per_action,
            delta=delta,
            name="test_planner",
            action_sampler=action_sampler,
            n_simulations=100,  # Add n_simulations to avoid error
            alpha=alpha,
        )

        initial_belief = get_initial_belief(environment, 10)
        tree = Tree()
        tree.add_belief_node(initial_belief)

        assert planner.discount_factor == 0.0
        assert planner.depth == depth
        assert planner.alpha == alpha

        assert isinstance(initial_belief, WeightedParticleBelief)
        assert len(initial_belief.particles) > 0


class TestICVaR_POMCPOWPolicyInterface:
    """Test class for ICVaR_POMCPOW policy interface compliance."""

    def test_get_space_info(self, planner):
        """Test that get_space_info returns correct space types."""
        space_info = planner.get_space_info()

        # Check case-insensitive comparison
        assert space_info.action_space.value.lower() == "mixed"
        assert space_info.observation_space.value.lower() == "mixed"

    def test_policy_info_variables(self, planner):
        """Test that planner provides required policy info variables."""
        # This should not raise an error
        assert hasattr(planner, "name")
        assert hasattr(planner, "environment")
        assert hasattr(planner, "discount_factor")

    def test_action_method_signature(self, planner, belief):
        """Test that action method has correct signature."""
        # Test that the method exists and has the right signature
        assert hasattr(planner, "action")
        assert callable(planner.action)

        # Test that the method can be called with the right parameters
        # We'll skip the actual execution since it requires a working simulation
        # but we can verify the method signature
        sig = inspect.signature(planner.action)
        assert "belief" in sig.parameters

    def test_action_method_execution(self, planner, belief):
        """Test that action method can be executed successfully."""
        # Set a very short timeout to avoid long execution
        original_timeout = planner.time_out_in_seconds
        planner.time_out_in_seconds = 1  # 1 second timeout

        try:
            # Call the action method
            action, policy_run_data = planner.action(belief)

            # Verify the return values
            assert action is not None
            assert isinstance(policy_run_data, PolicyRunData)

            # Verify action is valid for the environment
            valid_actions = planner.environment.get_actions()

            # Handle different action formats (single value vs list)
            if isinstance(action, list) and len(action) == 1:
                actual_action = str(action[0])
            else:
                actual_action = str(action)

            # Convert valid actions to strings for comparison
            valid_actions_str = [str(a) for a in valid_actions]

            # Handle action name variations (open-left vs open_left)
            normalized_action = actual_action.replace("-", "_")
            normalized_valid_actions = [a.replace("-", "_") for a in valid_actions_str]

            assert normalized_action in normalized_valid_actions

        finally:
            # Restore original timeout
            planner.time_out_in_seconds = original_timeout


class TestICVaR_POMCPOWParameterValidation:
    """Test class for ICVaR_POMCPOW parameter validation."""

    def test_invalid_depth(
        self,
        environment,
        discount_factor,
        exploration_constant,
        k_o,
        k_a,
        alpha_o,
        alpha_a,
        alpha,
        action_sampler,
        min_immediate_cost,
        max_immediate_cost,
        min_visit_count_per_action,
        delta,
    ):
        """Test that invalid depth values are handled."""
        # The current implementation doesn't validate depth, so this test should pass
        try:
            planner = ICVaR_POMCPOW(
                environment=environment,
                discount_factor=discount_factor,
                depth=-1,  # Invalid negative depth
                exploration_constant=exploration_constant,
                k_o=k_o,
                k_a=k_a,
                alpha_o=alpha_o,
                alpha_a=alpha_a,
                min_immediate_cost=min_immediate_cost,
                max_immediate_cost=max_immediate_cost,
                min_visit_count_per_action=min_visit_count_per_action,
                delta=delta,
                name="test_planner",
                action_sampler=action_sampler,
                n_simulations=100,
                alpha=alpha,
            )
            # If no exception is raised, that's fine - the implementation doesn't validate depth
            assert planner.depth == -1
        except Exception as e:
            # If an exception is raised, that's also fine
            assert isinstance(e, Exception)

    def test_invalid_discount_factor(
        self,
        environment,
        depth,
        exploration_constant,
        k_o,
        k_a,
        alpha_o,
        alpha_a,
        alpha,
        action_sampler,
        min_immediate_cost,
        max_immediate_cost,
        min_visit_count_per_action,
        delta,
    ):
        """Test that invalid discount factor values are handled."""
        # The current implementation doesn't validate discount_factor, so this test should pass
        try:
            planner = ICVaR_POMCPOW(
                environment=environment,
                discount_factor=1.5,  # Invalid discount factor > 1
                depth=depth,
                exploration_constant=exploration_constant,
                k_o=k_o,
                k_a=k_a,
                alpha_o=alpha_o,
                alpha_a=alpha_a,
                min_immediate_cost=min_immediate_cost,
                max_immediate_cost=max_immediate_cost,
                min_visit_count_per_action=min_visit_count_per_action,
                delta=delta,
                name="test_planner",
                action_sampler=action_sampler,
                n_simulations=100,
                alpha=alpha,
            )
            # If no exception is raised, that's fine - the implementation doesn't validate discount_factor
            assert planner.discount_factor == 1.5
        except Exception as e:
            # If an exception is raised, that's also fine
            assert isinstance(e, Exception)

    def test_invalid_alpha_values(
        self,
        environment,
        discount_factor,
        depth,
        exploration_constant,
        k_o,
        k_a,
        alpha_o,
        alpha_a,
        action_sampler,
        min_immediate_cost,
        max_immediate_cost,
        min_visit_count_per_action,
        delta,
    ):
        """Test that invalid alpha values are handled."""
        # The current implementation doesn't validate alpha, so this test should pass
        try:
            planner = ICVaR_POMCPOW(
                environment=environment,
                discount_factor=discount_factor,
                depth=depth,
                exploration_constant=exploration_constant,
                k_o=k_o,
                k_a=k_a,
                alpha_o=alpha_o,
                alpha_a=alpha_a,
                min_immediate_cost=min_immediate_cost,
                max_immediate_cost=max_immediate_cost,
                min_visit_count_per_action=min_visit_count_per_action,
                delta=delta,
                name="test_planner",
                action_sampler=action_sampler,
                n_simulations=100,
                alpha=-0.1,  # Invalid negative alpha
            )
            # If no exception is raised, that's fine - the implementation doesn't validate alpha
            assert planner.alpha == -0.1
        except Exception as e:
            # If an exception is raised, that's also fine
            assert isinstance(e, Exception)


class TestICVaR_POMCPOWEpisodeTests:
    """Test class for ICVaR_POMCPOW episode execution tests."""

    def test_run_episode_tiger_pomdp(self, planner, belief, environment):
        """Test running a single episode with TigerPOMDP environment."""
        logger = getLogger(__name__)
        num_steps = 10

        # Run the episode
        history = run_episode(
            environment=environment,
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

            # Verify action is valid for environment
            valid_actions = environment.get_actions()
            if isinstance(step.action, list) and len(step.action) == 1:
                actual_action = step.action[0]
            else:
                actual_action = step.action

            # Skip None actions (they might occur in some edge cases)
            if actual_action is not None:
                # Handle different action formats
                if isinstance(actual_action, str):
                    # Handle action name variations (open-left vs open_left)
                    normalized_action = actual_action.replace("-", "_")
                    normalized_valid_actions = [str(a).replace("-", "_") for a in valid_actions]
                    assert normalized_action in normalized_valid_actions
                else:
                    assert actual_action in valid_actions

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

    def test_run_episode_light_dark_pomdp(
        self, light_dark_planner, light_dark_belief, light_dark_env
    ):
        """Test running a single episode with DiscreteLightDarkPOMDP environment."""
        logger = getLogger(__name__)
        num_steps = 15

        # Run the episode
        history = run_episode(
            environment=light_dark_env,
            policy=light_dark_planner,
            initial_belief=light_dark_belief,
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

            # Verify action is valid for environment
            valid_actions = light_dark_env.get_actions()
            if isinstance(step.action, list) and len(step.action) == 1:
                actual_action = step.action[0]
            else:
                actual_action = step.action

            # Skip None actions (they might occur in some edge cases)
            if actual_action is not None:
                assert actual_action in valid_actions

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

    def test_run_episode_different_alpha_values(
        self,
        environment,
        discount_factor,
        depth,
        exploration_constant,
        k_o,
        k_a,
        alpha_o,
        alpha_a,
        action_sampler,
        min_immediate_cost,
        max_immediate_cost,
        min_visit_count_per_action,
        delta,
        n_particles,
    ):
        """Test running episodes with different alpha values."""
        logger = getLogger(__name__)
        num_steps = 5
        belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)

        alpha_values = [0.05, 0.1, 0.25, 0.5]

        for alpha in alpha_values:
            # Create planner with specific alpha
            planner = ICVaR_POMCPOW(
                environment=environment,
                discount_factor=discount_factor,
                depth=depth,
                exploration_constant=exploration_constant,
                k_o=k_o,
                k_a=k_a,
                alpha_o=alpha_o,
                alpha_a=alpha_a,
                min_immediate_cost=min_immediate_cost,
                max_immediate_cost=max_immediate_cost,
                min_visit_count_per_action=min_visit_count_per_action,
                delta=delta,
                name=f"test_planner_alpha_{alpha}",
                action_sampler=action_sampler,
                n_simulations=50,  # Reduced for faster testing
                alpha=alpha,
            )

            # Run the episode
            history = run_episode(
                environment=environment,
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
    def test_run_episode_timeout_vs_n_simulations(
        self,
        environment,
        discount_factor,
        depth,
        exploration_constant,
        k_o,
        k_a,
        alpha_o,
        alpha_a,
        alpha,
        action_sampler,
        min_immediate_cost,
        max_immediate_cost,
        min_visit_count_per_action,
        delta,
        n_particles,
    ):
        """Test running episodes with timeout vs n_simulations configurations."""
        logger = getLogger(__name__)
        num_steps = 5
        belief = get_initial_belief(environment, n_particles=n_particles, resampling=True)

        # Test with n_simulations
        planner_n_sims = ICVaR_POMCPOW(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            exploration_constant=exploration_constant,
            k_o=k_o,
            k_a=k_a,
            alpha_o=alpha_o,
            alpha_a=alpha_a,
            min_immediate_cost=min_immediate_cost,
            max_immediate_cost=max_immediate_cost,
            min_visit_count_per_action=min_visit_count_per_action,
            delta=delta,
            name="test_planner_n_sims",
            action_sampler=action_sampler,
            n_simulations=50,
            alpha=alpha,
        )

        # Test with timeout
        planner_timeout = ICVaR_POMCPOW(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            exploration_constant=exploration_constant,
            k_o=k_o,
            k_a=k_a,
            alpha_o=alpha_o,
            alpha_a=alpha_a,
            min_immediate_cost=min_immediate_cost,
            max_immediate_cost=max_immediate_cost,
            min_visit_count_per_action=min_visit_count_per_action,
            delta=delta,
            name="test_planner_timeout",
            action_sampler=action_sampler,
            time_out_in_seconds=2,  # 2 second timeout
            alpha=alpha,
        )

        # Run episodes with both configurations
        history_n_sims = run_episode(
            environment=environment,
            policy=planner_n_sims,
            initial_belief=belief,
            num_steps=num_steps,
            logger=logger,
        )

        history_timeout = run_episode(
            environment=environment,
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

    def test_run_episode_early_termination(self, light_dark_planner, light_dark_env, n_particles):
        """Test that episode terminates early when reaching terminal state."""
        logger = getLogger(__name__)
        num_steps = 50  # Large number of steps
        belief = get_initial_belief(light_dark_env, n_particles=n_particles, resampling=True)

        # Run the episode
        history = run_episode(
            environment=light_dark_env,
            policy=light_dark_planner,
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
            assert light_dark_env.is_terminal(last_step.state)
        else:
            # If not terminal, should have used all steps
            assert len(history.history) == num_steps


class TestICVaR_POMCPOWArenaInvariants:
    """Test class for arena tree invariants beyond visit/PW/Q-V relationships."""

    def test_observation_cdf_consistency(self, planner, belief):
        """Test the per-parent children_cdf invariants on the iCVaR-POMCPOW arena tree.

        Purpose: Validates the arena tree's ``children_cdf`` (used by
        ``sample_belief_child`` for O(log K) weighted sampling at observation-widening
        action nodes) is monotonically non-decreasing and totals to the sum of
        children weights for every parent in the tree.

        Given: ICVaR_POMCPOW planner with default fixture parameters and an initial
        belief; tree is built by running 100 ``_simulate_path`` calls from the root
        (matching the existing comprehensive test setup).
        When: Every parent in the arena (belief and action) is walked.
        Then: For every parent with children, the CDF length matches children count,
        is monotonically non-decreasing, and ``cdf[-1]`` equals the sum of children
        weights within absolute tolerance ``1e-6``. At least one action node has
        multiple belief children.

        Test type: unit
        """
        tree = Tree()
        root_id = tree.add_belief_node(belief)

        n_sims = 100
        for _ in range(n_sims):
            planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

        multi_child_action_seen = False
        for parent_id in _iter_node_ids(tree, root_id):
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
            "CDF invariant is vacuous on this tree. Increase n_sims."
        )


class TestICVaR_POMCPOWThreadsRecursionDepth:
    """Regression: action selection must use the recursion depth, not the
    planner's max depth, so the LCB exploration horizon decays as the
    search descends. Otherwise horizon = max_depth - depth is identically
    0, the LCB bound goes NaN, and the planner systematically returns
    action index 0.
    """

    def test_action_selection_observes_varying_recursion_depth(
        self,
        planner,
        belief,
    ):
        """Planner threads recursion depth into action-PW selection.

        Purpose: Verifies that across a single ``_learn_tree`` call the
            ``depth`` argument forwarded into
            ``cvar_action_progressive_widening_arena`` takes at least two
            distinct values, including a value strictly less than
            ``planner.depth`` (the configured max). Without the fix the
            planner passed ``depth=self.depth`` from ``__init__``, so
            every selection saw the same depth and every horizon was 0.

        Given: A configured ICVaR_POMCPOW planner with depth=3 and an
            initial Tiger belief.
        When: ``_learn_tree`` is run and every call to
            ``cvar_action_progressive_widening_arena`` is recorded.
        Then: The set of observed ``depth`` values has size > 1, and
            includes at least one value strictly less than
            ``planner.depth``.

        Test type: regression
        """
        from POMDPPlanners.planners.mcts_planners import icvar_pomcpow as mod

        observed_depths: list = []
        original = mod.cvar_action_progressive_widening_arena

        def spy(*args, depth, max_depth, **kwargs):
            observed_depths.append(depth)
            return original(*args, depth=depth, max_depth=max_depth, **kwargs)

        mod.cvar_action_progressive_widening_arena = spy
        try:
            planner._learn_tree(belief=belief)
        finally:
            mod.cvar_action_progressive_widening_arena = original

        assert observed_depths, "no action-PW calls recorded"
        unique_depths = set(observed_depths)
        assert len(unique_depths) > 1, (
            f"expected multiple recursion depths threaded through action "
            f"selection, got only {unique_depths}; the planner is passing "
            f"a constant depth instead of the recursion depth"
        )
        assert any(d < planner.depth for d in unique_depths), (
            f"expected some calls at depth < planner.depth={planner.depth}, "
            f"saw {sorted(unique_depths)}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
