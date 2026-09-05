# SPDX-License-Identifier: MIT

"""POMCP correctness tests: state threading, exact backups, isolation, bounds.

The existing ``test_pomcp.py`` checks that the returned action is legal, that
the tree has the expected shape, and that visit counts add up. None of that
distinguishes POMCP from a planner that discounts wrongly, counts a reward
twice, or feeds the generative model the wrong state. These tests pin the
arithmetic of Silver & Veness (2010) ``SIMULATE`` on fixtures small enough to
work out by hand.

Reference:
    Silver, D., & Veness, J. (2010). Monte-Carlo Planning in Large POMDPs.
    NeurIPS 23.
"""

import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import (
    UnweightedParticleBeliefStateUpdate,
    WeightedParticleBelief,
)
from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.tests.test_planners.planner_fixtures import (
    CHAIN_REWARDS,
    END,
    NEXT,
    ROOT,
    THREAD_A,
    THREAD_B,
    THREAD_REWARDS,
    ChainEnv,
    SelfLoopEnv,
    chain_belief,
    chain_state_update_belief,
)
from POMDPPlanners.tests.test_planners.tree_assertions import (
    action_ids,
    assert_subtree_unchanged,
    assert_values_within_bounds,
    belief_ids,
    running_mean,
    snapshot_subtree,
    walk_arena_tree,
)


DISCOUNT = 0.5
# Absolute tolerance for a sum of at most four exactly representable binary
# fractions (rewards are integers, the discount is 0.5). Any real arithmetic
# error in these fixtures is O(1), so this is purely float hygiene.
TOL = 1e-12


def _planner(environment, depth: int, n_simulations: int = 1, exploration_constant: float = 0.0):
    return POMCP(
        environment=environment,
        discount_factor=environment.discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        name="POMCP_correctness",
        n_simulations=n_simulations,
    )


# ---------------------------------------------------------------------------
# State threading (Silver & Veness 2010: SEARCH draws s once, SIMULATE reuses it)
# ---------------------------------------------------------------------------


def test_simulate_uses_the_threaded_state_not_a_redraw_from_the_node_belief():
    """The generative model is called on the state passed in, not a fresh draw.

    Purpose: Pins SIMULATE's use of ``s`` from the parent transition.

    Given: A belief node whose particle bag holds only ``THREAD_B`` (reward 10),
        an already-expanded action child so the leaf branch is not taken, and a
        *different* state ``THREAD_A`` (reward 1) threaded in. Each state is its
        own successor and depth is 0, so the return is exactly the source
        state's reward.
    When: ``_simulate_state_path`` runs one simulation.
    Then: The environment is asked to transition from ``THREAD_A``, the return
        is 1.0, and the action's Q is 1.0.

    A planner that re-draws the state from ``tree.get_belief(belief_id)``
    transitions from ``THREAD_B`` and returns 10.0, so this fixture separates
    the two rules instead of making them agree.

    Test type: unit
    """
    env = SelfLoopEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=0)

    tree = Tree()
    node_id = tree.add_belief_node(UnweightedParticleBeliefStateUpdate(particles=[THREAD_B]))
    action_id = tree.add_action_node(action="a", parent_id=node_id)

    return_sample = planner._simulate_state_path(
        tree=tree, state=THREAD_A, belief_id=node_id, depth=0
    )

    assert env.transitioned_states() == [THREAD_A], (
        f"POMCP transitioned from {env.transitioned_states()}; SIMULATE must call G(s, a) on "
        f"the threaded state {THREAD_A!r}, not on a state re-drawn from the node's belief bag "
        f"(which holds only {THREAD_B!r})"
    )
    assert return_sample == pytest.approx(THREAD_REWARDS[THREAD_A], abs=TOL), (
        f"return {return_sample} != reward({THREAD_A}) = {THREAD_REWARDS[THREAD_A]}; "
        f"a re-draw would have produced {THREAD_REWARDS[THREAD_B]}"
    )
    assert tree.q_value[action_id] == pytest.approx(THREAD_REWARDS[THREAD_A], abs=TOL)


def test_non_root_belief_bag_gains_the_threaded_state():
    """``B(h) <- B(h) union {s}`` adds the state the simulation arrived with.

    Purpose: Pins the particle-bag update, which is how POMCP's belief at a
        non-root history is represented at all.

    Given: A non-root belief node holding one particle ``THREAD_B`` and a
        simulation threaded in with ``THREAD_A``.
    When: One simulation completes and ``update_nodes`` runs.
    Then: The node's particle list is ``[THREAD_B, THREAD_A]``.

    A planner that re-draws from the bag before the backup appends a duplicate
    of ``THREAD_B``, so the bag can never gain a distinct particle and the
    belief collapses to its first state. That failure is invisible to any test
    that only reads visit counts.

    Test type: unit
    """
    env = SelfLoopEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=0)

    tree = Tree()
    root_id = tree.add_belief_node(UnweightedParticleBeliefStateUpdate(particles=[THREAD_A]))
    root_action_id = tree.add_action_node(action="a", parent_id=root_id)
    node_id = tree.add_belief_node(
        belief=UnweightedParticleBeliefStateUpdate(particles=[THREAD_B]),
        observation=THREAD_B,
        parent_id=root_action_id,
    )
    tree.add_action_node(action="a", parent_id=node_id)

    planner._simulate_state_path(tree=tree, state=THREAD_A, belief_id=node_id, depth=0)

    bag = tree.get_belief(node_id)
    # Narrow the declared ``Belief`` to the concrete type POMCP allocates for
    # its own nodes; ``particles`` is that subclass's field, not the base's.
    assert isinstance(bag, UnweightedParticleBeliefStateUpdate)
    assert bag.particles == [THREAD_B, THREAD_A], (
        f"belief bag at node {node_id} is {bag.particles}; "
        f"expected the arriving state {THREAD_A!r} appended to the existing {THREAD_B!r}"
    )


def test_root_belief_is_not_mutated_by_planning():
    """Planning leaves the caller's root belief object alone.

    Purpose: The root belief belongs to the episode runner; a planner that
        appends its own simulated particles to it corrupts the caller's state.

    Given: A root belief with a recorded particle list.
    When: ``action()`` runs a fixed number of simulations.
    Then: The caller's belief still holds exactly the particles it started with.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=2, n_simulations=5)
    belief = chain_belief(ROOT)
    before = list(belief.particles)

    planner.action(belief)

    assert belief.particles == before, (
        f"root belief particles changed from {before} to {belief.particles}; POMCP's "
        "update_nodes deliberately skips the root, so planning must not mutate it"
    )


# ---------------------------------------------------------------------------
# Exact backup arithmetic
# ---------------------------------------------------------------------------


def test_two_step_return_counts_each_reward_once_with_the_right_discount():
    """Rewards 2 then 4 at discount 0.5 give a root return of exactly 4.

    Purpose: Separates the correct backup from the three plausible wrong ones.

    Given: ``ChainEnv`` (reward 2 leaving ``root``, 4 leaving ``next``,
        ``end`` terminal), discount 0.5, planner depth 1, one simulation from a
        root belief concentrated on ``root``.
    When: One simulation runs through an already-expanded root.
    Then: The return is ``2 + 0.5 * 4 = 4``.

    Dropping the discount gives 6; double-counting the first reward gives 6;
    dropping the future term gives 2. All three differ from 4.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=1)

    tree = Tree()
    root_id = tree.add_belief_node(chain_state_update_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)

    return_sample = planner._simulate_state_path(tree=tree, state=ROOT, belief_id=root_id, depth=0)

    expected = CHAIN_REWARDS[ROOT] + DISCOUNT * CHAIN_REWARDS[NEXT]
    assert return_sample == pytest.approx(expected, abs=TOL), (
        f"return {return_sample} != {expected} = {CHAIN_REWARDS[ROOT]} + {DISCOUNT} * "
        f"{CHAIN_REWARDS[NEXT]}"
    )
    assert tree.q_value[action_id] == pytest.approx(expected, abs=TOL)
    assert env.transitioned_states() == [ROOT, NEXT], (
        f"transitions came from {env.transitioned_states()}; the recursion must descend "
        f"{ROOT!r} then {NEXT!r}"
    )


def test_terminal_transition_adds_no_continuation_and_bumps_only_visits():
    """A terminal state returns 0 and increments the node's visit count only.

    Purpose: Terminal reward is counted once, by the transition into the
        terminal state, and nothing accrues after it.

    Given: A belief node holding the terminal state ``end``.
    When: ``_simulate_state_path`` is called on it.
    Then: The return is 0, the visit count went from 0 to 1, no action child
        was created, and the environment was never asked to transition.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=3)

    tree = Tree()
    node_id = tree.add_belief_node(chain_state_update_belief(END))

    result = planner._simulate_state_path(tree=tree, state=END, belief_id=node_id, depth=0)

    assert result == 0
    assert tree.visit_count[node_id] == 1
    assert tree.children_ids[node_id] == [], "a terminal belief must not be expanded"
    assert env.transitioned_states() == [], (
        f"environment was asked to transition from {env.transitioned_states()} after a "
        "terminal state"
    )


def test_depth_cutoff_returns_zero_without_touching_the_tree():
    """Past the depth limit the simulation returns 0 and changes nothing.

    Purpose: The cutoff is a separate branch from termination and must not
        bump visits or expand actions.

    Given: Planner depth 1 and a call at depth 2.
    When: ``_simulate_state_path`` runs.
    Then: The return is 0 and the node's visit count and children are unchanged.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=1)

    tree = Tree()
    node_id = tree.add_belief_node(chain_state_update_belief(ROOT))

    result = planner._simulate_state_path(tree=tree, state=ROOT, belief_id=node_id, depth=2)

    assert result == 0
    assert tree.visit_count[node_id] == 0, "the depth cutoff must not count as a visit"
    assert tree.children_ids[node_id] == []


def test_action_q_is_the_running_mean_of_its_returns():
    """Q(h,a) after two returns is their mean, and the visit count is 2.

    Purpose: Pins the incremental-mean update ``q + (g - q)/n``.

    Given: An action node updated with returns 2.0 then 6.0.
    When: ``update_action_q_with_return`` is applied twice, as the backup does.
    Then: Q is 2.0 after the first and 4.0 after the second, the value the
        hand-written running mean gives, and it lies strictly between the old
        mean and the new sample.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    tree = Tree()
    root_id = tree.add_belief_node(chain_state_update_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    del env

    tree.update_action_q_with_return(action_id, 2.0)
    assert tree.q_value[action_id] == pytest.approx(2.0, abs=TOL), "first sample must be itself"
    assert tree.visit_count[action_id] == 1

    tree.update_action_q_with_return(action_id, 6.0)
    expected = running_mean(previous_mean=2.0, previous_count=1, new_sample=6.0)
    assert expected == 4.0
    assert tree.q_value[action_id] == pytest.approx(expected, abs=TOL)
    assert tree.visit_count[action_id] == 2
    assert 2.0 < tree.q_value[action_id] < 6.0, "a sample mean lies between the old mean and g"


def test_belief_value_is_the_maximum_over_visited_children_only():
    """POMCP's V(h) ignores zero-visit children, which carry no estimate.

    Purpose: POMCP is the one arena planner in this repository that filters to
        visited children, and its source says why: an unvisited child still
        holds the allocation-time ``q_value = 0.0`` sentinel, which would win
        every maximum over a tree of negative estimates. This test pins that
        documented departure so it cannot be "unified" away by accident.

    Given: A root with a visited action at Q = -2 and an unvisited sibling
        still at its 0.0 initializer.
    When: ``update_nodes`` backs a return of -2 up.
    Then: V equals -2, the visited child's Q, not the unvisited 0.0.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=1)

    tree = Tree()
    root_id = tree.add_belief_node(chain_state_update_belief(ROOT))
    visited_id = tree.add_action_node(action="a", parent_id=root_id)
    unvisited_id = tree.add_action_node(action="b", parent_id=root_id)

    planner.update_nodes(
        tree=tree, belief_id=root_id, action_id=visited_id, return_sample=-2.0, state=ROOT
    )

    assert tree.visit_count[unvisited_id] == 0
    assert tree.q_value[unvisited_id] == 0.0, "an untouched action keeps its 0.0 sentinel"
    assert tree.v_value[root_id] == pytest.approx(-2.0, abs=TOL), (
        f"V(root) = {tree.v_value[root_id]}; POMCP maximises over visited children only, so "
        "the unvisited sibling's 0.0 sentinel must not win"
    )


def test_a_simulation_down_one_branch_leaves_the_sibling_untouched():
    """Only the visited branch's statistics change.

    Purpose: Catches aliasing and stray writes across sibling subtrees.

    Given: A root with two action children, one of which already has a belief
        child of its own, and a simulation forced down the other.
    When: One simulation runs.
    Then: Every recorded field of the untouched branch — visits, Q, V, weight,
        cached immediate reward, child list, CDF and belief particles — is
        unchanged.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=1)

    tree = Tree()
    root_id = tree.add_belief_node(chain_state_update_belief(ROOT))
    taken_id = tree.add_action_node(action="a", parent_id=root_id)
    untouched_id = tree.add_action_node(action="b", parent_id=root_id)
    untouched_child = tree.add_belief_node(
        belief=UnweightedParticleBeliefStateUpdate(particles=[NEXT]),
        observation=NEXT,
        parent_id=untouched_id,
    )
    tree.visit_count[untouched_id] = 3
    tree.q_value[untouched_id] = 7.0
    tree.visit_count[untouched_child] = 2
    before = snapshot_subtree(tree, untouched_id)

    # Force the "a" branch: every child but ``taken_id`` already has visits, and
    # get_explored_action_node picks uniformly among the zero-visit children.
    planner._simulate_state_path(tree=tree, state=ROOT, belief_id=root_id, depth=0)

    assert tree.visit_count[taken_id] == 1, "the forced branch must be the one that got the visit"
    assert_subtree_unchanged(tree, before, label="untouched action branch")


# ---------------------------------------------------------------------------
# Whole-search structure, coverage counters and value ranges
# ---------------------------------------------------------------------------


def test_full_search_topology_reachability_and_value_bounds():
    """Every allocated node is reachable, well-linked, and inside its bounds.

    Purpose: Combines the structural walk (which also proves no orphan, cycle,
        broken reverse link or stale CDF exists) with a Q/V range check derived
        from the fixture's declared reward range rather than from the tree.

    Given: ``ChainEnv`` with rewards in [0, 4], discount 0.5, depth 2, eight
        simulations, fixed seeds.
    When: ``_learn_tree`` builds the tree.
    Then: The walk reaches exactly ``range(len(tree))``; it saw at least one
        expanded non-root belief, one visited and one unvisited action, so the
        conditional checks were not vacuous; and every visited action Q and
        belief V lies in the interval a three-term discounted sum of rewards in
        [0, 4] with a zero continuation allows.

    Test type: unit
    """
    np.random.seed(20260905)
    random.seed(20260905)
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=2, n_simulations=8)

    tree, root_id = planner._learn_tree(belief=chain_state_update_belief(ROOT))

    counters = walk_arena_tree(tree, root_id)
    assert counters.belief_nodes >= 2, f"only {counters.belief_nodes} belief nodes were built"
    assert counters.action_nodes >= 2
    assert counters.expanded_non_root_belief_nodes >= 1, (
        "no non-root belief node was expanded, so the recursion never went past the root and "
        "the per-node checks below never saw an interesting case"
    )
    assert counters.visited_action_nodes >= 1
    assert counters.unvisited_action_nodes >= 1, (
        "every action was visited, so the unvisited-child branch of the value rule was never "
        "exercised"
    )

    # Remaining reward terms: a belief node at edge depth d sits at planning
    # depth d // 2, and the planner allows planning depths 0..self.depth, so
    # ``self.depth + 1 - d // 2`` terms can still be collected. Action nodes
    # share their parent belief's horizon: Q(h,a) is one reward plus the
    # discounted value of the child belief, which is the same term count.
    def horizon_of(node_id: int, edge_depth: int):
        del node_id
        return max(planner.depth + 1 - edge_depth // 2, 0)

    checked = assert_values_within_bounds(
        tree,
        root_id,
        horizon_of=horizon_of,
        reward_min=0.0,
        reward_max=4.0,
        discount=DISCOUNT,
    )
    assert checked >= 3, f"only {checked} values were range-checked; the check was near-vacuous"


def test_visit_accounting_at_the_root():
    """The root's visit count and its children's visits agree with the trace.

    Purpose: Ties the counters to the fixed simulation count rather than
        asserting a loose inequality that any tree satisfies.

    Given: Four simulations on the deterministic chain with a fixed seed.
    When: ``_learn_tree`` runs.
    Then: The root has exactly four visits (one per completed simulation, none
        of which is cut off at the root), and the sum of the root actions'
        visits is three — the first simulation finds the root a leaf, expands
        it and rolls out without selecting any action.

    Test type: unit
    """
    np.random.seed(7)
    random.seed(7)
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=2, n_simulations=4)

    tree, root_id = planner._learn_tree(belief=chain_state_update_belief(ROOT))

    assert (
        tree.visit_count[root_id] == 4
    ), f"root visits {tree.visit_count[root_id]} != 4 completed simulations"
    root_action_visits = sum(tree.visit_count[cid] for cid in action_ids(tree, root_id))
    assert root_action_visits == 3, (
        f"root action visits sum to {root_action_visits}; the first simulation expands the root "
        "as a leaf and rolls out without crediting any action, so three of four remain"
    )
    assert root_action_visits <= tree.visit_count[root_id]


def test_observation_children_are_reused_not_duplicated():
    """Two simulations reaching the same observation share one belief child.

    Purpose: POMCP keys belief children by observation; a lookup that misses
        would grow a fresh child per simulation and destroy the particle bag.

    Given: The deterministic chain, where the observation after leaving
        ``root`` is always ``next``, and three simulations.
    When: ``_learn_tree`` runs.
    Then: Each visited root action has exactly one belief child, and that
        child's particle bag has one entry per simulation that reached it.

    Test type: unit
    """
    np.random.seed(11)
    random.seed(11)
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=2, n_simulations=6)

    tree, root_id = planner._learn_tree(belief=chain_state_update_belief(ROOT))

    reused = 0
    for action_id in action_ids(tree, root_id):
        children = belief_ids(tree, action_id)
        assert len(children) <= 1, (
            f"action {action_id} ({tree.action[action_id]!r}) has {len(children)} belief "
            "children, but the chain emits a single deterministic observation"
        )
        if children and tree.visit_count[action_id] > 1:
            reused += 1
            child_belief = tree.get_belief(children[0])
            assert isinstance(child_belief, UnweightedParticleBeliefStateUpdate)
            assert child_belief.particles, "a reached belief child must hold particles"
    assert reused >= 1, "no action was entered twice, so observation reuse was never exercised"


def test_kinds_alternate_and_only_belief_nodes_carry_beliefs():
    """Belief and action layers alternate and each carries only its own payload.

    Purpose: A payload written to the wrong node kind is invisible to value
        checks but breaks every consumer of the tree.

    Given: A small completed search.
    When: Every logical node is inspected.
    Then: Action nodes have an action and no belief; belief nodes have a belief
        and no action; the root alone has no parent.

    Test type: unit
    """
    np.random.seed(3)
    random.seed(3)
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=2, n_simulations=5)

    tree, root_id = planner._learn_tree(belief=chain_state_update_belief(ROOT))

    roots = [i for i in range(len(tree)) if tree.parent_id[i] is None]
    assert roots == [root_id], f"expected exactly one root, found {roots}"
    for node_id in range(len(tree)):
        if tree.kind[node_id] == ACTION:
            assert tree.action[node_id] in env.get_actions()
            assert tree.belief[node_id] is None, f"action node {node_id} carries a belief"
        else:
            assert tree.kind[node_id] == BELIEF
            assert tree.belief[node_id] is not None
            assert tree.action[node_id] is None, f"belief node {node_id} carries an action"


# ---------------------------------------------------------------------------
# Public contract
# ---------------------------------------------------------------------------


def test_action_returns_one_action_and_the_declared_metrics():
    """``action()`` returns a single legal action and every declared metric.

    Purpose: The episode runner executes the returned list and the statistics
        layer reads the metric names, so both shapes are contracts.

    Given: A small search on the chain.
    When: ``action()`` is called.
    Then: One legal action comes back, and the metric names exactly match
        ``get_info_variable_names()``.

    Test type: unit
    """
    np.random.seed(5)
    random.seed(5)
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=2, n_simulations=6)

    actions, run_data = planner.action(chain_belief(ROOT))

    assert len(actions) == 1 and actions[0] in env.get_actions()
    assert [v.name for v in run_data.info_variables] == POMCP.get_info_variable_names()


def test_terminal_belief_short_circuits_before_any_search():
    """A wholly terminal belief returns a legal action with no metrics and no tree.

    Purpose: The terminal branch must not run a search at all.

    Given: A belief every particle of which is the terminal ``end`` state.
    When: ``action()`` is called.
    Then: One legal action comes back, ``info_variables`` is empty, and the
        environment was never asked to transition.

    Test type: unit
    """
    np.random.seed(5)
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=2, n_simulations=6)
    belief = WeightedParticleBelief(particles=[END, END], log_weights=np.array([-1.0, -1.0]))

    actions, run_data = planner.action(belief)

    assert len(actions) == 1 and actions[0] in env.get_actions()
    assert run_data.info_variables == []
    assert (
        env.transitioned_states() == []
    ), "a terminal belief must not trigger any generative-model call"


def test_repeated_calls_build_independent_trees_and_leave_the_first_metrics_alone():
    """A second ``action()`` neither reuses nor rewrites the first call's data.

    Purpose: ``_learn_tree`` builds a fresh tree per call; the statistics layer
        keeps the first call's ``PolicyRunData`` and must still see the values
        it was given.

    Given: Two consecutive calls with the same simulation count and seed.
    When: The first call's metrics are snapshotted before the second runs.
    Then: The snapshot still holds the same names and values afterwards, and
        the second call's root visit count equals the simulation count again
        rather than twice it.

    Test type: unit
    """
    np.random.seed(13)
    random.seed(13)
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=2, n_simulations=4)
    belief = chain_belief(ROOT)

    _, first = planner.action(belief)
    snapshot = [(v.name, v.value) for v in first.info_variables]

    _, second = planner.action(belief)

    assert [
        (v.name, v.value) for v in first.info_variables
    ] == snapshot, "the first call's PolicyRunData changed when the second call ran"
    root_visits = dict((v.name, v.value) for v in second.info_variables)["root_visit_count"]
    assert root_visits == 4, (
        f"second call reports {root_visits} root visits; each action() must start a fresh tree, "
        "so the count is the simulation count and not an accumulation"
    )
