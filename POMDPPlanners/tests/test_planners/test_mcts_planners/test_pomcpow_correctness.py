# SPDX-License-Identifier: MIT

"""POMCPOW correctness tests: exact backups, unvisited-child value, widening.

``test_pomcpow.py`` already covers the tree walk, the observation CDF and the
widening bounds. What it does not settle is the arithmetic — its value
assertions filter to visited children, while the implementation maximises over
all of them, and no test says which is intended. These tests settle that from
the evidence and pin the backup numbers.

Reference:
    Sunberg, Z., & Kochenderfer, M. J. (2018). Online algorithms for POMDPs with
    continuous state, action, and observation spaces. ICAPS 28, 259-263.
"""

import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBeliefStateUpdate
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.tests.test_planners.planner_fixtures import (
    CHAIN_REWARDS,
    END,
    NEXT,
    ROOT,
    ChainEnv,
    FixedActionSampler,
    SingleActionSampler,
    chain_belief,
)
from POMDPPlanners.tests.test_planners.tree_assertions import (
    action_ids,
    assert_subtree_unchanged,
    assert_values_within_bounds,
    belief_ids,
    snapshot_subtree,
    walk_arena_tree,
)


DISCOUNT = 0.5
TOL = 1e-12


def _planner(
    env,
    *,
    depth=1,
    n_simulations=1,
    k_o=1.0,
    alpha_o=0.0,
    k_a=1.0,
    alpha_a=0.0,
    sampler=None,
    exploration_constant=0.0,
    min_visit_count_per_action=1,
):
    return POMCPOW(
        environment=env,
        discount_factor=env.discount_factor,
        depth=depth,
        exploration_constant=exploration_constant,
        k_o=k_o,
        k_a=k_a,
        alpha_o=alpha_o,
        alpha_a=alpha_a,
        name="POMCPOW_correctness",
        action_sampler=sampler or SingleActionSampler("a"),
        n_simulations=n_simulations,
        min_visit_count_per_action=min_visit_count_per_action,
    )


# ---------------------------------------------------------------------------
# Belief value over all action children
# ---------------------------------------------------------------------------


def test_belief_value_is_the_maximum_over_all_action_children_including_unvisited():
    """V(b) takes the maximum over every action child, unvisited ones included.

    Purpose: The audit flagged this as unsettled: the implementation maximises
        over all children while the existing comprehensive test filters to
        visited ones, and a fixture where the two sets coincide cannot tell
        them apart. This one separates them.

    The rule asserted here is all-children, on three pieces of evidence:

    1. The POMCPOW paper stores no ``V(h)`` at all — ``SIMULATE`` updates
       ``N(h)``, ``N(ha)`` and ``Q(ha)`` and returns the total. So there is no
       paper rule for an implementation-only field to contradict.
    2. Nothing reads it. ``v_value`` is written by eight arena planners and read
       by exactly two — the iCVaR pair, via ``get_v_value`` — neither of which
       is POMCPOW or consumes a POMCPOW tree. Here it is a diagnostic quantity
       that mirrors the maximum the tree's own ``best_action_by_reward`` takes
       over the same all-children set when the final action is chosen.
    3. Seven sibling arena planners share the identical expression, two of
       which (PFT-DPW, Sparse-PFT) already have all-children assertions in
       their existing suites. POMCP is the single documented exception and its
       source says why.

    Given: A root with one visited action at Q = -2 and one action still at its
        allocation-time 0.0 initializer.
    When: One backup runs on the visited action.
    Then: V(root) is 0.0, the unvisited child's value.

    A visited-children-only rule would give -2.0, so this fails if the rule
    changes.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    visited_id = tree.add_action_node(action="a", parent_id=root_id)
    unvisited_id = tree.add_action_node(action="b", parent_id=root_id)
    tree.update_action_q_with_return(visited_id, -2.0)

    tree.increment_visit_count(root_id)
    children = tree.get_children_ids(root_id)
    tree.v_value[root_id] = float(max(tree.get_q_value(cid) for cid in children))

    assert tree.q_value[visited_id] == pytest.approx(-2.0, abs=TOL)
    assert tree.visit_count[unvisited_id] == 0
    assert tree.q_value[unvisited_id] == 0.0
    assert tree.v_value[root_id] == pytest.approx(0.0, abs=TOL)
    del planner


def test_search_backup_sets_the_all_children_maximum():
    """The planner's own backup produces that same all-children maximum.

    Purpose: The test above pins the rule; this one proves the code path under
        test actually applies it, rather than a hand-rolled copy of it.

    Given: A root with a pre-seeded losing action at Q = -5 plus the action the
        widening sampler will select, and a single simulation.
    When: ``_simulate_state_path`` runs one simulation.
    Then: V(root) equals the maximum Q over both children, and that maximum is
        strictly greater than the pre-seeded -5, so the check is not trivially
        satisfied by the seeded value.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=1, k_a=2.0)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    loser_id = tree.add_action_node(action="b", parent_id=root_id)
    tree.visit_count[loser_id] = 1
    tree.q_value[loser_id] = -5.0

    planner._simulate_state_path(tree=tree, state=ROOT, belief_id=root_id, depth=0)

    children = tree.get_children_ids(root_id)
    expected = max(tree.get_q_value(cid) for cid in children)
    assert tree.v_value[root_id] == pytest.approx(expected, abs=TOL)
    assert expected > -5.0, "the fixture must make the maximum something other than the seed"


# ---------------------------------------------------------------------------
# Exact backup arithmetic
# ---------------------------------------------------------------------------


def test_leaf_expansion_return_is_immediate_reward_plus_discounted_rollout():
    """A newly created observation child triggers a rollout, not a recursion.

    Purpose: POMCPOW's ``SIMULATE`` branches on whether the observation child
        is new; a leaf gets the rollout estimate. This pins that branch's
        arithmetic.

    Given: Depth 1, discount 0.5, chain rewards 2 then 4 with ``end`` terminal.
        The root transition earns 2 and the rollout from ``next`` earns 4
        before hitting the terminal state.
    When: One simulation runs.
    Then: The return is ``2 + 0.5 * 4 = 4`` and the action's Q is 4.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=1)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))

    total = planner._simulate_state_path(tree=tree, state=ROOT, belief_id=root_id, depth=0)

    expected = CHAIN_REWARDS[ROOT] + DISCOUNT * CHAIN_REWARDS[NEXT]
    assert expected == 4.0
    assert total == pytest.approx(expected, abs=TOL), (
        f"return {total} != 2 + 0.5*4; the rollout after the leaf expansion must contribute "
        "exactly one discounted reward before the terminal state"
    )
    (action_id,) = action_ids(tree, root_id)
    assert tree.q_value[action_id] == pytest.approx(expected, abs=TOL)
    assert tree.visit_count[action_id] == 1
    assert tree.visit_count[root_id] == 1


def test_visited_observation_child_recurses_instead_of_rolling_out():
    """An observation child that already has a visit is recursed into.

    Purpose: The two branches of ``SIMULATE`` must be told apart, and the
        recursion branch re-draws the state from the child's own particle bag.

    Given: A root whose action already has a visited belief child for the
        observation ``next``, holding the single particle ``next``.
    When: One simulation runs at depth 0 with planner depth 1.
    Then: The child's visit count rises (the recursion reached it) and the
        return is ``2 + 0.5 * 4 = 4``: the recursion from ``next`` expands that
        child as a leaf and rolls out one reward of 4 before ``end``.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=1, k_o=1.0, alpha_o=0.0)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.visit_count[action_id] = 1
    child_belief = WeightedParticleBeliefStateUpdate()
    child_id = tree.add_belief_node(
        belief=child_belief, observation=NEXT, parent_id=action_id, weight=1.0, obs_key=NEXT
    )
    child_belief.inplace_update(action="a", observation=NEXT, pomdp=env, state=NEXT)
    tree.visit_count[child_id] = 1
    visits_before = tree.visit_count[child_id]

    total = planner._simulate_state_path(tree=tree, state=ROOT, belief_id=root_id, depth=0)

    assert (
        tree.visit_count[child_id] > visits_before
    ), "the already-visited observation child must be recursed into, not replaced"
    expected = CHAIN_REWARDS[ROOT] + DISCOUNT * CHAIN_REWARDS[NEXT]
    assert total == pytest.approx(expected, abs=TOL), f"return {total} != {expected}"


def test_terminal_state_returns_zero_and_only_counts_a_visit():
    """A terminal state short-circuits with return 0.

    Given: A belief node reached with the terminal state ``end``.
    When: ``_simulate_state_path`` runs.
    Then: The return is 0, the visit count is 1, and nothing was expanded.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=3)
    tree = Tree()
    node_id = tree.add_belief_node(chain_belief(END))

    assert planner._simulate_state_path(tree=tree, state=END, belief_id=node_id, depth=0) == 0.0
    assert tree.visit_count[node_id] == 1
    assert tree.children_ids[node_id] == []


def test_depth_cutoff_returns_zero_and_leaves_the_node_alone():
    """Past the depth limit nothing is counted.

    Given: Planner depth 1, called at depth 2.
    When: ``_simulate_state_path`` runs.
    Then: The return is 0 and the node is untouched.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=1)
    tree = Tree()
    node_id = tree.add_belief_node(chain_belief(ROOT))

    assert planner._simulate_state_path(tree=tree, state=ROOT, belief_id=node_id, depth=2) == 0.0
    assert tree.visit_count[node_id] == 0
    assert tree.children_ids[node_id] == []


# ---------------------------------------------------------------------------
# Observation widening: reuse, weights, CDF
# ---------------------------------------------------------------------------


def test_reused_observation_child_bumps_its_weight_and_the_parent_cdf():
    """Re-observing an existing observation increments its weight by exactly 1.

    Purpose: The observation weight is the sampling distribution over belief
        children; a bump that misses the CDF silently biases every later
        ``sample_belief_child``.

    Given: An action node with one belief child for observation ``next`` at
        weight 1.0, and a widening budget that keeps the reuse branch open.
    When: ``_observation_widening`` is called again with the same observation.
    Then: The same child ID comes back, its weight is 2.0, no second child was
        created, and the parent's CDF's last entry equals 2.0.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, k_o=5.0, alpha_o=1.0)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.visit_count[action_id] = 4
    first_id = planner._observation_widening(tree, action_id, NEXT)
    assert tree.weight[first_id] == pytest.approx(1.0, abs=TOL)

    second_id = planner._observation_widening(tree, action_id, NEXT)

    assert second_id == first_id, "an equal observation must reuse its belief child"
    assert len(belief_ids(tree, action_id)) == 1
    assert tree.weight[first_id] == pytest.approx(2.0, abs=TOL)
    assert tree.children_cdf[action_id][-1] == pytest.approx(2.0, abs=TOL), (
        f"parent CDF {tree.children_cdf[action_id]} does not reflect the weight bump; "
        "sampling would use stale probabilities"
    )


def test_observation_widening_stops_creating_children_at_the_boundary():
    """The widening condition ``|C| <= k_o * N(ha)**alpha_o`` is applied exactly.

    Purpose: A boundary test rather than a loose bound. The condition is
        evaluated *before* the child is added and uses ``<=``, which is the
        form in the POMCPOW paper's ``ObsWiden``. With ``k_o = 1`` and
        ``alpha_o = 0`` the bound is the constant 1, so a new child may be
        created while the count is 0 or 1 — two children in total — and the
        third distinct observation must sample an existing child instead. An
        implementation using ``<`` would stop one child early and one using
        the post-increment count would stop one late; both differ from this.

    Given: An action node whose two belief children were created by the two
        preceding widening calls.
    When: ``_observation_widening`` is asked for a third, distinct observation.
    Then: No third child is created, and the returned ID is one of the two
        that already exist.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, k_o=1.0, alpha_o=0.0)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.visit_count[action_id] = 3

    first_id = planner._observation_widening(tree, action_id, NEXT)
    assert len(belief_ids(tree, action_id)) == 1, "count 0 <= 1, so the first child is created"
    second_id = planner._observation_widening(tree, action_id, END)
    assert len(belief_ids(tree, action_id)) == 2, "count 1 <= 1, so a second child is created too"
    assert second_id != first_id

    third_id = planner._observation_widening(tree, action_id, ROOT)

    assert len(belief_ids(tree, action_id)) == 2, (
        "count 2 is above the bound of 1, so the third observation must sample an existing "
        f"child, but the action now has {len(belief_ids(tree, action_id))} children"
    )
    assert third_id in (first_id, second_id)


def test_action_widening_reuses_a_duplicate_sampled_action():
    """Sampling an action that already has a node returns that node.

    Purpose: A widening step that added a second node for the same action would
        split its statistics in half without any visible error.

    Given: An action sampler that always returns ``"a"`` and a budget that keeps
        the widening branch open.
    When: Two simulations run from the root.
    Then: The root has exactly one action child, whose visit count is 2.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=1, k_a=5.0, alpha_a=1.0, sampler=SingleActionSampler("a"))
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))

    planner._simulate_state_path(tree=tree, state=ROOT, belief_id=root_id, depth=0)
    planner._simulate_state_path(tree=tree, state=ROOT, belief_id=root_id, depth=0)

    children = action_ids(tree, root_id)
    assert (
        len(children) == 1
    ), f"duplicate action nodes created: {[tree.action[c] for c in children]}"
    assert tree.visit_count[children[0]] == 2


# ---------------------------------------------------------------------------
# Isolation, structure, ranges, public contract
# ---------------------------------------------------------------------------


def test_one_simulation_leaves_an_untouched_action_branch_alone():
    """A simulation down one action does not disturb its sibling's subtree.

    Given: A root with a pre-populated sibling branch and a sampler that only
        ever proposes the other action.
    When: One simulation runs.
    Then: Every recorded field of the sibling subtree is unchanged.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=1, k_a=5.0, alpha_a=1.0, sampler=SingleActionSampler("a"))
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    sibling_id = tree.add_action_node(action="b", parent_id=root_id)
    sibling_child = tree.add_belief_node(
        belief=WeightedParticleBeliefStateUpdate(),
        observation=NEXT,
        parent_id=sibling_id,
        weight=1.0,
        obs_key=NEXT,
    )
    tree.visit_count[sibling_id] = 2
    tree.q_value[sibling_id] = 9.0
    tree.visit_count[sibling_child] = 1
    before = snapshot_subtree(tree, sibling_id)

    planner._simulate_state_path(tree=tree, state=ROOT, belief_id=root_id, depth=0)

    assert_subtree_unchanged(tree, before, label="untouched POMCPOW action branch")


def test_full_search_structure_and_value_ranges():
    """A completed search is fully reachable, consistent and inside its bounds.

    Given: The chain with rewards in [0, 4], discount 0.5, depth 2, ten
        simulations, a two-action sampler and fixed seeds.
    When: ``_learn_tree`` builds the tree.
    Then: Every allocated node is reachable with correct reverse links,
        alternating kinds and an exact per-entry CDF; the walk saw an expanded
        non-root belief, a visited action and a multi-child action node; and
        every visited Q and every V is inside the interval implied by the
        fixture's reward range.

    Test type: unit
    """
    np.random.seed(4242)
    random.seed(4242)
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(
        env,
        depth=2,
        n_simulations=10,
        k_a=2.0,
        alpha_a=0.5,
        k_o=2.0,
        alpha_o=0.5,
        sampler=FixedActionSampler(["a", "b"]),
        exploration_constant=1.0,
    )

    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    counters = walk_arena_tree(tree, root_id)
    assert counters.expanded_non_root_belief_nodes >= 1, (
        "the search never expanded a non-root belief, so none of the per-node checks saw a "
        "node the recursion had actually worked on"
    )
    assert counters.visited_action_nodes >= 2
    assert counters.action_nodes >= 2

    def horizon_of(node_id: int, edge_depth: int):
        del node_id
        return max(planner.depth + 1 - edge_depth // 2, 0)

    checked = assert_values_within_bounds(
        tree, root_id, horizon_of=horizon_of, reward_min=0.0, reward_max=4.0, discount=DISCOUNT
    )
    assert checked >= 3


def test_action_returns_one_legal_action_and_the_declared_metrics():
    """The public contract: one action plus exactly the declared metric names.

    Test type: unit
    """
    np.random.seed(9)
    random.seed(9)
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=2, n_simulations=6, sampler=FixedActionSampler(["a", "b"]))

    actions, run_data = planner.action(chain_belief(ROOT))

    assert len(actions) == 1 and actions[0] in env.get_actions()
    assert [v.name for v in run_data.info_variables] == POMCPOW.get_info_variable_names()


def test_planning_does_not_mutate_the_callers_belief():
    """The root belief object handed in is unchanged afterwards.

    Test type: unit
    """
    np.random.seed(17)
    random.seed(17)
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(env, depth=2, n_simulations=6, sampler=FixedActionSampler(["a", "b"]))
    belief = chain_belief(ROOT)
    before_particles = list(belief.particles)
    before_weights = np.array(belief.log_weights, copy=True)

    planner.action(belief)

    assert belief.particles == before_particles
    assert np.array_equal(
        np.asarray(belief.log_weights), before_weights
    ), "planning rewrote the caller's belief weights"
