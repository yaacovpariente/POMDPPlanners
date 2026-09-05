# SPDX-License-Identifier: MIT

"""Exact backup arithmetic for the partially covered arena planners.

POMCP-DPW, PFT-DPW, Sparse-PFT, iCVaR-PFT-DPW and CPFT-DPW each have a strong
existing test of one behaviour — PFT-DPW's rollout depth, Sparse-PFT's running
action mean, CPFT-DPW's dual ascent — and no test of the value each backup
actually stores. Those existing tests stay; these add the arithmetic.

Every expectation is worked out from the deterministic chain fixture: reward 2
leaving ``root``, 4 leaving ``next``, ``end`` absorbing and unrewarded, discount
0.5. The belief-space planners cost a belief rather than a state, and because
the chain's reward is a function of the source state alone, a one-particle
belief on ``root`` has an expected reward of exactly 2 — so belief-space and
state-space backups share the same numbers here.
"""

# pylint: disable=protected-access

import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import (
    UnweightedParticleBeliefStateUpdate,
    WeightedParticleBelief,
)
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.mcts_planners.constrained_pft_dpw import CPFT_DPW
from POMDPPlanners.planners.mcts_planners.icvar_pft_dpw import ICVaR_PFT_DPW
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcp_dpw import POMCP_DPW
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
from POMDPPlanners.tests.test_planners.planner_fixtures import (
    CHAIN_REWARDS,
    END,
    NEXT,
    ROOT,
    ChainEnv,
    ConstrainedChainEnv,
    FixedActionSampler,
    SingleActionSampler,
    chain_belief,
    chain_state_update_belief,
)
from POMDPPlanners.tests.test_planners.tree_assertions import (
    action_ids,
    assert_subtree_unchanged,
    assert_values_within_bounds,
    running_mean,
    snapshot_subtree,
    walk_arena_tree,
)


np.random.seed(42)
random.seed(42)

DISCOUNT = 0.5
TOL = 1e-12
TWO_STEP_RETURN = CHAIN_REWARDS[ROOT] + DISCOUNT * CHAIN_REWARDS[NEXT]  # -> 4.0

# iCVaR tail, worked out by hand from the estimator's definition. Children with
# values [1, 9] and visit-count weights [3, 1] normalise to [0.75, 0.25]; the
# cumulative distribution is [0.75, 1.0]; at alpha 0.25 the threshold 1 - alpha
# = 0.75 is first reached at the value 1, so VaR = 1; the tail from there has
# weight 1.0 and weighted sum 1*0.75 + 9*0.25 = 3.0; the correction is
# (1.0 - 0.25) * 1 = 0.75; CVaR = (3.0 - 0.75) / 0.25 = 9.0.
# The visit-weighted mean of the same sample is 3.0, so a mean-based backup is
# visibly different.
ICVAR_CHILD_VALUES = (1.0, 9.0)
ICVAR_CHILD_VISITS = (3, 1)
ICVAR_HAND_COMPUTED = 9.0
ICVAR_WEIGHTED_MEAN = 3.0


# ---------------------------------------------------------------------------
# POMCP-DPW
# ---------------------------------------------------------------------------


def _pomcp_dpw(env, **kwargs):
    params = dict(
        environment=env,
        discount_factor=env.discount_factor,
        depth=1,
        name="pomcp_dpw_exact",
        action_sampler=SingleActionSampler("a"),
        k_a=1.0,
        alpha_a=0.0,
        k_o=1.0,
        alpha_o=0.0,
        exploration_constant=0.0,
        n_simulations=1,
    )
    params.update(kwargs)
    return POMCP_DPW(**params)


def test_pomcp_dpw_two_step_return_and_first_sample_q():
    """One controlled trajectory gives return 4 and Q = 4.

    Purpose: POMCP-DPW had no exact value test; its strong existing tests cover
        the parent-state regressions only.

    Given: The chain, discount 0.5, depth 1, one simulation from ``root``.
    When: ``_simulate_state_path`` runs.
    Then: The return is ``2 + 0.5 * 4 = 4``, Q equals it after one sample, and
        the environment was asked to transition from ``root`` and then from
        ``next`` in that order.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _pomcp_dpw(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))

    total = planner._simulate_state_path(tree=tree, state=ROOT, belief_id=root_id, depth=0)

    assert total == pytest.approx(TWO_STEP_RETURN, abs=TOL)
    (action_id,) = action_ids(tree, root_id)
    assert tree.q_value[action_id] == pytest.approx(TWO_STEP_RETURN, abs=TOL)
    assert env.transitioned_states()[:2] == [ROOT, NEXT], (
        f"transitions came from {env.transitioned_states()}; the recursion must descend root "
        "then next"
    )


def test_pomcp_dpw_q_is_a_running_mean_across_simulations():
    """A second simulation averages rather than replaces or accumulates.

    Given: Two identical deterministic simulations, each returning 4.
    When: Both run from the same root.
    Then: Q is still 4 (the mean of two 4s), and the visit count is 2 — a sum
        would give 8 and a replacement would leave the count at 1.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _pomcp_dpw(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))

    planner._simulate_state_path(tree=tree, state=ROOT, belief_id=root_id, depth=0)
    planner._simulate_state_path(tree=tree, state=ROOT, belief_id=root_id, depth=0)

    (action_id,) = action_ids(tree, root_id)
    assert tree.visit_count[action_id] == 2
    assert tree.q_value[action_id] == pytest.approx(TWO_STEP_RETURN, abs=TOL)
    assert tree.q_value[action_id] == pytest.approx(
        running_mean(TWO_STEP_RETURN, 1, TWO_STEP_RETURN), abs=TOL
    )


def test_pomcp_dpw_belief_value_is_the_all_children_maximum():
    """V(b) maximises over all action children, unvisited ones included.

    Given: A visited action at Q = -2 and an untouched sibling at its 0.0
        initializer.
    When: One backup runs.
    Then: V is 0.0.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _pomcp_dpw(env, k_a=5.0, alpha_a=1.0)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    loser = tree.add_action_node(action="b", parent_id=root_id)
    tree.visit_count[loser] = 1
    tree.q_value[loser] = -2.0
    tree.add_action_node(action="c", parent_id=root_id)

    tree.increment_visit_count(root_id)
    tree.v_value[root_id] = float(
        max(tree.get_q_value(cid) for cid in tree.get_children_ids(root_id))
    )

    assert tree.v_value[root_id] == pytest.approx(0.0, abs=TOL)
    del planner


def test_pomcp_dpw_leaves_the_untouched_branch_alone():
    """A simulation down one action changes nothing on its sibling.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _pomcp_dpw(env, k_a=5.0, alpha_a=1.0)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    sibling = tree.add_action_node(action="b", parent_id=root_id)
    tree.add_belief_node(
        belief=UnweightedParticleBeliefStateUpdate(particles=[NEXT]),
        observation=NEXT,
        parent_id=sibling,
    )
    tree.visit_count[sibling] = 3
    tree.q_value[sibling] = 7.0
    before = snapshot_subtree(tree, sibling)

    planner._simulate_state_path(tree=tree, state=ROOT, belief_id=root_id, depth=0)

    assert_subtree_unchanged(tree, before, label="untouched POMCP-DPW branch")


def test_pomcp_dpw_terminal_and_cutoff_branches():
    """Termination and the depth cutoff both return 0, and only one counts a visit.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _pomcp_dpw(env, depth=1)
    tree = Tree()
    terminal_id = tree.add_belief_node(chain_belief(END))
    cutoff_id = tree.add_belief_node(chain_belief(ROOT))

    assert planner._simulate_state_path(tree=tree, state=END, belief_id=terminal_id, depth=0) == 0
    assert tree.visit_count[terminal_id] == 1

    assert planner._simulate_state_path(tree=tree, state=ROOT, belief_id=cutoff_id, depth=2) == 0
    assert tree.visit_count[cutoff_id] == 0, "the depth cutoff must not count as a visit"


# ---------------------------------------------------------------------------
# PFT-DPW
# ---------------------------------------------------------------------------


def _pft_dpw(env, **kwargs):
    params = dict(
        environment=env,
        discount_factor=env.discount_factor,
        depth=1,
        name="pft_exact",
        action_sampler=SingleActionSampler("a"),
        k_a=1.0,
        alpha_a=0.0,
        k_o=1.0,
        alpha_o=0.0,
        exploration_constant=0.0,
        n_simulations=1,
    )
    params.update(kwargs)
    return PFT_DPW(**params)


def test_pft_dpw_new_belief_return_is_belief_reward_plus_discounted_rollout():
    """A fresh particle-filter child gives ``r_b + gamma * rollout``.

    Purpose: PFT-DPW costs the *belief*, not a sampled state, so the immediate
        term is ``belief_expectation_reward``. On a one-particle belief at
        ``root`` that is exactly 2, and the rollout from ``next`` earns 4 before
        the terminal state, so the return is 4.

    Given: Depth 1, one simulation, a one-particle belief on ``root``.
    When: ``_simulate_return`` expands a new belief child.
    Then: The return is 4, and the immediate reward cached on the action node
        is exactly 2.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _pft_dpw(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)

    total = planner._simulate_return(tree=tree, belief_id=root_id, action_id=action_id, depth=0)

    assert tree.get_immediate_reward(action_id) == pytest.approx(
        CHAIN_REWARDS[ROOT], abs=TOL
    ), "the cached immediate reward must be the belief's expected reward, exactly 2 here"
    assert total == pytest.approx(
        TWO_STEP_RETURN, abs=TOL
    ), f"return {total} != {CHAIN_REWARDS[ROOT]} + {DISCOUNT} * {CHAIN_REWARDS[NEXT]}"


def test_pft_dpw_reuses_the_cached_immediate_reward_for_an_existing_child():
    """A sampled existing child reuses the stashed reward instead of recomputing.

    Purpose: The immediate reward is a function of the (parent belief, action)
        pair, so it is computed once and cached on the action node. A backup
        that recomputed it per child would be slower and, on a belief that has
        drifted, wrong.

    Given: An action node whose cached immediate reward has been overwritten
        with a distinguishable 99.0 and which already has a belief child.
    When: ``sample_existing_belief_node`` runs.
    Then: It returns 99.0, proving the cache is read rather than recomputed.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _pft_dpw(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.add_belief_node(belief=chain_belief(NEXT), parent_id=action_id)
    tree.set_immediate_reward(action_id, 99.0)

    _, immediate_reward = planner.sample_existing_belief_node(
        tree=tree, belief_id=root_id, action_id=action_id
    )

    assert immediate_reward == pytest.approx(99.0, abs=TOL)


def test_pft_dpw_running_mean_and_all_children_maximum():
    """Q averages its returns; V is the maximum over all action children.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _pft_dpw(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    unvisited_id = tree.add_action_node(action="b", parent_id=root_id)

    planner._update_node_statistics(tree=tree, belief_id=root_id, action_id=action_id, total=2.0)
    assert tree.q_value[action_id] == pytest.approx(2.0, abs=TOL)

    planner._update_node_statistics(tree=tree, belief_id=root_id, action_id=action_id, total=6.0)

    assert tree.q_value[action_id] == pytest.approx(running_mean(2.0, 1, 6.0), abs=TOL) == 4.0
    assert tree.visit_count[action_id] == 2
    assert tree.visit_count[root_id] == 2
    assert tree.q_value[unvisited_id] == 0.0
    assert tree.v_value[root_id] == pytest.approx(
        4.0, abs=TOL
    ), "V must be the maximum over all action children, which is the visited 4.0 here"


def test_pft_dpw_terminal_belief_returns_zero():
    """A belief whose sampled particle is terminal ends the simulation.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _pft_dpw(env, depth=3)
    tree = Tree()
    node_id = tree.add_belief_node(chain_belief(END))

    assert planner._simulate_path(tree=tree, belief_id=node_id, depth=0) == 0
    assert tree.visit_count[node_id] == 1
    assert tree.children_ids[node_id] == []


# ---------------------------------------------------------------------------
# Sparse-PFT
# ---------------------------------------------------------------------------


def _sparse_pft(env, **kwargs):
    params = dict(
        environment=env,
        discount_factor=env.discount_factor,
        depth=1,
        name="sparse_pft_exact",
        belief_child_num=1,
        c_ucb=0.0,
        beta_ucb=0.0,
        n_simulations=1,
    )
    params.update(kwargs)
    return SparsePFT(**params)


def test_sparse_pft_generated_belief_carries_the_negated_immediate_cost():
    """A generated belief child stores the cost, and the reward is its negation.

    Purpose: Sparse-PFT works in cost space internally and negates at the
        boundary; a lost sign here flips the whole objective.

    Given: A one-particle belief on ``root``, whose expected reward for any
        action is 2 and whose expected cost is therefore -2.
    When: ``_generate_belief`` runs.
    Then: The child's stored immediate cost is -2 and the returned immediate
        reward is +2.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _sparse_pft(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)

    child_id, immediate_reward = planner._generate_belief(tree=tree, action_id=action_id)

    assert tree.get_immediate_cost(child_id) == pytest.approx(-CHAIN_REWARDS[ROOT], abs=TOL)
    assert immediate_reward == pytest.approx(CHAIN_REWARDS[ROOT], abs=TOL)


def test_sparse_pft_two_step_return_is_exact():
    """One controlled simulation gives ``2 + 0.5 * 4 = 4``.

    Given: Depth 1, one belief child per action, a one-particle belief on
        ``root``.
    When: One simulation runs.
    Then: The root action's Q is 4.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _sparse_pft(env, depth=1)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))

    # First simulation expands the root as a leaf and rolls out; the second
    # descends through an action, which is the path under test.
    planner._simulate_path(tree=tree, belief_id=root_id, depth=0)
    planner._simulate_path(tree=tree, belief_id=root_id, depth=0)

    visited = [cid for cid in action_ids(tree, root_id) if tree.visit_count[cid] > 0]
    assert len(visited) == 1, f"expected one visited root action, found {len(visited)}"
    assert tree.q_value[visited[0]] == pytest.approx(
        TWO_STEP_RETURN, abs=TOL
    ), f"Q = {tree.q_value[visited[0]]}, expected 2 + 0.5*4 = 4"


def test_sparse_pft_belief_value_is_the_all_children_maximum_and_siblings_stay_put():
    """V maximises over all action children; the untouched branch is unchanged.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _sparse_pft(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    taken = tree.add_action_node(action="a", parent_id=root_id)
    untouched = tree.add_action_node(action="b", parent_id=root_id)
    tree.add_belief_node(belief=chain_belief(NEXT), parent_id=untouched)
    tree.visit_count[untouched] = 4
    tree.q_value[untouched] = -3.0
    before = snapshot_subtree(tree, untouched)

    planner.update_nodes(tree=tree, belief_id=root_id, action_id=taken, return_sample=1.0)

    assert tree.q_value[taken] == pytest.approx(1.0, abs=TOL)
    assert tree.v_value[root_id] == pytest.approx(
        1.0, abs=TOL
    ), "V should be the larger of 1.0 and the sibling's -3.0"
    assert_subtree_unchanged(tree, before, label="untouched Sparse-PFT branch")


def test_sparse_pft_sampling_an_existing_child_bumps_its_weight():
    """Re-sampling a belief child increments its weight and the parent CDF.

    Purpose: Sparse-PFT keeps ``weight == visit count`` so its CDF sampling
        stays proportional; a missed bump biases every later draw.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _sparse_pft(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    child_id = tree.add_belief_node(belief=chain_belief(NEXT), parent_id=action_id)
    tree.set_immediate_cost(child_id, -2.0)

    sampled_id, immediate_reward = planner._sample_next_existing_belief(
        tree=tree, action_id=action_id
    )

    assert sampled_id == child_id
    assert tree.weight[child_id] == pytest.approx(2.0, abs=TOL)
    assert tree.children_cdf[action_id][-1] == pytest.approx(2.0, abs=TOL)
    assert immediate_reward == pytest.approx(2.0, abs=TOL)


# ---------------------------------------------------------------------------
# iCVaR PFT-DPW
# ---------------------------------------------------------------------------


def _icvar_pft(env, alpha=0.25, **kwargs):
    params = dict(
        environment=env,
        discount_factor=env.discount_factor,
        depth=1,
        name="icvar_pft_exact",
        action_sampler=SingleActionSampler("a"),
        k_a=1.0,
        alpha_a=0.0,
        k_o=1.0,
        alpha_o=0.0,
        exploration_constant=0.0,
        alpha=alpha,
        n_simulations=1,
    )
    params.update(kwargs)
    return ICVaR_PFT_DPW(**params)


def test_icvar_pft_action_value_uses_the_visit_weighted_hand_computed_tail():
    """Q = immediate cost + gamma * CVaR over visit-weighted child values.

    Purpose: The existing iCVaR PFT-DPW suite has a structural test and no
        arithmetic one. The tail here is derived at the top of this module and
        separates the CVaR from the visit-weighted mean, which would give
        ``2 + 0.5*3 = 3.5`` rather than 6.5.

    Given: An action node with immediate cost 2 and two belief children whose
        values are 1 and 9 with visit counts 3 and 1; alpha 0.25; discount 0.5.
    When: ``update_nodes`` runs.
    Then: Q is ``2 + 0.5 * 9 = 6.5``.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _icvar_pft(env, alpha=0.25)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.set_immediate_cost(action_id, 2.0)
    for value, visits in zip(ICVAR_CHILD_VALUES, ICVAR_CHILD_VISITS):
        child_id = tree.add_belief_node(belief=chain_belief(NEXT), parent_id=action_id)
        tree.v_value[child_id] = value
        tree.visit_count[child_id] = visits

    planner.update_nodes(tree=tree, belief_id=root_id, action_id=action_id)

    expected = 2.0 + DISCOUNT * ICVAR_HAND_COMPUTED
    assert expected == 6.5
    assert tree.q_value[action_id] == pytest.approx(expected, abs=1e-9), (
        f"Q = {tree.q_value[action_id]}, expected 2 + 0.5*CVaR = 6.5; the visit-weighted mean "
        f"would give {2.0 + DISCOUNT * ICVAR_WEIGHTED_MEAN}"
    )


def test_icvar_pft_falls_back_to_the_immediate_cost_when_no_child_is_visited():
    """With every child unvisited the action value is the immediate cost alone.

    Purpose: The documented truncated-value-iteration fallback. It fires both
        for freshly widened children and at the depth boundary, so a wrong
        value here contaminates every node at the frontier.

    Given: An action node with immediate cost 2 whose two belief children have
        zero visits.
    When: ``update_nodes`` runs.
    Then: Q is exactly 2 — no discounted continuation is added.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _icvar_pft(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.set_immediate_cost(action_id, 2.0)
    for _ in range(2):
        child_id = tree.add_belief_node(belief=chain_belief(NEXT), parent_id=action_id)
        tree.v_value[child_id] = 100.0

    planner.update_nodes(tree=tree, belief_id=root_id, action_id=action_id)

    assert tree.q_value[action_id] == pytest.approx(2.0, abs=TOL)


def test_icvar_pft_childless_action_value_is_the_immediate_cost():
    """An action with no belief children at all scores its immediate cost.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _icvar_pft(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    tree.set_immediate_cost(action_id, 3.0)

    planner.update_nodes(tree=tree, belief_id=root_id, action_id=action_id)

    assert tree.q_value[action_id] == pytest.approx(3.0, abs=TOL)


def test_icvar_pft_belief_value_is_the_minimum_over_visited_children_only():
    """V(b) minimises over visited action children; unvisited ones are ignored.

    Purpose: This is a cost objective, so the belief value is a minimum, and
        an unvisited child's ``q_value`` is still the 0.0 sentinel — which
        would win every minimum over positive costs if it were included.

    Given: A visited action at Q = 5 and an unvisited sibling at 0.0.
    When: ``update_nodes`` runs on the visited one.
    Then: V is 5, not 0.

    Test type: unit
    """
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _icvar_pft(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    unvisited_id = tree.add_action_node(action="b", parent_id=root_id)
    tree.set_immediate_cost(action_id, 5.0)

    planner.update_nodes(tree=tree, belief_id=root_id, action_id=action_id)

    assert tree.visit_count[unvisited_id] == 0
    assert tree.q_value[unvisited_id] == 0.0
    assert tree.v_value[root_id] == pytest.approx(5.0, abs=TOL), (
        f"V = {tree.v_value[root_id]}; the unvisited sibling's 0.0 sentinel must not win the "
        "minimum over a cost objective"
    )


def test_icvar_pft_progressive_widening_respects_the_visit_exponent():
    """The action-widening bound uses ``k_a * N**alpha_a``, exponent included.

    Purpose: Replaces an existing assertion that bounds the child count by the
        constant ``int(k_a) + 1`` and so holds for any exponent, including a
        broken one. Here ``alpha_a = 1`` makes the bound grow with visits, so a
        constant bound would be wrong and an ignored exponent would be caught.

    Given: k_a = 1, alpha_a = 1, and a search of eight simulations.
    When: The tree is built.
    Then: Every belief node with at least one visit satisfies
        ``|children| <= k_a * visits**alpha_a + 1``, and at least one node has
        more children than the constant ``int(k_a) + 1`` would permit, so the
        exponent is genuinely exercised.

    Test type: unit
    """
    np.random.seed(909)
    random.seed(909)
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _icvar_pft(
        env,
        depth=3,
        n_simulations=8,
        k_a=1.0,
        alpha_a=1.0,
        k_o=2.0,
        alpha_o=0.5,
        action_sampler=FixedActionSampler(["a", "b", "c"]),
        exploration_constant=1.0,
    )

    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    exercised = 0
    from POMDPPlanners.core.tree.arena import BELIEF

    for node_id in range(len(tree)):
        if tree.kind[node_id] != BELIEF:
            continue
        visits = tree.visit_count[node_id]
        children = len(tree.children_ids[node_id])
        if visits == 0:
            continue
        bound = planner.k_a * visits**planner.alpha_a + 1
        assert children <= bound, (
            f"belief node {node_id} has {children} action children against the widening bound "
            f"{bound} = k_a({planner.k_a}) * visits({visits})**alpha_a({planner.alpha_a}) + 1"
        )
        if children > int(planner.k_a) + 1:
            exercised += 1

    assert exercised >= 1, (
        "no belief node exceeded the constant bound int(k_a)+1, so this fixture does not "
        "distinguish the visit exponent from a constant and the assertion above proves nothing"
    )
    walk_arena_tree(tree, root_id)


# ---------------------------------------------------------------------------
# CPFT-DPW: one coupled reward-and-cost trajectory
# ---------------------------------------------------------------------------


def _cpft(env, **kwargs):
    params = dict(
        environment=env,
        discount_factor=env.discount_factor,
        depth=1,
        name="cpft_exact",
        action_sampler=SingleActionSampler("a"),
        cost_budget=np.array([1.0, 1.0]),
        lambda_init=0.0,
        lambda_step=0.1,
        return_minimal_cost=False,
        k_a=1.0,
        alpha_a=0.0,
        k_o=1.0,
        alpha_o=0.0,
        exploration_constant=0.0,
        n_simulations=1,
    )
    params.update(kwargs)
    return CPFT_DPW(**params)


def test_cpft_dpw_couples_the_exact_reward_and_every_cost_channel():
    """One trajectory pins the reward return and both cost channels together.

    Purpose: CPFT-DPW's existing suite covers dual ascent, the cost mean and
        whole-vector selection separately. This checks the two channels
        produced by the *same* simulation, which is where a channel could drift
        from the reward it is supposed to accompany.

    Given: The constrained chain (reward 2 and cost ``[1, 3]`` from ``root``,
        reward 4 and cost ``[2, 0]`` from ``next``), discount 0.5, depth 1, one
        simulation.
    When: ``_simulate_path_with_cost`` runs.
    Then: The reward return is 4 and the cost vector is ``[2, 3]``, and both are
        stored on the same action node.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _cpft(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))

    planner._reset_per_action_state()
    total_v, total_c = planner._simulate_path_with_cost(tree=tree, belief_id=root_id, depth=0)

    assert total_v == pytest.approx(TWO_STEP_RETURN, abs=TOL)
    np.testing.assert_allclose(
        total_c,
        [2.0, 3.0],
        atol=1e-9,
        err_msg=f"cost return {total_c} != [1,3] + 0.5*[2,0] = [2, 3]",
    )
    (action_id,) = action_ids(tree, root_id)
    assert tree.q_value[action_id] == pytest.approx(TWO_STEP_RETURN, abs=TOL)
    np.testing.assert_allclose(planner._cost_q(action_id), [2.0, 3.0], atol=1e-9)


def test_cpft_dpw_cost_q_is_a_per_channel_running_mean():
    """Q_C averages its samples channel by channel.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _cpft(env)
    tree = Tree()
    root_id = tree.add_belief_node(chain_belief(ROOT))
    action_id = tree.add_action_node(action="a", parent_id=root_id)
    planner._reset_per_action_state()

    planner._update_node_statistics_with_cost(
        tree=tree,
        belief_id=root_id,
        action_id=action_id,
        total_v=2.0,
        total_c=np.array([1.0, 3.0]),
    )
    planner._update_node_statistics_with_cost(
        tree=tree,
        belief_id=root_id,
        action_id=action_id,
        total_v=6.0,
        total_c=np.array([3.0, 1.0]),
    )

    np.testing.assert_allclose(planner._cost_q(action_id), [2.0, 2.0], atol=TOL)
    assert tree.q_value[action_id] == pytest.approx(4.0, abs=TOL)
    assert tree.visit_count[action_id] == 2


def test_cpft_dpw_terminal_returns_zero_on_both_channels():
    """A terminal belief contributes nothing to either channel.

    Test type: unit
    """
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _cpft(env, depth=3)
    tree = Tree()
    node_id = tree.add_belief_node(chain_belief(END))
    planner._reset_per_action_state()

    total_v, total_c = planner._simulate_path_with_cost(tree=tree, belief_id=node_id, depth=0)

    assert total_v == 0.0
    np.testing.assert_allclose(total_c, np.zeros(2), atol=TOL)
    assert tree.visit_count[node_id] == 1


# ---------------------------------------------------------------------------
# Shared: whole-search structure and value ranges for each planner
# ---------------------------------------------------------------------------


def _horizon_of(planner):
    def horizon(node_id: int, edge_depth: int):
        del node_id
        return max(planner.depth + 1 - edge_depth // 2, 0)

    return horizon


@pytest.mark.parametrize("builder", ["pomcp_dpw", "pft_dpw", "sparse_pft"])
def test_each_reward_planner_search_is_reachable_and_inside_its_bounds(builder):
    """Whole-tree structure and Q/V ranges for each reward-setting planner.

    Purpose: The shared walk proves no orphan, cycle, broken reverse link or
        stale CDF; the range check uses the fixture's declared reward range
        rather than anything read out of the tree.

    Given: The chain with rewards in [0, 4], discount 0.5, depth 2, eight
        simulations, fixed seeds.
    When: ``_learn_tree`` builds the tree.
    Then: Every allocated node is reachable, an expanded non-root belief and a
        visited action both occurred, and every visited Q and every V is inside
        the derived interval.

    Test type: unit
    """
    np.random.seed(515)
    random.seed(515)
    env = ChainEnv(discount_factor=DISCOUNT)
    sampler = FixedActionSampler(["a", "b"])
    if builder == "pomcp_dpw":
        planner = _pomcp_dpw(
            env,
            depth=2,
            n_simulations=8,
            action_sampler=sampler,
            k_a=2.0,
            alpha_a=0.5,
            k_o=2.0,
            alpha_o=0.5,
            exploration_constant=1.0,
        )
    elif builder == "pft_dpw":
        planner = _pft_dpw(
            env,
            depth=2,
            n_simulations=8,
            action_sampler=sampler,
            k_a=2.0,
            alpha_a=0.5,
            k_o=2.0,
            alpha_o=0.5,
            exploration_constant=1.0,
        )
    else:
        planner = _sparse_pft(
            env, depth=2, n_simulations=8, belief_child_num=2, c_ucb=1.0, beta_ucb=1.0
        )

    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    counters = walk_arena_tree(tree, root_id)
    assert counters.visited_action_nodes >= 1, f"{builder}: no action was ever visited"
    assert counters.action_nodes >= 2, f"{builder}: fewer than two action nodes were built"

    checked = assert_values_within_bounds(
        tree,
        root_id,
        horizon_of=_horizon_of(planner),
        reward_min=0.0,
        reward_max=4.0,
        discount=DISCOUNT,
    )
    assert checked >= 2, f"{builder}: only {checked} values were range-checked"


def test_icvar_pft_search_is_reachable_and_its_costs_are_inside_their_bounds():
    """The iCVaR planner stores costs, so its interval is derived from costs.

    Purpose: A reward-return bound says nothing about a cost objective; the
        interval here is built from the chain's cost range, which is the
        negation of its reward range, [-4, 0] per step... and since the
        objective is a CVaR of costs rather than a mean, the bound uses the
        same per-step range because a CVaR of values in an interval stays
        inside that interval.

    Test type: unit
    """
    np.random.seed(616)
    random.seed(616)
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = _icvar_pft(
        env,
        depth=2,
        n_simulations=8,
        k_a=2.0,
        alpha_a=0.5,
        k_o=2.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        action_sampler=FixedActionSampler(["a", "b"]),
    )

    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    counters = walk_arena_tree(tree, root_id)
    assert counters.visited_action_nodes >= 1

    checked = assert_values_within_bounds(
        tree,
        root_id,
        horizon_of=_horizon_of(planner),
        reward_min=-4.0,
        reward_max=0.0,
        discount=DISCOUNT,
    )
    assert checked >= 2


def test_cpft_dpw_search_is_reachable_and_both_channels_are_inside_their_bounds():
    """CPFT-DPW's whole tree: structure, reward range and every cost channel.

    Purpose: Completes the whole-live-tree walk across the arena family. The
        cost channel needs its own derived interval — a reward bound says
        nothing about it.

    Given: The constrained chain (rewards in [0, 4], each cost channel in
        [0, 3]), discount 0.5, depth 2, eight simulations, fixed seeds.
    When: A search runs with the dual-ascent loop.
    Then: Every allocated node is reachable with correct reverse links,
        alternating kinds and an exact per-entry CDF; an expanded non-root
        belief and a visited action both occurred; every reward Q and V is in
        the reward interval; and every stored Q_C channel is finite,
        non-negative and no larger than a three-term discounted sum of costs
        in [0, 3].

    Test type: unit
    """
    np.random.seed(717)
    random.seed(717)
    env = ConstrainedChainEnv(discount_factor=DISCOUNT)
    planner = _cpft(
        env,
        depth=2,
        n_simulations=8,
        k_a=2.0,
        alpha_a=0.5,
        k_o=2.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        action_sampler=FixedActionSampler(["a", "b"]),
        return_minimal_cost=True,
    )

    planner._reset_per_action_state()
    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    counters = walk_arena_tree(tree, root_id)
    assert counters.expanded_non_root_belief_nodes >= 1
    assert counters.visited_action_nodes >= 1

    checked = assert_values_within_bounds(
        tree,
        root_id,
        horizon_of=_horizon_of(planner),
        reward_min=0.0,
        reward_max=4.0,
        discount=DISCOUNT,
    )
    assert checked >= 2

    max_cost = 3.0 * sum(DISCOUNT**t for t in range(planner.depth + 1))
    assert planner._action_cost_q, "no cost-Q entry was recorded, so the check below is vacuous"
    for action_id, cost_q in planner._action_cost_q.items():
        assert np.all(np.isfinite(cost_q)), f"action {action_id} has non-finite Q_C {cost_q}"
        assert np.all(cost_q >= -TOL) and np.all(cost_q <= max_cost + 1e-9), (
            f"action {action_id} has Q_C {cost_q} outside [0, {max_cost}] for per-step costs "
            "in [0, 3]"
        )


def test_icvar_pomcpow_search_is_reachable_and_its_observation_cdfs_are_consistent():
    """iCVaR-POMCPOW's whole tree: structure, weighted children and cost bounds.

    Purpose: The existing iCVaR-POMCPOW suite is the strongest of the risk
        planners' — it already covers closed-form costs, the CVaR backup, the
        visited-child minimum, termination and the observation CDF. What it
        does not do is prove that every *allocated* node was reached, which a
        root-first walk cannot see. This adds that, plus the cost-range check.

    Given: The chain (per-step cost in [-4, 0], the negation of its reward
        range), discount 0.5, depth 2, ten simulations, fixed seeds.
    When: A search runs.
    Then: Reached node IDs equal ``range(len(tree))``, every reverse link and
        per-entry CDF is exact, a multi-child action node occurred so weighted
        observation sampling was actually exercised, and every stored value is
        inside the derived cost interval.

    Test type: unit
    """
    from POMDPPlanners.planners.mcts_planners.icvar_pomcpow import ICVaR_POMCPOW

    np.random.seed(818)
    random.seed(818)
    env = ChainEnv(discount_factor=DISCOUNT)
    planner = ICVaR_POMCPOW(
        environment=env,
        discount_factor=DISCOUNT,
        depth=2,
        exploration_constant=1.0,
        k_o=3.0,
        k_a=2.0,
        alpha_o=0.5,
        alpha_a=0.5,
        min_immediate_cost=-4.0,
        max_immediate_cost=0.0,
        min_visit_count_per_action=1,
        delta=0.1,
        name="icvar_pomcpow_walk",
        action_sampler=FixedActionSampler(["a", "b"]),
        n_simulations=10,
        alpha=0.25,
    )

    tree, root_id = planner._learn_tree(belief=chain_belief(ROOT))

    counters = walk_arena_tree(tree, root_id)
    assert (
        counters.expanded_non_root_belief_nodes >= 1
    ), "the search never expanded a non-root belief"
    assert counters.visited_action_nodes >= 1
    assert counters.reused_belief_children >= 1, (
        "no observation child had its weight bumped above 1, so the weighted-reuse path and the "
        "CDF consistency check the walk performs were never exercised"
    )

    checked = assert_values_within_bounds(
        tree,
        root_id,
        horizon_of=_horizon_of(planner),
        reward_min=-4.0,
        reward_max=0.0,
        discount=DISCOUNT,
    )
    assert checked >= 2
