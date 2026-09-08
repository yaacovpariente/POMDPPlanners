# SPDX-License-Identifier: MIT

"""AR-DESPOT correctness: the scale, the regularized value, blocking, anytime.

``test_ardespot.py`` checks that the planner returns a legal action and a
well-formed tree. That is also true of plain DESPOT, and of an AR-DESPOT whose
value scale is wrong, whose blocking never fires, or which descends on the
unregularized bound. These tests pin the eight things AR-DESPOT does
differently from :class:`DESPOT` (module docstring of
``POMDPPlanners.planners.scenario_tree_planners.ardespot``), each on a fixture
where AR-DESPOT's rule and DESPOT's plausible alternative give *different*
numbers -- so a test that passes cannot be passing for the wrong planner.

References:
    Ye, Somani, Hsu & Lee (2017), DESPOT: Online POMDP Planning with
    Regularization, JAIR 58, Algorithm 1 and equation (12).
"""

# pylint: disable=protected-access

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.belief import UnweightedParticleBeliefStateUpdate
from POMDPPlanners.core.tree.arena import BELIEF, Tree
from POMDPPlanners.planners.scenario_tree_planners.ardespot import (
    ARDESPOT,
    _ARActionData,
    _ARBeliefData,
)
from POMDPPlanners.planners.scenario_tree_planners.despot import DESPOT
from POMDPPlanners.tests.test_planners.planner_fixtures import (
    CHAIN_REWARDS,
    END,
    NEXT,
    ROOT,
    ChainEnv,
    chain_belief,
)


DISCOUNT = 0.5

#: Rewards are integers and the discount is 0.5, so every expected value below
#: is an exactly representable binary fraction. Anything above this tolerance
#: is a real arithmetic error, not float accumulation.
TOL = 1e-12

#: ``ChainEnv`` declares ``reward_range=(0.0, 4.0)``. Written out rather than
#: read off the planner, so the expected upper bounds do not come from the code
#: under test.
CHAIN_MAX_REWARD = 4.0


def _planner(
    environment: Any,
    depth: int,
    n_scenarios: int = 4,
    n_simulations: int = 10,
    **overrides: Any,
) -> ARDESPOT:
    params: Dict[str, Any] = dict(
        environment=environment,
        discount_factor=environment.discount_factor,
        depth=depth,
        name="ARDESPOT_correctness",
        n_scenarios=n_scenarios,
        n_simulations=n_simulations,
        pruning_constant=0.0,
    )
    params.update(overrides)
    return ARDESPOT(**params)


def _geometric(discount: float, n_terms: int) -> float:
    """``sum_{t<n} g^t``, written out so no test reads it off the planner."""
    return float(sum(discount**t for t in range(n_terms)))


def _belief_nodes(tree: Tree) -> List[int]:
    return [node_id for node_id in range(len(tree)) if tree.kind[node_id] == BELIEF]


# ---------------------------------------------------------------------------
# Synthetic trees: bounds set by hand, so the rules can be tested in isolation
# ---------------------------------------------------------------------------


def _add_belief(  # pylint: disable=too-many-arguments
    tree: Tree,
    depth: int,
    lower: float,
    regularized: float,
    upper: float,
    n_scenarios_here: int,
    n_scenarios_total: int,
    parent_id: Optional[int] = None,
    default_value: Optional[float] = None,
    observation: Any = None,
) -> int:
    """A belief node whose three quantities are whatever the test says.

    ``l`` lives in ``lower_confidence_bound``, ``U`` in
    ``upper_confidence_bound`` and ``mu`` in ``v_value``, matching the planner.
    """
    node_id = tree.add_belief_node(
        UnweightedParticleBeliefStateUpdate(particles=[NEXT] * n_scenarios_here),
        observation=observation,
        weight=n_scenarios_here / n_scenarios_total,
        parent_id=parent_id,
    )
    tree.lower_confidence_bound[node_id] = lower
    tree.v_value[node_id] = regularized
    tree.upper_confidence_bound[node_id] = upper
    tree.data[node_id] = _ARBeliefData(
        depth=depth,
        default_value=lower if default_value is None else default_value,
        initial_upper=upper,
        expanded=parent_id is None,
        scenario_ids=list(range(n_scenarios_here)),
    )
    return node_id


def _add_action(tree: Tree, parent_id: int, rho: float, reward_sum: float, label: str) -> int:
    action_id = tree.add_action_node(action=label, parent_id=parent_id)
    tree.set_immediate_reward(action_id, reward_sum)
    tree.data[action_id] = _ARActionData(rho=rho)
    return action_id


# ---------------------------------------------------------------------------
# 1-2. The unnormalized scale, and mu_0
# ---------------------------------------------------------------------------


def test_leaf_lower_bound_carries_weight_and_discount_inside_it():
    """``l_0(b) = (|Phi_b|/K) * gamma^Delta * L_0(b)``, not ``L_0(b)``.

    Purpose: this is the single change from which every other difference
    follows. DESPOT stores the conditional ``L_0`` and divides weights back out
    at each backup; AR-DESPOT multiplies them in once and never divides again.
    A node holding 2 of 4 scenarios at depth 1 separates the two by a factor of
    ``0.5 * 0.5 = 4``, so neither can pass the other's assertion.

    Given: ``ChainEnv`` at ``gamma = 0.5``, ``K = 4``, ``depth = 2``. A node at
        depth 1 holding scenarios ``[0, 1]``, both in state ``next``.
    Expect: the conditional default value is ``r(next) = 4`` over the one
        remaining step, so ``L_0 = 4`` and ``l_0 = (2/4) * 0.5^1 * 4 = 1.0``.
        DESPOT would have written ``4.0``.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=2)
    planner._streams = None
    planner.use_determinized_scenarios = False

    tree = Tree()
    node_id = planner._add_belief_node(
        tree=tree,
        states=[NEXT, NEXT],
        scenario_ids=[0, 1],
        depth=1,
        parent_id=None,
        observation=None,
        obs_key=None,
    )

    conditional = CHAIN_REWARDS[NEXT]
    assert tree.data[node_id].default_value == pytest.approx(
        (2 / 4) * DISCOUNT**1 * conditional, abs=TOL
    )
    assert tree.data[node_id].default_value == pytest.approx(1.0, abs=TOL)
    assert tree.data[node_id].default_value != pytest.approx(conditional, abs=TOL)


def test_leaf_upper_bound_stays_conditional():
    """``U`` is the one quantity that is *not* rescaled.

    Purpose: rescaling ``U`` too would look tidier and would break equation
    (12), which multiplies ``U`` by ``w * gamma^Delta`` itself. If the column
    already held the scaled value the blocker test would apply the factor
    twice and block far too eagerly.

    Given: the same depth-1 node holding 2 of 4 scenarios, one step from the
        horizon.
    Expect: ``U = R_max * sum_{t<1} gamma^t = 4.0``, unscaled -- not
        ``(2/4) * 0.5 * 4 = 1.0``.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=2)
    planner._streams = None
    planner.use_determinized_scenarios = False

    tree = Tree()
    node_id = planner._add_belief_node(
        tree=tree,
        states=[NEXT, NEXT],
        scenario_ids=[0, 1],
        depth=1,
        parent_id=None,
        observation=None,
        obs_key=None,
    )
    assert tree.upper_confidence_bound[node_id] == pytest.approx(
        CHAIN_MAX_REWARD * _geometric(DISCOUNT, 1), abs=TOL
    )


def test_mu_0_is_the_better_of_defaulting_and_paying_one_lambda_to_plan():
    """``mu_0(b) = max{ l_0(b), w * gamma^Delta * U_0(b) - lambda }``.

    Purpose: ``mu`` is the regularized value, and an unexpanded node's two
    options are "run the default policy" (free) and "plan here" (costs one
    ``lambda`` before it can earn anything). Initialising ``mu_0`` to the
    scaled ``U_0`` alone would forget the cost; initialising it to ``l_0``
    alone would make the root gap zero and stop the search before it began.

    Given: a root at depth 0 holding all 4 scenarios in state ``root``,
        ``depth = 2``, so ``L_0 = 2 + 0.5*4 = 4`` and
        ``U_0 = 4 * (1 + 0.5) = 6``.
    Expect: at ``lambda = 0.5``, ``mu_0 = max(4, 6 - 0.5) = 5.5``; at
        ``lambda = 3``, the ``-lambda`` term wins nothing and
        ``mu_0 = max(4, 3) = 4``, which is exactly ``l_0`` -- so the root gap
        is zero and no trial is worth running.
    """
    for pruning_constant, expected in ((0.5, 5.5), (3.0, 4.0)):
        environment = ChainEnv(discount_factor=DISCOUNT)
        planner = _planner(environment, depth=2, pruning_constant=pruning_constant)
        planner._streams = None
        planner.use_determinized_scenarios = False

        tree = Tree()
        node_id = planner._add_belief_node(
            tree=tree,
            states=[ROOT] * 4,
            scenario_ids=[0, 1, 2, 3],
            depth=0,
            parent_id=None,
            observation=None,
            obs_key=None,
        )
        assert tree.data[node_id].default_value == pytest.approx(4.0, abs=TOL)
        assert tree.upper_confidence_bound[node_id] == pytest.approx(6.0, abs=TOL)
        assert tree.v_value[node_id] == pytest.approx(expected, abs=TOL)


# ---------------------------------------------------------------------------
# 3. rho
# ---------------------------------------------------------------------------


def test_rho_is_scaled_by_gamma_depth_over_K_and_charged_one_lambda():
    """``rho(ba) = gamma^Delta * (sum_phi r) / K - lambda``.

    Purpose: three plausible wrong versions -- dividing by the node's own
    scenario count instead of ``K``, forgetting ``gamma^Delta``, and omitting
    ``lambda`` -- each give a different number on this fixture, so the test
    distinguishes all four.

    Given: ``ChainEnv``, ``K = 4``, a node at depth 1 holding 2 scenarios in
        state ``next``, ``lambda = 0.25``. Each scenario earns
        ``r(next) = 4``, so ``sum_phi r = 8``.
    Expect: ``rho = 0.5^1 * 8 / 4 - 0.25 = 0.75``. Dividing by 2 rather than
        ``K`` would give ``1.75``; dropping ``gamma^Delta`` would give
        ``1.75``; dropping ``lambda`` would give ``1.0``.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=3, pruning_constant=0.25)
    planner._streams = None
    planner.use_determinized_scenarios = False

    tree = Tree()
    node_id = planner._add_belief_node(
        tree=tree,
        states=[NEXT, NEXT],
        scenario_ids=[0, 1],
        depth=1,
        parent_id=None,
        observation=None,
        obs_key=None,
    )
    planner._expand(tree=tree, belief_id=node_id)

    for action_id in tree.get_children_ids(node_id):
        assert tree.data[action_id].rho == pytest.approx(0.75, abs=TOL)
        # The arena column keeps the raw environment reward, so the generic
        # tree metrics never report a lambda-penalised quantity as a reward.
        assert tree.get_immediate_reward(action_id) == pytest.approx(8.0, abs=TOL)


# ---------------------------------------------------------------------------
# 4-5. Action bounds: plain sums for l and mu, a weighted mean for U
# ---------------------------------------------------------------------------


def test_action_lower_and_regularized_bounds_are_plain_sums_over_children():
    """``ba_l = rho + sum_o l(b_o)``: no weights, because ``l`` already carries them.

    Purpose: DESPOT's rule is a weighted *mean*, ``rho + gamma * sum_o
    (w_o/w_b) l(b_o)``. On two children of unequal size with different bounds
    the mean and the sum differ, and the sum is additionally not discounted
    again -- the discount is inside the children.

    Given: a parent holding 4 of 4 scenarios; child A has 3 scenarios and
        ``l = 1.0``, child B has 1 and ``l = 5.0``; ``rho = 2.0``.
    Expect: ``ba_l = 2 + 1 + 5 = 8.0``. DESPOT's weighted, discounted mean
        would give ``2 + 0.5*(0.75*1 + 0.25*5) = 3.0``.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=3)

    tree = Tree()
    root_id = _add_belief(tree, 0, 0.0, 0.0, 0.0, 4, 4)
    action_id = _add_action(tree, root_id, rho=2.0, reward_sum=0.0, label="a")
    _add_belief(tree, 1, 1.0, 3.0, 9.0, 3, 4, parent_id=action_id)
    _add_belief(tree, 1, 5.0, 6.0, 7.0, 1, 4, parent_id=action_id)

    planner._update_action_bounds(tree=tree, action_id=action_id, parent_count=4)

    assert tree.lower_confidence_bound[action_id] == pytest.approx(8.0, abs=TOL)
    assert tree.data[action_id].mu == pytest.approx(2.0 + 3.0 + 6.0, abs=TOL)
    assert tree.lower_confidence_bound[action_id] != pytest.approx(3.0, abs=TOL)
    # ``q_value`` must track ``ba_l``, so the arena's generic best-action
    # helper and the planner's own final rule cannot disagree.
    assert tree.q_value[action_id] == pytest.approx(8.0, abs=TOL)


def test_action_upper_bound_is_a_scenario_weighted_mean_over_the_parent_count():
    """``ba_U = (sum_phi r + gamma * sum_o |Phi_o| U(b_o)) / |Phi_b|``.

    Purpose: ``U`` is the one conditional quantity, so it is the one place a
    weight and a discount still appear. Normalising by the *children's* total
    count instead of the parent's would be identical whenever no scenario
    terminates and silently wrong when one does -- which is why the fixture
    drops one.

    Given: parent with 4 scenarios, ``sum_phi r = 8``; child A has 3 scenarios
        with ``U = 9``, child B has 1 with ``U = 7``. One scenario of the four
        terminated and has no child at all.
    Expect: ``(8 + 0.5*(3*9 + 1*7)) / 4 = (8 + 17) / 4 = 6.25``. Dividing by
        the children's total (4... here 4 as well) is indistinguishable, so the
        fixture uses 3 + 1 = 4 children out of 5 parent scenarios below.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=3, n_scenarios=5)

    tree = Tree()
    root_id = _add_belief(tree, 0, 0.0, 0.0, 0.0, 5, 5)
    action_id = _add_action(tree, root_id, rho=0.0, reward_sum=8.0, label="a")
    _add_belief(tree, 1, 0.0, 0.0, 9.0, 3, 5, parent_id=action_id)
    _add_belief(tree, 1, 0.0, 0.0, 7.0, 1, 5, parent_id=action_id)

    planner._update_action_bounds(tree=tree, action_id=action_id, parent_count=5)

    # Normalising by the surviving children (4) would give 6.25; the parent's
    # own count (5) is the correct denominator and gives 5.0.
    assert tree.upper_confidence_bound[action_id] == pytest.approx(
        (8.0 + DISCOUNT * (3 * 9.0 + 1 * 7.0)) / 5, abs=TOL
    )
    assert tree.upper_confidence_bound[action_id] == pytest.approx(5.0, abs=TOL)
    assert tree.upper_confidence_bound[action_id] != pytest.approx(6.25, abs=TOL)


# ---------------------------------------------------------------------------
# 6-7. Backup
# ---------------------------------------------------------------------------


def test_backup_takes_the_maximum_over_actions_and_lets_the_lower_bound_fall():
    """``l(b) = max(l_0, max_a ba_l)``, not ``max(l(b), max_a ba_l)``.

    Purpose: this is the difference that makes blocking work at all. DESPOT's
    lower bound only ever rises, because in DESPOT nothing ever retracts a
    value. In AR-DESPOT blocking collapses a descendant back onto its default
    policy, and a running maximum would leave the retracted, higher number
    standing in every ancestor -- the tree would keep claiming a guarantee it
    had just withdrawn.

    Given: a root whose stored ``l`` is 20 (as if an earlier, better subtree
        had been backed up) and whose ``l_0`` is 1. Its single action has
        ``rho = 0`` and one child with ``l = 3``.
    Expect: ``l(root)`` becomes ``max(1, 3) = 3`` -- it *falls* from 20.
        DESPOT's rule would leave 20.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=3)

    tree = Tree()
    root_id = _add_belief(tree, 0, 20.0, 20.0, 50.0, 4, 4, default_value=1.0)
    planner._root_id = root_id
    action_id = _add_action(tree, root_id, rho=0.0, reward_sum=0.0, label="a")
    child_id = _add_belief(tree, 1, 3.0, 4.0, 9.0, 4, 4, parent_id=action_id)

    planner._backup(tree=tree, belief_id=child_id)

    assert tree.lower_confidence_bound[root_id] == pytest.approx(3.0, abs=TOL)
    assert tree.v_value[root_id] == pytest.approx(4.0, abs=TOL)


def test_backup_floors_both_l_and_mu_on_the_default_policy_value():
    """``max{ l_0, ... }``: planning can never be worth less than not planning.

    Purpose: without the floor, a heavily penalised subtree (large ``lambda``,
    many nodes) would drive the node's value below what simply running the
    default policy achieves, and the final action would be chosen against a
    number no policy corresponds to.

    Given: a root with ``l_0 = mu_0 = 10`` and a single action whose ``rho`` is
        ``-100``, so ``ba_l = ba_mu = -100``.
    Expect: both ``l(root)`` and ``mu(root)`` stay at 10.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=3)

    tree = Tree()
    root_id = _add_belief(tree, 0, 10.0, 10.0, 50.0, 4, 4, default_value=10.0)
    planner._root_id = root_id
    action_id = _add_action(tree, root_id, rho=-100.0, reward_sum=0.0, label="a")
    child_id = _add_belief(tree, 1, 0.0, 0.0, 0.0, 4, 4, parent_id=action_id)

    planner._backup(tree=tree, belief_id=child_id)

    assert tree.lower_confidence_bound[root_id] == pytest.approx(10.0, abs=TOL)
    assert tree.v_value[root_id] == pytest.approx(10.0, abs=TOL)
    assert tree.lower_confidence_bound[action_id] == pytest.approx(-100.0, abs=TOL)


def test_backup_walks_all_the_way_to_the_root():
    """A trial's result must reach the root, or the stopping test never moves.

    Given: a three-level chain root -> a -> b1 -> a -> b2, ``rho = 1`` on both
        actions, ``l(b2) = 4``, every ``l_0`` zero.
    Expect: ``l(b1) = 1 + 4 = 5`` and ``l(root) = 1 + 5 = 6``.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=3)

    tree = Tree()
    root_id = _add_belief(tree, 0, 0.0, 0.0, 0.0, 4, 4, default_value=0.0)
    planner._root_id = root_id
    action_0 = _add_action(tree, root_id, rho=1.0, reward_sum=0.0, label="a")
    mid_id = _add_belief(tree, 1, 0.0, 0.0, 0.0, 4, 4, parent_id=action_0, default_value=0.0)
    action_1 = _add_action(tree, mid_id, rho=1.0, reward_sum=0.0, label="a")
    leaf_id = _add_belief(tree, 2, 4.0, 4.0, 4.0, 4, 4, parent_id=action_1, default_value=4.0)

    planner._backup(tree=tree, belief_id=leaf_id)

    assert tree.lower_confidence_bound[mid_id] == pytest.approx(5.0, abs=TOL)
    assert tree.lower_confidence_bound[root_id] == pytest.approx(6.0, abs=TOL)
    assert tree.data[mid_id].in_tree is True
    assert tree.data[root_id].in_tree is True


# ---------------------------------------------------------------------------
# 4 (cont). Excess uncertainty
# ---------------------------------------------------------------------------


def test_excess_uncertainty_uses_mu_minus_l_and_has_no_inverse_discount():
    """``E(b) = (mu - l) - (|Phi_b|/K) * xi * (mu(root) - l(root))``.

    Purpose: DESPOT's version is ``(u - l) - eta * root_gap * gamma^(-d)``,
    which on this fixture is *positive* while AR-DESPOT's is *negative*. The
    two rules therefore send the trial to different places, and a test that
    only checked a sign on a node where both agree would not notice.

    Given: root gap ``mu - l = 10``, ``xi = 0.5``. A child at depth 2 holding 2
        of 4 scenarios with ``l = 1``, ``mu = 3``, ``U = 100``.
    Expect: AR-DESPOT's ``E = (3 - 1) - 0.5 * 0.5 * 10 = -0.5``, so the branch
        is resolved and the trial stops. DESPOT's would be
        ``(100 - 1) - 0.5 * 10 * 0.5^-2 = 79`` -- wildly positive.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=4, xi=0.5)

    tree = Tree()
    root_id = _add_belief(tree, 0, 0.0, 10.0, 100.0, 4, 4)
    planner._root_id = root_id
    action_id = _add_action(tree, root_id, rho=0.0, reward_sum=0.0, label="a")
    child_id = _add_belief(tree, 2, 1.0, 3.0, 100.0, 2, 4, parent_id=action_id)

    assert planner._excess_uncertainty(tree=tree, belief_id=child_id) == pytest.approx(
        -0.5, abs=TOL
    )
    despot_style = (100.0 - 1.0) - 0.5 * 10.0 * DISCOUNT ** (-2)
    assert despot_style > 0.0


def test_excess_uncertainty_is_weighted_by_the_nodes_share_of_the_scenarios():
    """A branch carrying few scenarios is allowed a looser bound.

    Given: two children of one action with identical ``mu - l = 2``, one
        holding 3 of 4 scenarios and one holding 1 of 4; root gap 4, xi 0.5.
    Expect: ``E = 2 - 0.75*0.5*4 = 0.5`` for the heavy one and
        ``2 - 0.25*0.5*4 = 1.5`` for the light one, so the light branch is the
        one a trial follows even though the widths are equal.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=4, xi=0.5)

    tree = Tree()
    root_id = _add_belief(tree, 0, 0.0, 4.0, 100.0, 4, 4)
    planner._root_id = root_id
    action_id = _add_action(tree, root_id, rho=0.0, reward_sum=0.0, label="a")
    heavy_id = _add_belief(tree, 1, 1.0, 3.0, 50.0, 3, 4, parent_id=action_id)
    light_id = _add_belief(tree, 1, 1.0, 3.0, 50.0, 1, 4, parent_id=action_id)

    assert planner._excess_uncertainty(tree=tree, belief_id=heavy_id) == pytest.approx(0.5, abs=TOL)
    assert planner._excess_uncertainty(tree=tree, belief_id=light_id) == pytest.approx(1.5, abs=TOL)
    assert planner._next_best(tree=tree, belief_id=root_id) == light_id


# ---------------------------------------------------------------------------
# 3 (cont). Descent
# ---------------------------------------------------------------------------


def test_descent_follows_the_regularized_action_not_the_optimistic_one():
    """``next_best`` picks ``argmax_a ba_mu``, not ``argmax_a ba_U`` or ``ba_l``.

    Purpose: the fixture makes the three maximisers three *different* actions,
    so DESPOT's rule (``Q_u``) and the final-decision rule (``ba_l``) both fail
    it. That is the point of carrying ``mu`` at all: an action whose subtree
    cannot pay for its own size is not descended into in the first place.

    Given: three actions at the root. Action 0 has the largest ``ba_U``,
        action 1 the largest ``ba_mu``, action 2 the largest ``ba_l``.
    Expect: the trial descends through action 1.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=4, xi=0.5)

    tree = Tree()
    root_id = _add_belief(tree, 0, 0.0, 10.0, 100.0, 4, 4)
    planner._root_id = root_id

    # (rho, child l, child mu, child U) chosen so the three maxima differ.
    specs = [
        (0.0, 0.0, 1.0, 99.0),  # ba_U largest
        (0.0, 0.0, 5.0, 10.0),  # ba_mu largest
        (0.0, 4.0, 2.0, 10.0),  # ba_l largest
    ]
    children: List[int] = []
    action_ids: List[int] = []
    for index, (rho, lower, regularized, upper) in enumerate(specs):
        action_id = _add_action(tree, root_id, rho=rho, reward_sum=0.0, label=f"a{index}")
        child_id = _add_belief(tree, 1, lower, regularized, upper, 4, 4, parent_id=action_id)
        planner._update_action_bounds(tree=tree, action_id=action_id, parent_count=4)
        action_ids.append(action_id)
        children.append(child_id)

    upper_max = max(range(3), key=lambda i: tree.upper_confidence_bound[action_ids[i]])
    mu_max = max(range(3), key=lambda i: tree.data[action_ids[i]].mu)
    lower_max = max(range(3), key=lambda i: tree.lower_confidence_bound[action_ids[i]])
    assert {upper_max, mu_max, lower_max} == {0, 1, 2}

    assert planner._next_best(tree=tree, belief_id=root_id) == children[mu_max]


# ---------------------------------------------------------------------------
# 5 (cont). Blocking -- equation (12)
# ---------------------------------------------------------------------------


def _blocking_chain(
    planner: ARDESPOT, uppers: List[float], defaults: List[float]
) -> Tuple[Tree, int, List[int]]:
    """A straight chain root -> b1 -> b2 -> ... with hand-set ``U`` and ``l_0``.

    Returns the tree, the root id, and the belief ids in depth order.
    """
    tree = Tree()
    root_id = _add_belief(tree, 0, 0.0, 10.0, 100.0, 4, 4, default_value=0.0)
    planner._root_id = root_id
    ids = [root_id]
    parent = root_id
    for depth, (upper, default_value) in enumerate(zip(uppers, defaults), start=1):
        action_id = _add_action(tree, parent, rho=0.0, reward_sum=0.0, label="a")
        parent = _add_belief(
            tree,
            depth,
            default_value,
            default_value,
            upper,
            4,
            4,
            parent_id=action_id,
            default_value=default_value,
        )
        ids.append(parent)
    return tree, root_id, ids


def test_find_blocker_is_equation_12_and_measures_distance_in_belief_levels():
    """``w * gamma^Delta * U(bp) - l_0(bp) <= lambda * len`` blocks.

    Purpose: the ``len`` factor is the whole idea -- the deeper a node sits
    below an ancestor, the more policy nodes it costs and the more the ancestor
    must be able to gain to justify it. A version that dropped ``len`` and
    compared against ``lambda`` alone gives a different answer here.

    Given: ``gamma = 0.5``, ``lambda = 1.0``, all nodes holding all 4 of 4
        scenarios. Node at depth 1 has ``U = 6``, ``l_0 = 1``, so its gain is
        ``1 * 0.5 * 6 - 1 = 2``. Node at depth 2 has ``U = 40``, ``l_0 = 0``,
        gain ``1 * 0.25 * 40 = 10``.
    Expect: asked about the node at depth 3, the depth-2 ancestor is at
        ``len = 1`` and its gain 10 > 1, so it does not block; the depth-1
        ancestor is at ``len = 2`` and its gain 2 > 2 is false, so
        ``2 <= 1 * 2`` holds and it *does* block. The returned blocker is the
        depth-1 node. Without the ``len`` factor the depth-1 test would be
        ``2 <= 1``, false, and nothing would block.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=6, pruning_constant=1.0)
    tree, _, ids = _blocking_chain(planner, uppers=[6.0, 40.0, 8.0], defaults=[1.0, 0.0, 0.0])

    assert planner._find_blocker(tree=tree, belief_id=ids[3]) == ids[1]
    # Asked about the depth-2 node, only the depth-1 ancestor is in range, at
    # len = 1: 2 <= 1 is false, so nothing blocks.
    assert planner._find_blocker(tree=tree, belief_id=ids[2]) is None


def test_the_root_is_never_a_blocker():
    """Blocking the root would mean refusing to plan, and an action still has to come out.

    Given: a root whose gain is 0 -- ``U = 0``, ``l_0 = 0`` -- which would
        block under any ``lambda >= 0`` if the root were eligible.
    Expect: ``_find_blocker`` on its depth-1 child returns ``None``.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=6, pruning_constant=1.0)

    tree = Tree()
    root_id = _add_belief(tree, 0, 0.0, 0.0, 0.0, 4, 4, default_value=0.0)
    planner._root_id = root_id
    action_id = _add_action(tree, root_id, rho=0.0, reward_sum=0.0, label="a")
    child_id = _add_belief(tree, 1, 0.0, 0.0, 0.0, 4, 4, parent_id=action_id)

    assert planner._find_blocker(tree=tree, belief_id=child_id) is None


def test_lambda_zero_blocks_exactly_when_there_is_nothing_left_to_gain():
    """At ``lambda = 0`` the test becomes ``gain <= 0``, which is still meaningful.

    Purpose: it would be easy to assume ``lambda = 0`` disables blocking. It
    does not -- it reduces it to "an ancestor whose upper bound already equals
    its default value has nothing to offer below it", which is correct pruning
    and not regularization. Recording that keeps a ``lambda = 0`` run from
    being mistaken for one with no pruning at all.

    Given: ``lambda = 0``; an ancestor at depth 1 with ``U = 4``, ``l_0 = 2``,
        so gain ``1*0.5*4 - 2 = 0``.
    Expect: it blocks. Raise its ``U`` to 6 (gain 1) and it does not.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=6, pruning_constant=0.0)

    tree, _, ids = _blocking_chain(planner, uppers=[4.0, 8.0], defaults=[2.0, 0.0])
    assert planner._find_blocker(tree=tree, belief_id=ids[2]) == ids[1]

    tree.upper_confidence_bound[ids[1]] = 6.0
    assert planner._find_blocker(tree=tree, belief_id=ids[2]) is None


def test_prune_collapses_the_blocked_node_onto_its_default_policy_and_backs_it_up():
    """``make_default``: ``mu = l = l_0``, and every ancestor learns about it.

    Purpose: collapsing without the backup would leave the ancestors quoting a
    value the collapsed node no longer offers -- exactly the stale optimism the
    non-monotone backup exists to prevent.

    Given: a chain whose depth-2 node has ``l = mu = 9`` and ``l_0 = 1``, with
        a depth-1 ancestor that blocks it. The root's action has ``rho = 0``,
        the depth-1 node's ``l_0`` is 1.
    Expect: after pruning, ``l`` and ``mu`` at the depth-2 node are both 1, the
        depth-1 node's ``l`` has fallen from 9 to ``max(1, 0 + 1) = 1``, and
        ``ardespot_n_blocked`` has counted it.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=6, pruning_constant=1.0)

    tree = Tree()
    root_id = _add_belief(tree, 0, 0.0, 10.0, 100.0, 4, 4, default_value=0.0)
    planner._root_id = root_id
    action_0 = _add_action(tree, root_id, rho=0.0, reward_sum=0.0, label="a")
    mid_id = _add_belief(tree, 1, 9.0, 9.0, 6.0, 4, 4, parent_id=action_0, default_value=1.0)
    action_1 = _add_action(tree, mid_id, rho=0.0, reward_sum=0.0, label="a")
    leaf_id = _add_belief(tree, 2, 9.0, 9.0, 40.0, 4, 4, parent_id=action_1, default_value=1.0)

    # gain at the depth-1 ancestor: 1 * 0.5 * 6 - 1 = 2 <= lambda * 1 = 1? No.
    # Lower its U so it does block: 1 * 0.5 * 4 - 1 = 1 <= 1. Yes.
    tree.upper_confidence_bound[mid_id] = 4.0

    assert planner._prune(tree=tree, belief_id=leaf_id) is True
    assert tree.lower_confidence_bound[leaf_id] == pytest.approx(1.0, abs=TOL)
    assert tree.v_value[leaf_id] == pytest.approx(1.0, abs=TOL)
    assert tree.data[leaf_id].defaulted_by == "blocked"
    assert tree.lower_confidence_bound[mid_id] == pytest.approx(1.0, abs=TOL)
    assert planner._n_blocked >= 1


def test_make_default_leaves_the_upper_bound_alone():
    """``U`` is untouched, so a collapse cannot change who blocks whom.

    Purpose: ``U`` is the left-hand side of equation (12). If collapsing a node
    also lowered its ``U``, the node's own collapse would retroactively make
    its descendants' blocker tests come out differently, and pruning would
    depend on the order trials happened to visit nodes.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=4, pruning_constant=1.0)

    tree = Tree()
    node_id = _add_belief(tree, 1, 9.0, 9.0, 40.0, 4, 4, default_value=2.0)
    planner._make_default(tree=tree, belief_id=node_id, reason="blocked")

    assert tree.lower_confidence_bound[node_id] == pytest.approx(2.0, abs=TOL)
    assert tree.v_value[node_id] == pytest.approx(2.0, abs=TOL)
    assert tree.upper_confidence_bound[node_id] == pytest.approx(40.0, abs=TOL)


# ---------------------------------------------------------------------------
# Terminated scenarios
# ---------------------------------------------------------------------------


def test_terminated_scenarios_are_dropped_rather_than_given_their_own_branch():
    """AR-DESPOT drops them; DESPOT gives them a ``TERMINAL_OBSERVATION`` child.

    Purpose: the two are both correct, each in its own scale, and confusing
    them is a real hazard -- an AR-DESPOT that kept DESPOT's terminal branch
    would double count nothing but would inflate the child count and break the
    ``ba_U`` denominator argument.

    Given: a node holding 4 scenarios, two of them already in the terminal
        state ``end``.
    Expect: each action gets exactly one observation child, holding the two
        live scenarios, and ``sum_o |Phi_o| = 2 < 4``. Plain DESPOT on the same
        states produces two children, one of them the terminal branch.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=3)
    planner._streams = None
    planner.use_determinized_scenarios = False

    tree = Tree()
    node_id = planner._add_belief_node(
        tree=tree,
        states=[NEXT, NEXT, END, END],
        scenario_ids=[0, 1, 2, 3],
        depth=0,
        parent_id=None,
        observation=None,
        obs_key=None,
    )
    planner._expand(tree=tree, belief_id=node_id)

    for action_id in tree.get_children_ids(node_id):
        children = tree.get_children_ids(action_id)
        assert len(children) == 1
        assert len(tree.data[children[0]].scenario_ids) == 2

    despot = DESPOT(
        environment=ChainEnv(discount_factor=DISCOUNT),
        discount_factor=DISCOUNT,
        depth=3,
        name="DESPOT_comparison",
        n_scenarios=4,
        n_simulations=1,
    )
    despot._streams = None
    despot.use_determinized_scenarios = False
    despot_tree = Tree()
    despot_root = despot._add_belief_node(
        tree=despot_tree,
        states=[NEXT, NEXT, END, END],
        scenario_ids=[0, 1, 2, 3],
        depth=0,
        parent_id=None,
        observation=None,
        obs_key=None,
    )
    despot._expand(tree=despot_tree, belief_id=despot_root)
    assert all(
        len(despot_tree.get_children_ids(action_id)) == 2
        for action_id in despot_tree.get_children_ids(despot_root)
    )


def test_dropping_terminated_scenarios_keeps_the_action_lower_bound_exact():
    """The dropped weight contributes zero, so no renormalization is needed.

    Given: ``ChainEnv``, ``K = 4``, ``depth = 2``. A root at depth 0 holding
        two scenarios in ``next`` and two in ``end`` (terminal). The live pair
        earns ``r(next) = 4`` each, so ``sum_phi r = 8`` and
        ``rho = 0.5^0 * 8/4 = 2``. Their successor ``end`` is terminal, so the
        child's ``l_0`` is 0.
    Expect: ``ba_l = 2 + 0 = 2``, which is exactly the true two-step value of
        the node in the unnormalized scale: two of four scenarios earn 4 once,
        i.e. ``(2/4) * 4 = 2``.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=2)
    planner._streams = None
    planner.use_determinized_scenarios = False

    tree = Tree()
    node_id = planner._add_belief_node(
        tree=tree,
        states=[NEXT, NEXT, END, END],
        scenario_ids=[0, 1, 2, 3],
        depth=0,
        parent_id=None,
        observation=None,
        obs_key=None,
    )
    planner._expand(tree=tree, belief_id=node_id)

    for action_id in tree.get_children_ids(node_id):
        assert tree.data[action_id].rho == pytest.approx(2.0, abs=TOL)
        assert tree.lower_confidence_bound[action_id] == pytest.approx(2.0, abs=TOL)


# ---------------------------------------------------------------------------
# 8. Final action
# ---------------------------------------------------------------------------


def test_final_action_is_argmax_ba_l_not_ba_mu_and_not_the_optimistic_one():
    """The answer is the regularized *lower* bound, already carrying ``-lambda``.

    Purpose: the search descends on ``ba_mu`` and the tree also holds
    ``ba_U``; both are plausible things to answer with, and both are wrong.
    The fixture makes all three maximisers different actions.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=4)

    tree = Tree()
    root_id = _add_belief(tree, 0, 0.0, 10.0, 100.0, 4, 4)
    planner._root_id = root_id
    specs = [
        (0.0, 0.0, 1.0, 99.0),  # ba_U largest
        (0.0, 0.0, 5.0, 10.0),  # ba_mu largest
        (0.0, 4.0, 2.0, 10.0),  # ba_l largest
    ]
    action_ids: List[int] = []
    for index, (rho, lower, regularized, upper) in enumerate(specs):
        action_id = _add_action(tree, root_id, rho=rho, reward_sum=0.0, label=f"a{index}")
        _add_belief(tree, 1, lower, regularized, upper, 4, 4, parent_id=action_id)
        planner._update_action_bounds(tree=tree, action_id=action_id, parent_count=4)
        action_ids.append(action_id)

    assert planner._select_final_action(tree=tree, root_id=root_id) == "a2"


def test_final_action_breaks_a_tie_at_random_rather_than_by_action_order():
    """A tie is not a decision, and returning index 0 makes it look like one.

    Purpose: job 04 found DESPOT returning the same action at every decision on
    RockSample because a too-short horizon left every lower bound at zero and
    the ``max`` tie-break always chose the first action. A random tie-break
    turns that into visible noise instead of a plausible-looking policy.

    Given: three actions with identical ``ba_l = 0``.
    Expect: over 200 selections, more than one distinct action is returned.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=4)

    tree = Tree()
    root_id = _add_belief(tree, 0, 0.0, 10.0, 100.0, 4, 4)
    planner._root_id = root_id
    for index in range(3):
        action_id = _add_action(tree, root_id, rho=0.0, reward_sum=0.0, label=f"a{index}")
        _add_belief(tree, 1, 0.0, 0.0, 0.0, 4, 4, parent_id=action_id)
        planner._update_action_bounds(tree=tree, action_id=action_id, parent_count=4)

    seen = {planner._select_final_action(tree=tree, root_id=root_id) for _ in range(200)}
    assert len(seen) > 1


# ---------------------------------------------------------------------------
# 7. Anytime: three stopping conditions
# ---------------------------------------------------------------------------


def test_the_search_stops_on_epsilon_0_and_says_so():
    """``mu(root) - l(root) <= epsilon_0`` ends the loop and sets ``gap_closed``.

    Given: ``ChainEnv`` at ``depth = 2``, where one trial resolves the root
        exactly (the chain is deterministic and one step from terminal).
    Expect: the loop stops with ``gap_closed = 1``, the other two stop flags 0,
        and strictly fewer trials than the budget allows.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=2, n_simulations=100)
    _, run_data = planner.action(chain_belief(ROOT))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}

    assert metrics["ardespot_gap_closed"] == 1
    assert metrics["ardespot_time_exhausted"] == 0
    assert metrics["ardespot_trials_exhausted"] == 0
    assert metrics["ardespot_n_trials"] < 100


def test_the_search_stops_on_the_trial_budget_and_says_so():
    """A budget-limited stop is reported as such, not as a closed gap.

    Given: a ``depth = 6`` chain and ``xi = 0.01``, so the target precision is
        far tighter than one trial can reach, with a budget of one trial.
    Expect: ``trials_exhausted = 1`` and exactly one trial.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=6, n_simulations=1, xi=0.01)
    _, run_data = planner.action(chain_belief(ROOT))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}

    assert metrics["ardespot_n_trials"] == 1
    assert metrics["ardespot_trials_exhausted"] == 1
    assert metrics["ardespot_gap_closed"] == 0


def test_both_budgets_may_be_given_at_once_which_despot_refuses():
    """The wall clock is a stopping condition, not an alternative to counting.

    Purpose: this is the constructor-level shape of "anytime". Plain DESPOT
    raises when both budgets are supplied, because for it they are two ways of
    saying the same thing; for AR-DESPOT they are two different guarantees and
    both are enforced.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = ARDESPOT(
        environment=environment,
        discount_factor=DISCOUNT,
        depth=4,
        name="ARDESPOT_both_budgets",
        n_scenarios=4,
        n_simulations=5,
        time_out_in_seconds=10,
    )
    assert planner.n_simulations == 5
    assert planner.time_out_in_seconds == 10

    with pytest.raises(ValueError):
        DESPOT(
            environment=ChainEnv(discount_factor=DISCOUNT),
            discount_factor=DISCOUNT,
            depth=4,
            name="DESPOT_both_budgets",
            n_scenarios=4,
            n_simulations=5,
            time_out_in_seconds=10,
        )


def test_every_budget_from_one_trial_upward_returns_a_legal_action():
    """Anytime: the tree holds a complete answer after every trial.

    Purpose: "returns a valid action if interrupted" is only meaningful if it
    holds at every interruption point, not just at the end. Running the planner
    at each budget from 1 to 8 is the operational form of that claim.

    Given: ``ChainEnv`` at ``depth = 6``, budgets 1..8.
    Expect: every budget returns an action in the environment's action set, and
        every run leaves a root with action children to read the answer off.
    """
    for budget in range(1, 9):
        environment = ChainEnv(discount_factor=DISCOUNT)
        planner = _planner(environment, depth=6, n_simulations=budget, xi=0.01)
        actions, _ = planner.action(chain_belief(ROOT))
        assert actions[0] in environment.get_actions()
        assert planner._last_tree is not None
        assert planner._last_root_id is not None
        assert planner._last_tree.get_children_ids(planner._last_root_id)


def test_more_budget_never_loosens_the_root_regularized_gap():
    """A longer search is at least as resolved as a shorter one.

    Purpose: the anytime contract is "improving with budget", and the gap
    ``mu(root) - l(root)`` is the quantity the algorithm's own stopping rule
    treats as progress. It is checked as non-increasing rather than strictly
    decreasing because a trial that only confirms an existing bound is allowed.

    Given: ``ChainEnv`` at ``depth = 6``, budgets 1, 2, 4, 8, 16.
    Expect: the reported ``ardespot_root_gap`` never rises as the budget grows.
    """
    gaps: List[float] = []
    for budget in (1, 2, 4, 8, 16):
        environment = ChainEnv(discount_factor=DISCOUNT)
        planner = _planner(environment, depth=6, n_simulations=budget, xi=0.01)
        _, run_data = planner.action(chain_belief(ROOT))
        metrics = {variable.name: variable.value for variable in run_data.info_variables}
        gaps.append(metrics["ardespot_root_gap"])

    for earlier, later in zip(gaps, gaps[1:]):
        assert later <= earlier + TOL


# ---------------------------------------------------------------------------
# The horizon
# ---------------------------------------------------------------------------


def test_the_tree_never_goes_deeper_than_the_search_horizon():
    """``depth`` means ``depth`` belief steps, unlike the reference's ``D + 1``.

    Purpose: the reference expands at ``Delta <= D`` and collects one more step
    of reward than DESPOT does at the same ``D``. Keeping that off-by-one would
    make every DESPOT-versus-AR-DESPOT comparison at one ``depth`` a comparison
    of two horizons.

    Given: ``ChainEnv`` with no terminal state, so nothing stops the descent
        except the horizon; ``depth = 3``, a generous trial budget.
    Expect: no belief node in the tree sits deeper than 3, and the node at
        depth 3 has no action children.
    """
    environment = ChainEnv(discount_factor=DISCOUNT, terminal_states=())
    planner = _planner(environment, depth=3, n_simulations=50, xi=0.01)
    planner.action(chain_belief(ROOT))

    tree = planner._last_tree
    assert tree is not None
    depths = [tree.data[node_id].depth for node_id in _belief_nodes(tree)]
    assert max(depths) <= 3
    for node_id in _belief_nodes(tree):
        if tree.data[node_id].depth == 3:
            assert not tree.get_children_ids(node_id)


def test_a_node_past_the_horizon_is_collapsed_onto_its_default_policy_and_counted():
    """The horizon default and the blocking default are counted separately.

    Purpose: both call ``make_default``, and a single counter would make a run
    where the regularizer never fired look identical to one where it fired
    constantly.

    Given: ``ChainEnv`` with no terminal state, ``depth = 3``, ``lambda = 0``
        with a generous budget so blocking is confined to "nothing to gain".
    Expect: ``ardespot_n_depth_defaults`` is positive, and every belief node
        marked ``defaulted_by == "depth"`` sits at depth 3.
    """
    environment = ChainEnv(discount_factor=DISCOUNT, terminal_states=())
    planner = _planner(environment, depth=3, n_simulations=50, xi=0.01)
    _, run_data = planner.action(chain_belief(ROOT))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}

    assert metrics["ardespot_n_depth_defaults"] > 0
    tree = planner._last_tree
    assert tree is not None
    for node_id in _belief_nodes(tree):
        if tree.data[node_id].defaulted_by == "depth":
            assert tree.data[node_id].depth == 3


# ---------------------------------------------------------------------------
# End to end on a hand-computable chain
# ---------------------------------------------------------------------------


def test_chain_root_values_match_the_hand_computation_at_lambda_zero():
    """Every number at the root of a two-step chain, worked out by hand.

    Given: ``ChainEnv``, ``gamma = 0.5``, ``K = 4``, ``depth = 2``,
        ``lambda = 0``. All four scenarios start in ``root``. Each action
        drives every scenario to ``next``, earning ``r(root) = 2``.
    Expect:
        ``rho   = 0.5^0 * (4*2) / 4 - 0 = 2``;
        the single child at depth 1 has ``L_0 = r(next) = 4`` over its one
        remaining step, so ``l_0 = (4/4) * 0.5 * 4 = 2``;
        ``ba_l = 2 + 2 = 4``; ``l(root) = max(l_0(root), 4)`` and
        ``l_0(root) = 1 * 1 * (2 + 0.5*4) = 4``, so ``l(root) = 4``, which is
        the true two-step return of the chain. ``mu(root)`` equals it, so the
        gap closes and the search stops after one trial.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=2, n_simulations=20, pruning_constant=0.0)
    _, run_data = planner.action(chain_belief(ROOT))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}

    assert metrics["ardespot_root_lower_bound"] == pytest.approx(4.0, abs=TOL)
    assert metrics["ardespot_root_regularized_value"] == pytest.approx(4.0, abs=TOL)
    assert metrics["ardespot_root_gap"] == pytest.approx(0.0, abs=TOL)
    assert metrics["ardespot_n_trials"] == 1

    tree = planner._last_tree
    root_id = planner._last_root_id
    assert tree is not None and root_id is not None
    for action_id in tree.get_children_ids(root_id):
        assert tree.data[action_id].rho == pytest.approx(2.0, abs=TOL)
        assert tree.lower_confidence_bound[action_id] == pytest.approx(4.0, abs=TOL)


def test_lambda_lowers_the_action_lower_bound_by_exactly_one_per_policy_node():
    """``ba_l`` falls by ``lambda`` per action node on the policy, not by more.

    Purpose: the size penalty is the whole content of "regularized". Charging
    it per *belief* node, or per scenario, or once for the whole tree, all
    produce a lower ``ba_l`` and would pass a test that only checked "it went
    down".

    Given: the same two-step chain, at ``lambda = 0`` and ``lambda = 0.5``. The
        policy under the root action is one action node deep (its child is a
        leaf that runs the default policy), so exactly one ``lambda`` is
        charged.
    Expect: ``ba_l`` drops from 4.0 to 3.5 -- exactly ``lambda``.
    """
    lowers: List[float] = []
    for pruning_constant in (0.0, 0.5):
        environment = ChainEnv(discount_factor=DISCOUNT)
        planner = _planner(
            environment, depth=2, n_simulations=20, pruning_constant=pruning_constant
        )
        planner.action(chain_belief(ROOT))
        tree = planner._last_tree
        root_id = planner._last_root_id
        assert tree is not None and root_id is not None
        lowers.append(
            max(
                tree.lower_confidence_bound[action_id]
                for action_id in tree.get_children_ids(root_id)
            )
        )

    assert lowers[0] == pytest.approx(4.0, abs=TOL)
    assert lowers[1] == pytest.approx(3.5, abs=TOL)


def test_l_never_exceeds_mu_anywhere_in_the_tree():
    """``l(b) <= mu(b)`` is the invariant blocking and backup must preserve.

    Purpose: ``mu`` bounds the regularized value from above and ``l`` from
    below, so a crossing means one of the two updates is inconsistent. The
    sweep covers every belief node the search touched, not just the root, and
    is run at three ``lambda`` values because the ordering is easiest to break
    where ``max(l_0, ...)`` and ``-lambda`` interact.
    """
    for pruning_constant in (0.0, 0.5, 5.0):
        environment = ChainEnv(discount_factor=DISCOUNT, terminal_states=())
        planner = _planner(
            environment,
            depth=4,
            n_simulations=40,
            xi=0.05,
            pruning_constant=pruning_constant,
        )
        planner.action(chain_belief(ROOT))
        tree = planner._last_tree
        assert tree is not None
        for node_id in _belief_nodes(tree):
            assert (
                tree.lower_confidence_bound[node_id] <= tree.v_value[node_id] + TOL
            ), f"l > mu at node {node_id}, lambda={pruning_constant}"


def test_gamma_one_is_not_a_special_case():
    """An undiscounted problem must not divide by ``1 - gamma`` anywhere.

    Given: ``ChainEnv`` at ``gamma = 1``, ``depth = 2``, ``K = 4``.
    Expect: ``l_0(root) = r(root) + r(next) = 6``, and the search returns a
        legal action with a finite root value.
    """
    environment = ChainEnv(discount_factor=1.0)
    planner = _planner(environment, depth=2, n_simulations=10)
    actions, run_data = planner.action(chain_belief(ROOT))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}

    assert actions[0] in environment.get_actions()
    assert metrics["ardespot_root_lower_bound"] == pytest.approx(6.0, abs=TOL)
    assert np.isfinite(metrics["ardespot_root_upper_bound"])


# ---------------------------------------------------------------------------
# Determinization, inherited but exercised through the new expansion
# ---------------------------------------------------------------------------


def test_one_scenario_draws_the_same_number_under_every_root_action():
    """Determinization survives the rewritten ``_expand``.

    Purpose: ``_expand`` was rewritten for the new scale, and the seeding call
    sits inside it. A version that seeded once per action rather than once per
    ``(scenario, depth)`` would still produce a plausible tree.

    Given: ``CoinEnv``-style behaviour is not needed here -- ``ChainEnv`` is
        deterministic -- so the check is made directly on the stream table the
        planner installs: every transition at depth 0 for scenario ``i`` must
        be preceded by ``seed_for(i, 0)``.
    Expect: the uniform drawn immediately after activation is the same for both
        root actions, for every scenario.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=3, n_scenarios=4)
    planner.action(chain_belief(ROOT))

    streams = planner._streams
    assert streams is not None
    for scenario_id in range(4):
        draws = []
        for _ in range(2):  # once per root action
            streams.activate(scenario_id=scenario_id, depth=0)
            draws.append(np.random.random())
        assert draws[0] == draws[1]


def test_planning_leaves_both_global_generators_where_it_found_them():
    """The planner seeds the global streams; it must not keep them.

    Purpose: without the restore, the episode's own randomness becomes a
    function of the planner's last transition, and two planners would face
    different episodes for reasons nothing records.
    """
    environment = ChainEnv(discount_factor=DISCOUNT)
    planner = _planner(environment, depth=3)

    np.random.seed(12345)
    random.seed(12345)
    numpy_before = np.random.get_state()
    python_before = random.getstate()

    planner.action(chain_belief(ROOT))

    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert python_before == random.getstate()


# ---------------------------------------------------------------------------
# Blocking, end to end -- what it does and does not reach here
# ---------------------------------------------------------------------------


class FlatRewardChain(ChainEnv):
    """``ChainEnv`` whose every transition pays exactly ``R_max``.

    Its purpose is to make the trivial upper bound *tight*: with
    ``r(s, a) = 4`` everywhere and ``reward_range = (4, 4)``, a node's
    ``U_0 = R_max * sum_t gamma^t`` equals its default policy's value exactly.
    """

    def __init__(self, discount_factor: float = DISCOUNT) -> None:
        super().__init__(discount_factor=discount_factor, terminal_states=())
        self.reward_range = (CHAIN_MAX_REWARD, CHAIN_MAX_REWARD)

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        del state, action, next_state
        return CHAIN_MAX_REWARD


def test_the_search_loop_actually_consults_the_blocker_test():
    """``_explore`` calls ``_prune`` on real, non-root ancestors.

    Purpose: every other blocking test drives ``_find_blocker`` and ``_prune``
    directly, which leaves the wiring untested -- an ``_explore`` that never
    called ``_prune``, or called it only where the ancestor chain is empty,
    would pass all of them and blocking would be dead code.

    Given: a ``ChainEnv`` with no terminal state, ``depth = 8``, ``K = 4``,
        ``xi = 0.001`` and a large trial budget, so trials descend deep enough
        for the ancestor walk to have something to walk.
    Expect: ``_find_blocker`` is called from nodes at several depths, and the
        equation-(12) gain is evaluated for at least one strict, non-root
        ancestor.
    """
    np.random.seed(2)
    random.seed(2)
    environment = ChainEnv(discount_factor=0.9, terminal_states=())
    planner = _planner(
        environment,
        depth=8,
        n_scenarios=4,
        n_simulations=500,
        xi=0.001,
        pruning_constant=0.2,
    )
    planner.discount_factor = 0.9

    called_at_depth: List[int] = []
    ancestors_checked: List[int] = []
    original = planner._find_blocker

    def spy(tree: Tree, belief_id: int) -> Optional[int]:
        called_at_depth.append(tree.data[belief_id].depth)
        ancestor_id = planner._parent_belief_id(tree=tree, belief_id=belief_id)
        while ancestor_id is not None and ancestor_id != planner._root_id:
            ancestors_checked.append(ancestor_id)
            ancestor_id = planner._parent_belief_id(tree=tree, belief_id=ancestor_id)
        return original(tree=tree, belief_id=belief_id)

    planner._find_blocker = spy  # type: ignore[method-assign]
    planner.action(chain_belief(ROOT))

    assert called_at_depth, "_explore never reached _prune"
    assert len(set(called_at_depth)) > 1, "_prune was only consulted at one depth"
    assert ancestors_checked, "equation (12) was never evaluated on a real ancestor"


@pytest.mark.parametrize("pruning_constant", [0.0, 0.2, 0.5, 1.0, 2.0])
def test_blocking_does_not_fire_under_the_trivial_r_max_upper_bound(pruning_constant):
    """A limit of the bound available here, recorded so it is not mistaken for a bug.

    Purpose: equation (12) compares ``w * gamma^Delta * U(bp)`` against
    ``l_0(bp)``. This repository's only leaf upper bound is ``R_max`` sustained
    for the remaining horizon, which on any environment whose default policy
    earns less than ``R_max`` every step leaves that difference large. To block,
    ``lambda`` would have to approach it -- and by then
    ``mu_0(root) = max(l_0, U_0 - lambda)`` has collapsed to ``l_0``, the root
    gap is zero and the search stops before running a trial. The two effects
    work against each other, so with this bound blocking never fires no matter
    how ``lambda`` is set.

    This is what makes AR-DESPOT here differ from plain DESPOT mainly through
    the ``lambda`` charged in ``rho`` and through the anytime loop, rather than
    through pruning. A tighter bound -- a fully observable value upper bound,
    say -- would change that, and would make this test fail, which is the point
    of writing it down.

    Expect: ``ardespot_n_blocked`` is 0 at every ``lambda`` tried.
    """
    np.random.seed(2)
    random.seed(2)
    environment = ChainEnv(discount_factor=0.9, terminal_states=())
    planner = _planner(
        environment,
        depth=8,
        n_scenarios=4,
        n_simulations=500,
        xi=0.001,
        pruning_constant=pruning_constant,
    )
    planner.discount_factor = 0.9
    _, run_data = planner.action(chain_belief(ROOT))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}

    assert metrics["ardespot_n_blocked"] == 0, (
        f"blocking fired at lambda={pruning_constant}; if the leaf upper bound "
        "was tightened, revisit the note in this test and in the module docstring"
    )


def test_a_perfectly_tight_upper_bound_leaves_nothing_to_search():
    """When ``U_0`` equals the default policy's value, the root gap is already zero.

    Purpose: this is the other end of the same trade-off, and it explains why
    "just tighten the bound until blocking fires" is not a knob. With a bound
    tight enough that equation (12)'s gain is zero everywhere,
    ``mu_0(root) = max(l_0, U_0 - lambda) = l_0``, so the stopping test is
    satisfied before the first trial and there is nothing left to prune.

    Given: ``FlatRewardChain``, where every transition pays exactly ``R_max``.
    Expect: zero trials, ``gap_closed`` set, and a legal action anyway -- the
        anytime contract has to hold at zero trials too.
    """
    np.random.seed(2)
    random.seed(2)
    environment = FlatRewardChain()
    planner = _planner(
        environment, depth=6, n_scenarios=8, n_simulations=40, xi=0.01, pruning_constant=0.1
    )
    actions, run_data = planner.action(chain_belief(ROOT))
    metrics = {variable.name: variable.value for variable in run_data.info_variables}

    assert metrics["ardespot_n_trials"] == 0
    assert metrics["ardespot_gap_closed"] == 1
    assert metrics["ardespot_root_gap"] == pytest.approx(0.0, abs=TOL)
    assert actions[0] in environment.get_actions()
