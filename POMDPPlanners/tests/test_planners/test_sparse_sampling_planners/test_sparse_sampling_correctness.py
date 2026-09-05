# SPDX-License-Identifier: MIT

"""Sparse-sampling correctness: hand-computed CVaR, exact backups, statistics.

Two problems in the existing suites are addressed here.

The iCVaR test computes its expected value by calling ``cvar_estimator_from_dist``
— the same production helper the planner calls — so it asserts only that the
function agrees with itself. Every CVaR figure below is instead worked through
by hand from the estimator's stated definition, and the derivation is written
out in the test so a reader can check it without running anything.

The statistics test in ``test_sparse_sampling.py`` reads node attributes
without first running the update it claims to test, so it passes on whatever
the tree happens to hold. The tests here run the update and then assert the
number it produced.

References:
    Kearns, M., Mansour, Y., & Ng, A. Y. (2002). A Sparse Sampling Algorithm for
    Near-Optimal Planning in Large MDPs. Machine Learning 49(2), 193-208.
"""

# pylint: disable=protected-access

import random

import numpy as np
import pytest
from anytree import PostOrderIter

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.sparse_sampling_planners.icvar_sparse_sampling import (
    ICVaRSparseSampling,
)
from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import (
    SparseSamplingDiscreteActionsPlanner,
)


np.random.seed(42)
random.seed(42)

DISCOUNT = 0.5
TOL = 1e-12

# --- Hand-computed CVaR ------------------------------------------------------
#
# The estimator returns the conditional expectation over the worst-alpha
# probability mass, treating values as costs, so the tail is the *upper* one.
# For child values [1, 1, 1, 9] with uniform weights 0.25 and alpha = 0.25:
#
#   duplicates aggregate to values [1, 9] with weights [0.75, 0.25];
#   the cumulative distribution is [0.75, 1.0];
#   VaR is the value where the cumulative mass first reaches 1 - alpha = 0.75,
#     which is the value 1 at index 0;
#   the tail from that index has weight 1.0 and weighted sum
#     1*0.75 + 9*0.25 = 3.0;
#   the correction for the mass above alpha is (1.0 - 0.25) * 1 = 0.75;
#   CVaR = (3.0 - 0.75) / 0.25 = 9.0.
#
# The mean of the same sample is 3.0, so a fixture that accidentally used the
# mean gives a visibly different answer.
TAIL_VALUES = [1.0, 1.0, 1.0, 9.0]
HAND_COMPUTED_CVAR = 9.0
MEAN_OF_TAIL_VALUES = 3.0


def _belief(env):
    return get_initial_belief(pomdp=env, n_particles=6)


def _leaf_action_with_children(env, values, immediate_cost):
    """An action node with one belief child per value, each carrying that v_value."""
    root = BeliefNode(belief=_belief(env))
    action = ActionNode(action="listen", parent=root, children=tuple(), data=None)
    action.immediate_cost = immediate_cost
    for value in values:
        child = BeliefNode(belief=_belief(env), parent=action, children=tuple(), data=None)
        child.v_value = value
    return root, action


# ---------------------------------------------------------------------------
# iCVaR sparse sampling: hand-computed tail
# ---------------------------------------------------------------------------


def test_icvar_action_value_uses_the_hand_computed_upper_tail():
    """Q = immediate cost + gamma * CVaR, with CVaR worked out by hand.

    Purpose: Replaces an expectation produced by the production helper itself.
        The numbers separate the CVaR from the mean (which would give
        ``2 + 0.5*3 = 3.5``), from the wrong tail direction (the lower tail of
        this sample is 1, giving 2.5), and from a missing discount (11).

    Given: An action node with immediate cost 2 and four belief children whose
        values are 1, 1, 1 and 9; alpha 0.25; discount 0.5. The module-level
        derivation gives CVaR = 9.
    When: ``_update_non_leaf_action_node_q_value`` runs.
    Then: Q is ``2 + 0.5 * 9 = 6.5``.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = ICVaRSparseSampling(
        environment=env, branching_factor=2, depth=2, alpha=0.25, name="icvar_ss_exact"
    )
    _, action = _leaf_action_with_children(env, TAIL_VALUES, immediate_cost=2.0)

    planner._update_non_leaf_action_node_q_value(action)

    expected = 2.0 + DISCOUNT * HAND_COMPUTED_CVAR
    assert expected == 6.5
    assert action.q_value == pytest.approx(expected, abs=TOL), (
        f"Q = {action.q_value}, expected 2 + 0.5 * CVaR(1,1,1,9; alpha=0.25) = 2 + 0.5*9 = 6.5; "
        f"using the mean {MEAN_OF_TAIL_VALUES} instead would give "
        f"{2.0 + DISCOUNT * MEAN_OF_TAIL_VALUES}"
    )


def test_icvar_at_alpha_one_reduces_to_the_mean():
    """alpha = 1 makes the tail the whole distribution, so CVaR is the mean.

    Purpose: A boundary that pins the alpha convention. If alpha were the
        *lower* tail probability, alpha = 1 would still give the mean, but
        alpha = 0.25 would give 1 rather than 9 — the test above separates
        that case, and this one fixes the scale.

    Given: The same four children, alpha 1.0, immediate cost 2, discount 0.5.
    When: The action value is computed.
    Then: Q is ``2 + 0.5 * mean(1,1,1,9) = 2 + 0.5*3 = 3.5``.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = ICVaRSparseSampling(
        environment=env, branching_factor=2, depth=2, alpha=1.0, name="icvar_ss_alpha1"
    )
    _, action = _leaf_action_with_children(env, TAIL_VALUES, immediate_cost=2.0)

    planner._update_non_leaf_action_node_q_value(action)

    assert action.q_value == pytest.approx(2.0 + DISCOUNT * MEAN_OF_TAIL_VALUES, abs=TOL)


def test_icvar_is_riskier_than_the_mean_on_a_skewed_sample():
    """A smaller alpha weights the worst outcomes more heavily.

    Purpose: A monotonicity check that would catch a sign flip even where the
        exact arithmetic happened to coincide.

    Given: The same skewed children at alpha 0.25 and alpha 1.0.
    When: Both action values are computed.
    Then: The risk-averse value is strictly larger, because these are costs.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    values = {}
    for alpha in (0.25, 1.0):
        planner = ICVaRSparseSampling(
            environment=env, branching_factor=2, depth=2, alpha=alpha, name=f"icvar_ss_{alpha}"
        )
        _, action = _leaf_action_with_children(env, TAIL_VALUES, immediate_cost=0.0)
        planner._update_non_leaf_action_node_q_value(action)
        values[alpha] = action.q_value

    assert values[0.25] > values[1.0], (
        f"risk-averse value {values[0.25]} is not above the mean-based {values[1.0]}; on a cost "
        "objective a smaller alpha must weight the expensive outcomes more"
    )


def test_icvar_belief_value_is_the_minimum_action_cost():
    """V(b) = min over action children, the cost-setting convention.

    Given: A belief with three action children at Q = 5, 2 and 9.
    When: ``_update_belief_node_v_value`` runs.
    Then: V is 2.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = ICVaRSparseSampling(
        environment=env, branching_factor=2, depth=2, alpha=0.25, name="icvar_ss_min"
    )
    root = BeliefNode(belief=_belief(env))
    for index, q in enumerate((5.0, 2.0, 9.0)):
        node = ActionNode(action=f"a{index}", parent=root, children=tuple(), data=None)
        node.q_value = q

    planner._update_belief_node_v_value(root)

    assert root.v_value == pytest.approx(2.0, abs=TOL)


def test_icvar_rejects_an_alpha_outside_its_range():
    """alpha must lie in (0, 1]; 0 and 1.5 are rejected with a reason.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    for bad_alpha in (0.0, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            ICVaRSparseSampling(
                environment=env, branching_factor=2, depth=2, alpha=bad_alpha, name="bad"
            )
    with pytest.raises(TypeError):
        ICVaRSparseSampling(environment=env, branching_factor=2, depth=2, alpha=1, name="bad_type")


# ---------------------------------------------------------------------------
# Plain sparse sampling: exact backups
# ---------------------------------------------------------------------------


def test_plain_action_value_is_immediate_cost_plus_discounted_child_mean():
    """Q = c + gamma * mean(child V), which is the mean, not a tail.

    Purpose: The counterpart to the CVaR test — it is what makes the two
        planners different, and the same fixture distinguishes them.

    Given: An action node with immediate cost 2 and children valued
        1, 1, 1 and 9; discount 0.5.
    When: ``_update_non_leaf_action_node_q_value`` runs.
    Then: Q is ``2 + 0.5 * 3 = 3.5``, and *not* the 6.5 the CVaR planner gives
        on identical input.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = SparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=2, name="ss_exact"
    )
    _, action = _leaf_action_with_children(env, TAIL_VALUES, immediate_cost=2.0)

    planner._update_non_leaf_action_node_q_value(action)

    assert action.q_value == pytest.approx(2.0 + DISCOUNT * MEAN_OF_TAIL_VALUES, abs=TOL)
    assert action.q_value != pytest.approx(
        6.5, abs=1e-6
    ), "the plain planner must not reproduce the CVaR planner's answer on the same input"


def test_leaf_action_value_is_the_immediate_cost_alone():
    """A leaf action's value has no continuation term.

    Given: A leaf action node under a belief.
    When: ``_update_leaf_node_statistics`` runs.
    Then: Its visit count is 1 and its Q equals its own immediate cost, with no
        discounted future contribution.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = SparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=1, name="ss_leaf"
    )
    root = BeliefNode(belief=_belief(env))
    action = ActionNode(action="listen", parent=root, children=tuple(), data=None)

    planner._update_leaf_node_statistics(action)

    assert action.visit_count == 1
    assert action.q_value == pytest.approx(action.immediate_cost, abs=TOL)
    # Tiger's listen reward is a constant -1 in every state, so the belief's
    # expected *cost* for listening is exactly +1 whatever the particle mix.
    assert action.immediate_cost == pytest.approx(1.0, abs=TOL)


def test_statistics_update_actually_runs_before_the_attributes_are_read():
    """Every node carries a value the update produced, not an initial default.

    Purpose: Replaces a test that inspected attributes without first running
        ``_update_node_statistics``, so it could pass on an untouched tree.
        Here the tree is built, every value is checked to be the untouched
        default first, the update is run, and only then are the results
        asserted — so the assertions cannot be satisfied by initialisation.

    Given: A depth-2, branching-factor-2 tree on Tiger.
    When: ``_update_node_statistics`` runs over it.
    Then: Every leaf action has visit count 1 and a Q equal to its immediate
        cost; every non-leaf action has a Q equal to its immediate cost plus
        the discounted mean of its children's values; and every belief node's
        V is the minimum of its action children's Q values and its visit count
        the sum of theirs.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = SparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=2, name="ss_statistics"
    )
    root = BeliefNode(belief=_belief(env), parent=None, children=tuple())
    planner._build_tree(belief_node=root, current_depth=0)

    untouched = [n for n in PostOrderIter(root) if isinstance(n, ActionNode)]
    assert untouched, "the fixture built no action nodes"
    assert all(node.visit_count == 0 for node in untouched), (
        "action nodes already carry visits before the update, so the assertions below could be "
        "satisfied without it running"
    )

    planner._update_node_statistics(root)

    leaves = non_leaves = beliefs = 0
    for node in PostOrderIter(root):
        if isinstance(node, ActionNode) and node.is_leaf:
            leaves += 1
            assert node.visit_count == 1
            assert node.q_value == pytest.approx(node.immediate_cost, abs=TOL)
        elif isinstance(node, ActionNode):
            non_leaves += 1
            expected = node.immediate_cost + DISCOUNT * float(
                np.mean([child.v_value for child in node.children])
            )
            assert node.q_value == pytest.approx(expected, abs=1e-9), (
                f"action {node.action!r} at height {node.height}: Q = {node.q_value}, expected "
                f"{expected} = immediate cost + {DISCOUNT} * mean of child values"
            )
        elif isinstance(node, BeliefNode) and not node.is_leaf:
            beliefs += 1
            assert node.v_value == pytest.approx(
                min(child.q_value for child in node.children), abs=TOL
            )
            assert node.visit_count == sum(child.visit_count for child in node.children)

    assert leaves > 0 and non_leaves > 0 and beliefs > 0, (
        f"the walk saw leaves={leaves}, non-leaf actions={non_leaves}, beliefs={beliefs}; each "
        "case must occur or its assertions never ran"
    )


def test_tree_shape_is_exactly_the_declared_branching_and_depth():
    """The sparse-sampling tree is full and fixed, with no widening.

    Purpose: Progressive widening does not apply to this planner; its shape is
        a contract instead. Asserting the exact counts is what makes the value
        tests above meaningful.

    Given: Three Tiger actions, branching factor 2, depth 2.
    When: ``_build_tree`` runs.
    Then: Every belief node has exactly 3 action children; every non-leaf
        action has exactly 2 belief children; and every leaf action sits at the
        same depth.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = SparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=2, name="ss_shape"
    )
    root = BeliefNode(belief=_belief(env), parent=None, children=tuple())
    planner._build_tree(belief_node=root, current_depth=0)

    n_actions = len(env.get_actions())
    leaf_depths = set()
    for node in PostOrderIter(root):
        if isinstance(node, BeliefNode):
            assert len(node.children) == n_actions, (
                f"belief node at depth {node.depth} has {len(node.children)} action children, "
                f"expected {n_actions}"
            )
        elif node.is_leaf:
            leaf_depths.add(node.depth)
        else:
            assert len(node.children) == planner.branching_factor

    assert len(leaf_depths) == 1, f"leaves sit at differing depths {sorted(leaf_depths)}"


def test_declared_metric_contract_is_the_empty_list():
    """Both sparse-sampling planners deliberately report no tree metrics.

    Purpose: The absence is a decision, not an omission — these planners build
        a fixed full tree, so the visit-distribution metrics the MCTS planners
        report would be constants. The contract is asserted rather than
        an invented metric set added.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    assert SparseSamplingDiscreteActionsPlanner.get_info_variable_names() == []
    assert ICVaRSparseSampling.get_info_variable_names() == []

    planner = SparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=1, name="ss_metrics"
    )
    _, run_data = planner.action(_belief(env))
    assert run_data.info_variables == []


def test_action_returns_one_legal_action_and_does_not_mutate_the_belief():
    """Public contract plus caller-belief immutability.

    Test type: unit
    """
    np.random.seed(5)
    random.seed(5)
    env = TigerPOMDP(discount_factor=DISCOUNT)
    planner = SparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=2, depth=2, name="ss_contract"
    )
    belief = _belief(env)
    before_particles = list(belief.particles)
    before_weights = np.array(belief.log_weights, copy=True)

    actions, _ = planner.action(belief)

    assert len(actions) == 1 and actions[0] in env.get_actions()
    assert belief.particles == before_particles
    assert np.array_equal(np.asarray(belief.log_weights), before_weights)


def test_invalid_construction_parameters_are_rejected():
    """Depth and branching factor must be positive integers.

    Test type: unit
    """
    env = TigerPOMDP(discount_factor=DISCOUNT)
    with pytest.raises(ValueError, match="Depth"):
        SparseSamplingDiscreteActionsPlanner(
            environment=env, branching_factor=2, depth=0, name="bad"
        )
    with pytest.raises(ValueError, match="Branching factor"):
        SparseSamplingDiscreteActionsPlanner(
            environment=env, branching_factor=0, depth=2, name="bad"
        )
    with pytest.raises(TypeError):
        SparseSamplingDiscreteActionsPlanner(
            environment=env, branching_factor=2.5, depth=2, name="bad"  # type: ignore[arg-type]
        )
