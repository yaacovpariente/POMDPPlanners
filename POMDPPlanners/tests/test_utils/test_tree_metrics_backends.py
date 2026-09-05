# SPDX-License-Identifier: MIT

"""Independent tree-metric tests for all three backends.

``compute_tree_metrics`` (object), ``compute_arena_tree_metrics`` (arena) and
``compute_vectorized_tree_metrics`` (flat tensor) are meant to report the same
:class:`TreeMetrics` set so planners on different backends can be compared. Only
the object backend had unit tests, and three of them were empty ``pass`` bodies
marked skipped on the false premise that the helper handles only leaves.

Every expected value here is worked out in the test — entropy from
``-sum(p log2 p)`` written out by hand, depth from an explicit walk. The
production helper is never called to produce its own expected result.

The three backends deliberately disagree on the depth unit, and that is
asserted rather than papered over:

* arena: maximum edge depth from the root, so one planning transition (belief
  to action to belief) counts as 2;
* object: ``tree.height - 1``, one less than the arena figure on the same shape;
* vectorized: maximum *active* belief-node depth, counted in planning steps,
  and root visits are the sum of the root actions' visits rather than a stored
  root counter.
"""

import math

import numpy as np
import pytest
import torch

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.core.tree.vectorized_belief_tree import VectorizedBeliefTree
from POMDPPlanners.utils.tree_statistics import (
    TreeMetrics,
    compute_arena_tree_metrics,
    compute_tree_metrics,
    compute_vectorized_tree_metrics,
)
from POMDPPlanners.tests.test_planners.tree_assertions import visit_entropy


# Visit counts 1 and 3 give proportions 0.25 and 0.75. Written out rather than
# imported so a wrong logarithm base in the helper cannot hide behind a shared
# constant: -(0.25*log2 0.25 + 0.75*log2 0.75).
SKEWED_VISITS = (1, 3)
SKEWED_ENTROPY = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))
# Entropy is a sum of two logarithms; float64 gives ~1e-16 relative error, so
# 1e-9 is comfortably tight while immune to the last-bit difference between
# scipy's implementation and this one.
ENTROPY_TOL = 1e-9


def _as_dict(metrics):
    names = [m.name for m in metrics]
    assert len(names) == len(set(names)), f"duplicate metric names: {names}"
    return {m.name: m.value for m in metrics}


def _belief():
    return WeightedParticleBelief(particles=[1, 2], log_weights=np.log(np.array([0.6, 0.4])))


def test_skewed_entropy_constant_is_what_it_claims():
    """The hand-written entropy agrees with the independent helper.

    Purpose: Both sides of every entropy assertion below are computed outside
        the production code; this pins the constant itself so a typo in it
        cannot silently relax the checks.

    Test type: unit
    """
    assert visit_entropy(SKEWED_VISITS) == pytest.approx(SKEWED_ENTROPY, abs=ENTROPY_TOL)
    assert SKEWED_ENTROPY == pytest.approx(0.8112781244591328, abs=1e-12)


# ---------------------------------------------------------------------------
# Object backend
# ---------------------------------------------------------------------------


def _object_tree_with_visits(visits):
    root = BeliefNode(belief=_belief())
    root.visit_count = sum(visits)
    for index, count in enumerate(visits):
        child = ActionNode(action=f"a{index}", parent=root, children=tuple(), data=None)
        child.visit_count = count
    return root


def test_object_tree_uniform_visits():
    """Uniform action visits give maximum entropy and equal extrema.

    Purpose: Replaces one of the three empty ``pass`` bodies in
        ``test_tree_statistics.py``, which were skipped on the incorrect claim
        that the helper only supports leaves.

    Given: A root with four action children at 2 visits each.
    When: ``compute_tree_metrics`` runs.
    Then: min = max = 2, the action count is 4, the root visit count is 8, and
        the entropy is exactly ``log2(4) = 2``, the maximum for four actions.

    Test type: unit
    """
    metrics = _as_dict(compute_tree_metrics(_object_tree_with_visits((2, 2, 2, 2))))

    assert metrics[TreeMetrics.MIN_ACTIONS_VISIT_COUNT.value] == 2
    assert metrics[TreeMetrics.MAX_ACTIONS_VISIT_COUNT.value] == 2
    assert metrics[TreeMetrics.N_ACTIONS_FROM_ROOT.value] == 4
    assert metrics[TreeMetrics.ROOT_VISIT_COUNT.value] == 8
    assert metrics[TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value] == pytest.approx(
        2.0, abs=ENTROPY_TOL
    )
    assert metrics[TreeMetrics.IS_LEAF.value] == 0


def test_object_tree_skewed_visits():
    """Skewed visits give the hand-computed entropy and the right extrema.

    Purpose: Replaces the second empty ``pass`` body. A wrong logarithm base
        (natural instead of base 2) would give 0.5623, and dividing by the root
        visit count instead of the action total would give a different figure
        again; the assertion separates all three.

    Given: A root with action children at 1 and 3 visits.
    When: ``compute_tree_metrics`` runs.
    Then: min 1, max 3, two actions, root visits 4, entropy 0.811278.

    Test type: unit
    """
    metrics = _as_dict(compute_tree_metrics(_object_tree_with_visits(SKEWED_VISITS)))

    assert metrics[TreeMetrics.MIN_ACTIONS_VISIT_COUNT.value] == 1
    assert metrics[TreeMetrics.MAX_ACTIONS_VISIT_COUNT.value] == 3
    assert metrics[TreeMetrics.N_ACTIONS_FROM_ROOT.value] == 2
    assert metrics[TreeMetrics.ROOT_VISIT_COUNT.value] == 4
    assert metrics[TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value] == pytest.approx(
        SKEWED_ENTROPY, abs=ENTROPY_TOL
    )


def test_object_tree_all_zero_visits_gives_zero_entropy():
    """Replaces the third empty ``pass`` body: an unvisited expanded root.

    Purpose: The entropy of an all-zero visit vector is undefined as a
        distribution; the documented answer is 0, not NaN.

    Given: A root with two action children, neither visited.
    When: ``compute_tree_metrics`` runs.
    Then: min, max, root visits and entropy are all 0, the action count is 2,
        and the entropy is a real number rather than NaN.

    Test type: unit
    """
    metrics = _as_dict(compute_tree_metrics(_object_tree_with_visits((0, 0))))

    assert metrics[TreeMetrics.MIN_ACTIONS_VISIT_COUNT.value] == 0
    assert metrics[TreeMetrics.MAX_ACTIONS_VISIT_COUNT.value] == 0
    assert metrics[TreeMetrics.N_ACTIONS_FROM_ROOT.value] == 2
    assert metrics[TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value] == 0.0
    assert not math.isnan(float(metrics[TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value]))


def test_object_leaf_emits_only_four_fields():
    """A leaf root reports the reduced metric set, and that is a contract.

    Purpose: The leaf branch omits ``n_actions_from_root``, ``root_visit_count``
        and ``tree_max_depth`` entirely. A caller that substitutes measured
        zeros for the absent fields is comparing different quantities, so the
        absence is asserted rather than tolerated.

    Given: A root belief node with no action children.
    When: ``compute_tree_metrics`` runs.
    Then: Exactly four metrics come back, ``is_leaf`` is 1, and the three
        non-leaf-only names are missing.

    Test type: unit
    """
    metrics = _as_dict(compute_tree_metrics(BeliefNode(belief=_belief())))

    assert set(metrics) == {
        TreeMetrics.MIN_ACTIONS_VISIT_COUNT.value,
        TreeMetrics.MAX_ACTIONS_VISIT_COUNT.value,
        TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value,
        TreeMetrics.IS_LEAF.value,
    }
    assert metrics[TreeMetrics.IS_LEAF.value] == 1


def test_object_depth_is_height_minus_one():
    """The object backend reports ``tree.height - 1``.

    Purpose: Pins the depth unit so a cross-backend comparison cannot silently
        mix it with the arena's edge count.

    Given: root -> action -> belief -> action, i.e. an anytree height of 3.
    When: ``compute_tree_metrics`` runs.
    Then: ``tree_max_depth`` is 2, one less than the 3 edges the arena walker
        would report on the same shape.

    Test type: unit
    """
    root = BeliefNode(belief=_belief())
    root.visit_count = 4
    action = ActionNode(action="a0", parent=root, children=tuple(), data=None)
    action.visit_count = 4
    grandchild = BeliefNode(belief=_belief(), parent=action, children=tuple(), data=None)
    ActionNode(action="a1", parent=grandchild, children=tuple(), data=None)

    metrics = _as_dict(compute_tree_metrics(root))

    assert root.height == 3
    assert metrics[TreeMetrics.TREE_MAX_DEPTH.value] == root.height - 1 == 2


def test_object_helper_rejects_a_non_belief_root():
    """A wrong root type fails loudly.

    Test type: unit
    """
    with pytest.raises(TypeError):
        compute_tree_metrics(ActionNode(action="a", parent=None, children=tuple(), data=None))


# ---------------------------------------------------------------------------
# Arena backend
# ---------------------------------------------------------------------------


def _arena_tree_with_visits(visits):
    tree = Tree()
    root_id = tree.add_belief_node(_belief())
    tree.visit_count[root_id] = sum(visits)
    for index, count in enumerate(visits):
        action_id = tree.add_action_node(action=f"a{index}", parent_id=root_id)
        tree.visit_count[action_id] = count
    return tree, root_id


def test_arena_skewed_visits_metrics():
    """The arena backend reports the same numbers as the object backend.

    Purpose: The arena helper had no independent unit test at all; every arena
        planner's reported metrics come from it.

    Given: A root with action visits 1 and 3 and a root visit count of 4.
    When: ``compute_arena_tree_metrics`` runs.
    Then: min 1, max 3, two actions, root visits 4, entropy 0.811278, and
        ``is_leaf`` 0.

    Test type: unit
    """
    tree, root_id = _arena_tree_with_visits(SKEWED_VISITS)

    metrics = _as_dict(compute_arena_tree_metrics(tree=tree, root_id=root_id))

    assert metrics[TreeMetrics.MIN_ACTIONS_VISIT_COUNT.value] == 1
    assert metrics[TreeMetrics.MAX_ACTIONS_VISIT_COUNT.value] == 3
    assert metrics[TreeMetrics.N_ACTIONS_FROM_ROOT.value] == 2
    assert metrics[TreeMetrics.ROOT_VISIT_COUNT.value] == 4
    assert metrics[TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value] == pytest.approx(
        SKEWED_ENTROPY, abs=ENTROPY_TOL
    )
    assert metrics[TreeMetrics.IS_LEAF.value] == 0


def test_arena_root_visit_count_is_the_stored_root_counter_not_a_child_sum():
    """The arena root visit count is read off the root, not summed from children.

    Purpose: The vectorized backend derives it as a child sum; the arena one
        does not, and mixing them up would silently change every reported
        figure for planners whose root gains visits its children do not.

    Given: A root whose own visit count is 10 while its actions sum to 4.
    When: ``compute_arena_tree_metrics`` runs.
    Then: The reported root visit count is 10.

    Test type: unit
    """
    tree, root_id = _arena_tree_with_visits(SKEWED_VISITS)
    tree.visit_count[root_id] = 10

    metrics = _as_dict(compute_arena_tree_metrics(tree=tree, root_id=root_id))

    assert metrics[TreeMetrics.ROOT_VISIT_COUNT.value] == 10


def test_arena_depth_is_maximum_edge_depth():
    """The arena backend counts edges, so one planning step is two.

    Given: root -> action -> belief -> action, three edges.
    When: ``compute_arena_tree_metrics`` runs.
    Then: ``tree_max_depth`` is 3, one more than the object backend's figure on
        the same shape.

    Test type: unit
    """
    tree = Tree()
    root_id = tree.add_belief_node(_belief())
    tree.visit_count[root_id] = 4
    action_id = tree.add_action_node(action="a0", parent_id=root_id)
    tree.visit_count[action_id] = 4
    child_id = tree.add_belief_node(_belief(), observation="o", parent_id=action_id)
    tree.add_action_node(action="a1", parent_id=child_id)

    metrics = _as_dict(compute_arena_tree_metrics(tree=tree, root_id=root_id))

    assert metrics[TreeMetrics.TREE_MAX_DEPTH.value] == 3


def test_arena_leaf_emits_only_four_fields():
    """An unexpanded arena root reports the same reduced set as the object one.

    Test type: unit
    """
    tree = Tree()
    root_id = tree.add_belief_node(_belief())

    metrics = _as_dict(compute_arena_tree_metrics(tree=tree, root_id=root_id))

    assert set(metrics) == {
        TreeMetrics.MIN_ACTIONS_VISIT_COUNT.value,
        TreeMetrics.MAX_ACTIONS_VISIT_COUNT.value,
        TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value,
        TreeMetrics.IS_LEAF.value,
    }
    assert metrics[TreeMetrics.IS_LEAF.value] == 1


def test_arena_all_zero_visits_gives_zero_entropy():
    """An expanded but unvisited arena root gives entropy 0, not NaN.

    Test type: unit
    """
    tree, root_id = _arena_tree_with_visits((0, 0))
    tree.visit_count[root_id] = 0

    metrics = _as_dict(compute_arena_tree_metrics(tree=tree, root_id=root_id))

    assert metrics[TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value] == 0.0
    assert metrics[TreeMetrics.N_ACTIONS_FROM_ROOT.value] == 2


# ---------------------------------------------------------------------------
# Vectorized backend
# ---------------------------------------------------------------------------


def _vector_tree_with_visits(visits, belief_capacity=64, action_capacity=64):
    tree = VectorizedBeliefTree(belief_capacity=belief_capacity, action_capacity=action_capacity)
    root = tree.root_index
    parents = torch.full((len(visits),), root, dtype=tree.index_dtype)
    keys = torch.arange(len(visits), dtype=tree.index_dtype)
    action_nodes, _ = tree.get_or_create_actions(parents, keys)
    tree.action_visit_count[action_nodes] = torch.as_tensor(
        list(visits), dtype=tree.action_visit_count.dtype
    )
    return tree, action_nodes


def test_vectorized_skewed_visits_metrics():
    """The vectorized backend agrees on min, max, entropy and action count.

    Purpose: VOPP's reported metrics come from here, and this helper had no
        arithmetic test of its own.

    Given: A root with two action nodes at 1 and 3 visits.
    When: ``compute_vectorized_tree_metrics`` runs.
    Then: min 1, max 3, two actions, entropy 0.811278, and the root visit count
        is the *sum* 4 — the documented vectorized convention, since no root
        counter is stored.

    Test type: unit
    """
    tree, _ = _vector_tree_with_visits(SKEWED_VISITS)

    metrics = _as_dict(compute_vectorized_tree_metrics(tree))

    assert metrics[TreeMetrics.MIN_ACTIONS_VISIT_COUNT.value] == 1
    assert metrics[TreeMetrics.MAX_ACTIONS_VISIT_COUNT.value] == 3
    assert metrics[TreeMetrics.N_ACTIONS_FROM_ROOT.value] == 2
    assert metrics[TreeMetrics.ROOT_VISIT_COUNT.value] == 4
    assert metrics[TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value] == pytest.approx(
        SKEWED_ENTROPY, abs=ENTROPY_TOL
    )
    assert metrics[TreeMetrics.IS_LEAF.value] == 0


def test_vectorized_metrics_ignore_padded_capacity():
    """Preallocated but unused tensor slots contribute nothing.

    Purpose: This is the failure mode unique to the flat-tensor backend. The
        columns are allocated to capacity and filled with sentinels; a helper
        that read the whole column instead of ``[:num_action_nodes]`` would
        report a minimum visit count of 0 and a depth taken from padding.

    Given: A tree with capacity 64 but only two live action nodes at 1 and 3
        visits, with the padded region deliberately poisoned with a large
        visit count and a large depth.
    When: ``compute_vectorized_tree_metrics`` runs.
    Then: The metrics are identical to the unpoisoned tree — minimum 1, maximum
        3, root visits 4 — so no padded slot leaked in.

    Test type: unit
    """
    tree, _ = _vector_tree_with_visits(SKEWED_VISITS, belief_capacity=64, action_capacity=64)
    live_actions = tree.num_action_nodes
    live_beliefs = tree.num_belief_nodes
    tree.action_visit_count[live_actions:] = 999
    tree.belief_depth[live_beliefs:] = 77
    tree.action_parent_belief[live_actions:] = tree.root_index

    metrics = _as_dict(compute_vectorized_tree_metrics(tree))

    assert (
        metrics[TreeMetrics.MIN_ACTIONS_VISIT_COUNT.value] == 1
    ), "a padded slot with 999 visits leaked into the minimum"
    assert (
        metrics[TreeMetrics.MAX_ACTIONS_VISIT_COUNT.value] == 3
    ), "a padded slot with 999 visits leaked into the maximum"
    assert metrics[TreeMetrics.N_ACTIONS_FROM_ROOT.value] == 2
    assert metrics[TreeMetrics.ROOT_VISIT_COUNT.value] == 4
    assert (
        metrics[TreeMetrics.TREE_MAX_DEPTH.value] == 0
    ), "a padded belief slot at depth 77 leaked into the reported depth"


def test_vectorized_depth_is_the_deepest_active_belief_node():
    """Depth is counted in planning steps over active belief nodes only.

    Given: root -> action -> belief, so the deepest active belief node is at
        depth 1.
    When: ``compute_vectorized_tree_metrics`` runs.
    Then: ``tree_max_depth`` is 1 — the vectorized unit, half the arena's
        2-edge figure for the same shape.

    Test type: unit
    """
    tree, action_nodes = _vector_tree_with_visits((1,))
    observation_keys = torch.zeros(1, dtype=tree.index_dtype)
    tree.get_or_create_beliefs(action_nodes[:1], observation_keys)

    metrics = _as_dict(compute_vectorized_tree_metrics(tree))

    assert metrics[TreeMetrics.TREE_MAX_DEPTH.value] == 1


def test_vectorized_leaf_root_emits_only_four_fields():
    """A vectorized root with no actions reports the reduced set.

    Test type: unit
    """
    tree = VectorizedBeliefTree(belief_capacity=8, action_capacity=8)

    metrics = _as_dict(compute_vectorized_tree_metrics(tree))

    assert set(metrics) == {
        TreeMetrics.MIN_ACTIONS_VISIT_COUNT.value,
        TreeMetrics.MAX_ACTIONS_VISIT_COUNT.value,
        TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value,
        TreeMetrics.IS_LEAF.value,
    }
    assert metrics[TreeMetrics.IS_LEAF.value] == 1


# ---------------------------------------------------------------------------
# Cross-backend agreement
# ---------------------------------------------------------------------------


def test_all_three_backends_agree_on_the_visit_statistics():
    """On the same visit vector the three helpers report the same four numbers.

    Purpose: This is the point of having three implementations. Depth is
        deliberately excluded — the three units differ, and each is pinned
        separately above.

    Given: Action visits 1 and 3 built on each backend.
    When: Each backend's helper runs.
    Then: min, max, action count and entropy agree exactly.

    Test type: unit
    """
    object_metrics = _as_dict(compute_tree_metrics(_object_tree_with_visits(SKEWED_VISITS)))
    arena_tree, arena_root = _arena_tree_with_visits(SKEWED_VISITS)
    arena_metrics = _as_dict(compute_arena_tree_metrics(tree=arena_tree, root_id=arena_root))
    vector_tree, _ = _vector_tree_with_visits(SKEWED_VISITS)
    vector_metrics = _as_dict(compute_vectorized_tree_metrics(vector_tree))

    shared = [
        TreeMetrics.MIN_ACTIONS_VISIT_COUNT.value,
        TreeMetrics.MAX_ACTIONS_VISIT_COUNT.value,
        TreeMetrics.N_ACTIONS_FROM_ROOT.value,
    ]
    for name in shared:
        assert object_metrics[name] == arena_metrics[name] == vector_metrics[name], (
            f"backends disagree on {name}: object={object_metrics[name]}, "
            f"arena={arena_metrics[name]}, vector={vector_metrics[name]}"
        )
    entropies = [
        float(object_metrics[TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value]),
        float(arena_metrics[TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value]),
        float(vector_metrics[TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value]),
    ]
    for value in entropies:
        assert value == pytest.approx(SKEWED_ENTROPY, abs=ENTROPY_TOL)


def test_every_backend_reports_every_declared_metric_name_when_not_a_leaf():
    """A non-leaf tree reports exactly the declared :class:`TreeMetrics` set.

    Purpose: ``get_info_variable_names`` promises this set to the statistics
        layer; a helper that dropped one would leave a column of NaNs in every
        results table.

    Test type: unit
    """
    declared = {metric.value for metric in TreeMetrics}
    arena_tree, arena_root = _arena_tree_with_visits(SKEWED_VISITS)
    vector_tree, _ = _vector_tree_with_visits(SKEWED_VISITS)

    for name, metrics in (
        ("object", compute_tree_metrics(_object_tree_with_visits(SKEWED_VISITS))),
        ("arena", compute_arena_tree_metrics(tree=arena_tree, root_id=arena_root)),
        ("vector", compute_vectorized_tree_metrics(vector_tree)),
    ):
        assert set(_as_dict(metrics)) == declared, f"{name} backend metric set differs"
