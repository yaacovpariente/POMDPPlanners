"""Tests for cvar_exploration LCB action selection."""

import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBeliefStateUpdate
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.planners_utils.cvar_exploration import (
    _sparse_sampling_guarantees_exploration_v2_arena,
)


def _build_tree_with_visited_action_children(q_values, visit_counts):
    tree = Tree()
    parent_id = tree.add_belief_node(
        belief=WeightedParticleBeliefStateUpdate(particles=[], weights=[]),
        parent_id=None,
        weight=1.0,
    )
    tree.visit_count[parent_id] = 100
    child_ids = []
    for i, (q, v) in enumerate(zip(q_values, visit_counts)):
        cid = tree.add_action_node(action=i, parent_id=parent_id)
        tree.visit_count[cid] = v
        tree.q_value[cid] = float(q)
        child_ids.append(cid)
    return tree, parent_id, child_ids


def test_sparse_sampling_lcb_is_deterministic_when_no_unvisited_children():
    """Picks lowest-LCB child when all action children have visit_count >= 1.

    Purpose: Validates the LCB code path actually fires once every action
    child has been visited at least once. With a clear Q-value gap and a
    small exploration_constant, LCB selection is deterministic, so the
    same call under different RNG seeds must return the same child.

    Given: A belief node with 3 visited action children. Two have
        q_value=5.0 and one has q_value=0.1; all share visit_count=10
        so the per-child exploration term collapses to a constant. The
        lowest-Q child therefore has the lowest LCB.
    When: _sparse_sampling_guarantees_exploration_v2_arena is called
        repeatedly under different np.random seeds.
    Then: Every call returns the same child (the lowest-Q one).

    Test type: unit
    """
    tree, parent_id, child_ids = _build_tree_with_visited_action_children(
        q_values=[5.0, 5.0, 0.1],
        visit_counts=[10, 10, 10],
    )
    expected = child_ids[2]
    results = set()
    for seed in range(20):
        np.random.seed(seed)
        results.add(
            _sparse_sampling_guarantees_exploration_v2_arena(
                tree=tree,
                belief_id=parent_id,
                exploration_constant=0.1,
                alpha=0.1,
                min_cost=-10.0,
                max_cost=10.0,
                horizon=5,
                delta=0.1,
                visit_count_penalty=0.0,
            )
        )
    assert results == {
        expected
    }, f"expected deterministic LCB pick of child[2]={expected}; got {results}"


def test_sparse_sampling_lcb_prefers_less_visited_child_when_q_tied():
    """With Q-values tied, LCB selection favors the least-visited child.

    Purpose: Validates that the per-child exploration bound (which scales
    with 1/sqrt(child_visits) inside the guarantees-bound formula)
    actually drives the choice toward less-visited children when Q is
    not a differentiator.

    Given: Three action children with identical q_value=1.0 but visit
        counts [50, 10, 100]; child[1] is the least visited.
    When: _sparse_sampling_guarantees_exploration_v2_arena is called
        with a large exploration_constant so the bonus dominates.
    Then: Every call returns child[1] regardless of RNG seed.

    Test type: unit
    """
    tree, parent_id, child_ids = _build_tree_with_visited_action_children(
        q_values=[1.0, 1.0, 1.0],
        visit_counts=[50, 10, 100],
    )
    expected = child_ids[1]
    results = set()
    for seed in range(20):
        np.random.seed(seed)
        results.add(
            _sparse_sampling_guarantees_exploration_v2_arena(
                tree=tree,
                belief_id=parent_id,
                exploration_constant=5.0,
                alpha=0.1,
                min_cost=-10.0,
                max_cost=10.0,
                horizon=5,
                delta=0.1,
                visit_count_penalty=0.0,
            )
        )
    assert results == {expected}, (
        f"expected deterministic LCB pick of least-visited child[1]={expected}; " f"got {results}"
    )


def test_sparse_sampling_lcb_falls_back_to_greedy_when_horizon_is_zero():
    """At horizon=0 the LCB bound is undefined — fall back to greedy q-min.

    Purpose: Validates that the LCB action selector does not return a
        systematic action-index-0 default when horizon=0. Without the
        fix, log(1 - belief_visits**0) = log(0) = -inf collapses every
        per-action score to NaN, and the comparison ``score < best_score``
        is False for NaN, so the kernel always returns index 0 — even
        when later actions have strictly lower q-values.

    Given: A belief node with 3 visited action children, all with
        visit_count=5; child[2] has the strictly lowest q_value (0.1
        vs. 5.0 for the others).
    When: _sparse_sampling_guarantees_exploration_v2_arena is called
        with horizon=0.
    Then: Returns child[2] (lowest q), not child[0] by default.

    Test type: unit
    """
    tree, parent_id, child_ids = _build_tree_with_visited_action_children(
        q_values=[5.0, 5.0, 0.1],
        visit_counts=[5, 5, 5],
    )
    expected = child_ids[2]
    np.random.seed(0)
    result = _sparse_sampling_guarantees_exploration_v2_arena(
        tree=tree,
        belief_id=parent_id,
        exploration_constant=1.0,
        alpha=0.1,
        min_cost=-10.0,
        max_cost=10.0,
        horizon=0,
        delta=0.1,
        visit_count_penalty=0.0,
    )
    assert result == expected, (
        f"horizon=0 should fall back to greedy q-min selection; "
        f"got {result} (children={child_ids}, expected child[2]={expected})"
    )
