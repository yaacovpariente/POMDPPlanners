"""Tests for cvar_exploration LCB action selection."""

import math

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


def test_sparse_sampling_lcb_handles_overflow_at_deep_horizon_and_many_visits():
    """LCB does not collapse to action-index-0 when belief_visits**horizon overflows.

    Purpose: Validates that the LCB action selector remains correct when
        ``belief_visits ** horizon`` exceeds float64 max. With the naive
        formula ``x1 = 1 - belief_visits**horizon`` overflows to -inf,
        ``x3 = log(x1 / x2)`` evaluates to +inf, ``bound`` evaluates to
        +inf, and every per-action score collapses to -inf. After the
        first iteration sets best_score=-inf the strict comparison
        ``score < best_score`` is False for all subsequent indices, so
        the kernel returns index 0 — even when later actions have
        strictly lower q-values. The fix must be a log-space rewrite
        that stays finite for any (belief_visits, horizon).

    Given: A belief node with visit_count=10_000 and 3 visited action
        children, all with visit_count=10; child[2] has the strictly
        lowest q_value (0.1 vs. 5.0 for the others). horizon=90 makes
        belief_visits**horizon = 10_000**90 = 1e360, which overflows
        float64 (max ≈ 1.8e308).
    When: _sparse_sampling_guarantees_exploration_v2_arena is called.
    Then: Returns child[2] (lowest q-value), not child[0] by default.

    Test type: unit
    """
    tree, parent_id, child_ids = _build_tree_with_visited_action_children(
        q_values=[5.0, 5.0, 0.1],
        visit_counts=[10, 10, 10],
    )
    tree.visit_count[parent_id] = 10_000
    expected = child_ids[2]
    np.random.seed(0)
    result = _sparse_sampling_guarantees_exploration_v2_arena(
        tree=tree,
        belief_id=parent_id,
        exploration_constant=1.0,
        alpha=0.1,
        min_cost=-10.0,
        max_cost=10.0,
        horizon=90,
        delta=0.1,
        visit_count_penalty=0.0,
    )
    assert result == expected, (
        f"deep-horizon LCB must not collapse to index 0 on overflow; "
        f"got {result} (children={child_ids}, expected child[2]={expected})"
    )


def test_sparse_sampling_lcb_matches_log_space_formula_on_safe_inputs():
    """LCB selection on inputs that do NOT overflow agrees with the log-space form.

    Purpose: Anchors the new log-space implementation against the
        analytical formula on inputs where the original ``belief_visits
        ** horizon`` form is safely representable in float64. This guards
        against the log-space rewrite silently changing the LCB ordering
        on the regime where the old code was already correct.

    Given: A belief node with visit_count=20 and 3 visited action
        children. q_values are spread enough that the LCB ordering is
        stable; visit counts differ to exercise the per-child bound.
        horizon=10 keeps 20**10 ≈ 1e13 well under float64 max.
    When: _sparse_sampling_guarantees_exploration_v2_arena is called and
        the LCB is recomputed in log-space (``horizon * log(belief_visits)
        - log(delta * (belief_visits - 1))``).
    Then: The selector returns the index with the lowest analytic LCB.

    Test type: unit
    """
    q_values = [5.0, 4.0, 3.0]
    visit_counts = [50, 5, 100]
    tree, parent_id, child_ids = _build_tree_with_visited_action_children(
        q_values=q_values,
        visit_counts=visit_counts,
    )
    belief_visits = 20
    tree.visit_count[parent_id] = belief_visits
    horizon = 10
    alpha = 0.1
    delta = 0.1
    min_cost = -10.0
    max_cost = 10.0
    exploration_constant = 1.0

    log_x3 = horizon * math.log(belief_visits) - math.log(delta * (belief_visits - 1.0))
    cost_range = max_cost - min_cost
    expected_scores = [
        q - exploration_constant * cost_range * math.sqrt(log_x3 / (alpha * v))
        for q, v in zip(q_values, visit_counts)
    ]
    expected_idx = expected_scores.index(min(expected_scores))
    expected = child_ids[expected_idx]

    np.random.seed(0)
    result = _sparse_sampling_guarantees_exploration_v2_arena(
        tree=tree,
        belief_id=parent_id,
        exploration_constant=exploration_constant,
        alpha=alpha,
        min_cost=min_cost,
        max_cost=max_cost,
        horizon=horizon,
        delta=delta,
        visit_count_penalty=0.0,
    )
    assert result == expected, (
        f"log-space LCB should match the analytic argmin; "
        f"got {result} (children={child_ids}, expected child[{expected_idx}]={expected}, "
        f"scores={expected_scores})"
    )
