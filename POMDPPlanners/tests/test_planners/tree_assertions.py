# SPDX-License-Identifier: MIT

"""Backend-independent structural and range checks for planner search trees.

The planner suites in this repository walk a tree from its root and stop. That
walk cannot see a node the search allocated but never linked in, a parent
pointer that disagrees with the child list, or a cumulative weight table left
stale by an in-place weight bump — all of which are silent corruptions that
change which child gets sampled next.

Everything here recomputes what it checks from first principles. Nothing calls
a production helper to produce its own expected value, and every walker
returns a :class:`WalkCounters` the caller is expected to assert on, so a check
that happened to visit no interesting node cannot pass by vacuous truth.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree


@dataclass
class WalkCounters:
    """How many of each interesting case the walk actually saw.

    A structural assertion inside a loop proves nothing if the loop ran zero
    times. Tests assert on these counters so "the tree had no expanded non-root
    belief" fails loudly instead of passing quietly.
    """

    belief_nodes: int = 0
    action_nodes: int = 0
    non_root_belief_nodes: int = 0
    expanded_non_root_belief_nodes: int = 0
    visited_action_nodes: int = 0
    unvisited_action_nodes: int = 0
    multi_child_action_nodes: int = 0
    reused_belief_children: int = 0
    max_edge_depth: int = 0
    depths: Dict[int, int] = field(default_factory=dict)


def walk_arena_tree(
    tree: Tree,
    root_id: int,
    *,
    expect_all_nodes_reachable: bool = True,
    weight_tolerance: float = 1e-9,
) -> WalkCounters:
    """Walk every node of an arena tree and check its structural invariants.

    Checks, in order of what they catch:

    * the root is a belief node with no parent and no stored observation;
    * every child ID is in range, its ``parent_id`` points back at the parent,
      and its kind alternates belief/action;
    * no node is reached twice (a cycle or a duplicated edge) and no parent
      lists the same child twice;
    * every reached node's visit count is a non-negative integer and its
      ``q_value``/``v_value`` are finite;
    * belief nodes carry a belief payload and action nodes carry an action;
    * each parent's ``children_cdf`` has one entry per child, is non-decreasing,
      and equals the running sum of the children's ``weight`` at *every*
      position — not only at the last one, which is what a stale in-place bump
      corrupts;
    * with ``expect_all_nodes_reachable``, the set of reached IDs equals
      ``range(len(tree))``, so a node allocated but never linked is caught.

    Args:
        tree: The arena tree to inspect.
        root_id: The tree's root belief node.
        expect_all_nodes_reachable: Set ``False`` only for a tree that
            deliberately retains detached storage; state the reason at the
            call site.
        weight_tolerance: Absolute tolerance for the CDF comparison. The CDF
            is a running float sum over a handful of small weights, so
            ``1e-9`` is far above the accumulated rounding error and far below
            any real bookkeeping mistake (the smallest real bump is ``1.0``).

    Returns:
        A :class:`WalkCounters` describing what the walk saw.
    """
    size = len(tree)
    assert 0 <= root_id < size, f"root_id {root_id} outside [0, {size})"
    assert (
        tree.kind[root_id] == BELIEF
    ), f"root {root_id} kind={tree.kind[root_id]}, expected BELIEF"
    assert tree.parent_id[root_id] is None, f"root {root_id} has parent {tree.parent_id[root_id]}"
    assert tree.observation[root_id] is None, (
        f"root {root_id} stores observation {tree.observation[root_id]!r}; the root belief "
        "is not reached by an observation"
    )

    counters = WalkCounters()
    pending: List[Tuple[int, int]] = [(root_id, 0)]
    while pending:
        node_id, depth = pending.pop()
        assert node_id not in counters.depths, (
            f"node {node_id} reached twice at depths {counters.depths[node_id]} and {depth}: "
            "cycle or repeated child edge"
        )
        counters.depths[node_id] = depth
        counters.max_edge_depth = max(counters.max_edge_depth, depth)

        kind = tree.kind[node_id]
        assert kind in (BELIEF, ACTION), f"node {node_id} has unknown kind {kind!r}"

        visits = tree.visit_count[node_id]
        assert isinstance(visits, int) and visits >= 0, (
            f"node {node_id} (kind={kind}, parent={tree.parent_id[node_id]}) has "
            f"visit_count {visits!r}; expected a non-negative int"
        )
        assert math.isfinite(
            tree.q_value[node_id]
        ), f"node {node_id} (kind={kind}) has non-finite q_value {tree.q_value[node_id]}"
        assert math.isfinite(
            tree.v_value[node_id]
        ), f"node {node_id} (kind={kind}) has non-finite v_value {tree.v_value[node_id]}"

        children = tree.children_ids[node_id]
        assert len(children) == len(
            set(children)
        ), f"node {node_id} lists a duplicate child edge: {children}"

        if kind == BELIEF:
            counters.belief_nodes += 1
            assert tree.belief[node_id] is not None, f"belief node {node_id} has no belief payload"
            if node_id != root_id:
                counters.non_root_belief_nodes += 1
                if children:
                    counters.expanded_non_root_belief_nodes += 1
            expected_child_kind = ACTION
        else:
            counters.action_nodes += 1
            assert tree.action[node_id] is not None, f"action node {node_id} has no action payload"
            if visits > 0:
                counters.visited_action_nodes += 1
            else:
                counters.unvisited_action_nodes += 1
            if len(children) > 1:
                counters.multi_child_action_nodes += 1
            expected_child_kind = BELIEF

        _check_children_cdf(tree, node_id, weight_tolerance)

        for child_id in children:
            assert 0 <= child_id < size, f"node {node_id} lists out-of-range child {child_id}"
            assert tree.parent_id[child_id] == node_id, (
                f"child {child_id} of {node_id} has parent_id {tree.parent_id[child_id]}; "
                "reverse link is broken"
            )
            assert tree.kind[child_id] == expected_child_kind, (
                f"child {child_id} of {node_id} (kind={kind}) has kind {tree.kind[child_id]}; "
                "belief and action nodes must alternate"
            )
            assert tree.position_in_parent[child_id] == children.index(child_id), (
                f"child {child_id} of {node_id} caches position "
                f"{tree.position_in_parent[child_id]} but sits at index {children.index(child_id)}"
            )
            if tree.kind[child_id] == BELIEF and tree.weight[child_id] > 1.0 + weight_tolerance:
                counters.reused_belief_children += 1
            pending.append((child_id, depth + 1))

    if expect_all_nodes_reachable:
        reached = set(counters.depths)
        expected: Set[int] = set(range(size))
        assert reached == expected, (
            f"unreachable logical nodes {sorted(expected - reached)} "
            f"(reached {len(reached)} of {size})"
        )
    return counters


def _check_children_cdf(tree: Tree, parent_id: int, tolerance: float) -> None:
    children = tree.children_ids[parent_id]
    cdf = tree.children_cdf[parent_id]
    assert len(cdf) == len(
        children
    ), f"node {parent_id} has {len(children)} children but a CDF of length {len(cdf)}"
    running = 0.0
    for index, child_id in enumerate(children):
        weight = tree.weight[child_id] if tree.kind[child_id] == BELIEF else 1.0
        assert (
            math.isfinite(weight) and weight >= 0.0
        ), f"child {child_id} of {parent_id} has weight {weight}; expected finite and >= 0"
        running += weight
        assert (
            cdf[index] >= (cdf[index - 1] if index else 0.0) - tolerance
        ), f"node {parent_id} CDF is not monotone at index {index}: {cdf}"
        assert abs(cdf[index] - running) <= tolerance, (
            f"node {parent_id} CDF entry {index} is {cdf[index]} but the running sum of child "
            f"weights is {running}; the CDF is stale relative to weight[{child_id}]"
        )


def discounted_horizon_bounds(
    reward_min: float,
    reward_max: float,
    discount: float,
    horizon: int,
    leaf_min: float = 0.0,
    leaf_max: float = 0.0,
) -> Tuple[float, float]:
    """Bounds on a discounted return with ``horizon`` reward terms plus a leaf.

    ``lower = r_min * sum_{t<h} g^t + g^h * L`` and the matching upper bound.
    The finite sum is used rather than ``1/(1-g)`` so a discount of exactly one
    and a zero horizon both come out right. Derived from the fixture's declared
    reward range, never from values observed in the tree under test.
    """
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    if not 0.0 <= discount <= 1.0:
        raise ValueError(f"discount must be in [0, 1], got {discount}")
    geometric = sum(discount**t for t in range(horizon))
    tail = discount**horizon
    return (reward_min * geometric + tail * leaf_min, reward_max * geometric + tail * leaf_max)


def assert_values_within_bounds(
    tree: Tree,
    root_id: int,
    *,
    horizon_of: Callable[[int, int], Optional[int]],
    reward_min: float,
    reward_max: float,
    discount: float,
    leaf_min: float = 0.0,
    leaf_max: float = 0.0,
    tolerance: float = 1e-9,
    skip_unvisited_actions: bool = True,
) -> int:
    """Check every live Q and V against an independently derived interval.

    Args:
        tree: The tree to check.
        root_id: Its root.
        horizon_of: ``(node_id, edge_depth) -> remaining reward terms``, or
            ``None`` to exclude the node. The caller maps the planner's depth
            bookkeeping onto a reward-term count; raw edge depth is not it,
            because one planning transition spans two edges.
        reward_min, reward_max: The fixture's proven per-step reward bounds.
        discount: The planner's discount.
        leaf_min, leaf_max: Bounds on whatever estimate sits past the horizon —
            zero for a truncated rollout, the network's proven output range for
            a learned leaf.
        tolerance: Absolute slack, to absorb float accumulation only.
        skip_unvisited_actions: Skip action nodes with zero visits. Their
            ``q_value`` is still the allocation-time ``0.0`` sentinel, not an
            estimate; a separate test pins that sentinel.

    Returns:
        The number of values actually checked, so the caller can assert the
        check was not vacuous.
    """
    counters = WalkCounters()
    pending: List[Tuple[int, int]] = [(root_id, 0)]
    checked = 0
    while pending:
        node_id, depth = pending.pop()
        counters.depths[node_id] = depth
        horizon = horizon_of(node_id, depth)
        if horizon is not None:
            is_action = tree.kind[node_id] == ACTION
            if not (is_action and skip_unvisited_actions and tree.visit_count[node_id] == 0):
                value = tree.q_value[node_id] if is_action else tree.v_value[node_id]
                low, high = discounted_horizon_bounds(
                    reward_min, reward_max, discount, horizon, leaf_min, leaf_max
                )
                assert low - tolerance <= value <= high + tolerance, (
                    f"node {node_id} "
                    f"(kind={'ACTION' if is_action else 'BELIEF'}, "
                    f"parent={tree.parent_id[node_id]}, edge_depth={depth}, "
                    f"visits={tree.visit_count[node_id]}, remaining_horizon={horizon}) "
                    f"has value {value!r} outside [{low}, {high}]"
                )
                checked += 1
        for child_id in tree.children_ids[node_id]:
            pending.append((child_id, depth + 1))
    return checked


def action_ids(tree: Tree, belief_id: int) -> List[int]:
    """The action children of ``belief_id``, in allocation order."""
    return [cid for cid in tree.children_ids[belief_id] if tree.kind[cid] == ACTION]


def belief_ids(tree: Tree, action_id: int) -> List[int]:
    """The belief children of ``action_id``, in allocation order."""
    return [cid for cid in tree.children_ids[action_id] if tree.kind[cid] == BELIEF]


def visit_entropy(visit_counts: Sequence[int]) -> float:
    """``-sum(p log2 p)`` over positive proportions; zero when nothing is visited.

    Written out here rather than imported so the tree-metrics tests never check
    the production helper against itself.
    """
    total = float(sum(visit_counts))
    if total <= 0.0:
        return 0.0
    result = 0.0
    for count in visit_counts:
        if count > 0:
            p = count / total
            result -= p * math.log2(p)
    return result


def running_mean(previous_mean: float, previous_count: int, new_sample: float) -> float:
    """``(n*q + g)/(n+1)`` — the incremental sample mean, written out by hand."""
    return (previous_count * previous_mean + new_sample) / (previous_count + 1)


def snapshot_subtree(tree: Tree, node_id: int) -> Dict[int, Dict[str, Any]]:
    """Record the mutable per-node fields of a subtree, for isolation checks.

    Belief payloads are recorded by their particle list where they expose one,
    so accidental aliasing between two branches' beliefs is detectable and not
    just their scalar statistics.
    """
    result: Dict[int, Dict[str, Any]] = {}
    pending = [node_id]
    while pending:
        current = pending.pop()
        belief = tree.belief[current]
        result[current] = {
            "visit_count": tree.visit_count[current],
            "q_value": tree.q_value[current],
            "v_value": tree.v_value[current],
            "weight": tree.weight[current],
            "immediate_reward": tree.immediate_reward[current],
            "immediate_cost": tree.immediate_cost[current],
            "children": list(tree.children_ids[current]),
            "children_cdf": list(tree.children_cdf[current]),
            "particles": list(getattr(belief, "particles", []) or []),
        }
        pending.extend(tree.children_ids[current])
    return result


def assert_subtree_unchanged(
    tree: Tree, snapshot: Dict[int, Dict[str, Any]], label: str = "sibling"
) -> None:
    """Assert every field recorded by :func:`snapshot_subtree` is still equal."""
    current = snapshot_subtree(tree, min(snapshot))
    assert set(current) == set(
        snapshot
    ), f"{label} subtree changed shape: was {sorted(snapshot)}, now {sorted(current)}"
    for node_id, before in snapshot.items():
        after = current[node_id]
        for key, old in before.items():
            assert after[key] == old, (
                f"{label} node {node_id} field {key!r} changed from {old!r} to {after[key]!r}; "
                "a simulation down another branch must not touch it"
            )
