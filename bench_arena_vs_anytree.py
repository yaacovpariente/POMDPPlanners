"""A/B benchmark: column-store Tree vs legacy anytree-based tree.

Same operations as a real MCTS planner performs on the tree. The arena
side now has no node objects — operations go directly through the
``Tree`` columns and methods.

Run:
    python bench_arena_vs_anytree.py
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, List, Tuple

import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.environment import Environment, SpaceInfo, SpaceType

# Legacy (anytree-based)
from POMDPPlanners.core.tree import ActionNode as LegacyActionNode
from POMDPPlanners.core.tree import BeliefNode as LegacyBeliefNode

# Arena (column-store, no node objects)
from POMDPPlanners.core.tree.arena import Tree


BELIEF = WeightedParticleBelief(particles=[1, 2], log_weights=np.log(np.array([0.6, 0.4])))


class _StubEnv(Environment):
    """Minimal env exposing only what get_belief_child needs."""

    def __init__(self) -> None:
        space_info = SpaceInfo(SpaceType.DISCRETE, SpaceType.DISCRETE)
        super().__init__(discount_factor=0.95, name="stub", space_info=space_info)

    def is_equal_observation(self, observation1, observation2):
        return observation1 == observation2

    def state_transition(self, state, action):
        del action
        return state

    def observation_model(self, next_state, action):  # pragma: no cover
        del next_state, action
        raise NotImplementedError

    def is_terminal(self, state):  # pragma: no cover
        del state
        return False

    def reward(self, state, action):  # pragma: no cover
        del state, action
        return 0.0

    def initial_state_dist(self):  # pragma: no cover
        raise NotImplementedError

    def initial_observation_dist(self):  # pragma: no cover
        raise NotImplementedError

    def state_transition_model(self, state, action):  # pragma: no cover
        del state, action
        raise NotImplementedError


ENV = _StubEnv()


def _median_time(fn: Callable[[], Any], repeats: int) -> float:
    samples: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


# ---------------- legacy helpers ----------------


def _legacy_build(depth: int, branching: int):
    root = LegacyBeliefNode(BELIEF)
    frontier = [root]
    for layer in range(depth):
        next_frontier = []
        is_belief_layer = layer % 2 == 0
        for parent in frontier:
            for i in range(branching):
                if is_belief_layer:
                    child = LegacyActionNode(f"a{layer}_{i}", parent=parent)
                else:
                    child = LegacyBeliefNode(BELIEF, observation=f"o{layer}_{i}", parent=parent)
                next_frontier.append(child)
        frontier = next_frontier
    return root


def _legacy_walk_visits(node) -> int:
    total = node.visit_count
    for child in node.children:
        total += _legacy_walk_visits(child)
    return total


def _legacy_walk_increment(node) -> None:
    node.visit_count = node.visit_count + 1
    for child in node.children:
        _legacy_walk_increment(child)


def _legacy_make_action_with_children(k: int):
    root = LegacyBeliefNode(BELIEF)
    action = LegacyActionNode("a", parent=root)
    for i in range(k):
        LegacyBeliefNode(BELIEF, observation=f"o{i}", parent=action)
    return action


def _legacy_make_belief_with_action_children(k: int):
    root = LegacyBeliefNode(BELIEF)
    for i in range(k):
        LegacyActionNode(f"a{i}", parent=root)
    return root


# ---------------- arena helpers ----------------


def _arena_build(depth: int, branching: int) -> Tuple[Tree, int]:
    tree = Tree()
    root = tree.add_belief_node(BELIEF)
    frontier = [root]
    for layer in range(depth):
        next_frontier: List[int] = []
        is_belief_layer = layer % 2 == 0
        for parent in frontier:
            for i in range(branching):
                if is_belief_layer:
                    cid = tree.add_action_node(f"a{layer}_{i}", parent_id=parent)
                else:
                    cid = tree.add_belief_node(
                        BELIEF, observation=f"o{layer}_{i}", parent_id=parent
                    )
                next_frontier.append(cid)
        frontier = next_frontier
    return tree, root


def _arena_walk_visits(tree: Tree, node_id: int) -> int:
    total = tree.visit_count[node_id]
    for cid in tree.children_ids[node_id]:
        total += _arena_walk_visits(tree, cid)
    return total


def _arena_walk_increment(tree: Tree, node_id: int) -> None:
    tree.visit_count[node_id] = tree.visit_count[node_id] + 1
    for cid in tree.children_ids[node_id]:
        _arena_walk_increment(tree, cid)


def _arena_make_action_with_children(k: int) -> Tuple[Tree, int]:
    tree = Tree()
    root = tree.add_belief_node(BELIEF)
    action = tree.add_action_node("a", parent_id=root)
    for i in range(k):
        tree.add_belief_node(BELIEF, observation=f"o{i}", parent_id=action)
    return tree, action


def _arena_make_belief_with_action_children(k: int) -> Tuple[Tree, int]:
    tree = Tree()
    root = tree.add_belief_node(BELIEF)
    for i in range(k):
        tree.add_action_node(f"a{i}", parent_id=root)
    return tree, root


# ---------------- formatting ----------------


def _ratio(arena_t: float, legacy_t: float) -> str:
    if legacy_t == 0:
        return "—"
    r = arena_t / legacy_t
    label = "faster" if r < 1.0 else "slower"
    return f"{r:5.2f}x {label}"


def _format_us(seconds: float) -> str:
    return f"{seconds * 1e6:>10.2f} us"


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    repeats = 7

    print("=" * 88)
    print(" A/B Benchmark: column-store Tree (no node objects) vs legacy anytree-based tree")
    print(" " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 88)
    print()

    # ---------------- 1. build cost ----------------
    print("[1] Build cost — alternating belief/action tree, branching=2")
    print(f"{'depth':<6} {'nodes':<8} {'legacy':<14} {'arena':<14} {'ratio'}")
    print("-" * 70)
    for depth in (8, 12, 14, 16):
        n_nodes = 2 ** (depth + 1) - 1
        legacy_t = _median_time(lambda d=depth: _legacy_build(d, 2), repeats)
        arena_t = _median_time(lambda d=depth: _arena_build(d, 2), repeats)
        print(
            f"{depth:<6} {n_nodes:<8} {_format_us(legacy_t):<14} "
            f"{_format_us(arena_t):<14} {_ratio(arena_t, legacy_t)}"
        )
    print()

    # ---------------- 2. full-tree DFS read ----------------
    print("[2] Full DFS — sum visit_count across all nodes")
    print(f"{'depth':<6} {'nodes':<8} {'legacy':<14} {'arena':<14} {'ratio'}")
    print("-" * 70)
    for depth in (8, 12, 14, 16):
        n_nodes = 2 ** (depth + 1) - 1
        legacy_root = _legacy_build(depth, 2)
        arena_tree, arena_root = _arena_build(depth, 2)
        legacy_t = _median_time(lambda r=legacy_root: _legacy_walk_visits(r), repeats)
        arena_t = _median_time(lambda t=arena_tree, r=arena_root: _arena_walk_visits(t, r), repeats)
        print(
            f"{depth:<6} {n_nodes:<8} {_format_us(legacy_t):<14} "
            f"{_format_us(arena_t):<14} {_ratio(arena_t, legacy_t)}"
        )
    print()

    # ---------------- 3. full-tree DFS write ----------------
    print("[3] Full DFS — increment visit_count on every node")
    print(f"{'depth':<6} {'nodes':<8} {'legacy':<14} {'arena':<14} {'ratio'}")
    print("-" * 70)
    for depth in (8, 12, 14, 16):
        n_nodes = 2 ** (depth + 1) - 1
        legacy_root = _legacy_build(depth, 2)
        arena_tree, arena_root = _arena_build(depth, 2)
        legacy_t = _median_time(lambda r=legacy_root: _legacy_walk_increment(r), repeats)
        arena_t = _median_time(
            lambda t=arena_tree, r=arena_root: _arena_walk_increment(t, r), repeats
        )
        print(
            f"{depth:<6} {n_nodes:<8} {_format_us(legacy_t):<14} "
            f"{_format_us(arena_t):<14} {_ratio(arena_t, legacy_t)}"
        )
    print()

    # ---------------- 4. get_belief_child (linear scan) ----------------
    print("[4] get_belief_child — K children, M lookups (M=100 for K=100000 to bound runtime)")
    print(f"{'K':<8} {'M':<8} {'legacy':<14} {'arena':<14} {'ratio'}")
    print("-" * 70)
    for k, m in ((10, 1000), (50, 1000), (200, 1000), (100000, 100)):
        legacy_action = _legacy_make_action_with_children(k)
        arena_tree, arena_action = _arena_make_action_with_children(k)
        observations = [f"o{i % (k * 2)}" for i in range(m)]
        legacy_t = _median_time(
            lambda a=legacy_action, obs=observations: [
                a.get_belief_node_child(o, ENV) for o in obs
            ],
            repeats,
        )
        arena_t = _median_time(
            lambda t=arena_tree, a=arena_action, obs=observations: [
                t.get_belief_child(a, o, ENV) for o in obs
            ],
            repeats,
        )
        print(
            f"{k:<8} {m:<8} {_format_us(legacy_t):<14} "
            f"{_format_us(arena_t):<14} {_ratio(arena_t, legacy_t)}"
        )
    print()

    # ---------------- 5. get_action_child (linear scan) ----------------
    print("[5] get_action_child — K children, M lookups (M=100 for K=100000 to bound runtime)")
    print(f"{'K':<8} {'M':<8} {'legacy':<14} {'arena':<14} {'ratio'}")
    print("-" * 70)
    for k, m in ((10, 1000), (50, 1000), (200, 1000), (100000, 100)):
        legacy_belief = _legacy_make_belief_with_action_children(k)
        arena_tree, arena_belief = _arena_make_belief_with_action_children(k)
        actions = [f"a{i % (k * 2)}" for i in range(m)]
        legacy_t = _median_time(
            lambda b=legacy_belief, acts=actions: [b.get_child(a) for a in acts],
            repeats,
        )
        arena_t = _median_time(
            lambda t=arena_tree, b=arena_belief, acts=actions: [
                t.get_action_child(b, a) for a in acts
            ],
            repeats,
        )
        print(
            f"{k:<8} {m:<8} {_format_us(legacy_t):<14} "
            f"{_format_us(arena_t):<14} {_ratio(arena_t, legacy_t)}"
        )
    print()

    # ---------------- 6. sample_belief_child (np.random.choice path) ----------------
    print("[6] sample_belief_child — K belief children, M samples (M=100 for K=100000)")
    print(f"{'K':<8} {'M':<8} {'legacy':<14} {'arena':<14} {'ratio'}")
    print("-" * 70)
    for k, m in ((10, 1000), (50, 1000), (200, 1000), (100000, 100)):
        legacy_action = _legacy_make_action_with_children(k)
        arena_tree, arena_action = _arena_make_action_with_children(k)
        legacy_t = _median_time(
            lambda a=legacy_action, m=m: [a.sample_child_node() for _ in range(m)],
            repeats,
        )
        arena_t = _median_time(
            lambda t=arena_tree, a=arena_action, m=m: [t.sample_belief_child(a) for _ in range(m)],
            repeats,
        )
        print(
            f"{k:<8} {m:<8} {_format_us(legacy_t):<14} "
            f"{_format_us(arena_t):<14} {_ratio(arena_t, legacy_t)}"
        )
    print()

    # ---------------- 7. indexed lookup vs linear-scan lookup ----------------
    print("[7] Indexed (O(1) dict) vs linear-scan (O(K)) lookup — both arena, M=1000")
    print(f"{'K':<8} {'M':<8} {'linear':<14} {'indexed':<14} {'ratio'}")
    print("-" * 70)
    for k, m in ((10, 1000), (50, 1000), (200, 1000), (10000, 1000), (100000, 1000)):
        arena_tree, arena_action = _arena_make_action_with_children(k)
        observations = [f"o{i % (k * 2)}" for i in range(m)]
        linear_t = _median_time(
            lambda t=arena_tree, a=arena_action, obs=observations: [
                t.get_belief_child(a, o, ENV) for o in obs
            ],
            repeats,
        )
        indexed_t = _median_time(
            lambda t=arena_tree, a=arena_action, obs=observations: [
                t.get_belief_child_indexed(a, o) for o in obs
            ],
            repeats,
        )
        print(
            f"{k:<8} {m:<8} {_format_us(linear_t):<14} "
            f"{_format_us(indexed_t):<14} {_ratio(indexed_t, linear_t)}"
        )
    print()

    print("=" * 88)
    print(" Note: ratios < 1.00 mean arena is faster; > 1.00 mean arena is slower.")
    print("=" * 88)


if __name__ == "__main__":
    main()
