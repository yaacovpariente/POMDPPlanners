# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""MCTS tree data structures.

Two implementations live here:

* :mod:`POMDPPlanners.core.tree.anytree_based` — node-object tree (each node
  is a Python object subclassing :class:`anytree.NodeMixin`). The original
  implementation, preserved for backward compatibility.
* :mod:`POMDPPlanners.core.tree.arena` — column-store SoA tree (``Tree``
  holds one list per node attribute, nodes are integer IDs). Faster on
  every measured tree-side operation.

The legacy classes (``ActionNode``, ``BeliefNode``, ``BaseNode``,
``print_tree``, ``get_optimal_action_*``) are re-exported from this
package so existing ``from POMDPPlanners.core.tree import ActionNode``
imports keep working.
"""

from POMDPPlanners.core.tree.anytree_based import (
    ActionNode,
    BaseNode,
    BeliefNode,
    get_optimal_action_cost_setting,
    get_optimal_action_reward_setting,
    print_tree,
)

__all__ = [
    "BaseNode",
    "ActionNode",
    "BeliefNode",
    "print_tree",
    "get_optimal_action_cost_setting",
    "get_optimal_action_reward_setting",
]
