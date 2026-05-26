# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

from typing import Optional

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.planners_utils.cvar_exploration import (
    get_explored_action_node,
    get_explored_action_node_arena,
)
from POMDPPlanners.planners.planners_utils.dpw import (
    ActionNode,
    ActionSampler,
    _get_or_add_action_child,
)


def cvar_action_progressive_widening(
    belief_node: BeliefNode,
    alpha_a: float,
    action_sampler: ActionSampler,
    exploration_constant: float,
    k_a: float,
    min_immediate_cost: float,
    max_immediate_cost: float,
    depth: int,
    max_depth: int,
    gamma: float,
    min_visit_count_per_action: int,
    alpha: float,
    delta: float,
    discrete_actions: bool = False,  # pylint: disable=unused-argument
    visit_count_penalty: float = 0.0,
) -> ActionNode:
    if (
        belief_node.is_leaf
        or belief_node.visit_count == 0
        or len(belief_node.children) <= k_a * belief_node.visit_count**alpha_a
    ):
        action = action_sampler.sample(belief_node=belief_node)
        action_node = belief_node.get_child(action=action)
        if action_node is None:
            action_node = ActionNode(action=action, parent=belief_node)

        return action_node

    return get_explored_action_node(
        belief_node=belief_node,
        min_immediate_cost=min_immediate_cost,
        max_immediate_cost=max_immediate_cost,
        depth=depth,
        max_depth=max_depth,
        gamma=gamma,
        exploration_constant=exploration_constant,
        min_visit_count_per_action=min_visit_count_per_action,
        alpha=alpha,
        delta=delta,
        visit_count_penalty=visit_count_penalty,
    )


def cvar_action_progressive_widening_arena(  # pylint: disable=too-many-arguments
    tree: Tree,
    belief_id: int,
    alpha_a: float,
    action_sampler: ActionSampler,
    exploration_constant: float,
    k_a: float,
    min_immediate_cost: float,
    max_immediate_cost: float,
    depth: int,
    max_depth: int,
    gamma: float,
    min_visit_count_per_action: int,
    alpha: float,
    delta: float,
    discrete_actions: bool = False,  # pylint: disable=unused-argument
    visit_count_penalty: float = 0.0,
    environment: Optional[Environment] = None,
) -> int:
    """Arena variant of :func:`cvar_action_progressive_widening`.

    Returns the action-node ID selected by CVaR-aware progressive widening
    from belief node ``belief_id`` in ``tree``.

    When ``environment`` is supplied, ``environment.hash_action`` is used to
    index action children in O(1). When omitted, falls back to the legacy
    linear-scan path for backwards compatibility.
    """
    children = tree.children_ids[belief_id]
    belief_visits = tree.visit_count[belief_id]
    is_leaf = len(children) == 0

    if is_leaf or belief_visits == 0 or len(children) <= k_a * belief_visits**alpha_a:
        action = action_sampler.sample()
        return _get_or_add_action_child(tree, belief_id, action, environment)

    return get_explored_action_node_arena(
        tree=tree,
        belief_id=belief_id,
        min_immediate_cost=min_immediate_cost,
        max_immediate_cost=max_immediate_cost,
        depth=depth,
        max_depth=max_depth,
        gamma=gamma,
        exploration_constant=exploration_constant,
        min_visit_count_per_action=min_visit_count_per_action,
        alpha=alpha,
        delta=delta,
        visit_count_penalty=visit_count_penalty,
    )
