# SPDX-License-Identifier: MIT

"""PUCT action selection and progressive widening for BetaZero.

Implements the Predictor Upper Confidence Trees (PUCT) selection rule used
in BetaZero, replacing the standard UCB1 criterion. PUCT biases exploration
towards actions favoured by the policy network.

Reference:
    Moss, R. J., Corso, A., Caers, J., & Kochenderfer, M. J. (2024). BetaZero:
    Belief-State Planning for Long-Horizon POMDPs using Learned Approximations.
    Reinforcement Learning Conference (RLC).

Functions:
    puct_selection: Select among existing children using PUCT.
    puct_action_progressive_widening: Progressive widening with PUCT selection.
    puct_selection_arena: Arena-tree variant of puct_selection.
    puct_action_progressive_widening_arena: Arena variant of widening+PUCT.
"""

from typing import Optional

import numpy as np

from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


def puct_selection(
    belief_node: BeliefNode,
    exploration_constant: float,
    action_priors: Optional[np.ndarray] = None,
) -> ActionNode:
    """Select an action child using the PUCT criterion.

    The selection rule is:

        a* = argmax  Q̄(b,a) + c · P(a|b) · √N(b) / (1 + N(b,a))

    where Q-values are normalised to [0, 1] for problem-independent exploration.

    Args:
        belief_node: Current belief node with at least one action child.
        exploration_constant: Exploration constant *c*.
        action_priors: Prior probabilities P(a|b) aligned with
            ``belief_node.children``. If ``None``, uniform priors are used.

    Returns:
        The action node with the highest PUCT score.
    """
    children = belief_node.children
    n_children = len(children)

    if action_priors is None:
        priors = np.ones(n_children) / n_children
    else:
        priors = action_priors

    q_values = np.array([child.q_value for child in children])
    visit_counts = np.array([child.visit_count for child in children])

    q_normalized = _normalize_q_values(q_values)

    parent_visits = max(belief_node.visit_count, 1)
    exploration = exploration_constant * priors * np.sqrt(parent_visits) / (1.0 + visit_counts)

    puct_scores = q_normalized + exploration
    return children[int(np.argmax(puct_scores))]


def puct_action_progressive_widening(
    belief_node: BeliefNode,
    alpha_a: float,
    action_sampler: ActionSampler,
    exploration_constant: float,
    k_a: float,
    action_priors: Optional[np.ndarray] = None,
    min_visit_count_per_action: int = 1,
) -> ActionNode:
    """Progressive widening with PUCT selection instead of UCB1.

    Follows the same widening logic as the standard
    ``action_progressive_widening`` but selects among existing actions
    using :func:`puct_selection` with neural network priors.

    Args:
        belief_node: Current belief node.
        alpha_a: Progressive widening exponent (0 < α_a ≤ 1).
        action_sampler: Sampler for generating new candidate actions.
        exploration_constant: PUCT exploration constant *c*.
        k_a: Progressive widening coefficient.
        action_priors: Prior probabilities for existing children. If ``None``,
            uniform priors are used.
        min_visit_count_per_action: At the root, ensure every child has been
            visited at least this many times before selecting via PUCT.

    Returns:
        Selected or newly created action node.
    """
    if belief_node.depth == 0:
        for action_node in belief_node.children:
            if action_node.visit_count < min_visit_count_per_action:
                return action_node

    if _should_widen(belief_node, k_a, alpha_a):
        action = action_sampler.sample()
        action_node = belief_node.get_child(action=action)
        if action_node is None:
            action_node = ActionNode(action=action, parent=belief_node)
        return action_node

    return puct_selection(
        belief_node=belief_node,
        exploration_constant=exploration_constant,
        action_priors=action_priors,
    )


def _should_widen(belief_node: BeliefNode, k_a: float, alpha_a: float) -> bool:
    return (
        belief_node.is_leaf
        or belief_node.visit_count == 0
        or len(belief_node.children) <= k_a * belief_node.visit_count**alpha_a
    )


def _normalize_q_values(q_values: np.ndarray) -> np.ndarray:
    q_min = q_values.min()
    q_max = q_values.max()
    if q_max - q_min < 1e-8:
        return np.full_like(q_values, 0.5)
    return (q_values - q_min) / (q_max - q_min)


def puct_selection_arena(
    tree: Tree,
    belief_id: int,
    exploration_constant: float,
    action_priors: Optional[np.ndarray] = None,
) -> int:
    """Arena variant of :func:`puct_selection`. Returns the action-node ID."""
    children = tree.children_ids[belief_id]
    n_children = len(children)

    if action_priors is None:
        priors = np.ones(n_children) / n_children
    else:
        priors = action_priors

    q_values = np.array([tree.q_value[cid] for cid in children])
    visit_counts = np.array([tree.visit_count[cid] for cid in children])

    q_normalized = _normalize_q_values(q_values)

    parent_visits = max(tree.visit_count[belief_id], 1)
    exploration = exploration_constant * priors * np.sqrt(parent_visits) / (1.0 + visit_counts)

    puct_scores = q_normalized + exploration
    return children[int(np.argmax(puct_scores))]


def puct_action_progressive_widening_arena(  # pylint: disable=too-many-arguments
    tree: Tree,
    belief_id: int,
    alpha_a: float,
    action_sampler: ActionSampler,
    exploration_constant: float,
    k_a: float,
    action_priors: Optional[np.ndarray] = None,
    min_visit_count_per_action: int = 1,
) -> int:
    """Arena variant of :func:`puct_action_progressive_widening`.

    Returns the action-node ID selected by progressive widening + PUCT.
    """
    if tree.parent_id[belief_id] is None:
        for cid in tree.children_ids[belief_id]:
            if tree.visit_count[cid] < min_visit_count_per_action:
                return cid

    if _should_widen_arena(tree, belief_id, k_a, alpha_a):
        action = action_sampler.sample()
        existing_id = tree.get_action_child_indexed(belief_id, action)
        if existing_id is not None:
            return existing_id
        existing_id = tree.get_action_child(belief_id, action)
        if existing_id is not None:
            return existing_id
        return tree.add_action_node(action=action, parent_id=belief_id)

    return puct_selection_arena(
        tree=tree,
        belief_id=belief_id,
        exploration_constant=exploration_constant,
        action_priors=action_priors,
    )


def _should_widen_arena(tree: Tree, belief_id: int, k_a: float, alpha_a: float) -> bool:
    children = tree.children_ids[belief_id]
    belief_visits = tree.visit_count[belief_id]
    return len(children) == 0 or belief_visits == 0 or len(children) <= k_a * belief_visits**alpha_a
