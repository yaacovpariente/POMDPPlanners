# SPDX-License-Identifier: MIT

"""Safety-constrained PUCT (SPUCT) action selection for ConstrainedZero.

Implements the SPUCT selection rule that masks unsafe actions based on their
estimated failure probability relative to an adaptive threshold Delta'.

References:
    Moss, R. J., Jamgochian, A., Fischer, J., Corso, A., & Kochenderfer, M. J. (2024).
    ConstrainedZero: Chance-Constrained POMDP Planning Using Learned Probabilistic Failure
    Surrogates and Adaptive Safety Constraints. Proceedings of the Thirty-Third International
    Joint Conference on Artificial Intelligence (IJCAI), 6752-6760. arXiv:2405.00644.
    https://arxiv.org/abs/2405.00644

Functions:
    spuct_selection: Select among existing children using safety-masked PUCT.
    spuct_action_progressive_widening: Progressive widening with SPUCT selection.
    spuct_selection_arena: Arena-tree variant of spuct_selection.
    spuct_action_progressive_widening_arena: Arena variant of widening+SPUCT.
"""

from typing import Dict, Optional

import numpy as np

from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.core.tree.arena import Tree
from POMDPPlanners.planners.mcts_planners.beta_zero.puct import (
    _normalize_q_values,
    _should_widen,
    _should_widen_arena,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


def spuct_selection(
    belief_node: BeliefNode,
    exploration_constant: float,
    failure_dict: Dict[int, float],
    delta_prime: float,
    action_priors: Optional[np.ndarray] = None,
) -> ActionNode:
    """Select an action child using the safety-constrained PUCT criterion.

    The selection rule is:

        a* = argmax  subject_to(a) * [Q_norm(b,a) + c * P(a|b) * sqrt(N(b)) / (1 + N(b,a))]

    where ``subject_to(a) = I(f(a) <= Delta')`` masks unsafe actions. If ALL
    actions are unsafe, falls back to unconstrained selection.

    Args:
        belief_node: Current belief node with at least one action child.
        exploration_constant: Exploration constant *c*.
        failure_dict: Maps ``id(action_node)`` to estimated failure probability.
        delta_prime: Adaptive failure threshold.
        action_priors: Prior probabilities P(a|b) aligned with
            ``belief_node.children``. If ``None``, uniform priors are used.

    Returns:
        The action node with the highest safety-masked PUCT score.
    """
    children = belief_node.children
    n_children = len(children)

    if action_priors is None:
        priors = np.ones(n_children) / n_children
    else:
        priors = action_priors

    q_values = np.array([child.q_value for child in children])
    visit_counts = np.array([child.visit_count for child in children])
    failure_probs = np.array([failure_dict.get(id(child), 0.0) for child in children])

    q_normalized = _normalize_q_values(q_values)

    parent_visits = max(belief_node.visit_count, 1)
    exploration = exploration_constant * priors * np.sqrt(parent_visits) / (1.0 + visit_counts)

    puct_scores = q_normalized + exploration
    safety_mask = _compute_safety_mask(failure_probs, delta_prime)
    masked_scores = safety_mask * puct_scores

    return children[int(np.argmax(masked_scores))]


def spuct_action_progressive_widening(
    belief_node: BeliefNode,
    alpha_a: float,
    action_sampler: ActionSampler,
    exploration_constant: float,
    k_a: float,
    failure_dict: Dict[int, float],
    delta_prime: float,
    action_priors: Optional[np.ndarray] = None,
    min_visit_count_per_action: int = 1,
) -> ActionNode:
    """Progressive widening with SPUCT selection instead of PUCT.

    Args:
        belief_node: Current belief node.
        alpha_a: Progressive widening exponent (0 < alpha_a <= 1).
        action_sampler: Sampler for generating new candidate actions.
        exploration_constant: PUCT exploration constant *c*.
        k_a: Progressive widening coefficient.
        failure_dict: Maps ``id(action_node)`` to estimated failure probability.
        delta_prime: Adaptive failure threshold.
        action_priors: Prior probabilities for existing children.
        min_visit_count_per_action: At the root, ensure every child has been
            visited at least this many times before selecting via SPUCT.

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

    return spuct_selection(
        belief_node=belief_node,
        exploration_constant=exploration_constant,
        failure_dict=failure_dict,
        delta_prime=delta_prime,
        action_priors=action_priors,
    )


def _compute_safety_mask(failure_probs: np.ndarray, delta_prime: float) -> np.ndarray:
    """Compute safety mask: 1 for safe actions, 0 for unsafe.

    If ALL actions are unsafe, returns all ones (fallback to unconstrained).
    """
    mask = (failure_probs <= delta_prime).astype(np.float64)
    if mask.sum() == 0:
        return np.ones_like(mask)
    return mask


def spuct_selection_arena(
    tree: Tree,
    belief_id: int,
    exploration_constant: float,
    failure_dict: Dict[int, float],
    delta_prime: float,
    action_priors: Optional[np.ndarray] = None,
) -> int:
    """Arena variant of :func:`spuct_selection`. Returns the action-node ID.

    ``failure_dict`` is keyed by action-node integer ID (not ``id(node)``).
    """
    children = tree.children_ids[belief_id]
    n_children = len(children)

    if action_priors is None:
        priors = np.ones(n_children) / n_children
    else:
        priors = action_priors

    q_values = np.array([tree.q_value[cid] for cid in children])
    visit_counts = np.array([tree.visit_count[cid] for cid in children])
    failure_probs = np.array([failure_dict.get(cid, 0.0) for cid in children])

    q_normalized = _normalize_q_values(q_values)

    parent_visits = max(tree.visit_count[belief_id], 1)
    exploration = exploration_constant * priors * np.sqrt(parent_visits) / (1.0 + visit_counts)

    puct_scores = q_normalized + exploration
    safety_mask = _compute_safety_mask(failure_probs, delta_prime)
    masked_scores = safety_mask * puct_scores

    return children[int(np.argmax(masked_scores))]


def spuct_action_progressive_widening_arena(  # pylint: disable=too-many-arguments
    tree: Tree,
    belief_id: int,
    alpha_a: float,
    action_sampler: ActionSampler,
    exploration_constant: float,
    k_a: float,
    failure_dict: Dict[int, float],
    delta_prime: float,
    action_priors: Optional[np.ndarray] = None,
    min_visit_count_per_action: int = 1,
) -> int:
    """Arena variant of :func:`spuct_action_progressive_widening`.

    Returns the action-node ID selected by progressive widening + SPUCT.
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

    return spuct_selection_arena(
        tree=tree,
        belief_id=belief_id,
        exploration_constant=exploration_constant,
        failure_dict=failure_dict,
        delta_prime=delta_prime,
        action_priors=action_priors,
    )
