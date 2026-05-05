import numpy as np

from POMDPPlanners.utils.numba_kernels import sparse_sampling_lcb_min_idx_kernel
from POMDPPlanners.utils.statistics_utils import get_min_and_max_cost
from POMDPPlanners.core.tree import BeliefNode, ActionNode
from POMDPPlanners.core.tree.arena import Tree


def _get_sparse_sampling_guarantees_exploration_v2(
    belief_node: BeliefNode,
    exploration_constant: float,
    alpha: float,
    min_cost: float,
    max_cost: float,
    horizon: int,
    delta: float,
    visit_count_penalty: float = 0.0,
) -> ActionNode:
    visit_count_array = np.array([child.visit_count for child in belief_node.children])
    # Bug-fix: the original predicate was `visit_count_array >= 1` with
    # variable name `unvisited_action_indices` — that always returned
    # visited children and skipped the LCB calculation entirely. Correct
    # semantic is "if any child is unvisited, explore it randomly".
    unvisited_action_indices = np.where(visit_count_array == 0)[0]
    if len(unvisited_action_indices) > 0:
        return belief_node.children[np.random.choice(unvisited_action_indices)]

    # Guard: the LCB formula below has `delta * (1 - belief_visits)` in the
    # denominator, which is 0 when belief_visits == 1 (the second visit to
    # this belief node). Fall back to random visited-child selection in
    # that edge case; LCB only becomes well-defined for belief_visits >= 2.
    if belief_node.visit_count <= 1:
        return belief_node.children[int(np.random.randint(len(belief_node.children)))]

    visit_count_penalty_array = 1 / (np.sqrt(visit_count_array) + 1)
    q_values = np.array([child.q_value for child in belief_node.children])

    if horizon == 0:
        # Remaining-horizon is zero, so the LCB bound below is
        # undefined (log(0) = -inf, sqrt of negative ⇒ NaN). Fall
        # back to greedy q-min with the visit-count tie-breaker.
        return belief_node.children[
            int(np.argmin(q_values + visit_count_penalty * visit_count_penalty_array))
        ]

    x1 = 1 - belief_node.visit_count**horizon
    x2 = delta * (1 - belief_node.visit_count)
    x3 = np.log(x1 / x2)
    x4 = alpha * visit_count_array

    guarantees_bound = (max_cost - min_cost) * np.sqrt(x3 / x4)

    lower_confidence_bounds = q_values - exploration_constant * guarantees_bound
    return belief_node.children[
        np.argmin(lower_confidence_bounds + visit_count_penalty * visit_count_penalty_array)
    ]


def get_explored_action_node(
    belief_node: BeliefNode,
    min_immediate_cost: float,
    max_immediate_cost: float,
    depth: int,
    max_depth: int,
    gamma: float,
    exploration_constant: float,
    min_visit_count_per_action: int,
    alpha: float,
    delta: float,
    visit_count_penalty: float = 0.0,
) -> ActionNode:
    assert isinstance(belief_node, BeliefNode), "belief_node must be an instance of BeliefNode"
    assert isinstance(min_immediate_cost, (int, float)), "min_immediate_cost must be a number"
    assert isinstance(max_immediate_cost, (int, float)), "max_immediate_cost must be a number"
    assert isinstance(depth, int), "depth must be an integer"
    assert isinstance(max_depth, int), "max_depth must be an integer"
    assert isinstance(gamma, (int, float)), "gamma must be a number"
    assert isinstance(exploration_constant, (int, float)), "exploration_constant must be a number"
    assert isinstance(
        min_visit_count_per_action, int
    ), "min_visit_count_per_action must be an integer"
    assert isinstance(alpha, (int, float)), "alpha must be a number"
    assert isinstance(delta, (int, float)), "delta must be a number"
    assert isinstance(visit_count_penalty, (int, float)), "visit_count_penalty must be a number"

    action_nodes_visits = np.array([child.visit_count for child in belief_node.children])
    unvisited_action_indices = np.where(action_nodes_visits == 0)[0]
    if len(unvisited_action_indices) > 0:
        return belief_node.children[np.random.choice(unvisited_action_indices)]

    if belief_node.depth == 0:
        for action_node in belief_node.children:
            if action_node.visit_count < min_visit_count_per_action:
                return action_node

    min_cost, max_cost = get_min_and_max_cost(
        min_immediate_cost=min_immediate_cost,
        max_immediate_cost=max_immediate_cost,
        depth=depth,
        max_depth=max_depth,
        gamma=gamma,
    )

    return _get_sparse_sampling_guarantees_exploration_v2(
        belief_node=belief_node,
        min_cost=min_cost,
        max_cost=max_cost,
        exploration_constant=exploration_constant,
        alpha=alpha,
        horizon=max_depth - depth,
        delta=delta,
        visit_count_penalty=visit_count_penalty,
    )


def _sparse_sampling_guarantees_exploration_v2_arena(
    tree: Tree,
    belief_id: int,
    exploration_constant: float,
    alpha: float,
    min_cost: float,
    max_cost: float,
    horizon: int,
    delta: float,
    visit_count_penalty: float = 0.0,
) -> int:
    """Arena variant of :func:`_get_sparse_sampling_guarantees_exploration_v2`."""
    children = tree.children_ids[belief_id]
    n_children = len(children)
    # Single Python loop fills both arrays — faster than two np.fromiter
    # calls over a Python-list source for small N (typical here).
    tree_visits = tree.visit_count
    tree_q = tree.q_value
    visit_count_array = np.empty(n_children, dtype=np.float64)
    q_values = np.empty(n_children, dtype=np.float64)
    unvisited_local: list = []
    for i, cid in enumerate(children):
        v = tree_visits[cid]
        visit_count_array[i] = v
        q_values[i] = tree_q[cid]
        if v == 0:
            unvisited_local.append(i)
    if unvisited_local:
        # Bug-fix: the original predicate was `visit_count_array >= 1` with
        # variable name `unvisited_indices` — that always returned visited
        # children and skipped the LCB calculation entirely. The intended
        # semantic is "if any child is unvisited, explore it randomly".
        return children[unvisited_local[int(np.random.randint(len(unvisited_local)))]]

    # Guard: the LCB formula below has `delta * (1 - belief_visits)` in the
    # denominator, which is 0 when belief_visits == 1 (the second visit to
    # this belief node). Fall back to random visited-child selection in
    # that edge case; LCB only becomes well-defined for belief_visits >= 2.
    belief_visits = tree.visit_count[belief_id]
    if belief_visits <= 1:
        return children[int(np.random.randint(n_children))]

    best_idx = sparse_sampling_lcb_min_idx_kernel(
        visit_count_array,
        q_values,
        float(belief_visits),
        int(horizon),
        float(alpha),
        float(delta),
        float(min_cost),
        float(max_cost),
        float(exploration_constant),
        float(visit_count_penalty),
    )
    return children[best_idx]


def get_explored_action_node_arena(  # pylint: disable=too-many-arguments
    tree: Tree,
    belief_id: int,
    min_immediate_cost: float,
    max_immediate_cost: float,
    depth: int,
    max_depth: int,
    gamma: float,
    exploration_constant: float,
    min_visit_count_per_action: int,
    alpha: float,
    delta: float,
    visit_count_penalty: float = 0.0,
) -> int:
    """Arena variant of :func:`get_explored_action_node`. Returns action-node ID."""
    children = tree.children_ids[belief_id]
    # Plain loop is faster than np.fromiter over a Python-list source for
    # the small N (action children) here. Builds the unvisited index set in
    # one pass; np.fromiter + np.where would do the same work in two.
    tree_visits = tree.visit_count
    unvisited_local: list = []
    for i, cid in enumerate(children):
        if tree_visits[cid] == 0:
            unvisited_local.append(i)
    if unvisited_local:
        return children[unvisited_local[int(np.random.randint(len(unvisited_local)))]]

    if tree.parent_id[belief_id] is None:
        for cid in children:
            if tree.visit_count[cid] < min_visit_count_per_action:
                return cid

    min_cost, max_cost = get_min_and_max_cost(
        min_immediate_cost=min_immediate_cost,
        max_immediate_cost=max_immediate_cost,
        depth=depth,
        max_depth=max_depth,
        gamma=gamma,
    )

    return _sparse_sampling_guarantees_exploration_v2_arena(
        tree=tree,
        belief_id=belief_id,
        min_cost=min_cost,
        max_cost=max_cost,
        exploration_constant=exploration_constant,
        alpha=alpha,
        horizon=max_depth - depth,
        delta=delta,
        visit_count_penalty=visit_count_penalty,
    )
