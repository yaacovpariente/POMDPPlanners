import numpy as np

from POMDPPlanners.utils.statistics_utils import get_min_and_max_cost
from POMDPPlanners.core.tree import BeliefNode, ActionNode


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
    unvisited_action_indices = np.where(visit_count_array >= 1)[0]
    if len(unvisited_action_indices) > 0:
        return belief_node.children[np.random.choice(unvisited_action_indices)]

    visit_count_penalty_array = 1 / (np.sqrt(visit_count_array) + 1)

    x1 = 1 - belief_node.visit_count**horizon
    x2 = delta * (1 - belief_node.visit_count)
    x3 = np.log(x1 / x2)
    x4 = alpha * visit_count_array

    guarantees_bound = (max_cost - min_cost) * np.sqrt(x3 / x4)

    lower_confidence_bounds = (
        np.array([child.q_value for child in belief_node.children])
        - exploration_constant * guarantees_bound
    )
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
