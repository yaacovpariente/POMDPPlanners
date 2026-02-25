from POMDPPlanners.planners.planners_utils.dpw import ActionSampler, ActionNode
from POMDPPlanners.core.tree import BeliefNode
from POMDPPlanners.planners.planners_utils.cvar_exploration import get_explored_action_node


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
