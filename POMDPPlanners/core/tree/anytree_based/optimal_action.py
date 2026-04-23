from typing import Any

import numpy as np

from POMDPPlanners.core.tree.anytree_based.belief_node import BeliefNode


def get_optimal_action_cost_setting(belief_node: BeliefNode) -> Any:
    actions = [child.action for child in belief_node.children]
    q_values = [child.q_value for child in belief_node.children]
    return actions[np.argmin(q_values)]


def get_optimal_action_reward_setting(belief_node: BeliefNode) -> Any:
    actions = [child.action for child in belief_node.children]
    q_values = [child.q_value for child in belief_node.children]
    return actions[np.argmax(q_values)]
