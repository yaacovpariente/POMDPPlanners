from typing import List

import numpy as np
from scipy.stats import entropy

from POMDPPlanners.core.tree import ActionNode, BeliefNode
from POMDPPlanners.core.policy import PolicyInfoVariable


def get_v_values_sample(action_node: ActionNode) -> np.ndarray:
    if not action_node.is_leaf:
        v_values_sample = [child.v_value for child in action_node.children]
        children_visit_counts = np.array([child.visit_count for child in action_node.children])
        v_values_sample = np.repeat(v_values_sample, children_visit_counts)
    else:
        v_values_sample = []
    
    return v_values_sample


def compute_tree_metrics(tree: BeliefNode) -> List[PolicyInfoVariable]:
    assert isinstance(tree, BeliefNode)
    
    if tree.is_leaf:
        return [
            PolicyInfoVariable(
                name="min_actions_visit_count",
                value=0,
            ),
            PolicyInfoVariable(
                name="max_actions_visit_count",
                value=0,
            ),
            PolicyInfoVariable(
                name="actions_visit_count_entropy",
                value=0,
            )
        ]
    
    visit_counts = np.array([node.visit_count for node in tree.children])
    
    # Calculate entropy of visit counts
    total_visits = np.sum(visit_counts)
    if total_visits > 0:
        probabilities = visit_counts / total_visits
        entropy_value = entropy(probabilities, base=2)
    else:
        entropy_value = 0.0
    
    return [
        PolicyInfoVariable(
            name="min_actions_visit_count",
            value=np.min(visit_counts),
        ),
        PolicyInfoVariable(
            name="max_actions_visit_count",
            value=np.max(visit_counts),
        ),
        PolicyInfoVariable(
            name="actions_visit_count_entropy",
            value=entropy_value,
        ),
    ]    
