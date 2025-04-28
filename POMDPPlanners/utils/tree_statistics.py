import numpy as np

from POMDPPlanners.planners.research_planners.icvar_pomcp import ActionNode

def get_v_values_sample(action_node: ActionNode) -> np.ndarray:
    if not action_node.is_leaf:
        v_values_sample = [child.v_value for child in action_node.children]
        children_visit_counts = np.array([child.visit_count for child in action_node.children])
        v_values_sample = np.repeat(v_values_sample, children_visit_counts)
    else:
        v_values_sample = []
    
    return v_values_sample
