from anytree import NodeMixin, RenderTree, PostOrderIter
from typing import Optional, List, Any
import numpy as np

from POMDPPlanners.core.belief import Belief


class ActionNode(NodeMixin):
    def __init__(self, action, parent=None, children=tuple(), data: Any = None):
        self.action = action

        self.parent = parent
        self.children = children
        self.data = data

        self.q_value = 0.0
        self.visit_count = 0
        self.immediate_cost = 0
        self.sample = []
        self.lower_confidence_bound = 0.0
        self.upper_confidence_bound = 0.0


class BeliefNode(NodeMixin):
    def __init__(self, belief: Belief, parent=None, children=tuple(), data: Any = None):
        assert isinstance(belief, Belief)

        self.belief = belief
        self.parent = parent
        self.children = children
        self.data = data

        self.v_value = 0.0
        self.visit_count = 0
        self.lower_confidence_bound = 0.0
        self.upper_confidence_bound = 0.0


def get_optimal_action(belief_node: BeliefNode) -> Any:
    actions = [child.action for child in belief_node.children]
    q_values = [child.q_value for child in belief_node.children]
    return actions[np.argmax(q_values)]
