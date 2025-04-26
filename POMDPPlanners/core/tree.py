from typing import Any, Union
import numpy as np
from anytree import NodeMixin
from anytree import RenderTree

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.belief import Belief

class BaseNode(NodeMixin):
    def __init__(self, parent=None, children=tuple(), data: Any = None):
        self.parent = parent
        self.children = children
        self.data = data
        self.visit_count = 0
        self.lower_confidence_bound = 0.0
        self.upper_confidence_bound = 0.0
        self.sample = []
class ActionNode(BaseNode):
    def __init__(self, action, parent=None, children=tuple(), data: Any = None):
        super().__init__(parent=parent, children=children, data=data)
        self.action = action

        self.q_value = 0.0
        self.immediate_cost = None

    @property
    def spec(self):
        return f"ActionNode: action={self.action}, q_value={self.q_value}, visit_count={self.visit_count}, immediate_cost={self.immediate_cost}, sample={self.sample}, lower_confidence_bound={self.lower_confidence_bound}, upper_confidence_bound={self.upper_confidence_bound}, depth={self.depth}"

    @property
    def name(self):
        return self.spec

    def print(self):
        print_tree(self)        
class BeliefNode(BaseNode):
    def __init__(self, belief: Belief, observation: Any = None, parent=None, children=tuple(), data: Any = None):
        assert isinstance(belief, Belief)
        super().__init__(parent=parent, children=children, data=data)

        self.belief = belief
        self.observation = observation
        self.v_value = 0.0
        
    @property
    def spec(self):
        return f"BeliefNode: v_value={self.v_value}, visit_count={self.visit_count}, lower_confidence_bound={self.lower_confidence_bound}, upper_confidence_bound={self.upper_confidence_bound}, depth={self.depth}"
        
    @property
    def name(self):
        return self.spec
        
    def print(self):
        print_tree(self)
        
    def update_belief(self, action: Any, observation: Any, pomdp: Environment, **kwargs):
        self.belief = self.belief.update(action=action, observation=observation, pomdp=pomdp, **kwargs)
        
        
def print_tree(tree: Union[BeliefNode, ActionNode]):
    for pre, fill, node in RenderTree(tree):
        if isinstance(node, BeliefNode):
            name = node.spec
        elif isinstance(node, ActionNode):
            name = node.spec

        print(f"{pre}{name}")

def get_optimal_action_cost_setting(belief_node: BeliefNode) -> Any:
    actions = [child.action for child in belief_node.children]
    q_values = [child.q_value for child in belief_node.children]
    return actions[np.argmin(q_values)]

def get_optimal_action_reward_setting(belief_node: BeliefNode) -> Any:
    actions = [child.action for child in belief_node.children]
    q_values = [child.q_value for child in belief_node.children]
    return actions[np.argmax(q_values)]
