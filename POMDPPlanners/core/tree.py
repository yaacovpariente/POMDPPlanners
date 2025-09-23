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
        self.immediate_cost = None
        self.immediate_reward = None
        self.sample = []


class ActionNode(BaseNode):
    def __init__(self, action, parent=None, children=tuple(), data: Any = None):
        super().__init__(parent=parent, children=children, data=data)
        self.action = action
        self.q_value = 0.0

    @property
    def spec(self):
        return f"""ActionNode:
                action: {self.action}
                q_value: {self.q_value}
                visit_count: {self.visit_count}
                immediate_cost: {self.immediate_cost}
                immediate_reward: {self.immediate_reward}
                lower_confidence_bound: {self.lower_confidence_bound}
                upper_confidence_bound: {self.upper_confidence_bound}
                depth: {self.depth}"""

    @property
    def name(self):
        return self.spec

    def print(self):
        print_tree(self)

    def sample_child_node(self) -> "BeliefNode":
        child_weights = np.array([child.weight for child in self.children])
        weights = child_weights / sum(child_weights)
        return np.random.choice(self.children, p=weights)

    def get_belief_node_child(
        self, observation: Any, environment: Environment
    ) -> Union["BeliefNode", None]:
        for child in self.children:
            if environment.is_equal_observation(child.observation, observation):
                return child

        return None


class BeliefNode(BaseNode):
    def __init__(
        self,
        belief: Belief,
        observation: Any = None,
        weight: Union[float, int] = 1.0,
        parent=None,
        children=tuple(),
        data: Any = None,
    ):
        if not isinstance(belief, Belief):
            raise TypeError("belief must be a Belief instance")
        super().__init__(parent=parent, children=children, data=data)

        self.belief = belief
        self.observation = observation
        self.weight = weight
        self.v_value = 0.0

    @property
    def spec(self):
        return f"""BeliefNode:
                observation: {self.observation}
                v_value: {self.v_value}
                visit_count: {self.visit_count}
                weight: {self.weight}
                lower_confidence_bound: {self.lower_confidence_bound}
                upper_confidence_bound: {self.upper_confidence_bound}
                depth: {self.depth}"""

    @property
    def name(self):
        return self.spec

    def print(self):
        print_tree(self)

    def update_belief(self, action: Any, observation: Any, pomdp: Environment, **kwargs):
        self.belief = self.belief.update(
            action=action, observation=observation, pomdp=pomdp, **kwargs
        )

    def get_child(self, action: Any) -> Union["ActionNode", None]:
        for child in self.children:
            if child.action == action:
                return child

        return None


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


def sample_belief_node_child(action_node: ActionNode) -> BeliefNode:
    child_visit_counts = np.array([child.visit_count for child in action_node.children])
    weights = child_visit_counts / sum(child_visit_counts)
    return np.random.choice(action_node.children, p=weights)
