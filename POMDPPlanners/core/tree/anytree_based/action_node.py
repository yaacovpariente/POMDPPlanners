from typing import Any, Union

import numpy as np

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.tree.anytree_based.base_node import BaseNode


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
        # Lazy import to avoid a cycle: visualization imports both node classes.
        from POMDPPlanners.core.tree.anytree_based.visualization import (
            print_tree,
        )  # pylint: disable=import-outside-toplevel

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


# Re-import at module bottom for runtime type resolution of forward refs above.
from POMDPPlanners.core.tree.anytree_based.belief_node import (
    BeliefNode,
)  # noqa: E402  pylint: disable=wrong-import-position
