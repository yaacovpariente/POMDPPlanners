from typing import Any, Union

import numpy as np

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.tree.anytree_based.base_node import BaseNode


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
            # Runtime guard for callers that bypass static typing.
            raise TypeError(
                "belief must be a Belief instance"
            )  # pyright: ignore[reportUnreachable]
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
        # Lazy import to avoid a cycle: visualization imports both node classes.
        from POMDPPlanners.core.tree.anytree_based.visualization import (
            print_tree,
        )  # pylint: disable=import-outside-toplevel

        print_tree(self)

    def update_belief(self, action: Any, observation: Any, pomdp: Environment, **kwargs):
        self.belief = self.belief.update(
            action=action, observation=observation, pomdp=pomdp, **kwargs
        )

    def get_child(self, action: Any) -> Union["ActionNode", None]:
        for child in self.children:
            # Handle numpy array comparisons properly
            if isinstance(child.action, np.ndarray) and isinstance(action, np.ndarray):
                if np.array_equal(child.action, action):
                    return child
            elif child.action == action:
                return child

        return None


# Re-import at module bottom for runtime type resolution of forward refs above.
from POMDPPlanners.core.tree.anytree_based.action_node import (
    ActionNode,
)  # noqa: E402  pylint: disable=wrong-import-position
