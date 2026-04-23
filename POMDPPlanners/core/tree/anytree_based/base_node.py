from typing import Any, List, Optional

from anytree import NodeMixin


class BaseNode(NodeMixin):
    def __init__(self, parent=None, children=tuple(), data: Any = None):
        self.parent = parent
        self.children = children
        self.data = data
        self.visit_count = 0
        self.lower_confidence_bound = 0.0
        self.upper_confidence_bound = 0.0
        self._immediate_cost: Optional[float] = None
        self._immediate_reward: Optional[float] = None
        self.sample: List[Any] = []

    @property
    def immediate_cost(self) -> Optional[float]:
        """Get the immediate cost value."""
        return self._immediate_cost

    @immediate_cost.setter
    def immediate_cost(self, value: Optional[float]):
        """Set immediate cost and automatically update immediate_reward to its negative value.

        Args:
            value: The cost value to set. When set, immediate_reward will be set to -value.
        """
        self._immediate_cost = value
        if value is not None:
            self._immediate_reward = -value

    @property
    def immediate_reward(self) -> Optional[float]:
        """Get the immediate reward value."""
        return self._immediate_reward

    @immediate_reward.setter
    def immediate_reward(self, value: Optional[float]):
        """Set immediate reward and automatically update immediate_cost to its negative value.

        Args:
            value: The reward value to set. When set, immediate_cost will be set to -value.
        """
        self._immediate_reward = value
        if value is not None:
            self._immediate_cost = -value
