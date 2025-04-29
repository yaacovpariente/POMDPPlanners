from abc import ABC, abstractmethod
from time import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.belief import Belief


class Policy(ABC):
    def __init__(self, environment: "Environment", discount_factor: float, name: str):
        self.environment = environment
        self.discount_factor = discount_factor
        self.name = name

    @abstractmethod
    def action(self, belief: "Belief"):
        pass
