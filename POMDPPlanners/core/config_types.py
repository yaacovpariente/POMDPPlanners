"""Configuration types for POMDP components."""

from typing import Any, List, Dict, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.policy import Policy
    from POMDPPlanners.core.belief import Belief

@dataclass
class EnvironmentConfig:
    class_name: str
    params: Dict[str, Any]

@dataclass
class PolicyConfig:
    class_name: str
    params: Dict[str, Any]

@dataclass
class BeliefConfig:
    class_name: str
    params: Dict[str, Any]

@dataclass
class ExperimentConfig:
    environment: 'Environment'
    policies: List['Policy']
    belief: 'Belief'
    num_episodes: int
    num_steps: int 