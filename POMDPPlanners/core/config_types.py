"""Configuration types for POMDP components."""

from typing import Any, List, Dict
from dataclasses import dataclass

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
    environment: EnvironmentConfig
    policies: List[PolicyConfig]
    belief: BeliefConfig
    num_episodes: int
    num_steps: int 