from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING
import hashlib
import json

if TYPE_CHECKING:
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.environment import SpaceType
    

@dataclass
class PolicySpaceInfo:
    action_space: 'SpaceType'
    observation_space: 'SpaceType'

class Policy(ABC):
    def __init__(self, environment: "Environment", discount_factor: float, name: str):
        self.environment = environment
        self.discount_factor = discount_factor
        self.name = name

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on policy configuration."""
        config_dict = {}
        
        # Include all public attributes that aren't callables
        for key, value in self.__dict__.items():
            if key.startswith('_') or callable(value):
                continue
                
            # Handle environment by using its config_id
            if key == 'environment':
                config_dict[key] = value.config_id
            # Handle basic Python types
            elif isinstance(value, (str, int, float, bool, list, tuple)):
                config_dict[key] = value
            # Handle dictionaries
            elif isinstance(value, dict):
                # Only include serializable values
                serializable_dict = {}
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool, list, tuple)):
                        serializable_dict[k] = v
                config_dict[key] = serializable_dict
                
        # Sort dictionary to ensure consistent ordering
        config_dict = dict(sorted(config_dict.items()))
        
        # Create a deterministic string representation and hash it
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    @abstractmethod
    def action(self, belief: "Belief"):
        pass
    
    @classmethod
    @abstractmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        pass
