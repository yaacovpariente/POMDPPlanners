from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Tuple, TYPE_CHECKING, Union
from typing import NamedTuple

import numpy as np
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.environment import SpaceType
    

@dataclass
class PolicySpaceInfo:
    action_space: 'SpaceType'
    observation_space: 'SpaceType'

class PolicyInfoVariable(NamedTuple):
    name: str
    value: Union[float, int]
    
class PolicyRunData(NamedTuple):
    info_variables: List[PolicyInfoVariable]

class Policy(ABC):
    def __init__(self, environment: "Environment", discount_factor: float, name: str):
        self.environment = environment
        self.discount_factor = discount_factor
        self.name = name

    @property
    def config_id(self) -> str:
        """Generate a deterministic identifier based on policy configuration."""
        def serialize_value(value):
            """Helper function to serialize values in a deterministic way."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {str(k): serialize_value(v) for k, v in sorted(value.items())}
            elif hasattr(value, '__dict__'):
                return serialize_value(value.__dict__)
            else:
                return str(value)
        
        config_dict = {}
        for key, value in self.__dict__.items():
            if key.startswith('_') or callable(value):
                continue
            config_dict[key] = serialize_value(value)
        config_dict = dict(sorted(config_dict.items()))
        config_dict['environment'] = self.environment.config_id
        
        return config_to_id(config_dict)

    def __hash__(self) -> int:
        return hash(self.config_id)

    @abstractmethod
    def action(self, belief: "Belief") -> Tuple[List[Any], PolicyRunData]:
        pass
    
    @classmethod
    @abstractmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        pass
