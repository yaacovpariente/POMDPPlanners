"""Module for POMDP policy abstractions and execution tracking.

This module provides the foundational interface for POMDP policies, including
abstract base classes for policy implementations and data structures for
tracking policy execution and performance metrics.

Classes:
    Policy: Abstract base class for all POMDP policies
    PolicySpaceInfo: Space type information for policy compatibility
    PolicyInfoVariable: Named tuple for policy execution metrics
    PolicyRunData: Container for policy execution information
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Tuple, TYPE_CHECKING, Union, Optional
from typing import NamedTuple
from pathlib import Path
import logging

import numpy as np
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.logger import get_logger

if TYPE_CHECKING:
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.environment import SpaceType
    

@dataclass
class PolicySpaceInfo:
    """Data class containing space type requirements for policy compatibility.
    
    This class specifies the action and observation space types that a policy
    is designed to work with, enabling compatibility checking with environments.
    
    Attributes:
        action_space: Required action space type (discrete, continuous, or mixed)
        observation_space: Required observation space type (discrete, continuous, or mixed)
    """
    action_space: 'SpaceType'
    observation_space: 'SpaceType'

class PolicyInfoVariable(NamedTuple):
    """Named tuple for storing policy execution metrics.
    
    This structure stores key-value pairs of policy performance metrics
    that are collected during policy execution.
    
    Attributes:
        name: Descriptive name of the metric (e.g., "nodes_expanded", "planning_time")
        value: Numeric value of the metric
    """
    name: str
    value: Union[float, int]
    
class PolicyRunData(NamedTuple):
    """Container for policy execution information and metrics.
    
    This class aggregates all the information collected during a policy's
    action selection process, including performance metrics and execution details.
    
    Attributes:
        info_variables: List of policy-specific metrics and performance data
    """
    info_variables: List[PolicyInfoVariable]

class Policy(ABC):
    """Abstract base class for POMDP policies.
    
    This class defines the interface for POMDP policies that select actions
    based on belief states. All concrete policy implementations must inherit
    from this class and implement the action selection and space information methods.
    
    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement the action() and get_space_info() methods.
        
    Attributes:
        environment: The POMDP environment this policy operates in
        discount_factor: Discount factor for future rewards
        name: Unique identifier for the policy
        log_path: Optional directory for logging output
        debug: Flag to enable debug logging
    """
    
    def __init__(
        self,
        environment: "Environment",
        discount_factor: float,
        name: str,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False
    ):
        """Initialize the POMDP policy.
        
        Args:
            environment: Environment that this policy will operate in
            discount_factor: Discount factor for future rewards (0 < discount_factor <= 1)
            name: Unique identifier for this policy instance
            log_path: Optional directory for logging output. Defaults to None.
            debug: Enable debug logging. Defaults to False.
        """
        self.environment = environment
        self.discount_factor = discount_factor
        self.name = name
        self.log_path = log_path
        self.debug = debug
        self.use_queue_logger = use_queue_logger
        
        # Initialize logger with the policy's name and user-specified settings
        self.logger.info(f"Initialized policy: {self.name} (debug={self.debug})")
        
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this policy.
        
        The logger is implemented as a property to maintain pickle compatibility,
        as logger objects cannot be pickled directly.
        
        Returns:
            Configured logger instance with hierarchical naming
        """
        return get_logger(
            name=f"policy.{self.name}",
            level=logging.INFO,
            output_dir=self.log_path,
            debug=self.debug,
            use_queue=self.use_queue_logger
        )

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
            elif isinstance(value, logging.Logger):
                # Exclude logger from config serialization
                return None
            elif hasattr(value, '__dict__'):
                # Skip logger objects in __dict__ to avoid recursion
                if isinstance(value, logging.Logger):
                    return serialize_value(value.name)
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
        """Select action(s) based on the current belief state.
        
        This is the core method that implements the policy's decision-making logic.
        It takes a belief state and returns the selected action(s) along with
        execution information and performance metrics.
        
        Args:
            belief: Current belief state representing uncertainty over states
            
        Returns:
            Tuple containing:
                - List of selected actions (typically single action, but supports multiple)
                - PolicyRunData with execution metrics and performance information
                
        Note:
            Subclasses must implement this method with their specific planning
            or decision-making algorithm.
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        """Get space type requirements for this policy class.
        
        This class method specifies what types of action and observation spaces
        this policy implementation can handle, enabling compatibility checking
        with environments.
        
        Returns:
            PolicySpaceInfo specifying required action and observation space types
            
        Note:
            Subclasses must implement this method to declare their space compatibility.
            This is used for validation when pairing policies with environments.
        """
        pass
