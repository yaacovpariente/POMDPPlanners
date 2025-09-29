"""Configuration data structures for POMDP components.

This module defines standardized configuration data structures used throughout
the POMDP planning framework for component specification and experiment setup.

Classes:
    EnvironmentConfig: Configuration specification for environments
    PolicyConfig: Configuration specification for policies
    BeliefConfig: Configuration specification for beliefs
    ExperimentConfig: Complete experiment specification with all components
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Sequence

if TYPE_CHECKING:
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.core.policy import Policy


@dataclass
class EnvironmentConfig:
    """Configuration specification for POMDP environments.

    This data class standardizes how environment configurations are specified,
    enabling dynamic creation of environment instances from configuration files.

    Attributes:
        class_name: Name of the environment class to instantiate
        params: Dictionary of parameters to pass to the environment constructor

    Example:
        Creating environment configurations:

        >>> # Tiger POMDP configuration
        >>> tiger_config = EnvironmentConfig(
        ...     class_name="TigerPOMDP",
        ...     params={
        ...         "discount_factor": 0.95,
        ...         "tiger_location": "left",
        ...         "reward_correct": 10.0,
        ...         "reward_incorrect": -100.0
        ...     }
        ... )
    """

    class_name: str
    params: Dict[str, Any]


@dataclass
class PolicyConfig:
    """Configuration specification for POMDP policies.

    This data class standardizes how policy configurations are specified,
    enabling dynamic creation of policy instances from configuration files.

    Attributes:
        class_name: Name of the policy class to instantiate
        params: Dictionary of parameters to pass to the policy constructor

    Example:
        Creating policy configurations:

        >>> # POMCP policy configuration
        >>> pomcp_config = PolicyConfig(
        ...     class_name="POMCP",
        ...     params={
        ...         "num_simulations": 1000,
        ...         "exploration_constant": 1.0,
        ...         "max_depth": 10
        ...     }
        ... )
    """

    class_name: str
    params: Dict[str, Any]


@dataclass
class BeliefConfig:
    """Configuration specification for belief representations.

    This data class standardizes how belief configurations are specified,
    enabling dynamic creation of belief instances from configuration files.

    Attributes:
        class_name: Name of the belief class to instantiate
        params: Dictionary of parameters to pass to the belief constructor
    """

    class_name: str
    params: Dict[str, Any]


@dataclass
class ExperimentConfig:
    """Complete experiment specification with all required components.

    This data class aggregates all the components needed to run a POMDP
    experiment, including the environment, policies, belief representation,
    and execution parameters.

    Attributes:
        environment: Configured environment instance
        policies: List of policy instances to evaluate
        belief: Initial belief representation
        num_episodes: Number of episodes to run per policy
        num_steps: Maximum number of steps per episode
    """

    environment: "Environment"
    policies: Sequence["Policy"]
    belief: "Belief"
    num_episodes: int
    num_steps: int
