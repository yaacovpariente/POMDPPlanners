"""Hyperparameter configuration packages for all POMDP planners.

This package contains hyperparameter optimization configurations for all planners
in the POMDPPlanners library. Each planner has its own module with configurations
for all supported environments.

Available Planners:
    - POMCP: Basic Monte Carlo Tree Search with UCB1
    - POMCP_DPW: POMCP with Double Progressive Widening
    - StandardSparseSamplingDiscreteActionsPlanner: Sparse sampling algorithm
    - DiscreteActionSequencesPlanner: Open-loop exhaustive search
    - SparsePFT: Sparse Progressive Function Transfer with enhanced UCB
    - POMCPOW: POMCP with Optimistic Weights (requires custom optimization)
    - PFT_DPW: Progressive Function Transfer with Double Progressive Widening (requires custom optimization)

Usage:
    Import configurations directly from submodules::

        from POMDPPlanners.experiments.configs.hyperparameter_configs.pomcp_configs import ALL_POMCP_CONFIGS
        from POMDPPlanners.experiments.configs.hyperparameter_configs.sparse_sampling_configs import ALL_SPARSE_SAMPLING_CONFIGS

    Or use the consolidated imports in this module::

        from POMDPPlanners.experiments.configs.hyperparameter_configs import ALL_CONFIGS
        from POMDPPlanners.experiments.configs.hyperparameter_configs import get_configs_by_planner
        from POMDPPlanners.experiments.configs.hyperparameter_configs import get_configs_by_environment
"""

from .pomcp_configs import ALL_POMCP_CONFIGS
from .pomcp_dpw_configs import ALL_POMCP_DPW_CONFIGS
from .sparse_sampling_configs import ALL_SPARSE_SAMPLING_CONFIGS
from .discrete_action_sequences_configs import ALL_DISCRETE_ACTION_SEQUENCES_CONFIGS
from .sparse_pft_configs import ALL_SPARSE_PFT_CONFIGS
from .pomcpow_configs import ALL_POMCPOW_CONFIGS, POMCPOW_CONFIG_INFO
from .pft_dpw_configs import ALL_PFT_DPW_CONFIGS, PFT_DPW_CONFIG_INFO

# Consolidated list of all hyperparameter configurations
# Note: POMCPOW and PFT_DPW configs are empty lists since they require custom optimization
ALL_CONFIGS = (
    ALL_POMCP_CONFIGS
    + ALL_POMCP_DPW_CONFIGS
    + ALL_SPARSE_SAMPLING_CONFIGS
    + ALL_DISCRETE_ACTION_SEQUENCES_CONFIGS
    + ALL_SPARSE_PFT_CONFIGS
    + ALL_POMCPOW_CONFIGS  # Empty list - requires custom optimization
    + ALL_PFT_DPW_CONFIGS  # Empty list - requires custom optimization
)


def get_configs_by_planner(planner_name: str):
    """Get all configurations for a specific planner.

    Args:
        planner_name: Name of the planner class (e.g., "POMCP", "StandardSparseSamplingDiscreteActionsPlanner")

    Returns:
        List of HyperParameterRunParams configurations for the specified planner

    Example:
        >>> pomcp_configs = get_configs_by_planner("POMCP")
        >>> len(pomcp_configs)
        9
    """
    return [config for config in ALL_CONFIGS if config.policy_cls.__name__ == planner_name]


def get_configs_by_environment(environment_name: str):
    """Get all configurations for a specific environment.

    Args:
        environment_name: Name of the environment class (e.g., "TigerPOMDP", "CartPolePOMDP")

    Returns:
        List of HyperParameterRunParams configurations for the specified environment

    Example:
        >>> tiger_configs = get_configs_by_environment("TigerPOMDP")
        >>> len(tiger_configs)
        4
    """
    return [
        config
        for config in ALL_CONFIGS
        if config.environment.__class__.__name__ == environment_name
    ]


def get_config_summary():
    """Get a summary of all available configurations.

    Returns:
        Dictionary with planner names as keys and lists of environment names as values

    Example:
        >>> summary = get_config_summary()
        >>> summary["POMCP"]
        ['TigerPOMDP', 'CartPolePOMDP', 'MountainCarPOMDP', ...]
    """
    summary = {}
    for config in ALL_CONFIGS:
        planner_name = config.policy_cls.__name__
        env_name = config.environment.__class__.__name__

        if planner_name not in summary:
            summary[planner_name] = []
        summary[planner_name].append(env_name)

    return summary


# Export main symbols
__all__ = [
    "ALL_CONFIGS",
    "ALL_POMCP_CONFIGS",
    "ALL_POMCP_DPW_CONFIGS",
    "ALL_SPARSE_SAMPLING_CONFIGS",
    "ALL_DISCRETE_ACTION_SEQUENCES_CONFIGS",
    "ALL_SPARSE_PFT_CONFIGS",
    "ALL_POMCPOW_CONFIGS",
    "ALL_PFT_DPW_CONFIGS",
    "POMCPOW_CONFIG_INFO",
    "PFT_DPW_CONFIG_INFO",
    "get_configs_by_planner",
    "get_configs_by_environment",
    "get_config_summary",
]
