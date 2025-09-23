from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration files for POMDP experiments.

    This utility function provides a standardized way to load experimental
    configurations from YAML files. It's commonly used to configure
    environments, planners, and simulation parameters for reproducible
    experiments.

    The function uses PyYAML's safe_load to prevent execution of arbitrary
    Python code, making it secure for loading untrusted configuration files.

    Args:
        config_path: Path to the YAML configuration file (absolute or relative)

    Returns:
        Dictionary containing the parsed configuration parameters

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the file contains invalid YAML syntax

    Example:
        Loading experiment configuration::

            from POMDPPlanners.utils.config_loader import load_config

            # Load experiment configuration
            config = load_config("experiments/tiger_pomdp_study.yaml")

            # Access configuration sections
            env_config = config['environment']
            planner_configs = config['planners']
            simulation_config = config['simulation']

            print(f"Environment: {env_config['name']}")
            print(f"Planners: {[p['name'] for p in planner_configs]}")
            print(f"Episodes per run: {simulation_config['episodes_per_run']}")

    Example:
        Using with environment configuration::

            # Example YAML content (tiger_config.yaml):
            # environment:
            #   name: "TigerPOMDP"
            #   discount_factor: 0.95
            #   observation_accuracy: 0.85
            # planners:
            #   - name: "POMCP"
            #     n_simulations: 1000
            #     depth: 20
            #     exploration_constant: 1.0
            #   - name: "PFT_DPW"
            #     n_simulations: 500
            #     depth: 15
            #     k_a: 2.0
            # simulation:
            #   episodes_per_run: 100
            #   num_runs: 10

            config = load_config("tiger_config.yaml")

            # Create environment from config
            from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
            env = TigerPOMDP(discount_factor=config['environment']['discount_factor'])

            # Create planners from config
            planners = []
            for planner_config in config['planners']:
                if planner_config['name'] == 'POMCP':
                    from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
                    planner = POMCP(
                        environment=env,
                        discount_factor=env.discount_factor,
                        n_simulations=planner_config['n_simulations'],
                        depth=planner_config['depth'],
                        exploration_constant=planner_config['exploration_constant'],
                        name=planner_config['name']
                    )
                    planners.append(planner)

    Example:
        Handling configuration hierarchies and defaults::

            # Example YAML with nested configuration:
            # defaults:
            #   simulation:
            #     episodes: 50
            #     particles: 100
            # experiments:
            #   quick_test:
            #     environment: "SanityPOMDP"
            #     planners: ["POMCP"]
            #   full_study:
            #     environment: "TigerPOMDP"
            #     planners: ["POMCP", "PFT_DPW", "SparsePFT"]
            #     simulation:
            #       episodes: 200
            #       particles: 500

            config = load_config("multi_experiment_config.yaml")

            # Access different experiment configurations
            defaults = config['defaults']
            quick_config = config['experiments']['quick_test']
            full_config = config['experiments']['full_study']

            # Merge defaults with specific configuration
            def merge_config(base_config, specific_config):
                merged = base_config.copy()
                merged.update(specific_config)
                return merged

            full_simulation_config = merge_config(
                defaults['simulation'],
                full_config.get('simulation', {})
            )
            print(f"Full study episodes: {full_simulation_config['episodes']}")

    Configuration Best Practices:
        **File Organization**:
        - Use descriptive filenames (e.g., `pomcp_tiger_baseline.yaml`)
        - Organize configs by environment or study type
        - Include version information in complex configurations

        **Parameter Naming**:
        - Use consistent naming conventions across configurations
        - Group related parameters under sections
        - Include comments explaining non-obvious parameters

        **Default Handling**:
        - Define sensible defaults for optional parameters
        - Use inheritance or merging for parameter variants
        - Validate required parameters after loading

    Security Considerations:
        - Uses `yaml.safe_load()` to prevent code execution
        - Suitable for loading user-provided configuration files
        - Automatically handles standard YAML data types safely
        - Does not support custom Python object instantiation
    """
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
