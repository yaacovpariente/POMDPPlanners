# SPDX-License-Identifier: MIT

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
        Loading experiment configuration:

        >>> from POMDPPlanners.utils.config_loader import load_config
        >>> import tempfile
        >>> import os

        >>> # Create a temporary config file for testing
        >>> config_content = '''
        ... environment:
        ...   name: "TigerPOMDP"
        ...   discount_factor: 0.95
        ... planners:
        ...   - name: "POMCP"
        ...     n_simulations: 1000
        ... simulation:
        ...   episodes_per_run: 100
        ... '''
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        ...     _ = f.write(config_content)
        ...     temp_config_path = f.name

        >>> # Load experiment configuration
        >>> config = load_config(temp_config_path)
        >>> config['environment']['name']
        'TigerPOMDP'
        >>> config['environment']['discount_factor']
        0.95
        >>> config['planners'][0]['name']
        'POMCP'
        >>> config['simulation']['episodes_per_run']
        100

        >>> # Clean up
        >>> os.unlink(temp_config_path)

    Example:
        Using with environment configuration:

        >>> # Create a temporary config file with environment and planners
        >>> config_content = '''
        ... environment:
        ...   name: "TigerPOMDP"
        ...   discount_factor: 0.95
        ...   observation_accuracy: 0.85
        ... planners:
        ...   - name: "POMCP"
        ...     n_simulations: 100
        ...     depth: 5
        ...     exploration_constant: 1.0
        ... simulation:
        ...   episodes_per_run: 10
        ...   num_runs: 2
        ... '''
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        ...     _ = f.write(config_content)
        ...     temp_config_path = f.name

        >>> config = load_config(temp_config_path)

        >>> # Create environment from config
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> env = TigerPOMDP(discount_factor=config['environment']['discount_factor'])
        >>> env.discount_factor
        0.95

        >>> # Access planner configuration
        >>> planner_config = config['planners'][0]
        >>> planner_config['name']
        'POMCP'
        >>> planner_config['n_simulations']
        100

        >>> # Clean up
        >>> os.unlink(temp_config_path)

    Example:
        Handling configuration hierarchies and defaults:

        >>> # Create a nested configuration file
        >>> nested_config_content = '''
        ... defaults:
        ...   simulation:
        ...     episodes: 50
        ...     particles: 100
        ... experiments:
        ...   quick_test:
        ...     environment: "SanityPOMDP"
        ...     planners: ["POMCP"]
        ...   full_study:
        ...     environment: "TigerPOMDP"
        ...     planners: ["POMCP", "PFT_DPW"]
        ...     simulation:
        ...       episodes: 200
        ...       particles: 500
        ... '''
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        ...     _ = f.write(nested_config_content)
        ...     temp_config_path = f.name

        >>> config = load_config(temp_config_path)

        >>> # Access different experiment configurations
        >>> defaults = config['defaults']
        >>> quick_config = config['experiments']['quick_test']
        >>> full_config = config['experiments']['full_study']

        >>> # Verify configuration structure
        >>> defaults['simulation']['episodes']
        50
        >>> quick_config['environment']
        'SanityPOMDP'
        >>> full_config['simulation']['episodes']
        200

        >>> # Simple config merging test
        >>> merged = {**defaults['simulation'], **full_config.get('simulation', {})}
        >>> merged['episodes']
        200
        >>> merged['particles']
        500

        >>> # Clean up
        >>> os.unlink(temp_config_path)

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
    # Load a YAML configuration file.
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
