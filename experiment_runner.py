import argparse
from pathlib import Path
import importlib.util
import sys
from typing import List, Dict, Any

from POMDPPlanners.core.config_types import ExperimentConfig
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.utils.logger import get_logger
from POMDPPlanners.simulations.simulations_api import SimulationsAPI

logger = get_logger(__name__)


def load_config_module(config_path: Path) -> Any:
    """Load a Python module from a file path.
    
    Args:
        config_path: Path to the Python module
        
    Returns:
        The loaded module
    """
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {config_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[config_path.stem] = module
    spec.loader.exec_module(module)
    return module


def load_experiment_configs(config_dir: Path) -> List[ExperimentConfig]:
    """Load all experiment configurations from a directory.
    
    Args:
        config_dir: Directory containing experiment configuration files
        
    Returns:
        List of ExperimentConfig objects
    """
    configs = []
    for config_file in config_dir.glob("*.py"):
        if config_file.name == "__init__.py":
            continue
            
        try:
            module = load_config_module(config_file)
            # Look for variables ending with _experiment_config
            for var_name in dir(module):
                if var_name.endswith("_experiment_config"):
                    config = getattr(module, var_name)
                    if isinstance(config, ExperimentConfig):
                        configs.append(config)
        except Exception as e:
            logger.warning(f"Error loading config from {config_file}: {str(e)}")
            continue
            
    return configs


def create_environment_run_params(configs: List[ExperimentConfig], debug_mode: bool = False) -> List[EnvironmentRunParams]:
    """Convert experiment configs to environment run parameters.
    
    Args:
        configs: List of experiment configurations
        debug_mode: If True, reduces number of episodes to 2 for debugging
        
    Returns:
        List of EnvironmentRunParams objects
    """
    return [
        EnvironmentRunParams(
            environment=config.environment,
            belief=config.belief,
            policies=config.policies,
            num_episodes=2 if debug_mode else config.num_episodes,
            num_steps=2 if debug_mode else config.num_steps
        )
        for config in configs
    ]


def main():
    parser = argparse.ArgumentParser(description="Run POMDP planning experiments")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory containing experiment configurations")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save experiment results")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha value for statistical tests")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence interval level")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs to run")
    parser.add_argument("--scheduler_address", type=str, help="Dask scheduler address for distributed computing")
    parser.add_argument("--cache_visualizations", action="store_true", help="Cache visualizations")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with reduced episodes")
    parser.add_argument("--clear_cache_on_start", action="store_true", help="Clear cache directory at the start of the experiment")
    args = parser.parse_args()

    # Setup logging
    logger = get_logger(__name__)
    logger.info("Starting experiment runner")
    logger.info(f"Arguments: {args}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config directory
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        raise ValueError(f"Config directory {config_dir} does not exist")
    
    # Load configurations
    logger.info(f"Loading configurations from {config_dir}")
    configs = load_experiment_configs(config_dir)
    if not configs:
        logger.error("No valid configurations found!")
        return
        
    logger.info(f"Loaded {len(configs)} configurations")
    
    # Convert to environment run parameters
    environment_run_params = create_environment_run_params(configs, debug_mode=args.debug)
    
    if args.debug:
        logger.info("Running in DEBUG mode - using only 2 episodes per configuration")
    
    # Run comparison
    logger.info("Starting experiment comparison...")
    api = SimulationsAPI()
    histories, statistics_df = api.run_multiple_environments_and_policies_local_run_with_initial_debug_run(
        environment_run_params=environment_run_params,
        alpha=args.alpha,
        confidence_interval_level=args.confidence,
        experiment_name=args.experiment_name,
        n_jobs=args.n_jobs,
        cache_dir_path=output_dir,
        clear_cache_on_start=args.clear_cache_on_start
    )
    
    logger.info("Experiment completed successfully!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("\nStatistics Summary:")
    logger.info(f"\n{statistics_df}")


if __name__ == "__main__":
    main()
