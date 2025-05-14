"""Module for running POMDP experiments."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from POMDPPlanners.utils.config_loader import load_config
from POMDPPlanners.environments import get_environment
from POMDPPlanners.planners import get_policy
from POMDPPlanners.simulations.simulations import (
    compare_multiple_environments_policies,
    EnvironmentRunParams,
    DeploymentType
)
from POMDPPlanners.core.simulation import History
from POMDPPlanners.utils.logger import logger
from POMDPPlanners.utils.weighted_particle_beliefs import create_belief
import pandas as pd


class ExperimentRunner:
    def __init__(self, config_path: str):
        """
        Initialize the experiment runner with a configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        logger.info(f"Initializing experiment runner with config: {config_path}")
        self.config = load_config(config_path)
        self.environment = None
        self.policies = []
        logger.debug(f"Loaded configuration: {self.config}")
        
    def setup_environment(self):
        """Initialize the environment from configuration."""
        logger.info("Setting up environment...")
        env_config = self.config['environment']
        logger.debug(f"Environment configuration: {env_config}")
        
        try:
            self.environment = get_environment(
                env_config['type'],
                **env_config['params']
            )
            logger.info(f"Successfully initialized environment: {self.environment.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize environment: {str(e)}")
            raise
        
    def setup_policies(self):
        """Initialize all policies from configuration."""
        logger.info("Setting up policies...")
        for i, policy_config in enumerate(self.config['policies']):
            logger.debug(f"Initializing policy {i+1}: {policy_config['type']}")
            
            # Add required parameters for policy initialization
            policy_params = policy_config['params'].copy()
            policy_params.update({
                'environment': self.environment,
                'discount_factor': self.environment.discount_factor,
                'name': f"{policy_config['type']}_{len(self.policies)}"
            })
            
            try:
                policy = get_policy(
                    policy_config['type'],
                    **policy_params
                )
                self.policies.append(policy)
                logger.info(f"Successfully initialized policy: {policy.name}")
            except Exception as e:
                logger.error(f"Failed to initialize policy {policy_config['type']}: {str(e)}")
                raise
            
    def run_experiment(self, n_jobs: int = 1, cache_dir_path: Path = None) -> Tuple[Dict[str, Dict[str, List[History]]], pd.DataFrame]:
        """
        Run the experiment for the specified number of episodes using parallel simulation.
        
        Args:
            n_jobs: Number of parallel jobs for simulation
            cache_dir_path: Path to store results (if None, uses current directory)
            
        Returns:
            Tuple containing:
            - Dictionary mapping environment names to dictionaries of policy histories
            - DataFrame with statistics and policy configurations
        """
        if 'num_episodes' not in self.config or 'num_steps' not in self.config:
            raise KeyError("Both 'num_episodes' and 'num_steps' must be specified as top-level keys in the YAML config.")
        num_episodes = self.config['num_episodes']
        num_steps = self.config['num_steps']
        logger.info(f"Starting experiment with {num_episodes} episodes, {num_steps} steps per episode, {n_jobs} parallel jobs")
        
        # Setup environment and policies
        self.setup_environment()
        self.setup_policies()
        
        # Create environment run parameters
        logger.info("Creating environment run parameters...")
        env_run_params = EnvironmentRunParams(
            environment=self.environment,
            belief=create_belief(self.environment, self.config['belief']),
            policies=self.policies,
            num_episodes=num_episodes,
            num_steps=num_steps
        )
        logger.debug(f"Environment run parameters: {env_run_params}")
        
        # Run comparison
        logger.info("Starting simulation comparison...")
        try:
            histories, statistics_df = compare_multiple_environments_policies(
                environment_run_params=[env_run_params],
                alpha=0.05,  # Standard significance level
                confidence_interval_level=0.95,
                n_jobs=n_jobs,
                cache_dir_path=cache_dir_path,
                experiment_name=f"{self.environment.__class__.__name__}_experiment",
                cache_visualizations=True,
                deployment_type=DeploymentType.LOCAL
            )
            logger.info("Simulation completed successfully")
            logger.debug(f"Statistics summary:\n{statistics_df}")
            
            return histories, statistics_df
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise 