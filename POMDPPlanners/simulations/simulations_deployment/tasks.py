from typing import Any, Union, Optional
from pathlib import Path
import hashlib
import json
import random
import logging

import numpy as np

from POMDPPlanners.core.simulation import SimulationTask, History
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.utils.logger import get_logger
from POMDPPlanners.utils.config_to_id import config_to_id


class EpisodeSimulationTask(SimulationTask):
    """A class to represent a single simulation task with caching capabilities."""
    
    def __init__(
        self,
        environment: Any,  # Use Any to avoid circular import
        policy: Any,
        initial_belief: Any,
        num_steps: int,
        episode_id: int,
        seed: int,
        discount_factor: float = 1.0,
        episode_number: int = 0,
        cache_dir: Optional[Path] = None,
        debug: bool = False,
        console_output: bool = True
    ):
        """Initialize a simulation task.
        
        Args:
            environment: The environment to simulate
            policy: The policy to use
            initial_belief: The initial belief state
            num_steps: Number of steps to simulate
            episode_id: Unique identifier for this episode
            seed: Random seed for reproducibility
            discount_factor: Discount factor for reward calculation
            episode_number: The episode number for this simulation
            cache_dir: Directory for caching results
            debug: Whether to enable debug logging
            console_output: Whether to enable console output (default: True).
                          Set to False to disable console output while keeping file logging.
            
        Raises:
            ValueError: If num_steps is not positive
        """
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
            
        self.environment = environment
        self.policy = policy
        self.initial_belief = initial_belief
        self.num_steps = num_steps
        self.episode_id = episode_id
        self.seed = seed
        self.discount_factor = discount_factor
        self.episode_number = episode_number
        self.debug = debug
        self.console_output = console_output
        self.cache_dir = cache_dir
        
        # Generate cache key after all attributes are set
        self._cache_key = self._generate_cache_key()
        
        # Log task creation
        temp_logger = get_logger(
            name=f"task.{self.environment.name}.{self.policy.name}.{self.episode_id}",
            debug=self.debug,
            output_dir=self.cache_dir / "logs" / "episodes" if self.cache_dir else None,
            console_output=self.console_output
        )
        temp_logger.debug(f"Creating EpisodeSimulationTask with episode_id={self.episode_id}, episode_number={self.episode_number}")
        temp_logger.debug(f"Task parameters: num_steps={self.num_steps}, seed={self.seed}, discount_factor={self.discount_factor}")
        temp_logger.debug(f"Cache directory: {self.cache_dir}")
        temp_logger.debug(f"Generated cache key: {self._cache_key}")
    
    @property
    def logger(self) -> logging.Logger:
        """All tasks should remain pickable and therefore the logger should be a property"""
        output_dir = None
        if self.cache_dir is not None:
            output_dir = self.cache_dir / "logs" / "episodes"
        
        return get_logger(
            name=f"task.{self.environment.name}.{self.policy.name}.{self.episode_id}",
            debug=self.debug,
            output_dir=output_dir,
            console_output=self.console_output
        )
    
    def _generate_cache_key(self) -> str:
        """Generate a unique cache key for this task."""
        
        components = {
            'env': self.environment.config_id,
            'policy': self.policy.config_id,
            'belief': self.initial_belief.config_id,
            'episode_id': self.episode_id,
            'episode_number': self.episode_number,
            'num_steps': self.num_steps,
            'seed': self.seed,
            'discount_factor': self.discount_factor
        }
        return config_to_id(components)
    
    def get_config_id(self) -> str:
        """Get the configuration ID for this task.
        
        Returns:
            str: The cache key used to identify this task
        """
        return self._cache_key
    
    def to_dict(self) -> dict:
        """Convert task to dictionary for serialization."""
        return {
            'environment': self.environment,
            'policy': self.policy,
            'initial_belief': self.initial_belief,
            'num_steps': self.num_steps,
            'episode_id': self.episode_id,
            'seed': self.seed,
            'discount_factor': self.discount_factor,
            'episode_number': self.episode_number
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EpisodeSimulationTask':
        """Create a SimulationTask instance from a dictionary."""
        return cls(**data)
    
    def run(self) -> Union[History, None]:
        """Run the simulation task.
        
        Returns:
            History: The simulation history
        """
        # Log task configuration
        self.logger.info(f"Starting episode {self.episode_id} (episode_number: {self.episode_number})")
        self.logger.info(f"Environment: {self.environment.name} (config_id: {self.environment.config_id})")
        self.logger.info(f"Policy: {self.policy.name} (config_id: {self.policy.config_id})")
        self.logger.info(f"Initial belief: {self.initial_belief.config_id}")
        self.logger.info(f"Configuration: num_steps={self.num_steps}, seed={self.seed}, discount_factor={self.discount_factor}")
        self.logger.info(f"Debug mode: {self.debug}, Console output: {self.console_output}")
        if self.cache_dir:
            self.logger.info(f"Cache directory: {self.cache_dir}")
        self.logger.info(f"Cache key: {self._cache_key}")
        
        # Log random state setup
        self.logger.debug(f"Setting random seed to {self.seed}")
        state = np.random.get_state()
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        try:
            # Run simulation with seed parameter
            self.logger.info("Starting episode simulation...")
            result = run_episode(
                environment=self.environment,
                policy=self.policy,
                initial_belief=self.initial_belief,
                num_steps=self.num_steps,
                logger=self.logger
            )
            
            if result is not None:
                self.logger.info(f"Episode {self.episode_id} completed successfully")
                self.logger.debug(f"Episode result type: {type(result)}")
                if hasattr(result, 'history') and result.history:
                    self.logger.debug(f"Episode had {len(result.history)} steps")
                    # Calculate total reward from history
                    total_reward = sum(step.reward for step in result.history if step.reward is not None)
                    self.logger.info(f"Total episode reward: {total_reward}")
                    self.logger.debug(f"Episode reached terminal state: {result.reach_terminal_state}")
                    self.logger.debug(f"Actual steps taken: {result.actual_num_steps}")
            else:
                self.logger.warning(f"Episode {self.episode_id} returned None result")
                
        except Exception as e:
            self.logger.error(f"Error running episode {self.episode_id}: {e}")
            self.logger.exception("Full exception details:")
            result = None
        finally:
            # Restore random state
            self.logger.debug("Restoring random state")
            np.random.set_state(state)
            self.logger.debug("Random state restored")
            
        return result
    
    def __eq__(self, other: 'EpisodeSimulationTask') -> bool:
        """Check if two tasks are equal."""
        if not isinstance(other, EpisodeSimulationTask):
            return False
        return self._cache_key == other._cache_key
    
    def __hash__(self) -> int:
        """Generate hash for the task."""
        return hash(self._cache_key)
