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
        self._cache_key = self._generate_cache_key()
        self.debug = debug
        self.console_output = console_output
        self.cache_dir = cache_dir
    
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
        
        # Get configuration IDs or string representations of objects
        env_id = getattr(self.environment, 'config_id', str(self.environment))
        policy_id = getattr(self.policy, 'config_id', str(self.policy))
        belief_id = getattr(self.initial_belief, 'config_id', str(self.initial_belief))
        
        components = {
            'env': env_id,
            'policy': policy_id,
            'belief': belief_id,
            'episode_id': self.episode_id,
            'episode_number': self.episode_number,
            'num_steps': self.num_steps,
            'seed': self.seed,
            'discount_factor': self.discount_factor
        }
        return f"simulation:{hashlib.md5(json.dumps(components, sort_keys=True).encode()).hexdigest()}"
    
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
        state = np.random.get_state()
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        try:
            # Run simulation with seed parameter
            result = run_episode(
                environment=self.environment,
                policy=self.policy,
                initial_belief=self.initial_belief,
                num_steps=self.num_steps,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f"Error running episode {self.episode_id}: {e}")
            result = None
        finally:
            # Restore random state
            np.random.set_state(state)
            
        return result
    
    def __eq__(self, other: 'EpisodeSimulationTask') -> bool:
        """Check if two tasks are equal."""
        if not isinstance(other, EpisodeSimulationTask):
            return False
        return self._cache_key == other._cache_key
    
    def __hash__(self) -> int:
        """Generate hash for the task."""
        return hash(self._cache_key)
