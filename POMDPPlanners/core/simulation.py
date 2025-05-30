from typing import Any, NamedTuple, Union, TYPE_CHECKING, Optional, List, Dict
from dataclasses import dataclass
from typing import List
import numpy as np
from pathlib import Path
import dask
from dask.distributed import Future
from dask.distributed import Client, LocalCluster
from dask.cache import Cache


if TYPE_CHECKING:
    from POMDPPlanners.core.belief import Belief
    from POMDPPlanners.core.policy import PolicyRunData

    
class StepData(NamedTuple):
    state: Any
    action: Any
    next_state: Any
    observation: Any
    reward: float
    belief: 'Belief'


@dataclass(frozen=True)
class History:
    history: List[StepData]
    discount_factor: float
    average_state_sampling_time: float
    average_action_time: float
    average_observation_time: float
    average_belief_update_time: float
    average_reward_time: float
    actual_num_steps: int
    reach_terminal_state: bool
    policy_run_data: 'PolicyRunData'

    def to_dict(self) -> dict:
        """Convert History object to dictionary."""
        history_data = []
        for step in self.history:
            step_dict = step._asdict()
            if hasattr(step_dict['belief'], 'to_dict'):
                belief_dict = step_dict['belief'].to_dict()
                belief_dict['type'] = step_dict['belief'].__class__.__name__
                step_dict['belief'] = belief_dict
            history_data.append(step_dict)

        return {
            'history': history_data,
            'discount_factor': self.discount_factor,
            'average_state_sampling_time': self.average_state_sampling_time,
            'average_action_time': self.average_action_time,
            'average_observation_time': self.average_observation_time,
            'average_belief_update_time': self.average_belief_update_time,
            'average_reward_time': self.average_reward_time,
            'actual_num_steps': self.actual_num_steps,
            'reach_terminal_state': self.reach_terminal_state
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'History':
        """Create a History instance from a dictionary.
        
        Args:
            data: Dictionary containing History data
            
        Returns:
            History: New History instance
        """
        # Convert history list of dictionaries back to StepData objects
        history = []
        for step_data in data['history']:
            # Handle belief deserialization
            if isinstance(step_data['belief'], dict) and 'type' in step_data['belief']:
                belief_type = step_data['belief']['type']
                # Import the belief class dynamically
                if belief_type == 'WeightedParticleBelief':
                    from POMDPPlanners.core.belief import WeightedParticleBelief
                    step_data['belief'] = WeightedParticleBelief(
                        particles=step_data['belief']['particles'],
                        log_weights=np.array(step_data['belief']['log_weights']),
                        resampling=step_data['belief'].get('resampling', False)
                    )
            history.append(StepData(**step_data))

        # Handle policy_run_data deserialization
        policy_run_data = data.get('policy_run_data', None)
        if isinstance(policy_run_data, dict):
            from POMDPPlanners.core.policy import PolicyRunData, PolicyInfoVariable
            info_variables = [
                PolicyInfoVariable(name=iv['name'], value=iv['value'])
                for iv in policy_run_data.get('info_variables', [])
            ]
            policy_run_data = PolicyRunData(info_variables=info_variables)

        # Create and return History instance
        return History(
            history=history,
            discount_factor=data['discount_factor'],
            average_state_sampling_time=data['average_state_sampling_time'],
            average_action_time=data['average_action_time'],
            average_observation_time=data['average_observation_time'],
            average_belief_update_time=data['average_belief_update_time'],
            average_reward_time=data['average_reward_time'],
            actual_num_steps=data['actual_num_steps'],
            reach_terminal_state=data['reach_terminal_state'],
            policy_run_data=policy_run_data
        )


class CategoricalHyperParameter(NamedTuple):
    choices: list[Any]
    name: str


class NumericalHyperParameter(NamedTuple):
    low: Union[int, float]
    high: Union[int, float]
    name: str


class MetricValue(NamedTuple):
    name: str
    value: float
    lower_confidence_bound: float
    upper_confidence_bound: float


class EnvironmentRunParams(NamedTuple):
    environment: Any  # Use Any to avoid circular import
    belief: Any
    policies: list[Any]
    num_episodes: int
    num_steps: int


@dask.delayed(pure=True)
def _run_simulation_task(
    environment: Any,
    policy: Any,
    initial_belief: Any,
    num_steps: int,
    seed: int,
    episode_number: int
) -> History:
    """Pure function to run a simulation task.
    
    Args:
        environment: The environment to simulate
        policy: The policy to use
        initial_belief: The initial belief state
        num_steps: Number of steps to simulate
        seed: Random seed for reproducibility
        episode_number: The episode number for this simulation
        
    Returns:
        History: The simulation history
    """
    import numpy as np
    from POMDPPlanners.simulations.simulations import run_episode
    
    # Set random seed for reproducibility
    state = np.random.get_state()
    np.random.seed(seed)
    
    try:
        # Run simulation
        return run_episode(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_steps=num_steps
        )
    finally:
        # Restore random state
        np.random.set_state(state)

class SimulationTask:
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
        episode_number: int = 0
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
    
    def _generate_cache_key(self) -> str:
        """Generate a unique cache key for this task."""
        import hashlib
        import json
        
        components = {
            'env': hash(self.environment),
            'policy': hash(self.policy),
            'belief': hash(self.initial_belief),
            'episode_id': self.episode_id,
            'episode_number': self.episode_number,
            'num_steps': self.num_steps,
            'seed': self.seed,
            'discount_factor': self.discount_factor
        }
        return f"simulation:{hashlib.md5(json.dumps(components, sort_keys=True).encode()).hexdigest()}"
    
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
    def from_dict(cls, data: dict) -> 'SimulationTask':
        """Create a SimulationTask instance from a dictionary."""
        return cls(**data)
    
    def run(self) -> History:
        """Run the simulation task.
        
        Returns:
            History: The simulation history
        """
        return _run_simulation_task(
            environment=self.environment,
            policy=self.policy,
            initial_belief=self.initial_belief,
            num_steps=self.num_steps,
            seed=self.seed,
            episode_number=self.episode_number
        )
    
    def __eq__(self, other: 'SimulationTask') -> bool:
        """Check if two tasks are equal."""
        if not isinstance(other, SimulationTask):
            return False
        return self._cache_key == other._cache_key
    
    def __hash__(self) -> int:
        """Generate hash for the task."""
        return hash(self._cache_key)

def history_to_discounted_return_value(history: History) -> float:
    return sum(step.reward * history.discount_factor ** i for i, step in enumerate(history.history) if step.reward is not None)

class TaskManager:
    """A class to manage simulation tasks and their execution."""
    
    def __init__(
        self,
        n_workers: int = None,
        scheduler_address: Optional[str] = None,
        cache_size: int = 2e9  # 2GB default cache size
    ):
        """Initialize the task manager.
        
        Args:
            n_workers: Number of worker processes (None for auto)
            scheduler_address: Address of Dask scheduler (None for local)
            cache_size: Size of cache in bytes
        """
        self.scheduler_address = scheduler_address
        self.n_workers = n_workers
        self.cache_size = cache_size
        self.client = None
        self.cache = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Dask client and cache."""
        
        if self.scheduler_address:
            # Connect to existing cluster
            self.client = Client(self.scheduler_address)
        else:
            # Create local cluster
            cluster = LocalCluster(n_workers=self.n_workers)
            self.client = Client(cluster)
        
        # Initialize cache
        self.cache = Cache(self.cache_size)
        self.cache.register()
    
    def submit_tasks(self, tasks: List[SimulationTask]) -> List[Future]:
        """Submit tasks for execution.
        
        Args:
            tasks: List of simulation tasks to execute
            
        Returns:
            List[Future]: List of Dask futures for the tasks
        """
        if not self.client:
            self._initialize_client()
            
        # Submit all tasks
        futures = []
        for task in tasks:
            future = self.client.submit(
                task.run,
                key=task._cache_key  # Use task's cache key for Dask caching
            )
            futures.append(future)
        
        return futures
    
    def gather_results(self, futures: List[Future]) -> List[History]:
        """Gather results from submitted tasks.
        
        Args:
            futures: List of Dask futures to gather results from
            
        Returns:
            List[History]: List of simulation histories
        """
        if not self.client:
            raise RuntimeError("No Dask client available")
            
        return self.client.gather(futures)
    
    def run_tasks(self, tasks: List[SimulationTask]) -> List[History]:
        """Submit and gather results for tasks.
        
        Args:
            tasks: List of simulation tasks to execute
            
        Returns:
            List[History]: List of simulation histories
        """
        futures = self.submit_tasks(tasks)
        return self.gather_results(futures)
    
    def get_task_status(self, futures: List[Future]) -> Dict[str, str]:
        """Get status of submitted tasks.
        
        Args:
            futures: List of Dask futures to check
            
        Returns:
            Dict[str, str]: Dictionary mapping task keys to their status
        """
        if not self.client:
            raise RuntimeError("No Dask client available")
            
        return {
            future.key: future.status
            for future in futures
        }
    
    def cancel_tasks(self, futures: List[Future]):
        """Cancel submitted tasks.
        
        Args:
            futures: List of Dask futures to cancel
        """
        if not self.client:
            raise RuntimeError("No Dask client available")
            
        for future in futures:
            self.client.cancel(future)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.client:
            self.client.close()
            self.client = None
        if self.cache:
            self.cache.unregister()
            self.cache = None
