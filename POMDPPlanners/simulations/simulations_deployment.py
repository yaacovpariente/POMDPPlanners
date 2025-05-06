from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List
from pathlib import Path
import redis
import json
import os
import subprocess

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.utils.distributed_computing import run_parallel_locally, run_distributed
from POMDPPlanners.core.simulation import History
from POMDPPlanners.utils.simulations_caching import cache_episode_simulation_results, load_episode_simulation_results

class DeploymentType(Enum):
    LOCAL = "local"
    REMOTE_RAY = "remote_ray"
    REMOTE_RAY_MULTI_GPU = "remote_ray_multi_gpu"
    
class SimulationDeployment(ABC):
    @abstractmethod
    def run_multiple_episodes(self, func: Callable, episode_configs: List[dict]):
        pass

    @abstractmethod
    def run_multiple_policies(self, func: Callable, policy_configs: List[dict]):
        pass

    @abstractmethod
    def load_episode_simulation_results(
        self, 
        environment: Environment,
        policy: Policy,
        initial_belief: Belief,
        cache_dir_path: Path,
        general_config: dict
    ) -> List[History]:
        pass

    @abstractmethod
    def save_episode_simulation_results(
        self, 
        environment: Environment,
        policy: Policy,
        initial_belief: Belief,
        results: List[History],
        cache_dir_path: Path,
        general_config: dict
    ) -> None:
        pass
    
class LocalSimulationDeployment(SimulationDeployment):
    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    def run_multiple_episodes(self, func: Callable, episode_configs: List[dict]):
        return run_parallel_locally(func=func, kwargs_list=episode_configs, n_jobs=self.n_jobs)
    
    def run_multiple_policies(self, func: Callable, policy_configs: List[dict]):
        return run_parallel_locally(func=func, kwargs_list=policy_configs, n_jobs=self.n_jobs)

    def load_episode_simulation_results(
        self, 
        environment: Environment,
        policy: Policy,
        initial_belief: Belief,
        cache_dir_path: Path,
        general_config: dict
    ) -> List[History]:
        return load_episode_simulation_results(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            cache_dir_path=cache_dir_path,
            general_config=general_config,
        )

    def save_episode_simulation_results(self, environment: Environment,
        policy: Policy,
        initial_belief: Belief,
        results: List[History],
        cache_dir_path: Path,
        general_config: dict
    ) -> None:
        cache_episode_simulation_results(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            results=results,
            cache_dir_path=cache_dir_path,
            general_config=general_config,
        )

class RemoteRaySimulationDeployment(SimulationDeployment):
    def __init__(self, num_cpus: int = 1, num_gpus: int = 0, redis_host: str = 'localhost', redis_port: int = 6379, redis_db_path: str = None):
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.redis_client = self._initialize_redis(redis_host, redis_port, redis_db_path)
        self.redis_process = None  # Store the Redis process reference

    def _initialize_redis(self, host: str, port: int, db_path: str = None) -> redis.Redis:
        """
        Initialize Redis server and client with custom database path.
        
        Args:
            host: Redis server host
            port: Redis server port
            db_path: Path where Redis database will be stored
            
        Returns:
            redis.Redis: Initialized Redis client
        """
        # Set up Redis database path
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'redis_data')
        os.makedirs(db_path, exist_ok=True)
        
        # Configure Redis to use the specified path
        redis_config = {
            'dir': db_path,
            'dbfilename': 'dump.rdb'
        }
        
        # Create Redis configuration file
        config_path = os.path.join(db_path, 'redis.conf')
        with open(config_path, 'w') as f:
            for key, value in redis_config.items():
                f.write(f"{key} {value}\n")
        
        # Start Redis server with the custom configuration
        try:
            self.redis_process = subprocess.Popen(['redis-server', config_path], 
                                                stdout=subprocess.DEVNULL,
                                                stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("Warning: Redis server not found. Make sure Redis is installed and in your PATH.")
        
        # Initialize and return Redis client
        return redis.Redis(
            host=host,
            port=port,
            db=0,
            decode_responses=True
        )

    def __del__(self):
        """Cleanup method to stop Redis server and close client when object is destroyed."""
        self.cleanup()

    def cleanup(self):
        """Close Redis client and stop Redis server."""
        if self.redis_client is not None:
            try:
                self.redis_client.close()
            except Exception as e:
                print(f"Error closing Redis client: {e}")
            self.redis_client = None

        if self.redis_process is not None:
            try:
                # Send SIGTERM to Redis process
                self.redis_process.terminate()
                # Wait for process to terminate (with timeout)
                self.redis_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # If process doesn't terminate gracefully, force kill it
                self.redis_process.kill()
            except Exception as e:
                print(f"Error stopping Redis server: {e}")
            self.redis_process = None

    def run_multiple_episodes(self, func: Callable, episode_configs: List[dict]):
        try:
            return run_distributed(func=func, kwargs_list=episode_configs, num_cpus=self.num_cpus, num_gpus=self.num_gpus)
        finally:
            self.cleanup()

    def run_multiple_policies(self, func: Callable, policy_configs: List[dict]):
        try:
            return run_distributed(func=func, kwargs_list=policy_configs, num_cpus=self.num_cpus, num_gpus=self.num_gpus)
        finally:
            self.cleanup()

    def _generate_cache_key(self, environment: Environment, policy: Policy, initial_belief: Belief, general_config: dict) -> str:
        """Generate a unique cache key based on the simulation parameters."""
        # Create a unique identifier using the relevant parameters
        key_components = {
            'env': environment.__class__.__name__,
            'policy': policy.__class__.__name__,
            'belief': initial_belief.__class__.__name__,
            'config': general_config
        }
        return f"simulation:{json.dumps(key_components, sort_keys=True)}"

    def load_episode_simulation_results(
        self, 
        environment: Environment,
        policy: Policy,
        initial_belief: Belief,
        cache_dir_path: Path,
        general_config: dict
    ) -> List[History]:
        cache_key = self._generate_cache_key(environment, policy, initial_belief, general_config)
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data is None:
            return []
            
        try:
            # Deserialize the cached data back into History objects
            histories_data = json.loads(cached_data)
            return [History.from_dict(h) for h in histories_data]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading cached results: {e}")
            return []

    def save_episode_simulation_results(
        self, 
        environment: Environment,
        policy: Policy,
        initial_belief: Belief,
        results: List[History],
        cache_dir_path: Path,
        general_config: dict
    ) -> None:
        cache_key = self._generate_cache_key(environment, policy, initial_belief, general_config)
        
        try:
            # Convert History objects to dictionaries for serialization
            histories_data = [h.to_dict() for h in results]
            
            # Serialize and store in Redis
            serialized_data = json.dumps(histories_data)
            self.redis_client.set(cache_key, serialized_data)
        except Exception as e:
            print(f"Error saving results to Redis: {e}")
