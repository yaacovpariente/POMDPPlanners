import os
import importlib
import inspect
from pathlib import Path

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief, get_initial_belief
from POMDPPlanners.simulations.simulations import simulation

class SimulationsAPI:
    def __init__(self):
        pass
    
    def run_simulation(
        self, 
        environment: Environment, 
        policy: Policy, 
        initial_belief: Belief, 
        discount_factor: float, 
        num_episodes: int, 
        num_steps: int,
        alpha: float,
        confidence_interval_level: float,
        cache_dir_path: Path = None
    ):
        assert isinstance(environment, Environment)
        assert isinstance(policy, Policy)
        assert isinstance(initial_belief, Belief)
        assert isinstance(discount_factor, float)
        assert isinstance(num_episodes, int)
        assert isinstance(num_steps, int)
        assert isinstance(alpha, float)
        assert isinstance(confidence_interval_level, float)
        assert isinstance(cache_dir_path, Path)
        
        assert num_episodes > 0
        assert num_steps > 0
        assert 1 >= alpha >= 0
        assert 1 >= confidence_interval_level >= 0
        
        return simulation(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            discount_factor=discount_factor,
            num_episodes=num_episodes,
            num_steps=num_steps,
            alpha=alpha,
            confidence_interval_level=confidence_interval_level,
            cache_dir_path=cache_dir_path
        )
        
    def evaluate_policy_over_all_environments(
        self, 
        policy: Policy, 
        discount_factor: float, 
        n_particles: int,
        num_episodes: int, 
        num_steps: int,
        alpha: float,
        confidence_interval_level: float,
        cache_dir_path: Path = None
    ):
        assert isinstance(policy, Policy)
        assert isinstance(discount_factor, float)
        assert isinstance(n_particles, int)
        assert isinstance(num_episodes, int)
        assert isinstance(num_steps, int)
        assert isinstance(alpha, float)
        assert isinstance(confidence_interval_level, float)
        assert isinstance(cache_dir_path, Path)
        
        assert num_episodes > 0
        assert num_steps > 0
        assert n_particles > 0
        assert 1 >= alpha >= 0
        assert 1 >= confidence_interval_level >= 0

        environment_classes = get_all_environment_classes()
        
        for environment_class in environment_classes:
            environment = environment_class()
            initial_belief = get_initial_belief(pomdp=environment, n_particles=n_particles)
            
            self.run_simulation(
                environment=environment, 
                policy=policy, 
                initial_belief=initial_belief, 
                discount_factor=discount_factor, 
                num_episodes=num_episodes, 
                num_steps=num_steps,
                alpha=alpha,
                confidence_interval_level=confidence_interval_level,
                cache_dir_path=cache_dir_path
            )


def get_all_environment_classes():
    # Get the directory containing the environment files
    env_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all Python files in the directory (excluding __init__.py)
    env_files = [f[:-3] for f in os.listdir(env_dir) 
                if f.endswith('.py') and f != '__init__.py']
    
    env_classes = []
    
    # Import each module and find Environment subclasses
    for module_name in env_files:
        # Import the module
        module = importlib.import_module(f'POMDPPlanners.environments.{module_name}')
        
        # Get all classes defined in the module
        for name, obj in inspect.getmembers(module):
            # Check if it's a class and a subclass of Environment (but not Environment itself)
            if (inspect.isclass(obj) and 
                issubclass(obj, Environment) and 
                obj != Environment):
                env_classes.append(obj)
    
    return env_classes