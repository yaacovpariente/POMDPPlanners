from pathlib import Path

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.utils.logger import get_logger


def visualize_planner_episode(
    planner: Policy,
    environment: Environment,
    policy: Policy,
    n_episodes: int,
    cache_dir: Path,
    num_steps: int = 20,
    n_particles: int = 1000
):
    """
    Visualize episodes of a planner by running episodes and caching visualizations.
    
    Args:
        planner: The planner policy (used for naming cache files)
        environment: The POMDP environment to run episodes in
        policy: The policy to use for action selection during episodes
        n_episodes: Number of episodes to run and visualize
        cache_dir: Directory to cache visualization files
        num_steps: Maximum number of steps per episode (default: 20)
        n_particles: Number of particles for initial belief (default: 1000)
    """
    # Set up logger and initial belief
    logger = get_logger("episode_visualization", debug=False)
    initial_belief = get_initial_belief(environment, n_particles=n_particles)
    
    # Run the episodes
    for episode_id in range(n_episodes):
        episode_result = run_episode(
            environment=environment,
            policy=policy,
            initial_belief=initial_belief,
            num_steps=num_steps,
            logger=logger
        )
        cache_path = cache_dir / f"{planner.name}_{episode_id}.gif"
        
        # Visualize the episode
        environment.cache_visualization(history=episode_result, cache_path=cache_path)