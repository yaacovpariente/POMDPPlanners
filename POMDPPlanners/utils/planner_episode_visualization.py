from pathlib import Path

from joblib import Parallel, delayed

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.belief import Belief
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.utils.logger import get_logger


def _run_single_episode(
    episode_id: int,
    planner: Policy,
    environment: Environment,
    belief: Belief,
    num_steps: int,
    cache_dir: Path,
):
    """Helper function to run a single episode and cache its visualization."""
    logger = get_logger("episode_visualization", debug=False)

    episode_result = run_episode(
        environment=environment,
        policy=planner,
        initial_belief=belief,
        num_steps=num_steps,
        logger=logger,
    )
    cache_path = cache_dir / f"{planner.name}_{episode_id}.gif"

    # Visualize the episode
    environment.cache_visualization(
        history=episode_result.history, cache_path=cache_path
    )


def visualize_planner_episode(
    planner: Policy,
    environment: Environment,
    belief: Belief,
    n_episodes: int,
    cache_dir: Path,
    num_steps: int = 20,
    n_jobs: int = 1,
):
    """
    Visualize episodes of a planner by running episodes and caching visualizations.

    Args:
        planner: The planner policy (used for naming cache files)
        environment: The POMDP environment to run episodes in
        belief: The initial belief to use for episodes
        n_episodes: Number of episodes to run and visualize
        cache_dir: Directory to cache visualization files
        num_steps: Maximum number of steps per episode (default: 20)
        n_jobs: Number of parallel jobs for episode execution (default: 1, sequential)
    """
    # Run episodes either sequentially or in parallel based on n_jobs
    if n_jobs == 1:
        # Sequential execution
        for episode_id in range(n_episodes):
            _run_single_episode(
                episode_id, planner, environment, belief, num_steps, cache_dir
            )
    else:
        # Parallel execution
        Parallel(n_jobs=n_jobs)(
            delayed(_run_single_episode)(
                episode_id, planner, environment, belief, num_steps, cache_dir
            )
            for episode_id in range(n_episodes)
        )
