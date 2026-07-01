# SPDX-License-Identifier: MIT

from pathlib import Path

from joblib import Parallel, delayed

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
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
    # The environment owns the output file name/extension; keep each planner's
    # visualizations separated by writing them into a per-planner subdirectory.
    planner_output_dir = cache_dir / planner.name
    planner_output_dir.mkdir(parents=True, exist_ok=True)
    environment.cache_visualization(
        history=episode_result.history,
        output_dir=planner_output_dir,
        episode_index=episode_id,
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
            _run_single_episode(episode_id, planner, environment, belief, num_steps, cache_dir)
    else:
        # Parallel execution
        Parallel(n_jobs=n_jobs)(
            delayed(_run_single_episode)(
                episode_id, planner, environment, belief, num_steps, cache_dir
            )
            for episode_id in range(n_episodes)
        )
