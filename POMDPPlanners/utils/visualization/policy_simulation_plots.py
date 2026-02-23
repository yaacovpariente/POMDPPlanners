"""Policy simulation and returns plotting utilities.

This module provides functions for simulating agent paths and visualizing
the distribution of returns for different policy trajectories.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.cost import belief_expectation_cost_particle_belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.utils.visualization.plot_utils import _log_or_print

matplotlib.use("Agg")  # Use non-interactive backend

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class AgentPath:
    """Data class to store agent path."""

    name: str
    state_sequence: List[Any]
    action_sequence: List[Any]
    n_particles: int


def _validate_plot_policy_returns_inputs(
    env: Environment,
    agent_paths: List[AgentPath],
    dir_path: Path,
    n_samples: int,
    n_jobs: int,
    logger: Optional[logging.Logger],
) -> None:
    """Validate inputs for plot_policy_returns function."""
    # Type validation
    if not isinstance(env, Environment):
        raise TypeError("env must be an instance of Environment")
    if not isinstance(agent_paths, list):
        raise TypeError("agent_paths must be a list")
    if not isinstance(dir_path, Path):
        raise TypeError("dir_path must be a Path object")
    if not isinstance(n_samples, int):
        raise TypeError("n_samples must be an integer")
    if not isinstance(n_jobs, int):
        raise TypeError("n_jobs must be an integer")
    if logger is not None and not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger instance or None")

    # Value validation
    if not agent_paths:
        raise ValueError("agent_paths cannot be empty")
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0")
    if n_jobs < -1:
        raise ValueError("n_jobs must be -1 or greater")


def _validate_agent_path(path: AgentPath, index: int) -> None:
    """Validate a single AgentPath object."""
    if not isinstance(path, AgentPath):
        raise TypeError(f"agent_paths[{index}] must be an instance of AgentPath")
    if not isinstance(path.name, str):
        raise TypeError(f"agent_paths[{index}].name must be a string")
    if not isinstance(path.state_sequence, list):
        raise TypeError(f"agent_paths[{index}].state_sequence must be a list")
    if not isinstance(path.action_sequence, list):
        raise TypeError(f"agent_paths[{index}].action_sequence must be a list")
    if not isinstance(path.n_particles, int):
        raise TypeError(f"agent_paths[{index}].n_particles must be an integer")
    if path.n_particles <= 0:
        raise ValueError(f"agent_paths[{index}].n_particles must be greater than 0")
    if len(path.state_sequence) != len(path.action_sequence):
        raise ValueError(f"agent_paths[{index}] has mismatched state and action sequence lengths")


def _test_environment_reward_function(
    env: Environment, agent_paths: List[AgentPath], logger: Optional[logging.Logger]
) -> None:
    """Test environment reward function with first path to catch issues early."""
    if not agent_paths:
        return

    test_state = agent_paths[0].state_sequence[0]
    test_action = agent_paths[0].action_sequence[0]
    try:
        test_reward = env.reward(test_state, test_action)
        if not np.isfinite(test_reward):
            _log_or_print(
                logger, f"Environment reward function returned invalid value: {test_reward}"
            )
        elif abs(test_reward) > 1e6:
            _log_or_print(
                logger, f"Environment reward function returned extreme value: {test_reward}"
            )
    except Exception as e:  # pylint: disable=broad-exception-caught
        _log_or_print(logger, f"Environment reward function failed: {e}")


def _validate_and_clip_reward(
    reward: float, path_name: str, step: Optional[int], logger: Optional[logging.Logger]
) -> float:
    """Validate reward value and clip if necessary."""
    step_info = f" at step {step}" if step is not None else ""

    if not np.isfinite(reward):
        _log_or_print(logger, f"Invalid reward {reward}{step_info} for path {path_name}")
        return 0.0
    if abs(reward) > 1e6:
        _log_or_print(logger, f"Extreme reward {reward}{step_info} for path {path_name}")
        return float(np.clip(reward, -1e6, 1e6))

    return reward


def _validate_returns_range(
    returns: np.ndarray, path_name: str, logger: Optional[logging.Logger]
) -> bool:
    """Validate returns have reasonable range. Returns True if valid, False otherwise."""
    if len(returns) == 0:
        return False

    min_val = float(np.min(returns))
    max_val = float(np.max(returns))

    # If the range is extremely large or small, skip plotting
    if max_val - min_val > 1e6 or (max_val - min_val < 1e-10 and max_val - min_val > 0):
        _log_or_print(
            logger,
            f"Returns for {path_name} have extreme range [{min_val}, {max_val}]. Skipping this path.",
        )
        return False

    return True


def plot_policy_returns(
    env: Environment,
    agent_paths: List[AgentPath],
    dir_path: Path,
    n_samples: int = 1000,
    n_jobs: int = -1,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Simulate and plot returns for multiple agent paths.

    Args:
        env: POMDP environment
        agent_paths: List of AgentPath objects containing path information
        dir_path: Directory path to save the plot
        n_samples: Number of simulations to run for each path
        n_jobs: Number of parallel jobs to run (-1 for all cores)
        logger: Logger instance for logging warnings and info messages

    Raises:
        ValueError: If any of the input parameters are invalid
        TypeError: If any of the input parameters are of incorrect type
    """
    # Input validation
    _validate_plot_policy_returns_inputs(env, agent_paths, dir_path, n_samples, n_jobs, logger)

    # Validate each agent path
    for i, path in enumerate(agent_paths):
        _validate_agent_path(path, i)

    # Create directory if it doesn't exist
    dir_path.mkdir(parents=True, exist_ok=True)

    # Test environment reward function with first path to catch issues early
    _test_environment_reward_function(env, agent_paths, logger)

    def simulate_sequence(agent_path: AgentPath):
        total_reward: float = 0.0

        for i, _ in enumerate(agent_path.action_sequence):
            # Create a weighted particle belief centered on the current state
            particles = [
                agent_path.state_sequence[i]
            ] * agent_path.n_particles  # Two identical particles
            log_weights = np.log(
                np.array(np.ones(agent_path.n_particles) / agent_path.n_particles)
            )  # One with log(1), one with log(exp(-1))
            belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)

            # Use belief_expectation_cost to compute the reward
            step_reward = -belief_expectation_cost_particle_belief(
                belief=belief, action=agent_path.action_sequence[i], env=env
            )

            # Validate and clip step reward
            step_reward = _validate_and_clip_reward(step_reward, agent_path.name, i, logger)
            total_reward += step_reward

        # Final validation of total reward
        total_reward = _validate_and_clip_reward(total_reward, agent_path.name, None, logger)
        return total_reward

    def run_simulation(path_idx):
        return simulate_sequence(agent_paths[path_idx])

    # Run simulations in parallel
    all_returns = []
    for i in range(len(agent_paths)):
        returns = Parallel(n_jobs=n_jobs)(delayed(run_simulation)(i) for _ in range(n_samples))
        all_returns.append(returns)

    # Create the plot
    plt.figure(figsize=(10, 6))
    colors = [
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    valid_paths_plotted = 0
    for i, (returns, agent_path) in enumerate(zip(all_returns, agent_paths)):
        # Filter out invalid values and convert to numpy array
        returns_array = np.array(returns)

        # Remove NaN and inf values
        valid_mask = np.isfinite(returns_array)
        if not np.any(valid_mask):
            _log_or_print(
                logger,
                f"All returns for {agent_path.name} are invalid (NaN/inf). Skipping this path.",
            )
            continue

        valid_returns = returns_array[valid_mask]

        # Validate returns range
        if not _validate_returns_range(valid_returns, agent_path.name, logger):
            continue

        # Use explicit bins to prevent automatic bin calculation issues
        try:
            sns.histplot(
                data=valid_returns,
                label=agent_path.name,
                alpha=0.5,
                color=colors[i % len(colors)],
                bins=50,
            )
            valid_paths_plotted += 1
        except Exception as e:  # pylint: disable=broad-exception-caught
            _log_or_print(
                logger, f"Failed to plot histogram for {agent_path.name}: {e}. Skipping this path."
            )
            continue

    # Check if we have any valid paths to plot
    if valid_paths_plotted == 0:
        plt.text(
            0.5,
            0.5,
            "No valid data to plot",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,  # type: ignore[attr-defined]
        )
        _log_or_print(
            logger, "No valid data could be plotted. Check the simulation results for issues."
        )
    else:
        plt.xlabel("Total Reward")
        plt.ylabel("Count")
        plt.title("Comparison of Returns for Different Agent Paths")
        plt.legend()

    # Save the plot
    output_path = dir_path / "policy_returns_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory
