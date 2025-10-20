from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.simulations.simulations_deployment.task_manager_configs import JoblibConfig
from POMDPPlanners.simulations.simulations_deployment.tasks.episode_simulation_task import (
    EpisodeSimulationTask,
)
import tempfile
from pathlib import Path

temp_dir = Path("./pacman_pomcp")

with POMDPSimulator(
    task_manager_config=JoblibConfig(n_jobs=1),
    cache_dir_path=temp_dir,
    experiment_name=f"Test_Experiment_pacman_pomcp",
    debug=False,
    enable_profiling=False,
) as simulator:
    # Do some basic operations
    env = PacManPOMDP(discount_factor=0.95)
    policy = POMCP(
        environment=env,
        discount_factor=0.95,
        depth=5,
        exploration_constant=1.0,
        name="TestPolicy",
        n_simulations=1000,
    )

    # Test some simulator methods
    belief = get_initial_belief(env, n_particles=10)
    env_params = [
        EnvironmentRunParams(
            environment=env, belief=belief, policies=[policy], num_episodes=2, num_steps=3
        )
    ]

    # Run a small simulation
    results, _ = simulator.compare_multiple_environments_policies(
        environment_run_params=env_params,
        alpha=0.1,
        n_jobs=-1,
        cache_visualizations=True,  # Disable to isolate context manager
    )

    print("Simulation completed successfully!")
    print(f"Results: {len(results)} environment-policy pairs")
