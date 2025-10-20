"""Simple script demonstrating benchmark evaluation across all environments."""

from pathlib import Path
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import (
    LocalSimulationsAPI,
)
from POMDPPlanners.core.policy import Policy, PolicySpaceInfo
from POMDPPlanners.core.environment import (
    Environment,
    DiscreteActionsEnvironment,
    SpaceType,
)
from POMDPPlanners.core.simulation.simulation_configs import PlannerGenerator
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler
import random


class DiscreteActionSampler(ActionSampler):
    """Action sampler for discrete action spaces."""

    def __init__(self, actions):
        self.actions = actions

    def sample(self, belief_node=None):
        """Sample a random action from the discrete action space."""
        return random.choice(self.actions)


class SimplePOMCPOWGenerator(PlannerGenerator):
    """Generator for POMCPOW planner with fixed hyperparameters."""

    def __init__(self, name: str = "POMCPOW"):
        self.planner_name = name

    def generate(self, environment: Environment) -> Policy:
        """Generate a POMCPOW policy for the given environment."""
        # Create action sampler for discrete actions
        if isinstance(environment, DiscreteActionsEnvironment):
            action_sampler = DiscreteActionSampler(environment.get_actions())
        else:
            action_sampler = UnitCircleActionSampler()

        return POMCPOW(
            environment=environment,
            discount_factor=0.95,
            depth=5,
            exploration_constant=1.0,
            k_o=3.0,
            k_a=3.0,
            alpha_o=0.5,
            alpha_a=0.5,
            action_sampler=action_sampler,
            name=f"{self.planner_name}_{environment.name}",
            n_simulations=50,
        )

    def get_planner_space_info(self) -> PolicySpaceInfo:
        """Return space info for discrete action spaces with mixed observations.

        Note: While POMCPOW can theoretically handle mixed spaces, this generator
        specifically requires DiscreteActionsEnvironment to get the action list.
        """
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.MIXED,
        )


def main():
    """Run benchmark evaluation on all compatible environments."""
    # Initialize API
    api = LocalSimulationsAPI(debug=True)

    # Create planner generators
    generators = [
        SimplePOMCPOWGenerator(name="POMCPOW_Config1"),
    ]

    # Run benchmarks
    results, stats_df = api.run_all_benchmark_environments_on_planner_generators(
        generators=generators,
        n_particles=30,
        num_episodes=5,  # Small number for quick testing
        num_steps=10,  # Small number for quick testing
        alpha=0.1,
        confidence_interval_level=0.95,
        experiment_name="Simple_Planner_Benchmark",
        n_jobs=1,
        cache_dir_path=Path("./benchmark_results"),
        enable_profiling=False,
        cache_visualizations=True,
    )

    # Display results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(stats_df)
    print("\n")


if __name__ == "__main__":
    main()
