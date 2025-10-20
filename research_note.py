import numpy as np
from pathlib import Path

from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import NumericalHyperParameter
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams,
    HyperParameterOptimizationDirection,
    HyperParamPlannerConfigGenerator,
    HyperParamPlannerConfig,
)

from typing import Optional
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.core.policy import PolicySpaceInfo
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

# Set random seed for reproducibility
np.random.seed(42)

N_JOBS = 1


# Create a concrete implementation of HyperParamPlannerConfigGenerator
class PFT_DPW_Generator(HyperParamPlannerConfigGenerator):
    def __init__(
        self,
        discount_factor: float,
        depth: int,
        name: str,
        action_sampler: ActionSampler,
        max_exploration_constant: float = 1.0,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        min_samples_per_node: int = 10,
        min_visit_count_per_action: int = 1,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        self.discount_factor = discount_factor
        self.depth = depth
        self.name = name
        self.action_sampler = action_sampler
        self.max_exploration_constant = max_exploration_constant
        self.time_out_in_seconds = time_out_in_seconds
        self.n_simulations = n_simulations
        self.min_samples_per_node = min_samples_per_node
        self.min_visit_count_per_action = min_visit_count_per_action

    def generate(self, environment) -> HyperParamPlannerConfig:
        hyper_parameters = [
            NumericalHyperParameter(0, self.max_exploration_constant, "exploration_constant"),
            NumericalHyperParameter(1, 10, "k_o"),
            NumericalHyperParameter(0.01, 0.5, "alpha_o"),
        ]

        constant_parameters = {
            "discount_factor": self.discount_factor,
            "environment": environment,
            "name": self.name,
            "depth": self.depth,
            "action_sampler": self.action_sampler,
            "n_simulations": self.n_simulations,
            "min_samples_per_node": self.min_samples_per_node,
            "min_visit_count_per_action": self.min_visit_count_per_action,
        }

        return HyperParamPlannerConfig(
            policy_cls=PFT_DPW,
            hyper_parameters=hyper_parameters,
            constant_parameters=constant_parameters,
        )

    def get_planner_space_info(self):
        # Return a basic policy space info for discrete actions
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.DISCRETE,
        )


# Initialize API
api = LocalSimulationsAPI(debug=True)

# Create environment
tiger = TigerPOMDP(discount_factor=0.95)
initial_belief = get_initial_belief(tiger, n_particles=30)

# Create configuration generator for benchmark
gen = PFT_DPW_Generator(
    discount_factor=0.95,
    depth=10,
    name="PFT_DPW",
    action_sampler=DiscreteActionSampler(tiger.get_actions()),
    max_exploration_constant=100.0,
    time_out_in_seconds=None,
    n_simulations=2000,
)

# Run comprehensive benchmark
results, stats_df = api.run_hyperparameter_tuning_comprehensive_benchmark_local(
    generators=[gen],
    particles=30,
    num_episodes=10,  # Episodes for optimization
    num_steps=20,  # Steps for optimization
    optuna_n_jobs=3,  # Parallel jobs for Optuna
    n_trials=20,  # Optimization trials per planner-environment pair
    evaluation_episodes=100,  # Episodes for final evaluation
    evaluation_steps=50,  # Steps for final evaluation
    evaluation_n_jobs=N_JOBS,  # Parallel jobs for evaluation
    optimization_n_jobs=2,  # Use all cores for optimization
    confidence_interval_level=0.95,
    alpha=0.05,
    cache_dir_path=Path("./benchmark_cache2"),
    experiment_name="Comprehensive_Benchmark",
    debug=True,
    cache_visualizations=True,
)

# Display results
print("\nBenchmark Results:")
print(stats_df[["environment", "policy", "average_return", "std_return"]])
