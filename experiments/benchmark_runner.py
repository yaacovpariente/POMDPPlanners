"""Benchmark runner for comprehensive POMDP planner evaluation.

This script orchestrates hyperparameter optimization and evaluation across all
available POMDP environments and their compatible planners. It automates the
process of discovering environments, identifying compatible planners, and
running comprehensive benchmarks.
"""

from pathlib import Path
from typing import List, Tuple

from POMDPPlanners.configs.environment_configs import get_all_environments
from POMDPPlanners.configs.planners_hyperparam_configs import PlannersHyperparamConfigs
from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
)
from POMDPPlanners.configs.experiment_configs import AverageReturnParameterToOptimizeMapper
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
from POMDPPlanners.utils.hyperparameter_tuning_and_eval import HyperParamPlannerConfig
from POMDPPlanners.utils.logger import get_logger


class BenchmarkRunner:
    """Orchestrates comprehensive benchmarking across all environments and planners.

    This class manages the process of discovering all available POMDP environments,
    identifying compatible planners for each environment, and running hyperparameter
    optimization followed by comprehensive evaluation.
    """

    def __init__(
        self,
        cache_dir: Path = Path("./experiments/benchmark_results"),
        discount_factor: float = 0.95,
        n_particles: int = 100,
        include_risk_averse: bool = True,
        debug: bool = False,
    ):
        """Initialize the benchmark runner.

        Args:
            cache_dir: Directory for storing benchmark results and artifacts
            discount_factor: Discount factor for future rewards
            n_particles: Number of particles for belief initialization
            include_risk_averse: Whether to include risk-averse environment variants
            debug: Whether to enable debug-level logging
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.discount_factor = discount_factor
        self.n_particles = n_particles
        self.include_risk_averse = include_risk_averse

        self.logger = get_logger(name="benchmark_runner", output_dir=cache_dir, debug=debug)
        self.simulations_api = LocalSimulationsAPI(cache_dir_path=cache_dir, debug=debug)
        self.planner_configs_api = PlannersHyperparamConfigs(discount_factor=discount_factor)

    def _get_all_environments_with_beliefs(
        self,
    ) -> List[Tuple[Environment, WeightedParticleBelief]]:
        """Get all available environments with their initial beliefs.

        Returns:
            List of (environment, belief) tuples for all available environments
        """
        self.logger.info(
            f"Discovering all environments (include_risk_averse={self.include_risk_averse})"
        )
        environments = get_all_environments(
            n_particles=self.n_particles, include_risk_averse=self.include_risk_averse
        )
        self.logger.info(f"Found {len(environments)} environments")
        return environments

    def _get_compatible_planners_for_environment(
        self, env: Environment, time_out_in_seconds: float = 3.0
    ) -> List[HyperParamPlannerConfig]:
        """Get all planners compatible with the given environment.

        Args:
            env: Environment to find compatible planners for
            time_out_in_seconds: Time limit for each planner

        Returns:
            List of hyperparameter planner configurations compatible with the environment
        """
        self.logger.info(
            f"Finding compatible planners for environment: {env.name} "
            f"(action_space={env.space_info.action_space}, "
            f"observation_space={env.space_info.observation_space})"
        )

        compatible_planners = self.planner_configs_api.get_compatible_planners(
            env=env, time_out_in_seconds=time_out_in_seconds
        )

        self.logger.info(f"Found {len(compatible_planners)} compatible planners for {env.name}")
        for planner_config in compatible_planners:
            self.logger.debug(f"  - {planner_config.policy_cls.__name__}")

        return compatible_planners

    def run_benchmark(
        self,
        optimization_episodes: int = 10,
        optimization_steps: int = 20,
        n_trials: int = 50,
        optimization_n_jobs: int = -1,
        evaluation_episodes: int = 100,
        evaluation_steps: int = 50,
        evaluation_n_jobs: int = 1,
        confidence_interval_level: float = 0.95,
        alpha: float = 0.05,
        time_out_in_seconds: float = 3.0,
        verbose: bool = True,
    ):
        """Run comprehensive benchmark across all environments and planners.

        This method orchestrates the complete benchmarking workflow:
        1. Discover all available environments
        2. For each environment, find compatible planners
        3. Run hyperparameter optimization and evaluation for each environment-planner pair

        Args:
            optimization_episodes: Number of episodes for optimization trials
            optimization_steps: Number of steps per optimization episode
            n_trials: Number of optimization trials per planner
            optimization_n_jobs: Number of parallel jobs for optimization
            evaluation_episodes: Number of episodes for evaluation
            evaluation_steps: Number of steps per evaluation episode
            evaluation_n_jobs: Number of parallel jobs for evaluation
            confidence_interval_level: Confidence level for statistical analysis
            alpha: Alpha value for risk metrics
            time_out_in_seconds: Time limit for each planner execution
            verbose: Whether to print progress messages
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Comprehensive POMDP Planner Benchmark")
        self.logger.info("=" * 80)

        # Get all environments
        environments = self._get_all_environments_with_beliefs()

        # Track overall benchmark progress
        total_benchmarks = 0
        successful_benchmarks = 0
        failed_benchmarks = []

        # Run benchmark for each environment
        for env_idx, (env, initial_belief) in enumerate(environments, 1):
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(f"Environment {env_idx}/{len(environments)}: {env.name}")
            self.logger.info("=" * 80)

            try:
                # Get compatible planners for this environment
                planner_configs = self._get_compatible_planners_for_environment(
                    env, time_out_in_seconds=time_out_in_seconds
                )

                if not planner_configs:
                    self.logger.warning(f"No compatible planners found for {env.name}. Skipping.")
                    continue

                # Create experiment name for this environment
                experiment_name = f"Benchmark_{env.name}"
                env_cache_dir = self.cache_dir / env.name
                env_cache_dir.mkdir(parents=True, exist_ok=True)

                self.logger.info(
                    f"Running optimization and evaluation for {env.name} "
                    f"with {len(planner_configs)} planners"
                )

                # Create hyperparameter run configurations
                parameter_mapper = AverageReturnParameterToOptimizeMapper()
                hyperparameter_configs = []

                for planner_config in planner_configs:
                    params_to_optimize = parameter_mapper.generate(env, planner_config.policy_cls)
                    hyperparameter_configs.append(
                        HyperParameterRunParams(
                            environment=env,
                            belief=initial_belief,
                            hyper_param_planner_config=planner_config,
                            num_episodes=optimization_episodes,
                            num_steps=optimization_steps,
                            n_trials=n_trials,
                            parameters_to_optimize=params_to_optimize,
                        )
                    )

                # Run hyperparameter optimization and evaluation
                results = self.simulations_api.run_optimize_and_evaluate(
                    configs=hyperparameter_configs,
                    evaluation_episodes=evaluation_episodes,
                    evaluation_steps=evaluation_steps,
                    evaluation_n_jobs=evaluation_n_jobs,
                    optimization_n_jobs=optimization_n_jobs,
                    confidence_interval_level=confidence_interval_level,
                    alpha=alpha,
                    cache_dir_path=env_cache_dir,
                    experiment_name=experiment_name,
                    debug=False,
                    cache_visualizations=True,
                )

                total_benchmarks += 1
                successful_benchmarks += 1

                self.logger.info(f"Successfully completed benchmark for {env.name}")

            except Exception as e:
                self.logger.error(
                    f"Benchmark failed for environment {env.name}: {e}",
                    exc_info=True,
                )
                failed_benchmarks.append((env.name, str(e)))
                total_benchmarks += 1

        # Print final summary
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Benchmark Summary")
        self.logger.info("=" * 80)
        self.logger.info(f"Total environments processed: {total_benchmarks}")
        self.logger.info(f"Successful benchmarks: {successful_benchmarks}")
        self.logger.info(f"Failed benchmarks: {len(failed_benchmarks)}")

        if failed_benchmarks:
            self.logger.warning("")
            self.logger.warning("Failed Benchmarks:")
            for env_name, error in failed_benchmarks:
                self.logger.warning(f"  - {env_name}: {error}")

        self.logger.info("")
        self.logger.info(f"All results saved to: {self.cache_dir}")
        self.logger.info("=" * 80)


def main():
    """Main entry point for benchmark runner."""
    # Configure benchmark parameters
    benchmark = BenchmarkRunner(
        cache_dir=Path("./benchmark_results"),
        discount_factor=0.95,
        n_particles=100,
        include_risk_averse=True,
        debug=False,
    )

    # Run comprehensive benchmark
    benchmark.run_benchmark(
        # Optimization parameters
        optimization_episodes=15,
        optimization_steps=20,
        n_trials=1000,
        optimization_n_jobs=-1,
        # Evaluation parameters
        evaluation_episodes=500,
        evaluation_steps=30,
        evaluation_n_jobs=-1,
        # Statistical parameters
        confidence_interval_level=0.95,
        alpha=0.1,
        # Planner execution time limit
        time_out_in_seconds=3.0,
        # Logging
        verbose=True,
    )


if __name__ == "__main__":
    main()
