"""Simple usage example for LocalSimulationsAPI.run_optimize_and_evaluate().

This script demonstrates how to:
1. Set up a POMDP environment (Continuous Light-Dark POMDP)
2. Configure hyperparameter optimization for POMCPOW planner
3. Run optimization and evaluation workflow
4. Analyze the results
"""

import numpy as np
from pathlib import Path

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import NumericalHyperParameter
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
    HyperParamPlannerConfig,
)
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDPDiscreteActions,
)
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import (
    LocalSimulationsAPI,
)
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler


def main():
    """Run optimization and evaluation example."""
    print("=" * 80)
    print("LocalSimulationsAPI.run_optimize_and_evaluate() Example")
    print("=" * 80)

    # Initialize the API
    api = LocalSimulationsAPI(debug=True)

    # Create Continuous Light-Dark POMDP environment with discrete actions
    discount_factor = 0.95
    goal_state = np.array([10.0, 5.0])
    start_state = np.array([2.0, 5.0])

    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=discount_factor,
        goal_state=goal_state,
        start_state=start_state,
        grid_size=11,
        goal_state_radius=1.5,
        fuel_cost=1.0,
        goal_reward=100.0,
        obstacle_reward=-50.0,
    )
    print(f"\n✓ Created environment: {env.name}")
    print(f"  - Start state: {start_state}")
    print(f"  - Goal state: {goal_state}")
    print(f"  - Available actions: {env.get_actions()}")

    # Create initial belief with particles
    n_particles = 100
    initial_belief = get_initial_belief(env, n_particles=n_particles)
    print(f"✓ Initialized belief with {n_particles} particles")

    # Create action sampler for discrete actions
    action_sampler = DiscreteActionSampler(env.get_actions())

    # Define hyperparameter search space for POMCPOW
    # Based on the config from planners_hyperparam_configs.py
    max_depth_for_tuning = 15
    exploration_constant_max = (
        (env.reward_range[1] - env.reward_range[0]) * max_depth_for_tuning
        if env.reward_range
        else 1.0 * max_depth_for_tuning
    )

    planner_config = HyperParamPlannerConfig(
        policy_cls=POMCPOW,
        hyper_parameters=[
            NumericalHyperParameter(0.0, exploration_constant_max, "exploration_constant"),
            NumericalHyperParameter(5, max_depth_for_tuning, "depth"),
            NumericalHyperParameter(1, 10, "k_a"),  # Action progressive widening coefficient
            NumericalHyperParameter(0.01, 0.5, "alpha_a"),  # Action progressive widening exponent
            NumericalHyperParameter(1, 10, "k_o"),  # Observation progressive widening coefficient
            NumericalHyperParameter(
                0.01, 0.5, "alpha_o"
            ),  # Observation progressive widening exponent
        ],
        constant_parameters={
            "discount_factor": discount_factor,
            "name": "OptimizedPOMCPOW_LightDark",
            "environment": env,
            "action_sampler": action_sampler,
            "time_out_in_seconds": 3.0,
        },
    )
    print("\n✓ Configured POMCPOW hyperparameter search space:")
    print(f"  - exploration_constant: [0.0, {exploration_constant_max:.1f}]")
    print(f"  - depth: [5, {max_depth_for_tuning}]")
    print(f"  - k_a (action widening coeff): [1, 10]")
    print(f"  - alpha_a (action widening exp): [0.01, 0.5]")
    print(f"  - k_o (obs widening coeff): [1, 10]")
    print(f"  - alpha_o (obs widening exp): [0.01, 0.5]")
    print(f"  - Fixed time_out: 3.0 seconds")

    # Create optimization configuration
    optimization_configs = [
        HyperParameterRunParams(
            environment=env,
            belief=initial_belief,
            hyper_param_planner_config=planner_config,
            num_episodes=10,  # Episodes for optimization
            num_steps=20,  # Steps per episode for optimization
            n_trials=15,  # Number of optimization trials
            parameters_to_optimize=[
                ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
            ],
        )
    ]
    print("\n✓ Created optimization configuration:")
    print(f"  - Optimization episodes: 10")
    print(f"  - Optimization steps: 20")
    print(f"  - Number of trials: 15")
    print(f"  - Objective: Maximize average return")

    # Run optimization and evaluation
    print("\n" + "=" * 80)
    print("Starting Optimization and Evaluation Workflow")
    print("=" * 80)

    results, stats_df = api.run_optimize_and_evaluate(
        configs=optimization_configs,
        evaluation_episodes=30,  # More episodes for reliable evaluation
        evaluation_steps=25,  # More steps for evaluation
        evaluation_n_jobs=-1,  # Parallel jobs for evaluation
        optimization_n_jobs=-1,  # Parallel jobs for optimization
        experiment_name="LightDark_POMCP_Optimize_Evaluate_Example",
        cache_dir_path=Path("./example_optimize_evaluate_results"),
        debug=True,
        cache_visualizations=True,
    )

    # Display results
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)

    print(f"\n✓ Optimization and evaluation completed successfully!")
    print(f"\n📊 Statistics DataFrame shape: {stats_df.shape}")
    print(f"   - Number of configurations evaluated: {len(stats_df)}")


if __name__ == "__main__":
    main()
