Planners Comparison Study
=========================

This example demonstrates how to conduct a comprehensive comparison of different POMDP planning algorithms using the SimulationsAPI. We'll compare POMCPOW and PFT-DPW planners on Push POMDP and Light-Dark POMDP environments, showcasing how to evaluate algorithm performance across different problem domains.

Overview
--------

**Planning Algorithms Tested:**
- **POMCPOW**: Monte Carlo Tree Search with double progressive widening
- **PFT-DPW**: Progressive Function Transfer with Double Progressive Widening

**Environments Tested:**
- **Push POMDP**: Object manipulation with continuous actions
- **Light-Dark POMDP**: Navigation with position-dependent observation noise

**Key Features Demonstrated:**
- Environment configuration using EnvironmentConfigsAPI
- Custom action samplers for continuous action spaces
- Statistical analysis with confidence intervals
- Multi-environment, multi-algorithm evaluation
- Performance profiling and result visualization

Complete Example
----------------

.. code-block:: python

    import numpy as np
    from pathlib import Path
    from typing import List

    # Core POMDPPlanners imports
    from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
    from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
    from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
    from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
    from POMDPPlanners.simulations.simulations_api import SimulationsAPI
    from POMDPPlanners.core.simulation import EnvironmentRunParams

    # Custom Action Samplers for Different Environments
    # ================================================

    class PushPOMDPActionSampler(ActionSampler):
        """Action sampler for Push POMDP with 2D continuous force vectors."""

        def __init__(self, max_force: float = 1.0):
            self.max_force = max_force

        def sample(self, belief_node=None):
            # Sample 2D force vector for pushing objects
            angle = np.random.uniform(0, 2 * np.pi)
            magnitude = np.random.uniform(0, self.max_force)
            return np.array([
                magnitude * np.cos(angle),
                magnitude * np.sin(angle)
            ])

    class LightDarkPOMDPActionSampler(ActionSampler):
        """Action sampler for Light-Dark POMDP with discrete movement actions."""

        def __init__(self):
            # Light-Dark POMDP typically uses discrete actions: North, South, East, West
            self.actions = [0, 1, 2, 3]  # Discrete movement directions

        def sample(self, belief_node=None):
            return np.random.choice(self.actions)

    # Environment Configuration
    # ========================

    def setup_environments_and_beliefs():
        """Configure Push POMDP and Light-Dark POMDP environments with initial beliefs."""

        print("Setting up environments...")

        # Initialize environment configuration API
        env_config = EnvironmentConfigsAPI(discount_factor=0.95, debug=False)

        # Configure Push POMDP environment
        push_env, push_belief = env_config.push_pomdp_config(n_particles=1000)
        print(f"Push POMDP configured: {push_env.name}")
        print(f"  - Action space: {push_env.space_info.action_space}")
        print(f"  - Observation space: {push_env.space_info.observation_space}")

        # Configure Light-Dark POMDP environment (discrete actions version)
        light_dark_env, light_dark_belief = env_config.continuous_observations_discrete_actions_light_dark_pomdp_config(
            n_particles=1000
        )
        print(f"Light-Dark POMDP configured: {light_dark_env.name}")
        print(f"  - Goal state: {light_dark_env.goal_state}")
        print(f"  - Start state: {light_dark_env.start_state}")

        return (push_env, push_belief), (light_dark_env, light_dark_belief)

    # Planner Configuration
    # ====================

    def setup_planners(push_env, light_dark_env):
        """Configure POMCPOW and PFT-DPW planners for both environments."""

        print("\\nSetting up planners...")

        # Create action samplers for each environment
        push_action_sampler = PushPOMDPActionSampler(max_force=1.5)
        light_dark_action_sampler = LightDarkPOMDPActionSampler()

        # Configure planners for Push POMDP
        push_planners = [
            POMCPOW(
                environment=push_env,
                discount_factor=0.95,
                depth=15,
                exploration_constant=1.41,  # √2 for balanced exploration
                k_o=3.0,                    # Observation progressive widening coefficient
                k_a=3.0,                    # Action progressive widening coefficient
                alpha_o=0.5,                # Observation progressive widening exponent
                alpha_a=0.5,                # Action progressive widening exponent
                action_sampler=push_action_sampler,
                n_simulations=1000,
                name="POMCPOW_Push"
            ),
            PFT_DPW(
                environment=push_env,
                discount_factor=0.95,
                depth=15,
                name="PFT_DPW_Push",
                action_sampler=push_action_sampler,
                k_a=2.0,                    # Action progressive widening coefficient
                alpha_a=0.6,                # Faster action space expansion
                k_o=1.5,                    # Observation progressive widening coefficient
                alpha_o=0.5,                # Observation progressive widening exponent
                exploration_constant=1.0,   # UCB1 exploration parameter
                n_simulations=1000
            )
        ]

        # Configure planners for Light-Dark POMDP
        light_dark_planners = [
            POMCPOW(
                environment=light_dark_env,
                discount_factor=0.95,
                depth=20,
                exploration_constant=2.0,   # Higher exploration for navigation
                k_o=4.0,                    # More observation branches for complex navigation
                k_a=2.0,                    # Conservative action expansion for discrete actions
                alpha_o=0.6,                # Faster observation expansion
                alpha_a=0.4,                # Slower action expansion (discrete space)
                action_sampler=light_dark_action_sampler,
                n_simulations=1500,
                name="POMCPOW_LightDark"
            ),
            PFT_DPW(
                environment=light_dark_env,
                discount_factor=0.95,
                depth=20,
                name="PFT_DPW_LightDark",
                action_sampler=light_dark_action_sampler,
                k_a=1.5,                    # Conservative action expansion
                alpha_a=0.4,                # Slower expansion for discrete actions
                k_o=2.0,                    # Observation progressive widening
                alpha_o=0.5,                # Standard observation expansion
                exploration_constant=1.5,   # Moderate exploration
                n_simulations=1500
            )
        ]

        print(f"Configured {len(push_planners)} planners for Push POMDP")
        print(f"Configured {len(light_dark_planners)} planners for Light-Dark POMDP")

        return push_planners, light_dark_planners

    # Simulation Configuration
    # =======================

    def create_simulation_configurations(environments_and_beliefs, planners):
        """Create environment run parameters for the simulation study."""

        (push_env, push_belief), (light_dark_env, light_dark_belief) = environments_and_beliefs
        push_planners, light_dark_planners = planners

        print("\\nCreating simulation configurations...")

        # Configure simulation parameters for each environment
        environment_run_params = [
            # Push POMDP configuration
            EnvironmentRunParams(
                environment=push_env,
                belief=push_belief,
                policies=push_planners,
                num_episodes=100,            # Number of episodes per policy
                num_steps=30                 # Maximum steps per episode
            ),

            # Light-Dark POMDP configuration
            EnvironmentRunParams(
                environment=light_dark_env,
                belief=light_dark_belief,
                policies=light_dark_planners,
                num_episodes=150,            # More episodes for navigation task
                num_steps=25                 # Steps to reach goal
            )
        ]

        total_configurations = sum(len(config.policies) for config in environment_run_params)
        print(f"Created {len(environment_run_params)} environment configurations")
        print(f"Total algorithm-environment combinations: {total_configurations}")

        return environment_run_params

    # Main Simulation Execution
    # ========================

    def run_planners_comparison_study():
        """Execute the complete planners comparison study."""

        print("=" * 60)
        print("POMDP Planners Comparison Study")
        print("Comparing POMCPOW vs PFT-DPW on Push and Light-Dark POMDPs")
        print("=" * 60)

        # Setup phase
        environments_and_beliefs = setup_environments_and_beliefs()
        push_env, light_dark_env = environments_and_beliefs[0][0], environments_and_beliefs[1][0]
        planners = setup_planners(push_env, light_dark_env)
        environment_run_params = create_simulation_configurations(environments_and_beliefs, planners)

        # Initialize SimulationsAPI
        print("\\nInitializing SimulationsAPI...")
        api = SimulationsAPI(
            cache_dir_path=Path("./planners_comparison_results"),
            debug=True
        )

        # Execute simulation with debug validation
        print("\\nStarting simulation with initial debug validation...")
        print("This will run a quick validation followed by the full study...")

        try:
            results, statistics_df = api.run_multiple_environments_and_policies_local_run_with_initial_debug_run(
                environment_run_params=environment_run_params,
                alpha=0.05,                      # 95% confidence intervals
                confidence_interval_level=0.95,
                experiment_name="Planners_Comparison_Study",
                n_jobs=-1,                       # Use all available CPU cores
                enable_profiling=True            # Enable performance profiling
            )

            print("\\n" + "=" * 60)
            print("SIMULATION COMPLETED SUCCESSFULLY!")
            print("=" * 60)

            return results, statistics_df

        except Exception as e:
            print(f"\\nSimulation failed with error: {e}")
            print("Check the logs for detailed error information.")
            raise

    # Results Analysis
    # ===============

    def analyze_results(results, statistics_df):
        """Analyze and display the simulation results."""

        print("\\nANALYZING RESULTS...")
        print("-" * 40)

        # Display basic statistics
        print(f"Total configurations tested: {len(statistics_df)}")
        print(f"Environments: {', '.join(statistics_df['environment'].unique())}")
        print(f"Planners: {', '.join(statistics_df['policy'].unique())}")

        print("\\nDETAILED PERFORMANCE COMPARISON:")
        print("-" * 40)

        # Compare performance by environment
        for env_name in statistics_df['environment'].unique():
            env_results = statistics_df[statistics_df['environment'] == env_name]
            print(f"\\n{env_name.upper()} RESULTS:")

            for _, row in env_results.iterrows():
                policy_name = row['policy']
                avg_return = row['average_return']
                ci_lower = row['average_return_ci_lower']
                ci_upper = row['average_return_ci_upper']
                std_return = row['std_return']
                total_episodes = row['total_episodes']

                print(f"  {policy_name}:")
                print(f"    Average Return: {avg_return:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"    Std Deviation: {std_return:.3f}")
                print(f"    Episodes: {total_episodes}")

        # Performance ranking
        print("\\nPERFORMANCE RANKING BY ENVIRONMENT:")
        print("-" * 40)

        for env_name in statistics_df['environment'].unique():
            env_results = statistics_df[statistics_df['environment'] == env_name]
            ranked = env_results.sort_values('average_return', ascending=False)

            print(f"\\n{env_name} - Best to Worst:")
            for i, (_, row) in enumerate(ranked.iterrows(), 1):
                print(f"  {i}. {row['policy']}: {row['average_return']:.3f}")

        # Statistical significance analysis
        print("\\nSTATISTICAL ANALYSIS:")
        print("-" * 40)

        for env_name in statistics_df['environment'].unique():
            env_results = statistics_df[statistics_df['environment'] == env_name]
            if len(env_results) >= 2:
                best_policy = env_results.loc[env_results['average_return'].idxmax()]
                worst_policy = env_results.loc[env_results['average_return'].idxmin()]

                # Check if confidence intervals overlap
                best_ci_lower = best_policy['average_return_ci_lower']
                worst_ci_upper = worst_policy['average_return_ci_upper']

                significant = best_ci_lower > worst_ci_upper
                significance_text = "SIGNIFICANT" if significant else "not significant"

                print(f"{env_name}: Performance difference is {significance_text}")
                print(f"  Best: {best_policy['policy']} ({best_policy['average_return']:.3f})")
                print(f"  Worst: {worst_policy['policy']} ({worst_policy['average_return']:.3f})")

    # Visualization and Reporting
    # ==========================

    def generate_comparison_report(results, statistics_df):
        """Generate a comprehensive comparison report."""

        print("\\n" + "=" * 60)
        print("COMPREHENSIVE COMPARISON REPORT")
        print("=" * 60)

        # Algorithm characteristics summary
        print("\\nALGORITHM CHARACTERISTICS:")
        print("-" * 30)
        print("POMCPOW:")
        print("  - Double progressive widening (actions + observations)")
        print("  - Weighted particle belief updates")
        print("  - UCB1-based action selection")
        print("  - Suitable for mixed discrete/continuous spaces")

        print("\\nPFT-DPW:")
        print("  - Progressive Function Transfer")
        print("  - Action and observation progressive widening")
        print("  - Designed for continuous action spaces")
        print("  - UCB1-style exploration with adaptive sampling")

        # Environment characteristics
        print("\\nENVIRONMENT CHARACTERISTICS:")
        print("-" * 30)
        print("Push POMDP:")
        print("  - Continuous action space (2D force vectors)")
        print("  - Object manipulation task")
        print("  - Noisy observations")
        print("  - Reward based on successful object pushing")

        print("\\nLight-Dark POMDP:")
        print("  - Discrete action space (4 directions)")
        print("  - Navigation task with position-dependent noise")
        print("  - Beacon-based observations")
        print("  - Goal-reaching with obstacle avoidance")

        # Key insights
        print("\\nKEY INSIGHTS:")
        print("-" * 15)

        for env_name in statistics_df['environment'].unique():
            env_results = statistics_df[statistics_df['environment'] == env_name]
            best_performer = env_results.loc[env_results['average_return'].idxmax()]

            print(f"\\n{env_name}:")
            print(f"  Best performer: {best_performer['policy']}")
            print(f"  Performance: {best_performer['average_return']:.3f}")

            # Environment-specific insights
            if "Push" in env_name:
                print("  Analysis: Performance on continuous action manipulation task")
            elif "LightDark" in env_name:
                print("  Analysis: Performance on discrete navigation with noisy observations")

        print("\\nRECOMMendations:")
        print("-" * 15)
        print("- For continuous action spaces: Consider the better-performing algorithm")
        print("- For discrete action spaces: Evaluate based on computational constraints")
        print("- For mixed environments: POMCPOW may offer better versatility")
        print("- Consider problem-specific tuning of progressive widening parameters")

    # Main Execution
    # =============

    if __name__ == "__main__":
        try:
            # Run the complete comparison study
            results, statistics_df = run_planners_comparison_study()

            # Analyze and report results
            analyze_results(results, statistics_df)
            generate_comparison_report(results, statistics_df)

            print("\\n" + "=" * 60)
            print("STUDY COMPLETE!")
            print("Check './planners_comparison_results' for detailed logs and data")
            print("MLflow tracking available for experiment details")
            print("=" * 60)

        except Exception as e:
            print(f"\\nStudy failed: {e}")
            print("Check logs for debugging information")

Expected Output and Analysis
---------------------------

**Performance Metrics:**
The simulation will generate comprehensive statistics including:

- **Average Return**: Mean cumulative reward across episodes
- **Confidence Intervals**: Statistical bounds on performance estimates
- **Standard Deviation**: Measure of performance variability
- **Episode Counts**: Number of episodes completed for each configuration

**Comparative Analysis:**
The study will reveal:

- **Algorithm Strengths**: Which planner excels in each environment type
- **Statistical Significance**: Whether performance differences are meaningful
- **Computational Efficiency**: Planning time and resource usage via profiling
- **Robustness**: Performance consistency across different episodes

**Expected Insights:**

1. **Push POMDP**:
   - PFT-DPW may perform better due to its continuous action space design
   - POMCPOW's double progressive widening may help with observation complexity

2. **Light-Dark POMDP**:
   - POMCPOW may excel due to its mixed space handling capabilities
   - PFT-DPW's discrete action sampling may be less optimal

3. **Overall**:
   - Progressive widening parameters significantly impact performance
   - Environment complexity affects relative algorithm performance
   - Statistical analysis provides confidence in comparative conclusions

Customization Options
--------------------

**Environment Modifications:**
.. code-block:: python

    # Modify environment parameters
    env_config = EnvironmentConfigsAPI(discount_factor=0.99, debug=True)

    # Use risk-averse environment configurations
    risk_averse_config = RiskAverseEnvironmentConfigsAPI(discount_factor=0.95)

**Planner Tuning:**
.. code-block:: python

    # Adjust progressive widening parameters
    pomcpow_tuned = POMCPOW(
        environment=env,
        k_o=5.0,        # More aggressive observation expansion
        k_a=1.5,        # Conservative action expansion
        alpha_o=0.7,    # Faster observation growth
        alpha_a=0.3,    # Slower action growth
        # ... other parameters
    )

**Simulation Scale:**
.. code-block:: python

    # Scale up for production studies
    environment_run_params = [
        EnvironmentRunParams(
            environment=env,
            belief=belief,
            policies=policies,
            num_episodes=500,    # More episodes for statistical power
            num_steps=50         # Longer episodes
        )
    ]

**Advanced Analysis:**
.. code-block:: python

    # Enable additional statistical analysis
    results, statistics_df = api.run_multiple_environments_and_policies_local_run(
        environment_run_params=environment_run_params,
        alpha=0.01,                      # 99% confidence intervals
        confidence_interval_level=0.99,
        experiment_name="Detailed_Study",
        enable_profiling=True,           # Performance analysis
        profiling_output_limit=100      # Detailed profiling data
    )

This comprehensive example demonstrates the power of the POMDPPlanners framework for conducting rigorous algorithm comparisons across different problem domains, providing both statistical rigor and practical insights for algorithm selection and tuning.