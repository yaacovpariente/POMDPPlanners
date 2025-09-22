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
- Pre-built action samplers from utils module for different action spaces
- Statistical analysis with confidence intervals
- Multi-environment, multi-algorithm evaluation
- Performance profiling and result visualization

Complete Example
----------------

.. code-block:: python

    import numpy as np
    from pathlib import Path

    # Core POMDPPlanners imports
    from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
    from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
    from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
    from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler, DiscreteActionSampler
    from POMDPPlanners.simulations.simulations_api import SimulationsAPI
    from POMDPPlanners.core.simulation import EnvironmentRunParams

    # Setup environments
    env_config = EnvironmentConfigsAPI(discount_factor=0.95, debug=False)
    push_env, push_belief = env_config.push_pomdp_config(n_particles=1000)
    light_dark_env, light_dark_belief = env_config.continuous_observations_discrete_actions_light_dark_pomdp_config(n_particles=1000)

    # Create action samplers
    push_action_sampler = UnitCircleActionSampler(max_action_magnitude=1.5)
    light_dark_action_sampler = DiscreteActionSampler(actions=[0, 1, 2, 3])

    # Configure planners for Push POMDP
    push_planners = [
        POMCPOW(
            environment=push_env,
            discount_factor=0.95,
            depth=15,
            exploration_constant=1.41,
            k_o=3.0,
            k_a=3.0,
            alpha_o=0.5,
            alpha_a=0.5,
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
            k_a=2.0,
            alpha_a=0.6,
            k_o=1.5,
            alpha_o=0.5,
            exploration_constant=1.0,
            n_simulations=1000
        )
    ]

    # Configure planners for Light-Dark POMDP
    light_dark_planners = [
        POMCPOW(
            environment=light_dark_env,
            discount_factor=0.95,
            depth=20,
            exploration_constant=2.0,
            k_o=4.0,
            k_a=2.0,
            alpha_o=0.6,
            alpha_a=0.4,
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
            k_a=1.5,
            alpha_a=0.4,
            k_o=2.0,
            alpha_o=0.5,
            exploration_constant=1.5,
            n_simulations=1500
        )
    ]

    # Create simulation configurations
    environment_run_params = [
        EnvironmentRunParams(
            environment=push_env,
            belief=push_belief,
            policies=push_planners,
            num_episodes=100,
            num_steps=30
        ),
        EnvironmentRunParams(
            environment=light_dark_env,
            belief=light_dark_belief,
            policies=light_dark_planners,
            num_episodes=150,
            num_steps=25
        )
    ]

    # Run simulation
    api = SimulationsAPI(cache_dir_path=Path("./planners_comparison_results"), debug=True)
    results, statistics_df = api.run_multiple_environments_and_policies_local_run_with_initial_debug_run(
        environment_run_params=environment_run_params,
        alpha=0.05,
        confidence_interval_level=0.95,
        experiment_name="Planners_Comparison_Study",
        n_jobs=-1,
        enable_profiling=True
    )

    # Display results
    print("\\nPERFORMANCE RESULTS:")
    for env_name in statistics_df['environment'].unique():
        env_results = statistics_df[statistics_df['environment'] == env_name]
        print(f"\\n{env_name}:")
        for _, row in env_results.iterrows():
            print(f"  {row['policy']}: {row['average_return']:.3f} [{row['average_return_ci_lower']:.3f}, {row['average_return_ci_upper']:.3f}]")

    print("\\nStudy complete! Check './planners_comparison_results' for detailed logs.")

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

**Action Sampler Customization:**
.. code-block:: python

    # Customize action samplers for different environments
    from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler, DiscreteActionSampler
    
    # Conservative movement for delicate tasks
    conservative_sampler = UnitCircleActionSampler(max_action_magnitude=0.5)
    
    # Aggressive movement for fast tasks
    aggressive_sampler = UnitCircleActionSampler(max_action_magnitude=2.0)
    
    # Custom discrete actions
    custom_discrete_sampler = DiscreteActionSampler(actions=[0, 1, 2, 3, 4, 5])

**Planner Tuning:**
.. code-block:: python

    # Adjust progressive widening parameters
    pomcpow_tuned = POMCPOW(
        environment=env,
        k_o=5.0,        # More aggressive observation expansion
        k_a=1.5,        # Conservative action expansion
        alpha_o=0.7,    # Faster observation growth
        alpha_a=0.3,    # Slower action growth
        action_sampler=conservative_sampler,  # Use customized sampler
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