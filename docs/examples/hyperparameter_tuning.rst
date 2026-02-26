Hyperparameter Tuning Examples
==============================

This page demonstrates how to perform hyperparameter optimization for POMDP planners using the POMDPPlanners framework. We'll show how to optimize different algorithms on various environments using Optuna-based optimization.

Overview
--------

Hyperparameter tuning is crucial for achieving optimal performance in POMDP planning. The framework provides a comprehensive hyperparameter optimization system that:

- Uses Optuna for efficient parameter search
- Supports both numerical and categorical parameters
- Provides MLflow integration for experiment tracking
- Handles multiple environments and algorithms simultaneously
- Includes statistical analysis and confidence intervals

Basic Hyperparameter Optimization
---------------------------------

Let's start with a simple example optimizing POMCP on the Tiger POMDP:

.. code-block:: python

   from pathlib import Path
   from POMDPPlanners.simulations.simulations_api import SimulationsAPI
   from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
   from POMDPPlanners.configs.planners_hyperparam_configs import PlannersHyperparamConfigs
   from POMDPPlanners.core.simulation import (
       NumericalHyperParameter, CategoricalHyperParameter
   )
   from POMDPPlanners.core.simulation.hyperparameter_tuning import (
       HyperParameterRunParams, HyperParameterOptimizationDirection
   )

   # Initialize the API
   api = SimulationsAPI(
       cache_dir_path=Path("./hyperparameter_results"),
       debug=True
   )

   # Create environment configuration
   env_configs = EnvironmentConfigsAPI(discount_factor=0.95)
   tiger_env, tiger_belief = env_configs.tiger_pomdp_config(n_particles=1000)

   # Create planner configuration
   planner_configs = PlannersHyperparamConfigs(discount_factor=0.95)

   # Define hyperparameter optimization configuration
   optimization_configs = [
       HyperParameterRunParams(
           environment=tiger_env,
           belief=tiger_belief,
           policy_cls=POMCP,
           hyper_parameters=[
               NumericalHyperParameter("exploration_constant", 0.1, 100.0),
               NumericalHyperParameter("depth", 5, 30),
               NumericalHyperParameter("min_samples_per_node", 1, 20)
           ],
           constant_parameters={
               "discount_factor": 0.95,
               "name": "OptimizedPOMCP_Tiger"
           },
           num_episodes=50,       # Episodes for final evaluation
           num_steps=30,          # Steps per episode
           n_trials=100,         # Number of optimization trials
           direction=HyperParameterOptimizationDirection.MAXIMIZE,
           parameter_to_optimize="average_return"
       )
   ]

   # Run hyperparameter optimization
   results = api.run_hyperparameter_optimization(
       environment_run_params=optimization_configs,
       experiment_name="Tiger_POMCP_Optimization",
       n_jobs=4,  # Use 4 CPU cores
   )

   # Analyze results
   for i, result in enumerate(results):
       print(f"Configuration {i+1} Results:")
       print(f"  Environment: {result.environment.__class__.__name__}")
       print(f"  Policy: {result.policy.__class__.__name__}")
       print(f"  Best hyperparameters: {result.chosen_hyper_parameters}")
       print(f"  Policy name: {result.policy.name}")

Multi-Environment Optimization
------------------------------

Now let's optimize multiple algorithms on different environments:

.. code-block:: python

   from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
   from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
   from POMDPPlanners.planners.planners_utils.dpw import SimpleActionSampler

   # Initialize APIs
   api = SimulationsAPI(
       cache_dir_path=Path("./multi_env_optimization"),
       debug=True
   )
   
   env_configs = EnvironmentConfigsAPI(discount_factor=0.95)
   planner_configs = PlannersHyperparamConfigs(discount_factor=0.95)

   # Create environments
   rock_sample_env, rock_sample_belief = env_configs.rock_sample_pomdp_config(n_particles=1000)
   laser_tag_env, laser_tag_belief = env_configs.laser_tag_pomdp_config(n_particles=1000)

   # Create action sampler for POMCPOW
   action_sampler = SimpleActionSampler()

   # Define multiple optimization configurations
   optimization_configs = [
       # SparsePFT on Rock Sample
       HyperParameterRunParams(
           environment=rock_sample_env,
           belief=rock_sample_belief,
           policy_cls=SparsePFT,
           hyper_parameters=[
               NumericalHyperParameter("depth", 5, 15),
               NumericalHyperParameter("c_ucb", 0.0, 50.0),
               NumericalHyperParameter("beta_ucb", 0.0, 50.0),
               NumericalHyperParameter("belief_child_num", 3, 15)
           ],
           constant_parameters={
               "discount_factor": 0.95,
               "gamma": 0.95,
               "name": "OptimizedSparsePFT_RockSample"
           },
           num_episodes=30,
           num_steps=25,
           n_trials=80,
           direction=HyperParameterOptimizationDirection.MAXIMIZE,
           parameter_to_optimize="average_return"
       ),
       
       # POMCPOW on Laser Tag
       HyperParameterRunParams(
           environment=laser_tag_env,
           belief=laser_tag_belief,
           policy_cls=POMCPOW,
           hyper_parameters=[
               NumericalHyperParameter("exploration_constant", 0.0, 50.0),
               NumericalHyperParameter("depth", 5, 15),
               NumericalHyperParameter("k_a", 1, 10),
               NumericalHyperParameter("alpha_a", 0.01, 0.5),
               NumericalHyperParameter("k_o", 1, 10),
               NumericalHyperParameter("alpha_o", 0.01, 0.5)
           ],
           constant_parameters={
               "discount_factor": 0.95,
               "name": "OptimizedPOMCPOW_LaserTag",
               "action_sampler": action_sampler,
               "time_out_in_seconds": 3.0
           },
           num_episodes=30,
           num_steps=25,
           n_trials=80,
           direction=HyperParameterOptimizationDirection.MAXIMIZE,
           parameter_to_optimize="average_return"
       ),
       
       # SparsePFT on Laser Tag
       HyperParameterRunParams(
           environment=laser_tag_env,
           belief=laser_tag_belief,
           policy_cls=SparsePFT,
           hyper_parameters=[
               NumericalHyperParameter("depth", 5, 15),
               NumericalHyperParameter("c_ucb", 0.0, 50.0),
               NumericalHyperParameter("beta_ucb", 0.0, 50.0),
               NumericalHyperParameter("belief_child_num", 3, 15)
           ],
           constant_parameters={
               "discount_factor": 0.95,
               "gamma": 0.95,
               "name": "OptimizedSparsePFT_LaserTag"
           },
           num_episodes=30,
           num_steps=25,
           n_trials=80,
           direction=HyperParameterOptimizationDirection.MAXIMIZE,
           parameter_to_optimize="average_return"
       )
   ]

   # Run multi-environment optimization
   results = api.run_hyperparameter_optimization(
       environment_run_params=optimization_configs,
       experiment_name="Multi_Environment_Algorithm_Optimization",
       n_jobs=4,
   )

   # Analyze and compare results
   print("=== Multi-Environment Optimization Results ===")
   for i, result in enumerate(results):
       env_name = result.environment.__class__.__name__
       policy_name = result.policy.__class__.__name__
       best_params = result.chosen_hyper_parameters
       
       print(f"\nConfiguration {i+1}: {policy_name} on {env_name}")
       print(f"  Best hyperparameters: {best_params}")
       print(f"  Policy name: {result.policy.name}")

Using Predefined Hyperparameter Configurations
----------------------------------------------

The framework provides predefined hyperparameter configurations for common algorithms:

.. code-block:: python

   from POMDPPlanners.configs.planners_hyperparam_configs import PlannersHyperparamConfigs
   from POMDPPlanners.planners.planners_utils.dpw import SimpleActionSampler

   # Initialize configuration APIs
   env_configs = EnvironmentConfigsAPI(discount_factor=0.95)
   planner_configs = PlannersHyperparamConfigs(discount_factor=0.95)

   # Create environments
   rock_sample_env, rock_sample_belief = env_configs.rock_sample_pomdp_config(n_particles=1000)
   laser_tag_env, laser_tag_belief = env_configs.laser_tag_pomdp_config(n_particles=1000)

   # Create action sampler
   action_sampler = SimpleActionSampler()

   # Use predefined configurations
   sparse_pft_config = planner_configs.sparse_pft_config(
       env=rock_sample_env, 
       name="PredefinedSparsePFT_RockSample"
   )
   
   pomcpow_config = planner_configs.pomcpow_config(
       env=laser_tag_env, 
       action_sampler=action_sampler, 
       name="PredefinedPOMCPOW_LaserTag"
   )

   # Convert to optimization parameters
   optimization_configs = [
       HyperParameterRunParams(
           environment=rock_sample_env,
           belief=rock_sample_belief,
           policy_cls=sparse_pft_config.policy_cls,
           hyper_parameters=sparse_pft_config.hyper_parameters,
           constant_parameters=sparse_pft_config.constant_parameters,
           num_episodes=40,
           num_steps=30,
           n_trials=100,
           direction=HyperParameterOptimizationDirection.MAXIMIZE,
           parameter_to_optimize="average_return"
       ),
       
       HyperParameterRunParams(
           environment=laser_tag_env,
           belief=laser_tag_belief,
           policy_cls=pomcpow_config.policy_cls,
           hyper_parameters=pomcpow_config.hyper_parameters,
           constant_parameters=pomcpow_config.constant_parameters,
           num_episodes=40,
           num_steps=30,
           n_trials=100,
           direction=HyperParameterOptimizationDirection.MAXIMIZE,
           parameter_to_optimize="average_return"
       )
   ]

   # Run optimization with predefined configurations
   results = api.run_hyperparameter_optimization(
       environment_run_params=optimization_configs,
       experiment_name="Predefined_Config_Optimization",
       n_jobs=4,
   )

Advanced Optimization with Risk-Averse Environments
----------------------------------------------------

Let's optimize algorithms on risk-averse environment configurations:

.. code-block:: python

   from POMDPPlanners.configs.environment_configs import RiskAverseEnvironmentConfigsAPI

   # Initialize risk-averse environment configurations
   risk_averse_env_configs = RiskAverseEnvironmentConfigsAPI(discount_factor=0.95)

   # Create risk-averse environments
   risk_rock_sample_env, risk_rock_sample_belief = risk_averse_env_configs.rock_sample_pomdp_config(n_particles=1000)
   risk_laser_tag_env, risk_laser_tag_belief = risk_averse_env_configs.laser_tag_pomdp_config(n_particles=1000)

   # Define risk-aware optimization configurations
   risk_optimization_configs = [
       # SparsePFT on Risk-Averse Rock Sample
       HyperParameterRunParams(
           environment=risk_rock_sample_env,
           belief=risk_rock_sample_belief,
           policy_cls=SparsePFT,
           hyper_parameters=[
               NumericalHyperParameter("depth", 5, 15),
               NumericalHyperParameter("c_ucb", 0.0, 50.0),
               NumericalHyperParameter("beta_ucb", 0.0, 50.0),
               NumericalHyperParameter("belief_child_num", 3, 15)
           ],
           constant_parameters={
               "discount_factor": 0.95,
               "gamma": 0.95,
               "name": "RiskAverseSparsePFT_RockSample"
           },
           num_episodes=50,
           num_steps=30,
           n_trials=120,
           direction=HyperParameterOptimizationDirection.MAXIMIZE,
           parameter_to_optimize="average_return"
       ),
       
       # POMCPOW on Risk-Averse Laser Tag
       HyperParameterRunParams(
           environment=risk_laser_tag_env,
           belief=risk_laser_tag_belief,
           policy_cls=POMCPOW,
           hyper_parameters=[
               NumericalHyperParameter("exploration_constant", 0.0, 50.0),
               NumericalHyperParameter("depth", 5, 15),
               NumericalHyperParameter("k_a", 1, 10),
               NumericalHyperParameter("alpha_a", 0.01, 0.5),
               NumericalHyperParameter("k_o", 1, 10),
               NumericalHyperParameter("alpha_o", 0.01, 0.5)
           ],
           constant_parameters={
               "discount_factor": 0.95,
               "name": "RiskAversePOMCPOW_LaserTag",
               "action_sampler": action_sampler,
               "time_out_in_seconds": 3.0
           },
           num_episodes=50,
           num_steps=30,
           n_trials=120,
           direction=HyperParameterOptimizationDirection.MAXIMIZE,
           parameter_to_optimize="average_return"
       )
   ]

   # Run risk-aware optimization
   risk_results = api.run_hyperparameter_optimization(
       environment_run_params=risk_optimization_configs,
       experiment_name="Risk_Averse_Optimization",
       n_jobs=4,
   )

   print("=== Risk-Averse Optimization Results ===")
   for i, result in enumerate(risk_results):
       env_name = result.environment.__class__.__name__
       policy_name = result.policy.__class__.__name__
       best_params = result.chosen_hyper_parameters
       
       print(f"\nRisk-Averse Configuration {i+1}: {policy_name} on {env_name}")
       print(f"  Best hyperparameters: {best_params}")

Optimization with Custom Parameter Types
-----------------------------------------

You can also use categorical parameters for algorithm selection:

.. code-block:: python

   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import SparseSamplingDiscreteActionsPlanner

   # Define optimization with categorical parameters
   categorical_optimization_configs = [
       HyperParameterRunParams(
           environment=tiger_env,
           belief=tiger_belief,
           policy_cls=POMCP,  # Base class, actual algorithm determined by categorical param
           hyper_parameters=[
               CategoricalHyperParameter("algorithm_type", ["POMCP", "SparseSampling"]),
               NumericalHyperParameter("exploration_constant", 0.1, 100.0),
               NumericalHyperParameter("depth", 5, 20),
               NumericalHyperParameter("num_simulations", 100, 2000)
           ],
           constant_parameters={
               "discount_factor": 0.95,
               "name": "CategoricalOptimized"
           },
           num_episodes=30,
           num_steps=25,
           n_trials=60,
           direction=HyperParameterOptimizationDirection.MAXIMIZE,
           parameter_to_optimize="average_return"
       )
   ]

   # Note: This requires custom policy class that handles categorical parameters
   # For demonstration purposes, we'll use separate configurations instead

Analyzing Optimization Results
------------------------------

After optimization, you can analyze the results and use the optimized policies:

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt

   # Extract optimization results
   optimized_policies = [result.policy for result in results]
   optimization_metadata = [result.optimization_metadata for result in results]

   # Create a comparison table
   comparison_data = []
   for i, (policy, metadata) in enumerate(zip(optimized_policies, optimization_metadata)):
       comparison_data.append({
           'Configuration': i+1,
           'Environment': policy.environment.__class__.__name__,
           'Algorithm': policy.__class__.__name__,
           'Policy_Name': policy.name,
           'Best_Parameters': str(metadata.get('best_params', {})),
           'Best_Value': metadata.get('best_value', 0.0),
           'N_Trials': metadata.get('n_trials', 0)
       })

   comparison_df = pd.DataFrame(comparison_data)
   print("=== Optimization Results Comparison ===")
   print(comparison_df.to_string(index=False))

   # Use optimized policies for further analysis
   print(f"\nSuccessfully optimized {len(optimized_policies)} policies:")
   for policy in optimized_policies:
       print(f"  - {policy.name} ({policy.__class__.__name__})")

   # You can now use these optimized policies in simulations
   from POMDPPlanners.core.simulation import EnvironmentRunParams
   
   # Create simulation configurations with optimized policies
   simulation_configs = []
   for i, policy in enumerate(optimized_policies):
       simulation_configs.append(
           EnvironmentRunParams(
               environment=policy.environment,
               belief=rock_sample_belief if "RockSample" in policy.name else laser_tag_belief,
               policies=[policy],
               num_episodes=100,
               num_steps=30
           )
       )

   # Run final evaluation simulation
   final_results, final_stats = api.run_multiple_environments_and_policies_local_run(
       environment_run_params=simulation_configs,
       alpha=0.05,
       confidence_interval_level=0.95,
       experiment_name="Final_Optimized_Policy_Evaluation",
       n_jobs=4
   )

   print("\n=== Final Evaluation Results ===")
   print(final_stats[['environment', 'policy', 'average_return', 'average_return_ci_lower', 'average_return_ci_upper']].to_string(index=False))

Best Practices for Hyperparameter Optimization
----------------------------------------------

**Parameter Range Selection**
   - Start with wide ranges and narrow down based on initial results
   - Use logarithmic scales for parameters that vary over orders of magnitude
   - Consider problem-specific constraints (e.g., depth should not exceed episode length)

**Trial Configuration**
   - Start with fewer trials (50-100) for initial exploration
   - Increase trials (200-500) for final optimization
   - Use more episodes (50-100) for reliable performance estimates

**Computational Resources**
   - Use parallel execution (``n_jobs=-1``) when available
   - Consider using distributed computing for large-scale optimization
   - Monitor memory usage with large particle counts

**Evaluation Strategy**
   - Use consistent evaluation metrics across all configurations
   - Consider multiple performance metrics (average return, success rate, planning time)
   - Validate optimized policies on held-out test episodes

**MLflow Integration**
   - All optimization runs are automatically tracked in MLflow
   - Use descriptive experiment names for easy organization
   - Compare results across different optimization runs

Troubleshooting Common Issues
-----------------------------

**Low Performance After Optimization**
   - Check if parameter ranges are appropriate for the problem
   - Verify that the optimization direction is correct (maximize vs minimize)
   - Ensure sufficient trials and episodes for reliable estimates

**Optimization Taking Too Long**
   - Reduce the number of trials or episodes
   - Use fewer particles in belief representation
   - Decrease the planning depth or timeout limits

**Memory Issues**
   - Reduce particle count in belief representation
   - Use smaller planning depths
   - Consider using sparse belief representations

**Convergence Problems**
   - Increase the number of trials
   - Adjust parameter ranges based on initial results
   - Consider using different optimization algorithms in Optuna

Next Steps
----------

- Try :doc:`planners_comparison` for comparing optimized policies
- See :doc:`basic_usage` for using optimized policies in simulations
- Check the :doc:`../api/simulations` for advanced simulation features
- Explore the :doc:`../api/core` for detailed API reference
