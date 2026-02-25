Quickstart Guide
================

This guide will get you up and running with POMDPPlanners in just a few minutes.

Your First POMDP Solution
--------------------------

Let's solve the classic Tiger POMDP problem using POMCP:

.. code-block:: python

   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.core.belief import get_initial_belief

   # Create the environment and initial belief
   env = TigerPOMDP(discount_factor=0.95)
   belief = get_initial_belief(env, n_particles=500)

   # Create the planner
   planner = POMCP(
       environment=env,
       discount_factor=0.95,
       depth=10,
       exploration_constant=50.0,
       name="tiger_planner",
       n_simulations=1000,
   )

   # Plan: returns a list of actions (length=1 for closed-loop planning)
   actions, run_data = planner.action(belief)
   action = actions[0]
   print(f"Recommended action: {action}")

   # Execute: sample the next state, observation, and reward
   state = belief.sample()
   next_state, observation, reward = env.sample_next_step(state=state, action=action)
   print(f"Observation: {observation}, Reward: {reward}")


Running a Complete Episode
---------------------------

Use ``run_episode`` to run a full episode with automatic belief updates:

.. code-block:: python

   from POMDPPlanners.simulations.episodes import run_episode
   from POMDPPlanners.utils.logger import get_logger

   logger = get_logger("quickstart")

   history = run_episode(
       environment=env,
       policy=planner,
       initial_belief=belief,
       num_steps=20,
       logger=logger,
   )

   total_reward = sum(step.reward for step in history.history if step.reward is not None)
   print(f"Steps: {len(history.history)}, Total reward: {total_reward:.2f}")

   # Each step exposes: action, observation, reward, state
   for i, step in enumerate(history.history[:5]):
       print(f"Step {i}: action={step.action}, obs={step.observation}, reward={step.reward}")


Core Concepts
-------------

**Environments**

Environments can be created directly or via ``EnvironmentConfigsAPI``:

.. code-block:: python

   # Direct construction
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   env = TigerPOMDP(discount_factor=0.95)

   # Via config API (also returns a ready-made initial belief)
   from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
   config_api = EnvironmentConfigsAPI(discount_factor=0.95)
   env, belief = config_api.tiger_pomdp_config(n_particles=500)

   # Discrete environments expose their state/action/observation spaces
   print(env.states)       # ['tiger_left', 'tiger_right']
   print(env.actions)      # ['listen', 'open_left', 'open_right']
   print(env.observations) # ['hear_left', 'hear_right', 'hear_nothing']

   # Core interaction method
   next_state, observation, reward = env.sample_next_step(state=state, action=action)
   done = env.is_terminal(next_state)

**Belief States**

.. code-block:: python

   from POMDPPlanners.core.belief import get_initial_belief

   belief = get_initial_belief(env, n_particles=500)

   # Sample a single state from the belief
   state = belief.sample()

   # Inspect the weighted distribution
   distribution = belief.to_unique_support_distribution()

**Planners**

All planners share the same interface: ``planner.action(belief)`` returns
``(List[action], PolicyRunData)``. A single-element list means closed-loop
(replans each step); a multi-element list means open-loop (executes the
sequence before replanning).

.. code-block:: python

   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP

   planner = POMCP(
       environment=env,
       discount_factor=0.95,
       depth=10,
       exploration_constant=50.0,
       name="my_planner",
       n_simulations=1000,
   )

   actions, run_data = planner.action(belief)
   action = actions[0]  # closed-loop: take the single planned action


Continuous Action Spaces
-------------------------

For environments with continuous actions, pair ``PFT_DPW`` with an action sampler:

.. code-block:: python

   import numpy as np
   from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
       ContinuousLightDarkPOMDP, RewardModelType,
   )
   from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
   from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
   from POMDPPlanners.core.belief import get_initial_belief

   env = ContinuousLightDarkPOMDP(
       discount_factor=0.95,
       goal_state=np.array([10, 5]),
       start_state=np.array([0, 5]),
       reward_model_type=RewardModelType.STANDARD,
   )

   class VelocityActionSampler(ActionSampler):
       def sample(self, belief_node=None):
           angle = np.random.uniform(0, 2 * np.pi)
           speed = np.random.uniform(0, 1.0)
           return np.array([speed * np.cos(angle), speed * np.sin(angle)])

   planner = PFT_DPW(
       environment=env,
       discount_factor=0.95,
       depth=10,
       name="navigation_planner",
       action_sampler=VelocityActionSampler(),
       n_simulations=500,
   )

   belief = get_initial_belief(env, n_particles=500)
   actions, _ = planner.action(belief)
   print(f"Navigation action: {actions[0]}")


Comparing Planners
-------------------

Use ``LocalSimulationsAPI`` to run a statistically rigorous multi-planner,
multi-environment comparison study:

.. code-block:: python

   from pathlib import Path
   from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
   from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
   from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
   from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
   from POMDPPlanners.core.simulation import EnvironmentRunParams
   from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

   config_api = EnvironmentConfigsAPI(discount_factor=0.95)
   env, belief = config_api.continuous_observations_discrete_actions_light_dark_pomdp_config(
       n_particles=500
   )

   action_sampler = DiscreteActionSampler(actions=env.get_actions())

   planners = [
       POMCPOW(
           environment=env, discount_factor=0.95, depth=10,
           exploration_constant=100.0, k_o=10, k_a=4,
           alpha_o=0.01, alpha_a=0.01,
           action_sampler=action_sampler, n_simulations=1500,
           name="POMCPOW",
       ),
       PFT_DPW(
           environment=env, discount_factor=0.95, depth=10,
           k_a=4, alpha_a=0.01, k_o=10, alpha_o=0.01,
           exploration_constant=100.0, action_sampler=action_sampler,
           n_simulations=1500, name="PFT_DPW",
       ),
   ]

   run_params = [
       EnvironmentRunParams(
           environment=env, belief=belief, policies=planners,
           num_episodes=100, num_steps=30,
       )
   ]

   api = LocalSimulationsAPI(cache_dir_path=Path("./results"))
   results, stats_df = api.run_multiple_environments_and_policies_with_initial_debug_run(
       environment_run_params=run_params,
       alpha=0.05,
       confidence_interval_level=0.95,
       experiment_name="planner_comparison",
       n_jobs=-1,
   )


Hyperparameter Tuning
----------------------

Automatically find the best hyperparameters using Optuna, then evaluate the
optimised policy:

.. code-block:: python

   from pathlib import Path
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.core.simulation import NumericalHyperParameter
   from POMDPPlanners.core.simulation.hyperparameter_tuning import (
       HyperParamPlannerConfig, HyperParameterRunParams,
       HyperParameterOptimizationDirection,
   )
   from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI

   env = TigerPOMDP(discount_factor=0.95)
   belief = get_initial_belief(env, n_particles=200)

   optimization_config = HyperParameterRunParams(
       environment=env,
       belief=belief,
       hyper_param_planner_config=HyperParamPlannerConfig(
           policy_cls=POMCP,
           hyper_parameters=[
               NumericalHyperParameter(0.1, 100.0, "exploration_constant"),
               NumericalHyperParameter(3, 10, "depth"),
           ],
           constant_parameters={
               "discount_factor": 0.95,
               "n_simulations": 500,
               "name": "OptimizedPOMCP",
           },
       ),
       num_episodes=20,
       num_steps=30,
       n_trials=50,
       parameters_to_optimize=[
           ("average_return", HyperParameterOptimizationDirection.MAXIMIZE)
       ],
   )

   api = LocalSimulationsAPI(cache_dir_path=Path("./tuning_results"), debug=True)

   # Optimise then evaluate in one call
   results, stats_df = api.run_optimize_and_evaluate(
       configs=[optimization_config],
       evaluation_episodes=100,
       evaluation_steps=30,
       evaluation_n_jobs=-1,
       optimization_n_jobs=-1,
       confidence_interval_level=0.95,
       alpha=0.05,
       experiment_name="tiger_pomcp_tuning",
   )

   print(stats_df[["environment_name", "policy_name", "mean_total_return", "ci_lower", "ci_upper"]])

Use predefined search spaces from ``PlannersHyperparamConfigs`` to skip
writing parameter ranges by hand:

.. code-block:: python

   from POMDPPlanners.configs.planners_hyperparam_configs import PlannersHyperparamConfigs
   from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

   action_sampler = DiscreteActionSampler(actions=env.get_actions())
   planner_configs = PlannersHyperparamConfigs(discount_factor=0.95)

   predefined = planner_configs.pomcpow_config(
       env=env, action_sampler=action_sampler, name="POMCPOW_Tuned"
   )


Viewing Results
----------------

All simulation runs and optimization trials are tracked in MLflow. After any
run, launch the UI from the cache directory:

.. code-block:: bash

   cd ./results   # or whichever cache_dir_path you used
   mlflow ui

Then open http://localhost:5000 to browse metrics, compare runs, and inspect
confidence intervals.


Available Environments
-----------------------

.. code-block:: python

   from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
   config_api = EnvironmentConfigsAPI(discount_factor=0.95)

   # Classic
   env, belief = config_api.tiger_pomdp_config(n_particles=500)

   # Navigation (discrete actions, continuous observations)
   env, belief = config_api.continuous_observations_discrete_actions_light_dark_pomdp_config(n_particles=500)

   # Navigation (fully continuous)
   env, belief = config_api.continuous_observations_continuous_actions_light_dark_pomdp_config(n_particles=500)

   # Manipulation
   env, belief = config_api.push_pomdp_config(n_particles=500)

   # Classic control
   env, belief = config_api.cartpole_pomdp_config(n_particles=500)
   env, belief = config_api.mountain_car_pomdp_config(n_particles=500)


Available Planners
-------------------

.. code-block:: python

   # POMCP — discrete actions and observations
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   planner = POMCP(environment=env, discount_factor=0.95, depth=10,
                   exploration_constant=50.0, name="pomcp", n_simulations=1000)

   # POMCPOW — continuous actions/observations via double progressive widening
   from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
   planner = POMCPOW(environment=env, discount_factor=0.95, depth=10,
                     exploration_constant=100.0,
                     k_o=10, k_a=4, alpha_o=0.01, alpha_a=0.01,
                     action_sampler=action_sampler, n_simulations=1500, name="pomcpow")

   # PFT-DPW — particle filter trees with double progressive widening
   from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
   planner = PFT_DPW(environment=env, discount_factor=0.95, depth=10,
                     k_a=4, alpha_a=0.01, k_o=10, alpha_o=0.01,
                     exploration_constant=100.0, action_sampler=action_sampler,
                     n_simulations=1500, name="pft_dpw")

   # Sparse Sampling — simple model-based baseline (depth=2, branching_factor=10)
   from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling_planner import SparseSamplingDiscreteActionsPlanner
   planner = SparseSamplingDiscreteActionsPlanner(env, branching_factor=10, depth=2)


Next Steps
----------

**Run the example notebooks**

.. code-block:: bash

   jupyter notebook docs/examples/basic_usage.ipynb
   jupyter notebook docs/examples/planners_comparison.ipynb
   jupyter notebook docs/examples/hyperparameter_tuning.ipynb
   jupyter notebook docs/examples/advanced_optimization.ipynb

**API Reference**

Browse the complete API documentation: :doc:`api/POMDPPlanners`
