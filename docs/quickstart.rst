Quickstart Guide
================

This guide will get you up and running with POMDPPlanners in just a few minutes.

Your First POMDP Solution
-------------------------

Let's solve the classic Tiger POMDP problem using POMCP:

.. code-block:: python

   from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.core.belief import WeightedParticleBelief

   # Step 1: Create the environment
   config_api = EnvironmentConfigsAPI()
   env, initial_belief = config_api.tiger_pomdp_config(n_particles=30)
   print(f"States: {env.states}")
   print(f"Actions: {env.actions}")
   print(f"Observations: {env.observations}")

   # Step 2: Create the planner
   planner = POMCP(
       environment=env,
       discount_factor=env.discount_factor,
       depth=10,
       exploration_constant=50.0,
       name="POMCP_Tiger",
       n_simulations=1000
   )

   # Step 3: Belief state is already initialized from config

   # Step 4: Get optimal action
   action, run_data = planner.action(initial_belief)
   print(f"Recommended action: {action[0]}")

   # Step 5: Take action and update belief
   next_state, observation, reward = env.sample_next_step(state=initial_belief.sample(), action=action[0])
   print(f"Observation: {observation}")
   print(f"Reward: {reward}")

Core Concepts
------------

**Environments**

Environments define the POMDP problem structure:

.. code-block:: python

   # All environments inherit from the base Environment class
   from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI

   config_api = EnvironmentConfigsAPI()
   env, belief = config_api.continuous_observations_continuous_actions_light_dark_pomdp_config()

   # Key methods
   state = env.reset()                    # Initialize environment
   obs, reward, done = env.step(action)   # Take action
   states = env.get_states()              # Get state space
   actions = env.get_actions()            # Get action space

**Planners**

Planners compute optimal actions given beliefs:

.. code-block:: python

   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner

   # MCTS-based planner
   pomcp = POMCP(
       environment=env,
       discount_factor=env.discount_factor,
       depth=10,
       exploration_constant=10.0,
       name="POMCP_Test",
       n_simulations=500
   )

   # Sparse sampling planner
   sparse = StandardSparseSamplingDiscreteActionsPlanner(env, branching_factor=100, depth=2)

   # Get action from planner
   action, run_data = pomcp.action(belief_state)

**Belief States**

Belief states represent uncertainty over the true state:

.. code-block:: python

   from POMDPPlanners.core.belief import WeightedParticleBelief

   # Create uniform belief over all states
   from POMDPPlanners.core.belief import get_initial_belief
   belief = get_initial_belief(env, n_particles=1000)

   # Sample from belief
   state_sample = belief.sample()

   # Get belief probabilities
   probabilities = belief.get_state_probabilities()

Running Simulations
------------------

For systematic evaluation, use the simulation framework:

.. code-block:: python

   from POMDPPlanners.simulations.simulator import POMDPSimulator
   from POMDPPlanners.utils.config_loader import load_config

   # Create configuration
   config = {
       'environment': {
           'type': 'TigerPOMDP'
       },
       'planner': {
           'type': 'POMCP',
           'num_simulations': 1000,
           'exploration_constant': 50.0
       },
       'simulation': {
           'num_episodes': 100,
           'max_steps_per_episode': 30
       }
   }

   # Run simulation
   simulator = Simulator(config)
   results = simulator.run()

   print(f"Average reward: {results['average_reward']:.2f}")
   print(f"Success rate: {results['success_rate']:.2f}")

Visualization Example
--------------------

Visualize planner performance and environment dynamics:

.. code-block:: python

   from POMDPPlanners.utils.visualization import plot_episode_results
   import matplotlib.pyplot as plt

   # Run episode and collect data
   episode_data = {
       'rewards': [],
       'actions': [],
       'observations': [],
       'beliefs': []
   }

   # Episode loop
   belief = initial_belief
   for step in range(20):
       action = planner.get_action(belief)
       obs, reward, done = env.step(action)
       belief = planner.update_belief(belief, action, obs)

       episode_data['rewards'].append(reward)
       episode_data['actions'].append(action)
       episode_data['observations'].append(obs)
       episode_data['beliefs'].append(belief.get_state_probabilities())

       if done:
           break

   # Plot results
   plot_episode_results(episode_data)
   plt.show()

Available Environments
---------------------

**Classic Problems**

.. code-block:: python

   # Tiger POMDP - classic two-door problem
   from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
   config_api = EnvironmentConfigsAPI()
   tiger_env, tiger_belief = config_api.tiger_pomdp_config()

**Navigation Tasks**

.. code-block:: python

   # Light-Dark POMDP - position-dependent observation noise
   from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
   config_api = EnvironmentConfigsAPI()
   light_dark_env, light_dark_belief = config_api.continuous_observations_continuous_actions_light_dark_pomdp_config()

**Control Problems**

.. code-block:: python

   # CartPole with partial observability
   from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
   config_api = EnvironmentConfigsAPI()
   cartpole_env, cartpole_belief = config_api.cartpole_pomdp_config()

   # Mountain Car with partial observability
   from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
   config_api = EnvironmentConfigsAPI()
   mountain_car_env, mountain_car_belief = config_api.mountain_car_pomdp_config()

**Manipulation Tasks**

.. code-block:: python

   # Object pushing with uncertainty
   from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
   config_api = EnvironmentConfigsAPI()
   push_env, push_belief = config_api.push_pomdp_config()

Available Planners
-----------------

**MCTS-Based Algorithms**

.. code-block:: python

   # POMCP - Partially Observable Monte Carlo Planning
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   pomcp = POMCP(env, num_simulations=1000, exploration_constant=50.0)

   # PFT-DPW - Progressive Widening with Particle Filter Trees
   from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
   pft_dpw = PFT_DPW(env, num_simulations=500, k_action=5.0, alpha_action=0.5)

   # Sparse PFT - Sparse Particle Filter Trees
   from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT
   sparse_pft = SparsePFT(env, num_simulations=500)

**Other Algorithms**

.. code-block:: python

   # Sparse Sampling
   from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
   sparse = StandardSparseSamplingDiscreteActionsPlanner(env, branching_factor=100, depth=10)

Configuration Management
-----------------------

Use YAML files for reproducible experiments:

.. code-block:: yaml

   # config.yaml
   environment:
     type: "TigerPOMDP"

   planner:
     type: "POMCP"
     num_simulations: 1000
     exploration_constant: 50.0
     discount_factor: 0.95

   simulation:
     num_episodes: 100
     max_steps_per_episode: 30
     random_seed: 42

.. code-block:: python

   from POMDPPlanners.utils.config_loader import load_config
   from POMDPPlanners.simulations.simulator import POMDPSimulator

   # Load configuration
   config = load_config("config.yaml")

   # Run simulation
   simulator = Simulator(config)
   results = simulator.run()

Hyperparameter Tuning
---------------------

Optimize planner performance automatically:

.. code-block:: python

   from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import HyperParameterTuningSimulations

   # Define hyperparameter space
   hyperparams = {
       "num_simulations": [500, 1000, 2000],
       "exploration_constant": [10.0, 50.0, 100.0],
       "discount_factor": [0.9, 0.95, 0.99]
   }

   # Run optimization
   tuner = HyperParameterTuningSimulations(
       base_config="config.yaml",
       hyperparameters=hyperparams,
       optimization_metric="average_reward"
   )

   best_params = tuner.optimize(num_trials=50)
   print(f"Best parameters: {best_params}")

Next Steps
----------

**Explore Examples**

Check out the visualization examples in the repository:

.. code-block:: bash

   python light_dark_pomdp_visualization_example.py
   python pomcpow_rock_sample_visualization_demo.py

**Read the Tutorials**

Dive deeper with comprehensive tutorials:

- :doc:`tutorials/environments` - Creating custom environments
- :doc:`tutorials/planners` - Implementing new algorithms
- :doc:`tutorials/experiments` - Advanced experiment design

**API Reference**

Browse the complete API documentation: :doc:`api/index`

**Get Help**

- `GitHub Issues <https://github.com/yaacovpariente/POMDPPlanners/issues>`_
- `Discussions <https://github.com/yaacovpariente/POMDPPlanners/discussions>`_
- :doc:`contributing` - Join the development community!