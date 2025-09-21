Quickstart Guide
================

This guide will get you up and running with POMDPPlanners in just a few minutes.

Your First POMDP Solution
-------------------------

Let's solve the classic Tiger POMDP problem using POMCP:

.. code-block:: python

   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.core.belief import WeightedParticleBelief

   # Step 1: Create the environment
   env = TigerPOMDP()
   print(f"States: {env.get_states()}")
   print(f"Actions: {env.get_actions()}")
   print(f"Observations: {env.get_observations()}")

   # Step 2: Create the planner
   planner = POMCP(env, num_simulations=1000, exploration_constant=50.0)

   # Step 3: Initialize belief state
   initial_belief = WeightedParticleBelief.create_uniform_belief(
       env.get_states(), num_particles=1000
   )

   # Step 4: Get optimal action
   action = planner.get_action(initial_belief)
   print(f"Recommended action: {action}")

   # Step 5: Take action and update belief
   observation, reward, done = env.step(action)
   updated_belief = planner.update_belief(initial_belief, action, observation)

   print(f"Observation: {observation}")
   print(f"Reward: {reward}")
   print(f"Episode done: {done}")

Core Concepts
------------

**Environments**

Environments define the POMDP problem structure:

.. code-block:: python

   # All environments inherit from the base Environment class
   from POMDPPlanners.environments.light_dark_pomdp import ContinuousLightDarkPOMDP

   env = ContinuousLightDarkPOMDP()

   # Key methods
   state = env.reset()                    # Initialize environment
   obs, reward, done = env.step(action)   # Take action
   states = env.get_states()              # Get state space
   actions = env.get_actions()            # Get action space

**Planners**

Planners compute optimal actions given beliefs:

.. code-block:: python

   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.planners.sparse_sampling_planner import SparseSamplingPlanner

   # MCTS-based planner
   pomcp = POMCP(env, num_simulations=500, exploration_constant=10.0)

   # Sparse sampling planner
   sparse = SparseSamplingPlanner(env, num_simulations=100, max_depth=10)

   # Get action from planner
   action = pomcp.get_action(belief_state)

**Belief States**

Belief states represent uncertainty over the true state:

.. code-block:: python

   from POMDPPlanners.core.belief import WeightedParticleBelief

   # Create uniform belief over all states
   belief = WeightedParticleBelief.create_uniform_belief(
       states=env.get_states(),
       num_particles=1000
   )

   # Sample from belief
   state_sample = belief.sample()

   # Get belief probabilities
   probabilities = belief.get_state_probabilities()

Running Simulations
------------------

For systematic evaluation, use the simulation framework:

.. code-block:: python

   from POMDPPlanners.simulations.simulator import Simulator
   from POMDPPlanners.utils.config_loader import create_config

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
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   tiger = TigerPOMDP()

**Navigation Tasks**

.. code-block:: python

   # Light-Dark POMDP - position-dependent observation noise
   from POMDPPlanners.environments.light_dark_pomdp import ContinuousLightDarkPOMDP
   light_dark = ContinuousLightDarkPOMDP(grid_size=10)

**Control Problems**

.. code-block:: python

   # CartPole with partial observability
   from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
   cartpole = CartPolePOMDP()

   # Mountain Car with partial observability
   from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
   mountain_car = MountainCarPOMDP()

**Manipulation Tasks**

.. code-block:: python

   # Object pushing with uncertainty
   from POMDPPlanners.environments.push_pomdp import PushPOMDP
   push = PushPOMDP()

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
   from POMDPPlanners.planners.sparse_sampling_planner import SparseSamplingPlanner
   sparse = SparseSamplingPlanner(env, num_simulations=100, max_depth=10)

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
   from POMDPPlanners.simulations.simulator import Simulator

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