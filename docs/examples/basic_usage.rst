Basic Usage Examples
===================

This page provides simple, working examples to get you started with POMDPPlanners quickly.

Your First POMDP Solution
-------------------------

Let's solve the Tiger POMDP problem step by step:

.. code-block:: python

   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.core.belief import get_initial_belief

   # Step 1: Create the environment
   env = TigerPOMDP(discount_factor=0.95)

   # Step 2: Create the planner
   planner = POMCP(
       environment=env,
       num_simulations=1000,
       exploration_constant=50.0,
       depth=30
   )

   # Step 3: Get initial belief
   initial_belief = get_initial_belief(env, n_particles=1000)

   # Step 4: Plan and act
   action, run_data = planner.action(initial_belief)
   print(f"Recommended action: {action}")
   print(f"Planning took {run_data.info_variables['planning_time']:.3f} seconds")

Running a Complete Episode
-------------------------

Here's how to run a full episode with belief updates:

.. code-block:: python

   from POMDPPlanners.simulations.episodes import run_episode
   from POMDPPlanners.utils.logger import get_logger

   # Setup logging
   logger = get_logger("basic_example")

   # Run complete episode
   history = run_episode(
       environment=env,
       policy=planner,
       initial_belief=initial_belief,
       num_steps=20,
       logger=logger
   )

   # Analyze results
   print(f"Episode completed in {len(history.history)} steps")
   print(f"Total reward: {history.total_reward:.2f}")
   print(f"Final state: {history.history[-1].state}")

   # Print step-by-step breakdown
   for i, step in enumerate(history.history):
       print(f"Step {i}: Action={step.action}, Observation={step.observation}, Reward={step.reward}")

Multiple Episodes with Statistics
--------------------------------

Run multiple episodes and compute statistics:

.. code-block:: python

   from POMDPPlanners.simulations.simulator import Simulator
   from POMDPPlanners.simulations.simulation_statistics import compute_statistics_environment_policy_pair

   # Configuration for multiple episodes
   config = {
       'environment': {
           'type': 'TigerPOMDP',
           'discount_factor': 0.95
       },
       'planner': {
           'type': 'POMCP',
           'num_simulations': 1000,
           'exploration_constant': 50.0
       },
       'simulation': {
           'num_episodes': 50,
           'max_steps_per_episode': 30,
           'random_seed': 42
       }
   }

   # Run simulation
   simulator = Simulator(config)
   results = simulator.run()

   # Print summary statistics
   print("=== Simulation Results ===")
   print(f"Episodes run: {results['num_episodes']}")
   print(f"Average reward: {results['average_reward']:.3f}")
   print(f"Standard deviation: {results['std_reward']:.3f}")
   print(f"Success rate: {results.get('success_rate', 'N/A')}")

Working with Different Environments
-----------------------------------

**Continuous Control (CartPole)**

.. code-block:: python

   from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
   from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
   from POMDPPlanners.planners.planners_utils.dpw import SimpleActionSampler
   import numpy as np

   # Create continuous control environment
   env = CartPolePOMDP(
       discount_factor=0.99,
       noise_cov=np.diag([0.1, 0.1, 0.1, 0.1])
   )

   # Action sampler for continuous actions
   class CartPoleActionSampler(SimpleActionSampler):
       def sample(self, belief_node=None):
           return np.random.choice([0, 1])  # Left or right force

   # Create planner suitable for continuous domains
   planner = PFT_DPW(
       environment=env,
       num_simulations=500,
       action_sampler=CartPoleActionSampler(),
       depth=20
   )

   # Run episode
   belief = get_initial_belief(env, n_particles=500)
   action, _ = planner.action(belief)
   print(f"CartPole action: {action}")

**Navigation (Light-Dark POMDP)**

.. code-block:: python

   from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
       ContinuousLightDarkPOMDP, RewardModelType
   )

   # Create navigation environment
   env = ContinuousLightDarkPOMDP(
       discount_factor=0.95,
       goal_state=np.array([10, 5]),
       start_state=np.array([0, 5]),
       reward_model_type=RewardModelType.STANDARD
   )

   # Use POMCP for navigation
   planner = POMCP(
       environment=env,
       num_simulations=800,
       exploration_constant=10.0
   )

   # Get action for navigation
   belief = get_initial_belief(env, n_particles=1000)
   action, _ = planner.action(belief)
   print(f"Navigation action: {action}")

Comparing Multiple Algorithms
----------------------------

.. code-block:: python

   from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling_planner import SparseSamplingDiscreteActionsPlanner

   # Test multiple algorithms on the same problem
   algorithms = {
       'POMCP': POMCP(
           environment=env,
           num_simulations=1000,
           exploration_constant=50.0
       ),
       'Sparse Sampling': SparseSamplingDiscreteActionsPlanner(
           environment=env,
           branching_factor=3,
           depth=10
       )
   }

   # Compare performance
   results = {}
   for name, planner in algorithms.items():
       # Run episodes with each algorithm
       config['planner'] = {'type': name}
       simulator = Simulator(config)
       result = simulator.run()
       results[name] = result['average_reward']

   # Print comparison
   print("=== Algorithm Comparison ===")
   for name, avg_reward in results.items():
       print(f"{name}: {avg_reward:.3f}")

Visualization Example
--------------------

.. code-block:: python

   from POMDPPlanners.utils.visualization import plot_episode_results
   import matplotlib.pyplot as plt

   # Collect data during episode
   episode_data = {
       'rewards': [],
       'actions': [],
       'observations': [],
       'planning_times': []
   }

   # Run episode and collect data
   belief = get_initial_belief(env, n_particles=1000)
   for step in range(15):
       action, run_data = planner.action(belief)
       observation, reward, done = env.step(action)

       # Update belief (simplified)
       belief = planner.update_belief(belief, action, observation)

       # Store data
       episode_data['rewards'].append(reward)
       episode_data['actions'].append(action)
       episode_data['observations'].append(observation)
       episode_data['planning_times'].append(run_data.info_variables.get('planning_time', 0))

       if done:
           break

   # Create visualization
   fig, axes = plt.subplots(2, 2, figsize=(12, 8))

   # Plot rewards over time
   axes[0, 0].plot(episode_data['rewards'])
   axes[0, 0].set_title('Rewards per Step')
   axes[0, 0].set_xlabel('Step')
   axes[0, 0].set_ylabel('Reward')

   # Plot planning times
   axes[0, 1].plot(episode_data['planning_times'])
   axes[0, 1].set_title('Planning Time per Step')
   axes[0, 1].set_xlabel('Step')
   axes[0, 1].set_ylabel('Time (seconds)')

   # Plot cumulative reward
   cumulative_rewards = np.cumsum(episode_data['rewards'])
   axes[1, 0].plot(cumulative_rewards)
   axes[1, 0].set_title('Cumulative Reward')
   axes[1, 0].set_xlabel('Step')
   axes[1, 0].set_ylabel('Cumulative Reward')

   # Action frequency
   from collections import Counter
   action_counts = Counter(episode_data['actions'])
   axes[1, 1].bar(action_counts.keys(), action_counts.values())
   axes[1, 1].set_title('Action Frequency')
   axes[1, 1].set_xlabel('Action')
   axes[1, 1].set_ylabel('Count')

   plt.tight_layout()
   plt.show()

Quick Configuration Tips
-----------------------

**Performance Tuning**
   - Start with ``num_simulations=100`` for quick testing
   - Increase to ``1000-2000`` for better performance
   - Use ``exploration_constant=50.0`` for Tiger POMDP
   - Adjust ``depth`` based on problem horizon

**Common Issues**
   - Low rewards? Increase ``num_simulations``
   - Slow planning? Decrease ``depth`` or ``num_simulations``
   - Poor exploration? Adjust ``exploration_constant``

**Memory Usage**
   - Use fewer particles (``n_particles=500``) for large state spaces
   - Reduce ``depth`` for memory-constrained environments

Next Steps
----------

- Try :doc:`environments` for more environment examples
- See :doc:`planners` for advanced planner usage
- Check :doc:`experiments` for large-scale experiment setup
- Explore the :doc:`../api/core` for detailed API reference