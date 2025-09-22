Simulations
===========

The simulation framework provides comprehensive tools for running experiments, managing episodes, tuning hyperparameters, and analyzing results across different environments and planners.

Simulation Components
--------------------

**Core Simulation**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.simulations.simulator.Simulator
   POMDPPlanners.simulations.episodes.run_episode

**Experiment Management**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.simulations.simulations_api.SimulationsAPI
   POMDPPlanners.simulations.hyper_parameter_tuning_simulations.HyperParameterTuningSimulations

**Statistics & Analysis**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.simulations.simulation_statistics.compute_statistics_environment_policy_pair
   POMDPPlanners.utils.statistics.compute_confidence_interval

Basic Simulation Usage
---------------------

**Single Episode**

.. code-block:: python

   from POMDPPlanners.simulations.episodes import run_episode
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.core.belief import get_initial_belief

   # Setup environment and planner
   env = TigerPOMDP()
   planner = POMCP(env, num_simulations=500)
   belief = get_initial_belief(env, n_particles=1000)

   # Run single episode
   history = run_episode(
       environment=env,
       policy=planner,
       initial_belief=belief,
       num_steps=20
   )

   print(f"Total reward: {history.total_reward}")
   print(f"Episode length: {len(history.history)}")

**Multiple Episodes with Simulator**

.. code-block:: python

   from POMDPPlanners.simulations.simulator import Simulator

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
           'num_episodes': 100,
           'max_steps_per_episode': 30,
           'random_seed': 42
       }
   }

   simulator = Simulator(config)
   results = simulator.run()

   print(f"Average reward: {results['average_reward']:.2f}")
   print(f"Success rate: {results['success_rate']:.2f}")

Configuration-Based Experiments
-------------------------------

**YAML Configuration Files**

Create ``experiment_config.yaml``:

.. code-block:: yaml

   environment:
     type: "TigerPOMDP"
     discount_factor: 0.95

   planner:
     type: "POMCP"
     num_simulations: 1000
     exploration_constant: 50.0
     depth: 30

   simulation:
     num_episodes: 50
     max_steps_per_episode: 25
     random_seed: 42
     parallel_episodes: 4

Load and run:

.. code-block:: python

   from POMDPPlanners.utils.config_loader import load_config
   from POMDPPlanners.simulations.simulator import Simulator

   config = load_config("experiment_config.yaml")
   simulator = Simulator(config)
   results = simulator.run()

Hyperparameter Tuning
---------------------

**Grid Search Optimization**

.. code-block:: python

   from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import HyperParameterTuningSimulations

   # Define hyperparameter space
   hyperparams = {
       "num_simulations": [500, 1000, 2000],
       "exploration_constant": [10.0, 50.0, 100.0],
       "depth": [20, 30, 40]
   }

   # Setup tuning experiment
   tuner = HyperParameterTuningSimulations(
       base_config="base_experiment.yaml",
       hyperparameters=hyperparams,
       optimization_metric="average_reward",
       num_episodes_per_config=20
   )

   # Run optimization
   best_params, all_results = tuner.optimize()

   print(f"Best parameters: {best_params}")
   print(f"Best performance: {all_results['best_score']:.3f}")

**Bayesian Optimization with Optuna**

.. code-block:: python

   import optuna
   from POMDPPlanners.simulations.simulator import Simulator

   def objective(trial):
       # Suggest hyperparameters
       num_sims = trial.suggest_int('num_simulations', 100, 2000)
       exploration = trial.suggest_float('exploration_constant', 1.0, 100.0)

       # Create config with suggested parameters
       config = {
           'environment': {'type': 'TigerPOMDP'},
           'planner': {
               'type': 'POMCP',
               'num_simulations': num_sims,
               'exploration_constant': exploration
           },
           'simulation': {'num_episodes': 20}
       }

       # Run simulation
       simulator = Simulator(config)
       results = simulator.run()

       return results['average_reward']

   # Run optimization
   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=50)

   print(f"Best parameters: {study.best_params}")

Distributed Computing
---------------------

**Ray-Based Parallel Execution**

.. code-block:: python

   import ray
   from POMDPPlanners.simulations.simulations_api import SimulationsAPI

   # Initialize Ray
   ray.init()

   # Create multiple experiment configurations
   experiments = [
       {"environment": {"type": "TigerPOMDP"}, "planner": {"type": "POMCP"}},
       {"environment": {"type": "CartPolePOMDP"}, "planner": {"type": "PFT_DPW"}},
       # ... more configurations
   ]

   # Run experiments in parallel
   api = SimulationsAPI()
   results = api.run_multiple_environments_and_policies_parallel(
       experiments,
       num_episodes=50,
       num_workers=4
   )

   ray.shutdown()

**Dask-Based Cluster Computing**

.. code-block:: python

   from dask.distributed import Client
   from POMDPPlanners.simulations.simulations_deployment.task_managers import DaskTaskManager

   # Connect to Dask cluster
   client = Client('scheduler-address:8786')

   # Setup task manager
   task_manager = DaskTaskManager(client)

   # Submit large-scale experiments
   futures = task_manager.submit_experiment_batch(
       experiment_configs=large_experiment_list,
       batch_size=10
   )

   # Collect results
   results = task_manager.gather_results(futures)

Statistical Analysis
-------------------

**Performance Metrics**

.. code-block:: python

   from POMDPPlanners.simulations.simulation_statistics import compute_statistics_environment_policy_pair

   # Compute comprehensive statistics
   stats = compute_statistics_environment_policy_pair(
       env=environment,
       histories=episode_histories,
       alpha=0.05,  # 95% confidence intervals
       confidence_interval_level=0.95
   )

   print(f"Mean reward: {stats['mean_total_reward']:.3f}")
   print(f"95% CI: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")
   print(f"Success rate: {stats['success_rate']:.3f}")

**Result Comparison**

.. code-block:: python

   from POMDPPlanners.utils.statistics import compare_algorithm_performance

   # Compare multiple algorithms
   comparison = compare_algorithm_performance([
       ("POMCP", pomcp_results),
       ("PFT-DPW", pft_dwp_results),
       ("Sparse Sampling", sparse_results)
   ])

   print("Algorithm Performance Comparison:")
   for algo, stats in comparison.items():
       print(f"{algo}: {stats['mean']:.3f} ± {stats['std']:.3f}")

Experiment Caching
------------------

**Automatic Result Caching**

.. code-block:: python

   from POMDPPlanners.utils.simulations_caching import CachedSimulator

   # Simulator with automatic caching
   cached_simulator = CachedSimulator(
       cache_dir="./experiment_cache",
       cache_results=True
   )

   # Results are automatically cached and reused
   results1 = cached_simulator.run(config)  # Runs experiment
   results2 = cached_simulator.run(config)  # Loads from cache

Visualization and Reporting
---------------------------

**Performance Plots**

.. code-block:: python

   from POMDPPlanners.utils.visualization import plot_performance_comparison
   import matplotlib.pyplot as plt

   # Plot algorithm comparison
   plot_performance_comparison(
       results_dict={
           'POMCP': pomcp_results,
           'PFT-DPW': pft_results
       },
       metric='total_reward',
       title='Algorithm Performance Comparison'
   )
   plt.show()

**Learning Curves**

.. code-block:: python

   from POMDPPlanners.utils.visualization import plot_learning_curve

   # Plot performance over episodes
   plot_learning_curve(
       episode_rewards=rewards_over_time,
       window_size=10,  # Moving average window
       title='Learning Progress'
   )
   plt.show()

See Also
--------

- :doc:`../examples/experiments` - Complete experiment examples
- :doc:`../api/simulations` - Full simulation API reference
- :doc:`../contributing` - Contributing simulation studies