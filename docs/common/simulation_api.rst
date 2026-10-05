Simulation API and Caching
==========================

The simulation API runs batches of episodes and hands back every episode's
record plus a table of statistics. Use it instead of writing your own episode
loop: it caches each finished episode on disk, gives every episode a fixed
seed, runs episodes in parallel, and logs the run to MLflow so the results
site can show it. A loop of your own gets none of that, and nothing warns you.

For a walk-through of a full study, see the :doc:`../core/simulations` guide.

Example
-------

Run four episodes of POMCP on Tiger and read the average return:

.. code-block:: python

   import tempfile
   from pathlib import Path

   import numpy as np
   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.core.simulation import EnvironmentRunParams
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.simulations import LocalSimulationsAPI

   np.random.seed(0)  # fixes the initial belief, and so the cache key
   env = TigerPOMDP(discount_factor=0.95)
   planner = POMCP(
       environment=env,
       discount_factor=0.95,
       depth=5,
       exploration_constant=1.0,
       name="POMCP",
       time_out_in_seconds=2.0,
   )
   run_params = EnvironmentRunParams(
       environment=env,
       belief=get_initial_belief(env, n_particles=100),
       policies=[planner],
       num_episodes=4,
       num_steps=10,
   )

   run_dir = Path(tempfile.mkdtemp()) / "tiger-demo"
   api = LocalSimulationsAPI(cache_dir_path=run_dir)
   histories, stats = api.run_multiple_environments_and_policies(
       environment_run_params=[run_params],
       alpha=0.05,                    # CVaR over the worst 5% of returns
       confidence_interval_level=0.95,
       experiment_name="tiger_demo",
       n_jobs=1,
       cache_dir_path=run_dir,
   )

   episodes = histories["TigerPOMDP"]["POMCP"]  # one History per episode
   print(len(episodes), stats[["policy", "average_return"]])

``histories`` maps environment name, then policy name, to a list of
:class:`~POMDPPlanners.core.simulation.history.History`. ``stats`` is a
pandas DataFrame with one row per environment and policy pair. Each metric
column comes with ``<metric>_ci_lower`` and ``<metric>_ci_upper`` columns;
:doc:`metrics` lists the metric names.

What a run leaves on disk
-------------------------

Everything goes under ``cache_dir_path``. It defaults to the current
directory, so always pass one.

- ``cache.db`` and the short hex-named folders next to it: the episode cache,
  one entry per episode.
- ``mlruns/``: the MLflow store, which ``pomdp-report serve`` reads.
- ``logs/``: logs of the simulator, the task manager and the cache.
- ``env_policy/logs/``: one log per environment and policy pair.

How caching works
-----------------

Each episode is one cached task. Its key is a hash of:

- the environment's ``config_id``,
- the policy's ``config_id``,
- the initial belief's ``config_id``,
- the episode index, the step limit and the discount factor,
- the episode's seed.

The seed comes from the environment name, the policy name and the episode
index, so the same episode always gets the same seed. Every ``config_id`` is
a hash of the object's public attributes, so changing any parameter of the
environment or the planner gives new keys, and those episodes run again.

The initial belief is part of the key. ``get_initial_belief`` draws random
particles, so seed NumPy before you build the belief, as in the example
above. Without that, every run builds a different belief and no episode is
ever reused.

How a killed run resumes
------------------------

An episode is written to the cache as soon as it finishes, not at the end of
the batch. If a run is killed, run the same script again with the same
``cache_dir_path``. Finished episodes load from the cache and only the
missing ones run. The log line ``Cache status: N tasks cached, M tasks
uncached`` says how many of each.

A cached episode recorded before an environment reported per-step
measurements would score every metric as zero. The task manager detects such
an entry and reruns that one episode, keeping the rest.

``clear_cache_on_start=True`` throws the whole cache away. Do not use it to
explain a surprising result: find which ``config_id`` changed first, because
clearing removes the finished episodes you would compare against.

Entry points
------------

``LocalSimulationsAPI`` runs on one machine with joblib. ``DaskSimulationsAPI``
and ``PBSSimulationsAPI`` have the same methods and run on a Dask cluster or a
PBS queue. All three implement ``SimulationsAPIInterface``.

.. autoclass:: POMDPPlanners.simulations.simulation_apis.local_simulations_api.LocalSimulationsAPI
   :members: run_multiple_environments_and_policies, run_multiple_environments_and_policies_with_initial_debug_run, run_hyperparameter_optimization, run_optimize_and_evaluate, run_all_benchmark_environments_on_planner_generators, run_hyperparameter_tuning_experiment_with_benchmarks, run_all_hyperparameter_benchmarks

.. autoclass:: POMDPPlanners.simulations.simulation_apis.dask_simulations_api.DaskSimulationsAPI

.. autoclass:: POMDPPlanners.simulations.simulation_apis.pbs_simulations_api.PBSSimulationsAPI

.. autoclass:: POMDPPlanners.simulations.simulation_apis.simulations_api_interface.SimulationsAPIInterface

.. autoclass:: POMDPPlanners.simulations.simulator.pomdp_simulator.POMDPSimulator

Run configuration
-----------------

.. autoclass:: POMDPPlanners.core.simulation.simulation_configs.EnvironmentRunParams

.. autoclass:: POMDPPlanners.core.simulation.hyperparameter_tuning.HyperParameterRunParams

.. autoclass:: POMDPPlanners.core.simulation.hyperparameter_tuning.NumericalHyperParameter

.. autoclass:: POMDPPlanners.core.simulation.hyperparameter_tuning.CategoricalHyperParameter

.. autoclass:: POMDPPlanners.core.simulation.hyperparameter_tuning.EarlyStoppingConfig

.. autoclass:: POMDPPlanners.core.simulation.hyperparameter_tuning.ParallelizationLevel

.. autoclass:: POMDPPlanners.simulations.simulations_deployment.task_manager_configs.JoblibConfig

Results
-------

.. autoclass:: POMDPPlanners.core.simulation.history.History
   :members: to_dict, from_dict

.. autoclass:: POMDPPlanners.core.simulation.history.StepData

.. autofunction:: POMDPPlanners.core.simulation.history.history_to_discounted_return_value
