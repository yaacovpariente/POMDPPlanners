Running Simulations
===================

To measure a planner, run many episodes through
:class:`~POMDPPlanners.simulations.simulation_apis.local_simulations_api.LocalSimulationsAPI`.
It runs them in parallel, caches each finished episode, computes confidence
intervals, and logs the run to MLflow. ``run_episode`` is for looking at one
episode by hand.

This page shows how to run each. The full API, the cache internals and the
statistics are in :doc:`../common/simulation_api`.

One episode
-----------

``run_episode`` plays one episode: the planner picks an action, the environment
samples the next state, observation and reward, and the belief is updated with
that observation. It returns a
:class:`~POMDPPlanners.core.simulation.history.History`.

.. code-block:: python

   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.simulations.episodes import run_episode

   env = TigerPOMDP(discount_factor=0.95)
   belief = get_initial_belief(env, n_particles=200)
   planner = POMCP(
       environment=env,
       discount_factor=0.95,
       depth=10,
       exploration_constant=50.0,
       name="POMCP",
       time_out_in_seconds=2.0,
   )

   history = run_episode(
       environment=env, policy=planner, initial_belief=belief, num_steps=10, logger=None
   )
   for step in history.history:
       print(step.action, step.observation, step.reward)

One episode says little about a planner. Returns vary a lot between episodes,
so a comparison needs a batch and a confidence interval.

A batch of episodes
-------------------

Describe each environment with an
:class:`~POMDPPlanners.core.simulation.simulation_configs.EnvironmentRunParams`: the environment,
its initial belief, the planners to run on it, and how many episodes of how many
steps. Then hand the list to ``LocalSimulationsAPI``.

.. code-block:: python

   from pathlib import Path

   import numpy as np

   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.core.simulation import EnvironmentRunParams
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import (
       SparseSamplingDiscreteActionsPlanner,
   )
   from POMDPPlanners.simulations.simulation_apis.local_simulations_api import (
       LocalSimulationsAPI,
   )

   np.random.seed(0)  # fixes the initial belief, so a rerun hits the cache
   env = TigerPOMDP(discount_factor=0.95)
   belief = get_initial_belief(env, n_particles=200)

   planners = [
       POMCP(
           environment=env,
           discount_factor=0.95,
           depth=10,
           exploration_constant=50.0,
           name="POMCP",
           time_out_in_seconds=2.0,
       ),
       SparseSamplingDiscreteActionsPlanner(env, branching_factor=3, depth=2),
   ]

   api = LocalSimulationsAPI()
   results, stats_df = api.run_multiple_environments_and_policies(
       environment_run_params=[
           EnvironmentRunParams(
               environment=env, belief=belief, policies=planners,
               num_episodes=5, num_steps=10,
           )
       ],
       alpha=0.05,
       confidence_interval_level=0.95,
       experiment_name="tiger_demo",
       n_jobs=1,
       cache_dir_path=Path("results/tiger-demo"),
   )

   print(stats_df[["environment", "policy", "average_return",
                   "average_return_ci_lower", "average_return_ci_upper"]])

The call returns two things:

- ``results[env_name][planner_name]`` is the list of ``History`` objects, one
  per episode. Use it for anything the table does not have.
- ``stats_df`` is a pandas table with one row per environment and planner. Each
  metric has a mean and ``_ci_lower`` / ``_ci_upper`` columns: the return,
  the environment's own metrics (Tiger reports ``task_completion_rate`` and
  ``average_listens``), the planner's search metrics with a ``policy_info_``
  prefix, and the time spent per step on each part of the loop.

``alpha`` sets the tail used by the risk metrics (``return_cvar``,
``return_value_at_risk``). ``confidence_interval_level`` sets the width of the
intervals.

Pass ``cache_dir_path`` to the run method
-----------------------------------------

The run method's ``cache_dir_path`` decides where the episodes, the MLflow
store and the logs are written. The ``LocalSimulationsAPI(cache_dir_path=...)``
constructor argument only places the API's own log file. If you leave the run
method's argument out, the results go to ``./cache`` in the current directory.
Give each study its own directory, such as ``results/<run-name>/``, so studies
do not mix.

Rerunning and resuming
----------------------

Every finished episode is stored in the cache under a key built from the
environment's, the planner's and the initial belief's ``config_id``. A rerun
with the same configuration loads those episodes instead of playing them
again. That is what makes a long batch safe to stop: run the same script again
and it finishes only what is missing. The log says how many episodes it found,
for example ``Cache status: 10 tasks cached, 0 tasks uncached``.

Two things change the key without you meaning to:

- **An unseeded initial belief.** ``get_initial_belief`` samples its particles
  at random, and the particles are part of the belief's ``config_id``. Seed
  NumPy before you build the belief, as above, or every rerun starts from
  zero.
- **Any planner parameter.** Changing the budget, the depth or the name makes
  a new planner configuration, with its own episodes.

Do not delete the cache to make a confusing result go away. The cache holds the
episodes that produced the result, and those are the evidence you need to find
out what changed. ``clear_cache_on_start=True`` exists for tests, not for
studies.

Parallel runs
-------------

``n_jobs`` sets how many episodes run at once. ``-1`` uses every core. Use
``n_jobs=1`` while you debug: joblib then runs the episodes one at a time in
the calling process, so errors arrive in order and the traceback comes from
that process, not from a worker.

``run_multiple_environments_and_policies_with_initial_debug_run`` takes the
same arguments, but first runs every pair for 2 episodes of 2 steps. A broken
configuration then fails in seconds instead of after the first long episode.

Planning on a model, executing in a simulator
---------------------------------------------

The episode runner keeps two environments apart. The ``environment`` you pass
to ``run_episode`` or to ``EnvironmentRunParams`` is the *world*: it produces
the true states, observations and rewards. The planner's own ``environment``,
the one it was built with, is its *model*: the search and the belief update use
it. Usually the two are the same object.

They differ when the world is a simulator that cannot serve as a model, such as
the wrappers in :doc:`../environments/realistic`. Those can only step forward
and have no transition or observation density, so a planner cannot search on
them. Build the planner and its initial belief on an approximated model
instead, and pass the simulator as the world:

- the true initial state comes from the world's own reset, not from the belief;
- each raw observation from the world goes through the model's
  ``encode_observation`` before the belief update, so the model decides how a
  sensor reading maps to its observation space;
- the world and the model must have the same ``discount_factor``, or the runner
  raises ``ValueError``. A mismatch would score the episode with one discount
  and plan with another.

Isaac Lab allows one simulator per process, so run its episodes with
``n_jobs=1``.

Comparing planners
------------------

A comparison is only fair if each planner got the same compute per decision.
Give every planner the same ``time_out_in_seconds``, or measure how long each
one's ``n_simulations`` takes and match them. :doc:`../examples/planners_comparison`
runs a full comparison.

To search a planner's parameters before comparing it, see
:doc:`../examples/hyperparameter_tuning`.

Looking at the results
----------------------

Every run writes an MLflow store under ``<cache_dir_path>/mlruns``. To browse
it, with the episode replays, run ``pomdp-report``:

.. code-block:: bash

   pomdp-report serve results/tiger-demo

See also
--------

- :doc:`../common/simulation_api` — the simulation API reference.
- :doc:`beliefs` — building the initial belief.
- :doc:`../environments/custom` — adding an environment to run studies on.
