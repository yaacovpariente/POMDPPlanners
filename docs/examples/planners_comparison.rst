Planners Comparison Study
=========================

This example compares two planners, POMCPOW and PFT-DPW, on two environments,
Push and Light-Dark, and reads the result with confidence intervals. It runs in
well under a minute on a laptop, so you can change it and run it again.

Both environments have discrete actions and continuous observations, which is
the case both planners were built for.
They differ in how they hold the belief inside the search. POMCPOW weights one
sampled state per simulation; PFT-DPW runs a particle filter update at every
new node. Which of the two pays off is a question for an experiment, not for
the planner pages.

The setup
---------

Three choices give each planner the same compute and let a rerun reuse the
cache:

- **Every planner gets the same time per decision.** Both use
  ``time_out_in_seconds``, not ``n_simulations``. A PFT-DPW simulation does
  more work than a POMCPOW one, so equal simulation counts would give them
  unequal compute.
- **Both planners share every other parameter**: depth, exploration constant
  and widening. A difference in the result then comes from the algorithm, not
  from one planner being tuned and the other not. Before you draw a conclusion
  from a comparison, tune each planner first (:doc:`hyperparameter_tuning`).
- **NumPy is seeded before the beliefs are built.** The initial belief is
  part of the cache key. With a fixed belief, a second run of the script loads
  the finished episodes from the cache instead of playing them again.

The script
----------

.. code-block:: python

   from pathlib import Path

   import numpy as np

   from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
   from POMDPPlanners.core.simulation import EnvironmentRunParams
   from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
   from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
   from POMDPPlanners.simulations.simulation_apis.local_simulations_api import (
       LocalSimulationsAPI,
   )
   from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

   DISCOUNT = 0.95
   TIME_PER_DECISION = 0.1  # seconds, the same for every planner
   DEPTH = 15


   def make_planners(env, tag):
       action_sampler = DiscreteActionSampler(actions=env.get_actions())
       return [
           POMCPOW(
               environment=env,
               discount_factor=DISCOUNT,
               depth=DEPTH,
               exploration_constant=10.0,
               k_a=4.0,
               alpha_a=0.5,
               k_o=4.0,
               alpha_o=0.5,
               action_sampler=action_sampler,
               time_out_in_seconds=TIME_PER_DECISION,
               name=f"POMCPOW_{tag}",
           ),
           PFT_DPW(
               environment=env,
               discount_factor=DISCOUNT,
               depth=DEPTH,
               exploration_constant=10.0,
               k_a=4.0,
               alpha_a=0.5,
               k_o=4.0,
               alpha_o=0.5,
               action_sampler=action_sampler,
               time_out_in_seconds=TIME_PER_DECISION,
               name=f"PFT_DPW_{tag}",
           ),
       ]


   np.random.seed(0)  # fixes both initial beliefs, so a rerun reuses the cache
   configs = EnvironmentConfigsAPI(discount_factor=DISCOUNT)
   push_env, push_belief = configs.push_pomdp_config(n_particles=100)
   light_dark_env, light_dark_belief = (
       configs.continuous_observations_discrete_actions_light_dark_pomdp_config(n_particles=100)
   )

   environment_run_params = [
       EnvironmentRunParams(
           environment=push_env,
           belief=push_belief,
           policies=make_planners(push_env, "Push"),
           num_episodes=20,
           num_steps=20,
       ),
       EnvironmentRunParams(
           environment=light_dark_env,
           belief=light_dark_belief,
           policies=make_planners(light_dark_env, "LightDark"),
           num_episodes=20,
           num_steps=20,
       ),
   ]

   api = LocalSimulationsAPI()
   results, stats_df = api.run_multiple_environments_and_policies(
       environment_run_params=environment_run_params,
       alpha=0.05,
       confidence_interval_level=0.95,
       experiment_name="planners_comparison",
       n_jobs=-1,
       cache_dir_path=Path("results/planners-comparison"),
   )

   columns = [
       "environment",
       "policy",
       "average_return",
       "average_return_ci_lower",
       "average_return_ci_upper",
       "task_completion_rate",
       "average_action_time",
       "policy_info_root_visit_count",
   ]
   print(stats_df[columns].to_string(index=False))

``EnvironmentConfigsAPI`` returns each environment together with an initial
belief, with preset parameters.
``run_multiple_environments_and_policies`` runs every planner on its
environment, ``num_episodes`` times, with ``n_jobs=-1`` running episodes on all
cores at once. Everything it writes goes under ``cache_dir_path``.

Reading the statistics table
----------------------------

``stats_df`` has one row per environment and planner. One run on a 10-core
Apple M5 laptop printed:

.. code-block:: text

                               environment            policy  average_return  average_return_ci_lower  average_return_ci_upper  task_completion_rate  average_action_time  policy_info_root_visit_count
                                 PushPOMDP      POMCPOW_Push      -76.375927              -102.743274               -50.008580                  0.20             0.115861                   3040.878424
                                 PushPOMDP      PFT_DPW_Push      -46.220239               -77.780618               -14.659859                  0.40             0.105867                   1506.417157
   ContinuousLightDarkPOMDPDiscreteActions POMCPOW_LightDark      -16.962864               -20.109607               -13.816121                  0.25             0.107339                   1664.634167
   ContinuousLightDarkPOMDPDiscreteActions PFT_DPW_LightDark      -14.700247               -18.393459               -11.007034                  0.60             0.101015                   3136.931667

Your numbers will differ: how many simulations fit in 0.1 seconds depends on
the machine and on how many episodes run at once. The columns mean:

``average_return``, ``average_return_ci_lower``, ``average_return_ci_upper``
   The mean discounted return over the episodes, and its confidence interval
   at ``confidence_interval_level`` (95% here).

``task_completion_rate``
   The fraction of episodes in which the goal was reached. Each environment
   defines its own goal.

``average_action_time``
   Seconds spent choosing an action, averaged over steps. It should sit close
   to ``TIME_PER_DECISION`` for every planner, which confirms the budgets
   matched. It runs slightly over because the clock is checked between
   simulations, and the last one is allowed to finish.

``policy_info_root_visit_count``
   How many simulations the planner ran per decision, averaged within each
   episode and then across episodes. Every column that starts with
   ``policy_info_`` is one of the search metrics the planner reports.

Every metric has ``_ci_lower`` and ``_ci_upper`` columns. Print
``stats_df.columns`` to see them all, including each environment's own metrics
and the time spent on each part of the episode loop.

What the run shows
------------------

On both environments the two planners' return intervals overlap, so with 20
episodes this run cannot say which planner has the higher mean return. That
is the usual result of a 20-episode study, and the reason to read the intervals rather than the
means. To separate them, raise ``num_episodes``: the interval's width shrinks
roughly in proportion to one over the square root of the number of episodes.

For single episodes, ``results[env_name][planner_name]`` holds the list of
``History`` objects, and ``pomdp-report serve results/planners-comparison``
serves the run as a website with the episode replays.

See also
--------

- :doc:`../core/simulations` — the simulation API, caching and parallel runs.
- :doc:`hyperparameter_tuning` — tuning each planner before comparing them.
