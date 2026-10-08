Metrics
=======

Every simulation run reports three groups of metrics for each environment and
planner pair: standard ones computed for every environment (return, CVaR,
timings), environment ones (task completion, collisions), and planner ones
(tree depth, visit counts). Each comes as a value with a confidence interval.
When two planners' intervals do not overlap, the gap between them is larger
than the spread from which episodes happened to be drawn.

Environment metrics are declared, not hand-computed. The environment reports
raw numbers each step through ``step_info``, and lists in ``get_metric_specs``
how each number becomes a metric. The shared code then does the per-episode
reduction, the averaging and the confidence interval.

Why per-step channels
---------------------

Episodes run in worker processes, and metrics are computed later in the
parent process on a different copy of the environment. A count kept on
``self`` during the episode never reaches the parent. ``step_info`` returns a
``{name: float}`` dict that the simulator stores on each
:class:`~POMDPPlanners.core.simulation.history.StepData` as ``info``, so the
values travel back with the episode.

``step_info`` is also called once more at the end of a terminated episode,
with ``action`` and ``next_state`` set to ``None``. Implementations must
accept that. :doc:`../environments/base` documents both methods on the
``Environment`` class.

Example
-------

The Tiger environment declares two metrics:

.. code-block:: python

   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

   env = TigerPOMDP(discount_factor=0.95)
   print(env.get_metric_names())  # ['task_completion_rate', 'average_listens']
   print(env.step_info(state="tiger_left", action="open_right", next_state="tiger_left"))
   # {'correct_door_opened': 1.0, 'listened': 0.0}

This is how two specs turn per-step channels into metrics. Here the per-step
values are written by hand; in a run they come from ``step_info``:

.. code-block:: python

   from POMDPPlanners.core.simulation.step_info_metrics import (
       EpisodeReduction,
       StepInfoMetric,
       aggregate_step_info_metrics,
   )

   specs = [
       # Did the episode ever reach the goal? Averaged over episodes, a rate.
       StepInfoMetric(name="task_completion_rate", channel="at_goal",
                      per_episode=EpisodeReduction.ANY),
       # How many collisions per episode, averaged over episodes.
       StepInfoMetric(name="average_collisions", channel="collided",
                      per_episode=EpisodeReduction.SUM),
   ]

   # What step_info returned on each step of three short episodes.
   episodes = [
       [{"at_goal": 0.0, "collided": 1.0}, {"at_goal": 1.0, "collided": 0.0}],
       [{"at_goal": 0.0, "collided": 0.0}, {"at_goal": 0.0, "collided": 1.0}],
       [{"at_goal": 1.0, "collided": 0.0}],
   ]
   for metric in aggregate_step_info_metrics(episodes, specs):
       print(metric.name, metric.value, metric.lower_confidence_bound,
             metric.upper_confidence_bound)

With three episodes the 95% interval is wide, about ``[-0.77, 2.10]`` around
``0.67``. The t-interval is that wide because it rests on three episodes; it
narrows as the episode count grows.

Use the names in :class:`~POMDPPlanners.core.simulation.metrics.CommonMetricName`
for concepts that every environment shares, such as task completion, so one
idea is not reported under several spellings.

Metric specs
------------

.. autoclass:: POMDPPlanners.core.simulation.step_info_metrics.StepInfoMetric

.. autoclass:: POMDPPlanners.core.simulation.step_info_metrics.EpisodeReduction

.. autofunction:: POMDPPlanners.core.simulation.step_info_metrics.aggregate_step_info_metrics

.. autoclass:: POMDPPlanners.core.simulation.metrics.MetricValue

Built-in metric names
---------------------

The standard metrics are computed for every pair. ``return_cvar`` is skipped
when the environment has no ``reward_range``, because its confidence interval
needs bounds on the return.

.. autoclass:: POMDPPlanners.simulations.simulation_statistics.StandardMetrics
   :members:
   :undoc-members:

.. autoclass:: POMDPPlanners.core.simulation.metrics.CommonMetricName

Planner metrics
---------------

A planner returns a
:class:`~POMDPPlanners.core.policy.PolicyRunData` with every action. Each of
its info variables is averaged within an episode, then across episodes, and
appears in the results as ``policy_info_<name>``.

Both classes are documented on :doc:`../planners/base`, next to the planner
interface that returns them.

Listing the metrics of a pair
-----------------------------

Use these to find the exact names to optimize in a tuning study.

.. autofunction:: POMDPPlanners.simulations.simulation_statistics.get_metric_names_from_environment_policy_pair

.. autofunction:: POMDPPlanners.simulations.simulation_statistics.get_available_optimization_metrics
