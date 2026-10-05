Statistics
==========

These helpers turn a list of episode returns into numbers you can compare: a
mean with a confidence interval, a tail risk (CVaR), and a quantile with its
own interval. The simulator already uses them to build its statistics table,
so call them yourself only when you analyse episodes outside a run, or need a
statistic the table does not have.

Example
-------

.. code-block:: python

   import numpy as np
   from POMDPPlanners.utils.statistics_utils import (
       confidence_interval,
       cvar_estimator,
       quantile_confidence_interval,
   )

   returns = np.array([12.5, 8.3, 15.7, -2.1, 9.8, 13.2, 6.4, 11.0, -1.5, 14.3])

   low, high = confidence_interval(returns, confidence=0.95)
   print(f"mean {returns.mean():.2f}, 95% CI [{low:.2f}, {high:.2f}]")
   # mean 8.76, 95% CI [4.31, 13.21]

   # cvar_estimator averages the *largest* alpha fraction of its input.
   # Negate twice to get the mean of the worst 20% of returns.
   print(f"CVaR of returns at 20%: {-cvar_estimator(-returns, alpha=0.2):.2f}")
   # CVaR of returns at 20%: -1.80

   low, high, _, _ = quantile_confidence_interval(returns, alpha=0.1, conf_level=0.95)
   print(f"10% quantile in [{low:.2f}, {high:.2f}]")

Notes
-----

- ``cvar_estimator(x, alpha)`` treats large values as bad, as for costs. For
  returns, where small is bad, pass ``-returns`` and negate the result. The
  simulator does this for ``return_cvar``.
- ``confidence_interval`` uses a Student t-interval. The t-distribution widens
  the interval to account for estimating the standard deviation from the
  sample, which matters at the 10 to 100 episodes of a planner comparison.
- ``cvar_confidence_interval`` is a finite-sample bound. It needs bounds on
  the data; pass the true ones when you know them. Without them it uses the
  sample minimum and maximum, which can make the interval narrower than it
  should be.

Comparing planners
------------------

To compare several planners, pass the ``histories`` from a simulation run to
``compute_statistics_environments_policies_comparison``. It returns the same
DataFrame that ``run_multiple_environments_and_policies`` returns, one row per
environment and planner pair with ``_ci_lower`` and ``_ci_upper`` columns. If
two planners' intervals do not overlap, the difference holds at that
confidence level. If they overlap, the table alone cannot settle it; run more
episodes.

.. autofunction:: POMDPPlanners.simulations.simulation_statistics.compute_statistics_environment_policy_pair

.. autofunction:: POMDPPlanners.simulations.simulation_statistics.compute_statistics_environments_policies_comparison

.. autofunction:: POMDPPlanners.simulations.simulation_statistics.metrics_dict_to_dataframe

Estimators and intervals
------------------------

.. autofunction:: POMDPPlanners.utils.statistics_utils.confidence_interval

.. autofunction:: POMDPPlanners.utils.statistics_utils.cvar_estimator

.. autofunction:: POMDPPlanners.utils.statistics_utils.cvar_confidence_interval

.. autofunction:: POMDPPlanners.utils.statistics_utils.quantile_confidence_interval

.. autofunction:: POMDPPlanners.utils.statistics_utils.cvar_estimator_from_dist

Distances between distributions
-------------------------------

.. autofunction:: POMDPPlanners.utils.statistics_utils.tv_distance
