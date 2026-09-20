AR-DESPOT
=========

Anytime Regularized DESPOT. It searches the same determinized scenario tree as
:doc:`despot`, but carries the regularization inside the tree instead of
applying it once at the end, and it is anytime: the tree holds a complete, legal
answer after every trial, so stopping early still returns an action and a longer
budget returns a better one.

Each belief node keeps three numbers rather than two — the lower bound ``l``,
the upper bound ``U``, and a regularized value ``mu``. Descent, the stopping
test and pruning all run on ``mu``; the final action is read off ``l``. Values
are kept unnormalized so a child's value can be added to its parent's without
renormalizing, which is what lets one ``lambda`` per action node be subtracted
directly.

Notes
-----

- Original paper: Ye, N., Somani, A., Hsu, D., & Lee, W. S. (2017). *DESPOT:
  Online POMDP Planning with Regularization*. JAIR 58, Sec. 5.
- ``pruning_constant`` is the regularization weight ``lambda``. At
  ``lambda = 0`` the search still differs from plain DESPOT: pruning and the
  stopping test remain regularized quantities.
- Subclasses :class:`DESPOT <POMDPPlanners.planners.scenario_tree_planners.despot.DESPOT>`
  and reuses its scenario resampling and leaf bounds.

Can I use?
----------

.. list-table::
   :header-rows: 1
   :widths: 34 14

   * - Capability
     - Supported
   * - Discrete actions
     - ✔️
   * - Continuous actions
     - ❌
   * - Action widening
     - ❌
   * - Observation widening
     - ❌
   * - Cost constraints
     - ❌
   * - GPU
     - ❌

Example
-------

.. code-block:: python

   import numpy as np
   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.scenario_tree_planners.ardespot import ARDESPOT

   np.random.seed(0)
   tiger = TigerPOMDP(discount_factor=0.95)

   planner = ARDESPOT(
       environment=tiger,
       discount_factor=0.95,
       depth=5,
       name="ExampleARDESPOT",
       n_scenarios=8,
       pruning_constant=0.01,
       n_simulations=20,
   )

   belief = get_initial_belief(tiger, n_particles=20)
   actions, run_data = planner.action(belief)

Parameters
----------

.. autoclass:: POMDPPlanners.planners.scenario_tree_planners.ardespot.ARDESPOT
   :members:
   :show-inheritance:
   :no-index:
