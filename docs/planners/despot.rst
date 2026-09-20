DESPOT
======

Determinized Sparse Partially Observable Tree search. DESPOT samples ``K``
scenarios from the current belief — a start state plus a fixed random number for
every depth — and searches only the tree those scenarios induce. A node is the
subset of scenarios that reach it, and it carries a lower and an upper bound on
its value rather than an average of sampled returns.

The search runs in trials. Each trial walks down the branch with the highest
upper bound and the largest remaining gap, expands the leaf it reaches, and
backs the bounds up the path. A branch is abandoned when its interval can no
longer matter — branch and bound, not exploration bonuses. The search ends when
the root's excess uncertainty falls to noise or the budget runs out.

Notes
-----

- Original papers: Somani, A., Ye, N., Hsu, D., & Lee, W. S. (2013). *DESPOT:
  Online POMDP Planning with Regularization*. NeurIPS 26; extended as Ye, N.,
  Somani, A., Hsu, D., & Lee, W. S. (2017), JAIR 58, 231-266.
- There is no exploration constant and no visit-count selection here. The knobs
  that matter are ``n_scenarios``, ``depth`` and the bounds.
- ``eta`` sets the target on the root gap; ``pruning_constant`` (``lambda``) is
  charged once, over the finished tree, purely to pick the final action. For
  regularization inside the search, use :doc:`ardespot`.

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
   from POMDPPlanners.planners.scenario_tree_planners.despot import DESPOT

   np.random.seed(0)
   tiger = TigerPOMDP(discount_factor=0.95)

   planner = DESPOT(
       environment=tiger,
       discount_factor=0.95,
       depth=5,
       name="ExampleDESPOT",
       n_scenarios=8,
       n_simulations=20,
   )

   belief = get_initial_belief(tiger, n_particles=20)
   actions, run_data = planner.action(belief)

Parameters
----------

.. autoclass:: POMDPPlanners.planners.scenario_tree_planners.despot.DESPOT
   :members:
   :show-inheritance:
   :no-index:
