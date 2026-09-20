Sparse PFT
==========

Sparse-PFT is PFT with a fixed observation branching factor instead of
progressive widening: every action node gets exactly ``belief_child_num``
belief children, sampled once. There is nothing to widen and no ``k_o`` /
``alpha_o`` to tune, so the shape of the tree is decided before the search
starts and the budget goes entirely into depth and value estimates.

Use it when the action set is discrete and the budget per decision is small.

Notes
-----

- ``c_ucb`` and ``beta_ucb`` set the exploration term; ``belief_child_num``
  sets the width.
- Takes a discrete-action environment: it enumerates the action set rather than
  sampling it.

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
   from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT

   np.random.seed(42)
   tiger = TigerPOMDP(discount_factor=0.95)

   planner = SparsePFT(
       environment=tiger,
       discount_factor=0.95,
       depth=5,
       c_ucb=1.0,
       beta_ucb=2.0,
       belief_child_num=3,
       n_simulations=10,
       name="ExamplePlanner",
   )

   belief = get_initial_belief(tiger, n_particles=10)
   actions, run_data = planner.action(belief)

Parameters
----------

.. autoclass:: POMDPPlanners.planners.mcts_planners.sparse_pft.SparsePFT
   :members:
   :show-inheritance:
   :no-index:
