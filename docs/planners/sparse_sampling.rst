Sparse sampling
===============

The classical forward-search baseline. Sparse sampling builds a full lookahead
tree of fixed depth: at every node it expands every action, samples
``branching_factor`` outcomes per action, and solves the tree by dynamic
programming from the leaves up. Nothing is adaptive — the tree's shape is a
function of ``branching_factor`` and ``depth`` alone, and the same tree is built
no matter what the values turn out to be.

That is exactly why it is useful as a comparison point: it has no exploration
heuristic to tune, so a planner that beats it is beating the search, not the
tuning.

Notes
-----

- Original paper: Kearns, M., Mansour, Y., & Ng, A. Y. (2002). *A Sparse
  Sampling Algorithm for Near-Optimal Planning in Large Markov Decision
  Processes*. Machine Learning 49, 193-208.
  https://link.springer.com/article/10.1023/A:1017932429737
- Cost grows as ``(|A| * branching_factor) ** depth``. Small depths only.
- The CVaR variant is in :doc:`constrained_and_cvar`.

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
   from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import (
       SparseSamplingDiscreteActionsPlanner,
   )

   np.random.seed(42)
   tiger = TigerPOMDP(discount_factor=0.95)

   planner = SparseSamplingDiscreteActionsPlanner(
       environment=tiger,
       branching_factor=2,
       depth=2,
       name="ExamplePlanner",
   )

   belief = get_initial_belief(tiger, n_particles=10)
   actions, run_data = planner.action(belief)

Parameters
----------

.. autoclass:: POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling.SparseSamplingDiscreteActionsPlanner
   :members:
   :show-inheritance:
   :no-index:
