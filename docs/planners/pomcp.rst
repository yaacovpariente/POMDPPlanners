POMCP
=====

Partially Observable Monte Carlo Planning. POMCP grows a history tree from the
current belief: each simulation descends the tree by UCB1, samples a state from
the node's particle set, steps the generative model, and finishes with a random
rollout. Returns are averaged back up the path it came down. The belief at a
node is the set of particles that reached it, so no explicit belief update is
needed during the search.

It is the default starting point for a discrete-action problem, and the
baseline every other MCTS planner here is a modification of.

Notes
-----

- Original paper: Silver, D., & Veness, J. (2010). *Monte-Carlo Planning in
  Large POMDPs*. NeurIPS 23.
  https://papers.nips.cc/paper_files/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html
- Set exactly one budget: ``n_simulations`` or ``time_out_in_seconds``.
- Observations index the tree by equality, so a problem with many distinct
  observations splits the tree faster than the budget can fill it. Use
  :doc:`pomcpow` or :doc:`pft_dpw` there.

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
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP

   np.random.seed(42)
   tiger = TigerPOMDP(discount_factor=0.95)

   planner = POMCP(
       environment=tiger,
       discount_factor=0.95,
       depth=5,
       exploration_constant=1.0,
       name="ExamplePlanner",
       n_simulations=10,
   )

   belief = get_initial_belief(tiger, n_particles=10)
   actions, run_data = planner.action(belief)

Parameters
----------

.. autoclass:: POMDPPlanners.planners.mcts_planners.pomcp.POMCP
   :members:
   :show-inheritance:
   :no-index:
