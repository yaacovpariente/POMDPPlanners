POMCP-DPW
=========

POMCP with double progressive widening. The number of action children and the
number of observation children of a node are both capped by ``k * n**alpha``,
where ``n`` is the node's visit count. A node therefore starts narrow and widens
only as it is visited, which keeps the tree deep enough to be useful when the
action or observation space is large or continuous.

Beliefs stay unweighted particle sets, as in :doc:`pomcp`. For weighted
particles, use :doc:`pomcpow`.

Notes
-----

- Original paper: Sunberg, Z. N., & Kochenderfer, M. J. (2018). *Online
  Algorithms for POMDPs with Continuous State, Action, and Observation Spaces*.
  ICAPS 28(1), 259-263. https://ojs.aaai.org/index.php/ICAPS/article/view/13882
- ``k_a`` / ``alpha_a`` control action widening, ``k_o`` / ``alpha_o``
  observation widening. ``alpha = 0`` freezes a branch at ``k`` children.
- Needs an ``action_sampler``; with a discrete environment that is
  ``DiscreteActionSampler(env.get_actions())``.

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
     - ✔️
   * - Action widening
     - ✔️
   * - Observation widening
     - ✔️
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
   from POMDPPlanners.planners.mcts_planners.pomcp_dpw import POMCP_DPW
   from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

   np.random.seed(42)
   tiger = TigerPOMDP(discount_factor=0.95)
   action_sampler = DiscreteActionSampler(tiger.get_actions())

   planner = POMCP_DPW(
       environment=tiger,
       discount_factor=0.95,
       depth=5,
       exploration_constant=1.0,
       k_o=3.0,
       k_a=3.0,
       alpha_o=0.5,
       alpha_a=0.5,
       action_sampler=action_sampler,
       n_simulations=10,
       name="ExamplePlanner",
   )

   belief = get_initial_belief(tiger, n_particles=10)
   actions, run_data = planner.action(belief)

Parameters
----------

.. autoclass:: POMDPPlanners.planners.mcts_planners.pomcp_dpw.POMCP_DPW
   :members:
   :show-inheritance:
   :no-index:
