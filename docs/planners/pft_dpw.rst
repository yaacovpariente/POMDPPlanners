PFT-DPW
=======

Particle Filter Tree with double progressive widening. PFT-DPW searches the
*belief MDP*: a node is a particle belief, and stepping an action runs a full
particle filter update to produce the child belief, which is then treated as a
single state of the belief MDP. Because a node is a belief rather than a
history, the value estimate at a node reflects the whole belief rather than one
sampled state, at the cost of one filter update per simulation step.

Notes
-----

- Original paper: Sunberg, Z. N., & Kochenderfer, M. J. (2018). *Online
  Algorithms for POMDPs with Continuous State, Action, and Observation Spaces*.
  ICAPS 28(1), 259-263. https://ojs.aaai.org/index.php/ICAPS/article/view/13882
- The per-simulation cost is higher than :doc:`pomcp`; the number of particles
  per belief node multiplies it. :doc:`sparse_pft` is the cheaper variant.
- This is the planner the constrained and risk-sensitive variants in
  :doc:`constrained_and_cvar` are built on.

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
   from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
   from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

   np.random.seed(42)
   tiger = TigerPOMDP(discount_factor=0.95)
   action_sampler = DiscreteActionSampler(tiger.get_actions())

   planner = PFT_DPW(
       environment=tiger,
       discount_factor=0.95,
       depth=5,
       name="ExamplePlanner",
       action_sampler=action_sampler,
       k_a=2.0,
       alpha_a=0.5,
       n_simulations=10,
   )

   belief = get_initial_belief(tiger, n_particles=10)
   actions, run_data = planner.action(belief)

Parameters
----------

.. autoclass:: POMDPPlanners.planners.mcts_planners.pft_dpw.PFT_DPW
   :members:
   :show-inheritance:
   :no-index:
