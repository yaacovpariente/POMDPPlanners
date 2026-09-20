POMCPOW
=======

POMCP with observation widening and *weighted* particles. Where
:doc:`pomcp_dpw` keeps an unweighted particle set per node, POMCPOW inserts the
sampled state into the observation child's belief with a weight, so the belief
at a node approximates the true posterior instead of the set of states that
happened to generate an identical observation. That is what makes it usable
when observations are continuous and no two are ever equal.

Notes
-----

- Original paper: Sunberg, Z. N., & Kochenderfer, M. J. (2018). *Online
  Algorithms for POMDPs with Continuous State, Action, and Observation Spaces*.
  ICAPS 28(1), 259-263. https://ojs.aaai.org/index.php/ICAPS/article/view/13882
- The weighting needs the environment's observation model, so an environment
  that cannot score an observation cannot be planned on with POMCPOW.
- Risk-sensitive and cost-constrained variants live in
  :doc:`constrained_and_cvar`.

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
   from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
   from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

   np.random.seed(42)
   tiger = TigerPOMDP(discount_factor=0.95)
   action_sampler = DiscreteActionSampler(tiger.get_actions())

   planner = POMCPOW(
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

.. autoclass:: POMDPPlanners.planners.mcts_planners.pomcpow.POMCPOW
   :members:
   :show-inheritance:
   :no-index:
