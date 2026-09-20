Open-loop planning
==================

``DiscreteActionSequencesPlanner`` enumerates every action sequence up to
``depth``, estimates each one's return from ``n_return_samples`` simulations,
and returns the best sequence. It is open loop: the sequence is committed to in
advance, so no observation received along the way can change it.

Use it as a lower bound on what closed-loop planning buys you on a problem — if
a closed-loop planner cannot beat this, the problem's observations are not worth
acting on.

Notes
-----

- ``action()`` returns the whole sequence, not one action. This is the one
  planner where the returned list is longer than one element.
- The number of sequences is ``|A| ** depth``, so depth stays small.

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

   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import (
       DiscreteActionSequencesPlanner,
   )

   tiger = TigerPOMDP(discount_factor=0.95)

   planner = DiscreteActionSequencesPlanner(
       environment=tiger,
       discount_factor=0.95,
       name="TestOpenLoop",
       depth=2,
       n_return_samples=5,
   )

   belief = get_initial_belief(pomdp=tiger, n_particles=10, resampling=True)
   actions, run_data = planner.action(belief)

Parameters
----------

.. autoclass:: POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner.DiscreteActionSequencesPlanner
   :members:
   :show-inheritance:
   :no-index:
