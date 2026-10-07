ConstrainedZero
===============

BetaZero for chance-constrained POMDPs. Three things change: the network grows
a third head that predicts failure probability, action selection becomes SPUCT
and masks actions whose predicted failure exceeds a threshold, and that
threshold is calibrated at run time by conformal inference rather than fixed by
hand. Training targets are constrained in the same way, so the policy the
network learns respects the same limit the search does.

Notes
-----

- Original paper: Moss, R. J., Jamgochian, A., Fischer, J., Corso, A., &
  Kochenderfer, M. J. (2024). *ConstrainedZero: Chance-Constrained POMDP
  Planning Using Learned Probabilistic Failure Surrogates and Adaptive Safety
  Constraints*. IJCAI, 6752-6760. https://www.ijcai.org/proceedings/2024/746
- Requires a ``failure_fn``: the predicate that says whether a state counts as
  a failure. Everything else it inherits from :doc:`beta_zero`.
- The constraint is on *probability of failure*, not on an accumulated cost.
  For a cost budget, use :doc:`constrained_and_cvar`.

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
   * - Discrete observations
     - ✔️
   * - Continuous observations
     - ✔️
   * - Cost constraints
     - ✔️
   * - GPU
     - ✔️

Example
-------

Tiger defines no failure state, so the predicate below is a toy one: it
counts the ``"tiger_left"`` state as a failure, to show where ``failure_fn``
goes. ``delta_0`` is the failure-probability threshold the search starts from;
the planner then adjusts it at a rate set by ``eta``. As with
:doc:`beta_zero`, Tiger's string states give the default network all-zero
input, so this example shows the call, not a trained planner.

.. code-block:: python

   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero import (
       ConstrainedZero,
   )
   from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

   tiger = TigerPOMDP(discount_factor=0.95)


   def is_failure(state):
       return state == "tiger_left"


   planner = ConstrainedZero(
       environment=tiger,
       discount_factor=0.95,
       depth=5,
       name="ConstrainedZero_Example",
       action_sampler=DiscreteActionSampler(tiger.get_actions()),
       failure_fn=is_failure,
       delta_0=0.1,
       time_out_in_seconds=2.0,
       state_dim=1,
   )

   belief = get_initial_belief(tiger, n_particles=20)
   actions, run_data = planner.action(belief)

Parameters
----------

.. autoclass:: POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero.ConstrainedZero
   :members:
   :show-inheritance:
