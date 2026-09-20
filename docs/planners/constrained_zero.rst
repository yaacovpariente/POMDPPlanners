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
   * - Action widening
     - ✔️
   * - Observation widening
     - ✔️
   * - Cost constraints
     - ✔️
   * - GPU
     - ✔️

Parameters
----------

.. autoclass:: POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero.ConstrainedZero
   :members:
   :show-inheritance:
   :no-index:
