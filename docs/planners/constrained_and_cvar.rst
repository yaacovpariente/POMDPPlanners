Constrained and risk-sensitive variants
=======================================

Five planners here change *what* the search optimizes, not how it searches.
Two keep an expected-cost budget; three replace the mean in the value backup
with a tail measure. All of them need the environment to report a cost
alongside its reward.

Cost-constrained (dual ascent)
------------------------------

``CPFT_DPW`` and ``CPOMCPOW`` are :doc:`pft_dpw` and :doc:`pomcpow` under a
Lagrangian: the search maximizes reward minus ``lambda`` times cost, and
``lambda`` is raised or lowered by dual ascent between simulations until the
expected cost sits at ``cost_budget``. The search machinery is untouched; only
the value being backed up changes.

Risk-sensitive (iterated CVaR)
------------------------------

``ICVaR_PFT_DPW``, ``ICVaR_POMCPOW`` and ``ICVaRSparseSampling`` back up the
Conditional Value at Risk of the child values instead of their mean, at every
node. The result is a policy judged on the worst ``alpha`` fraction of
outcomes, so it avoids branches whose *tail* is bad even when their average is
fine. ``alpha = 1`` recovers the risk-neutral planner.

Notes
-----

- Constrained planners: Jamgochian, A., Corso, A., & Kochenderfer, M. J.
  (2023). *Online Planning for Constrained POMDPs with Continuous Spaces
  through Dual Ascent*. ICAPS 33, 198-202.
  https://doi.org/10.1609/icaps.v33i1.27195
- CVaR planners: Pariente, Y., & Indelman, V. (2026). *Online Risk-Averse
  Planning in POMDPs Using Iterated CVaR Value Function*. arXiv:2601.20554.
  https://arxiv.org/abs/2601.20554
- ``CPFT_DPW`` and ``CPOMCPOW`` are not in the policy registry — import them
  directly.
- The CVaR planners need ``min_immediate_cost`` and ``max_immediate_cost``:
  the bounds the tail estimate is normalized against.

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
     - ❌

Example
-------

.. code-block:: python

   from POMDPPlanners.planners.mcts_planners.constrained_pft_dpw import CPFT_DPW
   from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler

   planner = CPFT_DPW(
       environment=env,                 # an environment that reports costs
       discount_factor=0.95,
       depth=5,
       name="example_cpft_dpw",
       action_sampler=UnitCircleActionSampler(max_action_magnitude=1.0),
       cost_budget=0.5,
       lambda_init=0.0,
       lambda_step=0.1,
       k_a=1.0,
       alpha_a=0.5,
       k_o=1.0,
       alpha_o=0.5,
       exploration_constant=1.0,
       n_simulations=50,
   )

Parameters
----------

.. autoclass:: POMDPPlanners.planners.mcts_planners.constrained_pft_dpw.CPFT_DPW
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: POMDPPlanners.planners.mcts_planners.constrained_pomcpow.CPOMCPOW
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: POMDPPlanners.planners.mcts_planners.icvar_pft_dpw.ICVaR_PFT_DPW
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: POMDPPlanners.planners.mcts_planners.icvar_pomcpow.ICVaR_POMCPOW
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: POMDPPlanners.planners.sparse_sampling_planners.icvar_sparse_sampling.ICVaRSparseSampling
   :members:
   :show-inheritance:
   :no-index:
