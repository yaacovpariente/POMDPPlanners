Constrained and risk-sensitive variants
=======================================

Five planners here change *what* the search optimizes, not how it searches.
Two keep an expected-cost budget; three replace the mean in the value backup
with a tail measure.

**Cost-constrained (dual ascent).** ``CPFT_DPW`` and ``CPOMCPOW`` are
:doc:`pft_dpw` and :doc:`pomcpow` under a Lagrangian: the search maximizes
reward minus ``lambda`` times cost. After every simulation, dual ascent moves
``lambda`` by ``lambda_step`` times the greedy root action's estimated cost
minus ``cost_budget``, clipped at zero, so ``lambda`` grows while the cost is
over budget and shrinks toward zero while it is under.
The search machinery is untouched; only the value being backed up changes. The
cost comes from the environment, which must subclass
:class:`~POMDPPlanners.core.environment.ConstrainedEnvironment` and implement
``constraint_cost(state, action, next_state)``.

**Risk-sensitive (iterated CVaR).** ``ICVaR_PFT_DPW``, ``ICVaR_POMCPOW`` and
``ICVaRSparseSampling`` back up the Conditional Value at Risk of the child
values instead of their mean, at every node. The result is a policy judged on
the worst ``alpha`` fraction of outcomes, so it avoids branches whose *tail* is
bad even when their mean is high. ``alpha = 1`` recovers the risk-neutral
planner. These planners need no extra cost from the environment: the cost they
minimize is the negative reward.

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
- ``ICVaR_PFT_DPW`` and ``ICVaR_POMCPOW`` take ``min_immediate_cost`` and
  ``max_immediate_cost``: the bounds the tail estimate is normalized against.
  ``ICVaR_POMCPOW`` requires them; ``ICVaR_PFT_DPW`` defaults them to ``0``
  and ``1``, so set them to your environment's cost range.

Can I use?
----------

.. list-table::
   :header-rows: 1
   :widths: 20 12 12

   * -
     - Discrete
     - Continuous
   * - State
     - ✔️
     - ✔️
   * - Action
     - ✔️
     - ✔️
   * - Observation
     - ✔️
     - ✔️

.. list-table::
   :header-rows: 1
   :widths: 20 12

   * - Also supports
     -
   * - Cost constraints
     - ✔️
   * - GPU
     - ❌

``ICVaRSparseSampling`` is the exception: it takes discrete actions only and
does no widening.

Example
-------

The example adds a cost to Light-Dark: every step longer than 0.5 costs 1.
``CPFT_DPW`` then plans to keep the expected cost at or below ``cost_budget``.

.. code-block:: python

   import numpy as np
   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.core.environment import ConstrainedEnvironment
   from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
       ContinuousLightDarkPOMDP,
   )
   from POMDPPlanners.planners.mcts_planners.constrained_pft_dpw import CPFT_DPW
   from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler


   class LightDarkWithStepCost(ContinuousLightDarkPOMDP, ConstrainedEnvironment):
       """Light-Dark with a cost of 1 for every step longer than 0.5."""

       def constraint_cost(self, state, action, next_state):
           return np.array([float(np.linalg.norm(action) > 0.5)])


   np.random.seed(0)
   env = LightDarkWithStepCost(discount_factor=0.95)

   planner = CPFT_DPW(
       environment=env,
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
       time_out_in_seconds=2.0,
   )

   belief = get_initial_belief(env, n_particles=50)
   actions, run_data = planner.action(belief)

Parameters
----------

.. autoclass:: POMDPPlanners.planners.mcts_planners.constrained_pft_dpw.CPFT_DPW
   :members:
   :show-inheritance:

.. autoclass:: POMDPPlanners.planners.mcts_planners.constrained_pomcpow.CPOMCPOW
   :members:
   :show-inheritance:

.. autoclass:: POMDPPlanners.planners.mcts_planners.icvar_pft_dpw.ICVaR_PFT_DPW
   :members:
   :show-inheritance:

.. autoclass:: POMDPPlanners.planners.mcts_planners.icvar_pomcpow.ICVaR_POMCPOW
   :members:
   :show-inheritance:

.. autoclass:: POMDPPlanners.planners.sparse_sampling_planners.icvar_sparse_sampling.ICVaRSparseSampling
   :members:
   :show-inheritance:
