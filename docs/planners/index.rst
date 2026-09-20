Planners
========

A planner takes a belief and returns the action to execute next. Every planner
here is an *online* planner: on each step it searches forward from the current
belief under a budget you set, acts, and throws the search away.

Every planner is constructed directly, or by name through the registry:

.. code-block:: python

   from POMDPPlanners.planners import get_policy

   planner = get_policy("POMCP", environment=env, discount_factor=0.95, ...)

.. note::

   ``get_policy`` covers the planners listed below except the constrained MCTS
   variants (``CPFT_DPW``, ``CPOMCPOW``) and ``VOPPPlanner`` — import those
   classes directly.

Which planner supports what
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 11 11 11 11 11 8

   * - Planner
     - Discrete
     - Continuous
     - Action PW
     - Obs. PW
     - Costs
     - GPU
   * - :doc:`POMCP <pomcp>`
     - ✔️
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * - :doc:`POMCP_DPW <pomcp_dpw>`
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ❌
     - ❌
   * - :doc:`POMCPOW <pomcpow>`
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ❌
     - ❌
   * - :doc:`PFT_DPW <pft_dpw>`
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ❌
     - ❌
   * - :doc:`SparsePFT <sparse_pft>`
     - ✔️
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * - :doc:`BetaZero <beta_zero>`
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ❌
     - ✔️
   * - :doc:`ConstrainedZero <constrained_zero>`
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✔️
   * - :doc:`CPFT_DPW <constrained_and_cvar>`
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ❌
   * - :doc:`CPOMCPOW <constrained_and_cvar>`
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ❌
   * - :doc:`ICVaR_PFT_DPW <constrained_and_cvar>`
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ❌
   * - :doc:`ICVaR_POMCPOW <constrained_and_cvar>`
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ❌
   * - :doc:`DESPOT <despot>`
     - ✔️
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * - :doc:`ARDESPOT <ardespot>`
     - ✔️
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * - :doc:`AdaOPS <adaops/index>`
     - ✔️
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * - :doc:`HypDESPOT <hyp_despot/index>`
     - ✔️
     - ❌
     - ❌
     - ❌
     - ❌
     - ✔️
   * - :doc:`SparseSamplingDiscreteActionsPlanner <sparse_sampling>`
     - ✔️
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * - :doc:`ICVaRSparseSampling <constrained_and_cvar>`
     - ✔️
     - ❌
     - ❌
     - ❌
     - ✔️
     - ❌
   * - :doc:`VOPPPlanner <vopp>`
     - ✔️
     - ❌
     - ❌
     - ❌
     - ❌
     - ✔️
   * - :doc:`DiscreteActionSequencesPlanner <open_loop>`
     - ✔️
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌

What the columns mean
---------------------

**Discrete** / **Continuous**
   The action spaces the planner accepts. A discrete-only planner takes a
   :class:`DiscreteActionsEnvironment
   <POMDPPlanners.core.environment.DiscreteActionsEnvironment>` and enumerates
   its actions. The others take an :class:`ActionSampler
   <POMDPPlanners.planners.planners_utils.dpw.ActionSampler>` and work in both.

**Action PW** / **Obs. PW**
   Progressive widening: the node's children are capped at ``k * n**alpha`` in
   its visit count ``n``, so the branch grows only as the node is visited.
   Without it the branching factor is fixed before the search starts.

**Costs**
   The planner reads a cost signal as well as a reward, and keeps a budget,
   a chance constraint, or a tail measure over it.

**GPU**
   The planner has a CUDA or PyTorch path. All of them run on CPU; these are
   the ones where a GPU changes what budget is reachable.

.. note::

   Comparing two planners only means something if both got the same wall-clock
   budget per decision. See :doc:`../examples/planners_comparison`.

Planner guides
--------------

.. toctree::
   :maxdepth: 1

   pomcp
   pomcp_dpw
   pomcpow
   pft_dpw
   sparse_pft
   beta_zero
   constrained_zero
   constrained_and_cvar
   despot
   ardespot
   adaops/index
   hyp_despot/index
   sparse_sampling
   vopp
   open_loop

The planner interface
---------------------

All planners implement the base ``Policy`` interface: ``action(belief)``
returns a list of actions — length one for closed-loop planning — and the run
data recorded for that decision.

.. autoclass:: POMDPPlanners.core.policy.Policy
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Writing your own planner
------------------------

Inherit from ``Policy`` and implement ``action``:

.. code-block:: python

   from POMDPPlanners.core.policy import Policy
   from POMDPPlanners.core.simulation import SimulationRunData

   class MyCustomPlanner(Policy):
       def __init__(self, environment, **kwargs):
           super().__init__(environment, **kwargs)

       def action(self, belief_state):
           action = self.select_action(belief_state)
           run_data = SimulationRunData()
           return [action], run_data

       def select_action(self, belief_state):
           ...

Where the code lives
--------------------

.. parsed-literal::

   planners/
   ├── mcts_planners/
   │   ├── :class:`~POMDPPlanners.planners.mcts_planners.pomcp.POMCP`
   │   ├── :class:`~POMDPPlanners.planners.mcts_planners.pomcp_dpw.POMCP_DPW`
   │   ├── :class:`~POMDPPlanners.planners.mcts_planners.pomcpow.POMCPOW`
   │   ├── :class:`~POMDPPlanners.planners.mcts_planners.pft_dpw.PFT_DPW`
   │   ├── :class:`~POMDPPlanners.planners.mcts_planners.sparse_pft.SparsePFT`
   │   ├── :class:`~POMDPPlanners.planners.mcts_planners.constrained_pft_dpw.CPFT_DPW`
   │   ├── :class:`~POMDPPlanners.planners.mcts_planners.constrained_pomcpow.CPOMCPOW`
   │   ├── :class:`~POMDPPlanners.planners.mcts_planners.icvar_pft_dpw.ICVaR_PFT_DPW`
   │   ├── :class:`~POMDPPlanners.planners.mcts_planners.icvar_pomcpow.ICVaR_POMCPOW`
   │   ├── beta_zero/
   │   │   └── :class:`~POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero.BetaZero`
   │   └── constrained_zero/
   │       └── :class:`~POMDPPlanners.planners.mcts_planners.constrained_zero.constrained_zero.ConstrainedZero`
   ├── scenario_tree_planners/
   │   ├── :class:`~POMDPPlanners.planners.scenario_tree_planners.despot.DESPOT`
   │   ├── :class:`~POMDPPlanners.planners.scenario_tree_planners.ardespot.ARDESPOT`
   │   ├── adaops/
   │   │   └── :class:`~POMDPPlanners.planners.scenario_tree_planners.adaops.adaops.AdaOPS`
   │   └── hyp_despot/
   │       └── :class:`~POMDPPlanners.planners.scenario_tree_planners.hyp_despot.hyp_despot.HypDESPOT`
   ├── sparse_sampling_planners/
   │   ├── :class:`~POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling.SparseSamplingDiscreteActionsPlanner`
   │   └── :class:`~POMDPPlanners.planners.sparse_sampling_planners.icvar_sparse_sampling.ICVaRSparseSampling`
   ├── vectorized_planners/
   │   └── vopp/
   │       └── :class:`~POMDPPlanners.planners.vectorized_planners.vopp.vopp.VOPPPlanner`
   └── open_loop_planners/
       └── :class:`~POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner.DiscreteActionSequencesPlanner`

See also
--------

- :doc:`../core/beliefs` — the belief representations these planners consume.
- :doc:`../environments/index` — the environments they run on.
- :doc:`../examples/planners_comparison` — running several of them against
  each other.
- :doc:`../api/POMDPPlanners.planners` — the complete API reference.
