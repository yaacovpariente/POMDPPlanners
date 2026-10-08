Base Planner Class
==================

A planner takes a belief and returns the action to execute next. Every planner
in this package is an *online* planner: at each step it searches forward from
the current belief under a budget you set, returns an action, and throws the
search away. All of them except :doc:`vopp` subclass
:class:`~POMDPPlanners.core.policy.Policy`, so they share one constructor
shape, one ``action(belief)`` call, and one kind of run data.

Create a planner directly from its class, or by name through the registry:

.. code-block:: python

   from POMDPPlanners.planners import get_policy

   planner = get_policy("POMCP", environment=env, discount_factor=0.95, ...)

``get_policy`` passes the keyword arguments straight to the class. It knows
every planner on these pages except ``CPFT_DPW``, ``CPOMCPOW`` and
``VOPPPlanner``; import those three directly.

To write a planner of your own, see :doc:`../guide/custom_planners`.

Shared constructor arguments
----------------------------

Every ``Policy`` takes these arguments:

``environment``
   The environment to plan in. The planner samples transitions, observations
   and rewards from it during the search.

``discount_factor``
   The discount applied to future rewards, in ``(0, 1]``. The sparse sampling
   planners are the exception: they take no ``discount_factor`` and read
   ``environment.discount_factor`` instead.

``name``
   A label for the planner. It appears in logs and in saved file paths, so give
   each planner in a comparison its own name.

``log_path``, ``debug``, ``use_queue_logger``
   Logging options. They do not change the search.

Almost every planner also takes ``depth``, the number of steps the search
looks ahead.

The constructor checks the planner against the environment. A planner built
for discrete actions raises ``ValueError`` on an environment whose action space
is continuous or mixed, and the same check runs for observations. The error
comes at construction rather than mid-episode, so a wrong pairing never costs
you a run.

The ``action(belief)`` contract
-------------------------------

.. code-block:: python

   actions, run_data = planner.action(belief)

``action`` returns a pair:

``actions``
   A list. For every closed-loop planner it has one element, the action to
   execute now. :doc:`open_loop` is the exception: it returns the whole
   planned sequence.

``run_data``
   A :class:`~POMDPPlanners.core.policy.PolicyRunData`. Its ``info_variables``
   field is a list of
   :class:`~POMDPPlanners.core.policy.PolicyInfoVariable`, each a ``name`` and
   a numeric ``value`` measured during this decision, such as the root visit
   count of an MCTS tree. ``get_info_variable_names()`` on the class lists the
   names a planner reports. The simulation framework stores these per step, so
   you can analyse the search after the episode ends.

The belief classes are described in :doc:`../common/beliefs`. Every example on
these pages starts from a particle belief built by ``get_initial_belief``.

Budget arguments
----------------

The budget decides how much search goes into each decision. Most planners take
one of two budgets, and you must set exactly one of them:

``n_simulations``
   A fixed amount of search per decision. For the MCTS planners this is the
   number of simulations. Use it when you need the same search on every run, for
   example in tests.

``time_out_in_seconds``
   A wall-clock limit per decision. Use it to compare planners: two planners
   are only comparable if both got the same time per decision.

Setting both raises ``ValueError`` at construction. The other planners have
their own budget:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Planner
     - Budget
   * - :doc:`sparse_sampling`, ``ICVaRSparseSampling``
     - ``depth`` and ``branching_factor`` fix the tree. There is no time limit.
   * - :doc:`open_loop`
     - ``depth`` and ``n_return_samples`` per sequence.
   * - :doc:`vopp`
     - ``num_planning_iterations``, each running ``num_particles`` episodes
       in parallel.

How to set a budget so two planners get the same time is covered in
:doc:`../examples/planners_comparison`.

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
   Progressive widening: a node's children are capped at ``k * n**alpha``,
   where ``n`` is its visit count, so a branch grows only as the node is
   visited. Without it the branching factor is fixed before the search starts.

**Costs**
   The planner optimizes something other than the expected return. The
   constrained planners (``CPFT_DPW``, ``CPOMCPOW``) keep an expected-cost
   budget over a cost the environment reports. ``ConstrainedZero`` keeps a
   bound on the probability of failure. The ICVaR planners optimize a tail
   (CVaR) measure of the cost, where the cost is the negative reward.

**GPU**
   The planner has a CUDA or PyTorch path. All of them run on CPU; these are
   the ones where a GPU changes what budget is reachable.

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
   ├── sparse_sampling_planners/
   │   ├── :class:`~POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling.SparseSamplingDiscreteActionsPlanner`
   │   └── :class:`~POMDPPlanners.planners.sparse_sampling_planners.icvar_sparse_sampling.ICVaRSparseSampling`
   ├── vectorized_planners/
   │   └── vopp/
   │       └── :class:`~POMDPPlanners.planners.vectorized_planners.vopp.vopp.VOPPPlanner`
   └── open_loop_planners/
       └── :class:`~POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner.DiscreteActionSequencesPlanner`

The base classes below live in ``POMDPPlanners/core/policy.py`` and
``POMDPPlanners/planners/planners_utils/path_simulations_policy_arena.py``.

Parameters
----------

.. autoclass:: POMDPPlanners.core.policy.Policy
   :members:
   :show-inheritance:

.. autoclass:: POMDPPlanners.core.policy.TrainablePolicy
   :members:
   :show-inheritance:

.. autoclass:: POMDPPlanners.planners.planners_utils.path_simulations_policy_arena.ArenaPathSimulationPolicy
   :members:
   :show-inheritance:

.. autoclass:: POMDPPlanners.core.policy.PolicyRunData

.. autoclass:: POMDPPlanners.core.policy.PolicyInfoVariable
