VOPP
====

The Vectorized Online POMDP Planner: the whole search is batched tensor
operations over a flat belief tree, with no Python loop per simulation and no
host/device synchronization inside a planning step. Instead of interleaving
optimization with estimation, VOPP solves its value function analytically — a
log-sum-exp over action *preferences* — and leaves only the expectations to
Monte Carlo.

It does not take an ``Environment``. It takes a vectorized generative model and
the number of actions, and the episode loop is driven by
:class:`~POMDPPlanners.planners.vectorized_planners.vopp.vopp_episode_runner.VOPPEpisodeRunner`.

Notes
-----

- Original paper: Hoerger, M., Sudrajat, M., & Kurniawati, H. (2026).
  *Vectorized Online POMDP Planning*. arXiv:2510.27191.
  https://arxiv.org/abs/2510.27191
- Needs PyTorch. It runs on CPU, but the design only pays off on a GPU with a
  model that steps batches of particles.
- Not in the policy registry, and it is not a ``Policy`` subclass — its entry
  point is ``plan()``, not ``action()``.

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
     - ✔️

Example
-------

.. code-block:: python

   import torch
   from POMDPPlanners.planners.vectorized_planners import VOPPPlanner

   planner = VOPPPlanner(
       model,                       # a vectorized generative model
       num_actions=model.num_actions,
       num_particles=1024,
       max_depth=20,
       num_planning_iterations=100,
       discount_factor=0.95,
       temperature=1.0,
   )

   action = planner.plan(root_particles)   # [num_particles, state_dim] tensor

Parameters
----------

.. autoclass:: POMDPPlanners.planners.vectorized_planners.vopp.vopp.VOPPPlanner
   :members:
   :show-inheritance:
   :no-index:
