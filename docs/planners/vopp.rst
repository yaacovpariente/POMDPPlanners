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
- Needs PyTorch. It runs on CPU, but its speed comes from stepping every
  particle in one batched tensor call, which pays off on a GPU with a model
  that steps batches of particles.
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

The example plans one step of Continuous Light-Dark through its vectorized
model. The belief is a tensor of state particles, all at ``(2, 2)`` here.

.. code-block:: python

   import torch
   from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
       ContinuousLightDarkPOMDP,
   )
   from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_vectorized_model import (
       ContinuousLightDarkVectorizedModel,
   )
   from POMDPPlanners.planners.vectorized_planners import VOPPPlanner

   torch.manual_seed(0)
   env = ContinuousLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=False)
   model = ContinuousLightDarkVectorizedModel(env, device=torch.device("cpu"))

   planner = VOPPPlanner(
       model,
       num_actions=model.num_actions,
       num_particles=256,
       max_depth=5,
       num_planning_iterations=8,
       discount_factor=0.95,
   )

   root_particles = torch.tensor([[2.0, 2.0]]).repeat(256, 1)  # [n, state_dim]
   action = planner.plan(root_particles)  # an index in [0, num_actions)

Parameters
----------

.. autoclass:: POMDPPlanners.planners.vectorized_planners.vopp.vopp.VOPPPlanner
   :members:
   :show-inheritance:
