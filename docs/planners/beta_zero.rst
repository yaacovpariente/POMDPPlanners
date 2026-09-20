BetaZero
========

AlphaZero moved into belief space. BetaZero is PFT-DPW with two of its parts
replaced by a network: action selection uses PUCT with a learned policy prior
instead of UCB1, and a leaf's value comes from a learned value head instead of a
random rollout. The network reads a fixed-size summary of the belief, so the
same weights serve every belief the search reaches.

Offline policy iteration — play episodes with the current network, train on the
search's visit counts and returns, repeat — is orchestrated by
:class:`~POMDPPlanners.training.PolicyTrainer`. An untrained planner still runs:
the network is created automatically and the priors start uninformative.

Notes
-----

- Original paper: Moss, R. J., Corso, A., Caers, J., & Kochenderfer, M. J.
  (2024). *BetaZero: Belief-State Planning for Long-Horizon POMDPs using
  Learned Approximations*. Reinforcement Learning Conference (RLC).
- Needs PyTorch. ``state_dim`` must match the environment's state vector, and
  the action sampler is a
  :class:`BetaZeroActionSampler <POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_action_sampler.BetaZeroActionSampler>`
  wrapping an ordinary sampler as fallback.
- The safety-constrained extension is :doc:`constrained_zero`.

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
     - ✔️

Example
-------

.. code-block:: python

   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero import BetaZero
   from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_action_sampler import (
       BetaZeroActionSampler,
   )
   from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

   tiger = TigerPOMDP(discount_factor=0.95)
   actions = tiger.get_actions()
   action_sampler = BetaZeroActionSampler(
       fallback_sampler=DiscreteActionSampler(actions),
       actions=actions,
   )

   planner = BetaZero(
       environment=tiger,
       discount_factor=0.95,
       depth=10,
       name="BetaZero_Example",
       action_sampler=action_sampler,
       n_simulations=50,
       state_dim=1,          # Tiger's state is one number
       k_a=1.0,
       alpha_a=0.5,
       k_o=1.0,
       alpha_o=0.5,
       exploration_constant=1.0,
   )

   belief = get_initial_belief(tiger, n_particles=50)
   actions, run_data = planner.action(belief)

Parameters
----------

.. autoclass:: POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero.BetaZero
   :members:
   :show-inheritance:
   :no-index:
