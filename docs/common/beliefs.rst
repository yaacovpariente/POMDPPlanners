Beliefs
=======

A belief is the agent's probability distribution over the hidden state. Every
planner takes a belief as input to ``action``, and every simulation step
updates it with the action taken and the observation received. This page
lists the belief classes, the updaters that some of them use, and the helpers
that build a first belief from an environment.

For when to pick which belief, and how planners use it, see the
:doc:`../core/beliefs` guide.

Which belief to use
-------------------

- :class:`~POMDPPlanners.core.belief.particle_beliefs.WeightedParticleBelief`
  is the default. It works for any environment that can sample states and
  score observations, and :func:`~POMDPPlanners.core.belief.belief_utils.get_initial_belief`
  returns one.
- :class:`~POMDPPlanners.core.belief.gaussian_belief.GaussianBelief` and
  :class:`~POMDPPlanners.core.belief.gaussian_mixture_belief.GaussianMixtureBelief`
  fit continuous states with near-Gaussian noise. They store a mean and a
  covariance per component and update them in one filter step, instead of
  moving and reweighting every particle, but they need an updater (a Kalman
  filter or similar) and
  do not work with POMCP or POMCP-DPW, which add particles one at a time.
- :class:`~POMDPPlanners.core.belief.vectorized_weighted_particle_belief.VectorizedWeightedParticleBelief`
  updates all particles through the updater's batched transition and
  observation-likelihood calls, not a Python loop per particle. Use it when an environment ships a
  vectorized updater and the particle update is your bottleneck.
- :class:`~POMDPPlanners.core.belief.batched_particle_belief.BatchedParticleBelief`
  holds many particle beliefs as torch tensors, for the GPU planners.
- The ``...StateUpdate`` classes grow one particle at a time. MCTS planners
  use them inside the search tree; you rarely build one yourself.

Example
-------

Build a particle belief for the Tiger problem, then update it twice with the
same observation:

.. code-block:: python

   import numpy as np
   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

   np.random.seed(0)
   env = TigerPOMDP(discount_factor=0.95)

   # 500 equally weighted particles drawn from the initial state distribution.
   belief = get_initial_belief(env, n_particles=500)
   print(belief.sample())

   # Listen twice and hear the tiger on the left both times.
   for _ in range(2):
       belief = belief.update(action="listen", observation="hear_left", pomdp=env)

   left = belief.normalized_weights[np.array(belief.particles) == "tiger_left"].sum()
   print(f"P(tiger_left) = {left:.2f}")  # about 0.97

A Gaussian belief takes its update rule as an object, so the same class
serves a linear Kalman filter, an EKF or a UKF:

.. code-block:: python

   import numpy as np
   from POMDPPlanners.core.belief import GaussianBelief, LinearKalmanFilterUpdater

   updater = LinearKalmanFilterUpdater(
       A=np.eye(2), B=np.zeros((2, 1)), H=np.eye(2),
       Q=0.1 * np.eye(2), R=0.5 * np.eye(2),
   )
   belief = GaussianBelief(mean=np.zeros(2), covariance=np.eye(2), updater=updater)

   # A Kalman update needs no environment, so ``pomdp`` can be None.
   belief = belief.update(action=np.zeros(1), observation=np.array([1.0, 1.0]), pomdp=None)
   print(belief.mean)  # [0.6875 0.6875]

Some environments register their own belief factory, which builds a belief
with a vectorized updater or a Gaussian instead of the generic particle filter.
``create_environment_belief`` calls that factory, and falls back to
``get_initial_belief`` for environments without one:

.. code-block:: python

   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief

   env = TigerPOMDP(discount_factor=0.95)
   belief = create_environment_belief(env, belief_type=BeliefType.PARTICLE, n_particles=200)

Creating a belief
-----------------

.. autofunction:: POMDPPlanners.core.belief.belief_utils.get_initial_belief

.. autofunction:: POMDPPlanners.utils.belief_factory.create_environment_belief

.. autoclass:: POMDPPlanners.utils.belief_factory.BeliefType

Base class
----------

Subclass :class:`~POMDPPlanners.core.belief.base_belief.Belief` to add a new
representation. You must implement ``update`` and ``sample``. Implement
``inplace_update`` too if the belief should work inside POMCP's tree.

.. autoclass:: POMDPPlanners.core.belief.base_belief.Belief
   :members: update, inplace_update, sample, config_id, from_config

Particle beliefs
----------------

.. autoclass:: POMDPPlanners.core.belief.particle_beliefs.WeightedParticleBelief
   :members: update, sample, to_dict, to_unique_support_distribution

.. autoclass:: POMDPPlanners.core.belief.particle_beliefs.UnweightedParticleBelief
   :members: update, sample, reinvigorate

.. autoclass:: POMDPPlanners.core.belief.particle_beliefs.WeightedParticleBeliefReinvigoration
   :members: update, reinvigorate
   :show-inheritance:

.. autoclass:: POMDPPlanners.core.belief.particle_beliefs.WeightedParticleBeliefStateUpdate
   :members: update, inplace_update, sample

.. autoclass:: POMDPPlanners.core.belief.particle_beliefs.UnweightedParticleBeliefStateUpdate
   :members: update, inplace_update, sample

Vectorized and batched beliefs
------------------------------

.. autoclass:: POMDPPlanners.core.belief.vectorized_weighted_particle_belief.VectorizedWeightedParticleBelief
   :members: update, sample

.. autoclass:: POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater
   :members:

.. autoclass:: POMDPPlanners.core.belief.batched_particle_belief.BatchedParticleBelief
   :members: from_root, update, propagate, reweight, resample, sample_states, effective_sample_size

Gaussian beliefs and updaters
-----------------------------

.. autoclass:: POMDPPlanners.core.belief.gaussian_belief.GaussianBelief
   :members: update, sample, entropy, dim

.. autoclass:: POMDPPlanners.core.belief.gaussian_belief_updaters.GaussianBeliefUpdater
   :members: update

.. autoclass:: POMDPPlanners.core.belief.gaussian_belief_updaters.LinearKalmanFilterUpdater

.. autoclass:: POMDPPlanners.core.belief.gaussian_belief_updaters.ExtendedKalmanFilterUpdater

.. autoclass:: POMDPPlanners.core.belief.gaussian_belief_updaters.UnscentedKalmanFilterUpdater

.. autoclass:: POMDPPlanners.core.belief.gaussian_mixture_belief.GaussianMixtureBelief
   :members: update, sample, dim, n_components

.. autoclass:: POMDPPlanners.core.belief.gaussian_mixture_belief.GaussianMixtureBeliefUpdater
   :members: update

Helpers
-------

.. autofunction:: POMDPPlanners.core.belief.belief_utils.sample_next_belief

.. autofunction:: POMDPPlanners.core.belief.belief_utils.is_terminal_belief

.. autofunction:: POMDPPlanners.core.belief.particle_beliefs.get_unique_support
