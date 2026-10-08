Belief States
=============

A planner never sees the true state. It sees a *belief*: a probability
distribution over states, built from the actions taken and the observations
received so far. In this package the default belief is a set of weighted
particles, and you rarely build one by hand: ``get_initial_belief`` makes the
first one and the episode runner updates it after every step.

This page shows how to create, inspect and update a belief. The class-by-class
reference is in :doc:`../common/beliefs`.

Creating the initial belief
---------------------------

``get_initial_belief`` draws ``n_particles`` states from the environment's
initial state distribution and gives them equal weight. It returns a
:class:`~POMDPPlanners.core.belief.particle_beliefs.WeightedParticleBelief` with resampling on.

.. code-block:: python

   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

   env = TigerPOMDP(discount_factor=0.95)
   belief = get_initial_belief(env, n_particles=500)

   state = belief.sample()  # one state, drawn by weight

The particles are drawn at random, so two calls give two different beliefs. If
you run a batch through the simulation API, seed NumPy before this call:
the belief is part of the cache key, and an unseeded belief makes every rerun
start from zero (see :doc:`simulations`).

Reading a belief
----------------

For a discrete state space, ``to_unique_support_distribution`` merges equal
particles and sums their weights:

.. code-block:: python

   dist = belief.to_unique_support_distribution()
   print(dict(zip(dist.values, dist.probs)))
   # e.g. {'tiger_left': 0.49, 'tiger_right': 0.51}

For a continuous state space, read the particles and their weights directly:
``belief.particles`` and ``belief.normalized_weights``.

Updating a belief
-----------------

``update`` applies one action and one observation and returns a new belief. It
does not change the old one. For a weighted particle belief, it moves every
particle through the transition model, then multiplies each weight by the
likelihood of the observation from that particle.

.. code-block:: python

   for observation in ["hear_left", "hear_left"]:
       belief = belief.update(action="listen", observation=observation, pomdp=env)

   dist = belief.to_unique_support_distribution()
   print(dict(zip(dist.values, dist.probs)))  # most weight now on tiger_left

The episode runner makes this call for you after each step. You only call it
yourself to inspect a belief, or in a custom episode of your own.

To simulate one step without an observation from the world,
``sample_next_belief`` draws a state from the belief, steps the environment,
and updates with the observation it sampled:

.. code-block:: python

   from POMDPPlanners.core.belief import sample_next_belief

   next_belief, observation = sample_next_belief(belief, action="listen", pomdp=env)

Resampling
----------

After a few updates, most of the weight sits on a few particles and the rest
carry almost none. The belief then acts as if it had far fewer particles. The
*effective sample size*, ``1 / sum(w**2)`` over the normalized weights,
measures this.

With resampling on, ``update`` resamples whenever the effective sample size
falls below ``ess_factor * n_particles``. It draws ``n_particles`` particles
from the current set by systematic resampling, so each particle is copied in
proportion to its weight, and gives every copy equal weight. ``ess_factor``
defaults to 0.5. Leave resampling on unless you are studying the filter
itself: without it the belief degrades into a few particles over a long
episode.

Other belief types
------------------

Particles work for any model the environment can sample from, which is why
they are the default. When the model is linear or close to it, a Gaussian
belief stores only a mean and a covariance. Its Kalman filter update is exact
for a linear model with Gaussian noise; the extended and unscented filters
approximate a nonlinear one:

- :class:`~POMDPPlanners.core.belief.gaussian_belief.GaussianBelief`, with a Kalman filter
  updater: ``LinearKalmanFilterUpdater``, ``ExtendedKalmanFilterUpdater`` or
  ``UnscentedKalmanFilterUpdater``.
- :class:`~POMDPPlanners.core.belief.gaussian_mixture_belief.GaussianMixtureBelief`, for a belief with
  several modes.

The ``belief_representations`` notebook in :doc:`../examples/index` builds each
of them and compares them on the same problem.

See also
--------

- :doc:`../common/beliefs` — the belief API reference.
- :doc:`simulations` — running episodes, where beliefs are updated for you.
