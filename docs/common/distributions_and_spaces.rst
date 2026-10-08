Distributions and Spaces
========================

Two classes in ``POMDPPlanners.core`` tie environments and planners
together. A ``Distribution`` is what an environment returns when it is asked
for its initial state, so beliefs and simulators can draw from it without
knowing the environment. ``SpaceInfo`` says whether an environment's actions
and observations are discrete or continuous, so a planner that only handles
discrete actions refuses a continuous environment at construction time
rather than failing mid-run.

The ``Environment`` base class that uses both is documented on
:doc:`../environments/base`.

Example
-------

.. code-block:: python

   import numpy as np
   from POMDPPlanners.core.distributions import DiscreteDistribution
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

   dist = DiscreteDistribution(["tiger_left", "tiger_right"], np.array([0.8, 0.2]))
   print(dist.sample(n_samples=3))          # a list of 3 values
   print(dist.probability(["tiger_left"]))  # [0.8]

   env = TigerPOMDP(discount_factor=0.95)
   print(env.space_info)                    # discrete actions, discrete observations
   print(env.initial_state_dist().sample(n_samples=2))

Writing a distribution
----------------------

An environment whose initial state is not a short list of values, such as a
random map layout, subclasses ``Distribution`` and implements ``sample``.
``sample`` always returns a list, even for one sample, because
``get_initial_belief`` asks for all particles in one call. Implement
``probability`` only if something needs the density; the default raises
``NotImplementedError``.

Distributions
-------------

.. autoclass:: POMDPPlanners.core.distributions.Distribution
   :members: sample, probability

.. autoclass:: POMDPPlanners.core.distributions.DiscreteDistribution
   :members: sample, probability

.. autoclass:: POMDPPlanners.core.distributions.Numpy2DDistribution
   :members: sample, probability

Space types
-----------

An environment declares its ``SpaceInfo``; a planner declares the
``PolicySpaceInfo`` it supports through ``get_space_info``. A planner built
for discrete actions raises an error when given an environment with
continuous or mixed actions, and the same holds for observations.

.. autoclass:: POMDPPlanners.core.environment.environment.SpaceType

.. autoclass:: POMDPPlanners.core.environment.environment.SpaceInfo

.. autoclass:: POMDPPlanners.core.policy.PolicySpaceInfo
