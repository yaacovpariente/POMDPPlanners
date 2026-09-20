Sanity
======

Two states, two actions, perfect observations, no terminal state. Action ``0``
moves to the "good" state and pays ``1.0``; action ``1`` moves to "bad" and pays
``0.0``. ``reward_range`` is ``(0.0, 1.0)``.

There is no partial observability and nothing to plan around. Its job is to fail
loudly: if a planner does not converge to always taking action ``0``, the bug is
in the planner, not in the problem. Run it first when a planner produces
nonsense somewhere else.

Formal definition
-----------------

.. math::

   S = \Omega = \{0, 1\}, \qquad A = \{0, 1\}

Everything is deterministic, and nothing depends on the current state:

.. math::

   T(s' \mid s, a) &= \mathbb{1}[s' = a] \\
   O(o \mid s', a) &= \mathbb{1}[o = s'] \\
   R(s, a) &= \mathbb{1}[a = 0]

with :math:`R \in [0, 1]`. The start is fixed rather than drawn:

.. math::

   b_0(0) = 1, \qquad o_0 = 0

:math:`\gamma` defaults to :math:`0.95`, and :math:`S_T = \emptyset`.

Because :math:`O` is the identity, the belief collapses to a point mass after
one step — this is an MDP wearing a POMDP interface, which is exactly what
makes it useful as a control.

Minimal example
---------------

.. code-block:: python

   from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP

   env = SanityPOMDP(discount_factor=0.95)
   state = env.initial_state_dist().sample(1)[0]
   print(env.reward(state, 0), env.reward(state, 1))

See also
--------

- :class:`POMDPPlanners.environments.sanity_pomdp.SanityPOMDP`
- :doc:`mountain_car` — the other small debugging environment.
- :doc:`index` — the full catalog.
