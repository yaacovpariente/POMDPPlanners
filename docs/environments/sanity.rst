Sanity
======

Two states, two actions, perfect observations, no terminal state. Action ``0``
moves to the "good" state and pays ``1.0``; action ``1`` moves to "bad" and pays
``0.0``.

There is no partial observability and nothing to plan around. Its job is to fail
loudly: if a planner does not converge to always taking action ``0``, the bug is
in the planner, not in the problem. Run it first when a planner produces
nonsense somewhere else.

What the agent sees and does
----------------------------

- **State** — a plain ``int``: ``0`` (good) or ``1`` (bad).
- **Actions** (discrete) — the ints ``0`` (go to good) and ``1`` (go to bad).
- **Observations** (discrete) — an ``int`` equal to the new state, always.

Formal definition
-----------------

The environment is the POMDP :math:`\langle S, A, \Omega, T, O, R, b_0, \gamma
\rangle`. Everything is deterministic, and nothing depends on the current
state.

**State space**

.. math::

   S = \{0, 1\}

**Action space**

.. math::

   A = \{0, 1\}

**Observation space**

.. math::

   \Omega = \{0, 1\}

**Transition model**

.. math::

   T(s' \mid s, a) = \mathbb{1}[s' = a]

**Observation model**

.. math::

   O(o \mid s', a) = \mathbb{1}[o = s']

**Reward function**

.. math::

   R(s, a) = \mathbb{1}[a = 0]

with :math:`R \in [0, 1]`.

**Initial belief.** The start is fixed rather than drawn:

.. math::

   b_0(0) = 1, \qquad o_0 = 0

**Discount.** :math:`\gamma` defaults to :math:`0.95`.

**Terminal set.** :math:`S_T = \emptyset`.

Because :math:`O` is the identity, the belief collapses to a point mass after
one step — this is an MDP wearing a POMDP interface, which is exactly what
makes it useful as a control.

Rewards
-------

======================  =========
Event                   Reward
======================  =========
Action ``0``            1.0
Action ``1``            0.0
======================  =========

``reward_range`` is ``(0.0, 1.0)``.

Key settings
------------

``discount_factor`` (default ``0.95``) is the only constructor argument that
changes the problem. An episode never ends on its own: ``is_terminal`` always
returns ``False``, so episode length comes from the horizon the caller sets.

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
