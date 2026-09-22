Tiger
=====

.. episode-viewer:: traces/tiger.json

   One real episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

The classic two-door POMDP: one door hides a tiger, the other hides a prize.
Listening is cheap but only 85 % accurate, so the whole problem is deciding how
much evidence to buy before committing.

Use it as the first check on any new planner. A planner that opens a door
immediately, or that listens forever, is broken in a way you can see in three
episodes.

What the agent sees and does
----------------------------

- **State** — one of the strings ``"tiger_left"``, ``"tiger_right"``.
- **Actions** (discrete) — ``"listen"``, ``"open_left"``, ``"open_right"``.
- **Observations** (discrete) — ``"hear_left"``, ``"hear_right"``,
  ``"hear_nothing"``. Listening returns the correct side with probability 0.85.
  Opening a door always returns ``"hear_nothing"``; conversely
  ``"hear_nothing"`` never follows a listen.

Formal definition
-----------------

The environment is the POMDP :math:`\langle S, A, \Omega, T, O, R, b_0, \gamma
\rangle`. Write :math:`\ell` for ``tiger_left`` and :math:`r` for
``tiger_right``.

**State space**

.. math::

   S = \{\ell,\; r\}

**Action space**

.. math::

   A = \{\textsf{listen},\; \textsf{open\_left},\; \textsf{open\_right}\}

**Observation space**

.. math::

   \Omega = \{\textsf{hear\_left},\; \textsf{hear\_right},\;
   \textsf{hear\_nothing}\}

**Transition model.** Listening leaves the tiger where it is; opening either
door resets it to a fresh coin flip.

.. math::

   T(s' \mid s, \textsf{listen}) &= \mathbb{1}[s' = s] \\
   T(s' \mid s, \textsf{open\_left}) = T(s' \mid s, \textsf{open\_right})
   &= \tfrac{1}{2}, \quad s' \in S

**Observation model.** Conditioned on the *successor* state, as everywhere in
this codebase.

.. math::

   O(\textsf{hear\_left} \mid \ell, \textsf{listen}) =
   O(\textsf{hear\_right} \mid r, \textsf{listen}) &= 0.85 \\
   O(\textsf{hear\_right} \mid \ell, \textsf{listen}) =
   O(\textsf{hear\_left} \mid r, \textsf{listen}) &= 0.15 \\
   O(\textsf{hear\_nothing} \mid s', a) &= 1, \quad a \neq \textsf{listen}

Note the two halves do not overlap: :math:`\textsf{hear\_nothing}` has
probability zero under :math:`\textsf{listen}`, and the directional
observations have probability zero under either open action.

**Reward function.** Paid on the state the action is taken *from*, so
:math:`R` does not depend on :math:`s'`.

.. math::

   R(s, \textsf{listen}) &= -1 \\
   R(s, \textsf{open\_left}) &= \begin{cases}
     -100 & s = \ell \\ +10 & s = r \end{cases} \\
   R(s, \textsf{open\_right}) &= \begin{cases}
     -100 & s = r \\ +10 & s = \ell \end{cases}

giving :math:`R \in [-100, 10]`.

**Initial belief**

.. math::

   b_0(\ell) = b_0(r) = \tfrac{1}{2}

with the initial observation fixed at :math:`\textsf{hear\_nothing}`.

**Discount.** :math:`\gamma \in [0, 1]` — the required ``discount_factor``
argument; the class has no default.

**Terminal set.** :math:`S_T = \emptyset`. No state ends an episode, so the
horizon is whatever the caller sets.

Rewards
-------

======================  =========
Event                   Reward
======================  =========
Listen                  -1.0
Open the correct door   +10.0
Open the wrong door     -100.0
======================  =========

``reward_range`` is ``(-100.0, 10.0)``. The asymmetry is the point: at a
discount factor near 1 the optimal policy listens several times before acting.

Key settings
------------

``discount_factor`` is the only constructor argument that changes the problem,
and it is **required** — there is no default. The 0.85 listening accuracy and
the three rewards are constants in the class, not arguments.

An episode never ends on its own: ``is_terminal`` always returns ``False``, and
opening a door resets the tiger to a fresh 50/50 draw. Episode length comes from
the horizon the caller sets.

Minimal example
---------------

.. code-block:: python

   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

   env = TigerPOMDP(discount_factor=0.95)

   state = env.initial_state_dist().sample(1)[0]
   # With n_samples=1 these return the value itself, not a list of one.
   observation = env.sample_observation(state, "listen")
   print(state, observation, env.reward(state, "listen"))

There is also a batched torch model,
``POMDPPlanners.environments.tiger_pomdp.tiger_pomdp_vectorized_model.TigerVectorizedModel``,
for the vectorized planners.

See also
--------

- :class:`POMDPPlanners.environments.tiger_pomdp.TigerPOMDP`
- :doc:`index` — the full catalog.
