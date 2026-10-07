Tiger
=====

.. episode-viewer:: traces/tiger.json

   One recorded episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

The two-door POMDP: one door hides a tiger, the other hides a prize.
Listening costs 1 but is only 85 % accurate, so the whole problem is deciding how
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

The environment is the POMDP :math:`\langle S, A, Z, T, O, R, b_0, \gamma
\rangle`.

**State space.** Two doors, with the tiger behind one: :math:`\ell` =
``tiger_left`` (behind the left door), :math:`r` = ``tiger_right`` (behind
the right door).

.. math::

   S = \{\ell,\; r\}

**Action space**

.. math::

   A = \{\textsf{listen},\; \textsf{open\_left},\; \textsf{open\_right}\}

:math:`\textsf{listen}` hears which side the tiger is on, with noise;
:math:`\textsf{open\_left}` and :math:`\textsf{open\_right}` open that door.

**Observation space**

.. math::

   Z = \{\textsf{hear\_left},\; \textsf{hear\_right},\;
   \textsf{hear\_nothing}\}

:math:`\textsf{hear\_left}` and :math:`\textsf{hear\_right}` say which side
the tiger was heard on; :math:`\textsf{hear\_nothing}` follows opening a door.

**Transition model.** Listening leaves the tiger where it is; opening either
door places it behind the left or right door with probability 1/2 each,
independent of where it was.

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
opening a door places the tiger behind either door with probability 1/2,
independent of where it was. Episode length comes from
the horizon the caller sets.

Can I use?
----------

.. list-table::
   :header-rows: 1
   :widths: 20 12 12

   * - ``TigerPOMDP``
     - Discrete
     - Continuous
   * - State
     - ✔️
     - ❌
   * - Action
     - ✔️
     - ❌
   * - Observation
     - ✔️
     - ❌

.. list-table::
   :header-rows: 1
   :widths: 34 30

   * - Also supports
     -
   * - Native C++ backend
     - ❌
   * - Vectorized (torch) model
     - ✔️ ``TigerVectorizedModel``
   * - In the ``get_environment`` registry
     - ✔️
   * - Optional dependencies
     - None

Example
-------

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

Parameters
----------

.. autoclass:: POMDPPlanners.environments.tiger_pomdp.TigerPOMDP
   :members:
   :show-inheritance:

See also
--------

- :class:`POMDPPlanners.environments.tiger_pomdp.TigerPOMDP`
- :doc:`base` — the full catalog and the environment interface.
