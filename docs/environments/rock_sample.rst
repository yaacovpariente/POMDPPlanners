RockSample
==========

.. image:: ../../POMDPPlanners/tests/test_environments/golden_visualizations/rock_sample_visualization.gif
   :alt: A robot crossing the RockSample grid, checking and sampling rocks.
   :width: 480px

A robot on a grid must sample the good rocks and skip the bad ones, then leave
by walking east off the right-hand edge. Whether a rock is good is hidden; a
long-range sensor answers, but its accuracy decays with distance as
``(1 + 2 ** (-distance / sensor_efficiency)) / 2`` — the accuracy of Smith &
Simmons, "Heuristic Search Value Iteration for POMDPs" (2004). It is ``1.0`` at
the rock and falls towards ``0.5``, never below, so a check from far away tells
you nothing rather than telling you the opposite of the truth.

This is the standard long-horizon information-gathering benchmark. Getting a
good score means walking *towards* a rock to make its reading trustworthy, which
costs steps — the trade-off that separates planners that reason about future
observations from ones that do not.

What the agent sees and does
----------------------------

- **State** — a ``float32`` vector ``[robot_row, robot_col, rock_0, …,
  rock_{R-1}]``, each rock ``1.0`` good or ``0.0`` bad. Build one with
  ``create_rock_sample_state``; read it with ``get_robot_pos`` and
  ``get_rocks``. With ``is_dangerous_area_hit_terminal=True`` a terminal-flag
  slot is appended.
- **Actions** (discrete) — integers: ``0`` sample, ``1`` north, ``2`` east,
  ``3`` south, ``4`` west, and ``5 … 4+R`` to check rock *i*. So the action set
  grows with the number of rocks; readable names are in ``env.action_names``.
- **Observations** (discrete) — ``"none"``, ``"good"`` or ``"bad"``.

Formal definition
-----------------

Fix a grid of :math:`M \times N` cells and rock positions :math:`\rho_1, \dots,
\rho_R`. Write a state as :math:`s = (x, c)` with position :math:`x \in
\mathbb{Z}^2` and rock qualities :math:`c \in \{0, 1\}^R` (:math:`1` good).

**State space.** The grid, the rock flags, and one absorbing exit state
:math:`\top` stored as the sentinel position :math:`(-1, -1)`:

.. math::

   S = \big(\{0, \dots, M-1\} \times \{0, \dots, N-1\} \times \{0,1\}^R\big)
   \;\cup\; \{\top\}

so :math:`|S| = MN2^R + 1`. With ``is_dangerous_area_hit_terminal=True`` a
further binary slot :math:`h` is appended and :math:`S_T` grows accordingly.

**Action space.** Five moves plus one check per rock:

.. math::

   A = \underbrace{\{0\}}_{\textsf{sample}} \cup
   \underbrace{\{1,2,3,4\}}_{\textsf{N,E,S,W}} \cup
   \underbrace{\{5, \dots, 4+R\}}_{\textsf{check rock } i}

**Observation space**

.. math::

   \Omega = \{\textsf{none},\; \textsf{good},\; \textsf{bad}\}

**Transition model.** Deterministic. Moves clamp at the north, south and west
walls; moving east off the last column exits:

.. math::

   T(s' \mid s, a) = \mathbb{1}[s' = f(s, a)]

with :math:`f(\top, a) = \top` and, for :math:`s = (x, c)` where
:math:`x = (r, k)`,

.. math::

   f(s, 1) &= \big((\max(r-1, 0),\, k),\; c\big) \\
   f(s, 3) &= \big((\min(r+1, M-1),\, k),\; c\big) \\
   f(s, 4) &= \big((r,\, \max(k-1, 0)),\; c\big) \\
   f(s, 2) &= \begin{cases}
     \top & k + 1 \geq N \\
     \big((r,\, k+1),\; c\big) & \text{otherwise}
   \end{cases} \\
   f(s, 0) &= \big(x,\; c \text{ with } c_i \leftarrow 0
     \text{ if } \rho_i = x\big) \\
   f(s, a) &= s \qquad a \geq 5

Sampling a rock consumes it: a good rock becomes bad, so sampling twice pays
the penalty the second time. Check actions never move the robot.

**Observation model.** Only a check returns information. For
:math:`a = 5 + i`, let :math:`d = \lVert x' - \rho_i \rVert_2` be the Euclidean
distance from the robot to rock :math:`i` and let :math:`\eta` be
``sensor_efficiency``. The accuracy is the Smith & Simmons (2004) law

.. math::

   \alpha(d) = \tfrac{1}{2}\big(1 + 2^{-d/\eta}\big)

which is :math:`1` at the rock, :math:`0.75` at :math:`d = \eta`, and decays to
:math:`\tfrac{1}{2}` — never below it, so a distant check is uninformative
rather than misleading. Then

.. math::

   O(\textsf{good} \mid s', 5+i) &= \begin{cases}
     \alpha(d) & c'_i = 1 \\ 1 - \alpha(d) & c'_i = 0 \end{cases} \\
   O(\textsf{bad} \mid s', 5+i) &= 1 - O(\textsf{good} \mid s', 5+i) \\
   O(\textsf{none} \mid s', a) &= 1 \qquad a < 5

**Reward function.** Terms are **added**, evaluated on the pre-transition
state :math:`s = (x, c)` except the hazard term, which uses the realised
:math:`x'`. Write :math:`\sigma` for ``step_penalty``, :math:`g` for
``good_rock_reward``, :math:`\beta` for ``bad_rock_penalty``, :math:`\varsigma`
for ``sensor_use_penalty`` and :math:`e` for ``exit_reward``:

.. math::

   R(s, a, s') = \sigma + \begin{cases}
     e & a = 2,\; k = N-1 \\
     g\,c_i + \beta(1 - c_i) & a = 0,\; x = \rho_i \\
     \varsigma & a \geq 5 \\
     0 & \text{otherwise}
   \end{cases} \;+\; D(x')

The exit term short-circuits: an exiting step pays :math:`\sigma + e` and no
hazard term. The hazard term :math:`D` is where the three
``reward_model_type`` variants differ. Let :math:`\delta(x')` be the distance
from :math:`x'` to the nearest hazard centre, :math:`\varrho` the
``dangerous_area_radius``, :math:`P` the ``dangerous_area_penalty`` and
:math:`q` the ``dangerous_area_hit_probability``:

.. math::

   D_{\text{constant}}(x') &= P \cdot \mathbb{1}[\delta(x') \leq \varrho]
     \cdot \mathrm{Bern}(q) \\
   D_{\text{decayed}}(x') &= P \cdot
     \mathrm{Bern}\big(e^{-\delta(x') / \lambda}\big) \\
   D_{\text{shock}}(x') &= \pm |P| \text{ with probability } \tfrac{1}{2}
     \text{ each}, \quad \delta(x') \leq \varrho

where :math:`\lambda` is ``penalty_decay``. The decayed variant has no radius
cutoff — it draws on every step. The shock variant has zero mean and exists to
separate risk-sensitive planners from risk-neutral ones, which cannot tell it
from :math:`D \equiv 0`.

.. warning::

   Under the constant and decayed variants with :math:`q < 1`, :math:`R` draws
   a fresh Bernoulli per call, so ``reward(s, a)`` is **not** a deterministic
   function of its arguments. Pass the realised ``next_state`` to keep the
   reward on the same outcome as the trajectory. Setting
   ``is_dangerous_area_hit_terminal=True`` removes the problem by moving the
   draw into the transition: the hazard slot :math:`h` is set there, and
   :math:`D` becomes deterministic given :math:`h`.

**Initial belief.** Position known, every rock configuration equally likely:

.. math::

   b_0\big((x_0, c)\big) = 2^{-R}, \qquad c \in \{0,1\}^R

with :math:`x_0` = ``init_pos`` and the initial observation fixed at
:math:`\textsf{none}`. This is the point of the benchmark: :math:`2^R` hidden
configurations, and the only way to narrow them is to spend steps walking
closer to the rocks.

**Discount.** :math:`\gamma` = ``discount_factor``, default :math:`0.95`.

**Terminal set.** :math:`S_T = \{\top\}`, plus :math:`\{s : h = 1\}` when the
hazard is terminal.

Rewards
-------

Terms are **added**, so the hazard penalty is passed as a negative number:

=========================================  ==================================
Event                                      Reward
=========================================  ==================================
Every step                                 ``step_penalty`` (default ``0.0``)
Sample a good rock                         ``good_rock_reward`` (``+10.0``)
Sample a bad rock                          ``bad_rock_penalty`` (``-10.0``)
Any check action                           ``sensor_use_penalty`` (``0.0``)
Move east off the right edge               ``exit_reward`` (``+10.0``)
Enter a dangerous area                     ``dangerous_area_penalty`` (``-5.0``)
=========================================  ==================================

Key settings
------------

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Argument
     - Default
     - What it changes
   * - ``map_size``
     - ``(5, 5)``
     - Grid size.
   * - ``rock_positions``
     - ``[(0,0), (2,2), (3,3)]``
     - Number and placement of rocks. This sets the action-space size.
   * - ``sensor_efficiency``
     - ``10.0``
     - Distance in cells at which a check is 75% accurate. Larger means the
       sensor stays accurate further away, so the problem gets easier.
   * - ``dangerous_areas``
     - ``None``
     - Hazard cells. Adds a safety dimension on top of the information problem.
   * - ``is_dangerous_area_hit_terminal``
     - ``False``
     - Whether entering a hazard ends the episode.
   * - ``reward_model_type``
     - ``CONSTANT_HAZARD_PENALTY``
     - Also ``DISTANCE_DECAYED_HAZARD_PENALTY`` and
       ``ZERO_MEAN_HAZARD_SHOCK``. The last one cannot be combined with
       ``is_dangerous_area_hit_terminal=True``.
   * - ``discount_factor``
     - ``0.95``
     -

An episode ends when the robot exits east (its position becomes the sentinel
``(-1, -1)``), or on a hazard hit if that was made terminal.

Minimal example
---------------

.. code-block:: python

   from POMDPPlanners.environments.rock_sample_pomdp import RockSamplePOMDP

   env = RockSamplePOMDP(map_size=(5, 5), rock_positions=[(0, 0), (2, 2), (3, 3)])

   state = env.initial_state_dist().sample(1)[0]
   check_first_rock = 5
   observation = env.sample_observation(state, check_first_rock)
   print(env.action_names[check_first_rock], observation)

See also
--------

- :class:`POMDPPlanners.environments.rock_sample_pomdp.RockSamplePOMDP`
- Batched torch model:
  ``POMDPPlanners.environments.rock_sample_pomdp.rocksample_vectorized_model.RockSampleVectorizedModel``
- :doc:`index` — the full catalog.
