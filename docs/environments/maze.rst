Maze
====

**DiscreteMazePOMDP** — a generated maze, one-cell moves.

.. episode-viewer:: traces/discrete_maze.json

   One recorded episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

**ContinuousMazePOMDP** — the same maze, with bounded displacements.

.. episode-viewer:: traces/continuous_maze.json

   One recorded episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

**TMazePOMDP** — a T-shaped corridor.

.. episode-viewer:: traces/t_maze.json

   One recorded episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

These three environments test memory of a noisy, single-use cue. The map is
known and movement is deterministic; only the rewarding goal side is hidden.
Entering either goal ends the episode, but only the correct goal counts as
success. The cue names that side with probability ``cue_accuracy``; later
observations are ``empty``.

What the agent sees and does
----------------------------

- **State** — a ``float64`` array ``[x, y, goal, cue_phase]``: the position,
  the goal side (``0.0`` left, ``1.0`` right) and the cue phase (``0.0``
  unseen, ``1.0`` emitting, ``2.0`` consumed). ``x`` and ``y`` are integer
  cells in the discrete variants and real numbers in the continuous one.
- **Actions** — discrete ``"up"``, ``"down"``, ``"left"``, ``"right"`` for
  ``DiscreteMazePOMDP`` and ``TMazePOMDP``; continuous ``[dx, dy]``, at most
  ``max_step_size`` long, for ``ContinuousMazePOMDP``.
- **Observations** (discrete) — ``"left_cue"``, ``"right_cue"``, ``"empty"``.
  Only the step that crosses the cue cell can return a cue.

Formal definition
-----------------

The environment is the POMDP :math:`\langle S, A, Z, T, O, R, b_0, \gamma
\rangle`. All three variants share one model and differ only in :math:`A` and
how a move is resolved.

**State space.** Let :math:`G` be the walkable cells. Position, the hidden
goal side, and the cue's delivery phase:

.. math::

   s = (x,\; y,\; \mathrm{side},\; \mathrm{phase})

.. math::

   S = \mathcal{P} \times \{\textsf{L}, \textsf{R}\} \times
   \{\textsf{UNSEEN}, \textsf{EMITTING}, \textsf{CONSUMED}\}

with :math:`(x, y)` the agent's position, :math:`\mathcal{P} = G` for the
discrete variants and :math:`\mathcal{P} \subseteq \mathbb{R}^2` the walkable
region for the continuous one (a point belongs to the cell it rounds to);
:math:`\mathrm{side}` the goal that pays, left (:math:`\textsf{L}`) or right
(:math:`\textsf{R}`); and :math:`\mathrm{phase}` whether the cue is not yet
seen (:math:`\textsf{UNSEEN}`), being read this step
(:math:`\textsf{EMITTING}`), or used up (:math:`\textsf{CONSUMED}`).

**Action space.**

.. math::

   A = \begin{cases}
     \{\textsf{up}, \textsf{down}, \textsf{left}, \textsf{right}\}
       & \texttt{DiscreteMazePOMDP},\ \texttt{TMazePOMDP} \\
     \{d \in \mathbb{R}^2 : \lVert d \rVert_2 \leq \texttt{max\_step\_size}\}
       & \texttt{ContinuousMazePOMDP}
   \end{cases}

A discrete action moves one cell: up :math:`y + 1`, down :math:`y - 1`,
left :math:`x - 1`, right :math:`x + 1`. A continuous action is the
displacement :math:`d` itself; a longer one is rescaled to the cap rather
than rejected.

**Observation space**

.. math::

   Z = \{\textsf{left\_cue},\; \textsf{right\_cue},\; \textsf{empty}\}

**Transition model.** Deterministic, and a goal is absorbing:

.. math::

   T(s' \mid s, a) = \mathbb{1}[s' = f(s, a)], \qquad
   f(s, a) = s \ \text{ for } s \in S_T

The position update refuses illegal moves without moving the agent, with
:math:`\Delta_{\textsf{up}} = (0, 1)`, :math:`\Delta_{\textsf{down}} = (0, -1)`,
:math:`\Delta_{\textsf{left}} = (-1, 0)`, :math:`\Delta_{\textsf{right}} =
(1, 0)` for the discrete variants and :math:`\Delta_d = d` for the continuous
one:

.. math::

   (x', y') = \begin{cases}
     (x, y) + \Delta_a & \text{the move is legal} \\
     (x, y) & \text{otherwise (wall collision)}
   \end{cases}

A discrete step is legal when the target cell is walkable. A continuous step
is legal only when the **whole swept segment** stays inside the walkable
region. The goal side never changes. With :math:`c \in G` the cue cell: The cue phase advances on every action, including one a
wall refused:

.. math::

   \mathrm{phase}' = \begin{cases}
     \textsf{EMITTING} & \mathrm{phase} = \textsf{UNSEEN}
       \text{ and the step crosses } c \\
     \textsf{CONSUMED} & \mathrm{phase} = \textsf{EMITTING} \\
     \mathrm{phase} & \text{otherwise}
   \end{cases}

So the cue is read once: it is consumed by whatever action follows it.

**Observation model**

.. math::

   O(\textsf{left\_cue} \mid s', \cdot) &= \begin{cases}
     p_{\text{cue}} & \mathrm{phase}' = \textsf{EMITTING},\ \mathrm{side} = \textsf{L} \\
     1 - p_{\text{cue}} & \mathrm{phase}' = \textsf{EMITTING},\ \mathrm{side} = \textsf{R} \\
     0 & \mathrm{phase}' \neq \textsf{EMITTING}
   \end{cases} \\
   O(\textsf{empty} \mid s', \cdot) &= \mathbb{1}[\mathrm{phase}' \neq \textsf{EMITTING}]

with :math:`p_{\text{cue}}` = ``cue_accuracy`` :math:`\in [0.5, 1]`, and
:math:`\textsf{right\_cue}` mirrored. The action does not enter.

**Reward function.** The terminal payout **replaces** the step cost rather
than stacking with it. :math:`g_L, g_R \in G` are the left and right goal
cells:

.. math::

   R(s, a) = \begin{cases}
     0 & s \in S_T \\
     +\texttt{goal\_reward} & \text{the step enters } g_{\mathrm{side}} \\
     -\texttt{wrong\_goal\_penalty} & \text{the step enters the other goal} \\
     -\texttt{step\_penalty} & \text{otherwise, wall collisions included}
   \end{cases}

**Initial belief.** Position and cue phase known, goal side a coin flip:

.. math::

   b_0\big((x_0, y_0, \textsf{L}, \textsf{UNSEEN})\big) =
   b_0\big((x_0, y_0, \textsf{R}, \textsf{UNSEEN})\big) = \tfrac{1}{2}

where :math:`(x_0, y_0)` is the fixed start cell. The opening observation is
fixed at :math:`\textsf{empty}`.

**Discount.** :math:`\gamma` = ``discount_factor``, default :math:`0.95`.

**Terminal set.** Either goal, correct or not:

.. math::

   S_T = \{s : (x, y) \in \{g_L, g_R\}\}

with :math:`g_L, g_R` the left and right goal cells.

Rewards
-------

======================================  =========
Event                                   Reward
======================================  =========
Step (including a wall collision)       -1.0
Enter the correct goal                  +10.0
Enter the wrong goal                    -10.0
======================================  =========

These are the defaults of ``step_penalty``, ``goal_reward`` and
``wrong_goal_penalty``, the same in all three variants. ``reward_range`` is
built from them, ``(-10.0, 10.0)`` by default.

Key settings
------------

All three classes take ``discount_factor`` (default ``0.95``), ``cue_accuracy``
(default ``0.9``), ``goal_reward`` (default ``10.0``), ``wrong_goal_penalty``
(default ``10.0``) and ``step_penalty`` (default ``1.0``). Every argument has a
default, so each class builds with no arguments. The two penalties are passed
as positive numbers and subtracted.

The layout arguments differ by class. ``DiscreteMazePOMDP`` and
``ContinuousMazePOMDP`` take ``maze_width`` (``7``), ``maze_height`` (``9``),
``maze_seed`` (``0``) and ``loop_fraction`` (``0.15``).
``ContinuousMazePOMDP`` also takes ``max_step_size`` (``1.0``). ``TMazePOMDP``
takes ``stem_length`` (``4``) and ``arm_length`` (``1``).

DiscreteMazePOMDP
~~~~~~~~~~~~~~~~~

``DiscreteMazePOMDP`` uses integer cell positions in a generated maze. The
four actions, ``up``, ``down``, ``left`` and ``right``, move one cell; a wall
blocks the move. ``maze_width``, ``maze_height``, ``maze_seed`` and
``loop_fraction`` control the layout.

ContinuousMazePOMDP
~~~~~~~~~~~~~~~~~~~

``ContinuousMazePOMDP`` uses real-valued positions and displacement actions
``[dx, dy]``, capped in length by ``max_step_size``. Collision checks cover
the whole movement path. With the same layout settings as DiscreteMazePOMDP,
it uses the same maze geometry.

TMazePOMDP
~~~~~~~~~~

``TMazePOMDP`` uses a T-shaped corridor and the same four one-cell actions as
the discrete maze. ``stem_length`` sets the distance to the junction and
``arm_length`` sets the distance from the junction to each endpoint. The
agent must remember the cue while walking up the stem, then choose an arm.

Belief
~~~~~~

``create_environment_belief`` returns a
``MazeVectorizedWeightedParticleBelief`` for either maze. The hidden state is
one bit -- which goal pays -- so the update is cheap per particle and the Python
loop around it is the whole cost; ``DiscreteMazeVectorizedUpdater`` and
``ContinuousMazeVectorizedUpdater`` do that loop's work in NumPy instead.

Both reproduce the environment's event rule rather than approximating it: the
discrete one by the same lookup table the environment builds, the continuous one
by the segment test written as array algebra. A belief that walked through walls
the world refuses would be searching a different maze. The continuous updater's
positions agree with the environment's to within its cell tolerance rather than
bit for bit, because it stops a step at the tolerance-widened cell boundary the
same code uses to decide membership.

Can I use?
----------

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Capability
     - ``DiscreteMazePOMDP``
     - ``ContinuousMazePOMDP``
     - ``TMazePOMDP``
   * - Action space
     - Discrete
     - Continuous
     - Discrete
   * - Observation space
     - Discrete
     - Discrete
     - Discrete
   * - Native C++ backend
     - ❌
     - ❌
     - ❌
   * - Vectorized (torch) model
     - ❌
     - ❌
     - ❌
   * - In the ``get_environment`` registry
     - ✔️
     - ✔️
     - ✔️
   * - Optional dependencies
     - None
     - None
     - None

Example
-------

.. code-block:: python

   import numpy as np

   from POMDPPlanners.environments.maze_pomdp import ContinuousMazePOMDP, DiscreteMazePOMDP

   env = DiscreteMazePOMDP(discount_factor=0.95, maze_seed=0)

   state = env.initial_state_dist().sample(1)[0]
   # With n_samples=1 these return the value itself, not a list of one.
   next_state = env.sample_next_state(state, "up")
   observation = env.sample_observation(next_state, "up")
   print(state, "->", next_state, observation, env.reward(state, "up"))

   # The continuous maze takes a displacement [dx, dy] instead.
   continuous_env = ContinuousMazePOMDP(discount_factor=0.95, max_step_size=1.0)
   state = continuous_env.initial_state_dist().sample(1)[0]
   print(continuous_env.sample_next_state(state, np.array([0.0, 0.5])))

Parameters
----------

.. autoclass:: POMDPPlanners.environments.maze_pomdp.DiscreteMazePOMDP
   :members:
   :show-inheritance:

.. autoclass:: POMDPPlanners.environments.maze_pomdp.ContinuousMazePOMDP
   :members:
   :show-inheritance:

.. autoclass:: POMDPPlanners.environments.maze_pomdp.TMazePOMDP
   :members:
   :show-inheritance:

See also
--------

- :class:`POMDPPlanners.environments.maze_pomdp.DiscreteMazePOMDP`
- :class:`POMDPPlanners.environments.maze_pomdp.ContinuousMazePOMDP`
- :class:`POMDPPlanners.environments.maze_pomdp.TMazePOMDP`
- :doc:`base` — the full catalog and the environment interface.
