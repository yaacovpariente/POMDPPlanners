PacMan
======

.. image:: ../../POMDPPlanners/tests/test_environments/golden_visualizations/pacman_visualization.gif
   :alt: PacMan collecting pellets in a small maze while a ghost pursues.
   :width: 480px

Clear every pellet in a walled maze while ghosts hunt you. PacMan's own position
is known; the ghosts' positions are only observed through noise that grows with
distance, so the agent has to reason about where a ghost probably is rather than
where it was last seen.

It is the package's largest discrete state space, which makes it a useful stress
test for tree-search planners: the belief is over ghost positions, and a
planner that collapses it to a point estimate walks into ghosts.

What the agent sees and does
----------------------------

- **State** — a flat ``float64`` vector
  ``[pac_row, pac_col, ghost positions…, pellet_mask…, score, terminal]``.
  Construct one with ``env.make_state(...)`` and read it with
  ``env.get_pacman_pos``, ``env.get_ghost_positions``, ``env.get_pellets``,
  ``env.get_score``, ``env.get_terminal``. These are methods on the
  environment, not module-level functions.
- **Actions** (discrete) — ``0`` north, ``1`` east, ``2`` south, ``3`` west,
  ``4`` stay.
- **Observations** (discrete) — a tuple of one noisy ``(row, col)`` per ghost.
  Convert to and from a flat array with ``env.observation_to_array`` and
  ``env.array_to_observation``.

Formal definition
-----------------

Let the maze be :math:`M \times N` with wall set :math:`\mathcal{W}`, free
cells :math:`G`, :math:`P` initial pellet cells :math:`\pi_1, \dots, \pi_P`,
and :math:`g` = ``num_ghosts``.

**State space.**

.. math::

   s = \big(x,\; (y_k)_{k=1}^{g},\; \mathbf{m},\; \varsigma,\; \top\big),
   \qquad
   S = G \times G^{g} \times \{0,1\}^{P} \times \mathbb{R} \times \{0,1\}

with :math:`x` PacMan's cell, :math:`y_k` ghost :math:`k`'s cell,
:math:`\mathbf{m}` the pellet mask (:math:`m_p = 1` still there),
:math:`\varsigma` the running score and :math:`\top` an absorbing terminal
flag. The state space is the package's largest discrete one:
:math:`|G|^{g+1} 2^{P}`.

**Action space**

.. math::

   A = \{\textsf{N}, \textsf{E}, \textsf{S}, \textsf{W}, \textsf{stay}\}
     = \{0,1,2,3,4\}

**Transition model.** :math:`\top` is absorbing. Otherwise one step resolves
in this order.

*PacMan moves,* deterministically; a move into a wall or off the grid keeps
the cell:

.. math::

   x' = \begin{cases}
     x + \Delta_a & x + \Delta_a \in G \\ x & \text{otherwise}
   \end{cases}

*Ghosts move,* each from its own stochastic policy, evaluated against
PacMan's **pre-move** cell :math:`x`. Let :math:`\mathcal{M}(y_k)` be the
valid moves from :math:`y_k`. An *aggressive* ghost is a Boltzmann pursuer
over Manhattan distance, with temperature :math:`\beta` =
``ghost_aggressiveness``:

.. math::

   \Pr[y'_k = u] = \frac{\exp\big(-\mathrm{d}(u, x)/\beta\big)}
   {\sum_{v \in \mathcal{M}(y_k)} \exp\big(-\mathrm{d}(v, x)/\beta\big)},
   \qquad u \in \mathcal{M}(y_k)

Small :math:`\beta` makes a near-greedy chaser; large :math:`\beta` tends to
a uniform random walk. A *patrol* ghost continues in its stored direction
while that move is valid, else rotates clockwise and picks uniformly from
:math:`\mathcal{M}(y_k)`. An *ambush* ghost is deterministic: it minimizes a
score that penalizes leaving the ring :math:`\mathrm{d} \in [2, 4]` around
PacMan by :math:`+10`, so it loiters at intercept range instead of closing.
``ghost_coordination`` selects whether ghosts also condition on each other.

*Collision.* The episode ends if PacMan and any ghost share a cell **or swap
cells** — both arcs of the standard rule, since without the swap arc a ghost
would walk through PacMan:

.. math::

   \top' = 1 \quad\text{if}\quad \exists k:\;
   y'_k = x' \;\vee\; \big(y_k = x' \wedge y'_k = x\big)

*Pellets.* Landing on an active pellet clears it and adds ``pellet_reward``
to :math:`\varsigma`. If no pellet remains, :math:`\top' = 1`.

*Hazard.* When ``is_dangerous_area_hit_terminal``, a final draw terminates
the episode if :math:`x'` lies in a hazard zone — taken last, and only when
the step has not already ended, so the terminal flag stays absorbing.

**Observation model.** PacMan's own cell is known and never reported; the
observation is one noisy cell per ghost:

.. math::

   \Omega = \big(\{0..M{-}1\} \times \{0..N{-}1\}\big)^{g}

The noise grows with the distance to the ghost and then saturates:

.. math::

   \sigma_k = \mathrm{clip}\big(\texttt{observation\_noise\_factor}
   \cdot \mathrm{d}(y'_k, x'),\;
   10^{-6},\; \texttt{max\_observation\_noise}\big)

Each coordinate is drawn, rounded and clamped to the grid independently:

.. math::

   \hat{y}_k = \mathrm{clip}\big(\mathrm{round}(y'_k + \varepsilon_k),\;
   0,\; (M{-}1, N{-}1)\big), \qquad
   \varepsilon_k \sim \mathcal{N}(0, \sigma_k^2 I)

so the likelihood of a reading is the Gaussian mass of its rounding bin, with
the two end bins absorbing the tails. A terminal state reports
:math:`(-1, -1)` for every ghost. A nearby ghost is seen almost exactly; a
distant one is a blur — which is what makes a point estimate of ghost
positions a losing policy.

**Reward function.** Terminal states pay :math:`0`. Otherwise, evaluated
against the realised transition:

.. math::

   R(s, a, s') = \;&\texttt{step\_penalty}
   \;+\; \texttt{pellet\_reward} \cdot \mathbb{1}[\text{pellet eaten}] \\
   &+\; \texttt{ghost\_collision\_penalty} \cdot
     \mathbb{1}[\text{collision}] \\
   &+\; \texttt{win\_reward} \cdot \mathbb{1}[\mathbf{m}' = \mathbf{0}]
   \;-\; D(x')

with the hazard term :math:`D` following the same three
``reward_model_type`` variants as :doc:`rock_sample`.

.. note::

   ``reward_batch`` called without ``next_states`` returns only the
   deterministic terms — step penalty, pellet, win — and **omits the
   collision penalty**, because that depends on the stochastic ghost draw.
   Pass the realised successors to get numbers that agree with the scalar
   path.

**Initial belief.** Fully known, a single state:

.. math::

   b_0 = \delta_{s_0}, \qquad
   s_0 = \big(x_0,\, (y_k^0),\, \mathbf{1},\, 0,\, 0\big)

Uncertainty does not come from the prior here — it accumulates from the
observation noise as the ghosts move. The initial observation distribution is
a live draw from :math:`O` at :math:`s_0`, not a point mass, so the opening
belief already carries the sensor's blur.

**Discount.** :math:`\gamma` = ``discount_factor``, default :math:`0.95`.

**Terminal set.** :math:`S_T = \{s : \top = 1\}` — reached by a collision, by
clearing the last pellet, or by a terminal hazard hit.

Rewards
-------

============================  ==============================================
Event                         Reward
============================  ==============================================
Every non-terminal step       ``step_penalty`` (default ``-1.0``)
Eat a pellet                  ``pellet_reward`` (``+10.0``)
Share a cell with a ghost     ``ghost_collision_penalty`` (``-100.0``)
Eat the last pellet           ``win_reward`` (``+100.0``)
Inside a dangerous area       ``-dangerous_area_penalty``
============================  ==============================================

.. warning::

   ``dangerous_area_penalty`` is a **positive magnitude that gets subtracted**
   here, the opposite of the convention in RockSample and Push, where the
   penalty is added and so must be passed negative.

Key settings
------------

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Argument
     - Default
     - What it changes
   * - ``maze_size``
     - ``(7, 7)``
     - Grid size.
   * - ``walls``
     - 5 fixed cells
     - Maze layout.
   * - ``num_ghosts``
     - ``1``
     - Each ghost adds two dimensions to every observation.
   * - ``ghost_strategies``
     - ``["aggressive"]``
     - Per ghost: ``"aggressive"``, ``"patrol"`` or ``"ambush"``.
   * - ``ghost_coordination``
     - ``"independent"``
     - Also ``"coordinated"`` or ``"mixed"``.
   * - ``observation_noise_factor``
     - ``0.3``
     - How fast ghost-position noise grows with distance, capped by
       ``max_observation_noise`` (``1.5``).
   * - ``discount_factor``
     - ``0.95``
     -

``create_simple_maze_pacman(maze_size=7, num_walls=5, num_ghosts=1, seed=None)``
builds a randomized maze if you want variation across episodes.

An episode ends on a ghost collision, on clearing the last pellet, or on a
hazard hit when ``is_dangerous_area_hit_terminal=True``.

Minimal example
---------------

.. code-block:: python

   from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP

   env = PacManPOMDP(maze_size=(7, 7), num_ghosts=1)

   state = env.initial_state_dist().sample(1)[0]
   east = 1
   observation = env.sample_observation(state, east)
   print(env.get_pacman_pos(state), "believes ghosts near", observation)

See also
--------

- :class:`POMDPPlanners.environments.pacman_pomdp.PacManPOMDP`
- Batched torch model:
  ``POMDPPlanners.environments.pacman_pomdp.pacman_vectorized_model.PacManVectorizedModel``
- :doc:`index` — the full catalog.
