Battleship
==========

.. episode-viewer:: traces/battleship.json

   One recorded episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

``BattleshipPOMDP`` searches for a hidden, fixed fleet on a square board.
Probe one cell at a time and hit every occupied cell to finish.

The sensor is exact and the fleet never moves, so all the uncertainty is in
the initial layout. Every probe removes the fleet layouts that disagree with
its reading, and nothing ever adds one back.

What the agent sees and does
----------------------------

- **State** — a ``float64`` vector of length ``2 * board_size ** 2``: fleet
  occupancy per cell, then a probed flag per cell. Occupancy is hidden from
  the agent.
- **Actions** (discrete) — an ``int``, ``row * board_size + column``, that
  probes one cell; coordinates start at zero.
- **Observations** (discrete) — exact: ``1`` for a hit and ``0`` for a miss.
  Probing changes only the record of visited cells, never the fleet.

Formal definition
-----------------

The environment is the POMDP :math:`\langle S, A, Z, T, O, R, b_0, \gamma
\rangle`. Let :math:`n` be ``board_size``, :math:`C = \{0, \dots, n^2 - 1\}`
the cells in row-major order, and :math:`\mathcal{L} \subseteq \{0,1\}^{C}` the
set of legal fleet layouts — the occupancy vectors reachable by placing every
ship in ``ship_lengths`` straight, within the board, without overlap (and
without touching when ``allow_adjacent_ships=False``).

**State space.** A layout paired with the set of cells probed so far:

.. math::

   S = \mathcal{L} \times \{0,1\}^{C}, \qquad s = (u, m)

where :math:`u_j = 1` means cell :math:`j` holds a ship and :math:`m_j = 1`
means it has been probed. The vector is stored flat as
:math:`[u_0 \dots u_{n^2-1},\, m_0 \dots m_{n^2-1}]`.

**Action space.** One probe per cell:

.. math::

   A = C, \qquad a = \text{row} \cdot n + \text{column}

**Observation space**

.. math::

   Z = \{\textsf{MISS}, \textsf{HIT}\} = \{0, 1\}

**Transition model.** Deterministic, and it never touches the fleet — probing
only records that a cell was visited:

.. math::

   T\big((u', m') \mid (u, m), a\big) =
   \mathbb{1}[u' = u] \cdot \mathbb{1}[m' = m + e_a]

where :math:`e_a` sets bit :math:`a`. Because :math:`u` is constant along a
trajectory, all uncertainty is in the initial draw; the agent only rules out layouts and
never tracks a moving target.

**Observation model.** Noiseless:

.. math::

   O(o \mid (u', m'), a) = \mathbb{1}[o = u'_a]

The sensor is exact, so every probe is a hard constraint that cuts
:math:`\mathcal{L}` down rather than reweighting it.

**Reward function.** Only a *new* hit pays:

.. math::

   R\big((u, m), a\big) = \begin{cases}
     +\texttt{hit\_reward} & u_a = 1 \text{ and } m_a = 0 \\
     -\texttt{miss\_penalty} & \text{otherwise}
   \end{cases}

so :math:`R \in [-\texttt{miss\_penalty},\, \texttt{hit\_reward}]`. Re-probing a
cell already known to hold a ship scores as water: exactly one branch fires per
step, and nothing stacks. Note :math:`R` reads the *pre*-probe mask, which is
what makes it a function of :math:`(s, a)` alone.

**Initial belief.** Uniform over legal layouts, nothing probed:

.. math::

   b_0\big((u, 0)\big) = \frac{1}{|\mathcal{L}|}, \qquad u \in \mathcal{L}

with the pre-probe observation fixed at :math:`\textsf{MISS}`, which carries no
information. ``max_layouts`` caps the enumeration of :math:`\mathcal{L}` used
by the exact belief.

**Discount.** :math:`\gamma` = ``discount_factor``, default :math:`0.99`.

**Terminal set.** Every occupied cell probed:

.. math::

   S_T = \{(u, m) : u_j \leq m_j \ \text{for all } j \in C\}

Hitting the runner's step limit first is a timeout, recorded separately from
completion.

Rewards
-------

A new hit earns ``hit_reward`` (default 1.0). Water and repeated probes of any
cell earn ``-miss_penalty`` (default -0.1), including another probe of a known
hit.

Key settings
------------

The default board is 5 by 5, with straight ships of lengths 3, 2 and 2. Ships
may touch, including diagonally, unless ``allow_adjacent_ships=False``.

Belief
~~~~~~

``BattleshipBelief`` tracks legal fleet layouts consistent with observed hits
and misses; its occupancy probabilities describe uncertainty about each cell.
These probabilities are not extra sensor readings.

``BattleshipVectorizedWeightedParticleBelief`` is the batched version, and it is
what ``create_environment_belief`` returns. It carries the same posterior --
its particles are redrawn from the consistent layouts on every probe, so the
two agree cell for cell -- through the vectorized updater interface, which is
what a vectorized planner needs to hold a belief at all.

Visualization
~~~~~~~~~~~~~

Runs write a trace of each episode through the environment's episode
visualizer. The results site replays it in 3D, as the replay on this page does.

Can I use?
----------

.. list-table::
   :header-rows: 1
   :widths: 34 30

   * - Capability
     - ``BattleshipPOMDP``
   * - Action space
     - Discrete
   * - Observation space
     - Discrete
   * - Native C++ backend
     - ❌
   * - Vectorized (torch) model
     - ❌
   * - In the ``get_environment`` registry
     - ❌
   * - Optional dependencies
     - None

Example
-------

.. code-block:: python

   from POMDPPlanners.environments.battleship_pomdp import BattleshipPOMDP

   env = BattleshipPOMDP(board_size=5, ship_lengths=(3, 2, 2))

Parameters
----------

.. autoclass:: POMDPPlanners.environments.battleship_pomdp.BattleshipPOMDP
   :members:
   :show-inheritance:

See also
--------

- :class:`POMDPPlanners.environments.battleship_pomdp.BattleshipPOMDP`
- :doc:`base` — the full catalog and the environment interface.
