Battleship
==========

``BattleshipPOMDP`` searches for a hidden, fixed fleet on a square board.
Probe one cell at a time and hit every occupied cell to finish. The default
board is 5 by 5, with straight ships of lengths 3, 2 and 2. Ships may touch,
including diagonally, unless ``allow_adjacent_ships=False``.

.. code-block:: python

   from POMDPPlanners.environments.battleship_pomdp import BattleshipPOMDP

   env = BattleshipPOMDP(board_size=5, ship_lengths=(3, 2, 2))

Actions, observations and rewards
---------------------------------

Action ``row * board_size + column`` probes a cell; coordinates start at zero.
The observation is exact: ``1`` for a hit and ``0`` for a miss. Probing changes
only the record of visited cells, never the fleet.

A new hit earns ``hit_reward`` (default 1.0). Water and repeated probes of any
cell earn ``-miss_penalty`` (default -0.1), including another probe of a known
hit. The episode ends when all ship cells have been hit. Reaching the runner's
step limit first is a timeout, recorded separately from completion.

The state contains fleet occupancy and probe flags. Occupancy is hidden from
the agent. ``BattleshipBelief`` tracks legal fleet layouts consistent with
observed hits and misses; its occupancy probabilities describe uncertainty
about each cell. These probabilities are not extra sensor readings.

``BattleshipVectorizedWeightedParticleBelief`` is the batched twin, and it is
what ``create_environment_belief`` returns. It carries the same posterior --
its particles are redrawn from the consistent layouts on every probe, so the
two agree cell for cell -- through the vectorized updater interface, which is
what a vectorized planner needs to hold a belief at all.

Formal definition
-----------------

Let :math:`n` be ``board_size``, :math:`C = \{0, \dots, n^2 - 1\}` the cells in
row-major order, and :math:`\mathcal{L} \subseteq \{0,1\}^{C}` the set of legal
fleet layouts — the occupancy vectors reachable by placing every ship in
``ship_lengths`` straight, within the board, without overlap (and without
touching when ``allow_adjacent_ships=False``).

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

   \Omega = \{\textsf{MISS}, \textsf{HIT}\} = \{0, 1\}

**Transition model.** Deterministic, and it never touches the fleet — probing
only records that a cell was visited:

.. math::

   T\big((u', m') \mid (u, m), a\big) =
   \mathbb{1}[u' = u] \cdot \mathbb{1}[m' = m + e_a]

where :math:`e_a` sets bit :math:`a`. Because :math:`u` is constant along a
trajectory, all uncertainty is in the initial draw; the agent is doing pure
hypothesis elimination, never tracking a moving target.

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

Episode replay
--------------

.. episode-viewer:: traces/battleship.json

   One real episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

Runs also write a GIF of each episode through ``cache_visualization``. Its left
board shows prior probes: crosses mark hits, dots mark misses, and pale cells
are unprobed. The amber ring marks the current action. The center board shows
recorded belief probabilities on a fixed 0–100% scale. Gray cells with dashes
mean belief data is unavailable. The right board shows the hidden fleet for
human review; the policy does not receive this view. Each ship shape marks one
occupied cell, without assigning ship identities.

Boards show the state before the displayed action. The caption reports that
action's observation and reward separately. The final record has no action
ring and says “Final recorded state”; that label alone does not mean the fleet
was sunk. Frames last 1.4 seconds, with 2.4 seconds for the final record.

The approved :download:`review GIF <../artifacts/battleship_redesign/review.gif>`
of that renderer replays 11 records from the seed-7 renderer fixture, using
real transitions and belief updates. The
:download:`contact sheet <../artifacts/battleship_redesign/contact-sheet.png>`
shows six decoded frames. See the
:download:`asset provenance <../artifacts/battleship_redesign/README.md>`
for source and hashes.
