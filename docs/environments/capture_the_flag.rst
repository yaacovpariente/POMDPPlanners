Capture the Flag
================

.. episode-viewer:: traces/capture_the_flag.json

   One recorded episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

``CaptureTheFlagPOMDP`` puts two teams on a grid field split by a midline.
Each team has a flag. The planner drives the blue team as one joint
controller; the red team belongs to the transition model and follows a
stochastic role policy. Blue wins by carrying the red flag to the blue base
while its own flag is still home.

Blue never sees the red players, and does not know which of ``K`` candidate
cells holds the red flag. Both are inferred from one observation vector, which
gives strong evidence about where the enemy is and weak evidence about where
the flag is.

What the agent sees and does
----------------------------

- **State** — a ``float64`` vector of length ``4 * n_blue + 4 * n_red + 5``
  (21 at the default two-a-side): blue cells, red cells, the red flag's
  candidate index, a carrier index per flag, a respawn-freeze and a
  tagger-cooldown counter per player, and the two scores.
- **Actions** (discrete) — an ``int`` in ``0 .. 6 ** n_blue - 1`` (36 at the
  default two-a-side): one joint action for the whole blue team, encoded in
  base 6. Each player has six actions — four moves (``0`` north, ``1`` east,
  ``2`` south, ``3`` west), ``4`` hold, and ``5`` scan. Scan does not move the
  player; it widens that player's flag detector for the step and costs more.
- **Observations** (continuous) — a tuple of ``4 * n_blue + n_blue * n_red +
  4`` numbers (16 at the default): blue's own cells, a noisy range to each red
  player per blue player, a flag-detector bit per blue player, who carries the
  red flag, whether blue's flag is taken, blue's freeze counters, and both
  scores. Blue sees its own team exactly -- positions, freeze counters, who is
  carrying -- plus the score and whether its own flag has been taken. It does
  not see the red players, the red flag cell, or which red player took its
  flag. A terminal state emits all ``-1``.

Formal definition
-----------------

The environment is the POMDP :math:`\langle S, A, Z, T, O, R, b_0, \gamma
\rangle`.

**State space.** The notation:

- **Field.** The field is a grid of :math:`W \times H` cells
  (``grid_size``). A cell is written :math:`(x, y)` with :math:`x \in
  \{0..W-1\}` the column and :math:`y \in \{0..H-1\}` the row.
- **Trees.** Some cells hold trees (``trees``), which no player can enter.
  :math:`G` is the set of cells inside the field and not a tree.
- **Flag candidates.** The red flag's home is one of :math:`K` candidate
  cells :math:`F = (f_1, \dots, f_K)` (``red_flag_candidates``).
- **Players.** There are :math:`n_{\text{blue}}` blue players, numbered
  :math:`1..n_{\text{blue}}`, and :math:`n_{\text{red}}` red players,
  numbered :math:`1..n_{\text{red}}`.

A state is one vector:

.. math::

   s = \big(&\underbrace{x^{\text{blue}}_{1:n_{\text{blue}}}}_{\text{blue cells}},\;
            \underbrace{x^{\text{red}}_{1:n_{\text{red}}}}_{\text{red cells}},\;
            k,\;
            c^{\text{red}}, c^{\text{blue}},\; \\
            &\text{freeze}^{\text{blue}}_{1:n_{\text{blue}}}, \text{freeze}^{\text{red}}_{1:n_{\text{red}}},\;
            \text{cool}^{\text{blue}}_{1:n_{\text{blue}}}, \text{cool}^{\text{red}}_{1:n_{\text{red}}},\;
            \text{score}^{\text{blue}}, \text{score}^{\text{red}} \big)

.. math::

   S = G^{n_{\text{blue}}} \times G^{n_{\text{red}}} \times \{1..K\} \times
   \{0..n_{\text{blue}}\} \times \{0..n_{\text{red}}\} \times
   \mathbb{Z}_{\geq 0}^{2(n_{\text{blue}} + n_{\text{red}})} \times \mathbb{Z}_{\geq 0}^2

with these components:

- :math:`x^{\text{blue}}_i \in G` — the cell of blue player :math:`i`;
  :math:`x^{\text{red}}_j \in G` — the cell of red player :math:`j`.
- :math:`k \in \{1..K\}` — which candidate :math:`f_k` is the red flag's
  home.
- :math:`c^{\text{red}} \in \{0..n_{\text{blue}}\}` — the blue player
  carrying the red flag, :math:`0` if the red flag is home;
  :math:`c^{\text{blue}} \in \{0..n_{\text{red}}\}` — the red player carrying
  the blue flag, :math:`0` if the blue flag is home.
- :math:`\text{freeze}^{\text{blue}}_i, \text{freeze}^{\text{red}}_j` — steps
  left before a tagged player may act again.
- :math:`\text{cool}^{\text{blue}}_i, \text{cool}^{\text{red}}_j` — steps left
  before a player who tagged someone may tag again.
- :math:`\text{score}^{\text{blue}}, \text{score}^{\text{red}}` — the number
  of flags each team has captured.

:math:`s` is the state an action is taken from and :math:`s'` the state after
the step; a prime marks a component of :math:`s'`, as in
:math:`x^{\text{blue}\prime}_i`.

**Action space.** One action per blue player, issued together as one joint
action:

.. math::

   A = \{0, \dots, 5\}^{n_{\text{blue}}}, \qquad |A| = 6^{n_{\text{blue}}}

The joint action is the base-6 integer
:math:`a = \sum_{i=1}^{n_{\text{blue}}} a_i 6^{i-1}`, where :math:`a_i` is
blue player :math:`i`'s action:

- :math:`0` — move north, :math:`y + 1`;
- :math:`1` — move east, :math:`x + 1`;
- :math:`2` — move south, :math:`y - 1`;
- :math:`3` — move west, :math:`x - 1`;
- :math:`4` — hold: stay in place;
- :math:`5` — scan: stay in place, and read the flag detector at longer
  range this step.

Only blue is controlled; the red players move as part of :math:`T`.

**Observation space.** Each observation is one vector with these
components, in order:

- :math:`x^{\text{blue}}_{1:n_{\text{blue}}} \in G^{n_{\text{blue}}}` —
  every blue player's cell, exact.
- :math:`\hat d_{ij} \in \{0..d_{\max}\}` for every blue player :math:`i`
  and red player :math:`j`, ordered by :math:`i` then :math:`j` — a noisy
  reading of the Manhattan distance from :math:`i` to :math:`j`, where
  :math:`d_{\max} = W + H - 2` is the largest distance on the field.
- :math:`z_i \in \{0, 1\}` for every blue player :math:`i` — a noisy
  flag-detector bit; :math:`z_i = 1` is more likely the closer player
  :math:`i` is to the red flag.
- :math:`c^{\text{red}} \in \{0..n_{\text{blue}}\}` — the blue player
  carrying the red flag, :math:`0` if none, exact.
- :math:`\mathbb{1}[c^{\text{blue}} \neq 0] \in \{0, 1\}` — :math:`1` if a
  red player is carrying the blue flag, exact; *which* red player is not
  observed.
- :math:`\text{freeze}^{\text{blue}}_{1:n_{\text{blue}}}` — every blue
  player's freeze counter, exact.
- :math:`\text{score}^{\text{blue}}, \text{score}^{\text{red}}` — both
  scores, exact.

.. math::

   Z = \;&G^{n_{\text{blue}}} \times \{0..d_{\max}\}^{n_{\text{blue}} n_{\text{red}}} \times \{0,1\}^{n_{\text{blue}}} \\
   &\times \{0..n_{\text{blue}}\} \times \{0,1\} \times \mathbb{Z}_{\geq 0}^{n_{\text{blue}}}
   \times \mathbb{Z}_{\geq 0}^2 \;\cup\; \{(-1, \dots, -1)\}

The red players' cells, :math:`k`, and the red counters are not observed. A
terminal state emits the all :math:`-1` vector.

**Transition model.** The model uses:

- :math:`p_s` = ``slip_probability``;
- :math:`p_r` = ``red_pursuit_probability``;
- :math:`r` = ``red_alert_radius``;
- :math:`n_d` = ``n_red_defenders``;
- :math:`F_{\max}` = ``freeze_steps``;
- :math:`C_{\max}` = ``tagger_cooldown_steps``;
- :math:`m` = ``midline``;
- :math:`b^{\text{blue}}, b^{\text{red}}` = ``blue_base``, ``red_base``;
- :math:`h^{\text{blue}}` = ``blue_flag_cell``, :math:`h^{\text{red}} = f_k`;
- :math:`\mathrm{d}`, the Manhattan distance.

The blue half is left of the midline, the red half right of it:

.. math::

   H^{\text{blue}} = \{(x, y) \in G : x < m\}, \qquad
   H^{\text{red}} = \{(x, y) \in G : x > m\}

:math:`\Delta_a` is the one-cell move of action :math:`a`, and
:math:`\mathrm{side}(a)` is the set of the two moves at right angles to it — a
north or south move slips east or west, an east or west move slips north or
south:

.. math::

   &\Delta_0 = (0, 1),\; \Delta_1 = (1, 0),\; \Delta_2 = (0, -1),\; \Delta_3 = (-1, 0), \\
   &\mathrm{side}(0) = \mathrm{side}(2) = \{1, 3\},\;
   \mathrm{side}(1) = \mathrm{side}(3) = \{0, 2\}

:math:`\mathrm{blk}_x(y)` keeps a player at :math:`x` when the target
:math:`y` is a tree or off the field; :math:`N(x)` is :math:`x` and its free
neighbours:

.. math::

   \mathrm{blk}_x(y) = \begin{cases} y & y \in G \\ x & y \notin G \end{cases},
   \qquad N(x) = \{x\} \cup \{x + \Delta_a \in G : a \in \{0..3\}\}

:math:`\delta_x` is the distribution that puts probability 1 on cell
:math:`x`, and :math:`\mathcal{U}(N)` is the uniform distribution over a set
of cells :math:`N`. A step applies, in order:

*Blue moves.* A frozen, holding or scanning player stays. Otherwise it
moves as chosen, or slips to a right angle with probability :math:`p_s`.
With :math:`x = x^{\text{blue}}_i`:

.. math::

   x^{\text{blue}\prime}_i \sim \begin{cases}
     \delta_x & \text{freeze}^{\text{blue}}_i > 0 \;\lor\; a_i \in \{4, 5\} \\
     (1 - p_s)\,\delta_{\mathrm{blk}_x(x + \Delta_{a_i})}
       + \sum_{a' \in \mathrm{side}(a_i)} \tfrac{p_s}{2}\,\delta_{\mathrm{blk}_x(x + \Delta_{a'})}
       & \text{otherwise}
   \end{cases}

*Red moves.* Each red player first picks a target :math:`q_j`:

- the red base if it carries the blue flag;
- the blue flag if it is an attacker (:math:`j > n_d`);
- otherwise, as a defender, the nearest blue player in the red half
  :math:`\iota_j` if one is within :math:`r`, else the free cell beside the
  red flag nearest the red base.

With :math:`x = x^{\text{red}}_j`:

.. math::

   q_j &= \begin{cases}
     b^{\text{red}} & c^{\text{blue}} = j \\
     h^{\text{blue}} & j > n_d \\
     \iota_j & j \le n_d,\; \mathrm{d}(x, \iota_j) \le r \\
     \operatorname{arg\,min}_{y \in N(h^{\text{red}}) \setminus \{h^{\text{red}}\}} \mathrm{d}(y, b^{\text{red}}) & \text{otherwise}
   \end{cases}, \\
   \iota_j &= \operatorname{arg\,min}_{x^{\text{blue}\prime}_i \in H^{\text{red}}} \mathrm{d}(x, x^{\text{blue}\prime}_i)

It then steps toward :math:`q_j` with probability :math:`p_r` and to a random
neighbour otherwise; a frozen red player stays:

.. math::

   x^{\text{red}\prime}_j &\sim \begin{cases}
     \delta_x & \text{freeze}^{\text{red}}_j > 0 \\
     (1 - p_r)\,\mathcal{U}\big(N(x)\big) + p_r\,\mathcal{U}\big(N^\star_j\big) & \text{otherwise}
   \end{cases}, \\
   N^\star_j &= \operatorname{arg\,min}_{y \in N(x)} \mathrm{d}(y, q_j)

*Pick-up.* An unfrozen player on the other team's flag takes it if no one
carries it; the lowest index wins a tie:

.. math::

   c^{\text{red}} \leftarrow \min\{i : x^{\text{blue}\prime}_i = h^{\text{red}},\; \text{freeze}^{\text{blue}}_i = 0\}
   \quad \text{if } c^{\text{red}} = 0, \\
   c^{\text{blue}} \leftarrow \min\{j : x^{\text{red}\prime}_j = h^{\text{blue}},\; \text{freeze}^{\text{red}}_j = 0\}
   \quad \text{if } c^{\text{blue}} = 0

(unchanged when the set is empty).

*Tagging.* A player caught on an opponent's cell in the opponent's half is
sent home, frozen, and drops the flag; the tagger cools down. Red tags
first: for :math:`j = 1..n_{\text{red}}`, red player :math:`j` tags the lowest
:math:`i` with

.. math::

   x^{\text{blue}\prime}_i = x^{\text{red}\prime}_j \in H^{\text{red}},\quad
   \text{freeze}^{\text{blue}}_i = \text{freeze}^{\text{red}}_j = \text{cool}^{\text{red}}_j = 0

.. math::

   \Rightarrow\; x^{\text{blue}\prime}_i \leftarrow b^{\text{blue}},\;
   \text{freeze}^{\text{blue}\prime}_i \leftarrow F_{\max},\;
   \text{cool}^{\text{red}\prime}_j \leftarrow C_{\max},\;
   c^{\text{red}} \leftarrow 0 \text{ if } c^{\text{red}} = i

Then the same with colours swapped (:math:`H^{\text{blue}}`,
:math:`b^{\text{red}}`, :math:`c^{\text{blue}}`), skipping blue players
tagged this step.

*Scoring.* A team scores when its carrier stands on its own base while its
own flag is home; the captured flag goes back home:

.. math::

   \sigma^{\text{blue}} &= \mathbb{1}\big[c^{\text{red}} \ne 0 \wedge x^{\text{blue}\prime}_{c^{\text{red}}} = b^{\text{blue}} \wedge c^{\text{blue}} = 0\big], \\
   \sigma^{\text{red}} &= \mathbb{1}\big[c^{\text{blue}} \ne 0 \wedge x^{\text{red}\prime}_{c^{\text{blue}}} = b^{\text{red}} \wedge c^{\text{red}} = 0\big]

.. math::

   \text{score}^{\text{blue}\prime} = \text{score}^{\text{blue}} + \sigma^{\text{blue}},\;
   \text{score}^{\text{red}\prime} = \text{score}^{\text{red}} + \sigma^{\text{red}}, \\
   c^{\text{red}\prime} = (1 - \sigma^{\text{blue}})\,c^{\text{red}},\;
   c^{\text{blue}\prime} = (1 - \sigma^{\text{red}})\,c^{\text{blue}}

*Counters.* Every freeze and cooldown not just set by a tag counts down by
one:

.. math::

   \text{freeze}' = \max(\text{freeze} - 1, 0), \qquad \text{cool}' = \max(\text{cool} - 1, 0)

**Observation model.** The observation is drawn from the successor state
:math:`s'` and the joint action :math:`a`:

.. math::

   o = \big(x^{\text{blue}\prime}_{1:n_{\text{blue}}},\; \underbrace{\hat{d}_{ij}}_{n_{\text{blue}} \times n_{\text{red}}},\;
   \underbrace{z_{1:n_{\text{blue}}}}_{\text{flag detector}},\;
   c^{\text{red}\prime},\; \mathbb{1}[c^{\text{blue}\prime} \neq 0],\; \text{freeze}^{\text{blue}\prime}_{1:n_{\text{blue}}},\; \text{score}^{\text{blue}\prime}, \text{score}^{\text{red}\prime}\big)

Blue's cells, the red-flag carrier, whether the blue flag is taken, blue's
freeze counters and the scores are copied exactly from :math:`s'`: an
observation that disagrees with them has likelihood zero, not merely a small
one. The range readings :math:`\hat d_{ij}` and the flag-detector bits
:math:`z_i` are noisy, and drawn independently:

*Range readings.* Blue player :math:`i` reads the Manhattan distance
:math:`\mathrm{d}` to red player :math:`j`, correct with probability :math:`1 - p_e` where :math:`p_e` =
``range_error_probability`` (default :math:`0.2`), and off by one
otherwise:

.. math::

   \Pr[\hat{d}_{ij} = v] = \begin{cases}
     1 - p_e & v = d_{ij} \\
     p_e / 2 & v = d_{ij} \pm 1
   \end{cases}, \qquad d_{ij} = \mathrm{d}(x^{\text{blue}\prime}_i, x^{\text{red}\prime}_j)

clipped to :math:`[0, d_{\max}]`, :math:`d_{\max} = W + H - 2`, with the
out-of-range mass folded back onto the endpoint, so it sums to one at the
field's extremes too.

*Flag detector.* Blue player :math:`i` reads one bit :math:`z_i`, a noisy
signal that the red flag is near: :math:`z_i = 1` is likelier the nearer
player :math:`i` is to the red flag's cell :math:`f_k`:

.. math::

   \Pr[z_i = 1] = \tfrac{1}{2}\big(1 + 2^{-d^f_i / d_0(a_i)}\big),
   \qquad d^f_i = \mathrm{d}(x^{\text{blue}\prime}_i, f_k)

where :math:`a_i` is player :math:`i`'s action and :math:`d_0(a_i)` is the
distance at which the reading is right with probability :math:`0.75`:
``detector_half_distance_scan`` (default :math:`4.0`) when :math:`a_i = 5`
(scan) and ``detector_half_distance_move`` (default :math:`1.5`) otherwise —
the same
:math:`\tfrac{1}{2}(1 + 2^{-d/d_0})` law RockSample uses, so it never drops
below :math:`\tfrac{1}{2}`.

A terminal state emits the sentinel :math:`o = (-1, \dots, -1)`.

The asymmetry between the range readings and the flag-detector bits is the
planning problem. The
:math:`n_{\text{blue}} n_{\text{red}}` range readings **multiply**: at the default two-a-side, moving
one red player a single cell changes two of the four readings and costs a
factor of :math:`((1-p_e)/(p_e/2))^2 = 64` in likelihood. One flag scan barely
separates the candidates. Strong evidence about where the enemy is, weak
evidence about where the flag is.

**Reward function.** Additive over the realised transition, so it
needs :math:`s'` (``reward_requires_next_state`` is ``True``):

.. math::

   R(s, a, s') = \;&\texttt{capture\_reward} \cdot \Delta\text{score}^{\text{blue}}
   \;-\; \texttt{concede\_penalty} \cdot \Delta\text{score}^{\text{red}} \\
   &-\; \texttt{tagged\_penalty} \cdot n_{\text{suffered}}
   \;+\; \texttt{tag\_reward} \cdot n_{\text{inflicted}} \\
   &+\; \texttt{pickup\_reward} \cdot
     \mathbb{1}[c^{\text{red}} = 0 \wedge c^{\text{red}\prime} \neq 0]
   \;-\; \sum_{i=1}^{n_{\text{blue}}} \mathrm{cost}(a_i)

where:

- :math:`\Delta\text{score}` is a team's score in :math:`s'` minus its
  score in :math:`s`;
- :math:`n_{\text{suffered}}` and :math:`n_{\text{inflicted}}` are the
  numbers of blue and red players tagged this step;
- :math:`a_i` is blue player :math:`i`'s action.

Every blue player pays for its action, frozen or not:

.. math::

   \mathrm{cost}(a_i) = \begin{cases}
     \texttt{move\_cost} \;(\text{default } 1) & a_i \in \{0, 1, 2, 3, 4\} \text{ (a move or hold)} \\
     \texttt{scan\_cost} \;(\text{default } 2) & a_i = 5 \text{ (scan)}
   \end{cases}

:math:`R(s, a, s') = 0` for terminal :math:`s`.
A tag is read off the freeze counters: a player that was not frozen in
:math:`s` and has freeze ``freeze_steps`` in :math:`s'` was tagged this step. None of these terms exclude each other,
so the declared ``reward_range`` is the joint worst case, not the largest
single term.

**Initial belief.** Everything is known except which candidate :math:`k`
holds the red flag:

.. math::

   b_0\big(s(k)\big) = \tfrac{1}{K}, \qquad k \in \{1, \dots, K\}

where :math:`s(k)` has:

- every blue player on the blue base;
- every red player on the red base;
- no flag carried;
- every freeze counter, cooldown and score at zero.

The opening observation is drawn from the
observation model averaged over the :math:`K` candidates, so a filter that
weights it stays uniform over the candidates instead of favouring the nearer
ones. When there are more than 8192 possible opening observations, the
environment returns the single most likely one instead of listing them all.

**Discount.** :math:`\gamma` = ``discount_factor``, default :math:`0.98`.

**Terminal set.** The states where either team has captured
``score_to_win`` flags:

.. math::

   S_T = \{s : \text{score}^{\text{blue}} \geq \texttt{score\_to\_win}
   \ \text{or}\ \text{score}^{\text{red}} \geq \texttt{score\_to\_win}\}

with ``score_to_win`` = 1 by default.

Step order
~~~~~~~~~~

One step resolves in a fixed order:

1. blue moves;
2. red moves;
3. flags are picked up;
4. tags are resolved;
5. scores are awarded;
6. counters tick.

Changing the order
changes the outcome. Pick-up runs before tagging, so a player tagged
while standing on the flag cell has already taken the flag and therefore drops
it. Both scoring conditions are judged against the same carrier indices, which
keeps blue scoring and red scoring mutually exclusive -- each needs the other
side's flag to be home.

Tagging and respawn
~~~~~~~~~~~~~~~~~~~

A player is tagged when an opponent shares its cell **in the enemy half**, and
only by a tagger that is neither frozen nor on cooldown. Being tagged does
not end the episode: the player returns to its own base, drops any flag it
carried, and is frozen for ``freeze_steps``. The tagger is put on cooldown for
``tagger_cooldown_steps``, which is what stops a defender camping the flag and
tagging repeatedly.

Rewards
-------

The reward terms:

- scoring earns ``capture_reward``;
- conceding costs ``concede_penalty``;
- each blue player tagged costs ``tagged_penalty``;
- each red player tagged earns ``tag_reward``;
- first pick-up earns ``pickup_reward``;
- every player pays its action's cost.

None of these exclude each other, so the declared reward
range is the joint worst case rather than the largest single term.

Default values, from the constructor:

========================================  ============================
Event                                     Reward
========================================  ============================
Blue scores (``capture_reward``)          +100.0
Red scores (``concede_penalty``)          -100.0
Blue player tagged (``tagged_penalty``)   -25.0 each
Red player tagged (``tag_reward``)        +10.0 each
First pick-up (``pickup_reward``)         +20.0
Move or hold (``move_cost``)              -1.0 per player
Scan (``scan_cost``)                      -2.0 per player
========================================  ============================

Key settings
------------

The defaults:

- a 9 by 7 field (``grid_size=(9, 7)``) split at column ``midline=4``;
- two players per team (``n_blue=2``, ``n_red=2``), one of them a red
  defender (``n_red_defenders=1``);
- blue moves slip sideways with probability ``slip_probability=0.1``;
- a range badge reads off by one with probability
  ``range_error_probability=0.2``;
- a red player steps toward its target with probability
  ``red_pursuit_probability=0.7``;
- ``score_to_win=1`` capture ends the episode;
- ``discount_factor`` defaults to ``0.98``.

The reward arguments are listed under Rewards above.

Metrics
~~~~~~~

- The completion metric is ``task_completion_rate``.
- Episodes are also split by why they ended -- ``ended_by_goal_rate``,
  ``ended_by_failure_rate`` and ``ended_by_timeout_rate`` -- because a
  completion rate alone cannot tell a planner taking bad risks from one given
  too small a step budget.
- Alongside episode length, the environment reports tags suffered and
  inflicted, steps spent holding the enemy flag, and exposure in the enemy half
  as both a count and a per-episode maximum.

Visualization
~~~~~~~~~~~~~

Runs write a trace of each episode through the environment's episode
visualizer. The results site replays it in 3D, as the replay on this page does.

Limits
~~~~~~

The environment has no torch vectorized model, so it cannot be run under VOPP.
``PFT_DPW`` runs on the scalar ``Environment`` API.

Can I use?
----------

.. list-table::
   :header-rows: 1
   :widths: 20 12 12

   * - ``CaptureTheFlagPOMDP``
     - Discrete
     - Continuous
   * - State
     - ✔️
     - ❌
   * - Action
     - ✔️
     - ❌
   * - Observation
     - ❌
     - ✔️

.. list-table::
   :header-rows: 1
   :widths: 34 30

   * - Also supports
     -
   * - Native C++ backend
     - ❌
   * - Vectorized (torch) model
     - ❌
   * - In the ``get_environment`` registry
     - ✔️
   * - Optional dependencies
     - None

Example
-------

.. code-block:: python

   from POMDPPlanners.environments.capture_the_flag_pomdp import CaptureTheFlagPOMDP

   env = CaptureTheFlagPOMDP(grid_size=(9, 7), midline=4, n_blue=2, n_red=2)

Parameters
----------

.. autoclass:: POMDPPlanners.environments.capture_the_flag_pomdp.CaptureTheFlagPOMDP
   :members:
   :show-inheritance:

See also
--------

- :class:`POMDPPlanners.environments.capture_the_flag_pomdp.CaptureTheFlagPOMDP`
- :doc:`base` — the full catalog and the environment interface.
