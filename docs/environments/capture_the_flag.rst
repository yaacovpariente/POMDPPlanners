Capture the Flag
================

.. episode-viewer:: traces/capture_the_flag.json

   One real episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
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

The environment is the POMDP :math:`\langle S, A, \Omega, T, O, R, b_0, \gamma
\rangle`. Write :math:`G` for the free cells (in bounds, not a tree), :math:`\text{Blue}` for
the blue players and :math:`\text{Red}` for red, with :math:`|\text{Blue}| = n_{\text{blue}}` and
:math:`|\text{Red}| = n_{\text{red}}`. Let :math:`F = (f_1, \dots, f_K)` be the red flag
candidates and :math:`\mathrm{d}` the Manhattan distance.

**State space.** One vector holding both teams and all the bookkeeping:

.. math::

   s = \big(\underbrace{x^{\text{blue}}_{1:n_{\text{blue}}}}_{\text{blue cells}},\;
            \underbrace{x^{\text{red}}_{1:n_{\text{red}}}}_{\text{red cells}},\;
            k,\;
            c^{\text{red}}, c^{\text{blue}},\;
            \text{freeze}^{\text{blue}}_{1:n_{\text{blue}}}, \text{freeze}^{\text{red}}_{1:n_{\text{red}}},\;
            \text{cool}^{\text{blue}}_{1:n_{\text{blue}}}, \text{cool}^{\text{red}}_{1:n_{\text{red}}},\;
            \text{score}^{\text{blue}}, \text{score}^{\text{red}} \big)

.. math::

   S = G^{n_{\text{blue}}} \times G^{n_{\text{red}}} \times \{1..K\} \times
   \{0..n_{\text{blue}}\} \times \{0..n_{\text{red}}\} \times
   \mathbb{Z}_{\geq 0}^{2(n_{\text{blue}} + n_{\text{red}})} \times \mathbb{Z}_{\geq 0}^2

Here :math:`k` indexes which candidate holds the red flag, :math:`c^{\text{red}}` is
the blue player carrying the red flag (:math:`0` = nobody) and :math:`c^{\text{blue}}` the
red player carrying the blue flag, :math:`\text{freeze}` are respawn-freeze counters,
:math:`\text{cool}` tagger cooldowns, and :math:`\text{score}` the scores. A flag is either
home or on a carrier's back — a dropped flag returns home at once — so one
index replaces a second position.

**Action space.** Six per player, issued as one joint action:

.. math::

   A = \{0, \dots, 5\}^{n_{\text{blue}}}, \qquad |A| = 6^{n_{\text{blue}}}

encoded as a base-6 integer :math:`a = \sum_i a_i 6^i`. Per player:
:math:`0..3` move, :math:`4` hold, :math:`5` scan. Only blue is controlled;
red is part of :math:`T`.

**Observation space.** Blue's own cells exactly, the two noisy channels, and
the exactly observed bookkeeping, plus a terminal sentinel:

.. math::

   \Omega = G^{n_{\text{blue}}} \times \{0..d_{\max}\}^{n_{\text{blue}} n_{\text{red}}} \times \{0,1\}^{n_{\text{blue}}}
   \times \{0..n_{\text{blue}}\} \times \{0,1\} \times \mathbb{Z}_{\geq 0}^{n_{\text{blue}}}
   \times \mathbb{Z}_{\geq 0}^2 \;\cup\; \{(-1, \dots, -1)\}

with :math:`d_{\max} = W + H - 2` for a :math:`W \times H` field, the
largest Manhattan distance on it.

**Transition model.** :math:`T` is the composition of four stages, in this
order. Let :math:`p_s` = ``slip_probability``, :math:`p_r` =
``red_pursuit_probability``.

*Stage 1 — blue moves.* A frozen player (:math:`\text{freeze}^{\text{blue}}_i > 0`) cannot move. A
move goes where intended with probability :math:`1 - p_s` and deflects to
either perpendicular direction with probability :math:`p_s / 2` each; a move
into a tree or off the field leaves the player in place:

.. math::

   \Pr[x^{\text{blue}\prime}_i = y] = \sum_{a'} w(a') \,
   \mathbb{1}\big[y = \mathrm{block}(x^{\text{blue}}_i + \Delta_{a'})\big], \quad
   w(a_i) = 1 - p_s,\; w(a_i^\perp) = \tfrac{p_s}{2}

where :math:`\mathrm{block}(y) = y` if :math:`y \in G`, else the current cell.
Two different slips can be blocked into the same cell, so the outcomes are
accumulated.

*Stage 2 — red moves.* Each red player picks a target :math:`q_j`
deterministically from its role: a carrier runs for its base; an attacker for
the blue flag cell; a defender guards the cell beside the red flag, switching
to the nearest blue intruder in the red half once that intruder is within
``red_alert_radius``. Let :math:`N_j` be the free 4-neighbours of
:math:`x^{\text{red}}_j` together with :math:`x^{\text{red}}_j` itself, and
:math:`N_j^\star \subseteq N_j` those minimizing :math:`\mathrm{d}(\cdot,
q_j)`. Then

.. math::

   \Pr[x^{\text{red}\prime}_j = y] = \frac{1 - p_r}{|N_j|}\mathbb{1}[y \in N_j]
   + \frac{p_r}{|N_j^\star|}\mathbb{1}[y \in N_j^\star]

so red closes on its target with probability :math:`p_r` and otherwise wanders
uniformly. A frozen red player stays put.

*Stages 3–6 — deterministic resolution.* Given both teams' realised cells:

1. **Pick-up.** An unfrozen blue player on the red flag cell takes the flag if
   :math:`c^{\text{red}} = 0`; symmetrically for red. Lowest index wins a tie.
2. **Tagging.** Blue player :math:`i` is tagged when it shares a cell with an
   unfrozen, off-cooldown red player **in the red half**: it teleports to the
   blue base, drops any flag, and gets :math:`\text{freeze}^{\text{blue}}_i \leftarrow`
   ``freeze_steps``, while the tagger gets :math:`\text{cool}^{\text{red}}_j \leftarrow`
   ``tagger_cooldown_steps``. Symmetric for red in the blue half.
3. **Scoring.** Both conditions are judged against the *same*, pre-scoring
   carrier indices:

   .. math::

      \text{blue scores} &\iff c^{\text{red}} \neq 0 \;\wedge\;
        x^{\text{blue}\prime}_{c^{\text{red}}} = \text{blue base} \;\wedge\; c^{\text{blue}} = 0 \\
      \text{red scores} &\iff c^{\text{blue}} \neq 0 \;\wedge\;
        x^{\text{red}\prime}_{c^{\text{blue}}} = \text{red base} \;\wedge\; c^{\text{red}} = 0

   which makes them mutually exclusive: each side needs the other's flag home.
4. **Counters.** :math:`\text{freeze}, \text{cool}` decrement toward zero, against the values
   carried in from :math:`s`, so a freeze set this step lasts its full length.

Pick-up strictly precedes tagging, so a player tagged on the flag cell has
already taken the flag and therefore drops it. The support is the product of
the per-player move outcomes — at most three per blue player, five per red —
so :math:`T` is enumerated exactly rather than sampled from.

**Observation model.** Blue sees its own team exactly and the red team only
through two noisy channels. The observation is the concatenation

.. math::

   o = \big(x^{\text{blue}}_{1:n_{\text{blue}}},\; \underbrace{\hat{d}_{ij}}_{n_{\text{blue}} \times n_{\text{red}}},\;
   \underbrace{z_{1:n_{\text{blue}}}}_{\text{flag detector}},\;
   c^{\text{red}},\; \mathbb{1}[c^{\text{blue}} \neq 0],\; \text{freeze}^{\text{blue}}_{1:n_{\text{blue}}},\; \text{score}^{\text{blue}}, \text{score}^{\text{red}}\big)

The exact components act as an indicator factor — an observation disagreeing with
them has likelihood zero, not merely a small one. The two noisy channels:

*Range badges.* Player :math:`i` reads the Manhattan distance to red player
:math:`j`, correct with probability :math:`1 - p_e` where :math:`p_e` =
``range_error_probability``, and off by one otherwise:

.. math::

   \Pr[\hat{d}_{ij} = v] = \begin{cases}
     1 - p_e & v = d_{ij} \\
     p_e / 2 & v = d_{ij} \pm 1
   \end{cases}, \qquad d_{ij} = \mathrm{d}(x^{\text{blue}\prime}_i, x^{\text{red}\prime}_j)

clipped to :math:`[0, d_{\max}]` with the out-of-range mass folded back onto
the endpoint, so it sums to one at the field's extremes too.

*Flag detector.* A binary reading per player whose accuracy decays with
distance to the red flag cell:

.. math::

   \Pr[z_i = 1] = \tfrac{1}{2}\big(1 + 2^{-d^f_i / d_0(a_i)}\big),
   \qquad d^f_i = \mathrm{d}(x^{\text{blue}\prime}_i, f_k)

with :math:`d_0(a_i) =` ``detector_half_distance_scan`` when :math:`a_i = 5`
and ``detector_half_distance_move`` otherwise — the same
:math:`\tfrac{1}{2}(1 + 2^{-d/d_0})` law RockSample uses, so it never drops
below :math:`\tfrac{1}{2}`.

A terminal state emits the sentinel :math:`o = (-1, \dots, -1)`.

The asymmetry between the two channels is the planning problem. The
:math:`n_{\text{blue}} n_{\text{red}}` range readings **multiply**: at the default two-a-side, moving
one red player a single cell changes two of the four readings and costs a
factor of :math:`((1-p_e)/(p_e/2))^2 = 64` in likelihood. One flag scan barely
separates the candidates. Strong evidence about where the enemy is, weak
evidence about where the flag is.

**Reward function.** Additive over the realised transition, so it genuinely
needs :math:`s'` (``reward_requires_next_state`` is ``True``):

.. math::

   R(s, a, s') = \;&\texttt{capture\_reward} \cdot \Delta\text{score}^{\text{blue}}
   \;-\; \texttt{concede\_penalty} \cdot \Delta\text{score}^{\text{red}} \\
   &-\; \texttt{tagged\_penalty} \cdot n_{\text{suffered}}
   \;+\; \texttt{tag\_reward} \cdot n_{\text{inflicted}} \\
   &+\; \texttt{pickup\_reward} \cdot
     \mathbb{1}[c^{\text{red}} = 0 \wedge c^{\text{red}\prime} \neq 0]
   \;-\; \sum_{i=1}^{n_{\text{blue}}} \mathrm{cost}(a_i)

with :math:`\mathrm{cost}(a_i) =` ``scan_cost`` for a scan and ``move_cost``
otherwise, and :math:`R(s, a, s') = 0` for terminal :math:`s`. A tag is read
off the counters: a player that was free and is now frozen for the full
``freeze_steps`` was tagged this step. None of these terms exclude each other,
so the declared ``reward_range`` is the joint worst case, not the largest
single term.

**Initial belief.** Everything known but the flag:

.. math::

   b_0\big(s(k)\big) = \tfrac{1}{K}, \qquad k \in \{1, \dots, K\}

where :math:`s(k)` spawns every blue player on the blue base, every red
player on the red base, all counters and scores at zero. The opening
observation is a genuine draw from :math:`O` — the mixture over candidates of
the noise each implies — so a filter that weights it stays uniform over the
candidates instead of favouring the nearer ones. When that support has more
than 8192 observations, the environment returns the single most likely one
instead of enumerating it.

**Discount.** :math:`\gamma` = ``discount_factor``, default :math:`0.98`.

**Terminal set.** Either side reaching the target score:

.. math::

   S_T = \{s : \text{score}^{\text{blue}} \geq \texttt{score\_to\_win}
   \ \text{or}\ \text{score}^{\text{red}} \geq \texttt{score\_to\_win}\}

with ``score_to_win`` = 1 by default.

Rewards
-------

Scoring earns ``capture_reward``; conceding costs ``concede_penalty``. Each
blue player tagged costs ``tagged_penalty``, each red player tagged earns
``tag_reward``, first pick-up earns ``pickup_reward``, and every player pays
its action's cost. None of these exclude each other, so the declared reward
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

Step order
----------

One step resolves in a fixed order: blue moves, red moves, flags are picked
up, tags are resolved, scores are awarded, counters tick. The order is
semantics rather than style. Pick-up runs before tagging, so a player tagged
while standing on the flag cell has already taken the flag and therefore drops
it. Both scoring conditions are judged against the same carrier indices, which
keeps blue scoring and red scoring mutually exclusive -- each needs the other
side's flag to be home.

Tagging and respawn
-------------------

A player is tagged when an opponent shares its cell **in the enemy half**, and
only by a tagger that is neither frozen nor on cooldown. Being tagged is a
setback, not a failure: the player returns to its own base, drops any flag it
carried, and is frozen for ``freeze_steps``. The tagger is put on cooldown for
``tagger_cooldown_steps``, which is what stops a defender camping the flag and
tagging repeatedly.

Belief
------

``create_environment_belief`` returns ``CaptureTheFlagVectorizedBelief``, a
particle filter whose transition and likelihood both run over the whole
particle set at once through ``CaptureTheFlagVectorizedUpdater``.

Two parts of a reading need different treatment. Blue's own positions, the
carrier ids, its freezes and both scores come back without noise, so the
posterior puts all its mass on them; the belief writes them onto every particle
rather than weighting by them, because weighting floors every particle whose
blue player slipped differently from the real one -- most of them, most steps.
And the flag candidate is static: nothing in the transition moves a particle
from one candidate to another, so resampling across candidates deletes
hypotheses permanently. Resampling therefore happens inside a candidate.

What remains is a genuine filter with a sharp likelihood, so it converges on
the true candidate in most episodes and over-commits to a wrong one in a few:
over twelve 15-step episodes on the default field it held a mean weight near
0.7 on the truth at 200 particles and near 0.8 at 400.

Metrics
-------

The completion metric is ``task_completion_rate``. Episodes are also split by
why they ended -- ``ended_by_goal_rate``, ``ended_by_failure_rate`` and
``ended_by_timeout_rate`` -- because a completion rate alone cannot tell a
planner taking bad risks from one given too small a step budget. Alongside
episode length, the environment reports tags suffered and inflicted, steps
spent holding the enemy flag, and exposure in the enemy half as both a count
and a per-episode maximum.

Visualization
-------------

Runs also write a GIF of each episode through ``cache_visualization``. Its
camera is isometric and the terrain is one continuous procedural field, so the
map reads as a section of a larger world rather than a board. Soldiers
interpolate between cells rather than teleporting, a scan plays its own ping,
and a tagged player is drawn translucent while frozen.

The belief is a separate overlay, never baked into the world art: coloured
markers give each red player's position marginal, sized by probability mass,
and translucent diamonds over the flag candidates carry the marginal over the
red flag's cell. Drawing only the true trajectory would give an MDP picture of
a POMDP -- it could not distinguish a planner that handled uncertainty from
one that got lucky. Particles are drawn rather than summarised by an ellipse,
because a belief over a hidden pursuer routinely goes multi-modal and an
ellipse would hide exactly that.

Limits
------

The environment has no torch vectorized model, so it cannot be run under VOPP.
``PFT_DPW`` runs on the scalar ``Environment`` API.

Minimal example
---------------

.. code-block:: python

   from POMDPPlanners.environments.capture_the_flag_pomdp import CaptureTheFlagPOMDP

   env = CaptureTheFlagPOMDP(grid_size=(9, 7), midline=4, n_blue=2, n_red=2)

See also
--------

- :class:`POMDPPlanners.environments.capture_the_flag_pomdp.CaptureTheFlagPOMDP`
- :doc:`index` — the full catalog.
