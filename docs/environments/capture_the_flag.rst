Capture the Flag
================

``CaptureTheFlagPOMDP`` puts two teams on a grid field split by a midline.
Each team has a flag. The planner drives the blue team as one joint
controller; the red team belongs to the transition model and follows a
stochastic role policy. Blue wins by carrying the red flag to the blue base
while its own flag is still home.

Blue never sees the red players, and does not know which of ``K`` candidate
cells holds the red flag. Both are inferred from one observation vector.

.. code-block:: python

   from POMDPPlanners.environments.capture_the_flag_pomdp import CaptureTheFlagPOMDP

   env = CaptureTheFlagPOMDP(grid_size=(9, 7), midline=4, n_blue=2, n_red=2)

State and actions
-----------------

The state carries every player on the field: blue cells, red cells, the red
flag's home cell, a carrier index per flag, a respawn-freeze and a
tagger-cooldown counter per player, and the score. A flag is either at its
home cell or on a carrier's back, because a dropped flag returns home at once,
so one carrier index replaces a second position.

Each player has six actions -- four moves, hold, and scan -- and the planner
issues one joint action for the whole blue team, so the action space is
``6 ** n_blue``: 36 at the default two-a-side. Scan does not move the player;
it widens that player's flag detector for the step and costs more. A move
slips to a perpendicular direction with probability ``slip_probability``, and
a move into a tree or off the field leaves the player where it stood.

One step resolves in a fixed order: blue moves, red moves, flags are picked
up, tags are resolved, scores are awarded, counters tick. The order is
semantics rather than style. Pick-up runs before tagging, so a player tagged
while standing on the flag cell has already taken the flag and therefore drops
it. Both scoring conditions are judged against the same carrier indices, which
keeps blue scoring and red scoring mutually exclusive -- each needs the other
side's flag to be home.

Formal definition
-----------------

Write :math:`G` for the free cells (in bounds, not a tree), :math:`B` for blue
and :math:`\mathcal{R}` for red, with :math:`|B| = n_b` and
:math:`|\mathcal{R}| = n_r`. Let :math:`F = (f_1, \dots, f_K)` be the red flag
candidates and :math:`\mathrm{d}` the Manhattan distance.

**State space.** One vector holding both teams and all the bookkeeping:

.. math::

   s = \big(\underbrace{x^b_{1:n_b}}_{\text{blue cells}},\;
            \underbrace{x^r_{1:n_r}}_{\text{red cells}},\;
            \kappa,\;
            c^r, c^b,\;
            \phi^b_{1:n_b}, \phi^r_{1:n_r},\;
            \psi^b_{1:n_b}, \psi^r_{1:n_r},\;
            \sigma^b, \sigma^r \big)

.. math::

   S = G^{n_b} \times G^{n_r} \times \{1..K\} \times
   \{0..n_b\} \times \{0..n_r\} \times
   \mathbb{Z}_{\geq 0}^{2(n_b + n_r)} \times \mathbb{Z}_{\geq 0}^2

Here :math:`\kappa` indexes which candidate holds the red flag, :math:`c^r` is
the blue player carrying the red flag (:math:`0` = nobody) and :math:`c^b` the
red player carrying the blue flag, :math:`\phi` are respawn-freeze counters,
:math:`\psi` tagger cooldowns, and :math:`\sigma` the scores. A flag is either
home or on a carrier's back — a dropped flag returns home at once — so one
index replaces a second position.

**Action space.** Six per player, issued as one joint action:

.. math::

   A = \{0, \dots, 5\}^{n_b}, \qquad |A| = 6^{n_b}

encoded as a base-6 integer :math:`a = \sum_i a_i 6^i`. Per player:
:math:`0..3` move, :math:`4` hold, :math:`5` scan. Only blue is controlled;
red is part of :math:`T`.

**Transition model.** :math:`T` is the composition of four stages, in this
order. Let :math:`p_s` = ``slip_probability``, :math:`p_r` =
``red_pursuit_probability``.

*Stage 1 — blue moves.* A frozen player (:math:`\phi^b_i > 0`) cannot move. A
move goes where intended with probability :math:`1 - p_s` and deflects to
either perpendicular direction with probability :math:`p_s / 2` each; a move
into a tree or off the field leaves the player in place:

.. math::

   \Pr[x^{b\prime}_i = y] = \sum_{a'} w(a') \,
   \mathbb{1}\big[y = \mathrm{block}(x^b_i + \Delta_{a'})\big], \quad
   w(a_i) = 1 - p_s,\; w(a_i^\perp) = \tfrac{p_s}{2}

where :math:`\mathrm{block}(y) = y` if :math:`y \in G`, else the current cell.
Two different slips can be blocked into the same cell, so the outcomes are
accumulated.

*Stage 2 — red moves.* Each red player picks a target :math:`\tau_j`
deterministically from its role: a carrier runs for its base; an attacker for
the blue flag cell; a defender guards the cell beside the red flag, switching
to the nearest blue intruder in the red half once that intruder is within
``red_alert_radius``. Let :math:`N_j` be the free 4-neighbours of
:math:`x^r_j` together with :math:`x^r_j` itself, and
:math:`N_j^\star \subseteq N_j` those minimizing :math:`\mathrm{d}(\cdot,
\tau_j)`. Then

.. math::

   \Pr[x^{r\prime}_j = y] = \frac{1 - p_r}{|N_j|}\mathbb{1}[y \in N_j]
   + \frac{p_r}{|N_j^\star|}\mathbb{1}[y \in N_j^\star]

so red closes on its target with probability :math:`p_r` and otherwise wanders
uniformly. A frozen red player stays put.

*Stages 3–6 — deterministic resolution.* Given both teams' realised cells:

1. **Pick-up.** An unfrozen blue player on the red flag cell takes the flag if
   :math:`c^r = 0`; symmetrically for red. Lowest index wins a tie.
2. **Tagging.** Blue player :math:`i` is tagged when it shares a cell with an
   unfrozen, off-cooldown red player **in the red half**: it teleports to the
   blue base, drops any flag, and gets :math:`\phi^b_i \leftarrow`
   ``freeze_steps``, while the tagger gets :math:`\psi^r_j \leftarrow`
   ``tagger_cooldown_steps``. Symmetric for red in the blue half.
3. **Scoring.** Both conditions are judged against the *same*, pre-scoring
   carrier indices:

   .. math::

      \text{blue scores} &\iff c^r \neq 0 \;\wedge\;
        x^{b\prime}_{c^r} = \text{blue base} \;\wedge\; c^b = 0 \\
      \text{red scores} &\iff c^b \neq 0 \;\wedge\;
        x^{r\prime}_{c^b} = \text{red base} \;\wedge\; c^r = 0

   which makes them mutually exclusive: each side needs the other's flag home.
4. **Counters.** :math:`\phi, \psi` decrement toward zero, against the values
   carried in from :math:`s`, so a freeze set this step lasts its full length.

Pick-up strictly precedes tagging, so a player tagged on the flag cell has
already taken the flag and therefore drops it. The support is the product of
the per-player move outcomes — at most three per blue player, five per red —
so :math:`T` is enumerated exactly rather than sampled from.

**Observation model.** Blue sees its own team exactly and the red team only
through two noisy channels. The observation is the concatenation

.. math::

   o = \big(x^b_{1:n_b},\; \underbrace{\hat{d}_{ij}}_{n_b \times n_r},\;
   \underbrace{\beta_{1:n_b}}_{\text{flag detector}},\;
   c^r,\; \mathbb{1}[c^b \neq 0],\; \phi^b_{1:n_b},\; \sigma^b, \sigma^r\big)

The exact components act as a delta factor — an observation disagreeing with
them has likelihood zero, not merely a small one. The two noisy channels:

*Range badges.* Player :math:`i` reads the Manhattan distance to red player
:math:`j`, correct with probability :math:`1 - p_e` where :math:`p_e` =
``range_error_probability``, and off by one otherwise:

.. math::

   \Pr[\hat{d}_{ij} = v] = \begin{cases}
     1 - p_e & v = d_{ij} \\
     p_e / 2 & v = d_{ij} \pm 1
   \end{cases}, \qquad d_{ij} = \mathrm{d}(x^{b\prime}_i, x^{r\prime}_j)

clipped to :math:`[0, d_{\max}]` with the out-of-range mass folded back onto
the endpoint, so it sums to one at the field's extremes too.

*Flag detector.* A binary reading per player whose accuracy decays with
distance to the red flag cell:

.. math::

   \Pr[\beta_i = 1] = \tfrac{1}{2}\big(1 + 2^{-d^f_i / d_0(a_i)}\big),
   \qquad d^f_i = \mathrm{d}(x^{b\prime}_i, f_\kappa)

with :math:`d_0(a_i) =` ``detector_half_distance_scan`` when :math:`a_i = 5`
and ``detector_half_distance_move`` otherwise — the same
:math:`\tfrac{1}{2}(1 + 2^{-d/d_0})` law RockSample uses, so it never drops
below :math:`\tfrac{1}{2}`.

A terminal state emits the sentinel :math:`o = (-1, \dots, -1)`.

The asymmetry between the two channels is the planning problem. The
:math:`n_b n_r` range readings **multiply**: at the default two-a-side, moving
one red player a single cell changes two of the four readings and costs a
factor of :math:`((1-p_e)/(p_e/2))^2 = 64` in likelihood. One flag scan barely
separates the candidates. Strong evidence about where the enemy is, weak
evidence about where the flag is.

**Reward function.** Additive over the realised transition, so it genuinely
needs :math:`s'` (``reward_requires_next_state`` is ``True``):

.. math::

   R(s, a, s') = \;&\texttt{capture\_reward} \cdot \Delta\sigma^b
   \;-\; \texttt{concede\_penalty} \cdot \Delta\sigma^r \\
   &-\; \texttt{tagged\_penalty} \cdot n_{\text{suffered}}
   \;+\; \texttt{tag\_reward} \cdot n_{\text{inflicted}} \\
   &+\; \texttt{pickup\_reward} \cdot
     \mathbb{1}[c^r = 0 \wedge c^{r\prime} \neq 0]
   \;-\; \sum_{i=1}^{n_b} \mathrm{cost}(a_i)

with :math:`\mathrm{cost}(a_i) =` ``scan_cost`` for a scan and ``move_cost``
otherwise, and :math:`R(s, a, s') = 0` for terminal :math:`s`. A tag is read
off the counters: a player that was free and is now frozen for the full
``freeze_steps`` was tagged this step. None of these terms exclude each other,
so the declared ``reward_range`` is the joint worst case, not the largest
single term.

**Initial belief.** Everything known but the flag:

.. math::

   b_0\big(s(\kappa)\big) = \tfrac{1}{K}, \qquad \kappa \in \{1, \dots, K\}

where :math:`s(\kappa)` spawns every blue player on the blue base, every red
player on the red base, all counters and scores at zero. The opening
observation is a genuine draw from :math:`O` — the mixture over candidates of
the noise each implies — so a filter that weights it stays uniform over the
candidates instead of favouring the nearer ones.

**Discount.** :math:`\gamma` = ``discount_factor``, default :math:`0.98`.

**Terminal set.** Either side reaching the target score:

.. math::

   S_T = \{s : \sigma^b \geq \texttt{score\_to\_win}
   \ \text{or}\ \sigma^r \geq \texttt{score\_to\_win}\}

Tagging and respawn
-------------------

A player is tagged when an opponent shares its cell **in the enemy half**, and
only by a tagger that is neither frozen nor on cooldown. Being tagged is a
setback, not a failure: the player returns to its own base, drops any flag it
carried, and is frozen for ``freeze_steps``. The tagger is put on cooldown for
``tagger_cooldown_steps``, which is what stops a defender camping the flag and
tagging repeatedly.

Observations
------------

Blue sees its own team exactly -- positions, freeze counters, who is carrying
-- plus the score and whether its own flag has been taken. It does not see the
red players, the red flag cell, or which red player took its flag.

Each blue player carries a range badge reporting a noisy Manhattan distance to
each red player, correct with probability ``1 - range_error_probability`` and
off by one otherwise, and a binary detector for the red flag cell that fires
with probability ``0.5 * (1 + 2 ** (-d / d0))``. The half-distance ``d0`` is
wider after a scan.

Because every blue player measures every red player, the ``n_blue * n_red``
range readings multiply: moving one red player a single cell changes two of
the four readings at the default team size, costing a factor of 64 in
likelihood. A single flag scan, by contrast, barely separates the candidates.
Strong evidence about where the enemy is, weak evidence about where the flag
is -- that asymmetry is the planning problem.

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

Rewards and metrics
-------------------

Scoring earns ``capture_reward``; conceding costs ``concede_penalty``. Each
blue player tagged costs ``tagged_penalty``, each red player tagged earns
``tag_reward``, first pick-up earns ``pickup_reward``, and every player pays
its action's cost. None of these exclude each other, so the declared reward
range is the joint worst case rather than the largest single term.

The completion metric is ``task_completion_rate``. Episodes are also split by
why they ended -- ``ended_by_goal_rate``, ``ended_by_failure_rate`` and
``ended_by_timeout_rate`` -- because a completion rate alone cannot tell a
planner taking bad risks from one given too small a step budget. Alongside
episode length, the environment reports tags suffered and inflicted, steps
spent holding the enemy flag, and exposure in the enemy half as both a count
and a per-episode maximum.

Recorded visualization
----------------------

.. image:: ../../POMDPPlanners/tests/test_environments/golden_visualizations/capture_the_flag_visualization.gif
   :alt: Isometric capture-the-flag field with two teams of soldiers, a river crossing, a scoreboard and a belief overlay over the hidden red players and flag.
   :width: 100%

The camera is isometric and the terrain is one continuous procedural field, so
the map reads as a section of a larger world rather than a board. Soldiers
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

The environment has no torch vectorized model, so it cannot be run under VOPP.
``PFT_DPW`` runs on the scalar ``Environment`` API.
