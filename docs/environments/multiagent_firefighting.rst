Multi-agent firefighting
========================

.. episode-viewer:: traces/multiagent_firefighting.json

   One real episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

``MultiAgentFirefightingPOMDP`` puts ``N`` firefighting robots on an ``R x C``
grid and asks them to put out a fire that spreads under a **hidden, constant
wind**. The robots see the fire only near themselves, carry a finite tank of
suppressant, and take heat damage for standing in flames. The task is complete
when no cell is burning.

The wind is never observed. It has to be inferred from how the fire spreads,
while the robots decide where to look, where to spray, and when to go back to
the depot to refill. Use it to test planning over a large joint action space
with a belief over a whole map.

What the agent sees and does
----------------------------

- **State** — one ``float64`` vector of length ``3 + 4N + R*C``: the step
  counter, ``(row, col, tank, health)`` per robot, the wind direction and
  strength, then one category per cell (``0`` unburnt to ``4`` wet). Length
  111 at the defaults.
- **Actions** (discrete) — one ``int`` in ``[0, 5 ** N)``. Its base-5 digits,
  least significant first, are the per-robot actions ``NORTH`` (0), ``EAST``
  (1), ``SOUTH`` (2), ``WEST`` (3), ``SUPPRESS`` (4). 25 actions at the default
  two robots.
- **Observations** (discrete) — one ``float64`` vector of length
  ``4N + R*C``: exact ``(row, col, tank, health)`` per robot, then a noisy
  category per cell within ``sensing_radius`` of a live robot and ``-1``
  (unknown) everywhere else. The wind never appears.

Formal definition
-----------------

The environment is the POMDP :math:`\langle S, A, \Omega, T, O, R, b_0, \gamma
\rangle`. Let the grid have :math:`H \times W` cells with obstacle set :math:`\mathcal{O}`,
and let :math:`M` = ``num_robots``. Cell categories are

.. math::

   \mathcal{C} = \{\textsf{UNBURNT}, \textsf{SMOLDERING}, \textsf{BURNING},
   \textsf{BURNT}, \textsf{WET}\} = \{0,1,2,3,4\}

with :math:`\mathcal{A}\ell = \{\textsf{SMOLDERING}, \textsf{BURNING}\}` the
alight categories. :math:`\textsf{BURNT}` and :math:`\textsf{WET}` are
absorbing — no rule maps either back — which is what makes "no cell is alight"
a genuine terminal state rather than a moment that can be undone.

**State space**

.. math::

   s = \big(t,\; (y_j, z_j, \text{tank}_j, \text{hp}_j)_{j=1}^{M},\;
   (w_{\text{dir}}, w_{\text{str}}),\; \mathbf{f}\big)

.. math::

   S = \mathbb{Z}_{\geq 0} \times
   \big(\{0..H{-}1\} \times \{0..W{-}1\} \times \{0..\texttt{max\_tank}\}
   \times \{0..\texttt{max\_health}\}\big)^{M}
   \times \mathcal{W} \times \mathcal{C}^{HW}

where :math:`(y_j, z_j)` is robot :math:`j`'s cell, :math:`\text{tank}_j` its tank,
:math:`\text{hp}_j` its health, :math:`\mathbf{f}` the fire map, and

.. math::

   \mathcal{W} = \{\textsf{N},\textsf{E},\textsf{S},\textsf{W}\}
   \times \{\textsf{LOW}, \textsf{HIGH}\}, \qquad |\mathcal{W}| = 8

is the **hidden wind**, which is never observed and never changes.

**Action space.** Five per robot, issued jointly:

.. math::

   A = \{\textsf{N}, \textsf{E}, \textsf{S}, \textsf{W},
   \textsf{SUPPRESS}\}^{M}, \qquad |A| = 5^{M}

encoded base-5. At the default :math:`M = 2` that is 25; at three robots 125,
which is where a tree search starts to feel the branching.

**Transition model.** Six stages, in this order. The order is not cosmetic.

1. **Motion.** A move is admissible if the target is on the grid, not an
   obstacle, and not :math:`\textsf{BURNT}`, judged against the *pre-step*
   fire map. An admissible move slips with probability :math:`p_s` =
   ``slip_probability``:

   .. math::

      \Pr[(y'_j, z'_j) = \text{target}] = 1 - p_s, \qquad
      \Pr[(y'_j, z'_j) = (y_j, z_j)] = p_s

   A disabled robot (:math:`\text{hp}_j = 0`) and a suppressing robot do not move,
   and take no draw.

2. **Suppression.** Robot :math:`j` sprays iff it is live, chose
   :math:`\textsf{SUPPRESS}`, and :math:`\text{tank}_j > 0`; an empty tank does
   nothing and costs nothing. A spray covers its own cell and its four
   neighbours, and coverage **counts add** — two robots covering one cell each
   get an independent attempt. With :math:`k` covering sprays,

   .. math::

      \Pr[\text{cell soaked}] = 1 - (1 - q_{\mathbf{f}})^{k}

   where :math:`q_c` is ``suppression_probability_*`` for category :math:`c`
   (and :math:`0` for the absorbing ones). A soaked cell becomes
   :math:`\textsf{WET}`. Each sprayer then loses one tank unit; a robot
   standing on the depot refills to ``max_tank``, which overrides the cost.

3. **Spread.** A cell catches unless *every* alight neighbour fails to ignite
   it. Write :math:`\mathcal{N}^{\uparrow}(k)` for the four-neighbours of
   :math:`k` that are alight after suppression. The one sitting directly
   upwind under :math:`w_{\text{dir}}` gets the boosted rate; the other three the
   attenuated one:

   .. math::

      \Pr[\text{$k$ ignites}] = 1 - \prod_{n \in \mathcal{N}^{\uparrow}(k)}
      \big(1 - p_{\text{ign}}(n, k)\big)

   .. math::

      p_{\text{ign}}(n,k) = \begin{cases}
        \min(1,\, p_0 \cdot g(w_{\text{str}})) & k - n = \Delta_{w_{\text{dir}}} \\
        p_0 \,(1 - \texttt{crosswind\_attenuation}) & \text{otherwise}
      \end{cases}

   with :math:`p_0` = ``spread_probability`` and :math:`g` the gain,
   ``wind_gain_low`` or ``wind_gain_high``. An ignited cell becomes
   :math:`\textsf{SMOLDERING}`. Only unburnt, non-obstacle cells can catch.

4. **Growth and burnout,** applied to cells alight *before* the spread stage,
   which is what stops a cell igniting and growing to burning in one step:

   .. math::

      \Pr[\textsf{SMOLDERING} \to \textsf{BURNING}] &=
        \texttt{growth\_probability} \\
      \Pr[\textsf{BURNING} \to \textsf{BURNT}] &=
        \texttt{burnout\_probability}

5. **Heat damage,** read off the final map at each robot's final cell:
   :math:`\text{hp}'_j = \max(0, \text{hp}_j - \mathrm{dmg}(\mathbf{f}'_{k_j}))`, with
   :math:`\mathrm{dmg} = (0, 1, 2, 0, 0)` over :math:`\mathcal{C}`. At the
   default ``max_health`` of 3 a robot survives one burning step and is
   disabled by the second — which is what makes fighting from an *adjacent*
   cell the intended play.

6. **Bookkeeping.** :math:`t' = t + 1`, and the wind is copied unchanged.

Suppression resolving before spread is what lets a robot stop a front by
soaking the cell ahead of it in the same step. The wind never changing is what
makes it identifiable from the spread pattern across an episode.

**Observation space.** Exact robot fields, then one reported category per
cell, with :math:`\textsf{UNKNOWN} = -1` for a cell no live robot senses:

.. math::

   \Omega = \big(\{0..H{-}1\} \times \{0..W{-}1\} \times \{0..\texttt{max\_tank}\}
   \times \{0..\texttt{max\_health}\}\big)^{M}
   \times \big(\mathcal{C} \cup \{\textsf{UNKNOWN}\}\big)^{HW}

**Observation model.** Robot fields are reported exactly; the fire map is seen
only inside the union of the live robots' Chebyshev footprints:

.. math::

   V(s') = \bigcup_{j : \text{hp}'_j > 0}
   \{k : \lVert k - (y'_j, z'_j) \rVert_\infty \leq \texttt{sensing\_radius}\}

A disabled robot sees nothing, and two robots standing together see barely
more than one — spreading out is what buys information. The reading is

.. math::

   o = \big((y'_j, z'_j, \text{tank}'_j, \text{hp}'_j)_{j=1}^{M},\; \hat{\mathbf{f}}\big),
   \qquad
   \hat{f}_k = \begin{cases}
     \textsf{UNKNOWN} = -1 & k \notin V(s') \\
     f'_k & \text{w.p. } 1 - p_{\text{err}} \\
     \mathrm{Unif}(\mathcal{C} \setminus \{f'_k\}) & \text{w.p. } p_{\text{err}}
   \end{cases}

with :math:`p_{\text{err}}` = ``observation_error_probability``: a symmetric
confusion matrix spreading its error mass evenly over the four wrong
categories. The observation depends on :math:`s'` alone, not on the action.

Note what is **never** observed: the wind. It has to be inferred from how the
fire spreads, and because only :math:`w_{\text{str}}` sets the downwind gain, a
strong wind of unknown direction is a different inference problem from a weak
one — the belief over the eight wind values need not collapse to a point for a
planner to act well.

**Reward function.** Every term reads the realised successor:

.. math::

   R(s, a, s') = \;&-\texttt{step\_cost}
   \;-\; \texttt{smoldering\_cell\_cost} \cdot |\{k : f'_k = \textsf{SMOLDERING}\}| \\
   &-\; \texttt{burning\_cell\_cost} \cdot |\{k : f'_k = \textsf{BURNING}\}| \\
   &-\; \texttt{burnt\_cell\_cost} \cdot
     |\{k : f'_k = \textsf{BURNT},\, f_k \neq \textsf{BURNT}\}| \\
   &-\; \texttt{damage\_cost} \cdot \textstyle\sum_j (\text{hp}_j - \text{hp}'_j)
   \;-\; \texttt{water\_cost} \cdot |\text{sprayers}| \\
   &+\; \texttt{success\_reward} \cdot
     \mathbb{1}\big[\{k : f'_k \in \mathcal{A}\ell\} = \emptyset\big]

The success bonus is paid on the transition *into* a fire-free state, and the
step cost is charged on that transition too — which is why the largest
reachable reward is :math:`\texttt{success\_reward} - \texttt{step\_cost}`.

**Initial belief.** Robots at known posts with full tank and health; the wind
uniform over its eight values; ``num_initial_fires`` cells drawn uniformly
without replacement from the non-obstacle cells and set to
:math:`\textsf{BURNING}`:

.. math::

   b_0 = \mathbb{1}\big[\text{robots} = \text{robots}_0\big] \otimes
   \mathrm{Unif}(\mathcal{W}) \otimes
   \mathrm{Unif}\big(\text{$n_0$-subsets of } \overline{\mathcal{O}}\big)

The opening observation is a sentinel — known robot fields, every cell
:math:`\textsf{UNKNOWN}` — not a scan; the first real reading arrives with the
first transition.

**Discount.** :math:`\gamma` = ``discount_factor``, default :math:`0.95`.

**Terminal set.**

.. math::

   S_T = \{s : \{k : f_k \in \mathcal{A}\ell\} = \emptyset\}
   \;\cup\; \{s : \forall j,\, \text{hp}_j = 0\}
   \;\cup\; \{s : t \geq \texttt{max\_steps}\}

the middle set only when ``is_all_robots_disabled_terminal``.

World and state
---------------

Every cell holds one of five categories: ``UNBURNT`` (has fuel), ``SMOLDERING``
(intensity 1), ``BURNING`` (intensity 2), ``BURNT`` (fuel consumed) and ``WET``
(soaked, cannot reignite). ``BURNT`` and ``WET`` are absorbing, and no rule maps
either back into an alight category. That is what makes "no cell is alight"
a genuine terminal state rather than a moment that can be undone, and therefore
what makes the completion metric mean anything.

The hidden wind is a direction in ``{N, E, S, W}`` crossed with a strength in
``{low, high}`` -- eight values, drawn uniformly at reset and constant for the
episode. It is never observed.

The state is one ``float64`` vector of length ``3 + 4N + R*C``::

    [step, (row, col, tank, health) x N, wind_direction, wind_strength, cells]

A robot's tank holds ``max_tank`` sprays (6 by default) and its health starts at
``max_health`` (3 by default). At zero health a robot is disabled: its digit of
the joint action is ignored and it sees nothing for the rest of the episode.

Actions
-------

Each robot has five actions -- ``NORTH``, ``EAST``, ``SOUTH``, ``WEST`` and
``SUPPRESS``. The environment receives one centralized joint action: a single
integer whose base-5 digits are the per-robot actions, least significant digit
first. Two robots give 25 joint actions, three give 125.

A move is refused by the grid edge, by an obstacle and by a ``BURNT`` cell, and
an otherwise admissible move fails with ``slip_probability`` (0.05 by default).
``SUPPRESS`` costs one tank unit and covers the robot's own cell *and its four
neighbours* -- which is the point of the design: a careful planner fights from
an adjacent cell and takes no damage, a careless one stands in the fire. An
``UNBURNT`` target becomes ``WET``, which is how a firebreak is built ahead of
the front. A robot with an empty tank does nothing and pays nothing. Entering
the depot refills the tank, with no separate action, so ``|A|`` stays at
``5 ** N``; a robot that sprays and steps into the depot on the same step ends
the step full.

Transition
----------

Six stages resolve in a fixed order, and the order is not cosmetic.

1. **Motion.** Admissible moves succeed with ``1 - slip_probability``.
2. **Suppression.** A cell covered by ``m`` sprays becomes ``WET`` with
   ``1 - (1 - q)^m``, where ``q`` depends on the cell's category: 1.0 unburnt,
   0.9 smoldering, 0.6 burning, 0 for burnt and wet. Burning resists one hose,
   which is why two robots on one cell are worth more than two robots dividing
   the work -- this is the only place in the model where they genuinely
   cooperate. Resolving suppression *before* spread is what lets a robot stop a
   front by soaking the cell ahead of it in the same step.
3. **Spread.** An unburnt cell catches unless every alight four-neighbour fails
   to ignite it. The one neighbour sitting directly upwind ignites it at
   ``min(1, spread_probability * gain)``; the other three at
   ``spread_probability * (1 - crosswind_attenuation)``. The gain is
   ``wind_gain_low`` (2.0) or ``wind_gain_high`` (3.5). That asymmetry is the
   whole reason the wind is identifiable: with no asymmetry the hidden wind
   would be unobservable noise rather than something to infer.
4. **Growth and burnout**, applied only to cells already alight *before* the
   spread stage, so a cell cannot ignite and reach full intensity in one step.
   Smoldering grows to burning with ``growth_probability``; burning burns out to
   ``BURNT`` with ``burnout_probability``.
5. **Heat damage.** A robot loses 1 health on a smoldering cell and 2 on a
   burning one. With the default health of 3 it survives one burning step and is
   disabled by the second.
6. **Bookkeeping.** The step counter advances and the wind is copied unchanged.

Observation
-----------

An observation is one ``float64`` vector of length ``4N + R*C``::

    [(row, col, tank, health) x N, reported category per cell]

Poses, tanks and healths are exact. Every cell within Chebyshev radius
``sensing_radius`` (2 by default) of any *live* robot reports its true category
with probability ``1 - observation_error_probability`` and one of the other four
uniformly otherwise. Every other cell reports ``-1``, the unknown marker, so the
observation has a fixed shape whatever the robots do. Two robots standing
together see barely more than one, so spreading out is what buys information.

The wind never enters the observation. It is observable only through its effect
on the fire, which is what makes this an inference problem rather than a lookup.
Both the transition and the observation density are available in closed form:
``transition_log_probability`` is exact, because the intermediate map is
recoverable from the pair of maps -- wet is reachable only by suppression, burnt
only by burnout, and a freshly ignited cell cannot also grow.

Reward and termination
----------------------

Every term reads the realised successor, so
``reward_requires_next_state`` is ``True``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Term
     - Value
   * - step cost
     - ``-step_cost`` (0.1) on every transition, the terminating one included
   * - active fire
     - ``-(0.5 * smoldering + 1.0 * burning)`` cells in the successor
   * - property loss
     - ``-5.0`` per cell newly burnt this step
   * - robot damage
     - ``-10.0`` per point of health lost, deliberately above the cell cost so
       a planner does not trade a robot for a cell
   * - suppressant
     - ``-0.1`` per robot that actually sprayed
   * - success
     - ``+100.0`` on the transition into a fire-free state

The declared reward range is computed from the constructor's own coefficients,
grid size and robot count -- never a constant. The **maximum** is
``success_reward - step_cost``, not ``success_reward``, because the step cost is
charged on the terminating transition too. The **minimum** is::

    -(step_cost
      + max(smoldering_cell_cost, burning_cell_cost, burnt_cell_cost) * R * C
      + damage_cost * N * min(max_health, 2)
      + water_cost * N)

A cell in the successor is smoldering, or burning, or newly burnt, or none of
those -- never two at once -- so those three terms share one budget of ``R * C``
cells rather than stacking, which is what allows the ``max`` instead of a sum.
With the defaults that is ``(-540.3, 99.9)``.
``is_all_robots_disabled_terminal`` changes only termination, so it moves
neither end of the bound.

Termination is evaluated in this order: **goal** (no cell alight), **failure**
(every robot disabled while fire is still active, when
``is_all_robots_disabled_terminal``), then **timeout** (``max_steps``, 100 by
default). Goal wins over failure, so a fire put out by robots that then burned
out is still a success.

Key settings
------------

The default world is 10 by 10 with two robots, one small obstacle blob just
past the middle, and a depot in the north-west corner.

``spread_probability`` defaults to **0.10** and ``burnout_probability`` to
**0.03**, and the pair was measured rather than proposed. At the originally
drafted 0.06 and 0.12 an *unattended* fire on the default world went out by
itself in 93% of episodes: a planner that did nothing would have "completed the
task" nine times in ten, and no margin over a random baseline would have been
measurable. At 0.10 and 0.03 the same unattended fire goes out 6% of the time, a
uniformly random policy 27%, and a hand-written greedy firefighter 89%, so the
completion rate reports what the planner did.

Minimal example
---------------

.. code-block:: python

   from POMDPPlanners.environments.multiagent_firefighting_pomdp import (
       MultiAgentFirefightingPOMDP,
   )
   from POMDPPlanners.utils.belief_factory import create_environment_belief

   env = MultiAgentFirefightingPOMDP()
   belief = create_environment_belief(env, n_particles=100)

Metrics
-------

``task_completion_rate`` reports a fire-free map, reduced with ``ANY``: wet and
burnt are absorbing, so a fire-free map cannot be undone and ``ANY`` and
``LAST`` agree. ``ended_by_goal``, ``ended_by_failure`` and ``ended_by_timeout``
report how each episode ended and sum to one. ``average_episode_length`` is a
constant channel summed.

The danger is reported both as a count -- ``robot_steps_in_fire``,
``robot_health_lost`` -- and as a severity --
``max_simultaneous_alight_cells``, ``max_burnt_cell_fraction``. A planner that
lets the fire reach forty cells and then beats it out is not the same as one
that never let it past five, and the totals alone would not distinguish them.
``suppressant_units_used`` and ``robots_disabled_at_end`` round out the picture.

Visualization
-------------

Runs also write a GIF of each episode through ``cache_visualization``. Its left
panel is the truth: the five categories, the obstacles, the depot, each live
robot with its health and tank badge and its sensing footprint, and the true
wind printed and labelled hidden. The middle panel is the per-cell probability
that the cell is alight, as the weighted mean over the belief's particles --
the exact marginal, not a summary. The right panel is the total particle weight
on each of the eight wind values, with the true one marked in red. A
firefighting run is unreadable without that last panel: you cannot tell a
planner that inferred the wind from one that guessed.

Filtering and limits
--------------------

There is **no torch vectorized model and no C++ native model**, so VOPP is
unsupported and the environment is deliberately absent from the vectorized
config contract. ``PFT_DPW`` takes the scalar API directly.

The belief is ``FirefightingVectorizedBelief``, which
``create_environment_belief`` returns. It runs every stage of the transition
over the particle axis and the grid at once, and it does two things a plain
bootstrap filter does not, both because a bootstrap filter over whole 100-cell
maps is degenerate here. The poses, tanks and healths come back from the sensor
exactly but depend on hidden state, so weighting by them puts every weight on
the floor and both belief panels render as noise; this belief writes the
reported values onto the particles instead. And the wind never changes, so
resampling across wind values deletes hypotheses no later evidence can restore;
resampling therefore happens inside a wind value.

That does not make the filter free of the usual cautions. Use enough particles
and inspect effective sample size rather than trusting the wind histogram on
sight. The wind *is* identifiable from the spread -- an exact posterior over the
eight values, given the map, puts most of its mass on the true wind within about
twenty-five steps -- but with random actions this filter is slower than that: in
a six-episode check it held about 0.4 of its weight on the true wind after
thirty steps, against a prior of 0.125. A flat histogram late in an episode is a
statement about the filter, not about the environment.

See also
--------

- :class:`POMDPPlanners.environments.multiagent_firefighting_pomdp.MultiAgentFirefightingPOMDP`
- :doc:`index` — the full catalog.
