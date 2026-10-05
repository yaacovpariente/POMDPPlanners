Firefighting
============

.. episode-viewer:: traces/firefighting.json

   One recorded episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

``FirefightingPOMDP`` puts ``N`` firefighting robots on an ``R x C``
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

The environment is the POMDP :math:`\langle S, A, Z, T, O, R, b_0, \gamma
\rangle`.

**State space.** The grid has :math:`H \times W` cells, indexed :math:`k` or
:math:`(y, z)` = (row, column), with obstacle cells :math:`\mathcal{O}`
(``obstacle_cells``); there are :math:`M` = ``num_robots`` robots. Each cell
holds one category

.. math::

   \mathcal{C} = \{\textsf{UNBURNT}, \textsf{SMOLDERING}, \textsf{BURNING},
   \textsf{BURNT}, \textsf{WET}\} = \{0,1,2,3,4\}

— fuel left, alight weakly, alight strongly, fuel consumed, soaked. A state is

.. math::

   s = \big(t,\; (y_j, z_j, \text{tank}_j, \text{hp}_j)_{j=1}^{M},\;
   (w_{\text{dir}}, w_{\text{str}}),\; \mathbf{f}\big)

.. math::

   S = \mathbb{Z}_{\geq 0} \times
   \big(\{0..H{-}1\} \times \{0..W{-}1\} \times \{0..\texttt{max\_tank}\}
   \times \{0..\texttt{max\_health}\}\big)^{M}
   \times \mathcal{W} \times \mathcal{C}^{HW}

where :math:`t` is the step count, :math:`(y_j, z_j)` is robot :math:`j`'s
cell, :math:`\text{tank}_j` the sprays left in its tank, :math:`\text{hp}_j`
its health, :math:`\mathbf{f} = (f_k)_k` the category of every cell, and

.. math::

   \mathcal{W} = \{\textsf{N},\textsf{E},\textsf{S},\textsf{W}\}
   \times \{\textsf{LOW}, \textsf{HIGH}\}, \qquad |\mathcal{W}| = 8

is the **hidden wind**: :math:`w_{\text{dir}}` is the direction it blows
toward and :math:`w_{\text{str}}` its strength.

**Action space.** Five per robot, issued jointly:

.. math::

   A = \{\textsf{N}, \textsf{E}, \textsf{S}, \textsf{W},
   \textsf{SUPPRESS}\}^{M}, \qquad |A| = 5^{M}

encoded as :math:`a = \sum_{j=1}^{M} a_j 5^{j-1}`, where robot :math:`j`'s
action :math:`a_j` is

- :math:`0` N — move one row up, :math:`y - 1`;
- :math:`1` E — move one column right, :math:`z + 1`;
- :math:`2` S — move one row down, :math:`y + 1`;
- :math:`3` W — move one column left, :math:`z - 1`;
- :math:`4` SUPPRESS — stay, and spray the robot's own cell and its four
  neighbours.

**Observation space.** For every robot :math:`j` its cell
:math:`(y_j, z_j)`, tank and health, exact; then for every cell a reported
category, or :math:`\textsf{UNKNOWN} = -1` for a cell no live robot
(:math:`\text{hp}_j > 0`) senses:

.. math::

   Z = \big(\{0..H{-}1\} \times \{0..W{-}1\} \times \{0..\texttt{max\_tank}\}
   \times \{0..\texttt{max\_health}\}\big)^{M}
   \times \big(\mathcal{C} \cup \{\textsf{UNKNOWN}\}\big)^{HW}

**Transition model.** A robot is *live* while :math:`\text{hp}_j > 0`; a cell
is *alight* if its category is in :math:`\mathcal{A}\ell = \{\textsf{SMOLDERING},
\textsf{BURNING}\}`; :math:`\Delta_d` is the one-cell offset of direction
:math:`d` (N :math:`(-1, 0)`, E :math:`(0, 1)`, S :math:`(1, 0)`, W
:math:`(0, -1)`); the depot is ``depot_cell``. One step resolves in this
order:

1. **Motion.** A move is admissible if the target is on the grid, not an
   obstacle, and not :math:`\textsf{BURNT}`, judged against the *pre-step*
   fire map. An admissible move slips with probability :math:`p_s` =
   ``slip_probability`` (default 0.05):

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

      \Pr[\text{cell soaked}] = 1 - (1 - q_{f})^{k}

   where :math:`f` is the cell's category and :math:`q_f` is
   ``suppression_probability_unburnt`` (default 1.0),
   ``suppression_probability_smoldering`` (0.9) or
   ``suppression_probability_burning`` (0.6), and :math:`0` for
   :math:`\textsf{BURNT}` and :math:`\textsf{WET}`. A soaked cell becomes
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
   ``wind_gain_low`` (default 2.0) or ``wind_gain_high`` (default 3.5). An ignited cell becomes
   :math:`\textsf{SMOLDERING}`. Only unburnt, non-obstacle cells can catch.

4. **Growth and burnout,** applied to cells alight *before* the spread, so a
   cell cannot ignite and grow to burning in one step:

   .. math::

      \Pr[\textsf{SMOLDERING} \to \textsf{BURNING}] &=
        \texttt{growth\_probability} \\
      \Pr[\textsf{BURNING} \to \textsf{BURNT}] &=
        \texttt{burnout\_probability}

5. **Heat damage,** read off the final map at each robot's final cell:
   :math:`\text{hp}'_j = \max(0, \text{hp}_j - \mathrm{dmg}(f'_{k_j}))`, with
   :math:`k_j` robot :math:`j`'s cell and :math:`\mathrm{dmg} = (0, 1, 2, 0, 0)`
   over :math:`\mathcal{C}`.

6. **Step count and wind.** :math:`t' = t + 1`, and the wind is copied
   unchanged.

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

where :math:`\mathcal{A}\ell = \{\textsf{SMOLDERING}, \textsf{BURNING}\}` and
the sprayers are the live robots that chose :math:`\textsf{SUPPRESS}` with a
non-empty tank. The success bonus is paid on the transition *into* a fire-free state, and the
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

with :math:`\text{robots}_0` the start cells (``robot_start_cells``) at full
tank and health, :math:`n_0` = ``num_initial_fires`` and
:math:`\overline{\mathcal{O}}` the non-obstacle cells.

The opening observation is a sentinel — known robot fields, every cell
:math:`\textsf{UNKNOWN}` — not a scan; the first sensor reading arrives with the
first transition.

**Discount.** :math:`\gamma` = ``discount_factor``, default :math:`0.95`.

**Terminal set.**

.. math::

   S_T = \{s : \{k : f_k \in \mathcal{A}\ell\} = \emptyset\}
   \;\cup\; \{s : \forall j,\, \text{hp}_j = 0\}
   \;\cup\; \{s : t \geq \texttt{max\_steps}\}

with :math:`\mathcal{A}\ell = \{\textsf{SMOLDERING}, \textsf{BURNING}\}`: every
fire out, every robot disabled (only when ``is_all_robots_disabled_terminal``),
or the step limit reached.

World and state
~~~~~~~~~~~~~~~

Every cell holds one of five categories: ``UNBURNT`` (has fuel), ``SMOLDERING``
(intensity 1), ``BURNING`` (intensity 2), ``BURNT`` (fuel consumed) and ``WET``
(soaked, cannot reignite). ``BURNT`` and ``WET`` are absorbing, and no rule maps
either back into an alight category. So once no cell is alight, no later step
can relight one, and a completed episode stays completed.

The hidden wind is a direction in ``{N, E, S, W}`` crossed with a strength in
``{low, high}`` -- eight values, drawn uniformly at reset and constant for the
episode. It is never observed.

The state is one ``float64`` vector of length ``3 + 4N + R*C``::

    [step, (row, col, tank, health) x N, wind_direction, wind_strength, cells]

A robot's tank holds ``max_tank`` sprays (6 by default) and its health starts at
``max_health`` (3 by default). At zero health a robot is disabled: its digit of
the joint action is ignored and it sees nothing for the rest of the episode.

Observation
~~~~~~~~~~~

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
only by burnout, and a cell ignited this step cannot also grow.

Rewards
-------

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
     - ``-0.1`` per robot that sprayed (an empty tank does not)
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

The default world is 10 by 10 with two robots, a 2 by 2 obstacle block at
rows 5–6 and columns 5–6, and a depot in the north-west corner.

``spread_probability`` defaults to **0.10** and ``burnout_probability`` to
**0.03**, and the pair was measured rather than proposed. At the originally
drafted 0.06 and 0.12 an *unattended* fire on the default world went out by
itself in 93% of episodes: a planner that did nothing would have "completed the
task" nine times in ten, and no margin over a random baseline would have been
measurable. At 0.10 and 0.03 the same unattended fire goes out 6% of the time, a
uniformly random policy 27%, and a hand-written greedy firefighter 89%, so the
completion rate reports what the planner did.

Metrics
~~~~~~~

``task_completion_rate`` reports a fire-free map, reduced with ``ANY``: wet and
burnt are absorbing, so a fire-free map cannot be undone and ``ANY`` and
``LAST`` agree. ``ended_by_goal_rate``, ``ended_by_failure_rate`` and ``ended_by_timeout_rate``
report how each episode ended and sum to one. ``average_episode_length`` is a
constant channel summed.

The danger is reported both as a count -- ``average_robot_steps_in_fire``,
``average_robot_health_lost`` -- and as a severity --
``max_simultaneous_alight_cells``, ``max_burnt_cell_fraction``. A planner that
lets the fire reach forty cells and then beats it out is not the same as one
that never let it past five, and the totals alone would not distinguish them.
``average_suppressant_units_used`` and ``final_robots_disabled`` are also reported.

Visualization
~~~~~~~~~~~~~

Runs write a trace of each episode through the environment's episode
visualizer. The results site replays it in 3D, as the replay on this page does.

Filtering and limits
~~~~~~~~~~~~~~~~~~~~

There is **no torch vectorized model and no C++ native model**, so VOPP is
unsupported and the environment is deliberately absent from the vectorized
config contract. ``PFT_DPW`` takes the scalar API directly.

The belief is ``FirefightingVectorizedBelief``, which
``create_environment_belief`` returns. It runs every part of the transition
over the particle axis and the grid at once, and it does two things a
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

Can I use?
----------

.. list-table::
   :header-rows: 1
   :widths: 34 30

   * - Capability
     - ``FirefightingPOMDP``
   * - Action space
     - Discrete
   * - Observation space
     - Discrete
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

   from POMDPPlanners.environments.firefighting_pomdp import (
       FirefightingPOMDP,
   )
   from POMDPPlanners.utils.belief_factory import create_environment_belief

   env = FirefightingPOMDP()
   belief = create_environment_belief(env, n_particles=100)

Parameters
----------

.. autoclass:: POMDPPlanners.environments.firefighting_pomdp.FirefightingPOMDP
   :members:
   :show-inheritance:

See also
--------

- :class:`POMDPPlanners.environments.firefighting_pomdp.FirefightingPOMDP`
- :doc:`base` — the full catalog and the environment interface.
