Firefighting
============

.. episode-viewer:: traces/firefighting.json

   One recorded episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

``FirefightingPOMDP`` puts ``N`` firefighters on an ``R x C``
grid and asks them to put out a fire that spreads under a **hidden, constant
wind**. The firefighters see the fire only near themselves, carry a finite tank of
suppressant, and take heat damage for standing in flames. The task is complete
when no cell is burning.

The wind is never observed. It has to be inferred from how the fire spreads,
while the firefighters decide where to look, where to spray, and when to go back to
the depot to refill. Use it to test planning over a large joint action space
with a belief over a whole map.

What the agent sees and does
----------------------------

- **State** — one ``float64`` vector of length ``3 + 4N + R*C``: the step
  counter, ``(row, col, tank, health)`` per firefighter, the wind direction and
  strength, then one category per cell (``0`` unburnt to ``4`` wet). Length
  111 at the defaults.
- **Actions** (discrete) — one ``int`` in ``[0, 5 ** N)``. Its base-5 digits,
  least significant first, are the per-firefighter actions ``NORTH`` (0), ``EAST``
  (1), ``SOUTH`` (2), ``WEST`` (3), ``SUPPRESS`` (4). 25 actions at the default
  two firefighters.
- **Observations** (discrete) — one ``float64`` vector of length
  ``4N + R*C``: exact ``(row, col, tank, health)`` per firefighter, then a noisy
  category per cell within ``sensing_radius`` of a live firefighter and ``-1``
  (unknown) everywhere else. The wind never appears.

Formal definition
-----------------

The environment is the POMDP :math:`\langle S, A, Z, T, O, R, b_0, \gamma
\rangle`. Names in typewriter font are constructor arguments of
``FirefightingPOMDP``; :math:`\{0..n\}` is the integers :math:`0` to
:math:`n`.

**State space.** The notation:

- **Grid.** :math:`H` = ``num_rows`` rows and :math:`W` = ``num_cols``
  columns. A cell is :math:`k = (y, z)`, row :math:`y` (0 at the top) and
  column :math:`z` (0 at the left). The cells are :math:`K = \{0..H{-}1\}
  \times \{0..W{-}1\}`, in row-major order.
- **Obstacles.** :math:`\mathcal{B} \subseteq K` = ``obstacle_cells``; the
  free cells are :math:`\overline{\mathcal{B}} = K \setminus \mathcal{B}`.
- **Firefighters.** :math:`M` = ``num_firefighters``, indexed :math:`j =
  1..M`.
- **Categories.** :math:`\mathcal{C} = \{\textsf{UNBURNT},
  \textsf{SMOLDERING}, \textsf{BURNING}, \textsf{BURNT}, \textsf{WET}\}`,
  coded :math:`0..4` in that order.
- **Directions.** :math:`D = \{\textsf{N}, \textsf{E}, \textsf{S},
  \textsf{W}\}`, coded :math:`0..3` in that order: :math:`\textsf{N}` is
  toward row 0, :math:`\textsf{E}` toward higher columns, :math:`\textsf{S}`
  toward higher rows, :math:`\textsf{W}` toward column 0.

A state holds:

- :math:`t`, the step count.
- For each firefighter :math:`j`: its cell :math:`p_j \in
  \overline{\mathcal{B}}`, its sprays left :math:`\text{tank}_j`, and its
  health :math:`\text{hp}_j`.
- The hidden wind: its direction :math:`w_{\text{dir}} \in D`, the way it
  blows toward, and its strength :math:`w_{\text{str}} \in \{\textsf{LOW},
  \textsf{HIGH}\}`. The wind lies in :math:`\Theta = D \times \{\textsf{LOW},
  \textsf{HIGH}\}`.
- :math:`f_k \in \mathcal{C}`, the category of each cell :math:`k`.

.. math::

   s = \big(t,\; (p_j, \text{tank}_j, \text{hp}_j)_{j=1}^{M},\;
   (w_{\text{dir}}, w_{\text{str}}),\; (f_k)_{k \in K}\big)

.. math::

   S = \{0..\texttt{max\_steps}\} \times
   \big(\overline{\mathcal{B}} \times \{0..\texttt{max\_tank}\}
   \times \{0..\texttt{max\_health}\}\big)^{M}
   \times \Theta \times \mathcal{C}^{K}

**Action space.** There are :math:`M` = ``num_firefighters`` firefighters.
A joint action :math:`a` packs one digit :math:`a_j \in \{0..4\}` per
firefighter :math:`j`, least significant first:

.. math::

   A = \{0..5^{M}{-}1\}, \qquad a = \textstyle\sum_{j=1}^{M} a_j\, 5^{j-1},
   \qquad a_j = \lfloor a / 5^{j-1} \rfloor \bmod 5

Firefighter :math:`j` is *live* while its health :math:`\text{hp}_j > 0`, and
*disabled* once :math:`\text{hp}_j = 0`. For
a live firefighter at cell :math:`(y, z)` (row :math:`y` from the top, column
:math:`z` from the left): :math:`a_j = 0` (north) moves to :math:`(y{-}1, z)`,
:math:`1` (east) to :math:`(y, z{+}1)`, :math:`2` (south) to :math:`(y{+}1,
z)`, :math:`3` (west) to :math:`(y, z{-}1)`; :math:`4`
(:math:`\textsf{SUPPRESS}`) stays and, if it has sprays left
(:math:`\text{tank}_j > 0`), sprays its own cell and the up to four
neighbouring cells one step north, east, south and west of it that lie on the
grid of ``num_rows`` by ``num_cols`` cells. A disabled firefighter's digit is ignored.

**Observation space.** An observation lists, for each of the :math:`M` =
``num_firefighters`` firefighters :math:`j`, its cell :math:`p_j`, its sprays
left :math:`\text{tank}_j` and its health :math:`\text{hp}_j`, where a cell is
a pair :math:`(y, z)`, row :math:`y` from the top and column :math:`z` from the
left; then, for every
cell :math:`k` of the grid :math:`K = \{0..H{-}1\} \times \{0..W{-}1\}`
(:math:`H` = ``num_rows``, :math:`W` = ``num_cols``), a reported category
:math:`\hat f_k`:

.. math::

   o = \big((p_j, \text{tank}_j, \text{hp}_j)_{j=1}^{M},\;
   (\hat f_k)_{k \in K}\big)

.. math::

   Z = \big(\overline{\mathcal{B}} \times \{0..\texttt{max\_tank}\}
   \times \{0..\texttt{max\_health}\}\big)^{M}
   \times \big(\mathcal{C} \cup \{\textsf{UNKNOWN}\}\big)^{K}

Here :math:`\overline{\mathcal{B}}` is the cells of :math:`K` not in
``obstacle_cells`` and :math:`\mathcal{C} = \{\textsf{UNBURNT}, \textsf{SMOLDERING},
\textsf{BURNING}, \textsf{BURNT}, \textsf{WET}\}`, coded :math:`0..4`. The
firefighter fields are exact for every firefighter, including one whose health
is zero.
:math:`\hat f_k` is a reported category, possibly
different from the cell's true category, for a cell within Chebyshev distance ``sensing_radius`` of the
cell :math:`p_j` of a firefighter with health above zero, and :math:`\textsf{UNKNOWN}` (coded
:math:`-1`) for every other cell; :math:`\textsf{UNKNOWN}` is not in
:math:`\mathcal{C}`. The step count and the wind are not observed.

**Transition model.** :math:`T` maps the state :math:`s` and the joint action
:math:`a` to the successor :math:`s'`; a prime marks a component of
:math:`s'`. The components of :math:`s` are the step count :math:`t`; for
each of the :math:`M` = ``num_firefighters`` firefighters :math:`j = 1..M`,
its cell :math:`p_j`, sprays left :math:`\text{tank}_j` and health
:math:`\text{hp}_j`; the hidden wind's direction :math:`w_{\text{dir}} \in D`
and strength :math:`w_{\text{str}} \in \{\textsf{LOW}, \textsf{HIGH}\}`; and
each cell :math:`k`'s category :math:`f_k \in \mathcal{C} =
\{\textsf{UNBURNT}, \textsf{SMOLDERING}, \textsf{BURNING}, \textsf{BURNT},
\textsf{WET}\}`. Firefighter :math:`j`'s action is the digit
:math:`a_j = \lfloor a / 5^{j-1} \rfloor \bmod 5` of the joint action :math:`a
\in A = \{0..5^M{-}1\}`, and the firefighter is
*live* while :math:`\text{hp}_j > 0`. A cell is :math:`k = (y, z)`, row
:math:`y` from the top and column :math:`z` from the left, in the grid
:math:`K = \{0..H{-}1\} \times \{0..W{-}1\}` (:math:`H` = ``num_rows``,
:math:`W` = ``num_cols``); :math:`\mathcal{B}` = ``obstacle_cells`` and
:math:`\overline{\mathcal{B}} = K \setminus \mathcal{B}`. The directions
:math:`D = \{\textsf{N}, \textsf{E}, \textsf{S}, \textsf{W}\}`, coded
:math:`0..3` in that order, have offsets :math:`\Delta_{\textsf{N}} = (-1, 0)`,
:math:`\Delta_{\textsf{E}} = (0, 1)`, :math:`\Delta_{\textsf{S}} = (1, 0)`,
:math:`\Delta_{\textsf{W}} = (0, -1)`, added to or subtracted from a cell
:math:`(y, z)` componentwise, and :math:`\mathcal{N}(k) = \{k +
\Delta_d : d \in D\} \cap K` are the neighbours of :math:`k`. One step, in
this order; all draws are independent.

1. **Motion.** For :math:`a_j \in \{0..3\}` let :math:`u_j = p_j +
   \Delta_{a_j}`. If firefighter :math:`j` is live, :math:`u_j \in
   \overline{\mathcal{B}}` and :math:`f_{u_j} \neq \textsf{BURNT}`, then
   :math:`p'_j = u_j` w.p. :math:`1 - \sigma` and :math:`p'_j = p_j` w.p.
   :math:`\sigma`, with :math:`\sigma` = ``slip_probability``. Otherwise
   :math:`p'_j = p_j`.

2. **Suppression.** Action digit :math:`4` is :math:`\textsf{SUPPRESS}`. The
   sprayers are :math:`J = \{j : \text{hp}_j > 0,\; a_j = 4,\; \text{tank}_j
   > 0\}`, and :math:`c_k` is the number of sprayers with :math:`k \in
   \{p'_j\} \cup \mathcal{N}(p'_j)`. Cell :math:`k` turns
   :math:`\textsf{WET}`, giving the map :math:`\mathbf{f}^{\text{sup}}`, w.p.

   .. math::

      1 - \big(1 - q(f_k)\big)^{c_k}

   with :math:`q(\textsf{UNBURNT})`, :math:`q(\textsf{SMOLDERING})`,
   :math:`q(\textsf{BURNING})` = ``suppression_probability_unburnt``,
   ``_smoldering``, ``_burning``, and :math:`q(\textsf{BURNT}) =
   q(\textsf{WET}) = 0`. Then, with :math:`k_{\text{dep}}` = ``depot_cell``,
   :math:`\text{tank}'_j = \texttt{max\_tank}` if :math:`p'_j =
   k_{\text{dep}}`, else :math:`\text{tank}_j - \mathbb{1}[j \in J]`.

3. **Spread.** A cell is *alight* if its category is in
   :math:`\mathcal{A}\ell = \{\textsf{SMOLDERING}, \textsf{BURNING}\}`. A
   cell :math:`k \notin \mathcal{B}` with :math:`f^{\text{sup}}_k =
   \textsf{UNBURNT}` becomes :math:`\textsf{SMOLDERING}` w.p.

   .. math::

      1 - \prod_{\nu \in \mathcal{N}(k),\; f^{\text{sup}}_\nu \in \mathcal{A}\ell}
      \big(1 - p_{\text{ign}}(\nu, k)\big), \qquad
      p_{\text{ign}}(\nu, k) = \begin{cases}
        \min\!\big(1,\; \lambda\, g(w_{\text{str}})\big) & k - \nu = \Delta_{w_{\text{dir}}} \\
        \lambda\, (1 - \texttt{crosswind\_attenuation}) & \text{otherwise}
      \end{cases}

   with :math:`\lambda` = ``spread_probability``, :math:`g(\textsf{LOW})` =
   ``wind_gain_low``, :math:`g(\textsf{HIGH})` = ``wind_gain_high``. The case
   :math:`k - \nu = \Delta_{w_{\text{dir}}}` is the neighbour :math:`\nu`
   directly upwind of :math:`k`.

4. **Growth and burnout.** A cell with :math:`f^{\text{sup}}_k =
   \textsf{SMOLDERING}` becomes :math:`\textsf{BURNING}` w.p.
   ``growth_probability``; one with :math:`f^{\text{sup}}_k =
   \textsf{BURNING}` becomes :math:`\textsf{BURNT}` w.p.
   ``burnout_probability``. Every other cell keeps its category from the
   spread, giving :math:`\mathbf{f}'`.

5. **Heat damage.** :math:`\text{hp}'_j = \max\big(0,\; \text{hp}_j -
   \mathrm{dmg}(f'_{p'_j})\big)`, with :math:`\mathrm{dmg}(\textsf{SMOLDERING})
   = 1`, :math:`\mathrm{dmg}(\textsf{BURNING}) = 2`, and :math:`0` otherwise.

6. **Clock and wind.** :math:`t' = t + 1`; the wind is unchanged.

**Observation model.** :math:`O` gives the observation from the successor
state :math:`s'`. In :math:`s'`, firefighter :math:`j \in \{1..M\}`, of the
:math:`M` = ``num_firefighters``, is at cell :math:`p'_j = (y'_j, z'_j)` with
sprays left :math:`\text{tank}'_j` and health :math:`\text{hp}'_j`, and cell
:math:`k = (y, z)` of the grid :math:`K` (row :math:`y`, column :math:`z`)
has category :math:`f'_k \in \mathcal{C} = \{\textsf{UNBURNT},
\textsf{SMOLDERING}, \textsf{BURNING}, \textsf{BURNT}, \textsf{WET}\}`. With
:math:`\rho` = ``sensing_radius`` and :math:`p_{\text{err}}` =
``observation_error_probability``, the visible cells are those within
Chebyshev distance :math:`\rho` of a firefighter whose health is above zero:

.. math::

   V(s') = \bigcup_{j :\, \text{hp}'_j > 0}
   \big\{k \in K : \max(|y - y'_j|,\, |z - z'_j|) \leq \rho\big\}

The observation is :math:`o = \big((p'_j, \text{tank}'_j,
\text{hp}'_j)_{j=1}^{M},\, (\hat f_k)_{k \in K}\big)`: the firefighter fields
are copied from :math:`s'`, and each cell is reported independently, a wrong
report being one of the four categories of :math:`\mathcal{C}` other than the
true one. :math:`\textsf{UNKNOWN}` (coded :math:`-1`) is not in
:math:`\mathcal{C}`:

.. math::

   \hat f_k = \begin{cases}
     \textsf{UNKNOWN} & k \notin V(s') \\
     f'_k & \text{w.p. } 1 - p_{\text{err}}, \quad k \in V(s') \\
     c \in \mathcal{C} \setminus \{f'_k\} & \text{w.p. } p_{\text{err}}/4 \text{ each}, \quad k \in V(s')
   \end{cases}

**Reward function.** The reward reads the successor :math:`s'`
(``reward_requires_next_state`` is ``True``). :math:`k` ranges over the grid
cells :math:`K = \{0..H{-}1\} \times \{0..W{-}1\}` (:math:`H` = ``num_rows``,
:math:`W` = ``num_cols``), and :math:`f_k`, :math:`f'_k` are cell :math:`k`'s category
in :math:`s` and :math:`s'`, from :math:`\mathcal{C} = \{\textsf{UNBURNT},
\textsf{SMOLDERING}, \textsf{BURNING}, \textsf{BURNT}, \textsf{WET}\}`.
:math:`j` ranges over the :math:`M` = ``num_firefighters`` firefighters,
:math:`j = 1..M`, and :math:`\text{hp}_j`, :math:`\text{hp}'_j` are
firefighter :math:`j`'s health in :math:`s` and :math:`s'`.
Firefighter :math:`j`'s action is the digit :math:`a_j = \lfloor a / 5^{j-1}
\rfloor \bmod 5` of the joint action, with :math:`4` meaning
:math:`\textsf{SUPPRESS}`, and :math:`\text{tank}_j` is its sprays left in
:math:`s`; the
sprayers are :math:`J = \{j : \text{hp}_j > 0,\; a_j = 4,\; \text{tank}_j >
0\}`. The alight categories are :math:`\mathcal{A}\ell =
\{\textsf{SMOLDERING}, \textsf{BURNING}\}`.

.. math::

   R(s, a, s') = \;&-\texttt{step\_cost}
   \;-\; \texttt{smoldering\_cell\_cost} \cdot |\{k : f'_k = \textsf{SMOLDERING}\}|
   \;-\; \texttt{burning\_cell\_cost} \cdot |\{k : f'_k = \textsf{BURNING}\}| \\
   &-\; \texttt{burnt\_cell\_cost} \cdot
     |\{k : f'_k = \textsf{BURNT},\, f_k \neq \textsf{BURNT}\}|
   \;-\; \texttt{damage\_cost} \cdot \textstyle\sum_j (\text{hp}_j - \text{hp}'_j)
   \;-\; \texttt{water\_cost} \cdot |J| \\
   &+\; \texttt{success\_reward} \cdot
     \mathbb{1}\big[f'_k \notin \mathcal{A}\ell \;\; \forall k\big]

**Initial belief.** A state :math:`s` holds the step count :math:`t`; for
each of the :math:`M` = ``num_firefighters`` firefighters :math:`j = 1..M`,
its cell :math:`p_j`, sprays left :math:`\text{tank}_j` and health
:math:`\text{hp}_j`; the hidden wind :math:`(w_{\text{dir}},
w_{\text{str}})`; and the category :math:`f_k \in \{\textsf{UNBURNT}, \textsf{SMOLDERING},
\textsf{BURNING}, \textsf{BURNT}, \textsf{WET}\}` of every cell :math:`k` of the
grid :math:`K = \{0..H{-}1\} \times \{0..W{-}1\}` (:math:`H` = ``num_rows``,
:math:`W` = ``num_cols``).
The step count is :math:`t = 0`. Each firefighter :math:`j` starts at cell :math:`p^0_j` = ``firefighter_start_cells[j-1]``
with a full tank, :math:`\text{tank}_j = \texttt{max\_tank}`, and full health,
:math:`\text{hp}_j = \texttt{max\_health}`. The wind :math:`(w_{\text{dir}},
w_{\text{str}})` is uniform over its eight values :math:`\Theta = \{\textsf{N}, \textsf{E}, \textsf{S},
\textsf{W}\} \times \{\textsf{LOW}, \textsf{HIGH}\}` (the direction it blows
toward, and its strength). :math:`n_0` = ``num_initial_fires`` distinct cells,
chosen uniformly from the free cells :math:`\overline{\mathcal{B}}` (the grid
cells not in ``obstacle_cells``), have category :math:`f_k =
\textsf{BURNING}`, and every other cell :math:`k` is :math:`\textsf{UNBURNT}`:

.. math::

   b_0(s) = \mathbb{1}\big[t = 0,\; (p_j, \text{tank}_j, \text{hp}_j) =
   (p^0_j, \texttt{max\_tank}, \texttt{max\_health}) \;\forall j\big]
   \cdot \frac{1}{8}
   \cdot \binom{|\overline{\mathcal{B}}|}{n_0}^{-1}
   \mathbb{1}\big[\{k : f_k = \textsf{BURNING}\} \subseteq
   \overline{\mathcal{B}},\; |\{k : f_k = \textsf{BURNING}\}| = n_0,\;
   f_k = \textsf{UNBURNT} \text{ otherwise}\big]

**Discount.** :math:`\gamma` = ``discount_factor``.

**Terminal set.** A state :math:`s` holds the step count :math:`t`, the
health :math:`\text{hp}_j` of each of the :math:`M` = ``num_firefighters``
firefighters :math:`j = 1..M`, and the category :math:`f_k \in
\{\textsf{UNBURNT}, \textsf{SMOLDERING}, \textsf{BURNING}, \textsf{BURNT},
\textsf{WET}\}` of every cell :math:`k` of the grid :math:`K = \{0..H{-}1\} \times \{0..W{-}1\}` (:math:`H` = ``num_rows``,
:math:`W` = ``num_cols``), among
other fields. An episode
ends when no cell :math:`k` has a category
:math:`f_k` in the alight set :math:`\mathcal{A}\ell = \{\textsf{SMOLDERING},
\textsf{BURNING}\}`; when ``is_all_firefighters_disabled_terminal`` is set and
every firefighter :math:`j` has health :math:`\text{hp}_j = 0`; or when the
step count :math:`t` reaches ``max_steps``:

.. math::

   S_T = \{s : f_k \notin \mathcal{A}\ell \;\, \forall k\}
   \;\cup\; \{s : \texttt{is\_all\_firefighters\_disabled\_terminal},\;
   \text{hp}_j = 0 \;\, \forall j\}
   \;\cup\; \{s : t \geq \texttt{max\_steps}\}

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

A firefighter's tank holds ``max_tank`` sprays (6 by default) and its health starts at
``max_health`` (3 by default). At zero health a firefighter is disabled: its digit of
the joint action is ignored and it sees nothing for the rest of the episode.

Observation
~~~~~~~~~~~

An observation is one ``float64`` vector of length ``4N + R*C``::

    [(row, col, tank, health) x N, reported category per cell]

Poses, tanks and healths are exact. Every cell within Chebyshev radius
``sensing_radius`` (2 by default) of any *live* firefighter reports its true category
with probability ``1 - observation_error_probability`` and one of the other four
uniformly otherwise. Every other cell reports ``-1``, the unknown marker, so the
observation has a fixed shape whatever the firefighters do. Two firefighters standing
together see barely more than one, so spreading out is what buys information.

The opening observation is a sentinel, not a scan: the known start cells, full
tanks and full health, with every cell ``-1``. The first sensor reading arrives
with the first transition.

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
   * - firefighter damage
     - ``-10.0`` per point of health lost, deliberately above the cell cost so
       a planner does not trade a firefighter for a cell
   * - suppressant
     - ``-0.1`` per firefighter that sprayed (an empty tank does not)
   * - success
     - ``+100.0`` on the transition into a fire-free state

The declared reward range is computed from the constructor's own coefficients,
grid size and firefighter count -- never a constant. The **maximum** is
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
``is_all_firefighters_disabled_terminal`` changes only termination, so it moves
neither end of the bound.

Termination is evaluated in this order: **goal** (no cell alight), **failure**
(every firefighter disabled while fire is still active, when
``is_all_firefighters_disabled_terminal``), then **timeout** (``max_steps``, 100 by
default). Goal wins over failure, so a fire put out by firefighters that then burned
out is still a success.

Key settings
------------

The default world is 10 by 10 with two firefighters, a 2 by 2 obstacle block at
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

- ``task_completion_rate`` reports a fire-free map, reduced with ``ANY``: wet
  and burnt are absorbing, so a fire-free map cannot be undone and ``ANY`` and
  ``LAST`` agree.
- ``ended_by_goal_rate``, ``ended_by_failure_rate`` and
  ``ended_by_timeout_rate`` report how each episode ended and sum to one.
- ``average_episode_length`` is a constant channel summed.

The danger is reported both as a count -- ``average_firefighter_steps_in_fire``,
``average_firefighter_health_lost`` -- and as a severity --
``max_simultaneous_alight_cells``, ``max_burnt_cell_fraction``. A planner that
lets the fire reach forty cells and then beats it out is not the same as one
that never let it past five, and the totals alone would not distinguish them.
``average_suppressant_units_used`` and ``final_firefighters_disabled`` are also reported.

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
maps is degenerate here:

- The poses, tanks and healths come back from the sensor exactly but depend on
  hidden state, so weighting by them puts every weight on the floor and both
  belief panels render as noise; this belief writes the reported values onto
  the particles instead.
- The wind never changes, so resampling across wind values deletes hypotheses
  no later evidence can restore; resampling therefore happens inside a wind
  value.

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
