LaserTag
========

.. episode-viewer:: traces/laser_tag.json

   Grid variant, ``LaserTagPOMDP``: one recorded episode planned by PFT-DPW,
   replayed in 3D. Drag to orbit, scroll to zoom, and use the bar to play,
   scrub and switch camera.

.. episode-viewer:: traces/continuous_laser_tag.json

   Continuous variant, ``ContinuousLaserTagPOMDP``: the same, in the
   continuous world. Drag to orbit, scroll to zoom, and use the bar to play,
   scrub and switch camera.

Chase an opponent through a walled arena and fire the tag action from its cell.
The only sensor is eight noisy laser ranges, one per compass direction. A ray
stops at a wall **or at the opponent**, so a ray that reaches the opponent
gives its distance up to sensor noise, while a ray that ends at a wall only
says the opponent is not on it.

So the robot learns where the opponent is only on steps when the two share a
row, column or diagonal with no wall between them. Between sightings the belief spreads out under
the opponent's movement model, so a planner has to decide whether to manoeuvre
for an unblocked ray to the opponent or to commit to a tag on evidence from
earlier steps. There is a
grid variant and a continuous one; see `Variants`_.

What the agent sees and does
----------------------------

- **State** — ``np.ndarray`` of shape ``(5,)``:
  ``[robot_row, robot_col, opponent_row, opponent_col, terminal_flag]`` (grid),
  or the same layout in continuous coordinates.
- **Actions** (discrete in the grid, continuous otherwise) — grid: integers
  ``0`` north, ``1`` south, ``2`` east, ``3`` west, ``4`` tag. Continuous:
  ``[dx, dy, tag_flag]``.
- **Observations** (continuous) — eight laser ranges with Gaussian noise
  ``measurement_noise``. The grid variant returns a tuple, the continuous one a
  length-8 ``np.ndarray``. Terminal grid states emit ``(-1.0,) * 8``. Both
  variants sweep the same eight headings but number them differently: the grid
  order is N, NE, E, SE, S, SW, W, NW, while the continuous order starts two
  places later, at E. Continuous beam :math:`i` is grid beam
  :math:`(i + 2) \bmod 8`, so a planner must not carry a beam index from one
  variant to the other.

Formal definition
-----------------

The environment is the POMDP :math:`\langle S, A, Z, T, O, R, b_0, \gamma \rangle`.
It is written for the grid variant.

**State space.** Let :math:`G` be the free cells of an :math:`M \times N` grid
with wall set :math:`\mathcal{W}`; a cell is written (row, column). Both
positions and an absorbing flag:

.. math::

   s = (\mathbf{u},\; \mathbf{v},\; \top), \qquad
   S = G \times G \times \{0, 1\}

with :math:`\mathbf{u}` the robot's cell, :math:`\mathbf{v}` the opponent's
cell, and :math:`\top = 1` once the episode has ended.

**Action space.**

.. math::

   A = \{0,1,2,3,4\} = \{\textsf{N}, \textsf{S}, \textsf{E}, \textsf{W},
   \textsf{tag}\}

:math:`0` moves one row up (row :math:`-1`), :math:`1` one row down (row
:math:`+1`), :math:`2` one column right (column :math:`+1`), :math:`3` one
column left (column :math:`-1`); :math:`4` (tag) stays and tries to tag the
opponent on the robot's cell.

The continuous variant uses :math:`(\mathrm{d}x, \mathrm{d}y, \text{tag flag})
\in \mathbb{R}^3` instead.

**Observation space.** Eight non-negative laser ranges, or the all
:math:`-1` vector a terminal state emits:

.. math::

   Z = \mathbb{R}_{\geq 0}^{8} \cup \{(-1, \dots, -1)\}

Component :math:`k` is the range along heading N, NE, E, SE, S, SW, W, NW
for :math:`k = 0..7`. The continuous variant returns the same eight ranges as
a length-8 array starting at E: its beam :math:`i` is grid beam
:math:`(i + 2) \bmod 8`.

**Transition model.** The robot's move first. With :math:`p` =
``transition_error_prob``, a movement action executes as commanded with
probability :math:`1 - p` and as one of the other three otherwise, chosen
uniformly. With :math:`a'` the executed move and :math:`\Delta_0 = (-1, 0)`,
:math:`\Delta_1 = (1, 0)`, :math:`\Delta_2 = (0, 1)`, :math:`\Delta_3 = (0, -1)`
its (row, column) offset:

.. math::

   \mathbf{u}' = \begin{cases}
     \mathbf{u} + \Delta_{a'} & \mathbf{u} + \Delta_{a'} \in G \\
     \mathbf{u} & \text{otherwise}
   \end{cases}

:math:`\textsf{tag}` does not move the robot. A tag on the opponent's cell
ends the episode immediately, :math:`\top' = 1`, with no opponent draw.

*The opponent* then moves under a stochastic policy with a fixed mass
schedule. Write :math:`\mathbf{w}` for the robot cell it conditions on —
post-move under ``PURSUE``, pre-move under ``EVADE`` and
``EVADE_WHEN_SPOTTED`` (for a tag these coincide, since tagging does not
move). Per axis independently, the policy puts :math:`0.4` on one neighbour:

.. math::

   \begin{array}{lll}
     \texttt{PURSUE} & 0.4 \text{ on the cell toward } \mathbf{w}
       & 0.0 \text{ away} \\
     \texttt{EVADE} & 0.0 \text{ toward} & 0.4 \text{ away} \\
     \text{axes aligned } (w_i = v_i) & 0.2 \text{ each way}
       & \text{(policy-invariant)}
   \end{array}

plus :math:`0.2` on staying put. Invalid neighbours are dropped and their mass
falls back onto *stay*, so the distribution always normalizes and a cornered
opponent stands still.

``EVADE_WHEN_SPOTTED`` switches on visibility: while the opponent is **not**
on an unoccluded laser ray, it walks uniformly (:math:`0.2` per valid
cardinal neighbour, remainder on stay); once spotted, it flees as
``EVADE``. That makes the robot's own sensing change the opponent's dynamics
— the reason this variant is harder than either fixed policy.

In short: the opponent moves with probability 0.4 along x, 0.4 along y and stays with
probability 0.2. Those are nominal weights: when the robot is aligned on an
axis the 0.4 splits 0.2/0.2 across both directions, and a blocked neighbour
folds its mass into "stay", so 0.2 is a floor rather than the actual stay
probability. ``opponent_policy`` selects ``EVADE`` (default; away from the
robot's pre-move position), ``PURSUE``, or ``EVADE_WHEN_SPOTTED``, which only
runs from the robot once a laser has seen it.

**Observation model.** Eight laser ranges, one per compass direction
:math:`\Delta_k` (N, NE, E, SE, S, SW, W, NW). The true range is the number of
free cells before the first blocker — a wall, the grid edge, **or the
opponent**:

.. math::

   d_k(s') = \min\{ j \geq 0 :\;
   \mathbf{u}' + (j{+}1)\Delta_k \notin G \ \text{ or }\
   \mathbf{u}' + (j{+}1)\Delta_k = \mathbf{v}' \}

Each is read through independent noise and clipped at zero:

.. math::

   o_k = \max\big(0,\; d_k(s') + n_k\big), \qquad
   n_k \sim \mathcal{N}(0, \sigma^2)

with :math:`\sigma` = ``measurement_noise``. This is the whole inference
problem: the opponent is visible only as a **shortened ray**, so a reading
short by one is ambiguous between an opponent and the sensor's noise, and a
ray blocked by a wall carries no information about the opponent at all.
Terminal states emit :math:`(-1, \dots, -1)`.

**Reward function.** Evaluated against the pre-transition positions for the
tag, the realised position for the hazard:

.. math::

   R(s, a, s') = \begin{cases}
     0 & \top = 1 \\
     +\texttt{tag\_reward} + H(\mathbf{u}')
       & a = \textsf{tag},\ \mathbf{u} = \mathbf{v} \\
     -\texttt{tag\_penalty} + H(\mathbf{u}')
       & a = \textsf{tag},\ \mathbf{u} \neq \mathbf{v} \\
     -\texttt{step\_cost} + H(\mathbf{u}') & \text{otherwise}
   \end{cases}

The hazard term charges **one** penalty on a wall *or* a danger zone, not one
for each:

.. math::

   H(\mathbf{u}') = -\texttt{dangerous\_area\_penalty} \cdot
   \mathbb{1}\big[\mathbf{u}' \in \mathcal{W} \ \text{ or }\
   \exists c:\ \lVert \mathbf{u}' - c \rVert_2 \leq \texttt{dangerous\_area\_radius} \big]

where :math:`c` ranges over the danger-zone centres ``dangerous_areas``.

A mistimed tag costs ``tag_penalty`` rather than merely a step, which is what
makes guessing expensive and the belief worth maintaining.

**Initial belief.** Uniform over every pair of distinct free cells:

.. math::

   b_0\big((\mathbf{u}, \mathbf{v}, 0)\big) = \frac{1}{|G|(|G| - 1)},
   \qquad \mathbf{u} \neq \mathbf{v}

so the robot's *own* position starts unknown too, and the first laser reading
must localize both. With ``initial_state`` supplied, :math:`b_0` is a point
mass on it instead.

.. note::

   The opening observation is a fixed mid-range placeholder
   :math:`(3, \dots, 3)`, not a draw from :math:`O` under :math:`b_0`. A
   filter that weights the first reading through
   ``observation_log_probability`` is therefore scoring a reading the
   observation model did not produce.

**Discount.** :math:`\gamma` = ``discount_factor``, required.

**Terminal set.** :math:`S_T = \{s : \top = 1\}` — set by a successful tag,
or by a hazard hit when ``is_dangerous_area_hit_terminal``.

Rewards
-------

=====================================  =========================================
Event                                  Reward
=====================================  =========================================
Tag on the opponent's cell             ``tag_reward`` (``+10.0``)
Tag and miss                           ``-tag_penalty`` (``-10.0``)
Any move                               ``-step_cost`` (``-1.0``)
Inside a wall or a dangerous area      ``-dangerous_area_penalty`` (``-5.0``)
=====================================  =========================================

.. warning::

   As in PacMan, ``dangerous_area_penalty`` is a **positive magnitude that is
   subtracted**, the opposite of the RockSample and Push convention.

Key settings
------------

.. list-table::
   :header-rows: 1
   :widths: 34 24 42

   * - Argument
     - Default
     - What it changes
   * - ``floor_shape`` (grid)
     - ``(11, 7)``
     - Arena size as ``(rows, cols)``. The module docstring says "7x11"; the
       code says ``(11, 7)``.
   * - ``grid_size`` (continuous)
     - ``(11.0, 7.0)``
     - Arena size as ``(width, height)`` — note the axis order differs from
       ``floor_shape``.
   * - ``walls``
     - 8 fixed cells / box list
     - Layout. Walls both block movement and shape the laser readings.
   * - ``measurement_noise``
     - ``1.0``
     - Laser noise — the partial-observability knob.
   * - ``opponent_policy``
     - ``OpponentPolicy.EVADE``
     - ``PURSUE`` makes the opponent close on the robot; ``EVADE_WHEN_SPOTTED``
       makes it flee only while a laser has line of sight.
   * - ``tag_reward`` / ``tag_penalty`` / ``step_cost``
     - ``10.0`` / ``10.0`` / ``1.0``
     - The impatience trade-off: raise ``tag_penalty`` to punish speculative
       tags.
   * - ``dangerous_areas``
     - ``{(5,3), (7,1), (2,5)}``
     -
   * - ``evasion_speed``
     - ``0.6`` (continuous only)
     -

An episode ends when the terminal flag is set — on a successful tag, or on a
hazard hit when ``is_dangerous_area_hit_terminal=True``.

Variants
~~~~~~~~

- :class:`LaserTagPOMDP
  <POMDPPlanners.environments.laser_tag_pomdp.LaserTagPOMDP>` —
  a grid, five discrete actions.
- :class:`ContinuousLaserTagPOMDP
  <POMDPPlanners.environments.laser_tag_pomdp.ContinuousLaserTagPOMDP>`
  — continuous positions, axis-aligned box walls, and a continuous
  ``[dx, dy, tag_flag]`` action. ``ContinuousLaserTagPOMDPDiscreteActions``
  gives that world a five-action set.

Can I use?
----------

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Capability
     - ``LaserTagPOMDP``
     - ``ContinuousLaserTagPOMDP``
     - ``ContinuousLaserTagPOMDPDiscreteActions``
   * - Action space
     - Discrete
     - Continuous
     - Discrete
   * - Observation space
     - Continuous
     - Continuous
     - Continuous
   * - Native C++ backend
     - ✔️ (loaded lazily; falls back to Python if the extension is missing)
     - ✔️
     - ✔️
   * - Vectorized (torch) model
     - ✔️ ``LaserTagVectorizedModel`` (some configurations; others raise NotImplementedError)
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

   from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP

   env = LaserTagPOMDP(discount_factor=0.95)

   state = env.initial_state_dist().sample(1)[0]
   north = 0
   next_state = env.sample_next_state(state, north)
   laser_ranges = env.sample_observation(next_state, north)
   print(next_state, laser_ranges)

Parameters
----------

.. autoclass:: POMDPPlanners.environments.laser_tag_pomdp.LaserTagPOMDP
   :members:
   :show-inheritance:

.. autoclass:: POMDPPlanners.environments.laser_tag_pomdp.ContinuousLaserTagPOMDP
   :members:
   :show-inheritance:

.. autoclass:: POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp.ContinuousLaserTagPOMDPDiscreteActions
   :members:
   :show-inheritance:

See also
--------

- Batched torch model:
  ``POMDPPlanners.environments.laser_tag_pomdp.laser_tag_vectorized_model.LaserTagVectorizedModel``
  (grid variant only).
- :doc:`base` — the full catalog and the environment interface.
