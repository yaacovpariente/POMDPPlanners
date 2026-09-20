LaserTag
========

.. image:: ../../POMDPPlanners/tests/test_environments/golden_visualizations/laser_tag_visualization.gif
   :alt: A robot chasing an evading opponent through a walled grid.
   :width: 480px

Chase an opponent through a walled arena and fire the tag action from its cell.
The only sensor is eight noisy laser ranges, one per compass direction. A ray
stops at a wall **or at the opponent**, so a ray that runs clear along the
opponent's row or column pins it down exactly, while every other configuration
says almost nothing.

That is what makes the problem interesting: information arrives in rare, sharp
bursts rather than continuously. Between sightings the belief spreads out under
the opponent's movement model, so a planner has to decide whether to manoeuvre
for a clean line of sight or to commit to a tag on stale evidence.

Two variants:

- :class:`LaserTagPOMDP
  <POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp.LaserTagPOMDP>` —
  a grid, five discrete actions.
- :class:`ContinuousLaserTagPOMDP
  <POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp.ContinuousLaserTagPOMDP>`
  — continuous positions, axis-aligned box walls, and a continuous
  ``[dx, dy, tag_flag]`` action. ``ContinuousLaserTagPOMDPDiscreteActions``
  gives that world a five-action set.

.. image:: ../../POMDPPlanners/tests/test_environments/golden_visualizations/continuous_laser_tag_visualization.gif
   :alt: The continuous LaserTag variant with box walls.
   :width: 480px

What the agent sees and does
----------------------------

- **State** — ``np.ndarray`` of shape ``(5,)``:
  ``[robot_row, robot_col, opponent_row, opponent_col, terminal_flag]`` (grid),
  or the same layout in continuous coordinates.
- **Actions** — grid: integers ``0`` north, ``1`` south, ``2`` east, ``3`` west,
  ``4`` tag. Continuous: ``[dx, dy, tag_flag]``.
- **Observations** — eight laser ranges with Gaussian noise
  ``measurement_noise``. The grid variant returns a tuple, the continuous one a
  length-8 ``np.ndarray``. Terminal grid states emit ``(-1.0,) * 8``. Both
  variants sweep the same eight headings but number them differently: the grid
  order is N, NE, E, SE, S, SW, W, NW, while the continuous order starts two
  places later, at E. Continuous beam :math:`i` is grid beam
  :math:`(i + 2) \bmod 8`, so a planner must not carry a beam index from one
  variant to the other.

The opponent moves with probability 0.4 along x, 0.4 along y and stays with
probability 0.2. Those are nominal weights: when the robot is aligned on an
axis the 0.4 splits 0.2/0.2 across both directions, and a blocked neighbour
folds its mass into "stay", so 0.2 is a floor rather than the actual stay
probability. ``opponent_policy`` selects ``EVADE`` (default; away from the
robot's pre-move position), ``PURSUE``, or ``EVADE_WHEN_SPOTTED``, which only
runs from the robot once a laser has actually seen it.

Formal definition
-----------------

Let :math:`G` be the free cells of an :math:`M \times N` grid with wall set
:math:`\mathcal{W}`.

**State space.** Both positions and an absorbing flag:

.. math::

   s = (\mathbf{u},\; \mathbf{v},\; \top), \qquad
   S = G \times G \times \{0, 1\}

with :math:`\mathbf{u}` the robot and :math:`\mathbf{v}` the opponent.

**Action space.**

.. math::

   A = \{0,1,2,3,4\} = \{\textsf{N}, \textsf{S}, \textsf{E}, \textsf{W},
   \textsf{tag}\}

The continuous variant uses :math:`(\mathrm{d}x, \mathrm{d}y, \text{tag flag})
\in \mathbb{R}^3` instead.

**Transition model.** The robot's move first. With :math:`\epsilon` =
``transition_error_prob``, a movement action executes as commanded with
probability :math:`1 - \epsilon` and as one of the other three otherwise:

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
opponent simply stands still.

``EVADE_WHEN_SPOTTED`` switches on visibility: while the opponent is **not**
on an unoccluded laser ray, it walks uniformly (:math:`0.2` per valid
cardinal neighbour, remainder on stay); once spotted, it flees as
``EVADE``. That makes the robot's own sensing change the opponent's dynamics
— the reason this variant is harder than either fixed policy.

**Observation model.** Eight laser ranges, one per compass direction
:math:`\Delta_k` (N, NE, E, SE, S, SW, W, NW). The true range is the number of
free cells before the first blocker — a wall, the grid edge, **or the
opponent**:

.. math::

   \rho_k(s') = \min\{ j \geq 0 :\;
   \mathbf{u}' + (j{+}1)\Delta_k \notin G \ \text{ or }\
   \mathbf{u}' + (j{+}1)\Delta_k = \mathbf{v}' \}

Each is read through independent noise and clipped at zero:

.. math::

   o_k = \max\big(0,\; \rho_k(s') + \varepsilon_k\big), \qquad
   \varepsilon_k \sim \mathcal{N}(0, \sigma^2)

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
   \exists c:\ \lVert \mathbf{u}' - c \rVert_2 \leq \varrho \big]

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

Minimal example
---------------

.. code-block:: python

   from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP

   env = LaserTagPOMDP(discount_factor=0.95)

   state = env.initial_state_dist().sample(1)[0]
   north = 0
   next_state = env.sample_next_state(state, north)
   laser_ranges = env.sample_observation(next_state, north)
   print(next_state, laser_ranges)

See also
--------

- Batched torch model:
  ``POMDPPlanners.environments.laser_tag_pomdp.laser_tag_vectorized_model.LaserTagVectorizedModel``
  (grid variant only).
- :doc:`index` — the full catalog.
