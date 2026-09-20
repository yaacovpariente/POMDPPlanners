Light-Dark
==========

.. image:: ../../POMDPPlanners/tests/test_environments/golden_visualizations/light_dark_visualization.gif
   :alt: An agent detouring through a light beacon before heading to the goal.
   :width: 480px

Navigate to a goal in a world where your position sensor is sharp near beacons
and vague everywhere else. The shortest path is rarely the best one: going the
long way through a beacon buys the localization needed to actually land on the
goal.

This is the cleanest test of whether a planner values information. A planner
optimizing expected reward under a point estimate drives straight at the goal
and misses; one that reasons over beliefs detours.

Two variants ship, sharing the same reward shape:

- :class:`ContinuousLightDarkPOMDP
  <POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp.ContinuousLightDarkPOMDP>`
  — continuous 2-D position, continuous action, continuous observation.
- :class:`DiscreteLightDarkPOMDP
  <POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp.DiscreteLightDarkPOMDP>`
  — integer grid, four moves, grid-cell observations.

There is also ``ContinuousLightDarkPOMDPDiscreteActions``: the continuous world
with the four discrete moves, for planners that need a finite action set. Note
its noise defaults differ — ``np.eye(2)`` rather than ``np.eye(2) * 0.05``.

Formal definition
-----------------

Both variants share one geometry: beacons :math:`\mathcal{B}`, obstacles
:math:`\mathcal{O}`, goal :math:`\mathbf{g}`, start :math:`\mathbf{x}_0`, and
a grid of side :math:`\mathcal{G}` = ``grid_size``.

Continuous variant
~~~~~~~~~~~~~~~~~~

**State space.** The plane, plus a terminal slot when
``is_obstacle_hit_terminal`` (which defaults to ``True``, so states are 3-D by
default):

.. math::

   S = \mathbb{R}^2 \times \{0, 1\}, \qquad s = (\mathbf{p}, \top)

Note :math:`S` is *not* restricted to the grid. Leaving it is penalized but
not terminal, and the sampler deliberately does not clip — clipping the
sampler while the observation density stayed unclipped would break importance
weights near the edges.

**Action space.** An unbounded displacement:

.. math::

   A = \mathbb{R}^2

**Transition model.** Additive Gaussian, with the commanded displacement as
the mean:

.. math::

   T(\mathbf{p}' \mid \mathbf{p}, \mathbf{a}) =
   \mathcal{N}\big(\mathbf{p}';\; \mathbf{p} + \mathbf{a},\; \Sigma_T\big),
   \qquad \Sigma_T = \texttt{state\_transition\_cov\_matrix}

A terminal state is absorbing and draws nothing. When the hazard flag is on,
the transition also draws :math:`\top'` from the obstacle hit probability at
:math:`\mathbf{p}'`, which is what makes the hazard penalty deterministic
given :math:`s'`.

**Observation model.** This is the environment's whole point. Let

.. math::

   \nu(\mathbf{p}) = \min_{\mathbf{b} \in \mathcal{B}}
   \lVert \mathbf{p} - \mathbf{b} \rVert_2

be the distance to the nearest beacon and :math:`r_\mathcal{B}` =
``beacon_radius``. Under ``NORMAL_NOISE`` (the default) the agent always sees
its position, but the covariance is **halved** inside a beacon:

.. math::

   O(\mathbf{o} \mid \mathbf{p}', \cdot) =
   \mathcal{N}\big(\mathbf{o};\; \mathbf{p}',\; \Sigma_O(\mathbf{p}')\big),
   \qquad
   \Sigma_O(\mathbf{p}') = \begin{cases}
     \tfrac{1}{2}\Sigma_O & \nu(\mathbf{p}') < r_\mathcal{B} \\
     \Sigma_O & \text{otherwise}
   \end{cases}

The other two models replace the reading in the dark with a null symbol
instead of a wider one:

.. math::

   \texttt{NORMAL\_NOISE\_NO\_OBS\_IN\_DARK}: \quad
   \mathbf{o} = \begin{cases}
     \mathcal{N}(\mathbf{p}', \tfrac{1}{2}\Sigma_O)
       & \nu(\mathbf{p}') < r_\mathcal{B} \\
     \textsf{None} & \text{otherwise}
   \end{cases}

.. math::

   \texttt{DISTANCE\_BASED}: \quad
   \mathbf{o} = \begin{cases}
     \textsf{None} & \nu(\mathbf{p}') > r_\mathcal{B} \\
     \mathcal{N}(\mathbf{p}', \tfrac{1}{2}\Sigma_O)
       & \nu(\mathbf{p}') < r_\mathcal{B} \\
     \mathcal{N}(\mathbf{p}', \Sigma_O) & \text{otherwise (on the boundary)}
   \end{cases}

.. note::

   That third branch is not a typo. The "near" test is a strict :math:`<` and
   the "no reading" test is a strict :math:`>`, so a position exactly at
   :math:`\nu = r_\mathcal{B}` falls through to the wide covariance. The
   mismatch is preserved from the original model.

**Reward function.** A fuel cost plus a distance-to-goal shaping term, with
one exclusive bonus or penalty on top:

.. math::

   R(s, \mathbf{a}, s') = -\texttt{fuel\_cost}
   - \lVert \mathbf{p}' - \mathbf{g} \rVert_2 + \begin{cases}
     +\texttt{goal\_reward} & \lVert \mathbf{p}' - \mathbf{g} \rVert_2
       \leq r_{\mathbf{g}} \\
     +\texttt{obstacle\_reward} & \text{hazard fires} \\
     +\texttt{obstacle\_reward} & \mathbf{p}' \notin [0, \mathcal{G}]^2 \\
     0 & \text{otherwise}
   \end{cases}

(``obstacle_reward`` is negative by default, so both middle branches are
penalties.) The shaping term is what a planner using a point estimate
follows straight into the dark.

.. note::

   The declared ``reward_range`` bounds *in-grid* states only, using the grid
   diagonal :math:`\sqrt{2}\,\mathcal{G}` as the largest in-grid distance to
   the goal. A state can drift arbitrarily far outside the grid, and its
   reward can fall below that minimum. Those states sit outside the region
   the bound is defined over.

**Initial belief.** :math:`b_0 = \delta_{\mathbf{x}_0}` — the start is known
exactly. Uncertainty is produced entirely by the process noise as the agent
moves, and removed only by visiting a beacon.

**Terminal set.**

.. math::

   S_T = \{s : \lVert \mathbf{p} - \mathbf{g} \rVert_2 \leq r_{\mathbf{g}}\}
   \;\cup\; \{s : \top = 1\}

Discrete variant
~~~~~~~~~~~~~~~~

**State space.** :math:`S = \mathbb{Z}^2` (with the optional terminal slot).

**Action space.**

.. math::

   A = \{\textsf{up}, \textsf{down}, \textsf{right}, \textsf{left}\},
   \qquad \Delta = \{(0,1), (0,-1), (1,0), (-1,0)\}

**Transition model.** The commanded move is executed with probability
:math:`1 - \epsilon_T`, and otherwise one of the other three fires, uniformly:

.. math::

   T(\mathbf{p}' \mid \mathbf{p}, a) = \begin{cases}
     1 - \epsilon_T & \mathbf{p}' = \mathbf{p} + \Delta_a \\
     \epsilon_T / 3 & \mathbf{p}' = \mathbf{p} + \Delta_{a'},\; a' \neq a
   \end{cases}

with :math:`\epsilon_T` = ``transition_error_prob``. There are no walls: the
agent can step outside the grid, and pays for it through the reward.

**Observation model.** Five outcomes — the true cell, or one of the four
neighbours:

.. math::

   O(\mathbf{o} \mid \mathbf{p}', \cdot) = \begin{cases}
     1 - \epsilon(\mathbf{p}') & \mathbf{o} = \mathbf{p}' \\
     \epsilon(\mathbf{p}') / 4 & \mathbf{o} = \mathbf{p}' + \Delta_{a'}
   \end{cases}

The beacon effect is a **five-fold** reduction in the error rate rather than
the continuous variant's halved covariance:

.. math::

   \epsilon(\mathbf{p}') = \begin{cases}
     0.2 \cdot \epsilon_O & \nu(\mathbf{p}') < r_\mathcal{B} \\
     \epsilon_O & \text{otherwise}
   \end{cases}

with :math:`\epsilon_O` = ``observation_error_prob``.

**Reward function.** The same shape, with the goal and obstacle tests by
exact cell equality rather than by radius, and the obstacle penalty gated by
a Bernoulli:

.. math::

   R = -\texttt{fuel\_cost} - \lVert \mathbf{p}' - \mathbf{g} \rVert_2
   + \begin{cases}
     +\texttt{goal\_reward} & \mathbf{p}' = \mathbf{g} \\
     \texttt{obstacle\_reward} \cdot
       \mathrm{Bern}(\texttt{obstacle\_hit\_probability})
       & \mathbf{p}' \in \mathcal{O} \\
     +\texttt{obstacle\_reward} & \mathbf{p}' \notin [0, \mathcal{G}]^2 \\
     0 & \text{otherwise}
   \end{cases}

The branches are exclusive, in that order. The Bernoulli makes ``reward``
non-deterministic in :math:`(s, a)`; ``is_obstacle_hit_terminal=True`` moves
the draw into the transition and removes that. This class declares no
``reward_range``.

Continuous variant
------------------

- **State** — ``np.array([x, y])``. Because ``is_obstacle_hit_terminal``
  defaults to ``True``, a third terminal-flag slot is appended, so states are
  3-D by default.
- **Actions** (continuous) — a displacement ``np.array([dx, dy])``. The next
  state is ``state + action + N(0, state_transition_cov_matrix)``. No bound on
  the step is enforced.
- **Observations** (continuous) — noisy position. The covariance is halved
  within ``beacon_radius`` of any beacon. Three models are selectable through
  ``observation_model_type``: ``NORMAL_NOISE``, ``NORMAL_NOISE_NO_OBS_IN_DARK``
  and ``DISTANCE_BASED``.

Discrete variant
----------------

- **State** — ``np.array([x, y])`` on an integer grid.
- **Actions** (discrete) — ``"up"``, ``"down"``, ``"right"``, ``"left"``.
- **Observations** (discrete) — a grid cell: the true cell with probability
  ``1 - error``, otherwise a neighbour. The error is
  ``observation_error_prob * 0.2`` near a beacon and ``observation_error_prob``
  away from one.
- Its ``observation_model_type`` is a *different* enum, with members ``NORMAL``,
  ``NO_OBS_IN_DARK`` and ``DISTANCE_BASED``.
- ``reward_range`` is ``None`` on this class — it does not declare one.

Rewards
-------

Both variants start from ``-fuel_cost - ||next_state - goal||`` and then follow
an **ordered chain** — the first branch that matches wins, so the terms never
stack:

.. code-block:: text

   base = -fuel_cost - ||next_state - goal||

   inside the goal radius   ->  base + goal_reward
   else on an obstacle      ->  base            (the hazard penalty is applied
                                                 separately, via the hazard draw)
   else outside the grid    ->  base + obstacle_reward
   otherwise                ->  base

Defaults: ``fuel_cost=2.0``, ``goal_reward=10.0``, ``obstacle_reward=-10.0``,
``obstacle_hit_probability=0.2``.

.. note::

   A step that leaves the grid can pay less than the declared ``reward_range``
   minimum on the continuous variant. This is known, documented in the source,
   and covered by a strict xfail test — do not treat it as a new bug.

Key settings
------------

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Argument
     - Default
     - What it changes
   * - ``grid_size``
     - ``11``
     - World extent.
   * - ``beacons``
     - 9 points on a 3×3 lattice
     - Where localization is cheap. Removing beacons makes the problem harder.
   * - ``start_state`` / ``goal_state``
     - ``[0, 5]`` / ``[10, 5]``
     -
   * - ``goal_state_radius`` (continuous only)
     - ``1.5``
     - How precisely you must arrive. Shrink it to force a beacon detour. The
       discrete variant has no such argument — it terminates on exact cell
       equality with ``goal_state``.
   * - ``obstacles``
     - ``[(3, 7), (5, 5)]``
     -
   * - ``state_transition_cov_matrix``
     - ``np.eye(2) * 0.05``
     - Motion noise (continuous variant).
   * - ``observation_cov_matrix``
     - ``np.eye(2) * 0.05``
     - Sensor noise away from a beacon (continuous variant).
   * - ``is_obstacle_hit_terminal``
     - ``True`` (continuous), ``False`` (discrete)
     - On the continuous variant, whether an obstacle ends the episode. On the
       discrete variant it chooses *which* termination rule applies, not
       whether: ``False`` is the legacy behaviour where obstacles are
       **always** terminal, ``True`` makes them Bernoulli-terminal through the
       hazard draw and adds the terminal slot.

An episode ends at the goal — inside ``goal_state_radius`` on the continuous
variant, on the exact cell on the discrete one — or on an obstacle hit under
the rule above. Leaving the grid is penalized but not terminal.

Minimal example
---------------

.. code-block:: python

   import numpy as np

   from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
       ContinuousLightDarkPOMDP,
   )

   env = ContinuousLightDarkPOMDP(discount_factor=0.95)

   state = env.initial_state_dist().sample(1)[0]
   step_right = np.array([1.0, 0.0])
   next_state = env.sample_next_state(state, step_right)
   observation = env.sample_observation(next_state, step_right)
   print(state[:2], "->", next_state[:2], "seen as", observation)

See also
--------

- Batched torch model:
  ``POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_vectorized_model.ContinuousLightDarkVectorizedModel``
  (continuous variant only — the discrete one has none).
- :doc:`index` — the full catalog.
