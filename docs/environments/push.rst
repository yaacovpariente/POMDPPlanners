Push
====

.. episode-viewer:: traces/push.json

   ``PushPOMDP``: one real episode planned by PFT-DPW, replayed in 3D. Drag to
   orbit, scroll to zoom, and use the bar to play, scrub and switch camera.

A robot moves around a square arena and pushes an object towards a fixed target
corner. Only the object's position is observed noisily; the robot always knows
where it is. Pushing is indirect — you have to get on the far side of the object
first — so the agent must plan several steps ahead through a contact model.

Two variants:

- :class:`PushPOMDP <POMDPPlanners.environments.push_pomdp.push_pomdp.PushPOMDP>`
  — four grid moves.
- :class:`ContinuousPushPOMDP
  <POMDPPlanners.environments.push_pomdp.continuous_push_pomdp.ContinuousPushPOMDP>`
  — a circular robot and free 2-D displacement actions, with square obstacles.

.. episode-viewer:: traces/continuous_push.json

   ``ContinuousPushPOMDP``: the same, in the continuous world. Drag to orbit,
   scroll to zoom, and use the bar to play, scrub and switch camera.

.. note::

   ``ContinuousPushPOMDP`` and ``ContinuousPushPOMDPDiscreteActions`` are **not**
   in ``ENVIRONMENT_REGISTRY``, so ``get_environment("ContinuousPushPOMDP")``
   fails. Import the class directly.

What the agent sees and does
----------------------------

- **State** — ``np.ndarray`` of shape ``(6,)``:
  ``[robot_x, robot_y, object_x, object_y, target_x, target_y]``. The continuous
  variant appends a terminal-flag slot when either hazard-terminal flag is on.
- **Actions** — ``PushPOMDP``: ``"up"``, ``"down"``, ``"right"``, ``"left"``.
  ``ContinuousPushPOMDP``: a 2-D displacement vector, with Gaussian noise
  ``state_transition_cov_matrix``.
- **Observations** (continuous) — the same vector as the state, with only the
  object's position noised by ``observation_noise``. It is 7-D on a
  ``ContinuousPushPOMDP`` that carries the terminal slot.

The target sits at ``(grid_size - 1, grid_size - 1)`` and is not configurable.

Formal definition
-----------------

Write the state as :math:`s = (\mathbf{r}, \mathbf{q}, \mathbf{t})` — robot,
object and target positions — and let :math:`\mathcal{G} = [0, n{-}1]^2` with
:math:`n` = ``grid_size``, :math:`\mathcal{O}` the obstacle discs.

**State space.** Six numbers, plus a terminal slot on the continuous variant
when a hazard-terminal flag is on:

.. math::

   S = \mathcal{G} \times \mathcal{G} \times \mathcal{G}
   \subseteq \mathbb{R}^6

:math:`\mathbf{t}` is a constant of the episode — it is carried in the state
so the model is self-contained, not because it moves.

**Action space.**

.. math::

   A = \begin{cases}
     \{\textsf{up}, \textsf{down}, \textsf{right}, \textsf{left}\}
       & \texttt{PushPOMDP} \\
     \{\mathbf{d} \in \mathbb{R}^2 :
       \lVert \mathbf{d} \rVert \leq \texttt{max\_push}\}
       & \texttt{ContinuousPushPOMDP}
   \end{cases}

**Transition model.** The robot moves, and *then* drags the object if it is
close enough. With displacement :math:`\mathbf{d}`, friction
:math:`\mu` = ``friction_coefficient`` and push radius :math:`\varrho` =
``push_threshold``:

.. math::

   \mathbf{r}' &= \Pi_\mathcal{G}\big(
     \mathrm{blk}(\mathbf{r} + \mathbf{d})\big) \\
   \mathbf{q}' &= \begin{cases}
     \Pi_\mathcal{G}\big(\mathrm{blk}(\mathbf{q} + (1 - \mu)\mathbf{d})\big)
       & \lVert \mathbf{r}' - \mathbf{q} \rVert_2 < \varrho \\
     \mathbf{q} & \text{otherwise}
   \end{cases} \\
   \mathbf{t}' &= \mathbf{t}

where :math:`\mathrm{blk}(\mathbf{y}) = \mathbf{y}` unless :math:`\mathbf{y}`
lies in an obstacle, in which case the mover stays put, and
:math:`\Pi_\mathcal{G}` clamps to the grid. The object moves a factor
:math:`1 - \mu` of the robot's displacement — friction is a *slip* between
robot and object, not a drag on the robot.

Note the push test uses :math:`\mathbf{r}'`, the robot's **post-move**
position: the robot must end its step near the object, not start there.

The discrete variant adds action noise. With probability
:math:`\epsilon` = ``transition_error_prob`` one of the other three moves
fires instead, uniformly:

.. math::

   \Pr[\text{executed} = a] = 1 - \epsilon, \qquad
   \Pr[\text{executed} = a'] = \epsilon / 3, \quad a' \neq a

The continuous variant instead perturbs the displacement by
:math:`\mathcal{N}(0, \Sigma_T)`.

**Observation model.** The robot knows where *it* is; only the object is
hidden:

.. math::

   o = \big(\mathbf{r}',\;
   \Pi_\mathcal{G}(\mathbf{q}' + \boldsymbol{\varepsilon}),\;
   \mathbf{t}\big), \qquad
   \boldsymbol{\varepsilon} \sim
   \mathcal{N}(0,\; \texttt{observation\_noise}^2 I_2)

Robot and target slices are exact; only the two object coordinates are
noised, then clamped to the grid. The clamp is what makes the likelihood
non-Gaussian at the walls.

**Reward function.** Distance shaping toward the target plus an exclusive
success bonus, then the hazard terms:

.. math::

   R(s, a, s') = -\lVert \mathbf{q}' - \mathbf{t} \rVert_2
   + 100 \cdot \mathbb{1}\big[\lVert \mathbf{q}' - \mathbf{t} \rVert_2
     < 0.5\big]
   + C(\mathbf{r}') + D(\mathbf{r}')

with

.. math::

   C(\mathbf{r}') &= \texttt{obstacle\_penalty} \cdot
     \mathbb{1}[\mathbf{r}' \in \mathcal{O}] \cdot
     \mathrm{Bern}(\texttt{obstacle\_hit\_probability}) \\
   D(\mathbf{r}') &= \texttt{dangerous\_area\_penalty} \cdot
     \mathbb{1}[\mathbf{r}' \in \text{hazard}] \cdot
     \mathrm{Bern}(\texttt{dangerous\_area\_hit\_probability})

The shaping term is on the **object**, the penalties on the **robot**. Both
hazard terms follow the same three ``reward_model_type`` variants as
:doc:`rock_sample`. With either hit probability below one, ``reward`` draws a
Bernoulli per call and is not a deterministic function of its arguments.

**Initial belief.** With ``initial_state`` supplied, :math:`b_0` is a point
mass on it. Otherwise robot and object positions are drawn at random over the
grid, clear of the obstacles and of the target. Note :math:`\mathbf{r}` is
then in the prior but observed exactly on the first reading, so the belief
concentrates on the object alone.

**Discount.** :math:`\gamma` = ``discount_factor``, required.

**Terminal set.** The *object* reaching the target, within a fixed half-cell
radius:

.. math::

   S_T = \{s : \lVert \mathbf{q} - \mathbf{t} \rVert_2 < 0.5\}

The target sits at :math:`(n{-}1, n{-}1)` and is not configurable.

Rewards
-------

============================================  =======================================
Event                                         Reward
============================================  =======================================
Every step                                    ``-distance(object, target)``
Object within 0.5 of the target               ``+100.0``
Robot inside an obstacle                      ``obstacle_penalty`` (``-10.0``)
Robot inside a dangerous area                 ``dangerous_area_penalty`` (``-10.0``)
============================================  =======================================

Penalties are **added**, so pass them negative.

Key settings
------------

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Argument
     - Default
     - What it changes
   * - ``grid_size``
     - ``10``
     - Arena size, and therefore the target corner.
   * - ``push_threshold``
     - ``1.0``
     - How close the robot must be to move the object.
   * - ``friction_coefficient``
     - ``0.3``
     - How far a push carries.
   * - ``observation_noise``
     - ``0.1``
     - Object-position sensor noise — the partial-observability knob.
   * - ``obstacles``
     - ``None``
     - Centres of circular obstacles of radius ``obstacle_radius``
       (discrete, default ``0.5``), or ``(cx, cy, half_extent)`` squares
       (continuous).
   * - ``max_push``
     - ``2.0`` (continuous only)
     - Cap on the force delivered to the object, not on the action itself.
   * - ``robot_radius``
     - ``0.3`` (continuous only)
     -
   * - ``initial_state``
     - ``None``
     - ``None`` randomizes robot and object placement each episode, keeping the
       object at least 2.0 away from the target.

An episode ends when the object is within 0.5 of the target. On
``ContinuousPushPOMDP`` the absorbing terminal slot is checked first, so an
obstacle or hazard hit also ends the episode when the matching
``is_*_hit_terminal`` flag is on.

Minimal example
---------------

.. code-block:: python

   from POMDPPlanners.environments.push_pomdp import PushPOMDP

   env = PushPOMDP(discount_factor=0.95, grid_size=10)

   state = env.initial_state_dist().sample(1)[0]
   next_state = env.sample_next_state(state, "right")
   observation = env.sample_observation(next_state, "right")
   print(next_state, observation, env.reward(state, "right", next_state))

See also
--------

- Batched torch model:
  ``POMDPPlanners.environments.push_pomdp.push_vectorized_model.PushVectorizedModel``
  (wraps the discrete variant).
- :doc:`index` — the full catalog.
