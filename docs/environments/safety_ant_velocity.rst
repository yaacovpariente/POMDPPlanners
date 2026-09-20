Safety Ant Velocity
===================

.. image:: ../../POMDPPlanners/tests/test_environments/golden_visualizations/safety_ant_velocity_visualization.gif
   :alt: A point mass accelerating across the plane while staying under the speed limit.
   :width: 480px

Move as fast as you can while keeping speed below a safety threshold, judging
your own speed only through a noisy sensor. Reward grows with speed and a heavy
penalty fires above the threshold, so the optimal behaviour sits just under a
line whose position you cannot see exactly.

Use it for constrained and risk-aware planners. It is the package's simplest
environment where the interesting question is not "what is the best expected
return" but "how much probability mass is over the limit".

.. note::

   Despite the name this is **not** a MuJoCo or Safety-Gymnasium wrapper. It is
   a self-contained 2-D point mass. Nothing in the package imports ``mujoco``
   or ``safety-gymnasium``.

What the agent sees and does
----------------------------

- **State** — ``np.ndarray`` of shape ``(4,)``:
  ``[position_x, position_y, velocity_x, velocity_y]``. The initial position is
  ``Uniform(-1, 1)`` in each axis with zero velocity.
- **Actions** (discrete) — ``0``–``3``, force magnitudes
  ``[0.0, 0.33, 0.67, 1.0] * max_force``. The force **direction is randomized
  by the transition kernel**; the agent chooses only how hard to push.
- **Observations** (continuous) — the 4-vector with diagonal Gaussian noise,
  ``position_noise`` on position and ``velocity_noise`` on velocity.

Formal definition
-----------------

Write the state as :math:`s = (\mathbf{p}, \mathbf{v})` with
:math:`\mathbf{p}, \mathbf{v} \in \mathbb{R}^2`, and let
:math:`\lVert \mathbf{v} \rVert` be the speed.

**Spaces**

.. math::

   S = \Omega = \mathbb{R}^4, \qquad A = \{0, 1, 2, 3\}

**Transition model.** The agent chooses a force *magnitude*; the **direction
is drawn by the environment**:

.. math::

   \phi = \kappa_a \cdot \texttt{max\_force}, \qquad
   \kappa = (0,\; 0.33,\; 0.67,\; 1.0)

.. math::

   \theta \sim \mathrm{Unif}(-\pi, \pi), \qquad
   \mathbf{F} = \phi\,(\cos\theta,\; \sin\theta)

The damped point-mass dynamics are then integrated semi-implicitly:

.. math::

   \mathbf{a} &= \frac{\mathbf{F} - c\,\mathbf{v}}{m} \\
   \mathbf{v}' &= \mathbf{v} + \mathbf{a}\,\Delta t \\
   \mathbf{p}' &= \mathbf{p} + \mathbf{v}'\,\Delta t

with :math:`m` = ``mass``, :math:`c` = ``damping``, :math:`\Delta t` = ``dt``.
Position is updated with the *new* velocity, not the old one.

.. note::

   The stochasticity here is unusual: it is in the force's **direction**, not
   an additive Gaussian on the state. So :math:`T(\cdot \mid s, a)` is
   supported on a *circle* in velocity space — a one-dimensional set in
   :math:`\mathbb{R}^4` — rather than having a density over it. For
   :math:`a = 0` the transition is deterministic, since :math:`\phi = 0`
   makes :math:`\theta` irrelevant.

**Observation model.** The full state with independent diagonal noise:

.. math::

   O(o \mid s', a) = \mathcal{N}(o;\; s',\; \Sigma_O), \qquad
   \Sigma_O = \mathrm{diag}\big(\sigma_p^2,\, \sigma_p^2,\,
   \sigma_v^2,\, \sigma_v^2\big)

with :math:`\sigma_p` = ``position_noise``, :math:`\sigma_v` =
``velocity_noise``. The action does not enter.

**Reward function.** Paid for speed, penalized for exceeding the limit — both
read off the state the action is taken **from**:

.. math::

   R(s, a) = \lVert \mathbf{v} \rVert \cdot
   \texttt{movement\_reward\_scale}
   + \texttt{safety\_violation\_penalty} \cdot
   \mathbb{1}\big[\lVert \mathbf{v} \rVert > \tau\big]

with :math:`\tau` = ``safe_velocity_threshold``. This is the whole tension:
:math:`R` grows linearly in speed right up to :math:`\tau`, then falls off a
cliff of :math:`-100` by default. Because the agent only ever sees
:math:`\lVert \mathbf{v} \rVert` through noise of width :math:`\sigma_v`, it
cannot know which side of :math:`\tau` it is on — it can only trade expected
speed against the probability of having crossed.

**Initial belief.** Position uniform in a unit box, velocity exactly zero:

.. math::

   b_0 = \mathrm{Unif}\big([-1, 1]^2\big) \otimes \delta_{\mathbf{0}}

.. note::

   ``initial_observation_dist`` returns the *state* distribution, not a draw
   through :math:`O`. The opening "observation" is therefore a noiseless
   state sample, which is not what the observation model would produce.

**Discount.** :math:`\gamma` = ``discount_factor``, required.

**Terminal set.** A 50 % margin above the penalty threshold:

.. math::

   S_T = \{s : \lVert \mathbf{v} \rVert > 1.5\,\tau\}

so there is a band :math:`\tau < \lVert \mathbf{v} \rVert \leq 1.5\tau` where
the agent is being penalized every step but the episode continues — it can
still brake back under the limit.

Rewards
-------

.. code-block:: text

   reward = speed * movement_reward_scale
            + safety_violation_penalty   if speed > safe_velocity_threshold

Defaults: ``movement_reward_scale=1.0``, ``safety_violation_penalty=-100.0``,
``safe_velocity_threshold=2.0``. ``reward_range`` is
``(safety_violation_penalty, safe_velocity_threshold * 1.5 * movement_reward_scale)``.

The reward is a function of the **current state only** — ``action`` and
``next_state`` are ignored.

Key settings
------------

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Argument
     - Default
     - What it changes
   * - ``safe_velocity_threshold``
     - ``2.0``
     - Where the penalty starts, and (at 1.5×) where the episode dies.
   * - ``safety_violation_penalty``
     - ``-100.0``
     - How risk-averse the optimal policy is.
   * - ``velocity_noise``
     - ``0.2``
     - How well the agent can tell it is speeding. This is what makes the
       problem a POMDP.
   * - ``position_noise``
     - ``0.1``
     -
   * - ``max_force`` / ``mass`` / ``damping`` / ``dt``
     - ``1.0`` / ``1.0`` / ``0.1`` / ``0.1``
     - Point-mass dynamics.

An episode ends when speed exceeds ``safe_velocity_threshold * 1.5`` — ``3.0``
at defaults, a 50 % margin above the penalty line.

.. note::

   The default ``name`` is ``"SafeVelocityPOMDP"``, not
   ``"SafeAntVelocityPOMDP"``. That string reaches result filenames and
   ``config_id``.

Minimal example
---------------

.. code-block:: python

   from POMDPPlanners.environments.safety_ant_velocity_pomdp import SafeAntVelocityPOMDP

   env = SafeAntVelocityPOMDP(discount_factor=0.95, safe_velocity_threshold=2.0)

   state = env.initial_state_dist().sample(1)[0]
   full_thrust = 3
   next_state = env.sample_next_state(state, full_thrust)
   observation = env.sample_observation(next_state, full_thrust)
   print(next_state, observation, env.reward(next_state, full_thrust))

See also
--------

- :class:`POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_pomdp.SafeAntVelocityPOMDP`
- Batched torch model:
  ``POMDPPlanners.environments.safety_ant_velocity_pomdp.safety_ant_velocity_vectorized_model.SafetyAntVelocityVectorizedModel``
- :doc:`index` — the full catalog.
