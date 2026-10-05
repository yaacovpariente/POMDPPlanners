Mountain Car
============

.. episode-viewer:: traces/mountain_car.json

   One recorded episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

A car whose engine force (:math:`k = 0.001`) is smaller than the largest pull of
gravity on the slope (:math:`g = 0.0025`), so it cannot climb the hill directly
and must rock back and forth to build momentum, while reading position and velocity through
noise.

The reward is -1 on every step until the goal, so it says nothing about
progress, and reaching the goal takes many steps. That makes it hard for
short-horizon search: a planner that cannot see past its horizon never
discovers that reversing first is what wins.

What the agent sees and does
----------------------------

- **State** — ``[position, velocity]``, position in ``[-1.2, 0.6]``, velocity in
  ``[-0.07, 0.07]``, with process noise ``diag([2.5e-5, 1e-6])``.
- **Actions** (discrete) — ``-1``, ``0``, ``1``.
- **Observations** (continuous) — noisy ``[position, velocity]``, standard
  deviations 0.1 and 0.01.

Formal definition
-----------------

The environment is the POMDP :math:`\langle S, A, Z, T, O, R, b_0, \gamma
\rangle`.

**State space**

.. math::

   S = [-1.2,\, 0.6] \times [-0.07,\, 0.07], \qquad s = (p, v)

with :math:`p` the car's position along the valley and :math:`v` its
velocity.

**Action space**

.. math::

   A = \{-1, 0, 1\}

:math:`-1` pushes left, :math:`0` does not push, :math:`1` pushes right.

**Observation space**

.. math::

   Z = \mathbb{R}^2

A noisy reading of :math:`(p, v)`, in that order.

**Transition model.** The deterministic part is the Mountain Car
map, with power :math:`k = 0.001` and gravity :math:`g = 0.0025`:

.. math::

   \tilde{v} &= \mathrm{clip}\big(v + ak - g\cos(3p),\; -0.07,\; 0.07\big) \\
   \tilde{p} &= \mathrm{clip}(p + \tilde{v},\; -1.2,\; 0.6) \\
   f(s, a) &= \begin{cases}
     (\tilde{p},\, 0) & \tilde{p} = -1.2 \text{ and } \tilde{v} < 0 \\
     (\tilde{p},\, \tilde{v}) & \text{otherwise}
   \end{cases}

The left wall is inelastic: hitting it zeroes the velocity. Gaussian process
noise is then added and the result projected back into :math:`S` by the same
clipping rule :math:`\mathrm{proj}`:

.. math::

   s' = \mathrm{proj}\big(f(s, a) + w\big), \qquad
   w \sim \mathcal{N}(0, \Sigma_T), \qquad
   \Sigma_T = \mathrm{diag}(2.5 \times 10^{-5},\; 10^{-6})

:math:`\Sigma_T` is the ``state_transition_cov`` constructor argument.

.. note::

   ``transition_log_probability`` returns the *unprojected* density
   :math:`\log \mathcal{N}(s'; f(s,a), \Sigma_T)`. At the boundary that
   density is not the density of the projected variable, which puts positive
   mass on the walls. It matters only for beliefs with particles at a wall.

**Observation model.** The full state, read through independent noise:

.. math::

   O(o \mid s', a) = \mathcal{N}(o;\, s',\, \Sigma_O), \qquad
   \Sigma_O = \mathrm{diag}(0.1^2,\; 0.01^2)

The action does not enter. Position noise of :math:`0.1` against a domain of
width :math:`1.8` is what makes this a POMDP rather than Mountain Car.

**Reward function.** One unit of cost per step until the goal:

.. math::

   R(s, a) = -\mathbb{1}[p < 0.5]

so :math:`R \in [-1, 0]`.

**Initial belief.** Position uniform on a band around the valley floor,
velocity exactly zero:

.. math::

   b_0:\quad p \sim \mathrm{Unif}([-0.6,\, -0.4]), \qquad v = 0

with the initial observation reported as :math:`(0, 0)` rather than sampled
from :math:`O`.

**Discount.** :math:`\gamma` = ``discount_factor``, required.

**Terminal set.** :math:`S_T = \{(p, v) \in S : p \geq 0.5\}`.

Rewards
-------

======================  =========
Event                   Reward
======================  =========
Step before the goal    -1.0
Step at the goal        0.0
======================  =========

The goal is ``position >= 0.5``. ``reward_range`` is ``(-1.0, 0.0)``, and the
episode ends at the goal.

Key settings
------------

``discount_factor`` is required; the class has no default.
``state_transition_cov`` sets the process noise on ``[position, velocity]``.
It defaults to ``None``, which means ``diag([2.5e-5, 1e-6])``. The observation
noise (standard deviations 0.1 and 0.01) is fixed in the class, not an
argument.

Metrics
~~~~~~~

It reports a ``task_completion_rate`` metric.

Can I use?
----------

.. list-table::
   :header-rows: 1
   :widths: 34 30

   * - Capability
     - ``MountainCarPOMDP``
   * - Action space
     - Discrete
   * - Observation space
     - Continuous
   * - Native C++ backend
     - ✔️
   * - Vectorized (torch) model
     - ✔️ ``MountainCarVectorizedModel``
   * - In the ``get_environment`` registry
     - ✔️
   * - Optional dependencies
     - None

Example
-------

.. code-block:: python

   from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp import (
       MountainCarPOMDP,
   )

   env = MountainCarPOMDP(discount_factor=0.99)
   state = env.initial_state_dist().sample(1)[0]
   next_state = env.sample_next_state(state, 1)
   print(state, next_state, env.sample_observation(next_state, 1))

Parameters
----------

.. autoclass:: POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp.MountainCarPOMDP
   :members:
   :show-inheritance:

See also
--------

- :class:`POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp.MountainCarPOMDP`
- :doc:`sanity` — the two-state debugging environment.
- :doc:`base` — the full catalog and the environment interface.
