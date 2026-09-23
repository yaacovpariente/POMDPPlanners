Mountain Car
============

.. episode-viewer:: traces/mountain_car.json

   One real episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

The classic underpowered car: it cannot climb the hill directly and must rock
back and forth to build momentum, while reading position and velocity through
noise.

The long horizon and the sparse, purely negative reward make it a hard problem
for short-horizon search: a planner that cannot see past its horizon never
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

The environment is the POMDP :math:`\langle S, A, \Omega, T, O, R, b_0, \gamma
\rangle`. Write the state as :math:`s = (p, v)`.

**State space**

.. math::

   S = [-1.2,\, 0.6] \times [-0.07,\, 0.07]

**Action space**

.. math::

   A = \{-1, 0, 1\}

**Observation space**

.. math::

   \Omega = \mathbb{R}^2

**Transition model.** The deterministic part is the standard Mountain Car
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
   mass on the walls. It matters only for beliefs pressed against a wall.

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

Metrics
-------

It reports a ``goal_reaching_rate`` metric.

Minimal example
---------------

.. code-block:: python

   from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp import (
       MountainCarPOMDP,
   )

   env = MountainCarPOMDP(discount_factor=0.99)
   state = env.initial_state_dist().sample(1)[0]
   next_state = env.sample_next_state(state, 1)
   print(state, next_state, env.sample_observation(next_state, 1))

See also
--------

- :class:`POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp.MountainCarPOMDP`
- :doc:`sanity` — the other small debugging environment.
- :doc:`index` — the full catalog.
