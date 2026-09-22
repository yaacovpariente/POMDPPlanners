CartPole
========

.. episode-viewer:: traces/cartpole.json

   One real episode planned by PFT-DPW, replayed in 3D. Drag to orbit, scroll
   to zoom, and use the bar to play, scrub and switch camera.

Balance a pole on a cart by pushing left or right, seeing only a noisy reading
of the four-dimensional state. The dynamics are the familiar Gym CartPole; the
partial observability comes from adding Gaussian sensor noise to every
observation.

Use it when you want a continuous-state control problem whose optimal behaviour
you can judge by eye — a good policy survives, a bad one drops the pole in a few
steps — without any of the information-gathering structure of Light-Dark or
RockSample.

What the agent sees and does
----------------------------

- **State** — ``np.ndarray`` of shape ``(4,)``:
  ``[cart_position, cart_velocity, pole_angle, pole_angular_velocity]``. The
  initial state is drawn ``Uniform(-0.05, 0.05)`` in each coordinate.
- **Actions** (discrete) — ``0`` push left, ``1`` push right.
- **Observations** (continuous) — the state plus zero-mean Gaussian noise with
  covariance ``noise_cov``.

Formal definition
-----------------

Write the state as :math:`s = (x, \dot{x}, \vartheta, \dot{\vartheta})`.

**State and observation spaces**

.. math::

   S = \Omega = \mathbb{R}^4, \qquad A = \{0, 1\}
   \quad (\text{push left, push right})

**Transition model.** The Gym CartPole physics, with :math:`m_c = 1.0`,
:math:`m_p = 0.1`, half-length :math:`\ell = 0.5`, :math:`g = 9.8`,
:math:`F = 10` and :math:`\tau = 0.02`. Let :math:`F_a = +F` for
:math:`a = 1` and :math:`-F` for :math:`a = 0`. The accelerations are

.. math::

   \varphi &= \frac{F_a + m_p \ell \dot{\vartheta}^2 \sin\vartheta}
     {m_c + m_p} \\
   \ddot{\vartheta} &= \frac{g \sin\vartheta - \varphi \cos\vartheta}
     {\ell\left(\tfrac{4}{3} -
     \dfrac{m_p \cos^2\vartheta}{m_c + m_p}\right)} \\
   \ddot{x} &= \varphi - \frac{m_p \ell \ddot{\vartheta} \cos\vartheta}
     {m_c + m_p}

integrated by explicit Euler (the configured default):

.. math::

   f(s, a) = \big(x + \tau\dot{x},\;\; \dot{x} + \tau\ddot{x},\;\;
   \vartheta + \tau\dot{\vartheta},\;\;
   \dot{\vartheta} + \tau\ddot{\vartheta}\big)

Gaussian process noise is then added:

.. math::

   T(s' \mid s, a) = \mathcal{N}\big(s';\; f(s, a),\; \Sigma_T\big),
   \qquad
   \Sigma_T = \mathrm{diag}(10^{-4}, 10^{-4}, 2.5 \times 10^{-5}, 10^{-4})

by default, overridable through ``state_transition_cov``.

**Observation model.** The full state read through noise, with no dependence
on the action:

.. math::

   O(o \mid s', a) = \mathcal{N}\big(o;\; s',\; \Sigma_O\big),
   \qquad \Sigma_O = \texttt{noise\_cov}

:math:`\Sigma_O` is a **required** constructor argument: the partial
observability is entirely this matrix, so there is no sensible default. With
:math:`\Sigma_O = 0` the problem is exactly Gym CartPole.

**Reward function.** One unit per step survived:

.. math::

   R(s, a) = \mathbb{1}[s \notin S_T]

so :math:`R \in [0, 1]`. Note it is evaluated on the state the action is taken
*from*, so the step that leaves the limits still pays :math:`1`.

**Initial belief.** Every coordinate independently uniform on a small band
around upright:

.. math::

   b_0 = \mathrm{Unif}\big([-0.05,\, 0.05]^4\big)

The opening observation is a real draw: a state from :math:`b_0` plus
:math:`\mathcal{N}(0, \Sigma_O)`, so the belief starts already blurred.

**Discount.** :math:`\gamma` = ``discount_factor``, required.

**Terminal set.** The cart leaving its track or the pole passing 12 degrees:

.. math::

   S_T = \{s : |x| > 2.4 \ \text{ or }\ |\vartheta| > 12^\circ\}

.. note::

   :math:`S_T` is a predicate on :math:`x` and :math:`\vartheta`, which the
   agent never observes exactly. A planner therefore cannot know it has
   terminated; it can only hold a belief about it.

Rewards
-------

``+1.0`` for every step in a non-terminal state, ``0.0`` in a terminal one.
``reward_range`` is ``(0.0, 1.0)``. There is no step cost and no goal bonus:
the return is simply how long you survived, discounted.

Key settings
------------

.. list-table::
   :header-rows: 1
   :widths: 34 26 40

   * - Argument
     - Default
     - What it changes
   * - ``discount_factor``
     - *required*
     -
   * - ``noise_cov``
     - *required*
     - Observation noise. This is the knob that sets how partially observable
       the problem is; at zero it degenerates to an MDP.
   * - ``state_transition_cov``
     - ``diag([1e-4, 1e-4, 2.5e-5, 1e-4])``
     - Process noise.

The physics constants are fixed in the class, not constructor arguments:
gravity 9.8, cart mass 1.0, pole mass 0.1, half-pole length 0.5, force
magnitude 10.0, timestep 0.02 s, Euler integration.

An episode ends when ``|cart_position| > 2.4`` or ``|pole_angle| > 0.2094`` rad
(12°).

Minimal example
---------------

.. code-block:: python

   import numpy as np

   from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP

   env = CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.01)

   state = env.initial_state_dist().sample(1)[0]
   push_right = 1
   # With n_samples=1 these return the value itself, not a list of one.
   next_state = env.sample_next_state(state, push_right)
   observation = env.sample_observation(next_state, push_right)
   print(next_state, observation, env.reward(state, push_right))

See also
--------

- :class:`POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp.CartPolePOMDP`
- Batched torch model:
  ``POMDPPlanners.environments.cartpole_pomdp.cartpole_vectorized_model.CartPoleVectorizedModel``
- :doc:`index` — the full catalog.
