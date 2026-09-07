Sanity and Mountain Car
=======================

Two small environments that get a short entry because there is little to
explain: neither is a benchmark you would report results on, and both exist to
make something else easier to debug.

SanityPOMDP
-----------

Two states, two actions, perfect observations, no terminal state. Action ``0``
moves to the "good" state and pays ``1.0``; action ``1`` moves to "bad" and pays
``0.0``. ``reward_range`` is ``(0.0, 1.0)``.

There is no partial observability and nothing to plan around. Its job is to fail
loudly: if a planner does not converge to always taking action ``0``, the bug is
in the planner, not in the problem. Run it first when a planner produces
nonsense somewhere else.

.. code-block:: python

   from POMDPPlanners.environments.sanity_pomdp import SanityPOMDP

   env = SanityPOMDP(discount_factor=0.95)
   state = env.initial_state_dist().sample(1)[0]
   print(env.reward(state, 0), env.reward(state, 1))

MountainCarPOMDP
----------------

The classic underpowered car: it cannot climb the hill directly and must rock
back and forth to build momentum, while reading position and velocity through
noise.

- **State** — ``[position, velocity]``, position in ``[-1.2, 0.6]``, velocity in
  ``[-0.07, 0.07]``, with process noise ``diag([2.5e-5, 1e-6])``.
- **Actions** (discrete) — ``-1``, ``0``, ``1``.
- **Observations** (continuous) — noisy ``[position, velocity]``, standard
  deviations 0.1 and 0.01.
- **Rewards** — ``-1.0`` per step, ``0.0`` once ``position >= 0.5``.
  ``reward_range`` is ``(-1.0, 0.0)``; the episode ends at the goal. It reports
  a ``goal_reaching_rate`` metric.

The long horizon and the sparse, purely negative reward make it a hard problem
for short-horizon search: a planner that cannot see past its horizon never
discovers that reversing first is what wins.

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

- :doc:`index` — the full catalog.
