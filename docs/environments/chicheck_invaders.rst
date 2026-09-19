Chicheck Invaders
=============

``ChicheckInvadersPOMDP`` is an arcade shooter written as a POMDP. A ship on row 0
of a grid has to clear a flock of chickens before one of them reaches it. The
default world is 8 columns by 7 rows with 4 chickens and a 60-step budget. The
ship starts in the middle column with an empty sky.

.. code-block:: python

   from POMDPPlanners.environments.chicheck_invaders_pomdp import ChicheckInvadersPOMDP
   from POMDPPlanners.utils.belief_factory import create_environment_belief

   env = ChicheckInvadersPOMDP()
   belief = create_environment_belief(env, n_particles=200)

Dynamics
--------

Actions are stay, left, right and fire, in that index order. A move is clamped
at the walls. ``FIRE`` discharges only when the cooldown has expired; otherwise
it behaves exactly like ``STAY`` and costs nothing, because nothing left the
ship.

**The gun is hitscan.** A shot resolves inside the step that fired it, killing
the lowest live chicken in the ship's column over rows 1 to ``num_rows - 1``, at
unlimited range. A shot into a column with no live chicken misses and is charged
the shot cost alone. ``fire_cooldown`` is the only rate limiter.

The shot is resolved **before** the dive coins are flipped and before the flock
moves, and that ordering is the design rather than an implementation detail. The
ship picks its action from an observation of where the chickens are *now*, so it
has to be able to hit what it aimed at. An earlier version fired a bolt that
climbed one row per step; against chickens stepping sideways every step it was
dodged by accident rather than by any decision the flock made, so aiming
collapsed into waiting and most shots missed for reasons the ship could not have
reasoned about.

One consequence worth naming: a chicken shot this step never gets to dive, so
firing is a live defence as well as an attack.

A chicken is either patrolling or diving, and which it is stays hidden. A
patrolling chicken steps one column along its direction and reverses at a wall;
a diving chicken drops one row. Each step every surviving patrolling chicken
switches into a dive with probability ``dive_probability``. The coin is flipped
after the shot but *before* the chickens move, so a chicken that switches this
step also drops this step.

A chicken that reaches row 0 in the ship's column destroys it and ends the
episode. One that reaches row 0 anywhere else **pulls up**: it goes back to
patrolling at the top row, keeping its column and direction. That rule is an
addition to the original design proposal, which left the case open. Without it a
dive would either carry the chicken off the grid -- letting the flock clear
itself and making the completion bonus free -- or park it on row 0, below the
rows the gun covers, which is unwinnable.

State and observation contract
------------------------------

The state is ``[step, ship column, cooldown, ship hit, (column, row, direction,
mode, alive) per chicken]``, so its length is ``4 + 5 * num_chickens``. A dead
chicken keeps its slot. Nothing records a shot: a hitscan shot never survives
the step that fired it, so there is nothing in flight for the state to carry and
the only thing the gun leaves behind is the cooldown.

An observation is ``[own-column reading, (camera reported, camera offset, radar
reported, radar rows, radar drop) per chicken]``, of length
``1 + 5 * num_chickens``. Every masked field is written as ``0.0``, so two
readings that report the same things compare equal and hash alike; the two
``reported`` flags are what separate a masked zero from a genuine reading of
zero.

The two sensors each give half an answer:

* the **camera** covers a cone opening upward from the ship, with half-slope
  ``camera_slope``. It reports a chicken's *column* offset and says nothing
  about its row.
* the **radar** covers a disc of radius ``radar_radius``. It reports a chicken's
  *row* distance and whether it is dropping, and says nothing about its column.

A chicken inside a sensor's reach is reported with probability
``camera_detection_probability`` or ``radar_detection_probability``; a dead one
or one outside the reach is never reported. Every integer reading is drawn from
a rounded Gaussian

.. math::

   G_\sigma(k; \mu) = \Phi\!\left(\frac{k + 1/2 - \mu}{\sigma}\right)
                    - \Phi\!\left(\frac{k - 1/2 - \mu}{\sigma}\right),
   \qquad k \in \mathbb{Z},

which is the law of ``round(mu + sigma * xi)``. It is not truncated at the grid
edge, so its masses sum to one over all integers and a reading may name a column
that does not exist. Truncating would make the normaliser depend on the hidden
quantity being inferred and would change every likelihood ratio in the belief
update for no gain in realism. ``sigma = 0`` collapses the law to a point mass.

The likelihood multiplies in one factor for **every** chicken slot, not only the
reported ones:

.. math::

   Z(o \mid s') = G_{\sigma_c}(\hat c; c')
                  \prod_{i=1}^{N} q_i^{\text{cam}} \, q_i^{\text{rad}},

where a chicken in reach and reported contributes its detection probability
times the noise mass, one in reach and silent contributes the miss chance, one
out of reach and silent contributes 1, and one out of reach but reported
contributes 0. Silence is therefore evidence: a chicken that a particle places
well inside both sensors, on a step where neither reported it, costs that
particle a factor of ``0.1 * 0.1``. Everything is computed in log space, with an
impossible reading floored rather than set to negative infinity, because
``0 * -inf`` becomes a NaN inside weight normalisation.

Modes and presets
-----------------

``observation_mode="full"`` makes the observation the state itself, for a fully
observable control baseline on identical dynamics and reward.
``noiseless_preset()`` returns the constructor keywords that make every sensor
always report, add no noise and never invert the drop flag, so the observation
becomes a deterministic function of the successor. The transition is untouched:
dives are still drawn, which is what keeps the preset a POMDP whose only
certainty is the reading.

.. code-block:: python

   from POMDPPlanners.environments.chicheck_invaders_pomdp import (
       ChicheckInvadersPOMDP,
       noiseless_preset,
   )

   deterministic_sensors = ChicheckInvadersPOMDP(**noiseless_preset())
   fully_observable = ChicheckInvadersPOMDP(observation_mode="full")

Reward and termination
----------------------

+10 per chicken killed, -1 per shot actually fired, -0.1 every step, -50 when a
chicken reaches the ship, +50 when the last chicken dies. A shot that connects
therefore pays ``-0.1 - 1 + 10`` on the step it was fired.

The declared reward range is enumerated rather than estimated. A step kills at
most one chicken, and a kill can only happen on a step that fired, so the best
step is the shot that takes the last chicken and collects the bonus with it:
``kill_reward + clear_reward - step_cost - shot_cost``.

The declared minimum stacks the step cost, the shot cost and the ship-hit
penalty, and that sum is deliberately **not reachable**. The gun kills the
lowest chicken in the ship's column, and a chicken can only reach the ship by
diving down that same column, so any step that could be overrun gave the shot a
target and the kill reward comes back; firing into a genuinely empty column
cannot be overrun at all. The worst a run can actually score is being overrun
without firing, exactly one shot cost above the bound. The wider bound is kept
because it costs nothing and survives a change to the gun's reach, where a tight
one derived from that argument would not.

``reward_requires_next_state`` is ``True``, but only because of the ship hit.
Under hitscan the kill and the completion bonus are functions of
``(state, action)`` alone -- the shot resolves before anything random happens --
so a call without a successor returns everything except the ``-50``. That
missing term is deliberately not replaced by its expectation: a planner
comparing actions at a belief node sees the real value of a shot that connects,
and the risk it took is charged on the step the flock actually gets through.

An episode ends when the flock is cleared, when a chicken reaches the ship, or
at ``max_steps``.

Belief
------

``ChicheckInvadersBelief`` is a weighted particle filter with reinvigoration, and it
is what ``create_environment_belief`` returns. The interesting half of the work
is done by the likelihood above rather than by any special code: particles that
put chickens where the sensors would have seen them die off on their own.

What a weight-only filter cannot do is invent a hypothesis it never held, and
two things here need that -- the opening placement is drawn from a large set of
cells that a few hundred particles only partly cover, and a chicken outside both
sensors for several steps drifts away from whichever patrol phase the particles
guessed. After each reweight and resample, a fraction of the particles therefore
have their *unreported* chickens re-drawn: a fresh direction, a fresh mode, and a
one-cell jitter of the position. Chickens the sensors just reported are left
exactly as the weights found them, because perturbing one would throw away the
only hard information the step produced.

Metrics and visualization
-------------------------

``task_completion_rate`` is the fraction of episodes that cleared the flock,
reduced with ``ANY`` -- clearing happens once and ends the episode.
``ended_by_goal``, ``ended_by_failure`` and ``ended_by_timeout`` reduce with
``LAST`` and sum to one per episode. ``average_episode_length``,
``average_chickens_killed`` and ``average_shots_fired`` are per-episode sums.

The danger is a chicken getting close, reported both ways:
``average_hits_taken`` counts the times the flock actually got through, and
``max_chicken_encroachment_cells`` is the severity. The latter is reported as
``max(num_columns - 1, num_rows - 1)`` minus the smallest Chebyshev distance
from the ship to a live chicken, so larger means closer and the distance itself
is recoverable by subtraction. It is reported this way round because the episode
reduction available for a severity is ``MAX`` and there is no ``MIN``.

``shot_accuracy`` is kills per shot, which under hitscan is simply hits per
shot. It is the one metric that is not a channel
reduction -- a ratio of two per-episode sums cannot be expressed as a reduction
over a single channel, and a mean of per-step ratios is not the episode's ratio
-- so it is computed in ``compute_metrics`` from the same two channels the
counts are built from. Episodes that never fired are left out of the average
rather than scored as zero.

.. image:: ../images/chicheck_invaders_visualization.gif
   :alt: The true sky with ship, chickens and shots, beside the belief's weighted chance of a chicken per cell.
   :width: 100%

The left panel is the true world: the ship, the chickens with their mode shown
by the sprite, the beam of a shot fired on that step, the edges of the camera
cone and the radar's ring. The beam appears only on steps that discharged the
gun and stops at the chicken it killed, which is ringed; a shot into an empty
column runs the full height and rings nothing. There is no bolt to follow,
because a shot never survives the step it was fired in. The right panel is the belief's weighted per-cell chance that a
chicken is there. Drawing the particles themselves is this repository's usual
choice for a low-dimensional state, and the belief here is a particle cloud --
but one particle is a whole flock, and a few hundred overlaid flocks are a smear
rather than a picture. The per-cell marginal is the projection the task turns on
and is still the belief rather than a fit to it. Each frame shows the state a
step was taken from; the caption names the action about to be taken.

Limits
------

There is no torch vectorized model and no C++ model, so VOPP cannot run on this
environment. Scalar ``PFT_DPW`` runs on it directly, which is what the QA gate
uses.
