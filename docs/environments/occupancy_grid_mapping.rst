Occupancy grid mapping
======================

``OccupancyGridMappingPOMDP`` models a robot mapping a hidden static grid with
known pose and noisy range scans. The default world is 10 by 10 cells with a
boundary wall and three random rectangular obstacles. The robot starts at the
centre, facing north. Actions are forward one cell, turn left and turn right.
Integer pose is a design simplification; particle-based MCTS also supports
continuous states.

.. code-block:: python

   from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
       OccupancyGridMappingPOMDP, OccupancyGridMappingBelief,
   )

   env = OccupancyGridMappingPOMDP()
   belief = OccupancyGridMappingBelief.initial(env, n_particles=30)

State and observation contract
------------------------------

The state contains the step, row, column, heading, hidden true occupancy,
observation-derived map log-odds, and last noisy scan. Its length is
``4 + 2 * num_cells + num_beams``. The hidden map constrains motion and produces
nominal ranges. Gaussian noise is drawn once in the transition, then the same
scan is stored, used for mapping, and revealed by the observation.

Observations contain ``[row, column, heading, ranges...]``. Pose is exact and
integer. Repeated observation calls on one successor reveal the same stored
scan. Gaussian ranges are unbounded; they are never clipped in the sampler.
The augmented observation kernel is a point mass. The predictive density
``p(observation | prior state, action)`` combines Gaussian ranges with the
probability of the observed motion outcome.

The initial observation contains known pose and zero range placeholders.
It is a sentinel before any scan and is never applied to the map.

Mapping and reward
------------------

Log-odds start at zero, or occupancy probability 0.5. The inverse update reads
only the previous log-odds and observed pose/ranges. A reading below maximum
range selects the nearest ray-cell centre; cells before it get free evidence,
the selected cell gets occupied evidence, and cells after it remain unchanged.
Negative readings select the first valid cell. Ties select the nearer cell.
A reading at or above maximum range marks the full in-grid ray free.
The robot's observed cell also receives free evidence. Log-odds are clamped.

A true hit exactly at maximum range and a miss have identical range laws.
They therefore produce identical updates for the same reading. Without an
observed hit flag, this ambiguity cannot be removed. Gaussian noise can place
an apparent hit beyond a real obstacle or before it; the mapper follows the
measurement, not hidden truth.

Simulation reward is the realised decrease in the observed inverse map's
summed binary entropy, minus ``step_cost``. The environment declares
``reward_requires_next_state=True`` so the runner supplies that realised map.
When a planner requests reward without a successor, the explicit fallback uses
eight fixed antithetic Gaussian samples per motion outcome to approximate the
expected decrease. This deterministic integration consumes no simulation RNG,
but has integration error. It updates each hypothetical map with the same
observation-based rule as the actual map.

This is an exploration surrogate inspired by entropy-reduction objectives,
not exact posterior information gain. The inverse map treats cells and beam
increments as independent. Repeated correlated beams can create confidence in
an incorrect map. Low entropy measures confidence, not map accuracy.

An episode completes when the observed inverse map's entropy reaches
``entropy_threshold_fraction`` of its initial value (25% by default).
``max_steps`` (40 by default) ends an unfinished episode. Reward and completion
use the inverse estimate; neither substitutes the true occupancy grid.

Filtering and limits
--------------------

Use ``OccupancyGridMappingBelief`` with PFT_DPW. Ordinary
``get_initial_belief`` returns a bootstrap filter which cannot condition this
stored continuous scan correctly. No core planner changes are needed.

The custom filter installs the observed scan and its map update in every
particle. It weights whole-map hypotheses by the predictive Gaussian density
and exact motion probability, with no epsilon floor for impossible poses.
Routine resampling is disabled to retain low-weight hypotheses. If every
particle contradicts observed motion, bounded prior replay tests up to 4096
fresh maps against the entire observation history and resamples surviving
weighted maps. It raises if that search finds no support.

Thirty particles is a QA resource choice, not a guarantee against degeneracy.
Reweighting cannot create missing maps. Prior replay is finite importance
sampling and may leave only one effective hypothesis. Inspect effective sample
size, unique maps, restarts and map error alongside completion. Unweighted
rejection filters are unsuitable for exact matching of continuous scans.

Metrics and visualization
-------------------------

``task_completion_rate`` reports threshold crossing. ``ended_by_goal``,
``ended_by_failure`` and ``ended_by_timeout`` report episode endings; failure
is always zero. Progress metrics include residual entropy and resolved-cell
fraction. ``average_obstacle_collisions`` counts blocked moves;
``average_successful_translations`` counts successful moves, including revisits.
It is not a unique-cell count.

.. image:: ../images/occupancy_grid_mapping_visualization.gif
   :alt: Observed map, the weighted map estimate, and the true map with the robot pose.
   :width: 100%

The left panel shows the observation-derived inverse map. The middle shows
weighted occupancy marginals over whole-map particles. The right shows the
hidden true map for review. Panels depict the state before the captioned action;
the caption's reward is the realised inverse-map entropy reduction.

There is no torch vectorized or C++ model. VOPP is unsupported. Scalar PFT_DPW
uses the environment and custom belief shown above. Sensor contract version 2
changes cache identity; old results and GIFs describe the earlier behavior.
