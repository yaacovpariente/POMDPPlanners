Occupancy grid mapping
======================

``OccupancyGridMappingPOMDP`` drops a robot into a 2-D world it has never seen
and pays it for finding out what the world looks like. There is no goal cell and
no object to fetch: the task is the map. The robot carries a range sensor, keeps
an occupancy grid of the cells around it, and is rewarded for the uncertainty it
drives out of that grid.

.. code-block:: python

   from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
       OccupancyGridMappingPOMDP,
   )

   env = OccupancyGridMappingPOMDP(num_rows=10, num_cols=10, num_beams=24)

The default world is a 10 by 10 grid walled on the outside, with three
rectangular obstacles of up to 2 by 2 cells placed at random inside it. The
robot starts at the centre facing north. The map is drawn afresh at the start of
each episode and never changes within one, so the map is what a belief is over.

Actions, observations and rewards
---------------------------------

Three actions: ``0`` move forward one cell along the heading, ``1`` turn left,
``2`` turn right. Headings are ``N``, ``E``, ``S``, ``W``; rows grow downwards.
A forward move into an occupied cell or off the grid leaves the robot where it
was, wastes the step, and is counted as a collision.

The observation is ``[row, column, heading, range × num_beams]``. The pose is
reported exactly: this is *mapping with known poses*, the classical setting for
occupancy grids, and adding localisation would make it a different problem. Each
beam reports the centre-to-centre distance to the first occupied cell it meets,
or the sensor's maximum range when it meets none, plus Gaussian noise of
``range_noise_std_cells``. The noisy ranges are not clipped back to the sensor's
range, because clipping would put a point mass at the ends that the Gaussian
likelihood cannot represent.

The reward is the reduction in the occupancy grid's binary entropy, in bits —
the information-gain exploration objective of Bourgault et al. (2002). It is a
*belief-dependent* reward, which no other environment here has, and it is
expressed by carrying the robot's own occupancy grid inside the state next to
the hidden true map. A belief particle therefore holds one candidate world and
the map the robot would have built in it, so averaging this reward over the
particles is exactly the expected information gain.

The grid is stored as log-odds, ``l = log(p / (1 - p))``, initialised to zero —
``p = 0.5``, unknown. The update is additive there, so a cell seen a hundred
times is a sum rather than a hundred multiplications of small numbers.
Probabilities appear only where entropy is computed and where the visualization
draws. Accumulated log-odds are clamped to ``±log_odds_clamp``, which stops a
cell swept by many beams from becoming unrevisable.

Per beam, the inverse sensor model marks the cells the beam passed through as
free, the cell it stopped on as occupied, and leaves everything behind that cell
alone — the beam saw nothing there. A beam that returns nothing marks its whole
length free and marks no cell occupied. That last case is what makes exploration
work: without it, looking into open space would be indistinguishable from not
looking. The cell the robot is standing in is marked free too, since the robot
is in it and no beam ever reports it.

An episode ends when the grid's entropy falls to
``entropy_threshold_fraction`` of its initial value (25% by default — an average
of a quarter of a bit per cell), or when ``max_steps`` transitions have been
taken. Nothing here can fail: a wall stops the robot and costs it a step, but
ends nothing.

Metrics
-------

``task_completion_rate`` is the fraction of episodes whose map was resolved.
``ended_by_goal``, ``ended_by_failure`` and ``ended_by_timeout`` say why each
episode stopped and sum to one; ``ended_by_failure`` is always zero here.
``final_residual_entropy_bits`` and ``max_resolved_cell_fraction`` say how far
an unfinished episode got, and ``average_obstacle_collisions`` and
``average_new_cells_visited`` describe how it moved.

Recorded visualization
----------------------

.. image:: ../images/occupancy_grid_mapping_visualization.gif
   :alt: Three grids side by side - the robot's occupancy grid, the belief's per-cell occupancy probability, and the true map with the robot's pose.
   :width: 100%

The left grid is the map the robot has built, shaded from white (believed free)
through grey (unknown) to dark (believed occupied). The middle grid is the
belief over whole worlds, projected to the chance each cell is occupied — a
particle here is an entire map, and a hundred overlaid maps are not a picture of
anything, so the per-cell marginal is what is drawn. The right grid is the real
map with the robot's pose and heading, shown for review; the planner never sees
it.

Panels show the state before the displayed action, and the caption reports that
action and the information it gained. A planner that is exploring well drives
the grey out of the left grid; a filter that is tracking well makes the middle
grid approach the right one.

No vectorized model
-------------------

This environment has no torch vectorized model and no C++ native model, so it
cannot be run under VOPP. ``PFT_DPW`` takes the scalar ``Environment`` directly
and works on it as it stands.
