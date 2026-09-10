Environments
============

An environment defines the problem a planner has to solve: what the hidden state
is, what the agent may do, what it gets to see, and what it is paid. Pick one
from the catalog below, then read its guide for the settings and the reward
numbers.

Every environment listed here is constructed directly, or by name through the
registry:

.. code-block:: python

   from POMDPPlanners.environments import get_environment

   env = get_environment("TigerPOMDP", discount_factor=0.95)

.. note::

   ``get_environment`` covers the classic suite only, which now includes
   ``OccupancyGridMappingPOMDP``. Import ``BattleshipPOMDP``
   directly. ``ContinuousPushPOMDP``,
   ``ContinuousPushPOMDPDiscreteActions`` and the realistic worlds (CARLA,
   Isaac Lab, nuPlan, Racetrack) are not in the registry — import those classes
   directly.

Environment catalog
-------------------

"State" is not a declared property of an environment — :class:`SpaceInfo
<POMDPPlanners.core.environment.SpaceInfo>` records only the action and
observation space types. The state column below describes what the code
actually stores.

.. list-table::
   :header-rows: 1
   :widths: 16 26 14 11 11 12 10

   * - Environment
     - Purpose
     - State
     - Actions
     - Observations
     - Extra dependency
     - Guide
   * - ``TigerPOMDP``
     - Decide which door hides the tiger from noisy listening.
     - discrete label
     - discrete
     - discrete
     - none
     - :doc:`tiger`
   * - ``RockSamplePOMDP``
     - Sample good rocks with a distance-degraded sensor, then exit east.
     - vector ``[row, col, rocks…]``
     - discrete
     - discrete
     - none
     - :doc:`rock_sample`
   * - ``BattleshipPOMDP``
     - Find every cell of a hidden fleet with exact hit/miss probes.
     - occupancy and probe flags
     - discrete
     - discrete
     - none
     - :doc:`battleship`
   * - ``OccupancyGridMappingPOMDP``
     - Explore an unknown grid world, paid for the entropy it maps away.
     - ``[step, pose, true map, log-odds map]``
     - discrete
     - continuous
     - none
     - :doc:`occupancy_grid_mapping`
   * - ``PacManPOMDP``
     - Clear every pellet while dodging noisily-observed ghosts.
     - vector (pac, ghosts, pellets)
     - discrete
     - discrete
     - none
     - :doc:`pacman`
   * - ``DiscreteLightDarkPOMDP``
     - Reach a goal on a grid, detouring through beacons to localize.
     - grid position
     - discrete
     - discrete
     - none
     - :doc:`light_dark`
   * - ``ContinuousLightDarkPOMDP``
     - The same trade-off in continuous 2-D.
     - ``[x, y]``
     - continuous
     - continuous
     - none
     - :doc:`light_dark`
   * - ``ContinuousLightDarkPOMDPDiscreteActions``
     - The continuous Light-Dark world with four discrete moves.
     - ``[x, y]``
     - discrete
     - continuous
     - none
     - :doc:`light_dark`
   * - ``DiscreteMazePOMDP``
     - Remember a cue while navigating a generated grid maze.
     - ``[x, y, goal_side, cue_phase]``
     - discrete
     - discrete
     - none
     - :doc:`maze`
   * - ``ContinuousMazePOMDP``
     - Navigate the same maze with bounded displacements.
     - ``[x, y, goal_side, cue_phase]``
     - continuous
     - discrete
     - none
     - :doc:`maze`
   * - ``TMazePOMDP``
     - Remember a cue until choosing an arm of a T corridor.
     - ``[x, y, goal_side, cue_phase]``
     - discrete
     - discrete
     - none
     - :doc:`maze`
   * - ``CartPolePOMDP``
     - Balance a pole seeing only noisy sensor readings.
     - ``[x, ẋ, θ, θ̇]``
     - discrete
     - continuous
     - none
     - :doc:`cartpole`
   * - ``PushPOMDP``
     - Push an object onto a target corner of a grid.
     - ``[robot, object, target]``
     - discrete
     - continuous
     - none
     - :doc:`push`
   * - ``ContinuousPushPOMDP``
     - The same task with a circular robot and free 2-D pushes.
     - ``[robot, object, target]``
     - continuous
     - continuous
     - none
     - :doc:`push`
   * - ``ContinuousPushPOMDPDiscreteActions``
     - The continuous Push world with four discrete moves.
     - ``[robot, object, target]``
     - discrete
     - continuous
     - none
     - :doc:`push`
   * - ``LaserTagPOMDP``
     - Corner an evading opponent on a walled grid and tag it.
     - ``[robot, opponent, done]``
     - discrete
     - continuous
     - none
     - :doc:`laser_tag`
   * - ``ContinuousLaserTagPOMDP``
     - The same pursuit in continuous space.
     - ``[robot, opponent, done]``
     - continuous
     - continuous
     - none
     - :doc:`laser_tag`
   * - ``ContinuousLaserTagPOMDPDiscreteActions``
     - The continuous LaserTag world with five discrete actions.
     - ``[robot, opponent, done]``
     - discrete
     - continuous
     - none
     - :doc:`laser_tag`
   * - ``SafeAntVelocityPOMDP``
     - Move fast while staying under a velocity safety limit.
     - ``[x, y, vx, vy]``
     - discrete
     - continuous
     - none
     - :doc:`safety_ant_velocity`
   * - ``SanityPOMDP``
     - Two states, perfect observations — a debugging baseline.
     - discrete label
     - discrete
     - discrete
     - none
     - :doc:`simple`
   * - ``MountainCarPOMDP``
     - Build momentum up a hill from noisy readings.
     - ``[position, velocity]``
     - discrete
     - continuous
     - none
     - :doc:`simple`
   * - ``RacetrackPOMDP``
     - A matched MDP/POMDP pair on one track, to isolate partial observability.
     - HighwayEnv vehicle state
     - discrete
     - continuous
     - ``highway-env``
     - :doc:`realistic`
   * - ``CarlaPOMDP``
     - Urban driving against a live CARLA server.
     - ego, agent, light and goal slots
     - discrete
     - dict of 5 sensors
     - CARLA server
     - :doc:`realistic`
   * - ``IsaacLabPOMDP``
     - Any registered ``Isaac-*-v0`` task, with a real sensor observation.
     - physics state vector
     - configurable
     - configurable
     - Isaac Sim
     - :doc:`realistic`
   * - ``NuPlanPOMDP``
     - Closed-loop driving on recorded nuPlan scenarios.
     - ego, agent and light slots
     - discrete
     - ``{ego, agents}`` dict
     - ``nuplan-devkit``
     - :doc:`realistic`

Guides
------

.. toctree::
   :maxdepth: 1

   tiger
   rock_sample
   battleship
   occupancy_grid_mapping
   pacman
   maze
   light_dark
   cartpole
   push
   laser_tag
   safety_ant_velocity
   simple
   realistic

API listing
-----------

**Classic benchmark problems**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.environments.tiger_pomdp.TigerPOMDP
   POMDPPlanners.environments.sanity_pomdp.SanityPOMDP

**Control and navigation**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp.CartPolePOMDP
   POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp.MountainCarPOMDP
   POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp.ContinuousLightDarkPOMDP
   POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp.DiscreteLightDarkPOMDP
   POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp.ContinuousLightDarkPOMDPDiscreteActions

**Manipulation**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.environments.push_pomdp.PushPOMDP
   POMDPPlanners.environments.push_pomdp.ContinuousPushPOMDP
   POMDPPlanners.environments.push_pomdp.continuous_push_pomdp.ContinuousPushPOMDPDiscreteActions
   POMDPPlanners.environments.safety_ant_velocity_pomdp.SafeAntVelocityPOMDP

**Information gathering and pursuit**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.environments.rock_sample_pomdp.RockSamplePOMDP
   POMDPPlanners.environments.occupancy_grid_mapping_pomdp.OccupancyGridMappingPOMDP
   POMDPPlanners.environments.pacman_pomdp.PacManPOMDP
   POMDPPlanners.environments.laser_tag_pomdp.LaserTagPOMDP
   POMDPPlanners.environments.laser_tag_pomdp.ContinuousLaserTagPOMDP
   POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp.ContinuousLaserTagPOMDPDiscreteActions

**Realistic simulators**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.environments.carla_pomdp.carla_pomdp.CarlaPOMDP
   POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp.IsaacLabPOMDP
   POMDPPlanners.environments.nuplan_pomdp.nuplan_pomdp.NuPlanPOMDP
   POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp.RacetrackPOMDP

See also
--------

- :doc:`custom` — write your own environment against the base interface.
- :doc:`../core/planners` — the planners that run on these environments.
