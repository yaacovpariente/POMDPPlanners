Base Environment Class
======================

An environment defines the problem a planner has to solve: what the hidden state
is, what the agent may do, what it gets to see, and what it is paid. Every
environment in the package subclasses
:class:`~POMDPPlanners.core.environment.Environment`, so any planner can run on
any of them. Pick one from the catalog below, then read its page for the
settings and the reward numbers.

Creating an environment
-----------------------

Construct an environment directly from its class, or by name through the
registry. The registry exists so a configuration file can name an environment
as a string.

.. code-block:: python

   from POMDPPlanners.environments import get_environment
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

   env = TigerPOMDP(discount_factor=0.95)
   same_env = get_environment("TigerPOMDP", discount_factor=0.95)

.. note::

   ``get_environment`` covers only the classes in ``ENVIRONMENT_REGISTRY``, which
   include
   ``OccupancyGridMappingPOMDP``, ``ChicheckInvadersPOMDP`` and
   ``FirefightingPOMDP``. Import ``BattleshipPOMDP`` and
   ``SnakePOMDP`` directly. ``ContinuousPushPOMDP``,
   ``ContinuousPushPOMDPDiscreteActions`` and the realistic worlds (CARLA,
   Isaac Lab, nuPlan, Racetrack) are not in the registry — import those classes
   directly.

Environment catalog
-------------------

"State" is not a declared property of an environment — :class:`SpaceInfo
<POMDPPlanners.core.environment.SpaceInfo>` records only the action and
observation space types. The state column below describes what the code
stores.

A world and its variants share one page. :doc:`maze` covers the discrete,
continuous and T-maze versions; :doc:`light_dark`, :doc:`push` and
:doc:`laser_tag` each cover their continuous and discrete-action forms; and
:doc:`realistic` covers Racetrack, CARLA, Isaac Lab and nuPlan together. The
Guide column says which page to open for a given class.

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
   * - ``CaptureTheFlagPOMDP``
     - Two teams, two flags: steal theirs and get home without being tagged.
     - ``[players, flag home, carriers, counters, score]``
     - discrete
     - continuous
     - none
     - :doc:`capture_the_flag`
   * - ``OccupancyGridMappingPOMDP``
     - Explore an unknown grid world, paid for the entropy it maps away.
     - ``[step, pose, true map, log-odds map]``
     - discrete
     - continuous
     - none
     - :doc:`occupancy_grid_mapping`
   * - ``ChicheckInvadersPOMDP``
     - Shoot down a diving flock seen through a split camera/radar reading.
     - ``[ship, chickens]``
     - discrete
     - discrete
     - none
     - :doc:`chicheck_invaders`
   * - ``FirefightingPOMDP``
     - Put out a wind-driven grid fire with several partially sighted robots.
     - ``[step, robots, wind, cells]``
     - discrete
     - discrete
     - none
     - :doc:`firefighting`
   * - ``SnakePOMDP``
     - Grow the snake to a target length while the food stays hidden.
     - ``[status, length, counter, food, body]``
     - discrete
     - discrete
     - none
     - :doc:`snake`
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
     - :doc:`sanity`
   * - ``MountainCarPOMDP``
     - Build momentum up a hill from noisy readings.
     - ``[position, velocity]``
     - discrete
     - continuous
     - none
     - :doc:`mountain_car`
   * - ``RacetrackPOMDP``
     - A matched MDP/POMDP pair on one track, to isolate partial observability.
     - HighwayEnv vehicle state
     - discrete
     - continuous
     - ``highway-env``
     - :doc:`realistic`
   * - ``CarlaPOMDP``
     - Urban driving against a running CARLA server.
     - ego, agent, light and goal slots
     - discrete
     - dict of 5 sensors
     - CARLA server
     - :doc:`realistic`
   * - ``IsaacLabPOMDP``
     - Any registered ``Isaac-*-v0`` task, observed through a sensor buffer.
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

The Environment interface
-------------------------

A planner talks to an environment only through the methods below. Write a new
environment against them and every planner in the package can run on it; see
:doc:`custom` for a worked example.

**Constructor.** ``Environment.__init__`` takes:

- ``discount_factor`` — discount for future rewards, in :math:`(0, 1]`.
- ``name`` — the environment's identifier, used in logs and results.
- ``space_info`` — a ``SpaceInfo`` naming the action and observation space
  types (see below).
- ``reward_range`` — optional ``(min_reward, max_reward)``. It is checked on
  construction: two numbers, no NaN, and min no larger than max.
- ``output_dir``, ``debug``, ``use_queue_logger`` — logging options.

**Methods every environment must implement.** These are abstract on the base
class:

- ``sample_next_state(state, action, n_samples=1)`` — draw next states.
- ``sample_observation(next_state, action, n_samples=1)`` — draw observations.
  The observation is conditioned on the state *after* the action.
- ``transition_log_probability(state, action, next_states)`` and
  ``observation_log_probability(next_state, action, observations)`` — log
  probabilities, one per candidate. Particle beliefs reweight with these.
- ``reward(state, action, next_state=None)`` — the immediate reward.
  ``next_state`` is passed when the caller already drew it, so a reward that
  depends on the outcome uses the same draw as the trajectory.
- ``is_terminal(state)`` — whether the episode ends in ``state``.
- ``initial_state_dist()`` and ``initial_observation_dist()`` — where an
  episode starts.
- ``is_equal_observation(observation1, observation2)`` — observation equality.
- ``hash_action(action)`` — a hashable key for an action. Tree planners index
  action children by this key.

With ``n_samples=1`` the two samplers return the value itself, not a list of
one.

**Methods with a default.** Override them when the default is too slow or does
not fit:

- ``hash_observation`` — a hashable key consistent with
  ``is_equal_observation``. The default returns the observation itself, so an
  environment with ``np.ndarray`` observations must override it.
- ``sample_next_state_batch``, ``observation_log_probability_per_state`` and
  ``reward_batch`` — batched forms used by particle filters. The defaults loop
  in Python over the single-state methods.
- ``sample_next_step(state, action)`` — draws the next state and observation
  and computes the reward in one call.
- ``step_info`` and ``get_metric_specs`` — the per-step measurements and the
  metrics built from them. See :doc:`custom` for what every environment should
  report.
- ``episode_visualizer()`` — how this environment's episodes are shown, or
  ``None``.

**Discrete actions.** ``DiscreteActionsEnvironment`` adds one abstract method,
``get_actions()``, which lists every action. Planners that loop over actions,
such as POMCP, need it.

**Configuration, equality and caching.** ``config_id`` is a deterministic
identifier built from the environment's public attributes; private attributes,
loggers and bound methods are skipped. Two environments of the same class with
the same public attributes compare equal and hash equal. The simulation cache
keys finished episodes on this identifier, so changing an attribute gives a
new cache entry instead of reusing results computed under the old value.

**Environment generators.** ``EnvironmentGenerator`` is a factory base
class with one abstract method, ``generate_environment()``. Use it to produce
environments with randomized parameters.

Space types
-----------

``SpaceInfo`` holds two ``SpaceType`` values: ``action_space`` and
``observation_space``. ``SpaceType`` is one of ``DISCRETE`` (a finite,
countable set), ``CONTINUOUS`` (real-valued) or ``MIXED`` (both). Some
planners read ``space_info`` to decide how to treat actions; the ICVaR planners,
for example, branch on whether the action space is discrete. There is no
state-space type — a state is whatever the environment stores.

The 3D episode viewer
---------------------

Most environment pages open with a 3D replay of one recorded episode, planned
by PFT-DPW. The page embeds it with the ``.. episode-viewer::`` directive, which
reads a trace JSON written by the environment's ``TraceVisualizer``. The
traces live in ``docs/environments/traces/`` and are regenerated by
``scripts/generate_docs_traces.py``.

The replay loads its trace with a web request, and a browser will not let a
``file://`` page do that. Serve the built docs over HTTP to see it:

.. code-block:: bash

   python -m http.server -d docs/_build/html

Example
-------

.. code-block:: python

   from POMDPPlanners.core.environment import SpaceType
   from POMDPPlanners.environments import get_environment
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

   env = TigerPOMDP(discount_factor=0.95)
   same_env = get_environment("TigerPOMDP", discount_factor=0.95)

   # Same class and same configuration: equal, with the same hash.
   assert env == same_env
   assert hash(env) == hash(same_env)
   assert env.config_id == same_env.config_id

   assert env.space_info.action_space == SpaceType.DISCRETE
   print(env.get_actions())

Parameters
----------

.. autoclass:: POMDPPlanners.core.environment.Environment
   :members:
   :show-inheritance:

.. autoclass:: POMDPPlanners.core.environment.DiscreteActionsEnvironment
   :members: get_actions
   :show-inheritance:

.. autoclass:: POMDPPlanners.core.environment.EnvironmentGenerator
   :members:

.. autoclass:: POMDPPlanners.core.environment.SpaceInfo
   :members:

.. autoclass:: POMDPPlanners.core.environment.SpaceType

See also
--------

- :doc:`custom` — write your own environment against this interface.
- :doc:`realistic` — environments that wrap an external simulator.
- :doc:`../planners/base` — the planners that run on these environments.
