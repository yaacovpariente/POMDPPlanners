Saving and Loading Policies
===========================

The planners here are online: they start a new search at every step and keep
nothing between decisions. So for most planners there is no learned model to
save. What you save is the *configuration*: the planner's parameters plus the
environment it was built for. That is enough to rebuild a planner with the same
``config_id`` later.

The learned planners, :doc:`../planners/beta_zero` and
:doc:`../planners/constrained_zero`, are the exception. They also carry network
weights. ``BetaZero.save`` writes those too, and ``ConstrainedZero`` inherits
it.

This page covers both, and the other things a run leaves on disk.

Saving a planner's configuration
--------------------------------

Every planner has ``save`` and ``load``. ``save`` writes one JSON file. It
holds the planner's class, its constructor parameters, the environment's class
and parameters, the action sampler if the planner has one, and the planner's
``config_id``.

.. code-block:: python

   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP

   env = TigerPOMDP(discount_factor=0.95)
   planner = POMCP(
       environment=env,
       discount_factor=0.95,
       depth=10,
       exploration_constant=50.0,
       name="tiger_pomcp",
       time_out_in_seconds=2.0,
   )

   path = planner.save("configs/tiger_pomcp.json")
   restored = POMCP.load(path)

   assert restored.config_id == planner.config_id

``load`` rebuilds the environment too, so the file alone is enough. It warns if
the rebuilt planner's ``config_id`` differs from the saved one, which means
some parameter did not survive the round trip.

With no path, ``save`` writes to
``saved_policies/<environment>/<planner class>/<name>_<timestamp>.json`` in the
current directory.

Only constructor parameters are saved, read back from the planner's attributes
of the same name. A custom planner must store each constructor argument under
its own name for this to work (see :doc:`custom_planners`).

Saving a trained BetaZero network
---------------------------------

``BetaZero.save`` writes a directory, not a file:

- ``policy_config.json`` — the search and network parameters,
- ``network_weights.pt`` — the PyTorch weights,
- ``normalization_stats.json`` — the input and value statistics, when
  normalization is on (the default).

There is no matching ``BetaZero.load``: the inherited ``Policy.load`` expects
the single-file format above and cannot read this directory. To restore a
trained planner, build it again with the same arguments, then load the weights
and the statistics into it:

.. code-block:: python

   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero import BetaZero
   from POMDPPlanners.planners.mcts_planners.beta_zero.beta_zero_action_sampler import (
       BetaZeroActionSampler,
   )
   from POMDPPlanners.utils.action_samplers import DiscreteActionSampler


   def make_planner(env):
       actions = env.get_actions()
       return BetaZero(
           environment=env,
           discount_factor=0.95,
           depth=10,
           name="tiger_betazero",
           action_sampler=BetaZeroActionSampler(
               fallback_sampler=DiscreteActionSampler(actions), actions=actions
           ),
           time_out_in_seconds=2.0,
           state_dim=1,
       )


   env = TigerPOMDP(discount_factor=0.95)
   planner = make_planner(env)
   # ... train it with POMDPPlanners.training.PolicyTrainer ...
   save_dir = planner.save("checkpoints/tiger_betazero")

   restored = make_planner(env)
   restored.network.load_weights(save_dir / "network_weights.pt")
   restored.load_normalization_stats(save_dir)

The statistics matter. The network was trained on normalized inputs and
predicts normalized values. Without the statistics it is fed raw inputs and its
values are not mapped back to the reward scale, so its estimates are wrong even
though the weights loaded without error.

During training, the ``ModelCheckpoint`` callback in
``POMDPPlanners.training`` calls ``save`` for you whenever a monitored loss
improves.

Saving an environment
---------------------

An environment converts to a dictionary and back. ``Policy.save`` uses
this; you can use it directly to store an environment next to your results.

.. code-block:: python

   from POMDPPlanners.core.environment import Environment

   data = env.to_dict()  # class, module, params, config_id
   same_env = Environment.from_dict(data)

What a simulation run saves
---------------------------

You do not save simulation results by hand. A run through
``LocalSimulationsAPI`` writes everything under the ``cache_dir_path`` you pass
to its run method:

- every finished episode, in a disk cache keyed by the environment, planner and
  initial belief, so a rerun reuses it (see :doc:`../core/simulations`);
- an MLflow store in ``mlruns/``, with the parameters, the metrics and the
  per-episode artifacts, which ``pomdp-report`` serves
  as a website;
- logs in ``logs/``.

A single episode's :class:`~POMDPPlanners.core.simulation.history.History` also has
``to_dict`` and ``History.from_dict``, if you need one episode as a dictionary.

See also
--------

- :doc:`custom_planners` — what a custom planner needs for ``save`` to work.
- :doc:`../core/simulations` — running batches, and how the cache is keyed.
