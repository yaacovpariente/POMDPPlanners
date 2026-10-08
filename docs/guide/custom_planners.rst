Custom Planners
===============

A planner is any subclass of :class:`~POMDPPlanners.core.policy.Policy`. If
your class implements three methods, the episode runner, the simulation API,
caching and hyperparameter tuning all accept it like a built-in planner.

The planner interface
---------------------

``Policy`` has one main method and two class methods that describe it:

``action(belief)``
   Takes the current belief and returns a pair: a list of actions and a
   :class:`~POMDPPlanners.core.policy.PolicyRunData`. A closed-loop planner
   returns a list of length one. An open-loop planner returns the whole
   sequence (see :doc:`../planners/open_loop`), and the episode runner executes
   it before it calls ``action`` again.

``get_space_info()``
   Returns a :class:`~POMDPPlanners.core.policy.PolicySpaceInfo` that names the
   action and observation spaces the planner accepts. The constructor checks it
   against the environment. A planner that declares ``SpaceType.DISCRETE``
   actions raises ``ValueError`` on an environment with continuous actions, so
   a bad pairing fails at construction, not halfway through a batch.

``get_info_variable_names()``
   Lists the names of the metrics that ``action`` reports in its
   ``PolicyRunData``. Hyperparameter tuning reads this list to know which
   metrics it can optimize before any episode runs.

The base constructor takes ``environment``, ``discount_factor`` and ``name``.
``name`` labels the planner in logs, results tables and MLflow.

Writing your own planner
------------------------

The planner below scores each action by its mean immediate reward over states
sampled from the belief, and picks the best. It scores only the next reward,
so it gives listening on Tiger no credit for what the observation reveals. It
is still a full example of every part a planner needs.

.. code-block:: python

   from typing import Any, List, Tuple

   import numpy as np

   from POMDPPlanners.core.environment import SpaceType
   from POMDPPlanners.core.policy import (
       Policy,
       PolicyInfoVariable,
       PolicyRunData,
       PolicySpaceInfo,
   )


   class OneStepLookahead(Policy):
       """Pick the action with the best mean immediate reward under the belief."""

       def __init__(self, environment, discount_factor: float, name: str, n_samples: int):
           super().__init__(environment=environment, discount_factor=discount_factor, name=name)
           self.n_samples = n_samples

       def action(self, belief) -> Tuple[List[Any], PolicyRunData]:
           actions = self.environment.get_actions()
           values = []
           for action in actions:
               rewards = [
                   self.environment.sample_next_step(state=belief.sample(), action=action)[2]
                   for _ in range(self.n_samples)
               ]
               values.append(np.mean(rewards))
           best = int(np.argmax(values))
           run_data = PolicyRunData(
               info_variables=[PolicyInfoVariable(name="best_value", value=float(values[best]))]
           )
           return [actions[best]], run_data

       @classmethod
       def get_space_info(cls) -> PolicySpaceInfo:
           return PolicySpaceInfo(
               action_space=SpaceType.DISCRETE, observation_space=SpaceType.MIXED
           )

       @classmethod
       def get_info_variable_names(cls) -> List[str]:
           return ["best_value"]

Run it like any other planner:

.. code-block:: python

   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.simulations.episodes import run_episode

   env = TigerPOMDP(discount_factor=0.95)
   belief = get_initial_belief(env, n_particles=200)
   planner = OneStepLookahead(env, discount_factor=0.95, name="lookahead", n_samples=50)

   actions, run_data = planner.action(belief)
   history = run_episode(
       environment=env, policy=planner, initial_belief=belief, num_steps=5, logger=None
   )

Four rules keep a custom planner working with the rest of the package:

- **Store every constructor argument as an attribute of the same name.**
  :meth:`Policy.save <POMDPPlanners.core.policy.Policy.save>` rebuilds the
  constructor call by reading each argument back from the attribute with its
  name, so an argument stored under another name is lost on save.
- **Keep run-time scratch in attributes that start with an underscore.**
  ``config_id`` hashes every public attribute to key the simulation cache, so
  a public attribute that changes during planning changes the key.
- **Report the same metric names you declare.** A name in
  ``get_info_variable_names`` that ``action`` never reports cannot be tuned
  against.
- **Use the belief and the environment's generative model, never the true
  state.** The planner only sees ``belief``. The episode runner keeps the
  true state outside the planner on purpose.

Reusing the MCTS machinery
--------------------------

A new tree-search planner does not have to write the budget loop.
:class:`~POMDPPlanners.planners.planners_utils.path_simulations_policy_arena.ArenaPathSimulationPolicy`
runs simulations until ``n_simulations`` or ``time_out_in_seconds`` is used up,
picks the final action from the root, and reports the standard tree metrics
(``root_visit_count``, ``tree_max_depth`` and the rest). A subclass implements
``_simulate_path(tree, belief_id, depth)``, one simulation from the root, plus
``get_space_info``. The tree is a column store addressed by integer node ids.
Built-in planners such as :doc:`../planners/pomcp`,
:doc:`../planners/pft_dpw` and :doc:`../planners/sparse_pft` subclass it or its progressive-widening variant
``ArenaDoubleProgressiveWideningMCTSPolicy``. Read one of them before you
start: their code shows how nodes are expanded and values are backed up.

``PathSimulationPolicy`` (in ``POMDPPlanners.planners.planners_utils.path_simulations_policy``)
is the same contract on a tree of node objects,
``_simulate_path(belief_node, depth)``.

To call a new planner by name through
``POMDPPlanners.planners.get_policy``, add it to ``POLICY_REGISTRY`` in
``POMDPPlanners/planners/__init__.py``.

See also
--------

- :doc:`../planners/base` — the built-in planners and what each supports.
- :doc:`saving_loading` — saving a planner's configuration to JSON.
