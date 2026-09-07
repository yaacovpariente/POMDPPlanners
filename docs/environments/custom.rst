Create a custom environment
===========================

A custom environment is a subclass of :class:`Environment
<POMDPPlanners.core.environment.Environment>`. If the action set is a finite
list, subclass :class:`DiscreteActionsEnvironment
<POMDPPlanners.core.environment.DiscreteActionsEnvironment>` instead — it adds
``get_actions`` and lets the discrete-action planners enumerate.

What you must implement
-----------------------

The base class is a generative model, not a table: planners sample from it far
more often than they ask for a probability, so the sampling methods are the ones
that need to be fast.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Method
     - What it returns
   * - ``initial_state_dist()``
     - Distribution over start states.
   * - ``initial_observation_dist()``
     - Distribution over the observation before the first action.
   * - ``sample_next_state(state, action, n_samples=1)``
     - ``n_samples`` draws from :math:`T(\cdot \mid s, a)`.
   * - ``sample_observation(next_state, action, n_samples=1)``
     - ``n_samples`` draws from :math:`Z(\cdot \mid s', a)`.
   * - ``transition_log_probability(state, action, next_states)``
     - Array of shape ``(N,)`` — log :math:`T(s'_i \mid s, a)`.
   * - ``observation_log_probability(next_state, action, observations)``
     - Array of shape ``(N,)`` — log :math:`Z(o_i \mid s', a)`. This is what
       weights the particles in a belief update.
   * - ``reward(state, action, next_state=None)``
     - Scalar reward.
   * - ``is_terminal(state)``
     - Whether the episode has ended.
   * - ``is_equal_observation(o1, o2)``
     - Observation equality. Needed because observations are often numpy
       arrays, where ``==`` is elementwise.
   * - ``hash_action(action)``
     - A hashable key for an action, used to index search-tree children.
   * - ``get_actions()``
     - Only on ``DiscreteActionsEnvironment``: the list of actions.

Traps worth knowing before you start
------------------------------------

- **``reward_range`` is only structurally validated.** The constructor checks
  the tuple's shape, not that your rewards actually fall inside it. The rollout
  check that catches a wrong range is a conformance test, and it runs only over
  the environments listed in ``ENV_BUILDERS`` in
  ``tests/test_environments/test_env_api_conformance.py`` — so you have to add
  yours there to get it. Until then, leave ``reward_range`` ``None`` rather
  than guessing.
- **``config_id`` must be stable.** It is derived from the attributes the
  instance stores, not from the constructor signature, and it is the cache key
  for simulation results. A constructor argument you never assign to ``self``
  has no effect on it; an attribute you set after construction does. Anything
  that changes between runs — a path, a timestamp, a seed drawn at
  construction — silently splits or, worse, merges cached results.
- **Log-probabilities, not probabilities.** Both ``*_log_probability`` methods
  return logs. Returning a probability produces beliefs that look plausible and
  are wrong.

A minimal working environment
-----------------------------

A hidden coin, a 90 %-accurate peek that costs 1, and a guess that pays ±10:

.. code-block:: python

   import numpy as np

   from POMDPPlanners.core.distributions import DiscreteDistribution
   from POMDPPlanners.core.environment import (
       DiscreteActionsEnvironment,
       SpaceInfo,
       SpaceType,
   )


   class CoinFlipPOMDP(DiscreteActionsEnvironment):
       """Guess whether a hidden coin is heads, seeing a 90%-accurate peek."""

       STATES = ["heads", "tails"]
       ACTIONS = ["peek", "say_heads", "say_tails"]

       def __init__(self, discount_factor: float = 0.95, peek_accuracy: float = 0.9):
           super().__init__(
               discount_factor=discount_factor,
               name="CoinFlipPOMDP",
               space_info=SpaceInfo(
                   action_space=SpaceType.DISCRETE,
                   observation_space=SpaceType.DISCRETE,
               ),
               reward_range=(-10.0, 10.0),
           )
           self.peek_accuracy = peek_accuracy

       def get_actions(self):
           return list(self.ACTIONS)

       def initial_state_dist(self):
           return DiscreteDistribution(list(self.STATES), np.array([0.5, 0.5]))

       def initial_observation_dist(self):
           return DiscreteDistribution(["saw_nothing"], np.array([1.0]))

       # The coin does not change; only the agent's knowledge does.
       def sample_next_state(self, state, action, n_samples=1):
           return [state] * n_samples

       def transition_log_probability(self, state, action, next_states):
           return np.log(np.array([s == state for s in next_states], dtype=float))

       def _observation_probs(self, next_state, action):
           if action != "peek":
               return {"saw_nothing": 1.0}
           correct = "saw_heads" if next_state == "heads" else "saw_tails"
           wrong = "saw_tails" if next_state == "heads" else "saw_heads"
           return {correct: self.peek_accuracy, wrong: 1.0 - self.peek_accuracy}

       def sample_observation(self, next_state, action, n_samples=1):
           probs = self._observation_probs(next_state, action)
           values = list(probs)
           weights = np.array([probs[v] for v in values])
           return DiscreteDistribution(values, weights).sample(n_samples)

       def observation_log_probability(self, next_state, action, observations):
           probs = self._observation_probs(next_state, action)
           with np.errstate(divide="ignore"):
               return np.log(np.array([probs.get(o, 0.0) for o in observations], dtype=float))

       def is_equal_observation(self, observation1, observation2):
           return observation1 == observation2

       def hash_action(self, action):
           return action

       def reward(self, state, action, next_state=None):
           if action == "peek":
               return -1.0
           guessed = "heads" if action == "say_heads" else "tails"
           return 10.0 if guessed == state else -10.0

       def is_terminal(self, state):
           return False


   env = CoinFlipPOMDP()
   state = env.initial_state_dist().sample(1)[0]
   observation = env.sample_observation(state, "peek")[0]

Metrics
-------

An environment reports its own metrics through ``get_metric_specs`` and
``step_info``. At minimum, report a task-completion rate and why each episode
ended; without them a simulation batch can only tell you the mean return, which
does not distinguish "solved slowly" from "failed safely".

The base interface
------------------

.. autoclass:: POMDPPlanners.core.environment.Environment
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: POMDPPlanners.core.environment.DiscreteActionsEnvironment
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Space types
-----------

``SpaceInfo`` records the action and observation space types. There is no
declared state-space type — a state is whatever the environment stores.

.. autoclass:: POMDPPlanners.core.environment.SpaceType
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: POMDPPlanners.core.environment.SpaceInfo
   :members:
   :undoc-members:
   :no-index:

See also
--------

- :doc:`index` — the environments that already exist.
- :doc:`../core/beliefs` — the belief representations that consume
  ``observation_log_probability``.
