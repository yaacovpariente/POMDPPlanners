Environments
============

POMDP environments define the problem structure, including states, actions, observations, transitions, and rewards. POMDPPlanners provides both classic benchmark problems and modern challenging environments.

Core Environment Types
----------------------

**Classic Benchmark Problems**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.environments.tiger_pomdp.TigerPOMDP
   POMDPPlanners.environments.sanity_pomdp.SanityPOMDP

**Control & Navigation**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp.CartPolePOMDP
   POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp.MountainCarPOMDP
   POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp.ContinuousLightDarkPOMDP
   POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp.DiscreteLightDarkPOMDP

**Manipulation**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.environments.push_pomdp.PushPOMDP
   POMDPPlanners.environments.safety_ant_velocity_pomdp.SafeAntVelocityPOMDP

Environment Interface
---------------------

All environments inherit from the base Environment class:

.. autoclass:: POMDPPlanners.core.environment.Environment
   :members:
   :undoc-members:
   :show-inheritance:

Space Types
-----------

Environments can have different action and observation space types:

.. autoclass:: POMDPPlanners.core.environment.SpaceType
   :members:
   :undoc-members:

Key Environment Features
-----------------------

**State Spaces**
   - Discrete states (Tiger, Sanity)
   - Continuous states (CartPole, Mountain Car, Light-Dark)
   - Mixed representations

**Action Spaces**
   - Discrete actions (Tiger: listen/open doors)
   - Continuous actions (Light-Dark: movement vectors)
   - Hybrid approaches

**Observation Models**
   - Perfect observability (Sanity POMDP)
   - Noisy observations (Tiger: 85% accuracy)
   - Position-dependent noise (Light-Dark)
   - Sensor noise models (CartPole, Mountain Car)

**Reward Structures**
   - Goal-reaching rewards
   - Action costs
   - Safety penalties
   - Shaped rewards for learning

Creating Custom Environments
---------------------------

To create a custom environment, inherit from the base Environment class and implement the required methods:

.. code-block:: python

   from POMDPPlanners.core.environment import Environment
   import numpy as np

   class MyCustomPOMDP(Environment):
       def __init__(self, discount_factor=0.95):
           super().__init__(discount_factor)
           # Initialize your environment

       def get_states(self):
           # Return list of possible states
           return ["state1", "state2", "state3"]

       def get_actions(self):
           # Return list of possible actions
           return ["action1", "action2"]

       def initial_state_dist(self):
           # Return initial state distribution
           pass

       def state_transition_model(self, state, action):
           # Return state transition model
           pass

       def observation_model(self, next_state, action):
           # Return observation model
           pass

       def reward(self, state, action):
           # Return reward for state-action pair
           pass

       def is_terminal(self, state):
           # Return whether state is terminal
           return False

See Also
--------

- :doc:`../examples/environments` - Environment usage examples
- :doc:`planners` - Planning algorithms for these environments
- :doc:`../api/environments` - Complete API reference