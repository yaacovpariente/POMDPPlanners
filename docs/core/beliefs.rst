Belief States
=============

Belief states represent the agent's uncertainty about the true state of the environment. POMDPPlanners provides flexible belief representations suitable for different problem types.

Belief Representations
----------------------

**Particle Filter Beliefs**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.core.belief.WeightedParticleBelief
   POMDPPlanners.core.belief.UnweightedParticleBelief

**Utility Functions**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.core.belief.get_initial_belief

Base Belief Interface
--------------------

All belief representations inherit from the base Belief class:

.. autoclass:: POMDPPlanners.core.belief.Belief
   :members:
   :undoc-members:
   :show-inheritance:

Particle Filter Beliefs
-----------------------

**WeightedParticleBelief**
   - Maintains particles with associated weights
   - Efficient for complex observation models
   - Supports importance sampling
   - Handles continuous state spaces well

**UnweightedParticleBelief**
   - All particles have equal weight
   - Simpler implementation
   - Good for uniform beliefs
   - Faster sampling operations

Belief Operations
----------------

**Sampling from Beliefs**

.. code-block:: python

   from POMDPPlanners.core.belief import WeightedParticleBelief
   import numpy as np

   # Create belief with weighted particles
   states = [0, 1, 2]
   particles = [0, 0, 1, 1, 2]
   weights = [0.3, 0.2, 0.2, 0.2, 0.1]

   belief = WeightedParticleBelief(
       particles=particles,
       weights=weights,
       state_space=states
   )

   # Sample from belief
   state_sample = belief.sample()
   print(f"Sampled state: {state_sample}")

   # Get state probabilities
   probabilities = belief.get_state_probabilities()
   print(f"State probabilities: {probabilities}")

**Creating Initial Beliefs**

.. code-block:: python

   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

   env = TigerPOMDP()

   # Create uniform initial belief
   belief = get_initial_belief(env, n_particles=1000)

   # Sample from initial belief
   initial_state = belief.sample()

Belief Updates
-------------

Beliefs are updated based on actions and observations:

.. code-block:: python

   # After taking action and receiving observation
   action = "listen"
   observation = "hear_left"

   # Update belief (typically done by planner)
   updated_belief = planner.update_belief(
       current_belief=belief,
       action=action,
       observation=observation
   )

Advanced Belief Operations
-------------------------

**State Probability Queries**

.. code-block:: python

   # Get probability of specific state
   prob_tiger_left = belief.get_state_probabilities()["tiger_left"]

   # Check if belief is concentrated
   max_prob = max(belief.get_state_probabilities().values())
   is_concentrated = max_prob > 0.8

**Effective Sample Size**

.. code-block:: python

   # For weighted particle beliefs
   if hasattr(belief, 'effective_sample_size'):
       eff_size = belief.effective_sample_size()
       if eff_size < 100:  # Threshold for resampling
           print("Consider particle resampling")

**Belief Entropy**

.. code-block:: python

   import numpy as np

   probs = list(belief.get_state_probabilities().values())
   entropy = -sum(p * np.log(p) for p in probs if p > 0)
   print(f"Belief entropy: {entropy:.3f}")

Working with Continuous States
------------------------------

For continuous state spaces, particles represent state samples:

.. code-block:: python

   from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
   import numpy as np

   env = CartPolePOMDP()

   # Create belief with continuous state particles
   particles = [
       np.array([0.1, 0.0, 0.05, 0.0]),  # [position, velocity, angle, angular_velocity]
       np.array([0.0, 0.1, -0.02, 0.1]),
       np.array([-0.05, -0.05, 0.0, -0.05])
   ]

   belief = WeightedParticleBelief(
       particles=particles,
       weights=[0.4, 0.3, 0.3],
       state_space=None  # Continuous space
   )

   # Sample continuous state
   continuous_state = belief.sample()
   print(f"Sampled state: {continuous_state}")

Custom Belief Implementations
-----------------------------

To create custom belief representations:

.. code-block:: python

   from POMDPPlanners.core.belief import Belief

   class GaussianBelief(Belief):
       def __init__(self, mean, covariance):
           self.mean = mean
           self.covariance = covariance

       def sample(self):
           return np.random.multivariate_normal(self.mean, self.covariance)

       def get_state_probabilities(self):
           # For continuous beliefs, this might return density estimates
           # or discretized approximations
           pass

Performance Considerations
-------------------------

**Particle Count**
   - More particles → better approximation, slower computation
   - Typical range: 100-10,000 particles
   - Adjust based on problem complexity

**Resampling**
   - Monitor effective sample size
   - Resample when weights become too uneven
   - Use systematic resampling for efficiency

**Memory Usage**
   - Particle beliefs scale with particle count
   - Consider state compression for large states
   - Use appropriate data types (float32 vs float64)

See Also
--------

- :doc:`../examples/beliefs` - Belief usage examples
- :doc:`planners` - How planners use beliefs
- :doc:`../api/core` - Complete API reference