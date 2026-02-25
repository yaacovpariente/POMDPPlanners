Planners
========

POMDP planners compute optimal actions given belief states. POMDPPlanners provides state-of-the-art algorithms from Monte Carlo Tree Search to sparse sampling approaches.

Planning Algorithm Categories
----------------------------

**Monte Carlo Tree Search (MCTS)**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.planners.mcts_planners.pomcp.POMCP
   POMDPPlanners.planners.mcts_planners.pft_dpw.PFT_DPW
   POMDPPlanners.planners.mcts_planners.sparse_pft.SparsePFT

**Sparse Sampling**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling_planner.SparseSamplingDiscreteActionsPlanner

**Open Loop Planning**

.. autosummary::
   :toctree: ../api/

   POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner.DiscreteActionSequencesPlanner

Planner Interface
----------------

All planners inherit from the base Policy class:

.. autoclass:: POMDPPlanners.core.policy.Policy
   :members:
   :undoc-members:
   :show-inheritance:

Algorithm Details
----------------

**POMCP (Partially Observable Monte Carlo Planning)**
   - Uses Upper Confidence Bounds (UCB) for action selection
   - Builds belief trees through simulation
   - Handles continuous observation spaces with particle filters
   - Excellent for problems with large observation spaces

**PFT-DPW (Particle Filter Trees with Double Progressive Widening)**
   - Extends POMCP with progressive widening
   - Gradually expands action and observation nodes
   - Better for continuous action spaces
   - Balances exploration and exploitation

**Sparse PFT**
   - Sparse sampling within particle filter trees
   - Efficient for large state/action spaces
   - Reduced computational requirements
   - Good performance with limited simulations

**Sparse Sampling**
   - Classical forward-search algorithm
   - Builds sparse lookahead trees
   - Provable performance guarantees
   - Simple and effective baseline

Choosing the Right Planner
--------------------------

**For Discrete Problems (Tiger, Sanity):**
   - POMCP: Excellent default choice
   - Sparse Sampling: Simple baseline
   - PFT-DPW: When you need progressive widening

**For Continuous Problems (CartPole, Light-Dark):**
   - PFT-DPW: Handles continuous actions well
   - POMCP: Good for continuous observations
   - Sparse PFT: When computational budget is limited

**For Large-Scale Problems:**
   - Sparse PFT: Efficient scaling
   - POMCP with limited simulations: Balance speed/quality

Basic Usage Example
------------------

.. code-block:: python

   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.core.belief import get_initial_belief

   # Create environment and planner
   env = TigerPOMDP()
   planner = POMCP(
       environment=env,
       num_simulations=1000,
       exploration_constant=50.0,
       depth=30
   )

   # Get initial belief and plan
   belief = get_initial_belief(env, n_particles=1000)
   action, run_data = planner.action(belief)

   print(f"Recommended action: {action}")
   print(f"Planning time: {run_data.info_variables['planning_time']:.3f}s")

Configuration Parameters
-----------------------

**Common Parameters:**
   - ``num_simulations``: Number of MCTS simulations
   - ``depth``: Maximum planning horizon
   - ``discount_factor``: Future reward discount
   - ``exploration_constant``: UCB exploration parameter

**POMCP-Specific:**
   - ``threshold``: Particle reinvigoration threshold
   - ``particle_filter_threshold``: Belief update threshold

**PFT-DPW-Specific:**
   - ``k_action``, ``alpha_action``: Action progressive widening
   - ``k_observation``, ``alpha_observation``: Observation progressive widening

Creating Custom Planners
------------------------

To implement a custom planner, inherit from the Policy base class:

.. code-block:: python

   from POMDPPlanners.core.policy import Policy
   from POMDPPlanners.core.simulation import SimulationRunData

   class MyCustomPlanner(Policy):
       def __init__(self, environment, **kwargs):
           super().__init__(environment, **kwargs)
           # Initialize your planner

       def action(self, belief_state):
           # Implement your planning algorithm
           # Return (action, run_data)
           action = self.select_action(belief_state)
           run_data = SimulationRunData()
           return action, run_data

       def select_action(self, belief_state):
           # Your action selection logic
           pass

See Also
--------

- :doc:`../examples/planners` - Planner usage examples
- :doc:`beliefs` - Belief state representations
- :doc:`../api/planners` - Complete API reference