# POMDPPlanners User Guide

A comprehensive guide to using the POMDPPlanners library for partially observable Markov decision process (POMDP) planning and simulation.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Quick Start Guide](#quick-start-guide)
4. [Core Concepts](#core-concepts)
5. [Environments](#environments)
6. [Planners](#planners)
7. [Simulations and Evaluation](#simulations-and-evaluation)
8. [Advanced Usage](#advanced-usage)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)
12. [Examples and Tutorials](#examples-and-tutorials)

---

## 1. Introduction

POMDPPlanners is a comprehensive Python library for partially observable Markov decision process (POMDP) planning, simulation, and evaluation. It provides implementations of state-of-the-art POMDP algorithms, benchmark environments, and tools for comparing different planning approaches.

### Key Features

- **Multiple Planning Algorithms**: POMCP, PFT-DPW, Sparse Sampling, and more
- **Benchmark Environments**: Tiger, CartPole, Mountain Car, Light-Dark, and others
- **Comprehensive Evaluation**: Statistical analysis, visualization, and comparison tools
- **Parallel Simulation**: Distributed execution using Dask or Joblib
- **Experiment Tracking**: MLflow integration for experiment management
- **Extensible Architecture**: Easy to add custom environments and planners

### Who Should Use This Library

- **Researchers** working on POMDP algorithms and applications
- **Students** learning about partially observable decision making
- **Practitioners** applying POMDP methods to real-world problems
- **Algorithm developers** who need a robust framework for testing and comparison

---

## 2. Installation and Setup

### Prerequisites

- Python 3.8 or higher
- NumPy, SciPy, pandas
- MLflow (for experiment tracking)
- Dask or Joblib (for parallel execution)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/POMDPPlanners.git
cd POMDPPlanners

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Verification

```python
# Test your installation
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP

# Create a simple test
env = TigerPOMDP(discount_factor=0.95)
planner = POMCP(
    environment=env,
    discount_factor=0.95,
    depth=3,
    exploration_constant=1.0,
    name="POMCP_Test",
    n_simulations=100
)

print("✅ Installation successful!")
```

---

## 3. Quick Start Guide

### Your First POMDP Experiment

Let's start with a simple example using the classic Tiger POMDP problem:

```python
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.belief import get_initial_belief

# Step 1: Create the environment
env = TigerPOMDP(discount_factor=0.95)

# Step 2: Create a planner
planner = POMCP(
    environment=env,
    discount_factor=0.95,
    depth=10,
    exploration_constant=1.0,
    name="POMCP_Tiger",
    n_simulations=1000
)

# Step 3: Get initial belief
initial_belief = get_initial_belief(env, n_particles=1000)

# Step 4: Plan an action
action, run_data = planner.action(initial_belief)
print(f"Recommended action: {action[0]}")

# Step 5: Examine planning results
print(f"Tree statistics: {[m.name for m in run_data.info_variables]}")
```

### Running a Simple Simulation

```python
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.utils.logger import get_logger

# Create logger
logger = get_logger("my_experiment")

# Run a single episode
history = run_episode(
    environment=env,
    policy=planner,
    initial_belief=initial_belief,
    num_steps=20,
    logger=logger
)

# Analyze results
total_reward = sum(step.reward for step in history.history if step.reward is not None)
print(f"Total reward: {total_reward}")
print(f"Episode length: {history.actual_num_steps}")
print(f"Average action time: {history.average_action_time:.4f}s")
```

---

## 4. Core Concepts

### 4.1 Partially Observable Markov Decision Processes (POMDPs)

A POMDP is defined by the tuple (S, A, O, T, Z, R, γ, b₀):

- **S**: State space
- **A**: Action space  
- **O**: Observation space
- **T**: Transition function T(s'|s,a)
- **Z**: Observation function Z(o|s',a)
- **R**: Reward function R(s,a)
- **γ**: Discount factor
- **b₀**: Initial belief distribution

### 4.2 Key Components

#### Environments
Environments define the POMDP problem structure:

```python
class Environment:
    def state_transition_model(self, state, action) -> StateTransitionModel
    def observation_model(self, next_state, action) -> ObservationModel  
    def reward(self, state, action) -> float
    def is_terminal(self, state) -> bool
    def get_actions() -> List[Any]
```

#### Planners
Planners implement algorithms for action selection:

```python
class Policy:
    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]
```

#### Beliefs
Beliefs represent probability distributions over states:

```python
class Belief:
    def sample(self) -> Any
    def update(self, action, observation, pomdp) -> Belief
```

### 4.3 Algorithm Categories

**Monte Carlo Tree Search (MCTS):**
- POMCP: Standard POMDP MCTS with UCB1
- PFT-DPW: Progressive widening for continuous actions
- Sparse-PFT: Memory-efficient sparse tree construction

**Finite Horizon:**
- Sparse Sampling: Builds finite-depth lookahead trees
- Open-Loop Planning: Sequences of actions without observation feedback

---

## 5. Environments

The library includes several benchmark POMDP environments for testing and comparison.

### 5.1 Tiger POMDP

Classic two-door problem with noisy observations.

**Problem Description:**
- Agent faces two doors, one has a tiger, one has treasure
- Actions: listen, open_left, open_right
- Observations: hear_left, hear_right, hear_nothing
- Goal: Open the correct door while avoiding the tiger

```python
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP

# Create Tiger environment
tiger = TigerPOMDP(discount_factor=0.95)

# Environment properties
print(f"Actions: {tiger.get_actions()}")
print(f"Discount factor: {tiger.discount_factor}")

# Sample initial state and take action
state_dist = tiger.initial_state_dist()
state = state_dist.sample()[0]
reward = tiger.reward(state, "listen")
print(f"Reward for listening: {reward}")
```

### 5.2 CartPole POMDP

Continuous control problem with noisy state observations.

**Problem Description:**
- Balance a pole on a moving cart
- Continuous 4D state: [position, velocity, angle, angular_velocity]  
- Discrete actions: left force, right force
- Noisy observations of the true state

```python
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
import numpy as np

# Create CartPole with observation noise
noise_cov = np.diag([0.1, 0.1, 0.1, 0.1])
cartpole = CartPolePOMDP(
    discount_factor=0.99,
    noise_cov=noise_cov
)

# Sample episode
initial_state_dist = cartpole.initial_state_dist()
state = initial_state_dist.sample()[0]
action = 1  # Apply right force
reward = cartpole.reward(state, action)
is_done = cartpole.is_terminal(state)

print(f"State: {state}")
print(f"Reward: {reward}")
print(f"Terminal: {is_done}")
```

### 5.3 Light-Dark POMDP

Navigation problem with position-dependent observation accuracy.

**Problem Description:**
- Agent navigates in 2D space to reach a goal
- Observation accuracy depends on distance from light source
- Continuous state and action spaces
- Trade-off between information gathering and goal reaching

```python
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP, RewardModelType
)

# Create Light-Dark environment
env = ContinuousLightDarkPOMDP(
    discount_factor=0.95,
    goal_state=np.array([10, 5]),
    start_state=np.array([0, 5]),
    reward_model_type=RewardModelType.STANDARD
)

# Sample state and take continuous action
state_dist = env.initial_state_dist()
state = state_dist.sample()[0]
action = np.array([1.0, 0.0])  # Move right
reward = env.reward(state, action)

print(f"Start state: {state}")
print(f"Goal state: {env.goal_state}")
print(f"Reward: {reward}")
```

### 5.4 Mountain Car POMDP

Classic control problem with partial observability.

**Problem Description:**
- Car must reach the top of a hill
- Insufficient power to drive straight up
- Must build momentum by rocking back and forth
- Noisy observations of position and velocity

```python
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP

# Create Mountain Car environment
mountain_car = MountainCarPOMDP(discount_factor=0.99)

# Get initial state and available actions
initial_state_dist = mountain_car.initial_state_dist()
state = initial_state_dist.sample()[0]  # [position, velocity]
actions = mountain_car.get_actions()  # [-1, 0, 1]

# Take action
action = 1  # Accelerate forward
reward = mountain_car.reward(state, action)
is_done = mountain_car.is_terminal(state)

print(f"State: {state}")
print(f"Actions: {actions}")
print(f"Reward: {reward}")
print(f"At goal: {is_done}")
```

### 5.5 Custom Environments

Create custom environments by inheriting from the base classes:

```python
from POMDPPlanners.core.environment import DiscreteActionsEnvironment
from POMDPPlanners.core.distributions import Distribution
import numpy as np

class MyCustomPOMDP(DiscreteActionsEnvironment):
    def __init__(self, discount_factor=0.9):
        # Define space information
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.CONTINUOUS
        )
        
        super().__init__(
            discount_factor=discount_factor,
            name="MyCustomPOMDP",
            space_info=space_info
        )
        
        # Initialize custom parameters
        self.state_dim = 2
        self.num_actions = 3
    
    def state_transition_model(self, state, action):
        # Return a StateTransitionModel instance
        return MyStateTransition(state, action)
    
    def observation_model(self, next_state, action):
        # Return an ObservationModel instance  
        return MyObservationModel(next_state, action)
    
    def reward(self, state, action):
        # Define reward function
        return -np.linalg.norm(state)  # Negative distance from origin
    
    def is_terminal(self, state):
        # Define termination condition
        return np.linalg.norm(state) < 0.1
    
    def get_actions(self):
        # Return list of available actions
        return [0, 1, 2]
    
    def initial_state_dist(self):
        # Return initial state distribution
        return MyInitialStateDistribution()
```

---

## 6. Planners

The library implements several state-of-the-art POMDP planning algorithms.

### 6.1 POMCP (Partially Observable Monte Carlo Planning)

POMCP uses Monte Carlo tree search with particle filtering for belief representation.

**Key Features:**
- UCB1 action selection
- Particle-based belief representation
- Handles large observation spaces
- Theoretical convergence guarantees

```python
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP

# Create POMCP planner
pomcp = POMCP(
    environment=env,
    discount_factor=0.95,
    depth=10,                    # Maximum planning depth
    exploration_constant=1.0,    # UCB1 exploration parameter
    name="POMCP_Planner",
    n_simulations=1000,         # Number of MCTS simulations
    min_samples_per_node=10     # Minimum samples for reliable estimates
)

# Plan action
action, run_data = pomcp.action(initial_belief)
print(f"Selected action: {action[0]}")

# Access tree statistics
tree_metrics = run_data.info_variables
for metric in tree_metrics:
    print(f"{metric.name}: {metric.value}")
```

**When to Use POMCP:**
- Discrete action spaces
- Large or continuous observation spaces
- Need theoretical guarantees
- Sufficient computation time available

### 6.2 PFT-DPW (Progressive Function Transfer with Double Progressive Widening)

PFT-DPW extends MCTS to continuous action spaces using progressive widening.

**Key Features:**
- Handles continuous action spaces
- Progressive action space expansion
- Custom action samplers
- Efficient exploration-exploitation balance

```python
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW, ActionSampler
import numpy as np

# Custom action sampler for continuous control
class ContinuousActionSampler(ActionSampler):
    def __init__(self, action_bounds=(-1.0, 1.0), action_dim=2):
        self.action_bounds = action_bounds
        self.action_dim = action_dim
    
    def sample(self, belief_node=None):
        low, high = self.action_bounds
        return np.random.uniform(low, high, size=self.action_dim)

# Create PFT-DPW planner
pft_dpw = PFT_DPW(
    environment=env,
    discount_factor=0.95,
    depth=10,
    name="PFT_DPW_Planner",
    action_sampler=ContinuousActionSampler(),
    k_a=2.0,                    # Action widening parameter
    alpha_a=0.5,                # Action widening exponent
    k_o=1.0,                    # Observation widening parameter
    alpha_o=0.5,                # Observation widening exponent
    exploration_constant=1.41,   # √2 for optimal exploration
    n_simulations=1000
)

# Plan action
action, run_data = pft_dpw.action(initial_belief)
print(f"Selected continuous action: {action[0]}")
```

**When to Use PFT-DPW:**
- Continuous action spaces
- Need adaptive action space exploration
- Custom domain knowledge for action sampling
- Complex control problems

### 6.3 Sparse Sampling

Finite-horizon algorithm that builds sparse lookahead trees.

**Key Features:**
- Finite-depth planning
- Provable approximation bounds
- Memory efficient
- Good for limited computation budgets

```python
from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner

# Create Sparse Sampling planner
sparse_planner = StandardSparseSamplingDiscreteActionsPlanner(
    environment=env,
    branching_factor=10,  # Number of samples per node
    depth=5,             # Planning depth
    name="SparseSampling_Planner"
)

# Plan action
action, run_data = sparse_planner.action(initial_belief)
print(f"Selected action: {action[0]}")
```

**When to Use Sparse Sampling:**
- Limited computation budget
- Need approximation bounds
- Discrete action spaces
- Shorter planning horizons

### 6.4 Sparse-PFT

Combines sparse sampling efficiency with MCTS tree structure.

**Key Features:**
- Memory-efficient sparse trees
- Enhanced UCB exploration
- Controlled branching factor
- Good balance of quality and efficiency

```python
from POMDPPlanners.planners.mcts_planners.sparse_pft import SparsePFT

# Create Sparse-PFT planner
sparse_pft = SparsePFT(
    environment=env,
    discount_factor=0.95,
    gamma=0.95,                  # Discount for recursive calls
    depth=12,                    # Maximum planning depth
    c_ucb=1.0,                   # Base exploration constant
    beta_ucb=2.0,                # Enhanced exploration parameter
    belief_child_num=5,          # Max belief children per action
    n_simulations=1000,
    name="SparsePFT_Planner"
)

# Plan action
action, run_data = sparse_pft.action(initial_belief)
print(f"Selected action: {action[0]}")
```

**When to Use Sparse-PFT:**
- Memory constraints
- Need exploration control
- Discrete action spaces  
- Balance between quality and efficiency

### 6.5 Algorithm Comparison

| Algorithm | Action Space | Memory | Guarantees | Best For |
|-----------|-------------|--------|------------|----------|
| POMCP | Discrete | High | Convergence | Large obs. spaces |
| PFT-DPW | Continuous | High | None | Continuous control |
| Sparse Sampling | Discrete | Low | Approximation | Limited compute |
| Sparse-PFT | Discrete | Medium | None | Memory efficiency |

### 6.6 Custom Planners

Implement custom planners by extending base classes:

```python
from POMDPPlanners.planners.mcts_planners.path_simulations_policy import PathSimulationPolicy
from POMDPPlanners.core.tree import BeliefNode, ActionNode
import numpy as np

class MyCustomMCTS(PathSimulationPolicy):
    def __init__(self, environment, discount_factor, name, n_simulations, 
                 exploration_constant=1.0):
        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            name=name,
            n_simulations=n_simulations,
            time_out_in_seconds=None
        )
        self.exploration_constant = exploration_constant
    
    def _simulate_path(self, belief_node: BeliefNode, depth: int) -> float:
        # Implement custom MCTS simulation logic
        if depth > 10:  # Max depth
            return 0.0
        
        if belief_node.is_leaf:
            # Expand node
            for action in self.environment.get_actions():
                ActionNode(action=action, parent=belief_node)
            return self._random_rollout(belief_node.belief.sample(), depth)
        
        # Custom action selection logic
        action_node = self._select_best_action(belief_node)
        
        # Simulate step and recurse
        # ... implement simulation logic ...
        
        return total_return
    
    def _select_best_action(self, belief_node):
        # Custom action selection (e.g., modified UCB)
        # ... implement selection logic ...
        pass
    
    def _random_rollout(self, state, depth):
        # Custom rollout policy
        # ... implement rollout logic ...
        pass
```

---

## 7. Simulations and Evaluation

The library provides comprehensive tools for running experiments and evaluating planner performance.

### 7.1 Single Episode Simulation

Run individual episodes to test planners:

```python
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.utils.logger import get_logger

# Create logger
logger = get_logger("episode_test", debug=True)

# Run single episode
history = run_episode(
    environment=env,
    policy=planner,
    initial_belief=initial_belief,
    num_steps=20,
    logger=logger
)

# Analyze results
total_reward = sum(step.reward for step in history.history if step.reward is not None)
print(f"Total reward: {total_reward}")
print(f"Episode length: {history.actual_num_steps}")
print(f"Average action time: {history.average_action_time:.4f}s")
print(f"Reached terminal: {history.reach_terminal_state}")

# Access step-by-step data
for i, step in enumerate(history.history):
    if step.reward is not None:
        print(f"Step {i}: action={step.action}, reward={step.reward}")
```

### 7.2 Multi-Episode Experiments

Compare multiple planners across many episodes:

```python
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.simulations.simulations_deployment.task_managers import TaskManagerType
from pathlib import Path

# Create multiple planners to compare
pomcp = POMCP(environment=env, discount_factor=0.95, depth=10, 
              exploration_constant=1.0, name="POMCP", n_simulations=1000)

sparse_sampling = StandardSparseSamplingDiscreteActionsPlanner(
    environment=env, branching_factor=5, depth=5, name="SparseSampling"
)

# Configure experiment parameters
env_params = [
    EnvironmentRunParams(
        environment=env,
        belief=initial_belief,
        policies=[pomcp, sparse_sampling],
        num_episodes=100,    # Run 100 episodes per planner
        num_steps=20         # Max 20 steps per episode
    )
]

# Run comparison with parallel execution
with POMDPSimulator(
    cache_dir_path=Path("./experiment_results"),
    experiment_name="Planner_Comparison",
    task_manager_type=TaskManagerType.JOBLIB,
    n_jobs=4,  # Use 4 parallel workers
    enable_profiling=True
) as simulator:
    results, statistics_df = simulator.compare_multiple_environments_policies(
        environment_run_params=env_params,
        alpha=0.05,  # 5% risk level for CVaR
        confidence_interval_level=0.95,
        n_jobs=4
    )

# Analyze results
print("Experiment Results:")
print(f"Results structure: {list(results.keys())}")
print(f"Statistics shape: {statistics_df.shape}")

# Compare average returns
for policy_name in ['POMCP', 'SparseSampling']:
    policy_stats = statistics_df[statistics_df['policy'] == policy_name]
    avg_return = policy_stats['average_return'].iloc[0]
    ci_lower = policy_stats['average_return_ci_lower'].iloc[0]
    ci_upper = policy_stats['average_return_ci_upper'].iloc[0]
    print(f"{policy_name}: {avg_return:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
```

### 7.3 High-Level API

Use the SimulationsAPI for simplified experiment execution:

```python
from POMDPPlanners.simulations.simulations_api import SimulationsAPI

# Initialize the API
api = SimulationsAPI(
    cache_dir_path=Path("./simulation_results"),
    debug=True
)

# Run comprehensive experiment
results, statistics_df = api.run_multiple_environments_and_policies_local_run(
    environment_run_params=env_params,
    alpha=0.05,
    confidence_interval_level=0.95,
    experiment_name="My_POMDP_Study",
    n_jobs=-1,  # Use all available cores
    enable_profiling=True
)

# Results are automatically logged to MLflow
print("Simulation completed! Check MLflow UI for detailed results.")
```

### 7.4 Statistical Analysis

The library computes comprehensive performance metrics:

```python
from POMDPPlanners.simulations.simulation_statistics import (
    compute_statistics_environment_policy_pair,
    metrics_dict_to_dataframe
)

# Compute detailed statistics for a single policy
histories = results['TigerPOMDP']['POMCP']  # Get POMCP histories
metrics = compute_statistics_environment_policy_pair(
    env=env,
    histories=histories,
    alpha=0.05,
    confidence_interval_level=0.95
)

# Available metrics include:
# - average_return: Mean discounted return
# - return_cvar: Conditional Value at Risk  
# - return_value_at_risk: Value at Risk
# - average_action_time: Planning time per step
# - average_actual_num_steps: Episode length
# - Environment-specific metrics

for metric in metrics:
    print(f"{metric.name}: {metric.value:.3f} "
          f"[{metric.lower_confidence_bound:.3f}, {metric.upper_confidence_bound:.3f}]")

# Convert to DataFrame for analysis
metrics_dict = {'TigerPOMDP': {'POMCP': metrics}}
df = metrics_dict_to_dataframe(metrics_dict)
print(df.head())
```

### 7.5 Visualization and Reporting

Visualizations are automatically generated and saved:

```python
# Results include:
# - Policy comparison histograms
# - Individual policy performance plots  
# - Episode trajectory visualizations (if supported by environment)
# - Statistical comparison charts

# Access visualization files
results_dir = Path("./experiment_results")
viz_files = list(results_dir.rglob("*.png"))
print(f"Generated {len(viz_files)} visualization files")

# MLflow UI provides interactive exploration
print("View results at: http://localhost:5000 (after running 'mlflow ui')")
```

---

## 8. Advanced Usage

### 8.1 Distributed Computing

Scale experiments using Dask for distributed execution:

```python
from POMDPPlanners.simulations.simulations_api import SimulationsAPI

# Run with Dask distributed scheduler
api = SimulationsAPI(cache_dir_path=Path("./results"))

results, df = api.run_multiple_environments_and_policies_remote_run(
    environment_run_params=env_params,
    alpha=0.05,
    confidence_interval_level=0.95,
    scheduler_address="tcp://scheduler-address:8786",  # Dask scheduler
    n_jobs=10  # Number of workers
)
```

### 8.2 Custom Metrics

Add environment-specific performance metrics:

```python
class MyEnvironment(DiscreteActionsEnvironment):
    # ... environment implementation ...
    
    def compute_metrics(self, histories):
        """Compute custom metrics for this environment."""
        custom_metrics = []
        
        # Example: Success rate metric
        successes = [h.reach_terminal_state for h in histories]
        success_rate = np.mean(successes)
        success_ci = confidence_interval(successes, 0.95)
        
        custom_metrics.append(MetricValue(
            name="success_rate",
            value=success_rate,
            lower_confidence_bound=success_ci[0],
            upper_confidence_bound=success_ci[1]
        ))
        
        # Example: Average path length for successful episodes
        successful_lengths = [h.actual_num_steps for h in histories 
                            if h.reach_terminal_state]
        if successful_lengths:
            avg_length = np.mean(successful_lengths)
            length_ci = confidence_interval(successful_lengths, 0.95)
            
            custom_metrics.append(MetricValue(
                name="avg_success_length",
                value=avg_length,
                lower_confidence_bound=length_ci[0],
                upper_confidence_bound=length_ci[1]
            ))
        
        return custom_metrics
```

### 8.3 Hyperparameter Optimization

While hyperparameter optimization is not currently active in the codebase, you can implement custom optimization:

```python
import optuna
from pathlib import Path

def optimize_pomcp_parameters(env, n_trials=50):
    """Optimize POMCP hyperparameters using Optuna."""
    
    def objective(trial):
        # Suggest hyperparameters
        depth = trial.suggest_int('depth', 5, 20)
        exploration_constant = trial.suggest_float('exploration_constant', 0.1, 5.0)
        n_simulations = trial.suggest_int('n_simulations', 100, 2000)
        
        # Create planner with suggested parameters
        planner = POMCP(
            environment=env,
            discount_factor=0.95,
            depth=depth,
            exploration_constant=exploration_constant,
            name=f"POMCP_trial_{trial.number}",
            n_simulations=n_simulations
        )
        
        # Run evaluation episodes
        total_reward = 0
        n_episodes = 10
        
        for _ in range(n_episodes):
            initial_belief = get_initial_belief(env, n_particles=500)
            history = run_episode(env, planner, initial_belief, 20, logger)
            episode_reward = sum(step.reward for step in history.history 
                               if step.reward is not None)
            total_reward += episode_reward
        
        return total_reward / n_episodes
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
    
    return study.best_params

# Usage
best_params = optimize_pomcp_parameters(tiger_env)
```

### 8.4 Memory Management

For large-scale experiments, manage memory usage:

```python
import gc
import psutil
import os

def monitor_memory_usage():
    """Monitor current memory usage."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

def run_large_experiment_with_memory_management():
    """Run experiment with memory management."""
    
    # Process environments in batches
    batch_size = 2  # Process 2 environments at a time
    all_env_params = [env_param1, env_param2, env_param3, env_param4]
    
    all_results = {}
    all_statistics = []
    
    for i in range(0, len(all_env_params), batch_size):
        batch = all_env_params[i:i+batch_size]
        
        monitor_memory_usage()
        print(f"Processing batch {i//batch_size + 1}/{(len(all_env_params)-1)//batch_size + 1}")
        
        # Run batch
        with POMDPSimulator(
            cache_dir_path=Path(f"./batch_{i//batch_size}"),
            experiment_name=f"Batch_{i//batch_size}"
        ) as simulator:
            results, stats_df = simulator.compare_multiple_environments_policies(
                environment_run_params=batch,
                alpha=0.05,
                confidence_interval_level=0.95
            )
        
        # Store results
        all_results.update(results)
        all_statistics.append(stats_df)
        
        # Force garbage collection
        gc.collect()
        monitor_memory_usage()
    
    # Combine all results
    final_statistics = pd.concat(all_statistics, ignore_index=True)
    return all_results, final_statistics
```

---

## 9. Best Practices

### 9.1 Environment Design

**State Space:**
- Keep state representations compact but sufficient
- Use continuous states for realistic modeling
- Ensure states capture all decision-relevant information

**Action Space:**
- Balance expressiveness with computational tractability  
- For continuous actions, provide meaningful bounds
- Consider action constraints and feasibility

**Observation Space:**
- Model realistic sensor limitations
- Include appropriate noise models
- Consider partial observability's impact on decision making

**Reward Design:**
- Provide clear incentives for desired behavior
- Avoid reward hacking through careful shaping
- Consider sparse vs. dense reward trade-offs

### 9.2 Planner Selection

**Choose POMCP when:**
- You have discrete actions
- Observation space is large or continuous
- You need theoretical guarantees
- Computation time is not severely limited

**Choose PFT-DPW when:**
- Actions are continuous
- You can design good action samplers
- Problem requires adaptive action exploration
- You're willing to trade guarantees for expressiveness

**Choose Sparse Sampling when:**
- Computation budget is limited
- You need approximation bounds
- Actions are discrete
- Planning horizon is relatively short

**Choose Sparse-PFT when:**
- Memory is constrained
- You need a balance of quality and efficiency
- Actions are discrete
- Standard POMCP is too memory-intensive

### 9.3 Parameter Tuning

**POMCP Parameters:**
- **depth**: Start with 2x expected episode length
- **exploration_constant**: Begin with 1.0, increase for more exploration
- **n_simulations**: Use as many as computational budget allows
- **min_samples_per_node**: 10-20 for reliable estimates

**PFT-DPW Parameters:**
- **k_a, alpha_a**: Control action space growth; start with (1.0, 0.5)
- **k_o, alpha_o**: Control observation space growth; start with (1.0, 0.5)  
- **exploration_constant**: Often √2 ≈ 1.41 works well
- **action_sampler**: Design domain-specific samplers

**Sparse Sampling Parameters:**
- **branching_factor**: Trade-off between quality and speed (5-20)
- **depth**: Usually shorter than MCTS approaches (3-8)

**General Guidelines:**
- Start with default parameters and tune systematically
- Use grid search or Bayesian optimization for hyperparameters
- Monitor both performance and computational cost
- Cross-validate parameter choices across different problem instances

### 9.4 Experimental Design

**Reproducibility:**
- Set random seeds for all components
- Document environment and planner configurations
- Save complete experimental configurations
- Use version control for code and configurations

**Statistical Rigor:**
- Run sufficient episodes for statistical significance
- Use appropriate confidence intervals
- Report both mean and risk measures (CVaR, VaR)
- Consider multiple problem instances

**Comparative Studies:**
- Use identical experimental conditions for all algorithms
- Report runtime alongside solution quality
- Include baseline algorithms for context
- Test across diverse problem characteristics

**Resource Management:**
- Monitor computation time and memory usage
- Use parallel execution for large experiments
- Implement checkpointing for long-running experiments
- Plan for experiment scalability

---

## 10. Troubleshooting

### 10.1 Common Issues

**Import Errors:**
```python
# Problem: ModuleNotFoundError
# Solution: Ensure proper installation and PYTHONPATH
import sys
sys.path.append('/path/to/POMDPPlanners')

# Verify installation
try:
    from POMDPPlanners.core.environment import Environment
    print("✅ Core modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Check requirements.txt and reinstall
```

**Memory Issues:**
```python
# Problem: Out of memory during large experiments
# Solution: Reduce parameters or use batching

# Reduce particle count
initial_belief = get_initial_belief(env, n_particles=100)  # Instead of 1000

# Reduce simulation count  
planner = POMCP(..., n_simulations=500)  # Instead of 2000

# Use memory monitoring
import psutil
process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {memory_mb:.1f} MB")
```

**Slow Performance:**
```python
# Problem: Experiments taking too long
# Solution: Optimize parameters and use parallelization

# Enable parallel execution
n_jobs = -1  # Use all available cores

# Reduce computational complexity
planner = POMCP(
    ...,
    depth=8,           # Reduce from 15
    n_simulations=500  # Reduce from 2000
)

# Profile performance
simulator = POMDPSimulator(..., enable_profiling=True)
# Check profiling results after experiment
```

**Numerical Issues:**
```python
# Problem: NaN or infinite values
# Solution: Add numerical stability checks

def safe_log(x, eps=1e-10):
    """Numerically stable logarithm."""
    return np.log(np.clip(x, eps, None))

def safe_divide(a, b, eps=1e-10):
    """Safe division avoiding division by zero."""
    return a / np.maximum(b, eps)

# Check for numerical issues in beliefs
belief_particles = belief.particles
if np.any(np.isnan(belief_particles)):
    print("❌ NaN detected in belief particles")
if np.any(np.isinf(belief_particles)):
    print("❌ Infinite values detected in belief particles")
```

### 10.2 Debugging Strategies

**Enable Debug Logging:**
```python
from POMDPPlanners.utils.logger import get_logger

# Create debug logger
logger = get_logger("debug_session", debug=True)

# Enable debug mode in components
planner = POMCP(..., debug=True)
simulator = POMDPSimulator(..., debug=True)

# Run with detailed logging
history = run_episode(..., logger=logger)
```

**Inspect Intermediate Results:**
```python
# Check belief consistency
print(f"Belief particles: {len(belief.particles)}")
print(f"Particle range: {np.min(belief.particles)} to {np.max(belief.particles)}")

# Check action selection
action, run_data = planner.action(belief)
print(f"Selected action: {action}")
print(f"Tree metrics: {[m.name for m in run_data.info_variables]}")

# Validate environment dynamics
state = env.initial_state_dist().sample()[0]
for action in env.get_actions():
    reward = env.reward(state, action)
    next_state = env.state_transition_model(state, action).sample()[0]
    observation = env.observation_model(next_state, action).sample()[0]
    print(f"Action {action}: reward={reward}, next_state={next_state}, obs={observation}")
```

**Validate Configurations:**
```python
def validate_environment(env):
    """Validate environment configuration."""
    # Test basic functionality
    try:
        actions = env.get_actions()
        print(f"✅ Actions: {actions}")
        
        # Test state sampling
        state_dist = env.initial_state_dist()
        state = state_dist.sample()[0]
        print(f"✅ Initial state: {state}")
        
        # Test dynamics
        action = actions[0]
        reward = env.reward(state, action)
        next_state = env.state_transition_model(state, action).sample()[0]
        observation = env.observation_model(next_state, action).sample()[0]
        
        print(f"✅ Dynamics work: reward={reward}")
        
        return True
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False

def validate_planner(planner, env):
    """Validate planner configuration."""
    try:
        initial_belief = get_initial_belief(env, n_particles=10)
        action, run_data = planner.action(initial_belief)
        
        print(f"✅ Planner works: action={action[0]}")
        print(f"✅ Tree metrics: {len(run_data.info_variables)}")
        
        return True
    except Exception as e:
        print(f"❌ Planner validation failed: {e}")
        return False

# Usage
if validate_environment(env) and validate_planner(planner, env):
    print("✅ Configuration validated successfully")
else:
    print("❌ Configuration has issues")
```

### 10.3 Performance Optimization

**Profile Critical Sections:**
```python
import cProfile
import pstats

def profile_experiment():
    """Profile experiment performance."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run experiment
    history = run_episode(env, planner, initial_belief, 20, logger)
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Show top 20 functions

profile_experiment()
```

**Optimize Belief Updates:**
```python
# Use efficient particle filtering
class OptimizedBelief(Belief):
    def __init__(self, particles, weights=None):
        self.particles = np.array(particles)
        if weights is None:
            self.weights = np.ones(len(particles)) / len(particles)
        else:
            self.weights = np.array(weights)
    
    def sample(self):
        """Efficient sampling using numpy."""
        idx = np.random.choice(len(self.particles), p=self.weights)
        return self.particles[idx]
    
    def update(self, action, observation, pomdp):
        """Efficient belief update."""
        # Vectorized likelihood computation
        likelihoods = np.array([
            pomdp.observation_model(p, action).probability([observation])[0]
            for p in self.particles
        ])
        
        # Update weights
        new_weights = self.weights * likelihoods
        new_weights /= new_weights.sum()
        
        return OptimizedBelief(self.particles, new_weights)
```

---

## 11. API Reference

### 11.1 Core Classes

#### Environment
```python
class Environment(ABC):
    """Base class for POMDP environments."""
    
    @abstractmethod
    def state_transition_model(self, state, action) -> StateTransitionModel:
        """Return state transition model for given state-action pair."""
        pass
    
    @abstractmethod  
    def observation_model(self, next_state, action) -> ObservationModel:
        """Return observation model for given next_state-action pair."""
        pass
    
    @abstractmethod
    def reward(self, state, action) -> float:
        """Return immediate reward for state-action pair.""" 
        pass
    
    @abstractmethod
    def is_terminal(self, state) -> bool:
        """Check if state is terminal."""
        pass
    
    def compute_metrics(self, histories) -> List[MetricValue]:
        """Compute environment-specific metrics from histories."""
        return []
```

#### Policy  
```python
class Policy(ABC):
    """Base class for POMDP planning algorithms."""
    
    @abstractmethod
    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]:
        """Select action given current belief state."""
        pass
    
    def get_space_info(self) -> PolicySpaceInfo:
        """Return information about action and observation spaces."""
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.DISCRETE
        )
```

#### Belief
```python
class Belief(ABC):
    """Base class for belief state representations."""
    
    @abstractmethod
    def sample(self) -> Any:
        """Sample a state from the belief distribution."""
        pass
    
    @abstractmethod
    def update(self, action, observation, pomdp) -> 'Belief':
        """Update belief given action and observation."""
        pass
```

### 11.2 Planning Algorithms

#### POMCP
```python
class POMCP(Policy):
    def __init__(self, environment, discount_factor, depth, exploration_constant,
                 name, time_out_in_seconds=None, n_simulations=None, 
                 min_samples_per_node=10, log_path=None, debug=False):
        """
        Args:
            environment: POMDP environment
            discount_factor: Discount factor (0 < γ ≤ 1)
            depth: Maximum search depth
            exploration_constant: UCB1 exploration parameter
            name: Policy identifier
            time_out_in_seconds: Time limit (mutually exclusive with n_simulations)
            n_simulations: Number of simulations (mutually exclusive with timeout)
            min_samples_per_node: Minimum samples for reliable estimates
            log_path: Optional logging path
            debug: Enable debug logging
        """
```

#### PFT_DPW
```python
class PFT_DPW(PathSimulationPolicy):
    def __init__(self, environment, discount_factor, depth, name, action_sampler,
                 k_a=1.0, alpha_a=0.5, k_o=1.0, alpha_o=0.5, 
                 exploration_constant=1.0, time_out_in_seconds=None, 
                 n_simulations=None, min_samples_per_node=10, 
                 min_visit_count_per_action=1, log_path=None, debug=False):
        """
        Args:
            action_sampler: ActionSampler instance for sampling new actions
            k_a, alpha_a: Action progressive widening parameters  
            k_o, alpha_o: Observation progressive widening parameters
            exploration_constant: UCB1 exploration parameter
            ... (other parameters same as POMCP)
        """
```

### 11.3 Simulation Tools

#### POMDPSimulator
```python
class POMDPSimulator(BaseSimulator):
    def compare_multiple_environments_policies(
        self, environment_run_params, alpha=0.1, confidence_interval_level=0.95,
        n_jobs=1, cache_visualizations=True
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """
        Compare multiple policies across multiple environments.
        
        Args:
            environment_run_params: List of EnvironmentRunParams
            alpha: Risk level for CVaR computation
            confidence_interval_level: Confidence level for intervals
            n_jobs: Number of parallel workers
            cache_visualizations: Whether to generate visualizations
            
        Returns:
            Tuple of (results_dict, statistics_dataframe)
        """
```

#### SimulationsAPI
```python
class SimulationsAPI:
    def run_multiple_environments_and_policies_local_run(
        self, environment_run_params, alpha, confidence_interval_level,
        experiment_name="POMDP_Planning_Comparison", debug=False, n_jobs=-1,
        cache_dir_path=None, clear_cache_on_start=False, enable_profiling=False,
        profiling_output_limit=50
    ) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
        """
        Run multi-environment experiment using local parallel execution.
        
        Args:
            environment_run_params: List of experiment configurations
            alpha: Risk level for statistical analysis
            confidence_interval_level: Confidence level for intervals
            experiment_name: MLflow experiment name
            debug: Enable debug logging
            n_jobs: Number of parallel workers (-1 for all cores)
            cache_dir_path: Path for storing results
            clear_cache_on_start: Clear cache before starting
            enable_profiling: Enable performance profiling
            profiling_output_limit: Number of functions in profiling output
            
        Returns:
            Tuple of (results_dict, statistics_dataframe)
        """
```

### 11.4 Utility Functions

#### Belief Creation
```python
def get_initial_belief(pomdp: Environment, n_particles: int = 1000) -> Belief:
    """
    Create initial belief for POMDP.
    
    Args:
        pomdp: POMDP environment
        n_particles: Number of particles for belief representation
        
    Returns:
        Initial belief state
    """
```

#### Statistics Computation
```python
def compute_statistics_environment_policy_pair(
    env: Environment, histories: List[History], alpha: float, 
    confidence_interval_level: float = 0.95
) -> List[MetricValue]:
    """
    Compute comprehensive statistics for environment-policy pair.
    
    Args:
        env: POMDP environment
        histories: List of episode histories
        alpha: Risk level for CVaR/VaR computation
        confidence_interval_level: Confidence level for intervals
        
    Returns:
        List of computed metrics with confidence intervals
    """
```

---

## 12. Examples and Tutorials

### 12.1 Tutorial 1: Your First POMDP Experiment

**Objective:** Learn basic library usage by solving the Tiger POMDP.

```python
# Step-by-step tutorial
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.utils.logger import get_logger

# 1. Understand the problem
print("Tiger POMDP:")
print("- Two doors: one has tiger, one has treasure")  
print("- Actions: listen, open_left, open_right")
print("- Goal: Open treasure door, avoid tiger")

# 2. Create environment
env = TigerPOMDP(discount_factor=0.95)
print(f"Available actions: {env.get_actions()}")

# 3. Create planner
planner = POMCP(
    environment=env,
    discount_factor=0.95,
    depth=10,
    exploration_constant=1.0,
    name="Tutorial_POMCP",
    n_simulations=1000
)

# 4. Initialize belief
initial_belief = get_initial_belief(env, n_particles=1000)
print("Initial belief: uniform over tiger locations")

# 5. Plan first action  
action, run_data = planner.action(initial_belief)
print(f"Recommended first action: {action[0]}")

# 6. Run complete episode
logger = get_logger("tutorial")
history = run_episode(env, planner, initial_belief, 20, logger)

# 7. Analyze results
total_reward = sum(step.reward for step in history.history if step.reward is not None)
print(f"Total reward: {total_reward}")
print(f"Episode length: {history.actual_num_steps}")

# 8. Examine decision sequence
print("\nDecision sequence:")
for i, step in enumerate(history.history):
    if step.action is not None:
        print(f"Step {i+1}: {step.action} (reward: {step.reward})")
```

### 12.2 Tutorial 2: Comparing Planning Algorithms

**Objective:** Compare multiple planners on the same problem.

```python
from POMDPPlanners.planners.sparse_sampling_planner import StandardSparseSamplingDiscreteActionsPlanner
from POMDPPlanners.simulations.simulator import POMDPSimulator
from POMDPPlanners.core.simulation import EnvironmentRunParams
from pathlib import Path

# Create multiple planners
planners = {
    'POMCP': POMCP(
        environment=env, discount_factor=0.95, depth=10,
        exploration_constant=1.0, name="POMCP", n_simulations=1000
    ),
    'Sparse_Sampling': StandardSparseSamplingDiscreteActionsPlanner(
        environment=env, branching_factor=5, depth=5, name="SparseSampling"
    ),
    'POMCP_Fast': POMCP(
        environment=env, discount_factor=0.95, depth=8,
        exploration_constant=0.5, name="POMCP_Fast", n_simulations=500
    )
}

# Configure experiment
env_params = [
    EnvironmentRunParams(
        environment=env,
        belief=get_initial_belief(env, n_particles=500),
        policies=list(planners.values()),
        num_episodes=50,
        num_steps=15
    )
]

# Run comparison
with POMDPSimulator(
    cache_dir_path=Path("./tutorial_comparison"),
    experiment_name="Algorithm_Comparison_Tutorial"
) as simulator:
    results, stats_df = simulator.compare_multiple_environments_policies(
        environment_run_params=env_params,
        alpha=0.05,
        confidence_interval_level=0.95
    )

# Analyze results
print("Algorithm Comparison Results:")
print("=" * 50)

for policy_name in planners.keys():
    policy_stats = stats_df[stats_df['policy'] == policy_name]
    
    avg_return = policy_stats['average_return'].iloc[0]
    ci_lower = policy_stats['average_return_ci_lower'].iloc[0]  
    ci_upper = policy_stats['average_return_ci_upper'].iloc[0]
    avg_time = policy_stats['average_action_time'].iloc[0]
    
    print(f"{policy_name}:")
    print(f"  Average Return: {avg_return:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  Planning Time: {avg_time:.4f}s per action")
    print()

# Find best performer
best_policy = stats_df.loc[stats_df['average_return'].idxmax(), 'policy']
print(f"Best performing algorithm: {best_policy}")
```

### 12.3 Tutorial 3: Continuous Control with PFT-DPW

**Objective:** Apply PFT-DPW to a continuous control problem.

```python
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW, ActionSampler
import numpy as np

# 1. Create continuous control environment  
cartpole = CartPolePOMDP(
    discount_factor=0.99,
    noise_cov=np.diag([0.1, 0.1, 0.1, 0.1])  # Observation noise
)

print("CartPole POMDP:")
print("- State: [position, velocity, angle, angular_velocity]")
print("- Actions: force magnitude (continuous)")
print("- Goal: Keep pole upright")

# 2. Design action sampler for CartPole
class CartPoleActionSampler(ActionSampler):
    def sample(self, belief_node=None):
        # Sample discrete force direction  
        return np.random.choice([0, 1])  # Left or right

# 3. Create PFT-DPW planner
pft_planner = PFT_DPW(
    environment=cartpole,
    discount_factor=0.99,
    depth=8,
    name="PFT_DPW_CartPole",
    action_sampler=CartPoleActionSampler(),
    k_a=2.0,                    # Action widening
    alpha_a=0.5,                # Action widening exponent  
    exploration_constant=1.41,   # √2
    n_simulations=500
)

# 4. Run episode
initial_belief = get_initial_belief(cartpole, n_particles=200)
history = run_episode(cartpole, pft_planner, initial_belief, 50, logger)

# 5. Analyze performance
episode_length = history.actual_num_steps
avg_planning_time = history.average_action_time

print(f"Episode Results:")
print(f"  Episode length: {episode_length} steps")
print(f"  Planning time: {avg_planning_time:.4f}s per action")
print(f"  Terminal reached: {history.reach_terminal_state}")

# 6. Examine action sequence
actions_taken = [step.action for step in history.history if step.action is not None]
print(f"Actions taken: {actions_taken[:10]}...")  # First 10 actions

# 7. Compare with discrete POMCP
pomcp_discrete = POMCP(
    environment=cartpole, discount_factor=0.99, depth=8,
    exploration_constant=1.0, name="POMCP_CartPole", n_simulations=500  
)

pomcp_history = run_episode(cartpole, pomcp_discrete, initial_belief, 50, logger)

print(f"\nComparison:")
print(f"  PFT-DPW length: {history.actual_num_steps}")
print(f"  POMCP length: {pomcp_history.actual_num_steps}")
print(f"  PFT-DPW time: {history.average_action_time:.4f}s")
print(f"  POMCP time: {pomcp_history.average_action_time:.4f}s")
```

### 12.4 Tutorial 4: Custom Environment Development

**Objective:** Create a custom POMDP environment.

```python
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceInfo, SpaceType
from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import StateTransitionModel, ObservationModel
import numpy as np

# 1. Define the problem
class GridWorldPOMDP(DiscreteActionsEnvironment):
    """Simple grid world with noisy observations."""
    
    def __init__(self, grid_size=5, discount_factor=0.95):
        # Create space info
        space_info = SpaceInfo(
            action_space=SpaceType.DISCRETE,
            observation_space=SpaceType.DISCRETE  
        )
        
        super().__init__(
            discount_factor=discount_factor,
            name="GridWorldPOMDP", 
            space_info=space_info
        )
        
        self.grid_size = grid_size
        self.goal = (grid_size-1, grid_size-1)  # Bottom-right corner
        
    def state_transition_model(self, state, action):
        return GridWorldTransition(state, action, self.grid_size)
    
    def observation_model(self, next_state, action):
        return GridWorldObservation(next_state, action)
    
    def reward(self, state, action):
        # Reward for reaching goal
        if state == self.goal:
            return 10.0
        else:
            return -0.1  # Small step penalty
    
    def is_terminal(self, state):
        return state == self.goal
    
    def get_actions(self):
        return ['up', 'down', 'left', 'right']
    
    def initial_state_dist(self):
        return GridWorldInitialState(self.grid_size)

# 2. Implement transition model
class GridWorldTransition(StateTransitionModel):
    def __init__(self, state, action, grid_size):
        super().__init__(state, action)
        self.grid_size = grid_size
    
    def sample(self, n_samples=1):
        x, y = self.state
        
        # Apply action with some noise
        if self.action == 'up' and y > 0:
            next_state = (x, y-1)
        elif self.action == 'down' and y < self.grid_size-1:
            next_state = (x, y+1)
        elif self.action == 'left' and x > 0:
            next_state = (x-1, y)
        elif self.action == 'right' and x < self.grid_size-1:
            next_state = (x+1, y)
        else:
            next_state = self.state  # No movement if hitting wall
        
        return [next_state] * n_samples
    
    def probability(self, values):
        expected = self.sample()[0]
        return np.array([1.0 if v == expected else 0.0 for v in values])

# 3. Implement observation model  
class GridWorldObservation(ObservationModel):
    def __init__(self, next_state, action):
        super().__init__(next_state, action)
    
    def sample(self, n_samples=1):
        # Noisy position observation
        x, y = self.next_state
        
        # 80% chance of correct observation, 20% chance of adjacent cell
        if np.random.random() < 0.8:
            obs = self.next_state
        else:
            # Random adjacent cell
            neighbors = [(x+dx, y+dy) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
                        if 0 <= x+dx < 5 and 0 <= y+dy < 5]
            obs = np.random.choice(neighbors) if neighbors else self.next_state
        
        return [obs] * n_samples
    
    def probability(self, values):
        # Simplified probability model
        true_state = self.next_state
        probs = []
        
        for v in values:
            if v == true_state:
                probs.append(0.8)  # Correct observation
            elif abs(v[0] - true_state[0]) + abs(v[1] - true_state[1]) == 1:
                probs.append(0.05)  # Adjacent cell
            else:
                probs.append(0.0)   # Impossible observation
        
        return np.array(probs)

# 4. Implement initial state distribution
class GridWorldInitialState(Distribution):
    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size
    
    def sample(self, n_samples=1):
        # Start at random position (excluding goal)
        positions = [(x, y) for x in range(self.grid_size) 
                    for y in range(self.grid_size)
                    if (x, y) != (self.grid_size-1, self.grid_size-1)]
        
        return [np.random.choice(positions) for _ in range(n_samples)]

# 5. Test the custom environment
print("Testing Custom Grid World POMDP:")

# Create environment
grid_env = GridWorldPOMDP(grid_size=5, discount_factor=0.95)

# Test basic functionality
initial_state = grid_env.initial_state_dist().sample()[0]
print(f"Initial state: {initial_state}")

action = 'right'
reward = grid_env.reward(initial_state, action)
print(f"Reward for {action}: {reward}")

# Test with planner
pomcp_grid = POMCP(
    environment=grid_env,
    discount_factor=0.95,
    depth=10,
    exploration_constant=1.0,
    name="POMCP_Grid",
    n_simulations=500
)

# Run episode
grid_belief = get_initial_belief(grid_env, n_particles=100)
grid_history = run_episode(grid_env, pomcp_grid, grid_belief, 20, logger)

print(f"Grid World Results:")
print(f"  Episode length: {grid_history.actual_num_steps}")
print(f"  Reached goal: {grid_history.reach_terminal_state}")

# Show path taken
path = [(0, 0)]  # Approximate starting position
for step in grid_history.history:
    if step.action is not None:
        print(f"  Action: {step.action}, Reward: {step.reward}")

print("✅ Custom environment working successfully!")
```

### 12.5 Tutorial 5: Large-Scale Experimental Study

**Objective:** Design and execute a comprehensive experimental study.

```python
from POMDPPlanners.simulations.simulations_api import SimulationsAPI
import numpy as np

# 1. Define experimental objectives
print("Large-Scale POMDP Algorithm Study")
print("Objective: Compare planning algorithms across multiple environments")
print("Metrics: Performance, computation time, scalability")

# 2. Create multiple environments
environments = {
    'Tiger': TigerPOMDP(discount_factor=0.95),
    'CartPole': CartPolePOMDP(
        discount_factor=0.99, 
        noise_cov=np.diag([0.1, 0.1, 0.1, 0.1])
    ),
    'GridWorld': GridWorldPOMDP(grid_size=5, discount_factor=0.95)
}

# 3. Create algorithm configurations
algorithm_configs = {
    'POMCP_Standard': {
        'class': POMCP,
        'params': {
            'depth': 10, 'exploration_constant': 1.0, 'n_simulations': 1000
        }
    },
    'POMCP_Fast': {
        'class': POMCP, 
        'params': {
            'depth': 8, 'exploration_constant': 0.5, 'n_simulations': 500
        }
    },
    'SparseSampling': {
        'class': StandardSparseSamplingDiscreteActionsPlanner,
        'params': {
            'branching_factor': 5, 'depth': 5
        }
    }
}

# 4. Generate experiment configurations
experiment_configs = []

for env_name, env in environments.items():
    # Create planners for this environment
    planners = []
    for alg_name, config in algorithm_configs.items():
        if config['class'] == POMCP:
            planner = config['class'](
                environment=env,
                discount_factor=env.discount_factor,
                name=f"{alg_name}_{env_name}",
                **config['params']
            )
        else:  # SparseSampling
            planner = config['class'](
                environment=env,
                name=f"{alg_name}_{env_name}",
                **config['params']
            )
        planners.append(planner)
    
    # Create experiment configuration
    env_config = EnvironmentRunParams(
        environment=env,
        belief=get_initial_belief(env, n_particles=500),
        policies=planners,
        num_episodes=100,  # Sufficient for statistical significance
        num_steps=25       # Reasonable episode length
    )
    experiment_configs.append(env_config)

# 5. Run comprehensive experiment
print(f"Running experiment with:")
print(f"  {len(environments)} environments")
print(f"  {len(algorithm_configs)} algorithms") 
print(f"  {len(experiment_configs)} total configurations")
print(f"  Total episodes: {sum(len(cfg.policies) * cfg.num_episodes for cfg in experiment_configs)}")

api = SimulationsAPI(
    cache_dir_path=Path("./comprehensive_study"),
    debug=False
)

# Run with debug mode first (smaller scale)
print("\n1. Running debug experiment...")
debug_configs = []
for cfg in experiment_configs:
    debug_cfg = EnvironmentRunParams(
        environment=cfg.environment,
        belief=cfg.belief, 
        policies=cfg.policies,
        num_episodes=5,   # Small number for debug
        num_steps=10      # Shorter episodes
    )
    debug_configs.append(debug_cfg)

debug_results, debug_stats = api.run_multiple_environments_and_policies_local_run(
    environment_run_params=debug_configs,
    alpha=0.05,
    confidence_interval_level=0.95,
    experiment_name="Debug_Study",
    n_jobs=2
)
print("✅ Debug experiment completed successfully")

# Run full experiment
print("\n2. Running full experiment...")
results, statistics_df = api.run_multiple_environments_and_policies_local_run(
    environment_run_params=experiment_configs,
    alpha=0.05,
    confidence_interval_level=0.95,
    experiment_name="Comprehensive_POMDP_Study", 
    n_jobs=-1,  # Use all cores
    enable_profiling=True
)

# 6. Comprehensive analysis
print("\n" + "="*60)
print("COMPREHENSIVE STUDY RESULTS")
print("="*60)

# Overall summary
total_configs = len(statistics_df)
environments_tested = statistics_df['environment'].nunique()
algorithms_tested = statistics_df['policy'].nunique()

print(f"Summary:")
print(f"  Total configurations tested: {total_configs}")
print(f"  Environments: {environments_tested}")
print(f"  Algorithms: {algorithms_tested}")

# Performance analysis by environment
print(f"\nPerformance by Environment:")
print("-" * 40)

for env_name in statistics_df['environment'].unique():
    env_data = statistics_df[statistics_df['environment'] == env_name]
    
    print(f"\n{env_name}:")
    # Sort by performance
    env_data_sorted = env_data.sort_values('average_return', ascending=False)
    
    for _, row in env_data_sorted.iterrows():
        policy_name = row['policy']
        avg_return = row['average_return']
        ci_lower = row['average_return_ci_lower']
        ci_upper = row['average_return_ci_upper'] 
        avg_time = row['average_action_time']
        
        print(f"  {policy_name:20}: {avg_return:8.3f} [{ci_lower:6.3f}, {ci_upper:6.3f}] ({avg_time:.4f}s)")

# Algorithm ranking across environments
print(f"\nAlgorithm Rankings (by average return):")
print("-" * 45)

algorithm_performance = {}
for alg_name in statistics_df['policy'].unique():
    alg_data = statistics_df[statistics_df['policy'] == alg_name]
    avg_performance = alg_data['average_return'].mean()
    avg_time = alg_data['average_action_time'].mean()
    algorithm_performance[alg_name] = (avg_performance, avg_time)

# Sort by performance
sorted_algorithms = sorted(algorithm_performance.items(), 
                         key=lambda x: x[1][0], reverse=True)

for rank, (alg_name, (performance, time)) in enumerate(sorted_algorithms, 1):
    print(f"{rank}. {alg_name:20}: {performance:8.3f} avg return ({time:.4f}s avg time)")

# Efficiency analysis (performance per unit time)
print(f"\nEfficiency Rankings (return per second):")
print("-" * 40)

efficiency_rankings = []
for alg_name, (performance, time) in algorithm_performance.items():
    if time > 0:
        efficiency = performance / time
        efficiency_rankings.append((alg_name, efficiency, performance, time))

efficiency_rankings.sort(key=lambda x: x[1], reverse=True)

for rank, (alg_name, efficiency, performance, time) in enumerate(efficiency_rankings, 1):
    print(f"{rank}. {alg_name:20}: {efficiency:8.1f} return/s ({performance:.3f} return, {time:.4f}s)")

# Statistical significance testing
print(f"\nStatistical Significance Analysis:")
print("-" * 35)

from scipy import stats

for env_name in statistics_df['environment'].unique():
    env_data = statistics_df[statistics_df['environment'] == env_name]
    
    if len(env_data) >= 2:
        print(f"\n{env_name}:")
        
        # Compare top two performers
        top_two = env_data.nlargest(2, 'average_return')
        
        if len(top_two) == 2:
            alg1 = top_two.iloc[0]
            alg2 = top_two.iloc[1]
            
            # Check confidence interval overlap
            ci1 = (alg1['average_return_ci_lower'], alg1['average_return_ci_upper'])
            ci2 = (alg2['average_return_ci_lower'], alg2['average_return_ci_upper'])
            
            overlap = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
            
            print(f"  Best: {alg1['policy']} ({alg1['average_return']:.3f})")
            print(f"  2nd:  {alg2['policy']} ({alg2['average_return']:.3f})")
            print(f"  CI Overlap: {'Yes' if overlap else 'No'} -> {'No significant difference' if overlap else 'Significant difference'}")

print(f"\n✅ Comprehensive study completed!")
print(f"📊 Results saved to: ./comprehensive_study/")
print(f"🔍 View detailed results with: mlflow ui --backend-store-uri ./comprehensive_study/mlruns")

# 7. Generate summary report
report_path = Path("./comprehensive_study/STUDY_REPORT.md")
with open(report_path, 'w') as f:
    f.write("# Comprehensive POMDP Algorithm Study Report\n\n")
    f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Total Configurations:** {total_configs}\n")
    f.write(f"**Environments Tested:** {environments_tested}\n") 
    f.write(f"**Algorithms Tested:** {algorithms_tested}\n\n")
    
    f.write("## Key Findings\n\n")
    f.write(f"1. **Best Overall Algorithm:** {sorted_algorithms[0][0]}\n")
    f.write(f"2. **Most Efficient Algorithm:** {efficiency_rankings[0][0]}\n")
    f.write(f"3. **Most Challenging Environment:** TBD based on performance variance\n\n")
    
    f.write("## Recommendations\n\n")
    f.write("- Use POMCP for problems requiring high solution quality\n")
    f.write("- Use Sparse Sampling when computation time is limited\n")
    f.write("- Consider environment characteristics when selecting algorithms\n\n")
    
    f.write("## Full Results\n\n")
    f.write("See MLflow UI for detailed interactive analysis.\n")

print(f"📋 Summary report generated: {report_path}")
```

---

This comprehensive user guide covers all major aspects of using the POMDPPlanners library, from basic concepts to advanced experimental studies. Users can follow the tutorials sequentially to build expertise, or jump to specific sections based on their needs.

The guide emphasizes practical usage with complete working examples while providing sufficient theoretical background to understand the algorithms and make informed choices for their applications.