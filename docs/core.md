# Core Module Documentation (`POMDPPlanners/core/`)

This directory contains the foundational abstractions and interfaces for the POMDP planning framework. All core components provide abstract base classes that define the essential contracts for environments, policies, beliefs, and supporting data structures.

## Module Overview

### `environment.py` - POMDP Environment Abstractions

**Key Classes:**
- `Environment`: Abstract base class for all POMDP environments
- `DiscreteActionsEnvironment`: Specialized for discrete action spaces
- `ObservationModel`: Abstract observation model inheriting from Distribution
- `StateTransitionModel`: Abstract state transition model inheriting from Distribution
- `EnvironmentGenerator`: Factory pattern for environment creation

**Key Features:**
- `SpaceType` enum (DISCRETE, CONTINUOUS, MIXED) for categorizing action/observation spaces
- Comprehensive equality comparison with numpy array handling
- Deterministic config_id generation for caching and reproducibility
- Integrated logging system with pickle compatibility
- Abstract methods for state transitions, observations, rewards, and terminal conditions

### `belief.py` - Belief State Representations

**Key Classes:**
- `Belief`: Abstract base class for all belief representations
- `WeightedParticleBelief`: Particle filter with weighted particles for continuous observation spaces
- `UnweightedParticleBelief`: Uniform particle filter for discrete observation spaces  
- `WeightedParticleBeliefReinvigoration`: Extended weighted filter with degeneracy handling

**Key Features:**
- Particle filter implementations with automatic resampling
- Effective Sample Size (ESS) based resampling strategies
- Belief serialization and deserialization support
- Reinvigoration mechanisms for particle diversity
- Configuration-based belief creation with factory methods
- Numerical stability through log-space computations

### `policy.py` - Policy Interface and Data Structures

**Key Classes:**
- `Policy`: Abstract base class for all planning policies
- `PolicyRunData`: Container for policy execution information
- `PolicyInfoVariable`: Named tuple for policy metrics
- `PolicySpaceInfo`: Space type information for policies

**Key Features:**
- Abstract action selection method with belief input
- Integrated logging with configurable debug levels
- Configuration management with deterministic IDs
- Policy execution tracking and metrics collection
- Space compatibility checking between policies and environments

### `distributions.py` - Probability Distribution Implementations

**Key Classes:**
- `Distribution`: Abstract base class for probability distributions
- `DiscreteDistribution`: Implementation for discrete probability distributions
- `Numpy2DDistribution`: Specialized for 2D numpy array distributions

**Key Features:**
- Sampling methods with configurable sample counts
- Probability calculation for given values
- Input validation and type checking
- Vectorized operations for efficiency
- Support for numpy arrays and generic value types

### `simulation.py` - Simulation Framework and Data Structures

**Key Classes:**
- `StepData`: NamedTuple containing state, action, observation, reward, and belief
- `History`: Comprehensive episode history with timing metrics and metadata
- `SimulationTask`: Abstract base for executable simulation tasks
- `TaskManager`: Abstract interface for distributed task execution
- `TaskManagerExternalDB`: Implementation with external database caching
- `DataBaseInterface`: Abstract interface for caching systems

**Key Data Types:**
- `MetricValue`: Named tuple for metrics with confidence intervals
- `CategoricalHyperParameter`: Discrete hyperparameter specification
- `NumericalHyperParameter`: Continuous hyperparameter specification
- `EnvironmentRunParams`: Complete specification for environment runs

**Key Features:**
- Comprehensive episode tracking with timing measurements
- History serialization and deserialization
- Distributed computing support with caching
- Task result management with failure handling
- Hyperparameter optimization support structures

### `tree.py` - Tree Data Structures for Planning

**Key Classes:**
- `BaseNode`: Foundation node class with visit counts and confidence bounds
- `ActionNode`: Node representing actions with Q-values
- `BeliefNode`: Node representing belief states with V-values
- `NodeMixin`: Inherited from anytree for tree operations

**Key Features:**
- Tree visualization with comprehensive node information
- Optimal action selection for both cost and reward settings
- Belief update integration within tree nodes
- Visit count tracking for exploration strategies
- Confidence bound management for algorithms like UCB

**Utility Functions:**
- `print_tree()`: Recursive tree printing with node specifications
- `get_optimal_action_cost_setting()`: Minimum cost action selection
- `get_optimal_action_reward_setting()`: Maximum reward action selection

### `cost.py` - Cost and Reward Calculations

**Key Functions:**
- `belief_expectation_cost(belief, action, env)`: Expected cost calculation for weighted particle beliefs
- `belief_expectation_reward(belief, action, env)`: Expected reward calculation (negative of cost)

**Key Features:**
- Weighted expectation calculations using particle weights
- Integration with WeightedParticleBelief for efficient computation
- Support for both cost-based and reward-based formulations

### `config_types.py` - Configuration Data Structures

**Key Classes:**
- `EnvironmentConfig`: Configuration specification for environments
- `PolicyConfig`: Configuration specification for policies
- `BeliefConfig`: Configuration specification for beliefs
- `ExperimentConfig`: Complete experiment specification

**Key Features:**
- Standardized configuration format with class names and parameters
- Type hints for component relationships
- Structured experiment definition support

### `model_trainer.py` - Model Learning Framework

**Key Classes:**
- `ModelTrainer`: Abstract base class for learning POMDP models from data

**Key Features:**
- Integration with Gymnasium environments for data collection
- Separate training for state transition and observation models
- Configurable sample count for training data
- Abstract interface for custom model learning implementations

## Architecture Patterns

### Abstract Base Classes
All core modules follow the Abstract Base Class pattern, defining clear contracts that implementations must follow while allowing flexibility in specific approaches.

### Configuration Management
Consistent configuration handling across all components with:
- Deterministic ID generation for caching and comparison
- Serialization support for complex data types
- Factory methods for object creation from configurations

### Logging Integration
Centralized logging system with:
- Hierarchical logger names for easy filtering
- Configurable debug levels
- Property-based loggers for pickle compatibility

### Type Safety
Extensive use of type hints and runtime validation:
- TYPE_CHECKING imports to avoid circular dependencies
- Comprehensive input validation with informative error messages
- Generic type support for flexible implementations

## Usage Patterns

### Environment Implementation
```python
class MyEnvironment(DiscreteActionsEnvironment):
    def __init__(self, param1, param2, **kwargs):
        super().__init__(
            discount_factor=0.95,
            name="MyEnvironment",
            space_info=SpaceInfo(SpaceType.DISCRETE, SpaceType.CONTINUOUS),
            **kwargs
        )
    
    def get_actions(self):
        return ["up", "down", "left", "right"]
    
    # Implement abstract methods...
```

### Policy Implementation
```python
class MyPolicy(Policy):
    def action(self, belief):
        # Planning logic here
        selected_actions = [best_action]
        info_vars = [PolicyInfoVariable("nodes_expanded", 1000)]
        return selected_actions, PolicyRunData(info_vars)
    
    @classmethod
    def get_space_info(cls):
        return PolicySpaceInfo(SpaceType.DISCRETE, SpaceType.CONTINUOUS)
```

### Belief Management
```python
# Create initial belief
belief = get_initial_belief(env, n_particles=1000, resampling=True)

# Update through episode
for action in episode_actions:
    next_belief, observation = sample_next_belief(belief, action, env)
    belief = next_belief
```

### Simulation Execution
```python
# Define simulation task
class MySimulationTask(SimulationTask):
    def run(self):
        # Execute simulation
        return results
    
    def get_config_id(self):
        return config_to_id(self.config)

# Execute with task manager
results, identifiers = task_manager.run_tasks(tasks, task_ids)
```

This core module provides the essential building blocks for implementing POMDP planning algorithms, with robust abstractions that support both research experimentation and production deployments.