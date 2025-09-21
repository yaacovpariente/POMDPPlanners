# Environment Module (`POMDPPlanners/core/environment.py`)

This module provides the foundational abstractions for POMDP environments, defining the core interface that all environments must implement.

## Core Classes and Enums

### `SpaceType` (Enum)
Categorizes the types of action and observation spaces:
- `DISCRETE`: Finite, countable spaces
- `CONTINUOUS`: Real-valued continuous spaces  
- `MIXED`: Combination of discrete and continuous elements

### `SpaceInfo` (Dataclass)
Contains space type information for an environment:
- `action_space`: SpaceType for actions
- `observation_space`: SpaceType for observations

## Abstract Base Classes

### `ObservationModel`
Abstract base class for observation models that inherit from `Distribution`.

**Constructor Parameters:**
- `next_state`: The state after taking an action
- `action`: The action that was taken

**Abstract Methods:**
- `sample(n_samples: int = 1) -> List[Any]`: Sample observations given the next state and action

**Methods:**
- `probability(values: List[Any]) -> np.ndarray`: Calculate observation probabilities (raises NotImplementedError by default)

### `StateTransitionModel`
Abstract base class for state transition models that inherit from `Distribution`.

**Constructor Parameters:**
- `state`: Current state
- `action`: Action to be taken

**Abstract Methods:**
- `sample(n_samples: int = 1) -> List[Any]`: Sample next states given current state and action

**Methods:**
- `probability(values: List[Any]) -> np.ndarray`: Calculate transition probabilities (raises NotImplementedError by default)

## Main Environment Classes

### `Environment` (Abstract Base Class)
The core abstract class that all POMDP environments must inherit from.

**Constructor Parameters:**
- `discount_factor`: Discount factor for future rewards (float)
- `name`: Environment identifier (str)
- `space_info`: SpaceInfo object describing action/observation spaces
- `output_dir`: Optional directory for logging output (Path)
- `debug`: Enable debug logging (bool, default: False)

**Properties:**
- `logger`: Returns a configured logger instance (property for pickling compatibility)
- `config_id`: Generates deterministic identifier based on configuration parameters

**Abstract Methods:**
- `state_transition_model(state, action) -> StateTransitionModel`: Return transition model
- `observation_model(next_state, action) -> ObservationModel`: Return observation model  
- `reward(state, action) -> float`: Calculate immediate reward
- `is_terminal(state) -> bool`: Check if state is terminal
- `initial_state_dist() -> Distribution`: Return initial state distribution
- `initial_observation_dist() -> Distribution`: Return initial observation distribution
- `is_equal_observation(observation1, observation2) -> bool`: Compare observations for equality

**Methods:**
- `sample_next_step(state, action)`: Sample next state, observation, and reward
- `cache_visualization(history, cache_path)`: Cache visualization data (optional override)
- `compute_metrics(histories)`: Compute environment-specific metrics (optional override)

**Special Methods:**
- `__eq__`: Comprehensive equality comparison handling numpy arrays and nested structures
- `__hash__`: Hash based on config_id for use in sets/dictionaries

### `DiscreteActionsEnvironment`
Specialized abstract class for environments with discrete action spaces.

**Additional Abstract Method:**
- `get_actions() -> List[Any]`: Return list of all possible actions

### `EnvironmentGenerator` (Abstract Base Class)
Factory pattern for generating environment instances.

**Constructor Parameters:**
- `name`: Generator identifier

**Abstract Methods:**
- `generate_environment() -> Environment`: Create and return an Environment instance

## Key Features

### Configuration Management
- Automatic generation of deterministic `config_id` from environment parameters
- Serialization support for complex data types including numpy arrays
- Exclusion of non-serializable objects (like loggers) from configuration

### Logging Integration
- Centralized logging system with configurable debug levels
- Logger as property to maintain pickle compatibility
- Hierarchical logger names for easy filtering

### Equality and Hashing
- Deep equality comparison with special handling for numpy arrays
- Hash-based on configuration for use in caching systems
- Support for nested data structures and custom types

## Usage Patterns

### Basic Environment Implementation
```python
class MyEnvironment(DiscreteActionsEnvironment):
    def __init__(self, param1, param2, **kwargs):
        super().__init__(
            discount_factor=0.95,
            name="MyEnvironment",
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.CONTINUOUS
            ),
            **kwargs
        )
        self.param1 = param1
        self.param2 = param2
    
    def get_actions(self):
        return ["up", "down", "left", "right"]
    
    # Implement other abstract methods...
```

### Environment Comparison
```python
env1 = MyEnvironment(param1=1.0, param2="test")
env2 = MyEnvironment(param1=1.0, param2="test")
assert env1 == env2  # True - same configuration
assert hash(env1) == hash(env2)  # True - same config_id
```

This module forms the foundation for all POMDP environments in the framework, providing consistent interfaces for state transitions, observations, rewards, and configuration management.