# Hyperparameter Optimization Configurations

This directory contains pre-configured hyperparameter optimization setups for all POMDP planners and environments in the POMDPPlanners library. Each planner has its own configuration file with optimized parameter ranges for all supported environments.

## Available Planners

### Implemented Planners

1. **POMCP** (`pomcp_configs.py`)
   - Basic Monte Carlo Tree Search with UCB1 action selection
   - Parameters: exploration_constant, depth, n_simulations, min_samples_per_node
   - 9 environment configurations

2. **POMCP_DPW** (`pomcp_dpw_configs.py`) 
   - POMCP with Double Progressive Widening
   - Parameters: exploration_constant, depth, n_simulations, k_a, k_o, alpha_a, alpha_o, min_samples_per_node
   - 9 environment configurations

3. **StandardSparseSamplingDiscreteActionsPlanner** (`sparse_sampling_configs.py`)
   - Sparse sampling algorithm with finite lookahead trees
   - Parameters: branching_factor, depth
   - 9 environment configurations

4. **DiscreteActionSequencesPlanner** (`discrete_action_sequences_configs.py`)
   - Open-loop exhaustive search for optimal action sequences
   - Parameters: depth, n_return_samples
   - 9 environment configurations

### Planners Requiring Custom Implementation

The following planners require additional implementation work due to their dependency on ActionSampler interfaces:

- **PFT_DPW**: Requires domain-specific ActionSampler implementations
- **POMCPOW**: Requires ActionSampler for progressive widening
- **SparsePFT**: Requires ActionSampler for action space exploration

## Supported Environments

All configurations support the following 9 POMDP environments:

1. **TigerPOMDP** - Classic tiger problem
2. **CartPolePOMDP** - Cart-pole balancing with partial observations  
3. **MountainCarPOMDP** - Mountain car climbing with partial observations
4. **PushPOMDP** - Object manipulation task
5. **SanityPOMDP** - Simple test environment
6. **LaserTagPOMDP** - Multi-agent laser tag game
7. **SafeAntVelocityPOMDP** - Safety-constrained ant locomotion
8. **DiscreteLightDarkPOMDP** - Discrete light-dark navigation
9. **ContinuousLightDarkPOMDPDiscreteActions** - Continuous light-dark with discrete actions

## Usage Examples

### Basic Usage

```python
from POMDPPlanners.experiments.configs.hyperparameter_configs import ALL_CONFIGS
from POMDPPlanners.simulations.hyper_parameter_tuning_simulations import HyperParameterOptimizer

# Use all configurations
optimizer = HyperParameterOptimizer(
    cache_dir_path=Path("./optimization_results"),
    experiment_name="Complete_Hyperparameter_Study"
)

results = optimizer.optimize(ALL_CONFIGS)
```

### Planner-Specific Usage

```python
from POMDPPlanners.experiments.configs.hyperparameter_configs.pomcp_configs import ALL_POMCP_CONFIGS

# Optimize only POMCP configurations
results = optimizer.optimize(ALL_POMCP_CONFIGS)
```

### Environment-Specific Usage

```python
from POMDPPlanners.experiments.configs.hyperparameter_configs import get_configs_by_environment

# Get all planner configurations for Tiger POMDP
tiger_configs = get_configs_by_environment("TigerPOMDP")
results = optimizer.optimize(tiger_configs)
```

### Custom Selection

```python
from POMDPPlanners.experiments.configs.hyperparameter_configs.pomcp_configs import POMCPTigerConfig
from POMDPPlanners.experiments.configs.hyperparameter_configs.sparse_sampling_configs import SparseSamplingTigerConfig

# Compare specific planner-environment combinations
comparison_configs = [
    POMCPTigerConfig(),
    SparseSamplingTigerConfig()
]

results = optimizer.optimize(comparison_configs)
```

## Configuration Structure

Each configuration file follows this pattern:

```python
# Define hyperparameter ranges for the planner
PLANNER_HYPERPARAMETERS = [
    NumericalHyperParameter(low=min_val, high=max_val, name="param_name"),
    # ... more parameters
]

# Create configuration classes for each environment
class PlannerEnvironmentConfig(HyperParameterRunParams):
    def __new__(cls):
        return super().__new__(
            cls,
            environment=EnvironmentClass(discount_factor=0.95),
            policy_cls=PlannerClass,
            hyper_parameters=PLANNER_HYPERPARAMETERS,
            num_episodes=episodes,
            num_steps=steps,
            direction=HyperParameterOptimizationDirection.MAXIMIZE,
            parameter_to_optimize="average_return"
        )

# Collect all configurations
ALL_PLANNER_CONFIGS = [config() for config in all_config_classes]
```

## Parameter Ranges

Parameter ranges have been chosen based on:

1. **Literature recommendations** for established algorithms
2. **Computational feasibility** for the given environments  
3. **Empirical testing** to ensure reasonable optimization times
4. **Environment characteristics** (episode length, complexity, etc.)

## Extending Configurations

To add new environments or modify existing configurations:

1. **Add new environments**: Create new config classes following the existing pattern
2. **Modify parameter ranges**: Update the hyperparameter lists at the top of each file
3. **Add new planners**: Create a new configuration file following the existing structure
4. **Update the index**: Add new configurations to `__init__.py`

## Notes

- All configurations use `average_return` as the optimization target
- Episode counts and step limits are tuned per environment for reasonable execution times
- All planners use a discount factor of 0.95 unless the environment requires otherwise
- Progressive widening parameters (k_a, k_o, alpha_a, alpha_o) are optimized for POMCP_DPW