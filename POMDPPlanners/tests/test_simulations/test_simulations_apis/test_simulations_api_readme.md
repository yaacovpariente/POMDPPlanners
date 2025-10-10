# Simulation API Tests - Mixin-Based Architecture

This directory contains tests for all simulation API implementations using a mixin-based architecture to maximize code reuse and ensure consistent behavior across implementations.

## Architecture Overview

The test architecture uses the **Mixin Pattern** to share common test methods across all API implementations:

```
test_simulations_apis/
├── api_test_fixtures.py           # Shared pytest fixtures and test data generators
├── api_test_mixins.py              # Mixin classes with concrete test methods
├── test_local_simulations_api_new.py   # LocalSimulationsAPI tests (using mixins)
├── test_dask_simulations_api.py    # DaskSimulationsAPI tests (existing, to be migrated)
└── test_pbs_simulations_api.py     # PBSSimulationsAPI tests (to be created)
```

## Key Components

### 1. **api_test_fixtures.py**

Contains reusable pytest fixtures for all API tests:

- `sample_environment`: Creates a TigerPOMDP environment for testing
- `sample_policy`: Creates a POMCP policy
- `sample_environment_params`: Generates `EnvironmentRunParams` configurations
- `sample_hyperparameter_configs`: Generates `HyperParameterRunParams` configurations
- `sample_planner_generators`: Generates `PlannerGenerator` objects for benchmark testing
- `create_temp_cache_dir`: Helper function for temporary cache directories
- `MockPlannerGenerator`: Mock implementation of PlannerGenerator for testing

These fixtures ensure all API tests use consistent test data.

### 2. **api_test_mixins.py**

Provides concrete test methods organized into logical mixins:

#### `InitializationTestsMixin`
- `test_api_initialization_with_default_params`
- `test_api_initialization_with_custom_cache_dir`
- `test_api_initialization_with_debug_mode`

#### `RunMultipleEnvironmentsTestsMixin`
- `test_run_multiple_environments_returns_correct_types`
- `test_run_multiple_environments_result_structure`
- `test_run_multiple_environments_dataframe_columns`
- `test_run_multiple_environments_with_profiling_enabled`
- `test_run_multiple_environments_cache_creation`

#### `HyperparameterOptimizationTestsMixin`
- `test_hyperparameter_optimization_returns_correct_type`
- `test_hyperparameter_optimization_result_structure`
- `test_hyperparameter_optimization_with_custom_experiment_name`
- `test_hyperparameter_optimization_cache_directory_creation`
- `test_hyperparameter_optimization_with_statistical_params`

#### `OptimizeAndEvaluateTestsMixin`
- `test_optimize_and_evaluate_basic_execution`
- `test_optimize_and_evaluate_returns_correct_types`
- `test_optimize_and_evaluate_empty_configs_raises_error`
- `test_optimize_and_evaluate_cache_directory_handling`

#### `BenchmarkEnvironmentsOnPlannerGeneratorsTestsMixin`
- `test_benchmark_environments_basic_execution`
- `test_benchmark_environments_returns_correct_types`
- `test_benchmark_environments_result_structure`
- `test_benchmark_environments_empty_generators_raises_error`
- `test_benchmark_environments_with_custom_experiment_name`
- `test_benchmark_environments_with_profiling_enabled`
- `test_benchmark_environments_cache_directory_handling`
- `test_benchmark_environments_with_custom_statistical_params`

#### `ErrorHandlingTestsMixin`
- `test_invalid_alpha_value_raises_error`
- `test_invalid_confidence_interval_raises_error`
- `test_negative_n_jobs_interpretation`

### 3. **Concrete Test Classes**

Each API implementation has its own test class that:
1. Inherits from **all relevant mixins**
2. Implements the **factory method** (`create_api`)
3. Adds **implementation-specific tests**

Example:

```python
class TestLocalSimulationsAPI(
    InitializationTestsMixin,
    RunMultipleEnvironmentsTestsMixin,
    HyperparameterOptimizationTestsMixin,
    OptimizeAndEvaluateTestsMixin,
    BenchmarkEnvironmentsOnPlannerGeneratorsTestsMixin,
    ErrorHandlingTestsMixin,
):
    """Test suite for LocalSimulationsAPI."""

    def create_api(self, cache_dir_path=None, debug=False, **kwargs):
        """Factory method to create LocalSimulationsAPI instance."""
        return LocalSimulationsAPI(cache_dir_path=cache_dir_path, debug=debug)

    # LocalSimulationsAPI-specific tests
    def test_local_api_scheduler_address_parameter_ignored(self, sample_environment_params):
        """Test that scheduler_address parameter is ignored."""
        # ... implementation-specific test
```

## Benefits of This Architecture

### 1. **DRY Principle (Don't Repeat Yourself)**
- Common test logic is written once in mixins
- All APIs are tested with the same scenarios automatically
- Reduces code duplication by ~70-80%

### 2. **Consistency**
- Ensures all APIs behave consistently for the same operations
- Identical test coverage across implementations
- Same test names and documentation across all APIs

### 3. **Maintainability**
- Changes to interface requirements propagate automatically
- Easy to add new common tests - just add to mixin
- Easy to add implementation-specific tests - just add to concrete class

### 4. **Flexibility**
- Each API can override specific tests if needed
- Implementation-specific tests are clearly separated
- Can selectively include only relevant mixins

### 5. **Clear Organization**
- Fixtures separate from tests separate from implementation
- Test purpose is clear from mixin names
- Easy to find related tests

## How to Use

### Running All Tests for a Specific API

```bash
# Run all LocalSimulationsAPI tests (including inherited from mixins)
pytest POMDPPlanners/tests/test_simulations/test_simulations_apis/test_local_simulations_api_new.py -v

# Run all DaskSimulationsAPI tests
pytest POMDPPlanners/tests/test_simulations/test_simulations_apis/test_dask_simulations_api.py -v
```

### Running Specific Test Category

```bash
# Run only initialization tests for all APIs
pytest POMDPPlanners/tests/test_simulations/test_simulations_apis/ -k "test_api_initialization" -v

# Run only hyperparameter optimization tests
pytest POMDPPlanners/tests/test_simulations/test_simulations_apis/ -k "hyperparameter_optimization" -v
```

### Running Implementation-Specific Tests Only

```bash
# Run only LocalSimulationsAPI-specific tests (not from mixins)
pytest POMDPPlanners/tests/test_simulations/test_simulations_apis/test_local_simulations_api_new.py -k "local_api" -v
```

## Adding New Common Tests

To add a new test that should run for all APIs:

1. **Choose or create appropriate mixin** in `api_test_mixins.py`
2. **Add the test method** with proper documentation
3. **The test automatically runs** for all API implementations that inherit the mixin

Example:

```python
# In api_test_mixins.py
class RunMultipleEnvironmentsTestsMixin:
    def test_new_common_feature(self, sample_environment_params):
        """Test new feature common to all APIs.

        Purpose: Validates new functionality

        Given: ...
        When: ...
        Then: ...

        Test type: integration
        """
        api = self.create_api()
        # Test implementation
```

## Adding Implementation-Specific Tests

To add a test for only one API:

1. **Add the test method** to the appropriate concrete test class
2. **Use descriptive name** that indicates it's implementation-specific
3. **Document why it's specific** to that implementation

Example:

```python
# In test_local_simulations_api_new.py
class TestLocalSimulationsAPI(...):
    def test_local_api_specific_feature(self):
        """Test LocalSimulationsAPI-specific method.

        Purpose: Validates unique feature only in LocalSimulationsAPI

        Given: LocalSimulationsAPI instance
        When: Specific method is called
        Then: Expected behavior occurs

        Test type: unit
        """
        api = self.create_api()
        # Test implementation
```

## Migration Plan

### Current State
- `test_local_simulations_api.py`: Existing comprehensive tests (to be replaced/migrated)
- `test_dask_simulations_api.py`: Existing tests (to be migrated to mixin pattern)
- `test_pbs_simulations_api.py`: Existing tests (to be migrated to mixin pattern)

### Migration Steps

1. ✅ Create `api_test_fixtures.py` with shared fixtures
2. ✅ Create `api_test_mixins.py` with common test methods
3. ✅ Create `test_local_simulations_api_new.py` using mixins
4. 🔄 Migrate `test_dask_simulations_api.py` to use mixins
5. 🔄 Migrate `test_pbs_simulations_api.py` to use mixins (or create if missing)
6. 🔄 Verify all tests pass
7. 🔄 Remove old test files and rename `*_new.py` files

### For Each API Migration

1. **Create factory method** `create_api()`
2. **Inherit from mixins** for common functionality
3. **Extract implementation-specific tests** from old file
4. **Remove redundant tests** that are covered by mixins
5. **Verify test coverage** is maintained or improved

## Best Practices

### Writing New Tests

1. **Use fixtures** from `api_test_fixtures.py` whenever possible
2. **Document tests** using the required template (Purpose/Given/When/Then)
3. **Keep tests focused** - one concept per test
4. **Use descriptive names** - test names should explain what they test
5. **Test actual behavior** - avoid testing implementation details

### Fixture Usage

```python
def test_example(self, sample_environment_params, tmp_path):
    """Use fixtures for test data and temporary directories."""
    api = self.create_api(cache_dir_path=tmp_path)
    results, stats_df = api.run_multiple_environments_and_policies(
        environment_run_params=sample_environment_params,  # From fixture
        alpha=0.05,
        confidence_interval_level=0.95,
        n_jobs=1,
    )
    # Assertions...
```

### Factory Method Pattern

Each test class must implement `create_api()`:

```python
def create_api(self, cache_dir_path=None, debug=False, **kwargs):
    """Create API instance for testing.

    Args:
        cache_dir_path: Optional cache directory
        debug: Enable debug mode
        **kwargs: Implementation-specific parameters

    Returns:
        API instance for this test class
    """
    # For LocalSimulationsAPI
    return LocalSimulationsAPI(cache_dir_path=cache_dir_path, debug=debug)

    # For PBSSimulationsAPI
    queue = kwargs.get('queue', 'test_queue')
    return PBSSimulationsAPI(
        queue=queue,
        cache_dir_path=cache_dir_path,
        debug=debug
    )
```

## Testing Strategy

### Unit Tests
- Test individual methods with mocked dependencies
- Focus on parameter validation and error handling
- Fast execution (< 1 second per test)

### Integration Tests
- Test actual execution with real environments
- Use minimal parameters for fast execution (1-2 episodes, 2-3 steps)
- Verify end-to-end workflows

### Test Coverage Goals
- All interface methods covered
- All error conditions tested
- All parameter combinations validated
- Implementation-specific features tested

## Troubleshooting

### Test Fails Only for One API

1. Check if the test is in a mixin that assumes certain behavior
2. Consider overriding the test in the specific API test class
3. Or add implementation-specific logic to handle the case

### Fixture Not Found

1. Ensure `api_test_fixtures.py` is in the same directory
2. Import fixtures explicitly if needed
3. Check fixture scope and naming

### Factory Method Errors

1. Verify `create_api()` signature matches across all test classes
2. Ensure **kwargs are handled for API-specific parameters
3. Check that required parameters are provided with defaults

## Summary

This mixin-based architecture provides:
- **Maximum code reuse** through shared test methods
- **Consistent testing** across all API implementations
- **Easy maintenance** with centralized common tests
- **Flexibility** for implementation-specific variations
- **Clear organization** with separation of concerns

The pattern ensures that as the `SimulationsAPIInterface` evolves, all implementations are automatically tested for compliance with new requirements.
