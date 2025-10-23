# Golden Visualization Files

This directory contains reference GIF files used for **snapshot testing** (also called **golden file testing**) of environment visualization outputs.

## What Are Snapshot Tests?

**Snapshot tests** capture the exact output of a system at a point in time and use it as a reference for future comparisons. Unlike traditional unit tests that assert specific values, snapshot tests:

1. **Capture complete output**: Store entire GIF files rather than testing individual properties
2. **Detect any change**: Any modification to the output (even a single pixel) triggers a test failure
3. **Easy to update**: When changes are intentional, simply regenerate the snapshot
4. **Self-documenting**: The golden files serve as examples of expected output

This approach is ideal for visualization testing where:
- Outputs are complex (multi-frame animated GIFs)
- Manual verification is needed (visual review of GIFs)
- Changes should be deliberate and reviewed

## Purpose

These golden files ensure that visualization outputs remain consistent across code changes. They serve as:
- **Regression tests**: Catch unintended changes to visualization output
- **Reference outputs**: Provide examples of expected visualization behavior
- **Determinism verification**: Ensure visualizers produce identical outputs given identical inputs
- **Visual documentation**: Show what each environment's visualization looks like

## Test Structure

The snapshot tests are organized in `test_environment_visualizations_golden_files.py` with the following structure:

### Test Classes

**`TestVisualizationConsistency`** - Golden file snapshot tests
- One test per environment (6 total)
- Each test generates a visualization and compares it against the golden file
- Tests are named: `test_<environment>_visualization_consistency`

**`TestVisualizationDeterminism`** - Determinism verification tests
- Verifies that repeated visualizations produce identical outputs
- Confirms the snapshot tests are reliable (no randomness)

### Test Components

Each snapshot test follows this pattern:

```python
def test_<environment>_visualization_consistency(self, temp_output_dir):
    # 1. Create deterministic episode with fixed seed
    history = create_deterministic_<environment>_episode(seed=42)

    # 2. Create environment and visualizer
    env = <Environment>POMDP(discount_factor=0.95)
    visualizer = <Environment>Visualizer(env)

    # 3. Generate visualization
    output_path = temp_output_dir / "<environment>_test.gif"
    visualizer.create_visualization(history, output_path)

    # 4. Compare against golden file (or create if missing)
    compare_or_create_golden_file(
        output_path,
        "<environment>_visualization.gif",
        "test_<environment>_visualization_consistency",
    )
```

### Determinism Requirements

For snapshot tests to be reliable, the following must be deterministic:
- ✅ **Random seed**: Fixed at 42 for all episode generation
- ✅ **Environment parameters**: Fixed and identical across runs
- ✅ **Action sequences**: Predetermined, not random
- ✅ **Matplotlib backend**: Uses 'Agg' for consistent rendering
- ✅ **Belief states**: Mock beliefs with fixed weights

## How It Works

### First Test Run (Golden File Creation)
1. Test runs and finds no golden file exists
2. Test generates visualization and saves it as golden file
3. Test emits warning and skips with message about golden file creation
4. **Action Required**: Review the generated GIF to ensure it's correct

### Subsequent Test Runs (Comparison)
1. Test runs and generates new visualization
2. Test compares SHA256 hash of new output against golden file
3. If hashes match: Test passes ✓
4. If hashes differ: Test fails with detailed error message

## Directory Structure

```
golden_visualizations/
├── README.md                              # This file
├── rock_sample_visualization.gif          # RockSample reference output
├── pacman_visualization.gif               # PacMan reference output
├── light_dark_visualization.gif           # LightDark reference output
├── push_visualization.gif                 # Push reference output
├── laser_tag_visualization.gif            # LaserTag reference output
└── safety_ant_velocity_visualization.gif  # SafeAntVelocity reference output
```

## Workflow Examples

### Initial Setup (First Run)

```bash
# Run tests - golden files will be created
pytest POMDPPlanners/tests/test_environments/test_environment_visualizations_golden_files.py -v

# Output:
# test_rock_sample_visualization_consistency SKIPPED
# Warning: GOLDEN FILE CREATED: .../rock_sample_visualization.gif

# Review generated files
ls -lh POMDPPlanners/tests/test_environments/golden_visualizations/
```

### Normal Testing (Subsequent Runs)

```bash
# Run tests - compares against golden files
pytest POMDPPlanners/tests/test_environments/test_environment_visualizations_golden_files.py -v

# Output if visualizations match:
# test_rock_sample_visualization_consistency PASSED ✓
# test_pacman_visualization_consistency PASSED ✓
# ...

# Output if visualization changed:
# test_rock_sample_visualization_consistency FAILED
# AssertionError: VISUALIZATION OUTPUT CHANGED!
# Expected hash: abc123...
# Actual hash:   def456...
```

### Updating Golden Files (After Intentional Changes)

```bash
# Scenario: You modified visualization logic and want to update reference

# Step 1: Delete old golden file
rm POMDPPlanners/tests/test_environments/golden_visualizations/rock_sample_visualization.gif

# Step 2: Re-run test to create new golden file
pytest POMDPPlanners/tests/test_environments/test_environment_visualizations_golden_files.py::TestVisualizationConsistency::test_rock_sample_visualization_consistency -v

# Step 3: Review new golden file
# (Open the GIF file and verify it looks correct)

# Step 4: Run test again to confirm it passes
pytest POMDPPlanners/tests/test_environments/test_environment_visualizations_golden_files.py::TestVisualizationConsistency::test_rock_sample_visualization_consistency -v
# PASSED ✓
```

### Regenerating All Golden Files

```bash
# Remove all golden files
rm POMDPPlanners/tests/test_environments/golden_visualizations/*.gif

# Re-run all tests to regenerate
pytest POMDPPlanners/tests/test_environments/test_environment_visualizations_golden_files.py -v

# Review all generated files
# ...

# Run again to confirm all pass
pytest POMDPPlanners/tests/test_environments/test_environment_visualizations_golden_files.py -v
```

## When Tests Fail

### Scenario 1: Unintentional Change (Bug)
```
VISUALIZATION OUTPUT CHANGED!
Test: test_rock_sample_visualization_consistency
Expected hash: abc123...
Actual hash:   def456...
```

**Actions:**
1. Review your recent code changes
2. Check if you modified visualization logic unintentionally
3. Compare the new GIF against the golden file visually
4. Fix the bug that caused the change
5. Re-run test to verify it passes

### Scenario 2: Intentional Change (Feature)
```
VISUALIZATION OUTPUT CHANGED!
Test: test_rock_sample_visualization_consistency
Expected hash: abc123...
Actual hash:   def456...
```

**Actions:**
1. Review the new output file (shown in error message)
2. Verify the change is correct and intentional
3. Delete the old golden file
4. Re-run test to create new golden file
5. Commit the new golden file to version control

### Scenario 3: Dependency Update
```
VISUALIZATION OUTPUT CHANGED!
```

This can happen when matplotlib or PIL/Pillow versions change.

**Actions:**
1. Check if dependency versions changed (pip list)
2. Review changes in library release notes
3. Verify new output looks correct
4. Update golden files if necessary
5. Document the dependency version change

## Best Practices

### ✅ DO
- Review golden files visually when first created
- Commit golden files to version control
- Update golden files when making intentional visualization changes
- Run tests locally before pushing changes
- Document why golden files were updated in commit messages

### ❌ DON'T
- Blindly update golden files without reviewing output
- Delete golden files and forget to regenerate them
- Ignore test failures without investigation
- Modify golden files manually (always regenerate via tests)
- Commit broken or incorrect golden files

## Troubleshooting

### Problem: Test always fails even though output looks identical

**Possible Causes:**
- Non-deterministic episode generation (check random seed)
- Matplotlib backend differences
- System-dependent rendering
- Floating-point precision differences

**Solutions:**
```python
# Ensure deterministic seeding
np.random.seed(42)

# Check matplotlib backend
import matplotlib
print(matplotlib.get_backend())  # Should be 'Agg' for headless

# Use exact environment parameters
env = RockSamplePOMDP(..., seed=42)  # Always use seed
```

### Problem: Golden file creation warning doesn't appear

**Possible Causes:**
- Golden file already exists
- Test is using wrong golden file path
- Permissions issue creating directory

**Solutions:**
```bash
# Check if file exists
ls POMDPPlanners/tests/test_environments/golden_visualizations/

# Check directory permissions
ls -ld POMDPPlanners/tests/test_environments/golden_visualizations/

# Remove file if exists
rm POMDPPlanners/tests/test_environments/golden_visualizations/rock_sample_visualization.gif
```

### Problem: Hash comparison fails immediately after golden file creation

**This is expected behavior!**

The first run creates the golden file and skips. Run the test a second time:

```bash
# First run: Creates golden file
pytest test_environment_visualizations_golden_files.py::test_rock_sample_visualization_consistency -v
# SKIPPED - golden file created

# Second run: Compares against golden file
pytest test_environment_visualizations_golden_files.py::test_rock_sample_visualization_consistency -v
# PASSED ✓
```

## Technical Details

### Hash Algorithm
- **Algorithm**: SHA256
- **Comparison**: Byte-for-byte file comparison via hash
- **Sensitivity**: Detects any change to the GIF file, no matter how small

### File Format
- **Format**: GIF (Graphics Interchange Format)
- **Generated by**: matplotlib + PIL/Pillow writer
- **Frame rate**: 0.8-2 fps (environment-specific)
- **Compression**: Standard GIF LZW compression

### Determinism Requirements

For tests to work correctly, all randomness must be controlled:
1. **NumPy seed**: `np.random.seed(42)` before episode generation
2. **Environment seed**: Pass `seed=42` to environment constructor
3. **Fixed action sequences**: Use predetermined action sequences
4. **Matplotlib backend**: Use 'Agg' backend for consistent rendering

## Version Control

### What to Commit
- ✅ Golden GIF files (`*.gif`)
- ✅ This README
- ✅ Test code (`test_environment_visualizations_golden_files.py`)

### What NOT to Commit
- ❌ Temporary test outputs
- ❌ Failed comparison outputs
- ❌ Debug/experimental GIF files

### Git LFS Recommendation

Since GIF files can be large (100KB - 5MB), consider using Git LFS:

```bash
# Install Git LFS
git lfs install

# Track GIF files
git lfs track "POMDPPlanners/tests/test_environments/golden_visualizations/*.gif"

# Commit .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for golden visualization files"

# Add and commit golden files
git add POMDPPlanners/tests/test_environments/golden_visualizations/*.gif
git commit -m "Add golden visualization reference files"
```

## Contact

For questions or issues with visualization testing:
- Review test file: `POMDPPlanners/tests/test_environments/test_environment_visualizations_golden_files.py`
- Check visualizer implementations: `POMDPPlanners/environments/*/visualizer.py`
- See CLAUDE.md for general testing guidelines
