# Pyright Type Check Summary

**Total:** 672 errors, 5 warnings, 0 informations

## Major Error Categories

### 1. **Matplotlib/Visualization Type Issues** (~150-200 errors)
- `Cannot access attribute "set_xlim" for class "ndarray[Any, dtype[Any]]"`
- `Cannot access attribute "scatter" for class "ndarray[Any, dtype[Any]]"`
- **Root cause**: Matplotlib `plt.subplots()` returns tuple but code expects axes object directly

### 2. **Optional/None Subscript Errors** (~80-100 errors)
- `Object of type "None" is not subscriptable`
- **Root cause**: Missing null checks before accessing dictionary/list elements

### 3. **NumPy/SciPy Type Issues** (~60-80 errors)
- `Argument of type "ndarray[Unknown, Unknown]" cannot be assigned to parameter "cov" of type "int"`
- `Type "bool_" is not assignable to declared type "float"`
- **Root cause**: NumPy type annotations not matching expected types

### 4. **List Type Invariance** (~40-60 errors)
- `"list[NumericalHyperParameter]" is not assignable to "List[HyperParameterFeature]"`
- **Root cause**: Python's List type is invariant, not covariant

### 5. **Import Resolution Issues** (~40-60 errors)
- `Import "anytree" could not be resolved`
- `Import "setuptools" could not be resolved`
- **Root cause**: Missing type stubs or development dependencies

### 6. **Cache File Corruption** (~30-40 errors)
- Generated files in `temp_sim_cache/` and `test_hyper_param_cache/`
- Syntax errors and undefined variables
- **Fix**: Exclude cache directories from type checking

### 7. **Method Override Issues** (~20-30 errors)
- `Method "probability" overrides class "ObservationModel" in an incompatible manner`
- Parameter name mismatches between base and override

### 8. **Belief Hierarchy Issues** (~20-30 errors)
- `Cannot access attribute "to_unique_support_distribution" for class "Belief"`
- `Cannot access attribute "inplace_update" for class "Belief"`
- **Root cause**: Abstract Belief class missing methods that concrete implementations have

### 9. **Missing Type Annotations** (~50-100 errors)
- Functions without proper return types
- Unbound variables

### 10. **Miscellaneous** (~50-100 errors)
- Various other type mismatches and issues

## Progress Made
✅ **Fixed numeric type variance issues** in `environment_configs.py` (reduced from 672+ to 672 errors)
- Converted `[(1, 1), ...]` to `[(1.0, 1.0), ...]` for beacon/obstacle coordinates

## Next Steps (by Priority)
1. **Exclude cache directories** from type checking (~30-40 error reduction)
2. **Fix matplotlib subplot type issues** (~150-200 error reduction)
3. **Add null checks for Optional types** (~80-100 error reduction)
4. **Fix NumPy type annotations** (~60-80 error reduction)
5. **Address list type invariance** (~40-60 error reduction)

## Files to Review
- Most errors in visualization/plotting code
- `planners_hyperparam_configs.py` has many Optional subscript issues
- Environment files have matplotlib type issues
- MCTS planner files have belief hierarchy issues