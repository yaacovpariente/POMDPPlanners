# Changelog

All notable changes to POMDPPlanners are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-20

### Added
- Vectorized belief updaters for RockSample, PacMan, LightDark (continuous and discrete), CartPole, MountainCar, Push, Continuous Push, Continuous LaserTag, and SafetyAnt POMDPs, with batched NumPy updates for significant throughput gains.
- Observation-model-aware vectorized belief updaters for the LightDark POMDP family.
- Shared belief-level equivalence test utilities that validate vectorized updaters against non-vectorized baselines across environments.
- `ParallelizationLevel` option for hyperparameter tuning, enabling episode-level parallelism alongside Optuna-level parallelism.
- Three-layer benchmark suite for planner and environment performance testing.
- Weekly CI workflow that runs the full slow-test suite; 117 tests marked as `slow`.
- Split Docker build pipeline: reusable base image plus a thin CI layer, with auto-build when the base image is missing from GHCR.
- Auto-rebase workflow for open PRs whenever `develop` is updated.
- Gaussian process noise in the CartPole and MountainCar state transition models.

### Changed
- Full test suite (including slow tests) now runs on pushes to `master`; other branches and PRs skip slow tests.
- `develop` branch added as a CI test workflow trigger.
- `PacManPOMDP` environment methods now accept NumPy array states.
- Refactored hyperparameter tuning to compute `optuna_n_jobs` and `episode_n_jobs` once in `__init__`.
- `CartPolePOMDP` and `MountainCarPOMDP` belief tests now compare against deterministic physics rather than noisy samples.

### Performance
- `PushPOMDP.sample_next_step`: ~5.2x speedup.
- `RockSamplePOMDP.sample_next_step`: ~6.35x speedup.
- `DiscreteLightDarkPOMDP.sample_next_step`: inlined sampling, pure-Python math for reward and beacon checks, squared-distance beacon proximity.
- `DiscreteDistribution`: faster init and sampling.
- `CovarianceParameterizedMultivariateNormal`: cached Cholesky transpose.

### Fixed
- CartPole and MountainCar vectorized updaters now correctly add process transition noise.
- Removed duplicated reward logic and fixed an RNG-stream divergence in `sample_next_step` paths.
- Post-run visualization no longer crashes Dask runs with `cannot pickle '_asyncio.Task'`. Visualization is now dispatched through the simulator's task manager as `EnvironmentVisualizationTask`s, scaling across the full cluster instead of being capped by a local joblib pool.
- Distributed task pipeline is OS-agnostic. Workers on a different OS than the client (e.g. a Linux scheduler with Windows workers over a Tailscale/meshnet) no longer die unpickling `pathlib.PosixPath`. `EnvironmentVisualizationTask` now returns `Dict[str, bytes]` from a worker-private scratch directory, and `EpisodeSimulationTask`/`HyperParameterTuningSimulationTask` ship `cache_dir` as `str` with a graceful fallback to console-only logging when the path doesn't resolve on the worker's OS.

## [0.1.0] - Initial release

- Initial public release of POMDPPlanners.

[0.2.0]: https://github.com/yaacovpariente/POMDPPlanners/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yaacovpariente/POMDPPlanners/releases/tag/v0.1.0
