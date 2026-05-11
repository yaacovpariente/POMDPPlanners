# Changelog

All notable changes to POMDPPlanners are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-05-11

### Added
- Stochastic obstacle-collision and dangerous-area mechanics in `PushPOMDP`, `ContinuousPushPOMDP`, `RockSamplePOMDP`, `LaserTagPOMDP`, and `ContinuousLaserTagPOMDP` (Bernoulli per-step penalty draws producing heavy-tailed return distributions for benchmarking risk-sensitive planners against expected-value MCTS on the same env).
- PacMan particle-belief visualization: ghost-position particle overlay rendered on top of the sprite-based viewer, with bundled DejaVu fonts for deterministic CI rendering.
- iCVaR-POMCPOW paper-formula tests pinning the LCB / CVaR-exploration formulas against the published reference.
- Arena-tree coverage and MCTS planner tree-structure tests.
- Env-API conformance tests for `hash_action` / `hash_observation` across all environments.
- Metric-invariants sanity-check suite (rate bounds, count non-negativity, CI bounds, return-shift linearity, belief invariants) wired into per-env metric tests.

### Changed
- Arena tree: typed accessors and compound mutation helpers (`increment_visit_count`, `update_action_q_with_return`); all planners migrated to the typed surface.
- PacMan state representation refactored to a raw NumPy `ndarray`; native batch path and obstacle/danger-penalty handling updated accordingly.
- `Environment.reward` extended with an optional `next_state` parameter so penalty terms (obstacles, dangerous areas) can be scored against the realised post-transition state rather than a fresh sample. Threaded through rollout and POMCP-DPW.
- PEP 639 license-file metadata; `setuptools>=77` build requirement.

### Performance
- PFT-DPW belief sampling: amortized O(log K) via inline CDF on weighted-particle beliefs.
- iCVaR CVaR-computation kernels migrated to Numba; faster beacon-likelihood and systematic resampling on the iCVaR path.
- Arena-tree column-store buffers pre-sized to avoid reallocations during tree growth.
- Tiger-pattern sampling fast-path on environment sampling.

### Fixed
- PFT-DPW / BetaZero: immediate-reward stash now keyed on `action_id` (was overwriting across sibling actions).
- POMCP-DPW: saturated branch-reward propagation.
- ConstrainedZero: failure-target alignment.
- CVaR exploration: corrected LCB formula; horizon-zero handling; LCB overflow on long horizons; vectorized-belief support.
- Sparse-sampling iCVaR: unvisited-action mask was inverted.
- Sparse-sampling: branching-factor loop off-by-one.
- BetaZero: unified continuous sampling across rollout and tree-expansion paths.
- PacMan: multiple audit-flagged environment bugs (state encoding, reward sign on capture, terminal handling, native/Python parity).
- CartPole and ContinuousLightDark: observation-model corrections.
- ContinuousLightDark: dropped the sampler grid-clip that biased particle weights.
- LaserTag: scalar/batch log-probability asymmetry; pickling regression on the continuous variant; observation-log-probability for the B1/B2 kernels; terminal-sentinel guard on `kernel.probability`.
- Tiger: listen-action impossible-observation handling and Push reward-range advertisement.
- RockSample: dangerous-area sign correction.
- Continuous Push (discrete-actions variant): obstacle-hit-probability handling.

### Removed
- `Tree.backup_belief_v_from_children` (per-algorithm V-backup formulas now inlined at the call sites — POMCP visited-only, iCVaR CVaR-over-children, others use `max`).
- Redundant gamma in sparse PFT.

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

[0.3.0]: https://github.com/yaacovpariente/POMDPPlanners/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yaacovpariente/POMDPPlanners/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yaacovpariente/POMDPPlanners/releases/tag/v0.1.0
