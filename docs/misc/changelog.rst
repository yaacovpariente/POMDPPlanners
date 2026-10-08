.. _changelog:

Changelog
=========

All notable changes to POMDPPlanners are documented here, newest release first.
This project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Each ``#NNN`` links to the pull request that introduced the change.


Release 0.6.0
-------------

**New environments, a results website with 3D episode viewers, and vectorized models across the board**

Breaking Changes:
^^^^^^^^^^^^^^^^^

- Common metrics now share one name across all environments, defined in
  ``CommonMetricName`` (for example ``task_completion_rate`` replaces
  ``success_rate``, ``goal_reaching_rate`` and ``win_rate``). Values are
  unchanged (:gh:`310`).
- Environments report metrics through the per-step ``step_info`` channel
  (:gh:`231`, :gh:`232`).
- Each environment's episodes are drawn by one ``episode_visualizer()``; the
  per-environment GIF renderers are gone (:gh:`304`).
- The Tiger and Sanity vectorized models and the Tiger visualizer moved into
  their own packages. The environment classes keep their import paths
  (:gh:`292`).
- ``compute_metrics`` raises ``ValueError`` for an episode cached before
  ``StepData.info`` existed, instead of scoring it as zeros. Names, values and
  ordering are unchanged for new histories.
- The same check runs per episode on any history whose steps lack the declared
  channels. Episodes with no transition steps are exempt.
- ``compute_metrics`` raises ``ValueError`` on an empty history list in every
  environment. Before, environments returned zeros, ``[]`` or an unrelated error.

New Features:
^^^^^^^^^^^^^

- New environments: Battleship (:gh:`252`), discrete and continuous Maze
  (:gh:`253`), Firefighting (:gh:`287`), CaptureTheFlag (:gh:`288`), Chicken
  Invaders (:gh:`289`) and Snake (:gh:`290`).
- ``RacetrackPOMDP`` on HighwayEnv's ``racetrack-v0``, with a matched
  fully-observed baseline that shares its dynamics and reward (:gh:`234`,
  :gh:`236`, :gh:`238`). The POMDP arm sees its own pose, speed, lane pose,
  road curvature ahead and noisy detections of nearby cars; only cars out of
  range (``max_detection_range_m``) or hidden behind others are withheld.
- Racetrack planner models: ``KnownTrackModel`` uses the track map and
  ``ObservedTrackModel`` uses only the camera's curvature readings. Both have
  matching torch versions for VOPP, and ``highway-env`` is a dev-only
  dependency.
- Results website, started with the new ``pomdp-report`` command. Browse runs
  and tuning studies and replay any episode in a 3D viewer (:gh:`296`,
  :gh:`318`, :gh:`319`).
- 3D episode viewers for every environment, also embedded in the docs and
  README (:gh:`297`, :gh:`298`, :gh:`299`, :gh:`300`, :gh:`305`, :gh:`323`,
  :gh:`324`).
- Torch vectorized models for ten more environments (:gh:`333`) and
  vectorized beliefs for the six that lacked one (:gh:`294`).
- C++ kernels for six more environments (:gh:`334`).
- Isaac Lab: factored generative and perception models, and analytic
  planner-side models for Franka reach and ANYmal navigation (:gh:`233`,
  :gh:`237`).
- Model learning: DAgger loop for transition models, batched transitions and
  rewards, evaluation by planning return, and MLflow logging of every fitted
  model (:gh:`245`–:gh:`250`).
- Optuna studies stop early once the Pareto front plateaus, and plot the
  evidence (:gh:`243`). ``EarlyStoppingConfig`` is now in
  ``POMDPPlanners.core.simulation`` (:gh:`244`).
- Danger zones are visible in Light-Dark, PacMan and Laser Tag episodes
  (:gh:`313`, :gh:`314`).
- Tiger, CartPole, MountainCar, RockSample, LaserTag and Push now compute
  metrics through ``step_info`` and ``get_metric_specs`` (:gh:`232`). PacMan
  and Light-Dark keep their own ``compute_metrics``.
- A resumed run reruns only cache entries written before the per-step channel
  existed, instead of discarding the whole cache.
- ``Environment.step_info`` is also called once on the terminal step, with
  ``action`` and ``next_state`` set to ``None``.
- Added ``order_and_fill_metrics``, which returns every declared metric in
  declaration order.

Bug Fixes:
^^^^^^^^^^

- Fixed CVaR return confidence intervals (:gh:`239`).
- Fixed reward ranges for zero-mean danger shocks (:gh:`240`).
- Fixed the RockSample rock-check sensor accuracy law (:gh:`286`).
- Fixed the violations found by the new cross-environment conformance and
  terminal-state tests (:gh:`241`, :gh:`315`).

Documentation:
^^^^^^^^^^^^^^

- Docs restructured into guide, common-API, environment and planner sections,
  with one page per environment (:gh:`266`, :gh:`295`, :gh:`321`).
- Every environment page has a precise, self-contained formal definition
  (:gh:`293`, :gh:`301`, :gh:`322`, :gh:`328`, :gh:`329`, :gh:`331`,
  :gh:`332`).
- README: every environment listed, a planner comparison on a time budget,
  pip install, the results site and a tuning walkthrough, with videos
  (:gh:`302`, :gh:`306`–:gh:`308`, :gh:`311`, :gh:`312`, :gh:`316`,
  :gh:`320`).

Others:
^^^^^^^

- Bumped the package version to 0.6.0.
- Tests: vectorized beliefs and models checked against their scalar
  environments, stronger planner correctness tests, visualization coverage,
  and per-env API checks folded into the cross-env harnesses (:gh:`220`,
  :gh:`254`, :gh:`277`, :gh:`283`, :gh:`309`, :gh:`317`).
- CI runs the Docker tests daily on develop (:gh:`275`).
- Extended the frozen metric baseline with a terminated-episode shape, a
  metric-ordering snapshot and a confidence-bounds snapshot, which is what
  makes the migration's "no name, value or ordering moved" claim checkable
  rather than assumed. The bounds matter on their own: a declared reduction
  that yields the right mean from the wrong per-episode samples leaves the
  point estimate intact and moves only the interval.
- Tiger now reports a real 95% t-interval instead of a zero-width one. An
  episode that never reports a metric's channel is left out of that metric's
  average.

Release 0.5.0
-------------

**CARLA, nuPlan and Isaac Lab environments; vectorized planning and batched GPU beliefs**

Breaking Changes:
^^^^^^^^^^^^^^^^^

- Environments now own their visualization file naming. Callers that built
  visualization paths themselves must go through the environment
  (:gh:`203`).

New Features:
^^^^^^^^^^^^^

- Added ``CarlaPOMDP``, a forward-only world environment backed by the CARLA
  simulator (:gh:`204`).
- Added a pluggable generative-model interface and a multi-agent perception
  world for CARLA (:gh:`207`).
- Added planner-side perception in the CARLA world observation model, so the
  planner model and the world emit identical raw observations while the
  belief filters and infers hidden per-agent intent (:gh:`208`).
- Added a destination/route option for CARLA with a success terminal and
  route metrics (:gh:`221`).
- Added a headless CARLA server pool for parallel episode simulation
  (:gh:`217`).
- Added a nuPlan POMDP environment — world, model, perception and belief —
  with a POMCPOW example (:gh:`210`).
- Added an Isaac Lab POMDP environment and visualizer (:gh:`206`).
- Added a vectorized belief tree to ``POMDPPlanners.core`` (:gh:`214`).
- Added the VOPP vectorized planner and vectorized generative models
  (:gh:`215`, :gh:`216`).
- Added ``BatchedParticleBelief``, a torch-backed belief for batched GPU
  belief filtering (:gh:`218`).
- Added opt-in ``is_*_hit_terminal`` hazard-terminates-episode flags across
  the hazard environments, with a draw-coupled termination gate
  (:gh:`211`, :gh:`212`).
- Added ``EVADE_WHEN_SPOTTED`` and ``PURSUE`` opponent-policy support to the
  vectorized LaserTag model (:gh:`219`).

Bug Fixes:
^^^^^^^^^^

- Fixed nuPlan control commands, perception densities and metrics
  (:gh:`224`).
- Fixed the PacMan vectorized belief updater to model motion slip, which the
  scalar updater already applied (:gh:`213`).

Documentation:
^^^^^^^^^^^^^^

- Rewrote the README to be user-focused (:gh:`223`) and added CARLA and
  Isaac Sim images rendered by the package (:gh:`222`).
- Cited the *Vectorized Online POMDP Planning* paper on the VOPP planner
  (:gh:`225`).
- Removed the nuPlan evaluation notebook (:gh:`226`).

Others:
^^^^^^^

- Bumped the package version to 0.5.0 across ``pyproject.toml``,
  ``POMDPPlanners/__init__.py``, ``test_setup.py`` and the Sphinx ``release``
  string, which had been stale at 0.2.0 (:gh:`228`).
- Refactored CARLA onto per-channel observation models with an
  ``encode_observation`` seam (:gh:`209`).
- Added coverage for the LaserTag belief updater honouring ``opponent_policy``
  (:gh:`202`) and for CARLA observation equality and collision-penalty reward
  (:gh:`205`).


Release 0.4.0 (2026-05-28)
--------------------------

**Constrained planners, hazard-centric reward variants, run-progress notifications**

Breaking Changes:
^^^^^^^^^^^^^^^^^

- Renamed the LightDark ``DANGEROUS_STATES`` reward model to
  ``HIGH_VARIANCE_STATES`` (:gh:`175`).
- Renamed reward variants to hazard-centric names across five environments
  (:gh:`186`).
- Dropped Python 3.8 and 3.9; the minimum supported version is now 3.10
  (:gh:`192`).

New Features:
^^^^^^^^^^^^^

- Added the constrained planners CPOMCPOW and CPFT-DPW together with a
  ``ConstrainedEnvironment`` ABC (:gh:`170`).
- Added run-progress tracking: Slack notifications on run start, finish and
  failure, backed by a SQLite progress DB with an external stall-detection
  watcher for deaths the process cannot witness itself (:gh:`172`).
- Added per-task progress callbacks for the Dask and PBS backends
  (:gh:`173`).
- Added opt-in circular dangerous areas to ``PacManPOMDP`` (:gh:`176`), plus
  a dangerous-area step counter and a total-danger-encounters aggregate
  (:gh:`187`).
- Added high-variance and decaying reward variants to RockSample — with
  batch acceleration — Push and PacMan (:gh:`180`, :gh:`181`, :gh:`182`).
- Added ``to_unique_support_distribution`` to
  ``VectorizedWeightedParticleBelief`` (:gh:`188`).
- Added a selectable LaserTag ``OpponentPolicy`` with evade, pursue and
  evade-when-spotted behaviours (:gh:`196`, :gh:`197`).

Bug Fixes:
^^^^^^^^^^

- Fixed the LaserTag opponent to evade and react to the robot's pre-move
  position (:gh:`195`).
- Aligned the C++ reward kernels with their Python counterparts across five
  environments (:gh:`183`) and closed Python-side reward-kernel bugs in
  PacMan and RockSample (:gh:`184`).
- Fixed three environment correctness bugs in the sanity, Tiger and
  LightDark environments (:gh:`185`).
- Fixed the PacMan visualizer to render the ghost belief heatmap for
  vectorized particle beliefs (:gh:`174`).

Documentation:
^^^^^^^^^^^^^^

- Rewrote the README Architecture and Running Experiments sections
  (:gh:`190`).
- Standardized planner ``References:`` sections — one reference per planner,
  conference/published links preferred over arXiv (:gh:`200`).

Others:
^^^^^^^

- Replaced the deprecated ``pkg_resources`` with ``importlib.metadata``
  (:gh:`171`).
- Extracted generic dangerous-area Numba kernels (:gh:`177`) and reward
  models for RockSample, Push, LaserTag and PacMan (:gh:`178`, :gh:`179`).
- Added MIT SPDX headers to all source files (:gh:`189`).
- The release workflow now builds an sdist only; the previous wheel carried
  an unportable ``linux_x86_64`` platform tag that PyPI rejects (:gh:`193`).
- Made environment tests independent of constructor defaults (:gh:`199`).


Release 0.3.1 (2026-05-12)
--------------------------

**Packaging hotfix for a broken 0.3.0 sdist**

Bug Fixes:
^^^^^^^^^^

- Bundled the C++ headers in the sdist, without which ``pip install`` from
  PyPI failed to compile. Added packaging regression tests (:gh:`168`).

Others:
^^^^^^^

- Removed benchmark files from the repository root (:gh:`167`).
- Removed ``CHANGELOG.md`` (:gh:`166`). Release notes returned as this page
  in 0.5.0.


Release 0.3.0 (2026-05-11)
--------------------------

**Risk-sensitive benchmarking mechanics, arena tree, large performance and correctness pass**

Breaking Changes:
^^^^^^^^^^^^^^^^^

- ``Environment.reward`` gained an optional ``next_state`` parameter so
  penalty terms (obstacles, dangerous areas) are scored against the realised
  post-transition state rather than a fresh sample. Threaded through rollout
  and POMCP-DPW.
- PacMan state is now a raw NumPy ``ndarray``; the native batch path and
  obstacle/danger-penalty handling were updated accordingly.
- Removed ``Tree.backup_belief_v_from_children``. Per-algorithm V-backup
  formulas are now inlined at the call sites — POMCP visited-only, iCVaR
  CVaR-over-children, others ``max``.
- Removed the redundant ``gamma`` parameter from Sparse PFT.
- PEP 639 license-file metadata now requires ``setuptools>=77`` at build
  time.

New Features:
^^^^^^^^^^^^^

- Added stochastic obstacle-collision and dangerous-area mechanics to
  ``PushPOMDP``, ``ContinuousPushPOMDP``, ``RockSamplePOMDP``,
  ``LaserTagPOMDP`` and ``ContinuousLaserTagPOMDP``. Bernoulli per-step
  penalty draws produce heavy-tailed return distributions for benchmarking
  risk-sensitive planners against expected-value MCTS on the same
  environment.
- Added a PacMan particle-belief visualization: a ghost-position particle
  overlay on the sprite viewer, with bundled DejaVu fonts for deterministic
  CI rendering.
- Added typed accessors and compound mutation helpers to the arena tree
  (``increment_visit_count``, ``update_action_q_with_return``); all planners
  migrated to the typed surface.

Bug Fixes:
^^^^^^^^^^

- PFT-DPW / BetaZero: the immediate-reward stash is now keyed on
  ``action_id``; it was overwriting across sibling actions.
- POMCP-DPW: corrected saturated branch-reward propagation.
- ConstrainedZero: aligned the failure target to a single episode-level
  target.
- CVaR exploration: corrected the LCB formula, horizon-zero handling, LCB
  overflow on long horizons, and vectorized-belief support.
- Sparse-sampling iCVaR: the unvisited-action mask was inverted.
- Sparse sampling: fixed the branching-factor loop off-by-one.
- BetaZero: unified continuous sampling across the rollout and
  tree-expansion paths.
- PacMan: fixed multiple audit-flagged bugs — state encoding, reward sign on
  capture, terminal handling, and native/Python parity.
- CartPole and ContinuousLightDark: observation-model corrections; dropped
  the ContinuousLightDark sampler grid-clip that biased particle weights.
- LaserTag: fixed scalar/batch log-probability asymmetry, a pickling
  regression on the continuous variant, observation log-probability for the
  B1/B2 kernels, and a terminal-sentinel guard on ``kernel.probability``.
- Tiger: fixed listen-action impossible-observation handling; Push now
  advertises a reward range that includes the obstacle penalty.
- RockSample: corrected the dangerous-area sign convention.
- Continuous Push (discrete-actions variant): fixed obstacle-hit-probability
  forwarding.

Performance:
^^^^^^^^^^^^

- PFT-DPW belief sampling is amortized O(log K) via an inline CDF on
  weighted-particle beliefs.
- Migrated the iCVaR CVaR-computation kernels to Numba; faster
  beacon-likelihood evaluation and systematic resampling on the iCVaR path.
- Arena-tree column-store buffers are pre-sized to avoid reallocation during
  tree growth.
- Added a Tiger-pattern sampling fast path to environment sampling.

Others:
^^^^^^^

- Added iCVaR-POMCPOW tests pinning the LCB / CVaR-exploration formulas
  against the published reference.
- Added arena-tree coverage and MCTS planner tree-structure tests.
- Added env-API conformance tests for ``hash_action`` / ``hash_observation``
  across all environments.
- Added a metric-invariants sanity suite — rate bounds, count
  non-negativity, CI bounds, return-shift linearity, belief invariants —
  wired into the per-environment metric tests.


Release 0.2.0 (2026-04-20)
--------------------------

**Vectorized belief updaters and distributed-execution robustness**

New Features:
^^^^^^^^^^^^^

- Added vectorized belief updaters for the RockSample, PacMan, LightDark
  (continuous and discrete), CartPole, MountainCar, Push, Continuous Push,
  Continuous LaserTag and SafetyAnt environments, with batched NumPy updates
  for significant throughput gains.
- Added observation-model-aware vectorized belief updaters for the LightDark
  family.
- Added a ``ParallelizationLevel`` option for hyperparameter tuning, enabling
  episode-level parallelism alongside Optuna-level parallelism.
- Added Gaussian process noise to the CartPole and MountainCar state
  transition models.

Bug Fixes:
^^^^^^^^^^

- The CartPole and MountainCar vectorized updaters now correctly add process
  transition noise.
- Removed duplicated reward logic and fixed an RNG-stream divergence in the
  ``sample_next_step`` paths.
- Post-run visualization no longer crashes Dask runs with
  ``cannot pickle '_asyncio.Task'``. Visualization is dispatched through the
  simulator's task manager as ``EnvironmentVisualizationTask``\ s, scaling
  across the full cluster instead of being capped by a local joblib pool.
- The distributed task pipeline is now OS-agnostic. Workers on a different OS
  than the client no longer die unpickling ``pathlib.PosixPath``:
  ``EnvironmentVisualizationTask`` returns ``Dict[str, bytes]`` from a
  worker-private scratch directory, and the episode / hyperparameter tuning
  tasks ship ``cache_dir`` as ``str`` with a graceful fallback to
  console-only logging when the path does not resolve on the worker's OS.

Performance:
^^^^^^^^^^^^

- ``PushPOMDP.sample_next_step``: ~5.2x speedup.
- ``RockSamplePOMDP.sample_next_step``: ~6.35x speedup.
- ``DiscreteLightDarkPOMDP.sample_next_step``: inlined sampling, pure-Python
  math for reward and beacon checks, squared-distance beacon proximity.
- ``DiscreteDistribution``: faster initialization and sampling.
- ``CovarianceParameterizedMultivariateNormal``: cached Cholesky transpose.

Others:
^^^^^^^

- Added shared belief-level equivalence test utilities that validate
  vectorized updaters against non-vectorized baselines across environments.
- Added a three-layer benchmark suite for planner and environment
  performance testing.
- Added a weekly CI workflow running the full slow-test suite; 117 tests are
  marked ``slow``. The full suite also runs on pushes to ``master``, while
  other branches and PRs skip slow tests.
- Split the Docker build into a reusable base image plus a thin CI layer,
  auto-building when the base image is missing from GHCR.
- Added an auto-rebase workflow for open PRs whenever ``develop`` is updated.
- ``PacManPOMDP`` methods now accept NumPy array states.
- Hyperparameter tuning computes ``optuna_n_jobs`` and ``episode_n_jobs``
  once in ``__init__``.
- CartPole and MountainCar belief tests compare against deterministic physics
  rather than noisy samples.


Release 0.1.0 (2026-03-21)
--------------------------

**Initial release**

- First public release of POMDPPlanners: core abstractions (Environment,
  Policy, Belief, Distributions), the MCTS planner family, the initial
  environment collection, and the simulation and hyperparameter-tuning
  framework.


Maintainers
-----------

POMDPPlanners is currently maintained by Yaacov Pariente
(`@yaacovpariente <https://github.com/yaacovpariente>`_).
