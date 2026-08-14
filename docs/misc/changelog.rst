.. _changelog:

Changelog
=========

All notable changes to POMDPPlanners are documented here, newest release first.
This project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Each ``#NNN`` links to the pull request that introduced the change.


Release 0.6.0 (WIP)
-------------------

**Existing environments migrated onto the shared per-step metrics channel**

Breaking Changes:
^^^^^^^^^^^^^^^^^

- Environment metrics are now derived from the per-step ``StepData.info``
  channel, so an episode ``History`` cached *before* that field existed can no
  longer be scored. ``compute_metrics`` raises ``ValueError`` for it rather
  than returning metrics, because every channel would be absent and a rate
  would read 0.0 — indistinguishable from a planner that never succeeded.
  Metric names, values and ordering are unchanged for any history produced by
  the current code.
- The same guard fires for a history built by hand or by a runner that does not
  call ``step_info``, and it is applied per episode, so a partially warm cache
  cannot let one measured episode vouch for unmeasured ones beside it. It keys
  on the environment's declared channels, not on ``info`` merely being
  non-empty. An episode with no transition steps is exempt: it legitimately
  measured nothing.
- ``compute_metrics`` now raises ``ValueError`` on an empty history list, in
  every environment. A metric over no episodes is an average over nothing, so
  any answer is invented: a zero reads like a genuine measurement of zero, an
  omitted name silently shortens the declared list, and an empty list claims the
  environment has no metrics. The three disagreed environment by environment
  before — Tiger, MountainCar, CartPole, RockSample and both Push variants
  returned zeros, both LaserTags, PacMan, CARLA and nuPlan returned ``[]``, and
  the light-dark pair raised ``Data must contain at least one element`` from
  ``confidence_interval``. Nothing in the simulation pipeline produces an empty
  batch: ``compute_statistics_environment_policy_pair`` already rejects one
  before metrics are reached, so this is a caller error, not a degenerate run.

New Features:
^^^^^^^^^^^^^

- Added ``RacetrackPOMDP``, a racing environment on HighwayEnv's ``racetrack-v0``
  with a **matched fully-observed baseline**. Both arms are selected by one
  ``ObservationMode`` argument and share a single dynamics path: the assembled
  simulator configurations are equal except for the ``observation`` key, which is
  asserted by a test rather than left as a convention. The point is attribution —
  every other driving environment here is partially observed by construction, so a
  planner's performance drop could never be pinned on partial observability alone
  rather than on a change of dynamics, reward or map. The POMDP arm observes a
  12x12 occupancy grid of presence and on-road flags over a ±18 m window,
  withholding every velocity, every vehicle identity and everything outside it; the
  baseline observes absolute position and velocity for the ego and the nearest
  vehicles. Ships with a forward-only world, a planner-side generative model whose
  ego dynamics reproduce the simulator's kinematic bicycle exactly, and a belief
  that recovers opponent motion by differencing consecutive grids.
- Two deliberate departures from what ``racetrack-v0`` ships, both worth knowing
  before reading a result off this environment. **Longitudinal control is enabled.**
  Under the shipped action configuration ``ContinuousAction`` is lateral-only:
  acceleration is pinned at zero and the ``target_speeds`` key is inert, because it
  belongs to ``DiscreteMetaAction``. The ego could therefore never brake for the
  opponent and could only swerve, which removes most of what partial observability
  costs. The flag is a dynamics key applied identically to both arms, so the matched
  pair is preserved. And **the baseline is a near-MDP, not a true MDP**: it still
  withholds the other vehicles' driver policy, so a gap measured against it is a
  lower bound on the cost of partial observability, not the whole of it.
- **The racetrack POMDP arm now observes its own speed.** Its reading is a
  ``(grid, ego_speed)`` pair rather than a bare occupancy grid: the same
  ``(2, 12, 12)`` array as before, plus a single signed number in metres per second.
  Withholding ego speed was never a decision — it fell out of picking
  ``OccupancyGridObservation``, which reports presence and on-road flags and nothing
  else. Every real car has a speedometer, and hiding it put a *second* source of
  partial observability on top of the one the matched pair exists to measure, so a gap
  between the arms was partly measuring a blindfold nobody meant to fit. The other
  vehicles' velocities and identities are still withheld; that is the hidden state
  under study. The belief deliberately does **not** stamp the reading into its
  particles — it lets the likelihood reweight them, because stamping a measurement and
  then scoring particles on agreeing with it is double-counting and would flatten the
  speed spread the term needs in order to discriminate.
- Two things worth knowing if you touch that observation block. **Scalar speed, not**
  ``(vx, vy)``: the pair would also reveal the ego's heading through ``atan2``, which
  the ego-aligned grid deliberately withholds, so the scalar adds exactly a speedometer
  and nothing more. And **the ego kinematics block asks for two vehicles, not one**. In
  highway-env 1.12.1 ``vehicles_count=1`` does not mean "the ego alone": the observer
  asks the road for ``count = vehicles_count - 1``, ``Road.close_objects_to`` guards its
  truncation with ``if count:`` so ``count=0`` means *no limit*, and the following slice
  ``[-vehicles_count + 1:]`` becomes ``[0:]``. Measured on a three-vehicle track it
  returns a ``(3, 2)`` block carrying two opponents' velocities. At 2 the block is the
  size it claims, and the world drops the second row before the observation leaves it.
- **The map-based racetrack model now draws the road and scores it.**
  ``KnownTrackModel`` used to predict the on-road layer as all ones — "drivable
  everywhere" — and then leave it out of the likelihood, on the argument that a
  term identical across particles vanishes at normalisation. The first half was
  simply wrong: measured against the live circuit over 113 steps, all-ones
  disagreed with the simulator on **79.2%** of cells. It now rebuilds the ego's
  pose on the track from its arclength, lane offset and lane-relative angle and
  walks the lane centrelines through the grid window, which brings that to
  **13.2%**, and the layer is scored, so the model's likelihood stops being blind
  to ``lat`` and ``ang``. The best lane offset moves from "any value, it cannot
  tell" to the true one on 34 of 113 steps with a median error of 0.00 m.
  ``ObservedTrackModel`` is untouched and still reads 18.3%.
- Two things to know before reading that 13.2%. **Highway-env draws lane
  centrelines, not filled corridors**: ``fill_road_layer_by_lanes`` walks each
  lane at one waypoint per cell and marks the cell it lands in, so only about 29
  of 144 cells are ever set and predicting *no* road anywhere already scores
  20.1%. Judge the number against that floor, not against zero. And **a lane
  change re-bases the arclength**, because the world numbers it from whichever
  lane the ego first occupied and re-anchors on the first visit to a new one.
  Rebuilding the ego's position from its Frenet pose lands within 0.38 m while it
  stays put and up to 58.7 m out afterwards; split on that, the same renderer
  scores 5.6% on the 73 steps that never re-based and 25.3% on the 40 that did.
  The re-basing is pre-existing and is not addressed here.
- ``TrackGeometry`` therefore carries the lane layout as well as the curvature:
  each segment's width and the signed lateral offset of every lane beside it,
  walked out of the lane graph by ``build_track_geometry``, plus a
  ``centreline_pose`` that integrates the curvature profile back into positions
  and headings for a renderer to draw. Offsets are stored **per segment** rather
  than once per lap, because ``next_lane`` can hand the walk onto the parallel
  lane partway round — three of the eight seeded starts do — and one lap-wide set
  would mix ``+5`` and ``-5`` into a third lane that does not exist. A geometry
  built by hand with no lane layout still works and reads as a single-lane road.
  Width is recorded because it is the road property callers ask for, but it plays
  no part in the render, for the centreline reason above.
- The torch model gains the matching ``TrackMapRoadLayer``, dispatched on the same
  signal as the curvature source. Both sides take their sample points from one
  ``centreline_pose`` call, and a parity test pins the rendered layers cell for
  cell and the likelihoods to 1e-9 — VOPP plans against the torch model, so a
  divergence would make the scalar model's road invisible to the planner that
  uses it.
- **The mapless racetrack model now fits the corridor rather than one guessed lane.**
  ``ObservedTrackModel`` predicted the on-road layer as a single centreline. The
  world's layer shows every lane, so that prediction could be slid sideways onto a
  *neighbour* lane and lose nothing — and it did: over 113 steps of random control
  on eight seeds, a state 3 m or more from the truth outscored the truth on **46**
  of them, which is a likelihood actively pointing the filter at the wrong lane.
  The fit now also measures the lanes lying either side of the ego's own and draws
  them all. A state 3 m out beats the truth on **20** steps, against 37 for the
  model holding the actual map; cell disagreement goes from **18.3% to 16.5%**; and
  the true lane offset is recovered as the single best hypothesis on 32 of 113
  steps rather than 20, with the mean error at -0.42 m rather than -0.69 m.
  ``KnownTrackModel`` is untouched and still reads 13.2% and 34 of 113.
- Two things to know about that fit. **Which lane is the ego's own is assumed, not
  measured** — it takes the nearest visible one, because an ego-centred layer says
  where the road is relative to the car and nothing more, and turning that into a
  lane index needs a map this arm does not have. Over those 113 steps the nearest
  visible centreline sat 0.84 m from the truth on average, inside the 3 m cell it is
  quantised to; a car more than half a lane out anchors on the wrong lane. And **the
  render now stops at** ``curvature_window_m``, the distance the curvature was
  actually measured over, instead of drawing to the edge of the 18 m window: past
  the fit the guessed parabola leaves the real road, and a cell left unmarked by
  every particle contributes the same constant to all of them, so declining to guess
  there costs almost no discrimination.
- The curvature estimator itself is unchanged, and it is now the limiting factor.
  Handed the map's curvature and changing nothing else, the same corridor render
  scores 14.8% and recovers the lane offset on 34 of 113 steps — matching the mapped
  model exactly. The gap between that and the shipped 16.5% is curvature error, not
  render error. ``LaneCorridorLayer`` mirrors all of this on the torch side, reading
  the lane offsets and the draw distance off the scalar model by name the same way it
  already read the curvature, so the parity test still pins the two together.
- **The racetrack POMDP arm now has a lane-keeping camera.** Its reading is a
  ``(grid, ego_speed, lane_pose)`` triple; ``lane_pose`` is two numbers, the ego's
  signed lateral offset from the lane centreline in metres and its heading relative
  to the lane in radians. This is the same accident as the missing speedometer, and a
  worse one: the racetrack reward *is* a function of lateral offset, so the agent was
  being scored on lane-centering while never being told its lane offset. It also
  replaces a measurably unreliable inference — ``ObservedTrackModel`` had to read
  lateral position out of the on-road layer, and its likelihood still peaked about
  1 m off the truth. With the camera observed directly, the best lane offset is the
  true one on **105 of 113** steps for ``KnownTrackModel`` (was 34) and **107 of 113**
  for ``ObservedTrackModel`` (was 32), with mean errors of +0.01 m and -0.00 m. Cell
  disagreement on the on-road layer is unchanged at 13.2% and 16.5% — this adds a
  channel, it does not change the road render — and the presence rasteriser still
  matches the world on every cell.
- **That reading is noisy on purpose, and the world is where the noise lives.**
  highway-env's ``lane_offset`` is exact. Emitting it unchanged would hand the planner
  a lane-relative pose no camera delivers, and would quietly do the work the on-road
  layer is there to do — ground truth smuggled into the arm the matched pair exists to
  compare against the fully-observed one. ``RacetrackPOMDP`` therefore takes
  ``lane_lateral_std_m`` (0.05) and ``lane_heading_std_rad`` (0.01, or 0.57 degrees):
  the centimetre-scale lateral and sub-degree heading accuracy a production
  mono-camera lane detector is specified at over the few metres this 3 m-cell window
  covers. The model takes the same two parameters and must be configured to match; a
  test pins that the readings are wrong, and wrong by the configured amount, rather
  than merely present.
- Note the asymmetry with ego speed, which the world still emits **exactly**: a real
  speedometer is accurate to well under a percent, so exactness there is defensible
  and noise would be invented. A lane camera's lateral offset is not.
- Relatedly, ``cell_flip_prob`` and ``ego_speed_std`` on the planner's model are now
  documented for what they are: **not sensor models**, despite the names. The presence
  rasteriser was measured to agree with highway-env exactly — 0 mismatches over 27,225
  swept offsets — and the world emits exact speed, so there is no sensor noise for
  either to describe. They are unfitted tolerances for how wrong a *particle's state*
  is, and fitting them to a sensor would be fitting them to noise that does not exist.
  Their values are unchanged. The new lane widths are the opposite case and are
  labelled as such.
- **The presence layer is now soft-rasterised, so the likelihood has a gradient inside
  a cell.** Hard binning made the presence term piecewise constant: an opponent marks
  one 3 m cell, so a particle 2.9 m from the truth landed in the same cell and scored
  identically to it, and the filter had nothing to follow over most of the resolution
  it has. Each tracked opponent now contributes the Gaussian mass of its position
  falling inside each cell, and the cells combine as "at least one opponent here".
  Measured over 40 seeds at 20 vehicles, nudging an in-window opponent by 1.0 m — a
  third of a cell — moved the log-likelihood on **22 of 22** steps, against **6 of 22**
  before, and the response is graded rather than the old fixed −5.89 cliff at a cell
  boundary. **Where a vehicle is deemed to be has not changed**: the hard bin still
  backs the sampler's mode and still matches the world with 0 missed and 0 invented
  cells over 113 steps, and the on-road mismatch rates are unchanged at 13.2% and
  16.5%. Mirrored in ``RacetrackVectorizedModel``, so VOPP scores the same likelihood;
  scalar and torch agree to 0.0.
- **The observation likelihood now scores whether a vehicle is there, not only where.**
  Presence was read from the state and never from the observation, so in the MDP arm a
  particle whose agent slots were empty was scored on its ego row alone and paid nothing
  for an observation full of traffic. A **detection model** fixes that — a miss rate
  ``presence_miss_prob`` (default 0.05) and a false-alarm rate
  ``presence_false_alarm_prob`` (default 0.02) — and it composes with the soft occupancy
  rather than sitting beside it: a cell reads occupied with probability
  ``q (1 - miss) + (1 - q) fa``. Setting the two rates equal reproduces the symmetric
  squash this replaced, exactly, so the POMDP arm's earlier measurements still stand;
  ``cell_mismatch_prob`` is left doing the one job it is the right shape for, the on-road
  layer's flip rate. Measured on a state holding one opponent 10.5 m ahead against one
  with empty slots, scored on the observation that shows it: the POMDP arm separates them
  by **3.34 nats** and the MDP arm by **14.06 nats**, where the MDP arm previously
  separated them by **−5.06** — that is, backwards. The converse direction costs 1.49 and
  2.98 nats. All four are finite. Mirrored in ``RacetrackVectorizedModel``; scalar and
  torch agree to 0.0.
- Neither rate may be a hard zero in a running filter, and that is the design rather than
  a caveat: an opponent crosses the ±18 m window every few steps, so a ``-inf`` there
  would collapse the filter on each crossing. Measured on 200 particles split half-and-half
  on presence and scored against an observation showing one vehicle, the disagreeing half
  keeps **21.7%** of the weight at the shipped rates and **0.0000%** at zero rates. Zero is
  still accepted, as the control the parity and sampler tests use.
- The false-alarm default is derived from the mechanism rather than fitted, because there
  is nothing to fit: the rasteriser agrees with highway-env exactly, so at the true state
  the observation misses nothing and invents nothing. What it stands for is the K-slot cap
  — the state carries ``max_tracked_agents`` slots, so a window holding more vehicles than
  that shows cells the model cannot account for. As a fraction of occupied window cells
  over 113 steps: **34.67%** at K=1, **1.33%** at K=2, **0.00%** at K=3 and K=4, and
  **0.00%** at K=4 with 30 vehicles. At the shipped K=4 the cap never binds, so 0.02 is the
  rate at the nearest configuration where it does. A model run at K=1 should raise it.
- The MDP arm additionally gains a **clutter model** — ``clutter_position_scale_m``
  (18.0 m) and ``clutter_velocity_scale`` (10.0 m/s), Cauchy — and it is not optional.
  Its slots hold four continuous numbers, so without a distribution over what a false
  alarm *reports*, the miss branch is a density and the false-alarm branch a bare
  probability, which is not a comparison: a perfectly matched slot scores a 4-D Gaussian
  whose peak at the default widths is ``exp(-5.06) = 0.0063``, below the 0.02 false-alarm
  rate, so "no vehicle, spurious detection" would beat a perfect match every time. Cauchy
  rather than the uniform-over-the-field-of-view of PDA and JPDA because these slots are
  ranked by range with no window to be uniform over.
- Both arms' samplers apply the detection model they score, so neither is the density of a
  different model. The MDP sampler drops a filled slot at the miss rate and fills an empty
  one with a Cauchy phantom at the false-alarm rate; the POMDP arm's Bernoulli draw over
  the cell probabilities already carried both halves.
- ``cell_flip_prob`` is renamed ``cell_mismatch_prob`` and joined by a new
  ``agent_position_std_m`` (default 1.0 m), which is the width the presence mass is
  integrated over. **Breaking**: ``cell_flip_prob=`` is no longer accepted, and no
  alias is kept — the parameter is a few days old and unreleased. The new width is
  derived rather than guessed: the belief reads an opponent off the grid as a cell
  centre, so its position is quantised to one 3 m cell (std 0.87 m), and the belief
  adds 0.5 m of stamping jitter, which is 1.0 m in quadrature. ``cell_mismatch_prob``
  keeps its 0.05. It carried the presence layer's floor and ceiling as well until the
  detection model took that over (see above); what is left is its real job, the flip rate
  the *rendered on-road layer* is scored under — the one place a flip model is the right
  shape, because the road error really is a disagreement rate rather than a positional
  spread. Setting ``agent_position_std_m=0`` reproduces the previous
  behaviour cell for cell, which is what the parity and sampler tests use as a control.
- ``TrackedAgentsBelief`` now **warns instead of silently dropping** opponents past its
  slot count. It keeps the nearest ``max_tracked_agents`` and always did, but the world
  stamps every vehicle inside the window, so beyond four the surplus became empty road
  to the planner — a false negative in the only channel the POMDP arm has, and one that
  looks exactly like success. The path had never been exercised: the most opponents
  visible at 20 vehicles was three.
- **The racetrack POMDP arm now observes the whole state except the vehicles it cannot
  see, and the occupancy grid is gone.** This is a redesign of *what the environment
  measures*, not a refactor of how it measures it: the numbers in the observation are
  different numbers, and so is the list of things the planner cannot see. The arm
  withholds **one** thing on purpose — a vehicle outside sensor range or line of sight —
  and reports everything else at near-exact widths. The reading is a
  ``RacetrackObservation(ego_pose, ego_speed, lane_pose, curvature_ahead, detections)``:

  * ``ego_pose`` is new: a ``(4,)`` block of ``x``, ``y``, heading and arclength around
    the lap, at the decimetre and sub-degree widths GPS/IMU and a wheel odometer deliver
    (``ego_position_std_m`` 0.1, ``ego_heading_std_rad`` 0.01, ``ego_arclength_std_m``
    0.1). Its arclength is read off the same lane walk the state's arclength slot is
    numbered against, so the two cannot disagree about where a corner is.
  * ``ego_speed`` and ``lane_pose`` are unchanged — a speedometer and a lane camera.
  * ``curvature_ahead`` is new: the lane's signed curvature at each of
    ``curvature_lookahead_m`` (default 10, 20 and 30 m) along the track from the ego's
    own arclength, which is the other thing a lane-detection camera outputs. The world
    reads it off the true lane graph — the same walk its arclength slot is numbered
    against, rebuilt in the same breath when a lane change re-bases it — and corrupts it
    at ``curvature_std_1pm``.
  * ``detections`` replaces the grid: a ``(K, 5)`` block of
    ``[detected, rel_x, rel_y, rel_vx, rel_vy]`` rows in the ego body frame, unlabeled
    and ordered by measured range. A vehicle that is reported is reported **whole** —
    relative position and both components of relative velocity.

  **The hidden state is therefore vehicles the sensor cannot see, plus driver intent and
  identity**, in place of "resolution and field of view", which is what a 3 m grid over a
  ±18 m window was actually withholding. A vehicle beyond ``max_detection_range_m`` (40 m)
  is absent; a vehicle behind a closer one is absent, under a deterministic geometric rule
  — a blocker is a disc of ``blocker_half_width_m`` (1.0 m, a 2 m car) and masks anything
  within the half-angle it subtends, ``arcsin(w / r)``, which is O(K²) per step and exact
  for a disc.
- **That makes the MDP and POMDP arms two ends of a continuum in one number.**
  ``max_detection_range_m`` is the dial: at ``R -> inf`` the POMDP reading is the state to
  within the sensor widths, and as R shrinks the traffic drops out of it first while
  everything else stays. Measured on a two-car scene with the shipped widths, ``R = 40``
  reports one of two cars and ``R = 1e9`` reports both, each row within 0.45 m and
  0.26 m/s of the truth. Sweeping R is now a legitimate experiment axis rather than a
  configuration detail.
- Two things this replaced, and why. The reading used to withhold the ego's own pose and
  the tangential half of relative velocity. Both were defensible as sensor realism and
  both were the wrong problem: withholding the pose made the arm a **localisation**
  problem on top of the tracking one, and withholding the crossing rate made it an
  **estimation** problem on top of it again, so a gap measured against the MDP baseline
  was measuring three things at once and could not be attributed to any of them. Radial-only
  Doppler is also not what a production stack reports — fusing a radar with a camera or a
  lidar gives both components. ``radial_velocities`` survives as a geometry helper for
  anyone writing a time-to-collision rule; nothing in the observation path calls it.
- **The payoff is that the likelihood got simpler, and discriminates on more.**
  ``observation_log_probability`` in POMDP mode is now a product of closed-form terms per
  particle: Gaussians over the four ego-pose entries (the heading one **wrapped**, so a
  particle on the far side of the branch cut is not charged 6.28 rad for a hundredth of
  one), the speedometer residual, the lane camera's two residuals and each curvature-ahead
  sample, then a per-rank Bernoulli over the detections with a Gaussian in each matched
  detection's position and full relative velocity. Scoring one particle touches
  ``7 + L + 5K`` residuals where the grid touched 288 cells twice, and the flat torch
  observation is 30 wide against 291. The ego-pose channel is what now pins a particle's
  **arclength**, which the curvature channel alone used to have to do — and which a mapless
  model could never do at all, since its curvature prediction came out of the very reading
  it was scoring.
- **Association is by range rank**, and that is a stated limit rather than a solved
  problem: detections carry no identity, so the particle's ``i``-th visible slot is scored
  against the ``i``-th detection. Two opponents at nearly equal range can swap order
  between the particle and the reading, and the residuals are then taken against the wrong
  pair. A joint-probabilistic association would fix it at ``K!`` cost.
- **The detection rates ship at zero, because this world's detection decision is
  deterministic.** The range gate and the occlusion rule run on the vehicles' true
  positions and the radar drops nothing it can see and invents nothing, so
  ``presence_miss_prob = 0.05`` and ``presence_false_alarm_prob = 0.02`` — carried over
  from the occupancy-grid arm — described a sensor that does not exist and that no
  measurement of this world could have fitted. Both are now ``0.0``, in the schema, the
  scalar model and the torch model alike. The consequence is that **prune-by-silence is
  sharp**: a particle whose visibility prediction contradicts the reading is excluded
  rather than discounted, which is what Bayes says to do with a hypothesis the data rules
  out. An unreported car inside the gate cost 2.98 nats and now costs 27.63; the occluded
  car behind a closer one still costs nothing, because the model predicts the occlusion.
- That 27.63 is a **numerical floor, not a miss rate**. Scoring clips a probability to
  ``PROBABILITY_EPS = 1e-12`` on both sides, because an all-zero weight vector crashes a
  finite particle set's normalisation rather than telling it anything. Measured on 200
  particles jittered 0.5 m around an opponent sitting on the 40 m boundary, where 15% of
  them straddle the gate: mean ESS over 20 drawn readings is **143.8** at the shipped
  rates against **128.9** at the old ones — the sharp rates leave the filter *healthier*,
  because the reading is no longer randomly corrupted by a sensor lie the world does not
  tell. Over a live 13-step episode no particle contradicted visibility at all, the belief
  stamping its slots from the same detections it is scored against.
- The rates remain parameters, and a lossy radar remains a legitimate thing to configure:
  set either above zero and both arms model one, in the sampler as well as in the density.
  The Cauchy clutter machinery is part of that configuration and is kept for it. At the
  shipped rate of zero no sampler draws a phantom and the clutter density is only reached
  by a detection the particle is already excluded by; with a rate configured it is what
  keeps the two branches comparable, for the same reason it was never optional on the MDP
  arm — a bare false-alarm rate is a probability and a matched detection is a probability
  *density*, and comparing the two inverts the likelihood.
- Nothing in the likelihood can be ``-inf``. A particle predicting a detection the reading
  does not show pays ``log(presence_miss_prob)``; a detection no slot explains costs
  ``log(presence_false_alarm_prob)`` **plus the clutter density of what was reported**;
  both are floored at the epsilon above.
- Both the world and the planner's model run **one** visibility rule, from
  ``racetrack_schema.detection_visibility``. That is what lets a particle placing an
  opponent behind a closer one keep its weight when the reading does not show it — the
  model predicts the occlusion rather than being punished for it. It is also where the
  range dial acts on the *inference* rather than on the reading: a particle that puts a
  car inside its own gate is ruled out when no row arrives, and one that puts a car
  outside pays nothing. That asymmetry is the mechanism by which an empty
  reading rules a hypothesis out, and it is exactly what shrinking R removes.
- The belief follows: ``TrackedAgentsBelief`` seeds its slots straight from the
  detections. Both velocity components are copied across as reported, with no
  line-of-sight projection to undo — a car crossing the ego's path abeam at 6 m/s is now
  stamped at 6 m/s where it used to be stamped at zero. ``agent_pose_jitter`` and
  ``agent_velocity_jitter`` therefore cover measurement noise and the constant-velocity
  drift's model error, and nothing structurally unmeasured; their defaults (0.5 m, 1.0
  m/s) are unchanged, which leaves the velocity jitter deliberately wider than the
  sensor's own 0.3 m/s. A slot with no detection behind it is still left empty rather
  than jittered, because there is no reading to spread.
- **Breaking, and deleted rather than deprecated.** ``ObservedTrackModel`` no longer fits
  anything: ``curvature_window_m``, the quadratic corridor fit, the own-lane trace and
  ``lane_offsets`` are gone, and ``curvature_estimate`` is now the nearest curvature
  sample the camera reported. That fit was biased and the bias depended on how well the
  car was being driven — 1.06x true curvature under 0.15 rad of lane-relative yaw, 0.84x
  between 0.35 and 0.50 — so a weak estimate under-steered, under-steering yawed the car,
  and a yawed car read the road worse still. ``KnownTrackModel`` loses
  ``_render_on_road_layer`` and its centreline tables and gains ``curvature_ahead``.
  ``RacetrackModelPOMDP`` loses ``agent_position_std_m`` and ``cell_mismatch_prob`` and
  gains nine sensor parameters, which **must match the world's**. The occupancy
  tracker module and the belief's frame buffer are deleted outright; the torch model
  loses its ``road_layer=`` argument and every ``RoadLayer`` implementation.
  ``TrackGeometry`` is untouched, though its ``centreline_pose`` and lane-offset API now
  has no shipped consumer.
- Also breaking, on both the world and the model: ``detection_radial_velocity_std`` is
  renamed ``detection_velocity_std`` (same 0.3 default, now applied to both components),
  ``DEFAULT_DETECTION_RADIAL_VELOCITY_STD`` follows it, and the schema's
  ``DETECTION_RADIAL_V`` becomes ``DETECTION_REL_VX`` / ``DETECTION_REL_VY`` with
  ``DETECTION_SLOT_WIDTH`` going 4 to 5. The flat torch offsets shift by the ego-pose
  channel's width; read them from the schema rather than assuming them.
- The world-side sensing moved into a new ``racetrack_world_sensors`` module —
  ``SensorConfig``, ``WorldSensors`` and ``relative_vehicles`` — because
  ``racetrack_pomdp`` crossed the 1000-line cap. It is the counterpart of
  ``racetrack_sensor_model``: that module says what a reading is *worth* to a particle,
  this one produces the reading. Like the rest of the package it imports nothing from
  highway-env and duck-types the vehicle objects, so it can be exercised without booting
  the simulator.
- **The matched-pair guarantee is re-expressed, not weakened.** The POMDP arm's reading is
  no longer something highway-env produces — the simulator supplies only the ego's own
  kinematics and the rest is measured world-side — so "the two configs differ on exactly
  the ``observation`` key" no longer covers what it was protecting. The test now asserts
  that the two arms' configurations are **byte-identical once the observation block is
  removed**, which is the same guarantee stated against the thing it actually guards: one
  dynamics path, one reward, one step rate, two readings.
- ``highway-env`` is a development dependency only. It is lazily imported, so a
  runtime install never loads it and nothing outside the racetrack package depends
  on it, but it is in the ``dev`` extra rather than absent entirely: the environment
  was chosen over the alternatives precisely because it needs no binary, no server
  and no GPU and therefore runs in CI, and tests that exercise a stand-in instead of
  the real simulator would give that up.
- Migrated eight environments — Tiger, CartPole, MountainCar, RockSample, both
  LaserTag variants and both Push variants — onto ``Environment.step_info``
  and ``get_metric_specs``, replacing eight hand-rolled ``compute_metrics``
  bodies and their inconsistent confidence-interval handling with the shared
  aggregator. Metrics that cannot be expressed as a per-episode reduction of a
  per-step channel are kept as-is: the LaserTag metrics defined in terms of a
  realised reward, and the Push ``*_rate`` metrics, which are pooled across
  episodes.
- PacMan and both light-dark environments are deliberately left on their own
  ``compute_metrics``. PacMan's rule that an episode ending in a malformed
  state reports zeros is an episode-level decision a stateless per-step channel
  cannot express, and reproducing it through the shared aggregator required
  rewriting each episode's measurements before aggregating — more code than the
  loop it replaced, in service of preserving a quirk. The light-dark metrics
  likewise come out of a single loop that stops at the goal.
- A resumed run no longer has to throw its cache away. The simulation caches
  are keyed on a task's configuration, which says nothing about the format of
  the episode it produced, so an entry written before the per-step channel
  existed still hits and unpickles cleanly with every measurement missing. Both
  layers now treat such an entry as a cache *miss* — ``run_tasks``, which
  consults the cache database before a task reaches the executor, and
  ``JoblibTaskManager``, for entries held only in the joblib store. Each logs a
  warning naming the task, reruns that one episode and replaces the entry.
  Everything recorded in the current format is still reused, so recovery
  survives the upgrade instead of costing a full re-run.
- ``Environment.step_info`` is now also called once per terminated episode,
  for the terminal bookkeeping step, with ``action`` and ``next_state`` both
  ``None``. Metrics that count every visited state need that final state, and
  it is recorded on no other step.
- Added ``order_and_fill_metrics`` to
  :mod:`POMDPPlanners.core.simulation.step_info_metrics`, so an environment
  with a fixed declared metric list always produces every declared name in
  declaration order, even when the aggregator has nothing to report for one.

Others:
^^^^^^^

- Extended the frozen metric baseline with a terminated-episode shape, a
  metric-ordering snapshot and a confidence-bounds snapshot, which is what
  makes the migration's "no name, value or ordering moved" claim checkable
  rather than assumed. The bounds matter on their own: a declared reduction
  that yields the right mean from the wrong per-episode samples leaves the
  point estimate intact and moves only the interval.
- Three deliberate behaviour changes came out of the migration, all confined to
  confidence bounds and degenerate inputs. No point estimate moved on any
  history an episode runner can produce.
  Tiger reported ``lower == upper == value`` — a placeholder its own comment
  called out — and now gets a real 95% t-interval. Every environment now rejects
  an empty history list rather than answering it, replacing three different
  answers with one error. Finally, an episode that reports a metric's channel on no step at all is now
  dropped from that metric's average rather than contributing a declared
  stand-in value. What an unmeasured episode would have measured is not a
  property of the metric, so there is no per-spec knob for it; an episode that
  ran but was never measured is rejected upstream instead of scored as zero.
  ``EpisodeRunner`` always records a measured step, including the terminal
  bookkeeping one, so this is reachable only from hand-built histories — where
  a mixed list of real and stepless episodes now reports the mean over the real
  ones alone.


Release 0.5.0 (WIP)
-------------------

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

- Bundled the C++ headers in the sdist. Without them ``pip install`` from
  PyPI failed to compile, because ``setup.py``'s ``include_dirs`` pointed at
  paths absent from the tarball. Added packaging regression tests
  (:gh:`168`).

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
