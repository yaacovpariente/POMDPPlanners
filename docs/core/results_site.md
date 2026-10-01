# Browsing results

`pomdp-report` serves a local website over a simulation run's own output. It
replaces the MLflow UI for reading results; MLflow itself stays exactly where it
is, as the tracking store the site reads.

```bash
pomdp-report serve results/my-study
# Indexed 1 experiment(s) in 1 store(s).
# Serving on http://127.0.0.1:8765/  (Ctrl-C to stop)
```

You can point it at several directories at once, or at a parent directory that
holds many runs:

```bash
pomdp-report serve results/ workspaces/ --port 9000
```

It is a server rather than a folder of HTML files, for two reasons that are not
a matter of taste. A browser refuses to let a `file://` page fetch a sibling
JSON file, so an episode viewer opened straight from disk could never load its
episode. And a run directory can hold gigabytes of video; the server streams it
from where it already is, in chunks and honouring byte ranges, so a `<video>`
element can seek within an episode and nothing is copied into an export.

The site binds to loopback by default. These are local results, and binding
every interface would publish them.

## What a run directory looks like

A simulation writes everything under the `cache_dir_path` you pass to
`LocalSimulationsAPI`. The parts the site reads:

```
results/my-study/
  mlruns/                                    <- the MLflow file store
    <experiment_id>/
      meta.yaml                              <- the experiment's name
      <run_id>/
        meta.yaml, params/, metrics/, tags/  <- what MLflow recorded
        artifacts/
          statistics/comparison_results.json
          <environment>/policy_comparison_histogram.png
          <environment>/<planner>/plots/discounted_returns_histogram.png
          <environment>/<planner>/visualizations/trace_<i>.json      <- or agent_path_<i>.mp4
  logs/, env_policy/, joblib_cache/, cache.db  <- run plumbing, not read by the site
```

Two things about this layout are worth knowing before you move results around.

**Every run directory holds its own MLflow store.** `BaseSimulator` points
`mlflow.set_tracking_uri` at `<cache_dir>/mlruns`, so there is no single
tracking server to configure — the site searches the directories you give it for
any `mlruns` inside them.

**`artifact_uri` inside a run's `meta.yaml` is an absolute path from the machine
that wrote it.** Copy results off a cluster and those paths point at nothing.
The site ignores them and resolves artifacts from the run directory on disk, so
a copied or renamed results tree still works.

One MLflow run can cover several environments and several planners — the
simulator logs them as `env_0_name`, `env_0_policy_0_name` and so on. The site
follows that: a run page lists its environments, an environment lists its
planners, and a planner lists its episodes.

## Episode artifacts, and the player each one gets

The site never asks which environment produced a file. It classifies the file
and picks a player from the classification:

| Kind | Files | Player |
| --- | --- | --- |
| `trace` | `trace_<i>.json` | the interactive 3D viewer |
| `video` | `.mp4`, `.webm`, `.mov` | a `<video>` element |
| `plot` | `.png`, `.jpg`, `.svg` | an image |

That indirection is the point. CARLA and Isaac Lab have no compact state to
replay — the render *is* the episode — so they write an MP4 and no trace, and
they appear in the site as first-class episodes with a video player without a
line of environment-specific code. When an episode produced several artifacts,
the page plays the most useful one: a trace beats a video because it is
interactive.

GIFs are not episode artifacts. Runs made before the GIF renderers were removed
still hold `agent_path_<i>.gif` files; the site ignores them.

## How an environment shows its episodes

Each environment shows its episodes in exactly one way, through the visualizer
`Environment.episode_visualizer()` returns
(`POMDPPlanners.core.simulation.episode_visualizers`):

- **`TraceVisualizer`**, for environments implemented in this repo. It writes
  `trace_<i>.json`, and a three.js scene replays it. The scene script lives in
  the environment's own visualization package next to the visualizer, as
  `<scene>.scene.js` — `battleship.scene.js` draws `battleship.v1`. The site
  serves every scene at `/static/scenes/<scene>.js`.
- **`VideoVisualizer`**, for worlds that wrap an external simulator (CARLA,
  Isaac Lab). It saves the simulator's own camera footage as
  `agent_path_<i>.mp4`.
- **`None`**, the default, for an environment with no visualization.

The simulator calls the visualizer's `write` once per episode, under the rule
that a failure loses the file and not the run.

## Episode traces

A picture is the end of the line: nothing can read a position, a reward or a
belief back out of it. A trace is the episode written as data.

A trace file has two halves:

- an **envelope** every environment fills in identically — schema version,
  environment name, episode index, policy, discount factor, and one record per
  step with the action, the reward and the environment's `step_info` channels.
  Anything that only indexes episodes reads this half and works for an
  environment it has never heard of;
- a **payload** the environment owns, named by `payload_kind` so a reader can
  decide whether it knows how to draw it before it tries. Light-Dark's
  `light_dark.v1` payload carries the world's geometry, the states, the
  observations, and the belief at each step.

To add a trace to an environment, subclass `TraceVisualizer` in the
environment's `<env>_visualization/` package, set `payload_kind`, implement
`build_payload(history)`, add the scene script beside it, and return the
visualizer from `episode_visualizer()`. The base class fills in the envelope.

### The belief

The belief is the field these pages exist to show, and the one most easily
faked by accident: a cloud regenerated in JavaScript around the true position
looks convincing and says nothing. So the trace carries the belief the run
actually held.

Serializing it is **not** an environment's job. `Belief` is a core abstraction
with a closed family of implementations, so
`POMDPPlanners.core.simulation.belief_payloads.belief_to_payload` writes any of
them, dispatching on the class. An environment's visualizer hands it a `Belief`
and gets a payload back, and is left with only what is genuinely its own: its
world and its states.

Dispatch is by class, and the resulting `kind` is a drawing contract rather
than a class name, so a viewer draws particles once instead of once per class:

| Belief class | `kind` |
| --- | --- |
| `WeightedParticleBelief`, and `WeightedParticleBeliefReinvigoration` under it | `particles` |
| `VectorizedWeightedParticleBelief` | `particles` |
| `WeightedParticleBeliefStateUpdate` | `particles` |
| `UnweightedParticleBelief` | `particles`, `weighted: false` |
| `UnweightedParticleBeliefStateUpdate` | `particles`, `weighted: false` |
| `BatchedParticleBelief` | `particle_batch` |
| `GaussianBelief` | `gaussian` |
| `GaussianMixtureBelief` | `gaussian_mixture` |
| anything else | `unsupported`, naming the class |

Some details that are decisions rather than accidents:

- **Weights are always normalized.** `WeightedParticleBeliefStateUpdate` keeps
  raw accumulated observation likelihoods, which never sum to one; writing them
  unchanged would hand a reader a distribution that is not one.
- **An unweighted belief says so** with `weighted: false`, so a viewer does not
  shade it by weight and draw structure the belief does not have.
- **A mixture keeps its components.** Collapsing it to one mean and covariance
  would draw a genuinely bimodal belief as a wide cloud centred between its
  modes, exactly where the belief says nothing is.
- **A batch is written as a batch.** `BatchedParticleBelief` holds several
  independent beliefs in one tensor; flattening them would show a cloud that
  was never anyone's belief.
- **A large cloud is trimmed to its heaviest `MAX_PAYLOAD_PARTICLES` (400)**,
  which keeps the mass and drops the tail. The payload records both
  `num_particles` and `num_written`, so a viewer can say what it is showing.
- **Particles are written as they are.** A discrete state such as Tiger's
  `"tiger_left"` survives; deciding what a particle means belongs to the
  environment and to the scene module, not to core.
- **`unsupported` names the class**, so a belief nobody has written a payload
  for yet is a diagnosable gap rather than a silent blank.

## What the viewer shows

The viewer replays a real episode from a real run, and it has no fallback
episode of any kind. If it is given no trace, or a payload kind it has no scene
for, it says so and draws nothing.

This means these pages are not curated illustrations. A planner that drove
straight into a hazard on episode 3 will be shown driving straight into a
hazard on episode 3. That is the point of wiring the viewer to real runs, and
nothing in the interface implies otherwise.

## Checking that the viewer animates

A still render is not a test of playback. Two viewers in this project shipped
frozen while every screenshot looked perfect, because the screenshots were
produced by a harness that drove frames by hand and never exercised the
animation loop.

`scripts/check_viewer_playback.mjs` loads a real episode page from a running
server, watches the step readout advance on its own, and exercises Pause and
Play. It needs Node and a Chrome binary, so it is not part of the Python test
suite:

```bash
pomdp-report serve results/my-study --port 8765 &
node scripts/check_viewer_playback.mjs \
  "http://127.0.0.1:8765/run/0/<experiment_id>/<run_id>/env/<env>/policy/<planner>/episode/0"
```
