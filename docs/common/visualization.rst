Visualization and Reporting
===========================

There are three ways to look at results, and each answers a different
question. An episode replay shows what one planner did in one episode, which
is how you catch a planner that scores well for the wrong reason. The results
site shows a whole run in the browser: metric tables, return histograms, and
every episode's replay. The tuning plots show whether a hyperparameter study
converged and what it traded away.

Episode replays
---------------

Each environment shows its episodes in exactly one way, returned by
``Environment.episode_visualizer()``:

- Environments written in this package return a ``TraceVisualizer``. It saves
  each episode as ``trace_<i>.json``, and a three.js scene replays the trace
  in the browser.
- Environments that wrap an external simulator (CARLA, Isaac Lab) return a
  ``VideoVisualizer``, which saves the simulator's own camera footage as
  ``agent_path_<i>.mp4``.

A simulation run writes these files for you. To produce replays without a
full run, use ``visualize_planner_episode``:

.. code-block:: python

   import tempfile
   from pathlib import Path

   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.core.simulation import EpisodeTrace
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.utils.planner_episode_visualization import visualize_planner_episode

   env = TigerPOMDP(discount_factor=0.95)
   planner = POMCP(environment=env, discount_factor=0.95, depth=5,
                   exploration_constant=1.0, name="POMCP", time_out_in_seconds=2.0)

   out = Path(tempfile.mkdtemp())
   visualize_planner_episode(planner, env, get_initial_belief(env, n_particles=100),
                             n_episodes=2, cache_dir=out, num_steps=10)

   # A trace is a JSON file, so it can be read back and checked.
   trace = EpisodeTrace.read(out / "POMCP" / "trace_0.json")
   print(trace.payload_kind, trace.num_steps, trace.discounted_return)

A trace has two parts. The envelope (actions, rewards, ``step_info`` values)
has the same shape for every environment. The payload holds the states,
observations and beliefs in whatever shape the environment's scene expects,
and ``payload_kind`` (for example ``"tiger.v1"``) names that shape.

.. autofunction:: POMDPPlanners.utils.planner_episode_visualization.visualize_planner_episode

.. autoclass:: POMDPPlanners.core.simulation.episode_visualizers.EpisodeVisualizer
   :members: write

.. autoclass:: POMDPPlanners.core.simulation.episode_visualizers.TraceVisualizer
   :members: build_payload, build_trace, metadata, reached_terminal_state, scene_script

.. autoclass:: POMDPPlanners.core.simulation.episode_visualizers.VideoVisualizer
   :members: write_video

.. autoclass:: POMDPPlanners.core.simulation.traces.EpisodeTrace
   :members: read, write, num_steps, total_reward, discounted_return

.. autoclass:: POMDPPlanners.core.simulation.traces.TraceStep

.. autoclass:: POMDPPlanners.core.simulation.traces.ArtifactKind
   :members:
   :undoc-members:

The results site
----------------

``pomdp-report serve`` starts a local website over one or more run
directories. It reads the MLflow store that every run writes under
``cache_dir_path/mlruns``; nothing needs exporting first. It is a server, not
a folder of HTML files, because a browser will not let a page opened from
disk load the trace files next to it.

.. code-block:: bash

   pomdp-report serve results/my-study
   # Serving on http://127.0.0.1:8765/

.. autofunction:: POMDPPlanners.reporting.server.serve

Tuning diagnostics
------------------

A tuning study writes ``trial_records.json`` and a set of PNG plots for each
tuned planner, by default under ``<cache_dir>/tuning_diagnostics/<config_id>/``
and also as MLflow artifacts under ``tuning/``. The plots are built from
:class:`~POMDPPlanners.utils.visualization.tuning_plots.TrialRecord` objects
rather than from the Optuna study, because the study holds episode histories
that no Optuna storage can save. So you can redraw every plot later from the
JSON file without rerunning anything:

.. code-block:: python

   import tempfile
   from pathlib import Path

   from POMDPPlanners.utils.visualization import (
       TrialRecord,
       load_trial_records,
       plot_tuning_diagnostics,
       save_trial_records,
   )

   # Normally a study writes this file; here three trials are made up by hand.
   records = [
       TrialRecord(number=i, state="COMPLETE",
                   params={"exploration_constant": c},
                   objective_values={"average_return": r},
                   metric_statistics={"average_return": (r, r - 2.0, r + 2.0)},
                   duration_seconds=10.0)
       for i, (c, r) in enumerate([(1.0, -20.0), (10.0, -5.0), (50.0, 3.0)])
   ]
   study_dir = Path(tempfile.mkdtemp())
   save_trial_records(records, study_dir / "trial_records.json")

   # Later, with no study in memory, redraw the plots from the file.
   records = load_trial_records(study_dir / "trial_records.json")
   written = plot_tuning_diagnostics(
       records,
       parameters_to_optimize=[("average_return", "maximize")],
       output_dir=study_dir / "plots",
   )
   print([p.name for p in written])  # only the plots that had data behind them

The front-quality curve answers "did the search converge, or did it run out
of budget". The secondary-metrics plot shows what the study gave up on
metrics it did not optimize, such as collision rate. See
:doc:`../examples/hyperparameter_tuning` for a full study.

.. autoclass:: POMDPPlanners.utils.visualization.tuning_plots.TrialRecord
   :members: to_dict, from_dict

.. autofunction:: POMDPPlanners.utils.visualization.tuning_plots.extract_trial_records

.. autofunction:: POMDPPlanners.utils.visualization.tuning_plots.save_trial_records

.. autofunction:: POMDPPlanners.utils.visualization.tuning_plots.load_trial_records

.. autofunction:: POMDPPlanners.utils.visualization.tuning_plots.plot_tuning_diagnostics

.. autofunction:: POMDPPlanners.utils.visualization.tuning_plots.plot_parameter_history

.. autofunction:: POMDPPlanners.utils.visualization.tuning_plots.plot_parameter_slices

.. autofunction:: POMDPPlanners.utils.visualization.tuning_plots.plot_parameter_importances

Result plots
------------

These draw static matplotlib figures from a run's statistics or histories,
for a paper or a report.

.. autofunction:: POMDPPlanners.utils.visualization.metrics_plots.plot_metrics_comparison

.. autofunction:: POMDPPlanners.utils.visualization.metrics_plots.plot_policies_comparison_on_environment

.. autofunction:: POMDPPlanners.utils.visualization.returns_plots.plot_discounted_returns_histogram

.. autofunction:: POMDPPlanners.utils.visualization.returns_plots.plot_discounted_returns_histogram_multiple_policies

.. autofunction:: POMDPPlanners.utils.visualization.returns_plots.plot_environment_policy_pair_comparison

.. autofunction:: POMDPPlanners.utils.visualization.tree_plots.plot_tree_graphs

.. autoclass:: POMDPPlanners.core.simulation.visualizers.ExperimentVisualizer
   :members: render
