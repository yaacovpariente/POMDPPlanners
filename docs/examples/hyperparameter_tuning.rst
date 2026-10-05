Hyperparameter Tuning
=====================

A planner's return depends on settings such as its exploration constant and
its widening coefficients, and the values that work on one environment do
not carry over to another. ``LocalSimulationsAPI.run_optimize_and_evaluate``
searches those settings with an Optuna study for each planner and
environment, then runs each tuned planner on new episodes, so you can compare
planners at their tuned settings rather than at hand-picked ones.

The example
-----------

The example tunes POMCPOW and PFT-DPW on RockSample and on Push, four studies
in all, and compares the tuned planners on each environment. Save it as
``tune_planners.py`` and run ``python tune_planners.py``.

.. code-block:: python

   from pathlib import Path

   from POMDPPlanners.environments import PushPOMDP, RockSamplePOMDP
   from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
   from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
   from POMDPPlanners.core.simulation.hyperparameter_tuning import (
       EarlyStoppingConfig,
       HyperParameterOptimizationDirection,
       HyperParameterRunParams,
       HyperParamPlannerConfig,
       NumericalHyperParameter,
   )
   from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
   from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
   from POMDPPlanners.utils.belief_factory import create_environment_belief

   N_TRIALS = 50            # at most this many Optuna trials per planner
   EPISODES_PER_TRIAL = 20  # episodes that score one trial
   EVAL_EPISODES = 30       # fresh episodes for each tuned planner
   NUM_STEPS = 30           # step limit per episode
   DISCOUNT_FACTOR = 0.95
   DEPTH = 10               # search depth, fixed rather than tuned

   # What each trial is scored on, and in which direction.
   OBJECTIVES = [
       ("average_return", HyperParameterOptimizationDirection.MAXIMIZE),
       ("task_completion_rate", HyperParameterOptimizationDirection.MAXIMIZE),
   ]


   def search_space(env):
       max_exploration = (env.reward_range[1] - env.reward_range[0]) * DEPTH
       return [
           NumericalHyperParameter(0.0, max_exploration, "exploration_constant"),  # UCB exploration
           NumericalHyperParameter(1, 10, "k_a"),  # action widening coefficient
           NumericalHyperParameter(0.01, 0.5, "alpha_a"),  # action widening exponent
           NumericalHyperParameter(1, 10, "k_o"),  # observation widening coefficient
           NumericalHyperParameter(0.01, 0.5, "alpha_o"),  # observation widening exponent
       ]


   studies = []
   for env in (
       RockSamplePOMDP(discount_factor=DISCOUNT_FACTOR),
       PushPOMDP(discount_factor=DISCOUNT_FACTOR),
   ):
       belief = create_environment_belief(env, n_particles=200)
       sampler = DiscreteActionSampler(env.get_actions())
       for policy_cls, name in ((POMCPOW, "POMCPOW"), (PFT_DPW, "PFT_DPW")):
           planner = HyperParamPlannerConfig(
               policy_cls=policy_cls,
               hyper_parameters=search_space(env),
               constant_parameters={  # settings that stay fixed in every trial
                   "discount_factor": DISCOUNT_FACTOR,
                   "depth": DEPTH,
                   "name": f"{name}_{env.name}",
                   "environment": env,
                   "action_sampler": sampler,
                   "time_out_in_seconds": 1,
               },
           )
           studies.append(
               HyperParameterRunParams(
                   environment=env,
                   belief=belief,
                   hyper_param_planner_config=planner,
                   num_episodes=EPISODES_PER_TRIAL,
                   num_steps=NUM_STEPS,
                   n_trials=N_TRIALS,
                   parameters_to_optimize=OBJECTIVES,
                   early_stopping=EarlyStoppingConfig(patience=20, min_trials=20),
               )
           )

   api = LocalSimulationsAPI()
   _, stats = api.run_optimize_and_evaluate(
       configs=studies,
       evaluation_episodes=EVAL_EPISODES,
       evaluation_steps=NUM_STEPS,
       optimization_n_jobs=-1,  # use every CPU core
       evaluation_n_jobs=-1,
       experiment_name="Tuning_RockSample_Push",
       cache_dir_path=Path("results"),
   )
   print(stats[["environment", "policy", "average_return", "task_completion_rate"]])

Before you run it
-----------------

At these constants the run takes hours. Each study allows up to 1,000
tuning episodes (``N_TRIALS`` × ``EPISODES_PER_TRIAL``) of up to 30 steps, at
1 second of planning per step, and there are four studies. That is up to
120,000 seconds of planning, split across your CPU cores, before the
evaluation episodes. Early stopping usually ends a study sooner, but plan for
the full budget.

For a first run that finishes in minutes, lower the constants:

.. code-block:: python

   N_TRIALS = 4
   EPISODES_PER_TRIAL = 3
   EVAL_EPISODES = 5
   NUM_STEPS = 10

and lower ``min_trials`` and ``patience`` in ``EarlyStoppingConfig`` to 2, so
early stopping does not ask for more trials than ``N_TRIALS``. The numbers
from such a run are too noisy to choose a planner by; use it to check that
the script runs and to see the results site.

What each part does
-------------------

**The search space.** ``search_space`` returns one
:class:`~POMDPPlanners.core.simulation.hyperparameter_tuning.NumericalHyperParameter`
per tuned setting, as ``(low, high, name)``. Each trial draws one value per
setting from its range. The upper end of ``exploration_constant`` is the
environment's reward range times ``DEPTH``, because the exploration bonus has
to be on the scale of the returns it competes with. For a setting with a few
fixed options, use
:class:`~POMDPPlanners.core.simulation.hyperparameter_tuning.CategoricalHyperParameter`
with a list of choices.

**The fixed settings.** ``constant_parameters`` holds every constructor
argument that is not tuned, and each trial passes them unchanged. Two of them
matter for a fair study:

- ``time_out_in_seconds`` gives every trial the same planning time per step.
  If the budget were tuned, the study would favour the settings that plan
  longest.
- ``name`` includes the environment, because the evaluation requires planner
  names to be unique across environments.

``depth`` is fixed rather than tuned for the first reason: a deeper search
costs more per simulation, so under a fixed time budget the study would trade
depth against simulation count instead of measuring either.

**The objectives.** ``OBJECTIVES`` lists the metrics each trial is scored on
and whether to maximize or minimize each. A trial's score for a metric is its
mean over ``EPISODES_PER_TRIAL`` episodes. With two objectives no single
trial is best on both, so the study keeps the Pareto front: the trials that
no other trial beats on both metrics. The tuned planner is the trial on the
front with the highest average z-score across the objectives, where each
metric is standardized by its mean and standard deviation over the front.

**Early stopping.**
:class:`~POMDPPlanners.core.simulation.hyperparameter_tuning.EarlyStoppingConfig`
ends a study once ``patience`` trials in a row fail to improve the Pareto
front, but never before ``min_trials`` trials have finished, so
``N_TRIALS`` is an upper bound. Set ``patience`` high: each trial's score is
a mean over a few episodes, so a run of trials with no improvement is often
noise rather than a study that has converged.

**The evaluation.** After tuning, each tuned planner runs
``EVAL_EPISODES`` episodes that were not used to score any trial. The
returned ``stats`` table, one row per environment and planner, comes from
these episodes, with a confidence interval for each metric. Scoring on the
tuning episodes instead would overstate the tuned planner, because the study
picked the trial whose episodes happened to score highest.

Reading the results
-------------------

The run writes everything under ``results/``. Open it in the results site:

.. code-block:: bash

   pomdp-report serve results --port 8765

The ``Tuning_RockSample_Push`` experiment shows the whole study as one card.
Its page has a "Compare Tuned planners" block for each environment, listing
every evaluation metric with its confidence interval. Each tuned planner
opens a tuning view with its trials, diagnostic charts and evaluation
episodes.

The video below walks through the whole example: copying the code, running the
full study, and reading the results on PushPOMDP in the results site. It shows
the planner comparison and its filters, a tuning view with its trials and
diagnostic charts, and the evaluation episodes played Live.

.. raw:: html

   <video controls preload="metadata" style="width:100%;border-radius:8px;" aria-label="Walkthrough of the parameter tuning example">
     <source src="../_static/videos/parameter_tuning_walkthrough.mp4" type="video/mp4">
     <a href="../_static/videos/parameter_tuning_walkthrough.mp4">Watch the video</a>.
   </video>

If the best trials sit at the edge of a setting's range, the best value may
lie outside it: widen that range and run again. If the tuned planners'
confidence intervals overlap, the evaluation cannot tell them apart; raise
``EVAL_EPISODES``.

See also
--------

- :doc:`planners_comparison` — comparing planners without tuning.
- :doc:`../common/simulation_api` — the reference for
  ``run_optimize_and_evaluate`` and the tuning classes.
