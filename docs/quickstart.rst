Quickstart Guide
================

This guide plans one action, runs one episode, then compares and tunes
planners.

Your First POMDP Solution
--------------------------

The following example solves the Tiger POMDP with POMCP: the planner
picks an action from the current belief, and the example then executes that
action in the environment.

.. code-block:: python

   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   from POMDPPlanners.core.belief import get_initial_belief

   # Create the environment and initial belief
   env = TigerPOMDP(discount_factor=0.95)
   belief = get_initial_belief(env, n_particles=500)

   # Create the planner
   planner = POMCP(
       environment=env,
       discount_factor=0.95,
       depth=10,
       exploration_constant=50.0,
       name="tiger_planner",
       time_out_in_seconds=2.0,
   )

   # Plan: returns a list of actions (length=1 for closed-loop planning)
   actions, run_data = planner.action(belief)
   action = actions[0]
   print(f"Recommended action: {action}")

   # Execute: sample the next state, observation, and reward
   state = belief.sample()
   next_state, observation, reward = env.sample_next_step(state=state, action=action)
   print(f"Observation: {observation}, Reward: {reward}")


Running a Complete Episode
---------------------------

Use ``run_episode`` to run a full episode. It updates the belief with each
action and observation:

.. code-block:: python

   from POMDPPlanners.simulations.episodes import run_episode
   from POMDPPlanners.utils.logger import get_logger

   logger = get_logger("quickstart")

   history = run_episode(
       environment=env,
       policy=planner,
       initial_belief=belief,
       num_steps=20,
       logger=logger,
   )

   total_reward = sum(step.reward for step in history.history if step.reward is not None)
   print(f"Steps: {len(history.history)}, Total reward: {total_reward:.2f}")

   # Each step exposes: action, observation, reward, state
   for i, step in enumerate(history.history[:5]):
       print(f"Step {i}: action={step.action}, obs={step.observation}, reward={step.reward}")


Core Concepts
-------------

**Environments**

.. code-block:: python

   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   env = TigerPOMDP(discount_factor=0.95)

   # Discrete environments expose their state/action/observation spaces
   print(env.states)       # ['tiger_left', 'tiger_right']
   print(env.actions)      # ['listen', 'open_left', 'open_right']
   print(env.observations) # ['hear_left', 'hear_right', 'hear_nothing']

   # Core interaction method
   next_state, observation, reward = env.sample_next_step(state=state, action=action)
   done = env.is_terminal(next_state)

**Belief States**

.. code-block:: python

   from POMDPPlanners.core.belief import get_initial_belief

   belief = get_initial_belief(env, n_particles=500)

   # Sample a single state from the belief
   state = belief.sample()

   # Inspect the weighted distribution
   distribution = belief.to_unique_support_distribution()

**Planners**

All planners share the same interface: ``planner.action(belief)`` returns
``(List[action], PolicyRunData)``. A single-element list means closed-loop
(replans each step); a multi-element list means open-loop (executes the
sequence before replanning).

.. code-block:: python

   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP

   planner = POMCP(
       environment=env,
       discount_factor=0.95,
       depth=10,
       exploration_constant=50.0,
       name="my_planner",
       time_out_in_seconds=2.0,
   )

   actions, run_data = planner.action(belief)
   action = actions[0]  # closed-loop: take the single planned action


Continuous Action Spaces
-------------------------

For environments with continuous actions, pair ``PFT_DPW`` with an action sampler:

.. code-block:: python

   import numpy as np
   from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
       ContinuousLightDarkPOMDP,
   )
   from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
   from POMDPPlanners.planners.planners_utils.dpw import ActionSampler
   from POMDPPlanners.core.belief import get_initial_belief

   env = ContinuousLightDarkPOMDP(
       discount_factor=0.95,
       goal_state=np.array([10, 5]),
       start_state=np.array([0, 5]),
   )

   class VelocityActionSampler(ActionSampler):
       def sample(self, belief_node=None):
           angle = np.random.uniform(0, 2 * np.pi)
           speed = np.random.uniform(0, 1.0)
           return np.array([speed * np.cos(angle), speed * np.sin(angle)])

   planner = PFT_DPW(
       environment=env,
       discount_factor=0.95,
       depth=10,
       name="navigation_planner",
       action_sampler=VelocityActionSampler(),
       time_out_in_seconds=2.0,
   )

   belief = get_initial_belief(env, n_particles=500)
   actions, _ = planner.action(belief)
   print(f"Navigation action: {actions[0]}")


Comparing Planners
------------------

``LocalSimulationsAPI`` is the recommended entry point for end-to-end
experiments. It runs episodes in parallel, caches every episode to disk, and
returns aggregated statistics: mean return, CVaR, VaR and confidence
intervals.

The example below compares POMCPOW and PFT-DPW on Light-Dark: a robot must
reach a goal, but its position readings are only precise near a light source,
so it has to detour toward the light before it can find the goal. Each planner
gets 2 seconds per decision, and the episodes (30 per planner) run in parallel
on every CPU core. Save it as ``compare_planners.py`` and run
``python compare_planners.py``.

.. code-block:: python

   from pathlib import Path

   from POMDPPlanners.environments import ContinuousLightDarkPOMDPDiscreteActions
   from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
   from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
   from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
   from POMDPPlanners.utils.belief_factory import create_environment_belief
   from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
   from POMDPPlanners.core.simulation import EnvironmentRunParams

   env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)
   sampler = DiscreteActionSampler(env.get_actions())

   pomcpow = POMCPOW(environment=env, discount_factor=0.95, depth=10,
                     exploration_constant=200.0, k_o=1.0, k_a=1.0,
                     alpha_o=0.5, alpha_a=0.5, time_out_in_seconds=2,
                     action_sampler=sampler, name="POMCPOW")
   pft_dpw = PFT_DPW(environment=env, discount_factor=0.95, depth=10,
                     exploration_constant=200.0, time_out_in_seconds=2,
                     action_sampler=sampler, name="PFT_DPW")
   belief = create_environment_belief(env, n_particles=200)

   api = LocalSimulationsAPI()
   _, stats = api.run_multiple_environments_and_policies(
       environment_run_params=[
           EnvironmentRunParams(
               environment=env,
               belief=belief,
               policies=[pomcpow, pft_dpw],
               num_episodes=30,
               num_steps=30,
           )
       ],
       alpha=0.1,
       confidence_interval_level=0.95,
       experiment_name="LightDark_Evaluation",
       n_jobs=-1,  # run episodes in parallel, one per CPU core
       cache_dir_path=Path("results"),
   )
   print(stats[["policy", "average_return", "task_completion_rate"]])

The run writes its results under ``results/``. Open them in the local results
site:

.. code-block:: bash

   pomdp-report serve results --port 8765

Then browse to http://127.0.0.1:8765, using the port you passed to ``--port``.
Open the ``LightDark_Evaluation`` experiment to compare the two planners side
by side on expected return and task completion rate, and replay any episode in
3D.

The video below walks through the whole example: copying the code, running it,
and comparing the planners in the results site. It also shows the List and
Table views, editing a chart's title and axes, exporting the chart as an
image, and playing every episode at once with Live.

.. raw:: html

   <video controls preload="metadata" style="width:100%;border-radius:8px;" aria-label="Walkthrough of the planner comparison example">
     <source src="_static/videos/compare_planners_walkthrough.mp4" type="video/mp4">
     <a href="_static/videos/compare_planners_walkthrough.mp4">Watch the video</a>.
   </video>

Long-running experiments can report progress to Slack and to a local progress
database, including detection of crashed or stalled runs. Set
``SLACK_WEBHOOK_URL`` in your environment and ``NotificationConfig`` reads it
and sends notifications there. See
:class:`~POMDPPlanners.simulations.simulations_deployment.run_progress.config.NotificationConfig`
for details.


Hyperparameter Tuning
---------------------

``LocalSimulationsAPI.run_optimize_and_evaluate`` tunes each planner with an
Optuna study, then runs each tuned planner for ``evaluation_episodes``
episodes, separate from the episodes that scored the trials.

:doc:`examples/hyperparameter_tuning` explains each part of the study and how
long it takes to run.

The example below tunes POMCPOW and PFT-DPW on two environments, RockSample
and Push, and compares the tuned planners on each. Everything the study uses
is written out so you can edit it in place: the search range of each
parameter, the planner settings that stay fixed, and the objectives each
trial is scored on (average return and task completion rate). Early stopping
ends a study once ``patience`` trials in a row fail to improve its best
results. Each planner's name carries its environment, because the evaluation
requires planner names to be unique across environments. Save it as ``tune_planners.py`` and run
``python tune_planners.py``.

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
   EVAL_EPISODES = 30       # evaluation episodes per tuned planner, run after tuning
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

These constants allow up to 1,000 tuning episodes per planner (50 × 20), then
30 evaluation episodes for each tuned planner. Early stopping ends a planner's
study once 20 trials in a row fail to improve its best results, but never
before 20 trials have run, so ``N_TRIALS`` is an upper bound.

In ``pomdp-report serve results``, the ``Tuning_RockSample_Push`` experiment
shows the whole study as one card. Its page has a "Compare Tuned planners"
block for each environment, listing every evaluation metric with its
confidence interval, and filters for metrics, planners and environments. Each
tuned planner opens a tuning view with its trials, diagnostic charts and
evaluation episodes, and both pages can build a figure and download it as SVG,
PNG or CSV.

The video below walks through the whole example: copying the code, running the
full study, and reading the results on PushPOMDP in the results site. It shows
the planner comparison and its filters, a tuning view with its trials and
diagnostic charts, and the evaluation episodes played Live.

.. raw:: html

   <video controls preload="metadata" style="width:100%;border-radius:8px;" aria-label="Walkthrough of the parameter tuning example">
     <source src="_static/videos/parameter_tuning_walkthrough.mp4" type="video/mp4">
     <a href="_static/videos/parameter_tuning_walkthrough.mp4">Watch the video</a>.
   </video>


Available Environments
-----------------------

Each environment is a class you construct directly, then build an initial
belief for. :doc:`environments/base` lists them all.

.. code-block:: python

   from POMDPPlanners.core.belief import get_initial_belief
   from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
   from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
       ContinuousLightDarkPOMDP,
       ContinuousLightDarkPOMDPDiscreteActions,
   )
   from POMDPPlanners.environments.push_pomdp import PushPOMDP

   # Classic
   env = TigerPOMDP(discount_factor=0.95)

   # Navigation (discrete actions, continuous observations)
   env = ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)

   # Navigation (fully continuous)
   env = ContinuousLightDarkPOMDP(discount_factor=0.95)

   # Manipulation
   env = PushPOMDP(discount_factor=0.95)

   belief = get_initial_belief(env, n_particles=500)


Available Planners
-------------------

.. code-block:: python

   # POMCP — discrete actions and observations
   from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
   planner = POMCP(environment=env, discount_factor=0.95, depth=10,
                   exploration_constant=50.0, name="pomcp", time_out_in_seconds=2.0)

   # POMCPOW — continuous actions/observations via double progressive widening
   from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
   planner = POMCPOW(environment=env, discount_factor=0.95, depth=10,
                     exploration_constant=100.0,
                     k_o=10, k_a=4, alpha_o=0.01, alpha_a=0.01,
                     action_sampler=action_sampler, time_out_in_seconds=2.0, name="pomcpow")

   # PFT-DPW — particle filter trees with double progressive widening
   from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
   planner = PFT_DPW(environment=env, discount_factor=0.95, depth=10,
                     k_a=4, alpha_a=0.01, k_o=10, alpha_o=0.01,
                     exploration_constant=100.0, action_sampler=action_sampler,
                     time_out_in_seconds=2.0, name="pft_dpw")

   # Sparse Sampling — simple model-based baseline (depth=2, branching_factor=10)
   from POMDPPlanners.planners.sparse_sampling_planners.sparse_sampling import SparseSamplingDiscreteActionsPlanner
   planner = SparseSamplingDiscreteActionsPlanner(env, branching_factor=10, depth=2)


Next Steps
----------

**Run the example notebooks**

.. code-block:: bash

   jupyter notebook docs/examples/basic_usage.ipynb
   jupyter notebook docs/examples/planners_comparison.ipynb
   jupyter notebook docs/examples/hyperparameter_tuning.ipynb
   jupyter notebook docs/examples/advanced_optimization.ipynb

**API Reference**

Each planner and environment page ends with its parameters. The shared APIs are
under *Common* in the sidebar: :doc:`common/beliefs`,
:doc:`common/simulation_api` and the rest.

HyP-DESPOT is available only for environments that expose a deterministic
``hyp_despot_cuda_model`` on ``cuda:0``. It never falls back to CPU DESPOT.
