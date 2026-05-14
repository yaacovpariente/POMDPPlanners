"""Optuna tuning + evaluation demo for POMCP on TigerPOMDP via ``LocalSimulationsAPI``.

Two phases, two separate API calls (each triggers its own parent-side run on
Slack):

1. **Tuning** via :meth:`LocalSimulationsAPI.run_hyperparameter_optimization`.
   Optuna tunes POMCP's ``exploration_constant`` over ``N_TRIALS`` trials.
   Milestone Slack messages fire every
   ``HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL`` completed trials.
2. **Evaluation** via :meth:`LocalSimulationsAPI.run_multiple_environments_and_policies`.
   The winning policy returned by the optimizer is re-run on a larger
   episode budget for a tighter point estimate.

Run with::

    source .venv/bin/activate
    export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
    export HYPERPARAM_TRIAL_NOTIFICATION_INTERVAL=3   # for demo visibility
    python scripts/tune_and_evaluate_pomcp_tiger_optuna.py
"""

from __future__ import annotations

import statistics
from pathlib import Path

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import (
    EnvironmentRunParams,
    NumericalHyperParameter,
)
from POMDPPlanners.core.simulation.history import history_to_discounted_return_value
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParamPlannerConfig,
    HyperParameterOptimizationDirection,
    HyperParameterRunParams,
    OptimizedPolicyResult,
)
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import (
    LocalSimulationsAPI,
)

CACHE_DIR = Path("./cache/pomcp_tiger_optuna_demo")
NUM_PARTICLES = 50
DISCOUNT_FACTOR = 0.95
NUM_STEPS = 10
TUNING_EPISODES = 10
N_TRIALS = 10
EVALUATION_EPISODES = 100
ROLLOUT_DEPTH = 5
N_SIMULATIONS = 50


def main() -> None:
    """Run the two-phase Optuna tuning + evaluation experiment."""
    tiger = TigerPOMDP(discount_factor=DISCOUNT_FACTOR)
    initial_belief = get_initial_belief(tiger, n_particles=NUM_PARTICLES)
    api = LocalSimulationsAPI(cache_dir_path=CACHE_DIR)

    optimized = _run_tuning_phase(api, tiger, initial_belief)
    _run_evaluation_phase(api, tiger, initial_belief, optimized)


def _run_tuning_phase(
    api: LocalSimulationsAPI,
    tiger: TigerPOMDP,
    initial_belief,
) -> OptimizedPolicyResult:
    print(
        f"[tuning] Optuna over POMCP exploration_constant: "
        f"{N_TRIALS} trials, {TUNING_EPISODES} episodes/trial, {NUM_STEPS} steps"
    )
    planner_config = HyperParamPlannerConfig(
        policy_cls=POMCP,
        hyper_parameters=[
            NumericalHyperParameter(0.1, 10.0, "exploration_constant"),
        ],
        constant_parameters={
            "discount_factor": DISCOUNT_FACTOR,
            "depth": ROLLOUT_DEPTH,
            "n_simulations": N_SIMULATIONS,
            "name": "POMCP_tuning",
        },
    )
    optimization_config = HyperParameterRunParams(
        environment=tiger,
        belief=initial_belief,
        hyper_param_planner_config=planner_config,
        num_episodes=TUNING_EPISODES,
        num_steps=NUM_STEPS,
        n_trials=N_TRIALS,
        parameters_to_optimize=[("average_return", HyperParameterOptimizationDirection.MAXIMIZE)],
    )

    results = api.run_hyperparameter_optimization(
        environment_run_params=[optimization_config],
        experiment_name="optuna_tune_pomcp_tiger",
        n_jobs=1,
        cache_dir_path=CACHE_DIR / "tuning",
    )
    best = results[0]
    print(f"[tuning] Best hyperparameters: {best.chosen_hyper_parameters}")
    print(f"[tuning] Optimized metric values: {best.optimized_metric_values}")
    return best


def _run_evaluation_phase(
    api: LocalSimulationsAPI,
    tiger: TigerPOMDP,
    initial_belief,
    optimized: OptimizedPolicyResult,
) -> None:
    print(
        f"[eval] Re-running winner on {EVALUATION_EPISODES} episodes "
        f"({optimized.chosen_hyper_parameters})"
    )
    env_params = [
        EnvironmentRunParams(
            environment=tiger,
            belief=initial_belief,
            policies=[optimized.policy],
            num_episodes=EVALUATION_EPISODES,
            num_steps=NUM_STEPS,
        )
    ]
    results, _ = api.run_multiple_environments_and_policies(
        environment_run_params=env_params,
        alpha=0.1,
        confidence_interval_level=0.95,
        experiment_name="optuna_evaluate_pomcp_tiger",
        n_jobs=1,
        cache_dir_path=CACHE_DIR / "evaluation",
    )
    histories = results[tiger.name][optimized.policy.name]
    returns = [history_to_discounted_return_value(h) for h in histories]
    print(
        f"[eval] mean={statistics.fmean(returns):.3f}, "
        f"stdev={statistics.stdev(returns):.3f} over {len(returns)} episodes"
    )


if __name__ == "__main__":
    main()
